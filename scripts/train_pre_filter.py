"""Pre-Filter Trainer Script.

Extracts training dataset from historical attempt logs (`artifacts/attempts/*.jsonl`)
and fits Stage B (XGBoost / RandomForest / Fallback Classifier + TF-IDF) and Stage C
(SetFit Transformer) model weights, saving the persisted artifacts to `artifacts/models/`.

Usage:
    uv run python scripts/train_pre_filter.py --attempts-dir artifacts/attempts --output-dir artifacts/models --no-setfit

    # or with setfit
    uv run python scripts/train_pre_filter.py --attempts-dir artifacts/attempts --output-dir artifacts/models --train-setfit
"""

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
import sys

# Ensure project root is in sys.path for scenarios import
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import scenarios.debate._thread_limits  # noqa: F401 (Enforce OpenMP thread limits on import)

from agentbeats.clock import RunClock
from agentbeats.tracing import trace_span
from scenarios.debate.pre_filter import (
    CODE_DELIMITER,
    PREDICATE_PREFIX,
    extract_domain_features_batch,
    FallbackStandardScaler,
    _combine_features,
    _extract_scenario_id,
    _parse_attempt_record,
    partition_dataset_by_scenario_stratified,
)
from scenarios.debate.graph_dataflow import evaluate_graph_reachability, is_sanitizer_valid_for_sink
from scenarios.debate.graph_extractor import extract_flow_graph_snapshot

# Optional imports for model training and persistence
try:
    import joblib
except ImportError:
    joblib = None

try:
    from datasets import Dataset
    from setfit import SetFitModel, Trainer
except ImportError:
    try:
        from datasets import Dataset
        from setfit import SetFitModel, SetFitTrainer as Trainer
    except ImportError:
        SetFitModel = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_pre_filter")

CVE_REGEX = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)

# Synthetic fallback examples for cold-start / empty attempts directory
SYNTHETIC_DATA = [
    ("Predicate: The code is vulnerable to a buffer overflow in parse_string | Code: void parse_string(char *s) { strcpy(buf, s); }", 1, "CVE-2023-0001"),
    ("Predicate: The code is vulnerable to integer overflow in malloc length calculation | Code: void *p = malloc(n * sizeof(int));", 1, "CVE-2023-0002"),
    ("Predicate: The code is vulnerable to use-after-free when closing descriptor | Code: free(ptr); ptr->field = 0;", 1, "CVE-2023-0003"),
    ("Predicate: The code is vulnerable to memory corruption in array index | Code: arr[index] = val;", 1, "CVE-2023-0004"),
    ("Predicate: buy followers instantly click here for miracle offer | Code: print('click link')", 0, "CVE-2023-0005"),
    ("Predicate: random non-vulnerable function comment | Code: return x + y;", 0, "CVE-2023-0006"),
    ("Predicate: invalid formatting predicate text | Code: int main() { return 0; }", 0, "CVE-2023-0007"),
    ("Predicate: harmless utility function | Code: int add(int a, int b) { return a + b; }", 0, "CVE-2023-0008"),
]


def _validate_safe_path(target_path: Path, *, allow_outside_project: bool = True) -> Path:
    """Sanitize target path to ensure it is contained within project directory tree."""
    resolved = target_path.resolve()
    project_root = Path(__file__).resolve().parent.parent
    try:
        resolved.relative_to(project_root)
    except ValueError:
        if not allow_outside_project:
            raise ValueError(f"Path '{target_path}' points outside project root '{project_root}'")
        logger.warning("Path '%s' points outside project root '%s'. Resolving safely.", target_path, project_root)
    return resolved


class FallbackCharTfidfVectorizer:
    """Pure Python/Numpy Character N-Gram TF-IDF Vectorizer fallback."""

    def __init__(self, ngram_range: Tuple[int, int] = (3, 5), max_features: int = 1000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.array([])

    def _extract_ngrams(self, text: str) -> List[str]:
        ngrams: List[str] = []
        text_padded = f" {text.lower()} "
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(text_padded) - n + 1):
                ngrams.append(text_padded[i : i + n])
        return ngrams

    def _compute_document_row(self, doc_counter: Counter[str]) -> np.ndarray:
        row = np.zeros(len(self.vocabulary_), dtype=np.float32)
        for ng, count in doc_counter.items():
            if ng not in self.vocabulary_ or count <= 0:
                continue
            v_idx = self.vocabulary_[ng]
            tf = 1.0 + math.log(count)
            row[v_idx] = tf * self.idf_[v_idx]
        return row

    def fit_transform(self, raw_documents: List[str]) -> np.ndarray:
        doc_counts: Counter[str] = Counter()
        all_doc_ngrams: List[Counter[str]] = []

        for doc in raw_documents:
            ngrams = self._extract_ngrams(doc)
            doc_counts.update(set(ngrams))
            all_doc_ngrams.append(Counter(ngrams))

        top_vocab = [ng for ng, _ in doc_counts.most_common(self.max_features)]
        self.vocabulary_ = {ng: idx for idx, ng in enumerate(top_vocab)}

        n_docs = len(raw_documents)
        idfs = [math.log((1 + n_docs) / (1 + doc_counts[ng])) + 1.0 for ng in top_vocab]
        self.idf_ = np.array(idfs)

        matrix = np.zeros((n_docs, len(top_vocab)), dtype=np.float32)
        for d_idx, doc_counter in enumerate(all_doc_ngrams):
            matrix[d_idx] = self._compute_document_row(doc_counter)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def transform(self, raw_documents: List[str]) -> np.ndarray:
        n_docs = len(raw_documents)
        if not self.vocabulary_:
            return np.zeros((n_docs, 0), dtype=np.float32)

        matrix = np.zeros((n_docs, len(self.vocabulary_)), dtype=np.float32)
        for d_idx, doc in enumerate(raw_documents):
            matrix[d_idx] = self._compute_document_row(Counter(self._extract_ngrams(doc)))

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class FallbackClassifier:
    """Pure Numpy Distance Classifier fallback when XGBoost and sklearn are unavailable."""

    def __init__(self):
        self.pos_mean: Optional[np.ndarray] = None
        self.neg_mean: Optional[np.ndarray] = None

    def fit(self, x_data: np.ndarray, y: np.ndarray) -> None:
        if x_data.shape[0] == 0:
            return
        pos_mask = y == 1
        neg_mask = y == 0

        self.pos_mean = np.mean(x_data[pos_mask], axis=0) if np.any(pos_mask) else np.zeros(x_data.shape[1])
        self.neg_mean = np.mean(x_data[neg_mask], axis=0) if np.any(neg_mask) else np.zeros(x_data.shape[1])

    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        if x_data.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        probs = []
        for row in x_data:
            pos_dist = float(np.linalg.norm(row - self.pos_mean)) if self.pos_mean is not None else 1.0
            neg_dist = float(np.linalg.norm(row - self.neg_mean)) if self.neg_mean is not None else 1.0

            tot = pos_dist + neg_dist + 1e-6
            # Closer to pos_mean means higher pos_prob (neg_dist / tot)
            pos_prob = neg_dist / tot
            # Standard sklearn probability ordering: [P(class=0), P(class=1)]
            probs.append([1.0 - pos_prob, pos_prob])

        return np.array(probs, dtype=np.float32)

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        if x_data.shape[0] == 0:
            return np.array([], dtype=np.int32)
        probs = self.predict_proba(x_data)
        return (probs[:, 1] >= 0.5).astype(int)


def _process_jsonl_file(jsonl_file: Path) -> Tuple[List[str], List[int], List[str]]:
    """Process a single JSONL file and extract texts, labels, and scenario_ids."""
    texts: List[str] = []
    labels: List[int] = []
    scenario_ids: List[str] = []

    try:
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text, label, scenario_id = _parse_attempt_record(record)
                    if text is not None and label is not None and scenario_id is not None:
                        texts.append(text)
                        labels.append(label)
                        scenario_ids.append(scenario_id)
                except Exception as e:
                    logger.debug("Skipping malformed record at line %d in '%s': %s", line_idx, jsonl_file.name, e)
                    continue
    except Exception as e:
        logger.warning("Failed to process attempt log '%s': %s", jsonl_file.name, e)

    return texts, labels, scenario_ids





def collapse_near_duplicate_samples(
    texts: List[str],
    labels: List[int],
    scenario_ids: List[str],
    similarity_threshold: float = 0.95,
) -> Tuple[List[str], List[int], List[str]]:
    """Collapse near-duplicate texts using pairwise TF-IDF cosine similarity >= threshold."""
    if len(texts) <= 1:
        return texts, labels, scenario_ids

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        tfidf_mat = vec.fit_transform(texts)
        sim_mat = cosine_similarity(tfidf_mat)

        keep_indices = []
        visited = set()
        n = len(texts)

        for i in range(n):
            if i in visited:
                continue
            keep_indices.append(i)
            visited.add(i)
            for j in range(i + 1, n):
                if j not in visited and sim_mat[i, j] >= similarity_threshold:
                    visited.add(j)

        collapsed_texts = [texts[i] for i in keep_indices]
        collapsed_labels = [labels[i] for i in keep_indices]
        collapsed_scenarios = [scenario_ids[i] for i in keep_indices]

        logger.info(
            "Near-duplicate collapsing (threshold=%.2f): reduced %d -> %d samples (%d removed).",
            similarity_threshold, len(texts), len(collapsed_texts), len(texts) - len(collapsed_texts)
        )
        return collapsed_texts, collapsed_labels, collapsed_scenarios
    except Exception as err:
        logger.warning("Near-duplicate collapsing failed (%s). Retaining un-collapsed dataset.", err)
        return texts, labels, scenario_ids


def extract_dataset_from_attempts(
    attempts_dir: Path,
    dedup_near_duplicates: bool = False,
    similarity_threshold: float = 0.95,
) -> Tuple[List[str], List[int], List[str]]:
    """Scan all .jsonl files in attempts_dir and extract (X_texts, y_labels, scenario_ids)."""
    texts: List[str] = []
    labels: List[int] = []
    scenario_ids: List[str] = []

    safe_attempts_dir = _validate_safe_path(attempts_dir)
    if not safe_attempts_dir.exists():
        logger.warning("Attempts directory '%s' does not exist. Using synthetic training set.", safe_attempts_dir)
        for text, label, cve in SYNTHETIC_DATA:
            texts.append(text)
            labels.append(label)
            scenario_ids.append(cve)
        return texts, labels, scenario_ids

    jsonl_files = sorted(safe_attempts_dir.glob("*.jsonl"))
    logger.info("Found %d attempt files in '%s'. Extracting records...", len(jsonl_files), safe_attempts_dir)

    for jsonl_file in jsonl_files:
        file_texts, file_labels, file_scenarios = _process_jsonl_file(jsonl_file)
        texts.extend(file_texts)
        labels.extend(file_labels)
        scenario_ids.extend(file_scenarios)

    # Deduplicate exact (text, label) pairs across batch attempt runs
    seen_hashes = set()
    dedup_texts: List[str] = []
    dedup_labels: List[int] = []
    dedup_scenarios: List[str] = []

    for t, l, s in zip(texts, labels, scenario_ids):
        item_hash = hashlib.sha256(f"{t}||{l}".encode("utf-8")).hexdigest()
        if item_hash not in seen_hashes:
            seen_hashes.add(item_hash)
            dedup_texts.append(t)
            dedup_labels.append(l)
            dedup_scenarios.append(s)

    removed_cnt = len(texts) - len(dedup_texts)
    if removed_cnt > 0:
        logger.info("Deduplicated dataset: removed %d exact duplicate attempt records across runs.", removed_cnt)

    texts, labels, scenario_ids = dedup_texts, dedup_labels, dedup_scenarios

    if dedup_near_duplicates:
        texts, labels, scenario_ids = collapse_near_duplicate_samples(
            texts, labels, scenario_ids, similarity_threshold=similarity_threshold
        )

    logger.info("Final extracted dataset: %d unique attempt samples (%d accepted, %d rejected across %d scenarios).",
                len(texts), sum(labels), len(labels) - sum(labels), len(set(scenario_ids)))

    if len(texts) < 4:
        logger.warning("Extracted sample count (%d) is below minimum (4). Appending synthetic samples.", len(texts))
        for text, label, cve in SYNTHETIC_DATA:
            texts.append(text)
            labels.append(label)
            scenario_ids.append(cve)

    return texts, labels, scenario_ids


def _save_artifact(obj: Any, target_path: Path) -> None:
    """Save model object using joblib or pickle atomically with path validation."""
    safe_path = _validate_safe_path(target_path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = safe_path.with_name(f"{safe_path.name}.tmp.{os.getpid()}")
    try:
        if joblib is not None:
            joblib.dump(obj, tmp_path)
        else:
            with tmp_path.open("wb") as f:
                pickle.dump(obj, f)
        os.replace(tmp_path, safe_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _write_model_manifest(
    output_dir: Path,
    sample_count: int,
    positive_count: int,
    feature_dim: int,
) -> None:
    """Generate and write a schema-validated model_manifest.json atomically."""
    safe_dir = _validate_safe_path(output_dir)
    manifest_path = safe_dir / "model_manifest.json"

    artifacts_info: Dict[str, Any] = {}
    for filename in ["xgb.joblib", "vectorizer.joblib", "domain_scaler.joblib"]:
        filepath = safe_dir / filename
        if filepath.exists():
            with filepath.open("rb") as f:
                content = f.read()
            artifacts_info[filename] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }

    setfit_dir = safe_dir / "setfit_model"
    if setfit_dir.exists() and setfit_dir.is_dir():
        for file_path in sorted(setfit_dir.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith(".tmp"):
                rel_path = str(file_path.relative_to(safe_dir))
                with file_path.open("rb") as f:
                    content = f.read()
                artifacts_info[rel_path] = {
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }


    manifest_data = {
        "schema_version": 1,
        "artifacts": artifacts_info,
        "feature_dimension": feature_dim,
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": max(0, sample_count - positive_count),
        "trained_at": RunClock.from_env().now_iso(),
    }

    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    os.replace(tmp_path, manifest_path)




def _train_stage_b_vectorizer(
    train_texts: List[str],
    test_texts: List[str],
    output_dir: Optional[Path] = None,
) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    """Train TF-IDF vectorizer and domain StandardScaler strictly on train_texts and transform train & test splits."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        logger.info("Fitting scikit-learn TF-IDF Vectorizer strictly on X_train (char_wb 3-5 n-grams)...")
        vectorizer: Any = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
    except ImportError:
        logger.info("Fitting FallbackCharTfidfVectorizer strictly on X_train...")
        vectorizer = FallbackCharTfidfVectorizer(ngram_range=(3, 5), max_features=1000)

    try:
        from sklearn.preprocessing import StandardScaler
        scaler: Any = StandardScaler()
    except ImportError:
        scaler = FallbackStandardScaler()

    # 1. Fit TF-IDF on train_texts
    x_train_tfidf = vectorizer.fit_transform(train_texts)
    x_test_tfidf = vectorizer.transform(test_texts) if test_texts else np.zeros((0, x_train_tfidf.shape[1]))

    # 2. Extract and scale domain features strictly on train_texts
    domain_train = extract_domain_features_batch(train_texts)
    domain_test = extract_domain_features_batch(test_texts) if test_texts else np.zeros((0, domain_train.shape[1]), dtype=np.float32)

    domain_train_scaled = scaler.fit_transform(domain_train)
    domain_test_scaled = scaler.transform(domain_test) if test_texts else np.zeros((0, domain_train.shape[1]), dtype=np.float32)

    # 3. Stack TF-IDF + Domain features
    x_train_stacked = _combine_features(x_train_tfidf, domain_train_scaled)
    x_test_stacked = _combine_features(x_test_tfidf, domain_test_scaled)

    if output_dir is not None:
        vec_path = output_dir / "vectorizer.joblib"
        scaler_path = output_dir / "domain_scaler.joblib"
        _save_artifact(vectorizer, vec_path)
        _save_artifact(scaler, scaler_path)
        logger.info("Saved isolated TF-IDF vectorizer and domain scaler to '%s' and '%s'.", vec_path, scaler_path)

    return vectorizer, scaler, x_train_stacked, x_test_stacked


def _train_stage_b_classifier(x_train: np.ndarray, y_train: np.ndarray, output_dir: Optional[Path] = None) -> Any:
    """Train and persist Stage B classifier (XGBoost / Fallback) on x_train."""
    try:
        from xgboost import XGBClassifier
        logger.info("Fitting XGBoost Classifier on X_train...")
        pos_cnt = int(np.sum(y_train == 1)) or 1
        neg_cnt = int(np.sum(y_train == 0)) or 1
        scale_pos_weight = float(neg_cnt / pos_cnt)

        classifier: Any = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
        )
    except ImportError:
        logger.info("Fitting FallbackClassifier on X_train...")
        classifier = FallbackClassifier()

    classifier.fit(x_train, y_train)
    if output_dir is not None:
        xgb_path = output_dir / "xgb.joblib"
        _save_artifact(classifier, xgb_path)
        logger.info("Saved Stage B classifier to '%s'.", xgb_path)

    return classifier


def _compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate balanced accuracy score safely."""
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    sens = float(np.mean(y_pred[pos_mask] == 1)) if np.any(pos_mask) else 0.5
    spec = float(np.mean(y_pred[neg_mask] == 0)) if np.any(neg_mask) else 0.5

    return (sens + spec) / 2.0


def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute explicit confusion matrix breakdown."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "matrix": [[tn, fp], [fn, tp]],
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
    }


def _run_null_model_sanity_check(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Train control model on randomly shuffled labels to verify zero target leakage."""
    if x_test.shape[0] == 0:
        return 0.5

    rng = np.random.default_rng(seed=42)
    y_shuffled = rng.permutation(y_train)
    try:
        from xgboost import XGBClassifier
        control_clf: Any = XGBClassifier(n_estimators=20, max_depth=2, tree_method="hist", n_jobs=1)
    except ImportError:
        control_clf = FallbackClassifier()

    control_clf.fit(x_train, y_shuffled)
    y_pred_null = control_clf.predict(x_test) if hasattr(control_clf, "predict") else (control_clf.predict_proba(x_test)[:, 0] >= 0.5).astype(int)

    balanced_acc = _compute_balanced_accuracy(y_test, y_pred_null)
    logger.info("Null-Model Control Check (shuffled labels) Balanced Accuracy: %.4f", balanced_acc)

    if balanced_acc > 0.55:
        logger.warning("NULL-MODEL WARNING: Shuffled-label balanced accuracy (%.4f) exceeded 0.55 threshold!", balanced_acc)

    return balanced_acc


def _evaluate_stage_b(classifier: Any, x_eval: np.ndarray, y_eval: np.ndarray) -> Dict[str, Any]:
    """Compute Stage B (XGBoost) accuracy metrics, confusion matrix, and predictions."""
    probs_b = classifier.predict_proba(x_eval)[:, 1] if hasattr(classifier, "predict_proba") else None
    if hasattr(classifier, "predict"):
        y_pred_b = classifier.predict(x_eval)
    elif probs_b is not None:
        y_pred_b = (probs_b >= 0.5).astype(int)
    else:
        y_pred_b = np.ones(len(y_eval), dtype=int)

    acc = float(np.mean(y_pred_b == y_eval))
    bal_acc = _compute_balanced_accuracy(y_eval, y_pred_b)
    cm = _compute_confusion_matrix(y_eval, y_pred_b)

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "predictions": y_pred_b.tolist(),
        "probabilities": [round(float(p), 4) for p in probs_b] if probs_b is not None else [],
        "confusion_matrix": cm,
    }


def _evaluate_stage_c(model_dir: Optional[Path], eval_texts: Optional[List[str]], y_eval: np.ndarray, setfit_model: Any = None) -> Optional[Dict[str, Any]]:
    """Compute Stage C (SetFit) accuracy metrics."""
    if not eval_texts or SetFitModel is None:
        return None

    if setfit_model is None and model_dir is not None and (model_dir / "setfit_model").exists():
        try:
            setfit_model = SetFitModel.from_pretrained(str(model_dir / "setfit_model"))
        except Exception:
            return None

    if setfit_model is None:
        return None

    try:
        preds_c = setfit_model.predict(eval_texts)
        y_pred_c = np.array([int(p) for p in preds_c])
        acc_c = float(np.mean(y_pred_c == y_eval))
        bal_acc_c = _compute_balanced_accuracy(y_eval, y_pred_c)
        cm_c = _compute_confusion_matrix(y_eval, y_pred_c)
        return {
            "accuracy": round(acc_c, 4),
            "balanced_accuracy": round(bal_acc_c, 4),
            "predictions": y_pred_c.tolist(),
            "confusion_matrix": cm_c,
        }
    except Exception as e:
        logger.warning("SetFit holdout evaluation failed: %s", e)
        return None


def _train_stage_c_setfit(texts: List[str], labels: List[int], output_dir: Optional[Path], train_setfit: bool) -> Any:
    """Train and persist Stage C SetFit transformer model, logging in-sample fit performance."""
    if not train_setfit:
        logger.info("Stage C (SetFit) training disabled via --no-setfit flag.")
        return None

    if SetFitModel is None:
        logger.warning("SetFit package is not installed. Skipping Stage C training.")
        return None

    logger.info("Fitting SetFit Transformer Model on %d training samples...", len(texts))
    try:
        setfit_data = Dataset.from_dict({"text": texts, "label": labels})
        setfit_model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        trainer = Trainer(
            model=setfit_model,
            train_dataset=setfit_data,
        )
        trainer.train()

        # In-sample fit sanity check
        in_sample_preds = setfit_model.predict(texts)
        y_pred_in = np.array([int(p) for p in in_sample_preds])
        y_true_in = np.array(labels)
        in_acc = float(np.mean(y_pred_in == y_true_in))
        in_bal_acc = _compute_balanced_accuracy(y_true_in, y_pred_in)

        head_norm = None
        if hasattr(setfit_model, "model_head") and hasattr(setfit_model.model_head, "coef_"):
            head_norm = float(np.linalg.norm(setfit_model.model_head.coef_))

        logger.info("SetFit In-Sample Fit Verification:")
        logger.info("  Training Accuracy: %.4f | Balanced Accuracy: %.4f", in_acc, in_bal_acc)
        if head_norm is not None:
            logger.info("  Classification Head Weight Norm: %.4f", head_norm)

        if output_dir is not None:
            setfit_dir = output_dir / "setfit_model"
            safe_setfit_dir = _validate_safe_path(setfit_dir)
            setfit_model.save_pretrained(str(safe_setfit_dir))
            logger.info("Saved SetFit model to '%s'.", safe_setfit_dir)

        return setfit_model
    except Exception as e:
        logger.warning("SetFit model training failed: %s", e)
        return None


def _process_single_cv_fold(
    fold_info: Dict[str, Any],
    n_splits: int,
    train_setfit: bool,
) -> Tuple[Dict[str, Any], Tuple[List[int], List[int], List[float]], Optional[List[int]]]:
    """Execute training and evaluation on a single CV fold split."""
    f_idx = fold_info["fold"]
    tr_texts, tr_labels = fold_info["train_texts"], fold_info["train_labels"]
    te_texts, te_labels = fold_info["test_texts"], fold_info["test_labels"]
    test_scenarios = fold_info["test_scenario_ids"]

    logger.info("\n--- FOLD %d/%d ---", f_idx, n_splits)
    logger.info("  Train: %d samples (%d acc, %d rej) across %d scenarios",
                len(tr_texts), sum(tr_labels), len(tr_labels) - sum(tr_labels), len(fold_info["train_scenario_ids"]))
    logger.info("  Test:  %d samples (%d acc, %d rej) across %d scenarios (%s)",
                len(te_texts), sum(te_labels), len(te_labels) - sum(te_labels), len(test_scenarios), test_scenarios[:4])

    y_tr = np.array(tr_labels)
    y_te = np.array(te_labels)

    # 1. Per-fold isolated vectorization & training
    _, _, x_tr, x_te = _train_stage_b_vectorizer(tr_texts, te_texts, output_dir=None)
    clf_b = _train_stage_b_classifier(x_tr, y_tr, output_dir=None)

    # 2. Evaluate Stage B
    b_res = _evaluate_stage_b(clf_b, x_te, y_te)
    b_probs = b_res["probabilities"] or []

    logger.info("  Fold %d Stage B XGBoost: Acc=%.4f, BalAcc=%.4f | CM: %s",
                f_idx, b_res["accuracy"], b_res["balanced_accuracy"], b_res["confusion_matrix"]["matrix"])

    # 3. Evaluate Stage C (SetFit) if enabled
    c_res = None
    c_preds = None
    if train_setfit:
        setfit_clf = _train_stage_c_setfit(tr_texts, tr_labels, output_dir=None, train_setfit=True)
        if setfit_clf is not None:
            c_res = _evaluate_stage_c(model_dir=None, eval_texts=te_texts, y_eval=y_te, setfit_model=setfit_clf)
            if c_res is not None:
                c_preds = c_res["predictions"]
                logger.info("  Fold %d Stage C SetFit:  Acc=%.4f, BalAcc=%.4f | CM: %s",
                            f_idx, c_res["accuracy"], c_res["balanced_accuracy"], c_res["confusion_matrix"]["matrix"])

    fold_record = {
        "fold": f_idx,
        "test_scenario_ids": test_scenarios,
        "test_sample_count": len(te_labels),
        "test_positive_count": sum(te_labels),
        "test_negative_count": len(te_labels) - sum(te_labels),
        "stage_b_xgboost": b_res,
        "stage_c_setfit": c_res,
    }
    return fold_record, (te_labels, b_res["predictions"], b_probs), c_preds


def _compute_macro_cv_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate macro-averaged mean ± std across all folds."""
    b_accs = [f["stage_b_xgboost"]["accuracy"] for f in fold_results]
    b_bal_accs = [f["stage_b_xgboost"]["balanced_accuracy"] for f in fold_results]

    macro = {
        "stage_b_xgboost": {
            "mean_accuracy": round(float(np.mean(b_accs)), 4),
            "std_accuracy": round(float(np.std(b_accs)), 4),
            "mean_balanced_accuracy": round(float(np.mean(b_bal_accs)), 4),
            "std_balanced_accuracy": round(float(np.std(b_bal_accs)), 4),
        }
    }

    c_folds = [f for f in fold_results if f.get("stage_c_setfit") is not None]
    if c_folds:
        c_accs = [f["stage_c_setfit"]["accuracy"] for f in c_folds]
        c_bal_accs = [f["stage_c_setfit"]["balanced_accuracy"] for f in c_folds]
        macro["stage_c_setfit"] = {
            "mean_accuracy": round(float(np.mean(c_accs)), 4),
            "std_accuracy": round(float(np.std(c_accs)), 4),
            "mean_balanced_accuracy": round(float(np.mean(c_bal_accs)), 4),
            "std_balanced_accuracy": round(float(np.std(c_bal_accs)), 4),
        }

    return macro


def _compute_pooled_cv_metrics(
    oof_y_true: List[int],
    oof_preds_b: List[int],
    oof_probs_b: List[float],
    oof_y_true_c: List[int],
    oof_preds_c: List[int],
) -> Dict[str, Any]:
    """Calculate micro-averaged out-of-fold pooled metrics across all samples."""
    y_true_arr = np.array(oof_y_true)
    y_pred_b_arr = np.array(oof_preds_b)

    b_pooled_cm = _compute_confusion_matrix(y_true_arr, y_pred_b_arr)
    b_pooled_acc = float(np.mean(y_pred_b_arr == y_true_arr))
    b_pooled_bal_acc = _compute_balanced_accuracy(y_true_arr, y_pred_b_arr)

    roc_auc = None
    pr_auc = None
    if oof_probs_b and len(oof_probs_b) == len(oof_y_true) and len(set(oof_y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
            probs_arr = np.array(oof_probs_b)
            roc_auc = round(float(roc_auc_score(y_true_arr, probs_arr)), 4)
            precisions, recalls, _ = precision_recall_curve(y_true_arr, probs_arr)
            pr_auc = round(float(auc(recalls, precisions)), 4)
        except Exception:
            pass

    pooled: Dict[str, Any] = {
        "stage_b_xgboost": {
            "accuracy": round(b_pooled_acc, 4),
            "balanced_accuracy": round(b_pooled_bal_acc, 4),
            "confusion_matrix": b_pooled_cm,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "out_of_fold_probabilities": oof_probs_b,
        }
    }

    if oof_preds_c and oof_y_true_c and len(oof_preds_c) == len(oof_y_true_c):
        y_true_c_arr = np.array(oof_y_true_c)
        y_pred_c_arr = np.array(oof_preds_c)
        c_pooled_cm = _compute_confusion_matrix(y_true_c_arr, y_pred_c_arr)
        c_pooled_acc = float(np.mean(y_pred_c_arr == y_true_c_arr))
        c_pooled_bal_acc = _compute_balanced_accuracy(y_true_c_arr, y_pred_c_arr)
        pooled["stage_c_setfit"] = {
            "accuracy": round(c_pooled_acc, 4),
            "balanced_accuracy": round(c_pooled_bal_acc, 4),
            "confusion_matrix": c_pooled_cm,
        }

    return pooled


def _extract_code_from_combined_text(text: str) -> str:
    """Return the code portion from a pre-filter combined text record."""
    if CODE_DELIMITER not in text:
        return ""
    return text.split(CODE_DELIMITER, 1)[1]


def _graph_proof_state(bucket: str) -> str:
    """Map extraction buckets to proof states without changing fail-closed decisions."""
    if bucket == "guarded_or_safe":
        return "proven_safe"
    if bucket in {"missing_sanitizer", "wrong_sanitizer", "graph_high_risk"}:
        return "proven_unsafe"
    return "unsupported_or_unproven"


def graph_acceptance_probability_from_bucket(bucket: str) -> float:
    """Map graph proof state to vulnerability-candidate acceptance probability."""
    proof_state = _graph_proof_state(bucket)
    if proof_state == "proven_unsafe":
        return 0.95
    return 0.0


def _classify_graph_extraction_bucket(text: str, sample_idx: int) -> Tuple[str, float]:
    """Classify graph extraction outcome for fold diagnostics without changing model decisions."""
    code_text = _extract_code_from_combined_text(text)
    snapshot = extract_flow_graph_snapshot(
        code_text=code_text,
        scenario_id=f"cv_sample_{sample_idx}",
        snapshot_id=f"cv_snapshot_{sample_idx}",
        version=1,
        created_at=0.0,
    )

    if not snapshot.is_complete or snapshot.parse_error is not None:
        return "unsupported_syntax", 1.0

    source_nodes = [node for node in snapshot.nodes.values() if node.get("kind") == "source"]
    if not source_nodes:
        return "missing_source", 1.0

    if not snapshot.signatures:
        return "missing_sink", 1.0

    risk_score = evaluate_graph_reachability(snapshot)
    if risk_score < 0.10:
        return "guarded_or_safe", risk_score

    for sig in snapshot.signatures:
        if sig.source_type != "UNTRUSTED_INPUT":
            continue
        if not sig.sanitizer_type:
            return "missing_sanitizer", risk_score
        if not is_sanitizer_valid_for_sink(sig.sink_type, sig.sanitizer_type):
            return "wrong_sanitizer", risk_score

        sink_node = snapshot.nodes.get(sig.sink_id, {})
        if sig.guarded_target != sink_node.get("target_var"):
            return "wrong_sanitizer", risk_score

    return "graph_high_risk", risk_score


def collect_graph_bucket_examples(
    texts: List[str],
    labels: List[int],
    scenario_ids: List[str],
    limit_per_bucket: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect representative graph diagnostic examples for corpus-driven extractor work."""
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for idx, (text, label, scenario_id) in enumerate(zip(texts, labels, scenario_ids, strict=True)):
        bucket, risk_score = _classify_graph_extraction_bucket(text, idx)
        if len(examples[bucket]) >= limit_per_bucket:
            continue

        code_text = _extract_code_from_combined_text(text)
        examples[bucket].append(
            {
                "sample_idx": idx,
                "label": int(label),
                "scenario_id": scenario_id,
                "bucket": bucket,
                "proof_state": _graph_proof_state(bucket),
                "risk_score": risk_score,
                "code_text": code_text,
            }
        )

    return dict(sorted(examples.items()))


def _compute_graph_fold_diagnostics(
    texts: List[str],
    labels: List[int],
    predictions: List[int],
) -> Dict[str, Any]:
    """Compute graph parser coverage and error buckets for fold reports."""
    bucket_counts: Counter[str] = Counter()
    error_bucket_counts: Counter[str] = Counter()
    proof_state_counts: Counter[str] = Counter()
    high_risk_count = 0
    low_risk_count = 0
    positive_evidence_count = 0
    no_positive_evidence_count = 0
    parse_complete_count = 0

    for idx, (text, label, prediction) in enumerate(zip(texts, labels, predictions, strict=True)):
        bucket, risk_score = _classify_graph_extraction_bucket(text, idx)
        bucket_counts[bucket] += 1
        proof_state_counts[_graph_proof_state(bucket)] += 1
        if graph_acceptance_probability_from_bucket(bucket) >= 0.90:
            positive_evidence_count += 1
        else:
            no_positive_evidence_count += 1
        if bucket != "unsupported_syntax":
            parse_complete_count += 1
        if risk_score >= 0.10:
            high_risk_count += 1
        else:
            low_risk_count += 1
        if int(label) != int(prediction):
            error_bucket_counts[bucket] += 1

    total = len(texts)
    return {
        "total_samples": total,
        "parse_complete_count": parse_complete_count,
        "parse_failed_count": total - parse_complete_count,
        "parser_coverage": round(float(parse_complete_count / total), 4) if total else 0.0,
        "graph_high_risk_count": high_risk_count,
        "graph_low_risk_count": low_risk_count,
        "graph_positive_evidence_count": positive_evidence_count,
        "graph_no_positive_evidence_count": no_positive_evidence_count,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "proof_state_counts": dict(sorted(proof_state_counts.items())),
        "prediction_error_bucket_counts": dict(sorted(error_bucket_counts.items())),
        "accepted_logic_error_rate": None,
        "accepted_logic_error_note": "not_computed_from_fold_texts_only",
    }


def _json_default_encoder(obj: Any) -> Any:
    """Convert numpy types or non-serializable objects for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return str(obj)


def _train_final_production_models(
    texts: List[str],
    labels: List[int],
    output_dir: Path,
    train_setfit: bool = False,
) -> None:
    """Train and persist final Stage B (and optional Stage C) production models on full dataset."""
    logger.info("Training final production models on full dataset (%d samples)...", len(texts))
    _, _, x_full, _ = _train_stage_b_vectorizer(texts, [], output_dir=output_dir)
    y_full = np.array(labels)
    _train_stage_b_classifier(x_full, y_full, output_dir=output_dir)

    if train_setfit:
        _train_stage_c_setfit(texts, labels, output_dir=output_dir, train_setfit=True)

    pos_cnt = int(np.sum(y_full == 1))
    feat_dim = int(x_full.shape[1]) if len(x_full.shape) > 1 else 0
    _write_model_manifest(
        output_dir=output_dir,
        sample_count=len(texts),
        positive_count=pos_cnt,
        feature_dim=feat_dim,
    )



def _print_cv_summary_report(
    n_folds: int,
    texts: List[str],
    scenario_ids: List[str],
    macro_metrics: Dict[str, Any],
    pooled_metrics: Dict[str, Any],
) -> None:
    """Print clean CV metrics summary to console stdout."""
    b_pooled_cm = pooled_metrics["stage_b_xgboost"]["confusion_matrix"]
    b_pooled_acc = pooled_metrics["stage_b_xgboost"]["accuracy"]
    b_pooled_bal_acc = pooled_metrics["stage_b_xgboost"]["balanced_accuracy"]

    print("\n=======================================================")
    print(f"=== {n_folds}-FOLD STRATIFIED SCENARIO-GROUPED CV RESULTS ===")
    print("=======================================================")
    print(f"Total Samples: {len(texts)} | Total Unique Scenarios: {len(set(scenario_ids))}")
    print("\n--- Stage B (XGBoost) ---")
    print(f"  Macro Mean Balanced Accuracy: {macro_metrics['stage_b_xgboost']['mean_balanced_accuracy']:.4f} ± {macro_metrics['stage_b_xgboost']['std_balanced_accuracy']:.4f}")
    print(f"  Macro Mean Accuracy:          {macro_metrics['stage_b_xgboost']['mean_accuracy']:.4f} ± {macro_metrics['stage_b_xgboost']['std_accuracy']:.4f}")
    print(f"  Pooled Out-of-Fold BalAcc:   {b_pooled_bal_acc:.4f} (Accuracy: {b_pooled_acc:.4f})")
    print(f"  Pooled Confusion Matrix:      TN={b_pooled_cm['tn']}, FP={b_pooled_cm['fp']}, FN={b_pooled_cm['fn']}, TP={b_pooled_cm['tp']}")
    print(f"  Pooled Sensitivity (TPR):     {b_pooled_cm['sensitivity']:.4f}")
    print(f"  Pooled Specificity (TNR):     {b_pooled_cm['specificity']:.4f}")
    if pooled_metrics["stage_b_xgboost"].get("roc_auc") is not None:
        print(f"  Pooled ROC-AUC Score:         {pooled_metrics['stage_b_xgboost']['roc_auc']:.4f}")
    if pooled_metrics["stage_b_xgboost"].get("pr_auc") is not None:
        print(f"  Pooled PR-AUC Score:          {pooled_metrics['stage_b_xgboost']['pr_auc']:.4f}")

    if "stage_c_setfit" in macro_metrics:
        print("\n--- Stage C (SetFit) ---")
        print(f"  Macro Mean Balanced Accuracy: {macro_metrics['stage_c_setfit']['mean_balanced_accuracy']:.4f} ± {macro_metrics['stage_c_setfit']['std_balanced_accuracy']:.4f}")
        print(f"  Macro Mean Accuracy:          {macro_metrics['stage_c_setfit']['mean_accuracy']:.4f} ± {macro_metrics['stage_c_setfit']['std_accuracy']:.4f}")
        print(f"  Pooled Out-of-Fold BalAcc:   {pooled_metrics['stage_c_setfit']['balanced_accuracy']:.4f}")
        print(f"  Pooled Confusion Matrix:      TN={pooled_metrics['stage_c_setfit']['confusion_matrix']['tn']}, FP={pooled_metrics['stage_c_setfit']['confusion_matrix']['fp']}, FN={pooled_metrics['stage_c_setfit']['confusion_matrix']['fn']}, TP={pooled_metrics['stage_c_setfit']['confusion_matrix']['tp']}")

    print("=======================================================\n")


def run_kfold_cross_validation(
    texts: List[str],
    labels: List[int],
    scenario_ids: List[str],
    output_dir: Path,
    n_splits: int = 5,
    train_setfit: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute Stratified Scenario-Grouped K-Fold CV with macro/pooled metrics and per-fold tracking."""
    logger.info("Starting %d-Fold Stratified Scenario-Grouped Cross-Validation...", n_splits)
    folds = partition_dataset_by_scenario_stratified(texts, labels, scenario_ids, n_splits=n_splits, seed=seed)

    fold_results = []
    oof_y_true: List[int] = []
    oof_preds_b: List[int] = []
    oof_probs_b: List[float] = []
    oof_texts: List[str] = []
    oof_y_true_c: List[int] = []
    oof_preds_c: List[int] = []

    for fold_info in folds:
        fold_rec, (te_labels, b_preds, b_probs), c_preds = _process_single_cv_fold(fold_info, n_splits, train_setfit)
        fold_results.append(fold_rec)
        oof_y_true.extend(te_labels)
        oof_preds_b.extend(b_preds)
        oof_probs_b.extend(b_probs)
        oof_texts.extend(fold_info["test_texts"])
        if c_preds is not None:
            oof_y_true_c.extend(te_labels)
            oof_preds_c.extend(c_preds)

    macro_metrics = _compute_macro_cv_metrics(fold_results)
    pooled_metrics = _compute_pooled_cv_metrics(oof_y_true, oof_preds_b, oof_probs_b, oof_y_true_c, oof_preds_c)
    graph_diagnostics = _compute_graph_fold_diagnostics(oof_texts, oof_y_true, oof_preds_b)

    full_report = {
        "total_samples": len(texts),
        "total_scenarios": len(set(scenario_ids)),
        "n_folds": len(folds),
        "requested_n_folds": n_splits,
        "partition_seed": seed,
        "macro_metrics": macro_metrics,
        "pooled_micro_metrics": pooled_metrics,
        "graph_diagnostics": graph_diagnostics,
        "fold_details": fold_results,
    }

    _print_cv_summary_report(len(folds), texts, scenario_ids, macro_metrics, pooled_metrics)

    # Persist full report
    report_path = output_dir / "cv_holdout_metrics.json"
    safe_report_path = _validate_safe_path(report_path)
    with safe_report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=_json_default_encoder)

    logger.info("Saved Stratified Scenario-Grouped CV report to '%s'.", safe_report_path)

    # Train final production model on full dataset
    _train_final_production_models(texts, labels, output_dir=output_dir, train_setfit=train_setfit)

    return full_report


def train_pre_filter(
    attempts_dir: Path,
    output_dir: Path,
    train_setfit: bool = False,
    k_folds: int = 5,
    dedup_near_duplicates: bool = False,
    similarity_threshold: float = 0.95,
) -> bool:
    """Train pre-filter models with Stratified Scenario-Grouped K-Fold Cross Validation."""
    try:
        with trace_span("train_pre_filter", attributes={"attempts_dir": str(attempts_dir), "output_dir": str(output_dir), "train_setfit": train_setfit, "k_folds": k_folds, "dedup_near_duplicates": dedup_near_duplicates}):
            safe_output_dir = _validate_safe_path(output_dir)
            safe_output_dir.mkdir(parents=True, exist_ok=True)

            safe_attempts_dir = _validate_safe_path(attempts_dir)
            with trace_span("train_pre_filter.dataset_extraction", attributes={"attempts_dir": str(safe_attempts_dir)}):
                texts, labels, scenario_ids = extract_dataset_from_attempts(
                    safe_attempts_dir,
                    dedup_near_duplicates=dedup_near_duplicates,
                    similarity_threshold=similarity_threshold,
                )
            if not texts:
                raise ValueError(f"No valid attempt data found in '{safe_attempts_dir}'.")

            if k_folds > 1:
                run_kfold_cross_validation(texts, labels, scenario_ids, safe_output_dir, n_splits=k_folds, train_setfit=train_setfit)
            else:
                logger.info("k_folds <= 1 specified. Training single split...")
                _train_final_production_models(texts, labels, output_dir=safe_output_dir, train_setfit=train_setfit)

            return True
    except Exception as e:
        logger.exception("Pre-filter model training failed: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BARRED 3-Stage Acceptance Pre-Filter Models with Stratified Scenario-Grouped K-Fold CV.")
    parser.add_argument("--attempts-dir", type=str, default="artifacts/attempts", help="Directory containing attempt log JSONL files.")
    parser.add_argument("--output-dir", type=str, default="artifacts/models", help="Directory to save trained model artifacts.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of Stratified Scenario-Grouped CV folds (default: 5).")
    parser.add_argument("--dedup-near-duplicates", action="store_true", default=False, help="Enable cosine-similarity near-duplicate sample collapsing.")
    parser.add_argument("--similarity-threshold", type=float, default=0.95, help="Cosine similarity threshold for near-duplicate collapsing (default: 0.95).")
    parser.add_argument("--train-setfit", action="store_true", default=False, help="Enable heavy SetFit transformer model training.")
    parser.add_argument("--no-setfit", dest="train_setfit", action="store_false", help="Disable SetFit transformer model training.")

    args = parser.parse_args()
    attempts_path = Path(args.attempts_dir)
    output_path = Path(args.output_dir)

    if train_pre_filter(
        attempts_path,
        output_path,
        train_setfit=args.train_setfit,
        k_folds=args.k_folds,
        dedup_near_duplicates=args.dedup_near_duplicates,
        similarity_threshold=args.similarity_threshold,
    ):
        logger.info("Pre-filter training pipeline completed successfully!")
    else:
        logger.error("Pre-filter training failed.")


if __name__ == "__main__":
    main()
