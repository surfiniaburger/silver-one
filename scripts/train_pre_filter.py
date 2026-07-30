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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scenarios.debate._thread_limits  # noqa: F401 (Enforce OpenMP thread limits on import)

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


def _validate_safe_path(target_path: Path) -> Path:
    """Sanitize target path to ensure it is contained within project directory tree."""
    resolved = target_path.resolve()
    project_root = Path(__file__).resolve().parent.parent
    try:
        resolved.relative_to(project_root)
    except ValueError:
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] == 0:
            return
        pos_mask = y == 1
        neg_mask = y == 0

        self.pos_mean = np.mean(X[pos_mask], axis=0) if np.any(pos_mask) else np.zeros(X.shape[1])
        self.neg_mean = np.mean(X[neg_mask], axis=0) if np.any(neg_mask) else np.zeros(X.shape[1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        probs = []
        for row in X:
            pos_dist = float(np.linalg.norm(row - self.pos_mean)) if self.pos_mean is not None else 1.0
            neg_dist = float(np.linalg.norm(row - self.neg_mean)) if self.neg_mean is not None else 1.0

            tot = pos_dist + neg_dist + 1e-6
            # Closer to pos_mean means higher pos_prob (neg_dist / tot)
            pos_prob = neg_dist / tot
            # Standard sklearn probability ordering: [P(class=0), P(class=1)]
            probs.append([1.0 - pos_prob, pos_prob])

        return np.array(probs, dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.array([], dtype=np.int32)
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def _extract_cve_id(record: dict, predicate: str) -> str:
    """Resolve CVE identifier or generate deterministic hash fallback."""
    seed_val = record.get("seed")
    seed_dict = seed_val if isinstance(seed_val, dict) else {}
    cve_id = record.get("cve_id") or seed_dict.get("cve_id")
    if cve_id:
        return str(cve_id)

    cve_match = CVE_REGEX.search(predicate)
    if cve_match:
        return cve_match.group(0).upper()

    return f"HASH-{hashlib.sha256(predicate.encode('utf-8')).hexdigest()[:10]}"


def _extract_code_snippet(record: dict, judge_dict: dict) -> str:
    """Extract code anchors or fall back to raw input_block."""
    anchors = record.get("anchors_normalized") or judge_dict.get("anchors", [])
    if isinstance(anchors, list) and anchors:
        return " ".join(str(a) for a in anchors)
    if anchors and not isinstance(anchors, list):
        return str(anchors)
    return str(record.get("input_block", ""))


def _parse_attempt_record(record: dict) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Extract (combined_text, label, cve_id) from a single attempt record dictionary."""
    if not isinstance(record, dict):
        return None, None, None

    decision = str(record.get("decision", "")).lower()
    if decision not in ("accepted", "rejected"):
        return None, None, None

    judge_eval = record.get("judge_eval")
    judge_dict = judge_eval if isinstance(judge_eval, dict) else {}
    predicate = record.get("predicate") or judge_dict.get("predicate", "")
    if not predicate or not isinstance(predicate, str):
        return None, None, None

    label = 1 if decision == "accepted" else 0
    cve_id = _extract_cve_id(record, predicate)
    code_snippet = _extract_code_snippet(record, judge_dict)

    combined_text = f"Predicate: {predicate} | Code: {code_snippet[:1000]}"
    return combined_text, label, cve_id


def _process_jsonl_file(jsonl_file: Path) -> Tuple[List[str], List[int], List[str]]:
    """Process a single JSONL file and extract texts, labels, and cve_ids."""
    texts: List[str] = []
    labels: List[int] = []
    cve_ids: List[str] = []

    try:
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text, label, cve_id = _parse_attempt_record(record)
                    if text is not None and label is not None and cve_id is not None:
                        texts.append(text)
                        labels.append(label)
                        cve_ids.append(cve_id)
                except Exception as e:
                    logger.debug("Skipping malformed record at line %d in '%s': %s", line_idx, jsonl_file.name, e)
                    continue
    except Exception as e:
        logger.warning("Failed to process attempt log '%s': %s", jsonl_file.name, e)

    return texts, labels, cve_ids


def extract_dataset_from_attempts(attempts_dir: Path) -> Tuple[List[str], List[int], List[str]]:
    """Scan all .jsonl files in attempts_dir and extract (X_texts, y_labels, cve_ids)."""
    texts: List[str] = []
    labels: List[int] = []
    cve_ids: List[str] = []

    safe_attempts_dir = _validate_safe_path(attempts_dir)
    if not safe_attempts_dir.exists():
        logger.warning("Attempts directory '%s' does not exist. Using synthetic training set.", safe_attempts_dir)
        for text, label, cve in SYNTHETIC_DATA:
            texts.append(text)
            labels.append(label)
            cve_ids.append(cve)
        return texts, labels, cve_ids

    jsonl_files = sorted(safe_attempts_dir.glob("*.jsonl"))
    logger.info("Found %d attempt files in '%s'. Extracting records...", len(jsonl_files), safe_attempts_dir)

    for jsonl_file in jsonl_files:
        file_texts, file_labels, file_cves = _process_jsonl_file(jsonl_file)
        texts.extend(file_texts)
        labels.extend(file_labels)
        cve_ids.extend(file_cves)

    logger.info("Extracted %d valid attempt samples (%d accepted, %d rejected).",
                len(texts), sum(labels), len(labels) - sum(labels))

    if len(texts) < 4:
        logger.warning("Extracted sample count (%d) is below minimum (4). Appending synthetic samples.", len(texts))
        for text, label, cve in SYNTHETIC_DATA:
            texts.append(text)
            labels.append(label)
            cve_ids.append(cve)

    return texts, labels, cve_ids


def partition_dataset_by_cve(
    texts: List[str],
    labels: List[int],
    cve_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """Group attempt records strictly by cve_id into Train, Validation, and Test sets."""
    grouped: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for text, label, cve in zip(texts, labels, cve_ids):
        grouped[cve].append((text, label))

    unique_cves = sorted(grouped.keys())
    # Deterministic shuffle (fixed seed=1337) so lexicographic order does not bias split labels
    rng = np.random.default_rng(seed=1337)
    cve_arr = np.array(unique_cves)
    rng.shuffle(cve_arr)
    unique_cves = list(cve_arr)

    n_cves = len(unique_cves)

    if n_cves < 3:
        # Fallback to sample-level split if insufficient unique CVEs
        n_train = max(1, int(len(texts) * train_ratio))
        n_val = max(1, int(len(texts) * val_ratio)) if len(texts) >= 3 else 0
        return {
            "train": (texts[:n_train], labels[:n_train]),
            "val": (texts[n_train : n_train + n_val], labels[n_train : n_train + n_val]),
            "test": (texts[n_train + n_val :], labels[n_train + n_val :]),
        }

    n_train_cves = max(1, int(n_cves * train_ratio))
    n_val_cves = max(1, int(n_cves * val_ratio))

    train_cve_set = set(unique_cves[:n_train_cves])
    val_cve_set = set(unique_cves[n_train_cves : n_train_cves + n_val_cves])

    splits: Dict[str, Tuple[List[str], List[int]]] = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }

    for cve, items in grouped.items():
        if cve in train_cve_set:
            target = "train"
        elif cve in val_cve_set:
            target = "val"
        else:
            target = "test"

        t_texts, t_labels = splits[target]
        for txt, lbl in items:
            t_texts.append(txt)
            t_labels.append(lbl)

    return splits


def _save_artifact(obj: Any, target_path: Path) -> None:
    """Save model object using joblib or pickle with path validation."""
    safe_path = _validate_safe_path(target_path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(obj, safe_path)
    else:
        with safe_path.open("wb") as f:
            pickle.dump(obj, f)


def _train_stage_b_vectorizer(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    output_dir: Path,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Train TF-IDF vectorizer strictly on train_texts and transform all splits."""
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

    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts) if val_texts else np.zeros((0, x_train.shape[1]))
    x_test = vectorizer.transform(test_texts) if test_texts else np.zeros((0, x_train.shape[1]))

    vec_path = output_dir / "vectorizer.joblib"
    _save_artifact(vectorizer, vec_path)
    logger.info("Saved isolated TF-IDF vectorizer to '%s'.", vec_path)
    return vectorizer, x_train, x_val, x_test


def _train_stage_b_classifier(x_train: np.ndarray, y_train: np.ndarray, output_dir: Path) -> Any:
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


def _evaluate_stage_b(classifier: Any, x_eval: np.ndarray, y_eval: np.ndarray) -> Dict[str, float]:
    """Compute Stage B (XGBoost) accuracy metrics."""
    if hasattr(classifier, "predict"):
        y_pred_b = classifier.predict(x_eval)
    elif hasattr(classifier, "predict_proba"):
        probs_b = classifier.predict_proba(x_eval)[:, 1]
        y_pred_b = (probs_b >= 0.5).astype(int)
    else:
        y_pred_b = np.ones(len(y_eval), dtype=int)

    acc = float(np.mean(y_pred_b == y_eval))
    bal_acc = _compute_balanced_accuracy(y_eval, y_pred_b)
    return {"accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4)}


def _evaluate_stage_c(model_dir: Path, eval_texts: Optional[List[str]], y_eval: np.ndarray, setfit_model: Any = None) -> Optional[Dict[str, float]]:
    """Compute Stage C (SetFit) accuracy metrics."""
    if not eval_texts or SetFitModel is None:
        return None

    if setfit_model is None and (model_dir / "setfit_model").exists():
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
        return {"accuracy": round(acc_c, 4), "balanced_accuracy": round(bal_acc_c, 4)}
    except Exception as e:
        logger.warning("SetFit holdout evaluation failed: %s", e)
        return None


def _evaluate_cascade(model_dir: Path, eval_texts: Optional[List[str]], y_eval: np.ndarray) -> Optional[Dict[str, Any]]:
    """Compute full 3-stage cascade accuracy metrics."""
    if not eval_texts:
        return None

    try:
        from scenarios.debate.pre_filter import BarredPreFilter
        cascade = BarredPreFilter(model_dir=model_dir)
        cascade_preds = [1 if cascade.predict(predicate=t, input_block="").accept else 0 for t in eval_texts]
        y_pred = np.array(cascade_preds)
        acc = float(np.mean(y_pred == y_eval))
        bal_acc = _compute_balanced_accuracy(y_eval, y_pred)
        return {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "intercepted_count": int(np.sum(y_pred == 0)),
        }
    except Exception as e:
        logger.warning("Cascade evaluation failed: %s", e)
        return None


def _evaluate_holdout_performance(
    classifier: Any,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    output_dir: Path,
    eval_texts: Optional[List[str]] = None,
    setfit_model: Any = None,
) -> Dict[str, Any]:
    """Compute and persist independent holdout metrics for Stage B, Stage C, and Cascade."""
    if x_eval.shape[0] == 0:
        logger.warning("Holdout test set is empty. Skipping detailed metric export.")
        return {}

    y_arr = np.array(y_eval)
    stage_b_metrics = _evaluate_stage_b(classifier, x_eval, y_arr)

    metrics: Dict[str, Any] = {
        "test_samples": int(len(y_arr)),
        "accepted_samples": int(np.sum(y_arr == 1)),
        "rejected_samples": int(np.sum(y_arr == 0)),
        "stage_b_xgboost": stage_b_metrics,
        "accuracy": stage_b_metrics["accuracy"],
        "balanced_accuracy": stage_b_metrics["balanced_accuracy"],
    }

    stage_c_metrics = _evaluate_stage_c(output_dir, eval_texts, y_arr, setfit_model=setfit_model)
    if stage_c_metrics is not None:
        metrics["stage_c_setfit"] = stage_c_metrics

    cascade_metrics = _evaluate_cascade(output_dir, eval_texts, y_arr)
    if cascade_metrics is not None:
        metrics["cascade_overall"] = cascade_metrics

    metrics_path = output_dir / "holdout_metrics.json"
    safe_metrics_path = _validate_safe_path(metrics_path)
    with safe_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Holdout Evaluation Metrics saved to '%s':\n%s", safe_metrics_path, json.dumps(metrics, indent=2))
    return metrics


def _train_stage_c_setfit(texts: List[str], labels: List[int], output_dir: Path, train_setfit: bool) -> Any:
    """Train and persist Stage C SetFit transformer model if requested."""
    if not train_setfit:
        logger.info("Stage C (SetFit) training disabled via --no-setfit flag.")
        return None

    if SetFitModel is None:
        logger.warning("SetFit package is not installed. Skipping Stage C training.")
        return None

    setfit_dir = output_dir / "setfit_model"
    safe_setfit_dir = _validate_safe_path(setfit_dir)
    logger.info("Fitting SetFit Transformer Model...")
    try:
        setfit_data = Dataset.from_dict({"text": texts, "label": labels})
        setfit_model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        trainer = Trainer(
            model=setfit_model,
            train_dataset=setfit_data,
        )
        trainer.train()
        setfit_model.save_pretrained(str(safe_setfit_dir))
        logger.info("Saved SetFit model to '%s'.", safe_setfit_dir)
        return setfit_model
    except Exception as e:
        logger.warning("SetFit model training failed: %s", e)
        return None


def train_pre_filter(
    attempts_dir: Path,
    output_dir: Path,
    train_setfit: bool = False,
) -> bool:
    """Train pre-filter models (Stage B XGBoost & Stage C SetFit) with leak-proof evaluation."""
    safe_output_dir = _validate_safe_path(output_dir)
    safe_output_dir.mkdir(parents=True, exist_ok=True)

    safe_attempts_dir = _validate_safe_path(attempts_dir)
    texts, labels, cve_ids = extract_dataset_from_attempts(safe_attempts_dir)
    if not texts:
        logger.error("No valid attempt data available for training.")
        return False

    try:
        splits = partition_dataset_by_cve(texts, labels, cve_ids)
        train_texts, train_labels = splits["train"]
        val_texts, val_labels = splits["val"]
        test_texts, test_labels = splits["test"]

        # If test set is empty, fall back to evaluating on validation or train split
        if test_texts:
            eval_texts, eval_labels = test_texts, test_labels
        elif val_texts:
            eval_texts, eval_labels = val_texts, val_labels
        else:
            eval_texts, eval_labels = train_texts, train_labels

        y_train = np.array(train_labels)
        y_eval = np.array(eval_labels)

        # Stage B Isolated Vectorization
        _, x_train, _, x_eval = _train_stage_b_vectorizer(train_texts, val_texts, eval_texts, safe_output_dir)

        # Stage B Classifier Training
        classifier = _train_stage_b_classifier(x_train, y_train, safe_output_dir)

        # Null Model Control Check
        _run_null_model_sanity_check(x_train, y_train, x_eval, y_eval)

        # Stage C SetFit Training
        setfit_model = _train_stage_c_setfit(train_texts, train_labels, safe_output_dir, train_setfit)

        # Multi-Stage Holdout Benchmarking
        _evaluate_holdout_performance(classifier, x_eval, y_eval, safe_output_dir, eval_texts=eval_texts, setfit_model=setfit_model)

        return True
    except Exception as e:
        logger.exception("Pre-filter model training failed: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BARRED 3-Stage Acceptance Pre-Filter Models.")
    parser.add_argument("--attempts-dir", type=str, default="artifacts/attempts", help="Directory containing attempt log JSONL files.")
    parser.add_argument("--output-dir", type=str, default="artifacts/models", help="Directory to save trained model artifacts.")
    parser.add_argument("--train-setfit", action="store_true", default=False, help="Enable heavy SetFit transformer model training.")
    parser.add_argument("--no-setfit", dest="train_setfit", action="store_false", help="Disable SetFit transformer model training.")

    args = parser.parse_args()
    attempts_path = Path(args.attempts_dir)
    output_path = Path(args.output_dir)

    if train_pre_filter(attempts_path, output_path, train_setfit=args.train_setfit):
        logger.info("Pre-filter training pipeline completed successfully!")
    else:
        logger.error("Pre-filter training failed.")


if __name__ == "__main__":
    main()
