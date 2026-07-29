"""Pre-Filter Trainer Script.

Extracts training dataset from historical attempt logs (`artifacts/attempts/*.jsonl`)
and fits Stage B (XGBoost / RandomForest / Fallback Classifier + TF-IDF) and Stage C
(SetFit Transformer) model weights, saving the persisted artifacts to `artifacts/models/`.

Usage:
    python scripts/train_pre_filter.py --attempts-dir artifacts/attempts --output-dir artifacts/models --no-setfit
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

import numpy as np

# Optional imports for model training and persistence
try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
except ImportError:
    SklearnTfidfVectorizer = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None

try:
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
except ImportError:
    SetFitModel = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_pre_filter")

# Synthetic fallback examples for cold-start / empty attempts directory
SYNTHETIC_DATA = [
    ("Predicate: The code is vulnerable to a buffer overflow in parse_string | Code: void parse_string(char *s) { strcpy(buf, s); }", 1),
    ("Predicate: The code is vulnerable to integer overflow in malloc length calculation | Code: void *p = malloc(n * sizeof(int));", 1),
    ("Predicate: The code is vulnerable to use-after-free when closing descriptor | Code: free(ptr); ptr->field = 0;", 1),
    ("Predicate: The code is vulnerable to memory corruption in array index | Code: arr[index] = val;", 1),
    ("Predicate: buy followers instantly click here for miracle offer | Code: print('click link')", 0),
    ("Predicate: random non-vulnerable function comment | Code: return x + y;", 0),
    ("Predicate: invalid formatting predicate text | Code: int main() { return 0; }", 0),
    ("Predicate: harmless utility function | Code: int add(int a, int b) { return a + b; }", 0),
]


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

    def fit_transform(self, raw_documents: List[str]) -> np.ndarray:
        doc_counts: Counter[str] = Counter()
        all_doc_ngrams: List[Counter[str]] = []

        for doc in raw_documents:
            ngrams = self._extract_ngrams(doc)
            unique_ngrams = set(ngrams)
            for ng in unique_ngrams:
                doc_counts[ng] += 1
            all_doc_ngrams.append(Counter(ngrams))

        top_vocab = [ng for ng, _ in doc_counts.most_common(self.max_features)]
        self.vocabulary_ = {ng: idx for idx, ng in enumerate(top_vocab)}

        n_docs = len(raw_documents)
        idfs = [math.log((1 + n_docs) / (1 + doc_counts[ng])) + 1.0 for ng in top_vocab]
        self.idf_ = np.array(idfs)

        matrix = np.zeros((n_docs, len(top_vocab)), dtype=np.float32)
        for d_idx, doc_counter in enumerate(all_doc_ngrams):
            for ng, count in doc_counter.items():
                if ng in self.vocabulary_:
                    v_idx = self.vocabulary_[ng]
                    tf = 1.0 + math.log(count) if count > 0 else 0.0
                    matrix[d_idx, v_idx] = tf * self.idf_[v_idx]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def transform(self, raw_documents: List[str]) -> np.ndarray:
        n_docs = len(raw_documents)
        matrix = np.zeros((n_docs, len(self.vocabulary_)), dtype=np.float32)

        for d_idx, doc in enumerate(raw_documents):
            ngrams = self._extract_ngrams(doc)
            doc_counter = Counter(ngrams)
            for ng, count in doc_counter.items():
                if ng in self.vocabulary_:
                    v_idx = self.vocabulary_[ng]
                    tf = 1.0 + math.log(count) if count > 0 else 0.0
                    matrix[d_idx, v_idx] = tf * self.idf_[v_idx]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class FallbackClassifier:
    """Pure Numpy Distance Classifier fallback when XGBoost and sklearn are unavailable."""

    def __init__(self):
        self.pos_mean: Optional[np.ndarray] = None
        self.neg_mean: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pos_mask = y == 1
        neg_mask = y == 0

        self.pos_mean = np.mean(X[pos_mask], axis=0) if np.any(pos_mask) else np.zeros(X.shape[1])
        self.neg_mean = np.mean(X[neg_mask], axis=0) if np.any(neg_mask) else np.zeros(X.shape[1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for row in X:
            pos_dist = float(np.linalg.norm(row - self.pos_mean)) if self.pos_mean is not None else 1.0
            neg_dist = float(np.linalg.norm(row - self.neg_mean)) if self.neg_mean is not None else 1.0

            tot = pos_dist + neg_dist + 1e-6
            probs.append([pos_dist / tot, neg_dist / tot])

        return np.array(probs, dtype=np.float32)


def _parse_attempt_record(record: dict) -> Tuple[Optional[str], Optional[int]]:
    """Extract (combined_text, label) from a single attempt record dictionary."""
    decision = record.get("decision", "").lower()
    if decision not in ("accepted", "rejected"):
        return None, None

    label = 1 if decision == "accepted" else 0
    predicate = record.get("predicate") or record.get("judge_eval", {}).get("predicate", "")
    if not predicate:
        return None, None

    anchors = record.get("anchors_normalized") or record.get("judge_eval", {}).get("anchors", [])
    code_snippet = " ".join(anchors) if isinstance(anchors, list) else str(anchors)
    if not code_snippet:
        code_snippet = record.get("input_block", "")

    combined_text = f"Predicate: {predicate} | Code: {code_snippet[:1000]}"
    return combined_text, label


def _process_jsonl_file(jsonl_file: Path) -> Tuple[List[str], List[int]]:
    """Process a single JSONL file and extract texts and labels."""
    texts: List[str] = []
    labels: List[int] = []

    try:
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                text, label = _parse_attempt_record(record)
                if text is not None and label is not None:
                    texts.append(text)
                    labels.append(label)
    except Exception as e:
        logger.warning("Failed to process attempt log '%s': %s", jsonl_file.name, e)

    return texts, labels


def extract_dataset_from_attempts(attempts_dir: Path) -> Tuple[List[str], List[int]]:
    """Scan all .jsonl files in attempts_dir and extract (X_texts, y_labels)."""
    texts: List[str] = []
    labels: List[int] = []

    if not attempts_dir.exists():
        logger.warning("Attempts directory '%s' does not exist. Using synthetic training set.", attempts_dir)
        for text, label in SYNTHETIC_DATA:
            texts.append(text)
            labels.append(label)
        return texts, labels

    jsonl_files = sorted(attempts_dir.glob("*.jsonl"))
    logger.info("Found %d attempt files in '%s'. Extracting records...", len(jsonl_files), attempts_dir)

    for jsonl_file in jsonl_files:
        file_texts, file_labels = _process_jsonl_file(jsonl_file)
        texts.extend(file_texts)
        labels.extend(file_labels)

    logger.info("Extracted %d valid attempt samples (%d accepted, %d rejected).",
                len(texts), sum(labels), len(labels) - sum(labels))

    if len(texts) < 4:
        logger.warning("Extracted sample count (%d) is below minimum (4). Appending synthetic samples.", len(texts))
        for text, label in SYNTHETIC_DATA:
            texts.append(text)
            labels.append(label)

    return texts, labels


def _save_artifact(obj: Any, target_path: Path) -> None:
    """Save model object using joblib or pickle."""
    if joblib is not None:
        joblib.dump(obj, target_path)
    else:
        with target_path.open("wb") as f:
            pickle.dump(obj, f)


def _train_stage_b_vectorizer(texts: List[str], output_dir: Path) -> Tuple[Any, np.ndarray]:
    """Train and persist Stage B TF-IDF vectorizer."""
    if SklearnTfidfVectorizer is not None:
        logger.info("Fitting scikit-learn TF-IDF Vectorizer (char_wb 3-5 n-grams)...")
        vectorizer = SklearnTfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
    else:
        logger.info("Fitting FallbackCharTfidfVectorizer...")
        vectorizer = FallbackCharTfidfVectorizer(ngram_range=(3, 5), max_features=1000)

    X = vectorizer.fit_transform(texts)
    vec_path = output_dir / "vectorizer.joblib"
    _save_artifact(vectorizer, vec_path)
    logger.info("Saved TF-IDF vectorizer to '%s'.", vec_path)
    return vectorizer, X


def _train_stage_b_classifier(X: np.ndarray, y: np.ndarray, output_dir: Path) -> Any:
    """Train and persist Stage B classifier (XGBoost / RandomForest / Fallback)."""
    if XGBClassifier is not None:
        logger.info("Fitting XGBoost Classifier...")
        classifier: Any = XGBClassifier(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
        )
    else:
        logger.info("Fitting FallbackClassifier...")
        classifier = FallbackClassifier()

    classifier.fit(X, y)
    xgb_path = output_dir / "xgb.joblib"
    _save_artifact(classifier, xgb_path)
    logger.info("Saved Stage B classifier to '%s'.", xgb_path)
    return classifier


def _train_stage_c_setfit(texts: List[str], labels: List[int], output_dir: Path, train_setfit: bool) -> None:
    """Train and persist Stage C SetFit transformer model if requested."""
    setfit_dir = output_dir / "setfit_model"
    if train_setfit and SetFitModel is not None:
        logger.info("Fitting SetFit Transformer Model...")
        try:
            setfit_data = Dataset.from_dict({"text": texts, "label": labels})
            setfit_model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            trainer = Trainer(
                model=setfit_model,
                train_dataset=setfit_data,
                args=TrainingArguments(
                    batch_size=8,
                    num_epochs=1,
                    num_iterations=5,
                    learning_rate=2e-5,
                ),
            )
            trainer.train()
            setfit_model.save_pretrained(str(setfit_dir))
            logger.info("Saved SetFit model to '%s'.", setfit_dir)
        except Exception as e:
            logger.warning("SetFit model training failed: %s", e)
    else:
        if train_setfit:
            logger.warning("SetFit package is not installed. Skipping Stage C training.")
        else:
            logger.info("Stage C (SetFit) training disabled via --no-setfit flag.")


def train_pre_filter(
    attempts_dir: Path,
    output_dir: Path,
    train_setfit: bool = False,
) -> bool:
    """Train pre-filter models (Stage B XGBoost & Stage C SetFit) and persist artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    texts, labels = extract_dataset_from_attempts(attempts_dir)
    if not texts:
        logger.error("No valid attempt data available for training.")
        return False

    y = np.array(labels)

    # Stage B
    _, X = _train_stage_b_vectorizer(texts, output_dir)
    _train_stage_b_classifier(X, y, output_dir)

    # Stage C
    _train_stage_c_setfit(texts, labels, output_dir, train_setfit)

    return True


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
