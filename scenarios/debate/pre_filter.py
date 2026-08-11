"""Layered High-Precision Pre-Filter Module.

This module implements a 3-Stage Layered Acceptance Pre-Filter (`BarredPreFilter`)
to intercept unviable candidate seeds and doomed refinement loops before making
expensive LLM API calls.

Public Schema:
    PreFilterDecision:
        accept (bool): Whether the candidate seed is accepted (True) or rejected (False).
        probability (float): Predictor confidence/probability score (0.0 <= p <= 1.0).
        stage (str): Decision stage vocabulary ("heuristic" | "xgboost" | "setfit" | "default_pass").
        elapsed_ms (float): Measured execution latency in milliseconds.

Fallback Contract:
    When model binaries (XGBoost / SetFit) are missing or fail to load, the filter
    logs an explicit warning and defaults gracefully to `default_pass` (accept=True,
    probability=1.0) so existing batch workflows are never interrupted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path before importing scenarios packages
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import scenarios.debate._thread_limits  # noqa: F401 (Enforce OpenMP thread limits on import)

from agentbeats.tracing import trace_span

logger = logging.getLogger("pre_filter")

# Attempt optional imports for Stage B / C models
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

try:
    from setfit import SetFitModel
except ImportError:
    SetFitModel = None

CODE_DELIMITER = " | Code: "
PREDICATE_PREFIX = "Predicate: "
CVE_REGEX = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def _extract_scenario_id(record: dict, predicate: str) -> str:
    """Resolve scenario identifier from record/seed metadata or generate deterministic predicate hash fallback."""
    seed_val = record.get("seed") if isinstance(record, dict) else {}
    seed_dict = seed_val if isinstance(seed_val, dict) else {}
    cve_id = (record.get("cve_id") if isinstance(record, dict) else None) or seed_dict.get("cve_id")
    if cve_id:
        normalized_cve = str(cve_id).strip().upper()
        if normalized_cve:
            return normalized_cve
    cve_match = CVE_REGEX.search(predicate)
    if cve_match:
        return cve_match.group(0).upper()
    return f"HASH-{hashlib.sha256(predicate.encode('utf-8')).hexdigest()[:10]}"


def _extract_code_snippet(record: dict, judge_dict: dict) -> str:
    """Extract code anchors or fall back to raw input_block."""
    anchors = record.get("anchors_normalized") if isinstance(record, dict) else None
    if not anchors:
        anchors = judge_dict.get("anchors", []) if isinstance(judge_dict, dict) else []
    if isinstance(anchors, list) and anchors:
        return " ".join(str(a) for a in anchors)
    if anchors and not isinstance(anchors, list):
        return str(anchors)
    return str(record.get("input_block", "")) if isinstance(record, dict) else ""


def _parse_attempt_record(record: dict) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Extract (combined_text, label, scenario_id) from a single attempt record dictionary."""
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
    scenario_id = _extract_scenario_id(record, predicate)
    code_snippet = _extract_code_snippet(record, judge_dict)

    combined_text = f"{PREDICATE_PREFIX}{predicate}{CODE_DELIMITER}{code_snippet[:1000]}"
    return combined_text, label, scenario_id


def _fallback_bucket_partitioning(
    texts: List[str],
    labels: List[int],
    scenario_ids: List[str],
    effective_n_splits: int,
) -> List[Dict[str, Any]]:
    """Fallback scenario bucket partitioning when StratifiedGroupKFold is unavailable."""
    if not (len(texts) == len(labels) == len(scenario_ids)):
        raise ValueError("Input lists to fallback partitioning must have equal lengths.")

    grouped: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for t, l, s in zip(texts, labels, scenario_ids):
        grouped[s].append((t, l))

    scenario_acc_rates = []
    for s_id, items in grouped.items():
        tot = len(items)
        acc = sum(l for _, l in items)
        rate = acc / tot if tot > 0 else 0.0
        scenario_acc_rates.append((rate, tot, s_id))

    scenario_acc_rates.sort(reverse=True)
    fold_buckets: List[List[str]] = [[] for _ in range(effective_n_splits)]
    for idx, (_, _, s_id) in enumerate(scenario_acc_rates):
        fold_buckets[idx % effective_n_splits].append(s_id)

    folds = []
    texts_arr = np.array(texts, dtype=object)
    labels_arr = np.array(labels, dtype=int)
    scenarios_arr = np.array(scenario_ids, dtype=object)

    for fold_idx in range(1, effective_n_splits + 1):
        test_scenarios_set = set(fold_buckets[fold_idx - 1])
        test_mask = np.array([s in test_scenarios_set for s in scenario_ids])
        train_mask = ~test_mask

        train_idx = np.nonzero(train_mask)[0]
        test_idx = np.nonzero(test_mask)[0]

        folds.append({
            "fold": fold_idx,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_texts": list(texts_arr[train_idx]),
            "train_labels": list(labels_arr[train_idx]),
            "test_texts": list(texts_arr[test_idx]),
            "test_labels": list(labels_arr[test_idx]),
            "test_scenario_ids": sorted(test_scenarios_set),
            "train_scenario_ids": sorted(set(scenarios_arr[train_mask])),
        })

    return folds


def partition_dataset_by_scenario_stratified(
    texts: List[str],
    labels: List[int],
    scenario_ids: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Partition dataset into Stratified Scenario-Grouped folds (zero scenario-predicate leakage across splits)."""
    if not (len(texts) == len(labels) == len(scenario_ids)):
        raise ValueError(
            f"Input lists must have equal lengths: len(texts)={len(texts)}, "
            f"len(labels)={len(labels)}, len(scenario_ids)={len(scenario_ids)}."
        )

    unique_scenarios = len(set(scenario_ids))
    if unique_scenarios < 2:
        raise ValueError(
            f"Scenario-grouped cross-validation requires at least 2 unique scenario IDs, but found {unique_scenarios}."
        )

    effective_n_splits = max(2, min(n_splits, len(texts), unique_scenarios))
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=effective_n_splits, shuffle=True, random_state=seed)
        x_arr = np.array(texts, dtype=object)
        y_arr = np.array(labels, dtype=int)
        groups_arr = np.array(scenario_ids, dtype=object)

        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(x_arr, y_arr, groups=groups_arr), 1):
            train_scenarios = set(groups_arr[train_idx])
            test_scenarios = set(groups_arr[test_idx])
            folds.append({
                "fold": fold_idx,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_texts": list(x_arr[train_idx]),
                "train_labels": list(y_arr[train_idx]),
                "test_texts": list(x_arr[test_idx]),
                "test_labels": list(y_arr[test_idx]),
                "test_scenario_ids": sorted(test_scenarios),
                "train_scenario_ids": sorted(train_scenarios),
            })
        return folds
    except (ImportError, ValueError):
        logger.warning("scikit-learn StratifiedGroupKFold unavailable or failed. Using fallback scenario bucket partitioning.")
        return _fallback_bucket_partitioning(texts, labels, scenario_ids, effective_n_splits)

SAFETY_KEYWORDS = [
    "free", "malloc", "memcpy", "memset", "sizeof", "null", "nullptr",
    "overflow", "use-after-free", "race condition", "out-of-bounds",
    "script inclusion", "leakage", "unsanitized", "disclosure", "privilege",
    "bypass", "pointer", "buffer", "integer", "format string", "injection",
    "dereference", "vulnerable", "bounds", "uaf", "double free", "type confusion"
]

CONTROL_KEYWORDS = [
    "if", "while", "for", "switch", "goto", "return", "struct", "typedef", "def", "class"
]

SYMBOLS = ["*", "&", "->", "[", "]", "(", ")", ";", "=", "+", "-", "<", ">", "/"]


def extract_domain_features(text: str) -> np.ndarray:
    """Extract domain-specific structural, vulnerability, and syntax numerical features."""
    if CODE_DELIMITER in text:
        parts = text.split(CODE_DELIMITER, 1)
        pred = parts[0].replace(PREDICATE_PREFIX, "", 1)
        code = parts[1]
    else:
        pred = text
        code = ""

    code_lower = code.lower()
    pred_lower = pred.lower()

    lines = code.split("\n") if code else []
    line_count = float(len(lines))
    char_count = float(len(code))
    avg_line_len = float(char_count / max(line_count, 1.0))
    max_indent = float(max([len(line) - len(line.lstrip()) for line in lines] or [0]) / 4.0)

    control_counts = [float(code_lower.split().count(kw)) for kw in CONTROL_KEYWORDS]

    full_text_lower = f"{pred_lower} {code_lower}"
    safety_counts = [float(full_text_lower.count(kw)) for kw in SAFETY_KEYWORDS]

    symbol_counts = [float(code.count(sym)) for sym in SYMBOLS]

    pred_words = pred.split()
    pred_word_count = float(len(pred_words))
    pred_char_count = float(len(pred))
    pred_upper_ratio = float(sum(1 for c in pred if c.isupper()) / max(len(pred), 1.0))
    pred_digit_ratio = float(sum(1 for c in pred if c.isdigit()) / max(len(pred), 1.0))

    feat_list = [
        line_count,
        char_count,
        avg_line_len,
        max_indent,
        *control_counts,
        *safety_counts,
        *symbol_counts,
        pred_word_count,
        pred_char_count,
        pred_upper_ratio,
        pred_digit_ratio,
    ]
    return np.array(feat_list, dtype=np.float32)


def extract_domain_features_batch(texts: list[str]) -> np.ndarray:
    """Extract domain features for a batch of text documents."""
    if not texts:
        return np.zeros((0, 60), dtype=np.float32)
    rows = [extract_domain_features(t) for t in texts]
    return np.vstack(rows)


class FallbackStandardScaler:
    """Numpy Standard Scaler for domain numerical features with fallback if sklearn is absent."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> FallbackStandardScaler:
        if X.shape[0] == 0:
            return self
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[np.isclose(std, 0.0)] = 1.0
        self.scale_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0 or self.mean_ is None or self.scale_ is None:
            return X
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _combine_features(x_tfidf: Any, x_domain_scaled: np.ndarray) -> Any:
    """Stack TF-IDF sparse matrix with scaled dense domain features."""
    try:
        from scipy.sparse import issparse, hstack, csr_matrix
        if issparse(x_tfidf):
            x_domain_sparse = csr_matrix(x_domain_scaled)
            return hstack([x_tfidf, x_domain_sparse]).tocsr()
    except ImportError:
        pass
    return np.hstack([x_tfidf, x_domain_scaled])


@dataclass(frozen=True)
class PreFilterDecision:
    """Public decision object returned by BarredPreFilter.predict().

    Attributes:
        accept: Whether the candidate seed is accepted (True) or rejected (False).
        probability: Probability score in [0.0, 1.0].
        stage: One of 'heuristic', 'xgboost', 'setfit', or 'default_pass'.
        elapsed_ms: Execution duration in milliseconds.
    """
    accept: bool
    probability: float
    stage: str
    elapsed_ms: float


# Stage A Heuristic Rules
NEGATIVE_RULES = (
    re.compile(r"^\s*$", re.I),  # Empty or whitespace only
    re.compile(r"^.{0,14}$", re.I),  # Fewer than 15 characters
)

POSITIVE_RULES = (
    re.compile(
        r"\b(buffer overflow|integer overflow|use after free|out of bounds|memory corruption|race condition|denial of service|injection)\b",
        re.I,
    ),
)

# Multi-language code token regex (SonarQube S5843 complexity <= 20)
CODE_TOKEN_REGEX = re.compile(
    r"\b(def|class|function|if|return|void|int|char|struct|for|while|import|include|fn|const|curl)\b",
    re.I,
)


class BarredPreFilter:
    """3-Stage Layered Acceptance Pre-Filter.

    Stage A (Heuristics): Sub-millisecond regex/syntax rules (<0.1ms).
    Stage B (XGBoost + TF-IDF): Fast sparse feature classifier (~1ms).
    Stage C (SetFit Transformer): Dense semantic model (~10ms) for ambiguous inputs.
    """

    def __init__(
        self,
        vectorizer_path: str = "artifacts/models/vectorizer.joblib",
        xgb_path: str = "artifacts/models/xgb.joblib",
        setfit_dir: str = "artifacts/models/setfit_model",
        xgb_high_threshold: float = 0.995,
        xgb_low_threshold: float = 0.05,
        setfit_threshold: float = 0.65,
        model_dir: str | os.PathLike[str] | None = None,
    ):
        if model_dir:
            from pathlib import Path
            base_dir = Path(model_dir)
            vectorizer_path = str(base_dir / "vectorizer.joblib")
            xgb_path = str(base_dir / "xgb.joblib")
            setfit_dir = str(base_dir / "setfit_model")

        from pathlib import Path
        model_dir_path = Path(vectorizer_path).parent
        scaler_path = str(model_dir_path / "domain_scaler.joblib")

        self.vectorizer_path = vectorizer_path
        self.xgb_path = xgb_path
        self.setfit_dir = setfit_dir
        self.xgb_high_threshold = xgb_high_threshold
        self.xgb_low_threshold = xgb_low_threshold
        self.setfit_threshold = setfit_threshold

        manifest_valid = self._verify_manifest(model_dir_path)

        if manifest_valid:
            self.vectorizer: Any = self._load_joblib(vectorizer_path)
            self.domain_scaler: Any = self._load_joblib(scaler_path)
            self.xgb: Any = self._load_joblib(xgb_path)
            self.setfit: Any = self._load_setfit(setfit_dir)
        else:
            logger.error("Skipping Stage B/C model weights due to manifest verification failure.")
            self.vectorizer = None
            self.domain_scaler = None
            self.xgb = None
            self.setfit = None

        if not self.vectorizer or not self.xgb:
            logger.warning(
                "Pre-filter Stage B (XGBoost/TF-IDF) weights missing at '%s' / '%s'. "
                "Stage B will be skipped.",
                vectorizer_path,
                xgb_path,
            )

        if not self.setfit:
            logger.warning(
                "Pre-filter Stage C (SetFit) model directory missing at '%s'. "
                "Stage C will be skipped.",
                setfit_dir,
            )

    def _validate_single_artifact(self, model_dir: Path, resolved_model_dir: Path, rel_path: str, info: Any) -> bool:
        """Validate a single artifact entry against expected hash and file presence."""
        import hashlib
        if not isinstance(info, dict):
            logger.error("Manifest artifact entry '%s' is invalid.", rel_path)
            return False

        expected_sha = info.get("sha256")
        if not expected_sha or not isinstance(expected_sha, str):
            logger.error("Manifest artifact entry '%s' missing valid sha256 checksum.", rel_path)
            return False

        filepath = (model_dir / rel_path).resolve()
        try:
            filepath.relative_to(resolved_model_dir)
        except ValueError:
            logger.error("Manifest artifact entry '%s' escapes model directory. Rejected.", rel_path)
            return False

        if not filepath.exists() or not filepath.is_file():
            logger.error("Declared manifest artifact file '%s' is missing on disk.", filepath)
            return False

        hasher = hashlib.sha256()
        with filepath.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_sha = hasher.hexdigest()
        if actual_sha != expected_sha:
            logger.error(
                "Manifest checksum mismatch for '%s': expected %s, got %s. Artifact corrupted.",
                rel_path,
                expected_sha,
                actual_sha,
            )
            return False
        return True

    def _verify_manifest(self, model_dir: Path) -> bool:
        """Validate model artifact checksums against model_manifest.json if present.

        Fails closed (returns False) for any present but malformed, incomplete, or corrupted manifest.
        Returns True only when model_manifest.json is absent (legacy path).
        """
        manifest_path = model_dir / "model_manifest.json"
        if not manifest_path.exists():
            return True
        try:
            import json

            resolved_model_dir = model_dir.resolve()
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            if not isinstance(manifest_data, dict):
                logger.error("Manifest at '%s' is not a valid JSON object.", manifest_path)
                return False

            artifacts = manifest_data.get("artifacts")
            if not isinstance(artifacts, dict) or not artifacts:
                logger.error("Manifest at '%s' missing valid non-empty 'artifacts' mapping.", manifest_path)
                return False

            return all(
                self._validate_single_artifact(model_dir, resolved_model_dir, rel_path, info)
                for rel_path, info in artifacts.items()
            )
        except Exception as err:
            logger.exception("Manifest validation encountered exception for '%s': %s", manifest_path, err)
            return False



    def _load_joblib(self, path: str) -> Any:
        if os.path.exists(path):
            e_joblib: Optional[Exception] = None
            if joblib is not None:
                try:
                    return joblib.load(path)
                except Exception as e:
                    e_joblib = e

            try:
                import pickle
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e_pickle:
                if e_joblib is not None:
                    logger.warning(
                        "Failed to load artifact from '%s' via joblib (%s) and pickle (%s)",
                        path,
                        e_joblib,
                        e_pickle,
                    )
                else:
                    logger.warning("Failed to load artifact from '%s': %s", path, e_pickle)
        return None

    def _load_setfit(self, path: str) -> Any:
        if SetFitModel is not None and os.path.exists(path):
            try:
                return SetFitModel.from_pretrained(path)
            except Exception as e:
                logger.warning("Failed to load SetFit model from '%s': %s", path, e)
        return None

    def heuristic_score(self, predicate: str, input_block: Optional[str] = None) -> Optional[float]:
        """Stage A: Deterministic rules (<0.1ms).

        Returns 0.01 for clear negatives, 0.99 for clear positives, or None if ambiguous.
        """
        pred_text = predicate.strip() if predicate else ""
        if any(rule.search(pred_text) for rule in NEGATIVE_RULES):
            return 0.01

        # Check for minimum valid code tokens in input block if present
        input_code = input_block.strip() if input_block else ""
        if input_code and not CODE_TOKEN_REGEX.search(input_code):
            return 0.01

        if any(rule.search(pred_text) for rule in POSITIVE_RULES):
            return 0.99

        return None

    def predict(
        self,
        predicate: str,
        input_block: Optional[str] = None,
        attempt_number: Optional[int] = None,
    ) -> PreFilterDecision:
        """Run 3-Stage Cascade evaluation on candidate seed / retry code."""
        input_code = input_block if input_block is not None else ""
        normalized_attempt = attempt_number or 1
        attributes = {
            "predicate_len": len(predicate) if predicate else 0,
            "input_block_len": len(input_code),
            "attempt_number": normalized_attempt,
        }

        with trace_span("pre_filter_evaluation", stage="pre_filter", attributes=attributes) as span:
            start_time = perf_counter()
            decision = self._run_cascade(predicate, input_code, start_time, attempt_number=normalized_attempt)

            span.attributes["pre_filter.accept"] = decision.accept
            span.attributes["pre_filter.probability"] = decision.probability
            span.attributes["pre_filter.decision_stage"] = decision.stage
            span.attributes["pre_filter.elapsed_ms"] = decision.elapsed_ms

            return decision

    def _get_combined_text(self, predicate: str, input_block: str) -> str:
        if predicate.startswith(PREDICATE_PREFIX) and CODE_DELIMITER in predicate:
            parts = predicate.split(CODE_DELIMITER, 1)
            pred_part = parts[0]
            code_part = parts[1][:1000] if len(parts) > 1 else ""
            return f"{pred_part}{CODE_DELIMITER}{code_part}"
        snippet = input_block[:1000] if input_block else ""
        return f"{PREDICATE_PREFIX}{predicate}{CODE_DELIMITER}{snippet}"

    def _eval_stage_b(self, combined_text: str) -> Optional[float]:
        if self.vectorizer is None or self.xgb is None:
            return None
        try:
            tfidf_feat = self.vectorizer.transform([combined_text])
            if self.domain_scaler is not None:
                domain_feat = extract_domain_features_batch([combined_text])
                scaled_domain_feat = self.domain_scaler.transform(domain_feat)
                features = _combine_features(tfidf_feat, scaled_domain_feat)
            else:
                features = tfidf_feat
            probs = self.xgb.predict_proba(features)
            return float(probs[0][1])
        except Exception as e:
            logger.warning("Stage B prediction failed: %s", e)
            return None

    def _eval_stage_c(self, combined_text: str) -> Optional[float]:
        if self.setfit is None:
            return None
        try:
            setfit_probs = self.setfit.predict_proba([combined_text])
            return float(setfit_probs[0][1])
        except Exception as e:
            logger.warning("Stage C prediction failed: %s", e)
            return None

    def _run_cascade(
        self,
        predicate: str,
        input_block: str,
        start_time: float,
        attempt_number: Optional[int] = None,
    ) -> PreFilterDecision:
        # Stage A: Heuristics (<0.1ms)
        rule_prob = self.heuristic_score(predicate, input_block)
        if rule_prob is not None:
            elapsed = round((perf_counter() - start_time) * 1000.0, 3)
            return PreFilterDecision(
                accept=rule_prob >= 0.90,
                probability=rule_prob,
                stage="heuristic",
                elapsed_ms=elapsed,
            )

        # Option C: ML Stage B (XGBoost) & Stage C (SetFit) evaluate candidate code on retry attempts.
        # Initial seed prompts (attempt_number == 1) pass Stage B/C to allow initial LLM code synthesis.
        if attempt_number == 1:
            elapsed = round((perf_counter() - start_time) * 1000.0, 3)
            return PreFilterDecision(
                accept=True,
                probability=1.0,
                stage="initial_seed_pass",
                elapsed_ms=elapsed,
            )

        combined_text = self._get_combined_text(predicate, input_block)

        # Stage B: TF-IDF + XGBoost (~1ms)
        xgb_prob = self._eval_stage_b(combined_text)
        if xgb_prob is not None:
            if xgb_prob >= self.xgb_high_threshold:
                elapsed = round((perf_counter() - start_time) * 1000.0, 3)
                return PreFilterDecision(
                    accept=True,
                    probability=xgb_prob,
                    stage="xgboost",
                    elapsed_ms=elapsed,
                )
            if xgb_prob <= self.xgb_low_threshold:
                elapsed = round((perf_counter() - start_time) * 1000.0, 3)
                return PreFilterDecision(
                    accept=False,
                    probability=xgb_prob,
                    stage="xgboost",
                    elapsed_ms=elapsed,
                )

        # Stage C: SetFit Transformer (~10ms)
        setfit_prob = self._eval_stage_c(combined_text)
        if setfit_prob is not None:
            elapsed = round((perf_counter() - start_time) * 1000.0, 3)
            return PreFilterDecision(
                accept=setfit_prob >= self.setfit_threshold,
                probability=setfit_prob,
                stage="setfit",
                elapsed_ms=elapsed,
            )

        # Fallback: Default pass with explicit warning logging
        logger.warning("Pre-filter models unavailable or passed through. Executing default pass fallback.")
        elapsed = round((perf_counter() - start_time) * 1000.0, 3)
        return PreFilterDecision(
            accept=True,
            probability=1.0,
            stage="default_pass",
            elapsed_ms=elapsed,
        )
