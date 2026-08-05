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

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional
import sys
import os
import re
import logging
from pathlib import Path

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

    def _verify_manifest(self, model_dir: Path) -> bool:
        """Validate model artifact checksums against model_manifest.json if present."""
        manifest_path = model_dir / "model_manifest.json"
        if not manifest_path.exists():
            return True
        try:
            import json
            import hashlib
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            artifacts = manifest_data.get("artifacts", {})
            for filename, info in artifacts.items():
                filepath = model_dir / filename
                if filepath.exists():
                    with filepath.open("rb") as f:
                        content = f.read()
                    actual_sha = hashlib.sha256(content).hexdigest()
                    expected_sha = info.get("sha256")
                    if expected_sha and actual_sha != expected_sha:
                        logger.error(
                            "Manifest checksum mismatch for '%s': expected %s, got %s. Artifact corrupted.",
                            filename,
                            expected_sha,
                            actual_sha,
                        )
                        return False
            return True
        except Exception as err:
            logger.warning("Manifest validation encountered exception for '%s': %s", manifest_path, err)
            return True


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
