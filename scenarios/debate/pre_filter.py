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
import os
import re
import logging
from typing import Optional, Union, Any

import scenarios.debate._thread_limits  # noqa: F401 (Enforce OpenMP thread limits on import)

from agentbeats.tracing import trace_span

logger = logging.getLogger("pre_filter")

# Attempt optional imports for Stage B / C models
try:
    import joblib
except ImportError:
    joblib = None

try:
    from setfit import SetFitModel
except ImportError:
    SetFitModel = None


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
        r"\b(vulnerable|buffer overflow|integer overflow|use after free|out of bounds|memory corruption|race condition|denial of service|injection)\b",
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

        self.vectorizer_path = vectorizer_path
        self.xgb_path = xgb_path
        self.setfit_dir = setfit_dir
        self.xgb_high_threshold = xgb_high_threshold
        self.xgb_low_threshold = xgb_low_threshold
        self.setfit_threshold = setfit_threshold

        self.vectorizer: Any = self._load_joblib(vectorizer_path)
        self.xgb: Any = self._load_joblib(xgb_path)
        self.setfit: Any = self._load_setfit(setfit_dir)

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

    def predict(self, predicate: str, input_block: Optional[str] = None) -> PreFilterDecision:
        """Run 3-Stage Cascade evaluation on candidate seed."""
        input_code = input_block if input_block is not None else ""
        attributes = {
            "predicate_len": len(predicate) if predicate else 0,
            "input_block_len": len(input_code),
        }

        with trace_span("pre_filter_evaluation", stage="pre_filter", attributes=attributes) as span:
            start_time = perf_counter()
            decision = self._run_cascade(predicate, input_code, start_time)

            span.attributes["pre_filter.accept"] = decision.accept
            span.attributes["pre_filter.probability"] = decision.probability
            span.attributes["pre_filter.decision_stage"] = decision.stage
            span.attributes["pre_filter.elapsed_ms"] = decision.elapsed_ms

            return decision

    def _get_combined_text(self, predicate: str, input_block: str) -> str:
        if predicate.startswith("Predicate: ") and " | Code: " in predicate:
            return predicate
        snippet = input_block[:1000] if input_block else ""
        return f"Predicate: {predicate} | Code: {snippet}"

    def _eval_stage_b(self, combined_text: str) -> Optional[float]:
        if self.vectorizer is None or self.xgb is None:
            return None
        try:
            features = self.vectorizer.transform([combined_text])
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

    def _run_cascade(self, predicate: str, input_block: str, start_time: float) -> PreFilterDecision:
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
