from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import os
import re
import logging
from typing import Optional, Any

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


class BarredPreFilter:
    """3-Stage Layered Acceptance Pre-Filter.

    Stage A (Heuristics): Sub-millisecond regex/syntax rules.
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
    ):
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
        if joblib is not None and os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                logger.warning("Failed to load joblib artifact from '%s': %e", path, e)
        return None

    def _load_setfit(self, path: str) -> Any:
        if SetFitModel is not None and os.path.exists(path):
            try:
                return SetFitModel.from_pretrained(path)
            except Exception as e:
                logger.warning("Failed to load SetFit model from '%s': %e", path, e)
        return None

    def heuristic_score(self, predicate: str, input_block: str) -> Optional[float]:
        """Stage A: Deterministic rules (<0.1ms).

        Returns 0.01 for clear negatives, 0.99 for clear positives, or None if ambiguous.
        """
        pred_text = predicate.strip() if predicate else ""
        if any(rule.search(pred_text) for rule in NEGATIVE_RULES):
            return 0.01

        # Check for minimum valid code tokens in input block if present
        if input_block and not re.search(r"\b(def|class|if|return|void|int|char|struct|for|while|import|include)\b", input_block, re.I):
            return 0.01

        if any(rule.search(pred_text) for rule in POSITIVE_RULES):
            return 0.99

        return None

    def predict(self, predicate: str, input_block: str = "") -> PreFilterDecision:
        """Run 3-Stage Cascade evaluation on candidate seed."""
        attributes = {
            "predicate_len": len(predicate) if predicate else 0,
            "input_block_len": len(input_block) if input_block else 0,
        }

        with trace_span("pre_filter_evaluation", stage="pre_filter", attributes=attributes) as span:
            start_time = perf_counter()
            decision = self._run_cascade(predicate, input_block, start_time)

            span.attributes["pre_filter.accept"] = decision.accept
            span.attributes["pre_filter.probability"] = decision.probability
            span.attributes["pre_filter.decision_stage"] = decision.stage
            span.attributes["pre_filter.elapsed_ms"] = decision.elapsed_ms

            return decision

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

        combined_text = f"Predicate: {predicate} | Code: {input_block[:300]}"

        # Stage B: TF-IDF + XGBoost (~1ms)
        if self.vectorizer is not None and self.xgb is not None:
            try:
                features = self.vectorizer.transform([combined_text])
                probs = self.xgb.predict_proba(features)
                xgb_prob = float(probs[0][1])

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
            except Exception as e:
                logger.warning("Stage B prediction failed: %s", e)

        # Stage C: SetFit Transformer (~10ms)
        if self.setfit is not None:
            try:
                setfit_probs = self.setfit.predict_proba([combined_text])
                setfit_prob = float(setfit_probs[0][1])
                elapsed = round((perf_counter() - start_time) * 1000.0, 3)
                return PreFilterDecision(
                    accept=setfit_prob >= self.setfit_threshold,
                    probability=setfit_prob,
                    stage="setfit",
                    elapsed_ms=elapsed,
                )
            except Exception as e:
                logger.warning("Stage C prediction failed: %s", e)

        # Fallback: Default pass with explicit warning logging
        logger.warning("Pre-filter models unavailable or passed through. Executing default pass fallback.")
        elapsed = round((perf_counter() - start_time) * 1000.0, 3)
        return PreFilterDecision(
            accept=True,
            probability=1.0,
            stage="default_pass",
            elapsed_ms=elapsed,
        )
