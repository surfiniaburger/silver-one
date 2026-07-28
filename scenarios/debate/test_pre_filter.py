from unittest.mock import MagicMock
from time import perf_counter
import logging

from scenarios.debate.pre_filter import BarredPreFilter, PreFilterDecision


def test_pre_filter_decision_schema():
    decision = PreFilterDecision(accept=True, probability=0.95, stage="xgboost", elapsed_ms=1.2)
    assert hasattr(decision, "accept")
    assert hasattr(decision, "probability")
    assert hasattr(decision, "stage")
    assert hasattr(decision, "elapsed_ms")
    assert decision.stage in {"heuristic", "xgboost", "setfit", "default_pass"}


def test_stage_a_heuristics_negative_empty():
    pre_filter = BarredPreFilter(
        vectorizer_path="artifacts/models/non_existent.joblib",
        xgb_path="artifacts/models/non_existent.joblib",
        setfit_dir="artifacts/models/non_existent",
    )
    decision = pre_filter.predict("")
    assert decision.stage == "heuristic"
    assert decision.accept is False
    assert decision.probability == 0.01


def test_stage_a_heuristics_negative_short():
    pre_filter = BarredPreFilter(
        vectorizer_path="artifacts/models/non_existent.joblib",
        xgb_path="artifacts/models/non_existent.joblib",
        setfit_dir="artifacts/models/non_existent",
    )
    decision = pre_filter.predict("Too short")
    assert decision.stage == "heuristic"
    assert decision.accept is False
    assert decision.probability == 0.01


def test_stage_a_heuristics_ungrounded_code():
    pre_filter = BarredPreFilter(
        vectorizer_path="artifacts/models/non_existent.joblib",
        xgb_path="artifacts/models/non_existent.joblib",
        setfit_dir="artifacts/models/non_existent",
    )
    decision = pre_filter.predict("Analyzes vulnerability", input_block="just text without code keywords")
    assert decision.stage == "heuristic"
    assert decision.accept is False


def test_stage_a_heuristics_positive():
    pre_filter = BarredPreFilter(
        vectorizer_path="artifacts/models/non_existent.joblib",
        xgb_path="artifacts/models/non_existent.joblib",
        setfit_dir="artifacts/models/non_existent",
    )
    decision = pre_filter.predict("The code is vulnerable to a buffer overflow in parse_string", input_block="void parse_string(char *s) { }")
    assert decision.stage == "heuristic"
    assert decision.accept is True
    assert decision.probability == 0.99


def test_default_pass_fallback_when_models_missing(caplog):
    pre_filter = BarredPreFilter(
        vectorizer_path="artifacts/models/non_existent_vec.joblib",
        xgb_path="artifacts/models/non_existent_xgb.joblib",
        setfit_dir="artifacts/models/non_existent_setfit",
    )
    with caplog.at_level(logging.WARNING):
        decision = pre_filter.predict("An ambiguous test predicate for processing", input_block="def process_data(data): return data")

    assert decision.stage == "default_pass"
    assert decision.accept is True
    assert decision.probability == 1.0
    assert "Pre-filter models unavailable" in caplog.text


def test_stage_b_xgboost_accept_and_reject_thresholds():
    pre_filter = BarredPreFilter(
        vectorizer_path="artifacts/models/non_existent.joblib",
        xgb_path="artifacts/models/non_existent.joblib",
        setfit_dir="artifacts/models/non_existent",
    )
    pre_filter.vectorizer = MagicMock()
    pre_filter.vectorizer.transform.return_value = "mock_features"

    # High probability -> Accept via xgboost
    mock_xgb_high = MagicMock()
    mock_xgb_high.predict_proba.return_value = [[0.002, 0.998]]
    pre_filter.xgb = mock_xgb_high

    decision_accept = pre_filter.predict("An ambiguous test predicate for processing", input_block="def process_data(data): return data")
    assert decision_accept.stage == "xgboost"
    assert decision_accept.accept is True
    assert decision_accept.probability == 0.998

    # Low probability -> Reject via xgboost
    mock_xgb_low = MagicMock()
    mock_xgb_low.predict_proba.return_value = [[0.98, 0.02]]
    pre_filter.xgb = mock_xgb_low

    decision_reject = pre_filter.predict("An ambiguous test predicate for processing", input_block="def process_data(data): return data")
    assert decision_reject.stage == "xgboost"
    assert decision_reject.accept is False
    assert decision_reject.probability == 0.02


def test_stage_c_setfit_fallback_on_ambiguous_xgboost():
    pre_filter = BarredPreFilter(
        vectorizer_path="artifacts/models/non_existent.joblib",
        xgb_path="artifacts/models/non_existent.joblib",
        setfit_dir="artifacts/models/non_existent",
    )
    pre_filter.vectorizer = MagicMock()
    pre_filter.vectorizer.transform.return_value = "mock_features"

    # Stage B returns ambiguous 0.50 probability
    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = [[0.50, 0.50]]
    pre_filter.xgb = mock_xgb

    # Stage C SetFit handles the ambiguous input
    mock_setfit = MagicMock()
    mock_setfit.predict_proba.return_value = [[0.20, 0.80]]
    pre_filter.setfit = mock_setfit

    decision = pre_filter.predict("An ambiguous test predicate for processing", input_block="def process_data(data): return data")
    assert decision.stage == "setfit"
    assert decision.accept is True
    assert decision.probability == 0.80


def test_performance_latency_budget():
    pre_filter = BarredPreFilter(
        vectorizer_path="artifacts/models/non_existent.joblib",
        xgb_path="artifacts/models/non_existent.joblib",
        setfit_dir="artifacts/models/non_existent",
    )
    sample_predicate = "An ambiguous test predicate for performance benchmarking"
    sample_code = "def benchmark_fn(x, y):\n    return x + y"

    start_wall = perf_counter()
    decisions = [pre_filter.predict(sample_predicate, sample_code) for _ in range(100)]
    total_elapsed_ms = (perf_counter() - start_wall) * 1000.0
    avg_external_latency_ms = total_elapsed_ms / len(decisions)

    assert len(decisions) == 100
    # Average wall-clock latency per prediction including trace_span entry/exit must be <10ms
    assert avg_external_latency_ms < 10.0
