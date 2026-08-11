"""
Unit tests for statistical hypothesis testing functions in debate_telemetry.py.
"""

import math
import pytest
from scripts.debate_telemetry import (
    _build_stage_breakdown,
    _coerce_finite_metric,
    _percentile,
    apply_holm_step_down,
    compute_ab_benchmark_comparison,
    compute_hodges_lehmann,
    compute_statistical_hypothesis_test,
    compute_t_interval,
)


def test_percentile_telemetry_fields():
    # Empty input
    assert _percentile([], 50.0) == 0.0
    assert _percentile([], 99.0) == 0.0

    # Single-sample input
    assert _percentile([100.0], 50.0) == 100.0
    assert _percentile([100.0], 99.0) == 100.0

    # Multi-sample input
    samples = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert math.isclose(_percentile(samples, 50.0), 55.0, abs_tol=1e-1)
    assert math.isclose(_percentile(samples, 95.0), 95.5, abs_tol=1e-1)

    # Stage breakdown schema verification with P50 and P99
    stage_durations = {"stage_1": [10.0, 20.0, 30.0, 40.0, 50.0]}
    breakdown = _build_stage_breakdown(stage_durations, {"stage_1": 100}, {"stage_1": 50}, {"stage_1": 5})
    assert breakdown["stage_1"]["p50_duration_ms"] == 30.0
    assert breakdown["stage_1"]["p95_duration_ms"] == 48.0
    assert breakdown["stage_1"]["p99_duration_ms"] == 49.6



def test_coerce_finite_metric():
    assert _coerce_finite_metric("0") == 0.0
    assert _coerce_finite_metric(0.0) == 0.0
    assert _coerce_finite_metric("42.5") == 42.5
    assert _coerce_finite_metric(True) is None
    assert _coerce_finite_metric(False) is None
    assert _coerce_finite_metric(None) is None
    assert _coerce_finite_metric("invalid") is None
    assert _coerce_finite_metric(float("nan")) is None
    assert _coerce_finite_metric(float("inf")) is None


def test_compute_t_interval_normal_deltas():
    deltas = [1.0, 2.0, 1.5, 2.5, 3.0]
    mean_val, ci_low, ci_high = compute_t_interval(deltas, alpha=0.05)
    assert math.isclose(mean_val, 2.0, abs_tol=1e-3)
    assert ci_low < mean_val < ci_high


def test_compute_hodges_lehmann():
    deltas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    hl_med, ci_low, ci_high = compute_hodges_lehmann(deltas, alpha=0.05)
    assert math.isclose(hl_med, 0.0, abs_tol=1e-2)
    assert ci_low <= hl_med <= ci_high


def test_apply_holm_step_down_sorting_and_thresholds():
    p_values = [0.001, 0.03, 0.04]
    results = apply_holm_step_down(p_values, alpha=0.05)
    assert len(results) == 3

    # Rank 1: p=0.001 < 0.05/3 (0.01667) -> True
    assert results[0]["is_significant"] is True
    # Rank 2: p=0.03 >= 0.05/2 (0.025) -> False
    assert results[1]["is_significant"] is False
    # Rank 3: p=0.04 -> False (stopped)
    assert results[2]["is_significant"] is False


def test_hypothesis_test_insufficient_sample_size():
    b_vals = [10.0, 12.0]
    c_vals = [8.0, 9.0]
    res = compute_statistical_hypothesis_test(b_vals, c_vals, metric_name="tokens", num_metrics=1)

    assert res["sample_size"] == 2
    assert res["test_type"] == "insufficient_sample_size"
    assert res["is_significant"] is False
    assert res["p_value"] == 1.0


def test_hypothesis_test_normal_paired_deltas():
    # Large consistent shift (normal deltas)
    b_vals = [100.0, 102.0, 101.0, 103.0, 100.0, 102.0, 104.0, 101.0, 103.0, 102.0]
    c_vals = [80.0, 81.0, 82.0, 80.0, 81.0, 82.0, 83.0, 81.0, 82.0, 81.0]

    res = compute_statistical_hypothesis_test(b_vals, c_vals, metric_name="tokens", alpha=0.05, num_metrics=1)

    assert res["sample_size"] == 10
    assert res["is_normal"] is True
    assert res["test_type"] == "paired_t_test"
    assert res["p_value"] < 0.001
    assert res["is_significant"] is True
    assert res["ci_95"][0] < res["mean_delta"] < res["ci_95"][1]


def test_ab_comparison_with_paired_seed_telemetry():
    baseline = {
        "run_id": "base-run",
        "summary": {"total_wall_clock_ms": 1000.0},
        "token_efficiency": {
            "total_tokens_total": 10000,
            "prompt_tokens_total": 8000,
            "tokens_per_accepted_row": 500.0,
        },
        "cache_performance": {"cache_hit_rate_pct": 50.0},
        "per_seed_metrics": {
            "seed_42": {"tokens_per_accepted_row": 500.0, "total_tokens": 10000, "total_wall_clock_ms": 1000.0},
            "seed_1337": {"tokens_per_accepted_row": 520.0, "total_tokens": 10400, "total_wall_clock_ms": 1020.0},
            "seed_2026": {"tokens_per_accepted_row": 510.0, "total_tokens": 10200, "total_wall_clock_ms": 1010.0},
            "seed_9999": {"tokens_per_accepted_row": 490.0, "total_tokens": 9800, "total_wall_clock_ms": 990.0},
        },
    }

    candidate = {
        "run_id": "cand-run",
        "summary": {"total_wall_clock_ms": 600.0},
        "token_efficiency": {
            "total_tokens_total": 6000,
            "prompt_tokens_total": 4800,
            "tokens_per_accepted_row": 300.0,
        },
        "cache_performance": {"cache_hit_rate_pct": 80.0},
        "per_seed_metrics": {
            "seed_42": {"tokens_per_accepted_row": 300.0, "total_tokens": 6000, "total_wall_clock_ms": 600.0},
            "seed_1337": {"tokens_per_accepted_row": 310.0, "total_tokens": 6200, "total_wall_clock_ms": 610.0},
            "seed_2026": {"tokens_per_accepted_row": 305.0, "total_tokens": 6100, "total_wall_clock_ms": 605.0},
            "seed_9999": {"tokens_per_accepted_row": 295.0, "total_tokens": 5900, "total_wall_clock_ms": 595.0},
        },
    }

    ab_res = compute_ab_benchmark_comparison(baseline, candidate)

    assert ab_res["baseline_run_id"] == "base-run"
    assert ab_res["candidate_run_id"] == "cand-run"

    stat_tests = ab_res.get("statistical_tests", [])
    assert len(stat_tests) == 3
    for st in stat_tests:
        assert st["sample_size"] == 4
        assert st["test_type"] in {"paired_t_test", "wilcoxon_signed_rank"}
        assert st["is_significant_holm"] is True


def test_hypothesis_test_zero_delta_identical():
    b_vals = [500.0, 500.0, 500.0, 500.0]
    c_vals = [500.0, 500.0, 500.0, 500.0]
    res = compute_statistical_hypothesis_test(b_vals, c_vals, metric_name="tokens", num_metrics=1)

    assert res["test_type"] == "zero_delta_identical"
    assert res["mean_delta"] == 0.0
    assert res["p_value"] == 1.0
    assert res["is_significant"] is False
    assert res["ci_95"] == [0.0, 0.0]


def test_ab_comparison_defensive_seed_metrics_handling():
    baseline = {
        "run_id": "base-run",
        "summary": {"total_wall_clock_ms": 1000.0},
        "token_efficiency": {"total_tokens_total": 1000, "prompt_tokens_total": 800, "tokens_per_accepted_row": 500.0},
        "cache_performance": {"cache_hit_rate_pct": 50.0},
        "per_seed_metrics": {
            "seed_1": None,  # Malformed non-dict
            "seed_2": {"tokens_per_accepted_row": "invalid_number", "total_tokens": None, "total_wall_clock_ms": 100.0},
        },
    }

    candidate = {
        "run_id": "cand-run",
        "summary": {"total_wall_clock_ms": 600.0},
        "token_efficiency": {"total_tokens_total": 600, "prompt_tokens_total": 480, "tokens_per_accepted_row": 300.0},
        "cache_performance": {"cache_hit_rate_pct": 80.0},
        "per_seed_metrics": {
            "seed_1": {"tokens_per_accepted_row": 300.0, "total_tokens": 600.0, "total_wall_clock_ms": 60.0},
            "seed_2": {"tokens_per_accepted_row": 300.0, "total_tokens": 600.0, "total_wall_clock_ms": 60.0},
        },
    }

    ab_res = compute_ab_benchmark_comparison(baseline, candidate)
    stat_tests = ab_res.get("statistical_tests", [])
    assert len(stat_tests) == 3
    for st in stat_tests:
        # Invalid/missing seed entries dropped -> sample_size = 0 or 1 (< 3)
        assert st["sample_size"] < 3
        assert st["test_type"] == "insufficient_sample_size"
        assert st["is_significant"] is False
        assert st["is_significant_holm"] is False

