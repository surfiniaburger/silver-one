"""Unit tests for swarm reliability metric schema, empty-set behavior, canonical identity, and statistical protocol.

Tests all calculations, edge cases, and invariants defined in docs/MULTIAGENT_VULNERABILITY_SWARM_HYPOTHESES.md.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import pytest

from scenarios.debate.offline_b_gate import check_anti_gaming_invariants
from scripts.debate_telemetry import (
    apply_holm_step_down,
    compute_hodges_lehmann,
    compute_statistical_hypothesis_test,
)


def compute_valid_accepts_per_1m_tokens(valid_accepts: int, total_tokens: int) -> float:
    """Computes valid_accepts_per_1m_tokens with safe zero-denominator handling."""
    if total_tokens <= 0 or valid_accepts <= 0:
        return 0.0
    return (valid_accepts / total_tokens) * 1_000_000


def compute_tokens_per_valid_accept(valid_accepts: int, total_tokens: int, token_budget: int = 10_000_000) -> float:
    """Computes tokens_per_valid_accept with infinite penalty on zero accepts."""
    if valid_accepts <= 0:
        return float("inf")
    return total_tokens / valid_accepts


def compute_relative_token_improvement(baseline_tokens_per_accept: float, candidate_tokens_per_accept: float) -> Optional[float]:
    """Computes relative token efficiency improvement (baseline - cand) / baseline."""
    if baseline_tokens_per_accept <= 0.0 or math.isinf(baseline_tokens_per_accept):
        return None
    if math.isinf(candidate_tokens_per_accept):
        return -1.0  # complete degradation
    return (baseline_tokens_per_accept - candidate_tokens_per_accept) / baseline_tokens_per_accept


def compute_canonical_identity_key(scenario_id: str, vuln_family: str, signature_triple_or_anchor: str) -> Tuple[str, str, str]:
    """Builds canonical identity key tuple (S, P, K)."""
    return (scenario_id.strip(), vuln_family.strip().upper(), signature_triple_or_anchor.strip().lower())


def compute_duplicate_rates(valid_accept_keys: List[Tuple[str, str, str]]) -> Tuple[int, float]:
    """Returns (unique_valid_accepts, duplicate_valid_accept_rate)."""
    total = len(valid_accept_keys)
    if total == 0:
        return 0, 0.0
    unique = len(set(valid_accept_keys))
    dup_rate = (total - unique) / total
    return unique, dup_rate


def compute_diagnostic_triage_efficiency_gain(
    tokens_doomed_diagnostic: float,
    tokens_doomed_baseline: float,
) -> Optional[float]:
    """Computes diagnostic triage efficiency gain, returning None if baseline is zero."""
    if tokens_doomed_baseline <= 0.0:
        return None
    return 1.0 - (tokens_doomed_diagnostic / tokens_doomed_baseline)


def compute_verifier_contradiction_count(
    disagreement_verifier_pass_but_anchor_strict_fail_count: int,
    disagreement_judge_accept_but_verifier_missing_or_parse_fail_count: int,
) -> int:
    """Computes verifier contradiction count from exact telemetry identifiers."""
    return (
        max(0, disagreement_verifier_pass_but_anchor_strict_fail_count)
        + max(0, disagreement_judge_accept_but_verifier_missing_or_parse_fail_count)
    )


def is_viable_yield(
    valid_accepts_candidate: int,
    valid_accepts_baseline_c1: int,
    valid_accepts_per_1m_candidate: float,
    min_yield_fraction: float = 0.50,
    min_rate_threshold: float = 10.0,
) -> bool:
    """
    Non-vacuous viable yield predicate:
    - If C1 baseline count is positive (> 0), requires candidate count >= min_yield_fraction * C1 count.
    - If C1 baseline count is zero (0), requires candidate valid_accepts_per_1m_tokens >= min_rate_threshold.
    """
    if valid_accepts_baseline_c1 > 0:
        return valid_accepts_candidate >= (valid_accepts_baseline_c1 * min_yield_fraction)
    return valid_accepts_per_1m_candidate >= min_rate_threshold


def evaluate_viable_yield_routing(
    valid_accepts_candidate: int,
    valid_accepts_baseline_c1: int,
    valid_accepts_per_1m_candidate: float,
    ppv_candidate: float,
    anti_gaming_valid: bool = True,
    has_logic_error: bool = False,
    min_yield_fraction: float = 0.50,
    min_rate_threshold: float = 10.0,
) -> str:
    """Determines whether candidate strategy is primary acceptance gate, routed to advisory triage lane, or rejected."""
    if has_logic_error or not anti_gaming_valid:
        return "REJECTED_ANTI_GAMING_INVARIANT"
    if is_viable_yield(
        valid_accepts_candidate=valid_accepts_candidate,
        valid_accepts_baseline_c1=valid_accepts_baseline_c1,
        valid_accepts_per_1m_candidate=valid_accepts_per_1m_candidate,
        min_yield_fraction=min_yield_fraction,
        min_rate_threshold=min_rate_threshold,
    ):
        return "PRIMARY_ACCEPTANCE_GATE"
    if ppv_candidate >= 0.80:
        return "SECONDARY_ADVISORY_TRIAGE_LANE"
    return "REJECTED_LOW_PRECISION_AND_YIELD"


# ==============================================================================
# TESTS
# ==============================================================================

def test_efficiency_metrics_zero_denominators():
    """Verify zero valid accepts or zero tokens evaluate safely."""
    # Zero tokens consumed
    assert compute_valid_accepts_per_1m_tokens(valid_accepts=0, total_tokens=0) == 0.0
    assert compute_tokens_per_valid_accept(valid_accepts=0, total_tokens=0) == float("inf")

    # Positive tokens, zero accepts
    assert compute_valid_accepts_per_1m_tokens(valid_accepts=0, total_tokens=1_000_000) == 0.0
    assert compute_tokens_per_valid_accept(valid_accepts=0, total_tokens=1_000_000) == float("inf")

    # Normal values
    assert compute_valid_accepts_per_1m_tokens(valid_accepts=10, total_tokens=1_000_000) == 10.0
    assert compute_tokens_per_valid_accept(valid_accepts=10, total_tokens=1_000_000) == 100_000.0


def test_relative_token_improvement():
    """Verify relative improvement calculation and edge cases."""
    # 25% improvement: baseline 100k, candidate 75k
    rel_imp = compute_relative_token_improvement(100_000.0, 75_000.0)
    assert rel_imp is not None
    assert math.isclose(rel_imp, 0.25, abs_tol=1e-5)

    # 50% improvement
    rel_imp_50 = compute_relative_token_improvement(100_000.0, 50_000.0)
    assert rel_imp_50 is not None
    assert math.isclose(rel_imp_50, 0.50, abs_tol=1e-5)

    # Candidate infinite (0 valid accepts)
    rel_imp_inf = compute_relative_token_improvement(100_000.0, float("inf"))
    assert rel_imp_inf == -1.0

    # Baseline zero or infinite -> None
    assert compute_relative_token_improvement(0.0, 50_000.0) is None
    assert compute_relative_token_improvement(float("inf"), 50_000.0) is None


def test_canonical_identity_and_duplicate_rates():
    """Verify canonical identity tuple extraction and duplicate rate computation."""
    # Empty input
    unique_cnt, dup_rate = compute_duplicate_rates([])
    assert unique_cnt == 0
    assert dup_rate == 0.0

    # 4 items: 2 distinct, 2 duplicates
    k1 = compute_canonical_identity_key("CVE-2026-0001", "BUFFER_OVERFLOW", "sink_ptr_write")
    k2 = compute_canonical_identity_key("CVE-2026-0001", "buffer_overflow", "SINK_PTR_WRITE")  # same key normalized
    k3 = compute_canonical_identity_key("CVE-2026-0002", "OOB_WRITE", "array_index_assign")
    k4 = compute_canonical_identity_key("CVE-2026-0001", "BUFFER_OVERFLOW", "sink_ptr_write")

    assert k1 == k2 == k4
    assert k1 != k3

    keys = [k1, k2, k3, k4]
    unique_cnt, dup_rate = compute_duplicate_rates(keys)
    assert unique_cnt == 2
    assert math.isclose(dup_rate, 2 / 4, abs_tol=1e-5)  # 50% duplicate rate


def test_diagnostic_triage_efficiency_gain_zero_baseline():
    """Verify diagnostic triage efficiency gain excludes zero-baseline pairs."""
    # Both 0 tokens -> None (undefined, must not award 1.0)
    assert compute_diagnostic_triage_efficiency_gain(0.0, 0.0) is None

    # Positive baseline, zero diagnostic tokens -> 100% gain
    gain_100 = compute_diagnostic_triage_efficiency_gain(0.0, 10_000.0)
    assert gain_100 is not None
    assert math.isclose(gain_100, 1.0, abs_tol=1e-5)

    # 20% gain: baseline 10,000 tokens, diagnostic 8,000 tokens
    gain_20 = compute_diagnostic_triage_efficiency_gain(8_000.0, 10_000.0)
    assert gain_20 is not None
    assert math.isclose(gain_20, 0.20, abs_tol=1e-5)


def test_verifier_contradiction_count():
    """Verify contradiction count summation from contract identifiers."""
    count = compute_verifier_contradiction_count(
        disagreement_verifier_pass_but_anchor_strict_fail_count=3,
        disagreement_judge_accept_but_verifier_missing_or_parse_fail_count=2,
    )
    assert count == 5

    # Negative defensive handling
    count_zero = compute_verifier_contradiction_count(-1, 0)
    assert count_zero == 0


def test_viable_yield_routing_positive_and_zero_baseline():
    """Verify precision gate vs acceptance lane routing with positive, zero, and invariant-failing baselines."""
    # Positive baseline: High yield (80 >= 50) and valid -> primary gate
    res = evaluate_viable_yield_routing(
        valid_accepts_candidate=80,
        valid_accepts_baseline_c1=100,
        valid_accepts_per_1m_candidate=80.0,
        ppv_candidate=0.85,
        anti_gaming_valid=True,
    )
    assert res == "PRIMARY_ACCEPTANCE_GATE"

    # Positive baseline: Low yield (30 < 50) but high precision and valid invariants -> secondary advisory triage lane
    res_triage = evaluate_viable_yield_routing(
        valid_accepts_candidate=30,
        valid_accepts_baseline_c1=100,
        valid_accepts_per_1m_candidate=30.0,
        ppv_candidate=0.85,
        anti_gaming_valid=True,
    )
    assert res_triage == "SECONDARY_ADVISORY_TRIAGE_LANE"

    # Positive baseline: Low yield (< 50%) and low precision (< 80%) -> rejected
    res_rej = evaluate_viable_yield_routing(
        valid_accepts_candidate=30,
        valid_accepts_baseline_c1=100,
        valid_accepts_per_1m_candidate=30.0,
        ppv_candidate=0.60,
        anti_gaming_valid=True,
    )
    assert res_rej == "REJECTED_LOW_PRECISION_AND_YIELD"

    # Zero baseline (C1 count == 0): Candidate rate >= 10.0 -> primary gate
    res_zero_base_pass = evaluate_viable_yield_routing(
        valid_accepts_candidate=15,
        valid_accepts_baseline_c1=0,
        valid_accepts_per_1m_candidate=15.0,
        ppv_candidate=0.90,
        anti_gaming_valid=True,
    )
    assert res_zero_base_pass == "PRIMARY_ACCEPTANCE_GATE"

    # Zero baseline (C1 count == 0): Candidate rate < 10.0 but PPV >= 80% and valid invariants -> secondary triage
    res_zero_base_triage = evaluate_viable_yield_routing(
        valid_accepts_candidate=5,
        valid_accepts_baseline_c1=0,
        valid_accepts_per_1m_candidate=5.0,
        ppv_candidate=0.85,
        anti_gaming_valid=True,
    )
    assert res_zero_base_triage == "SECONDARY_ADVISORY_TRIAGE_LANE"

    # Invariant failure (e.g. anchor grounding or verifier parse fail) -> rejected immediately even with 100% precision
    res_inv_fail = evaluate_viable_yield_routing(
        valid_accepts_candidate=30,
        valid_accepts_baseline_c1=100,
        valid_accepts_per_1m_candidate=30.0,
        ppv_candidate=1.0,
        anti_gaming_valid=False,
    )
    assert res_inv_fail == "REJECTED_ANTI_GAMING_INVARIANT"

    # Logic error -> rejected immediately
    res_err = evaluate_viable_yield_routing(
        valid_accepts_candidate=80,
        valid_accepts_baseline_c1=100,
        valid_accepts_per_1m_candidate=80.0,
        ppv_candidate=0.95,
        has_logic_error=True,
    )
    assert res_err == "REJECTED_ANTI_GAMING_INVARIANT"


def test_anti_gaming_invariants_zero_yield_gate():
    """Verify check_anti_gaming_invariants rejects zero-row metrics."""
    metrics_zero = {
        "accepted_rows": 0,
        "accepted_corpus_logic_error_rate": 0.0,
        "verifier_parse_ok_rate": 1.0,
        "b2_anchor_match_rate": 1.0,
    }
    is_valid, violations = check_anti_gaming_invariants(metrics_zero)
    assert is_valid is False
    assert len(violations) == 1
    assert "accepted_rows (0) < min (1) [zero-yield collapse]" in violations[0]


def test_holm_step_down_8_metrics_full_ranks_and_boundary():
    """Verify Holm step-down procedure over M=8 primary endpoint family including equality boundary and stopping rule."""
    # 8 p-values with rank 4 on equality boundary: p = 0.010 == 0.05 / 5
    p_vals = [0.001, 0.003, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040]
    results = apply_holm_step_down(p_vals, alpha=0.05)
    assert len(results) == 8

    # Rank 1: p=0.001 < 0.05/8 (0.00625) -> True
    assert results[0]["rank"] == 1
    assert results[0]["is_significant"] is True
    assert math.isclose(results[0]["alpha_k"], 0.05 / 8, abs_tol=1e-5)

    # Rank 2: p=0.003 < 0.05/7 (0.00714) -> True
    assert results[1]["rank"] == 2
    assert results[1]["is_significant"] is True
    assert math.isclose(results[1]["alpha_k"], 0.05 / 7, abs_tol=1e-5)

    # Rank 3: p=0.005 < 0.05/6 (0.00833) -> True
    assert results[2]["rank"] == 3
    assert results[2]["is_significant"] is True
    assert math.isclose(results[2]["alpha_k"], 0.05 / 6, abs_tol=1e-5)

    # Rank 4: p=0.010 == 0.05/5 (0.01000) -> False (strict inequality p < alpha_k fails at boundary)
    assert results[3]["rank"] == 4
    assert math.isclose(results[3]["alpha_k"], 0.05 / 5, abs_tol=1e-5)
    assert results[3]["is_significant"] is False

    # Ranks 5-8: Stopped by step-down rule -> False
    for r in range(4, 8):
        assert results[r]["is_significant"] is False
