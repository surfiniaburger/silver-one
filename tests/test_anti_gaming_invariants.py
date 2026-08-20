"""Unit tests for check_anti_gaming_invariants in scenarios/debate/offline_b_gate.py."""

from __future__ import annotations

from scenarios.debate.offline_b_gate import check_anti_gaming_invariants


def test_anti_gaming_invariants_pass():
    """Verify clean passing metrics produce True and no violations."""
    metrics = {
        "accepted_rows": 10,
        "accepted_corpus_logic_error_rate": 0.0,
        "verifier_parse_ok_rate": 1.0,
        "b2_anchor_match_rate": 1.0,
    }
    is_valid, violations = check_anti_gaming_invariants(metrics)
    assert is_valid is True
    assert len(violations) == 0


def test_anti_gaming_invariants_zero_yield_collapse_violation():
    """Verify accepted_rows == 0 triggers anti-gaming violation even when rates pass."""
    metrics = {
        "accepted_rows": 0,
        "accepted_corpus_logic_error_rate": 0.0,
        "verifier_parse_ok_rate": 1.0,
        "b2_anchor_match_rate": 1.0,
    }
    is_valid, violations = check_anti_gaming_invariants(metrics)
    assert is_valid is False
    assert len(violations) == 1
    assert "accepted_rows (0) < min (1) [zero-yield collapse]" in violations[0]


def test_anti_gaming_invariants_logic_error_violation():
    """Verify logic error rate > 0.0 triggers anti-gaming violation."""
    metrics = {
        "accepted_rows": 10,
        "accepted_corpus_logic_error_rate": 0.05,
        "verifier_parse_ok_rate": 1.0,
        "b2_anchor_match_rate": 1.0,
    }
    is_valid, violations = check_anti_gaming_invariants(metrics)
    assert is_valid is False
    assert len(violations) == 1
    assert "accepted_logic_error_rate (0.0500) > max (0.0000)" in violations[0]


def test_anti_gaming_invariants_parse_ok_violation():
    """Verify verifier_parse_ok_rate < 0.95 triggers anti-gaming violation."""
    metrics = {
        "accepted_rows": 10,
        "accepted_corpus_logic_error_rate": 0.0,
        "verifier_parse_ok_rate": 0.90,
        "b2_anchor_match_rate": 1.0,
    }
    is_valid, violations = check_anti_gaming_invariants(metrics)
    assert is_valid is False
    assert len(violations) == 1
    assert "verifier_parse_ok_rate (0.9000) < min (0.9500)" in violations[0]


def test_anti_gaming_invariants_anchor_match_violation():
    """Verify b2_anchor_match_rate < 0.80 triggers anti-gaming violation."""
    metrics = {
        "accepted_rows": 10,
        "accepted_corpus_logic_error_rate": 0.0,
        "verifier_parse_ok_rate": 1.0,
        "b2_anchor_match_rate": 0.75,
    }
    is_valid, violations = check_anti_gaming_invariants(metrics)
    assert is_valid is False
    assert len(violations) == 1
    assert "b2_anchor_match_rate (0.7500) < min (0.8000)" in violations[0]


def test_anti_gaming_invariants_custom_min_accepted_rows():
    """Verify non-default min_accepted_rows rejects metrics with fewer accepted rows."""
    metrics = {
        "accepted_rows": 3,
        "accepted_corpus_logic_error_rate": 0.0,
        "verifier_parse_ok_rate": 1.0,
        "b2_anchor_match_rate": 1.0,
    }
    # Passes with default min_accepted_rows=1
    is_valid_default, violations_default = check_anti_gaming_invariants(metrics, min_accepted_rows=1)
    assert is_valid_default is True
    assert len(violations_default) == 0

    # Fails with custom min_accepted_rows=5
    is_valid_custom, violations_custom = check_anti_gaming_invariants(metrics, min_accepted_rows=5)
    assert is_valid_custom is False
    assert len(violations_custom) == 1
    assert "accepted_rows (3) < min (5) [zero-yield collapse]" in violations_custom[0]


def test_anti_gaming_invariants_invalid_min_accepted_rows_raises():
    """Verify zero or negative min_accepted_rows raises ValueError."""
    import pytest
    from scenarios.debate.offline_b_gate import BGateThresholds

    metrics = {"accepted_rows": 5}
    with pytest.raises(ValueError, match="min_accepted_rows must be >= 1"):
        check_anti_gaming_invariants(metrics, min_accepted_rows=0)

    with pytest.raises(ValueError, match="min_accepted_rows must be >= 1"):
        check_anti_gaming_invariants(metrics, min_accepted_rows=-2)

    with pytest.raises(ValueError, match="min_accepted_rows must be >= 1"):
        BGateThresholds(min_accepted_rows=0)

    with pytest.raises(ValueError, match="min_accepted_rows must be >= 1"):
        BGateThresholds(min_accepted_rows=-1)
