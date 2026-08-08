"""Unit tests for check_anti_gaming_invariants in scenarios/debate/offline_b_gate.py."""

from __future__ import annotations

from scenarios.debate.offline_b_gate import check_anti_gaming_invariants


def test_anti_gaming_invariants_pass():
    """Verify clean passing metrics produce True and no violations."""
    metrics = {
        "accepted_corpus_logic_error_rate": 0.0,
        "verifier_parse_ok_rate": 1.0,
        "b2_anchor_match_rate": 1.0,
    }
    is_valid, violations = check_anti_gaming_invariants(metrics)
    assert is_valid is True
    assert len(violations) == 0


def test_anti_gaming_invariants_logic_error_violation():
    """Verify logic error rate > 0.0 triggers anti-gaming violation."""
    metrics = {
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
        "accepted_corpus_logic_error_rate": 0.0,
        "verifier_parse_ok_rate": 1.0,
        "b2_anchor_match_rate": 0.75,
    }
    is_valid, violations = check_anti_gaming_invariants(metrics)
    assert is_valid is False
    assert len(violations) == 1
    assert "b2_anchor_match_rate (0.7500) < min (0.8000)" in violations[0]
