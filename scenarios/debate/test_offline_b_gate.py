import json
import tempfile

from offline_b_gate import BGateConfig, BGateThresholds, compute_b_metrics


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_gate_passes_on_grounded_sample():
    rows = [
        {
            "instruction": "Analyze ...",
            "input": "int foo(){ return bar(x); }",
            "output": {
                "predicate": "X",
                "anchors": ["bar(x)", "foo()"],
                "counterfactual": "Replace bar(x) with baz(x).",
                "verifier_report": "not_applicable",
                "support_level": "supported",
            },
        }
    ]
    with tempfile.TemporaryDirectory() as td:
        p = td + "/c.jsonl"
        _write_jsonl(p, rows)
        cfg = BGateConfig(
            thresholds=BGateThresholds(
                max_unsupported_in_accepted_rate=0.05,
                max_inconclusive_in_accepted_rate=0.20,
                min_anchor_match_rate=0.80,
            )
        )
        metrics = compute_b_metrics(input_path=p, config=cfg)
        assert metrics["pass"] is True


def test_gate_fails_when_no_anchor_matches_input():
    rows = [
        {
            "instruction": "Analyze ...",
            "input": "int foo(){ return bar(x); }",
            "output": {
                "predicate": "X",
                "anchors": ["not_present", "also_missing"],
                "counterfactual": "Replace bar(x) with baz(x).",
                "verifier_report": "not_applicable",
                "support_level": "supported",
            },
        }
    ]
    with tempfile.TemporaryDirectory() as td:
        p = td + "/c.jsonl"
        _write_jsonl(p, rows)
        cfg = BGateConfig(
            require_anchor_match=True,
            thresholds=BGateThresholds(
                max_unsupported_in_accepted_rate=0.05,
                max_inconclusive_in_accepted_rate=0.20,
                min_anchor_match_rate=0.80,
            ),
        )
        metrics = compute_b_metrics(input_path=p, config=cfg)
        assert metrics["pass"] is False
        assert any(f["reason"] == "no_anchor_matches_input" for f in metrics["failures"])


def test_verifier_metrics_and_disagreement_counts():
    rows = [
        {
            "instruction": "Analyze ...",
            "input": "int foo(){ return bar(x); }",
            "output": {
                "predicate": "X",
                "anchors": ["bar(x)", "foo("],
                "counterfactual": "Replace bar(x) with baz(x).",
                "verifier_report": "not_applicable",
                "support_level": "supported",
            },
        }
    ]
    attempts = [
        {
            "decision": "accepted",
            "support_level": "supported",
            "judge_eval": {"winner": "pro_debater"},
            "verifier": {"called": True, "parse_ok": False, "passes_audit": None},
        },
        {
            "decision": "rejected",
            "reject_reason": "anchors_no_match",
            "support_level": "supported",
            "judge_eval": {"winner": "pro_debater"},
            "verifier": {"called": True, "parse_ok": True, "passes_audit": True},
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        p = td + "/c.jsonl"
        a = td + "/attempts.jsonl"
        _write_jsonl(p, rows)
        _write_jsonl(a, attempts)
        cfg = BGateConfig(
            thresholds=BGateThresholds(
                max_unsupported_in_accepted_rate=0.05,
                max_inconclusive_in_accepted_rate=0.20,
                min_anchor_match_rate=0.80,
            )
        )
        metrics = compute_b_metrics(input_path=p, attempts_path=a, config=cfg)
        assert metrics["verifier_called_rate"] == 1.0
        assert metrics["verifier_called_on_prowin_rate"] == 1.0
        assert metrics["verifier_parse_ok_rate"] == 0.5
        assert metrics["verifier_pass_rate"] == 1.0
        assert metrics["accepted_with_not_applicable_rate"] == 1.0
        assert metrics["disagreement_judge_accept_but_verifier_missing_or_parse_fail_count"] == 1
        assert metrics["disagreement_verifier_pass_but_anchor_strict_fail_count"] == 1


def test_strict_b2_and_verifier_threshold_checks():
    rows = [
        {
            "instruction": "Analyze ...",
            "input": "int foo(){ return bar(x); }",
            "output": {
                "predicate": "X",
                "anchors": ["bar(x)", "foo("],
                "counterfactual": "Replace bar(x) with baz(x).",
                "verifier_report": {"passes_audit": True},
                "support_level": "supported",
            },
        }
    ]
    attempts = [
        {
            "decision": "accepted",
            "support_level": "supported",
            "judge_eval": {"winner": "pro_debater"},
            "verifier": {"called": True, "parse_ok": True, "passes_audit": True},
        },
        {
            "decision": "rejected",
            "reject_reason": "mechanism_evidence_failed",
            "support_level": "supported",
            "judge_eval": {"winner": "pro_debater"},
            "verifier": {"called": True, "parse_ok": True, "passes_audit": True},
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        p = td + "/c.jsonl"
        a = td + "/attempts.jsonl"
        _write_jsonl(p, rows)
        _write_jsonl(a, attempts)
        cfg = BGateConfig(
            thresholds=BGateThresholds(
                max_unsupported_in_accepted_rate=0.05,
                max_inconclusive_in_accepted_rate=0.20,
                min_anchor_match_rate=0.80,  # strict fail-rate must be <= 0.20
                min_verifier_pass_rate=0.9,
                min_verifier_parse_ok_rate=1.0,
            )
        )
        metrics = compute_b_metrics(input_path=p, attempts_path=a, config=cfg)
        assert metrics["b2_strict_fail_rate"] == 0.5
        assert metrics["checks"]["min_anchor_match_rate"] is False
        assert metrics["checks"]["min_verifier_pass_rate"] is True
        assert metrics["checks"]["min_verifier_parse_ok_rate"] is True


if __name__ == "__main__":
    test_gate_passes_on_grounded_sample()
    test_gate_fails_when_no_anchor_matches_input()
    test_verifier_metrics_and_disagreement_counts()
    test_strict_b2_and_verifier_threshold_checks()
    print("ok")
