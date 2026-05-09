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


if __name__ == "__main__":
    test_gate_passes_on_grounded_sample()
    test_gate_fails_when_no_anchor_matches_input()
    print("ok")

