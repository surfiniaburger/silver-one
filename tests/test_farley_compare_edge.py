import pytest
import json
from pathlib import Path
from scripts.farley_compare import (
    compute_suite_summary,
    get_usage_totals,
    top_regressions,
    _write_no_baseline_report,
    CASSETTE_ROOT,
    REPORT_ROOT,
)


def test_top_regressions_handles_missing_ids():
    base = {'tests': [
        {'id': 't1', 'farley_index': 8.0},
        {'farley_index': 5.0},  # missing id
    ]}
    pr = {'tests': [
        {'id': 't1', 'farley_index': 6.0},
        {'farley_index': 4.0},  # missing id
    ]}
    regs = top_regressions(base, pr, top_n=10)
    # should not raise and should include regression for t1
    assert any(r[1].get('id') == 't1' for r in regs)

def test_compute_suite_summary_empty():
    summary = compute_suite_summary({'tests': []})
    assert summary['avg_index'] == pytest.approx(0.0)
    assert summary['count'] == 0


def test_get_usage_totals_reads_farley_metadata():
    cassette = {
        "__metadata__": {
            "farley_usage_summary": {
                "usage": {
                    "totals": {
                        "calls": 2,
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                        "cost_usd": 0.01,
                    }
                }
            }
        }
    }

    totals = get_usage_totals(cassette)

    assert totals["calls"] == 2
    assert totals["total_tokens"] == 120


def test_write_no_baseline_report_creates_file(tmp_path, monkeypatch):
    """_write_no_baseline_report should write a PASS report when no baseline exists."""
    monkeypatch.setattr("scripts.farley_compare.REPORT_ROOT", tmp_path)

    pr = {
        "tests": [
            {"farley_index": 7.5, "farley_breakdown": {}},
            {"farley_index": 8.0, "farley_breakdown": {}},
        ]
    }
    out_path = str(tmp_path / "report.md")
    _write_no_baseline_report(out_path, pr)

    report = Path(out_path).read_text()
    assert "No baseline cassette found" in report
    assert "PASS" in report
    assert "7.75" in report  # average of 7.5 and 8.0


def test_main_exits_0_when_baseline_missing(tmp_path, monkeypatch):
    """main() should exit 0 and write an informational report when --baseline is absent."""
    import sys
    from scripts import farley_compare

    # Patch both roots to tmp_path so path validation works
    monkeypatch.setattr(farley_compare, "CASSETTE_ROOT", tmp_path)
    monkeypatch.setattr(farley_compare, "REPORT_ROOT", tmp_path)

    # Create a valid PR cassette in tmp_path
    pr_cassette = tmp_path / "pr.json"
    pr_cassette.write_text(json.dumps({
        "tests": [{"farley_index": 7.0, "farley_breakdown": {}}]
    }))
    out_path = tmp_path / "report.md"

    monkeypatch.setattr(
        sys, "argv",
        [
            "farley_compare.py",
            "--baseline", str(tmp_path / "baseline_does_not_exist.json"),
            "--pr", str(pr_cassette),
            "--out", str(out_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        farley_compare.main()

    assert exc_info.value.code == 0
    assert out_path.exists()
    report = out_path.read_text()
    assert "No baseline cassette found" in report
    assert "PASS" in report


@pytest.mark.asyncio
async def test_farley_evaluator_saves_tests_to_cassette(tmp_path, monkeypatch):
    """Verify that farley_score_evaluator writes the tests array to the cassette file."""
    import sys
    from scripts import farley_score_evaluator

    # 1. Setup a dummy test file
    test_file = tmp_path / "test_dummy.py"
    test_file.write_text("def test_dummy_example():\n    assert True\n", encoding="utf-8")

    # 2. Patch evaluate_test_case to return a mock breakdown.
    # *args/**kwargs intentionally mirrors the real signature so monkeypatch works
    # without coupling this helper to the exact call signature of evaluate_test_case.
    async def mock_evaluate_test_case(*args, **kwargs):
        import asyncio
        await asyncio.sleep(0)  # Required to use async features (SonarQube S7503)
        from scripts.farley_score_evaluator import FarleyScoreBreakdown, PropertyEvaluation
        pe = PropertyEvaluation(score=8, rationale="Good", suggestions=[])
        return FarleyScoreBreakdown(
            understandable=pe, maintainable=pe, repeatable=pe, atomic=pe,
            necessary=pe, granular=pe, fast=pe, first_tdd=pe,
            summary="Mocked report"
        )


    monkeypatch.setattr(farley_score_evaluator, "evaluate_test_case", mock_evaluate_test_case)
    monkeypatch.setattr(farley_score_evaluator, "TEST_ROOT", tmp_path)
    monkeypatch.setattr(farley_score_evaluator, "CASSETTE_ROOT", tmp_path)
    monkeypatch.setattr(farley_score_evaluator, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(farley_score_evaluator, "METRICS_ROOT", tmp_path)

    # 3. Define target cassette file path
    cassette_path = tmp_path / "farley_score_dummy.json"

    # 4. Invoke main_async with argv arguments
    monkeypatch.setattr(
        sys, "argv",
        [
            "farley_score_evaluator.py",
            str(test_file),
            "--cassette", str(cassette_path),
            "--mode", "record"
        ]
    )

    await farley_score_evaluator.main_async()

    # 5. Assert the evaluator exited successfully and saved the tests key
    assert cassette_path.exists()
    with cassette_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert "tests" in data
    assert len(data["tests"]) == 1
    assert data["tests"][0]["test_name"] == "test_dummy_example"
    assert data["tests"][0]["farley_index"] == pytest.approx(8.0)



