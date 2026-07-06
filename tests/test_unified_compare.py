import pytest
import json

from scripts.unified_compare import (
    combine_token_spend,
    parse_compatibility_result,
    resolve_farley_baseline,
)


def test_combine_token_spend_empty():
    res = combine_token_spend({}, {})
    assert "0 total tokens" in res


def test_combine_token_spend_with_values():
    cr = {
        "__metadata__": {
            "code_review_usage_summary": {
                "usage": {
                    "totals": {
                        "calls": 2,
                        "prompt_tokens": 1000,
                        "completion_tokens": 200,
                        "total_tokens": 1200,
                        "cost_usd": 0.012,
                    }
                }
            }
        }
    }
    farley = {
        "__metadata__": {
            "farley_usage_summary": {
                "usage": {
                    "totals": {
                        "calls": 1,
                        "prompt_tokens": 500,
                        "completion_tokens": 100,
                        "total_tokens": 600,
                        "cost_usd": 0.006,
                    }
                }
            }
        }
    }

    res = combine_token_spend(cr, farley)
    assert "1800 total tokens" in res
    assert "1500 prompt" in res
    assert "300 completion" in res
    assert "3 LLM call(s)" in res
    assert "0.018000" in res


def test_write_unified_report_basic(tmp_path):
    from scripts.unified_compare import write_unified_report
    out_file = tmp_path / "report.md"
    write_unified_report(
        out_path=out_file,
        unified_verdict="PASS",
        reasons=[],
        spend_str="**Token spend**: 0 total tokens\n\n",
        cr_data={"units": [], "verdict": "PASS"},
        farley_data={
            "bsum": {"avg_index": 0.0, "count": 0},
            "psum": {"avg_index": 8.5},
            "delta": 0.0,
            "verdict": "PASS",
            "regressions": [],
            "baseline_exists": False,
        },
        compat_data={"ok": True, "score": 10.0, "regressions": []}
    )
    content = out_file.read_text(encoding="utf-8")
    assert "# MSEC Unified Quality Report" in content
    assert "Verdict: **PASS**" in content
    assert "Token spend" in content
    assert "N/A (no Python code changes)" in content
    assert "Score: 10.0/10" in content


def test_parse_compatibility_result_legacy_pass():
    result = parse_compatibility_result({
        "pass": True,
        "compatibility_index": 10.0,
        "regressions": [],
    })

    assert result.state == "PASS"
    assert result.compatible is True
    assert result.score == pytest.approx(10.0)


def test_parse_compatibility_result_check_failed():
    result = parse_compatibility_result({
        "state": "CHECK_FAILED",
        "pass": False,
        "compatibility_index": 0.0,
        "reason": "Parser failed.",
        "details": ["bad syntax"],
    })

    assert result.state == "CHECK_FAILED"
    assert result.compatible is False
    assert result.reason == "Parser failed."
    assert result.details == ["bad syntax"]


def test_resolve_farley_baseline_first_run():
    result, path, cassette = resolve_farley_baseline("missing-baseline.json", required=False)

    assert result.state == "FIRST_RUN"
    assert result.available is False
    assert result.required is False
    assert path is None
    assert cassette == {"tests": []}


def test_resolve_farley_baseline_missing_required():
    result, path, cassette = resolve_farley_baseline("missing-baseline.json", required=True)

    assert result.state == "BASELINE_MISSING"
    assert result.available is False
    assert result.required is True
    assert path is None
    assert cassette == {"tests": []}


def test_resolve_farley_baseline_available(tmp_path, monkeypatch):
    from scripts import unified_compare

    monkeypatch.setattr(unified_compare, "PROJECT_ROOT", tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"tests": []}), encoding="utf-8")

    result, path, cassette = resolve_farley_baseline("baseline.json", required=True)

    assert result.state == "AVAILABLE"
    assert result.available is True
    assert result.required is True
    assert path == baseline
    assert cassette == {"tests": []}


def test_resolve_farley_baseline_corrupted_json(tmp_path, monkeypatch):
    from scripts import unified_compare

    monkeypatch.setattr(unified_compare, "PROJECT_ROOT", tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{not-json", encoding="utf-8")

    result, path, cassette = resolve_farley_baseline("baseline.json", required=True)

    assert result.state == "BASELINE_CORRUPTED"
    assert result.available is False
    assert path == baseline
    assert cassette == {"tests": []}


def test_resolve_farley_baseline_corrupted_shape(tmp_path, monkeypatch):
    from scripts import unified_compare

    monkeypatch.setattr(unified_compare, "PROJECT_ROOT", tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"not_tests": []}), encoding="utf-8")

    result, path, cassette = resolve_farley_baseline("baseline.json", required=False)

    assert result.state == "BASELINE_CORRUPTED"
    assert result.available is False
    assert path == baseline
    assert cassette == {"tests": []}


def test_write_unified_report_full(tmp_path):
    from scripts.unified_compare import write_unified_report
    out_file = tmp_path / "report.md"

    cr_units = [
        {
            "file_path": "src/utils.py",
            "name": "helper",
            "review": {
                "severity": "WARN",
                "summary": "Fix this function",
                "complexity": {"score": 6.5, "rationale": "High complexity", "suggestions": ["Simplify structure"]}
            }
        }
    ]

    farley_regressions = [
        (-1.5, {"file_path": "tests/test_x.py", "test_name": "test_one", "farley_index": 9.0}, {"farley_index": 7.5})
    ]

    compat_regressions = ["Removed method foo"]

    write_unified_report(
        out_path=out_file,
        unified_verdict="FAIL",
        reasons=["Code review quality warning", "API Compatibility regressions"],
        spend_str="**Token spend**: 100 total tokens\n\n",
        cr_data={"units": cr_units, "verdict": "WARN"},
        farley_data={
            "bsum": {"avg_index": 8.0},
            "psum": {"avg_index": 7.5},
            "delta": -0.5,
            "verdict": "PASS",
            "regressions": farley_regressions,
            "baseline_exists": True,
        },
        compat_data={"ok": False, "score": 8.0, "regressions": compat_regressions}
    )
    content = out_file.read_text(encoding="utf-8")
    assert "# MSEC Unified Quality Report" in content
    assert "Verdict: **FAIL**" in content
    assert "Code review quality warning" in content
    assert "Fix this function" in content
    assert "test_one" in content
    assert "Removed method foo" in content
    assert "FAIL (Score: 8.0/10)" in content


def test_write_unified_report_null_values(tmp_path):
    from scripts.unified_compare import write_unified_report
    out_file = tmp_path / "report.md"

    cr_units = [
        {
            "file_path": "src/utils.py",
            "name": "helper",
            "review": None
        }
    ]

    farley_regressions = [
        (-1.5, {"file_path": "tests/test_x.py", "test_name": "test_one", "farley_index": None}, {"farley_index": None})
    ]

    compat_regressions = ["Removed method foo"]

    write_unified_report(
        out_path=out_file,
        unified_verdict="FAIL",
        reasons=["Code review quality warning", "API Compatibility regressions"],
        spend_str="**Token spend**: 100 total tokens\n\n",
        cr_data={"units": cr_units, "verdict": "WARN"},
        farley_data={
            "bsum": {"avg_index": 8.0},
            "psum": {"avg_index": 7.5},
            "delta": -0.5,
            "verdict": "PASS",
            "regressions": farley_regressions,
            "baseline_exists": True,
        },
        compat_data={"ok": False, "score": None, "regressions": compat_regressions}
    )
    content = out_file.read_text(encoding="utf-8")
    assert "# MSEC Unified Quality Report" in content
    assert "Score: 10.0/10" in content


def test_write_unified_report_with_findings(tmp_path):
    from scripts.unified_compare import write_unified_report
    out_file = tmp_path / "report.md"

    cr_units = [
        {
            "file_path": "src/utils.py",
            "name": "helper",
            "review": {
                "severity": "WARN",
                "summary": "Some issues with helper",
                "complexity": {"score": 6.5, "rationale": "High complexity", "suggestions": ["Simplify"]},
                "findings": [
                    {
                        "title": "Uncaught ValueError",
                        "category": "Correctness",
                        "severity": "WARN",
                        "evidence": {
                            "location_type": "code",
                            "path": "src/utils.py",
                            "details": {"start_line": 15, "end_line": 20, "function_name": "helper"}
                        },
                        "engineering_rationale": "Parsing raw strings directly can throw ValueError.",
                        "engineering_consequence": "Uncaught exceptions will crash the program.",
                        "impact": {"correctness": "MEDIUM"},
                        "confidence": 0.9,
                        "recommended_action": "Wrap statement in a try-except block."
                    }
                ]
            }
        }
    ]

    write_unified_report(
        out_path=out_file,
        unified_verdict="FAIL",
        reasons=["Code review quality warning"],
        spend_str="**Token spend**: 100 total tokens\n\n",
        cr_data={"units": cr_units, "verdict": "WARN"},
        farley_data={
            "bsum": {"avg_index": 8.0},
            "psum": {"avg_index": 7.5},
            "delta": -0.5,
            "verdict": "PASS",
            "regressions": [],
            "baseline_exists": True,
        },
        compat_data={"ok": True, "score": 10.0, "regressions": []}
    )
    content = out_file.read_text(encoding="utf-8")
    assert "##### Structured Engineering Findings" in content
    assert "Uncaught ValueError" in content
    assert "Lines 15-20" in content
    assert "Uncaught exceptions will crash the program." in content
    assert "Wrap statement in a try-except block." in content


def test_write_unified_report_displays_effective_severity(tmp_path):
    from scripts.unified_compare import write_unified_report
    out_file = tmp_path / "report.md"

    cr_units = [
        {
            "file_path": "tests/test_code_review.py",
            "name": "test_unsubstantiated_block",
            "review": {
                "severity": "BLOCK",
                "summary": "OK",
                "findings": [],
            },
        }
    ]

    write_unified_report(
        out_path=out_file,
        unified_verdict="PASS",
        reasons=[],
        spend_str="**Token spend**: 100 total tokens\n\n",
        cr_data={"units": cr_units, "verdict": "PASS"},
        farley_data={
            "bsum": {"avg_index": 8.0},
            "psum": {"avg_index": 8.0},
            "delta": 0.0,
            "verdict": "PASS",
            "regressions": [],
            "baseline_exists": True,
        },
        compat_data={"ok": True, "score": 10.0, "regressions": []},
    )

    content = out_file.read_text(encoding="utf-8")
    assert "| tests/test_code_review.py | test_unsubstantiated_block | INVALID (MISSING_DIMENSION) | **WARN** | OK |" in content
    assert "**BLOCK**" not in content


def test_write_unified_report_marks_invalid_cqi_metric_failed(tmp_path):
    from scripts.unified_compare import write_unified_report
    out_file = tmp_path / "report.md"

    cr_units = [
        {
            "file_path": "src/utils.py",
            "name": "invalid_review",
            "review": {
                "severity": "OK",
                "summary": "Incomplete review",
            },
        }
    ]

    write_unified_report(
        out_path=out_file,
        unified_verdict="FAIL",
        reasons=["INVALID CQI: src/utils.py -> invalid_review: Required CQI dimension 'readability' is missing."],
        spend_str="**Token spend**: 100 total tokens\n\n",
        cr_data={"units": cr_units, "verdict": "PASS"},
        farley_data={
            "bsum": {"avg_index": 8.0},
            "psum": {"avg_index": 8.0},
            "delta": 0.0,
            "verdict": "PASS",
            "regressions": [],
            "baseline_exists": True,
        },
        compat_data={"ok": True, "score": 10.0, "regressions": []},
    )

    content = out_file.read_text(encoding="utf-8")
    assert "| Code Quality Index (CQI) | INVALID (1/1 unit(s)) | **FAIL** |" in content
    assert "| src/utils.py | invalid_review | INVALID (MISSING_DIMENSION) | **OK** | Incomplete review |" in content


def test_write_unified_report_ignores_malformed_cqi_units(tmp_path):
    from scripts.unified_compare import write_unified_report
    out_file = tmp_path / "report.md"

    write_unified_report(
        out_path=out_file,
        unified_verdict="PASS",
        reasons=[],
        spend_str="**Token spend**: 100 total tokens\n\n",
        cr_data={"units": ["not-a-unit"], "verdict": "PASS"},
        farley_data={
            "bsum": {"avg_index": 8.0},
            "psum": {"avg_index": 8.0},
            "delta": 0.0,
            "verdict": "PASS",
            "regressions": [],
            "baseline_exists": True,
        },
        compat_data={"ok": True, "score": 10.0, "regressions": []},
    )

    content = out_file.read_text(encoding="utf-8")
    assert "| Code Quality Index (CQI) | N/A (no Python code changes) | **PASS** |" in content


def test_write_unified_report_displays_baseline_state(tmp_path):
    from scripts.unified_compare import write_unified_report
    out_file = tmp_path / "report.md"

    write_unified_report(
        out_path=out_file,
        unified_verdict="FAIL",
        reasons=["FARLEY BASELINE: Farley baseline is required but was not found."],
        spend_str="**Token spend**: 100 total tokens\n\n",
        cr_data={"units": [], "verdict": "PASS"},
        farley_data={
            "bsum": {"avg_index": 0.0},
            "psum": {"avg_index": 8.0},
            "delta": 0.0,
            "verdict": "PASS",
            "regressions": [],
            "baseline_exists": False,
            "baseline_state": "BASELINE_MISSING",
            "baseline_reason": "Farley baseline is required but was not found.",
            "baseline_details": ["missing baseline.json"],
        },
        compat_data={"ok": True, "score": 10.0, "regressions": [], "state": "PASS"},
    )

    content = out_file.read_text(encoding="utf-8")
    assert "| Farley Test Quality | PR avg: 8.00 (baseline missing) | **FAIL** |" in content
    assert "State: **BASELINE_MISSING**" in content
    assert "missing baseline.json" in content


def test_write_unified_report_displays_compatibility_check_failed(tmp_path):
    from scripts.unified_compare import write_unified_report
    out_file = tmp_path / "report.md"

    write_unified_report(
        out_path=out_file,
        unified_verdict="FAIL",
        reasons=["API COMPATIBILITY CHECK_FAILED: Parser failed."],
        spend_str="**Token spend**: 100 total tokens\n\n",
        cr_data={"units": [], "verdict": "PASS"},
        farley_data={
            "bsum": {"avg_index": 8.0},
            "psum": {"avg_index": 8.0},
            "delta": 0.0,
            "verdict": "PASS",
            "regressions": [],
            "baseline_exists": True,
            "baseline_state": "AVAILABLE",
        },
        compat_data={
            "ok": False,
            "score": 0.0,
            "regressions": [],
            "state": "CHECK_FAILED",
            "reason": "Parser failed.",
            "details": ["bad syntax"],
        },
    )

    content = out_file.read_text(encoding="utf-8")
    assert "| API Compatibility | CHECK_FAILED (Score: 0.0/10) | **FAIL** |" in content
    assert "State: **CHECK_FAILED**" in content
    assert "bad syntax" in content
