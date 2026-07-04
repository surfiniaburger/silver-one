import pytest
from scripts.unified_compare import combine_token_spend


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
    assert "| tests/test_code_review.py | test_unsubstantiated_block | 10.00/10 | **WARN** | OK |" in content
    assert "**BLOCK**" not in content


