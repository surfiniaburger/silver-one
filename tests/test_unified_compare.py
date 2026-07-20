import pytest
import json

from scripts.unified_compare import (
    combine_token_spend,
    get_code_review_coverage,
    get_code_review_validation_summary,
    parse_compatibility_result,
    resolve_farley_baseline,
    write_unified_report,
)
from scripts.code_review_compare import get_reviews


def farley_report_data(**overrides):
    data = {
        "bsum": {"avg_index": 8.0},
        "psum": {"avg_index": 8.0},
        "delta": 0.0,
        "verdict": "PASS",
        "regressions": [],
        "baseline_exists": True,
    }
    data.update(overrides)
    return data


def compat_report_data(**overrides):
    data = {"ok": True, "score": 10.0, "regressions": []}
    data.update(overrides)
    return data


def render_unified_report(
    tmp_path,
    *,
    cr_units=None,
    cr_verdict="PASS",
    unified_verdict="PASS",
    reasons=None,
    spend_str="**Token spend**: 100 total tokens\n\n",
    farley_data=None,
    compat_data=None,
    review_coverage=None,
    validation_summary=None,
):
    """Render the public unified report markdown with explicit test defaults."""
    out_file = tmp_path / "report.md"
    write_unified_report(
        out_path=out_file,
        unified_verdict=unified_verdict,
        reasons=reasons or [],
        spend_str=spend_str,
        cr_data={
            "units": cr_units or [],
            "verdict": cr_verdict,
            "review_coverage": review_coverage or {},
            "validation_summary": validation_summary or {},
        },
        farley_data=farley_data or farley_report_data(),
        compat_data=compat_data or compat_report_data(),
    )
    return out_file.read_text(encoding="utf-8")


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


def test_get_code_review_coverage_from_metadata():
    coverage = {
        "total_extracted_units": 33,
        "reviewed_units": 33,
        "skipped_units": 0,
        "batch_count": 2,
        "max_units_per_batch": 20,
        "max_tokens_per_batch": 80000,
    }

    result = get_code_review_coverage({
        "__metadata__": {
            "code_review_usage_summary": {
                "review_coverage": coverage,
            }
        }
    })

    assert result == coverage


def test_get_code_review_validation_summary_from_metadata():
    validation_summary = {
        "valid_units": 3,
        "repaired_units": 1,
        "normalized_units": 0,
        "invalid_units": 1,
        "details": {
            "structured_output": {
                "repair_attempts": 2,
                "validation_retries": 1,
                "final_failures": 1,
            },
            "provider_error": {"failures": 0},
        },
    }

    result = get_code_review_validation_summary({
        "__metadata__": {
            "code_review_usage_summary": {
                "validation_summary": validation_summary,
            }
        }
    })

    assert result == validation_summary


def test_write_unified_report_basic(tmp_path):
    content = render_unified_report(
        tmp_path,
        spend_str="**Token spend**: 0 total tokens\n\n",
        farley_data=farley_report_data(
            bsum={"avg_index": 0.0, "count": 0},
            psum={"avg_index": 8.5},
            baseline_exists=False,
        ),
    )
    assert "# MSEC Unified Quality Report" in content
    assert "Verdict: **PASS**" in content
    assert "Token spend" in content
    assert "N/A (no Python code changes)" in content
    assert "Score: 10.0/10" in content


def test_write_unified_report_displays_code_review_coverage(tmp_path):
    content = render_unified_report(
        tmp_path,
        review_coverage={
            "total_extracted_units": 33,
            "reviewed_units": 33,
            "skipped_units": 0,
            "batch_count": 2,
            "max_units_per_batch": 20,
            "max_tokens_per_batch": 80000,
        },
    )

    assert "| Review Coverage | Changed units reviewed by the evaluator | 33/33 unit(s) reviewed across 2 batch(es); 0 skipped | **PASS** |" in content


def test_write_unified_report_separates_structured_output_health(tmp_path):
    content = render_unified_report(
        tmp_path,
        unified_verdict="FAIL",
        reasons=["structured output final failure"],
        validation_summary={
            "valid_units": 3,
            "repaired_units": 1,
            "normalized_units": 0,
            "invalid_units": 1,
            "details": {
                "repaired_confidence_count": 4,
                "repaired_score_count": 2,
                "repaired_default_count": 1,
                "dropped_finding_count": 0,
                "normalized_path_count": 3,
                "normalized_text_count": 5,
                "invalid_field_count": 0,
                "structured_output": {
                    "repair_attempts": 2,
                    "validation_retries": 1,
                    "final_failures": 1,
                },
                "provider_error": {"failures": 0},
            },
        },
    )

    assert "| Reviewer Quality | Model judgment over reviewed code | N/A (no Python code changes) | **PASS** |" in content
    assert (
        "| Structured Output Health | Schema validity, repairs, retries, final failures "
        "| 3 valid, 1 repaired, 0 normalized, 1 invalid; 2 repair attempt(s), "
        "1 retry/retries, 1 final failure(s) | **FAIL** |"
    ) in content
    assert (
        "| Repair Breakdown | Schema repair categories applied during validation "
        "| 4 confidence, 2 score, 1 default, 0 dropped finding(s), 0 invalid field(s); "
        "3 path(s) normalized, 5 text(s) normalized | **WARN** |"
    ) in content


def test_write_unified_report_repair_breakdown_normalization_only_passes(tmp_path):
    content = render_unified_report(
        tmp_path,
        validation_summary={
            "valid_units": 0,
            "repaired_units": 0,
            "normalized_units": 2,
            "invalid_units": 0,
            "details": {
                "repaired_confidence_count": 0,
                "repaired_score_count": 0,
                "repaired_default_count": 0,
                "dropped_finding_count": 0,
                "normalized_path_count": 2,
                "normalized_text_count": 3,
                "invalid_field_count": 0,
                "structured_output": {
                    "repair_attempts": 0,
                    "validation_retries": 0,
                    "final_failures": 0,
                },
                "provider_error": {"failures": 0},
            },
        },
    )

    assert (
        "| Repair Breakdown | Schema repair categories applied during validation "
        "| 0 confidence, 0 score, 0 default, 0 dropped finding(s), 0 invalid field(s); "
        "2 path(s) normalized, 3 text(s) normalized | **PASS** |"
    ) in content


def test_write_unified_report_displays_provider_runtime_lane(tmp_path):
    content = render_unified_report(
        tmp_path,
        unified_verdict="FAIL",
        reasons=["provider failure"],
        validation_summary={
            "valid_units": 0,
            "repaired_units": 0,
            "normalized_units": 0,
            "invalid_units": 0,
            "details": {
                "structured_output": {
                    "repair_attempts": 0,
                    "validation_retries": 0,
                    "final_failures": 0,
                },
                "provider_error": {"failures": 2},
            },
        },
    )

    assert "| Provider Runtime | Local model/provider execution health | 2 provider failure(s) | **FAIL** |" in content


def test_write_unified_report_coerces_malformed_telemetry_counts(tmp_path):
    content = render_unified_report(
        tmp_path,
        review_coverage={
            "total_extracted_units": "bad-total",
            "reviewed_units": "3",
            "skipped_units": "bad-skipped",
            "batch_count": "1",
        },
        validation_summary={
            "valid_units": "bad-valid",
            "repaired_units": "2",
            "normalized_units": None,
            "invalid_units": "bad-invalid",
            "details": {
                "repaired_confidence_count": "bad-confidence",
                "repaired_score_count": "2",
                "repaired_default_count": None,
                "dropped_finding_count": "bad-dropped",
                "normalized_path_count": "1",
                "normalized_text_count": "bad-text",
                "invalid_field_count": "bad-invalid-field",
                "structured_output": {
                    "repair_attempts": "bad-repairs",
                    "validation_retries": "1",
                    "final_failures": "bad-failures",
                },
                "provider_error": {"failures": "bad-provider"},
            },
        },
    )

    assert "0 valid, 2 repaired, 0 normalized, 0 invalid; 0 repair attempt(s), 1 retry/retries, 0 final failure(s)" in content
    assert "0 confidence, 2 score, 0 default, 0 dropped finding(s), 0 invalid field(s); 1 path(s) normalized, 0 text(s) normalized" in content
    assert "| Provider Runtime | Local model/provider execution health | 0 provider failure(s) | **PASS** |" in content
    assert "| Review Coverage | Changed units reviewed by the evaluator | 3/0 unit(s) reviewed across 1 batch(es); 0 skipped | **WARN** |" in content


def test_write_unified_report_coerces_infinite_telemetry_counts(tmp_path):
    content = render_unified_report(
        tmp_path,
        review_coverage={
            "total_extracted_units": float("inf"),
            "reviewed_units": 1,
            "skipped_units": 0,
            "batch_count": 1,
        },
        validation_summary={
            "valid_units": float("inf"),
            "repaired_units": 0,
            "normalized_units": 0,
            "invalid_units": 0,
            "details": {
                "repaired_confidence_count": float("inf"),
                "repaired_score_count": 0,
                "repaired_default_count": 0,
                "dropped_finding_count": 0,
                "normalized_path_count": 0,
                "normalized_text_count": float("-inf"),
                "invalid_field_count": 0,
                "structured_output": {
                    "repair_attempts": float("-inf"),
                    "validation_retries": 0,
                    "final_failures": 0,
                },
                "provider_error": {"failures": float("inf")},
            },
        },
    )

    assert "0 valid, 0 repaired, 0 normalized, 0 invalid; 0 repair attempt(s), 0 retry/retries, 0 final failure(s)" in content
    assert "0 confidence, 0 score, 0 default, 0 dropped finding(s), 0 invalid field(s); 0 path(s) normalized, 0 text(s) normalized" in content
    assert "| Provider Runtime | Local model/provider execution health | 0 provider failure(s) | **PASS** |" in content
    assert "| Review Coverage | Changed units reviewed by the evaluator | 1/0 unit(s) reviewed across 1 batch(es); 0 skipped | **WARN** |" in content


def test_write_unified_report_handles_malformed_nested_telemetry_objects(tmp_path):
    content = render_unified_report(
        tmp_path,
        validation_summary={
            "valid_units": 1,
            "repaired_units": 0,
            "normalized_units": 0,
            "invalid_units": 0,
            "details": "not-a-dict",
        },
    )

    assert "1 valid, 0 repaired, 0 normalized, 0 invalid" in content
    assert "0 confidence, 0 score, 0 default, 0 dropped finding(s), 0 invalid field(s); 0 path(s) normalized, 0 text(s) normalized" in content
    assert "| Provider Runtime | Local model/provider execution health | 0 provider failure(s) | **PASS** |" in content

    content = render_unified_report(
        tmp_path,
        validation_summary={
            "valid_units": 1,
            "repaired_units": 0,
            "normalized_units": 0,
            "invalid_units": 0,
            "details": {
                "structured_output": "not-a-dict",
                "provider_error": "not-a-dict",
            },
        },
    )

    assert "1 valid, 0 repaired, 0 normalized, 0 invalid" in content
    assert "| Provider Runtime | Local model/provider execution health | 0 provider failure(s) | **PASS** |" in content


def test_write_unified_report_keeps_legacy_reports_without_validation_summary(tmp_path):
    content = render_unified_report(tmp_path)

    assert "| Reviewer Quality | Model judgment over reviewed code | N/A (no Python code changes) | **PASS** |" in content
    assert "Structured Output Health" not in content
    assert "Provider Runtime" not in content


def test_write_unified_report_displays_run_context(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567890")

    content = render_unified_report(tmp_path)

    assert "### Run Context" in content
    assert "- GitHub run: `123456`" in content
    assert "- Attempt: `2`" in content
    assert "- Commit: `abcdef123456`" in content


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

    content = render_unified_report(
        tmp_path,
        unified_verdict="FAIL",
        reasons=["Code review quality warning", "API Compatibility regressions"],
        cr_units=cr_units,
        cr_verdict="WARN",
        farley_data=farley_report_data(
            psum={"avg_index": 7.5},
            delta=-0.5,
            regressions=farley_regressions,
        ),
        compat_data=compat_report_data(ok=False, score=8.0, regressions=compat_regressions),
    )
    assert "# MSEC Unified Quality Report" in content
    assert "Verdict: **FAIL**" in content
    assert "Code review quality warning" in content
    assert "Fix this function" in content
    assert "test_one" in content
    assert "Removed method foo" in content
    assert "FAIL (Score: 8.0/10)" in content


def test_write_unified_report_null_values(tmp_path):
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

    content = render_unified_report(
        tmp_path,
        unified_verdict="FAIL",
        reasons=["Code review quality warning", "API Compatibility regressions"],
        cr_units=cr_units,
        cr_verdict="WARN",
        farley_data=farley_report_data(
            psum={"avg_index": 7.5},
            delta=-0.5,
            regressions=farley_regressions,
        ),
        compat_data=compat_report_data(ok=False, score=None, regressions=compat_regressions),
    )
    assert "# MSEC Unified Quality Report" in content
    assert "Score: 10.0/10" in content


def test_write_unified_report_with_findings(tmp_path):
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
                        "reference_principle": "Catch parser errors at the boundary before parsing raw strings.",
                        "recommended_action": "Wrap statement in a try-except block."
                    }
                ]
            }
        }
    ]

    content = render_unified_report(
        tmp_path,
        unified_verdict="FAIL",
        reasons=["Code review quality warning"],
        cr_units=cr_units,
        cr_verdict="WARN",
        farley_data=farley_report_data(
            psum={"avg_index": 7.5},
            delta=-0.5,
        ),
    )
    assert "##### Structured Engineering Findings" in content
    assert "| Category | Severity | Confidence | Finding | Location | Principle | Consequence | Recommendation |" in content
    assert "Uncaught ValueError" in content
    assert "HIGH" in content
    assert "Lines 15-20" in content
    assert "Catch parser errors at the boundary before parsing raw strings." in content
    assert "Uncaught exceptions will crash the program." in content
    assert "Wrap statement in a try-except block." in content


def test_write_unified_report_displays_effective_severity(tmp_path):
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

    content = render_unified_report(tmp_path, cr_units=cr_units)
    assert "| tests/test_code_review.py | test_unsubstantiated_block | INVALID (MISSING_DIMENSION) | **WARN** | OK |" in content
    assert "**BLOCK**" not in content


def test_write_unified_report_marks_invalid_cqi_metric_failed(tmp_path):
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

    content = render_unified_report(
        tmp_path,
        unified_verdict="FAIL",
        reasons=["INVALID CQI: src/utils.py -> invalid_review: Required CQI dimension 'readability' is missing."],
        cr_units=cr_units,
    )
    assert "| Reviewer Quality | Model judgment over reviewed code | INVALID (1/1 unit(s)) | **FAIL** |" in content
    assert "| src/utils.py | invalid_review | INVALID (MISSING_DIMENSION) | **OK** | Incomplete review |" in content


def test_write_unified_report_ignores_malformed_cqi_units(tmp_path):
    content = render_unified_report(tmp_path, cr_units=["not-a-unit"])
    assert "| Reviewer Quality | Model judgment over reviewed code | N/A (no Python code changes) | **PASS** |" in content


def test_write_unified_report_accepts_legacy_review_payload(tmp_path, valid_review_payload):
    cr_units = get_reviews({"reviews": [valid_review_payload()]})

    content = render_unified_report(tmp_path, cr_units=cr_units)
    assert "| Reviewer Quality | Model judgment over reviewed code | 8.00/10 (average of 1 unit(s)) | **PASS** |" in content
    assert "| legacy-cassette | legacy_review_1 | 8.00/10 | **OK** | legacy review |" in content


def test_write_unified_report_excludes_recoverable_failure_from_cqi(tmp_path, valid_review_payload):
    cr_units = get_reviews({
        "reviews": [
            valid_review_payload(),
            {
                "file_path": "scripts/code_review_compare.py",
                "name": "<module>",
                "review": None,
                "recoverable_failure": {
                    "type": "structured_output",
                    "message": "Failed to validate structured output",
                },
            },
        ],
    })

    content = render_unified_report(tmp_path, cr_units=cr_units)
    assert (
        "| Reviewer Quality | Model judgment over reviewed code | 8.00/10 "
        "(average of 1 unit(s); 1 recoverable failure(s) excluded) | **PASS** |"
    ) in content
    assert "| scripts/code_review_compare.py | &lt;module&gt; | N/A (recoverable failure) | **N/A** | Failed to validate structured output |" in content


def test_write_unified_report_displays_provider_error_units(tmp_path, valid_review_payload):
    cr_units = get_reviews({
        "reviews": [
            valid_review_payload(),
            {
                "file_path": "scripts/unified_compare.py",
                "name": "review_summary",
                "review": None,
                "provider_error": {
                    "type": "provider_error",
                    "message": "OllamaException - llama-server process has terminated",
                    "recoverable": True,
                },
                "recoverable_failure": {
                    "type": "provider_error",
                    "message": "OllamaException - llama-server process has terminated",
                },
            },
        ],
    })

    content = render_unified_report(tmp_path, cr_units=cr_units)
    assert (
        "| Reviewer Quality | Model judgment over reviewed code | 8.00/10 "
        "(average of 1 unit(s); 1 recoverable failure(s) excluded) | **PASS** |"
    ) in content
    assert (
        "| scripts/unified_compare.py | review_summary | N/A (provider error) "
        "| **PROVIDER_ERROR** | OllamaException - llama-server process has terminated |"
    ) in content


def test_write_unified_report_uses_summary_fallback_for_empty_summary(tmp_path, valid_review_payload):
    review = valid_review_payload()
    review["summary"] = "   "
    cr_units = [{
        "file_path": "src/summary.py",
        "name": "missing_summary",
        "review": review,
    }]

    content = render_unified_report(tmp_path, cr_units=cr_units)
    assert "| src/summary.py | missing_summary | 8.00/10 | **OK** | No summary provided |" in content
    assert "_Review summary fallback used for 1 unit(s)._" in content


def test_write_unified_report_counts_empty_summaries_without_recoverable_failures(tmp_path, valid_review_payload):
    empty_review = valid_review_payload()
    empty_review["summary"] = ""
    useful_review = valid_review_payload()
    useful_review["summary"] = "Useful review"

    cr_units = [
        {
            "file_path": "src/empty.py",
            "name": "empty_summary",
            "review": empty_review,
        },
        {
            "file_path": "src/useful.py",
            "name": "useful_summary",
            "review": useful_review,
        },
        {
            "file_path": "src/recoverable.py",
            "name": "recoverable",
            "review": None,
            "recoverable_failure": {
                "type": "structured_output",
                "message": "Failed to validate structured output",
            },
        },
    ]

    content = render_unified_report(tmp_path, cr_units=cr_units)
    assert "_Review summary fallback used for 1 unit(s)._" in content
    assert "| src/empty.py | empty_summary | 8.00/10 | **OK** | No summary provided |" in content
    assert "| src/useful.py | useful_summary | 8.00/10 | **OK** | Useful review |" in content
    assert "| src/recoverable.py | recoverable | N/A (recoverable failure) | **N/A** | Failed to validate structured output |" in content


def test_write_unified_report_displays_baseline_state(tmp_path):
    """A required missing Farley baseline must fail the gate and explain the missing input."""
    expected_reason = "Farley baseline is required but was not found."
    missing_baseline_details = ["missing baseline.json"]
    missing_required_baseline = farley_report_data(
        bsum={"avg_index": 0.0},
        baseline_exists=False,
        baseline_state="BASELINE_MISSING",
        baseline_reason=expected_reason,
        baseline_details=missing_baseline_details,
    )

    content = render_unified_report(
        tmp_path,
        unified_verdict="FAIL",
        reasons=[f"FARLEY BASELINE: {expected_reason}"],
        farley_data=missing_required_baseline,
        compat_data=compat_report_data(state="PASS"),
    )

    assert "| Test Quality | Farley comparison against baseline | PR avg: 8.00 (baseline missing) | **FAIL** |" in content
    assert "State: **BASELINE_MISSING**" in content
    assert "missing baseline.json" in content


def test_write_unified_report_displays_compatibility_check_failed(tmp_path):
    content = render_unified_report(
        tmp_path,
        unified_verdict="FAIL",
        reasons=["API COMPATIBILITY CHECK_FAILED: Parser failed."],
        farley_data=farley_report_data(
            baseline_state="AVAILABLE",
        ),
        compat_data=compat_report_data(
            ok=False,
            score=0.0,
            state="CHECK_FAILED",
            reason="Parser failed.",
            details=["bad syntax"],
        ),
    )

    assert "| API Compatibility | Public API compatibility state | CHECK_FAILED (Score: 0.0/10) | **FAIL** |" in content
    assert "State: **CHECK_FAILED**" in content
    assert "bad syntax" in content


def test_combine_token_spend_handles_malformed_types():
    # String totals, float string total_tokens, none values, string cost, boolean calls
    cr = {
        "__metadata__": {
            "code_review_usage_summary": {
                "usage": {
                    "totals": {
                        "calls": "2",
                        "prompt_tokens": "1000",
                        "completion_tokens": None,
                        "total_tokens": "1200.0",
                        "cost_usd": "0.012",
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
                        "calls": True, # Should coerce to 0
                        "prompt_tokens": "banana", # Should coerce to 0
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
    assert "1000 prompt" in res
    assert "100 completion" in res
    assert "2 LLM call(s)" in res
    assert "0.018000" in res


def test_format_finding_confidence_advanced_parsing():
    from scripts.unified_compare import _format_finding_confidence, _format_finding_confidence_strict
    # Lenient helper (schema-like) tests
    assert _format_finding_confidence("85%") == "HIGH"
    assert _format_finding_confidence("4/5") == "HIGH"
    assert _format_finding_confidence("nan") == "MEDIUM"
    assert _format_finding_confidence(float("nan")) == "MEDIUM"
    assert _format_finding_confidence(None) == "UNKNOWN"
    assert _format_finding_confidence(True) == "UNKNOWN"
    assert _format_finding_confidence({"nested": "value"}) == "UNKNOWN"

    # Strict helper tests
    assert _format_finding_confidence_strict("85%") == "HIGH"
    assert _format_finding_confidence_strict("4/5") == "HIGH"
    assert _format_finding_confidence_strict("nan") == "UNKNOWN"
    assert _format_finding_confidence_strict(float("nan")) == "UNKNOWN"
    assert _format_finding_confidence_strict(float("inf")) == "UNKNOWN"
    assert _format_finding_confidence_strict("banana") == "UNKNOWN"
    assert _format_finding_confidence_strict("certain") == "UNKNOWN"
    assert _format_finding_confidence_strict(None) == "UNKNOWN"
    assert _format_finding_confidence_strict(True) == "UNKNOWN"
    assert _format_finding_confidence_strict({"nested": "value"}) == "UNKNOWN"


def test_calculate_confidence_distribution():
    from scripts.unified_compare import calculate_confidence_distribution

    # Empty list
    assert calculate_confidence_distribution([]) == {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}

    # Non-list inputs (defensive check)
    assert calculate_confidence_distribution(None) == {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    assert calculate_confidence_distribution("not-a-list") == {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}

    # Recoverable failure or non-dict unit
    assert calculate_confidence_distribution([
        None,
        {"recoverable_failure": {"type": "provider_error"}},
    ]) == {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}

    # Units with findings
    units = [
        {
            "review": {
                "findings": [
                    {"confidence": "HIGH"},
                    {"confidence": "certain"}, # strict -> UNKNOWN
                    {"confidence": 0.9},       # strict -> HIGH
                ]
            }
        },
        {
            "review": {
                "findings": [
                    {"confidence": "MEDIUM"},
                    {"confidence": "4/5"},     # strict -> HIGH
                    {"confidence": "banana"},   # strict -> UNKNOWN
                ]
            }
        },
        {
            "review": {
                "findings": [
                    {"confidence": "LOW"},
                    {"confidence": 0.2},       # strict -> LOW
                    {"confidence": None},      # strict -> UNKNOWN
                    {"confidence": True},      # strict -> UNKNOWN
                ]
            }
        }
    ]

    expected = {
        "HIGH": 3,   # HIGH, 0.9, 4/5
        "MEDIUM": 1, # MEDIUM
        "LOW": 2,    # LOW, 0.2
        "UNKNOWN": 4 # certain, banana, None, True
     }
    assert calculate_confidence_distribution(units) == expected


def test_render_unified_report_confidence_distribution_empty(tmp_path):
    report_content = render_unified_report(tmp_path, cr_units=[])
    assert "### Finding Confidence Distribution" in report_content
    assert "No structured engineering findings reported in this run." in report_content


def test_render_unified_report_confidence_distribution_with_findings(tmp_path):
    cr_units = [
        {
            "review": {
                "findings": [
                    {"confidence": "HIGH"},
                    {"confidence": "MEDIUM"},
                    {"confidence": "LOW"},
                    {"confidence": "HIGH"},
                ]
            }
        }
    ]
    report_content = render_unified_report(tmp_path, cr_units=cr_units)
    assert "### Finding Confidence Distribution" in report_content
    assert "- **HIGH Confidence**: 2 finding(s) (50.0%)" in report_content
    assert "- **MEDIUM Confidence**: 1 finding(s) (25.0%)" in report_content
    assert "- **LOW Confidence**: 1 finding(s) (25.0%)" in report_content
    assert "- **UNKNOWN/UNPARSED Confidence**: 0 finding(s) (0.0%)" in report_content
    assert "*Note:" not in report_content


def test_calculate_confidence_distribution_malformed_and_non_confidence():
    from scripts.unified_compare import calculate_confidence_distribution

    # "banana" -> UNKNOWN (strict)
    # "85%" -> HIGH (strict)
    # None, True/False, and non-scalar types -> UNKNOWN
    units = [
        {
            "review": {
                "findings": [
                    {"confidence": "banana"},  # -> UNKNOWN
                    {"confidence": "85%"},     # -> HIGH
                    {"confidence": None},      # -> UNKNOWN
                    {"confidence": False},     # -> UNKNOWN
                    {"confidence": {"nested": "value"}},  # -> UNKNOWN
                    {"confidence": [1, 2]},               # -> UNKNOWN
                ]
            }
        }
    ]

    expected = {
        "HIGH": 1,
        "MEDIUM": 0,
        "LOW": 0,
        "UNKNOWN": 5
    }
    assert calculate_confidence_distribution(units) == expected


def test_calculate_authored_confidence_distribution_uses_validation_provenance():
    from scripts.unified_compare import calculate_authored_confidence_distribution

    units = [
        {
            "review": {
                "findings": [
                    {"confidence": "MEDIUM"},
                    {"confidence": "MEDIUM"},
                    {"confidence": "HIGH"},
                    {"confidence": "MEDIUM"},
                    {"confidence": "LOW"},
                ]
            },
            "validation": {
                "fields": [
                    {
                        "field_name": "findings[0].confidence",
                        "status": "REPAIRED",
                        "raw_value": None,
                        "repaired_value": "MEDIUM",
                    },
                    {
                        "field_name": "findings[1].confidence",
                        "status": "REPAIRED",
                        "raw_value": "banana",
                        "repaired_value": "MEDIUM",
                    },
                    {
                        "field_name": "findings[2].confidence",
                        "status": "REPAIRED",
                        "raw_value": "0.9",
                        "repaired_value": "HIGH",
                    },
                    {
                        "field_name": "findings[3].confidence",
                        "status": "NORMALIZED",
                        "raw_value": " medium ",
                        "repaired_value": "MEDIUM",
                    },
                ]
            },
        },
        {
            "review": {
                "findings": [
                    {"confidence": "HIGH"},
                ]
            }
        },
    ]

    assert calculate_authored_confidence_distribution(units) == {
        "HIGH": 2,
        "MEDIUM": 1,
        "LOW": 1,
        "UNKNOWN_REPAIRED": 2,
    }


def test_render_unified_report_confidence_distribution_regression(tmp_path):
    # Regression test for exact skew scenario
    cr_units = [{
        "review": {
            "findings": [
                {"confidence": "banana"},  # -> UNKNOWN
                {"confidence": "nan"},     # -> UNKNOWN
                {"confidence": None},      # -> UNKNOWN
                {"confidence": "85%"},     # -> HIGH
                {"confidence": "MEDIUM"},  # -> MEDIUM
            ]
        }
    }]
    report_content = render_unified_report(tmp_path, cr_units=cr_units)
    assert "### Finding Confidence Distribution" in report_content
    assert "- **HIGH Confidence**: 1 finding(s) (20.0%)" in report_content
    assert "- **MEDIUM Confidence**: 1 finding(s) (20.0%)" in report_content
    assert "- **LOW Confidence**: 0 finding(s) (0.0%)" in report_content
    assert "- **UNKNOWN/UNPARSED Confidence**: 3 finding(s) (60.0%)" in report_content


def test_render_unified_report_authored_confidence_signal_shows_repaired_skew(tmp_path):
    cr_units = [{
        "review": {
            "findings": [
                {"confidence": "MEDIUM"},
                {"confidence": "MEDIUM"},
                {"confidence": "MEDIUM"},
            ]
        },
        "validation": {
            "fields": [
                {
                    "field_name": "findings[0].confidence",
                    "status": "REPAIRED",
                    "raw_value": None,
                    "repaired_value": "MEDIUM",
                },
                {
                    "field_name": "findings[1].confidence",
                    "status": "REPAIRED",
                    "raw_value": "banana",
                    "repaired_value": "MEDIUM",
                },
                {
                    "field_name": "findings[2].confidence",
                    "status": "REPAIRED",
                    "raw_value": "0.5",
                    "repaired_value": "MEDIUM",
                },
            ]
        },
    }]

    report_content = render_unified_report(tmp_path, cr_units=cr_units)

    assert "### Finding Confidence Distribution (Post-Schema)" in report_content
    assert "- **MEDIUM Confidence**: 3 finding(s) (100.0%)" in report_content
    assert "### Authored Confidence Signal" in report_content
    assert "- **HIGH Authored**: 0 finding(s) (0.0%)" in report_content
    assert "- **MEDIUM Authored**: 1 finding(s) (33.3%)" in report_content
    assert "- **LOW Authored**: 0 finding(s) (0.0%)" in report_content
    assert "- **UNKNOWN/REPAIRED Authored**: 2 finding(s) (66.7%)" in report_content
