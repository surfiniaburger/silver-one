#!/usr/bin/env python3
import argparse
import html
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Optional, Tuple

# Enable relative imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.path_utils import validate_input_path, validate_output_path
from scripts.finding_schema import BaselineCheckResult, CompatibilityCheckResult
from scripts.farley_compare import (
    compute_suite_summary as farley_compute_summary,
    merge_virtual_suite,
    compute_drops_and_pct,
    determine_verdict_and_reasons as farley_determine_verdict,
    top_regressions as farley_top_regressions,
)
from scripts.code_review_compare import (
    calculate_cqi_result,
    collect_cqi_failure_reasons,
    effective_review_severity,
    format_cqi_result,
    get_reviews,
    group_units_by_severity,
    is_recoverable_review_failure,
    determine_verdict as cr_determine_verdict,
    format_unit_name,
    write_dimension_rows,
)

# Project roots and limits
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASSETTE_ROOT = (PROJECT_ROOT / "artifacts" / "cassettes").resolve()
METRICS_ROOT = (PROJECT_ROOT / "artifacts" / "metrics").resolve()
REPORT_ROOT = (PROJECT_ROOT / "reports").resolve()

# Allowed file extensions
JSON_EXT = frozenset({".json"})
MD_EXT = frozenset({".md"})
EMPTY_SUMMARY_FALLBACK = "No summary provided"


def safe_load_json(path: Path) -> Dict[str, Any]:
    """Helper to safely load a JSON file, returning empty dict if missing."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load JSON from {path}: {e}", file=sys.stderr)
        return {}


def load_json_strict(path: Path) -> Dict[str, Any]:
    """Load JSON and raise when the file is missing, malformed, or not an object."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def resolve_farley_baseline(
    baseline_arg: str,
    required: bool,
) -> Tuple[BaselineCheckResult, Path | None, Dict[str, Any]]:
    """Resolve and load the Farley baseline with explicit baseline semantics."""
    try:
        baseline_path = validate_input_path(baseline_arg, PROJECT_ROOT)
    except ValueError as exc:
        if required:
            return (
                BaselineCheckResult(
                    state="BASELINE_MISSING",
                    available=False,
                    required=True,
                    reason="Farley baseline is required but was not found.",
                    details=[str(exc)],
                ),
                None,
                {"tests": []},
            )
        return (
            BaselineCheckResult(
                state="FIRST_RUN",
                available=False,
                required=False,
                reason="No Farley baseline exists yet; first-run mode is active.",
                details=[str(exc)],
            ),
            None,
            {"tests": []},
        )

    try:
        baseline_data = load_json_strict(baseline_path)
    except Exception as exc:
        return (
            BaselineCheckResult(
                state="BASELINE_CORRUPTED",
                available=False,
                required=required,
                reason="Farley baseline exists but could not be loaded as valid JSON.",
                details=[str(exc)],
            ),
            baseline_path,
            {"tests": []},
        )

    tests = baseline_data.get("tests")
    if not isinstance(tests, list):
        return (
            BaselineCheckResult(
                state="BASELINE_CORRUPTED",
                available=False,
                required=required,
                reason="Farley baseline exists but does not contain a tests list.",
                details=[f"{baseline_path}: missing or invalid 'tests' field."],
            ),
            baseline_path,
            {"tests": []},
        )

    return (
        BaselineCheckResult(
            state="AVAILABLE",
            available=True,
            required=required,
            reason="Farley baseline is available.",
        ),
        baseline_path,
        baseline_data,
    )


def parse_compatibility_result(compat_results: Dict[str, Any]) -> CompatibilityCheckResult:
    """Normalize compatibility result JSON into the explicit state model."""
    regressions = compat_results.get("regressions")
    if not isinstance(regressions, list):
        regressions = []

    details = compat_results.get("details")
    if not isinstance(details, list):
        details = []

    state = compat_results.get("state")
    if state in {"PASS", "FAIL", "NOT_EXECUTED", "CHECK_FAILED"}:
        compatible = compat_results.get("pass")
        if compatible is None:
            compatible = state in {"PASS", "NOT_EXECUTED"}
    else:
        compatible = compat_results.get("pass") if compat_results.get("pass") is not None else True
        state = "PASS" if compatible else "FAIL"

    raw_score = compat_results.get("compatibility_index")
    default_score = 0.0
    if compatible:
        default_score = 10.0
    score = raw_score if raw_score is not None else default_score

    return CompatibilityCheckResult(
        state=state,
        compatible=bool(compatible),
        score=float(score),
        regressions=[str(item) for item in regressions],
        reason=compat_results.get("reason"),
        details=[str(item) for item in details],
    )


def get_token_totals(cassette: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Extract token spend totals from a cassette's metadata."""
    if not isinstance(cassette, dict):
        return {}
    metadata = cassette.get("__metadata__")
    if not isinstance(metadata, dict):
        return {}
    summary = metadata.get(key)
    if not isinstance(summary, dict):
        return {}
    usage = summary.get("usage")
    if not isinstance(usage, dict):
        return {}
    totals = usage.get("totals")
    if not isinstance(totals, dict):
        return {}
    return totals


def get_code_review_coverage(cassette: Dict[str, Any]) -> Dict[str, Any]:
    """Extract review coverage telemetry from a code-review cassette."""
    if not isinstance(cassette, dict):
        return {}
    metadata = cassette.get("__metadata__")
    if not isinstance(metadata, dict):
        return {}
    summary = metadata.get("code_review_usage_summary")
    if not isinstance(summary, dict):
        return {}
    coverage = summary.get("review_coverage")
    if not isinstance(coverage, dict):
        return {}
    return coverage


def get_code_review_validation_summary(cassette: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured-output validation telemetry from a code-review cassette."""
    if not isinstance(cassette, dict):
        return {}
    metadata = cassette.get("__metadata__")
    if not isinstance(metadata, dict):
        return {}
    summary = metadata.get("code_review_usage_summary")
    if not isinstance(summary, dict):
        return {}
    validation = summary.get("validation_summary")
    if not isinstance(validation, dict):
        return {}
    return validation


def combine_token_spend(cr_cassette: Dict[str, Any], farley_cassette: Dict[str, Any]) -> str:
    """Summarize total token spend across both Code Review and Farley runs."""
    cr_totals = get_token_totals(cr_cassette, "code_review_usage_summary")
    farley_totals = get_token_totals(farley_cassette, "farley_usage_summary")

    total_tokens = int((cr_totals.get("total_tokens") or 0) + (farley_totals.get("total_tokens") or 0))
    prompt_tokens = int((cr_totals.get("prompt_tokens") or 0) + (farley_totals.get("prompt_tokens") or 0))
    comp_tokens = int((cr_totals.get("completion_tokens") or 0) + (farley_totals.get("completion_tokens") or 0))
    calls = int((cr_totals.get("calls") or 0) + (farley_totals.get("calls") or 0))
    cost = float((cr_totals.get("cost_usd") or 0.0) + (farley_totals.get("cost_usd") or 0.0))

    if total_tokens == 0:
        return "**Token spend**: 0 total tokens (all replayed via cassettes)\n\n"

    return (
        "**Combined Token Spend**: "
        f"{total_tokens} total tokens "
        f"({prompt_tokens} prompt, "
        f"{comp_tokens} completion) "
        f"across {calls} LLM call(s); "
        f"estimated cost ${cost:.6f}\n\n"
    )


def format_suggestions_short(suggestions: Any) -> str:
    if not suggestions:
        return "None"
    return " ".join([f"- {s}" for s in suggestions])


def _write_header_and_summary(f, unified_verdict: str, reasons: List[str], spend_str: str) -> None:
    f.write("# MSEC Unified Quality Report\n\n")
    status_color = "🔴" if unified_verdict == "FAIL" else "🟢"
    f.write(f"## {status_color} Verdict: **{unified_verdict}**\n\n")

    _write_run_context(f)

    if reasons:
        f.write("### Summary Notes\n")
        for reason in reasons:
            f.write(f"- {reason}\n")
        f.write("\n")

    f.write(spend_str)


def _write_run_context(f) -> None:
    run_id = os.getenv("GITHUB_RUN_ID")
    run_attempt = os.getenv("GITHUB_RUN_ATTEMPT")
    sha = os.getenv("GITHUB_SHA")
    if not any([run_id, run_attempt, sha]):
        return

    f.write("### Run Context\n")
    if run_id:
        f.write(f"- GitHub run: `{run_id}`\n")
    if run_attempt:
        f.write(f"- Attempt: `{run_attempt}`\n")
    if sha:
        f.write(f"- Commit: `{sha[:12]}`\n")
    f.write("\n")


def _get_cqi_metric(cr_data: Dict[str, Any]) -> Tuple[str, str]:
    """Build CQI metric and status values."""
    cr_units = [
        unit for unit in (cr_data.get("units") or [])
        if isinstance(unit, dict)
    ]
    cr_verdict = cr_data.get("verdict") or "PASS"
    if not cr_units:
        return "N/A (no Python code changes)", "PASS"

    cqi_units = [
        unit for unit in cr_units
        if not is_recoverable_review_failure(unit)
    ]
    recoverable_count = len(cr_units) - len(cqi_units)
    if not cqi_units:
        return f"N/A ({recoverable_count} recoverable evaluation failure(s))", "PASS"

    cqi_results = [calculate_cqi_result(u.get("review") or {}) for u in cqi_units]
    invalid_count = sum(1 for result in cqi_results if not result.valid)
    if invalid_count:
        return f"INVALID ({invalid_count}/{len(cqi_units)} unit(s))", "FAIL"

    scores = [result.value for result in cqi_results if result.value is not None]
    avg_cqi = mean(scores) if scores else 10.0
    if recoverable_count:
        return (
            f"{avg_cqi:.2f}/10 (average of {len(cqi_units)} unit(s); "
            f"{recoverable_count} recoverable failure(s) excluded)"
        ), cr_verdict
    return f"{avg_cqi:.2f}/10 (average of {len(cr_units)} unit(s))", cr_verdict


def _get_farley_metric(farley_data: Dict[str, Any]) -> Tuple[str, str]:
    """Build Farley metric and status values."""
    baseline_state = farley_data.get("baseline_state") or ("AVAILABLE" if farley_data.get("baseline_exists") else "FIRST_RUN")
    farley_baseline_exists = baseline_state == "AVAILABLE"
    farley_bsum = farley_data.get("bsum") or {}
    farley_psum = farley_data.get("psum") or {}
    farley_delta = farley_data.get("delta") if farley_data.get("delta") is not None else 0.0
    farley_verdict = farley_data.get("verdict") or "PASS"

    farley_bsum_avg = farley_bsum.get("avg_index") if farley_bsum.get("avg_index") is not None else 0.0
    farley_psum_avg = farley_psum.get("avg_index") if farley_psum.get("avg_index") is not None else 0.0

    if farley_baseline_exists:
        return (
            f"Baseline avg: {farley_bsum_avg:.2f} \\| PR avg: {farley_psum_avg:.2f} \\| Delta: {farley_delta:+.2f}",
            farley_verdict,
        )
    if baseline_state == "BASELINE_MISSING":
        return f"PR avg: {farley_psum_avg:.2f} (baseline missing)", "FAIL"
    if baseline_state == "BASELINE_CORRUPTED":
        return f"PR avg: {farley_psum_avg:.2f} (baseline corrupted)", "FAIL"
    return f"PR avg: {farley_psum_avg:.2f} (no baseline to compare)", farley_verdict


def _get_compat_metric(compat_data: Dict[str, Any]) -> Tuple[str, str]:
    """Build API Compatibility metric and status values."""
    state = compat_data.get("state")
    compat_score = compat_data.get("score") if compat_data.get("score") is not None else 10.0
    compat_ok = compat_data.get("ok") if compat_data.get("ok") is not None else True
    if state == "NOT_EXECUTED":
        return "NOT_EXECUTED (no changed non-test Python source)", "PASS"
    if state == "CHECK_FAILED":
        return f"CHECK_FAILED (Score: {compat_score:.1f}/10)", "FAIL"
    verdict = "PASS" if compat_ok else "FAIL"
    return f"{state or verdict} (Score: {compat_score:.1f}/10)", verdict


def _coerce_metric_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _get_code_review_coverage_metric(cr_data: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    coverage = cr_data.get("review_coverage")
    if not isinstance(coverage, dict):
        return None

    total = _coerce_metric_int(coverage.get("total_extracted_units"))
    reviewed = _coerce_metric_int(coverage.get("reviewed_units"))
    skipped = _coerce_metric_int(coverage.get("skipped_units"))
    batches = _coerce_metric_int(coverage.get("batch_count"))
    if total == 0 and reviewed == 0 and batches == 0:
        return None

    metric = f"{reviewed}/{total} unit(s) reviewed across {batches} batch(es); {skipped} skipped"
    status = "PASS" if skipped == 0 and reviewed <= total else "WARN"
    return metric, status


def _get_structured_output_metric(cr_data: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    summary = cr_data.get("validation_summary")
    if not isinstance(summary, dict) or not summary:
        return None

    details = summary.get("details")
    if not isinstance(details, dict):
        details = {}
    structured = details.get("structured_output")
    if not isinstance(structured, dict):
        structured = {}

    valid = _coerce_metric_int(summary.get("valid_units"))
    repaired = _coerce_metric_int(summary.get("repaired_units"))
    normalized = _coerce_metric_int(summary.get("normalized_units"))
    invalid = _coerce_metric_int(summary.get("invalid_units"))
    repair_attempts = _coerce_metric_int(structured.get("repair_attempts"))
    retries = _coerce_metric_int(structured.get("validation_retries"))
    final_failures = _coerce_metric_int(structured.get("final_failures"))

    metric = (
        f"{valid} valid, {repaired} repaired, {normalized} normalized, "
        f"{invalid} invalid; {repair_attempts} repair attempt(s), "
        f"{retries} retry/retries, {final_failures} final failure(s)"
    )
    status = "FAIL" if final_failures or invalid else "PASS"
    return metric, status


def _get_provider_runtime_metric(cr_data: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    summary = cr_data.get("validation_summary")
    if not isinstance(summary, dict) or not summary:
        return None

    details = summary.get("details")
    if not isinstance(details, dict):
        details = {}
    provider = details.get("provider_error")
    if not isinstance(provider, dict):
        provider = {}
    failures = _coerce_metric_int(provider.get("failures"))
    status = "FAIL" if failures else "PASS"
    return f"{failures} provider failure(s)", status


def _write_metrics_overview(
    f,
    cr_data: Dict[str, Any],
    farley_data: Dict[str, Any],
    compat_data: Dict[str, Any],
) -> None:
    f.write("## Metrics Overview\n\n")
    f.write("| Lane | What It Measures | Metric / Score | Status |\n")
    f.write("|---|---|---|---|\n")

    cqi_metric, cqi_status = _get_cqi_metric(cr_data)
    f.write(
        f"| Reviewer Quality | Model judgment over reviewed code "
        f"| {cqi_metric} | **{cqi_status}** |\n"
    )

    structured_metric = _get_structured_output_metric(cr_data)
    if structured_metric:
        metric, status = structured_metric
        f.write(
            f"| Structured Output Health | Schema validity, repairs, retries, final failures "
            f"| {metric} | **{status}** |\n"
        )

    provider_metric = _get_provider_runtime_metric(cr_data)
    if provider_metric:
        metric, status = provider_metric
        f.write(
            f"| Provider Runtime | Local model/provider execution health "
            f"| {metric} | **{status}** |\n"
        )

    coverage_metric = _get_code_review_coverage_metric(cr_data)
    if coverage_metric:
        metric, status = coverage_metric
        f.write(
            f"| Review Coverage | Changed units reviewed by the evaluator "
            f"| {metric} | **{status}** |\n"
        )

    farley_metric, farley_status = _get_farley_metric(farley_data)
    f.write(
        f"| Test Quality | Farley comparison against baseline "
        f"| {farley_metric} | **{farley_status}** |\n"
    )

    compat_metric, compat_status = _get_compat_metric(compat_data)
    f.write(
        f"| API Compatibility | Public API compatibility state "
        f"| {compat_metric} | **{compat_status}** |\n\n"
    )


def _esc(value: Any) -> str:
    text = html.escape(str(value or ""))
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _recoverable_failure_summary(unit: Dict[str, Any]) -> str:
    failure = unit.get("recoverable_failure") or {}
    return failure.get("message") or failure.get("type") or "Recoverable evaluation failure"


def _recoverable_failure_cqi_label(unit: Dict[str, Any]) -> str:
    failure = unit.get("recoverable_failure") or {}
    if failure.get("type") == "provider_error":
        return "N/A (provider error)"
    return "N/A (recoverable failure)"


def _recoverable_failure_severity_label(unit: Dict[str, Any]) -> str:
    failure = unit.get("recoverable_failure") or {}
    if failure.get("type") == "provider_error":
        return "PROVIDER_ERROR"
    return "N/A"


def review_summary(review: Any) -> str:
    if not isinstance(review, dict):
        return EMPTY_SUMMARY_FALLBACK
    summary = review.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return EMPTY_SUMMARY_FALLBACK


def count_empty_review_summaries(cr_units: List[Dict[str, Any]]) -> int:
    count = 0
    for unit in cr_units:
        if not isinstance(unit, dict) or is_recoverable_review_failure(unit):
            continue
        review = unit.get("review") or {}
        if review_summary(review) == EMPTY_SUMMARY_FALLBACK:
            count += 1
    return count


def _write_cr_units_overview(f, cr_units: List[Dict[str, Any]]) -> None:
    f.write("### Code Units Overview\n\n")
    f.write("| File | Unit | CQI | Severity | Summary |\n")
    f.write("|---|---|---:|---|---|\n")
    for unit in cr_units:
        if is_recoverable_review_failure(unit):
            f.write(
                f"| {_esc(unit.get('file_path'))} "
                f"| {_esc(format_unit_name(unit))} "
                f"| {_esc(_recoverable_failure_cqi_label(unit))} "
                f"| **{_esc(_recoverable_failure_severity_label(unit))}** "
                f"| {_esc(_recoverable_failure_summary(unit))} |\n"
            )
            continue
        review = unit.get("review") or {}
        severity = effective_review_severity(review)
        f.write(
            f"| {_esc(unit.get('file_path'))} "
            f"| {_esc(format_unit_name(unit))} "
            f"| {_esc(format_cqi_result(review))} "
            f"| **{_esc(severity)}** "
            f"| {_esc(review_summary(review))} |\n"
        )
    f.write("\n")


def _write_cr_findings_table(f, findings: List[Dict[str, Any]]) -> None:
    if not findings:
        return
    f.write("\n##### Structured Engineering Findings\n\n")
    f.write("| Category | Severity | Finding | Location | Consequence | Recommendation |\n")
    f.write("|---|---|---|---|---|---|\n")
    for finding in findings:
        title = _esc(finding.get("title", ""))
        cat = _esc(finding.get("category", ""))
        sev = _esc(finding.get("severity", "INFO"))
        conseq = _esc(finding.get("engineering_consequence", ""))
        recom = _esc(finding.get("recommended_action", ""))
        
        evidence = finding.get("evidence") or {}
        details = evidence.get("details") or {}
        start_line = details.get("start_line")
        end_line = details.get("end_line")
        loc = "General"
        if start_line is not None and end_line is not None:
            loc = f"Lines {start_line}-{end_line}"
        elif start_line is not None:
            loc = f"Line {start_line}"
            
        f.write(f"| {cat} | **{sev}** | {title} | {_esc(loc)} | {conseq} | {recom} |\n")
    f.write("\n")


def _write_code_review_details(f, cr_data: Dict[str, Any]) -> None:
    cr_units = [
        unit for unit in cr_data.get("units", [])
        if isinstance(unit, dict)
    ]
    if not cr_units:
        return
    f.write("## 🔍 Code Review Findings\n\n")
    f.write("<details>\n<summary>Click to view detailed Code Review feedback</summary>\n\n")
    block_units, warn_units, _ = group_units_by_severity(cr_units)
    empty_summary_count = count_empty_review_summaries(cr_units)
    if empty_summary_count:
        f.write(f"_Review summary fallback used for {empty_summary_count} unit(s)._\n\n")

    _write_cr_units_overview(f, cr_units)

    problematic = block_units + warn_units
    if problematic:
        f.write("### Detailed Feedback\n\n")
        for unit in problematic:
            review = unit.get("review") or {}
            severity = effective_review_severity(review)
            f.write(f"#### ⚠️ `{unit.get('file_path')}` -> `{format_unit_name(unit)}` ({severity})\n")
            f.write(f"**CQI**: {format_cqi_result(review)}\n\n")
            f.write(f"{review_summary(review)}\n\n")
            f.write("| Dimension | Score | Rationale | Suggestions |\n")
            f.write("|---|---|---|---|\n")
            write_dimension_rows(f, review)

            _write_cr_findings_table(f, review.get("findings"))

            f.write("\n---\n\n")
    f.write("</details>\n\n")


def _write_farley_details(f, farley_data: Dict[str, Any]) -> None:
    baseline_state = farley_data.get("baseline_state") or ("AVAILABLE" if farley_data.get("baseline_exists") else "FIRST_RUN")
    baseline_reason = farley_data.get("baseline_reason")
    if baseline_state in {"FIRST_RUN", "BASELINE_MISSING", "BASELINE_CORRUPTED"}:
        f.write("## 🧪 Farley Baseline State\n\n")
        f.write(f"- State: **{baseline_state}**\n")
        if baseline_reason:
            f.write(f"- Reason: {baseline_reason}\n")
        for detail in farley_data.get("baseline_details", []):
            f.write(f"- Detail: {detail}\n")
        f.write("\n")

    farley_baseline_exists = baseline_state == "AVAILABLE"
    farley_regressions = farley_data.get("regressions", [])
    if farley_baseline_exists and farley_regressions:
        f.write("## 🧪 Farley Test Regressions\n\n")
        f.write("<details>\n<summary>Click to view test score drops</summary>\n\n")
        f.write("| Delta | File | Test | Base | PR |\n")
        f.write("|---|---|---|---:|---:|\n")
        for delta_val, base_test, pr_test in farley_regressions:
            f.write(
                f"| {delta_val:.2f} "
                f"| {base_test.get('file_path', '')} "
                f"| {base_test.get('test_name', '')} "
                f"| {base_test.get('farley_index') or 0.0:.2f} "
                f"| {pr_test.get('farley_index') or 0.0:.2f} |\n"
            )
        f.write("\n</details>\n\n")


def _write_compatibility_details(f, compat_data: Dict[str, Any]) -> None:
    state = compat_data.get("state")
    reason = compat_data.get("reason")
    details = compat_data.get("details") or []
    if state in {"NOT_EXECUTED", "CHECK_FAILED"}:
        f.write("## 🔌 API Compatibility State\n\n")
        f.write(f"- State: **{state}**\n")
        if reason:
            f.write(f"- Reason: {reason}\n")
        for detail in details:
            f.write(f"- Detail: {detail}\n")
        f.write("\n")

    compat_regressions = compat_data.get("regressions", [])
    if compat_regressions:
        f.write("## 🔌 API Compatibility Regressions\n\n")
        f.write("<details open>\n<summary>Breaking API changes detected</summary>\n\n")
        for reg in compat_regressions:
            f.write(f"- ❌ {reg}\n")
        f.write("\n</details>\n\n")


def write_unified_report(
    out_path: Path,
    unified_verdict: str,
    reasons: List[str],
    spend_str: str,
    cr_data: Dict[str, Any],
    farley_data: Dict[str, Any],
    compat_data: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        _write_header_and_summary(f, unified_verdict, reasons, spend_str)
        _write_metrics_overview(f, cr_data, farley_data, compat_data)
        _write_code_review_details(f, cr_data)
        _write_farley_details(f, farley_data)
        _write_compatibility_details(f, compat_data)


def parse_args():
    parser = argparse.ArgumentParser(description="MSEC Unified Quality Report compiler")
    parser.add_argument("--code-review-pr", required=True, help="PR code review cassette path")
    parser.add_argument("--farley-baseline", required=True, help="Baseline Farley cassette path")
    parser.add_argument("--farley-pr", required=True, help="PR Farley cassette path")
    parser.add_argument("--compatibility-results", required=True, help="API compatibility JSON results path")
    parser.add_argument("--out", required=True, help="Markdown report output path")
    parser.add_argument("--warn-threshold", type=int, default=3, help="Max allowed WARN code review units")
    parser.add_argument(
        "--require-farley-baseline",
        action="store_true",
        help="Fail the unified report when the Farley baseline is missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    try:
        cr_pr_path = validate_input_path(args.code_review_pr, PROJECT_ROOT)
        farley_pr_path = validate_input_path(args.farley_pr, PROJECT_ROOT)
        compat_path = validate_input_path(args.compatibility_results, PROJECT_ROOT)
        out_path = validate_output_path(args.out, PROJECT_ROOT)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Load cassettes & results
    cr_cassette = safe_load_json(cr_pr_path)
    farley_pr_cassette = safe_load_json(farley_pr_path)
    compat_results = safe_load_json(compat_path)
    baseline_result, _, farley_base_cassette = resolve_farley_baseline(
        args.farley_baseline,
        args.require_farley_baseline,
    )

    # 1. Process Code Review
    cr_units = get_reviews(cr_cassette)
    block_units, warn_units, _ = group_units_by_severity(cr_units)
    cqi_failure_reasons = collect_cqi_failure_reasons(cr_units)
    cr_verdict, cr_exit, cr_reasons = cr_determine_verdict(
        block_units,
        warn_units,
        args.warn_threshold,
        cqi_failure_reasons,
    )

    # 2. Process Farley Score
    farley_bsum = {"avg_index": 0.0, "count": 0, "per_property": {}}
    farley_psum = farley_compute_summary(farley_pr_cassette)
    farley_delta = 0.0
    farley_verdict = "PASS"
    farley_exit = 0
    farley_reasons = []
    farley_regressions = []

    if baseline_result.available:
        farley_bsum = farley_compute_summary(farley_base_cassette)
        
        virtual_suite = merge_virtual_suite(farley_base_cassette, farley_pr_cassette)
        farley_psum = farley_compute_summary(virtual_suite)
        farley_delta = farley_psum["avg_index"] - farley_bsum["avg_index"]

        _, pct_val, _ = compute_drops_and_pct(
            farley_base_cassette.get("tests", []),
            farley_pr_cassette.get("tests", []),
        )

        farley_verdict, farley_exit, farley_reasons = farley_determine_verdict(
            farley_delta, farley_bsum, farley_psum, pct_val
        )
        farley_regressions = farley_top_regressions(farley_base_cassette, farley_pr_cassette, top_n=10)

    # 3. Process Compatibility Check
    compat_result = parse_compatibility_result(compat_results)

    # Compile Unified Verdict & Reasons
    unified_reasons = []
    unified_verdict = "PASS"
    exit_code = 0

    if cr_exit != 0:
        unified_verdict = "FAIL"
        exit_code = 2
        unified_reasons.extend(cr_reasons)

    if farley_exit != 0:
        unified_verdict = "FAIL"
        exit_code = 2
        unified_reasons.extend(farley_reasons)

    if baseline_result.state in {"BASELINE_MISSING", "BASELINE_CORRUPTED"}:
        unified_verdict = "FAIL"
        exit_code = 2
        unified_reasons.append(f"FARLEY BASELINE: {baseline_result.reason}")

    if compat_result.state == "CHECK_FAILED":
        unified_verdict = "FAIL"
        exit_code = 2
        unified_reasons.append(f"API COMPATIBILITY CHECK_FAILED: {compat_result.reason}")
    elif not compat_result.compatible:
        unified_verdict = "FAIL"
        exit_code = 2
        unified_reasons.append(f"API COMPATIBILITY: {len(compat_result.regressions)} backward-compatibility breaking change(s) found.")

    spend_str = combine_token_spend(cr_cassette, farley_pr_cassette)

    cr_data = {
        "units": cr_units,
        "verdict": cr_verdict,
        "review_coverage": get_code_review_coverage(cr_cassette),
        "validation_summary": get_code_review_validation_summary(cr_cassette),
    }
    farley_data = {
        "bsum": farley_bsum,
        "psum": farley_psum,
        "delta": farley_delta,
        "verdict": farley_verdict,
        "regressions": farley_regressions,
        "baseline_exists": baseline_result.available,
        "baseline_state": baseline_result.state,
        "baseline_reason": baseline_result.reason,
        "baseline_details": baseline_result.details,
    }
    compat_data = {
        "ok": compat_result.compatible,
        "score": compat_result.score,
        "regressions": compat_result.regressions,
        "state": compat_result.state,
        "reason": compat_result.reason,
        "details": compat_result.details,
    }

    # Write unified report
    write_unified_report(
        out_path,
        unified_verdict,
        unified_reasons,
        spend_str,
        cr_data,
        farley_data,
        compat_data,
    )

    print(f"Generated Unified Quality Report: {out_path} | Verdict: {unified_verdict}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
