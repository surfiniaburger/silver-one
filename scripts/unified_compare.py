#!/usr/bin/env python3
import argparse
import html
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Optional, Tuple

# Enable relative imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.path_utils import validate_input_path, validate_output_path
from scripts.finding_schema import BaselineCheckResult, CompatibilityCheckResult
from scripts.telemetry_utils import coerce_int, coerce_float
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


def _resolve_compatibility(pass_val: Any, state_val: Any) -> bool:
    if isinstance(pass_val, bool):
        return pass_val
    if isinstance(state_val, str) and state_val in {"PASS", "FAIL", "NOT_EXECUTED", "CHECK_FAILED"}:
        return state_val in {"PASS", "NOT_EXECUTED"}
    return True


def _resolve_state(state_val: Any, compatible: bool) -> str:
    if isinstance(state_val, str) and state_val in {"PASS", "FAIL", "NOT_EXECUTED", "CHECK_FAILED"}:
        return state_val
    return "PASS" if compatible else "FAIL"


def _resolve_score(score_val: Any, compatible: bool) -> float:
    if score_val is not None:
        try:
            return float(score_val)
        except (TypeError, ValueError):
            pass
    return 10.0 if compatible else 0.0


def _coerce_list(items: Any) -> List[str]:
    if isinstance(items, list):
        return [str(item) for item in items]
    return []


def parse_compatibility_result(compat_results: Dict[str, Any]) -> CompatibilityCheckResult:
    """Normalize compatibility result JSON into the explicit state model."""
    compatible = _resolve_compatibility(compat_results.get("pass"), compat_results.get("state"))
    state = _resolve_state(compat_results.get("state"), compatible)
    score = _resolve_score(compat_results.get("compatibility_index"), compatible)
    regressions = _coerce_list(compat_results.get("regressions"))
    details = _coerce_list(compat_results.get("details"))

    return CompatibilityCheckResult(
        state=state,
        compatible=compatible,
        score=score,
        regressions=regressions,
        reason=compat_results.get("reason"),
        details=details,
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


def _aggregate_latency(cr_totals: Dict[str, Any], farley_totals: Dict[str, Any], calls: int) -> Tuple[float, float, float]:
    total_dur_ms = coerce_float(cr_totals.get("total_duration_ms")) + coerce_float(farley_totals.get("total_duration_ms"))
    max_dur_ms = max(coerce_float(cr_totals.get("max_duration_ms")), coerce_float(farley_totals.get("max_duration_ms")))
    avg_dur_ms = round(total_dur_ms / calls, 1) if calls > 0 else 0.0
    return total_dur_ms, max_dur_ms, avg_dur_ms


def combine_token_spend(cr_cassette: Dict[str, Any], farley_cassette: Dict[str, Any]) -> str:
    """Summarize total token spend and latency across both Code Review and Farley runs."""
    cr_totals = get_token_totals(cr_cassette, "code_review_usage_summary")
    farley_totals = get_token_totals(farley_cassette, "farley_usage_summary")

    total_tokens = coerce_int(cr_totals.get("total_tokens")) + coerce_int(farley_totals.get("total_tokens"))
    prompt_tokens = coerce_int(cr_totals.get("prompt_tokens")) + coerce_int(farley_totals.get("prompt_tokens"))
    comp_tokens = coerce_int(cr_totals.get("completion_tokens")) + coerce_int(farley_totals.get("completion_tokens"))
    calls = coerce_int(cr_totals.get("calls")) + coerce_int(farley_totals.get("calls"))
    cost = coerce_float(cr_totals.get("cost_usd")) + coerce_float(farley_totals.get("cost_usd"))

    total_dur_ms, max_dur_ms, avg_dur_ms = _aggregate_latency(cr_totals, farley_totals, calls)

    latency_text = ""
    if total_dur_ms > 0:
        latency_text = f"; latency: {total_dur_ms/1000.0:.2f}s total (avg {avg_dur_ms:.1f}ms/call, max {max_dur_ms:.1f}ms)"

    if total_tokens == 0:
        return f"**Token spend**: 0 total tokens (all replayed via cassettes){latency_text}\n\n"

    return (
        "**Combined Token Spend**: "
        f"{total_tokens} total tokens "
        f"({prompt_tokens} prompt, "
        f"{comp_tokens} completion) "
        f"across {calls} LLM call(s){latency_text}; "
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


def _get_code_review_coverage_metric(cr_data: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    coverage = cr_data.get("review_coverage")
    if not isinstance(coverage, dict):
        return None

    total = coerce_int(coverage.get("total_extracted_units"))
    reviewed = coerce_int(coverage.get("reviewed_units"))
    skipped = coerce_int(coverage.get("skipped_units"))
    batches = coerce_int(coverage.get("batch_count"))
    is_empty_coverage = (total == 0 and reviewed == 0 and batches == 0)
    if is_empty_coverage:
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

    valid = coerce_int(summary.get("valid_units"))
    repaired = coerce_int(summary.get("repaired_units"))
    normalized = coerce_int(summary.get("normalized_units"))
    invalid = coerce_int(summary.get("invalid_units"))
    repair_attempts = coerce_int(structured.get("repair_attempts"))
    retries = coerce_int(structured.get("validation_retries"))
    final_failures = coerce_int(structured.get("final_failures"))

    metric = (
        f"{valid} valid, {repaired} repaired, {normalized} normalized, "
        f"{invalid} invalid; {repair_attempts} repair attempt(s), "
        f"{retries} retry/retries, {final_failures} final failure(s)"
    )
    status = "FAIL" if final_failures or invalid else "PASS"
    return metric, status


def _get_repair_breakdown_metric(cr_data: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    summary = cr_data.get("validation_summary")
    if not isinstance(summary, dict) or not summary:
        return None

    details = summary.get("details")
    if not isinstance(details, dict):
        details = {}

    confidence = coerce_int(details.get("repaired_confidence_count"))
    score = coerce_int(details.get("repaired_score_count"))
    default = coerce_int(details.get("repaired_default_count"))
    dropped = coerce_int(details.get("dropped_finding_count"))
    invalid = coerce_int(details.get("invalid_field_count"))
    normalized_path = coerce_int(details.get("normalized_path_count"))
    normalized_text = coerce_int(details.get("normalized_text_count"))

    metric = (
        f"{confidence} confidence, {score} score, {default} default, "
        f"{dropped} dropped finding(s), {invalid} invalid field(s); "
        f"{normalized_path} path(s) normalized, {normalized_text} text(s) normalized"
    )
    status = "WARN" if any([confidence, score, default, dropped, invalid]) else "PASS"
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
    failures = coerce_int(provider.get("failures"))
    status = "FAIL" if failures else "PASS"
    return f"{failures} provider failure(s)", status


def _get_latency_slo_metric(cr_data: Dict[str, Any], farley_data: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, str]]:
    cr_totals = get_token_totals(cr_data, "code_review_usage_summary")
    farley_totals = get_token_totals(farley_data or {}, "farley_usage_summary")

    cr_calls = coerce_int(cr_totals.get("calls"))
    farley_calls = coerce_int(farley_totals.get("calls"))
    calls = cr_calls + farley_calls

    total_dur_ms, max_dur_ms, avg_dur_ms = _aggregate_latency(cr_totals, farley_totals, calls)

    max_call_slo = float(os.getenv("LATENCY_SLO_MAX_CALL_MS", "30000"))
    max_batch_slo = float(os.getenv("LATENCY_SLO_MAX_BATCH_MS", "300000"))

    status = "PASS"
    if max_dur_ms > max_call_slo or total_dur_ms > max_batch_slo:
        status = "WARN"

    if calls == 0 and (math.isclose(total_dur_ms, 0.0, abs_tol=1e-6) or total_dur_ms <= 0):
        return None

    metric = f"{total_dur_ms/1000.0:.2f}s total ({avg_dur_ms:.1f}ms avg/call, max {max_dur_ms:.1f}ms)"
    return metric, status


def _format_finding_confidence(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return "UNKNOWN"

    from scripts.finding_schema import _parse_numeric_confidence, _parse_string_confidence
    if isinstance(value, (int, float)):
        return _parse_numeric_confidence(float(value))

    if isinstance(value, str):
        return _parse_string_confidence(value)

    return "UNKNOWN"


def _format_numeric_confidence_strict(value: Any) -> str:
    from scripts.finding_schema import _parse_numeric_confidence
    try:
        val_float = float(value)
    except (TypeError, ValueError, OverflowError):
        return "UNKNOWN"
    if not math.isfinite(val_float):
        return "UNKNOWN"
    return _parse_numeric_confidence(val_float)


def _format_string_confidence_strict(value: str) -> str:
    from scripts.finding_schema import _parse_confidence_literal
    cleaned = value.strip().upper()
    if cleaned in {"LOW", "MEDIUM", "HIGH"}:
        return cleaned
    return _parse_confidence_literal(cleaned) or "UNKNOWN"


def _format_finding_confidence_strict(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return "UNKNOWN"

    if isinstance(value, (int, float)):
        return _format_numeric_confidence_strict(value)

    if isinstance(value, str):
        return _format_string_confidence_strict(value)

    return "UNKNOWN"


def _extract_unit_findings(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(unit, dict) or is_recoverable_review_failure(unit):
        return []
    review = unit.get("review")
    if not isinstance(review, dict):
        return []
    findings = review.get("findings")
    return findings if isinstance(findings, list) else []


def _classify_finding_confidence(finding: Dict[str, Any]) -> str:
    formatted_conf = _format_finding_confidence_strict(finding.get("confidence"))
    if formatted_conf in {"HIGH", "MEDIUM", "LOW"}:
        return formatted_conf
    return "UNKNOWN"


def calculate_confidence_distribution(cr_units: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    if not isinstance(cr_units, list):
        return counts
    for unit in cr_units:
        findings = _extract_unit_findings(unit)
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            category = _classify_finding_confidence(finding)
            counts[category] += 1
    return counts


def _extract_confidence_validation_fields(unit: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    validation = unit.get("validation")
    if not isinstance(validation, dict):
        return {}
    fields = validation.get("fields")
    if not isinstance(fields, list):
        return {}

    by_index: Dict[int, Dict[str, Any]] = {}
    for field in fields:
        if not isinstance(field, dict):
            continue
        field_name = field.get("field_name")
        if not isinstance(field_name, str):
            continue
        prefix = "findings["
        suffix = "].confidence"
        if not field_name.startswith(prefix) or not field_name.endswith(suffix):
            continue
        index_text = field_name[len(prefix):-len(suffix)]
        try:
            by_index[int(index_text)] = field
        except ValueError:
            continue
    return by_index


def _classify_authored_confidence(finding: Dict[str, Any], validation_field: Optional[Dict[str, Any]]) -> str:
    if validation_field:
        raw_category = _format_finding_confidence_strict(validation_field.get("raw_value"))
        if raw_category in {"HIGH", "MEDIUM", "LOW"}:
            return raw_category

        repaired_category = _format_finding_confidence_strict(validation_field.get("repaired_value"))
        if validation_field.get("status") == "NORMALIZED" and repaired_category in {"HIGH", "MEDIUM", "LOW"}:
            return repaired_category

        return "UNKNOWN_REPAIRED"

    category = _classify_finding_confidence(finding)
    if category in {"HIGH", "MEDIUM", "LOW"}:
        return category
    return "UNKNOWN_REPAIRED"


def calculate_authored_confidence_distribution(cr_units: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN_REPAIRED": 0}
    if not isinstance(cr_units, list):
        return counts
    for unit in cr_units:
        if not isinstance(unit, dict):
            continue
        findings = _extract_unit_findings(unit)
        validation_fields = _extract_confidence_validation_fields(unit)
        for idx, finding in enumerate(findings):
            if not isinstance(finding, dict):
                continue
            category = _classify_authored_confidence(finding, validation_fields.get(idx))
            counts[category] += 1
    return counts






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

    repair_metric = _get_repair_breakdown_metric(cr_data)
    if repair_metric:
        metric, status = repair_metric
        f.write(
            f"| Repair Breakdown | Schema repair categories applied during validation "
            f"| {metric} | **{status}** |\n"
        )

    provider_metric = _get_provider_runtime_metric(cr_data)
    if provider_metric:
        metric, status = provider_metric
        f.write(
            f"| Provider Runtime | Local model/provider execution health "
            f"| {metric} | **{status}** |\n"
        )

    latency_metric = _get_latency_slo_metric(cr_data, farley_data)
    if latency_metric:
        metric, status = latency_metric
        f.write(
            f"| Latency & Speed SLA | Latency budget evaluation and response times "
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

    # Confidence Distribution Section
    cr_units = cr_data.get("units") or []
    dist = calculate_confidence_distribution(cr_units)
    total_findings = sum(dist.values())

    f.write("### Finding Confidence Distribution (Post-Schema)\n\n")
    if total_findings == 0:
        f.write("No structured engineering findings reported in this run.\n\n")
    else:
        f.write(f"- **HIGH Confidence**: {dist['HIGH']} finding(s) ({dist['HIGH']/total_findings:.1%})\n")
        f.write(f"- **MEDIUM Confidence**: {dist['MEDIUM']} finding(s) ({dist['MEDIUM']/total_findings:.1%})\n")
        f.write(f"- **LOW Confidence**: {dist['LOW']} finding(s) ({dist['LOW']/total_findings:.1%})\n")
        f.write(f"- **UNKNOWN/UNPARSED Confidence**: {dist['UNKNOWN']} finding(s) ({dist['UNKNOWN']/total_findings:.1%})\n\n")

    authored_dist = calculate_authored_confidence_distribution(cr_units)
    total_authored = sum(authored_dist.values())

    f.write("### Authored Confidence Signal\n\n")
    if total_authored == 0:
        f.write("No structured engineering findings reported in this run.\n\n")
    else:
        f.write(f"- **HIGH Authored**: {authored_dist['HIGH']} finding(s) ({authored_dist['HIGH']/total_authored:.1%})\n")
        f.write(f"- **MEDIUM Authored**: {authored_dist['MEDIUM']} finding(s) ({authored_dist['MEDIUM']/total_authored:.1%})\n")
        f.write(f"- **LOW Authored**: {authored_dist['LOW']} finding(s) ({authored_dist['LOW']/total_authored:.1%})\n")
        f.write(
            f"- **UNKNOWN/REPAIRED Authored**: {authored_dist['UNKNOWN_REPAIRED']} "
            f"finding(s) ({authored_dist['UNKNOWN_REPAIRED']/total_authored:.1%})\n\n"
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
    f.write("| Category | Severity | Confidence | Finding | Location | Principle | Consequence | Recommendation |\n")
    f.write("|---|---|---:|---|---|---|---|---|\n")
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        title = _esc(finding.get("title", ""))
        cat = _esc(finding.get("category", ""))
        sev = _esc(finding.get("severity", "INFO"))
        confidence = _esc(_format_finding_confidence_strict(finding.get("confidence")))
        principle = _esc(finding.get("reference_principle", ""))
        conseq = _esc(finding.get("engineering_consequence", ""))
        recom = _esc(finding.get("recommended_action", ""))

        evidence = finding.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
        details = evidence.get("details")
        if not isinstance(details, dict):
            details = {}
        start_line = details.get("start_line")
        end_line = details.get("end_line")
        loc = "General"
        if start_line is not None and end_line is not None:
            loc = f"Lines {start_line}-{end_line}"
        elif start_line is not None:
            loc = f"Line {start_line}"

        f.write(
            f"| {cat} | **{sev}** | {confidence} | {title} | {_esc(loc)} "
            f"| {principle} | {conseq} | {recom} |\n"
        )
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

        drops_count, pct_val, _ = compute_drops_and_pct(
            farley_base_cassette.get("tests", []),
            farley_pr_cassette.get("tests", []),
        )

        farley_verdict, farley_exit, farley_reasons = farley_determine_verdict(
            farley_delta, farley_bsum, farley_psum, pct_val, drops_count
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
