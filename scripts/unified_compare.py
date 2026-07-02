#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Tuple

# Enable relative imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.path_utils import validate_input_path, validate_output_path
from scripts.farley_compare import (
    compute_suite_summary as farley_compute_summary,
    merge_virtual_suite,
    compute_drops_and_pct,
    determine_verdict_and_reasons as farley_determine_verdict,
    top_regressions as farley_top_regressions,
)
from scripts.code_review_compare import (
    calculate_cqi,
    group_units_by_severity,
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

    if reasons:
        f.write("### Summary Notes\n")
        for reason in reasons:
            f.write(f"- {reason}\n")
        f.write("\n")

    f.write(spend_str)


def _get_cqi_metric(cr_data: Dict[str, Any]) -> Tuple[str, str]:
    """Build CQI metric and status values."""
    cr_units = cr_data.get("units") or []
    cr_verdict = cr_data.get("verdict") or "PASS"
    if not cr_units:
        return "N/A (no Python code changes)", "PASS"

    scores = [calculate_cqi(u.get("review") or {}) for u in cr_units]
    avg_cqi = mean(scores) if scores else 10.0
    return f"{avg_cqi:.2f}/10 (average of {len(cr_units)} unit(s))", cr_verdict


def _get_farley_metric(farley_data: Dict[str, Any]) -> Tuple[str, str]:
    """Build Farley metric and status values."""
    farley_baseline_exists = farley_data.get("baseline_exists") or False
    farley_bsum = farley_data.get("bsum") or {}
    farley_psum = farley_data.get("psum") or {}
    farley_delta = farley_data.get("delta") if farley_data.get("delta") is not None else 0.0
    farley_verdict = farley_data.get("verdict") or "PASS"

    farley_bsum_avg = farley_bsum.get("avg_index") if farley_bsum.get("avg_index") is not None else 0.0
    farley_psum_avg = farley_psum.get("avg_index") if farley_psum.get("avg_index") is not None else 0.0

    if farley_baseline_exists:
        return (
            f"Baseline avg: {farley_bsum_avg:.2f} \| PR avg: {farley_psum_avg:.2f} \| Delta: {farley_delta:+.2f}",
            farley_verdict,
        )
    return f"PR avg: {farley_psum_avg:.2f} (no baseline to compare)", farley_verdict


def _get_compat_metric(compat_data: Dict[str, Any]) -> Tuple[str, str]:
    """Build API Compatibility metric and status values."""
    compat_score = compat_data.get("score") if compat_data.get("score") is not None else 10.0
    compat_ok = compat_data.get("ok") if compat_data.get("ok") is not None else True
    verdict = "PASS" if compat_ok else "FAIL"
    return f"Score: {compat_score:.1f}/10", verdict


def _write_metrics_overview(
    f,
    cr_data: Dict[str, Any],
    farley_data: Dict[str, Any],
    compat_data: Dict[str, Any],
) -> None:
    f.write("## Metrics Overview\n\n")
    f.write("| Quality Domain | Metric / Score | Status |\n")
    f.write("|---|---|---|\n")

    cqi_metric, cqi_status = _get_cqi_metric(cr_data)
    f.write(f"| Code Quality Index (CQI) | {cqi_metric} | **{cqi_status}** |\n")

    farley_metric, farley_status = _get_farley_metric(farley_data)
    f.write(f"| Farley Test Quality | {farley_metric} | **{farley_status}** |\n")

    compat_metric, compat_status = _get_compat_metric(compat_data)
    f.write(f"| API Compatibility | {compat_metric} | **{compat_status}** |\n\n")


def _write_code_review_details(f, cr_data: Dict[str, Any]) -> None:
    cr_units = cr_data.get("units", [])
    if not cr_units:
        return
    f.write("## 🔍 Code Review Findings\n\n")
    f.write("<details>\n<summary>Click to view detailed Code Review feedback</summary>\n\n")
    block_units, warn_units, _ = group_units_by_severity(cr_units)

    f.write("### Code Units Overview\n\n")
    f.write("| File | Unit | CQI | Severity | Summary |\n")
    f.write("|---|---|---:|---|---|\n")
    for unit in cr_units:
        review = unit.get("review") or {}
        f.write(
            f"| {unit.get('file_path')} "
            f"| {format_unit_name(unit)} "
            f"| {calculate_cqi(review):.2f}/10 "
            f"| **{review.get('severity', 'OK')}** "
            f"| {review.get('summary', '')} |\n"
        )
    f.write("\n")

    problematic = block_units + warn_units
    if problematic:
        f.write("### Detailed Feedback\n\n")
        for unit in problematic:
            review = unit.get("review") or {}
            severity = review.get("severity", "OK")
            f.write(f"#### ⚠️ `{unit.get('file_path')}` -> `{format_unit_name(unit)}` ({severity})\n")
            f.write(f"**CQI**: {calculate_cqi(review):.2f}/10\n\n")
            f.write(f"{review.get('summary', '')}\n\n")
            f.write("| Dimension | Score | Rationale | Suggestions |\n")
            f.write("|---|---|---|---|\n")
            write_dimension_rows(f, review)
            f.write("\n---\n\n")
    f.write("</details>\n\n")


def _write_farley_details(f, farley_data: Dict[str, Any]) -> None:
    farley_baseline_exists = farley_data.get("baseline_exists", False)
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

    # Farley baseline exists check
    farley_baseline_exists = True
    try:
        farley_base_path = validate_input_path(args.farley_baseline, PROJECT_ROOT)
    except ValueError:
        farley_baseline_exists = False

    # Load cassettes & results
    cr_cassette = safe_load_json(cr_pr_path)
    farley_pr_cassette = safe_load_json(farley_pr_path)
    compat_results = safe_load_json(compat_path)

    # 1. Process Code Review
    cr_units = cr_cassette.get("reviews", [])
    block_units, warn_units, _ = group_units_by_severity(cr_units)
    cr_verdict, cr_exit, cr_reasons = cr_determine_verdict(block_units, warn_units, args.warn_threshold)

    # 2. Process Farley Score
    farley_bsum = {"avg_index": 0.0, "count": 0, "per_property": {}}
    farley_psum = farley_compute_summary(farley_pr_cassette)
    farley_delta = 0.0
    farley_verdict = "PASS"
    farley_exit = 0
    farley_reasons = []
    farley_regressions = []
    farley_base_cassette = {"tests": []}

    if farley_baseline_exists:
        farley_base_cassette = safe_load_json(farley_base_path)
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
    compat_ok = compat_results.get("pass") if compat_results.get("pass") is not None else True
    compat_score = compat_results.get("compatibility_index") if compat_results.get("compatibility_index") is not None else 10.0
    compat_regressions = compat_results.get("regressions", [])

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

    if not compat_ok:
        unified_verdict = "FAIL"
        exit_code = 2
        unified_reasons.append(f"API COMPATIBILITY: {len(compat_regressions)} backward-compatibility breaking change(s) found.")

    spend_str = combine_token_spend(cr_cassette, farley_pr_cassette)

    cr_data = {
        "units": cr_units,
        "verdict": cr_verdict,
    }
    farley_data = {
        "bsum": farley_bsum,
        "psum": farley_psum,
        "delta": farley_delta,
        "verdict": farley_verdict,
        "regressions": farley_regressions,
        "baseline_exists": farley_baseline_exists,
    }
    compat_data = {
        "ok": compat_ok,
        "score": compat_score,
        "regressions": compat_regressions,
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
