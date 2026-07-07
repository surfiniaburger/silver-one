import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Enable relative imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.path_utils import validate_input_path, validate_output_path
from scripts.finding_schema import CQIResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CQI_WEIGHTS = {
    "readability": 1.5,
    "maintainability": 1.5,
    "correctness": 2.0,
    "complexity": 1.0,
    "security": 2.0,
    "test_coverage": 1.0,
}

LEGACY_REVIEW_WRAPPER_KEYS = frozenset({
    "file_path",
    "name",
    "class_name",
    "validation",
    "raw_response",
    "structured_output",
    "recoverable_failure",
})


def _invalid_cqi(error_code: str, reason: str) -> CQIResult:
    return CQIResult(valid=False, error_code=error_code, reason=reason)


def calculate_cqi_result(review: Dict[str, Any]) -> CQIResult:
    if not isinstance(review, dict) or not review:
        return _invalid_cqi("MISSING_REVIEW", "Review payload is missing or empty.")

    total_weighted = 0.0
    total_weight = sum(CQI_WEIGHTS.values())

    for prop, weight in CQI_WEIGHTS.items():
        prop_val = review.get(prop)
        if not isinstance(prop_val, dict):
            return _invalid_cqi("MISSING_DIMENSION", f"Required CQI dimension '{prop}' is missing.")
        score_val = prop_val.get("score")
        if score_val is None:
            return _invalid_cqi("MISSING_SCORE", f"Required CQI score for '{prop}' is missing.")
        if isinstance(score_val, bool):
            return _invalid_cqi("INVALID_SCORE", f"CQI score for '{prop}' must be numeric, not boolean.")
        try:
            score = float(score_val)
        except (TypeError, ValueError):
            return _invalid_cqi("INVALID_SCORE", f"CQI score for '{prop}' is not numeric.")
        if not 0.0 <= score <= 10.0:
            return _invalid_cqi("INVALID_SCORE", f"CQI score for '{prop}' is outside [0.0, 10.0].")
        total_weighted += score * weight

    return CQIResult(valid=True, value=total_weighted / total_weight)


def calculate_cqi(review: Dict[str, Any]) -> float:
    """Compatibility wrapper for legacy callers that expect a float."""
    result = calculate_cqi_result(review)
    return result.value if result.valid and result.value is not None else 10.0


def format_cqi_result(review: Dict[str, Any]) -> str:
    result = calculate_cqi_result(review)
    if result.valid and result.value is not None:
        return f"{result.value:.2f}/10"
    return f"INVALID ({result.error_code or 'INVALID_CQI'})"


def get_usage_totals(cassette: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cassette, dict):
        return {}
    metadata = cassette.get("__metadata__")
    if not isinstance(metadata, dict):
        return {}
    summary = metadata.get("code_review_usage_summary")
    if not isinstance(summary, dict):
        return {}
    usage = summary.get("usage")
    if not isinstance(usage, dict):
        return {}
    totals = usage.get("totals")
    if not isinstance(totals, dict):
        return {}
    return totals


def parse_args():
    parser = argparse.ArgumentParser(description="MSEC Code Review Gate & Reporter")
    parser.add_argument("--pr", required=True, help="PR cassette filename relative to project root")
    parser.add_argument("--out", required=True, help="Report markdown filename relative to project root")
    parser.add_argument("--warn-threshold", type=int, default=3, help="Max allowed WARN units")
    return parser.parse_args()


def resolve_cli_paths(args) -> Tuple[Path, Path]:
    try:
        pr_path = validate_input_path(args.pr, PROJECT_ROOT)
        out_path = validate_output_path(args.out, PROJECT_ROOT)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    return pr_path, out_path


def load_cassette(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            cassette_data = json.load(f)
    except Exception as exc:
        print(f"Error loading PR cassette: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(cassette_data, dict):
        print("Error: Cassette data must be a JSON object.", file=sys.stderr)
        sys.exit(1)
    return cassette_data


def _default_validation_context() -> Dict[str, Any]:
    return {
        "repaired": False,
        "normalized": False,
        "fields": [],
    }


def _is_wrapped_review_unit(unit: Dict[str, Any]) -> bool:
    return isinstance(unit.get("review"), dict)


def _is_legacy_review_payload(unit: Dict[str, Any]) -> bool:
    return any(dim in unit for dim in CQI_WEIGHTS)


def _legacy_review_payload(unit: Dict[str, Any]) -> Dict[str, Any]:
    if _is_legacy_review_payload(unit):
        return {
            key: value
            for key, value in unit.items()
            if key not in LEGACY_REVIEW_WRAPPER_KEYS
        }
    return unit


def normalize_review_unit(unit: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Return a UnitReviewArtifact-shaped dict for current or legacy review entries."""
    if _is_wrapped_review_unit(unit):
        normalized = dict(unit)
        normalized.setdefault("validation", _default_validation_context())
        normalized.setdefault("raw_response", json.dumps(normalized.get("review") or {}, sort_keys=True))
        return normalized

    review = _legacy_review_payload(unit)
    return {
        "file_path": unit.get("file_path") or "legacy-cassette",
        "name": unit.get("name") or f"legacy_review_{index + 1}",
        "class_name": unit.get("class_name"),
        "review": review,
        "validation": _default_validation_context(),
        "raw_response": json.dumps(review, sort_keys=True),
    }


def get_reviews(cassette_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    reviews = cassette_data.get("reviews", [])
    if not isinstance(reviews, list):
        return []
    return [
        normalize_review_unit(unit, index)
        for index, unit in enumerate(reviews)
        if isinstance(unit, dict)
    ]


def _has_structured_block_finding(review: Dict[str, Any]) -> bool:
    findings = review.get("findings")
    if not isinstance(findings, list):
        return False
    return any(
        isinstance(finding, dict) and finding.get("severity") == "BLOCK"
        for finding in findings
    )


def effective_review_severity(review: Dict[str, Any]) -> str:
    """Return the gateable severity, requiring evidence for BLOCK labels."""
    severity = review.get("severity", "OK")
    if severity == "BLOCK" and not _has_structured_block_finding(review):
        return "WARN"
    if severity in {"OK", "WARN", "BLOCK"}:
        return severity
    return "OK"


def group_units_by_severity(reviews: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    block_units = []
    warn_units = []
    ok_units = []
    for unit in reviews:
        if not isinstance(unit, dict):
            continue
        review = unit.get("review") or {}
        severity = effective_review_severity(review)
        if severity == "BLOCK":
            block_units.append(unit)
        elif severity == "WARN":
            warn_units.append(unit)
        else:
            ok_units.append(unit)
    return block_units, warn_units, ok_units


def collect_cqi_failure_reasons(reviews: List[Dict[str, Any]]) -> List[str]:
    reasons = []
    for unit in reviews:
        if not isinstance(unit, dict):
            continue
        review = unit.get("review") or {}
        result = calculate_cqi_result(review)
        if result.valid:
            continue
        unit_name = format_unit_name(unit)
        location = f"{unit.get('file_path')} -> {unit_name}"
        reasons.append(f"INVALID CQI: {location}: {result.reason}")
    return reasons


def determine_verdict(
    block_units: List[Dict[str, Any]],
    warn_units: List[Dict[str, Any]],
    warn_threshold: int,
    cqi_failure_reasons: Optional[List[str]] = None,
) -> Tuple[str, int, List[str]]:
    exit_code = 0
    reasons = []

    if block_units:
        exit_code = 2
        reasons.append(f"CRITICAL: {len(block_units)} code unit(s) were flagged as BLOCK.")

    if cqi_failure_reasons:
        exit_code = 2
        reasons.extend(cqi_failure_reasons)

    if len(warn_units) > warn_threshold:
        reasons.append(
            f"WARNING: {len(warn_units)} code unit(s) flagged as WARN (exceeded threshold of {warn_threshold})."
        )

    verdict = "FAIL" if exit_code != 0 else "PASS"
    return verdict, exit_code, reasons


def format_token_spend(usage_totals: Dict[str, Any]) -> str:
    return (
        "**Token spend**: "
        f"{int(usage_totals.get('total_tokens') or 0)} total tokens "
        f"({int(usage_totals.get('prompt_tokens') or 0)} prompt, "
        f"{int(usage_totals.get('completion_tokens') or 0)} completion) "
        f"across {int(usage_totals.get('calls') or 0)} LLM call(s); "
        f"estimated cost ${float(usage_totals.get('cost_usd') or 0.0):.6f}\n\n"
    )


def format_unit_name(unit: Dict[str, Any]) -> str:
    class_str = f"{unit.get('class_name') or ''}." if unit.get("class_name") else ""
    return f"{class_str}{unit.get('name') or ''}"


def write_summary_notes(f, reasons: List[str]) -> None:
    if not reasons:
        return
    f.write("### Summary Notes\n")
    for reason in reasons:
        f.write(f"- {reason}\n")
    f.write("\n")


def write_usage_stats(f, cassette_data: Dict[str, Any]) -> None:
    usage_totals = get_usage_totals(cassette_data)
    if usage_totals:
        f.write(format_token_spend(usage_totals))


def write_overview(f, reviews: List[Dict[str, Any]]) -> None:
    f.write("## Overview of Units\n\n")
    if not reviews:
        f.write("No units reviewed.\n")
        return

    f.write("| File | Unit | CQI Score | Severity | Summary |\n")
    f.write("|---|---|---:|---|---|\n")
    for unit in reviews:
        review = unit.get("review") or {}
        severity = effective_review_severity(review)
        f.write(
            f"| {unit.get('file_path')} "
            f"| {format_unit_name(unit)} "
            f"| {format_cqi_result(review)} "
            f"| **{severity}** "
            f"| {review.get('summary', '')} |\n"
        )
    f.write("\n")


def format_suggestions(suggestions: Any) -> str:
    if not suggestions:
        return "None"
    return " ".join([f"- {s}" for s in suggestions])


def write_dimension_rows(f, review: Dict[str, Any]) -> None:
    dimensions = ["readability", "maintainability", "correctness", "complexity", "security", "test_coverage"]
    for dim in dimensions:
        dim_data = review.get(dim)
        if not isinstance(dim_data, dict):
            continue
        score = dim_data.get("score", 10)
        rationale = dim_data.get("rationale", "")
        sug_str = format_suggestions(dim_data.get("suggestions", []))
        f.write(f"| {dim.capitalize()} | {score}/10 | {rationale} | {sug_str} |\n")


def write_detailed_feedback(f, reviews: List[Dict[str, Any]]) -> None:
    if not reviews:
        return
    f.write("## Detailed Feedback\n\n")
    for unit in reviews:
        review = unit.get("review") or {}
        severity = effective_review_severity(review)
        if severity == "OK":
            continue

        f.write(f"### ⚠️ {unit.get('file_path')} -> `{format_unit_name(unit)}` ({severity})\n")
        f.write(f"**Overall CQI**: {format_cqi_result(review)}\n\n")
        f.write(f"{review.get('summary', '')}\n\n")
        f.write("| Dimension | Score | Rationale | Suggestions |\n")
        f.write("|---|---|---|---|\n")
        write_dimension_rows(f, review)
        f.write("\n---\n\n")


def write_report(
    out_path: Path,
    *,
    cassette_data: Dict[str, Any],
    reviews: List[Dict[str, Any]],
    verdict: str,
    reasons: List[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Code Review Report\n\n")
        f.write(f"**Verdict**: {verdict}\n\n")
        write_summary_notes(f, reasons)
        write_usage_stats(f, cassette_data)
        write_overview(f, reviews)
        write_detailed_feedback(f, reviews)



def main():
    args = parse_args()
    pr_path, out_path = resolve_cli_paths(args)
    cassette_data = load_cassette(pr_path)
    reviews = get_reviews(cassette_data)
    block_units, warn_units, _ = group_units_by_severity(reviews)
    cqi_failure_reasons = collect_cqi_failure_reasons(reviews)
    verdict, exit_code, reasons = determine_verdict(
        block_units,
        warn_units,
        args.warn_threshold,
        cqi_failure_reasons,
    )
    write_report(
        out_path,
        cassette_data=cassette_data,
        reviews=reviews,
        verdict=verdict,
        reasons=reasons,
    )
    print(f"Generated code review report: {out_path}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
