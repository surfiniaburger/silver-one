#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Dict

# Enable relative imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.path_utils import validate_input_path, validate_output_path

FAIL_DELTA = 0.25
FAIL_PROP = 0.5
FAIL_PERCENT_TESTS = 0.05

# Restrict all file operations to these directories
CASSETTE_ROOT = Path("./cassettes").resolve()
REPORT_ROOT = Path("./reports").resolve()

# Allowed file extensions
JSON_EXTENSIONS = frozenset({".json"})
MD_EXTENSIONS = frozenset({".md"})

# Ensure directories exist
CASSETTE_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)



def load_cassette(path: str):
    try:
        safe_path = validate_input_path(path, CASSETTE_ROOT, JSON_EXTENSIONS)

        with safe_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        print(
            f"Warning: cassette file not found at {path}. Returning empty cassette.",
            file=sys.stderr,
        )
        return {"tests": []}

    except (ValueError) as exc:
        print(f"Error loading cassette '{path}': {exc}", file=sys.stderr)
        return {"tests": []}


def compute_suite_summary(cassette):
    tests = cassette.get("tests", [])

    if not tests:
        return {
            "avg_index": 0.0,
            "count": 0,
            "per_property": {},
        }

    indices = [t.get("farley_index", 0.0) for t in tests]

    per_prop = {}
    props = [
        "understandable",
        "maintainable",
        "repeatable",
        "atomic",
        "necessary",
        "granular",
        "fast",
        "first_tdd",
    ]

    for prop in props:
        vals = []

        for test in tests:
            breakdown = test.get("farley_breakdown", {})
            score = breakdown.get(prop, {}).get("score")

            if score is not None:
                vals.append(score)

        per_prop[prop] = mean(vals) if vals else 0.0

    return {
        "avg_index": mean(indices),
        "count": len(indices),
        "per_property": per_prop,
    }


def get_usage_totals(cassette):
    if not isinstance(cassette, dict):
        return {}
    metadata = cassette.get("__metadata__")
    if not isinstance(metadata, dict):
        return {}
    summary = metadata.get("farley_usage_summary")
    if not isinstance(summary, dict):
        return {}
    usage = summary.get("usage")
    if not isinstance(usage, dict):
        return {}
    totals = usage.get("totals")
    if not isinstance(totals, dict):
        return {}
    return totals


def format_token_spend(pr) -> str:
    usage_totals = get_usage_totals(pr)
    if not usage_totals:
        return ""
    return (
        "**Token spend**: "
        f"{int(usage_totals.get('total_tokens') or 0)} total tokens "
        f"({int(usage_totals.get('prompt_tokens') or 0)} prompt, "
        f"{int(usage_totals.get('completion_tokens') or 0)} completion) "
        f"across {int(usage_totals.get('calls') or 0)} LLM call(s); "
        f"estimated cost ${float(usage_totals.get('cost_usd') or 0.0):.6f}\n\n"
    )


def _compare_tests(base_tests, pr_tests):
    """Align tests by ID and compute deltas (PR - Base)."""
    base_map = {
        test.get("id"): test
        for test in base_tests
        if test.get("id") is not None
    }
    deltas = []
    for test in pr_tests:
        test_id = test.get("id")
        baseline = base_map.get(test_id)
        if baseline:
            delta = test.get("farley_index", 0.0) - baseline.get("farley_index", 0.0)
            deltas.append((delta, baseline, test))
    return deltas


def merge_virtual_suite(base: dict, pr: dict) -> dict:
    """Build a Virtual PR Suite for mathematically valid suite-wide comparison.

    When the PR cassette only contains a subset of tests (diff-only evaluation
    mode), a direct average comparison against the full baseline is invalid.
    This function solves that by overlaying PR scores onto the baseline suite
    by test ID, producing a complete virtual suite where:

    * Tests evaluated in the PR  → use their new PR score.
    * Tests NOT evaluated in the PR → keep their original baseline score.
    * Tests NEW in the PR (no baseline entry) → added as new entries.

    If the PR cassette contains the full suite (legacy full-suite mode), the
    result is identical to a plain PR comparison — backward-compatible.
    """
    base_map: Dict[str, dict] = {
        t["id"]: t for t in base.get("tests", []) if t.get("id") is not None
    }
    pr_map: Dict[str, dict] = {
        t["id"]: t for t in pr.get("tests", []) if t.get("id") is not None
    }

    # Start from the full baseline; overlay any PR results.
    merged: Dict[str, dict] = {**base_map, **pr_map}
    return {"tests": list(merged.values())}


def top_regressions(base, pr, top_n=5):
    deltas = _compare_tests(base.get("tests", []), pr.get("tests", []))
    regressions = [item for item in deltas if item[0] < 0]
    regressions.sort(key=lambda item: item[0])
    return regressions[:top_n]


def compute_drops_and_pct(base_tests, pr_tests):
    deltas = _compare_tests(base_tests, pr_tests)
    drops = 0
    biggest = []
    for d, baseline, test in deltas:
        drop_val = -d
        if drop_val >= 2.0:
            drops += 1
            biggest.append((drop_val, baseline, test))

    total = len(pr_tests)
    pct = (drops / total) if total else 0.0
    return drops, pct, biggest


def determine_verdict_and_reasons(delta, bsum, psum, pct_val):
    verdict = "PASS"
    exit_code = 0
    reasons = []

    if delta <= -FAIL_DELTA:
        verdict = "FAIL"
        exit_code = 2
        reasons.append(
            f"Suite Farley Index decreased by {abs(delta):.2f} >= {FAIL_DELTA}"
        )

    understandable_delta = (
        psum["per_property"].get("understandable", 0.0)
        - bsum["per_property"].get("understandable", 0.0)
    )

    if understandable_delta <= -FAIL_PROP:
        verdict = "FAIL"
        exit_code = 2
        reasons.append("Understandable dropped too much")

    maintainable_delta = (
        psum["per_property"].get("maintainable", 0.0)
        - bsum["per_property"].get("maintainable", 0.0)
    )

    if maintainable_delta <= -FAIL_PROP:
        verdict = "FAIL"
        exit_code = 2
        reasons.append("Maintainable dropped too much")

    if pct_val > FAIL_PERCENT_TESTS:
        verdict = "FAIL"
        exit_code = 2
        reasons.append(
            f"{pct_val * 100:.1f}% of tests dropped by >=2 points"
        )

    return verdict, exit_code, reasons


def write_report(out_path: str, bsum, psum, delta, verdict, reasons, base, pr):
    safe_path = validate_output_path(out_path, REPORT_ROOT, frozenset({".md"}))

    # Create any allowed nested directories
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    with safe_path.open("w", encoding="utf-8") as f:
        f.write("# Farley Compare Report\n\n")
        f.write(f"**Baseline avg**: {bsum['avg_index']:.2f}\n")
        f.write(f"**PR avg**: {psum['avg_index']:.2f}\n")
        f.write(f"**Delta**: {delta:+.2f}\n\n")

        f.write(format_token_spend(pr))

        f.write(f"**Verdict**: {verdict}\n")

        if reasons:
            f.write("\n**Reasons**:\n")

            for reason in reasons:
                f.write(f"- {reason}\n")

        f.write("\n## Top regressions\n")

        regressions = top_regressions(base, pr, top_n=10)

        if not regressions:
            f.write("No regressions found.\n")
        else:
            f.write("| Delta | File | Test | Base | PR |\n")
            f.write("|---|---|---|---:|---:|\n")

            for delta_val, base_test, pr_test in regressions:
                f.write(
                    f"| {delta_val:.2f} "
                    f"| {base_test.get('file_path', '')} "
                    f"| {base_test.get('test_name', '')} "
                    f"| {base_test.get('farley_index', 0.0):.2f} "
                    f"| {pr_test.get('farley_index', 0.0):.2f} |\n"
                )


def _write_no_baseline_report(out_path: str, pr: dict) -> None:
    """Write a minimal report when no baseline cassette exists yet."""
    out = validate_output_path(out_path, REPORT_ROOT, MD_EXTENSIONS)
    out.parent.mkdir(parents=True, exist_ok=True)
    psum = compute_suite_summary(pr)
    with out.open("w", encoding="utf-8") as f:
        f.write("# Farley Compare Report\n\n")
        f.write(
            "> ⚠️ **No baseline cassette found.** "
            "This is likely the first run on this branch. "
            "A baseline will be established when this branch merges to `main`.\n\n"
        )
        f.write(f"**PR avg Farley Index**: {psum['avg_index']:.2f} "
                f"({psum['count']} test(s) evaluated)\n\n")
        f.write(format_token_spend(pr))
        f.write("**Verdict**: PASS _(no baseline to compare against)_\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline cassette filename relative to ./cassettes",
    )
    parser.add_argument(
        "--pr",
        required=True,
        help="PR cassette filename relative to ./cassettes",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Report filename relative to ./reports",
    )

    args = parser.parse_args()

    # Validate the PR cassette and output path — these must always be present/valid.
    try:
        validate_input_path(args.pr, CASSETTE_ROOT, JSON_EXTENSIONS)
        validate_output_path(args.out, REPORT_ROOT, MD_EXTENSIONS)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # The baseline is optional: if it doesn't exist yet (first run / no prior merge),
    # emit a warning and produce an informational report without failing.
    baseline_exists = True
    try:
        validate_input_path(args.baseline, CASSETTE_ROOT, JSON_EXTENSIONS)
    except ValueError as exc:
        # Only treat "does not exist" as a soft warning; other errors (bad path,
        # wrong extension) are still fatal.
        if "does not exist" in str(exc):
            print(
                f"Warning: baseline cassette not found ({exc}). "
                "Generating a first-run report with no comparison.",
                file=sys.stderr,
            )
            baseline_exists = False
        else:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    pr = load_cassette(args.pr)

    if not baseline_exists:
        _write_no_baseline_report(args.out, pr)
        sys.exit(0)

    base = load_cassette(args.baseline)

    bsum = compute_suite_summary(base)

    # Build Virtual PR Suite: overlay PR-evaluated tests onto the full baseline
    # by test ID.  This makes the delta calculation valid even when the PR
    # cassette only contains a diff-only subset of the test suite.
    # If the PR cassette has the full suite (legacy mode), the merge is a no-op.
    virtual_suite = merge_virtual_suite(base, pr)
    psum = compute_suite_summary(virtual_suite)

    delta = psum["avg_index"] - bsum["avg_index"]

    _, pct_val, _ = compute_drops_and_pct(
        base.get("tests", []),
        pr.get("tests", []),
    )

    verdict, exit_code, reasons = determine_verdict_and_reasons(
        delta,
        bsum,
        psum,
        pct_val,
    )

    write_report(
        args.out,
        bsum,
        psum,
        delta,
        verdict,
        reasons,
        base,
        pr,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
