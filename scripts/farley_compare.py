#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

FAIL_DELTA = 0.25
FAIL_PROP = 0.5
FAIL_PERCENT_TESTS = 0.05

# Restrict all file operations to these directories
CASSETTE_ROOT = Path("./cassettes").resolve()
REPORT_ROOT = Path("./reports").resolve()

# Ensure directories exist
CASSETTE_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def validate_input_path(path: str) -> Path:
    """
    Resolve and validate a cassette input path to prevent path traversal.
    Paths are treated as relative to CASSETTE_ROOT.
    """
    candidate = (CASSETTE_ROOT / path).resolve()

    try:
        candidate.relative_to(CASSETTE_ROOT)
    except ValueError:
        raise ValueError(
            f"Invalid input path '{path}': path escapes the cassette directory"
        )

    if candidate.suffix.lower() != ".json":
        raise ValueError(
            f"Invalid input path '{path}': expected a .json file"
        )

    return candidate


def validate_output_path(path: str) -> Path:
    """
    Resolve and validate an output report path to prevent path traversal.
    Paths are treated as relative to REPORT_ROOT.
    """
    candidate = (REPORT_ROOT / path).resolve()

    try:
        candidate.relative_to(REPORT_ROOT)
    except ValueError:
        raise ValueError(
            f"Invalid output path '{path}': path escapes the reports directory"
        )

    if candidate.suffix.lower() != ".md":
        raise ValueError(
            f"Invalid output path '{path}': expected a .md file"
        )

    return candidate


def load_cassette(path: str):
    try:
        safe_path = validate_input_path(path)

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


def top_regressions(base, pr, top_n=5):
    base_map = {
        test.get("id"): test
        for test in base.get("tests", [])
        if test.get("id") is not None
    }

    regressions = []

    for test in pr.get("tests", []):
        test_id = test.get("id")
        baseline = base_map.get(test_id)

        if baseline:
            delta = (
                test.get("farley_index", 0.0)
                - baseline.get("farley_index", 0.0)
            )

            if delta < 0:
                regressions.append((delta, baseline, test))

    regressions.sort(key=lambda item: item[0])

    return regressions[:top_n]


def compute_drops_and_pct(base_tests, pr_tests):
    base_map = {
        test.get("id"): test
        for test in base_tests
        if test.get("id") is not None
    }

    drops = 0
    total = len(pr_tests)
    biggest = []

    for test in pr_tests:
        test_id = test.get("id")
        baseline = base_map.get(test_id)

        if baseline:
            delta = (
                baseline.get("farley_index", 0.0)
                - test.get("farley_index", 0.0)
            )

            if delta >= 2.0:
                drops += 1
                biggest.append((delta, baseline, test))

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
    safe_path = validate_output_path(out_path)

    # Create any allowed nested directories
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    with safe_path.open("w", encoding="utf-8") as f:
        f.write("# Farley Compare Report\n\n")
        f.write(f"**Baseline avg**: {bsum['avg_index']:.2f}\n")
        f.write(f"**PR avg**: {psum['avg_index']:.2f}\n")
        f.write(f"**Delta**: {delta:+.2f}\n\n")
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

    try:
        # Validate all paths up front
        validate_input_path(args.baseline)
        validate_input_path(args.pr)
        validate_output_path(args.out)

    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    base = load_cassette(args.baseline)
    pr = load_cassette(args.pr)

    bsum = compute_suite_summary(base)
    psum = compute_suite_summary(pr)

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