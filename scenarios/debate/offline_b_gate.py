import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


GENERIC_ANCHOR_PATTERNS = [
    r"\bmissing\b",
    r"\bno\b.*\bvalidation\b",
    r"\bbounds\s*check(ing)?\b",
    r"\bmemory\s+safety\b",
    r"\blogic\s+error(s)?\b",
    r"\bvulnerab(le|ility)\b",
    r"\bunsafe\b",
    r"\bexploit\b",
    r"\battack(er)?\b",
]


def _iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {lineno}: {e}") from e


def _get(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _anchor_matches_input(anchor: str, input_text: str, *, case_insensitive: bool) -> bool:
    if not _is_nonempty_str(anchor) or not _is_nonempty_str(input_text):
        return False
    if case_insensitive:
        return anchor.lower() in input_text.lower()
    return anchor in input_text


def _is_generic_anchor(anchor: str) -> bool:
    if not _is_nonempty_str(anchor):
        return True
    lowered = anchor.lower()
    for pat in GENERIC_ANCHOR_PATTERNS:
        if re.search(pat, lowered):
            return True
    return False


@dataclass
class BGateThresholds:
    max_unsupported_in_accepted_rate: float = 0.05
    max_inconclusive_in_accepted_rate: float = 0.20
    min_anchor_match_rate: float = 0.80


@dataclass
class BGateConfig:
    case_insensitive_anchor_match: bool = True
    require_fields: bool = True
    require_min_anchors: int = 2
    require_anchor_match: bool = True
    thresholds: Optional[BGateThresholds] = None


def compute_b_metrics(
    *,
    input_path: str,
    attempts_path: Optional[str] = None,
    config: BGateConfig,
) -> Dict[str, Any]:
    total = 0
    failures: List[Dict[str, Any]] = []

    accepted = 0
    unsupported = 0
    inconclusive = 0

    anchors_rows_total = 0
    anchors_with_match = 0
    anchors_items_total = 0
    anchors_generic_total = 0

    required_fields = [
        "output.predicate",
        "output.anchors",
        "output.counterfactual",
        "output.verifier_report",
        "output.support_level",
    ]

    for lineno, row in _iter_jsonl(input_path):
        total += 1

        input_text = _get(row, "input")
        output_obj = _get(row, "output")
        if not isinstance(output_obj, dict):
            failures.append(
                {"line": lineno, "reason": "missing_or_invalid_output_object"}
            )
            continue

        missing_fields = []
        if config.require_fields:
            for fpath in required_fields:
                val = _get(row, fpath)
                if fpath == "output.anchors":
                    if not isinstance(val, list):
                        missing_fields.append(fpath)
                elif not _is_nonempty_str(val):
                    missing_fields.append(fpath)
            support_level = _get(row, "output.support_level")
            if support_level not in ("supported", "unsupported", "inconclusive"):
                missing_fields.append("output.support_level(valid)")

        anchors = _get(row, "output.anchors")
        if not isinstance(anchors, list):
            anchors = []

        if missing_fields:
            failures.append(
                {
                    "line": lineno,
                    "reason": "missing_required_fields",
                    "missing": missing_fields,
                }
            )
            continue

        accepted += 1

        support_level = _get(row, "output.support_level")
        if support_level == "unsupported":
            unsupported += 1
        elif support_level == "inconclusive":
            inconclusive += 1

        # B2 Anchor adequacy
        anchors_rows_total += 1
        anchors_items_total += len(anchors)
        if len(anchors) < config.require_min_anchors:
            failures.append(
                {
                    "line": lineno,
                    "reason": "anchors_too_few",
                    "anchors_len": len(anchors),
                }
            )
        else:
            # compute match + generic rate
            any_match = any(
                _anchor_matches_input(
                    str(a),
                    str(input_text or ""),
                    case_insensitive=config.case_insensitive_anchor_match,
                )
                for a in anchors
            )
            if any_match:
                anchors_with_match += 1
            if config.require_anchor_match and not any_match:
                failures.append(
                    {
                        "line": lineno,
                        "reason": "no_anchor_matches_input",
                    }
                )

        anchors_generic_total += sum(1 for a in anchors if _is_generic_anchor(str(a)))

    accepted_samples = max(accepted, 1)

    attempts_total = 0
    attempts_unsupported = 0
    attempts_inconclusive = 0
    if attempts_path:
        for _, attempt in _iter_jsonl(attempts_path):
            attempts_total += 1
            lvl = attempt.get("support_level")
            if lvl == "unsupported":
                attempts_unsupported += 1
            elif lvl == "inconclusive":
                attempts_inconclusive += 1

    metrics: Dict[str, Any] = {
        "input_path": input_path,
        "attempts_path": attempts_path,
        "total_rows": total,
        "accepted_rows": accepted,
        "failures": failures,
        "b0_structural_completeness_pass_rate": (accepted / total) if total else 0.0,
        "b1_unsupported_in_accepted_rate": unsupported / accepted_samples,
        "b1_inconclusive_in_accepted_rate": inconclusive / accepted_samples,
        "b1_unsupported_predicate_rate": (attempts_unsupported / max(attempts_total, 1)) if attempts_path else None,
        "b1_inconclusive_predicate_rate": (attempts_inconclusive / max(attempts_total, 1)) if attempts_path else None,
        "attempts_total": attempts_total if attempts_path else None,
        "b2_anchor_match_rate": anchors_with_match / max(anchors_rows_total, 1),
        "b2_generic_anchor_fraction": anchors_generic_total / max(anchors_items_total, 1),
    }

    thresholds = config.thresholds
    if thresholds:
        unsupported_rate_for_check = (
            metrics["b1_unsupported_predicate_rate"]
            if attempts_path
            else metrics["b1_unsupported_in_accepted_rate"]
        )
        inconclusive_rate_for_check = (
            metrics["b1_inconclusive_predicate_rate"]
            if attempts_path
            else metrics["b1_inconclusive_in_accepted_rate"]
        )
        checks = {
            "max_unsupported_in_accepted_rate": unsupported_rate_for_check
            <= thresholds.max_unsupported_in_accepted_rate,
            "max_inconclusive_in_accepted_rate": inconclusive_rate_for_check
            <= thresholds.max_inconclusive_in_accepted_rate,
            "min_anchor_match_rate": metrics["b2_anchor_match_rate"]
            >= thresholds.min_anchor_match_rate,
        }
        metrics["thresholds"] = {
            "max_unsupported_in_accepted_rate": thresholds.max_unsupported_in_accepted_rate,
            "max_inconclusive_in_accepted_rate": thresholds.max_inconclusive_in_accepted_rate,
            "min_anchor_match_rate": thresholds.min_anchor_match_rate,
        }
        metrics["checks"] = checks
        metrics["pass"] = all(checks.values()) and len(failures) == 0
    else:
        metrics["pass"] = len(failures) == 0

    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description="Offline B Gate (Grounding) for training_corpus.jsonl")
    p.add_argument("--input", required=True, help="Path to training_corpus.jsonl")
    p.add_argument("--attempts", default=None, help="Optional attempts.jsonl to compute true B1 (unsupported/total_attempts)")
    p.add_argument("--metrics-out", default=None, help="Write metrics JSON to this path")
    p.add_argument("--fail", action="store_true", help="Exit non-zero if gate fails")

    p.add_argument("--case-insensitive-anchor-match", action="store_true", default=True)
    p.add_argument("--case-sensitive-anchor-match", action="store_true", default=False)

    p.add_argument("--min-anchors", type=int, default=2)
    p.add_argument("--require-anchor-match", action="store_true", default=True)
    p.add_argument("--no-require-anchor-match", action="store_true", default=False)

    p.add_argument("--max-unsupported-rate", type=float, default=0.05)
    p.add_argument("--max-inconclusive-rate", type=float, default=0.20)
    p.add_argument("--min-anchor-match-rate", type=float, default=0.80)
    args = p.parse_args()

    case_insensitive = args.case_insensitive_anchor_match and not args.case_sensitive_anchor_match
    require_anchor_match = args.require_anchor_match and not args.no_require_anchor_match

    thresholds = BGateThresholds(
        max_unsupported_in_accepted_rate=args.max_unsupported_rate,
        max_inconclusive_in_accepted_rate=args.max_inconclusive_rate,
        min_anchor_match_rate=args.min_anchor_match_rate,
    )
    config = BGateConfig(
        case_insensitive_anchor_match=case_insensitive,
        require_min_anchors=args.min_anchors,
        require_anchor_match=require_anchor_match,
        thresholds=thresholds,
    )

    metrics = compute_b_metrics(input_path=args.input, attempts_path=args.attempts, config=config)

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    else:
        print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.fail and not metrics.get("pass", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
