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


def _valid_verifier_report(value: Any) -> bool:
    if isinstance(value, dict):
        return len(value) > 0
    if isinstance(value, str):
        return value.strip() != ""
    return False


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
    min_verifier_pass_rate: float = 0.0
    min_verifier_parse_ok_rate: float = 0.0
    max_cost_per_accepted_row: Optional[float] = None
    max_tokens_per_accepted_row: Optional[float] = None


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
                elif fpath == "output.verifier_report":
                    if not _valid_verifier_report(val):
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
    attempts_soft_aboutness_total = 0
    attempts_soft_aboutness_fail = 0
    attempts_soft_mech_total = 0
    attempts_soft_mech_fail = 0
    verifier_called = 0
    verifier_parse_ok = 0
    verifier_pass = 0
    prowin_total = 0
    verifier_called_on_prowin = 0
    accepted_with_not_applicable = 0
    disagreement_missing_or_parse_fail = 0
    disagreement_verifier_pass_anchor_strict_fail = 0
    attempts_b2_strict_total = 0
    attempts_b2_strict_fail = 0
    attempts_anchor_too_few = 0
    attempts_anchor_no_match = 0
    attempts_mechanism_evidence_failed = 0
    usage_calls_total = 0
    usage_prompt_tokens_total = 0
    usage_completion_tokens_total = 0
    usage_total_tokens_total = 0
    usage_cost_usd_total = 0.0
    usage_missing_usage_calls_total = 0
    usage_stage_totals: Dict[str, Dict[str, float]] = {}
    usage_model_totals: Dict[str, Dict[str, float]] = {}
    usage_source_totals: Dict[str, int] = {}
    generation_config_values: Dict[str, List[Any]] = {}
    generation_config_missing_attempts = 0
    if attempts_path:
        for _, attempt in _iter_jsonl(attempts_path):
            attempts_total += 1
            lvl = attempt.get("support_level")
            if lvl == "unsupported":
                attempts_unsupported += 1
            elif lvl == "inconclusive":
                attempts_inconclusive += 1

            soft = attempt.get("soft_checks") if isinstance(attempt, dict) else None
            if isinstance(soft, dict):
                pa = soft.get("predicate_aboutness")
                if isinstance(pa, dict) and "pass" in pa:
                    attempts_soft_aboutness_total += 1
                    if not bool(pa.get("pass")):
                        attempts_soft_aboutness_fail += 1

                mg = soft.get("mechanism_grounding")
                if isinstance(mg, dict) and "pass" in mg:
                    attempts_soft_mech_total += 1
                    if not bool(mg.get("pass")):
                        attempts_soft_mech_fail += 1

            verifier = attempt.get("verifier") if isinstance(attempt, dict) else None
            if isinstance(verifier, dict):
                if bool(verifier.get("called")):
                    verifier_called += 1
                if bool(verifier.get("parse_ok")):
                    verifier_parse_ok += 1
                if verifier.get("passes_audit") is True:
                    verifier_pass += 1

            judge_eval = attempt.get("judge_eval") if isinstance(attempt, dict) else None
            if isinstance(judge_eval, dict) and judge_eval.get("winner") == "pro_debater":
                prowin_total += 1
                if isinstance(verifier, dict) and bool(verifier.get("called")):
                    verifier_called_on_prowin += 1

            # Disagreement audit #1: judge accepted while verifier is missing/failed parse.
            if attempt.get("decision") == "accepted":
                parse_ok = bool(verifier.get("parse_ok")) if isinstance(verifier, dict) else False
                called = bool(verifier.get("called")) if isinstance(verifier, dict) else False
                if (not called) or (not parse_ok):
                    disagreement_missing_or_parse_fail += 1

            # Disagreement audit #2: verifier passes but strict anchor gate fails.
            if attempt.get("reject_reason") == "anchors_no_match":
                v_pass = False
                if isinstance(verifier, dict):
                    v_pass = verifier.get("passes_audit") is True
                if v_pass:
                    disagreement_verifier_pass_anchor_strict_fail += 1

            # Runtime-strict B2 from judge outcomes (attempt-level).
            # Only count attempts that reached judge output checks.
            reason = attempt.get("reject_reason")
            decision = attempt.get("decision")
            judge_eval = attempt.get("judge_eval")
            if isinstance(judge_eval, dict) or decision == "accepted":
                attempts_b2_strict_total += 1
                if reason in (
                    "anchors_too_few_after_normalization",
                    "anchors_no_match",
                    "mechanism_evidence_failed",
                ):
                    attempts_b2_strict_fail += 1
                if reason == "anchors_too_few_after_normalization":
                    attempts_anchor_too_few += 1
                elif reason == "anchors_no_match":
                    attempts_anchor_no_match += 1
                elif reason == "mechanism_evidence_failed":
                    attempts_mechanism_evidence_failed += 1

            llm_usage = attempt.get("llm_usage") if isinstance(attempt, dict) else None
            if isinstance(llm_usage, dict):
                generation_config = llm_usage.get("generation_config")
                if isinstance(generation_config, dict) and generation_config:
                    for key, value in generation_config.items():
                        bucket = generation_config_values.setdefault(str(key), [])
                        if value not in bucket:
                            bucket.append(value)
                else:
                    generation_config_missing_attempts += 1

                totals = llm_usage.get("totals")
                if isinstance(totals, dict):
                    usage_calls_total += int(totals.get("calls") or 0)
                    usage_prompt_tokens_total += int(totals.get("prompt_tokens") or 0)
                    usage_completion_tokens_total += int(totals.get("completion_tokens") or 0)
                    usage_total_tokens_total += int(totals.get("total_tokens") or 0)
                    usage_cost_usd_total += float(totals.get("cost_usd") or 0.0)
                    usage_missing_usage_calls_total += int(totals.get("missing_usage_calls") or 0)

                events = llm_usage.get("events")
                if isinstance(events, list):
                    for ev in events:
                        if not isinstance(ev, dict):
                            continue
                        src = str(ev.get("usage_source", "unknown"))
                        usage_source_totals[src] = usage_source_totals.get(src, 0) + 1

                # Backward compatibility: if event-level detail does not exist,
                # approximate source attribution from aggregate counters.
                if not isinstance(events, list):
                    calls = 0
                    missing = 0
                    if isinstance(totals, dict):
                        calls = int(totals.get("calls") or 0)
                        missing = int(totals.get("missing_usage_calls") or 0)
                    estimated_or_provider = max(calls - missing, 0)
                    if estimated_or_provider:
                        usage_source_totals["estimated_or_provider"] = (
                            usage_source_totals.get("estimated_or_provider", 0) + estimated_or_provider
                        )
                    if missing:
                        usage_source_totals["missing"] = usage_source_totals.get("missing", 0) + missing

                by_stage = llm_usage.get("by_stage")
                if isinstance(by_stage, dict):
                    for stage, vals in by_stage.items():
                        if not isinstance(vals, dict):
                            continue
                        slot = usage_stage_totals.setdefault(
                            str(stage),
                            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
                        )
                        slot["calls"] += int(vals.get("calls") or 0)
                        slot["prompt_tokens"] += int(vals.get("prompt_tokens") or 0)
                        slot["completion_tokens"] += int(vals.get("completion_tokens") or 0)
                        slot["total_tokens"] += int(vals.get("total_tokens") or 0)
                        slot["cost_usd"] += float(vals.get("cost_usd") or 0.0)

                by_model = llm_usage.get("by_model")
                if isinstance(by_model, dict):
                    for model, vals in by_model.items():
                        if not isinstance(vals, dict):
                            continue
                        slot = usage_model_totals.setdefault(
                            str(model),
                            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
                        )
                        slot["calls"] += int(vals.get("calls") or 0)
                        slot["prompt_tokens"] += int(vals.get("prompt_tokens") or 0)
                        slot["completion_tokens"] += int(vals.get("completion_tokens") or 0)
                        slot["total_tokens"] += int(vals.get("total_tokens") or 0)
                        slot["cost_usd"] += float(vals.get("cost_usd") or 0.0)

        # accepted_with_not_applicable_rate comes from training rows
        for _, row in _iter_jsonl(input_path):
            out = _get(row, "output")
            if not isinstance(out, dict):
                continue
            if out.get("verifier_report") == "not_applicable":
                accepted_with_not_applicable += 1

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
        "b2_predicate_aboutness_fail_rate": (
            attempts_soft_aboutness_fail / max(attempts_soft_aboutness_total, 1)
            if attempts_path
            else None
        ),
        "b2_predicate_aboutness_total": attempts_soft_aboutness_total if attempts_path else None,
        "b2_mechanism_grounding_fail_rate": (
            attempts_soft_mech_fail / max(attempts_soft_mech_total, 1) if attempts_path else None
        ),
        "b2_mechanism_grounding_total": attempts_soft_mech_total if attempts_path else None,
        "verifier_called_rate": (verifier_called / max(attempts_total, 1)) if attempts_path else None,
        "verifier_called_on_prowin_rate": (
            verifier_called_on_prowin / max(prowin_total, 1) if attempts_path else None
        ),
        "prowin_total": prowin_total if attempts_path else None,
        "verifier_parse_ok_rate": (verifier_parse_ok / max(verifier_called, 1)) if attempts_path else None,
        "verifier_pass_rate": (verifier_pass / max(verifier_parse_ok, 1)) if attempts_path else None,
        "accepted_with_not_applicable_rate": (
            accepted_with_not_applicable / max(accepted, 1) if attempts_path else None
        ),
        "disagreement_judge_accept_but_verifier_missing_or_parse_fail_count": (
            disagreement_missing_or_parse_fail if attempts_path else None
        ),
        "disagreement_verifier_pass_but_anchor_strict_fail_count": (
            disagreement_verifier_pass_anchor_strict_fail if attempts_path else None
        ),
        "b2_strict_fail_rate": (
            attempts_b2_strict_fail / max(attempts_b2_strict_total, 1) if attempts_path else None
        ),
        "b2_strict_total": attempts_b2_strict_total if attempts_path else None,
        "b2_strict_failures": {
            "anchors_too_few_after_normalization": attempts_anchor_too_few,
            "anchors_no_match": attempts_anchor_no_match,
            "mechanism_evidence_failed": attempts_mechanism_evidence_failed,
        }
        if attempts_path
        else None,
        "b2_anchor_match_rate": anchors_with_match / max(anchors_rows_total, 1),
        "b2_generic_anchor_fraction": anchors_generic_total / max(anchors_items_total, 1),
        "usage_calls_total": usage_calls_total if attempts_path else None,
        "usage_prompt_tokens_total": usage_prompt_tokens_total if attempts_path else None,
        "usage_completion_tokens_total": usage_completion_tokens_total if attempts_path else None,
        "usage_total_tokens_total": usage_total_tokens_total if attempts_path else None,
        "usage_cost_usd_total": usage_cost_usd_total if attempts_path else None,
        "usage_missing_usage_calls_total": usage_missing_usage_calls_total if attempts_path else None,
        "usage_by_stage_totals": usage_stage_totals if attempts_path else None,
        "usage_by_model_totals": usage_model_totals if attempts_path else None,
        "usage_by_source_totals": usage_source_totals if attempts_path else None,
        "usage_by_source_rates": (
            {
                k: (v / max(usage_calls_total, 1))
                for k, v in usage_source_totals.items()
            }
            if attempts_path
            else None
        ),
        "efficiency_tokens_per_accepted_row": (
            usage_total_tokens_total / max(accepted, 1) if attempts_path else None
        ),
        "efficiency_cost_per_accepted_row": (
            usage_cost_usd_total / max(accepted, 1) if attempts_path else None
        ),
        "efficiency_tokens_per_attempt": (
            usage_total_tokens_total / max(attempts_total, 1) if attempts_path else None
        ),
        "efficiency_cost_per_attempt": (
            usage_cost_usd_total / max(attempts_total, 1) if attempts_path else None
        ),
        "efficiency_tokens_per_verifier_called_attempt": (
            usage_total_tokens_total / max(verifier_called, 1) if attempts_path else None
        ),
        "efficiency_cost_per_verifier_called_attempt": (
            usage_cost_usd_total / max(verifier_called, 1) if attempts_path else None
        ),
        "generation_config_values": generation_config_values if attempts_path else None,
        "generation_config_missing_attempts": generation_config_missing_attempts if attempts_path else None,
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
            "min_anchor_match_rate": (
                # When attempts are present, use runtime strict B2 signal:
                # pass if strict fail-rate is within inverse of desired anchor match.
                (metrics["b2_strict_fail_rate"] <= (1.0 - thresholds.min_anchor_match_rate))
                if attempts_path
                else (metrics["b2_anchor_match_rate"] >= thresholds.min_anchor_match_rate)
            ),
            "min_verifier_pass_rate": (
                True
                if not attempts_path
                else (
                    metrics["verifier_pass_rate"] is not None
                    and metrics["verifier_pass_rate"] >= thresholds.min_verifier_pass_rate
                )
            ),
            "min_verifier_parse_ok_rate": (
                True
                if not attempts_path
                else (
                    metrics["verifier_parse_ok_rate"] is not None
                    and metrics["verifier_parse_ok_rate"] >= thresholds.min_verifier_parse_ok_rate
                )
            ),
            "max_cost_per_accepted_row": (
                True
                if (not attempts_path or thresholds.max_cost_per_accepted_row is None)
                else (
                    metrics["efficiency_cost_per_accepted_row"] is not None
                    and metrics["efficiency_cost_per_accepted_row"] <= thresholds.max_cost_per_accepted_row
                )
            ),
            "max_tokens_per_accepted_row": (
                True
                if (not attempts_path or thresholds.max_tokens_per_accepted_row is None)
                else (
                    metrics["efficiency_tokens_per_accepted_row"] is not None
                    and metrics["efficiency_tokens_per_accepted_row"] <= thresholds.max_tokens_per_accepted_row
                )
            ),
        }
        metrics["thresholds"] = {
            "max_unsupported_in_accepted_rate": thresholds.max_unsupported_in_accepted_rate,
            "max_inconclusive_in_accepted_rate": thresholds.max_inconclusive_in_accepted_rate,
            "min_anchor_match_rate": thresholds.min_anchor_match_rate,
            "min_verifier_pass_rate": thresholds.min_verifier_pass_rate,
            "min_verifier_parse_ok_rate": thresholds.min_verifier_parse_ok_rate,
            "max_cost_per_accepted_row": thresholds.max_cost_per_accepted_row,
            "max_tokens_per_accepted_row": thresholds.max_tokens_per_accepted_row,
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
    p.add_argument("--min-verifier-pass-rate", type=float, default=0.0)
    p.add_argument("--min-verifier-parse-ok-rate", type=float, default=0.0)
    p.add_argument("--max-cost-per-accepted-row", type=float, default=-1.0)
    p.add_argument("--max-tokens-per-accepted-row", type=float, default=-1.0)
    args = p.parse_args()

    case_insensitive = args.case_insensitive_anchor_match and not args.case_sensitive_anchor_match
    require_anchor_match = args.require_anchor_match and not args.no_require_anchor_match

    thresholds = BGateThresholds(
        max_unsupported_in_accepted_rate=args.max_unsupported_rate,
        max_inconclusive_in_accepted_rate=args.max_inconclusive_rate,
        min_anchor_match_rate=args.min_anchor_match_rate,
        min_verifier_pass_rate=args.min_verifier_pass_rate,
        min_verifier_parse_ok_rate=args.min_verifier_parse_ok_rate,
        max_cost_per_accepted_row=(
            None if args.max_cost_per_accepted_row < 0 else args.max_cost_per_accepted_row
        ),
        max_tokens_per_accepted_row=(
            None if args.max_tokens_per_accepted_row < 0 else args.max_tokens_per_accepted_row
        ),
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
