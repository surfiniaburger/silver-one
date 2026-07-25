#!/usr/bin/env python3
"""
Debate Telemetry & Performance Benchmark Extractor.

Aggregates execution spans, attempt logs, token usage, and quality gate metrics
for debate scenario benchmark runs. Supports standalone benchmark extraction as well as
A/B delta comparisons between baseline and candidate runs (e.g. measuring KV-cache prefill speedups,
token spend reductions, and wall-clock execution time).
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

# Enable relative imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.path_utils import validate_input_path, validate_output_path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_ROOT = (PROJECT_ROOT / "artifacts" / "metrics").resolve()
ATTEMPTS_ROOT = (PROJECT_ROOT / "artifacts" / "attempts").resolve()
REPORT_ROOT = (PROJECT_ROOT / "reports").resolve()

# Allowed file extensions
JSON_EXT = frozenset({".json"})
JSONL_EXT = frozenset({".jsonl"})
MD_EXT = frozenset({".md"})


def _percentile(values: List[float], p: float) -> float:
    """Calculate the p-th percentile of a list of floats."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_v[int(k)])
    d0 = sorted_v[int(f)] * (c - k)
    d1 = sorted_v[int(c)] * (k - f)
    return float(d0 + d1)


def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator == 0:
        return fallback
    return numerator / denominator


def load_spans_for_run(spans_path: Path, run_id: str) -> List[Dict[str, Any]]:
    """Load trace spans matching run_id from spans.jsonl."""
    if not spans_path.exists():
        return []
    matching = []
    with spans_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                attrs = data.get("attributes", {})
                if attrs.get("run_id") == run_id:
                    matching.append(data)
            except Exception as exc:
                logger.warning("[debate-telemetry] Failed to parse span line %d in %s: %s", lineno, spans_path, exc)
                continue
    return matching


def load_attempts_for_run(attempts_path: Path) -> List[Dict[str, Any]]:
    """Load attempt records from attempts JSONL file."""
    if not attempts_path.exists():
        return []
    attempts = []
    with attempts_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                attempts.append(json.loads(line))
            except Exception as exc:
                logger.warning("[debate-telemetry] Failed to parse attempt line %d in %s: %s", lineno, attempts_path, exc)
                continue
    return attempts


def load_b_gate_metrics(b_gate_path: Path) -> Dict[str, Any]:
    """Load B-gate metrics JSON if available."""
    if not b_gate_path.exists():
        return {}
    try:
        with b_gate_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("[debate-telemetry] Failed to load b_gate JSON from %s: %s", b_gate_path, exc)
        return {}


def _determine_accepted_rows(b_gate: Dict[str, Any], attempts: List[Dict[str, Any]]) -> int:
    accepted_rows = b_gate.get("accepted_rows", 0)
    if not accepted_rows and attempts:
        accepted_rows = sum(
            1 for a in attempts
            if a.get("decision") in {"accepted", "accept", "pass"}
            or a.get("status") in {"accepted", "pass"}
            or a.get("b2_pass") is True
        )
    return accepted_rows


def _record_stage_stats(
    stage_durations: Dict[str, List[float]],
    stage_prompt_tokens: Dict[str, int],
    stage_completion_tokens: Dict[str, int],
    stage_calls: Dict[str, int],
    stage: str,
    calls: int,
    p_tok: int,
    c_tok: int,
    dur_ms: float,
) -> None:
    """Record call metrics and duration for a single stage."""
    stage_calls[stage] = stage_calls.get(stage, 0) + calls
    stage_prompt_tokens[stage] = stage_prompt_tokens.get(stage, 0) + p_tok
    stage_completion_tokens[stage] = stage_completion_tokens.get(stage, 0) + c_tok
    if dur_ms > 0:
        stage_durations.setdefault(stage, []).append(dur_ms)


def _process_attempt_events(
    events: List[Dict[str, Any]],
    stage_durations: Dict[str, List[float]],
    stage_prompt_tokens: Dict[str, int],
    stage_completion_tokens: Dict[str, int],
    stage_calls: Dict[str, int],
) -> None:
    """Process event-level telemetry records for an attempt."""
    for ev in events:
        stage = ev.get("stage", "unknown")
        dur_ms = ev.get("duration_ms", 0.0)
        p_tok = ev.get("prompt_tokens", 0)
        c_tok = ev.get("completion_tokens", 0)
        _record_stage_stats(stage_durations, stage_prompt_tokens, stage_completion_tokens, stage_calls, stage, 1, p_tok, c_tok, dur_ms)


def _process_attempt_by_stage(
    by_stage: Dict[str, Any],
    stage_durations: Dict[str, List[float]],
    stage_prompt_tokens: Dict[str, int],
    stage_completion_tokens: Dict[str, int],
    stage_calls: Dict[str, int],
) -> None:
    """Process stage-level aggregate telemetry records for an attempt."""
    for stage, stats in by_stage.items():
        calls = stats.get("calls", 0)
        p_tok = stats.get("prompt_tokens", 0)
        c_tok = stats.get("completion_tokens", 0)
        avg_dur = stats.get("avg_duration_ms")
        if avg_dur is None and calls > 0 and stats.get("total_duration_ms"):
            avg_dur = stats["total_duration_ms"] / calls
        dur_ms = avg_dur or stats.get("duration_ms") or 0.0
        _record_stage_stats(stage_durations, stage_prompt_tokens, stage_completion_tokens, stage_calls, stage, calls, p_tok, c_tok, dur_ms)


def _aggregate_attempt_usage(attempts: List[Dict[str, Any]], b_gate: Dict[str, Any]) -> Dict[str, Any]:
    stage_durations: Dict[str, List[float]] = {}
    stage_prompt_tokens: Dict[str, int] = {}
    stage_completion_tokens: Dict[str, int] = {}
    stage_calls: Dict[str, int] = {}

    total_prompt_tokens = b_gate.get("usage_prompt_tokens_total", 0)
    total_completion_tokens = b_gate.get("usage_completion_tokens_total", 0)
    total_tokens = b_gate.get("usage_total_tokens_total", 0)
    total_calls = b_gate.get("usage_calls_total", 0)
    total_wall_clock_ms = 0.0

    has_b_gate_totals = bool(total_prompt_tokens or total_tokens)

    for attempt in attempts:
        llm_usage = attempt.get("llm_usage", {})
        events = llm_usage.get("events", [])
        if events:
            _process_attempt_events(events, stage_durations, stage_prompt_tokens, stage_completion_tokens, stage_calls)
        else:
            _process_attempt_by_stage(
                llm_usage.get("by_stage", {}),
                stage_durations,
                stage_prompt_tokens,
                stage_completion_tokens,
                stage_calls,
            )

        tot = llm_usage.get("totals", {})
        dur_tot = tot.get("total_duration_ms") or tot.get("duration_ms") or 0.0
        total_wall_clock_ms += dur_tot
        if not has_b_gate_totals:
            total_prompt_tokens += tot.get("prompt_tokens", 0)
            total_completion_tokens += tot.get("completion_tokens", 0)
            total_tokens += tot.get("total_tokens", 0)
            total_calls += tot.get("calls", 0)

    if not total_tokens:
        total_tokens = total_prompt_tokens + total_completion_tokens

    return {
        "stage_durations": stage_durations,
        "stage_prompt_tokens": stage_prompt_tokens,
        "stage_completion_tokens": stage_completion_tokens,
        "stage_calls": stage_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_calls": total_calls,
        "total_wall_clock_ms": total_wall_clock_ms,
    }


def _build_stage_breakdown(
    stage_durations: Dict[str, List[float]],
    stage_prompt_tokens: Dict[str, int],
    stage_completion_tokens: Dict[str, int],
    stage_calls: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    stage_breakdown = {}
    for stage, dur_list in stage_durations.items():
        stage_breakdown[stage] = {
            "calls": stage_calls.get(stage, len(dur_list)),
            "prompt_tokens": stage_prompt_tokens.get(stage, 0),
            "completion_tokens": stage_completion_tokens.get(stage, 0),
            "total_tokens": stage_prompt_tokens.get(stage, 0) + stage_completion_tokens.get(stage, 0),
            "avg_duration_ms": round(mean(dur_list), 2) if dur_list else 0.0,
            "p95_duration_ms": round(_percentile(dur_list, 95.0), 2) if dur_list else 0.0,
            "min_duration_ms": round(min(dur_list), 2) if dur_list else 0.0,
            "max_duration_ms": round(max(dur_list), 2) if dur_list else 0.0,
        }
    return stage_breakdown


def _aggregate_cache_spans(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    cache_hits = 0
    cache_misses = 0
    span_durations: List[float] = []

    for span in spans:
        attrs = span.get("attributes", {})
        dur = span.get("duration_ms", 0.0)
        if dur > 0:
            span_durations.append(dur)
        if attrs.get("cache_hit") is True:
            cache_hits += 1
        elif attrs.get("cache_hit") is False:
            cache_misses += 1

    total_span_calls = cache_hits + cache_misses
    cache_hit_rate_pct = _safe_div(cache_hits * 100.0, total_span_calls)
    return {
        "span_count": len(spans),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "total_span_calls": total_span_calls,
        "cache_hit_rate_pct": round(cache_hit_rate_pct, 2),
        "avg_span_duration_ms": round(mean(span_durations), 2) if span_durations else 0.0,
        "p95_span_duration_ms": round(_percentile(span_durations, 95.0), 2) if span_durations else 0.0,
    }


def compute_debate_benchmark_metrics(
    run_id: str,
    attempts_path: Path,
    spans_path: Optional[Path] = None,
    b_gate_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive benchmark telemetry for a debate run.
    Aggregates attempt logs, spans (if available), and B-gate quality metrics.
    """
    attempts = load_attempts_for_run(attempts_path)
    spans = load_spans_for_run(spans_path, run_id) if spans_path else []
    b_gate = load_b_gate_metrics(b_gate_path) if b_gate_path else {}

    total_attempts = len(attempts)
    accepted_rows = _determine_accepted_rows(b_gate, attempts)
    yield_rate = _safe_div(accepted_rows, total_attempts)

    usage = _aggregate_attempt_usage(attempts, b_gate)
    cache = _aggregate_cache_spans(spans)
    stage_breakdown = _build_stage_breakdown(
        usage["stage_durations"],
        usage["stage_prompt_tokens"],
        usage["stage_completion_tokens"],
        usage["stage_calls"],
    )

    total_tokens = usage["total_tokens"]
    tokens_per_accepted_row = _safe_div(total_tokens, accepted_rows) if accepted_rows > 0 else 0.0
    tokens_per_attempt = _safe_div(total_tokens, total_attempts) if total_attempts > 0 else 0.0
    wall_ms = usage["total_wall_clock_ms"]

    return {
        "run_id": run_id,
        "summary": {
            "total_attempts": total_attempts,
            "accepted_rows": accepted_rows,
            "yield_rate": round(yield_rate, 4),
            "total_wall_clock_ms": round(wall_ms, 2),
            "total_wall_clock_seconds": round(wall_ms / 1000.0, 2),
        },
        "token_efficiency": {
            "prompt_tokens_total": usage["total_prompt_tokens"],
            "completion_tokens_total": usage["total_completion_tokens"],
            "total_tokens_total": total_tokens,
            "tokens_per_accepted_row": round(tokens_per_accepted_row, 2),
            "tokens_per_attempt": round(tokens_per_attempt, 2),
            "total_llm_calls": usage["total_calls"] or cache["total_span_calls"],
        },
        "cache_performance": cache,
        "stage_breakdown": stage_breakdown,
        "b_gate_quality": {
            "verifier_pass_rate": b_gate.get("verifier_pass_rate", 0.0),
            "verifier_parse_ok_rate": b_gate.get("verifier_parse_ok_rate", 0.0),
            "anchor_match_rate": b_gate.get("b2_anchor_match_rate", 0.0),
            "pass": b_gate.get("pass", False),
        },
    }


def compute_ab_benchmark_comparison(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute A/B delta comparison between baseline and candidate debate benchmark runs."""
    base_tok = baseline["token_efficiency"]["total_tokens_total"]
    cand_tok = candidate["token_efficiency"]["total_tokens_total"]
    tok_diff = cand_tok - base_tok
    tok_pct = _safe_div(tok_diff * 100.0, base_tok)

    base_p_tok = baseline["token_efficiency"]["prompt_tokens_total"]
    cand_p_tok = candidate["token_efficiency"]["prompt_tokens_total"]
    p_tok_diff = cand_p_tok - base_p_tok
    p_tok_pct = _safe_div(p_tok_diff * 100.0, base_p_tok)

    base_per_acc = baseline["token_efficiency"]["tokens_per_accepted_row"]
    cand_per_acc = candidate["token_efficiency"]["tokens_per_accepted_row"]
    per_acc_diff = cand_per_acc - base_per_acc
    per_acc_pct = _safe_div(per_acc_diff * 100.0, base_per_acc)

    base_clock = baseline["summary"]["total_wall_clock_ms"]
    cand_clock = candidate["summary"]["total_wall_clock_ms"]
    clock_diff = cand_clock - base_clock
    clock_pct = _safe_div(clock_diff * 100.0, base_clock)

    base_hit = baseline["cache_performance"]["cache_hit_rate_pct"]
    cand_hit = candidate["cache_performance"]["cache_hit_rate_pct"]
    hit_diff = cand_hit - base_hit

    stage_deltas = {}
    cand_stages = candidate.get("stage_breakdown", {})
    base_stages = baseline.get("stage_breakdown", {})
    all_stages = set(cand_stages.keys()).union(base_stages.keys())

    for stage in sorted(all_stages):
        base_avg = base_stages.get(stage, {}).get("avg_duration_ms", 0.0)
        cand_avg = cand_stages.get(stage, {}).get("avg_duration_ms", 0.0)
        dur_diff = cand_avg - base_avg
        dur_pct = _safe_div(dur_diff * 100.0, base_avg)
        stage_deltas[stage] = {
            "baseline_avg_ms": base_avg,
            "candidate_avg_ms": cand_avg,
            "diff_ms": round(dur_diff, 2),
            "speedup_pct": round(-dur_pct, 2),  # positive means candidate is faster
        }

    return {
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "baseline_metrics": {
            "total_tokens": base_tok,
            "prompt_tokens": base_p_tok,
            "tokens_per_accepted_row": base_per_acc,
            "wall_clock_ms": base_clock,
            "cache_hit_rate_pct": base_hit,
        },
        "candidate_metrics": {
            "total_tokens": cand_tok,
            "prompt_tokens": cand_p_tok,
            "tokens_per_accepted_row": cand_per_acc,
            "wall_clock_ms": cand_clock,
            "cache_hit_rate_pct": cand_hit,
        },
        "deltas": {
            "total_tokens_diff": tok_diff,
            "total_tokens_pct": round(tok_pct, 2),
            "prompt_tokens_diff": p_tok_diff,
            "prompt_tokens_pct": round(p_tok_pct, 2),
            "tokens_per_accepted_row_diff": round(per_acc_diff, 2),
            "tokens_per_accepted_row_pct": round(per_acc_pct, 2),
            "wall_clock_ms_diff": round(clock_diff, 2),
            "wall_clock_pct": round(clock_pct, 2),
            "cache_hit_rate_diff_pct": round(hit_diff, 2),
        },
        "stage_deltas": stage_deltas,
    }


def _format_summary_section(run_id: str, summary: Dict[str, Any], tok: Dict[str, Any], cache: Dict[str, Any]) -> List[str]:
    wall_sec = summary.get("total_wall_clock_seconds", 0.0)
    if not wall_sec and summary.get("total_wall_clock_ms"):
        wall_sec = summary["total_wall_clock_ms"] / 1000.0
    wall_str = f"{wall_sec:.1f}s" if wall_sec and wall_sec > 0 else "n/a"
    lines = [
        f"# Debate Benchmark Telemetry: `{run_id}`",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Run ID** | `{run_id}` |",
        f"| **Total Attempts** | {summary.get('total_attempts', 0)} |",
        f"| **Accepted Rows** | {summary.get('accepted_rows', 0)} |",
        f"| **Yield Pass Rate** | {summary.get('yield_rate', 0.0) * 100:.1f}% |",
        f"| **Total Wall-Clock Time** | {wall_str} |",
        f"| **Total Prompt Tokens** | {tok.get('prompt_tokens_total', 0):,} |",
        f"| **Total Completion Tokens** | {tok.get('completion_tokens_total', 0):,} |",
        f"| **Tokens / Accepted Row** | {tok.get('tokens_per_accepted_row', 0.0):,} |",
        f"| **Cache Hit Rate** | {cache.get('cache_hit_rate_pct', 0.0):.1f}% ({cache.get('cache_hits', 0)} hits / {cache.get('cache_misses', 0)} misses) |",
        "",
    ]
    return lines


def _format_ab_comparison_section(comparison: Dict[str, Any]) -> List[str]:
    lines = []
    c_delta = comparison.get("deltas", {})
    c_stages = comparison.get("stage_deltas", {})
    b_metrics = comparison.get("baseline_metrics", {})
    cand_metrics = comparison.get("candidate_metrics", {})

    b_tot = f"{b_metrics.get('total_tokens', 0):,}" if "total_tokens" in b_metrics else "—"
    c_tot = f"{cand_metrics.get('total_tokens', 0):,}" if "total_tokens" in cand_metrics else "—"
    b_ptot = f"{b_metrics.get('prompt_tokens', 0):,}" if "prompt_tokens" in b_metrics else "—"
    c_ptot = f"{cand_metrics.get('prompt_tokens', 0):,}" if "prompt_tokens" in cand_metrics else "—"
    b_per_acc = f"{b_metrics.get('tokens_per_accepted_row', 0.0):,.2f}" if "tokens_per_accepted_row" in b_metrics else "—"
    c_per_acc = f"{cand_metrics.get('tokens_per_accepted_row', 0.0):,.2f}" if "tokens_per_accepted_row" in cand_metrics else "—"
    b_wall = b_metrics.get("wall_clock_ms", 0.0)
    c_wall = cand_metrics.get("wall_clock_ms", 0.0)
    b_dur = f"{b_wall / 1000.0:.1f}s" if b_wall and b_wall > 0 else "n/a"
    c_dur = f"{c_wall / 1000.0:.1f}s" if c_wall and c_wall > 0 else "n/a"
    b_hit = f"{b_metrics.get('cache_hit_rate_pct', 0.0):.1f}%" if "cache_hit_rate_pct" in b_metrics else "—"
    c_hit = f"{cand_metrics.get('cache_hit_rate_pct', 0.0):.1f}%" if "cache_hit_rate_pct" in cand_metrics else "—"

    lines.append(f"## A/B Comparison: Candidate `{comparison.get('candidate_run_id', '')}` vs Baseline `{comparison.get('baseline_run_id', '')}`")
    lines.append("")
    lines.append("| Dimension | Baseline | Candidate | Delta / Speedup |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| **Total Tokens** | {b_tot} | {c_tot} | `{c_delta.get('total_tokens_pct', 0.0):+.2f}%` |")
    lines.append(f"| **Prompt Tokens** | {b_ptot} | {c_ptot} | `{c_delta.get('prompt_tokens_pct', 0.0):+.2f}%` |")
    lines.append(f"| **Tokens / Accepted Row** | {b_per_acc} | {c_per_acc} | `{c_delta.get('tokens_per_accepted_row_pct', 0.0):+.2f}%` |")
    lines.append(f"| **Wall-Clock Duration** | {b_dur} | {c_dur} | `{c_delta.get('wall_clock_pct', 0.0):+.2f}%` |")
    lines.append(f"| **Cache Hit Rate** | {b_hit} | {c_hit} | `{c_delta.get('cache_hit_rate_diff_pct', 0.0):+.2f}%` |")
    lines.append("")
    lines.append("### Stage Latency Speedups")
    lines.append("")
    lines.append("| Stage | Baseline Avg (ms) | Candidate Avg (ms) | Speedup (%) |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for s_name, s_data in c_stages.items():
        lines.append(f"| `{s_name}` | {s_data.get('baseline_avg_ms', 0.0):.1f}ms | {s_data.get('candidate_avg_ms', 0.0):.1f}ms | `{s_data.get('speedup_pct', 0.0):+.2f}%` |")
    lines.append("")
    return lines


def _format_stage_breakdown_section(stages: Dict[str, Any]) -> List[str]:
    lines = [
        "## Stage Breakdown",
        "",
        "| Stage | Calls | Prompt Tokens | Completion Tokens | Avg Duration | P95 Duration |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for s_name, s_data in stages.items():
        lines.append(f"| `{s_name}` | {s_data.get('calls', 0)} | {s_data.get('prompt_tokens', 0):,} | {s_data.get('completion_tokens', 0):,} | {s_data.get('avg_duration_ms', 0.0):.1f}ms | {s_data.get('p95_duration_ms', 0.0):.1f}ms |")
    lines.append("")
    return lines


def _format_quality_signals_section(b_gate: Dict[str, Any]) -> List[str]:
    return [
        "## Quality Gate Signals",
        "",
        f"- **Verifier Pass Rate:** {b_gate.get('verifier_pass_rate', 0.0) * 100:.1f}%",
        f"- **Verifier Parse OK Rate:** {b_gate.get('verifier_parse_ok_rate', 0.0) * 100:.1f}%",
        f"- **Anchor Match Rate:** {b_gate.get('anchor_match_rate', 0.0) * 100:.1f}%",
        f"- **B-Gate Pass Status:** `{'PASS' if b_gate.get('pass') else 'FAIL'}`",
        "",
    ]


def generate_markdown_report(
    benchmark: Dict[str, Any],
    comparison: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a clean GitHub-style Markdown report for debate benchmark telemetry."""
    lines = []
    run_id = benchmark["run_id"]
    summary = benchmark.get("summary", {})
    tok = benchmark.get("token_efficiency", {})
    cache = benchmark.get("cache_performance", {})
    stages = benchmark.get("stage_breakdown", {})
    b_gate = benchmark.get("b_gate_quality", {})

    lines.extend(_format_summary_section(run_id, summary, tok, cache))
    if comparison:
        lines.extend(_format_ab_comparison_section(comparison))
    lines.extend(_format_stage_breakdown_section(stages))
    lines.extend(_format_quality_signals_section(b_gate))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debate Telemetry & Benchmark Extractor")
    parser.add_argument("--run-id", required=True, help="Run ID (e.g. pilot-v1-calibrated-i)")
    parser.add_argument("--attempts", help="Path to attempts JSONL file")
    parser.add_argument("--spans", help="Path to spans.jsonl file")
    parser.add_argument("--b-gate", help="Path to b_gate JSON metrics file")
    parser.add_argument("--baseline-json", help="Optional path to baseline debate benchmark JSON for A/B comparison")
    parser.add_argument("--output-json", help="Path to write output benchmark JSON")
    parser.add_argument("--output-markdown", help="Path to write output markdown report")

    args = parser.parse_args()

    # Resolve input paths securely
    attempts_path = validate_input_path(
        args.attempts or f"artifacts/attempts/{args.run_id}.jsonl",
        PROJECT_ROOT,
        JSONL_EXT,
    )
    spans_path = validate_input_path(
        args.spans or "artifacts/metrics/spans.jsonl",
        PROJECT_ROOT,
        JSONL_EXT,
    )
    b_gate_path = validate_input_path(
        args.b_gate or f"artifacts/metrics/b_gate-{args.run_id}.json",
        PROJECT_ROOT,
        JSON_EXT,
    )

    metrics = compute_debate_benchmark_metrics(
        run_id=args.run_id,
        attempts_path=attempts_path,
        spans_path=spans_path if spans_path.exists() else None,
        b_gate_path=b_gate_path if b_gate_path.exists() else None,
    )

    comparison = None
    if args.baseline_json:
        baseline_path = validate_input_path(args.baseline_json, PROJECT_ROOT, JSON_EXT)
        with baseline_path.open("r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        comparison = compute_ab_benchmark_comparison(baseline_data, metrics)
        metrics["ab_comparison"] = comparison

    # Output JSON
    out_json_path = validate_output_path(
        args.output_json or f"artifacts/metrics/debate_benchmark-{args.run_id}.json",
        PROJECT_ROOT,
        JSON_EXT,
    )
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[debate-telemetry] Wrote benchmark JSON: {out_json_path}")

    # Output Markdown if requested
    if args.output_markdown:
        out_md_path = validate_output_path(args.output_markdown, PROJECT_ROOT, MD_EXT)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        md_text = generate_markdown_report(metrics, comparison)
        with out_md_path.open("w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"[debate-telemetry] Wrote benchmark report markdown: {out_md_path}")


if __name__ == "__main__":
    main()
