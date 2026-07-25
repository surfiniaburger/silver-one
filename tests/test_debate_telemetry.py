#!/usr/bin/env python3
"""Unit tests for scripts/debate_telemetry.py."""

import json
import pytest
from pathlib import Path

from scripts.debate_telemetry import (
    compute_debate_benchmark_metrics,
    compute_ab_benchmark_comparison,
    generate_markdown_report,
    load_spans_for_run,
    load_attempts_for_run,
)


def test_load_attempts_and_spans(tmp_path):
    attempts_file = tmp_path / "test_attempts.jsonl"
    attempts_file.write_text(
        json.dumps({
            "seed": 1,
            "decision": "accepted",
            "llm_usage": {
                "by_stage": {
                    "judge_adjudication": {
                        "calls": 1,
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "duration_ms": 200.0,
                    }
                },
                "totals": {
                    "calls": 1,
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "duration_ms": 200.0,
                },
            },
        }) + "\n",
        encoding="utf-8",
    )

    spans_file = tmp_path / "spans.jsonl"
    spans_file.write_text(
        json.dumps({
            "name": "llm_completion",
            "stage": "judge_adjudication",
            "duration_ms": 200.0,
            "attributes": {
                "run_id": "test-run-1",
                "cache_hit": True,
            },
        }) + "\n",
        encoding="utf-8",
    )

    attempts = load_attempts_for_run(attempts_file)
    assert len(attempts) == 1

    spans = load_spans_for_run(spans_file, "test-run-1")
    assert len(spans) == 1
    assert spans[0]["attributes"]["cache_hit"] is True


def test_compute_debate_benchmark_metrics(tmp_path):
    attempts_file = tmp_path / "test_attempts.jsonl"
    attempts_file.write_text(
        json.dumps({
            "seed": 1,
            "decision": "accepted",
            "llm_usage": {
                "by_stage": {
                    "judge_adjudication": {
                        "calls": 1,
                        "prompt_tokens": 1000,
                        "completion_tokens": 200,
                        "duration_ms": 500.0,
                    }
                },
                "totals": {
                    "calls": 1,
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "total_tokens": 1200,
                    "duration_ms": 500.0,
                },
            },
        }) + "\n",
        encoding="utf-8",
    )

    b_gate_file = tmp_path / "b_gate-test-run.json"
    b_gate_file.write_text(
        json.dumps({
            "accepted_rows": 1,
            "usage_prompt_tokens_total": 1000,
            "usage_completion_tokens_total": 200,
            "usage_total_tokens_total": 1200,
            "verifier_pass_rate": 1.0,
            "pass": True,
        }),
        encoding="utf-8",
    )

    metrics = compute_debate_benchmark_metrics(
        run_id="test-run",
        attempts_path=attempts_file,
        b_gate_path=b_gate_file,
    )

    assert metrics["run_id"] == "test-run"
    assert metrics["summary"]["accepted_rows"] == 1
    assert metrics["summary"]["yield_rate"] == 1.0
    assert metrics["token_efficiency"]["total_tokens_total"] == 1200
    assert metrics["token_efficiency"]["tokens_per_accepted_row"] == 1200.0
    assert metrics["b_gate_quality"]["pass"] is True


def test_ab_benchmark_comparison():
    baseline = {
        "run_id": "base-run",
        "summary": {"total_wall_clock_ms": 1000.0},
        "token_efficiency": {
            "total_tokens_total": 1000,
            "prompt_tokens_total": 800,
            "tokens_per_accepted_row": 1000.0,
        },
        "cache_performance": {"cache_hit_rate_pct": 0.0},
        "stage_breakdown": {
            "judge_adjudication": {"avg_duration_ms": 500.0}
        },
    }

    candidate = {
        "run_id": "cand-run",
        "summary": {"total_wall_clock_ms": 600.0},
        "token_efficiency": {
            "total_tokens_total": 500,
            "prompt_tokens_total": 400,
            "tokens_per_accepted_row": 500.0,
        },
        "cache_performance": {"cache_hit_rate_pct": 50.0},
        "stage_breakdown": {
            "judge_adjudication": {"avg_duration_ms": 250.0}
        },
    }

    comp = compute_ab_benchmark_comparison(baseline, candidate)
    assert comp["baseline_run_id"] == "base-run"
    assert comp["candidate_run_id"] == "cand-run"
    assert comp["deltas"]["total_tokens_pct"] == -50.0
    assert comp["deltas"]["prompt_tokens_pct"] == -50.0
    assert comp["stage_deltas"]["judge_adjudication"]["speedup_pct"] == 50.0

    report = generate_markdown_report(candidate, comp)
    assert "# Debate Benchmark Telemetry: `cand-run`" in report
    assert "A/B Comparison" in report
    assert "-50.00%" in report
