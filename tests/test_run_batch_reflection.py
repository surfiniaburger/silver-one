"""
Unit & Integration Tests for Step 3: GEPA Graph-Powered Reflector Batch Integration.

Tests:
  - CLI argument parsing for reflector flags.
  - _build_payload Pareto prompt injection and fallback.
  - _process_seed GEPA work-memory trace recording.
  - Deterministic cassette replay with ReplayManager.
  - Concurrency safety with multiple asynchronous batch workers.
"""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentbeats.clock import RunClock
from scenarios.debate.pareto_registry import ParetoRegistry
from scenarios.debate.reflector_agent import ReflectorClient
from scenarios.debate.reflector_schemas import (
    GraphDiagnosticSignature,
    ReflectRequest,
    ReflectResponse,
    TaxonomyBucket,
    classify_taxonomy_bucket,
    get_static_baseline_prompt,
)
from scenarios.debate.run_batch import (
    BatchContext,
    _build_payload,
    _parse_args,
    _process_seed,
)


@pytest.fixture
def temp_gepa_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_gepa_batch_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_parse_args_reflector_flags():
    """Verify CLI parser correctly parses reflector arguments."""
    args = _parse_args([])
    assert args.reflector is True
    assert args.reflector_in_process is True
    assert args.reflector_url == "http://127.0.0.1:8004"
    assert args.gepa_dir == "artifacts/gepa"

    custom_args = _parse_args([
        "--no-reflector",
        "--no-reflector-in-process",
        "--reflector-url", "http://127.0.0.1:9999",
        "--gepa-dir", "/tmp/custom_gepa",
    ])
    assert custom_args.reflector is False
    assert custom_args.reflector_in_process is False
    assert custom_args.reflector_url == "http://127.0.0.1:9999"
    assert custom_args.gepa_dir == "/tmp/custom_gepa"


def test_build_payload_without_reflector():
    """When reflector_client is None, fallback baseline prompt is injected."""
    args = _parse_args([])
    seed = {
        "topic": "char buf[10];",
        "predicate": "buffer_overflow in memory write",
    }
    payload = _build_payload(
        args=args,
        item_seed=42,
        checkpoint_path="artifacts/checkpoints/42.json",
        record_path="artifacts/runs/42.json",
        batch_started_at="2026-08-23T12:00:00Z",
        seed=seed,
        reflector_client=None,
    )
    assert payload["config"]["taxonomy_bucket"] == "memory_safety"
    assert payload["config"]["active_mutation_id"] == "baseline_v0"
    assert payload["config"]["reflector_prompt"] == get_static_baseline_prompt("memory_safety")


def test_build_payload_with_active_pareto_prompt(temp_gepa_dir):
    """When reflector_client has an active Pareto prompt, it is injected into the payload."""
    registry = ParetoRegistry(gepa_dir=temp_gepa_dir)
    registry.register_pareto_prompt(
        taxonomy="integer_arithmetic",
        prompt="CUSTOM PARETO PROMPT FOR INTEGER ARITHMETIC",
        variant_id="var_int_custom_12345",
        score=1.5,
        rationale="Proven fix for integer overflow",
        topological_rule="RULE_CHECK_OVERFLOW",
    )
    client = ReflectorClient(registry=registry, in_process=True)

    args = _parse_args([])
    seed = {
        "topic": "int x = a + b;",
        "predicate": "integer_overflow leading to wraparound",
    }
    payload = _build_payload(
        args=args,
        item_seed=100,
        checkpoint_path="artifacts/checkpoints/100.json",
        record_path="artifacts/runs/100.json",
        batch_started_at="2026-08-23T12:00:00Z",
        seed=seed,
        reflector_client=client,
    )
    assert payload["config"]["taxonomy_bucket"] == "integer_arithmetic"
    assert payload["config"]["active_mutation_id"] == "var_int_custom_12345"
    assert payload["config"]["reflector_prompt"] == "CUSTOM PARETO PROMPT FOR INTEGER ARITHMETIC"


@pytest.mark.asyncio
async def test_process_seed_records_trace_on_completion(temp_gepa_dir):
    """Verify that _process_seed durable records attempt outcome to traces.jsonl."""
    registry = ParetoRegistry(gepa_dir=temp_gepa_dir)
    client = ReflectorClient(registry=registry, in_process=True)

    args = _parse_args(["--run-id", "test-run-step3", "--output", "/tmp/test_out.jsonl"])
    manifest = {
        "items": [
            {"index": 0, "status": "pending", "seed": 42}
        ]
    }
    ctx = BatchContext(
        args=args,
        batch_started_at="2026-08-23T12:00:00Z",
        processed_predicates=set(),
        manifest=manifest,
        manifest_path=str(temp_gepa_dir / "manifest.json"),
        manifest_lock=asyncio.Lock(),
        total_seeds=1,
        reflector_client=client,
        pareto_registry=registry,
    )

    fake_judge_response = {
        "status": "completed",
        "decision": "accepted",
        "response": "Adjudication succeeded",
        "verifier_logic_error": False,
    }

    with patch("scenarios.debate.run_batch.send_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = fake_judge_response
        sem = asyncio.Semaphore(1)
        seed = {
            "cve_id": "CVE-2026-0001",
            "topic": "memcpy(dst, src, len);",
            "predicate": "buffer_overflow in memory write",
        }
        await _process_seed(sem, 0, seed, ctx)

    # Check that traces.jsonl was populated
    traces = registry.get_recent_traces("memory_safety", limit=10)
    assert len(traces) == 1
    assert traces[0]["outcome"] == "VALID_ACCEPT"
    assert traces[0]["scenario_id"] == "CVE-2026-0001"
    assert traces[0]["attempt_index"] == 1


@pytest.mark.asyncio
async def test_process_seed_concurrent_execution(temp_gepa_dir):
    """Verify concurrent batch execution does not corrupt traces.jsonl or Pareto frontier."""
    registry = ParetoRegistry(gepa_dir=temp_gepa_dir)
    client = ReflectorClient(registry=registry, in_process=True)

    num_seeds = 10
    args = _parse_args(["--run-id", "test-run-concurrent", "--output", "/tmp/test_out_concurrent.jsonl"])
    manifest = {
        "items": [
            {"index": i, "status": "pending", "seed": 42 + i}
            for i in range(num_seeds)
        ]
    }
    ctx = BatchContext(
        args=args,
        batch_started_at="2026-08-23T12:00:00Z",
        processed_predicates=set(),
        manifest=manifest,
        manifest_path=str(temp_gepa_dir / "manifest_concurrent.json"),
        manifest_lock=asyncio.Lock(),
        total_seeds=num_seeds,
        reflector_client=client,
        pareto_registry=registry,
    )

    async def fake_send_message(payload_str, url):
        await asyncio.sleep(0.01)
        return {
            "status": "completed",
            "decision": "accepted",
            "response": "Adjudication succeeded",
            "verifier_logic_error": False,
        }

    with patch("scenarios.debate.run_batch.send_message", side_effect=fake_send_message):
        sem = asyncio.Semaphore(4)
        seeds = [
            {
                "cve_id": f"CVE-2026-{1000+i}",
                "topic": f"void vuln_{i}() {{ ... }}",
                "predicate": "concurrency data_race on shared state" if i % 2 == 0 else "buffer_overflow in memory",
            }
            for i in range(num_seeds)
        ]
        await asyncio.gather(*(_process_seed(sem, i, seeds[i], ctx) for i in range(num_seeds)))

    # Verify traces were cleanly recorded for both buckets
    concurrency_traces = registry.get_recent_traces("concurrency", limit=20)
    memory_traces = registry.get_recent_traces("memory_safety", limit=20)
    assert len(concurrency_traces) + len(memory_traces) == num_seeds
