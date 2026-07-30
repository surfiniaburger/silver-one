"""Unit tests for BARRED pre-filter integration in run_batch.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest

from scenarios.debate.run_batch import _parse_args, _build_payload, _process_seed
from scenarios.debate.pre_filter import BarredPreFilter, PreFilterDecision


def test_parse_args_pre_filter_flags():
    """Verify --pre-filter / --no-pre-filter CLI flag parsing."""
    args_default = _parse_args([])
    assert args_default.pre_filter is True
    assert args_default.model_dir == "artifacts/models"

    args_disabled = _parse_args(["--no-pre-filter"])
    assert args_disabled.pre_filter is False

    args_custom = _parse_args(["--pre-filter", "--model-dir", "custom/models"])
    assert args_custom.pre_filter is True
    assert args_custom.model_dir == "custom/models"


def test_build_payload_structure():
    """Verify _build_payload creates valid green agent request payload."""
    class DummyArgs:
        run_id = "test-run"
        mode = "record"
        resume = False
        cassette_path = ""
        record_path = ""
        record_dir = "artifacts/runs"
        attempts_out = ""
        output = "output.jsonl"

    seed = {"topic": "test code", "predicate": "buffer overflow check"}
    payload = _build_payload(DummyArgs(), 42, "checkpoint.json", "record.json", "2026-07-30T12:00:00Z", seed)

    assert payload["config"]["run_id"] == "test-run"
    assert payload["config"]["seed"] == 42
    assert payload["config"]["topic"] == "test code"
    assert payload["config"]["predicate"] == "buffer overflow check"


@pytest.mark.asyncio
async def test_process_seed_pre_filter_rejection(tmp_path: Path):
    """Verify _process_seed intercepts rejected seeds and appends attempt records."""
    from unittest.mock import MagicMock
    from scenarios.debate.run_batch import BatchContext, _process_seed

    attempts_file = tmp_path / "attempts.jsonl"
    manifest_file = tmp_path / "manifest.json"

    class DummyArgs:
        run_id = "test-run"
        seed = 42
        attempts_out = str(attempts_file)
        checkpoint_dir = str(tmp_path / "checkpoints")
        record_path = ""
        record_dir = str(tmp_path / "runs")

    mock_pre_filter = MagicMock()
    mock_pre_filter.predict.return_value = PreFilterDecision(
        accept=False,
        probability=0.01,
        stage="xgboost",
        elapsed_ms=1.2,
    )

    manifest = {"items": [{"status": "pending"}]}
    ctx = BatchContext(
        args=DummyArgs(),
        batch_started_at="2026-07-30T12:00:00Z",
        processed_predicates=set(),
        manifest=manifest,
        manifest_path=str(manifest_file),
        manifest_lock=asyncio.Lock(),
        total_seeds=1,
        pre_filter=mock_pre_filter,
    )

    sem = asyncio.Semaphore(1)
    seed = {
        "predicate": "buffer overflow in memcpy",
        "input_block": "void vuln() { memcpy(a, b, c); }",
        "topic": "test code",
        "cve_id": "CVE-2024-9999",
    }

    await _process_seed(sem, 0, seed, ctx)

    # Verify pre-filter was called with correct predicate and input_block
    mock_pre_filter.predict.assert_called_once_with(
        "buffer overflow in memcpy",
        "void vuln() { memcpy(a, b, c); }",
    )

    # Verify manifest was updated to skipped_pre_filter
    assert manifest["items"][0]["status"] == "skipped_pre_filter"
    assert "Rejected at xgboost" in manifest["items"][0]["response_excerpt"]

    # Verify attempts log record was created
    assert attempts_file.exists()
    records = [json.loads(line) for line in attempts_file.read_text("utf-8").strip().splitlines()]
    assert len(records) == 1
    assert records[0]["decision"] == "rejected"
    assert records[0]["pre_filter_stage"] == "xgboost"
    assert records[0]["skipped_pre_filter"] is True
    assert records[0]["input_block"] == "void vuln() { memcpy(a, b, c); }"
    assert records[0]["cve_id"] == "CVE-2024-9999"
