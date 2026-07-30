"""Unit tests for BARRED pre-filter integration in run_batch.py."""

from __future__ import annotations

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
