import json
import os
import sys
import time
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agentbeats.tracing import trace_span, set_current_trace_id, reset_current_trace_id
from agentbeats.replay import ReplayManager, RunRecord, LLMCassette
from scripts.unified_compare import _get_latency_slo_metric


def test_trace_span_context(tmp_path):
    reset_current_trace_id()
    set_current_trace_id("test_trace_12345")
    
    with trace_span("test_span_op", stage="test_stage", attributes={"key": "val"}, spans_dir=tmp_path) as span:
        time.sleep(0.01)
        assert span.trace_id == "test_trace_12345"
        assert span.name == "test_span_op"

    spans_file = tmp_path / "spans.jsonl"
    assert spans_file.exists()
    
    lines = spans_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["trace_id"] == "test_trace_12345"
    assert data["name"] == "test_span_op"
    assert data["stage"] == "test_stage"
    assert data["status"] == "OK"
    assert data["duration_ms"] > 0.0
    assert data["attributes"]["key"] == "val"
    reset_current_trace_id()


def test_trace_span_error(tmp_path):
    spans_file = tmp_path / "spans.jsonl"

    def _trigger_span_error():
        with trace_span("failing_op", stage="test_stage", spans_dir=tmp_path):
            raise ValueError("Boom")

    with pytest.raises(ValueError, match="Boom"):
        _trigger_span_error()

    assert spans_file.exists()
    data = json.loads(spans_file.read_text(encoding="utf-8").strip())
    assert data["status"] == "ERROR"
    assert "Boom" in data["error_message"]


def test_replay_manager_latency_aggregation(tmp_path):
    cassette_file = tmp_path / "test_cassette.json"
    record = RunRecord(
        run_id="test-run",
        rng_seed=42,
        models={"default": "test-model"},
        created_at="2026-06-01T00:00:00Z",
    )
    cassette = LLMCassette(str(cassette_file), mode="record")
    rm = ReplayManager(record, cassette)

    rm._record_usage_event(
        stage="test_stage_1",
        model="test-model",
        messages=[],
        response_obj={"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        source="provider",
        duration_ms=150.0,
    )
    rm._record_usage_event(
        stage="test_stage_2",
        model="test-model",
        messages=[],
        response_obj={"usage": {"prompt_tokens": 5, "completion_tokens": 15}},
        source="provider",
        duration_ms=350.0,
    )

    summary = rm.get_usage_summary()
    totals = summary["totals"]
    assert totals["total_duration_ms"] == pytest.approx(500.0)
    assert totals["avg_duration_ms"] == pytest.approx(250.0)
    assert totals["max_duration_ms"] == pytest.approx(350.0)
    assert totals["min_duration_ms"] == pytest.approx(150.0)

    by_stage = summary["by_stage"]
    assert by_stage["test_stage_1"]["total_duration_ms"] == pytest.approx(150.0)
    assert by_stage["test_stage_2"]["total_duration_ms"] == pytest.approx(350.0)


def test_unified_compare_latency_slo_metric(monkeypatch):
    cr_data = {
        "__metadata__": {
            "code_review_usage_summary": {
                "usage": {
                    "totals": {
                        "calls": 2,
                        "total_duration_ms": 2500.0,
                        "avg_duration_ms": 1250.0,
                        "max_duration_ms": 1800.0,
                    }
                }
            }
        }
    }

    result = _get_latency_slo_metric(cr_data)
    assert result is not None
    metric_str, status = result
    assert status == "PASS"
    assert "2.50s total" in metric_str
    assert "1250.0ms avg/call" in metric_str

    # Test SLA violation status WARN
    monkeypatch.setenv("LATENCY_SLO_MAX_CALL_MS", "1000")
    result_warn = _get_latency_slo_metric(cr_data)
    assert result_warn is not None
    _, warn_status = result_warn
    assert warn_status == "WARN"
