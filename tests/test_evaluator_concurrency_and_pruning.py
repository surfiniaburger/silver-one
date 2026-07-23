import asyncio
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import scripts.code_review_evaluator
from scripts import telemetry_utils
from scripts.code_review_evaluator import (
    _prune_unit_code,
    estimate_unit_tokens,
    evaluate_units,
    format_unit,
)
from scripts.farley_score_evaluator import _prune_test_code, evaluate_files


def test_prune_unit_code_short():
    short_code = "def foo():\n    return 42\n"
    assert _prune_unit_code(short_code, max_lines=10) == short_code


def test_prune_unit_code_long():
    lines = [f"    line_{i} = {i}" for i in range(150)]
    long_code = "\n".join(lines)
    pruned = _prune_unit_code(long_code, max_lines=50)
    assert "... [100 lines truncated for context efficiency] ..." in pruned
    assert "line_0" in pruned
    assert "line_149" in pruned


def test_prune_test_code():
    lines = [f"def test_{i}(): pass" for i in range(120)]
    code = "\n".join(lines)
    pruned = _prune_test_code(code, max_lines=40)
    assert "... [80 lines truncated for context efficiency] ..." in pruned


def test_prune_code_text_edge_cases():
    assert telemetry_utils.prune_code_text(None) == ""
    assert telemetry_utils.prune_code_text("") == ""
    assert telemetry_utils.prune_code_text(12345) == "12345"


def test_format_unit_uses_pruned_code():
    unit = {
        "file_path": "foo.py",
        "name": "bar",
        "class_name": None,
        "start_line": 1,
        "end_line": 200,
        "lines_changed": 150,
        "code": "\n".join([f"line_{i}" for i in range(150)]),
    }
    formatted = format_unit(unit)
    assert "... [50 lines truncated for context efficiency] ..." in formatted


def test_estimate_unit_tokens_uses_pruned_code():
    long_unit = {
        "file_path": "large.py",
        "name": "large_func",
        "code": "\n".join([f"line_{i} = 'some long string content here'" for i in range(1000)]),
    }
    estimated = estimate_unit_tokens(long_unit)
    pruned_code = _prune_unit_code(long_unit["code"])
    expected = len(pruned_code) // 4 + (len(scripts.code_review_evaluator.SYSTEM_PROMPT) // 4 + 200)
    assert estimated == expected


def test_coerce_int_defensive_defaults():
    assert telemetry_utils.coerce_int(None, default=2) == 2
    assert telemetry_utils.coerce_int("", default=2) == 2
    assert telemetry_utils.coerce_int("invalid", default=2) == 2
    assert telemetry_utils.coerce_int("5", default=2) == 5
    assert telemetry_utils.coerce_int(4, default=2) == 4


@pytest.mark.asyncio
async def test_evaluate_units_concurrency_and_error_handling():
    units = [
        {"file_path": f"file_{i}.py", "name": f"func_{i}", "code": f"def func_{i}(): pass"}
        for i in range(5)
    ]
    # Add a non-dict unit to test error scoping
    units.append("invalid_unit_string")  # type: ignore

    active_tasks = 0
    max_observed_concurrency = 0

    async def mock_call_structured(*args, **kwargs):
        nonlocal active_tasks, max_observed_concurrency
        active_tasks += 1
        max_observed_concurrency = max(max_observed_concurrency, active_tasks)
        await asyncio.sleep(0.05)
        active_tasks -= 1
        return None, '{"readability": {"score": 9.0, "rationale": "ok", "suggestions": []}}', {}

    with patch("scripts.llm_adapter.call_structured_with_raw_and_diagnostics", side_effect=mock_call_structured):
        results = await evaluate_units(replay_mgr=None, model="test-model", units=units, max_concurrency=2)

    assert len(results) == 6
    assert max_observed_concurrency <= 2
    # Check that the invalid unit was handled gracefully as provider error
    assert results[-1]["file_path"] == "unknown"


@pytest.mark.asyncio
async def test_farley_evaluate_files_concurrency():
    active_tasks = 0
    max_observed_concurrency = 0

    async def mock_eval_single(replay_mgr, model, tc, filepath):
        nonlocal active_tasks, max_observed_concurrency
        active_tasks += 1
        max_observed_concurrency = max(max_observed_concurrency, active_tasks)
        await asyncio.sleep(0.05)
        active_tasks -= 1
        return {"farley_index": 8.5}

    test_cases = [{"name": f"test_{i}", "code": "pass", "class_name": None} for i in range(6)]

    with patch("scripts.farley_score_evaluator.extract_tests_from_file", return_value=test_cases), \
         patch("scripts.farley_score_evaluator._evaluate_single_test", side_effect=mock_eval_single):
        indices, count, _ = await evaluate_files(
            replay_mgr=None,
            model="test-model",
            target_files=["tests/test_dummy.py"],
            max_concurrency=2
        )

    assert count == 6
    assert len(indices) == 6
    assert max_observed_concurrency <= 2
