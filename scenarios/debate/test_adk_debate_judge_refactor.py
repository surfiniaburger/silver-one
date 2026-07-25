import json
import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock

from agentbeats.models import EvalRequest
from debate_judge_common import DebateEval, DebaterScore, VerifierReport
from scenarios.debate.adk_debate_judge import (
    DebateJudgeADK,
    VerifierMeta,
    _EvalContext,
    _AcceptedSamplePayload,
    _mechanism_evidence_gate,
)


def test_eval_context_initialization_and_controls():
    with tempfile.TemporaryDirectory() as td:
        req = EvalRequest(
            participants={"pro_debater": "http://localhost:9001", "con_debater": "http://localhost:9002"},
            config={
                "run_id": "test-run-1",
                "seed": 42,
                "mode": "passthrough",
                "record_path": os.path.join(td, "run.json"),
                "attempts_path": os.path.join(td, "attempts.jsonl"),
                "checkpoint_path": os.path.join(td, "checkpoint.json"),
                "topic": "def foo(): pass",
            }
        )
        judge = DebateJudgeADK()
        ctx = _EvalContext(req, judge)

        assert ctx.run_id == "test-run-1"
        assert ctx.seed == 42
        assert ctx.mode == "passthrough"
        assert ctx.record_path == os.path.join(td, "run.json")
        assert ctx.attempts_path == os.path.join(td, "attempts.jsonl")


@pytest.mark.asyncio
async def test_get_sample_data_refinement_without_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        req = EvalRequest(
            participants={"pro_debater": "http://localhost:9001", "con_debater": "http://localhost:9002"},
            config={
                "run_id": "test-run-sample",
                "seed": 42,
                "mode": "passthrough",
                "checkpoint_path": os.path.join(td, "ckpt.json"),
                "topic": "def initial(): pass",
            }
        )
        judge = DebateJudgeADK()
        ctx = _EvalContext(req, judge)

        judge.generator = MagicMock()
        judge.generator.generate_boundary_sample = AsyncMock(return_value={"revised_input_block": "def gen_0(): pass"})
        judge.generator.refine_sample = AsyncMock(return_value={"revised_input_block": "def gen_1(): pass"})

        updater = MagicMock()
        updater.update_status = AsyncMock()

        # Pass 0: initial generation
        s_data_0, block_0 = await judge._get_sample_data(ctx, 0, None, "start", "", updater)
        assert block_0 == "def gen_0(): pass"

        # Pass 1: refinement (no active checkpoint)
        s_data_1, block_1 = await judge._get_sample_data(ctx, 1, None, "start", "Rejected by judge", updater)
        assert block_1 == "def gen_1(): pass"
        judge.generator.refine_sample.assert_called_once_with(
            "def initial(): pass",
            ctx.predicate,
            ctx.target_dimension,
            ctx.target_verdict,
            "def gen_0(): pass",
            "Rejected by judge",
        )


def test_mechanism_evidence_gate_refactored():
    # Test invalid non-string mechanism
    res = _mechanism_evidence_gate(123, "def foo(): pass", ["foo"])
    assert res["pass"] is False

    # Test valid mechanism with operation anchors
    input_text = "def check_auth(user):\n    if not user.is_admin:\n        raise PermissionError('Denied')"
    mechanism = "Calls check_auth(user) and raises PermissionError when not admin."
    res = _mechanism_evidence_gate(mechanism, input_text, ["check_auth(user)"])
    assert res["pass"] is True
    assert res["has_code_token"] is True
    assert res["has_operation_anchor"] is True


def test_evaluate_verifier_audit_passed_and_failed():
    with tempfile.TemporaryDirectory() as td:
        attempts_path = os.path.join(td, "attempts.jsonl")
        req = EvalRequest(
            participants={"pro_debater": "http://localhost:9001", "con_debater": "http://localhost:9002"},
            config={
                "run_id": "test-run-2",
                "seed": 42,
                "mode": "passthrough",
                "attempts_path": attempts_path,
            }
        )
        judge = DebateJudgeADK()
        ctx = _EvalContext(req, judge)

        pro = DebaterScore(technical_accuracy=1.0, logic_soundness=1.0, evidence_strength=1.0, total_score=1.0, critique="Pro critique")
        con = DebaterScore(technical_accuracy=0.5, logic_soundness=0.5, evidence_strength=0.5, total_score=0.5, critique="Con critique")
        debate_eval = DebateEval(
            thinking_process="Thinking...",
            predicate="Is vulnerable",
            anchors=["check_auth", "user"],
            support_level="supported",
            verifier_report="Audit passed",
            winner="pro_debater",
            reason="Clear vulnerability found",
            mechanism="Calls check_auth()",
            counterfactual="Fix check_auth()",
            pro_debater=pro,
            con_debater=con,
        )

        verifier_meta: VerifierMeta = {
            "called": True,
            "from_cache": False,
            "parse_ok": True,
            "passes_audit": True,
            "logic_error": None,
            "error": None,
            "raw_response": "{}",
            "llm_usage": None,
            "model": "verifier-v1",
            "url": "http://localhost:9020",
        }

        # Case 1: Passed audit
        passed_report = VerifierReport(
            thinking_process="Thinking...",
            passes_audit=True,
            anchor_analysis="Anchors verified",
            logic_error=None,
            suggested_correction=None,
        )
        ok, reason = judge._evaluate_verifier_audit(
            ctx, 0, passed_report, verifier_meta, debate_eval, ["check_auth"], {}, "def foo(): pass"
        )
        assert ok is True
        assert reason == "Clear vulnerability found"

        # Case 2: Failed audit due to logic error
        failed_report = VerifierReport(
            thinking_process="Thinking...",
            passes_audit=False,
            anchor_analysis="Anchor analysis failed",
            logic_error="Null pointer exception unhandled",
            suggested_correction="Add null check",
        )
        ok_fail, reason_fail = judge._evaluate_verifier_audit(
            ctx, 0, failed_report, verifier_meta, debate_eval, ["check_auth"], {}, "def foo(): pass"
        )
        assert ok_fail is False
        assert "VERIFIER AUDIT FAILED" in reason_fail
        assert os.path.exists(attempts_path)


@pytest.mark.asyncio
async def test_export_accepted_sample_payload():
    with tempfile.TemporaryDirectory() as td:
        output_file = os.path.join(td, "corpus.jsonl")
        attempts_path = os.path.join(td, "attempts.jsonl")
        record_path = os.path.join(td, "record.json")
        req = EvalRequest(
            participants={"pro_debater": "http://localhost:9001", "con_debater": "http://localhost:9002"},
            config={
                "run_id": "test-run-3",
                "seed": 100,
                "mode": "passthrough",
                "output_file": output_file,
                "attempts_path": attempts_path,
                "record_path": record_path,
                "predicate": "Input contains SQL injection",
                "target_verdict": "True",
            }
        )
        judge = DebateJudgeADK()
        ctx = _EvalContext(req, judge)

        pro = DebaterScore(technical_accuracy=1.0, logic_soundness=1.0, evidence_strength=1.0, total_score=1.0, critique="Pro critique")
        con = DebaterScore(technical_accuracy=0.5, logic_soundness=0.5, evidence_strength=0.5, total_score=0.5, critique="Con critique")
        debate_eval = DebateEval(
            thinking_process="Thinking...",
            predicate="Input contains SQL injection",
            anchors=["query", "input_str"],
            support_level="supported",
            verifier_report="Passed",
            winner="pro_debater",
            reason="Unsanitized query input",
            mechanism="Calls query()",
            counterfactual="Use parameterized query",
            pro_debater=pro,
            con_debater=con,
        )

        verifier_meta: VerifierMeta = {
            "called": True,
            "from_cache": False,
            "parse_ok": True,
            "passes_audit": True,
            "logic_error": None,
            "error": None,
            "raw_response": "{}",
            "llm_usage": None,
            "model": "verifier-v1",
            "url": "http://localhost:9020",
        }

        payload = _AcceptedSamplePayload(
            current_sample_block="db.query(input_str)",
            debate_eval=debate_eval,
            normalized_anchors=["query"],
            verifier_meta=verifier_meta,
            verifier_audit=VerifierReport(
                thinking_process="Thinking...",
                passes_audit=True,
                anchor_analysis="Anchors verified",
                logic_error=None,
                suggested_correction=None,
            ),
            sample_data={"topic": "SQLi"},
            debate={"transcript": []},
            soft_checks={"predicate_quality": True},
            anchor_stats={"hits": 1},
            last_judge_reason="Unsanitized query input",
        )

        updater = MagicMock()
        updater.update_status = AsyncMock()
        updater.add_artifact = AsyncMock()

        await judge._export_accepted_sample(ctx, 0, payload, updater)

        assert os.path.exists(output_file)
        with open(output_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["input"] == "db.query(input_str)"
            assert data["output"]["verdict"] == "1"


def test_merge_usage_summaries():
    from scenarios.debate.adk_debate_judge import _merge_usage_summaries

    primary = {
        "totals": {"calls": 2, "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost_usd": 0.01},
        "by_stage": {"judge": {"calls": 1, "prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70, "cost_usd": 0.005}},
        "by_model": {"m1": {"calls": 2, "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost_usd": 0.01}},
        "events": [{"event": "e1"}],
        "models": {"judge": "m1"},
        "generation_config": {"temp": 0.2},
        "notes": {},
    }

    secondary = {
        "totals": {"calls": 1, "prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50, "cost_usd": 0.002},
        "by_stage": {"verifier": {"calls": 1, "prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50, "cost_usd": 0.002}},
        "by_model": {"m2": {"calls": 1, "prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50, "cost_usd": 0.002}},
        "events": [{"event": "e2"}],
        "models": {"verifier": "m2"},
        "generation_config": {"verifier_temp": 0.0},
    }

    merged = _merge_usage_summaries(primary, secondary)

    assert merged["totals"]["calls"] == 3
    assert merged["totals"]["prompt_tokens"] == 140
    assert merged["totals"]["completion_tokens"] == 60
    assert merged["totals"]["total_tokens"] == 200
    assert abs(merged["totals"]["cost_usd"] - 0.012) < 1e-6
    assert "judge" in merged["by_stage"]
    assert "verifier" in merged["by_stage"]
    assert len(merged["events"]) == 2
    assert merged["models"]["verifier"] == "m2"

