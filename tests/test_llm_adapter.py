import os
import asyncio
import json

import pytest
import sys
import os

# Ensure scripts directory is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
import llm_adapter
import importlib
farley = importlib.import_module('farley_score_evaluator')
evaluator = farley

# Sample minimal Farley JSON matching the fallback schema
SAMPLE_FARLEY = {
    "understandable": {"score": 9, "rationale": "Clear name", "suggestions": []},
    "maintainable": {"score": 8, "rationale": "Good", "suggestions": []},
    "repeatable": {"score": 10, "rationale": "Deterministic", "suggestions": []},
    "atomic": {"score": 9, "rationale": "One assert", "suggestions": []},
    "necessary": {"score": 10, "rationale": "Necessary", "suggestions": []},
    "granular": {"score": 9, "rationale": "Focused", "suggestions": []},
    "fast": {"score": 10, "rationale": "Fast", "suggestions": []},
    "first_tdd": {"score": 8, "rationale": "TDD style", "suggestions": []},
    "summary": "Good test"
}


class DummyReplay:
    def __init__(self):
        self.store = {}
    def lookup(self, stage, model, messages):
        # no lookup
        return None
    def get(self, request_id):
        return None
    def save_response(self, stage, model, messages, payload):
        key = f"{stage}:{model}:{messages[-1]['content'][:40]}"
        self.store[key] = payload


@pytest.mark.asyncio
async def test_call_structured_litellm(monkeypatch, tmp_path):
    # Patch internal litellm caller to return JSON string
    async def fake_litellm(model, messages, params):
        return json.dumps(SAMPLE_FARLEY)
    monkeypatch.setattr(llm_adapter, "_call_litellm_async", fake_litellm)

    replay = DummyReplay()
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do it"}]
    result = await llm_adapter.call_structured(replay, "litellm/qwen3.5:2b", messages, "FarleyScoreBreakdown", evaluator.FarleyScoreBreakdown, "test_stage")
    assert result.summary == "Good test"
    assert result.understandable.score == 9


@pytest.mark.asyncio
async def test_call_structured_nebius(monkeypatch):
    async def fake_nebius(model, messages, params):
        return json.dumps(SAMPLE_FARLEY)
    monkeypatch.setattr(llm_adapter, "_call_nebius_async", fake_nebius)

    result = await llm_adapter.call_structured(None, "nebius/some-model", [{"role":"system","content":"s"},{"role":"user","content":"u"}], "FarleyScoreBreakdown", evaluator.FarleyScoreBreakdown, "s")
    assert result.summary == "Good test"


def test_select_provider_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nebius")
    assert llm_adapter._select_provider("anystring") == "nebius"
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


@pytest.mark.asyncio
async def test_replay_saved(monkeypatch):
    async def fake_litellm(model, messages, params):
        return json.dumps(SAMPLE_FARLEY)
    monkeypatch.setattr(llm_adapter, "_call_litellm_async", fake_litellm)

    replay = DummyReplay()
    messages = [{"role":"system","content":"sys"},{"role":"user","content":"unique_input_xyz"}]
    res = await llm_adapter.call_structured(replay, "litellm/qwen3.5:2b", messages, "FarleyScoreBreakdown", evaluator.FarleyScoreBreakdown, "s")
    # ensure something was saved
    assert any("unique_input_xyz" in k for k in replay.store.keys())
