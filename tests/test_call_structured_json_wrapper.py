import asyncio
import json

import pytest

from scripts import llm_adapter


class DummyModel:
    def __init__(self, data):
        self.data = data

    @classmethod
    def model_validate_json(cls, raw_json: str):
        payload = json.loads(raw_json)
        return cls(payload)


@pytest.mark.asyncio
async def test_call_structured_handles_markdown_wrapped_json(monkeypatch):
    async def fake_litellm(model, messages, params, schema_model=None):
        # return JSON wrapped in a markdown code fence and extra commentary
        return "Some preface\n```json\n{\"understandable\": {\"score\": 8, \"rationale\": \"ok\", \"suggestions\": []}, \"maintainable\": {\"score\": 7, \"rationale\": \"ok\", \"suggestions\": []}, \"repeatable\": {\"score\": 9, \"rationale\": \"ok\", \"suggestions\": []}, \"atomic\": {\"score\": 8, \"rationale\": \"ok\", \"suggestions\": []}, \"necessary\": {\"score\": 6, \"rationale\": \"ok\", \"suggestions\": []}, \"granular\": {\"score\": 7, \"rationale\": \"ok\", \"suggestions\": []}, \"fast\": {\"score\": 9, \"rationale\": \"ok\", \"suggestions\": []}, \"first_tdd\": {\"score\": 5, \"rationale\": \"ok\", \"suggestions\": []}, \"summary\": \"ok\"}\n```\nSome suffix"

    monkeypatch.setattr(llm_adapter, "_call_litellm_async", fake_litellm)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    validated = await llm_adapter.call_structured(None, "litellm/foo", messages, "FarleyScoreBreakdown", DummyModel, "stage")
    assert isinstance(validated, DummyModel)
    assert validated.data.get('summary') == 'ok'
