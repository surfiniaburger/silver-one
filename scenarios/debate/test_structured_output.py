import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agentbeats.structured_output import call_structured


class MiniModel(BaseModel):
    answer: str


@dataclass
class _DummyChoice:
    message: Any


@dataclass
class _DummyMessage:
    content: str


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [_DummyChoice(message=_DummyMessage(content=content))]


class _DummyCassette:
    def __init__(self):
        self.saved: List[Dict[str, Any]] = []

    def save_response(self, model: str, messages: list, params: Dict[str, Any], response: Any):
        self.saved.append(
            {"model": model, "messages": messages, "params": params, "response": response}
        )


class _DummyReplayManager:
    def __init__(
        self,
        *,
        first: Optional[str] = None,
        repair: Optional[str] = None,
        reject_schema: bool = False,
        retry: Optional[str] = None,
    ):
        self.cassette = _DummyCassette()
        self._first = first
        self._repair = repair
        self._reject_schema = reject_schema
        self._retry = retry
        self._calls = 0

    async def acompletion(self, model: str, messages: list, **kwargs):
        self._calls += 1
        if self._reject_schema and "response_format" in kwargs:
            raise RuntimeError("BadRequestError: response_format json_schema not supported")

        if self._calls == 1:
            return _DummyResponse(self._first or '{"answer":"ok"}')
        return _DummyResponse(self._repair or self._retry or '{"answer":"ok"}')


def test_validation_failure_repair():
    rm = _DummyReplayManager(first="not json", repair='{"answer":"fixed"}')
    result = asyncio.run(
        call_structured(
            replay_manager=rm,
            model="dummy",
            messages=[{"role": "user", "content": "hi"}],
            schema_name="mini",
            schema_model=MiniModel,
            strict=True,
            repair_on_fail=True,
        )
    )
    assert result.answer == "fixed"
    assert any(
        item["model"] == "event/structured_output_validation_error" for item in rm.cassette.saved
    )


def test_response_format_rejected_retry():
    rm = _DummyReplayManager(reject_schema=True, first='{"answer":"ok"}', retry='{"answer":"retry"}')
    result = asyncio.run(
        call_structured(
            replay_manager=rm,
            model="dummy",
            messages=[{"role": "user", "content": "hi"}],
            schema_name="mini",
            schema_model=MiniModel,
            strict=True,
            repair_on_fail=True,
        )
    )
    assert result.answer in ("ok", "retry")
    assert any(
        item["model"] == "event/structured_output_response_format_rejected"
        for item in rm.cassette.saved
    )


def test_markdown_fenced_json_parses():
    rm = _DummyReplayManager(first='```json\n{"answer":"ok"}\n```')
    result = asyncio.run(
        call_structured(
            replay_manager=rm,
            model="dummy",
            messages=[{"role": "user", "content": "hi"}],
            schema_name="mini",
            schema_model=MiniModel,
            strict=True,
            repair_on_fail=False,
        )
    )
    assert result.answer == "ok"


def test_invalid_escape_json_parses():
    rm = _DummyReplayManager(first='{"answer":"IP\\-Header"}')
    result = asyncio.run(
        call_structured(
            replay_manager=rm,
            model="dummy",
            messages=[{"role": "user", "content": "hi"}],
            schema_name="mini",
            schema_model=MiniModel,
            strict=True,
            repair_on_fail=False,
        )
    )
    assert result.answer == "IP\\-Header"


if __name__ == "__main__":
    test_validation_failure_repair()
    test_response_format_rejected_retry()
    print("ok")
