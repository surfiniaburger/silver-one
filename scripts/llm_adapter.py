import os
import json
import asyncio
import time
import math
from typing import Any, Dict, List, Type

try:
    # Optional dependency: litellm async client
    from litellm import acompletion
except Exception:
    acompletion = None

import requests

LITELLM_PREFIX = "litellm/"

PRESETS = {
    "instruct": {"temperature": 0.0, "top_p": 0.8, "max_tokens": 1024},
    "qwen3.5:thinking-general": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    "qwen3.5:thinking-precise-coding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
}


def _select_provider(model: str) -> str:
    model = model or ""
    if model.startswith("nebius/") or os.environ.get("LLM_PROVIDER") == "nebius":
        return "nebius"
    if model.startswith(LITELLM_PREFIX) or model.startswith("local/") or os.environ.get("LLM_PROVIDER") == "litellm":
        return "litellm"
    # fallback preference
    return os.environ.get("LLM_PROVIDER", "litellm")


async def _call_nebius_async(model: str, messages: List[Dict], params: Dict[str, Any]) -> str:
    """Call an OpenAI-compatible Nebius endpoint synchronously in a thread.
    Expects `NEBIUS_API_KEY` and optionally `NEBIUS_BASE_URL` in env.
    Returns the model's raw text output (string).
    """
    def _sync_call():
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise RuntimeError("NEBIUS_API_KEY not set")
        base = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
        url = f"{base.rstrip('/')}/responses"
        payload = {
            "model": model.replace("nebius/", ""),
            "input": messages[-1]["content"],
            "instructions": messages[0]["content"] if messages else "",
        }
        # merge params for backwards compat
        payload.update(params or {})
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Try common shapes
        if "output" in data:
            return data["output"]
        if "choices" in data and data["choices"]:
            c = data["choices"][0]
            return c.get("message", c.get("text", ""))
        return json.dumps(data)

    return await asyncio.to_thread(_sync_call)


def _retry_sync(fn, retries=2, backoff=0.5):
    last = None
    for i in range(retries+1):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** i))
    raise last


def _parse_ollama_response(d: Any) -> str:
    if isinstance(d, dict):
        # Ollama /api/generate returns a 'response' field
        if "response" in d:
            return d["response"]
        if "text" in d:
            return d["text"]
        if "choices" in d and d["choices"]:
            choice = d["choices"][0]
            msg = choice.get("message", "")
            if isinstance(msg, dict):
                return msg.get("content", "")
            return msg or choice.get("text", "")
    return json.dumps(d)


def _local_ollama_sync_call(model: str, messages: List[Dict]) -> str:
    url = os.environ.get("LITELLM_HTTP", "http://localhost:11434/api/generate")
    clean_model = model.replace(LITELLM_PREFIX, "")
    if "/" in clean_model:
        clean_model = clean_model.split("/", 1)[1]
    payload = {
        "model": clean_model,
        # Ollama /api/generate expects 'prompt', not 'input'
        "prompt": messages[-1].get("content", "") if messages else "",
        "system": messages[0].get("content", "") if messages else "",
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return _parse_ollama_response(r.json())


async def _call_litellm_async(model: str, messages: List[Dict], params: Dict[str, Any]) -> str:
    """Call litellm's async completion if available, else try local HTTP endpoints (ollama/litellm)."""
    if acompletion is not None:
        clean_model = model.replace(LITELLM_PREFIX, "")
        resp = await acompletion(
            model=clean_model,
            messages=messages,
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 1024)
        )
        if hasattr(resp, "choices") and resp.choices:
            msg = resp.choices[0].message
            # Qwen3 / thinking models put the answer in reasoning_content when
            # content is None or empty (extended-thinking mode).
            content = getattr(msg, "content", None) or ""
            if not content:
                content = (
                    getattr(msg, "reasoning_content", None)
                    or getattr(msg, "thinking", None)
                    or ""
                )
            return content
        return str(resp)

    fn = lambda: _local_ollama_sync_call(model, messages)
    return await asyncio.to_thread(lambda: _retry_sync(fn, retries=3, backoff=0.2))


def estimate_tokens(text: str) -> int:
    # Very small heuristic: 1 token ~= 4 chars
    return max(1, math.ceil(len(text) / 4))


def _replay_lookup(replay_manager: Any, stage: str, model: str, messages: List[Dict], schema_model: Type) -> Any:
    # ReplayManager exposes its cassette via .cassette (an LLMCassette instance)
    cassette = getattr(replay_manager, "cassette", None)
    if cassette is None or not hasattr(cassette, "get_response"):
        return None
    try:
        recorded = cassette.get_response(model, messages, {})
        if recorded:
            if hasattr(schema_model, "model_validate"):
                return schema_model.model_validate(recorded)
            elif hasattr(schema_model, "model_validate_json"):
                return schema_model.model_validate_json(json.dumps(recorded))
            else:
                return schema_model(**recorded)
    except Exception:
        pass
    return None


def _extract_json_text(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text


def _validate_response(text: str, schema_model: Type, schema_name: str) -> Any:
    json_text = _extract_json_text(text)
    try:
        if hasattr(schema_model, "model_validate_json"):
            return schema_model.model_validate_json(json_text)
        payload = json.loads(json_text)
        return schema_model(**payload)
    except Exception:
        try:
            start = text.find("{")
            if start != -1:
                payload = json.loads(text[start:])
                if hasattr(schema_model, "model_validate"):
                    return schema_model.model_validate(payload)
                return schema_model(**payload)
            raise ValueError("No '{' found in response")
        except Exception as e:
            raise RuntimeError(f"Failed to validate LLM response as {schema_name}: {e}\nResponse:\n{text}")


def _save_response(replay_manager: Any, stage: str, model: str, messages: List[Dict], validated: Any):
    """Persist validated response.

    Supports two shapes:
    - Real ``ReplayManager``: has a ``.cassette`` attribute whose
      ``LLMCassette.save_response(model, messages, params, payload)`` does the
      actual write.
    - Test ``DummyReplay`` (and any legacy adapter): exposes ``.save_response``
      directly with signature ``(stage, model, messages, payload)``.
    """
    try:
        payload = None
        if hasattr(validated, "model_dump"):
            payload = validated.model_dump()
        elif hasattr(validated, "model_dump_json"):
            payload = json.loads(validated.model_dump_json())
        elif hasattr(validated, "dict"):
            payload = validated.dict()
        else:
            import dataclasses
            if dataclasses.is_dataclass(validated):
                payload = dataclasses.asdict(validated)
            else:
                payload = json.loads(json.dumps(validated.__dict__))

        # Prefer the cassette path (real ReplayManager)
        cassette = getattr(replay_manager, "cassette", None)
        if cassette is not None and hasattr(cassette, "save_response"):
            cassette.save_response(model, messages, {}, payload)
            return

        # Fallback: direct save_response on the replay_manager itself (DummyReplay / legacy)
        if replay_manager is not None and hasattr(replay_manager, "save_response"):
            replay_manager.save_response(stage, model, messages, payload)
    except Exception:
        pass


async def call_structured(
    replay_manager: Any,
    model: str,
    messages: List[Dict],
    schema_name: str,
    schema_model: Type,
    stage: str
) -> Any:
    """Main entry used by farley_score_evaluator. Selects provider, applies presets, calls provider, and validates response.

    Returns an instance of `schema_model` (pydantic or fallback) or raises on error.
    """
    preset = os.environ.get("FARLEY_MODEL_PRESET", "instruct")
    params = PRESETS.get(preset, {})
    provider = _select_provider(model)

    replay_hit = _replay_lookup(replay_manager, stage, model, messages, schema_model)
    if replay_hit is not None:
        return replay_hit

    if provider == "nebius":
        raw = await _call_nebius_async(model, messages, params)
    else:
        raw = await _call_litellm_async(model, messages, params)

    validated = _validate_response(str(raw), schema_model, schema_name)
    _save_response(replay_manager, stage, model, messages, validated)
    return validated
