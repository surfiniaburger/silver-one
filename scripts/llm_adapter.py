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


async def _call_litellm_async(model: str, messages: List[Dict], params: Dict[str, Any]) -> str:
    """Call litellm's async completion if available, else try local HTTP endpoints (ollama/litellm)."""
    # Simple retry/backoff wrapper
    def _retry_sync(fn, retries=2, backoff=0.5):
        last = None
        for i in range(retries+1):
            try:
                return fn()
            except Exception as e:
                last = e
                time.sleep(backoff * (2 ** i))
        raise last

    if acompletion is not None:
        # Use the litellm library directly
        resp = await acompletion(model=model.replace(LITELLM_PREFIX, ""), messages=messages, temperature=params.get("temperature", 0.0), max_tokens=params.get("max_tokens", 1024))
        # normalize
        if hasattr(resp, "choices") and resp.choices:
            msg = resp.choices[0].message
            if hasattr(msg, "content"):
                return msg.content
            return str(msg)
        return str(resp)

    # Fallback: try local Ollama API
    def _sync_call():
        try:
            url = os.environ.get("LITELLM_HTTP", "http://localhost:11434/api/generate")
            payload = {"model": model.replace(LITELLM_PREFIX, ""), "input": messages[-1]["content"], "system": messages[0]["content"], "stream": False}
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            d = r.json()
            # Ollama-like shape
            if isinstance(d, dict) and "text" in d:
                return d["text"]
            if isinstance(d, dict) and "choices" in d and d["choices"]:
                return d["choices"][0].get("message", d["choices"][0].get("text", ""))
            return json.dumps(d)
        except Exception as e:
            raise

    return await asyncio.to_thread(lambda: _retry_sync(_sync_call, retries=3, backoff=0.2))


def estimate_tokens(text: str) -> int:
    # Very small heuristic: 1 token ~= 4 chars
    return max(1, math.ceil(len(text) / 4))



async def call_structured(replay_manager: Any, model: str, messages: List[Dict], schema_name: str, schema_model: Type, stage: str, *, include_prompts: bool = False, timeout: int = 60) -> Any:
    """Main entry used by farley_score_evaluator. Selects provider, applies presets, calls provider, and validates response.

    Returns an instance of `schema_model` (pydantic or fallback) or raises on error.
    """
    preset = os.environ.get("FARLEY_MODEL_PRESET", "instruct")
    params = PRESETS.get(preset, {})
    provider = _select_provider(model)

    # Replay lookup (best-effort; ReplayManager API may vary)
    def _replay_lookup():
        if replay_manager is None or not hasattr(replay_manager, "lookup"):
            return None
        try:
            request_id = replay_manager.lookup(stage, model, messages)
            if request_id:
                recorded = replay_manager.get(request_id)
                if recorded:
                    return schema_model.model_validate_json(json.dumps(recorded))
        except Exception:
            return None
        return None

    replay_hit = _replay_lookup()
    if replay_hit is not None:
        return replay_hit

    # Call provider
    if provider == "nebius":
        raw = await _call_nebius_async(model, messages, params)
    else:
        raw = await _call_litellm_async(model, messages, params)

    # Optionally extract JSON from raw text
    text = raw if isinstance(raw, str) else str(raw)
    # Trim surrounding whitespace
    text = text.strip()

    # Try to isolate JSON payload if the model wrapped it in markdown or commentary
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_text = text[start:end+1]
    else:
        json_text = text

    # Try to validate using schema_model.model_validate_json if available
    try:
        if hasattr(schema_model, "model_validate_json"):
            validated = schema_model.model_validate_json(json_text)
        else:
            # fallback: parse json then instantiate
            payload = json.loads(json_text)
            validated = schema_model(**payload)
    except Exception:
        # Last resort: try to extract JSON block from text
        try:
            start = text.find("{")
            if start != -1:
                payload = json.loads(text[start:])
                validated = schema_model.model_validate(payload) if hasattr(schema_model, "model_validate") else schema_model(**payload)
            else:
                raise
        except Exception as e:
            raise RuntimeError(f"Failed to validate LLM response as {schema_name}: {e}\nResponse:\n{text}")

    # Save to replay if possible
    def _save_response():
        if replay_manager is None or not hasattr(replay_manager, "save_response"):
            return
        try:
            payload = None
            if hasattr(validated, "model_dump_json"):
                payload = json.loads(validated.model_dump_json())
            else:
                payload = json.loads(json.dumps(validated.__dict__))
            replay_manager.save_response(stage, model, messages, payload)
        except Exception:
            return

    _save_response()
    return validated
