import os
import json
import asyncio
import time
import math
from typing import Any, Dict, List, Type, Tuple

try:
    # Optional dependency: litellm async client
    from litellm import acompletion
except Exception:
    acompletion = None

import requests

class OfflineReplayError(RuntimeError):
    """Raised when there is no matching recorded response in the replay cassette."""
    pass


class StructuredOutputError(RuntimeError):
    """Raised when an LLM response cannot be parsed or validated after retries."""

    def __init__(self, schema_name: str, message: str, raw_response: str, diagnostics: Dict[str, Any]):
        super().__init__(message)
        self.schema_name = schema_name
        self.raw_response = raw_response
        self.diagnostics = diagnostics


LITELLM_PREFIX = "litellm/"

PRESETS = {
    "instruct": {"temperature": 0.0, "top_p": 0.8, "max_tokens": 1024},
    # Qwen3 (and other extended-thinking models) require temperature >= 0.6.
    # At 0.0 the greedy sampler stalls after </think> and emits no final answer.
    "qwen3-thinking": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 2048},
    "qwen3.5:thinking-general": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    "qwen3.5:thinking-precise-coding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
}

# Model substrings that indicate an extended-thinking model needing temp > 0.
_THINKING_MODEL_HINTS = ("qwen3",)


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
    assert last is not None  # loop always runs ≥1 time, so last is always set
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


def _local_ollama_sync_call(model: str, messages: List[Dict], schema_model: Type = None) -> str:
    url = os.environ.get("LITELLM_HTTP", "http://localhost:11434/api/generate")
    clean_model = model.replace(LITELLM_PREFIX, "")
    if "/" in clean_model:
        clean_model = clean_model.split("/", 1)[1]
    payload = {
        "model": clean_model,
        # Ollama /api/generate expects 'prompt', not 'input'
        "prompt": messages[-1].get("content", "") if messages else "",
        # Only include system if there is a distinct first message
        "system": messages[0].get("content", "") if len(messages) > 1 else "",
        "stream": False
    }
    if schema_model is not None:
        if hasattr(schema_model, "model_json_schema"):
            payload["format"] = schema_model.model_json_schema()
        elif hasattr(schema_model, "schema"):
            payload["format"] = schema_model.schema()
        else:
            payload["format"] = "json"
    if any(hint in model.lower() for hint in _THINKING_MODEL_HINTS):
        payload["think"] = False
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return _parse_ollama_response(r.json())


def _extract_choice_dict_content(choice: Any) -> str:
    if not isinstance(choice, dict):
        return ""
    msg = choice.get("message", {})
    if isinstance(msg, dict):
        return msg.get("content") or msg.get("reasoning_content") or msg.get("thinking") or ""
    return choice.get("text", "")


def _extract_litellm_dict_content(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        return _extract_choice_dict_content(choices[0])
    return str(resp)


def _extract_litellm_object_content(resp: Any) -> str:
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    if not content:
        content = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "thinking", None)
            or ""
        )
    return content


def _extract_litellm_content(resp: Any) -> str:
    """Extract text content from a LiteLLM completion response object."""
    if isinstance(resp, dict):
        return _extract_litellm_dict_content(resp)
    if hasattr(resp, "choices") and resp.choices:
        return _extract_litellm_object_content(resp)
    return str(resp)


def _build_response_format(schema_name: str, schema_model: Type) -> Any:
    if hasattr(schema_model, "model_json_schema"):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema_model.model_json_schema(),
                "strict": False,
            },
        }
    if hasattr(schema_model, "schema"):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema_model.schema(),
                "strict": False,
            },
        }
    return {"type": "json_object"}


def _likely_response_format_unsupported(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "response_format",
            "json_schema",
            "unsupported",
            "not supported",
            "badrequest",
            "invalid request",
        )
    )


def _build_litellm_kwargs(model: str, messages: List[Dict], params: Dict[str, Any], schema_model: Type) -> Dict[str, Any]:
    """Build the kwargs dict for a LiteLLM acompletion call."""
    clean_model = model.replace(LITELLM_PREFIX, "")
    kwargs: Dict[str, Any] = {
        "model": clean_model,
        "messages": messages,
        "temperature": params.get("temperature", 0.0),
        "max_tokens": params.get("max_tokens", 1024),
    }
    if schema_model is not None:
        if hasattr(schema_model, "model_json_schema") or hasattr(schema_model, "schema"):
            kwargs["response_format"] = schema_model
        else:
            kwargs["response_format"] = {"type": "json_object"}
    if any(hint in model.lower() for hint in _THINKING_MODEL_HINTS):
        kwargs["extra_body"] = {"think": False}
        kwargs["drop_params"] = True
    return kwargs


async def _call_litellm_async(model: str, messages: List[Dict], params: Dict[str, Any], schema_model: Type = None) -> str:
    """Call litellm's async completion if available, else try local HTTP endpoints (ollama/litellm)."""
    if acompletion is not None:
        kwargs = _build_litellm_kwargs(model, messages, params, schema_model)
        resp = await acompletion(**kwargs)
        return _extract_litellm_content(resp)

    fn = lambda: _local_ollama_sync_call(model, messages, schema_model)
    return await asyncio.to_thread(lambda: _retry_sync(fn, retries=3, backoff=0.2))


async def _call_replay_manager_async(
    replay_manager: Any,
    model: str,
    messages: List[Dict],
    params: Dict[str, Any],
    schema_name: str,
    schema_model: Type,
    stage: str,
) -> str:
    clean_model = model.replace(LITELLM_PREFIX, "")
    kwargs: Dict[str, Any] = {
        "temperature": params.get("temperature", 0.0),
        "max_tokens": params.get("max_tokens", 1024),
        "response_format": _build_response_format(schema_name, schema_model),
        "stage": stage,
    }
    if any(hint in model.lower() for hint in _THINKING_MODEL_HINTS):
        kwargs["extra_body"] = {"think": False}
        kwargs["drop_params"] = True

    try:
        response = await replay_manager.acompletion(
            model=clean_model,
            messages=messages,
            **kwargs,
        )
    except Exception as exc:
        if not _likely_response_format_unsupported(exc):
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("response_format", None)
        fallback_messages = [
            {
                "role": "system",
                "content": (
                    "Output MUST be a single valid JSON object and NOTHING else. "
                    "Do not wrap in markdown. Do not include any non-JSON text."
                ),
            },
            *messages,
        ]
        response = await replay_manager.acompletion(
            model=clean_model,
            messages=fallback_messages,
            **fallback_kwargs,
        )
    return _extract_litellm_content(response)


def estimate_tokens(text: str) -> int:
    # Very small heuristic: 1 token ~= 4 chars
    return max(1, math.ceil(len(text) / 4))


def _replay_lookup(replay_manager: Any, model: str, messages: List[Dict], schema_model: Type, params: Dict[str, Any] = None) -> Any:
    # ReplayManager exposes its cassette via .cassette (an LLMCassette instance)
    cassette = getattr(replay_manager, "cassette", None)
    if cassette is None or not hasattr(cassette, "get_response"):
        return None
    try:
        recorded = cassette.get_response(model, messages, params or {})
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


def _diagnostics_template() -> Dict[str, Any]:
    return {
        "invalid_json_detected": False,
        "repair_attempts": 0,
        "repair_succeeded": False,
        "validation_retries": 0,
        "final_failure": False,
        "failure_reason": None,
    }


def _merge_structured_diagnostics(cumulative: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(cumulative)
    merged["invalid_json_detected"] = bool(
        cumulative.get("invalid_json_detected") or current.get("invalid_json_detected")
    )
    merged["repair_attempts"] = int(cumulative.get("repair_attempts", 0) or 0) + int(
        current.get("repair_attempts", 0) or 0
    )
    merged["repair_succeeded"] = bool(
        cumulative.get("repair_succeeded") or current.get("repair_succeeded")
    )
    merged["validation_retries"] = int(current.get("validation_retries", 0) or 0)
    merged["final_failure"] = bool(current.get("final_failure", False))
    merged["failure_reason"] = current.get("failure_reason") or cumulative.get("failure_reason")
    return merged


def _validate_payload(payload: Any, schema_model: Type) -> Any:
    if hasattr(schema_model, "model_validate"):
        return schema_model.model_validate(payload)
    if hasattr(schema_model, "model_validate_json"):
        return schema_model.model_validate_json(json.dumps(payload))
    return schema_model(**payload)


def _parse_json_payload(json_text: str) -> Any:
    return json.loads(json_text)


def _has_unclosed_string(json_text: str) -> bool:
    escaped = False
    in_string = False
    for char in json_text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
    return in_string


def _next_string_scan_state(char: str, escaped: bool, in_string: bool) -> Tuple[bool, bool, bool]:
    if escaped:
        return False, in_string, True
    if char == "\\":
        return True, in_string, True
    if char == '"':
        return False, not in_string, True
    return False, in_string, in_string


def _close_open_containers(json_text: str) -> str:
    stack: List[str] = []
    escaped = False
    in_string = False
    for char in json_text:
        escaped, in_string, handled = _next_string_scan_state(char, escaped, in_string)
        if handled:
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack and char == stack[-1]:
            stack.pop()
    return json_text + "".join(reversed(stack))


def _trailing_backslash_count(text: str) -> int:
    return len(text) - len(text.rstrip("\\"))


def _strip_trailing_commas_outside_strings(json_text: str) -> str:
    chars: List[str] = []
    escaped = False
    in_string = False
    length = len(json_text)

    for idx, char in enumerate(json_text):
        escaped, in_string, handled = _next_string_scan_state(char, escaped, in_string)
        if not in_string and not handled and char == ",":
            next_idx = idx + 1
            while next_idx < length and json_text[next_idx].isspace():
                next_idx += 1
            if next_idx < length and json_text[next_idx] in "}]":
                continue
        chars.append(char)

    return "".join(chars)


def _repair_json_text(json_text: str) -> str:
    repaired = json_text.strip()
    repaired = _strip_trailing_commas_outside_strings(repaired)
    if _has_unclosed_string(repaired):
        repaired = repaired.rstrip()
        if _trailing_backslash_count(repaired) % 2 == 1:
            repaired += '\\"'
        else:
            repaired += '"'
    return _close_open_containers(repaired)


def _validate_response_with_diagnostics(text: str, schema_model: Type, schema_name: str) -> Tuple[Any, Dict[str, Any]]:
    diagnostics = _diagnostics_template()
    json_text = _extract_json_text(text)
    try:
        payload = _parse_json_payload(json_text)
    except json.JSONDecodeError as json_exc:
        diagnostics["invalid_json_detected"] = True
        diagnostics["repair_attempts"] += 1
        repaired_json_text = _repair_json_text(json_text)
        try:
            payload = _parse_json_payload(repaired_json_text)
            diagnostics["repair_succeeded"] = True
        except Exception as repair_exc:
            diagnostics["final_failure"] = True
            diagnostics["failure_reason"] = f"invalid_json: {repair_exc}"
            message = f"Failed to parse LLM response as JSON before {schema_name} validation: {json_exc}"
            raise StructuredOutputError(schema_name, message, text, diagnostics) from repair_exc
    except Exception as exc:
        diagnostics["final_failure"] = True
        diagnostics["failure_reason"] = f"invalid_json: {exc}"
        message = f"Failed to parse LLM response as JSON before {schema_name} validation: {exc}"
        raise StructuredOutputError(schema_name, message, text, diagnostics) from exc

    try:
        return _validate_payload(payload, schema_model), diagnostics
    except Exception as exc:
        diagnostics["final_failure"] = True
        diagnostics["failure_reason"] = f"schema_validation: {exc}"
        message = f"Failed to validate LLM response as {schema_name}: {exc}"
        raise StructuredOutputError(schema_name, message, text, diagnostics) from exc


def _validate_response(text: str, schema_model: Type, schema_name: str) -> Any:
    validated, _ = _validate_response_with_diagnostics(text, schema_model, schema_name)
    return validated


def _save_response(replay_manager: Any, stage: str, model: str, messages: List[Dict], validated: Any, params: Dict[str, Any] = None):
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
            cassette.save_response(model, messages, params or {}, payload)
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
    validated, _ = await call_structured_with_raw(
        replay_manager, model, messages, schema_name, schema_model, stage
    )
    return validated


def _replay_lookup_raw(
    replay_manager: Any,
    model: str,
    messages: List[Dict],
    params: Dict[str, Any],
    schema_model: Type,
    schema_name: str
) -> Any:
    cassette = getattr(replay_manager, "cassette", None)
    if cassette is not None and hasattr(cassette, "get_response"):
        recorded = cassette.get_response(model, messages, params or {})
        if recorded:
            raw_str = json.dumps(recorded)
            validated = _validate_response(raw_str, schema_model, schema_name)
            return validated, raw_str
    raise OfflineReplayError("Offline Replay Error: No matching recorded response found.")


async def _call_litellm_or_nebius(
    model: str,
    messages: List[Dict],
    provider: str,
    params: Dict[str, Any],
    schema_model: Type
) -> str:
    if provider == "nebius":
        return await _call_nebius_async(model, messages, params)
    return await _call_litellm_async(model, messages, params, schema_model=schema_model)


async def call_structured_with_raw(
    replay_manager: Any,
    model: str,
    messages: List[Dict],
    schema_name: str,
    schema_model: Type,
    stage: str
) -> Any:
    """Selects provider, calls provider, validates response, and returns both parsed model and raw output string."""
    validated, raw, _ = await call_structured_with_raw_and_diagnostics(
        replay_manager, model, messages, schema_name, schema_model, stage
    )
    return validated, raw


def _structured_retry_count() -> int:
    raw_value = os.environ.get("LLM_STRUCTURED_MAX_RETRIES", "2")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 2


def _validate_raw_attempt(
    raw: Any,
    schema_model: Type,
    schema_name: str,
    attempt: int,
    cumulative_diagnostics: Dict[str, Any],
) -> Tuple[Any, str, Dict[str, Any]]:
    raw_str = str(raw)
    validated, diagnostics = _validate_response_with_diagnostics(raw_str, schema_model, schema_name)
    diagnostics["validation_retries"] = attempt
    diagnostics = _merge_structured_diagnostics(cumulative_diagnostics, diagnostics)
    return validated, raw_str, diagnostics


def _record_structured_attempt_failure(
    exc: StructuredOutputError,
    attempt: int,
    cumulative_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    exc.diagnostics["validation_retries"] = attempt
    return _merge_structured_diagnostics(cumulative_diagnostics, exc.diagnostics)


def _replay_lookup_with_diagnostics(
    replay_manager: Any,
    model: str,
    messages: List[Dict],
    params: Dict[str, Any],
    schema_model: Type,
    schema_name: str,
) -> Tuple[Any, str, Dict[str, Any]]:
    validated, raw_str = _replay_lookup_raw(replay_manager, model, messages, params, schema_model, schema_name)
    return validated, raw_str, _diagnostics_template()


async def _call_replay_manager_with_retries(
    replay_manager: Any,
    model: str,
    messages: List[Dict],
    params: Dict[str, Any],
    schema_name: str,
    schema_model: Type,
    stage: str,
    max_retries: int,
) -> Tuple[Any, str, Dict[str, Any]]:
    cumulative_diagnostics = _diagnostics_template()
    for attempt in range(max_retries + 1):
        try:
            raw = await _call_replay_manager_async(
                replay_manager,
                model,
                messages,
                params,
                schema_name,
                schema_model,
                stage,
            )
            return _validate_raw_attempt(raw, schema_model, schema_name, attempt, cumulative_diagnostics)
        except StructuredOutputError as exc:
            cumulative_diagnostics = _record_structured_attempt_failure(
                exc,
                attempt,
                cumulative_diagnostics,
            )
            if attempt < max_retries:
                continue
            exc.diagnostics = cumulative_diagnostics
            raise exc
        except RuntimeError as exc:
            if "Offline Replay Error" not in str(exc):
                raise
            return _replay_lookup_with_diagnostics(
                replay_manager,
                model,
                messages,
                params,
                schema_model,
                schema_name,
            )

    raise AssertionError("structured replay retry loop exited unexpectedly")


async def _call_provider_with_retries(
    replay_manager: Any,
    model: str,
    messages: List[Dict],
    provider: str,
    params: Dict[str, Any],
    schema_name: str,
    schema_model: Type,
    stage: str,
    max_retries: int,
) -> Tuple[Any, str, Dict[str, Any]]:
    cumulative_diagnostics = _diagnostics_template()
    for attempt in range(max_retries + 1):
        raw = await _call_litellm_or_nebius(model, messages, provider, params, schema_model)
        try:
            validated, raw_str, diagnostics = _validate_raw_attempt(
                raw,
                schema_model,
                schema_name,
                attempt,
                cumulative_diagnostics,
            )
            _save_response(replay_manager, stage, model, messages, validated, params)
            return validated, raw_str, diagnostics
        except StructuredOutputError as exc:
            cumulative_diagnostics = _record_structured_attempt_failure(
                exc,
                attempt,
                cumulative_diagnostics,
            )
            if attempt < max_retries:
                continue
            exc.diagnostics = cumulative_diagnostics
            raise exc

    raise AssertionError("structured provider retry loop exited unexpectedly")


async def call_structured_with_raw_and_diagnostics(
    replay_manager: Any,
    model: str,
    messages: List[Dict],
    schema_name: str,
    schema_model: Type,
    stage: str
) -> Tuple[Any, str, Dict[str, Any]]:
    """Selects provider, validates structured output, and returns validation diagnostics."""
    preset = os.environ.get("FARLEY_MODEL_PRESET", "instruct")

    params = dict(PRESETS.get(preset, PRESETS["instruct"]))  # copy so mutations are local
    provider = _select_provider(model)
    max_retries = _structured_retry_count()

    if replay_manager is not None and hasattr(replay_manager, "acompletion"):
        return await _call_replay_manager_with_retries(
            replay_manager,
            model,
            messages,
            params,
            schema_name,
            schema_model,
            stage,
            max_retries,
        )

    try:
        return _replay_lookup_with_diagnostics(
            replay_manager,
            model,
            messages,
            params,
            schema_model,
            schema_name,
        )
    except OfflineReplayError:
        pass

    return await _call_provider_with_retries(
        replay_manager,
        model,
        messages,
        provider,
        params,
        schema_name,
        schema_model,
        stage,
        max_retries,
    )
