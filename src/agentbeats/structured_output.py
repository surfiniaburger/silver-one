import json
import hashlib
import logging
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger("agentbeats.structured_output")

T = TypeVar("T", bound=BaseModel)


def strip_markdown_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```"):
        if lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return s


def _update_string_state(ch: str, in_string: bool, escaped: bool) -> tuple[bool, bool, bool]:
    if escaped:
        return in_string, False, True
    if ch == "\\":
        return in_string, in_string, True
    if ch == '"':
        return not in_string, False, True
    return in_string, False, in_string


def _scan_json_object_bounds(text: str) -> Optional[tuple[int, int]]:
    first_brace = text.find("{")
    if first_brace == -1:
        return None

    in_string = False
    escaped = False
    depth = 0

    for idx, ch in enumerate(text[first_brace:], start=first_brace):
        in_string, escaped, skip = _update_string_state(ch, in_string, escaped)
        if skip:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (first_brace, idx + 1)

    return None


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Return the first complete top-level JSON object found in `text`.
    """
    bounds = _scan_json_object_bounds(text)
    return text[bounds[0] : bounds[1]] if bounds is not None else None


def escape_invalid_backslashes(text: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _json_candidates(raw_text: str) -> list[str]:
    s0 = raw_text.strip()
    s1 = strip_markdown_fence(s0)
    extracted = extract_first_json_object(s1)
    s2 = extracted if extracted is not None else s1
    s3 = escape_invalid_backslashes(s2)
    out: list[str] = []
    for candidate in (s0, s1, s2, s3):
        if candidate and candidate not in out:
            out.append(candidate)
        try:
            parsed = json.loads(candidate, strict=False)
            reencoded = json.dumps(parsed)
            if reencoded not in out:
                out.append(reencoded)
        except Exception:
            pass
    return out


def _schema_dict(schema_name: str, model: Type[BaseModel], strict: bool = True) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": model.model_json_schema(),
            "strict": strict,
        },
    }


def _schema_fingerprint(schema: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()


def _stringify_exc(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:
        return repr(exc)


def _likely_schema_unsupported(exc: BaseException) -> bool:
    """
    Best-effort heuristic: some providers reject `response_format={"type":"json_schema",...}`.
    When that happens we fall back to "JSON-only" prompting (still validated by Pydantic)
    and we record the event for auditability.
    """
    msg = _stringify_exc(exc).lower()
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


def _safe_record_event(
    replay_manager: Any,
    *,
    model: str,
    name: str,
    params: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    try:
        cassette = getattr(replay_manager, "cassette", None)
        if not cassette:
            return
        cassette.save_response(
            model=f"event/{name}",
            messages=[{"role": "user", "content": json.dumps({"name": name})}],
            params={"target_model": model, **params},
            response=payload,
        )
    except Exception:
        logger.exception("Failed to record structured-output event %s.", name)


def _try_parse_schema_candidates(raw_text: str, schema_model: Type[T]) -> Optional[T]:
    for candidate in _json_candidates(raw_text):
        try:
            return schema_model.model_validate_json(candidate)
        except Exception:
            pass
    return None


async def _request_completion_with_schema_fallback(
    replay_manager: Any,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    schema_name: str,
    schema_fp: str,
    strict: bool,
    stage: str,
    **kwargs,
) -> tuple[str, bool]:
    try:
        response = await replay_manager.acompletion(
            model=model,
            messages=messages,
            response_format=schema,
            stage=stage,
            **kwargs,
        )
        return response.choices[0].message.content.strip(), True
    except Exception as e:
        if not _likely_schema_unsupported(e):
            raise

        _safe_record_event(
            replay_manager,
            model=model,
            name="structured_output_response_format_rejected",
            params={"schema_fp": schema_fp, "schema_name": schema_name, "strict": strict},
            payload={"error": _stringify_exc(e)},
        )

        retry_messages = [
            {
                "role": "system",
                "content": (
                    "Output MUST be a single valid JSON object and NOTHING else. "
                    "Do not wrap in markdown. Do not include any non-JSON text."
                ),
            },
            *messages,
        ]
        response = await replay_manager.acompletion(model=model, messages=retry_messages, stage=stage, **kwargs)
        return response.choices[0].message.content.strip(), False


async def _execute_repair_completion(
    replay_manager: Any,
    model: str,
    repair_model: Optional[str],
    raw_text: str,
    schema: dict[str, Any],
    schema_model: Type[T],
    structured_supported: bool,
    strict: bool,
    stage: str,
    **kwargs,
) -> T:
    repair_messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON repair tool. Convert the input text into a single valid JSON object "
                "that conforms EXACTLY to the provided JSON schema. Output ONLY JSON. No markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"JSON Schema (strict={strict}):\n{json.dumps(schema_model.model_json_schema(), indent=2)}\n\n"
                f"Text to convert:\n{raw_text}"
            ),
        },
    ]

    repair_kwargs: dict[str, Any] = {
        "model": repair_model or model,
        "messages": repair_messages,
        "stage": stage,
        **kwargs,
    }
    if structured_supported:
        repair_kwargs["response_format"] = schema

    response2 = await replay_manager.acompletion(**repair_kwargs)
    raw_text2 = response2.choices[0].message.content.strip()
    parsed = _try_parse_schema_candidates(raw_text2, schema_model)
    if parsed is not None:
        return parsed
    return schema_model.model_validate_json(raw_text2)


async def call_structured(
    *,
    replay_manager: Any,
    model: str,
    messages: list[dict[str, str]],
    schema_name: str,
    schema_model: Type[T],
    strict: bool = True,
    repair_on_fail: bool = True,
    repair_model: Optional[str] = None,
    stage: str = "structured_output",
    **kwargs,
) -> T:
    """
    Structured output helper with *recorded* fallback.
    """
    schema = _schema_dict(schema_name, schema_model, strict=strict)
    schema_fp = _schema_fingerprint(schema)

    raw_text, structured_supported = await _request_completion_with_schema_fallback(
        replay_manager, model, messages, schema, schema_name, schema_fp, strict, stage, **kwargs
    )

    parsed = _try_parse_schema_candidates(raw_text, schema_model)
    if parsed is not None:
        return parsed

    try:
        return schema_model.model_validate_json(raw_text)
    except Exception as e:
        _safe_record_event(
            replay_manager,
            model=model,
            name="structured_output_validation_error",
            params={"schema_fp": schema_fp, "schema_name": schema_name, "strict": strict},
            payload={
                "error": _stringify_exc(e),
                "schema_fp": schema_fp,
                "schema_name": schema_name,
                "raw_text": raw_text,
            },
        )

        if not repair_on_fail:
            raise

        return await _execute_repair_completion(
            replay_manager,
            model,
            repair_model,
            raw_text,
            schema,
            schema_model,
            structured_supported,
            strict,
            stage,
            **kwargs,
        )
