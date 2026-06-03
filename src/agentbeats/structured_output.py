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


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Return the first complete top-level JSON object found in `text`.
    """
    in_string = False
    escaped = False
    depth = 0
    start = -1
    for idx, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
            if depth == 0 and start >= 0:
                return text[start : idx + 1]
    return None


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

    Behavior:
    - Primary call: request strict JSON via `response_format` and validate with Pydantic.
    - If the provider rejects `response_format`:
      - record an event
      - retry without `response_format` but with JSON-only instructions, then validate/repair
    - On validation failure:
      - emit a deterministic failure event into the cassette
      - optionally run a second "repair" call that converts raw text to valid JSON for the same schema

    Determinism rules:
    - In replay mode, any cache miss is a hard failure (handled by ReplayManager).
    - Both primary and repair calls go through ReplayManager, so they are recorded/replayed.
    - Failure events are written to the cassette as data (no side effects beyond recording).
    """
    schema = _schema_dict(schema_name, schema_model, strict=strict)
    schema_fp = _schema_fingerprint(schema)

    structured_supported = True
    raw_text: str
    try:
        response = await replay_manager.acompletion(
            model=model,
            messages=messages,
            response_format=schema,
            stage=stage,
            **kwargs,
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        if not _likely_schema_unsupported(e):
            raise

        structured_supported = False
        _safe_record_event(
            replay_manager,
            model=model,
            name="structured_output_response_format_rejected",
            params={"schema_fp": schema_fp, "schema_name": schema_name, "strict": strict},
            payload={"error": _stringify_exc(e)},
        )

        # Retry without response_format, but demand JSON only.
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
        raw_text = response.choices[0].message.content.strip()

    for candidate in _json_candidates(raw_text):
        try:
            return schema_model.model_validate_json(candidate)
        except Exception:
            pass

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

        # Repair: ask for JSON matching the same schema, with no extra text.
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
        for candidate in _json_candidates(raw_text2):
            try:
                return schema_model.model_validate_json(candidate)
            except Exception:
                pass
        return schema_model.model_validate_json(raw_text2)
