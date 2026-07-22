import json
import time
import uuid
import contextlib
import contextvars
from pathlib import Path
from typing import Dict, Any, Optional, Generator

from agentbeats.clock import RunClock

_CURRENT_TRACE_ID_VAR: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_CURRENT_TRACE_ID_VAR", default=None)


def get_current_trace_id() -> str:
    trace_id = _CURRENT_TRACE_ID_VAR.get()
    if trace_id is None:
        trace_id = uuid.uuid4().hex
        _CURRENT_TRACE_ID_VAR.set(trace_id)
    return trace_id


def set_current_trace_id(trace_id: str) -> None:
    _CURRENT_TRACE_ID_VAR.set(trace_id)


def reset_current_trace_id() -> None:
    _CURRENT_TRACE_ID_VAR.set(None)


class TraceSpan:
    def __init__(
        self,
        name: str,
        stage: str = "unknown",
        attributes: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        spans_dir: Optional[Path] = None,
    ):
        self.name = name
        self.stage = stage
        self.attributes = attributes or {}
        self.trace_id = trace_id or get_current_trace_id()
        self.span_id = uuid.uuid4().hex[:16]
        self.start_time_iso = RunClock.from_env().now_iso()
        self.start_perf = time.perf_counter()
        self.end_time_iso: Optional[str] = None
        self.duration_ms: float = 0.0
        self.status: str = "OK"
        self.error_message: Optional[str] = None
        self.spans_dir = spans_dir

    def finish(self, status: str = "OK", error: Optional[Exception] = None) -> Dict[str, Any]:
        self.end_time_iso = RunClock.from_env().now_iso()
        self.duration_ms = round((time.perf_counter() - self.start_perf) * 1000.0, 3)
        self.status = status
        if error:
            self.error_message = str(error)
            self.attributes["error.type"] = error.__class__.__name__

        payload = self.to_dict()
        self.export_span(payload)
        return payload

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "stage": self.stage,
            "start_time": self.start_time_iso,
            "end_time": self.end_time_iso,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
        }
        if self.error_message:
            data["error_message"] = self.error_message
        return data

    def export_span(self, payload: Dict[str, Any]) -> None:
        try:
            if self.spans_dir:
                metrics_dir = self.spans_dir
            else:
                project_root = Path(__file__).resolve().parent.parent.parent
                metrics_dir = project_root / "artifacts" / "metrics"

            metrics_dir.mkdir(parents=True, exist_ok=True)
            spans_path = metrics_dir / "spans.jsonl"
            with spans_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            pass


@contextlib.contextmanager
def trace_span(
    name: str,
    stage: str = "unknown",
    attributes: Optional[Dict[str, Any]] = None,
    spans_dir: Optional[Path] = None,
) -> Generator[TraceSpan, None, None]:
    span = TraceSpan(name=name, stage=stage, attributes=attributes, spans_dir=spans_dir)
    try:
        yield span
        span.finish(status="OK")
    except Exception as exc:
        span.finish(status="ERROR", error=exc)
        raise
