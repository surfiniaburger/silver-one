import hashlib
import json
import fcntl
import tempfile
import os
import time
import logging

# ``litellm`` is required for the async completion wrapper and the token / cost helpers.
# Importing it once at module load time avoids repeated import overhead and makes the
# dependency explicit. If the package is missing we set a placeholder so the module can be
# imported (useful for tests that mock the provider).
try:
    import litellm  # type: ignore
except Exception:  # pragma: no cover – the library may be absent in some environments
    litellm = None  # type: ignore

logger = logging.getLogger(__name__)
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel
from agentbeats.clock import RunClock
from agentbeats.tracing import trace_span

class RunRecord(BaseModel):
    run_id: str
    rng_seed: int
    models: Dict[str, str]
    generation_config: Dict[str, Any] = {}
    git_commit: Optional[str] = None
    created_at: str


GEMMA4_OLLAMA_SAMPLING_CONFIG = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}

GENERATION_PARAM_KEYS = {"temperature", "top_p", "top_k", "max_tokens"}

class LLMCassette:
    """Records and replays LLM responses based on full prompt/param hashes."""
    def __init__(self, path: str, mode: str = "record"):
        self.path = path
        self.mode = mode
        self.data: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.data = json.load(f)
            except Exception as e:
                # Use logging instead of printing directly. This respects any
                # logging configuration the host may have set and makes the
                # warning easily testable.
                logger.warning("Failed to load cassette %s: %s", path, e)

    def _hash(self, model: str, messages: list, params: Dict[str, Any]) -> str:
        # Include all generation-affecting params in the hash
        payload = {
            "model": model,
            "messages": messages,
            "params": {k: v for k, v in sorted(params.items()) if k not in ["api_key", "base_url", "timeout", "request_timeout"]}
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _lock(self, file):
        fcntl.flock(file, fcntl.LOCK_EX)

    def _unlock(self, file):
        fcntl.flock(file, fcntl.LOCK_UN)

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self._lock(f)
                    self.data = json.load(f)
                    self._unlock(f)
            except Exception as e:
                logger.warning("Failed to load cassette %s: %s", self.path, e)

    def get_response(self, model: str, messages: list, params: Dict[str, Any]) -> Optional[Any]:
        # Always reload before reading in case another process updated it
        self.load()
        h = self._hash(model, messages, params)
        return self.data.get(h)

    def save_response(self, model: str, messages: list, params: Dict[str, Any], response: Any):
        if self.mode != "record":
            return
        
        h = self._hash(model, messages, params)
        
        # Atomically update and save
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        # Use a lock file to coordinate between processes
        lock_path = self.path + ".lock"
        with open(lock_path, "w") as lock_file:
            self._lock(lock_file)
            
            # Reload existing data to avoid clobbering concurrent writes
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    try:
                        existing = json.load(f)
                        if isinstance(existing, dict):
                            self.data.update(existing)
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning("Failed to reload existing cassette %s: %s", self.path, e)
            
            self.data[h] = response
            
            # Atomic write via temp file
            temp_dir = os.path.dirname(self.path)
            tempname = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=temp_dir, delete=False, suffix=".tmp") as tf:
                    json.dump(self.data, tf, indent=2)
                    tempname = tf.name
                os.replace(tempname, self.path)
                tempname = None
            finally:
                if tempname and os.path.exists(tempname):
                    try:
                        os.remove(tempname)
                    except Exception:
                        pass
                self._unlock(lock_file)

class ReplayManager:
    """Global manager for determinism state (A1, A5)."""
    def __init__(self, run_record: RunRecord, cassette: LLMCassette):
        self.run_record = run_record
        self.cassette = cassette
        self.usage_events: List[Dict[str, Any]] = []
        import random
        self.rng = random.Random(run_record.rng_seed)

    def reset_usage_events(self) -> None:
        self.usage_events = []

    @staticmethod
    def _compute_duration_stats(events: List[Dict[str, Any]]) -> Dict[str, float]:
        durations = [float(ev.get("duration_ms") or 0.0) for ev in events]
        if not durations:
            return {
                "total_duration_ms": 0.0,
                "avg_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "min_duration_ms": 0.0,
                "p95_duration_ms": 0.0,
            }
        total_dur = sum(durations)
        sorted_dur = sorted(durations)
        p95_idx = min(int(0.95 * len(sorted_dur)), len(sorted_dur) - 1)
        return {
            "total_duration_ms": round(total_dur, 3),
            "avg_duration_ms": round(total_dur / len(durations), 3),
            "max_duration_ms": round(max(durations), 3),
            "min_duration_ms": round(min(durations), 3),
            "p95_duration_ms": round(sorted_dur[p95_idx], 3),
        }

    @staticmethod
    def _update_group_metric(group_dict: Dict[str, Dict[str, Any]], key: str, ev: Dict[str, Any]) -> None:
        entry = group_dict.setdefault(
            key,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "total_duration_ms": 0.0,
                "max_duration_ms": 0.0,
            },
        )
        dur = float(ev.get("duration_ms") or 0.0)
        entry["calls"] += 1
        entry["prompt_tokens"] += int(ev.get("prompt_tokens") or 0)
        entry["completion_tokens"] += int(ev.get("completion_tokens") or 0)
        entry["total_tokens"] += int(ev.get("total_tokens") or 0)
        entry["cost_usd"] += float(ev.get("cost_usd") or 0.0)
        entry["total_duration_ms"] = round(entry["total_duration_ms"] + dur, 3)
        entry["max_duration_ms"] = max(entry["max_duration_ms"], dur)

    @staticmethod
    def _finalize_group_stats(group_dict: Dict[str, Dict[str, Any]]) -> None:
        for entry in group_dict.values():
            calls = entry.get("calls", 0)
            entry["avg_duration_ms"] = round(entry["total_duration_ms"] / calls, 3) if calls else 0.0

    def get_usage_summary(self) -> Dict[str, Any]:
        by_stage: Dict[str, Dict[str, Any]] = {}
        by_model: Dict[str, Dict[str, Any]] = {}
        totals = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "missing_usage_calls": 0,
            **self._compute_duration_stats(self.usage_events),
        }
        for ev in self.usage_events:
            totals["calls"] += 1
            totals["prompt_tokens"] += int(ev.get("prompt_tokens") or 0)
            totals["completion_tokens"] += int(ev.get("completion_tokens") or 0)
            totals["total_tokens"] += int(ev.get("total_tokens") or 0)
            totals["cost_usd"] += float(ev.get("cost_usd") or 0.0)
            if not ev.get("has_usage"):
                totals["missing_usage_calls"] += 1

            stage = str(ev.get("stage", "unknown"))
            model = str(ev.get("model", "unknown"))
            self._update_group_metric(by_stage, stage, ev)
            self._update_group_metric(by_model, model, ev)

        self._finalize_group_stats(by_stage)
        self._finalize_group_stats(by_model)

        return {
            "totals": totals,
            "by_stage": by_stage,
            "by_model": by_model,
            "events": self.usage_events,
            "generation_config": dict(self.run_record.generation_config),
            "models": dict(self.run_record.models),
            "notes": {
                "tracked_paths": "calls through ReplayManager.acompletion",
                "untracked_paths": ["direct A2A tool calls (debater/verifier transport) unless wrapped"],
            },
        }

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value or 0)
        except Exception:
            return 0

    def _to_response_dict(self, response_obj: Any) -> Dict[str, Any]:
        if isinstance(response_obj, dict):
            return response_obj
        # Use json() + loads() to bypass Pydantic serialization warnings for non-standard fields
        if hasattr(response_obj, "json"):
            try:
                return json.loads(response_obj.json())
            except Exception:
                pass
        if hasattr(response_obj, "model_dump"):
            try:
                return response_obj.model_dump()
            except Exception:
                pass
        return {}

    def _normalize_usage_dict(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        prompt_tokens = self._safe_int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_eval_count")
        )
        completion_tokens = self._safe_int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("eval_count")
        )
        total_tokens = self._safe_int(
            usage.get("total_tokens")
            or usage.get("tokens")
            or (prompt_tokens + completion_tokens)
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _parse_content_field(content: Any) -> Optional[str]:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
            return "\n".join(parts)
        return None

    def _extract_message_text(self, response_obj: Any) -> str:
        payload = self._to_response_dict(response_obj)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}

        msg = first.get("message")
        if isinstance(msg, dict):
            text = self._parse_content_field(msg.get("content"))
            if text is not None:
                return text

        text = first.get("text")
        return text if isinstance(text, str) else ""

    def _estimate_tokens(self, model: str, messages: List[Dict[str, Any]], response_obj: Any) -> Tuple[int, int, int]:
        try:
            if litellm is not None:
                prompt_tokens = int(litellm.token_counter(model=model, messages=messages) or 0)
            else:
                prompt_tokens = 0
        except Exception:
            prompt_tokens = 0
        try:
            completion_text = self._extract_message_text(response_obj)
            if litellm is not None and completion_text:
                completion_tokens = int(litellm.token_counter(model=model, text=completion_text) or 0)
            else:
                completion_tokens = 0
        except Exception:
            completion_tokens = 0
        total_tokens = prompt_tokens + completion_tokens
        return prompt_tokens, completion_tokens, total_tokens

    def _extract_usage(self, model: str, messages: List[Dict[str, Any]], response_obj: Any) -> Dict[str, Any]:
        payload = self._to_response_dict(response_obj)
        usage = payload.get("usage")
        if isinstance(usage, dict):
            normalized = self._normalize_usage_dict(usage)
            if normalized["total_tokens"] > 0:
                return {**normalized, "has_usage": True, "usage_source": "provider"}

        # Provider-specific fallback if usage exists at top-level.
        if payload:
            normalized = self._normalize_usage_dict(payload)
            if normalized["total_tokens"] > 0:
                return {**normalized, "has_usage": True, "usage_source": "provider_fallback"}

        # Deterministic local estimation when provider usage is unavailable.
        est_prompt, est_completion, est_total = self._estimate_tokens(model, messages, response_obj)
        if est_total > 0:
            return {
                "prompt_tokens": est_prompt,
                "completion_tokens": est_completion,
                "total_tokens": est_total,
                "has_usage": True,
                "usage_source": "estimated",
            }

        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "has_usage": False,
            "usage_source": "missing",
        }

    def _estimate_cost_usd(self, model: str, response_obj: Any, usage: Dict[str, Any]) -> float:
        try:
            if litellm is not None:
                return float(litellm.completion_cost(completion_response=response_obj, model=model) or 0.0)
        except Exception:
            pass
        try:
            if litellm is not None:
                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
                return float(
                    litellm.cost_per_token(
                        model=model,
                        prompt_tokens=pt,
                        completion_tokens=ct,
                    )
                    or 0.0
                )
        except Exception:
            pass
        return 0.0

    def _record_usage_event(
        self,
        *,
        stage: str,
        model: str,
        messages: List[Dict[str, Any]],
        response_obj: Any,
        source: str,
        duration_ms: float = 0.0,
    ) -> Dict[str, Any]:
        usage = self._extract_usage(model=model, messages=messages, response_obj=response_obj)
        cost_usd = self._estimate_cost_usd(model, response_obj, usage)
        event = {
            "stage": stage,
            "model": model,
            "source": source,
            "duration_ms": round(duration_ms, 3),
            "generation_params": self.effective_generation_config(model),
            **usage,
            "cost_usd": cost_usd,
        }
        self.usage_events.append(event)
        return event

    def _completion_span_attributes(
        self,
        *,
        stage: str,
        model: str,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "run_id": self.run_record.run_id,
            "seed": self.run_record.rng_seed,
            "stage": stage,
            "model": model,
            "cassette_mode": self.cassette.mode,
            "cassette_path": self.cassette.path,
            "message_count": len(messages) if isinstance(messages, list) else 0,
            "generation_params": self.effective_generation_config(model),
        }

    @staticmethod
    def _annotate_completion_span(
        span: Any,
        *,
        event: Dict[str, Any],
        cache_hit: bool,
    ) -> None:
        span.attributes.update(
            {
                "source": event.get("source"),
                "cache_hit": cache_hit,
                "prompt_tokens": event.get("prompt_tokens"),
                "completion_tokens": event.get("completion_tokens"),
                "total_tokens": event.get("total_tokens"),
                "cost_usd": event.get("cost_usd"),
                "has_usage": event.get("has_usage"),
                "usage_source": event.get("usage_source"),
            }
        )

    @staticmethod
    def _optional_float_env(name: str) -> Optional[float]:
        value = os.getenv(name, "").strip()
        if not value:
            return None
        return float(value)

    @staticmethod
    def _optional_int_env(name: str) -> Optional[int]:
        value = os.getenv(name, "").strip()
        if not value:
            return None
        return int(value)

    @classmethod
    def generation_config_from_env(cls, model_config: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        default_config: Dict[str, Any] = {}
        float_fields = {
            "temperature": "LLM_TEMPERATURE",
            "top_p": "LLM_TOP_P",
        }
        int_fields = {
            "top_k": "LLM_TOP_K",
            "max_tokens": "LLM_MAX_TOKENS",
        }
        for param, env_name in float_fields.items():
            value = cls._optional_float_env(env_name)
            if value is not None:
                default_config[param] = value
        for param, env_name in int_fields.items():
            value = cls._optional_int_env(env_name)
            if value is not None:
                default_config[param] = value

        profile = os.getenv("LLM_SAMPLING_PROFILE", "").strip().lower()
        model_values = list((model_config or {}).values())
        use_gemma4_profile = profile in ("ollama_gemma4", "gemma4_ollama", "gemma4") or (
            not profile and any("gemma4" in str(model).lower() for model in model_values)
        )

        model_overrides: Dict[str, Dict[str, Any]] = {}
        if use_gemma4_profile:
            model_overrides["gemma4"] = dict(GEMMA4_OLLAMA_SAMPLING_CONFIG)

        return {
            "default": default_config,
            "model_overrides": model_overrides,
            "sampling_profile": profile or ("auto_gemma4_ollama" if use_gemma4_profile else "env"),
        }

    def effective_generation_config(self, model: str) -> Dict[str, Any]:
        config = self.run_record.generation_config or {}

        # Backward compatibility for older flat RunRecord generation_config values.
        if any(key in config for key in GENERATION_PARAM_KEYS):
            effective = {k: v for k, v in config.items() if k in GENERATION_PARAM_KEYS}
        else:
            default = config.get("default")
            effective = dict(default) if isinstance(default, dict) else {}

        overrides = config.get("model_overrides")
        if isinstance(overrides, dict):
            model_lower = str(model).lower()
            for pattern, values in overrides.items():
                if str(pattern).lower() in model_lower and isinstance(values, dict):
                    effective.update(values)

        return effective

    def _apply_generation_config(self, model: str, kwargs: Dict[str, Any]) -> None:
        for key, value in self.effective_generation_config(model).items():
            kwargs.setdefault(key, value)

    def save_record(self, path: str):
        """Persist the RunRecord (A1)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.run_record.model_dump_json(indent=2))

    @classmethod
    def from_config(
        cls,
        run_id: str,
        seed: int,
        cassette_path: str,
        mode: str = "record",
        model_config: Dict[str, str] = None,
        generation_config: Dict[str, Any] = None,
        created_at: Optional[str] = None,
    ):
        record = RunRecord(
            run_id=run_id,
            rng_seed=seed,
            models=model_config or {},
            generation_config=generation_config if generation_config is not None else cls.generation_config_from_env(model_config),
            created_at=created_at or RunClock.from_env().now_iso(),
        )
        cassette = LLMCassette(cassette_path, mode)
        return cls(record, cassette)

    async def acompletion(self, model: str, messages: list, **kwargs):
        """Wrapper for litellm.acompletion with record/replay logic (A3)."""
        start_perf = time.perf_counter()
        # Use the module-level `litellm` binding (may be None in test environments)
        stage = str(kwargs.pop("stage", "unknown"))
        self._apply_generation_config(model, kwargs)

        span_attrs = self._completion_span_attributes(
            stage=stage,
            model=model,
            messages=messages,
        )
        with trace_span("llm_completion", stage=stage, attributes=span_attrs) as span:
            # In replay mode, fail loudly if not in cassette
            cached = self.cassette.get_response(model, messages, kwargs)
            if cached:
                duration_ms = (time.perf_counter() - start_perf) * 1000.0
                # Try to use litellm.ModelResponse when available for compatibility,
                # otherwise return the cached dict/object directly.
                model_response_cls = None
                if litellm is not None:
                    try:
                        from litellm.utils import ModelResponse as _ModelResponse
                        model_response_cls = _ModelResponse
                    except (ImportError, AttributeError):
                        model_response_cls = None

                if model_response_cls is not None and isinstance(cached, dict) and "choices" in cached:
                    response = model_response_cls(**cached)
                    event = self._record_usage_event(stage=stage, model=model, messages=messages, response_obj=response, source="cache", duration_ms=duration_ms)
                    self._annotate_completion_span(span, event=event, cache_hit=True)
                    return response

                # Fallback: return the cached object/dict and record usage using it.
                # _record_usage_event accepts either dict-like provider responses or model objects.
                event = self._record_usage_event(stage=stage, model=model, messages=messages, response_obj=cached, source="cache_fallback", duration_ms=duration_ms)
                self._annotate_completion_span(span, event=event, cache_hit=True)
                return cached

            if self.cassette.mode == "replay":
                span.attributes.update({"source": "replay_miss", "cache_hit": False})
                raise RuntimeError(f"Offline Replay Error: No cached response for {model} with hash {self.cassette._hash(model, messages, kwargs)}")

            # Make real call with increased timeout
            kwargs.setdefault("timeout", 1200)
            if litellm is None:
                span.attributes.update({"source": "provider", "cache_hit": False})
                raise RuntimeError("litellm is not available in this environment; cannot perform live completion")
            response = await litellm.acompletion(model=model, messages=messages, **kwargs)
            duration_ms = (time.perf_counter() - start_perf) * 1000.0

            # Save full response
            res_dict = response.model_dump() if hasattr(response, "model_dump") else response
            self.cassette.save_response(model, messages, kwargs, res_dict)
            event = self._record_usage_event(stage=stage, model=model, messages=messages, response_obj=response, source="provider", duration_ms=duration_ms)
            self._annotate_completion_span(span, event=event, cache_hit=False)
            return response
