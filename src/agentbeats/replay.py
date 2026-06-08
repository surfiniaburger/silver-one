import hashlib
import json
import fcntl
import tempfile
import os
import logging

logger = logging.getLogger(__name__)
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel
from agentbeats.clock import RunClock

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
                logger = logging.getLogger(__name__)
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
                logger = logging.getLogger(__name__)
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
                        self.data.update(json.load(f))
                    except: pass
            
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

    def get_usage_summary(self) -> Dict[str, Any]:
        by_stage: Dict[str, Dict[str, float]] = {}
        by_model: Dict[str, Dict[str, float]] = {}
        totals = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "missing_usage_calls": 0,
        }
        for ev in self.usage_events:
            stage = str(ev.get("stage", "unknown"))
            model = str(ev.get("model", "unknown"))
            pt = int(ev.get("prompt_tokens") or 0)
            ct = int(ev.get("completion_tokens") or 0)
            tt = int(ev.get("total_tokens") or 0)
            cost = float(ev.get("cost_usd") or 0.0)
            has_usage = bool(ev.get("has_usage"))

            totals["calls"] += 1
            totals["prompt_tokens"] += pt
            totals["completion_tokens"] += ct
            totals["total_tokens"] += tt
            totals["cost_usd"] += cost
            if not has_usage:
                totals["missing_usage_calls"] += 1

            s = by_stage.setdefault(stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0})
            s["calls"] += 1
            s["prompt_tokens"] += pt
            s["completion_tokens"] += ct
            s["total_tokens"] += tt
            s["cost_usd"] += cost

            m = by_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0})
            m["calls"] += 1
            m["prompt_tokens"] += pt
            m["completion_tokens"] += ct
            m["total_tokens"] += tt
            m["cost_usd"] += cost

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

    def _extract_message_text(self, response_obj: Any) -> str:
        payload = self._to_response_dict(response_obj)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                out: List[str] = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        out.append(part["text"])
                return "\n".join(out)
        text = first.get("text")
        if isinstance(text, str):
            return text
        return ""

    def _estimate_tokens(self, model: str, messages: List[Dict[str, Any]], response_obj: Any) -> Tuple[int, int, int]:
        try:
            import litellm
            prompt_tokens = int(litellm.token_counter(model=model, messages=messages) or 0)
        except Exception:
            prompt_tokens = 0
        try:
            import litellm
            completion_text = self._extract_message_text(response_obj)
            completion_tokens = int(
                litellm.token_counter(model=model, text=completion_text) or 0
            ) if completion_text else 0
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
            import litellm
            return float(litellm.completion_cost(completion_response=response_obj, model=model) or 0.0)
        except Exception:
            try:
                import litellm
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
                return 0.0

    def _record_usage_event(self, *, stage: str, model: str, messages: List[Dict[str, Any]], response_obj: Any, source: str) -> None:
        usage = self._extract_usage(model=model, messages=messages, response_obj=response_obj)
        cost_usd = self._estimate_cost_usd(model, response_obj, usage)
        self.usage_events.append(
            {
                "stage": stage,
                "model": model,
                "source": source,
                "generation_params": self.effective_generation_config(model),
                **usage,
                "cost_usd": cost_usd,
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
        import litellm
        stage = str(kwargs.pop("stage", "unknown"))
        self._apply_generation_config(model, kwargs)
        
        # In replay mode, fail loudly if not in cassette
        cached = self.cassette.get_response(model, messages, kwargs)
        if cached:
            from litellm.utils import ModelResponse
            if isinstance(cached, dict) and "choices" in cached:
                # Return a proper litellm ModelResponse object for serialization compatibility
                response = ModelResponse(**cached)
                self._record_usage_event(stage=stage, model=model, messages=messages, response_obj=response, source="cache")
                return response
            else:
                # Legacy or string-only fallback
                response = ModelResponse(choices=[{"message": {"content": str(cached), "role": "assistant"}}])
                self._record_usage_event(stage=stage, model=model, messages=messages, response_obj=response, source="cache_legacy")
                return response

        if self.cassette.mode == "replay":
            raise RuntimeError(f"Offline Replay Error: No cached response for {model} with hash {self.cassette._hash(model, messages, kwargs)}")

        # Make real call with increased timeout
        kwargs.setdefault("timeout", 1200)
        response = await litellm.acompletion(model=model, messages=messages, **kwargs)
        
        # Save full response
        res_dict = response.model_dump() if hasattr(response, "model_dump") else response
        self.cassette.save_response(model, messages, kwargs, res_dict)
        self._record_usage_event(stage=stage, model=model, messages=messages, response_obj=response, source="provider")
        return response
