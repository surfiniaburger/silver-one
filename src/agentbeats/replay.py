import hashlib
import json
import fcntl
import tempfile
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

class RunRecord(BaseModel):
    run_id: str
    rng_seed: int
    models: Dict[str, str]
    git_commit: Optional[str] = None
    created_at: str

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
                print(f"Warning: Failed to load cassette {path}: {e}")

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
                print(f"Warning: Failed to load cassette {self.path}: {e}")

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
        import random
        self.rng = random.Random(run_record.rng_seed)

    def save_record(self, path: str):
        """Persist the RunRecord (A1)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.run_record.model_dump_json(indent=2))

    @classmethod
    def from_config(cls, run_id: str, seed: int, cassette_path: str, mode: str = "record", model_config: Dict[str, str] = None):
        from datetime import datetime
        record = RunRecord(
            run_id=run_id,
            rng_seed=seed,
            models=model_config or {},
            created_at=datetime.now().isoformat()
        )
        cassette = LLMCassette(cassette_path, mode)
        return cls(record, cassette)

    async def acompletion(self, model: str, messages: list, **kwargs):
        """Wrapper for litellm.acompletion with record/replay logic (A3)."""
        import litellm
        
        # In replay mode, fail loudly if not in cassette
        cached = self.cassette.get_response(model, messages, kwargs)
        if cached:
            from litellm.utils import ModelResponse
            if isinstance(cached, dict) and "choices" in cached:
                # Return a proper litellm ModelResponse object for serialization compatibility
                return ModelResponse(**cached)
            else:
                # Legacy or string-only fallback
                return ModelResponse(choices=[{"message": {"content": str(cached), "role": "assistant"}}])

        if self.cassette.mode == "replay":
            raise RuntimeError(f"Offline Replay Error: No cached response for {model} with hash {self.cassette._hash(model, messages, kwargs)}")

        # Make real call with increased timeout
        kwargs.setdefault("timeout", 1200)
        response = await litellm.acompletion(model=model, messages=messages, **kwargs)
        
        # Save full response
        res_dict = response.model_dump() if hasattr(response, "model_dump") else response
        self.cassette.save_response(model, messages, kwargs, res_dict)
        return response
