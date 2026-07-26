import json
import os
import tempfile
from typing import Any, Dict, Optional

from agentbeats.clock import RunClock


class CheckpointError(RuntimeError):
    pass


def utc_timestamp() -> str:
    return RunClock.from_env().now_iso()


def _validate_path(path: str, base_dir: Optional[str] = None) -> str:
    """Validate and canonicalize file path to prevent path traversal vulnerability (CWE-22)."""
    if not path or not isinstance(path, str):
        raise CheckpointError("Invalid file path: path must be a non-empty string.")
    
    resolved_path = os.path.realpath(os.path.abspath(path))
    
    if base_dir:
        resolved_base = os.path.realpath(os.path.abspath(base_dir))
        if not resolved_path.startswith(resolved_base + os.sep) and resolved_path != resolved_base:
            raise CheckpointError(f"Path traversal detected: path '{path}' escapes base directory '{base_dir}'.")
            
    return resolved_path


def save_checkpoint(path: str, payload: Dict[str, Any], *, clock_now: Optional[str] = None, base_dir: Optional[str] = None) -> None:
    path = _validate_path(path, base_dir=base_dir)
    target_dir = _validate_path(os.path.dirname(path) or ".", base_dir=base_dir)
    os.makedirs(target_dir, exist_ok=True)
    data = dict(payload)
    data.setdefault("schema_version", 1)
    data["updated_at"] = RunClock.from_value(clock_now).now_iso()

    tempname: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir or ".",
            delete=False,
            suffix=".tmp",
        ) as tf:
            json.dump(data, tf, indent=2, sort_keys=True)
            tf.write("\n")
            tempname = tf.name
        os.replace(tempname, path)
        tempname = None
    finally:
        if tempname and os.path.exists(tempname):
            try:
                os.remove(tempname)
            except Exception:
                pass


def load_checkpoint(path: str, base_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = _validate_path(path, base_dir=base_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint {path}: {e}") from e
    if not isinstance(data, dict):
        raise CheckpointError(f"Invalid checkpoint {path}: expected JSON object")
    return data


def validate_checkpoint(checkpoint: Dict[str, Any], expected: Dict[str, Any]) -> None:
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            mismatches.append(
                {
                    "key": key,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    if mismatches:
        raise CheckpointError(
            "Checkpoint does not match current run controls: "
            + json.dumps(mismatches, sort_keys=True)
        )
