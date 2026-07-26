import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

from agentbeats.clock import RunClock


class CheckpointError(RuntimeError):
    pass


def _validate_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Validate and canonicalize file path to prevent path traversal vulnerability (CWE-22)."""
    if not path:
        raise CheckpointError("Invalid file path: path must be a non-empty string or Path.")
    
    resolved_path = Path(path).resolve()
    
    if base_dir:
        resolved_base = Path(base_dir).resolve()
        try:
            resolved_path.relative_to(resolved_base)
        except ValueError:
            raise CheckpointError(f"Path traversal detected: path '{path}' escapes base directory '{base_dir}'.")
            
    return resolved_path


def save_checkpoint(path: Union[str, Path], payload: Dict[str, Any], *, clock_now: Optional[str] = None, base_dir: Optional[Union[str, Path]] = None) -> None:
    valid_path = _validate_path(path, base_dir=base_dir)
    target_dir = valid_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    data = dict(payload)
    data.setdefault("schema_version", 1)
    data["updated_at"] = RunClock.from_value(clock_now).now_iso()

    tempname: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir,
            delete=False,
            suffix=".tmp",
        ) as tf:
            json.dump(data, tf, indent=2, sort_keys=True)
            tf.write("\n")
            tempname = tf.name
        os.replace(tempname, str(valid_path))
        tempname = None
    finally:
        if tempname and os.path.exists(tempname):
            try:
                os.remove(tempname)
            except Exception:
                pass


def load_checkpoint(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    valid_path = _validate_path(path, base_dir=base_dir)
    try:
        with open(valid_path, "r", encoding="utf-8") as f:
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
