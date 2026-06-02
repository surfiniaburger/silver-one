import json
import os
import tempfile
from typing import Any, Dict, Optional

from agentbeats.clock import RunClock


class CheckpointError(RuntimeError):
    pass


def utc_timestamp() -> str:
    return RunClock.from_env().now_iso()


def save_checkpoint(path: str, payload: Dict[str, Any], *, clock_now: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = dict(payload)
    data.setdefault("schema_version", 1)
    data["updated_at"] = RunClock.from_value(clock_now).now_iso()

    tempname: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(path) or ".",
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


def load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
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
