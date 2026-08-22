"""
Shared utilities for Hugging Face Hub synchronization scripts.
Provides safe path resolution, file hashing, and provenance logging.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_REPO_ID = "surfiniaburger/cve-decision-seeds"
DEFAULT_FILENAME = "cve_seeds_500_clean.jsonl"
DEFAULT_LOCAL_PATH = "scenarios/debate/cve_seeds_500_clean.jsonl"


def safe_resolve(path_str: str, base_dir: Path) -> Path:
    """Validate and safely resolve a path within base_dir to prevent path traversal."""
    resolved = (base_dir / path_str).resolve()
    base_resolved = base_dir.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Path traversal error: '{path_str}' escapes base directory '{base_dir}'")
    return resolved


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def log_provenance(entry: Dict[str, Any], base_dir: Path) -> None:
    """Append operation provenance record to append-only JSONL log."""
    log_path = base_dir / "artifacts" / "provenance" / "hub_operations.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
