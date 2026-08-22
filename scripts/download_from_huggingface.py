#!/usr/bin/env python3
"""
Download clean CVE seeds dataset from Hugging Face Hub.
Repository: surfiniaburger/cve-decision-seeds
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


def _safe_resolve(path_str: str, base_dir: Path) -> Path:
    """Validate and safely resolve a path within base_dir to prevent path traversal."""
    resolved = (base_dir / path_str).resolve()
    base_resolved = base_dir.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Path traversal error: '{path_str}' escapes base directory '{base_dir}'")
    return resolved


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _log_provenance(entry: dict, base_dir: Path) -> None:
    """Append operation provenance record to append-only JSONL log."""
    log_path = base_dir / "artifacts" / "provenance" / "hub_operations.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download clean CVE seeds dataset from Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default="surfiniaburger/cve-decision-seeds",
        help="Hugging Face dataset repository ID (default: surfiniaburger/cve-decision-seeds)",
    )
    parser.add_argument(
        "--filename",
        default="cve_seeds_500_clean.jsonl",
        help="Target filename inside dataset repo",
    )
    parser.add_argument(
        "--output",
        default="scenarios/debate/cve_seeds_500_clean.jsonl",
        help="Local destination file path",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Immutable revision, branch or commit SHA (default: main)",
    )
    args = parser.parse_args()

    workspace_root = Path.cwd().resolve()
    try:
        target_path = _safe_resolve(args.output, workspace_root)
    except ValueError as e:
        print(f"Security error: {e}", file=sys.stderr)
        sys.exit(1)

    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize remote filename against directory traversal
    clean_filename = os.path.basename(args.filename)

    print(f"Downloading {clean_filename} (revision: {args.revision}) from {args.repo_id} to {target_path}...")
    downloaded_path_str = hf_hub_download(
        repo_id=args.repo_id,
        filename=clean_filename,
        revision=args.revision,
        repo_type="dataset",
        local_dir=str(target_dir),
    )
    downloaded_path = Path(downloaded_path_str).resolve()
    file_sha256 = _compute_sha256(downloaded_path)

    # Log immutable operation provenance
    _log_provenance(
        {
            "operation": "download",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repo_id": args.repo_id,
            "filename": clean_filename,
            "revision": args.revision,
            "file_sha256": file_sha256,
            "target_path": str(downloaded_path.relative_to(workspace_root)),
        },
        workspace_root,
    )

    print(f"Downloaded successfully: {downloaded_path} (SHA-256: {file_sha256[:12]}...)")


if __name__ == "__main__":
    main()
