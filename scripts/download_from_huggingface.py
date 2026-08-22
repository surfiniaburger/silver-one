#!/usr/bin/env python3
"""
Download clean CVE seeds dataset from Hugging Face Hub.
Repository: surfiniaburger/cve-decision-seeds
"""

import argparse
import datetime
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Ensure repository root is on sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hub_common import (
    DEFAULT_FILENAME,
    DEFAULT_LOCAL_PATH,
    DEFAULT_REPO_ID,
    compute_sha256,
    log_provenance,
    safe_resolve,
)


def main():
    parser = argparse.ArgumentParser(description="Download clean CVE seeds dataset from Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help=f"Target filename inside dataset repo (default: {DEFAULT_FILENAME})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_LOCAL_PATH,
        help=f"Local destination file path (default: {DEFAULT_LOCAL_PATH})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Immutable revision, branch or commit SHA (default: main)",
    )
    args = parser.parse_args()

    workspace_root = Path.cwd().resolve()
    try:
        target_path = safe_resolve(args.output, workspace_root)
    except ValueError as e:
        print(f"Security error: {e}", file=sys.stderr)
        sys.exit(1)

    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize remote filename against directory traversal
    clean_filename = os.path.basename(args.filename)

    # 1. Log pre-network attempt record (started)
    log_provenance(
        {
            "operation": "download",
            "phase": "download_started",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repo_id": args.repo_id,
            "filename": clean_filename,
            "revision": args.revision,
            "target_path": str(target_path.relative_to(workspace_root)),
        },
        workspace_root,
    )

    print(f"Downloading {clean_filename} (revision: {args.revision}) from {args.repo_id} to {target_path}...")
    try:
        downloaded_path_str = hf_hub_download(
            repo_id=args.repo_id,
            filename=clean_filename,
            revision=args.revision,
            repo_type="dataset",
            local_dir=str(target_dir),
        )
        downloaded_path = Path(downloaded_path_str).resolve()
        file_sha256 = compute_sha256(downloaded_path)

        # 2. Log success record with immutable file digest
        log_provenance(
            {
                "operation": "download",
                "phase": "download_success",
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
    except Exception as e:
        # 3. Log failure record
        log_provenance(
            {
                "operation": "download",
                "phase": "download_failed",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "repo_id": args.repo_id,
                "filename": clean_filename,
                "revision": args.revision,
                "error": str(e),
                "target_path": str(target_path.relative_to(workspace_root)),
            },
            workspace_root,
        )
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
