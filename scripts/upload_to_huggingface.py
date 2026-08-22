#!/usr/bin/env python3
"""
Upload CVE Clean Seeds Dataset to Hugging Face Hub.
Repository: surfiniaburger/cve-decision-seeds
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple
from huggingface_hub import HfApi, create_repo, get_token


DATASET_CARD = """---
language:
- en
- code
license: mit
task_categories:
- text-generation
tags:
- c
- cpp
- code-security
- cve
- reasoning
- gepa
- vulnerability-detection
- synthetic-data
size_categories:
- n<1K
---

# CVE Decision Seeds (500 Clean Verified Seeds)

This dataset contains **500 high-fidelity, verified C/C++ vulnerability seeds** generated using the **GEPA-First (Generative Explanation of Program Anomalies)** framework for the BARRED synthetic debate pipeline.

## Overview

- **Source Corpus:** Extracted from [CVEFixes](https://www.kaggle.com/datasets/girish17019/cvefixes-vulnerable-and-fixed-code).
- **Clean-Room Anti-Leakage Partitioning:** Excluded against all 5,000 held-out evaluation scenarios in [cve-decision](https://www.kaggle.com/datasets/surfiniaburger/cve-decision) using exact, normalized, and 5-gram fuzzy shingling ($J < 0.80$).
- **Strict Length Limits:** $500 \\le \\text{code length} \\le 12,000$ characters.
- **Structured Explanations:** Generated using `call_structured(GepaExplanation)` with `gpt-oss:120b-cloud`.
- **Anchor Grounding:** $100\\%$ non-generic AST syntax tokens (no generic stopwords).

## Schema

Each JSONL line represents one vulnerability seed:

```json
{
  "topic": "int func(int a) { ... }",
  "predicate": "The code is vulnerable to an out-of-bounds write in `func`...",
  "gepa_info": {
    "predicate": "The code is vulnerable to an out-of-bounds write in `func`...",
    "evidence_hooks": ["`func`", "`arr[idx]`"],
    "uncertainty": "Low",
    "proof_requirements": "Provide idx >= size"
  },
  "language": "c",
  "original_safety": "vulnerable",
  "anchors": ["func", "arr[idx]"]
}
```

## Quick Start

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="https://huggingface.co/datasets/surfiniaburger/cve-decision-seeds/resolve/main/cve_seeds_500_clean.jsonl")
print(f"Loaded {len(dataset['train'])} seeds.")
```
"""

REQUIRED_SCHEMA_KEYS = {"topic", "predicate", "gepa_info", "language", "original_safety", "anchors"}
SUPPORTED_LANGUAGES = {"c", "cpp", "c++"}


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


def validate_seed_corpus(file_path: Path, expected_min_count: int = 10) -> Tuple[bool, str, int]:
    """
    Validate all JSONL records before publication.
    Enforces documented schema, length bounds, allowed languages, and required GEPA fields.
    """
    if not file_path.is_file():
        return False, f"File does not exist: {file_path}", 0

    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                return False, f"Empty line at line {lineno}", count
            try:
                record = json.loads(line)
            except Exception as e:
                return False, f"Malformed JSON at line {lineno}: {e}", count

            # Validate required schema keys
            missing_keys = REQUIRED_SCHEMA_KEYS - set(record.keys())
            if missing_keys:
                return False, f"Line {lineno} missing required keys: {missing_keys}", count

            # Validate language
            lang = (record.get("language") or "").strip().lower()
            if lang not in SUPPORTED_LANGUAGES:
                return False, f"Line {lineno} unsupported language: '{lang}'", count

            # Validate topic length
            topic = record.get("topic", "")
            if not isinstance(topic, str) or len(topic) < 50 or len(topic) > 12000:
                return False, f"Line {lineno} invalid topic length: {len(topic)} (must be 50-12000)", count

            # Validate predicate
            pred = record.get("predicate", "")
            if not isinstance(pred, str) or not pred.strip():
                return False, f"Line {lineno} missing or empty predicate", count

            # Validate GEPA info
            gepa = record.get("gepa_info", {})
            if not isinstance(gepa, dict) or not gepa.get("predicate"):
                return False, f"Line {lineno} missing or invalid gepa_info payload", count

            # Validate anchors
            anchors = record.get("anchors", [])
            if not isinstance(anchors, list) or len(anchors) < 1:
                return False, f"Line {lineno} missing anchors list", count

            count += 1

    if count < expected_min_count:
        return False, f"Record count {count} is below expected minimum {expected_min_count}", count

    return True, "Valid", count


def main():
    parser = argparse.ArgumentParser(description="Upload clean CVE seeds dataset to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default="surfiniaburger/cve-decision-seeds",
        help="Hugging Face dataset repository ID (default: surfiniaburger/cve-decision-seeds)",
    )
    parser.add_argument(
        "--file",
        default="scenarios/debate/cve_seeds_500_clean.jsonl",
        help="Path to clean seeds JSONL file",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make dataset private",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=500,
        help="Expected record count in dataset (default: 500)",
    )
    args = parser.parse_args()

    workspace_root = Path.cwd().resolve()
    try:
        target_file = _safe_resolve(args.file, workspace_root)
    except ValueError as e:
        print(f"Security error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate seed corpus before making any Hub calls
    is_valid, reason, total_records = validate_seed_corpus(target_file, expected_min_count=min(10, args.expected_count))
    if not is_valid:
        print(f"Corpus validation failed: {reason}", file=sys.stderr)
        sys.exit(1)

    print(f"Verified corpus: {total_records} clean, valid records in {target_file.name}.")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
    if not token:
        print(
            "Error: Hugging Face authentication token is required.\n"
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment or log in via `hf auth login`.\n"
            "Create a write token at: https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
    except Exception as e:
        print(f"Authentication failed: {e}", file=sys.stderr)
        sys.exit(1)

    username = user_info.get("name")
    print(f"Authenticated as Hugging Face user: {username}")

    print(f"Creating / verifying dataset repo: {args.repo_id}...")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        token=token,
        exist_ok=True,
    )

    # Explicitly enforce and verify private visibility when requested
    if args.private:
        api.update_repo_settings(repo_id=args.repo_id, repo_type="dataset", private=True)
        info = api.dataset_info(repo_id=args.repo_id, token=token)
        if not info.private:
            print(f"Security error: Repository {args.repo_id} is not private as requested. Aborting upload.", file=sys.stderr)
            sys.exit(1)

    source_sha256 = _compute_sha256(target_file)
    print(f"Uploading {target_file.name} to {args.repo_id}/cve_seeds_500_clean.jsonl...")
    commit_data = api.upload_file(
        path_or_fileobj=str(target_file),
        path_in_repo="cve_seeds_500_clean.jsonl",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"Add {total_records} clean verified CVE debate seeds (GEPA-first)",
    )
    data_commit_oid = commit_data.oid if hasattr(commit_data, "oid") else str(commit_data)

    print("Uploading README.md dataset card...")
    commit_card = api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card and documentation",
    )
    card_commit_oid = commit_card.oid if hasattr(commit_card, "oid") else str(commit_card)

    # Log immutable operation provenance
    _log_provenance(
        {
            "operation": "upload",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repo_id": args.repo_id,
            "filename": target_file.name,
            "total_records": total_records,
            "source_sha256": source_sha256,
            "data_commit_oid": data_commit_oid,
            "card_commit_oid": card_commit_oid,
            "private": bool(args.private),
        },
        workspace_root,
    )

    print(f"\nSuccess! Dataset is published at: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Commit OID: {data_commit_oid} | SHA-256: {source_sha256[:12]}...")


if __name__ == "__main__":
    main()
