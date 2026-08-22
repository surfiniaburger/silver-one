#!/usr/bin/env python3
"""
Upload CVE Clean Seeds Dataset to Hugging Face Hub.
Repository: surfiniaburger/cve-decision-seeds
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Tuple
from huggingface_hub import HfApi, create_repo, get_token

# Ensure repository root is on sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hub_common import (
    DEFAULT_LOCAL_PATH,
    DEFAULT_REPO_ID,
    compute_sha256,
    log_provenance,
    safe_resolve,
)


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
- **Strict Length Limits:** $200 \\le \\text{code length} \\le 12,000$ characters.
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
REQUIRED_GEPA_KEYS = {"predicate", "evidence_hooks", "uncertainty", "proof_requirements"}
SUPPORTED_LANGUAGES = {"c", "cpp", "c++"}


def _validate_gepa_payload(gepa: Any, lineno: int) -> Tuple[bool, str]:
    """Validate GEPA payload subfields."""
    if not isinstance(gepa, dict):
        return False, f"Line {lineno} gepa_info must be a JSON object"
    missing_gepa = REQUIRED_GEPA_KEYS - set(gepa.keys())
    if missing_gepa:
        return False, f"Line {lineno} gepa_info missing required subfields: {sorted(missing_gepa)}"
    if not isinstance(gepa.get("predicate"), str) or not gepa["predicate"].strip():
        return False, f"Line {lineno} gepa_info.predicate is empty"
    if not isinstance(gepa.get("evidence_hooks"), list) or len(gepa["evidence_hooks"]) < 1:
        return False, f"Line {lineno} gepa_info.evidence_hooks must be a non-empty list"
    if not isinstance(gepa.get("proof_requirements"), str) or not gepa["proof_requirements"].strip():
        return False, f"Line {lineno} gepa_info.proof_requirements is empty"
    return True, ""


def _validate_record(record: Any, lineno: int) -> Tuple[bool, str]:
    """Validate a single parsed JSON record against the dataset contract."""
    if not isinstance(record, dict):
        return False, f"Line {lineno} is not a JSON object"

    missing_keys = REQUIRED_SCHEMA_KEYS - set(record.keys())
    if missing_keys:
        return False, f"Line {lineno} missing required keys: {sorted(missing_keys)}"

    raw_language = record.get("language")
    if not isinstance(raw_language, str):
        return False, f"Line {lineno} language must be a string"
    lang = raw_language.strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        return False, f"Line {lineno} unsupported language: '{lang}'"

    topic = record.get("topic")
    if not isinstance(topic, str):
        return False, f"Line {lineno} topic must be a string"
    if len(topic) < 200 or len(topic) > 12000:
        return False, f"Line {lineno} invalid topic length: {len(topic)} (must be 200-12000)"

    pred = record.get("predicate", "")
    if not isinstance(pred, str) or not pred.strip():
        return False, f"Line {lineno} missing or empty predicate"

    anchors = record.get("anchors", [])
    if not isinstance(anchors, list) or len(anchors) < 1:
        return False, f"Line {lineno} missing anchors list"

    return _validate_gepa_payload(record.get("gepa_info"), lineno)


def validate_seed_corpus(file_path: Path, expected_count: int = 500) -> Tuple[bool, str, int]:
    """
    Validate all JSONL records before publication.
    Enforces documented schema, exact record count, length bounds (200-12000),
    allowed languages, non-empty anchors, and complete GEPA payload subfields.
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

            is_valid, err_msg = _validate_record(record, lineno)
            if not is_valid:
                return False, err_msg, count

            count += 1

    if count != expected_count:
        return False, f"Corpus record count ({count}) does not match expected count ({expected_count})", count

    return True, "Valid", count


def main():
    parser = argparse.ArgumentParser(description="Upload clean CVE seeds dataset to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_LOCAL_PATH,
        help=f"Path to clean seeds JSONL file (default: {DEFAULT_LOCAL_PATH})",
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
        help="Expected exact record count in dataset (default: 500)",
    )
    args = parser.parse_args()

    workspace_root = Path.cwd().resolve()
    try:
        target_file = safe_resolve(args.file, workspace_root)
    except ValueError as e:
        print(f"Security error: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Validate seed corpus before making any Hub calls
    is_valid, reason, total_records = validate_seed_corpus(target_file, expected_count=args.expected_count)
    if not is_valid:
        print(f"Corpus validation failed: {reason}", file=sys.stderr)
        sys.exit(1)

    print(f"Verified corpus: {total_records} clean, valid records in {target_file.name}.")
    source_sha256 = compute_sha256(target_file)

    # 2. Log pre-operation started record
    log_provenance(
        {
            "operation": "upload",
            "phase": "upload_started",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repo_id": args.repo_id,
            "filename": target_file.name,
            "total_records": total_records,
            "source_sha256": source_sha256,
            "private": bool(args.private),
        },
        workspace_root,
    )

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
        log_provenance(
            {
                "operation": "upload",
                "phase": "auth_failed",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "repo_id": args.repo_id,
                "error": str(e),
            },
            workspace_root,
        )
        print(f"Authentication failed: {e}", file=sys.stderr)
        sys.exit(1)

    username = user_info.get("name")
    print(f"Authenticated as Hugging Face user: {username}")

    print(f"Creating / verifying dataset repo: {args.repo_id}...")
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            token=token,
            exist_ok=True,
        )
        if args.private:
            api.update_repo_settings(repo_id=args.repo_id, repo_type="dataset", private=True)
            info = api.dataset_info(repo_id=args.repo_id, token=token)
            if not info.private:
                raise RuntimeError(f"Repository {args.repo_id} is not private as requested.")
    except Exception as e:
        log_provenance(
            {
                "operation": "upload",
                "phase": "repo_setup_failed",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "repo_id": args.repo_id,
                "error": str(e),
            },
            workspace_root,
        )
        print(f"Repository setup failed: {e}", file=sys.stderr)
        sys.exit(1)

    data_commit_oid = None
    card_commit_oid = None

    # 3. Upload data file
    print(f"Uploading {target_file.name} to {args.repo_id}/cve_seeds_500_clean.jsonl...")
    try:
        commit_data = api.upload_file(
            path_or_fileobj=str(target_file),
            path_in_repo="cve_seeds_500_clean.jsonl",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Add {total_records} clean verified CVE debate seeds (GEPA-first)",
        )
        data_commit_oid = commit_data.oid if hasattr(commit_data, "oid") else str(commit_data)
        log_provenance(
            {
                "operation": "upload",
                "phase": "data_upload_success",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "repo_id": args.repo_id,
                "data_commit_oid": data_commit_oid,
                "source_sha256": source_sha256,
            },
            workspace_root,
        )
    except Exception as e:
        log_provenance(
            {
                "operation": "upload",
                "phase": "data_upload_failed",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "repo_id": args.repo_id,
                "error": str(e),
            },
            workspace_root,
        )
        print(f"Data upload failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Upload dataset card
    print("Uploading README.md dataset card...")
    try:
        commit_card = api.upload_file(
            path_or_fileobj=DATASET_CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card and documentation",
        )
        card_commit_oid = commit_card.oid if hasattr(commit_card, "oid") else str(commit_card)
        log_provenance(
            {
                "operation": "upload",
                "phase": "upload_complete",
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
    except Exception as e:
        log_provenance(
            {
                "operation": "upload",
                "phase": "card_upload_failed",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "repo_id": args.repo_id,
                "data_commit_oid": data_commit_oid,
                "error": str(e),
            },
            workspace_root,
        )
        print(f"Dataset card upload failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nSuccess! Dataset is published at: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Data Commit OID: {data_commit_oid} | Card Commit OID: {card_commit_oid} | SHA-256: {source_sha256[:12]}...")


if __name__ == "__main__":
    main()
