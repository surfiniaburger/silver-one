#!/usr/bin/env python3
"""
Upload CVE Clean Seeds Dataset to Hugging Face Hub.
Repository: surfiniaburger/cve-decision-seeds
"""

import argparse
import os
import sys
from huggingface_hub import HfApi, create_repo


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
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face API write token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make dataset private",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: Seed file not found at {args.file}", file=sys.stderr)
        sys.exit(1)

    token = args.token
    if not token:
        print(
            "Error: Hugging Face authentication token is required.\n"
            "Provide it via --token <your_hf_token> or set HF_TOKEN in your environment.\n"
            "Create a write token at: https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami()
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

    print(f"Uploading {args.file} to {args.repo_id}/cve_seeds_500_clean.jsonl...")
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo="cve_seeds_500_clean.jsonl",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add 500 clean verified CVE debate seeds (GEPA-first)",
    )

    print("Uploading README.md dataset card...")
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card and documentation",
    )

    print(f"\nSuccess! Dataset is published at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
