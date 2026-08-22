#!/usr/bin/env python3
"""
Download clean CVE seeds dataset from Hugging Face Hub.
Repository: surfiniaburger/cve-decision-seeds
"""

import argparse
import os
from huggingface_hub import hf_hub_download


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
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"Downloading {args.filename} from {args.repo_id} to {args.output}...")
    downloaded_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        repo_type="dataset",
        local_dir=os.path.dirname(os.path.abspath(args.output)),
    )
    print(f"Downloaded successfully: {downloaded_path}")


if __name__ == "__main__":
    main()
