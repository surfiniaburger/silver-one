import json
import asyncio
import sys
import os
import argparse
from datetime import datetime

# Add src to PYTHONPATH to import agentbeats
sys.path.append(os.path.join(os.getcwd(), "src"))

from agentbeats.client import send_message

async def run_batch():
    parser = argparse.ArgumentParser(description="BARRED Batch Runner (Deterministic)")
    parser.add_argument("--seeds", default="scenarios/debate/cve_seeds_50.jsonl")
    parser.add_argument("--output", default="test_corpus_50.jsonl")
    parser.add_argument("--run-id", default=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["record", "replay"], default="record")
    parser.add_argument("--cassette-path", default="", help="Optional cassette path for the judge (defaults to artifacts/cassettes/<run-id>.json)")
    parser.add_argument("--record-path", default="", help="Optional run record path for the judge (defaults to artifacts/runs/<run-id>.json)")
    args = parser.parse_args()

    judge_url = "http://127.0.0.1:9009"
    
    if not os.path.exists(args.seeds):
        print(f"Error: {args.seeds} not found.")
        return

    # Load existing results to support resume
    processed_predicates = set()
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_predicates.add(data.get("instruction", ""))
                except:
                    pass

    with open(args.seeds, "r") as f:
        seeds = [json.loads(line) for line in f]
    
    print(f"Loaded {len(seeds)} seeds. {len(processed_predicates)} already processed.")
    print(f"Run ID: {args.run_id} | Seed: {args.seed} | Mode: {args.mode}")
    
    for i, seed in enumerate(seeds):
        instruction = f"Analyze this input for the condition: {seed['predicate']}"
        if instruction in processed_predicates:
            print(f"Skipping seed {i+1} (already processed).")
            continue

        print(f"\n>>> [{i+1}/{len(seeds)}] Seed Predicate: {seed.get('predicate')[:80]}...")
        
        payload = {
            "participants": {
                "pro_debater": "http://127.0.0.1:9019/",
                "con_debater": "http://127.0.0.1:9018/"
            },
            "config": {
                "run_id": args.run_id,
                "seed": args.seed + i, # Increment seed per item for variety but stay deterministic
                "mode": args.mode,
                **({"cassette_path": args.cassette_path} if args.cassette_path else {}),
                **({"record_path": args.record_path} if args.record_path else {}),
                "topic": seed["topic"],
                "predicate": seed["predicate"],
                "target_verdict": "True",
                "target_dimension": "Security Invariants",
                "num_rounds": 2,
                "max_refinements": 1,
                "output_file": args.output
            }
        }
        
        try:
            result = await send_message(json.dumps(payload), judge_url)
            print(f"  Result received. Status: {result.get('status')}")
        except Exception as e:
            print(f"  ERROR: Seed {i+1} failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_batch())

