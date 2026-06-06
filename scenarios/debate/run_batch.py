import json
import asyncio
import sys
import os
import argparse

# Add src to PYTHONPATH to import agentbeats
sys.path.append(os.path.join(os.getcwd(), "src"))

from agentbeats.client import send_message
from agentbeats.checkpoint import save_checkpoint
from agentbeats.clock import RunClock

async def run_batch():
    parser = argparse.ArgumentParser(description="BARRED Batch Runner (Deterministic)")
    parser.add_argument("--seeds", default="scenarios/debate/cve_seeds_50.jsonl")
    parser.add_argument("--output", default="test_corpus_50.jsonl")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["record", "replay"], default="record")
    parser.add_argument("--cassette-path", default="", help="Optional cassette path for the judge (defaults to artifacts/cassettes/<run-id>.json)")
    parser.add_argument("--record-path", default="", help="Legacy override: use one run record path for all seeds")
    parser.add_argument("--record-dir", default="artifacts/runs", help="Directory for per-seed run records")
    parser.add_argument("--attempts-out", default="", help="Optional attempts log path (defaults to artifacts/attempts/<run-id>.jsonl)")
    parser.add_argument("--resume", action="store_true", help="Resume each seed from its latest checkpoint when available")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints", help="Directory for per-seed checkpoint files")
    parser.add_argument("--manifest-out", default="", help="Optional batch manifest path")
    parser.add_argument("--clock-now", default="", help="Inject a fixed ISO timestamp for run records/checkpoints/manifests")
    args = parser.parse_args()

    clock = RunClock.from_value(args.clock_now or os.getenv("RUN_CLOCK_NOW", ""))
    batch_started_at = clock.now_iso()
    if not args.run_id:
        args.run_id = f"run-{clock.compact_timestamp()}"
    manifest_path = args.manifest_out or os.path.join(args.record_dir, args.run_id, "batch_manifest.json")

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

    seeds = []
    with open(args.seeds, "r") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                seeds.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {args.seeds} at line {lineno}: {exc}") from exc

    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "mode": args.mode,
        "base_seed": args.seed,
        "seed_schedule": "item_seed = base_seed + zero_based_index",
        "started_at": batch_started_at,
        "clock_now": batch_started_at,
        "seeds_path": args.seeds,
        "output_path": args.output,
        "attempts_path": args.attempts_out or f"artifacts/attempts/{args.run_id}.jsonl",
        "cassette_path": args.cassette_path or f"artifacts/cassettes/{args.run_id}.json",
        "checkpoint_dir": args.checkpoint_dir,
        "record_dir": args.record_dir,
        "items": [],
    }

    def write_manifest() -> None:
        save_checkpoint(manifest_path, manifest, clock_now=batch_started_at)
    
    print(f"Loaded {len(seeds)} seeds. {len(processed_predicates)} already processed.")
    print(f"Run ID: {args.run_id} | Base seed: {args.seed} | Mode: {args.mode}")
    write_manifest()
    
    for i, seed in enumerate(seeds):
        item_seed = args.seed + i
        instruction = f"Analyze this input for the condition: {seed['predicate']}"
        checkpoint_path = os.path.join(args.checkpoint_dir, args.run_id, f"{item_seed}.json")
        record_path = args.record_path or os.path.join(args.record_dir, args.run_id, f"{item_seed}.json")
        manifest_item = {
            "index": i,
            "seed": item_seed,
            "instruction": instruction,
            "predicate": seed["predicate"],
            "checkpoint_path": checkpoint_path,
            "record_path": record_path,
            "status": "pending",
        }
        manifest["items"].append(manifest_item)
        write_manifest()

        if instruction in processed_predicates:
            print(f"Skipping seed {i+1} (already processed).")
            manifest_item["status"] = "skipped_existing_output"
            write_manifest()
            continue

        print(f"\n>>> [{i+1}/{len(seeds)}] Seed Predicate: {seed.get('predicate')[:80]}...")
        manifest_item["status"] = "running"
        write_manifest()
        
        payload = {
            "participants": {
                "pro_debater": "http://127.0.0.1:9019/",
                "con_debater": "http://127.0.0.1:9018/"
            },
            "config": {
                "run_id": args.run_id,
                "seed": item_seed, # Increment seed per item for variety but stay deterministic
                "mode": args.mode,
                "resume": args.resume,
                "checkpoint_path": checkpoint_path,
                "clock_now": batch_started_at,
                **({"cassette_path": args.cassette_path} if args.cassette_path else {}),
                "record_path": record_path,
                **({"attempts_path": args.attempts_out} if args.attempts_out else {}),
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
            manifest_item["status"] = result.get("status") or "completed"
            manifest_item["response_excerpt"] = str(result.get("response", ""))[:500]
        except Exception as e:
            print(f"  ERROR: Seed {i+1} failed: {e}")
            manifest_item["status"] = "error"
            manifest_item["error"] = str(e)
        finally:
            write_manifest()

if __name__ == "__main__":
    asyncio.run(run_batch())
