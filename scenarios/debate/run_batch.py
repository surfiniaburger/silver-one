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


def _load_processed_predicates(output_path: str) -> set:
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data.get("instruction", ""))
                except (TypeError, ValueError):
                    pass
    return processed


def _load_seeds(seeds_path: str) -> list:
    seeds = []
    with open(seeds_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                seeds.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {seeds_path} at line {lineno}: {exc}") from exc
    return seeds


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--max-concurrency", type=int, default=1, help="Max concurrent seed executions (default: 1)")
    return parser.parse_args()


def _build_payload(
    args: argparse.Namespace,
    item_seed: int,
    checkpoint_path: str,
    record_path: str,
    batch_started_at: str,
    seed: dict,
) -> dict:
    return {
        "participants": {
            "pro_debater": "http://127.0.0.1:9019/",
            "con_debater": "http://127.0.0.1:9018/",
        },
        "config": {
            "run_id": args.run_id,
            "seed": item_seed,
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
            "output_file": args.output,
        },
    }


async def _process_seed(
    sem: asyncio.Semaphore,
    manifest_lock: asyncio.Lock,
    i: int,
    seed: dict,
    args: argparse.Namespace,
    batch_started_at: str,
    processed_predicates: set,
    manifest: dict,
    manifest_path: str,
    total_seeds: int,
    judge_url: str = "http://127.0.0.1:9009",
) -> None:
    async def write_manifest(status: str, response_excerpt: str | None = None, error: str | None = None) -> None:
        async with manifest_lock:
            manifest_item = manifest["items"][i]
            manifest_item["status"] = status
            if response_excerpt is not None:
                manifest_item["response_excerpt"] = response_excerpt
            if error is not None:
                manifest_item["error"] = error
            await asyncio.to_thread(save_checkpoint, manifest_path, manifest, clock_now=batch_started_at)

    async with sem:
        item_seed = args.seed + i
        instruction = f"Analyze this input for the condition: {seed['predicate']}"
        checkpoint_path = os.path.join(args.checkpoint_dir, args.run_id, f"{item_seed}.json")
        record_path = args.record_path or os.path.join(args.record_dir, args.run_id, f"{item_seed}.json")

        if instruction in processed_predicates:
            print(f"Skipping seed {i+1} (already processed).")
            await write_manifest("skipped_existing_output")
            return

        print(f"\n>>> [{i+1}/{total_seeds}] Seed Predicate: {seed.get('predicate')[:80]}...")
        await write_manifest("running")

        payload = _build_payload(args, item_seed, checkpoint_path, record_path, batch_started_at, seed)

        try:
            result = await send_message(json.dumps(payload), judge_url)
            print(f"  Result received for seed {i+1}. Status: {result.get('status')}")
            status = result.get("status") or "completed"
            excerpt = str(result.get("response", ""))[:500]
            await write_manifest(status, response_excerpt=excerpt)
        except Exception as e:
            print(f"  ERROR: Seed {i+1} failed: {e}")
            await write_manifest("error", error=str(e))


async def run_batch():
    args = _parse_args()

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
    processed_predicates = await asyncio.to_thread(_load_processed_predicates, args.output)
    seeds = await asyncio.to_thread(_load_seeds, args.seeds)

    manifest_items = [
        {
            "index": i,
            "seed": args.seed + i,
            "instruction": f"Analyze this input for the condition: {s['predicate']}",
            "predicate": s["predicate"],
            "checkpoint_path": os.path.join(args.checkpoint_dir, args.run_id, f"{args.seed + i}.json"),
            "record_path": args.record_path or os.path.join(args.record_dir, args.run_id, f"{args.seed + i}.json"),
            "status": "pending",
        }
        for i, s in enumerate(seeds)
    ]

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
        "items": manifest_items,
    }

    save_checkpoint(manifest_path, manifest, clock_now=batch_started_at)
    
    print(f"Loaded {len(seeds)} seeds. {len(processed_predicates)} already processed.")
    print(f"Run ID: {args.run_id} | Base seed: {args.seed} | Mode: {args.mode} | Concurrency: {args.max_concurrency}")
    
    sem = asyncio.Semaphore(args.max_concurrency)
    manifest_lock = asyncio.Lock()

    await asyncio.gather(
        *(
            _process_seed(
                sem,
                manifest_lock,
                i,
                s,
                args,
                batch_started_at,
                processed_predicates,
                manifest,
                manifest_path,
                len(seeds),
                judge_url,
            )
            for i, s in enumerate(seeds)
        )
    )

if __name__ == "__main__":
    asyncio.run(run_batch())
