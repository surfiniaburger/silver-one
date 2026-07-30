import json
import asyncio
import sys
import os
import argparse

# Add src and scenarios/debate to PYTHONPATH
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.path.join(os.getcwd(), "scenarios", "debate"))

from pathlib import Path
from typing import Any

from agentbeats.client import send_message
from agentbeats.checkpoint import save_checkpoint
from agentbeats.clock import RunClock

try:
    from scenarios.debate.pre_filter import BarredPreFilter
except ModuleNotFoundError:
    from pre_filter import BarredPreFilter


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


def _parse_args(cmd_args: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--pre-filter", action="store_true", default=True, help="Enable BARRED 3-Stage Pre-Filter Cascade.")
    parser.add_argument("--no-pre-filter", dest="pre_filter", action="store_false", help="Disable BARRED 3-Stage Pre-Filter Cascade.")
    parser.add_argument("--model-dir", default="artifacts/models", help="Directory containing pre-filter model weights.")
    return parser.parse_args(cmd_args)


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



from dataclasses import dataclass, field


@dataclass
class BatchContext:
    args: argparse.Namespace
    batch_started_at: str
    processed_predicates: set
    manifest: dict
    manifest_path: str
    manifest_lock: asyncio.Lock
    total_seeds: int
    attempts_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    judge_url: str = "http://127.0.0.1:9009"
    pre_filter: BarredPreFilter | None = None


def _append_attempt_record(attempts_path: str, record: dict) -> None:
    dirname = os.path.dirname(attempts_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(attempts_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


async def _handle_pre_filter_rejection(
    i: int,
    seed: dict,
    item_seed: int,
    decision: Any,
    ctx: BatchContext,
    write_manifest_fn: Any,
) -> None:
    print(f"  Skipping seed {i+1} (rejected by BARRED pre-filter at {decision.stage}, prob={decision.probability:.4f}).")
    attempts_path = ctx.args.attempts_out or f"artifacts/attempts/{ctx.args.run_id}.jsonl"
    attempt_record = {
        "decision": "rejected",
        "pre_filter_stage": decision.stage,
        "pre_filter_probability": decision.probability,
        "skipped_pre_filter": True,
        "predicate": seed.get("predicate", ""),
        "cve_id": seed.get("cve_id"),
        "run_id": ctx.args.run_id,
        "seed": item_seed,
    }
    async with ctx.attempts_lock:
        await asyncio.to_thread(_append_attempt_record, attempts_path, attempt_record)
    await write_manifest_fn("skipped_pre_filter", response_excerpt=f"Rejected at {decision.stage} (p={decision.probability:.4f})")


async def _process_seed(
    sem: asyncio.Semaphore,
    i: int,
    seed: dict,
    ctx: BatchContext,
) -> None:
    async def write_manifest(status: str, response_excerpt: str | None = None, error: str | None = None) -> None:
        async with ctx.manifest_lock:
            manifest_item = ctx.manifest["items"][i]
            manifest_item["status"] = status
            if response_excerpt is not None:
                manifest_item["response_excerpt"] = response_excerpt
            if error is not None:
                manifest_item["error"] = error
            await asyncio.to_thread(save_checkpoint, ctx.manifest_path, ctx.manifest, clock_now=ctx.batch_started_at)

    async with sem:
        item_seed = ctx.args.seed + i
        instruction = f"Analyze this input for the condition: {seed['predicate']}"
        checkpoint_path = os.path.join(ctx.args.checkpoint_dir, ctx.args.run_id, f"{item_seed}.json")
        record_path = ctx.args.record_path or os.path.join(ctx.args.record_dir, ctx.args.run_id, f"{item_seed}.json")

        if instruction in ctx.processed_predicates:
            print(f"Skipping seed {i+1} (already processed).")
            await write_manifest("skipped_existing_output")
            return

        if ctx.pre_filter is not None:
            input_block = seed.get("topic") or seed.get("input_block") or ""
            decision = ctx.pre_filter.predict(seed.get("predicate", ""), input_block)
            if not decision.accept:
                await _handle_pre_filter_rejection(i, seed, item_seed, decision, ctx, write_manifest)
                return

        print(f"\n>>> [{i+1}/{ctx.total_seeds}] Seed Predicate: {seed.get('predicate')[:80]}...")
        await write_manifest("running")

        payload = _build_payload(ctx.args, item_seed, checkpoint_path, record_path, ctx.batch_started_at, seed)

        try:
            result = await send_message(json.dumps(payload), ctx.judge_url)
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

    pre_filter = BarredPreFilter(model_dir=Path(args.model_dir)) if args.pre_filter else None
    if pre_filter:
        print(f"BARRED 3-Stage Pre-Filter enabled (models loaded from '{args.model_dir}').")

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

    ctx = BatchContext(
        args=args,
        batch_started_at=batch_started_at,
        processed_predicates=processed_predicates,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_lock=manifest_lock,
        total_seeds=len(seeds),
        judge_url=judge_url,
        pre_filter=pre_filter,
    )

    await asyncio.gather(
        *(
            _process_seed(
                sem,
                i,
                s,
                ctx,
            )
            for i, s in enumerate(seeds)
        )
    )

if __name__ == "__main__":
    asyncio.run(run_batch())
