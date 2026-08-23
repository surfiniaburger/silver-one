import json
import asyncio
import sys
import os
import hashlib
import argparse
from pathlib import Path

# Ensure project root is in sys.path before importing scenarios packages
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import scenarios.debate._thread_limits  # noqa: F401 (Enforce OpenMP thread limits on import)
sys.path.append(os.path.join(os.getcwd(), "src"))

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("run_batch")

from agentbeats.client import send_message
from agentbeats.checkpoint import save_checkpoint
from agentbeats.clock import RunClock
from scenarios.debate.pareto_registry import ParetoRegistry
from scenarios.debate.pre_filter import BarredPreFilter
from scenarios.debate.reflector_agent import ReflectorClient
from scenarios.debate.reflector_schemas import (
    TaxonomyBucket,
    classify_taxonomy_bucket,
    get_static_baseline_prompt,
)


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


def _load_seeds_with_hash(seeds_path: str) -> tuple[str, list]:
    """Read seed file bytes once, compute SHA-256 digest, and parse JSONL items."""
    with open(seeds_path, "rb") as f:
        raw_bytes = f.read()

    digest = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8")

    seeds = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            seeds.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {seeds_path} at line {lineno}: {exc}") from exc

    return digest, seeds


def _load_seeds(seeds_path: str) -> list:
    """Backward-compatible wrapper returning seeds list."""
    _, seeds = _load_seeds_with_hash(seeds_path)
    return seeds


def _compute_seeds_sha256(seeds_path: str) -> str:
    """Backward-compatible wrapper returning seeds SHA-256 digest."""
    digest, _ = _load_seeds_with_hash(seeds_path)
    return digest


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
    parser.add_argument("--reflector", action="store_true", default=True, help="Enable Graph-Powered GEPA Pareto Reflector.")
    parser.add_argument("--no-reflector", dest="reflector", action="store_false", help="Disable Graph-Powered GEPA Pareto Reflector.")
    parser.add_argument("--reflector-in-process", action="store_true", default=True, help="Execute ReflectorClient in-process.")
    parser.add_argument("--no-reflector-in-process", dest="reflector_in_process", action="store_false", help="Execute ReflectorClient over HTTP network.")
    parser.add_argument("--reflector-url", default="http://127.0.0.1:8004", help="Base URL for Reflector microservice.")
    parser.add_argument("--gepa-dir", default="artifacts/gepa", help="Directory for GEPA ledger, Pareto frontier, and traces.")
    return parser.parse_args(cmd_args)


def _build_payload(
    args: argparse.Namespace,
    item_seed: int,
    checkpoint_path: str,
    record_path: str,
    batch_started_at: str,
    seed: dict,
    reflector_client: ReflectorClient | None = None,
) -> dict:
    predicate = seed.get("predicate", "")
    taxonomy = classify_taxonomy_bucket(predicate)
    pareto_prompt = ""
    active_mutation_id = "baseline_v0"

    if reflector_client is not None and reflector_client.registry is not None:
        try:
            pareto_prompt = reflector_client.registry.get_pareto_prompt(taxonomy)
            active_mutation_id = reflector_client.registry.get_pareto_variant_id(taxonomy)
        except Exception:
            pareto_prompt = get_static_baseline_prompt(taxonomy)
            active_mutation_id = "baseline_v0"
    else:
        pareto_prompt = get_static_baseline_prompt(taxonomy)

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
            "taxonomy_bucket": taxonomy,
            "reflector_prompt": pareto_prompt,
            "active_mutation_id": active_mutation_id,
            "gepa_dir": getattr(args, "gepa_dir", "artifacts/gepa"),
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
    reflector_client: ReflectorClient | None = None
    pareto_registry: ParetoRegistry | None = None


import errno

_UNSUPPORTED_FSYNC_ERRNOS = {
    getattr(errno, name)
    for name in ("EINVAL", "ENOTSUP", "EBADF", "EOPNOTSUPP")
    if hasattr(errno, name)
}


def _append_attempt_record(attempts_path: str, record: dict) -> None:
    if not isinstance(record, dict) or not record:
        return
    dirname = os.path.dirname(attempts_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    line_data = json.dumps(record, ensure_ascii=False) + "\n"
    with open(attempts_path, "a", encoding="utf-8") as f:
        f.write(line_data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError as exc:
            if exc.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
                raise RuntimeError(f"Durable attempt log file sync failed for '{attempts_path}': {exc}") from exc


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
    input_code = seed.get("input_block") or seed.get("topic") or ""
    attempt_record = {
        "decision": "rejected",
        "pre_filter_stage": decision.stage,
        "pre_filter_probability": decision.probability,
        "skipped_pre_filter": True,
        "predicate": seed.get("predicate", ""),
        "topic": seed.get("topic", ""),
        "input_block": input_code,
        "cve_id": seed.get("cve_id"),
        "run_id": ctx.args.run_id,
        "seed": item_seed,
    }
    async with ctx.attempts_lock:
        await asyncio.to_thread(_append_attempt_record, attempts_path, attempt_record)
    await write_manifest_fn("skipped_pre_filter", response_excerpt=f"Rejected at {decision.stage} (p={decision.probability:.4f})")


async def _record_gepa_reflector_trace(
    ctx: BatchContext,
    seed: dict,
    item_seed: int,
    payload: dict,
    result: dict,
    status: str,
    i: int,
) -> None:
    if ctx.reflector_client is None:
        return

    predicate = seed.get("predicate", "")
    taxonomy = classify_taxonomy_bucket(predicate)
    is_valid = status == "completed" and result.get("decision") != "rejected"
    verifier_logic_error = bool(result.get("verifier_logic_error", False))
    active_mutation_id = payload["config"].get("active_mutation_id", "baseline_v0")
    prompt_used = payload["config"].get("reflector_prompt", "")

    try:
        await ctx.reflector_client.record_attempt_and_classify_outcome(
            taxonomy_bucket=taxonomy,
            predicate_family=seed.get("predicate_family") or seed.get("topic") or "GENERAL",
            seed_id=str(item_seed),
            scenario_id=str(seed.get("cve_id") or f"scenario_{item_seed}"),
            prompt=prompt_used,
            attempt_index=1,
            is_valid=is_valid,
            verifier_logic_error=verifier_logic_error,
            observed_at=ctx.batch_started_at,
            evaluated_at=ctx.batch_started_at,
            canonical_mutation_id=active_mutation_id,
            details={
                "run_id": ctx.args.run_id,
                "status": status,
                "decision": result.get("decision"),
                "reject_reason": result.get("reject_reason"),
            },
        )
    except Exception as exc:
        print(f"  Warning: Reflector trace recording failed for seed {i+1}: {exc}")


async def _should_skip_seed(
    ctx: BatchContext,
    seed: dict,
    item_seed: int,
    i: int,
    write_manifest_fn: Any,
) -> bool:
    instruction = f"Analyze this input for the condition: {seed['predicate']}"
    if instruction in ctx.processed_predicates:
        print(f"Skipping seed {i+1} (already processed).")
        await write_manifest_fn("skipped_existing_output")
        return True

    if ctx.pre_filter is not None:
        attempt_number = seed.get("attempt_number") or 1
        input_block = seed.get("input_block") or ""
        decision = ctx.pre_filter.predict(
            seed.get("predicate", ""),
            input_block=input_block,
            attempt_number=attempt_number,
        )
        if not decision.accept:
            await _handle_pre_filter_rejection(i, seed, item_seed, decision, ctx, write_manifest_fn)
            return True

    return False


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
        if await _should_skip_seed(ctx, seed, item_seed, i, write_manifest):
            return

        checkpoint_path = os.path.join(ctx.args.checkpoint_dir, ctx.args.run_id, f"{item_seed}.json")
        record_path = ctx.args.record_path or os.path.join(ctx.args.record_dir, ctx.args.run_id, f"{item_seed}.json")

        print(f"\n>>> [{i+1}/{ctx.total_seeds}] Seed Predicate: {seed.get('predicate')[:80]}...")
        await write_manifest("running")

        payload = _build_payload(
            ctx.args, item_seed, checkpoint_path, record_path, ctx.batch_started_at, seed, ctx.reflector_client
        )

        try:
            result = await send_message(json.dumps(payload), ctx.judge_url)
            print(f"  Result received for seed {i+1}. Status: {result.get('status')}")
            status = result.get("status") or "completed"
            excerpt = str(result.get("response", ""))[:500]
            await write_manifest(status, response_excerpt=excerpt)
            await _record_gepa_reflector_trace(ctx, seed, item_seed, payload, result, status, i)
        except Exception as e:
            print(f"  ERROR: Seed {i+1} failed: {e}")
            await write_manifest("error", error=str(e))


def _init_gepa_components(args: argparse.Namespace) -> tuple[Optional[ParetoRegistry], Optional[ReflectorClient]]:
    if not args.reflector:
        return None, None
    gepa_path = Path(args.gepa_dir)
    pareto_registry = ParetoRegistry(gepa_dir=gepa_path)
    reflector_client = ReflectorClient(
        registry=pareto_registry,
        base_url=args.reflector_url,
        in_process=args.reflector_in_process,
    )
    print(f"GEPA Graph-Powered Reflector enabled (in_process={args.reflector_in_process}, gepa_dir='{args.gepa_dir}').")
    return pareto_registry, reflector_client


def _build_batch_manifest(
    args: argparse.Namespace,
    seeds: list[dict],
    seeds_sha256: str,
    batch_started_at: str,
) -> dict:
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

    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "mode": args.mode,
        "base_seed": args.seed,
        "seed_schedule": "item_seed = base_seed + zero_based_index",
        "started_at": batch_started_at,
        "clock_now": batch_started_at,
        "seeds_path": args.seeds,
        "seeds_sha256": seeds_sha256,
        "output_path": args.output,
        "attempts_path": args.attempts_out or f"artifacts/attempts/{args.run_id}.jsonl",
        "cassette_path": args.cassette_path or f"artifacts/cassettes/{args.run_id}.json",
        "checkpoint_dir": args.checkpoint_dir,
        "record_dir": args.record_dir,
        "items": manifest_items,
    }


def _check_batch_failures(manifest: dict, mode: str) -> None:
    failed_items = [item for item in manifest["items"] if item.get("status") == "error"]
    if failed_items:
        print(f"\n[ERROR] {len(failed_items)} item(s) failed during batch execution.")
        if mode == "replay":
            raise RuntimeError(f"Replay mode failed for {len(failed_items)} seed(s). Batch execution aborted.")
        sys.exit(1)


async def run_batch():
    args = _parse_args()

    clock = RunClock.from_value(args.clock_now or os.getenv("RUN_CLOCK_NOW", ""))
    batch_started_at = clock.now_iso()
    if not args.run_id:
        args.run_id = f"run-{clock.compact_timestamp()}"
    manifest_path = args.manifest_out or os.path.join(args.record_dir, args.run_id, "batch_manifest.json")

    if not os.path.exists(args.seeds):
        print(f"Error: {args.seeds} not found.")
        return

    processed_predicates = await asyncio.to_thread(_load_processed_predicates, args.output)
    seeds_sha256, seeds = await asyncio.to_thread(_load_seeds_with_hash, args.seeds)

    pre_filter = BarredPreFilter(model_dir=Path(args.model_dir)) if args.pre_filter else None
    if pre_filter:
        print(f"BARRED 3-Stage Pre-Filter enabled (models loaded from '{args.model_dir}').")

    pareto_registry, reflector_client = _init_gepa_components(args)
    manifest = _build_batch_manifest(args, seeds, seeds_sha256, batch_started_at)
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
        judge_url="http://127.0.0.1:9009",
        pre_filter=pre_filter,
        reflector_client=reflector_client,
        pareto_registry=pareto_registry,
    )

    await asyncio.gather(*(_process_seed(sem, i, s, ctx) for i, s in enumerate(seeds)))

    if pareto_registry is not None:
        try:
            pareto_registry.sync_pareto_frontier()
            print(f"GEPA Pareto frontier synced to '{args.gepa_dir}/pareto_frontier.json'.")
        except Exception as e:
            logger.warning("Could not sync Pareto frontier: %s", e)

    _check_batch_failures(manifest, args.mode)


if __name__ == "__main__":
    asyncio.run(run_batch())
