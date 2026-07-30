import asyncio
import json
import tempfile
import os
import pytest
from argparse import Namespace
from scenarios.debate.run_batch import _process_seed
from agentbeats.checkpoint import load_checkpoint

@pytest.mark.asyncio
async def test_thread_offloaded_manifest_checkpointing():
    """Verify that _process_seed offloads manifest writes to a thread safely under concurrency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "batch_manifest.json")
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        record_dir = os.path.join(tmpdir, "records")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(record_dir, exist_ok=True)

        seeds = [{"predicate": f"Predicate {i}", "topic": f"Topic {i}"} for i in range(5)]
        manifest = {
            "schema_version": 1,
            "run_id": "test-run",
            "mode": "record",
            "items": [
                {
                    "index": i,
                    "seed": 100 + i,
                    "instruction": f"Analyze this input for the condition: Predicate {i}",
                    "predicate": f"Predicate {i}",
                    "checkpoint_path": os.path.join(checkpoint_dir, f"{100+i}.json"),
                    "record_path": os.path.join(record_dir, f"{100+i}.json"),
                    "status": "pending",
                }
                for i in range(5)
            ],
        }

        args = Namespace(
            run_id="test-run",
            seed=100,
            mode="record",
            resume=False,
            seeds="dummy.jsonl",
            output="dummy_out.jsonl",
            attempts_out=None,
            cassette_path=None,
            checkpoint_dir=checkpoint_dir,
            record_dir=record_dir,
            record_path=None,
            max_concurrency=4,
        )

        sem = asyncio.Semaphore(4)
        manifest_lock = asyncio.Lock()
        processed_predicates = {"Analyze this input for the condition: Predicate 0"}

        from scenarios.debate.run_batch import BatchContext

        ctx = BatchContext(
            args=args,
            batch_started_at="2026-07-26T00:00:00Z",
            processed_predicates=processed_predicates,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_lock=manifest_lock,
            total_seeds=5,
            judge_url="http://127.0.0.1:99999",
            pre_filter=None,
        )

        # Process all 5 seeds concurrently (seed 0 will skip, others will hit error)
        tasks = [
            _process_seed(
                sem=sem,
                i=i,
                seed=seeds[i],
                ctx=ctx,
            )
            for i in range(5)
        ]

        await asyncio.gather(*tasks)

        # Load saved checkpoint
        saved_manifest = load_checkpoint(manifest_path)
        assert saved_manifest is not None
        assert len(saved_manifest["items"]) == 5
        assert saved_manifest["items"][0]["status"] == "skipped_existing_output"
        for item in saved_manifest["items"][1:]:
            assert item["status"] == "error"
            assert "error" in item
