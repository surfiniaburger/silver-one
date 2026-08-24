"""
Unit tests for Step 4 Acceptance Audit & 5-Fold Cross Validation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.evaluate_step4_acceptance import (
    audit_attempt_records,
    audit_leak_proof_partitions,
    run_full_acceptance_audit,
)
from scripts.evaluate_graphify_cv import EvaluationDataset


def test_audit_attempt_records_empty_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = audit_attempt_records(Path(tmp_dir))
        assert res["total_attempts"] == 0
        assert res["accepted_logic_error_rate"] == 0.0
        assert res["b2_anchor_match_rate"] == 0.0
        assert res["verifier_parse_ok_rate"] == 0.0
        assert res["token_reduction_pct"] == 0.0
        assert res["refinement_correction_success_rate"] == 0.0


def test_audit_attempt_records_with_mock_data():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        attempt_file = tmp_path / "pilot-v6-warmer.jsonl"
        records = [
            {
                "run_id": "pilot-v6-warmer",
                "seed": 42,
                "decision": "accepted",
                "refinement_round": 1,
                "sample_sha256": "sha_1",
                "anchor_stats": {"total": 2, "matched": 2},
                "verifier": {"called": True, "parse_ok": True, "logic_error": None},
                "llm_usage": {"totals": {"total_tokens": 10000}},
            },
            {
                "run_id": "pilot-v6-warmer",
                "seed": 42,
                "decision": "rejected",
                "refinement_round": 0,
                "sample_sha256": "sha_2",
                "anchor_stats": {"total": 2, "matched": 1},
                "verifier": {"called": False, "parse_ok": False, "logic_error": None},
                "llm_usage": {"totals": {"total_tokens": 5000}},
            },
        ]
        with open(attempt_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        res = audit_attempt_records(tmp_path)
        assert res["total_attempts"] == 2
        assert res["accepted_attempts"] == 1
        assert res["accepted_logic_error_rate"] == 0.0
        assert res["b2_anchor_match_rate"] == 0.75
        assert res["verifier_parse_ok_rate"] == 1.0


def test_audit_leak_proof_partitions():
    dataset = EvaluationDataset(
        texts=["sample code 1", "sample code 2", "sample code 3", "sample code 4", "sample code 5"],
        labels=[1, 0, 1, 0, 1],
        scenario_ids=["scen_1", "scen_2", "scen_3", "scen_4", "scen_5"],
    )
    res = audit_leak_proof_partitions(dataset, seeds=[42, 43])
    assert res["all_leak_free"] is True
    assert res["total_folds_audited"] == 10
    assert len(res["details"]) == 10


def test_audit_leak_proof_partitions_detects_scenario_leak_with_distinct_texts():
    dataset = EvaluationDataset(
        texts=["text A", "text B", "text C", "text D", "text E"],
        labels=[1, 0, 1, 0, 1],
        scenario_ids=["scen_1", "scen_2", "scen_3", "scen_4", "scen_5"],
    )
    # Mock partitioner to return a fold with leaked scenario_id but completely different text
    leaked_folds = [
        {
            "train_texts": ["completely different text 1"],
            "test_texts": ["completely different text 2"],
            "train_scenario_ids": ["scen_LEAK", "scen_OTHER"],
            "test_scenario_ids": ["scen_LEAK", "scen_ANOTHER"],
        }
        for _ in range(5)
    ]
    with patch("scripts.evaluate_step4_acceptance.partition_dataset_by_scenario_stratified", return_value=leaked_folds):
        res = audit_leak_proof_partitions(dataset, seeds=[42])
        assert res["all_leak_free"] is False
        assert res["details"][0]["overlap_count"] == 1
        assert res["details"][0]["is_leak_free"] is False


def test_audit_attempt_records_missing_sha256_fails_identity_coverage():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        attempt_file = tmp_path / "pilot-v6-warmer.jsonl"
        records = [
            {
                "run_id": "pilot-v6-warmer",
                "seed": 42,
                "decision": "accepted",
                # sample_sha256 missing
                "anchor_stats": {"total": 2, "matched": 2},
                "verifier": {"called": True, "parse_ok": True, "logic_error": None},
                "llm_usage": {"totals": {"total_tokens": 10000}},
            },
            {
                "run_id": "pilot-v6-warmer",
                "seed": 43,
                "decision": "accepted",
                "sample_sha256": "sha_valid_1",
                "anchor_stats": {"total": 2, "matched": 2},
                "verifier": {"called": True, "parse_ok": True, "logic_error": None},
                "llm_usage": {"totals": {"total_tokens": 10000}},
            },
        ]
        with open(attempt_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        res = audit_attempt_records(tmp_path)
        assert res["accepted_attempts"] == 2
        assert res["accepted_sha256_coverage"] == 0.50
        # Duplicate rate among identified rows is 0.0, but coverage is 50%
        assert res["duplicate_valid_accept_rate"] == 0.0
