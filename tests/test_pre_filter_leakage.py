"""Unit & Integration Tests for Leak-Proof Pre-Filter Training Pipeline.

Verifies:
1. Strict Scenario Grouped data partitioning (zero cross-split prompt bleeding).
2. Isolated vectorization (vocabulary built strictly from X_train).
3. Class-balanced null-model sanity check (shuffled label performance <= 0.55).
4. Zero target-derived feature leakage in text extraction.
5. Independent evaluation metrics calculation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scenarios.debate.pre_filter import (
    _extract_scenario_id,
    _parse_attempt_record,
    partition_dataset_by_scenario_stratified,
)
from scripts.train_pre_filter import (
    _train_stage_b_vectorizer,
    _run_null_model_sanity_check,
    _evaluate_stage_b,
    FallbackClassifier,
)


def test_cve_grouping_isolation():
    """Verify all attempt records sharing a scenario ID reside strictly in the same split."""
    texts = [
        "Predicate: vuln A | Code: code A1",
        "Predicate: vuln A | Code: code A2",
        "Predicate: vuln B | Code: code B1",
        "Predicate: vuln C | Code: code C1",
        "Predicate: vuln D | Code: code D1",
    ]
    labels = [1, 1, 0, 1, 0]
    scenario_ids = ["HASH-0001", "HASH-0001", "HASH-0002", "HASH-0003", "HASH-0004"]

    folds = partition_dataset_by_scenario_stratified(texts, labels, scenario_ids, n_splits=3)
    assert len(folds) == 3

    # Verify HASH-0001 items are never split across train and test in any fold
    for fold in folds:
        train_scenarios = set(fold["train_scenario_ids"])
        test_scenarios = set(fold["test_scenario_ids"])
        assert train_scenarios.isdisjoint(test_scenarios), "Scenario ID leaked between train and test splits!"


def test_vectorizer_vocabulary_isolation(tmp_path: Path):
    """Verify TF-IDF vectorizer vocabulary is fit strictly on training texts."""
    train_texts = ["vulnerable buffer overflow in parse_string"]
    test_texts = ["integer overflow in malloc"]

    vectorizer, _scaler, _, _ = _train_stage_b_vectorizer(train_texts, test_texts, output_dir=None)

    # Vocabulary keys should come strictly from train_texts
    train_ngrams = set()
    padded = f" {train_texts[0].lower()} "
    for n in range(3, 6):
        for i in range(len(padded) - n + 1):
            train_ngrams.add(padded[i : i + n])

    for k in vectorizer.vocabulary_:
        assert k in train_ngrams, f"Vocabulary term '{k}' did not originate from X_train!"


def test_null_model_sanity_check():
    """Verify control model trained on randomly shuffled labels achieves near-random performance."""
    rng = np.random.default_rng(42)
    x_train = rng.standard_normal((50, 10), dtype=np.float32)
    y_train = np.array([1] * 25 + [0] * 25)

    x_test = rng.standard_normal((20, 10), dtype=np.float32)
    y_test = np.array([1] * 10 + [0] * 10)

    balanced_acc = _run_null_model_sanity_check(x_train, y_train, x_test, y_test)
    assert balanced_acc <= 0.55, f"Null model balanced accuracy {balanced_acc} was unexpectedly high!"


def test_no_target_derived_feature_leakage():
    """Verify attempt record parser excludes adjudication and outcome fields."""
    raw_record = {
        "decision": "accepted",
        "predicate": "vulnerable function check",
        "input_block": "int main() { return 0; }",
        "verifier_status": "SUCCESS_VERIFIED",
        "soft_checks": ["check_1_pass", "check_2_pass"],
        "winner": "agent_green",
    }

    text, label, scenario_id = _parse_attempt_record(raw_record)
    assert text is not None
    assert label == 1

    # Verify no target/outcome metadata leaked into text feature
    assert "SUCCESS_VERIFIED" not in text
    assert "check_1_pass" not in text
    assert "agent_green" not in text
    assert text.startswith("Predicate: vulnerable function check")


def test_holdout_metrics_export(tmp_path: Path):
    """Verify _evaluate_stage_b creates evaluation metrics dict with required keys."""
    classifier = FallbackClassifier()
    x_train = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    y_train = np.array([1, 1, 0, 0])
    classifier.fit(x_train, y_train)

    x_test = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)
    y_test = np.array([1, 0])

    metrics = _evaluate_stage_b(classifier, x_test, y_test)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_scenario_id_resolution_tiers():
    """Verify 3-tier scenario identifier resolution (cve_id -> CVE regex -> SHA-256 fallback)."""
    # Tier 1: Explicit cve_id field (uppercase canonicalization)
    rec_cve = {"cve_id": "cve-2024-1234"}
    assert _extract_scenario_id(rec_cve, "any predicate") == "CVE-2024-1234"

    # Tier 1b: seed_dict.cve_id with lowercase & whitespace
    rec_seed_cve = {"seed": {"cve_id": "  cve-2025-5678  "}}
    assert _extract_scenario_id(rec_seed_cve, "any predicate") == "CVE-2025-5678"

    # Tier 2: Regex extraction from predicate
    rec_regex = {}
    assert _extract_scenario_id(rec_regex, "vulnerability in cve-2023-9999 parse") == "CVE-2023-9999"

    # Tier 3: SHA-256 hash fallback
    import hashlib
    rec_hash = {}
    pred_text = "custom vulnerability predicate without cve"
    expected_hash = f"HASH-{hashlib.sha256(pred_text.encode('utf-8')).hexdigest()[:10]}"
    scen_id = _extract_scenario_id(rec_hash, pred_text)
    assert scen_id == expected_hash
    assert scen_id.startswith("HASH-")
    assert len(scen_id) == 15  # "HASH-" + 10 chars


def test_partition_dataset_length_mismatch_validation():
    """Verify partition_dataset_by_scenario_stratified rejects unaligned input lengths."""
    import pytest
    texts = ["text 1", "text 2"]
    labels = [1, 0]
    scenario_ids = ["HASH-1"]  # Mismatched length

    with pytest.raises(ValueError, match="Input lists must have equal lengths"):
        partition_dataset_by_scenario_stratified(texts, labels, scenario_ids)

