"""Unit & Integration Tests for Leak-Proof Pre-Filter Training Pipeline.

Verifies:
1. Strict CVE ID grouped data partitioning (zero cross-split prompt bleeding).
2. Isolated vectorization (vocabulary built strictly from X_train).
3. Class-balanced null-model sanity check (shuffled label performance <= 0.55).
4. Zero target-derived feature leakage in text extraction.
5. Independent holdout benchmark metrics export.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.train_pre_filter import (
    _parse_attempt_record,
    partition_dataset_by_cve,
    _train_stage_b_vectorizer,
    _run_null_model_sanity_check,
    _evaluate_holdout_performance,
    FallbackClassifier,
)


def test_cve_grouping_isolation():
    """Verify all attempt records sharing a CVE ID reside strictly in the same split."""
    texts = [
        "Predicate: vuln A | Code: code A1",
        "Predicate: vuln A | Code: code A2",
        "Predicate: vuln B | Code: code B1",
        "Predicate: vuln C | Code: code C1",
        "Predicate: vuln D | Code: code D1",
    ]
    labels = [1, 1, 0, 1, 0]
    cve_ids = ["CVE-2024-0001", "CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003", "CVE-2024-0004"]

    splits = partition_dataset_by_cve(texts, labels, cve_ids, train_ratio=0.6, val_ratio=0.2)
    train_texts, _ = splits["train"]
    val_texts, _ = splits["val"]
    test_texts, _ = splits["test"]

    # CVE-2024-0001 must not appear in both train and test/val
    train_cve_1 = any("vuln A" in t for t in train_texts)
    val_cve_1 = any("vuln A" in t for t in val_texts)
    test_cve_1 = any("vuln A" in t for t in test_texts)

    # Exactly one partition must hold vuln A
    assert sum([train_cve_1, val_cve_1, test_cve_1]) == 1


def test_vectorizer_vocabulary_isolation(tmp_path: Path):
    """Verify TF-IDF vectorizer vocabulary is fit strictly on training texts."""
    train_texts = ["vulnerable buffer overflow in parse_string"]
    val_texts = ["integer overflow in malloc"]
    test_texts = ["use after free in close"]

    vectorizer, _, _, _ = _train_stage_b_vectorizer(train_texts, val_texts, test_texts, tmp_path)

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

    text, label, cve_id = _parse_attempt_record(raw_record)
    assert text is not None
    assert label == 1

    # Verify no target/outcome metadata leaked into text feature
    assert "SUCCESS_VERIFIED" not in text
    assert "check_1_pass" not in text
    assert "agent_green" not in text
    assert text.startswith("Predicate: vulnerable function check")


def test_holdout_metrics_export(tmp_path: Path):
    """Verify _evaluate_holdout_performance creates holdout_metrics.json with required keys."""
    classifier = FallbackClassifier()
    x_train = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    y_train = np.array([1, 1, 0, 0])
    classifier.fit(x_train, y_train)

    x_test = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)
    y_test = np.array([1, 0])

    metrics = _evaluate_holdout_performance(classifier, x_test, y_test, tmp_path)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0

    metrics_file = tmp_path / "holdout_metrics.json"
    assert metrics_file.exists()
    with metrics_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["test_samples"] == 2
    assert data["accuracy"] == 1.0
