import json
import logging
from pathlib import Path
import pytest
import joblib

from scripts.train_pre_filter import extract_dataset_from_attempts, train_pre_filter
from scenarios.debate.pre_filter import BarredPreFilter


def test_extract_dataset_from_attempts(tmp_path):
    attempts_dir = tmp_path / "attempts"
    attempts_dir.mkdir()

    # Create mock attempt jsonl file
    log_file = attempts_dir / "test_attempts.jsonl"
    records = [
        {
            "decision": "accepted",
            "predicate": "The code is vulnerable to a buffer overflow in parse_string",
            "anchors_normalized": ["strcpy(buf, s);"],
        },
        {
            "decision": "rejected",
            "predicate": "buy followers instantly click here for miracle offer",
            "input_block": "print('click link')",
        },
        {
            "decision": "skipped_pre_filter",
            "predicate": "ignored pre-filter record",
        },
        {
            "decision": "accepted",
            "judge_eval": {
                "predicate": "The code is vulnerable to integer overflow in malloc",
                "anchors": ["malloc(n * sizeof(int));"],
            },
        },
        {
            "decision": "rejected",
            "predicate": "regular non-vulnerable math function",
            "input_block": "int x = a + b;",
        },
        "invalid json line string",
    ]

    with log_file.open("w", encoding="utf-8") as f:
        for rec in records:
            if isinstance(rec, str):
                f.write(rec + "\n")
            else:
                f.write(json.dumps(rec) + "\n")

    texts, labels, cve_ids = extract_dataset_from_attempts(attempts_dir)

    assert len(texts) == 4
    assert len(labels) == 4
    assert len(cve_ids) == 4
    assert labels == [1, 0, 1, 0]
    assert "buffer overflow" in texts[0]
    assert "miracle offer" in texts[1]
    assert "integer overflow" in texts[2]


def test_extract_dataset_synthetic_fallback(tmp_path):
    empty_dir = tmp_path / "empty_attempts"
    texts, labels, cve_ids = extract_dataset_from_attempts(empty_dir)

    assert len(texts) >= 4
    assert len(labels) >= 4
    assert len(cve_ids) >= 4
    assert set(labels) == {0, 1}


def test_train_pre_filter_and_artifact_persistence(tmp_path):
    attempts_dir = tmp_path / "attempts"
    attempts_dir.mkdir()

    # Add mock attempt samples
    log_file = attempts_dir / "sample_attempts.jsonl"
    records = [
        {"decision": "accepted", "predicate": "The code is vulnerable to buffer overflow in memcpy", "input_block": "memcpy(dest, src, len);"},
        {"decision": "accepted", "predicate": "The code is vulnerable to use-after-free in cleanup", "input_block": "free(ptr); ptr->val = 1;"},
        {"decision": "rejected", "predicate": "buy followers instantly click here", "input_block": "return 0;"},
        {"decision": "rejected", "predicate": "harmless calculation without issues", "input_block": "int add(int a, int b) { return a + b; }"},
    ]
    with log_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    models_dir = tmp_path / "models"
    success = train_pre_filter(attempts_dir, models_dir, train_setfit=False)

    assert success is True
    vec_path = models_dir / "vectorizer.joblib"
    xgb_path = models_dir / "xgb.joblib"

    assert vec_path.exists()
    assert xgb_path.exists()

    # Verify model binaries can be loaded and executed
    vectorizer = joblib.load(vec_path)
    classifier = joblib.load(xgb_path)

    features = vectorizer.transform(["Predicate: Test predicate | Code: void test() { }"])
    probs = classifier.predict_proba(features)

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0][1] <= 1.0


def test_pre_filter_integration_with_trained_models(tmp_path):
    attempts_dir = tmp_path / "attempts"
    attempts_dir.mkdir()

    log_file = attempts_dir / "sample.jsonl"
    records = [
        {"decision": "accepted", "predicate": "Vulnerable to heap buffer overflow in parse_json", "input_block": "char *buf = malloc(10);"},
        {"decision": "accepted", "predicate": "Vulnerable to stack corruption in parse_path", "input_block": "char path[32]; strcpy(path, input);"},
        {"decision": "rejected", "predicate": "buy miracle followers click here now", "input_block": "print('click link')"},
        {"decision": "rejected", "predicate": "regular math addition function", "input_block": "int x = 1 + 2;"},
    ]
    with log_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    models_dir = tmp_path / "models"
    train_pre_filter(attempts_dir, models_dir, train_setfit=False)

    # Initialize BarredPreFilter pointing to trained binaries
    pre_filter = BarredPreFilter(
        vectorizer_path=str(models_dir / "vectorizer.joblib"),
        xgb_path=str(models_dir / "xgb.joblib"),
        setfit_dir=str(models_dir / "non_existent_setfit"),
        xgb_high_threshold=0.80,
        xgb_low_threshold=0.20,
    )

    assert pre_filter.vectorizer is not None
    assert pre_filter.xgb is not None

    decision = pre_filter.predict("Vulnerable to heap buffer overflow in parse_json", input_block="char *buf = malloc(10);")
    # Should evaluate via Stage A or Stage B
    assert decision.stage in ("heuristic", "xgboost")
