import json
import pytest
from pathlib import Path
from scripts.hub_common import safe_resolve, compute_sha256, log_provenance
from scripts.upload_to_huggingface import validate_seed_corpus


def _make_valid_record(topic_len: int = 550) -> dict:
    code = "int process_security_context(struct context *ctx, char *payload, int len) {\n"
    code += "    // Buffer padding to reach realistic minimum code bounds\n"
    while len(code) < topic_len:
        code += "    if (!payload || len <= 0) { return -1; }\n"
    code += "    return 0;\n}"
    return {
        "topic": code,
        "predicate": "The code is not vulnerable to null pointer dereference in process_security_context",
        "gepa_info": {
            "predicate": "The code is not vulnerable to null pointer dereference in process_security_context",
            "evidence_hooks": ["if (!payload || len <= 0)"],
            "uncertainty": "Low",
            "proof_requirements": "Demonstrate payload == NULL bypass reaching dereference"
        },
        "language": "c",
        "original_safety": "safe",
        "anchors": ["process_security_context", "payload", "len", "context"]
    }


def test_safe_resolve_valid_path(tmp_path: Path):
    safe_target = safe_resolve("scenarios/debate/seeds.jsonl", tmp_path)
    assert safe_target == (tmp_path / "scenarios/debate/seeds.jsonl").resolve()


def test_safe_resolve_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="escapes base directory"):
        safe_resolve("../../etc/passwd", tmp_path)


def test_validate_seed_corpus_valid(tmp_path: Path):
    test_file = tmp_path / "valid_seeds.jsonl"
    records = [_make_valid_record(550) for _ in range(500)]
    with open(test_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_count=500)
    assert is_valid is True
    assert reason == "Valid"
    assert count == 500


def test_validate_seed_corpus_boundary_counts(tmp_path: Path):
    # Test 499 records (must reject when expecting 500)
    file_499 = tmp_path / "seeds_499.jsonl"
    with open(file_499, "w", encoding="utf-8") as f:
        for _ in range(499):
            f.write(json.dumps(_make_valid_record(550)) + "\n")

    is_valid, reason, count = validate_seed_corpus(file_499, expected_count=500)
    assert is_valid is False
    assert "Corpus record count (499) does not match expected count (500)" in reason

    # Test 501 records (must reject when expecting 500)
    file_501 = tmp_path / "seeds_501.jsonl"
    with open(file_501, "w", encoding="utf-8") as f:
        for _ in range(501):
            f.write(json.dumps(_make_valid_record(550)) + "\n")

    is_valid, reason, count = validate_seed_corpus(file_501, expected_count=500)
    assert is_valid is False
    assert "Corpus record count (501) does not match expected count (500)" in reason


def test_validate_seed_corpus_rejects_non_dict_json(tmp_path: Path):
    test_file = tmp_path / "non_dict.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(["not", "an", "object"]) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_count=1)
    assert is_valid is False
    assert "is not a JSON object" in reason


def test_validate_seed_corpus_rejects_missing_keys(tmp_path: Path):
    test_file = tmp_path / "missing_keys.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"topic": "int a = 0;", "predicate": "vulnerable"}) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_count=1)
    assert is_valid is False
    assert "missing required keys" in reason


def test_validate_seed_corpus_rejects_incomplete_gepa(tmp_path: Path):
    test_file = tmp_path / "bad_gepa.jsonl"
    record = _make_valid_record(550)
    # Remove proof_requirements from gepa_info
    del record["gepa_info"]["proof_requirements"]
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_count=1)
    assert is_valid is False
    assert "gepa_info missing required subfields" in reason


def test_validate_seed_corpus_rejects_short_topic(tmp_path: Path):
    test_file = tmp_path / "short_topic.jsonl"
    record = _make_valid_record(100) # Short topic < 200 chars
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_count=1)
    assert is_valid is False
    assert "invalid topic length" in reason


def test_validate_seed_corpus_rejects_long_topic(tmp_path: Path):
    test_file = tmp_path / "long_topic.jsonl"
    record = _make_valid_record(13000) # Exceeds 12000 chars
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_count=1)
    assert is_valid is False
    assert "invalid topic length" in reason


def test_validate_seed_corpus_rejects_unsupported_language(tmp_path: Path):
    test_file = tmp_path / "bad_lang.jsonl"
    record = _make_valid_record(550)
    record["language"] = "python"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_count=1)
    assert is_valid is False
    assert "unsupported language" in reason


def test_validate_seed_corpus_rejects_empty_anchors(tmp_path: Path):
    test_file = tmp_path / "empty_anchors.jsonl"
    record = _make_valid_record(550)
    record["anchors"] = []
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_count=1)
    assert is_valid is False
    assert "missing anchors list" in reason


def test_provenance_logging(tmp_path: Path):
    entry = {"operation": "download", "phase": "download_started", "repo_id": "test/repo"}
    log_provenance(entry, tmp_path)
    log_file = tmp_path / "artifacts" / "provenance" / "hub_operations.jsonl"
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 1
    assert lines[0]["operation"] == "download"
    assert lines[0]["phase"] == "download_started"
