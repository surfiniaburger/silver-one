import json
import pytest
from pathlib import Path
from scripts.upload_to_huggingface import _safe_resolve, validate_seed_corpus, _compute_sha256, _log_provenance


def test_safe_resolve_valid_path(tmp_path: Path):
    safe_target = _safe_resolve("scenarios/debate/seeds.jsonl", tmp_path)
    assert safe_target == (tmp_path / "scenarios/debate/seeds.jsonl").resolve()


def test_safe_resolve_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="escapes base directory"):
        _safe_resolve("../../etc/passwd", tmp_path)


def test_validate_seed_corpus_valid(tmp_path: Path):
    test_file = tmp_path / "valid_seeds.jsonl"
    records = [
        {
            "topic": "int process_packet(char *buf, int len) {\n    if (!buf) return -1;\n    return 0;\n}",
            "predicate": "The code is not vulnerable to null pointer dereference in process_packet",
            "gepa_info": {
                "predicate": "The code is not vulnerable to null pointer dereference in process_packet",
                "evidence_hooks": ["if (!buf) return -1;"],
                "uncertainty": "Low",
                "proof_requirements": "Demonstrate buf == NULL bypass"
            },
            "language": "c",
            "original_safety": "safe",
            "anchors": ["if (!buf) return -1;", "process_packet", "buf", "len"]
        }
        for _ in range(12)
    ]
    with open(test_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_min_count=10)
    assert is_valid is True
    assert reason == "Valid"
    assert count == 12


def test_validate_seed_corpus_rejects_missing_keys(tmp_path: Path):
    test_file = tmp_path / "missing_keys.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"topic": "int a = 0;", "predicate": "vulnerable"}) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_min_count=1)
    assert is_valid is False
    assert "missing required keys" in reason


def test_validate_seed_corpus_rejects_unsupported_language(tmp_path: Path):
    test_file = tmp_path / "bad_lang.jsonl"
    record = {
        "topic": "def foo():\n    return 'invalid language for this c/c++ corpus'",
        "predicate": "vulnerable to something",
        "gepa_info": {"predicate": "vulnerable"},
        "language": "python",
        "original_safety": "vulnerable",
        "anchors": ["def foo"]
    }
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_min_count=1)
    assert is_valid is False
    assert "unsupported language" in reason


def test_validate_seed_corpus_rejects_short_code(tmp_path: Path):
    test_file = tmp_path / "short_code.jsonl"
    record = {
        "topic": "int x;",
        "predicate": "vulnerable to undefined value",
        "gepa_info": {"predicate": "vulnerable"},
        "language": "c",
        "original_safety": "vulnerable",
        "anchors": ["int x"]
    }
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_min_count=1)
    assert is_valid is False
    assert "invalid topic length" in reason


def test_validate_seed_corpus_rejects_empty_anchors(tmp_path: Path):
    test_file = tmp_path / "empty_anchors.jsonl"
    record = {
        "topic": "int process_data(int *arr, int len) {\n    if (len <= 0) return 0;\n    return arr[0];\n}",
        "predicate": "The code is vulnerable to out of bounds read",
        "gepa_info": {"predicate": "vulnerable"},
        "language": "c",
        "original_safety": "vulnerable",
        "anchors": []
    }
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    is_valid, reason, count = validate_seed_corpus(test_file, expected_min_count=1)
    assert is_valid is False
    assert "missing anchors list" in reason


def test_provenance_logging(tmp_path: Path):
    entry = {"operation": "test", "repo_id": "test/repo", "sha256": "abc123"}
    _log_provenance(entry, tmp_path)
    log_file = tmp_path / "artifacts" / "provenance" / "hub_operations.jsonl"
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 1
    assert lines[0]["operation"] == "test"
    assert lines[0]["repo_id"] == "test/repo"
