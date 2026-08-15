import json
import logging
from pathlib import Path
import pytest
import joblib

from scripts.train_pre_filter import (
    collect_graph_bucket_examples,
    _compute_graph_fold_diagnostics,
    extract_dataset_from_attempts,
    graph_acceptance_probability_from_bucket,
    train_pre_filter,
)
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
    assert (models_dir / "domain_scaler.joblib").exists()
    assert xgb_path.exists()

    # Verify model binaries can be loaded and executed
    vectorizer = joblib.load(vec_path)
    scaler = joblib.load(models_dir / "domain_scaler.joblib")
    classifier = joblib.load(xgb_path)

    from scenarios.debate.pre_filter import extract_domain_features_batch, _combine_features

    test_texts = ["Predicate: Test predicate | Code: void test() { }"]
    tfidf_feat = vectorizer.transform(test_texts)
    domain_feat = extract_domain_features_batch(test_texts)
    scaled_domain = scaler.transform(domain_feat)
    features = _combine_features(tfidf_feat, scaled_domain)
    probs = classifier.predict_proba(features)

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0][1] <= 1.0


def _setup_attempts_and_train(tmp_path: Path) -> tuple[Path, Path]:
    attempts_dir = tmp_path / "attempts"
    attempts_dir.mkdir(exist_ok=True)

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
    return attempts_dir, models_dir


def test_pre_filter_integration_with_trained_models(tmp_path):
    _, models_dir = _setup_attempts_and_train(tmp_path)

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


def test_extract_domain_features_dimensions():
    from scenarios.debate.pre_filter import extract_domain_features, extract_domain_features_batch, FallbackStandardScaler
    import numpy as np

    sample_text = "Predicate: Vulnerable to buffer overflow | Code: void f() { memcpy(a, b, n); }"
    feat = extract_domain_features(sample_text)

    assert isinstance(feat, np.ndarray)
    assert feat.shape == (60,)
    assert not np.isnan(feat).any()

    batch_feat = extract_domain_features_batch([sample_text, sample_text])
    assert batch_feat.shape == (2, 60)

    scaler = FallbackStandardScaler()
    scaled = scaler.fit_transform(batch_feat)
    assert scaled.shape == (2, 60)
    assert not np.isnan(scaled).any()


def test_model_manifest_generation_and_validation(tmp_path):
    _, models_dir = _setup_attempts_and_train(tmp_path)

    manifest_path = models_dir / "model_manifest.json"
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["schema_version"] == 1
    assert "xgb.joblib" in manifest_data["artifacts"]
    assert "vectorizer.joblib" in manifest_data["artifacts"]
    assert "domain_scaler.joblib" in manifest_data["artifacts"]
    assert manifest_data["sample_count"] >= 4

    # Test manifest checksum validation pass
    filter_valid = BarredPreFilter(model_dir=models_dir)
    assert filter_valid.vectorizer is not None
    assert filter_valid.xgb is not None

    # Test tampering with artifact content triggers manifest checksum rejection
    xgb_path = models_dir / "xgb.joblib"
    with xgb_path.open("ab") as f:
        f.write(b"\n# corrupted extra bytes")

    filter_tampered = BarredPreFilter(model_dir=models_dir)
    assert filter_tampered.vectorizer is None
    assert filter_tampered.xgb is None


def test_graph_fold_diagnostics_bucket_parser_and_prediction_errors():
    texts = [
        "Predicate: vuln | Code: def f(data, i):\n    buf[i] = data",
        "Predicate: guarded | Code: def f(data, i):\n    if i < MAX_LEN:\n        buf[i] = data",
        "Predicate: no sink | Code: def f(data):\n    return data",
        "Predicate: c syntax | Code: : : invalid {{{ syntax",
    ]
    labels = [1, 1, 0, 0]
    predictions = [1, 0, 1, 0]

    diagnostics = _compute_graph_fold_diagnostics(texts, labels, predictions)

    assert diagnostics["total_samples"] == 4
    assert diagnostics["parse_complete_count"] == 3
    assert diagnostics["parse_failed_count"] == 1
    assert diagnostics["parser_coverage"] == 0.75
    assert diagnostics["bucket_counts"]["missing_sanitizer"] == 1
    assert diagnostics["bucket_counts"]["guarded_or_safe"] == 1
    assert diagnostics["bucket_counts"]["missing_sink"] == 1
    assert diagnostics["bucket_counts"]["unsupported_syntax"] == 1
    assert diagnostics["graph_positive_evidence_count"] == 1
    assert diagnostics["graph_no_positive_evidence_count"] == 3
    assert diagnostics["proof_state_counts"]["proven_unsafe"] == 1
    assert diagnostics["proof_state_counts"]["proven_safe"] == 1
    assert diagnostics["proof_state_counts"]["unsupported_or_unproven"] == 2
    assert diagnostics["prediction_error_bucket_counts"]["guarded_or_safe"] == 1
    assert diagnostics["prediction_error_bucket_counts"]["missing_sink"] == 1


def test_graph_acceptance_probability_only_accepts_proven_unsafe():
    assert graph_acceptance_probability_from_bucket("missing_sanitizer") == 0.95
    assert graph_acceptance_probability_from_bucket("wrong_sanitizer") == 0.95
    assert graph_acceptance_probability_from_bucket("graph_high_risk") == 0.95
    assert graph_acceptance_probability_from_bucket("guarded_or_safe") == 0.0
    assert graph_acceptance_probability_from_bucket("unsupported_syntax") == 0.0
    assert graph_acceptance_probability_from_bucket("missing_source") == 0.0
    assert graph_acceptance_probability_from_bucket("missing_sink") == 0.0


def test_collect_graph_bucket_examples_limits_and_metadata():
    texts = [
        "Predicate: vuln | Code: def f(data, i):\n    buf[i] = data",
        "Predicate: guarded | Code: def f(data, i):\n    if i < MAX_LEN:\n        buf[i] = data",
        "Predicate: no sink | Code: def f(data):\n    return data",
        "Predicate: c syntax | Code: : : invalid {{{ syntax",
    ]
    labels = [1, 1, 0, 0]
    scenario_ids = ["sc_1", "sc_2", "sc_3", "sc_4"]

    examples = collect_graph_bucket_examples(texts, labels, scenario_ids, limit_per_bucket=1)

    assert set(examples) == {"guarded_or_safe", "missing_sanitizer", "missing_sink", "unsupported_syntax"}
    guarded = examples["guarded_or_safe"][0]
    assert guarded["sample_idx"] == 1
    assert guarded["label"] == 1
    assert guarded["scenario_id"] == "sc_2"
    assert guarded["proof_state"] == "proven_safe"
    assert guarded["risk_score"] == 0.05
    assert "if i < MAX_LEN" in guarded["code_text"]


def test_evaluate_graph_pre_filter_persisted_report_schema(tmp_path):
    from scripts.evaluate_graph_pre_filter import run_graph_cv_evaluation
    attempts_dir = tmp_path / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_file = attempts_dir / "attempt_1.jsonl"
    records = [
        {
            "scenario_id": "sc_vuln_1",
            "attempt_index": 1,
            "combined_text": "Predicate: vuln | Code: def f(data, i):\n    buf[i] = data",
            "predicate_label": "VULNERABLE",
            "accepted": True,
            "round_index": 0,
        },
        {
            "scenario_id": "sc_vuln_2",
            "attempt_index": 1,
            "combined_text": "Predicate: vuln | Code: def g(data, i):\n    buf[i] = data",
            "predicate_label": "VULNERABLE",
            "accepted": True,
            "round_index": 0,
        },
        {
            "scenario_id": "sc_safe_1",
            "attempt_index": 1,
            "combined_text": "Predicate: safe | Code: def h(data, i):\n    if i < 10:\n        buf[i] = data",
            "predicate_label": "SAFE",
            "accepted": False,
            "round_index": 0,
        },
        {
            "scenario_id": "sc_safe_2",
            "attempt_index": 1,
            "combined_text": "Predicate: safe | Code: def k(data, i):\n    if i < 10:\n        buf[i] = data",
            "predicate_label": "SAFE",
            "accepted": False,
            "round_index": 0,
        },
    ]
    with attempt_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    output_path = tmp_path / "artifacts" / "metrics" / "graph_report.json"
    project_root = Path(__file__).resolve().parent.parent
    examples_path = project_root / "artifacts" / "metrics" / "test_graph_examples_schema.json"
    try:
        report = run_graph_cv_evaluation(
            attempts_dir=attempts_dir,
            output_path=output_path,
            n_splits=2,
            seeds=[42],
            bucket_examples_path=examples_path,
            bucket_example_limit=1,
        )
        assert examples_path.exists()
        persisted_examples = json.loads(examples_path.read_text(encoding="utf-8"))
    finally:
        if examples_path.exists():
            examples_path.unlink()

    assert output_path.exists()
    persisted_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted_report == report
    assert persisted_examples
    assert all(len(rows) <= 1 for rows in persisted_examples.values())

    assert persisted_report["schema_version"] == "1.0"
    assert "dataset_summary" in persisted_report
    assert "evaluation_protocol" in persisted_report
    assert "graph_pre_filter_metrics" in persisted_report
    assert "seed_breakdown" in persisted_report

    assert persisted_report["evaluation_protocol"]["requested_n_splits"] == 2
    assert persisted_report["evaluation_protocol"]["effective_n_splits"] == 2
    assert persisted_report["evaluation_protocol"]["seeds"] == [42]

    metrics = persisted_report["graph_pre_filter_metrics"]
    assert "roc_auc_95_percentile_ci" in metrics
    assert "pr_auc_95_percentile_ci" in metrics

    assert len(persisted_report["seed_breakdown"]) == 1
    assert "diagnostics" in persisted_report["seed_breakdown"][0]


def test_evaluate_graph_pre_filter_rejects_external_bucket_examples_path(tmp_path):
    from scripts.evaluate_graph_pre_filter import run_graph_cv_evaluation

    attempts_dir = tmp_path / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_file = attempts_dir / "attempt_1.jsonl"
    attempt_file.write_text(
        json.dumps(
            {
                "scenario_id": "sc_vuln_1",
                "attempt_index": 1,
                "combined_text": "Predicate: vuln | Code: def f(data, i):\n    buf[i] = data",
                "predicate_label": "VULNERABLE",
                "accepted": True,
                "round_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "graph_report.json"
    project_root = Path(__file__).resolve().parent.parent
    bucket_examples_path = project_root.parent / "graph_examples_outside_project.json"

    with pytest.raises(ValueError, match="outside project root"):
        run_graph_cv_evaluation(
            attempts_dir=attempts_dir,
            output_path=output_path,
            n_splits=2,
            seeds=[42],
            bucket_examples_path=bucket_examples_path,
        )
