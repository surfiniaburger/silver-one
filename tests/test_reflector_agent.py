"""
Unit tests for the GEPA Reflector Agent, Work-Memory Engine, and Pareto Registry.

Tests:
  - Spec §4.1: Time-decayed signed scoring (compute_time_decay, calculate_rule_score).
  - Spec §4.1: Corroboration threshold (is_mutation_preferred >= 2 seeds).
  - Spec §4.2: Cross-seed dead-end suppression (classify_attempt_outcome -> DEAD_END_CHAIN).
  - Spec §5.2: Concurrency lock (gepa_lock, atomic JSON writing).
  - Spec §3.4: Topological prompt mutation across all failure buckets.
  - FastAPI endpoints (/health, /reflect, /record_attempt) and ReflectorClient.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from scenarios.debate.pareto_registry import (
    GepaLockTimeoutError,
    ParetoRegistry,
    atomic_write_json,
    gepa_lock,
)
from scenarios.debate.reflector_agent import (
    ReflectorClient,
    calculate_rule_score,
    classify_attempt_outcome,
    compute_time_decay,
    count_corroborating_seeds,
    create_app,
    is_cross_seed_dead_end,
    is_mutation_preferred,
    mutate_system_prompt,
    outcome_sign,
)
from scenarios.debate.reflector_schemas import (
    GraphDiagnosticSignature,
    ReflectRequest,
    ReflectResponse,
    get_static_baseline_prompt,
)


# ═════════════════════════════════════════════════════════════════════════════
# §4.1  Work-Memory Reflection Engine Math Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestWorkMemoryScoring:
    """Tests for time decay, outcome signs, and aggregated scoring."""

    def test_compute_time_decay_half_life(self) -> None:
        """Weight must equal 0.5 after exactly 30 days."""
        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t_30d = t0 + timedelta(days=30)
        t_60d = t0 + timedelta(days=60)

        assert pytest.approx(compute_time_decay(t0, t0, half_life_days=30.0), rel=1e-4) == 1.0
        assert pytest.approx(compute_time_decay(t0, t_30d, half_life_days=30.0), rel=1e-4) == 0.5
        assert pytest.approx(compute_time_decay(t0, t_60d, half_life_days=30.0), rel=1e-4) == 0.25

    def test_outcome_sign_weights(self) -> None:
        """Outcome signs must match Spec §4.1."""
        assert outcome_sign("VALID_ACCEPT") == 1.0
        assert outcome_sign("LOGIC_ERROR") == -1.5
        assert outcome_sign("DEAD_END_CHAIN") == -1.5
        assert outcome_sign("RETRYABLE_FAILURE") == -0.5

    def test_calculate_rule_score_aggregation(self) -> None:
        """Verify aggregated score with time decay."""
        now = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
        records = [
            {
                "outcome": "VALID_ACCEPT",
                "evaluated_at": now.isoformat(),
            },
            {
                "outcome": "RETRYABLE_FAILURE",
                "evaluated_at": (now - timedelta(days=30)).isoformat(),
            },
        ]
        # score = 1.0 * 1.0 + (-0.5) * 0.5 = 1.0 - 0.25 = 0.75
        score = calculate_rule_score(records, now=now, half_life_days=30.0)
        assert pytest.approx(score, rel=1e-4) == 0.75

    def test_corroboration_tracking(self) -> None:
        """Corroboration requires >= 2 distinct successful seeds."""
        single_seed = [
            {"seed_id": "42", "outcome": "VALID_ACCEPT"},
            {"seed_id": "42", "outcome": "VALID_ACCEPT"},
        ]
        assert count_corroborating_seeds(single_seed) == 1
        assert not is_mutation_preferred(single_seed, min_corroboration=2)

        multi_seed = [
            {"seed_id": "42", "outcome": "VALID_ACCEPT"},
            {"seed_id": "43", "outcome": "VALID_ACCEPT"},
        ]
        assert count_corroborating_seeds(multi_seed) == 2
        assert is_mutation_preferred(multi_seed, min_corroboration=2)

    def test_cross_seed_dead_end_detection(self) -> None:
        """>= 3 consecutive failures across distinct seeds triggers DEAD_END_CHAIN."""
        now = datetime.now(timezone.utc)
        traces = [
            {
                "seed_id": "42",
                "outcome": "RETRYABLE_FAILURE",
                "evaluated_at": (now - timedelta(minutes=10)).isoformat(),
            },
            {
                "seed_id": "43",
                "outcome": "RETRYABLE_FAILURE",
                "evaluated_at": (now - timedelta(minutes=5)).isoformat(),
            },
            {
                "seed_id": "44",
                "outcome": "RETRYABLE_FAILURE",
                "evaluated_at": now.isoformat(),
            },
        ]
        assert is_cross_seed_dead_end(traces)

    def test_classify_attempt_outcome_hierarchy(self) -> None:
        """Classify attempt outcome according to precedence."""
        # 1. Valid accept
        assert classify_attempt_outcome(is_valid=True, verifier_logic_error=False, prior_mutation_traces=[]) == "VALID_ACCEPT"

        # 2. Logic error
        assert classify_attempt_outcome(is_valid=False, verifier_logic_error=True, prior_mutation_traces=[]) == "LOGIC_ERROR"

        # 3. Dead end chain
        now = datetime.now(timezone.utc)
        prior_failures = [
            {"seed_id": "42", "outcome": "RETRYABLE_FAILURE", "evaluated_at": (now - timedelta(minutes=10)).isoformat()},
            {"seed_id": "43", "outcome": "RETRYABLE_FAILURE", "evaluated_at": (now - timedelta(minutes=5)).isoformat()},
        ]
        assert classify_attempt_outcome(is_valid=False, verifier_logic_error=False, prior_mutation_traces=prior_failures) == "DEAD_END_CHAIN"

        # 4. Standard retryable failure
        assert classify_attempt_outcome(is_valid=False, verifier_logic_error=False, prior_mutation_traces=[]) == "RETRYABLE_FAILURE"


# ═════════════════════════════════════════════════════════════════════════════
# §5.2  Concurrency Lock & Atomic File I/O Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestParetoRegistryAndLock:
    """Tests for ParetoRegistry, gepa_lock, and atomic writes."""

    def test_atomic_write_json(self, tmp_path: Path) -> None:
        """Atomic write correctly serializes and publishes JSON data."""
        target = tmp_path / "test.json"
        data = {"hello": "world", "numbers": [1, 2, 3]}
        atomic_write_json(target, data)

        assert target.exists()
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded == data

    def test_gepa_lock_mutual_exclusion(self, tmp_path: Path) -> None:
        """gepa_lock raises GepaLockTimeoutError when lock is held."""
        lock_file = tmp_path / "gepa_ledger.lock"

        with gepa_lock(lock_file, timeout=1.0):
            # Second attempt to acquire same lock should timeout
            with pytest.raises(GepaLockTimeoutError):
                with gepa_lock(lock_file, timeout=0.1, initial_sleep=0.01):
                    pass

    def test_pareto_registry_fallback_to_baseline(self, tmp_path: Path) -> None:
        """ParetoRegistry returns static baseline prompt if empty."""
        reg = ParetoRegistry(gepa_dir=tmp_path / "gepa")
        prompt = reg.get_pareto_prompt("memory_safety")
        assert prompt == get_static_baseline_prompt("memory_safety")
        assert reg.get_pareto_variant_id("memory_safety") == "baseline_v0"

    def test_pareto_registry_register_and_retrieve(self, tmp_path: Path) -> None:
        """ParetoRegistry updates and stores Pareto frontier prompts."""
        reg = ParetoRegistry(gepa_dir=tmp_path / "gepa")
        reg.register_pareto_prompt(
            taxonomy="memory_safety",
            prompt="Mutated prompt for buffer overflow",
            variant_id="var_abc123",
            score=1.5,
            rationale="Added bounds check",
        )

        assert reg.get_pareto_prompt("memory_safety") == "Mutated prompt for buffer overflow"
        assert reg.get_pareto_variant_id("memory_safety") == "var_abc123"

        # Check mutations log
        mutations_log = (tmp_path / "gepa" / "mutations.jsonl").read_text(encoding="utf-8")
        assert "var_abc123" in mutations_log

    def test_pareto_registry_dead_ends(self, tmp_path: Path) -> None:
        """ParetoRegistry tracks negative dead-end constraints."""
        reg = ParetoRegistry(gepa_dir=tmp_path / "gepa")
        reg.add_dead_end_constraint("concurrency", "use sleep() instead of pthread_mutex_lock")

        dead_ends = reg.get_known_dead_ends("concurrency")
        assert "use sleep() instead of pthread_mutex_lock" in dead_ends


# ═════════════════════════════════════════════════════════════════════════════
# §3.4  Topological Prompt Mutation Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestPromptMutation:
    """Tests for mutate_system_prompt across failure buckets."""

    @pytest.mark.parametrize(
        "bucket,expected_term",
        [
            ("B_UNSUPPORTED_SYNTAX", "AST syntax"),
            ("B_LOGIC_ERROR", "factual contradictions"),
            ("B_ANCHOR_UNMATCHED", "source line anchors"),
            ("B_SOURCE_MISSING", "tainted data-flow"),
            ("B_SINK_MISSING", "security-sensitive sink"),
            ("B_SANITIZER_MISMATCH", "guard validation"),
            ("B_SANITIZER_TARGET_MISMATCH", "sanitizer guard protects"),
        ],
    )
    def test_mutation_for_each_failure_bucket(
        self, tmp_path: Path, bucket: str, expected_term: str
    ) -> None:
        reg = ParetoRegistry(gepa_dir=tmp_path / "gepa")
        diag = GraphDiagnosticSignature(
            scenario_id="scenario_001",
            predicate_family="BUFFER_OVERFLOW",
            failure_bucket=bucket,
            required_sanitizer="BOUNDS_CHECK",
            found_sanitizer="NULL_CHECK",
            target_var="dest_buf",
            guarded_target="src_buf",
        )
        req = ReflectRequest(
            attempt_index=1,
            scenario_id="scenario_001",
            predicate_family="BUFFER_OVERFLOW",
            taxonomy_bucket="memory_safety",
            code_text="char buf[10];",
            graph_diagnostic=diag,
            current_system_prompt="Base prompt",
        )

        resp = mutate_system_prompt(req, reg)
        assert resp.status == "SUCCESS"
        assert expected_term.lower() in resp.mutated_system_prompt.lower()
        assert resp.pareto_variant_id.startswith("var_")
        assert 0.0 <= resp.estimated_correction_success_probability <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# §3.4  FastAPI Microservice & ReflectorClient Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestFastAPIServiceAndClient:
    """Tests for FastAPI endpoints and ReflectorClient."""

    def test_health_endpoint(self, tmp_path: Path) -> None:
        reg = ParetoRegistry(gepa_dir=tmp_path / "gepa")
        app = create_app(reg)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
        assert resp.json()["port"] == 8004

    def test_reflect_endpoint(self, tmp_path: Path) -> None:
        reg = ParetoRegistry(gepa_dir=tmp_path / "gepa")
        app = create_app(reg)
        client = TestClient(app)

        diag = GraphDiagnosticSignature(
            scenario_id="s1",
            predicate_family="TEST",
            failure_bucket="B_SINK_MISSING",
        )
        req = ReflectRequest(
            attempt_index=1,
            scenario_id="s1",
            predicate_family="TEST",
            taxonomy_bucket="input_validation",
            code_text="int x;",
            graph_diagnostic=diag,
            current_system_prompt="Initial prompt",
        )

        resp = client.post("/reflect", json=req.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert "Initial prompt" in data["mutated_system_prompt"]
        assert data["taxonomy_bucket"] == "input_validation"

    def test_record_attempt_endpoint(self, tmp_path: Path) -> None:
        reg = ParetoRegistry(gepa_dir=tmp_path / "gepa")
        app = create_app(reg)
        client = TestClient(app)

        payload = {
            "taxonomy_bucket": "concurrency",
            "predicate_family": "RACE_CONDITION",
            "seed_id": "seed_101",
            "scenario_id": "scenario_101",
            "attempt_index": 1,
            "is_valid": True,
            "canonical_mutation_id": "var_123",
        }

        resp = client.post("/record_attempt", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["outcome"] == "VALID_ACCEPT"
        assert data["rule_score"] > 0
        assert data["corroborating_seeds"] == 1

    @pytest.mark.asyncio
    async def test_reflector_client_in_process(self, tmp_path: Path) -> None:
        """ReflectorClient in in_process mode operates cleanly without network."""
        reg = ParetoRegistry(gepa_dir=tmp_path / "gepa")
        client = ReflectorClient(registry=reg, in_process=True)

        prompt = await client.get_pareto_prompt("integer_arithmetic")
        assert prompt == get_static_baseline_prompt("integer_arithmetic")

        diag = GraphDiagnosticSignature(
            scenario_id="s2",
            predicate_family="INT_OVERFLOW",
            failure_bucket="B_LOGIC_ERROR",
        )
        req = ReflectRequest(
            attempt_index=1,
            scenario_id="s2",
            predicate_family="INT_OVERFLOW",
            taxonomy_bucket="integer_arithmetic",
            code_text="int a + b;",
            graph_diagnostic=diag,
            current_system_prompt="Prompt v0",
        )

        resp = await client.reflect(req)
        assert resp.status == "SUCCESS"
        assert "Prompt v0" in resp.mutated_system_prompt

        outcome = await client.record_attempt_and_classify_outcome(
            taxonomy_bucket="integer_arithmetic",
            predicate_family="INT_OVERFLOW",
            seed_id="42",
            scenario_id="s2",
            prompt="Prompt v0",
            attempt_index=1,
            is_valid=False,
            verifier_logic_error=True,
            observed_at=datetime.now(timezone.utc).isoformat(),
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )
        assert outcome == "LOGIC_ERROR"
