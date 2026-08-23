"""
Unit tests for Reflector Schemas & Deterministic Classifiers.

Tests cover:
  - Taxonomy bucket classification (§5.1)
  - Failure bucket precedence (§3.2)
  - Pydantic schema validation (§3.3, §3.4)
  - Static baseline prompts (§3.4)
  - GraphDiagnosticSignature.to_flow_dict() backward compat
"""

from __future__ import annotations

from types import SimpleNamespace

from pydantic import ValidationError
import pytest

from scenarios.debate.graph_dataflow import FlowGraphSnapshot, FlowSignature
from scenarios.debate.reflector_schemas import (
    GraphDiagnosticSignature,
    ReflectRequest,
    ReflectResponse,
    classify_graph_diagnostic,
    classify_taxonomy_bucket,
    get_static_baseline_prompt,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_snapshot(
    *,
    is_complete: bool = True,
    parse_error: str | None = None,
    nodes: dict | None = None,
    signatures: list[FlowSignature] | None = None,
    scenario_id: str = "test-scenario",
) -> FlowGraphSnapshot:
    """Build a FlowGraphSnapshot with sensible defaults for testing."""
    return FlowGraphSnapshot(
        snapshot_id="snap-test",
        scenario_id=scenario_id,
        version=1,
        created_at=1000.0,
        nodes=nodes or {},
        signatures=signatures or [],
        is_complete=is_complete,
        parse_error=parse_error,
    )


def _make_debate_result(
    *,
    verifier_logic_error: bool = False,
    b2_strict_fail: bool = False,
    b2_anchor_match: bool = True,
    failed_anchor_lines: list[str] | None = None,
    verifier_report: dict | None = None,
    judge_rationale: str = "",
) -> SimpleNamespace:
    """Build a lightweight debate result object for classifier tests."""
    return SimpleNamespace(
        verifier_logic_error=verifier_logic_error,
        b2_strict_fail=b2_strict_fail,
        b2_anchor_match=b2_anchor_match,
        failed_anchor_lines=failed_anchor_lines or [],
        verifier_report=verifier_report or {},
        judge_rationale=judge_rationale,
    )


def _make_sig(
    source_id: str = "src_1",
    sink_id: str = "sink_1",
    source_type: str = "UNTRUSTED_INPUT",
    sink_type: str = "MEMORY_WRITE",
    sanitizer_type: str | None = None,
    guarded_target: str | None = None,
) -> FlowSignature:
    return FlowSignature(
        source_id=source_id,
        sink_id=sink_id,
        source_type=source_type,
        sink_type=sink_type,
        flow_type="data_flow",
        sanitizer_type=sanitizer_type,
        guarded_target=guarded_target,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §5.1  Taxonomy Classification Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestClassifyTaxonomyBucket:
    """Tests for classify_taxonomy_bucket."""

    @pytest.mark.parametrize(
        "predicate, expected",
        [
            ("BUFFER_OVERFLOW in memcpy call", "memory_safety"),
            ("Heap overflow via strcpy", "memory_safety"),
            ("Use-after-free on pointer p", "memory_safety"),
            ("Double free of allocated buffer", "memory_safety"),
            ("Off-by-one write past buffer end", "memory_safety"),
            ("Null dereference in handler", "memory_safety"),
            ("Out-of-bounds read from array", "memory_safety"),
        ],
    )
    def test_memory_safety_keywords(self, predicate: str, expected: str) -> None:
        assert classify_taxonomy_bucket(predicate) == expected

    @pytest.mark.parametrize(
        "predicate, expected",
        [
            ("Integer overflow in size calculation", "integer_arithmetic"),
            ("Unsigned wrap-around on counter", "integer_arithmetic"),
            ("Width truncation from uint64 to uint32", "integer_arithmetic"),
            ("Signedness comparison bug in loop", "integer_arithmetic"),
        ],
    )
    def test_integer_arithmetic_keywords(self, predicate: str, expected: str) -> None:
        assert classify_taxonomy_bucket(predicate) == expected

    @pytest.mark.parametrize(
        "predicate, expected",
        [
            ("TOCTOU race condition in file check", "concurrency"),
            ("Deadlock in mutex acquisition", "concurrency"),
            ("Data race on shared counter", "concurrency"),
            ("Double-checked locking pattern", "concurrency"),
        ],
    )
    def test_concurrency_keywords(self, predicate: str, expected: str) -> None:
        assert classify_taxonomy_bucket(predicate) == expected

    @pytest.mark.parametrize(
        "predicate, expected",
        [
            ("Command injection via system() call", "input_validation"),
            ("Path traversal in file open", "input_validation"),
            ("Format string vulnerability in printf", "input_validation"),
            ("SQL injection in query builder", "input_validation"),
        ],
    )
    def test_input_validation_keywords(self, predicate: str, expected: str) -> None:
        assert classify_taxonomy_bucket(predicate) == expected

    def test_unknown_predicate_defaults_to_input_validation(self) -> None:
        """Predicates with no recognized keywords fall back to input_validation."""
        assert classify_taxonomy_bucket("some obscure vulnerability") == "input_validation"
        assert classify_taxonomy_bucket("") == "input_validation"

    def test_case_insensitive(self) -> None:
        """Classification is case-insensitive."""
        assert classify_taxonomy_bucket("BUFFER_OVERFLOW") == "memory_safety"
        assert classify_taxonomy_bucket("buffer_overflow") == "memory_safety"
        assert classify_taxonomy_bucket("Buffer_Overflow") == "memory_safety"

    def test_memory_safety_takes_precedence_over_input_validation(self) -> None:
        """When both memory and input keywords present, memory wins (higher precedence)."""
        predicate = "Buffer overflow via command injection"
        assert classify_taxonomy_bucket(predicate) == "memory_safety"


# ═════════════════════════════════════════════════════════════════════════════
# §3.2  Failure Bucket Precedence Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestClassifyGraphDiagnostic:
    """Tests for classify_graph_diagnostic §3.2 precedence."""

    def test_bucket_1_unsupported_syntax(self) -> None:
        """Incomplete parse with no nodes → B_UNSUPPORTED_SYNTAX."""
        snap = _make_snapshot(is_complete=False, parse_error="Parse failed", nodes={})
        debate = _make_debate_result()
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_UNSUPPORTED_SYNTAX"

    def test_bucket_1_takes_precedence_over_logic_error(self) -> None:
        """B_UNSUPPORTED_SYNTAX outranks B_LOGIC_ERROR when both apply."""
        snap = _make_snapshot(is_complete=False, nodes={})
        debate = _make_debate_result(verifier_logic_error=True)
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_UNSUPPORTED_SYNTAX"

    def test_bucket_2_logic_error(self) -> None:
        """Verifier logic error with complete parse → B_LOGIC_ERROR."""
        snap = _make_snapshot(
            is_complete=True,
            nodes={"src_1": {"kind": "source"}, "sink_1": {"kind": "sink", "target_var": "buf"}},
            signatures=[_make_sig()],
        )
        debate = _make_debate_result(verifier_logic_error=True)
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_LOGIC_ERROR"
        assert diag.verifier_logic_error is True

    def test_bucket_3_anchor_unmatched_strict_fail(self) -> None:
        """b2_strict_fail → B_ANCHOR_UNMATCHED."""
        snap = _make_snapshot(
            nodes={"src_1": {"kind": "source"}, "sink_1": {"kind": "sink", "target_var": "buf"}},
            signatures=[_make_sig()],
        )
        debate = _make_debate_result(b2_strict_fail=True)
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_ANCHOR_UNMATCHED"

    def test_bucket_3_anchor_unmatched_no_match(self) -> None:
        """b2_anchor_match=False → B_ANCHOR_UNMATCHED."""
        snap = _make_snapshot(
            nodes={"src_1": {"kind": "source"}, "sink_1": {"kind": "sink", "target_var": "buf"}},
            signatures=[_make_sig()],
        )
        debate = _make_debate_result(b2_anchor_match=False)
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_ANCHOR_UNMATCHED"

    def test_bucket_4_source_missing(self) -> None:
        """Sink present but all sources are UNKNOWN_ORIGIN → B_SOURCE_MISSING."""
        sig = _make_sig(source_type="UNKNOWN_ORIGIN")
        snap = _make_snapshot(
            nodes={"src_1": {"kind": "source"}, "sink_1": {"kind": "sink", "target_var": "buf"}},
            signatures=[sig],
        )
        debate = _make_debate_result()
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_SOURCE_MISSING"
        assert diag.sink_id == "sink_1"

    def test_bucket_5_sink_missing(self) -> None:
        """No recognized sink operations → B_SINK_MISSING."""
        sig = FlowSignature(
            source_id="src_1",
            sink_id="sink_1",
            source_type="UNTRUSTED_INPUT",
            sink_type="UNKNOWN_SINK",
            flow_type="data_flow",
        )
        snap = _make_snapshot(
            nodes={"src_1": {"kind": "source"}, "sink_1": {"kind": "sink"}},
            signatures=[sig],
        )
        debate = _make_debate_result()
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_SINK_MISSING"

    def test_bucket_5_no_signatures(self) -> None:
        """Complete parse but zero signatures → B_SINK_MISSING."""
        snap = _make_snapshot(
            nodes={"n1": {"kind": "source"}},
            signatures=[],
        )
        debate = _make_debate_result()
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_SINK_MISSING"

    def test_bucket_6_sanitizer_mismatch(self) -> None:
        """Wrong sanitizer type for sink → B_SANITIZER_MISMATCH."""
        # NULL_CHECK on MEMORY_WRITE (needs BOUNDS_CHECK)
        sig = _make_sig(
            sink_type="MEMORY_WRITE",
            sanitizer_type="NULL_CHECK",
            guarded_target="buf",
        )
        snap = _make_snapshot(
            nodes={
                "src_1": {"kind": "source", "target_var": "buf"},
                "sink_1": {"kind": "sink", "target_var": "buf"},
            },
            signatures=[sig],
        )
        debate = _make_debate_result()
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_SANITIZER_MISMATCH"
        assert diag.required_sanitizer == "BOUNDS_CHECK"
        assert diag.found_sanitizer == "NULL_CHECK"

    def test_bucket_6_no_sanitizer_at_all(self) -> None:
        """No sanitizer on unsanitized path → B_SANITIZER_MISMATCH (missing)."""
        sig = _make_sig(sink_type="MEMORY_WRITE")
        snap = _make_snapshot(
            nodes={
                "src_1": {"kind": "source", "target_var": "buf"},
                "sink_1": {"kind": "sink", "target_var": "buf"},
            },
            signatures=[sig],
        )
        debate = _make_debate_result()
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_SANITIZER_MISMATCH"
        assert diag.found_sanitizer is None

    def test_bucket_7_sanitizer_target_mismatch(self) -> None:
        """Correct sanitizer type but guards wrong variable → B_SANITIZER_TARGET_MISMATCH."""
        sig = _make_sig(
            sink_type="MEMORY_WRITE",
            sanitizer_type="BOUNDS_CHECK",
            guarded_target="other_buf",  # Guards wrong var
        )
        snap = _make_snapshot(
            nodes={
                "src_1": {"kind": "source", "target_var": "buf"},
                "sink_1": {"kind": "sink", "target_var": "buf"},  # Sink targets 'buf'
            },
            signatures=[sig],
        )
        debate = _make_debate_result()
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_SANITIZER_TARGET_MISMATCH"
        assert diag.target_var == "buf"
        assert diag.guarded_target == "other_buf"

    def test_bucket_4_mixed_signature_unknown_origin_recognized_sink(self) -> None:
        """Recognized sink with UNKNOWN_ORIGIN alongside an unrelated tracked signature → B_SOURCE_MISSING."""
        # Unrelated signature has tracked source, but is not a recognized sink
        unrelated_sig = FlowSignature(
            source_id="src_unrelated",
            sink_id="sink_unrelated",
            source_type="UNTRUSTED_INPUT",
            sink_type="UNKNOWN_SINK",
            flow_type="data_flow",
        )
        # Recognized sink has UNKNOWN_ORIGIN
        rec_sig = _make_sig(
            source_id="src_1",
            sink_id="sink_1",
            source_type="UNKNOWN_ORIGIN",
            sink_type="MEMORY_WRITE",
        )
        snap = _make_snapshot(
            nodes={"src_1": {"kind": "source"}, "sink_1": {"kind": "sink", "target_var": "buf"}},
            signatures=[unrelated_sig, rec_sig],
        )
        debate = _make_debate_result()
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_SOURCE_MISSING"
        assert diag.sink_id == "sink_1"

    def test_fully_sanitized_flow_rejected_reports_logic_error(self) -> None:
        """Fully sanitized graph with valid sink rejected by verifier → B_LOGIC_ERROR, not B_SINK_MISSING."""
        sig = _make_sig(
            sink_type="MEMORY_WRITE",
            sanitizer_type="BOUNDS_CHECK",
            guarded_target="buf",
        )
        snap = _make_snapshot(
            nodes={
                "src_1": {"kind": "source", "target_var": "buf"},
                "sink_1": {"kind": "sink", "target_var": "buf"},
            },
            signatures=[sig],
        )
        debate = _make_debate_result()  # rejected attempt (e.g. verifier failed)
        diag = classify_graph_diagnostic(debate, snap)
        assert diag.failure_bucket == "B_LOGIC_ERROR"
        assert diag.sink_id == "sink_1"
        assert diag.found_sanitizer == "BOUNDS_CHECK"

    def test_scenario_id_passthrough(self) -> None:
        """scenario_id from args is used; fallback to snapshot's scenario_id."""
        snap = _make_snapshot(is_complete=False, nodes={}, scenario_id="snap-id")
        debate = _make_debate_result()

        diag_explicit = classify_graph_diagnostic(debate, snap, scenario_id="explicit-id")
        assert diag_explicit.scenario_id == "explicit-id"

        diag_fallback = classify_graph_diagnostic(debate, snap)
        assert diag_fallback.scenario_id == "snap-id"

    def test_predicate_family_passthrough(self) -> None:
        """predicate_family is included in the diagnostic."""
        snap = _make_snapshot(is_complete=False, nodes={})
        debate = _make_debate_result()
        diag = classify_graph_diagnostic(debate, snap, predicate_family="BUFFER_OVERFLOW")
        assert diag.predicate_family == "BUFFER_OVERFLOW"


# ═════════════════════════════════════════════════════════════════════════════
# §3.3  Schema Validation Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGraphDiagnosticSignature:
    """Tests for the Pydantic GraphDiagnosticSignature model."""

    def test_to_flow_dict_shape(self) -> None:
        """to_flow_dict produces backward-compatible FlowSignature dict keys and roundtrips."""
        diag = GraphDiagnosticSignature(
            scenario_id="test",
            predicate_family="BUFFER_OVERFLOW",
            failure_bucket="B_SANITIZER_MISMATCH",
            source_id="src_1",
            sink_id="sink_1",
            source_type="UNTRUSTED_INPUT",
            sink_type="MEMORY_WRITE",
            flow_type="data_flow",
            required_sanitizer="BOUNDS_CHECK",
            found_sanitizer="NULL_CHECK",
            target_var="buf",
            guarded_target="other",
            failed_anchor_lines=["line 42"],
            invalid_at=100.5,
        )
        d = diag.to_flow_dict()
        assert d["source_id"] == "src_1"
        assert d["sink_id"] == "sink_1"
        assert d["source_type"] == "UNTRUSTED_INPUT"
        assert d["sink_type"] == "MEMORY_WRITE"
        assert d["flow_type"] == "data_flow"
        assert d["sanitizer_type"] == "NULL_CHECK"
        assert d["target_var"] == "buf"
        assert d["guarded_target"] == "other"
        assert d["failure_bucket"] == "B_SANITIZER_MISMATCH"
        assert d["failed_anchor_lines"] == ["line 42"]
        assert d["invalid_at"] == 100.5

        # Verify full round-trip through FlowSignature.from_dict()
        flow_sig = FlowSignature.from_dict(d)
        assert flow_sig.source_id == "src_1"
        assert flow_sig.sink_id == "sink_1"
        assert flow_sig.source_type == "UNTRUSTED_INPUT"
        assert flow_sig.sink_type == "MEMORY_WRITE"
        assert flow_sig.flow_type == "data_flow"
        assert flow_sig.sanitizer_type == "NULL_CHECK"
        assert flow_sig.guarded_target == "other"
        assert flow_sig.invalid_at == 100.5

    def test_optional_fields_default_none(self) -> None:
        """Optional fields default to None/empty when not provided."""
        diag = GraphDiagnosticSignature(
            scenario_id="test",
            predicate_family="TEST",
            failure_bucket="B_SINK_MISSING",
        )
        assert diag.source_id is None
        assert diag.sink_id is None
        assert diag.target_var is None
        assert diag.failed_anchor_lines == []
        assert diag.verifier_logic_error is False


class TestReflectResponse:
    """Tests for the Pydantic ReflectResponse model."""

    def test_success_status(self) -> None:
        resp = ReflectResponse(
            status="SUCCESS",
            mutated_system_prompt="You are a security analyst...",
            mutation_rationale="Added bounds check guidance",
            applied_topological_rule="rule_bounds_check_v1",
            taxonomy_bucket="memory_safety",
            pareto_variant_id="variant_abc123",
            estimated_correction_success_probability=0.75,
        )
        assert resp.status == "SUCCESS"
        assert resp.pareto_variant_id == "variant_abc123"

    def test_fallback_baseline_status(self) -> None:
        resp = ReflectResponse(
            status="FALLBACK_BASELINE",
            mutated_system_prompt="You are a security analyst...",
            mutation_rationale="LLM output unrecoverable",
            applied_topological_rule="none",
            taxonomy_bucket="memory_safety",
            pareto_variant_id="baseline_v0",
            estimated_correction_success_probability=0.0,
        )
        assert resp.status == "FALLBACK_BASELINE"
        assert resp.pareto_variant_id == "baseline_v0"

    def test_probability_bounds(self) -> None:
        """Probability must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            ReflectResponse(
                status="SUCCESS",
                mutated_system_prompt="prompt",
                mutation_rationale="test",
                applied_topological_rule="rule",
                taxonomy_bucket="memory_safety",
                pareto_variant_id="v1",
                estimated_correction_success_probability=1.5,
            )

    def test_empty_prompt_rejected(self) -> None:
        """mutated_system_prompt must be non-empty (min_length=1)."""
        with pytest.raises(ValidationError):
            ReflectResponse(
                status="SUCCESS",
                mutated_system_prompt="",
                mutation_rationale="test",
                applied_topological_rule="rule",
                taxonomy_bucket="memory_safety",
                pareto_variant_id="v1",
                estimated_correction_success_probability=0.5,
            )


class TestReflectRequest:
    """Tests for the Pydantic ReflectRequest model."""

    def test_attempt_index_minimum(self) -> None:
        """attempt_index must be >= 1."""
        with pytest.raises(ValidationError):
            ReflectRequest(
                attempt_index=0,
                scenario_id="test",
                predicate_family="TEST",
                taxonomy_bucket="memory_safety",
                code_text="int x;",
                graph_diagnostic=GraphDiagnosticSignature(
                    scenario_id="test",
                    predicate_family="TEST",
                    failure_bucket="B_SINK_MISSING",
                ),
                current_system_prompt="prompt",
            )


# ═════════════════════════════════════════════════════════════════════════════
# §3.4  Static Baseline Prompt Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGetStaticBaselinePrompt:
    """Tests for get_static_baseline_prompt."""

    @pytest.mark.parametrize(
        "taxonomy",
        ["memory_safety", "integer_arithmetic", "concurrency", "input_validation"],
    )
    def test_all_buckets_return_nonempty_prompt(self, taxonomy: str) -> None:
        prompt = get_static_baseline_prompt(taxonomy)
        assert isinstance(prompt, str)
        assert len(prompt) > 50  # Substantive prompt, not a stub

    def test_all_buckets_return_distinct_prompts(self) -> None:
        """Each taxonomy bucket has a distinct specialist prompt."""
        prompts = {
            t: get_static_baseline_prompt(t)
            for t in ["memory_safety", "integer_arithmetic", "concurrency", "input_validation"]
        }
        assert len(set(prompts.values())) == 4

    def test_unknown_taxonomy_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown taxonomy bucket"):
            get_static_baseline_prompt("unknown_bucket")

    def test_memory_safety_prompt_mentions_memory_ops(self) -> None:
        """Sanity check: memory_safety prompt mentions relevant operations."""
        prompt = get_static_baseline_prompt("memory_safety")
        assert "memcpy" in prompt or "buffer" in prompt.lower()

    def test_input_validation_prompt_mentions_injection(self) -> None:
        """Sanity check: input_validation prompt mentions injection vectors."""
        prompt = get_static_baseline_prompt("input_validation")
        assert "injection" in prompt.lower() or "traversal" in prompt.lower()
