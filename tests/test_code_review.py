import json
import pytest
from pathlib import Path
from scripts import diff_extractor
from scripts import code_review_evaluator
from scripts import code_review_compare

SAMPLE_CODE = """# Module comment
import os

def hello_world(name: str):
    print(f"Hello, {name}!")
    return name

class MathOps:
    def add(self, a: int, b: int) -> int:
        return a + b

    async def multiply(self, x: int, y: int) -> int:
        return x * y
"""


def test_function_visitor():
    visitor = diff_extractor.FunctionVisitor(SAMPLE_CODE)
    import ast
    tree = ast.parse(SAMPLE_CODE)
    visitor.visit(tree)

    assert len(visitor.functions) == 3
    names = [fn["name"] for fn in visitor.functions]
    assert "hello_world" in names
    assert "add" in names
    assert "multiply" in names

    # check class names
    math_fns = [fn for fn in visitor.functions if fn["class_name"] == "MathOps"]
    assert len(math_fns) == 2
    assert {f["name"] for f in math_fns} == {"add", "multiply"}


def test_extract_units_from_file(tmp_path):
    temp_file = tmp_path / "sample_code.py"
    temp_file.write_text(SAMPLE_CODE, encoding="utf-8")

    # Change inside hello_world (lines 4-6)
    units = diff_extractor.extract_units_from_file(temp_file, [5])
    assert len(units) == 1
    assert units[0]["name"] == "hello_world"
    assert units[0]["class_name"] is None
    assert "def hello_world" in units[0]["code"]

    # Change inside MathOps.add (lines 9-10)
    units = diff_extractor.extract_units_from_file(temp_file, [10])
    assert len(units) == 1
    assert units[0]["name"] == "add"
    assert units[0]["class_name"] == "MathOps"

    # Module-level change (lines 1-2)
    units = diff_extractor.extract_units_from_file(temp_file, [1])
    assert len(units) == 1
    assert units[0]["name"] == "<module>"


def test_estimate_pr_tokens_empty():
    """An empty collection of code units should estimate to zero tokens."""
    assert code_review_evaluator.estimate_pr_tokens([]) == 0


def test_estimate_pr_tokens_scales_with_code_length():
    """Token estimation should scale monotonically with the length of the code."""
    short_unit = [{"code": "def short(): pass"}]
    long_unit = [{"code": "def longer():\n" + "    print('line')\n" * 50}]

    short_tokens = code_review_evaluator.estimate_pr_tokens(short_unit)
    long_tokens = code_review_evaluator.estimate_pr_tokens(long_unit)

    assert long_tokens > short_tokens


def test_estimate_pr_tokens_accumulates_overhead():
    """Estimating multiple units should account for cumulative per-unit overhead."""
    single_unit = [{"code": "def foo(): pass"}]
    multiple_units = [{"code": "def foo(): pass"}, {"code": "def bar(): pass"}]

    single_tokens = code_review_evaluator.estimate_pr_tokens(single_unit)
    multiple_tokens = code_review_evaluator.estimate_pr_tokens(multiple_units)

    assert single_tokens == 403
    assert multiple_tokens == 806


def test_filter_units_by_budget():
    units = [
        {"code": "a" * 4000, "lines_changed": 10},  # ~1400 tokens
        {"code": "b" * 8000, "lines_changed": 20},  # ~2400 tokens
        {"code": "c" * 200, "lines_changed": 5},    # ~450 tokens
    ]
    # Max units 2, max tokens 3000
    # sorted: b (20 lines changed), a (10 lines changed), c (5 lines changed)
    # b: 2400 tokens. Remaining budget: 600.
    # a: 1400 tokens -> exceeds remaining budget. Skip.
    # c: 450 tokens -> fits. Selected: [b, c].
    selected, _ = code_review_evaluator.filter_units_by_budget(units, max_tokens=3000, max_units=2)
    assert len(selected) == 2
    assert selected[0]["lines_changed"] == 20
    assert selected[1]["lines_changed"] == 5


def test_calculate_cqi():
    review = {
        "readability": {"score": 8, "rationale": "ok"},
        "maintainability": {"score": 9, "rationale": "ok"},
        "correctness": {"score": 7, "rationale": "ok"},
        "complexity": {"score": 10, "rationale": "ok"},
        "security": {"score": 6, "rationale": "ok"},
        "test_coverage": {"score": 8, "rationale": "ok"},
    }
    # Weighted score = (1.5*8 + 1.5*9 + 2.0*7 + 1.0*10 + 2.0*6 + 1.0*8) / 9.0
    # = (12.0 + 13.5 + 14.0 + 10.0 + 12.0 + 8.0) / 9.0 = 69.5 / 9.0 = 7.72222...
    cqi = code_review_compare.calculate_cqi(review)
    assert cqi == pytest.approx(7.72222, rel=1e-4)


def test_calculate_cqi_result_rejects_missing_dimension():
    review = {
        "readability": {"score": 8, "rationale": "ok"},
    }

    result = code_review_compare.calculate_cqi_result(review)

    assert result.valid is False
    assert result.value is None
    assert result.error_code == "MISSING_DIMENSION"
    assert "maintainability" in result.reason


def test_calculate_cqi_result_rejects_invalid_score():
    review = {
        "readability": {"score": 8, "rationale": "ok"},
        "maintainability": {"score": 9, "rationale": "ok"},
        "correctness": {"score": "banana", "rationale": "ok"},
        "complexity": {"score": 10, "rationale": "ok"},
        "security": {"score": 6, "rationale": "ok"},
        "test_coverage": {"score": 8, "rationale": "ok"},
    }

    result = code_review_compare.calculate_cqi_result(review)

    assert result.valid is False
    assert result.error_code == "INVALID_SCORE"
    assert "correctness" in result.reason


def test_code_review_verdict_fails_on_invalid_cqi():
    invalid_unit = {
        "file_path": "x.py",
        "name": "broken_review",
        "review": {"severity": "OK", "summary": "missing dimensions"},
    }

    cqi_reasons = code_review_compare.collect_cqi_failure_reasons([invalid_unit])
    verdict, exit_code, reasons = code_review_compare.determine_verdict([], [], 3, cqi_reasons)

    assert verdict == "FAIL"
    assert exit_code == 2
    assert reasons == cqi_reasons
    assert "INVALID CQI" in reasons[0]


def test_cqi_failure_collection_skips_malformed_units():
    invalid_unit = {
        "file_path": "x.py",
        "name": "broken_review",
        "review": {"severity": "OK", "summary": "missing dimensions"},
    }

    reasons = code_review_compare.collect_cqi_failure_reasons([
        "not-a-unit",
        invalid_unit,
        None,
    ])

    assert len(reasons) == 1
    assert "x.py -> broken_review" in reasons[0]


def test_code_review_block_requires_structured_block_finding():
    unsupported_block = {
        "review": {
            "severity": "BLOCK",
            "summary": "OK",
            "findings": [],
        }
    }
    supported_block = {
        "review": {
            "severity": "BLOCK",
            "summary": "Critical issue",
            "findings": [{"severity": "BLOCK"}],
        }
    }

    block_units, warn_units, ok_units = code_review_compare.group_units_by_severity([
        unsupported_block,
        supported_block,
    ])

    assert block_units == [supported_block]
    assert warn_units == [unsupported_block]
    assert ok_units == []


def test_get_reviews_wraps_legacy_review_payload(valid_review_payload):
    legacy_review = valid_review_payload()

    reviews = code_review_compare.get_reviews({"reviews": [legacy_review]})

    assert len(reviews) == 1
    unit = reviews[0]
    assert unit["file_path"] == "legacy-cassette"
    assert unit["name"] == "legacy_review_1"
    assert unit["class_name"] is None
    assert unit["review"] == legacy_review
    assert unit["validation"] == {
        "repaired": False,
        "normalized": False,
        "fields": [],
    }
    assert json.loads(unit["raw_response"]) == legacy_review
    assert code_review_compare.calculate_cqi_result(unit["review"]).valid is True


def test_get_reviews_preserves_legacy_unit_metadata(valid_review_payload):
    legacy_unit = {
        **valid_review_payload("WARN"),
        "file_path": "src/legacy.py",
        "name": "legacy_helper",
        "class_name": "Legacy",
    }

    unit = code_review_compare.get_reviews({"reviews": [legacy_unit]})[0]

    assert unit["file_path"] == "src/legacy.py"
    assert unit["name"] == "legacy_helper"
    assert unit["class_name"] == "Legacy"
    assert "file_path" not in unit["review"]
    assert unit["review"]["severity"] == "WARN"


def test_get_reviews_supplies_defaults_for_null_wrapped_fields(valid_review_payload):
    wrapped_unit = {
        "file_path": "src/current.py",
        "name": "current_helper",
        "review": valid_review_payload(),
        "validation": None,
        "raw_response": None,
    }

    unit = code_review_compare.get_reviews({"reviews": [wrapped_unit]})[0]

    assert unit["validation"] == {
        "repaired": False,
        "normalized": False,
        "fields": [],
    }
    assert json.loads(unit["raw_response"]) == wrapped_unit["review"]


def test_get_reviews_handles_non_dict_cassette_data():
    assert code_review_compare.get_reviews(None) == []
    assert code_review_compare.get_reviews([]) == []


def test_cqi_failure_collection_skips_recoverable_review_failures():
    recoverable_unit = {
        "file_path": "scripts/code_review_compare.py",
        "name": "<module>",
        "review": None,
        "recoverable_failure": {
            "type": "structured_output",
            "message": "Failed to validate structured output",
        },
    }

    normalized_unit = code_review_compare.normalize_review_unit(recoverable_unit, 0)
    reasons = code_review_compare.collect_cqi_failure_reasons([normalized_unit])

    assert reasons == []


def test_validate_path_escapes(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    
    # Valid relative path
    safe = diff_extractor.validate_path("scripts/test.py", root)
    assert safe == root / "scripts/test.py"

    # Escaping path
    with pytest.raises(ValueError, match="escapes allowed directory"):
        diff_extractor.validate_path("../outside.py", root)


def test_pydantic_validators():
    from scripts.finding_schema import EngineeringFinding, Evidence
    from scripts.code_review_evaluator import PropertyEvaluation
    from pydantic import ValidationError

    # 1. Test score limits (out of bounds raises ValidationError)
    with pytest.raises(ValidationError):
        PropertyEvaluation.model_validate({"score": 150, "rationale": "Over limit"})
    
    with pytest.raises(ValidationError):
        PropertyEvaluation.model_validate({"score": "N/A", "rationale": "Invalid"})

    prop_ok = PropertyEvaluation.model_validate({"score": 8.5, "rationale": "Ok"})
    assert prop_ok.score == pytest.approx(8.5)

    prop_percent = PropertyEvaluation.model_validate({"score": 85, "rationale": "Ok"})
    assert prop_percent.score == pytest.approx(8.5)

    # 2. Test confidence parsing (98 -> 0.98)
    finding = EngineeringFinding.model_validate({
        "title": "Test Finding",
        "category": "Correctness",
        "severity": "WARN",
        "evidence": {
            "location_type": "code",
            "path": "/main.py",
            "details": {"start_line": 1}
        },
        "engineering_rationale": "Some rationale",
        "engineering_consequence": "Some consequence",
        "confidence": 98,
        "recommended_action": "Fix it"
    })
    assert finding.confidence == pytest.approx(0.98)
    assert finding.evidence.path == "main.py"  # path leading slash lstrip

    # 3. Test confidence invalid rejections
    with pytest.raises(ValidationError):
        EngineeringFinding.model_validate({
            "title": "Test Finding",
            "category": "Correctness",
            "severity": "WARN",
            "evidence": {"location_type": "code", "path": "main.py", "details": {}},
            "engineering_rationale": "Some rationale",
            "engineering_consequence": "Some consequence",
            "confidence": "banana",
            "recommended_action": "Fix it"
        })

    # 4. Test empty strings are preserved (not fabricated)
    finding_empty = EngineeringFinding.model_validate({
        "title": "Test Finding",
        "category": "Correctness",
        "severity": "WARN",
        "evidence": {
            "location_type": "code",
            "path": "main.py",
            "details": {}
        },
        "engineering_rationale": "  ",
        "engineering_consequence": "",
        "confidence": 0.8,
        "recommended_action": ""
    })
    assert finding_empty.engineering_rationale == ""
    assert finding_empty.engineering_consequence == ""
    assert finding_empty.recommended_action == ""


def test_unit_review_artifact():
    from scripts.finding_schema import UnitReviewArtifact, build_validation_context
    from scripts.code_review_evaluator import CodeReviewBreakdown
    import json

    raw_json_str = """{
        "readability": {"score": 8.0, "rationale": "Good readability"},
        "maintainability": {"score": 7.5, "rationale": "Moderate maintainability"},
        "correctness": {"score": 9.0, "rationale": "Correct"},
        "complexity": {"score": 6.0, "rationale": "Ok complexity"},
        "security": {"score": 10.0, "rationale": "Secure"},
        "test_coverage": {"score": 8.0, "rationale": "High coverage"},
        "summary": "Overall good quality",
        "severity": "OK",
        "findings": [
            {
                "title": "Test issue",
                "category": "Style",
                "severity": "INFO",
                "evidence": {
                    "location_type": "code",
                    "path": "/src/foo.py"
                },
                "engineering_rationale": " Whitespace ",
                "engineering_consequence": "None",
                "confidence": 95,
                "recommended_action": "Trim"
            }
        ]
    }"""
    
    raw_dict = json.loads(raw_json_str)
    breakdown = CodeReviewBreakdown.model_validate(raw_dict)
    context = build_validation_context(raw_dict, breakdown)

    artifact = UnitReviewArtifact(
        file_path="src/foo.py",
        name="foo",
        class_name="Foo",
        review=breakdown,
        validation=context,
        raw_response=raw_json_str
    )

    # Verify serialization
    dumped = artifact.model_dump()
    assert dumped["file_path"] == "src/foo.py"
    assert dumped["name"] == "foo"
    assert dumped["class_name"] == "Foo"
    assert dumped["raw_response"] == raw_json_str
    assert dumped["validation"]["repaired"] is True
    assert dumped["validation"]["normalized"] is True

    # Assert specific field validations
    fields = {f["field_name"]: f for f in dumped["validation"]["fields"]}
    assert fields["findings[0].confidence"]["status"] == "REPAIRED"
    assert fields["findings[0].confidence"]["raw_value"] == 95
    assert fields["findings[0].confidence"]["repaired_value"] == pytest.approx(0.95)
    assert fields["findings[0].evidence.path"]["status"] == "NORMALIZED"
    assert fields["findings[0].evidence.path"]["raw_value"] == "/src/foo.py"
    assert fields["findings[0].evidence.path"]["repaired_value"] == "src/foo.py"


def test_code_review_breakdown_recovers_common_llm_output_mistakes():
    from scripts.finding_schema import build_validation_context
    from scripts.code_review_evaluator import CodeReviewBreakdown

    raw_dict = {
        "readability": {"score": 85, "rationale": "Good readability"},
        "maintainability": {"score": 90, "rationale": "Good maintainability"},
        "correctness": {"score": 75, "rationale": "Mostly correct"},
        "complexity": {"score": 70, "rationale": "Moderate complexity"},
        "security": {"score": 95, "rationale": "No security issue"},
        "test_coverage": {"score": 80, "rationale": "Covered"},
        "findings": [
            {
                "title": "Missing consequence",
                "category": "Maintainability",
                "severity": "WARN",
                "evidence": {"location_type": "code", "path": "src/foo.py", "details": {}},
                "engineering_rationale": "The rationale exists.",
                "confidence": 0.8,
                "recommended_action": "Add the missing consequence.",
            },
            {
                "title": "Complete finding",
                "category": "Readability",
                "severity": "INFO",
                "evidence": {"location_type": "code", "path": "/src/foo.py", "details": {}},
                "engineering_rationale": "A complete rationale.",
                "engineering_consequence": "Readers understand the concern.",
                "confidence": 90,
                "recommended_action": "Keep the finding complete.",
            },
        ],
    }

    breakdown = CodeReviewBreakdown.model_validate(raw_dict)
    context = build_validation_context(raw_dict, breakdown)

    assert breakdown.summary == ""
    assert breakdown.severity == "OK"
    assert breakdown.readability.score == pytest.approx(8.5)
    assert breakdown.maintainability.score == pytest.approx(9.0)
    assert len(breakdown.findings) == 1
    assert breakdown.findings[0].title == "Complete finding"

    fields = {field.field_name: field for field in context.fields}
    assert fields["summary"].status == "REPAIRED"
    assert fields["summary"].repair_type == "missing_default"
    assert fields["severity"].status == "REPAIRED"
    assert fields["severity"].repaired_value == "OK"
    assert fields["readability.score"].status == "REPAIRED"
    assert fields["readability.score"].repaired_value == pytest.approx(8.5)
    assert fields["findings[0]"].status == "REPAIRED"
    assert fields["findings[0]"].repair_type == "dropped_invalid_finding"
    # Field indexes refer to the original raw findings list, not the filtered parsed findings list.
    assert fields["findings[1].confidence"].status == "REPAIRED"
    assert fields["findings[1].evidence.path"].status == "NORMALIZED"
    assert context.repaired is True
    assert context.normalized is True


def test_code_review_breakdown_preserves_prebuilt_findings():
    from scripts.finding_schema import EngineeringFinding, build_validation_context
    from scripts.code_review_evaluator import CodeReviewBreakdown

    prebuilt_finding = EngineeringFinding.model_validate({
        "title": "Prebuilt finding",
        "category": "Readability",
        "severity": "INFO",
        "evidence": {"location_type": "code", "path": "src/foo.py", "details": {}},
        "engineering_rationale": "Already validated.",
        "engineering_consequence": "Preserving it avoids data loss.",
        "confidence": 0.9,
        "recommended_action": "Keep the validated object.",
    })
    valid_finding_dict = {
        "title": "Dictionary finding",
        "category": "Maintainability",
        "severity": "WARN",
        "evidence": {"location_type": "code", "path": "src/bar.py", "details": {}},
        "engineering_rationale": "Valid dict input.",
        "engineering_consequence": "Validation should happen once in the pre-validator.",
        "confidence": 90,
        "recommended_action": "Return the validated object.",
    }
    invalid_finding_dict = {
        "title": "Invalid finding",
        "category": "Correctness",
    }
    raw_dict = {
        "readability": {"score": 8, "rationale": "Good readability"},
        "maintainability": {"score": 9, "rationale": "Good maintainability"},
        "correctness": {"score": 7, "rationale": "Mostly correct"},
        "complexity": {"score": 7, "rationale": "Moderate complexity"},
        "security": {"score": 9, "rationale": "No security issue"},
        "test_coverage": {"score": 8, "rationale": "Covered"},
        "summary": "Valid review",
        "severity": "OK",
        "findings": [prebuilt_finding, valid_finding_dict, invalid_finding_dict, "bad"],
    }

    breakdown = CodeReviewBreakdown.model_validate(raw_dict)

    assert len(breakdown.findings) == 2
    assert breakdown.findings[0] == prebuilt_finding
    assert isinstance(breakdown.findings[1], EngineeringFinding)
    assert breakdown.findings[1].title == "Dictionary finding"
    assert breakdown.findings[1].confidence == pytest.approx(0.9)

    context = build_validation_context(raw_dict, breakdown)
    dropped_fields = [
        field for field in context.fields
        if field.repair_type == "dropped_invalid_finding"
    ]
    assert len(dropped_fields) == 2
    assert {field.field_name for field in dropped_fields} == {"findings[2]", "findings[3]"}
    assert "findings[0]" not in {field.field_name for field in context.fields}
    assert context.repaired is True


def test_code_review_breakdown_still_rejects_irreparable_reviews():
    from scripts.code_review_evaluator import CodeReviewBreakdown
    from pydantic import ValidationError

    raw_dict = {
        "readability": {"score": "not-a-score", "rationale": "Bad score"},
        "maintainability": {"score": 90, "rationale": "Good maintainability"},
        "correctness": {"score": 75, "rationale": "Mostly correct"},
        "complexity": {"score": 70, "rationale": "Moderate complexity"},
        "security": {"score": 95, "rationale": "No security issue"},
        "test_coverage": {"score": 80, "rationale": "Covered"},
    }

    with pytest.raises(ValidationError):
        CodeReviewBreakdown.model_validate(raw_dict)


def test_strict_type_coercion_regressions():
    from scripts.finding_schema import EngineeringFinding, Evidence
    from scripts.code_review_evaluator import PropertyEvaluation
    from pydantic import ValidationError

    # --- Score validator tests ---
    # Accept 5, 5.5, "5.5", and percentage-style scores.
    p1 = PropertyEvaluation.model_validate({"score": 5, "rationale": "ok"})
    assert p1.score == pytest.approx(5.0)
    p2 = PropertyEvaluation.model_validate({"score": 5.5, "rationale": "ok"})
    assert p2.score == pytest.approx(5.5)
    p3 = PropertyEvaluation.model_validate({"score": "5.5", "rationale": "ok"})
    assert p3.score == pytest.approx(5.5)
    p4 = PropertyEvaluation.model_validate({"score": 85, "rationale": "ok"})
    assert p4.score == pytest.approx(8.5)

    # Reject True, False, None, [], {}, "banana", and values outside 0-100.
    for invalid in [True, False, None, [], {}, "banana"]:
        with pytest.raises(ValidationError):
            PropertyEvaluation.model_validate({"score": invalid, "rationale": "ok"})
    with pytest.raises(ValidationError):
        PropertyEvaluation.model_validate({"score": 101, "rationale": "too high"})

    # --- Confidence validator tests ---
    # Helper to build raw finding structure
    def make_finding_dict(confidence_val, **kwargs):
        return {
            "title": kwargs.get("title", "Test Finding"),
            "category": kwargs.get("category", "Correctness"),
            "severity": "WARN",
            "evidence": {
                "location_type": "code",
                "path": "main.py",
                "details": {}
            },
            "engineering_rationale": kwargs.get("engineering_rationale", "Some rationale"),
            "engineering_consequence": kwargs.get("engineering_consequence", "Some consequence"),
            "confidence": confidence_val,
            "recommended_action": kwargs.get("recommended_action", "Fix it"),
        }

    # Accept 0.7, "0.7", 95
    payload_f1 = make_finding_dict(0.7)
    f1 = EngineeringFinding.model_validate(payload_f1)
    assert f1.confidence == pytest.approx(0.7)
    
    payload_f2 = make_finding_dict("0.7")
    f2 = EngineeringFinding.model_validate(payload_f2)
    assert f2.confidence == pytest.approx(0.7)
    
    payload_f3 = make_finding_dict(95)
    f3 = EngineeringFinding.model_validate(payload_f3)
    assert f3.confidence == pytest.approx(0.95)

    # Reject True, False, None, [], {}, "banana"
    for invalid in [True, False, None, [], {}, "banana"]:
        payload_invalid = make_finding_dict(invalid)
        with pytest.raises(ValidationError):
            EngineeringFinding.model_validate(payload_invalid)

    # --- Engineering text fields tests ---
    # Accept "" and "   text   " -> "text"
    payload_f4 = make_finding_dict(0.8, engineering_rationale="")
    f4 = EngineeringFinding.model_validate(payload_f4)
    assert f4.engineering_rationale == ""
    
    payload_f5 = make_finding_dict(0.8, engineering_rationale="   text   ")
    f5 = EngineeringFinding.model_validate(payload_f5)
    assert f5.engineering_rationale == "text"

    # Reject None, {}, [], 123, True for engineering_rationale
    for invalid in [None, {}, [], 123, True]:
        payload_rat = make_finding_dict(0.8, engineering_rationale=invalid)
        with pytest.raises(ValidationError):
            EngineeringFinding.model_validate(payload_rat)

        # Check other string fields: title, category, engineering_consequence, recommended_action
        payload_title = make_finding_dict(0.8, title=invalid)
        with pytest.raises(ValidationError):
            EngineeringFinding.model_validate(payload_title)
            
        payload_cat = make_finding_dict(0.8, category=invalid)
        with pytest.raises(ValidationError):
            EngineeringFinding.model_validate(payload_cat)
            
        payload_conseq = make_finding_dict(0.8, engineering_consequence=invalid)
        with pytest.raises(ValidationError):
            EngineeringFinding.model_validate(payload_conseq)
            
        payload_recom = make_finding_dict(0.8, recommended_action=invalid)
        with pytest.raises(ValidationError):
            EngineeringFinding.model_validate(payload_recom)

    # Reject non-string, None for Evidence location_type and path
    with pytest.raises(ValidationError):
        Evidence.model_validate({"location_type": None, "path": "main.py"})
    with pytest.raises(ValidationError):
        Evidence.model_validate({"location_type": 123, "path": "main.py"})
    with pytest.raises(ValidationError):
        Evidence.model_validate({"location_type": "code", "path": 123})
    
    # Path is Optional, so path=None is fine, but non-string like True, [], {} must fail
    ev_none_path = Evidence.model_validate({"location_type": "code", "path": None})
    assert ev_none_path.path is None
    for invalid_path in [True, [], {}]:
        with pytest.raises(ValidationError):
            Evidence.model_validate({"location_type": "code", "path": invalid_path})


def test_provenance_consistency():
    from scripts.finding_schema import UnitReviewArtifact, build_validation_context
    from scripts.code_review_evaluator import CodeReviewBreakdown
    import json

    raw_json_str = """{
        "readability": {"score": 8.0, "rationale": "  Whitespace trimmed readability  "},
        "maintainability": {"score": 7.5, "rationale": "Moderate maintainability"},
        "correctness": {"score": 9.0, "rationale": "Correct"},
        "complexity": {"score": 6.0, "rationale": "Ok complexity"},
        "security": {"score": 10.0, "rationale": "Secure"},
        "test_coverage": {"score": 8.0, "rationale": "High coverage"},
        "summary": "Overall good quality",
        "severity": "OK",
        "findings": [
            {
                "title": "Test issue",
                "category": "Style",
                "severity": "INFO",
                "evidence": {
                    "location_type": "code",
                    "path": "/src/foo.py"
                },
                "engineering_rationale": "  Some rationale with whitespace  ",
                "engineering_consequence": "None",
                "confidence": 95,
                "recommended_action": "Trim"
            }
        ]
    }"""
    
    raw_dict = json.loads(raw_json_str)
    breakdown = CodeReviewBreakdown.model_validate(raw_dict)
    context = build_validation_context(raw_dict, breakdown)

    artifact = UnitReviewArtifact(
        file_path="src/foo.py",
        name="foo",
        review=breakdown,
        validation=context,
        raw_response=raw_json_str
    )

    dumped = artifact.model_dump()
    assert dumped["raw_response"] == raw_json_str
    
    # Check that confidence was repaired to 0.95
    assert dumped["review"]["findings"][0]["confidence"] == pytest.approx(0.95)
    
    # Check that evidence path was normalized
    assert dumped["review"]["findings"][0]["evidence"]["path"] == "src/foo.py"
    
    # Check that readability rationale was normalized
    assert dumped["review"]["readability"]["rationale"] == "Whitespace trimmed readability"

    # Verify that the validation fields map contains only the modified/repaired/normalized fields (Option A)
    fields = {f["field_name"]: f for f in dumped["validation"]["fields"]}
    
    # Verify confidence was repaired
    assert fields["findings[0].confidence"]["status"] == "REPAIRED"
    assert fields["findings[0].confidence"]["raw_value"] == 95
    assert fields["findings[0].confidence"]["repaired_value"] == pytest.approx(0.95)
    
    # Verify path was normalized
    assert fields["findings[0].evidence.path"]["status"] == "NORMALIZED"
    assert fields["findings[0].evidence.path"]["raw_value"] == "/src/foo.py"
    assert fields["findings[0].evidence.path"]["repaired_value"] == "src/foo.py"
    
    # Verify readability rationale was normalized
    assert fields["readability.rationale"]["status"] == "NORMALIZED"
    assert fields["readability.rationale"]["raw_value"] == "  Whitespace trimmed readability  "
    assert fields["readability.rationale"]["repaired_value"] == "Whitespace trimmed readability"

    # Option A constraint check: ensure that valid fields (like correctness.score, readability.score, etc.) are NOT present
    assert "correctness.score" not in fields
    assert "findings[0].severity" not in fields


def test_telemetry_aggregation():
    from scripts.code_review_evaluator import build_validation_summary
    
    dummy_results = [
        {
            "file_path": "a.py",
            "name": "a",
            "validation": {
                "repaired": True,
                "normalized": True,
                "fields": [
                    {"field_name": "findings[0].confidence", "status": "REPAIRED", "raw_value": 95, "repaired_value": 0.95},
                    {"field_name": "findings[0].evidence.path", "status": "NORMALIZED", "raw_value": "/a.py", "repaired_value": "a.py"},
                    {"field_name": "readability.rationale", "status": "NORMALIZED", "raw_value": "  foo  ", "repaired_value": "foo"},
                    {
                        "field_name": "summary",
                        "status": "REPAIRED",
                        "raw_value": None,
                        "repaired_value": "",
                        "repair_type": "missing_default",
                    },
                    {
                        "field_name": "findings[1]",
                        "status": "REPAIRED",
                        "raw_value": {},
                        "repaired_value": None,
                        "repair_type": "dropped_invalid_finding",
                    },
                ]
            }
        },
        {
            "file_path": "b.py",
            "name": "b",
            "validation": {
                "repaired": False,
                "normalized": False,
                "fields": [
                    {"field_name": "correctness.score", "status": "VALID", "raw_value": 8.0, "repaired_value": 8.0}
                ]
            }
        }
    ]
    
    summary = build_validation_summary(dummy_results)
    assert summary["total_units"] == 2
    assert summary["valid_units"] == 1
    assert summary["repaired_units"] == 1
    assert summary["normalized_units"] == 0
    assert summary["details"]["repaired_confidence_count"] == 1
    assert summary["details"]["repaired_score_count"] == 0
    assert summary["details"]["repaired_default_count"] == 1
    assert summary["details"]["dropped_finding_count"] == 1
    assert summary["details"]["normalized_path_count"] == 1
    assert summary["details"]["normalized_text_count"] == 1


def test_validation_summary_terminal_states_are_exclusive():
    from scripts.code_review_evaluator import build_validation_summary

    def unit(repaired=False, normalized=False, fields=None):
        return {
            "file_path": "x.py",
            "name": "x",
            "validation": {
                "repaired": repaired,
                "normalized": normalized,
                "fields": fields or [],
            },
        }

    invalid_field = {
        "field_name": "llm_response",
        "status": "INVALID",
        "raw_value": "bad",
        "repaired_value": None,
    }
    repaired_field = {
        "field_name": "findings[0].confidence",
        "status": "REPAIRED",
        "raw_value": 95,
        "repaired_value": 0.95,
    }
    normalized_field = {
        "field_name": "findings[0].evidence.path",
        "status": "NORMALIZED",
        "raw_value": "/x.py",
        "repaired_value": "x.py",
    }

    summary = build_validation_summary([
        unit(),
        unit(fields=[invalid_field]),
        unit(repaired=True, fields=[repaired_field]),
        unit(normalized=True, fields=[normalized_field]),
        unit(repaired=True, fields=[invalid_field, repaired_field]),
        unit(normalized=True, fields=[invalid_field, normalized_field]),
    ])

    assert summary["total_units"] == 6
    assert summary["valid_units"] == 1
    assert summary["repaired_units"] == 1
    assert summary["normalized_units"] == 1
    assert summary["invalid_units"] == 3
    assert (
        summary["valid_units"]
        + summary["repaired_units"]
        + summary["normalized_units"]
        + summary["invalid_units"]
    ) == summary["total_units"]
    assert summary["details"]["invalid_field_count"] == 3
    assert summary["details"]["repaired_confidence_count"] == 2
    assert summary["details"]["normalized_path_count"] == 2


def test_structured_output_telemetry_aggregation():
    from scripts.code_review_evaluator import build_validation_summary

    summary = build_validation_summary([
        {
            "file_path": "a.py",
            "name": "a",
            "validation": None,
            "structured_output": {
                "invalid_json_detected": True,
                "repair_attempts": 1,
                "repair_succeeded": True,
                "validation_retries": 0,
                "final_failure": False,
            },
        },
        {
            "file_path": "b.py",
            "name": "b",
            "validation": {"repaired": False, "normalized": False, "fields": []},
            "structured_output": {
                "invalid_json_detected": True,
                "repair_attempts": 2,
                "repair_succeeded": False,
                "validation_retries": 2,
                "final_failure": True,
            },
        },
    ])

    structured = summary["details"]["structured_output"]
    assert summary["total_units"] == 2
    assert summary["valid_units"] == 1
    assert summary["invalid_units"] == 1
    assert structured["invalid_json_detected"] == 2
    assert structured["repair_attempts"] == 3
    assert structured["repair_successes"] == 1
    assert structured["validation_retries"] == 2
    assert structured["final_failures"] == 1


@pytest.mark.asyncio
async def test_evaluate_units_marks_structured_output_failure_recoverable(monkeypatch):
    diagnostics = {
        "invalid_json_detected": True,
        "repair_attempts": 1,
        "repair_succeeded": False,
        "validation_retries": 2,
        "final_failure": True,
        "failure_reason": "invalid_json: Unterminated string",
    }

    async def fake_call_structured_with_raw_and_diagnostics(**kwargs):
        raise code_review_evaluator.llm_adapter.StructuredOutputError(
            "CodeReviewBreakdown",
            "Failed to parse LLM response as JSON before CodeReviewBreakdown validation",
            '{"summary": "truncated',
            diagnostics,
        )

    monkeypatch.setattr(
        code_review_evaluator.llm_adapter,
        "call_structured_with_raw_and_diagnostics",
        fake_call_structured_with_raw_and_diagnostics,
    )

    units = [{
        "name": "broken",
        "file_path": "scripts/broken.py",
        "class_name": "Broken",
        "start_line": 1,
        "end_line": 1,
        "lines_changed": 1,
        "code": "def broken(): pass",
    }]

    results = await code_review_evaluator.evaluate_units(None, "litellm/foo", units)

    assert len(results) == 1
    assert results[0]["class_name"] == "Broken"
    assert results[0]["review"] is None
    assert results[0]["recoverable_failure"]["type"] == "structured_output"
    assert results[0]["validation"]["fields"][0]["status"] == "INVALID"
    assert results[0]["structured_output"]["final_failure"] is True
