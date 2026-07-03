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
        PropertyEvaluation.model_validate({"score": 15, "rationale": "Over limit"})
    
    with pytest.raises(ValidationError):
        PropertyEvaluation.model_validate({"score": "N/A", "rationale": "Invalid"})

    prop_ok = PropertyEvaluation.model_validate({"score": 8.5, "rationale": "Ok"})
    assert prop_ok.score == pytest.approx(8.5)

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

