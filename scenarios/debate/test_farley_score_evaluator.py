import os
import tempfile
import sys

# Ensure src and scripts directories are in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))

from farley_score_evaluator import extract_tests_from_file, FarleyScoreBreakdown, PropertyEvaluation

def test_ast_extractor_finds_tests():
    sample_code = """
import pytest

def test_single_function():
    assert 1 + 1 == 2

class TestMathOperations:
    def test_add(self):
        assert True
        
    def test_sub(self):
        assert False

def helper_method():
    pass
"""
    with tempfile.TemporaryDirectory() as td:
        filepath = os.path.join(td, "test_sample.py")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sample_code)
        
        extracted = extract_tests_from_file(filepath)
        
        assert len(extracted) == 3
        
        names = [tc["name"] for tc in extracted]
        assert "test_single_function" in names
        assert "test_add" in names
        assert "test_sub" in names
        
        # Verify class contexts are correct
        for tc in extracted:
            if tc["name"] == "test_single_function":
                assert tc["class_name"] is None
            elif tc["name"] in ("test_add", "test_sub"):
                assert tc["class_name"] == "TestMathOperations"

def test_ast_extractor_fallback_for_no_tests():
    sample_code = """
# Just a simple script with no standard test functions
x = 10
y = 20
print(x + y)
"""
    with tempfile.TemporaryDirectory() as td:
        filepath = os.path.join(td, "simple_script.py")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sample_code)
        
        extracted = extract_tests_from_file(filepath)
        
        # Should fallback to treating the whole file as a test case
        assert len(extracted) == 1
        assert extracted[0]["name"] == "simple_script.py"
        assert "x = 10" in extracted[0]["code"]

def test_farley_score_breakdown_schema():
    # Test valid parsing of FarleyScoreBreakdown from JSON
    raw_json = """
    {
        "understandable": {"score": 9, "rationale": "Clear test name.", "suggestions": []},
        "maintainable": {"score": 8, "rationale": "Decoupled.", "suggestions": []},
        "repeatable": {"score": 10, "rationale": "Pure function.", "suggestions": []},
        "atomic": {"score": 9, "rationale": "Single assert.", "suggestions": []},
        "necessary": {"score": 10, "rationale": "Unique behavioral check.", "suggestions": []},
        "granular": {"score": 9, "rationale": "Targets single add operation.", "suggestions": []},
        "fast": {"score": 10, "rationale": "Executes instantly.", "suggestions": []},
        "first_tdd": {"score": 8, "rationale": "Clean AAA style.", "suggestions": []},
        "summary": "Excellent unit test."
    }
    """
    report = FarleyScoreBreakdown.model_validate_json(raw_json)
    assert report.understandable.score == 9
    assert report.repeatable.score == 10
    assert report.summary == "Excellent unit test."

if __name__ == "__main__":
    test_ast_extractor_finds_tests()
    test_ast_extractor_fallback_for_no_tests()
    test_farley_score_breakdown_schema()
    print("ok")

