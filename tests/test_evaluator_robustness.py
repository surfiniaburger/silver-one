import sys
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from scripts.farley_score_evaluator import (
    _parse_and_validate_args,
    _resolve_target_files,
    _filter_changed_test_files,
    _build_test_id,
    extract_tests_from_file,
    TEST_ROOT,
    PROJECT_ROOT,
)


def test_build_test_id():
    """Verify test ID generation covers both with and without class names."""
    id_without_class = _build_test_id("tests/test_demo.py", None, "test_func")
    assert id_without_class == "tests/test_demo.py::test_func"

    id_with_class = _build_test_id("tests/test_demo.py", "TestClass", "test_func")
    assert id_with_class == "tests/test_demo.py::TestClass::test_func"


def test_parse_and_validate_args_valid():
    """Verify that parsing valid command line args works as expected."""
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--cassette", type=str)
    parser.add_argument("--run-id", type=str)

    # We mock validate_path and sanitize_run_id
    mock_path = Path("/mock/cassettes/farley_score.json")
    with patch("scripts.farley_score_evaluator.validate_path", return_value=mock_path) as mock_val, \
         patch("scripts.farley_score_evaluator.sanitize_run_id", return_value="farley_run") as mock_san:
        args, cassette_path, safe_run_id = _parse_and_validate_args(
            parser,
            ["tests/", "--cassette", "farley_score.json", "--run-id", "farley-run"]
        )
        assert safe_run_id == "farley_run"
        assert cassette_path == mock_path
        mock_val.assert_called_once()
        mock_san.assert_called_once_with("farley-run")


def test_parse_and_validate_args_invalid():
    """Verify that invalid arguments cause sys.exit(1)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--cassette", type=str, default="")
    parser.add_argument("--run-id", type=str, default="")

    with pytest.raises(SystemExit) as excinfo:
        _parse_and_validate_args(parser, ["--cassette", "", "--run-id", ""])
    assert excinfo.value.code == 1


def test_filter_changed_test_files_containment(tmp_path: Path):
    """Verify _filter_changed_test_files properly filters out files outside of TEST_ROOT."""
    # Create test files inside and outside TEST_ROOT
    inside_test_dir = TEST_ROOT
    outside_dir = PROJECT_ROOT / "scripts"

    inside_test_file = inside_test_dir / "test_dummy_inside.py"
    outside_file = outside_dir / "test_dummy_outside.py"

    # Make sure parent directories exist (TEST_ROOT is already created, but let's be safe)
    inside_test_dir.mkdir(parents=True, exist_ok=True)
    outside_dir.mkdir(parents=True, exist_ok=True)

    inside_test_file.touch()
    outside_file.touch()

    try:
        # Create raw_changed dictionary mapping rel_path -> line list
        # To get relative path, we compute them relative to PROJECT_ROOT
        rel_inside = str(inside_test_file.relative_to(PROJECT_ROOT))
        rel_outside = str(outside_file.relative_to(PROJECT_ROOT))

        raw_changed = {
            rel_inside: [10, 15],
            rel_outside: [2, 5]
        }

        filtered = _filter_changed_test_files(raw_changed)

        # The inside test file should be present in the filtered map (under absolute string path)
        assert str(inside_test_file.resolve()) in filtered
        # The outside test file must be filtered out because it's not inside TEST_ROOT
        assert str(outside_file.resolve()) not in filtered

    finally:
        # Cleanup
        if inside_test_file.exists():
            inside_test_file.unlink()
        if outside_file.exists():
            outside_file.unlink()


def test_resolve_target_files_full_suite():
    """Verify full-suite resolution mode."""
    args = MagicMock()
    args.base = None
    args.paths = ["tests/"]

    mock_files = ["/absolute/tests/test_demo.py"]
    with patch("scripts.farley_score_evaluator.find_target_files", return_value=mock_files) as mock_find:
        target_files, changed_lines = _resolve_target_files(args, Path("dummy.json"))
        assert target_files == mock_files
        assert changed_lines is None
        mock_find.assert_called_once_with(["tests/"])


def test_resolve_target_files_diff_empty_or_no_changes():
    """Verify that when no changes are found, evaluator exits gracefully with 0."""
    args = MagicMock()
    args.base = "origin/main"

    # Mock _get_changed_lines to return empty dict
    with patch("scripts.farley_score_evaluator._DIFF_AVAILABLE", True), \
         patch("scripts.farley_score_evaluator._get_changed_lines", return_value={}) as mock_get_changed, \
         patch("scripts.farley_score_evaluator.save_farley_cassette") as mock_save:
        with pytest.raises(SystemExit) as excinfo:
            _resolve_target_files(args, Path("dummy.json"))
        assert excinfo.value.code == 0
        mock_get_changed.assert_called_once_with("origin/main", PROJECT_ROOT)
        mock_save.assert_called_once_with(Path("dummy.json"), [])


def test_extract_tests_from_file_range_optimization(tmp_path: Path):
    """Verify optimized diff-intersection logic in extract_tests_from_file."""
    p = tmp_path / "test_range_sample.py"
    # Create test content with known lines:
    # Line 1: empty
    # Line 2: def test_one():
    # Line 3:     pass
    # Line 4:
    # Line 5: def test_two():
    # Line 6:     pass
    p.write_text("\ndef test_one():\n    pass\n\ndef test_two():\n    pass\n")

    # Mock validate_path to return this temp file and pretend TEST_ROOT is the temp path
    with patch("scripts.farley_score_evaluator.validate_path", return_value=p):
        # Scenario A: Changed line is 2 (inside test_one)
        tests_one = extract_tests_from_file(str(p), changed_lines=[2])
        names_one = [t["name"] for t in tests_one]
        assert "test_one" in names_one
        assert "test_two" not in names_one

        # Scenario B: Changed line is 5 (inside test_two)
        tests_two = extract_tests_from_file(str(p), changed_lines=[5])
        names_two = [t["name"] for t in tests_two]
        assert "test_two" in names_two
        assert "test_one" not in names_two

        # Scenario C: Changed lines has no overlap (e.g. line 10)
        tests_none = extract_tests_from_file(str(p), changed_lines=[10])
        assert len(tests_none) == 0

        # Scenario D: changed_lines is None -> full suite (both extracted)
        tests_all = extract_tests_from_file(str(p), changed_lines=None)
        names_all = [t["name"] for t in tests_all]
        assert "test_one" in names_all
        assert "test_two" in names_all
