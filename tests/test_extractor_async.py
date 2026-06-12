import textwrap
from pathlib import Path
from scripts.farley_score_evaluator import extract_tests_from_file


def test_async_test_is_extracted(tmp_path: Path):
    p = tmp_path / "test_async_sample.py"
    content = textwrap.dedent('''
    import pytest

    async def test_async_behavior():
        assert True

    def test_sync():
        assert True
    ''')
    p.write_text(content)
    tests = extract_tests_from_file(str(p))
    names = [t['name'] for t in tests]
    assert 'test_async_behavior' in names
    assert 'test_sync' in names
