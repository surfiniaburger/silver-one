import pytest
from scripts.compatibility_check import extract_api_signatures, compare_signatures


def test_extract_api_signatures_simple():
    code = """
def public_fn(a, b=2):
    pass

def _private_fn(x):
    pass

class MyClass:
    def method(self, x, *, y=3):
        pass
    
    def _private_method(self):
        pass

class _PrivateClass:
    def method(self):
        pass
"""
    sigs = extract_api_signatures(code)

    # Public function
    assert "public_fn" in sigs
    assert sigs["public_fn"]["type"] == "function"
    params = sigs["public_fn"]["params"]
    assert len(params) == 2
    assert params[0]["name"] == "a"
    assert not params[0]["has_default"]
    assert params[1]["name"] == "b"
    assert params[1]["has_default"]

    # Private function ignored
    assert "_private_fn" not in sigs

    # Public class and its public method
    assert "class:MyClass" in sigs
    assert sigs["class:MyClass"]["type"] == "class"
    assert "MyClass.method" in sigs
    assert sigs["MyClass.method"]["type"] == "function"
    
    method_params = sigs["MyClass.method"]["params"]
    assert len(method_params) == 3
    assert method_params[0]["name"] == "self"
    assert method_params[1]["name"] == "x"
    assert method_params[2]["name"] == "y"
    assert method_params[2]["kind"] == "keyword_only"
    assert method_params[2]["has_default"]

    # Private methods and private classes ignored
    assert "MyClass._private_method" not in sigs
    assert "class:_PrivateClass" not in sigs
    assert "_PrivateClass.method" not in sigs


def test_compare_signatures_no_changes():
    base = extract_api_signatures("def fn(a, b=1): pass")
    pr = extract_api_signatures("def fn(a, b=1): pass")
    regs = compare_signatures(base, pr, "test.py")
    assert len(regs) == 0


def test_compare_signatures_deleted_class():
    base = extract_api_signatures("class A: pass")
    pr = extract_api_signatures("")
    regs = compare_signatures(base, pr, "test.py")
    assert len(regs) == 1
    assert "Deleted public class `A`" in regs[0]


def test_compare_signatures_deleted_function():
    base = extract_api_signatures("def fn(): pass")
    pr = extract_api_signatures("")
    regs = compare_signatures(base, pr, "test.py")
    assert len(regs) == 1
    assert "Deleted public function/method `fn`" in regs[0]


def test_compare_signatures_removed_parameter():
    base = extract_api_signatures("def fn(a, b): pass")
    pr = extract_api_signatures("def fn(a): pass")
    regs = compare_signatures(base, pr, "test.py")
    assert len(regs) == 1
    assert "removed or renamed parameter `b`" in regs[0]


def test_compare_signatures_renamed_parameter():
    base = extract_api_signatures("def fn(a): pass")
    pr = extract_api_signatures("def fn(b): pass")
    regs = compare_signatures(base, pr, "test.py")
    assert len(regs) == 2
    assert any("removed or renamed parameter `a`" in r for r in regs)


def test_compare_signatures_added_parameter_no_default():
    base = extract_api_signatures("def fn(a): pass")
    pr = extract_api_signatures("def fn(a, b): pass")
    regs = compare_signatures(base, pr, "test.py")
    assert len(regs) == 1
    assert "added parameter `b` without a default value" in regs[0]


def test_compare_signatures_added_parameter_with_default():
    base = extract_api_signatures("def fn(a): pass")
    pr = extract_api_signatures("def fn(a, b=2): pass")
    regs = compare_signatures(base, pr, "test.py")
    assert len(regs) == 0


def test_compare_signatures_order_changed():
    base = extract_api_signatures("def fn(a, b): pass")
    pr = extract_api_signatures("def fn(b, a): pass")
    regs = compare_signatures(base, pr, "test.py")
    # This triggers both a kind/rename/order difference or order altered
    assert len(regs) > 0
    assert any("positional parameter ordering" in r for r in regs)


def test_compare_signatures_removed_default():
    base = extract_api_signatures("def fn(a=1): pass")
    pr = extract_api_signatures("def fn(a): pass")
    regs = compare_signatures(base, pr, "test.py")
    assert len(regs) == 1
    assert "parameter `a` removed its default value" in regs[0]


def test_compare_signatures_positional_inserted_before_existing():
    base = extract_api_signatures("def fn(b): pass")
    pr = extract_api_signatures("def fn(a=1, b=2): pass")
    regs = compare_signatures(base, pr, "test.py")
    assert len(regs) == 1
    assert "new positional parameter was inserted before existing ones" in regs[0]


def test_get_base_file_content_validators():
    from scripts.compatibility_check import get_base_file_content, _is_safe_git_ref, _is_safe_relative_path
    from pathlib import Path
    
    assert _is_safe_git_ref("main")
    assert _is_safe_git_ref("origin/main")
    assert _is_safe_git_ref("feature-branch-1.2.3")
    assert not _is_safe_git_ref("-flag")
    assert not _is_safe_git_ref("main; rm -rf /")

    assert _is_safe_relative_path("scripts/compatibility_check.py")
    assert not _is_safe_relative_path("/absolute/path")
    assert not _is_safe_relative_path("-flag")
    assert not _is_safe_relative_path("scripts/../../escaped")

    path_val = Path(".")
    with pytest.raises(ValueError, match="Unsafe git ref"):
        get_base_file_content("-unsafe", "scripts/compatibility_check.py", path_val)

    with pytest.raises(ValueError, match="Unsafe relative path"):
        get_base_file_content("main", "/unsafe/path", path_val)
