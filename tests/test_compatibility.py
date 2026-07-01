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
