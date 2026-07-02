#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable

# Enable relative imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.diff_extractor import get_changed_lines
from scripts.path_utils import validate_path, validate_output_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_ROOT = (PROJECT_ROOT / "artifacts" / "metrics").resolve()
METRICS_ROOT.mkdir(parents=True, exist_ok=True)

CLASS_PREFIX = "class:"


def extract_api_signatures(code: str) -> Dict[str, Any]:
    """
    Extract public functions, classes, and methods from Python code AST.
    Returns a dict mapping signature keys (e.g., 'foo' or 'MyClass.bar')
    to information about parameters and types.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    signatures: Dict[str, Any] = {}

    class APISignatureVisitor(ast.NodeVisitor):
        def __init__(self):
            self.class_stack: List[str] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            if not node.name.startswith("_"):
                self.class_stack.append(node.name)
                class_key = CLASS_PREFIX + ".".join(self.class_stack)
                signatures[class_key] = {"type": "class"}
                self.generic_visit(node)
                self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            if not node.name.startswith("_"):
                prefix = ".".join(self.class_stack) + "." if self.class_stack else ""
                func_name = prefix + node.name
                
                args = node.args
                params = []
                
                # Check for posonlyargs attribute (Python 3.8+)
                posonlyargs = getattr(args, "posonlyargs", [])
                posonly_count = len(posonlyargs)
                args_count = len(args.args)
                
                total_pos_args = posonly_count + args_count
                defaults_count = len(args.defaults)
                first_default_idx = total_pos_args - defaults_count

                def get_pos_default(idx: int) -> bool:
                    return idx >= first_default_idx

                idx = 0
                for arg in posonlyargs:
                    params.append({
                        "name": arg.arg,
                        "kind": "positional_only",
                        "has_default": get_pos_default(idx)
                    })
                    idx += 1

                for arg in args.args:
                    params.append({
                        "name": arg.arg,
                        "kind": "positional_or_keyword",
                        "has_default": get_pos_default(idx)
                    })
                    idx += 1

                # keyword-only arguments and defaults
                kwonlyargs = getattr(args, "kwonlyargs", [])
                kw_defaults = getattr(args, "kw_defaults", [])
                for arg, default in zip(kwonlyargs, kw_defaults):
                    params.append({
                        "name": arg.arg,
                        "kind": "keyword_only",
                        "has_default": default is not None
                    })

                if args.vararg:
                    params.append({
                        "name": args.vararg.arg,
                        "kind": "var_positional",
                        "has_default": True
                    })

                if args.kwarg:
                    params.append({
                        "name": args.kwarg.arg,
                        "kind": "var_keyword",
                        "has_default": True
                    })

                signatures[func_name] = {
                    "type": "function",
                    "params": params
                }
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    APISignatureVisitor().visit(tree)
    return signatures


def _is_safe_git_ref(ref: str) -> bool:
    # Git references can contain alphanumeric characters, slashes, dashes, underscores, and dots.
    # Must not start with a dash.
    return bool(re.match(r"^[a-zA-Z0-9_/.-]+$", ref)) and not ref.startswith("-")


def _is_safe_relative_path(path: str) -> bool:
    # A safe relative path should only contain alphanumeric characters, slashes, dashes, underscores, dots.
    # Must not start with a dash, must not be absolute, and must not escape the repository (no ".." components).
    if path.startswith("/") or path.startswith("-") or ".." in path:
        return False
    return bool(re.match(r"^[a-zA-Z0-9_/.-]+$", path))


def get_base_file_content(base_ref: str, rel_path: str, cwd: Path) -> Optional[str]:
    """Retrieve file content from base_ref branch using git show."""
    if not _is_safe_git_ref(base_ref):
        raise ValueError(f"Unsafe git ref: {base_ref}")
    if not _is_safe_relative_path(rel_path):
        raise ValueError(f"Unsafe relative path: {rel_path}")

    try:
        cmd = ["git", "show", f"{base_ref}:{rel_path}"]
        return subprocess.check_output(cmd, text=True, cwd=str(cwd), stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def _check_missing_or_changed_type(key: str, base_val: Dict[str, Any], pr_sigs: Dict[str, Any], file_path: str) -> Optional[str]:
    if key not in pr_sigs:
        if key.startswith(CLASS_PREFIX):
            class_name = key.split(CLASS_PREFIX)[1]
            return f"{file_path}: Deleted public class `{class_name}`"
        else:
            return f"{file_path}: Deleted public function/method `{key}`"

    pr_val = pr_sigs[key]
    if base_val["type"] != pr_val["type"]:
        return f"{file_path}: Changed type of `{key}` from `{base_val['type']}` to `{pr_val['type']}`"
    return None


def _check_function_parameter_regressions(
    key: str,
    base_val: Dict[str, Any],
    pr_val: Dict[str, Any],
    file_path: str,
) -> List[str]:
    regressions = []
    base_params = base_val["params"]
    pr_params = pr_val["params"]

    # Map parameters by name
    base_param_map = {p["name"]: p for p in base_params}
    pr_param_map = {p["name"]: p for p in pr_params}

    # 1. Check for removed/renamed parameters or kind changes, or default value removals
    for base_p in base_params:
        name = base_p["name"]
        if name not in pr_param_map:
            regressions.append(f"{file_path}: In `{key}`, removed or renamed parameter `{name}`")
            continue

        pr_p = pr_param_map[name]
        if base_p["kind"] != pr_p["kind"]:
            regressions.append(
                f"{file_path}: In `{key}`, parameter `{name}` changed kind from "
                f"`{base_p['kind']}` to `{pr_p['kind']}`"
            )
        elif base_p["has_default"] and not pr_p["has_default"]:
            regressions.append(
                f"{file_path}: In `{key}`, parameter `{name}` removed its default value"
            )

    # 2. Check for added parameters that don't have defaults
    for pr_p in pr_params:
        name = pr_p["name"]
        if name not in base_param_map and not pr_p["has_default"]:
            regressions.append(f"{file_path}: In `{key}`, added parameter `{name}` without a default value")

    # 3. Check for changed parameter order for positional arguments
    base_pos = [p["name"] for p in base_params if p["kind"] in ("positional_only", "positional_or_keyword")]
    pr_pos = [p["name"] for p in pr_params if p["kind"] in ("positional_only", "positional_or_keyword")]

    base_pos_in_pr = [p for p in pr_pos if p in base_pos]
    expected_pos_in_pr = [p for p in base_pos if p in pr_pos]
    if base_pos_in_pr != expected_pos_in_pr:
        regressions.append(f"{file_path}: In `{key}`, positional parameter ordering was altered")
    elif base_pos_in_pr:
        last_active_base = base_pos_in_pr[-1]
        last_active_idx = pr_pos.index(last_active_base)
        if any(p not in base_pos for p in pr_pos[:last_active_idx]):
            regressions.append(f"{file_path}: In `{key}`, a new positional parameter was inserted before existing ones")

    return regressions


def compare_signatures(base_sigs: Dict[str, Any], pr_sigs: Dict[str, Any], file_path: str) -> List[str]:
    """Compare two sets of API signatures and return any compatibility regressions."""
    regressions = []

    for key, base_val in base_sigs.items():
        type_reg = _check_missing_or_changed_type(key, base_val, pr_sigs, file_path)
        if type_reg:
            regressions.append(type_reg)
            continue

        pr_val = pr_sigs[key]
        if base_val["type"] == "function":
            func_regs = _check_function_parameter_regressions(key, base_val, pr_val, file_path)
            regressions.extend(func_regs)

    return regressions


def _filter_source_files(changed_files: Dict[str, Any]) -> List[str]:
    files_to_check = []
    for rel_path in changed_files.keys():
        p = Path(rel_path)
        if p.suffix == ".py" and not (p.name.startswith("test_") or p.name.endswith("_test.py")):
            files_to_check.append(rel_path)
    return files_to_check


def _check_deleted_file_regressions(rel_path: str, base_content: str) -> List[str]:
    regressions = []
    base_sigs = extract_api_signatures(base_content)
    for key in base_sigs.keys():
        if key.startswith(CLASS_PREFIX):
            regressions.append(f"{rel_path}: Deleted public class `{key.split(CLASS_PREFIX)[1]}`")
        else:
            regressions.append(f"{rel_path}: Deleted public function/method `{key}`")
    return regressions


def _check_file_compatibility(base_ref: str, rel_path: str, cwd: Path) -> List[str]:
    base_content = get_base_file_content(base_ref, rel_path, cwd)
    if base_content is None:
        return []

    try:
        safe_path = validate_path(rel_path, cwd)
        if not safe_path.exists():
            return _check_deleted_file_regressions(rel_path, base_content)
        
        with safe_path.open("r", encoding="utf-8") as f:
            pr_content = f.read()
    except Exception as e:
        print(f"Warning: could not read {rel_path}: {e}", file=sys.stderr)
        return []

    base_sigs = extract_api_signatures(base_content)
    pr_sigs = extract_api_signatures(pr_content)
    return compare_signatures(base_sigs, pr_sigs, rel_path)


def run_compatibility_check(base_ref: str, cwd: Path = PROJECT_ROOT) -> Tuple[bool, List[str], float]:
    """Run AST compatibility check over changed files."""
    try:
        changed_files = get_changed_lines(base_ref, cwd)
    except ValueError as exc:
        print(f"Error extracting git diff: {exc}", file=sys.stderr)
        return False, [str(exc)], 0.0

    all_regressions = []
    files_to_check = _filter_source_files(changed_files)

    for rel_path in files_to_check:
        file_regs = _check_file_compatibility(base_ref, rel_path, cwd)
        all_regressions.extend(file_regs)

    is_compatible = len(all_regressions) == 0
    compatibility_index = max(0.0, 10.0 - 2.0 * len(all_regressions)) if not is_compatible else 10.0

    return is_compatible, all_regressions, compatibility_index


def main():
    parser = argparse.ArgumentParser(description="MSEC API Compatibility Gate")
    parser.add_argument("--base", type=str, default="origin/main", help="Git reference to compare against")
    parser.add_argument("--out", type=str, default="artifacts/metrics/compatibility_results.json", help="Output JSON filename relative to project root")
    args = parser.parse_args()

    try:
        out_path = validate_output_path(args.out, PROJECT_ROOT, {".json"})
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    is_ok, regressions, score = run_compatibility_check(args.base, PROJECT_ROOT)

    results = {
        "pass": is_ok,
        "regressions": regressions,
        "compatibility_index": score,
    }

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as exc:
        print(f"Error saving compatibility results: {exc}", file=sys.stderr)

    if regressions:
        print(f"\033[91mCompatibility Check: FAILED ({len(regressions)} regression(s) found)\033[0m")
        for reg in regressions:
            print(f"  - {reg}")
    else:
        print("\033[92mCompatibility Check: PASSED\033[0m")

    # We exit 0 so that unified_compare can aggregate the verdict and report in detail.
    sys.exit(0)


if __name__ == "__main__":
    main()
