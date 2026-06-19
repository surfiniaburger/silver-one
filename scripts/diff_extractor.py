import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def validate_path(path: str, root: Path) -> Path:
    """Validate that a path does not escape the allowed root directory."""
    p = Path(path)
    if p.is_absolute():
        candidate = p.resolve()
    else:
        candidate = (root / path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError(f"Path escapes allowed directory: {path}")
    return candidate


class FunctionVisitor(ast.NodeVisitor):
    """AST Visitor to extract functions, classes, line ranges and source code."""

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.functions: List[Dict[str, Any]] = []
        self.class_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        code_segment = ast.get_source_segment(self.source_code, node)

        self.functions.append({
            "name": node.name,
            "class_name": ".".join(self.class_stack) if self.class_stack else None,
            "code": code_segment or "",
            "start_line": start,
            "end_line": end,
        })
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def _parse_range(range_str: str) -> List[int]:
    """Parse a range string like +14,5 or +14 and return a list of line numbers."""
    if not range_str.startswith("+"):
        return []
    try:
        range_parts = range_str[1:].split(",")
        start = int(range_parts[0])
        count = int(range_parts[1]) if len(range_parts) > 1 else 1
        if count == 0:
            return [start]
        return list(range(start, start + count))
    except (ValueError, IndexError):
        return []


def _parse_diff_output(diff_output: str) -> Dict[str, List[int]]:
    """Parse raw git diff output to map file paths to changed line numbers."""
    changed_files: Dict[str, List[int]] = {}
    current_file: str = ""

    for line in diff_output.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            changed_files[current_file] = []
        elif line.startswith("@@ ") and current_file:
            parts = line.split(" ")
            if len(parts) >= 3:
                changed_lines = _parse_range(parts[2])
                changed_files[current_file].extend(changed_lines)

    return changed_files


def get_changed_lines(base_ref: str = "origin/main", cwd: Path = PROJECT_ROOT) -> Dict[str, List[int]]:
    """
    Run git diff base_ref --unified=0 and parse the line ranges that changed.
    Returns a dict mapping relative file paths (as strings) to lists of changed line numbers.
    """
    cmd = ["git", "diff", base_ref, "--unified=0", "--", "*.py"]
    try:
        diff_output = subprocess.check_output(cmd, text=True, cwd=str(cwd))
    except subprocess.CalledProcessError as exc:
        print(f"Error running git diff: {exc}", file=sys.stderr)
        return {}

    return _parse_diff_output(diff_output)


def extract_units_from_file(
    filepath: Path, changed_lines: List[int], project_root: Path = PROJECT_ROOT
) -> List[Dict[str, Any]]:
    """
    Parse a Python file and return a list of changed function/module units.
    """
    try:
        with filepath.open("r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return []

    try:
        tree = ast.parse(content)
        visitor = FunctionVisitor(content)
        visitor.visit(tree)
    except SyntaxError as e:
        print(f"Syntax error parsing {filepath}: {e}", file=sys.stderr)
        return []

    try:
        file_path_str = str(filepath.relative_to(project_root))
    except ValueError:
        file_path_str = str(filepath)

    units: List[Dict[str, Any]] = []
    covered_lines: Set[int] = set()

    for fn in visitor.functions:
        fn_lines = set(range(fn["start_line"], fn["end_line"] + 1))
        intersect = fn_lines.intersection(changed_lines)
        if intersect:
            units.append({
                "file_path": file_path_str,
                "name": fn["name"],
                "class_name": fn["class_name"],
                "code": fn["code"],
                "start_line": fn["start_line"],
                "end_line": fn["end_line"],
                "lines_changed": len(intersect),
            })
            covered_lines.update(fn_lines)

    # Check for uncovered changed lines (e.g., module-level changes)
    uncovered = set(changed_lines) - covered_lines
    if uncovered:
        lines = content.splitlines()
        units.append({
            "file_path": file_path_str,
            "name": "<module>",
            "class_name": None,
            "code": content,
            "start_line": 1,
            "end_line": len(lines),
            "lines_changed": len(uncovered),
        })

    return units


def get_all_changed_units(base_ref: str = "origin/main", cwd: Path = PROJECT_ROOT) -> List[Dict[str, Any]]:
    """
    Extract all changed units across the git diff.
    """
    changed_lines = get_changed_lines(base_ref, cwd)
    all_units: List[Dict[str, Any]] = []

    for rel_path, lines in changed_lines.items():
        try:
            safe_path = validate_path(rel_path, cwd)
            if not safe_path.exists():
                continue
            all_units.extend(extract_units_from_file(safe_path, lines, cwd))
        except ValueError as exc:
            print(f"Skipping invalid path '{rel_path}': {exc}", file=sys.stderr)

    return all_units
