"""
AST Graph Data-Flow Extractor Module.
Parses code attempt strings using Python's ast module and produces deterministic FlowGraphSnapshot objects.
Implements the technical specification in docs/SPEC_GRAPH_EXTRACTOR.md.
"""

import ast
import math
from typing import Dict, List, Optional, Set, Tuple

import tree_sitter
import tree_sitter_c

from scenarios.debate.graph_dataflow import (
    SUPPORTED_SINKS,
    VALID_SANITIZERS,
    FlowGraphSnapshot,
    FlowSignature,
)


SYSTEM_CALL_FUNCTIONS = {"os.system", "subprocess.Popen", "subprocess.run", "eval", "exec", "system", "Popen", "run"}
MEMORY_FUNCTIONS = {"memcpy", "strcpy", "memset", "memmove"}
EXPLICIT_INPUT_SOURCES = {"input", "sys.argv", "request.args", "request.get_json", "socket.recv", "file.read"}
COMMAND_SANITIZER_FUNCTIONS = {"shlex.quote", "quote"}


def _is_finite_numeric(val: float) -> bool:
    """Returns True if val is a finite int or float (excluding bool and non-numeric types)."""
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        return math.isfinite(val)
    return False


class SecurityFlowVisitor(ast.NodeVisitor):
    """
    AST Visitor that identifies data-flow sources, sinks, and enclosing guard sanitizers.
    Strictly enforces operand identity and branch dominance bindings.
    """

    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.signatures: List[FlowSignature] = []
        self.sources: Dict[str, str] = {}  # var_name -> source_node_id
        self.guard_stack: List[Dict[str, Set[str]]] = []
        self.node_counter = 0

    def _gen_id(self, prefix: str) -> str:
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._register_function_args(node)
        self._visit_statement_block(node.body)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _register_function_args(self, node: ast.AST):
        args_node = getattr(node, "args", None)
        if not args_node:
            return
        all_args = []
        all_args.extend(getattr(args_node, "posonlyargs", []))
        all_args.extend(getattr(args_node, "args", []))
        if getattr(args_node, "vararg", None):
            all_args.append(args_node.vararg)
        all_args.extend(getattr(args_node, "kwonlyargs", []))
        if getattr(args_node, "kwarg", None):
            all_args.append(args_node.kwarg)

        for arg in all_args:
            arg_name = arg.arg
            src_id = self._gen_id("src")
            self.nodes[src_id] = {
                "kind": "source",
                "label": f"Param({arg_name})",
                "type": "UNTRUSTED_INPUT",
                "target_var": arg_name,
                "source_kind": "function_parameter",
                **self._location_metadata(arg),
            }
            self.sources[arg_name] = src_id

    def visit_Module(self, node: ast.Module):
        self._visit_statement_block(node.body)

    def _visit_statement_block(self, statements: List[ast.stmt]):
        block_guards: Dict[str, Set[str]] = {}
        self.guard_stack.append(block_guards)
        try:
            for stmt in statements:
                if isinstance(stmt, ast.Assert):
                    self.visit(stmt.test)
                    self._merge_guards(block_guards, self._collect_guard_condition(stmt.test))
                    continue
                self.visit(stmt)
        finally:
            self.guard_stack.pop()

    def _merge_guards(self, target: Dict[str, Set[str]], source: Dict[str, Set[str]]):
        for var_name, sanitizer_types in source.items():
            target.setdefault(var_name, set()).update(sanitizer_types)

    def _record_current_guard(self, var_name: str, sanitizer_type: str):
        if not self.guard_stack:
            self.guard_stack.append({})
        self.guard_stack[-1].setdefault(var_name, set()).add(sanitizer_type)

    def _active_guards_for(self, var_name: str) -> Set[str]:
        guards: Set[str] = set()
        for guard_frame in self.guard_stack:
            guards.update(guard_frame.get(var_name, set()))
        return guards

    def _location_metadata(self, node: ast.AST) -> dict:
        return {
            "lineno": getattr(node, "lineno", None),
            "col_offset": getattr(node, "col_offset", None),
            "ast_type": type(node).__name__,
        }

    def _clear_guards_for_rebound_targets(self, targets: List[ast.expr]):
        if not self.guard_stack:
            return
        for target in targets:
            var_name = self._extract_var_name(target)
            if var_name:
                for frame in self.guard_stack:
                    frame.pop(var_name, None)

    def visit_Assign(self, node: ast.Assign):
        self._clear_guards_for_rebound_targets(node.targets)
        self._track_explicit_input_assignment(node)
        self._track_sanitizer_call_assignment(node)
        self._propagate_source_bindings(node)
        self._check_memory_write_assignment(node)
        self.generic_visit(node)

    def _track_explicit_input_assignment(self, node: ast.Assign):
        if not isinstance(node.value, ast.Call):
            return
        func_name = self._get_call_name(node.value)
        is_explicit = func_name in EXPLICIT_INPUT_SOURCES or any(func_name.endswith(src) for src in EXPLICIT_INPUT_SOURCES)
        if not is_explicit:
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                src_id = self._gen_id("src")
                self.nodes[src_id] = {
                    "kind": "source",
                    "label": f"InputSource({func_name})",
                    "type": "UNTRUSTED_INPUT",
                    "target_var": target.id,
                    "source_kind": "explicit_input",
                    **self._location_metadata(target),
                }
                self.sources[target.id] = src_id

    def _track_sanitizer_call_assignment(self, node: ast.Assign):
        if not isinstance(node.value, ast.Call):
            return
        func_name = self._get_call_name(node.value)
        callee = func_name.rsplit(".", 1)[-1]
        if callee not in COMMAND_SANITIZER_FUNCTIONS and func_name not in COMMAND_SANITIZER_FUNCTIONS:
            return
        for arg in node.value.args:
            arg_var = self._extract_var_name(arg)
            if not arg_var or arg_var not in self.sources:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._record_current_guard(target.id, "COMMAND_SANITIZATION")

    def _propagate_source_bindings(self, node: ast.Assign):
        for v in self._collect_assigned_value_vars(node.value):
            if v in self.sources:
                self._bind_targets_to_sources(node.targets, self.sources[v])

    def _collect_assigned_value_vars(self, node_value: ast.AST) -> List[str]:
        vars_found = []
        if isinstance(node_value, ast.Call):
            for arg in node_value.args:
                arg_name = self._extract_var_name(arg)
                if arg_name:
                    vars_found.append(arg_name)
        val_var = self._extract_var_name(node_value)
        if val_var:
            vars_found.append(val_var)
        return vars_found

    def _bind_targets_to_sources(self, targets: List[ast.expr], source_id: str):
        for target in targets:
            if isinstance(target, ast.Name):
                self.sources[target.id] = source_id

    def _check_memory_write_assignment(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                setattr(target, "_is_assign_target", True)
                self._check_memory_write(target, node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._clear_guards_for_rebound_targets([node.target])
        if isinstance(node.target, ast.Subscript):
            setattr(node.target, "_is_assign_target", True)
            self._check_memory_write(node.target, node.value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if getattr(node, "_is_assign_target", False):
            self.generic_visit(node)
            return

        idx_var = self._extract_var_name(node.slice)
        base_var = self._extract_var_name(node.value)
        source_id = self.sources.get(idx_var) or self.sources.get(base_var)

        if idx_var and source_id:
            sink_id = self._gen_id("sink")
            self.nodes[sink_id] = {
                "kind": "sink",
                "label": f"ArrayIndex({idx_var})",
                "type": "ARRAY_INDEX",
                "sink_expr_kind": "subscript_index",
                "target_var": idx_var,
                **self._location_metadata(node),
            }
            sanitizer_type, guarded_target = self._resolve_sanitizer("ARRAY_INDEX", idx_var)
            _add_signature(
                self.signatures,
                source_id=source_id,
                sink_id=sink_id,
                sink_type="ARRAY_INDEX",
                flow_type="SUBSCRIPT_INDEX",
                sanitizer_type=sanitizer_type,
                guarded_target=guarded_target,
            )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        base_var = self._extract_var_name(node.value)
        if base_var and base_var in self.sources:
            source_id = self.sources[base_var]
            sink_id = self._gen_id("sink")
            self.nodes[sink_id] = {
                "kind": "sink",
                "label": f"PointerDeref({base_var}.{node.attr})",
                "type": "POINTER_DEREF",
                "sink_expr_kind": "attribute_access",
                "target_var": base_var,
                **self._location_metadata(node),
            }
            sanitizer_type, guarded_target = self._resolve_sanitizer("POINTER_DEREF", base_var)
            _add_signature(
                self.signatures,
                source_id=source_id,
                sink_id=sink_id,
                sink_type="POINTER_DEREF",
                flow_type="ATTRIBUTE_ACCESS",
                sanitizer_type=sanitizer_type,
                guarded_target=guarded_target,
            )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = self._get_call_name(node)
        self._check_system_call_sink(node, func_name)
        self._check_memory_call_sink(node, func_name)
        self.generic_visit(node)

    def _check_system_call_sink(self, node: ast.Call, func_name: str):
        is_sys_fn = func_name in SYSTEM_CALL_FUNCTIONS or any(func_name.endswith("." + sys_fn) for sys_fn in SYSTEM_CALL_FUNCTIONS)
        if not is_sys_fn:
            return
        for arg in node.args:
            cmd_var = self._extract_var_name(arg)
            if cmd_var and cmd_var in self.sources:
                source_id = self.sources[cmd_var]
                sink_id = self._gen_id("sink")
                self.nodes[sink_id] = {
                    "kind": "sink",
                    "label": f"SystemCall({func_name})",
                    "type": "SYSTEM_CALL",
                    "sink_expr_kind": "command_execution",
                    "target_var": cmd_var,
                    **self._location_metadata(node),
                }
                sanitizer_type, guarded_target = self._resolve_sanitizer("SYSTEM_CALL", cmd_var)
                _add_signature(
                    self.signatures,
                    source_id=source_id,
                    sink_id=sink_id,
                    sink_type="SYSTEM_CALL",
                    flow_type="COMMAND_EXECUTION",
                    sanitizer_type=sanitizer_type,
                    guarded_target=guarded_target,
                )

    def _check_memory_call_sink(self, node: ast.Call, func_name: str):
        if func_name not in MEMORY_FUNCTIONS or len(node.args) < 2:
            return
        dest_var = self._extract_var_name(node.args[0])
        src_var = self._extract_var_name(node.args[1])

        size_var = None
        if len(node.args) >= 3:
            size_var = self._extract_var_name(node.args[2])

        source_id = self.sources.get(src_var) or self.sources.get(dest_var) or (self.sources.get(size_var) if size_var else None)
        target_var = size_var or dest_var

        if source_id and target_var:
            sink_id = self._gen_id("sink")
            self.nodes[sink_id] = {
                "kind": "sink",
                "label": f"MemoryWrite({func_name})",
                "type": "MEMORY_WRITE",
                "sink_expr_kind": "memory_copy_call",
                "target_var": target_var,
                **self._location_metadata(node),
            }
            sanitizer_type, guarded_target = self._resolve_sanitizer("MEMORY_WRITE", target_var)
            _add_signature(
                self.signatures,
                source_id=source_id,
                sink_id=sink_id,
                sink_type="MEMORY_WRITE",
                flow_type="MEMORY_COPY_CALL",
                sanitizer_type=sanitizer_type,
                guarded_target=guarded_target,
            )

    def visit_If(self, node: ast.If):
        self.visit(node.test)
        guard_map = self._collect_guard_condition(node.test)
        self.guard_stack.append(guard_map)
        try:
            self._visit_statement_block(node.body)
        finally:
            self.guard_stack.pop()

        if node.orelse:
            self._visit_statement_block(node.orelse)

    def _check_memory_write(self, target: ast.Subscript, value: ast.AST):
        idx_var = self._extract_var_name(target.slice)
        val_var = self._extract_var_name(value)
        base_var = self._extract_var_name(target.value)

        source_id = self.sources.get(val_var) or self.sources.get(idx_var) or self.sources.get(base_var)
        if source_id:
            sink_id = self._gen_id("sink")
            target_var = idx_var or base_var
            self.nodes[sink_id] = {
                "kind": "sink",
                "label": f"MemoryWrite({base_var}[{idx_var}])",
                "type": "MEMORY_WRITE",
                "sink_expr_kind": "subscript_write",
                "target_var": target_var,
                **self._location_metadata(target),
            }
            sanitizer_type, guarded_target = self._resolve_sanitizer("MEMORY_WRITE", target_var)
            _add_signature(
                self.signatures,
                source_id=source_id,
                sink_id=sink_id,
                sink_type="MEMORY_WRITE",
                flow_type="SUBSCRIPT_WRITE",
                sanitizer_type=sanitizer_type,
                guarded_target=guarded_target,
            )

    def _collect_guard_condition(self, test: ast.AST) -> Dict[str, Set[str]]:
        if isinstance(test, ast.Compare):
            return self._analyze_compare_guard(test)
        if isinstance(test, ast.Name):
            return {test.id: {"NULL_CHECK"}}
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            var = self._extract_var_name(test.operand)
            if var:
                return {var: {"NULL_CHECK"}}
        return {}

    def _analyze_compare_guard(self, test: ast.Compare) -> Dict[str, Set[str]]:
        guards: Dict[str, Set[str]] = {}
        operand_vars = [self._extract_var_name(test.left)] + [self._extract_var_name(c) for c in test.comparators]
        is_chained = len(test.ops) > 1

        for idx, op in enumerate(test.ops):
            self._process_compare_operator(op, idx, operand_vars, is_chained, guards)

        return guards

    def _process_compare_operator(
        self,
        op: ast.cmpop,
        idx: int,
        operand_vars: List[Optional[str]],
        is_chained: bool,
        guards: Dict[str, Set[str]],
    ):
        if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            self._add_inequality_guard(idx, operand_vars, is_chained, guards)
        elif isinstance(op, (ast.IsNot, ast.NotEq)):
            self._add_single_target_guard(operand_vars[idx] or operand_vars[idx + 1], "NULL_CHECK", guards)
        elif isinstance(op, ast.In):
            self._add_single_target_guard(operand_vars[idx] or operand_vars[idx + 1], "ALLOWLIST_CHECK", guards)

    def _add_single_target_guard(self, target_var: Optional[str], sanitizer_type: str, guards: Dict[str, Set[str]]):
        if target_var:
            guards.setdefault(target_var, set()).add(sanitizer_type)

    def _add_inequality_guard(
        self,
        idx: int,
        operand_vars: List[Optional[str]],
        is_chained: bool,
        guards: Dict[str, Set[str]],
    ):
        if is_chained:
            target_var = operand_vars[1] if len(operand_vars) >= 2 else None
            self._add_single_target_guard(target_var, "RANGE_VALIDATION", guards)
        else:
            target_var = operand_vars[idx] or operand_vars[idx + 1]
            self._add_single_target_guard(target_var, "BOUNDS_CHECK", guards)

    def _extract_var_name(self, node: Optional[ast.AST]) -> Optional[str]:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._extract_var_name(node.value)
        if isinstance(node, ast.Subscript):
            return self._extract_var_name(node.value)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "len" and node.args:
                return self._extract_var_name(node.args[0])
            return self._extract_var_name(node.func)
        return None

    def _get_call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            val_name = self._extract_var_name(node.func.value)
            return f"{val_name}.{node.func.attr}" if val_name else node.func.attr
        return ""

    def _resolve_sanitizer(self, sink_type: str, target_var: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not target_var:
            return None, None

        guards = self._active_guards_for(target_var)

        if sink_type in ("MEMORY_WRITE", "ARRAY_INDEX"):
            if "RANGE_VALIDATION" in guards:
                return "RANGE_VALIDATION", target_var
            if "BOUNDS_CHECK" in guards:
                return "BOUNDS_CHECK", target_var

        if sink_type == "POINTER_DEREF" and "NULL_CHECK" in guards:
            return "NULL_CHECK", target_var

        if sink_type == "SYSTEM_CALL":
            if "COMMAND_SANITIZATION" in guards:
                return "COMMAND_SANITIZATION", target_var
            if "ALLOWLIST_CHECK" in guards:
                return "ALLOWLIST_CHECK", target_var

        return None, None

def _add_signature(
    signatures: List[FlowSignature],
    source_id: str,
    sink_id: str,
    sink_type: str,
    flow_type: str,
    sanitizer_type: Optional[str] = None,
    guarded_target: Optional[str] = None,
):
    if sink_type not in SUPPORTED_SINKS:
        raise ValueError(f"Emitted sink_type '{sink_type}' is not in SUPPORTED_SINKS")
    if sanitizer_type and sanitizer_type not in VALID_SANITIZERS:
        raise ValueError(f"Emitted sanitizer_type '{sanitizer_type}' is not in VALID_SANITIZERS")

    signatures.append(
        FlowSignature(
            source_id=source_id,
            sink_id=sink_id,
            source_type="UNTRUSTED_INPUT",
            sink_type=sink_type,
            flow_type=flow_type,
            sanitizer_type=sanitizer_type,
            guarded_target=guarded_target,
        )
    )


import re


def _strip_markdown_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _is_parseable(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except Exception:
        return False


def _replace_c_keywords_outside_quotes(text: str) -> str:
    parts = re.split(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', text)
    transformed_parts = []
    for part in parts:
        if part.startswith(('"', "'")):
            transformed_parts.append(part)
        else:
            w_sub = re.sub(r"\bNULL\b", "None", part)
            w_sub = re.sub(r"\btrue\b", "True", w_sub)
            w_sub = re.sub(r"\bfalse\b", "False", w_sub)
            transformed_parts.append(w_sub)
    return "".join(transformed_parts)


def _advance_quote_state(char: str, in_quote: Optional[str], escaped: bool) -> Tuple[Optional[str], bool]:
    if escaped:
        return in_quote, False
    if char == "\\":
        return in_quote, True
    if in_quote:
        return (None, False) if char == in_quote else (in_quote, False)
    if char in ("'", '"'):
        return char, False
    return None, False


def _split_c_comment_quote_aware(line_str: str) -> Tuple[str, Optional[str]]:
    in_quote = None
    escaped = False
    n = len(line_str)

    for i in range(n):
        char = line_str[i]
        if not in_quote and not escaped and char == "/" and i + 1 < n and line_str[i + 1] == "/":
            return line_str[:i], line_str[i + 2 :]
        in_quote, escaped = _advance_quote_state(char, in_quote, escaped)

    return line_str, None


def _transform_tokens_outside_quotes(line_str: str) -> str:
    """
    Strips C-style line comments (//) and replaces C keywords (NULL, true, false)
    strictly outside single/double-quoted string literals.
    """
    code_part, comment_part = _split_c_comment_quote_aware(line_str)
    unquoted_code = _replace_c_keywords_outside_quotes(code_part.rstrip())

    if comment_part is not None:
        comment_prefix = "  # " if unquoted_code else "# "
        return unquoted_code + comment_prefix + comment_part

    return unquoted_code


def _transform_c_function_header(line_str: str) -> str:
    if line_str.startswith(("int ", "void ", "char ")) and "(" in line_str and ")" in line_str:
        header, rest = line_str.split(")", 1)
        parts = header.split("(", 1)
        func_name = parts[0].split()[-1].replace("*", "")
        args_str = parts[1]
        clean_args = [
            arg.strip().split()[-1].replace("*", "")
            for arg in args_str.split(",")
            if arg.strip().split()
        ]
        body_part = rest.strip()
        return f"def {func_name}({', '.join(clean_args)}):" + (f" {body_part}" if body_part else "")
    return line_str


def _transform_c_if_condition(line_str: str) -> str:
    if line_str.startswith("if (") and line_str.endswith(")"):
        return "if " + line_str[4:-1] + ":"
    if line_str.startswith("if (") and ")" in line_str and line_str.endswith("):"):
        return "if " + line_str[4:-2] + ":"
    return line_str


def _transform_c_var_declaration(line_str: str) -> str:
    c_types = ("char ", "int ", "float ", "double ", "void ", "const ")
    if line_str.startswith(c_types) and "=" in line_str and not line_str.startswith(("if ", "def ")):
        parts = line_str.split("=", 1)
        var_token = parts[0].strip().split()[-1].replace("*", "")
        return f"{var_token} = {parts[1].strip()}"
    return line_str


def _transform_single_c_line(stripped: str) -> Tuple[str, bool, bool]:
    has_open_brace = "{" in stripped
    line_str = _transform_tokens_outside_quotes(stripped)
    line_str = line_str.replace("{", "").replace("}", "").strip()

    had_semicolon = line_str.endswith(";")
    if had_semicolon:
        line_str = line_str[:-1].strip()

    line_str = _transform_c_if_condition(line_str)
    line_str = _transform_c_function_header(line_str)
    line_str = _transform_c_var_declaration(line_str)
    return line_str, had_semicolon, has_open_brace


def _apply_c_syntax_transformations(text: str) -> str:
    lines = text.splitlines()
    norm_lines: List[str] = []
    indent_level = 0

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            norm_lines.append("")
            continue

        if stripped.startswith("}"):
            indent_level = max(0, indent_level - 1)

        line_str, had_semicolon, has_open_brace = _transform_single_c_line(stripped)

        indent = "    " * indent_level
        norm_lines.append(f"{indent}{line_str}" if line_str else "")

        if line_str.endswith(":") and had_semicolon:
            norm_lines.append(f"{indent}    pass")

        if has_open_brace and not stripped.startswith("}"):
            indent_level += 1

    return "\n".join(norm_lines)


def _wrap_in_candidate_wrapper(text: str) -> str:
    lines = text.splitlines()
    wrapped = ["def candidate_wrapper():"] + [f"    {line_str}" if line_str else "" for line_str in lines]
    wrapped_text = "\n".join(wrapped)
    return wrapped_text if _is_parseable(wrapped_text) else text


def normalize_code_for_ast(code_text: str) -> str:
    """
    Normalizes raw snippet text into valid Python AST-parseable source code.
    Strips markdown code fences, translates C/C++ keywords and comments into Python equivalents,
    and wraps loose top-level statements inside a candidate wrapper function.
    """
    if not code_text or not code_text.strip():
        return ""

    text = _strip_markdown_fences(code_text.strip())
    if _is_parseable(text):
        return text

    processed_text = _apply_c_syntax_transformations(text)
    if _is_parseable(processed_text):
        return processed_text

    return _wrap_in_candidate_wrapper(processed_text)


C_LANGUAGE = tree_sitter.Language(tree_sitter_c.language())
C_PARSER = tree_sitter.Parser(C_LANGUAGE)


class TreeSitterFlowVisitor:
    """
    Traverses Tree-Sitter C AST nodes and extracts sources, sinks, and flow signatures.
    """

    def __init__(self, code_bytes: bytes):
        self.code_bytes = code_bytes
        self.nodes: Dict[str, dict] = {}
        self.signatures: List[FlowSignature] = []
        self.node_counter = 0
        self.guard_stack: List[Dict[str, Set[str]]] = []

    def _gen_id(self, prefix: str) -> str:
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def _get_node_text(self, node: tree_sitter.Node) -> str:
        return self.code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()

    def visit(self, node: tree_sitter.Node):
        method_name = f"visit_{node.type}"
        visitor_fn = getattr(self, method_name, self.generic_visit)
        visitor_fn(node)

    def generic_visit(self, node: tree_sitter.Node):
        for child in node.children:
            self.visit(child)

    def visit_parameter_declaration(self, node: tree_sitter.Node):
        param_name = self._find_first_identifier(node)
        if param_name:
            param_id = self._gen_id(f"src_ts_{param_name}")
            if param_id not in self.nodes:
                self.nodes[param_id] = {
                    "id": param_id,
                    "kind": "source",
                    "label": f"Param({param_name})",
                    "type": "UNTRUSTED_INPUT",
                    "target_var": param_name,
                    "source_kind": "function_parameter",
                    "lineno": node.start_point[0] + 1,
                    "col_offset": node.start_point[1],
                }
        self.generic_visit(node)

    def _find_first_identifier(self, node: tree_sitter.Node) -> Optional[str]:
        if node.type == "identifier":
            return self._get_node_text(node)
        for child in node.children:
            found = self._find_first_identifier(child)
            if found:
                return found
        return None

    def visit_call_expression(self, node: tree_sitter.Node):
        func_name = self._extract_func_name(node)
        if func_name:
            self._handle_call_sink(node, func_name)
        self.generic_visit(node)

    def _extract_func_name(self, node: tree_sitter.Node) -> Optional[str]:
        first_child = node.child_by_field_name("function") or (node.children[0] if node.children else None)
        if first_child:
            return self._get_node_text(first_child)
        return None

    def _handle_call_sink(self, node: tree_sitter.Node, func_name: str):
        sink_type = self._determine_sink_type(func_name)
        if not sink_type:
            return

        arg_list = node.child_by_field_name("arguments") or self._find_child_of_type(node, "argument_list")
        target_var = self._extract_first_arg_var(arg_list) if arg_list else None

        sink_id = self._gen_id(f"sink_ts_{func_name}")
        self.nodes[sink_id] = {
            "id": sink_id,
            "kind": "sink",
            "type": sink_type,
            "target_var": target_var,
            "sink_expr_kind": "call_expression",
        }

        flow_type = "COMMAND_EXECUTION" if sink_type == "SYSTEM_CALL" else "MEMORY_COPY_CALL"
        sanitizer_type, guarded_target = self._resolve_sanitizer(sink_type, target_var)

        arg_vars = self._extract_identifiers_from_node(arg_list) if arg_list else set()
        for src_id, src in self.nodes.items():
            if src.get("kind") == "source" and src.get("target_var") in arg_vars:
                _add_signature(
                    self.signatures,
                    source_id=src_id,
                    sink_id=sink_id,
                    sink_type=sink_type,
                    flow_type=flow_type,
                    sanitizer_type=sanitizer_type,
                    guarded_target=guarded_target,
                )

    def _determine_sink_type(self, func_name: str) -> Optional[str]:
        if func_name in {"open", "fopen", "system", "popen", "execl", "execv", "execve", "execvp"}:
            return "SYSTEM_CALL"
        if func_name in {"memcpy", "memmove", "memset", "strcpy", "strcat", "sprintf"}:
            return "MEMORY_WRITE"
        return None

    def _find_child_of_type(self, node: tree_sitter.Node, node_type: str) -> Optional[tree_sitter.Node]:
        for child in node.children:
            if child.type == node_type:
                return child
        return None

    def _extract_first_arg_var(self, arg_list: tree_sitter.Node) -> Optional[str]:
        for child in arg_list.children:
            if child.type in ("identifier", "field_expression", "pointer_declarator"):
                return self._get_node_text(child)
            id_text = self._find_first_identifier(child)
            if id_text and id_text not in ("(", ")", ","):
                return id_text
        return None

    def _active_guards_for(self, target_var: str) -> Set[str]:
        active: Set[str] = set()
        for frame in reversed(self.guard_stack):
            if target_var in frame:
                active.update(frame[target_var])
        return active

    def _resolve_sanitizer(self, sink_type: str, target_var: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not target_var:
            return None, None

        guards = self._active_guards_for(target_var)

        if sink_type in ("MEMORY_WRITE", "ARRAY_INDEX"):
            if "RANGE_VALIDATION" in guards:
                return "RANGE_VALIDATION", target_var
            if "BOUNDS_CHECK" in guards:
                return "BOUNDS_CHECK", target_var

        if sink_type in ("POINTER_DEREF", "SYSTEM_CALL"):
            if "COMMAND_SANITIZATION" in guards:
                return "COMMAND_SANITIZATION", target_var
            if "ALLOWLIST_CHECK" in guards:
                return "ALLOWLIST_CHECK", target_var
            if "NULL_CHECK" in guards:
                return "NULL_CHECK", target_var

        return None, None

    def _build_if_guard_map(self, cond_node: Optional[tree_sitter.Node]) -> Dict[str, Set[str]]:
        guard_map: Dict[str, Set[str]] = {}
        if not cond_node:
            return guard_map

        cond_text = self._get_node_text(cond_node)
        identifiers = self._extract_identifiers_from_node(cond_node)

        if any(op in cond_text for op in ("<", ">", "<=", ">=")):
            for var_name in identifiers:
                guard_map.setdefault(var_name, set()).add("BOUNDS_CHECK")

        if "NULL" in cond_text or "== 0" in cond_text or "!= 0" in cond_text or "None" in cond_text:
            for var_name in identifiers:
                guard_map.setdefault(var_name, set()).add("NULL_CHECK")

        return guard_map

    def visit_if_statement(self, node: tree_sitter.Node):
        cond_node = node.child_by_field_name("condition") or self._find_child_of_type(node, "parenthesized_expression")
        guard_map = self._build_if_guard_map(cond_node)

        self.guard_stack.append(guard_map)
        try:
            self._visit_if_consequence(node)
        finally:
            self.guard_stack.pop()

        self._visit_if_alternative(node)

    def _visit_if_consequence(self, node: tree_sitter.Node):
        consequence = node.child_by_field_name("consequence") or (node.children[2] if len(node.children) > 2 else None)
        if consequence:
            self.visit(consequence)
        else:
            self.generic_visit(node)

    def _visit_if_alternative(self, node: tree_sitter.Node):
        alternative = node.child_by_field_name("alternative")
        if alternative:
            self.visit(alternative)

    def _extract_identifiers_from_node(self, node: tree_sitter.Node) -> Set[str]:
        ids: Set[str] = set()
        if node.type == "identifier":
            ids.add(self._get_node_text(node))
        for child in node.children:
            ids.update(self._extract_identifiers_from_node(child))
        return ids


def extract_flow_graph_snapshot_treesitter(
    code_text: str,
    scenario_id: str,
    snapshot_id: str,
    version: int,
    created_at: float,
) -> Optional[FlowGraphSnapshot]:
    """
    Parses code_text using Tree-Sitter (C grammar) and extracts FlowGraphSnapshot.
    """
    if not code_text or not code_text.strip():
        return None

    clean_text = _strip_markdown_fences(code_text.strip())
    code_bytes = clean_text.encode("utf-8")

    try:
        tree = C_PARSER.parse(code_bytes)
        if tree.root_node.has_error:
            return None

        visitor = TreeSitterFlowVisitor(code_bytes)
        visitor.visit(tree.root_node)

        if not visitor.signatures:
            return None

        return FlowGraphSnapshot(
            snapshot_id=snapshot_id,
            scenario_id=scenario_id,
            version=version,
            created_at=float(created_at),
            nodes=visitor.nodes,
            signatures=visitor.signatures,
            is_complete=True,
            parse_error=None,
        )
    except Exception:
        return None


def extract_flow_graph_snapshot(
    code_text: str,
    scenario_id: str,
    snapshot_id: str,
    version: int,
    created_at: float,
) -> FlowGraphSnapshot:
    """
    Parses code_text using AST inspection and builds a deterministic FlowGraphSnapshot.
    Falls back to Tree-Sitter multi-language parser when native Python AST parsing fails.
    """
    if not _is_finite_numeric(created_at):
        return FlowGraphSnapshot(
            snapshot_id=snapshot_id,
            scenario_id=scenario_id,
            version=version,
            created_at=0.0,
            nodes={},
            signatures=[],
            is_complete=False,
            parse_error="Invalid non-numeric or non-finite created_at timestamp",
        )

    normalized_text = normalize_code_for_ast(code_text)

    try:
        tree = ast.parse(normalized_text)
        visitor = SecurityFlowVisitor()
        visitor.visit(tree)

        for sig in visitor.signatures:
            if sig.source_id not in visitor.nodes or sig.sink_id not in visitor.nodes:
                return FlowGraphSnapshot(
                    snapshot_id=snapshot_id,
                    scenario_id=scenario_id,
                    version=version,
                    created_at=float(created_at),
                    nodes={},
                    signatures=[],
                    is_complete=False,
                    parse_error="Signature endpoint missing from node registry",
                )

        return FlowGraphSnapshot(
            snapshot_id=snapshot_id,
            scenario_id=scenario_id,
            version=version,
            created_at=float(created_at),
            nodes=visitor.nodes,
            signatures=visitor.signatures,
            is_complete=True,
            parse_error=None,
        )

    except ValueError:
        raise

    except Exception as exc:
        ts_snapshot = extract_flow_graph_snapshot_treesitter(
            code_text=code_text,
            scenario_id=scenario_id,
            snapshot_id=snapshot_id,
            version=version,
            created_at=created_at,
        )
        if ts_snapshot is not None and ts_snapshot.is_complete:
            return ts_snapshot

        return FlowGraphSnapshot(
            snapshot_id=snapshot_id,
            scenario_id=scenario_id,
            version=version,
            created_at=float(created_at),
            nodes={},
            signatures=[],
            is_complete=False,
            parse_error=f"AST and Tree-Sitter extraction error: {type(exc).__name__}: {exc}",
        )
