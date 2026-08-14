"""
AST Graph Data-Flow Extractor Module.
Parses code attempt strings using Python's ast module and produces deterministic FlowGraphSnapshot objects.
Implements the technical specification in docs/SPEC_GRAPH_EXTRACTOR.md.
"""

import ast
import math
from typing import Dict, List, Optional, Set, Tuple

from scenarios.debate.graph_dataflow import (
    SUPPORTED_SINKS,
    VALID_SANITIZERS,
    FlowGraphSnapshot,
    FlowSignature,
)


SYSTEM_CALL_FUNCTIONS = {"os.system", "subprocess.Popen", "subprocess.run", "eval", "exec", "system", "Popen", "run"}
MEMORY_FUNCTIONS = {"memcpy", "strcpy", "memset", "memmove"}
EXPLICIT_INPUT_SOURCES = {"input", "sys.argv", "request.args", "request.get_json", "socket.recv", "file.read"}
SANITIZER_FUNCTIONS = {"quote", "escape", "sanitize", "quote_plus", "escape_string", "shlex.quote", "html.escape"}


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

    def visit_Assign(self, node: ast.Assign):
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
        if callee not in SANITIZER_FUNCTIONS and func_name not in SANITIZER_FUNCTIONS:
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
            self._add_signature(
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
            self._add_signature(
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
                self._add_signature(
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
            self._add_signature(
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
            self._add_signature(
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
        self,
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

        self.signatures.append(
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


def extract_flow_graph_snapshot(
    code_text: str,
    scenario_id: str,
    snapshot_id: str,
    version: int,
    created_at: float,
) -> FlowGraphSnapshot:
    """
    Parses code_text using AST inspection and builds a deterministic FlowGraphSnapshot.
    
    Fails closed (returns is_complete=False) if syntax errors or parse failures occur,
    or if non-numeric/non-finite created_at timestamp is provided.
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

    try:
        tree = ast.parse(code_text)
    except Exception as exc:
        return FlowGraphSnapshot(
            snapshot_id=snapshot_id,
            scenario_id=scenario_id,
            version=version,
            created_at=float(created_at),
            nodes={},
            signatures=[],
            is_complete=False,
            parse_error=f"AST parse error: {exc}",
        )

    try:
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

    except Exception as exc:
        return FlowGraphSnapshot(
            snapshot_id=snapshot_id,
            scenario_id=scenario_id,
            version=version,
            created_at=float(created_at),
            nodes={},
            signatures=[],
            is_complete=False,
            parse_error=f"AST extraction error: {exc}",
        )
