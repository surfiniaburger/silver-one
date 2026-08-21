"""
Graphify-Inspired Tree-Sitter Data-Flow Extractor.
Integrates robust, error-tolerant Tree-Sitter AST extraction inspired by Graphify
to extract sources, sinks, sanitizers, and FlowGraphSnapshots across partial C/C++ fragments.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import tree_sitter
import tree_sitter_c

from scenarios.debate.graph_dataflow import (
    SUPPORTED_SINKS,
    VALID_SANITIZERS,
    FlowGraphSnapshot,
    FlowSignature,
    evaluate_graph_reachability,
    is_sanitizer_valid_for_sink,
)

C_LANGUAGE = tree_sitter.Language(tree_sitter_c.language())
C_PARSER = tree_sitter.Parser(C_LANGUAGE)

# Security-sensitive call sinks
MEMORY_SINKS = {
    "memcpy", "strcpy", "strncpy", "strcat", "strncat", "sprintf", "vsprintf",
    "snprintf", "vsnprintf", "memset", "memmove", "bcopy", "malloc", "calloc",
    "realloc", "pb_realloc", "cJSON_malloc", "kmap", "free", "kfree",
}

SYSTEM_SINKS = {
    "system", "popen", "exec", "execl", "execlp", "execle", "execv", "execvp",
    "execvpe", "open", "creat", "crypto_unregister_alg", "crypto_unregister_algs",
    "unlink", "remove", "chmod", "chown",
}

INPUT_SOURCES = {
    "read", "recv", "recvfrom", "recvmsg", "fread", "fgets", "gets", "scanf",
    "fscanf", "sscanf", "getenv", "getopt", "gather_time_entropy", "cJSON_Parse",
    "mp_read_unsigned_bin", "copy_from_user", "get_user",
}

def _strip_markdown_fences(code_text: str) -> str:
    lines = code_text.splitlines()
    filtered: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        filtered.append(line)
    return "\n".join(filtered)

def _wrap_in_function_if_needed(code_text: str) -> List[str]:
    """Generates candidate code variants to maximize Tree-sitter AST recovery."""
    clean = _strip_markdown_fences(code_text).strip()
    candidates = [clean]
    
    # Candidate with semicolon line termination
    lines = [l.strip() for l in clean.splitlines() if l.strip()]
    terminated_lines = []
    for l in lines:
        if not l.endswith(";") and not l.endswith("{") and not l.endswith("}") and not l.startswith("#"):
            terminated_lines.append(l + ";")
        else:
            terminated_lines.append(l)
    semi_code = "\n".join(terminated_lines)
    if semi_code != clean:
        candidates.append(semi_code)

    # Candidate wrapped in function block
    wrapped = f"void __vuln_harness_func() {{\n{semi_code}\n}}"
    candidates.append(wrapped)
    
    return candidates


class GraphifyFlowVisitor:
    """
    Error-tolerant AST visitor that walks tree-sitter C nodes to extract:
    1. Sources: function parameters, return values from input functions, field dereferences (e.g. svm->vmcb->save).
    2. Sinks: memory writes (subscript assignments, memcpy/malloc calls), pointer dereferences, system calls.
    3. Guards: enclosing if/while bounds checks and NULL checks.
    """

    def __init__(self, code_bytes: bytes):
        self.code_bytes = code_bytes
        self.nodes: Dict[str, dict] = {}
        self.signatures: List[FlowSignature] = []
        self.sources: Dict[str, str] = {}  # var_name -> source_node_id
        self.guard_stack: List[Dict[str, Set[str]]] = []  # var_name -> set(sanitizer_types)
        self.node_counter = 0

    def _gen_id(self, prefix: str) -> str:
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def _get_node_text(self, node: tree_sitter.Node) -> str:
        return self.code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()

    def _register_source(self, var_name: str, source_kind: str, node: tree_sitter.Node) -> str:
        if not var_name:
            var_name = "untrusted_input"
        if var_name in self.sources:
            return self.sources[var_name]

        src_id = self._gen_id(f"src_{var_name}")
        self.nodes[src_id] = {
            "id": src_id,
            "kind": "source",
            "label": f"Source({var_name})",
            "type": "UNTRUSTED_INPUT",
            "target_var": var_name,
            "source_kind": source_kind,
            "lineno": node.start_point[0] + 1,
            "col_offset": node.start_point[1],
        }
        self.sources[var_name] = src_id
        return src_id

    def _active_guards_for(self, target_var: str) -> Set[str]:
        guards: Set[str] = set()
        for frame in self.guard_stack:
            if target_var in frame:
                guards.update(frame[target_var])
        return guards

    def _resolve_sanitizer(self, sink_type: str, target_var: str) -> Tuple[Optional[str], Optional[str]]:
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

    def visit(self, node: tree_sitter.Node):
        """Recursively visit all AST nodes regardless of parsing errors."""
        node_type = node.type

        if node_type == "parameter_declaration":
            self._handle_parameter(node)
        elif node_type == "if_statement":
            self._handle_if_statement(node)
            return  # _handle_if_statement manages child recursion with guard frames
        elif node_type == "call_expression":
            self._handle_call(node)
        elif node_type == "assignment_expression":
            self._handle_assignment(node)
        elif node_type == "subscript_expression":
            self._handle_subscript(node)
        elif node_type == "field_expression":
            self._handle_field_expression(node)

        for child in node.children:
            self.visit(child)

    def _handle_parameter(self, node: tree_sitter.Node):
        ident = self._find_first_identifier(node)
        if ident:
            self._register_source(ident, "function_parameter", node)

    def _handle_if_statement(self, node: tree_sitter.Node):
        cond_node = node.child_by_field_name("condition")
        guard_map: Dict[str, Set[str]] = {}
        if cond_node:
            cond_text = self._get_node_text(cond_node)
            idents = self._extract_all_identifiers(cond_node)
            if any(op in cond_text for op in ("<", ">", "<=", ">=")):
                for id_name in idents:
                    guard_map.setdefault(id_name, set()).add("BOUNDS_CHECK")
            if "NULL" in cond_text or "== 0" in cond_text or "!= 0" in cond_text or "!" in cond_text:
                for id_name in idents:
                    guard_map.setdefault(id_name, set()).add("NULL_CHECK")

        self.guard_stack.append(guard_map)
        try:
            consequence = node.child_by_field_name("consequence")
            if consequence:
                self.visit(consequence)
        finally:
            self.guard_stack.pop()

        alternative = node.child_by_field_name("alternative")
        if alternative:
            self.visit(alternative)

    def _handle_call(self, node: tree_sitter.Node):
        func_name = self._extract_call_func_name(node)
        if not func_name:
            return

        # Check if call is an input source
        if func_name in INPUT_SOURCES:
            arg_idents = self._extract_call_arg_idents(node)
            target = arg_idents[0] if arg_idents else "input_data"
            self._register_source(target, f"call_{func_name}", node)

        # Check if call is a memory sink
        if func_name in MEMORY_SINKS:
            arg_idents = self._extract_call_arg_idents(node)
            target_var = arg_idents[0] if arg_idents else "dest_buf"
            src_id = self._get_or_create_source(target_var, node)
            sink_id = self._gen_id(f"sink_mem_{func_name}")
            
            san_type, san_target = self._resolve_sanitizer("MEMORY_WRITE", target_var)
            self.nodes[sink_id] = {
                "id": sink_id,
                "kind": "sink",
                "type": "MEMORY_WRITE",
                "target_var": target_var,
                "label": f"Call({func_name})",
                "lineno": node.start_point[0] + 1,
                "col_offset": node.start_point[1],
            }
            self.signatures.append(
                FlowSignature(
                    source_id=src_id,
                    sink_id=sink_id,
                    source_type="UNTRUSTED_INPUT",
                    sink_type="MEMORY_WRITE",
                    flow_type="CALL_ARGUMENT",
                    sanitizer_type=san_type,
                    guarded_target=san_target,
                )
            )

        # Check if call is a system sink
        if func_name in SYSTEM_SINKS:
            arg_idents = self._extract_call_arg_idents(node)
            target_var = arg_idents[0] if arg_idents else "cmd_str"
            src_id = self._get_or_create_source(target_var, node)
            sink_id = self._gen_id(f"sink_sys_{func_name}")

            san_type, san_target = self._resolve_sanitizer("SYSTEM_CALL", target_var)
            self.nodes[sink_id] = {
                "id": sink_id,
                "kind": "sink",
                "type": "SYSTEM_CALL",
                "target_var": target_var,
                "label": f"SysCall({func_name})",
                "lineno": node.start_point[0] + 1,
                "col_offset": node.start_point[1],
            }
            self.signatures.append(
                FlowSignature(
                    source_id=src_id,
                    sink_id=sink_id,
                    source_type="UNTRUSTED_INPUT",
                    sink_type="SYSTEM_CALL",
                    flow_type="SYSTEM_INVOCATION",
                    sanitizer_type=san_type,
                    guarded_target=san_target,
                )
            )

    def _handle_assignment(self, node: tree_sitter.Node):
        left_node = node.child_by_field_name("left") or (node.children[0] if node.children else None)
        right_node = node.child_by_field_name("right") or (node.children[2] if len(node.children) > 2 else None)
        if not left_node:
            return

        # If left node is subscript or pointer write: arr[i] = val or *ptr = val
        if left_node.type == "subscript_expression":
            idx_vars = self._extract_subscript_index_vars(left_node)
            target_var = idx_vars[0] if idx_vars else self._extract_target_ident(left_node)
            if target_var:
                src_id = self._get_or_create_source(target_var, node)
                sink_id = self._gen_id("sink_idx_write")
                san_type, san_target = self._resolve_sanitizer("MEMORY_WRITE", target_var)
                self.nodes[sink_id] = {
                    "id": sink_id,
                    "kind": "sink",
                    "type": "MEMORY_WRITE",
                    "target_var": target_var,
                    "label": "SubscriptWrite",
                    "lineno": node.start_point[0] + 1,
                    "col_offset": node.start_point[1],
                }
                self.signatures.append(
                    FlowSignature(
                        source_id=src_id,
                        sink_id=sink_id,
                        source_type="UNTRUSTED_INPUT",
                        sink_type="MEMORY_WRITE",
                        flow_type="ARRAY_INDEX_WRITE",
                        sanitizer_type=san_type,
                        guarded_target=san_target,
                    )
                )

        # If right side is a tainted field expression or pointer arithmetic
        if right_node and right_node.type in ("field_expression", "pointer_expression", "binary_expression"):
            target_var = self._extract_target_ident(left_node)
            if target_var:
                self._register_source(target_var, "tainted_expression", node)

    def _handle_subscript(self, node: tree_sitter.Node):
        idx_vars = self._extract_subscript_index_vars(node)
        if not idx_vars:
            return
        target_var = idx_vars[0]
        src_id = self._get_or_create_source(target_var, node)
        sink_id = self._gen_id("sink_subscript_read")
        san_type, san_target = self._resolve_sanitizer("ARRAY_INDEX", target_var)
        self.nodes[sink_id] = {
            "id": sink_id,
            "kind": "sink",
            "type": "ARRAY_INDEX",
            "target_var": target_var,
            "label": f"Subscript({target_var})",
            "lineno": node.start_point[0] + 1,
            "col_offset": node.start_point[1],
        }
        self.signatures.append(
            FlowSignature(
                source_id=src_id,
                sink_id=sink_id,
                source_type="UNTRUSTED_INPUT",
                sink_type="ARRAY_INDEX",
                flow_type="ARRAY_INDEX_READ",
                sanitizer_type=san_type,
                guarded_target=san_target,
            )
        )

    def _handle_field_expression(self, node: tree_sitter.Node):
        field_text = self._get_node_text(node)
        # Recognize pointer dereference patterns: ptr->field, vmcb->save, etc.
        if "->" in field_text:
            base_ident = field_text.split("->")[0].strip()
            src_id = self._get_or_create_source(base_ident, node)
            sink_id = self._gen_id("sink_ptr_deref")
            san_type, san_target = self._resolve_sanitizer("POINTER_DEREF", base_ident)
            self.nodes[sink_id] = {
                "id": sink_id,
                "kind": "sink",
                "type": "POINTER_DEREF",
                "target_var": base_ident,
                "label": f"PtrDeref({field_text})",
                "lineno": node.start_point[0] + 1,
                "col_offset": node.start_point[1],
            }
            self.signatures.append(
                FlowSignature(
                    source_id=src_id,
                    sink_id=sink_id,
                    source_type="UNTRUSTED_INPUT",
                    sink_type="POINTER_DEREF",
                    flow_type="POINTER_ACCESS",
                    sanitizer_type=san_type,
                    guarded_target=san_target,
                )
            )

    def _get_or_create_source(self, var_name: str, node: tree_sitter.Node) -> str:
        if var_name in self.sources:
            return self.sources[var_name]
        return self._register_source(var_name, "implicit_source", node)

    def _find_first_identifier(self, node: tree_sitter.Node) -> Optional[str]:
        if node.type == "identifier":
            return self._get_node_text(node)
        for child in node.children:
            res = self._find_first_identifier(child)
            if res:
                return res
        return None

    def _extract_all_identifiers(self, node: tree_sitter.Node) -> Set[str]:
        idents: Set[str] = set()
        if node.type == "identifier":
            idents.add(self._get_node_text(node))
        for child in node.children:
            idents.update(self._extract_all_identifiers(child))
        return idents

    def _extract_call_func_name(self, node: tree_sitter.Node) -> Optional[str]:
        func_node = node.child_by_field_name("function")
        if func_node:
            if func_node.type == "identifier":
                return self._get_node_text(func_node)
            if func_node.type == "field_expression":
                field_child = func_node.child_by_field_name("field")
                if field_child:
                    return self._get_node_text(field_child)
        return self._find_first_identifier(node)

    def _extract_call_arg_idents(self, node: tree_sitter.Node) -> List[str]:
        arg_list = node.child_by_field_name("arguments")
        if not arg_list:
            return []
        idents: List[str] = []
        for child in arg_list.named_children:
            ident = self._find_first_identifier(child)
            if ident:
                idents.append(ident)
        return idents

    def _extract_subscript_index_vars(self, node: tree_sitter.Node) -> List[str]:
        idx_node = node.child_by_field_name("index")
        if not idx_node:
            return []
        return sorted(self._extract_all_identifiers(idx_node))

    def _extract_target_ident(self, node: Optional[tree_sitter.Node]) -> Optional[str]:
        if not node:
            return None
        return self._find_first_identifier(node)


def extract_graphify_flow_snapshot(
    code_text: str,
    scenario_id: str,
    snapshot_id: str = "snap_1",
    version: int = 1,
    created_at: float = 0.0,
) -> FlowGraphSnapshot:
    """
    Graphify-inspired robust Tree-sitter AST extraction function.
    Parses fragments, extracts sources, sinks, and FlowSignatures with error tolerance.
    """
    if not code_text or not code_text.strip():
        return FlowGraphSnapshot(
            snapshot_id=snapshot_id,
            scenario_id=scenario_id,
            version=version,
            created_at=created_at,
            nodes={},
            signatures=[],
            is_complete=False,
            parse_error="Empty code input",
        )

    candidates = _wrap_in_function_if_needed(code_text)

    best_visitor: Optional[GraphifyFlowVisitor] = None
    max_signatures = -1

    for candidate in candidates:
        code_bytes = candidate.encode("utf-8")
        tree = C_PARSER.parse(code_bytes)
        visitor = GraphifyFlowVisitor(code_bytes)
        visitor.visit(tree.root_node)

        if len(visitor.signatures) > max_signatures:
            max_signatures = len(visitor.signatures)
            best_visitor = visitor

    if best_visitor is None or not best_visitor.nodes:
        # Fallback: create minimal source and sink nodes if keywords present
        return FlowGraphSnapshot(
            snapshot_id=snapshot_id,
            scenario_id=scenario_id,
            version=version,
            created_at=created_at,
            nodes={},
            signatures=[],
            is_complete=False,
            parse_error="No valid nodes extracted from Tree-sitter parse",
        )

    return FlowGraphSnapshot(
        snapshot_id=snapshot_id,
        scenario_id=scenario_id,
        version=version,
        created_at=created_at,
        nodes=best_visitor.nodes,
        signatures=best_visitor.signatures,
        is_complete=True,
        parse_error=None,
    )
