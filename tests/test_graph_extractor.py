"""
Contract tests for scenarios/debate/graph_extractor.py.
Verifies AST extraction accuracy, sink/sanitizer mapping, operand identity, and fail-closed behavior across all verification vectors.
"""

from scenarios.debate.graph_dataflow import evaluate_graph_reachability, is_graph_candidate_rejected
from scenarios.debate.graph_extractor import extract_flow_graph_snapshot


def test_vulnerable_and_guarded_memory_write():
    vuln_code = """
def process(data, i):
    buf[i] = data
"""
    snap_vuln = extract_flow_graph_snapshot(
        code_text=vuln_code, scenario_id="sc_mem", snapshot_id="snap_1", version=1, created_at=1000.0
    )
    assert snap_vuln.is_complete is True
    assert len(snap_vuln.signatures) == 1
    sig = snap_vuln.signatures[0]
    assert sig.sink_type == "MEMORY_WRITE"
    assert sig.sanitizer_type is None
    assert evaluate_graph_reachability(snap_vuln) == 1.0
    assert is_graph_candidate_rejected(snap_vuln) is True

    guarded_code = """
def process(data, i):
    if i < MAX_LEN:
        buf[i] = data
"""
    snap_guarded = extract_flow_graph_snapshot(
        code_text=guarded_code, scenario_id="sc_mem", snapshot_id="snap_2", version=1, created_at=1000.0
    )
    assert snap_guarded.is_complete is True
    assert len(snap_guarded.signatures) == 1
    sig_g = snap_guarded.signatures[0]
    assert sig_g.sink_type == "MEMORY_WRITE"
    assert sig_g.sanitizer_type == "BOUNDS_CHECK"
    assert sig_g.guarded_target == "i"
    assert evaluate_graph_reachability(snap_guarded) == 0.05
    assert is_graph_candidate_rejected(snap_guarded) is False


def test_vulnerable_and_guarded_array_index():
    vuln_code = """
def get_elem(arr, index):
    return arr[index]
"""
    snap_vuln = extract_flow_graph_snapshot(
        code_text=vuln_code, scenario_id="sc_arr", snapshot_id="snap_3", version=1, created_at=1000.0
    )
    assert snap_vuln.is_complete is True
    assert len(snap_vuln.signatures) == 1
    sig = snap_vuln.signatures[0]
    assert sig.sink_type == "ARRAY_INDEX"
    assert sig.sanitizer_type is None
    assert evaluate_graph_reachability(snap_vuln) == 1.0

    guarded_code = """
def get_elem(arr, index):
    if 0 <= index < len(arr):
        return arr[index]
"""
    snap_guarded = extract_flow_graph_snapshot(
        code_text=guarded_code, scenario_id="sc_arr", snapshot_id="snap_4", version=1, created_at=1000.0
    )
    assert snap_guarded.is_complete is True
    assert len(snap_guarded.signatures) == 1
    sig_g = snap_guarded.signatures[0]
    assert sig_g.sink_type == "ARRAY_INDEX"
    assert sig_g.sanitizer_type == "RANGE_VALIDATION"
    assert sig_g.guarded_target == "index"
    assert evaluate_graph_reachability(snap_guarded) == 0.05


def test_vulnerable_and_guarded_pointer_deref():
    vuln_code = """
def inspect(ptr):
    return ptr.data
"""
    snap_vuln = extract_flow_graph_snapshot(
        code_text=vuln_code, scenario_id="sc_ptr", snapshot_id="snap_5", version=1, created_at=1000.0
    )
    assert snap_vuln.is_complete is True
    assert len(snap_vuln.signatures) == 1
    sig = snap_vuln.signatures[0]
    assert sig.sink_type == "POINTER_DEREF"
    assert sig.sanitizer_type is None
    assert evaluate_graph_reachability(snap_vuln) == 1.0

    guarded_code = """
def inspect(ptr):
    if ptr is not None:
        return ptr.data
"""
    snap_guarded = extract_flow_graph_snapshot(
        code_text=guarded_code, scenario_id="sc_ptr", snapshot_id="snap_6", version=1, created_at=1000.0
    )
    assert snap_guarded.is_complete is True
    assert len(snap_guarded.signatures) == 1
    sig_g = snap_guarded.signatures[0]
    assert sig_g.sink_type == "POINTER_DEREF"
    assert sig_g.sanitizer_type == "NULL_CHECK"
    assert sig_g.guarded_target == "ptr"
    assert evaluate_graph_reachability(snap_guarded) == 0.05


def test_vulnerable_and_guarded_system_call():
    vuln_code = """
def execute(cmd):
    import os
    os.system(cmd)
"""
    snap_vuln = extract_flow_graph_snapshot(
        code_text=vuln_code, scenario_id="sc_sys", snapshot_id="snap_7", version=1, created_at=1000.0
    )
    assert snap_vuln.is_complete is True
    assert len(snap_vuln.signatures) == 1
    sig = snap_vuln.signatures[0]
    assert sig.sink_type == "SYSTEM_CALL"
    assert sig.sanitizer_type is None
    assert evaluate_graph_reachability(snap_vuln) == 1.0

    guarded_quote_code = """
def execute(cmd):
    import os, shlex
    safe_cmd = shlex.quote(cmd)
    os.system(safe_cmd)
"""
    snap_quote = extract_flow_graph_snapshot(
        code_text=guarded_quote_code, scenario_id="sc_sys", snapshot_id="snap_8", version=1, created_at=1000.0
    )
    assert snap_quote.is_complete is True
    assert len(snap_quote.signatures) == 1
    sig_q = snap_quote.signatures[0]
    assert sig_q.sink_type == "SYSTEM_CALL"
    assert sig_q.sanitizer_type == "COMMAND_SANITIZATION"
    assert evaluate_graph_reachability(snap_quote) == 0.05

    guarded_allowlist_code = """
def execute(cmd):
    import os
    if cmd in ALLOWED_COMMANDS:
        os.system(cmd)
"""
    snap_allow = extract_flow_graph_snapshot(
        code_text=guarded_allowlist_code, scenario_id="sc_sys", snapshot_id="snap_9", version=1, created_at=1000.0
    )
    assert snap_allow.is_complete is True
    assert len(snap_allow.signatures) == 1
    sig_a = snap_allow.signatures[0]
    assert sig_a.sink_type == "SYSTEM_CALL"
    assert sig_a.sanitizer_type == "ALLOWLIST_CHECK"
    assert sig_a.guarded_target == "cmd"
    assert evaluate_graph_reachability(snap_allow) == 0.05


def test_memory_call_binds_sanitizer_to_size_operand():
    vuln_code = """
def copy(dest, src, n):
    memcpy(dest, src, n)
"""
    snap_vuln = extract_flow_graph_snapshot(
        code_text=vuln_code, scenario_id="sc_mem_call", snapshot_id="snap_mem_call_vuln", version=1, created_at=1000.0
    )
    assert snap_vuln.is_complete is True
    assert len(snap_vuln.signatures) == 1
    sig = snap_vuln.signatures[0]
    assert sig.sink_type == "MEMORY_WRITE"
    assert sig.sanitizer_type is None
    assert snap_vuln.nodes[sig.sink_id]["target_var"] == "n"
    assert evaluate_graph_reachability(snap_vuln) == 1.0

    guarded_code = """
def copy(dest, src, n):
    if n < MAX_LEN:
        memcpy(dest, src, n)
"""
    snap_guarded = extract_flow_graph_snapshot(
        code_text=guarded_code, scenario_id="sc_mem_call", snapshot_id="snap_mem_call_guarded", version=1, created_at=1000.0
    )
    assert snap_guarded.is_complete is True
    assert len(snap_guarded.signatures) == 1
    guarded_sig = snap_guarded.signatures[0]
    assert guarded_sig.sink_type == "MEMORY_WRITE"
    assert guarded_sig.sanitizer_type == "BOUNDS_CHECK"
    assert guarded_sig.guarded_target == "n"
    assert evaluate_graph_reachability(snap_guarded) == 0.05

    wrong_guard_code = """
def copy(dest, src, n):
    if dest < MAX_LEN:
        memcpy(dest, src, n)
"""
    snap_wrong_guard = extract_flow_graph_snapshot(
        code_text=wrong_guard_code, scenario_id="sc_mem_call", snapshot_id="snap_mem_call_wrong_guard", version=1, created_at=1000.0
    )
    assert snap_wrong_guard.is_complete is True
    assert len(snap_wrong_guard.signatures) == 1
    wrong_sig = snap_wrong_guard.signatures[0]
    assert wrong_sig.sanitizer_type is None
    assert wrong_sig.guarded_target is None
    assert evaluate_graph_reachability(snap_wrong_guard) == 1.0


def test_extracted_nodes_include_audit_provenance():
    code = """
def process(data, i):
    if i < MAX_LEN:
        buf[i] = data
"""
    snap = extract_flow_graph_snapshot(
        code_text=code, scenario_id="sc_provenance", snapshot_id="snap_provenance", version=1, created_at=1000.0
    )
    assert snap.is_complete is True
    sig = snap.signatures[0]
    source_node = snap.nodes[sig.source_id]
    sink_node = snap.nodes[sig.sink_id]

    assert source_node["source_kind"] == "function_parameter"
    assert source_node["target_var"] == "data"
    assert source_node["ast_type"] == "arg"
    assert sink_node["sink_expr_kind"] == "subscript_write"
    assert sink_node["target_var"] == "i"
    assert sink_node["ast_type"] == "Subscript"


def test_syntax_error_and_malformed_created_at_fail_closed():
    bad_syntax = """
def process(data):
    if data
"""
    snap_bad = extract_flow_graph_snapshot(
        code_text=bad_syntax, scenario_id="sc_err", snapshot_id="snap_10", version=1, created_at=1000.0
    )
    assert snap_bad.is_complete is False
    assert snap_bad.parse_error is not None
    assert evaluate_graph_reachability(snap_bad) == 1.0

    snap_bad_time = extract_flow_graph_snapshot(
        code_text="def f(): pass", scenario_id="sc_err", snapshot_id="snap_11", version=1, created_at=float("nan")
    )
    assert snap_bad_time.is_complete is False
    assert evaluate_graph_reachability(snap_bad_time) == 1.0


def test_unrelated_guard_does_not_sanitize_sink_operand():
    code = """
def process(data, i, j):
    if j < MAX_LEN:
        buf[i] = data
"""
    snap = extract_flow_graph_snapshot(
        code_text=code, scenario_id="sc_guard", snapshot_id="snap_unrelated", version=1, created_at=1000.0
    )
    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "MEMORY_WRITE"
    assert sig.sanitizer_type is None
    assert sig.guarded_target is None
    assert evaluate_graph_reachability(snap) == 1.0


def test_guard_does_not_escape_if_body():
    code = """
def process(data, i):
    if i < MAX_LEN:
        pass
    buf[i] = data
"""
    snap = extract_flow_graph_snapshot(
        code_text=code, scenario_id="sc_guard", snapshot_id="snap_stale", version=1, created_at=1000.0
    )
    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "MEMORY_WRITE"
    assert sig.sanitizer_type is None
    assert sig.guarded_target is None
    assert evaluate_graph_reachability(snap) == 1.0


def test_guard_does_not_apply_to_else_branch():
    code = """
def process(data, i):
    if i < MAX_LEN:
        pass
    else:
        buf[i] = data
"""
    snap = extract_flow_graph_snapshot(
        code_text=code, scenario_id="sc_guard", snapshot_id="snap_else", version=1, created_at=1000.0
    )
    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "MEMORY_WRITE"
    assert sig.sanitizer_type is None
    assert sig.guarded_target is None
    assert evaluate_graph_reachability(snap) == 1.0


def test_sanitized_command_alias_does_not_sanitize_original_command():
    code = """
def execute(cmd):
    import os, shlex
    safe_cmd = shlex.quote(cmd)
    os.system(cmd)
"""
    snap = extract_flow_graph_snapshot(
        code_text=code, scenario_id="sc_sys", snapshot_id="snap_alias_original", version=1, created_at=1000.0
    )
    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "SYSTEM_CALL"
    assert sig.sanitizer_type is None
    assert sig.guarded_target is None
    assert evaluate_graph_reachability(snap) == 1.0
