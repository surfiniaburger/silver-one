"""
Contract tests for scenarios/debate/graph_extractor.py.
Verifies AST extraction accuracy, sink/sanitizer mapping, operand identity, and fail-closed behavior across all verification vectors.
"""

import pytest

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


def test_html_escape_does_not_provide_command_sanitization():
    code = """
def execute(cmd):
    import os, html
    safe_cmd = html.escape(cmd)
    os.system(safe_cmd)
"""
    snap = extract_flow_graph_snapshot(
        code_text=code, scenario_id="sc_sys_html", snapshot_id="snap_html", version=1, created_at=1000.0
    )
    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "SYSTEM_CALL"
    assert sig.sanitizer_type is None
    assert sig.guarded_target is None
    assert evaluate_graph_reachability(snap) == 1.0


def test_rebound_variable_clears_prior_command_sanitization():
    code = """
def execute(cmd, attacker):
    import os, shlex
    cmd = shlex.quote(cmd)
    cmd = attacker
    os.system(cmd)
"""
    snap = extract_flow_graph_snapshot(
        code_text=code, scenario_id="sc_sys_rebound", snapshot_id="snap_rebound", version=1, created_at=1000.0
    )
    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "SYSTEM_CALL"
    assert sig.sanitizer_type is None
    assert sig.guarded_target is None
    assert evaluate_graph_reachability(snap) == 1.0


def test_normalize_code_for_ast_loose_statements_and_markdown_fences():
    from scenarios.debate.graph_extractor import normalize_code_for_ast

    c_fenced = "```c\nif (ptr == NULL);\n```"
    norm = normalize_code_for_ast(c_fenced)
    assert "None" in norm

    c_loose = "int f(char *s) { return 0; }"
    norm_loose = normalize_code_for_ast(c_loose)
    assert "def f(s):" in norm_loose

    snap = extract_flow_graph_snapshot(
        code_text="def store(data, i):\n    buf[i] = data",
        scenario_id="sc_loose",
        snapshot_id="snap_loose",
        version=1,
        created_at=1000.0,
    )
    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    assert snap.signatures[0].sink_type == "MEMORY_WRITE"


def test_normalize_code_quote_awareness_and_multiline_braces():
    import ast
    from scenarios.debate.graph_extractor import normalize_code_for_ast

    # 1. Quote awareness: URLs and string literals containing //, NULL, true are preserved
    c_quotes = 'char *url = "http://example.com/api?val=NULL&flag=true";'
    norm_quotes = normalize_code_for_ast(c_quotes)
    assert '"http://example.com/api?val=NULL&flag=true"' in norm_quotes
    ast.parse(norm_quotes)

    # 2. Multiline C block brace structure and indentation
    c_multiline = """
int process_data(char *data, int len) {
    if (ptr == NULL);
    if (len > 0) {
        buf[i] = data;
    }
    return 0;
}
"""
    norm_multi = normalize_code_for_ast(c_multiline)
    tree = ast.parse(norm_multi)
    process_data = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "process_data"
    )
    empty_guard, bounds_guard, return_stmt = process_data.body[-3:]
    assert isinstance(empty_guard, ast.If)
    assert isinstance(empty_guard.body[0], ast.Pass)
    assert isinstance(bounds_guard, ast.If)
    assert isinstance(bounds_guard.body[0], ast.Assign)
    assert isinstance(return_stmt, ast.Return)


def test_treesitter_c_extraction_sources_sinks_and_guards():
    from scenarios.debate.graph_extractor import extract_flow_graph_snapshot_treesitter

    # 1. Guarded C open call: error_log parameter paired with open(error_log)
    c_guarded = """
int process_c_file(char *error_log, int len) {
    if (error_log != NULL) {
        open(error_log, O_WRONLY | O_CREAT);
    }
}
"""
    snap_guarded = extract_flow_graph_snapshot_treesitter(
        code_text=c_guarded,
        scenario_id="sc_c_open",
        snapshot_id="snap_c_open",
        version=1,
        created_at=1000.0,
    )

    assert snap_guarded is not None
    assert snap_guarded.is_complete is True
    assert len(snap_guarded.signatures) == 1
    sig = snap_guarded.signatures[0]
    assert sig.sink_type == "SYSTEM_CALL"
    assert sig.flow_type == "COMMAND_EXECUTION"
    assert sig.sanitizer_type == "NULL_CHECK"
    assert sig.guarded_target == "error_log"
    assert snap_guarded.nodes[sig.source_id]["target_var"] == "error_log"

    # 2. Unguarded C open call: sanitizer_type is None
    c_unguarded = """
int process_c_file(char *error_log, int len) {
    open(error_log, O_WRONLY);
}
"""
    snap_unguarded = extract_flow_graph_snapshot_treesitter(
        code_text=c_unguarded,
        scenario_id="sc_c_un",
        snapshot_id="snap_c_un",
        version=1,
        created_at=1000.0,
    )
    assert snap_unguarded is not None
    assert len(snap_unguarded.signatures) == 1
    assert snap_unguarded.signatures[0].sanitizer_type is None

    # 3. C memcpy sample: MEMORY_WRITE sink with MEMORY_COPY_CALL flow type
    c_memcpy = """
void copy_buffer(char *dest, char *src, int n) {
    memcpy(dest, src, n);
}
"""
    snap_memcpy = extract_flow_graph_snapshot_treesitter(
        code_text=c_memcpy,
        scenario_id="sc_memcpy",
        snapshot_id="snap_memcpy",
        version=1,
        created_at=1000.0,
    )
    assert snap_memcpy is not None
    assert any(sig.sink_type == "MEMORY_WRITE" for sig in snap_memcpy.signatures)

    # 4. Non-C Python input: returns None
    snap_non_c = extract_flow_graph_snapshot_treesitter(
        code_text="def foo(): pass",
        scenario_id="sc_non_c",
        snapshot_id="snap_non_c",
        version=1,
        created_at=1000.0,
    )
    assert snap_non_c is None


def test_treesitter_corpus_getenv_to_execlp_source_sink_flow():
    from scenarios.debate.graph_extractor import extract_flow_graph_snapshot_treesitter

    code = """
void run_editor() {
    editor = getenv("EDITOR");
    execlp(editor, "sh", "-c", NULL);
}
"""

    snap = extract_flow_graph_snapshot_treesitter(
        code_text=code,
        scenario_id="HASH-6fa94f75c2",
        snapshot_id="snap_getenv_execlp",
        version=1,
        created_at=1000.0,
    )

    assert snap is not None
    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "SYSTEM_CALL"
    assert sig.sanitizer_type is None
    assert snap.nodes[sig.source_id]["target_var"] == "editor"
    assert snap.nodes[sig.source_id]["source_kind"] == "getenv"
    assert snap.nodes[sig.sink_id]["target_var"] == "editor"


def test_top_level_extraction_falls_back_to_treesitter_for_c_like_empty_ast():
    code = """
void run_editor() {
    editor = getenv("EDITOR");
    execlp(editor, "sh", "-c", NULL);
}
"""

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="HASH-6fa94f75c2",
        snapshot_id="snap_top_getenv_execlp",
        version=1,
        created_at=1000.0,
    )

    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    assert snap.signatures[0].sink_type == "SYSTEM_CALL"
    assert evaluate_graph_reachability(snap) == 1.0


def test_treesitter_registers_standalone_read_buffer_source():
    from scenarios.debate.graph_extractor import extract_flow_graph_snapshot_treesitter

    code = """
void run_command(int fd) {
    char buf[256];
    read(fd, buf, sizeof(buf));
    system(buf);
}
"""

    snap = extract_flow_graph_snapshot_treesitter(
        code_text=code,
        scenario_id="sc_read_buf",
        snapshot_id="snap_read_buf",
        version=1,
        created_at=1000.0,
    )

    assert snap is not None
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "SYSTEM_CALL"
    assert snap.nodes[sig.source_id]["target_var"] == "buf"
    assert snap.nodes[sig.source_id]["source_kind"] == "read"
    assert snap.nodes[sig.sink_id]["target_var"] == "buf"


def test_treesitter_registers_assigned_recv_buffer_source():
    from scenarios.debate.graph_extractor import extract_flow_graph_snapshot_treesitter

    code = """
void run_command(int sock) {
    char buf[256];
    n = recv(sock, buf, sizeof(buf), 0);
    system(buf);
}
"""

    snap = extract_flow_graph_snapshot_treesitter(
        code_text=code,
        scenario_id="sc_recv_buf",
        snapshot_id="snap_recv_buf",
        version=1,
        created_at=1000.0,
    )

    assert snap is not None
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "SYSTEM_CALL"
    assert snap.nodes[sig.source_id]["target_var"] == "buf"
    assert snap.nodes[sig.source_id]["source_kind"] == "recv"


def test_treesitter_registers_fread_first_argument_source():
    from scenarios.debate.graph_extractor import extract_flow_graph_snapshot_treesitter

    code = """
void run_command(FILE *fp) {
    char buf[256];
    fread(buf, 1, sizeof(buf), fp);
    system(buf);
}
"""

    snap = extract_flow_graph_snapshot_treesitter(
        code_text=code,
        scenario_id="sc_fread_buf",
        snapshot_id="snap_fread_buf",
        version=1,
        created_at=1000.0,
    )

    assert snap is not None
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "SYSTEM_CALL"
    assert snap.nodes[sig.source_id]["target_var"] == "buf"
    assert snap.nodes[sig.source_id]["source_kind"] == "fread"


def test_treesitter_rejects_benchmark_specific_sink_names():
    from scenarios.debate.graph_extractor import extract_flow_graph_snapshot_treesitter

    code = """
void f(char *error_log) {
    fpm_stdio_open_error_log(error_log);
}
"""

    snap = extract_flow_graph_snapshot_treesitter(
        code_text=code,
        scenario_id="HASH-d01f0f1dd4",
        snapshot_id="snap_project_specific_sink",
        version=1,
        created_at=1000.0,
    )

    assert snap is None


def test_treesitter_rejects_recovery_tree_syntax_errors():
    from scenarios.debate.graph_extractor import extract_flow_graph_snapshot_treesitter

    snap = extract_flow_graph_snapshot_treesitter(
        code_text="open(error_log, O_WRONLY | O_APPEND | O_CREAT",
        scenario_id="HASH-d01f0f1dd4",
        snapshot_id="snap_syntax_error",
        version=1,
        created_at=1000.0,
    )

    assert snap is None


def test_corpus_unsupported_positive_index_write_from_derived_pointer_read():
    code = "r1.i = *(uint64_t*)(ip + 1); mem[r1.i] = 0;"

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="HASH-c966428621",
        snapshot_id="snap_idx82",
        version=1,
        created_at=1000.0,
    )

    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "MEMORY_WRITE"
    assert sig.flow_type == "INDEX_WRITE"
    assert snap.nodes[sig.source_id]["target_var"] == "r1.i"
    assert snap.nodes[sig.source_id]["source_kind"] == "fragment_derived_value"
    assert snap.nodes[sig.sink_id]["target_var"] == "r1.i"
    assert evaluate_graph_reachability(snap) == 1.0


def test_corpus_unsupported_positive_output_index_write_from_derived_offset():
    code = (
        "pixeloutstart = ((ADAM7_IY[i] + y * ADAM7_DY[i]) * w + ADAM7_IX[i] + x * ADAM7_DX[i]) * bytewidth; "
        "out[pixeloutstart + b] = in[pixelinstart + b];"
    )

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="HASH-94bb66ef2c",
        snapshot_id="snap_idx102",
        version=1,
        created_at=1000.0,
    )

    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "MEMORY_WRITE"
    assert sig.flow_type == "INDEX_WRITE"
    assert snap.nodes[sig.source_id]["target_var"] == "pixeloutstart"
    assert snap.nodes[sig.source_id]["source_kind"] == "fragment_derived_value"
    assert snap.nodes[sig.sink_id]["target_var"] == "pixeloutstart"
    assert evaluate_graph_reachability(snap) == 1.0


@pytest.mark.parametrize(
    ("scenario_id", "code"),
    [
        (
            "HASH-db77a7663a",
            "item_len = btrfs_item_size_nr(leaf, i); memcpy(tmp_buf, &sh, sizeof(sh));",
        ),
        (
            "HASH-c153a60648",
            "if (pos > dp->realSize) memcpy((void *)(tmp + (dp->pos)), src, size) dp->pos = pos;",
        ),
    ],
)
def test_corpus_unsupported_positive_selected_copy_fragments_remain_unproven_without_source(scenario_id, code):
    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id=scenario_id,
        snapshot_id=f"snap_{scenario_id}",
        version=1,
        created_at=1000.0,
    )
    assert evaluate_graph_reachability(snap) == 1.0
    assert not snap.signatures


def test_corpus_unsupported_positive_sprintf_from_underallocated_buffer():
    code = (
        'mac_tmp_len = strlen(mac_exe) + strlen(MAC_PATH_VALUE) mac_tmp = malloc(mac_tmp_len) '
        'sprintf(mac_tmp, "%s%s%s", mac_exe, MAC_PATH_VALUE, mac_exe) if (mac_tmp_len <= arg0_len)'
    )

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="HASH-9c691500ed",
        snapshot_id="snap_sprintf_underallocated",
        version=1,
        created_at=1000.0,
    )

    assert snap.is_complete is True
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "MEMORY_WRITE"
    assert sig.flow_type == "MEMORY_COPY_CALL"
    assert snap.nodes[sig.source_id]["target_var"] == "mac_tmp"
    assert snap.nodes[sig.source_id]["source_kind"] == "underallocated_format_buffer"
    assert snap.nodes[sig.sink_id]["target_var"] == "mac_tmp"
    assert evaluate_graph_reachability(snap) == 1.0


def test_adversarial_sprintf_literal_only_is_not_positive_evidence():
    code = 'buf = malloc(32); sprintf(buf, "constant")'

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="sprintf-literal-only",
        snapshot_id="snap_sprintf_literal_only",
        version=1,
        created_at=1000.0,
    )

    assert evaluate_graph_reachability(snap) == 1.0
    assert not snap.signatures


def test_adversarial_sprintf_matching_allocation_terms_is_not_positive_evidence():
    code = 'buf_len = strlen(name); buf = malloc(buf_len); sprintf(buf, "%s", name)'

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="sprintf-matching-allocation",
        snapshot_id="snap_sprintf_matching_allocation",
        version=1,
        created_at=1000.0,
    )

    assert evaluate_graph_reachability(snap) == 1.0
    assert not snap.signatures


def test_adversarial_sprintf_escaped_percent_is_not_positive_evidence():
    code = 'buf_len = strlen(name); buf = malloc(buf_len); sprintf(buf, "%%s%s", name)'

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="sprintf-escaped-percent",
        snapshot_id="snap_sprintf_escaped_percent",
        version=1,
        created_at=1000.0,
    )

    assert evaluate_graph_reachability(snap) == 1.0
    assert not snap.signatures


def test_adversarial_project_macro_fragment_stays_unsupported():
    code = "char buffer[BUFF_SIG_SIZE + in_buffer_size] BUFFER_ADD (pea.username, username_len)"

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="HASH-115fc070e2",
        snapshot_id="snap_macro_adversarial",
        version=1,
        created_at=1000.0,
    )

    assert evaluate_graph_reachability(snap) == 1.0
    assert not snap.signatures


def test_adversarial_scalar_computed_index_is_not_positive_evidence():
    code = "safe_idx = i & 7; mem[safe_idx] = 0;"

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="scalar-computed-index",
        snapshot_id="snap_scalar_computed_index",
        version=1,
        created_at=1000.0,
    )

    assert evaluate_graph_reachability(snap) == 1.0
    assert not snap.signatures


def test_adversarial_unrelated_index_write_is_not_positive_evidence():
    code = "offset = table[i] + 1; mem[fixed_idx] = 0;"

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="unrelated-index-write",
        snapshot_id="snap_unrelated_index_write",
        version=1,
        created_at=1000.0,
    )

    assert evaluate_graph_reachability(snap) == 1.0
    assert not snap.signatures


def test_adversarial_sizeof_index_is_not_positive_evidence():
    code = "idx = sizeof(buf); mem[idx] = 0;"

    snap = extract_flow_graph_snapshot(
        code_text=code,
        scenario_id="sizeof-index-write",
        snapshot_id="snap_sizeof_index_write",
        version=1,
        created_at=1000.0,
    )

    assert evaluate_graph_reachability(snap) == 1.0
    assert not snap.signatures


def test_treesitter_registers_execv_tainted_argv_argument():
    from scenarios.debate.graph_extractor import extract_flow_graph_snapshot_treesitter

    code = """
void run_exec(char **argv) {
    execv("/bin/sh", argv);
}
"""
    snap = extract_flow_graph_snapshot_treesitter(
        code_text=code,
        scenario_id="sc_execv",
        snapshot_id="snap_execv",
        version=1,
        created_at=1000.0,
    )
    assert snap is not None
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert sig.sink_type == "SYSTEM_CALL"
    assert snap.nodes[sig.source_id]["target_var"] == "argv"


def test_treesitter_preserves_positional_arguments_with_literals():
    from scenarios.debate.graph_extractor import extract_flow_graph_snapshot_treesitter

    code = """
void run_read_literal_fd() {
    char buf[128];
    read(0, buf, 128);
    system(buf);
}
"""
    snap = extract_flow_graph_snapshot_treesitter(
        code_text=code,
        scenario_id="sc_literal_read",
        snapshot_id="snap_literal_read",
        version=1,
        created_at=1000.0,
    )
    assert snap is not None
    assert len(snap.signatures) == 1
    sig = snap.signatures[0]
    assert snap.nodes[sig.source_id]["target_var"] == "buf"
    assert snap.nodes[sig.source_id]["source_kind"] == "read"


def test_treesitter_size_calculation_on_struct_field_is_not_fragment_derived_source():
    from scenarios.debate.graph_extractor import extract_flow_graph_snapshot_treesitter

    matching_code = """
len = strlen(hdr->name) + 1;
buf = malloc(len);
sprintf(buf, "%s", hdr->name);
"""
    snap_matching = extract_flow_graph_snapshot_treesitter(
        code_text=matching_code,
        scenario_id="sc_hdr_len_safe",
        snapshot_id="snap_hdr_len_safe",
        version=1,
        created_at=1000.0,
    )
    assert snap_matching is None

    underallocated_code = """
len = strlen(hdr->name) + 1;
buf = malloc(len);
sprintf(buf, "%s%s", hdr->name, hdr->extra);
"""
    snap_underallocated = extract_flow_graph_snapshot_treesitter(
        code_text=underallocated_code,
        scenario_id="sc_hdr_len_vuln",
        snapshot_id="snap_hdr_len_vuln",
        version=1,
        created_at=1000.0,
    )
    assert snap_underallocated is not None
    assert len(snap_underallocated.signatures) == 1
    assert snap_underallocated.nodes[snap_underallocated.signatures[0].source_id]["source_kind"] == "underallocated_format_buffer"
