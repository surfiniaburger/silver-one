"""
Unit tests for the Graphify-inspired Tree-sitter Flow Extractor and Evaluation Dataset.
"""

from __future__ import annotations

from scenarios.debate.graphify_flow_extractor import (
    extract_graphify_flow_snapshot,
    strip_markdown_fences,
    wrap_in_function_if_needed,
)
from scripts.evaluate_graphify_cv import (
    EvaluationDataset,
    extract_code_from_sample_text,
    run_5fold_cv_for_extractor,
)


def test_empty_and_whitespace_code_input():
    """Verify that empty and whitespace-only strings return incomplete snapshots."""
    snap_empty = extract_graphify_flow_snapshot("", "test_empty")
    assert not snap_empty.is_complete
    assert snap_empty.parse_error == "Empty code input"
    assert len(snap_empty.signatures) == 0

    snap_space = extract_graphify_flow_snapshot("   \n\t  ", "test_space")
    assert not snap_space.is_complete
    assert snap_space.parse_error == "Empty code input"


def test_markdown_fence_stripping():
    """Verify markdown fences are cleanly stripped."""
    fenced = "```c\nint x = 42;\nmemcpy(buf, src, 10);\n```"
    stripped = strip_markdown_fences(fenced)
    assert "```" not in stripped
    assert "memcpy(buf, src, 10);" in stripped


def test_wrap_in_function_candidates():
    """Verify candidates generate valid compilable function wrappers."""
    snippet = "int a = 1;\nint b = 2"
    candidates = wrap_in_function_if_needed(snippet)
    assert len(candidates) >= 2
    assert any("__vuln_harness_func" in c for c in candidates)


def test_unguarded_memory_write():
    """Verify extraction of an unguarded memcpy memory sink."""
    code = """
    void handle_packet(char *user_input, int len) {
        char buffer[64];
        memcpy(buffer, user_input, len);
    }
    """
    snap = extract_graphify_flow_snapshot(code, "scenario_mem_1")
    assert snap.is_complete
    assert snap.parse_error is None
    assert len(snap.signatures) >= 1

    mem_sigs = [s for s in snap.signatures if s.sink_type == "MEMORY_WRITE"]
    assert len(mem_sigs) >= 1
    sig = mem_sigs[0]
    assert sig.sanitizer_type is None
    assert sig.guarded_target is None


def test_guarded_memory_write_bounds_check():
    """Verify that bounds check in if-statement correctly attaches to guarded sink."""
    code = """
    void safe_copy(char *src, int len) {
        char dest[128];
        if (len <= 128) {
            memcpy(dest, src, len);
        }
    }
    """
    snap = extract_graphify_flow_snapshot(code, "scenario_safe_mem")
    assert snap.is_complete

    mem_sigs = [s for s in snap.signatures if s.sink_type == "MEMORY_WRITE"]
    assert len(mem_sigs) >= 1
    # Check that at least one signature detected the BOUNDS_CHECK
    guarded = [s for s in mem_sigs if s.sanitizer_type == "BOUNDS_CHECK"]
    assert len(guarded) >= 1
    assert guarded[0].guarded_target in ("len", "dest")


def test_unguarded_pointer_dereference():
    """Verify detection of raw pointer dereference."""
    code = """
    struct vmcb_save_area *save = svm->vmcb->save;
    save->rip = next_rip;
    """
    snap = extract_graphify_flow_snapshot(code, "scenario_ptr_1")
    assert snap.is_complete
    assert snap.parse_error is None
    ptr_sigs = [s for s in snap.signatures if s.sink_type == "POINTER_DEREF"]
    assert len(ptr_sigs) >= 1
    assert ptr_sigs[0].flow_type == "POINTER_ACCESS"


def test_guarded_pointer_dereference_null_check():
    """Verify NULL check correctly attaches to pointer dereference."""
    code = """
    void update_state(struct node *ptr) {
        if (ptr != NULL) {
            ptr->val = 42;
        }
    }
    """
    snap = extract_graphify_flow_snapshot(code, "scenario_ptr_safe")
    assert snap.is_complete
    ptr_sigs = [s for s in snap.signatures if s.sink_type == "POINTER_DEREF"]
    assert len(ptr_sigs) >= 1
    assert ptr_sigs[0].sanitizer_type == "NULL_CHECK"
    assert ptr_sigs[0].guarded_target == "ptr"


def test_sanitizer_preference_sink_rejection():
    """Verify that NULL_CHECK does not sanitize SYSTEM_CALL and BOUNDS_CHECK does not sanitize POINTER_DEREF."""
    # NULL check around a system call -> should remain unsanitized because NULL_CHECK is invalid for SYSTEM_CALL
    code_sys_null = """
    void run_cmd(char *cmd) {
        if (cmd != NULL) {
            system(cmd);
        }
    }
    """
    snap_sys = extract_graphify_flow_snapshot(code_sys_null, "scenario_sys_null")
    assert snap_sys.is_complete
    sys_sigs = [s for s in snap_sys.signatures if s.sink_type == "SYSTEM_CALL"]
    assert len(sys_sigs) >= 1
    assert sys_sigs[0].sanitizer_type is None

    # Bounds check around a pointer dereference -> should remain unsanitized because BOUNDS_CHECK is invalid for POINTER_DEREF
    code_ptr_bounds = """
    void access_node(struct node *ptr, int len) {
        if (len <= 10) {
            ptr->val = 42;
        }
    }
    """
    snap_ptr = extract_graphify_flow_snapshot(code_ptr_bounds, "scenario_ptr_bounds")
    assert snap_ptr.is_complete
    ptr_sigs = [s for s in snap_ptr.signatures if s.sink_type == "POINTER_DEREF"]
    assert len(ptr_sigs) >= 1
    assert ptr_sigs[0].sanitizer_type is None


def test_system_call_sink():
    """Verify detection of system command execution sink."""
    code = """
    void execute_user_cmd(char *cmd) {
        system(cmd);
    }
    """
    snap = extract_graphify_flow_snapshot(code, "scenario_sys_1")
    sys_sigs = [s for s in snap.signatures if s.sink_type == "SYSTEM_CALL"]
    assert len(sys_sigs) >= 1
    assert sys_sigs[0].flow_type == "SYSTEM_INVOCATION"


def test_array_subscript_write():
    """Verify subscript index writes are captured as memory writes."""
    code = """
    void write_index(int *arr, int idx, int val) {
        arr[idx] = val;
    }
    """
    snap = extract_graphify_flow_snapshot(code, "scenario_arr_1")
    assert len(snap.signatures) >= 1
    assert any(s.sink_type == "MEMORY_WRITE" and s.flow_type == "ARRAY_INDEX_WRITE" for s in snap.signatures)


def test_evaluation_dataset_properties():
    """Verify EvaluationDataset count and scenario summary properties."""
    dataset = EvaluationDataset(
        texts=["txt1", "txt2", "txt3", "txt4"],
        labels=[1, 0, 0, 1],
        scenario_ids=["sc_1", "sc_1", "sc_2", "sc_3"],
    )
    assert dataset.total_samples == 4
    assert dataset.positive_count == 2
    assert dataset.negative_count == 2
    assert dataset.unique_scenarios == 3

    # Test empty dataset
    empty_dataset = EvaluationDataset(texts=[], labels=[], scenario_ids=[])
    assert empty_dataset.total_samples == 0
    assert empty_dataset.positive_count == 0
    assert empty_dataset.negative_count == 0
    assert empty_dataset.unique_scenarios == 0


def test_run_5fold_cv_for_extractor_empty():
    """Verify run_5fold_cv_for_extractor handles empty datasets without error."""
    empty_dataset = EvaluationDataset(texts=[], labels=[], scenario_ids=[])
    res = run_5fold_cv_for_extractor(extract_graphify_flow_snapshot, empty_dataset)
    assert res["mean_roc_auc"] == 0.5
    assert res["mean_pr_auc"] == 0.0
    assert res["seed_breakdown"] == []


def test_extract_code_from_sample_text():
    """Verify code snippet parsing from delimiter-formatted attempt string."""
    text_with_delim = "PREDICATE_PREFIX:some_pred | Code: memcpy(a, b, c);"
    extracted = extract_code_from_sample_text(text_with_delim)
    assert extracted == "memcpy(a, b, c);"

    plain_text = "int x = 10;"
    assert extract_code_from_sample_text(plain_text) == "int x = 10;"
