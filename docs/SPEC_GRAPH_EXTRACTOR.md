# Specification: AST Graph Data-Flow Extractor

**Status**: Proposal / Draft Specification  
**Target Implementation**: `scenarios/debate/graph_extractor.py`  
**Test Suite**: `tests/test_graph_extractor.py`  
**RFC Alignment**: [RFC_GRAPH_DATAFLOW_PRE_FILTER.md](RFC_GRAPH_DATAFLOW_PRE_FILTER.md)  
**Evaluator Target**: [graph_dataflow.py](../scenarios/debate/graph_dataflow.py)  

---

## 1. Executive Summary

This document specifies the technical contract for the **AST Graph Data-Flow Extractor** (`scenarios/debate/graph_extractor.py`). The extractor parses code attempt snippets using Python's Abstract Syntax Tree (`ast` module) and produces deterministic `FlowGraphSnapshot` instances.

The extracted snapshots are evaluated by `evaluate_graph_reachability` in `scenarios/debate/graph_dataflow.py` to classify security risk without relying on fragile TF-IDF text features.

---

## 2. Extraction Pipeline Architecture

```text
┌───────────────────────────┐
│ Code Attempt Snippet (str)│
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   ast.parse(code_text)    │ ── (On SyntaxError) ──► Fail Closed Snapshot (is_complete=False)
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   SecurityFlowVisitor     │
│   (ast.NodeVisitor)       │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Dynamic Variable Tracker  │ ── Map untrusted parameters, bounds checks, NULL checks
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ FlowGraphSnapshot Output  │ ── Guaranteed endpoint node presence & sink matching
└───────────────────────────┘
```

---

## 3. AST Identification Taxonomy

### 3.1 Source Node Identification (`UNTRUSTED_INPUT`)
An AST node is tagged as `UNTRUSTED_INPUT` if it satisfies any of the following rules:
1. **Function Parameters**: Any `ast.arg` in the top-level function signature (e.g. `def process(user_data, buf_len):`).
2. **Explicit Input Sources**: Calls to `input()`, `sys.argv`, `request.args`, `request.get_json()`, `socket.recv()`, or `file.read()`.
3. **External Variables**: Assignment targets receiving data from external function calls.

### 3.2 Sink Node Identification

The extractor recognizes four supported sink categories (`SUPPORTED_SINKS`) defined using semantic AST predicates:

| Sink Category | Target AST Node Types | Semantic Predicate & Pattern Description |
| :--- | :--- | :--- |
| `MEMORY_WRITE` | `ast.Assign`, `ast.AugAssign`, `ast.Call` | `ast.Assign` / `ast.AugAssign` where `target` is `ast.Subscript` (`buf[i] = val`), or `ast.Call` to qualified allowlisted memory callees (`memcpy`, `strcpy`, `memset`). |
| `ARRAY_INDEX` | `ast.Subscript` | Subscript index access on sequence types where the index variable is an untrusted operand (`data[index]`). |
| `POINTER_DEREF` | `ast.Attribute` | Attribute access on tracked pointer base variables (`ptr.value`). |
| `SYSTEM_CALL` | `ast.Call` | Invocations of qualified allowlisted system command functions (`os.system`, `subprocess.Popen`, `subprocess.run`, `eval`, `exec`). |

> [!NOTE]
> Every production sink node MUST populate the proof-bearing `target_var` metadata field (e.g. `sink_node["target_var"] = "i"`). The evaluator verifies `sig.guarded_target == sink_node.get("target_var")` to enforce target identity matching. `ast.UnaryOp` and `*ptr` syntax are excluded from `POINTER_DEREF` because Python expressions do not support C-style pointer dereference operator syntax. `ast.Assign` and `ast.AugAssign` act as `MEMORY_WRITE` sinks strictly when writing to subscripted array/buffer targets.

### 3.3 Sanitizer Evidence Mapping & Dominance Binding

Sanitizer evidence must be extracted from enclosing conditional guards (`ast.If`, `ast.Assert`, or ternary expressions `ast.IfExp`):

```python
VALID_SANITIZERS = {
    "BOUNDS_CHECK",
    "RANGE_VALIDATION",
    "NULL_CHECK",
    "COMMAND_SANITIZATION",
    "ALLOWLIST_CHECK",
}
```

#### Strict Dominance & Operand Identity Binding Rules
To prevent false-positive sanitization, `sanitizer_type` is assigned to a flow signature **only when both criteria are met**:
1. **Dominance**: The enclosing conditional guard dominates the sink node on the execution branch containing the sink.
2. **Operand Identity**: The variable evaluated in the guard is **identity-equivalent** to the sink's target index, pointer, or command argument.

- **`BOUNDS_CHECK` / `RANGE_VALIDATION`**:
  - AST Pattern: `ast.Compare` in `ast.If` checking length or upper/lower bound index (`idx < MAX_LEN`, `MIN_VAL <= index <= MAX_VAL`).
  - Target Binding (`guarded_target`): Must match the exact index variable evaluated at the `ARRAY_INDEX` or `MEMORY_WRITE` sink (e.g. `"idx"`). An unrelated length check like `len(data) <= BUF_SIZE` for `buf[i]` **does not** sanitize `i`.
- **`NULL_CHECK`**:
  - AST Pattern: Identity or truthiness checks (`ptr is not None`, `if ptr:`, `if not ptr: return`).
  - Target Binding (`guarded_target`): Must match the exact pointer base evaluated at the `POINTER_DEREF` sink (e.g. `"ptr"`).
- **`COMMAND_SANITIZATION` / `ALLOWLIST_CHECK`**:
  - AST Pattern: Membership checks (`cmd in ALLOWLIST`) or escaping functions (`shlex.quote(cmd)`).
  - Target Binding (`guarded_target`): Must match the exact command argument passed to the `SYSTEM_CALL` sink. An unrelated check like `arg2 in ALLOWLIST` for `subprocess.run(cmd)` **does not** sanitize `cmd`.

---

## 4. Snapshot Construction & Integrity Contract

The extractor must return a valid `FlowGraphSnapshot` complying with the following guarantees:

1. **Explicit Timestamps**: `created_at` must be passed explicitly from caller context (no wall-clock defaults).
2. **Endpoint Node Registry Integrity**: Every `source_id` and `sink_id` present in `signatures` **MUST** exist as a key in `snapshot.nodes`. Missing endpoints cause `evaluate_graph_reachability` to fail closed.
3. **Snapshot Immutability**: `snapshot.nodes` and `snapshot.signatures` must be stored as detached copies.
4. **Sink-Specific Sanitizer Matching**:
   - `MEMORY_WRITE` / `ARRAY_INDEX` $\rightarrow$ Requires `BOUNDS_CHECK` or `RANGE_VALIDATION`.
   - `POINTER_DEREF` $\rightarrow$ Requires `NULL_CHECK`.
   - `SYSTEM_CALL` $\rightarrow$ Requires `COMMAND_SANITIZATION` or `ALLOWLIST_CHECK`.

---

## 5. Fail-Closed Error Handling

If any of the following occur during extraction:
- `ast.parse` throws `SyntaxError` or `IndentationError`.
- Unhandled AST structures or malformed inputs are encountered.
- Non-numeric or non-finite metadata is provided.

The extractor **MUST** catch the exception and return:
```python
FlowGraphSnapshot(
    snapshot_id=snapshot_id,
    scenario_id=scenario_id,
    version=version,
    created_at=created_at,
    nodes={},
    signatures=[],
    is_complete=False,
    parse_error=str(error),
)
```
This guarantees that `evaluate_graph_reachability` evaluates incomplete or unparseable code snippets as `1.0` (High Risk / Reject).

---

## 6. Primary API Interface

```python
def extract_flow_graph_snapshot(
    code_text: str,
    scenario_id: str,
    snapshot_id: str,
    version: int,
    created_at: float,
) -> FlowGraphSnapshot:
    """
    Parses code_text using AST inspection and builds a deterministic FlowGraphSnapshot.
    
    Fails closed (returns is_complete=False) if syntax errors or parse failures occur.
    """
```

---

## 7. Verification Vectors & Test Plan

```python
# Multiline reference function snippets used in verification table below:

VULN_MEM_WRITE = """
def process(data, i):
    buf[i] = data
"""

GUARDED_MEM_WRITE = """
def process(data, i):
    if i < MAX_LEN:
        buf[i] = data
"""

VULN_ARRAY_INDEX = """
def get_elem(arr, index):
    return arr[index]
"""

GUARDED_ARRAY_INDEX = """
def get_elem(arr, index):
    if 0 <= index < len(arr):
        return arr[index]
"""

VULN_POINTER_DEREF = """
def inspect(ptr):
    return ptr.data
"""

GUARDED_POINTER_DEREF = """
def inspect(ptr):
    if ptr is not None:
        return ptr.data
"""

VULN_SYSTEM_CALL = """
def execute(cmd):
    import os
    os.system(cmd)
"""

GUARDED_SYSTEM_CALL_QUOTE = """
def execute(cmd):
    import os, shlex
    safe_cmd = shlex.quote(cmd)
    os.system(safe_cmd)
"""

GUARDED_SYSTEM_CALL_ALLOWLIST = """
def execute(cmd):
    import os
    if cmd in ALLOWED_COMMANDS:
        os.system(cmd)
"""
```

| Test Case | Code Snippet Pattern | Expected `FlowSignature` | Evaluator Risk Score |
| :--- | :--- | :--- | :--- |
| **Vulnerable Memory Write** | `VULN_MEM_WRITE` | `sink_type="MEMORY_WRITE"`, `sanitizer_type=None` | `1.0` (Reject) |
| **Guarded Memory Write** | `GUARDED_MEM_WRITE` | `sink_type="MEMORY_WRITE"`, `sanitizer_type="BOUNDS_CHECK"`, `guarded_target="i"` | `0.05` (Pass) |
| **Vulnerable Array Index** | `VULN_ARRAY_INDEX` | `sink_type="ARRAY_INDEX"`, `sanitizer_type=None` | `1.0` (Reject) |
| **Guarded Array Index** | `GUARDED_ARRAY_INDEX` | `sink_type="ARRAY_INDEX"`, `sanitizer_type="RANGE_VALIDATION"`, `guarded_target="index"` | `0.05` (Pass) |
| **Vulnerable Pointer Deref** | `VULN_POINTER_DEREF` | `sink_type="POINTER_DEREF"`, `sanitizer_type=None` | `1.0` (Reject) |
| **Guarded Pointer Deref** | `GUARDED_POINTER_DEREF` | `sink_type="POINTER_DEREF"`, `sanitizer_type="NULL_CHECK"`, `guarded_target="ptr"` | `0.05` (Pass) |
| **Vulnerable System Call** | `VULN_SYSTEM_CALL` | `sink_type="SYSTEM_CALL"`, `sanitizer_type=None` | `1.0` (Reject) |
| **Guarded System Call (Quote)** | `GUARDED_SYSTEM_CALL_QUOTE` | `sink_type="SYSTEM_CALL"`, `sanitizer_type="COMMAND_SANITIZATION"`, `guarded_target="cmd"` | `0.05` (Pass) |
| **Guarded System Call (Allowlist)** | `GUARDED_SYSTEM_CALL_ALLOWLIST` | `sink_type="SYSTEM_CALL"`, `sanitizer_type="ALLOWLIST_CHECK"`, `guarded_target="cmd"` | `0.05` (Pass) |
| **Syntax Error Code** | `def process(data): if data` | `is_complete=False`, `parse_error="..."` | `1.0` (Reject) |
