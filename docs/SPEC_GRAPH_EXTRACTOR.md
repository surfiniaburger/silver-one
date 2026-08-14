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

```
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

The extractor recognizes four supported sink categories (`SUPPORTED_SINKS`):

| Sink Category | Target AST Node Types | Pattern Description |
| :--- | :--- | :--- |
| `MEMORY_WRITE` | `ast.Assign`, `ast.AugAssign`, `ast.Call` | Array slice writes (`buf[i] = val`), buffer allocations, or calls to memory functions (`memcpy`, `strcpy`, `memset`). |
| `ARRAY_INDEX` | `ast.Subscript` | Index access on sequence types (`data[index]`). |
| `POINTER_DEREF` | `ast.Attribute`, `ast.UnaryOp` | Attribute access or dereference on pointer/reference variables (`ptr.value`, `*ptr`). |
| `SYSTEM_CALL` | `ast.Call` | Invocation of system commands (`os.system`, `subprocess.Popen`, `subprocess.run`, `eval`, `exec`). |

### 3.3 Sanitizer Evidence Mapping

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

- **`BOUNDS_CHECK` / `RANGE_VALIDATION`**:
  - AST Pattern: `ast.Compare` in `ast.If` checking length or upper/lower bound index (`idx < MAX_LEN`, `len(data) <= BUF_SIZE`).
  - Target Binding (`guarded_target`): The specific index or length variable checked (e.g. `"idx"`).
- **`NULL_CHECK`**:
  - AST Pattern: Identity or truthiness checks (`ptr is not None`, `if ptr:`, `if not ptr: return`).
  - Target Binding (`guarded_target`): The specific pointer variable checked (e.g. `"ptr"`).
- **`COMMAND_SANITIZATION` / `ALLOWLIST_CHECK`**:
  - AST Pattern: Membership checks (`cmd in ALLOWLIST`) or calls to escaping utilities (`shlex.quote(arg)`).
  - Target Binding (`guarded_target`): The command string argument.

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

| Test Case | Code Snippet Pattern | Expected `FlowSignature` | Evaluator Risk Score |
| :--- | :--- | :--- | :--- |
| **Vulnerable Memory Write** | `def f(data): buf[i] = data` | `sink_type="MEMORY_WRITE"`, `sanitizer_type=None` | `1.0` (Reject) |
| **Guarded Memory Write** | `def f(data): if i < MAX: buf[i] = data` | `sink_type="MEMORY_WRITE"`, `sanitizer_type="BOUNDS_CHECK"`, `guarded_target="i"` | `0.05` (Pass) |
| **Vulnerable Pointer Deref** | `def f(ptr): val = ptr.data` | `sink_type="POINTER_DEREF"`, `sanitizer_type=None` | `1.0` (Reject) |
| **Guarded Pointer Deref** | `def f(ptr): if ptr is not None: val = ptr.data` | `sink_type="POINTER_DEREF"`, `sanitizer_type="NULL_CHECK"`, `guarded_target="ptr"` | `0.05` (Pass) |
| **Syntax Error Code** | `def f(data): if data` | `is_complete=False`, `parse_error="..."` | `1.0` (Reject) |
