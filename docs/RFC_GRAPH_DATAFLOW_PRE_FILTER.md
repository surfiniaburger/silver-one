# RFC: Graph-Topology Semantic Data-Flow Pre-Filter

- **Status:** Proposed / Under Review  
- **Date:** August 2026  
- **Author:** Silver-One Development & AI Research Team  
- **Target Component:** `scenarios/debate/pre_filter.py` & `scripts/train_pre_filter.py`  
- **Primary References:** [PROPOSAL_PROBLEM_DEFINITION_OPEN_EXPLORATION.md](PROPOSAL_PROBLEM_DEFINITION_OPEN_EXPLORATION.md), [RFC_PRE_FILTER_STRATIFIED_CV.md](RFC_PRE_FILTER_STRATIFIED_CV.md), [GRAPH_PRE_FILTER_HYPOTHESES.md](GRAPH_PRE_FILTER_HYPOTHESES.md), and `getzep/graphiti` Open-Source Architecture.

---

## 1. Executive Summary & Context

Diagnostic evaluations of the 3-stage acceptance pre-filter in [RFC_PRE_FILTER_STRATIFIED_CV.md](RFC_PRE_FILTER_STRATIFIED_CV.md) demonstrated that surface text representations (TF-IDF char n-grams, XGBoost, SetFit embeddings) fail to discriminate between valid and invalid vulnerability candidates on unseen scenario holdouts ($ROC-AUC = 0.4052$, $PR-AUC = 0.2864$, $n=247$ deduplicated samples).

This RFC proposes re-architecting Stage B of the pre-filter from surface text classification to a **Deterministic Graph-Topology Semantic Data-Flow Classifier**, drawing directly from the open-source **Graphiti** context engine architecture (`getzep/graphiti`).

By mapping code snippets into **semantic flow nodes and signature triples** rather than high-dimensional text embeddings, the pre-filter evaluates **untrusted-source-to-vulnerable-sink reachability** deterministically, eliminating false positives caused by surface keywords (`malloc`, `memcpy`, `overflow`).

---

## 2. Motivation & Lessons from Graphiti

### 2.1 Why Text Embeddings Fail on Code Guardrails
Standard vector embeddings group inputs by **topic similarity**. In security evaluation, valid code attempts and invalid counterfactual attempts both contain security keywords and structural boilerplates. Consequently, embedding models group them into the same vector neighborhood, producing random-chance out-of-fold performance ($ROC-AUC = 0.4052$).

### 2.2 Graphiti’s "Graph Topology Without Embeddings" Blueprint
Zep’s open-source Graphiti engine demonstrates that pattern recognition across complex events can be achieved without vector clustering by:
1. **Extracting Deterministic Fact Signatures**: Reducing entity interactions into canonical tuples: `(Source_Entity, Target_Entity, Relation_Type)`.
2. **Building Episode Graphs**: Treating episodes as nodes and shared signature pairs as edges.
3. **Graph Topology Clustering**: Finding connected components via pure graph algorithms (BFS/DFS graph topology), isolating structural chains before passing them to an LLM.

---

## 3. Architecture & Data-Flow Abstraction

```text
                              Code Attempt Input Snippet
                                           │
                                           ▼
                            Stage A: Negative Rule Sanity Check
                               (Sub-ms junk rejection)
                                           │ (Pass)
                                           ▼
                            AST & Control/Data-Flow Parser
                                           │
                                           ▼
                            Semantic Flow Node Extraction
                       (SourceNode, TransformNode, SinkNode)
                                           │
                                           ▼
                           Deterministic Signature Generation
              (source_id, sink_id, flow_type, sanitizer_type, guarded_target)
                                           │
                                           ▼
                        Stage B: Graph Reachability Classifier
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          Unsanitized Reachable Path                      Guarded / Safe Flow
               (High Risk => 1.0)                          (Low Risk => 0.05)
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                          Probability Score & Advisory Filter
                           (graph_risk_threshold = 0.10)
               [Risk >= 0.10 => Flag/Reject | Risk < 0.10 => Pass]
```

### 3.1 Flow Nodes, Graph Snapshots & Sink-Specific Evidence

The code AST is mapped into a lightweight **Semantic Flow Graph Snapshot**:

- **Flow Nodes**:
  - `SourceNode`: External inputs, parameters, network buffers, user-controlled pointers.
  - `TransformNode`: Arithmetic operations, bitwise shifts, memory allocations (`malloc`), string copies (`strcpy`).
  - `SanitizerNode`: Explicit bounds checks (`if (len > MAX)`), range validations, NULL checks, command sanitization, allowlist checks.
  - `SinkNode`: Memory writes, pointer dereferences, array index access, system calls.

- **Sink-Specific Sanitizer Evidence & Signature Triples**:
  Generic `is_guarded` flags (such as treating a NULL check as a guard for a buffer overflow) are prohibited. Flow signatures require **sink-matched sanitizer evidence**:
  - `ARRAY_INDEX` and `MEMORY_WRITE` sinks require `BOUNDS_CHECK` or `RANGE_VALIDATION` targeting the index/length variable.
  - `POINTER_DEREF` sinks require `NULL_CHECK` targeting the specific pointer.
  - `SYSTEM_CALL` sinks require `COMMAND_SANITIZATION` or `ALLOWLIST_CHECK` targeting the command argument.

  ```python
  @dataclass(frozen=True)
  class FlowSignature:
      source_id: str                      # Stable source node identifier
      sink_id: str                        # Stable sink node identifier
      source_type: str                    # e.g., "UNTRUSTED_INPUT"
      sink_type: str                      # "MEMORY_WRITE", "POINTER_DEREF", "ARRAY_INDEX", "SYSTEM_CALL"
      flow_type: str                      # e.g., "DIRECT_ASSIGN", "ARITHMETIC_OFFSET"
      sanitizer_type: str | None = None   # "BOUNDS_CHECK", "RANGE_VALIDATION", "NULL_CHECK", "COMMAND_SANITIZATION", "ALLOWLIST_CHECK"
      guarded_target: str | None = None   # Specific variable/property proven safe by sanitizer
      invalid_at: float | None = None     # Timestamp if edge invalidated by counterfactual patch
  ```

- **Graph Snapshot Identity & Integrity Contract**:
  ```python
  @dataclass
  class FlowGraphSnapshot:
      snapshot_id: str
      scenario_id: str
      version: int
      nodes: dict[str, dict]
      signatures: list[FlowSignature]
      created_at: float
      is_complete: bool = True
      parse_error: str | None = None
  ```

---

## 4. Stage B Graph Reachability Classifier Algorithm

Instead of fitting an XGBoost model on raw TF-IDF text features, Stage B evaluates the topological properties of the extracted `FlowGraphSnapshot`:

```python
SUPPORTED_SINKS = {"MEMORY_WRITE", "POINTER_DEREF", "ARRAY_INDEX", "SYSTEM_CALL"}

def is_sanitizer_valid_for_sink(sink_type: str, sanitizer_type: str | None) -> bool:
    """Verifies that sanitizer proof matches specific sink requirements."""
    if not sanitizer_type:
        return False
    if sink_type in ("MEMORY_WRITE", "ARRAY_INDEX"):
        return sanitizer_type in ("BOUNDS_CHECK", "RANGE_VALIDATION")
    if sink_type == "POINTER_DEREF":
        return sanitizer_type == "NULL_CHECK"
    if sink_type == "SYSTEM_CALL":
        return sanitizer_type in ("COMMAND_SANITIZATION", "ALLOWLIST_CHECK")
    return False

def evaluate_graph_reachability(
    graph_snapshot: FlowGraphSnapshot,
    as_of: float | None = None,
) -> float:
    """
    Computes deterministic risk score based on source-to-sink graph topology.
    Fails closed (returns 1.0 / High Risk) if evidence is incomplete, parse errors occur,
    or signature endpoint nodes are missing from graph_snapshot.nodes.
    
    - Returns 1.0 (High Risk) for unsanitized paths, incomplete graphs, or invalid endpoints.
    - Returns 0.05 (Low Risk) for verified guarded or safe flows.
    """
    # Fail closed on incomplete extraction or parse error
    if not graph_snapshot.is_complete or graph_snapshot.parse_error is not None:
        return 1.0

    eval_time = as_of if as_of is not None else graph_snapshot.created_at

    active_signatures = [
        sig for sig in graph_snapshot.signatures
        if sig.invalid_at is None or sig.invalid_at > eval_time
    ]

    # Validate that signature endpoints exist in graph_snapshot.nodes (fail closed if missing)
    if graph_snapshot.nodes:
        for sig in active_signatures:
            if sig.source_id not in graph_snapshot.nodes or sig.sink_id not in graph_snapshot.nodes:
                return 1.0

    # Check for unhandled or unknown sink types in active signatures (fail closed)
    for sig in active_signatures:
        if sig.sink_type not in SUPPORTED_SINKS:
            return 1.0

    for sig in active_signatures:
        if sig.source_type == "UNTRUSTED_INPUT" and sig.sink_type in SUPPORTED_SINKS:
            # Check if sink-specific sanitizer proof is present on the path
            if not is_sanitizer_valid_for_sink(sig.sink_type, sig.sanitizer_type):
                return 1.0  # Confirmed unsanitized reachability path -> High Risk (Reject)

    return 0.05  # All flows guarded or safe -> Low Risk (Pass)

def is_graph_candidate_rejected(
    graph_snapshot: FlowGraphSnapshot,
    as_of: float | None = None,
    risk_threshold: float = 0.10,
) -> bool:
    """
    Evaluates whether candidate should be rejected based on advisory risk_threshold (default 0.10).
    Returns True if risk_score >= risk_threshold, False otherwise.
    """
    score = evaluate_graph_reachability(graph_snapshot, as_of=as_of)
    return score >= risk_threshold

```

### 4.1 Advisory Threshold Contract
- **Classifier Threshold**: `graph_risk_threshold = 0.10`.
- **Decision Rule**:
  - `risk_score >= 0.10` $\rightarrow$ **Reject** (un-sanitized source-to-sink reachability path detected, or incomplete extraction).
  - `risk_score < 0.10` $\rightarrow$ **Pass** (guarded flow score $0.05$ or disconnected safe flow score $0.05$).

---

## 5. Handling Counterfactuals & Bi-Temporal Invalidation

In multi-agent debate, debaters frequently introduce counterfactual patches (e.g., adding `if (offset < max_size)` to an insecure snippet).

Following Graphiti's **fact invalidation model**:
- **Fact Addition**: When a debater adds a bounds check node, a `SanitizerNode` is inserted into the graph snapshot.
- **Edge Invalidation**: The reachability edge `(source_id, sink_id)` is marked `invalid_at` timestamp $T$, breaking the un-guarded reachability path.
- **Dynamic Snapshot Re-evaluation**: When evaluated at timestamp $\text{as\_of} \ge T$, the reachability classifier skips invalidated signatures. If no other active unguarded reachability paths remain, the risk score updates to $0.05$ (guarded/pass). If another active path still lacks valid sanitizer evidence, the risk score remains $1.0$ (reject).

---

## 6. Experimental Verification Plan & Authoritative Acceptance Checklist

Evaluation will follow the rigorous statistical protocol in [GRAPH_PRE_FILTER_HYPOTHESES.md](GRAPH_PRE_FILTER_HYPOTHESES.md).

### 6.1 Authoritative Acceptance Bounds
Candidate graph models must meet the identical authoritative acceptance checklist across $N=5$ paired partition seeds (`--seed 42, 1337, 2026, 7, 99`):

1. **Zero Logic Error Rate**: $\text{accepted\_logic\_error\_rate} = \frac{\text{Accepted Corpus Rows with Verifier Logic Error}}{\text{Total Accepted Corpus Rows}} = 0.0$. If $\text{Total Accepted Corpus Rows} == 0$, the criterion fails hard (declared unacceptable due to zero-yield collapse).
2. **ROC-AUC Floor**: Pooled Out-of-Fold $\text{ROC-AUC} \ge 0.7000$ (Baseline: $0.4052$).
3. **PR-AUC Floor**: Pooled Out-of-Fold $\text{PR-AUC} \ge 0.6000$ (Baseline: $0.2864$).
4. **Sensitivity / TPR Floor**: Sensitivity (TPR) at advisory threshold ($0.10$) $\ge 0.9000$.
5. **Statistical Significance**: Paired-seed statistical significance ($p < \alpha_{\text{adjusted}}$ via Holm Step-Down Procedure).
6. **Non-Parametric Confidence Interval**: Hodges-Lehmann 95% Confidence Interval for $\Delta\text{ROC-AUC}$ strictly excludes $0.0$.


