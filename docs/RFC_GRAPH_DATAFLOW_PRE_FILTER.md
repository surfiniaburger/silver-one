# RFC: Graph-Topology Semantic Data-Flow Pre-Filter

- **Status:** Proposed / Under Review  
- **Date:** August 2026  
- **Author:** Silver-One Development & AI Research Team  
- **Target Component:** `scenarios/debate/pre_filter.py` & `scripts/train_pre_filter.py`  
- **Primary References:** [PROPOSAL_PROBLEM_DEFINITION_OPEN_EXPLORATION.md](file:///Users/surfiniaburger/Desktop/modular-metacog-swarm-v3/agent_training/silver-one/docs/PROPOSAL_PROBLEM_DEFINITION_OPEN_EXPLORATION.md), [RFC_PRE_FILTER_STRATIFIED_CV.md](file:///Users/surfiniaburger/Desktop/modular-metacog-swarm-v3/agent_training/silver-one/docs/RFC_PRE_FILTER_STRATIFIED_CV.md), [GRAPH_PRE_FILTER_HYPOTHESES.md](file:///Users/surfiniaburger/Desktop/modular-metacog-swarm-v3/agent_training/silver-one/docs/GRAPH_PRE_FILTER_HYPOTHESES.md), and `getzep/graphiti` Open-Source Architecture.

---

## 1. Executive Summary & Context

Diagnostic evaluations of the 3-stage acceptance pre-filter in [RFC_PRE_FILTER_STRATIFIED_CV.md](file:///Users/surfiniaburger/Desktop/modular-metacog-swarm-v3/agent_training/silver-one/docs/RFC_PRE_FILTER_STRATIFIED_CV.md) demonstrated that surface text representations (TF-IDF char n-grams, XGBoost, SetFit embeddings) fail to discriminate between valid and invalid vulnerability candidates on unseen scenario holdouts ($ROC-AUC = 0.4052$, $PR-AUC = 0.2864$, $n=247$ deduplicated samples).

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

```
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
                    (Source_Var, Sink_Var, Flow_Type, Guard_State)
                                           │
                                           ▼
                        Stage B: Graph Reachability Classifier
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          Unsanitized Reachable Path                      Guarded / Disconnected
               (High Risk => 1.0)                          (Low Risk => 0.0)
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                           Probability Score & Advisory Filter
                              (xgb_low_threshold = 0.10)
```

### 3.1 Flow Nodes & Edge Types

The code AST is mapped into a lightweight **Semantic Flow Graph**:

- **Flow Nodes**:
  - `SourceNode`: External inputs, parameters, network buffers, user-controlled pointers.
  - `TransformNode`: Arithmetic operations, bitwise shifts, memory allocations (`malloc`), string copies (`strcpy`).
  - `SanitizerNode`: Explicit bounds checks (`if (len > MAX)`), range validations, NULL checks.
  - `SinkNode`: Memory writes, pointer dereferences, array index access, system calls.

- **Signature Triples**:
  ```python
  @dataclass(frozen=True)
  class FlowSignature:
      source_type: str       # e.g., "UNTRUSTED_INPUT"
      sink_type: str         # e.g., "MEMORY_WRITE"
      flow_type: str         # e.g., "DIRECT_ASSIGN", "ARITHMETIC_OFFSET"
      is_guarded: bool       # True if SanitizerNode traverses the path
  ```

---

## 4. Stage B Graph Reachability Classifier Algorithm

Instead of fitting an XGBoost model on raw TF-IDF text features, Stage B evaluates the topological properties of the extracted Semantic Flow Graph:

```python
def evaluate_graph_reachability(signature_list: list[FlowSignature]) -> float:
    """
    Computes deterministic risk score based on source-to-sink graph topology.
    """
    if not signature_list:
        return 0.0  # Disconnected graph -> Low risk
        
    for sig in signature_list:
        # High risk: Untrusted source reaches vulnerable sink without a sanitizer guard
        if sig.source_type == "UNTRUSTED_INPUT" and sig.sink_type == "MEMORY_WRITE":
            if not sig.is_guarded:
                return 1.0  # Confirmed reachability path -> High probability
                
    return 0.05  # Guarded or safe flow -> Low probability
```

---

## 5. Handling Counterfactuals & Bi-Temporal Invalidation

In multi-agent debate, debaters frequently introduce counterfactual patches (e.g., adding `if (offset < max_size)` to an insecure snippet).

Following Graphiti's **fact invalidation model**:
- **Fact Addition**: When a debater adds a bounds check node, a `SanitizerNode` is inserted into the flow graph.
- **Edge Invalidation**: The reachability edge `(UNTRUSTED_INPUT, MEMORY_WRITE, DIRECT_ASSIGN)` is marked `invalid_at` timestamp $T$, breaking the un-guarded reachability path.
- **Dynamic Re-evaluation**: The reachability classifier immediately re-evaluates the graph, updating the risk score from $1.0$ (vulnerable) to $0.05$ (guarded).

---

## 6. Experimental Verification Plan

Evaluation will follow the rigorous statistical protocol in [GRAPH_PRE_FILTER_HYPOTHESES.md](file:///Users/surfiniaburger/Desktop/modular-metacog-swarm-v3/agent_training/silver-one/docs/GRAPH_PRE_FILTER_HYPOTHESES.md):

1. **Benchmark Corpus**: $n=247$ near-deduplicated out-of-fold samples across 51 unique scenario groups (`artifacts/models_near_dedup_test/`).
2. **Cross-Validation Harness**: 5-Fold Stratified Scenario-Grouped CV (`scripts/train_pre_filter.py`).
3. **Primary Acceptance Bounds**:
   - `accepted_logic_error_rate == 0.0`
   - Pooled Out-of-Fold $\text{ROC-AUC} \ge 0.7000$ (Baseline: $0.4052$)
   - Pooled Out-of-Fold $\text{PR-AUC} \ge 0.6000$ (Baseline: $0.2864$)
   - Paired-seed statistical significance ($p < \alpha_{\text{adjusted}}$ via Holm Step-Down).
