# GEPA-First Seed Generation: Technical Guide

This document explains the methodology and goals behind the 50 high-fidelity seeds generated for the BARRED synthetic data pipeline.

## 1. Methodology: The GEPA-First Flow

Traditional synthetic data generation often starts with a generic prompt ("Generate a SQL injection example"). Our approach, **GEPA-First (Generative Explanation of Program Anomalies)**, flips this by starting with **ground-truth reality**.

### Phase A: Clean-Room Selection
We pull "Seeds" from the 1.5GB **[CVEFixes](https://www.kaggle.com/datasets/girish17019/cvefixes-vulnerable-and-fixed-code)** dataset. To protect the integrity of our evaluation set, we implement a **3-Tier Deduplication** layer:
1.  **Exact Hash**: Rejects identical code snippets.
2.  **Normalized Hash**: Rejects code that differs only in whitespace, comments, or indentation.
3.  **Fuzzy Shingling**: Rejects "near-duplicates" (similarly structured code) using Jaccard similarity on 5-gram tokens.
*This ensures our training data never "leaks" snippets from the 5,000-sample evaluation benchmark.*

### Phase B: Predicate Discovery (The GEPA Signal)
Raw CVE data only provides a binary "Vulnerable/Safe" label. This is a low-signal starting point for a debate.
We use a **Frontier Model (gpt-oss:120b-cloud)** acting as the **GEPA Explainer** to analyze the raw seed and generate:
- **A Falsifiable Predicate**: A specific technical claim (e.g., *"The code is vulnerable to an integer overflow in the packet length calculation at line 42"*).
- **Evidence Hooks**: Concrete code features that support the claim.
- **Proof Requirements**: What a Skeptic would need to see to definitively refute the claim.

## 2. Goals: Why GEPA-First?

### Goal 1: Moving Beyond "Vulnerable vs Safe"
By transforming a binary label into a specific technical mechanism, we force the BARRED pipeline to debate **subtle technical invariants** rather than rhetorical style.

### Goal 2: High-Fidelity Training (SecurityDecisionGuard)
The ultimate goal of this pipeline is to generate a **Deep Reasoning Corpus**. 
- The **Seeds** provide real-world complexity (C/C++ memory safety).
- The **BARRED Generator** creates "Boundary Variants" (subtle changes that flip the verdict).
- The **Judge** provides a "GEPA Feedback Trace" (a multi-paragraph explanation of the mechanism).

When we fine-tune a 7B model on this data, it doesn't just learn to guess "Vulnerable"; it learns to **think like a Senior Security Architect**, adjudicating code based on evidence hooks and mechanism-level invariants.

### Goal 3: Teacher-Student Capacity Alignment
By using a **Frontier Teacher** for the "Predicate Discovery" but a **Mid-Tier/Local Teacher** for the "Variant Generation," we ensure the student model learns patterns it can actually replicate without being overwhelmed by frontier-level abstraction.

## 3. Seed Dataset Schema

Each entry in the generated JSONL seed dataset conforms to the following schema:

```json
{
  "topic": "int func(int a) { ... }",
  "predicate": "The code is vulnerable to an out-of-bounds write in `func`...",
  "gepa_info": {
    "predicate": "The code is vulnerable to an out-of-bounds write in `func`...",
    "evidence_hooks": ["`func`", "`arr[idx]`"],
    "uncertainty": "Low",
    "proof_requirements": "Provide idx >= size"
  },
  "language": "c",
  "original_safety": "vulnerable",
  "anchors": ["func", "arr[idx]"]
}
```

### Schema Fields
- **`topic`** (`str`): The verbatim source code snippet under investigation.
- **`predicate`** (`str`): The primary falsifiable claim about the snippet's vulnerability or safety mechanism.
- **`gepa_info`** (`dict`): Structured payload containing the model explanation, evidence hooks, uncertainty rating, and proof requirements.
- **`language`** (`str`): Source programming language identifier (`c`, `cpp`, etc.).
- **`original_safety`** (`str`): Ground-truth baseline label (`vulnerable` or `safe`).
- **`anchors`** (`list[str]`): List of verified, non-generic identifier tokens and verbatim code spans used by the B-gate verifier to audit debate groundedness.

