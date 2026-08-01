# Proposal: Integrating GEPA (Genetic Execution-Trace Prompt Adaptation) into BARRED Multi-Agent Swarm

## Executive Summary
This document proposes adopting **GEPA (Genetic Execution-Trace Prompt Adaptation, Berkeley / ICLR 2026)** to optimize prompt strategies and candidate refinement loops in the BARRED Multi-Agent Vulnerability Dataset Generation Swarm. Based on research highlighted by Avi Chawla (Daily Dose of Data Science), GEPA replaces scalar-reward RL (like GRPO) and naive single-turn retries by utilizing **full natural language execution traces** (debate arguments, judge rationales, verifier failure audits) to dynamically mutate system prompts across a Pareto frontier.

---

## 1. Context & Motivation

### The Signal Compression Problem in Standard RL & Retries
In LLM-based agent swarms, each execution rollout generates rich diagnostic artifacts:
- Pro and Con technical arguments
- Source, sink, and guard anchor extractions
- Judge adjudication rationales
- Verifier structured error audits (`verifier_logic_error`, `verifier_failed`, `suggested_correction`)

Standard Reinforcement Learning methods (such as GRPO) collapse this multi-thousand-token trace into a single scalar reward ($0$ or $1$). This discards thousands of bits of structured diagnostic feedback, requiring tens of thousands of rollouts to converge.

Similarly, naive retry loops attempt single-turn fixes without accumulating structural lessons across failure modes.

### The GEPA Paradigm (Avi Chawla / Berkeley)
GEPA treats the execution rollout trace as a first-class natural language artifact:
1. **Reflection LLM**: Rather than computing a numerical gradient, a Reflection LLM reads the full execution trace and feedback.
2. **Targeted Prompt Mutation**: The Reflection LLM diagnoses the exact failure mechanism (e.g., missing bounds check, unvalidated length parameter) and mutates the generator or debater system prompt.
3. **Pareto Selection**: Instead of collapsing population selection to a single global average, GEPA maintains a Pareto frontier of candidates that excel at specific sub-tasks or vulnerability classes.

---

## 2. Technical Comparison: Optimization Paradigms

| Feature / Axis | GRPO (RL) | MIPROv2 (DSPy) | GEPA (ICLR 2026) | BARRED Proposed Integration |
| :--- | :--- | :--- | :--- | :--- |
| **Feedback Signal** | Scalar Reward ($0.0 - 1.0$) | Task Score + Few-Shot Examples | Full Natural Language Trace ($\mu_f$) | Debate Adjudication + Verifier Audit |
| **Optimization Target** | Model Weights via Policy Gradients | Instructions & Static Examples | Dynamic Multi-Module System Prompts | Generator, Pro/Con Debaters, Judge Prompts |
| **Sample Efficiency** | Low ($10,000+$ rollouts) | Medium (Hundreds of examples) | **High ($10-50\times$ compute reduction)** | **High ($20-100$ failure traces)** |
| **Population Selection** | Mean Baseline Advantage | Bayesian Optimization | **Pareto Frontier Selection** | **Taxonomy-based Pareto Frontier** |
| **Infrastructure Needed** | GPU Training Cluster | Python Execution | Pure LLM Reflection / Zero GPU Training | Lightweight Python Orchestrator |

---

## 3. Proposed Architecture for BARRED Integration

```
                 [ CVE Seed + Predicate ]
                            │
                            ▼
              ┌───────────────────────────┐
              │ Candidate Code Generator  │◄───────────────────────┐
              └─────────────┬─────────────┘                        │
                            │                                      │
                            ▼                                      │
              ┌───────────────────────────┐                        │
              │   BARRED Debater Swarm    │                        │
              │   (Pro vs Con Debaters)   │                        │
              └─────────────┬─────────────┘                        │  Feedback Function (μ_f)
                            │                                      │  - Verifier audit report
                            ▼                                      │  - Judge rationale
              ┌───────────────────────────┐                        │  - Anchor mismatches
              │   Judge & Verifier Audit  │                        │
              └─────────────┬─────────────┘                        │
                            │                                      │
                   Accepted?│                                      │
             ┌──────────────┴──────────────┐                       │
             │                             │                       │
           [ YES ]                       [ NO ] ───────────────────┘
             │                             │
             ▼                             ▼
   [ Final Corpus Row ]          [ GEPA Reflection LLM ]
                                 (Diagnoses trace & updates
                                  Debater / Generator Prompts)
```

### Component Details

#### A. Structured Feedback Function ($\mu_f$)
When an attempt fails during debate or verification, the system constructs a rich feedback tuple $\mu_f$:
$$\mu_f = \{ \text{reject\_reason}, \text{pro\_argument}, \text{con\_argument}, \text{judge\_adjudication}, \text{verifier\_report} \}$$

#### B. Reflection Model Diagnosis Loop
The Reflection LLM (e.g., `gemma4:31b-cloud` or `gpt-oss:120b-cloud`) inspects $\mu_f$ and executes a diagnosis step:
- **Failure Diagnosis**: Identifies why the generated candidate code or debater argument failed.
- **Instruction Strategy Mutation**: Generates a targeted system prompt update for the Pro generator or debater.

#### C. Pareto Frontier Selection across Vulnerability Taxonomies
Rather than enforcing a single global system prompt across all CVE types, GEPA maintains candidate prompt specialists for specific vulnerability taxonomy buckets:
1. **Memory Safety & Pointer Arithmetic**: Buffer overflows, use-after-free, double-free.
2. **Integer Safety & Length Calculation**: Integer overflow, length truncation, wrap-around.
3. **Concurrency & State Invariants**: Race conditions, SRCU/RCU lock-free primitives.
4. **Logic & Input Validation**: Script inclusion, path traversal, untrusted deserialization.

Pareto selection ensures that prompt variants excelling at complex kernel race conditions are retained even if their global average across simpler buffer overflows is lower.

---

## 4. A2A Housed Agent Architecture & Coupling Constraints

In accordance with **Dave Farley's Coupling Principles** in `AI_REVIEWER_ENGINEERING_PRINCIPLES.md`, the Reflection engine is implemented as a **decoupled A2A (Agent-to-Agent) micro-service (`ReflectorAgent`)** running on Port `8004`, managed directly by `start_stack.sh`.

```
        ┌─────────────────────────────────────────────────────────────┐
        │                 ReflectorAgent (Port 8004)                  │
        │  - Pareto Prompt Pool per CVE Taxonomy                      │
        │  - Mutation History Ledger (Loop Memory)                    │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                         1. Get Adapted System Prompt
                         (Attempt 1: default prompt)
                         (Attempt 2+: reflected prompt)
                                       │
                                       ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                         BARRED Execution Swarm                           │
 │                                                                          │
 │   [Generator] ──► [Pro / Con Debaters] ──► [Judge] ──► [Verifier]        │
 └─────────────────────────────────────┬────────────────────────────────────┘
                                       │
                         2. Execution Trace (μ_f)
                         (If attempt fails audit)
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                 ReflectorAgent (Port 8004)                  │
        │  - Analyzes μ_f (verifier failure + judge rationale)         │
        │  - Diagnoses failure mechanism                              │
        │  - Mutates System Prompt for next retry                     │
        │  - Updates Pareto Frontier Registry                         │
        └─────────────────────────────────────────────────────────────┘
```

### Key Architectural Boundaries

1. **Decoupled A2A Service Boundary (`ReflectorAgent`)**:
   - `run_batch.py` does not contain inline prompt mutation logic. Instead, it communicates with `ReflectorAgent` via explicit HTTP Pydantic payloads (`ReflectRequest`, `ReflectResponse`).
   - If `ReflectorAgent` is offline or times out, `run_batch.py` falls back gracefully to the static baseline prompt without failing the batch run.

2. **Execution Lifecycle & Loop Memory**:
   - **Initial Attempt (Attempt 1)**: Before starting a seed debate, `run_batch.py` asks `ReflectorAgent` for the best historical prompt variant for that CVE taxonomy. If no history exists, it uses the default baseline.
   - **Refinement Attempt (Attempt 2+)**: If Attempt 1 fails verifier/judge audit, `run_batch.py` posts the failure trace $\mu_f$ to `ReflectorAgent`. The reflector diagnoses the defect (e.g. `anchors_too_few_after_normalization`), mutates the prompt directive for Attempt 2, and updates its Pareto ledger (`artifacts/gepa/pareto_frontier.json`).

3. **Isolated Memory Ledger**:
   - The Pareto state and mutation history reside strictly within `artifacts/gepa/`, preventing unstable LLM contract churn from leaking into core attempt log schemas.

---

## 5. Implementation Steps & Roadmap

1. **Step 1: Metric & Trace Extractor**: Update `run_batch.py` and `scenarios/debate/run_batch.py` to extract full $\mu_f$ execution traces on failed attempts.
2. **Step 2: GEPA Reflection Operator (`reflector_agent.py`)**: Implement the housed A2A Reflector Agent on Port `8004` (managed in `start_stack.sh`) using FastAPI and Pydantic contract validation.
3. **Step 3: Pareto Population Registry**: Implement local Pareto prompt pool persistence in `artifacts/gepa/pareto_frontier.json` per CVE taxonomy bucket.
4. **Step 4: Benchmarking & Telemetry**: Measure token spend per accepted row and anchor completeness rates across standard retries vs. GEPA-reflected retries.
