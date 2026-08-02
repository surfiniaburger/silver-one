# Proposal: Integrating GEPA (Genetic-Pareto Prompt Adaptation) into BARRED Multi-Agent Swarm

## Executive Summary
This document proposes adopting **GEPA (Genetic-Pareto Prompt Adaptation)**—based on the research paper *"GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"* by Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, et al. (ICLR 2026 / [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)) as highlighted by Avi Chawla (Daily Dose of Data Science)—to optimize prompt strategies and candidate refinement loops in the BARRED Multi-Agent Vulnerability Dataset Generation Swarm. GEPA replaces scalar-reward RL (like GRPO) and naive single-turn retries by utilizing **full natural language execution traces** (debate arguments, judge rationales, verifier failure audits) to dynamically mutate system prompts across a Pareto frontier.

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

### The GEPA Paradigm (Agrawal et al. / UC Berkeley & Stanford)
GEPA treats the execution rollout trace as a first-class natural language artifact:
1. **Reflection LLM**: Rather than computing a numerical gradient, a Reflection LLM reads the full execution trace and feedback ($\mu_f$).
2. **Targeted Prompt Mutation**: The Reflection LLM diagnoses the exact failure mechanism (e.g., missing bounds check, unvalidated length parameter) and mutates the generator or debater system prompt.
3. **Pareto Selection**: Instead of collapsing population selection to a single global average, GEPA maintains a Pareto frontier of candidate prompts that excel at specific sub-tasks or vulnerability classes.

---

## 2. Technical Comparison & Metric Definitions

### Token Efficiency Metric Definition
To evaluate GEPA's performance in Step 4, **Token Efficiency Ratio** is formally defined as:

$$\text{Token Efficiency Ratio} = \frac{\sum (\text{Prompt Tokens} + \text{Completion Tokens across all attempts for a seed})}{\text{Number of Accepted, Verifier-Passed Corpus Rows Produced}}$$

- **Baseline**: Baseline swarm runs without prompt adaptation (`pilot-v1-calibrated-v`).
- **Benchmark Sample Size**: Evaluated across a benchmark suite of $20 - 100$ failure traces.
- **Target Metric**: A $10 - 50\times$ improvement in Token Efficiency Ratio (equivalent to a $90 - 98\%$ reduction in tokens per accepted row) compared to the un-adapted baseline. In contrast, the GEPA paper reports up to a $35\times$ reduction in LLM rollouts on standard benchmarks.

### Comparison Table

| Feature / Axis | GRPO (RL) | MIPROv2 (DSPy) | GEPA (Agrawal et al., ICLR 2026) | BARRED Proposed Integration |
| :--- | :--- | :--- | :--- | :--- |
| **Feedback Signal** | Scalar Reward ($0.0 - 1.0$) | Task Score + Few-Shot Examples | Full Natural Language Trace ($\mu_f$) | Debate Adjudication + Verifier Audit |
| **Optimization Target** | Model Weights via Policy Gradients | Instructions & Static Examples | Dynamic Multi-Module System Prompts | Generator, Pro/Con Debaters, Judge Prompts |
| **Sample Efficiency** | Low ($10,000+$ rollouts) | Medium (Hundreds of examples) | **High (Up to $35\times$ rollout reduction)** | **Target: $10-50\times$ token efficiency ratio improvement over $20-100$ traces** |
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

In accordance with **Dave Farley's Coupling Principles** in `AI_REVIEWER_ENGINEERING_PRINCIPLES.md`, the Reflection engine will be implemented as a **decoupled A2A (Agent-to-Agent) micro-service (`ReflectorAgent`)** running on Port `8004`, managed directly by `start_stack.sh`.

```
        ┌─────────────────────────────────────────────────────────────┐
        │                 ReflectorAgent (Port 8004)                  │
        │  - Pareto Prompt Pool per CVE Taxonomy                      │
        │  - Mutation History Ledger (Loop Memory)                    │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                         1. Query Pareto Prompt Specialist
                         (Returns taxonomy history prompt if available,
                          or default baseline if history is empty)
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
   - `run_batch.py` will not contain inline prompt mutation logic. Instead, it will communicate with `ReflectorAgent` via explicit HTTP Pydantic payloads (`ReflectRequest`, `ReflectResponse`).
   - All reflector LLM calls in the BARRED flow route through `ReplayManager.acompletion(...)`, while structured JSON response parsing uses `call_structured()`.
   - **Mode-Specific Fallback Policy**:
     - **Normal Live Mode**: If `ReflectorAgent` is offline or times out, `run_batch.py` falls back gracefully to the static baseline prompt without failing the batch run.
     - **Replay Mode**: In record/replay mode (`ReplayManager`), a lookup cache miss raises an explicit error rather than silently falling back, maintaining strict experimental reproducibility.

2. **Execution Lifecycle & Loop Memory**:
   - **Initial Attempt (Attempt 1)**: Before starting a seed debate, `run_batch.py` queries `ReflectorAgent` for the current Pareto-optimal prompt variant corresponding to that seed's CVE taxonomy bucket. If no history exists for that taxonomy bucket, `ReflectorAgent` returns the default baseline system prompt.
   - **Refinement Attempt (Attempt 2+)**: If Attempt 1 fails verifier/judge audit, `run_batch.py` posts the failure trace $\mu_f$ to `ReflectorAgent`. The reflector diagnoses the defect (e.g. `anchors_too_few_after_normalization`), mutates the prompt directive for Attempt 2, and updates its Pareto ledger.

3. **Concurrency-Safe Memory Ledger**:
   - The Pareto state and mutation history reside strictly within `artifacts/gepa/`.
   - **File Locking for Read-Modify-Write Operations**: The complete read-modify-write cycle on `artifacts/gepa/pareto_frontier.json` and the coordinated append to `artifacts/gepa/mutations.jsonl` are protected inside a file lock (`fcntl.flock`) to guarantee that concurrent seed tasks in `run_batch.py` do not overwrite mutations.
   - **Atomic Final Publication**: Atomic file replacement (`write_to_temp_and_rename`) is used strictly as the final publication step once the lock is acquired and state is updated.
   - **Append-Only Attempt History**: Historical mutation logs are maintained as append-only JSONL files (`artifacts/gepa/mutations.jsonl`).

---

## 5. Implementation Steps & Roadmap

1. **Step 1: Metric & Trace Extractor**: Update `run_batch.py` and `scenarios/debate/run_batch.py` to extract full $\mu_f$ execution traces on failed attempts.
2. **Step 2: Proposed GEPA Reflection Operator (`reflector_agent.py`)**: Implement the housed A2A Reflector Agent on Port `8004` (managed in `start_stack.sh`) using FastAPI, Pydantic contract validation, `ReplayManager.acompletion(...)` for model calls, and `call_structured()` for JSON parsing.
3. **Step 3: Concurrency-Safe Pareto Registry**: Implement local Pareto prompt pool persistence in `artifacts/gepa/pareto_frontier.json` per CVE taxonomy bucket protected by `fcntl.flock` read-modify-write locks and append-only `mutations.jsonl` history.
4. **Step 4: Benchmarking & Telemetry**: Measure Token Efficiency Ratio and anchor completeness rates across standard retries vs. GEPA-reflected retries using the formal token accounting definition over $20-100$ failure traces.
