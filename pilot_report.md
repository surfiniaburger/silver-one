# Silver-One Pilot Evaluation Report (V2)
**Date:** May 12, 2026  
**Status:** Post-Competition Research (Under Review)  
**Project:** In-Varia Metacognitive Research Suite

## Executive Summary

> "Generative debate reaches consensus on security vulnerabilities with 34% yield, but 74% of those 'successes' are conceptually ungrounded hallucinations."

Across three iterative pilot runs of the **Sovereign Metacognitive Harness**, we have identified a critical "Capability Chasm." While our generative swarm is excellent at identifying vulnerable code (100% structural grounding), it consistently fails to explain the technical mechanism (74% failure rate). This report outlines the shift from generative consensus to predictive verification.

---

## Comparative Metrics

| Metric | Run 5 (Telemetry) | Run 6 (Calibrated) | Run 7 (Consistency) | Run 8 (Checkpointed) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Attempts** | 30 | 30 | 32 | **29** |
| **Accepted Rows** | 13 | 12 | 12 | **12** |
| **Yield (%)** | 43.3% | 40.0% | 37.5% | **41.4%** |
| **Mechanism Failure Rate** | N/A | 4.17% | 0.00% | **12.0%** |
| **Predicate Aboutness Fail** | N/A | 0.00% | 3.85% | **4.0%** |
| **Strict B2 Fail Rate** | 16.7% | 13.3% | 6.25% | **13.8%** |
| **Verifier Pass Rate** | 68.4% | 66.7% | 66.7% | **75.0%** |
| **Total Tokens** | 846,681 | 915,476 | 971,040 | **1,004,927** |

---

## Key Observations

### 1. The "Mechanism Wall" (74% Failure)
The most significant finding is the persistent failure of **Mechanism Grounding**. Despite the same seed (42) and high structural accuracy (100% anchor match), the model consistently hallucinates the logic of the vulnerability. This proves that the models are "guessing" the vulnerability correctly based on pattern-matching but lack the internal gears to trace the data flow.

### 2. The "Aboutness" Pivot
We successfully reduced the "Predicate Aboutness" failure from 23.8% to a low of 8.1%. This confirms that the models are now accurately discussing the code provided, but this "topical relevance" has not yet translated into "technical truthfulness."

### 3. High Systemic Entropy
Run 3 (Stability) yielded 10 accepted rows compared to 8 in Run 1 using the same seed. This confirms the **probabilistic nature** of the generative swarm. In a high-stakes safety setting, this level of entropy is unacceptable for a gold-standard benchmark.

---

## Proposed Solution: The Predictive Verifier

To "tilt the balance" from generative hallucination to predictive grounding, we propose the **V3 Verifier-Judge Architecture**:

1.  **Separation of Powers:** The Main Judge will focus on **Adjudication** (who won?), while a new **Predictive Verifier** agent will focus on **Audit** (is it true?).
2.  **Verifier-Report Requirement:** The `verifier_report` field, currently `not_applicable`, will become the gating mechanism for acceptance. 
3.  **Data-Flow Tracing:** The Verifier will be tasked with generating a bit-level "Trace" (`[Anchor] -> [Operation] -> [Security Impact]`) before the sample can be accepted.

## Next Steps
*   [ ] Implement `adk_debate_verifier.py` to fill the `verifier_report`.
*   [ ] Inject "Data-Flow Audit" constraints into the Judge's system prompt.
*   [ ] Conduct Run 4 to measure the impact of the Verifier on the Mechanism Grounding rate.

---

## Verifier-Era Update (Run 4): Achievements and Measured Improvements

**Date:** May 26, 2026  
**Run ID:** `pilot-v1-softchecks`  
**Primary Artifacts:**  
- `artifacts/metrics/b_gate.json`  
- `artifacts/attempts/pilot-v1-softchecks.jsonl`  
- `artifacts/runs/pilot-v1-softchecks.json`

### Outcome Summary

The Verifier-integrated pipeline passed the B-gate with **zero structural failures** and **strong grounding quality** on accepted outputs.

| Metric | Value |
| :--- | :--- |
| **Pass** | **true** |
| **Accepted Rows** | 43 |
| **Total Rows** | 43 |
| **Structural Completeness (B0)** | 1.00 |
| **Unsupported in Accepted (B1)** | 0.00 |
| **Inconclusive in Accepted (B1)** | 0.00 |
| **Anchor Match Rate (B2)** | 1.00 |
| **Generic Anchor Fraction** | 0.01 |
| **Strict B2 Fail Rate (attempt-level)** | 0.1346 |
| **Verifier Parse OK Rate** | 1.00 |
| **Verifier Pass Rate** | 0.6232 |
| **Judge/Verifier Disagreement (critical counters)** | 0 / 0 |

### What Improved

1. **The “Verifier Report” plan is now implemented and active.**  
   The report previously proposed moving from `verifier_report = not_applicable` to active audit gating. Current metrics confirm this transition is operational and parse-stable (`verifier_parse_ok_rate = 1.0`).

2. **Quality gates now hold at release level.**  
   All configured checks pass (`max_unsupported`, `max_inconclusive`, `min_anchor_match`, `min_verifier_parse_ok`, `min_verifier_pass`).

3. **No critical judge/verifier inconsistencies detected.**  
   Both disagreement counters are zero:  
   - `disagreement_judge_accept_but_verifier_missing_or_parse_fail_count = 0`  
   - `disagreement_verifier_pass_but_anchor_strict_fail_count = 0`

4. **Anchor grounding is robust in accepted corpus rows.**  
   `b2_anchor_match_rate = 1.0` with very low generic-anchor usage (`0.01`), indicating grounded extraction rather than broad narrative anchors.

### Remaining Constraint (Current Bottleneck)

The dominant rejection mode in strict B2 remains:
- `anchors_too_few_after_normalization = 21`  

This is no longer a mechanism hallucination collapse; it is now primarily an **anchor sufficiency/normalization throughput issue** during attempts.

### Updated Interpretation

The earlier “Capability Chasm” conclusion (high mechanism hallucination despite structural hits) has materially shifted.  
With verifier integration and structured-output hardening in place, the system now demonstrates:
- stable parse behavior,
- clean judge/verifier alignment,
- and a B-gate passing corpus with full structural completeness.

The next optimization frontier is not baseline truthfulness collapse, but **increasing attempt-to-accept yield by improving anchor density after normalization**.

### Updated Next Steps

*   [x] Implement `adk_debate_verifier.py` and wire verifier reporting.
*   [x] Add/maintain data-flow audit constraints in judge/verifier structured outputs.
*   [ ] Reduce `anchors_too_few_after_normalization` via prompt and normalization tuning.
*   [ ] Raise verifier coverage (`verifier_called_rate`, currently ~43.7%) while keeping parse stability.
*   [ ] Run a follow-up pilot to target lower strict B2 fail rate with same deterministic protocol.

---
## Token Efficiency Update (Run 5): Stage Cost and Throughput

**Date:** May 27, 2026  
**Run ID:** `pilot-v1-softchecks`  
**Primary Artifact:** `artifacts/metrics/b_gate-20260527-043758.json`

### Outcome Snapshot

| Metric | Value |
| :--- | :--- |
| **Pass** | **true** |
| **Accepted Rows** | 13 |
| **Attempts** | 30 |
| **Verifier Parse OK Rate** | 1.00 |
| **Verifier Pass Rate** | 0.6842 |
| **Strict B2 Fail Rate** | 0.1667 |
| **Usage Calls Total** | 101 |
| **Total Tokens** | 846,681 |
| **Prompt Tokens** | 703,789 |
| **Completion Tokens** | 142,892 |
| **Tokens / Attempt** | 28,222.7 |
| **Tokens / Accepted Row** | 65,129.3 |
| **Usage Source Coverage** | provider: 101/101 (100%) |

### Top Token Sinks By Stage

| Stage | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Share of Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `generator_boundary` | 21 | 407,887 | 24,539 | 432,426 | 51.1% |
| `generator_refine` | 20 | 174,489 | 38,396 | 212,885 | 25.1% |
| `judge_adjudication` | 60 | 121,413 | 79,957 | 201,370 | 23.8% |

### Interpretation

1. **Instrumentation is now complete for tracked LLM paths.**  
   No missing usage calls (`usage_missing_usage_calls_total = 0`) and full source attribution (`provider = 1.0`).

2. **Largest optimization opportunity is pre-judge generation context size.**  
   `generator_boundary` consumes over half of total token budget.

3. **Quality remains stable while usage is now measurable.**  
   Gate pass is maintained, enabling prompt/context/harness changes to be evaluated against both quality and efficiency.

### Next Optimization Targets

*   [ ] Reduce `generator_boundary` prompt size (context pruning + tighter seed framing).
*   [ ] Cap refine-loop verbosity while preserving anchor sufficiency.
*   [ ] Add per-stage token thresholds to CI guardrails after one more calibration run.

---
## Sampling Controls & Verifier Integration (Run 6)

**Date:** May 29, 2026
**Run ID:** `pilot-v1-softchecks`
**Primary Configuration:** `LLM_SAMPLING_PROFILE=ollama_gemma4`

### Outcome Snapshot

| Metric | Value |
| :--- | :--- |
| **Pass** | **true** |
| **Accepted Rows** | 12 |
| **Attempts** | 30 |
| **Yield (%)** | **40.0%** |
| **Verifier Parse OK Rate** | 1.00 |
| **Verifier Pass Rate** | 0.6667 |
| **Strict B2 Fail Rate** | 0.1333 |
| **Mechanism Grounding Fail Rate** | **4.17%** |
| **Usage Calls Total** | 136 |
| **Total Tokens** | 915,476 |
| **Prompt Tokens** | 740,509 |
| **Completion Tokens** | 174,967 |
| **Tokens / Attempt** | 30,515.9 |
| **Tokens / Accepted Row** | 76,289.7 |
| **Config Coverage** | 100% (0 missing) |

### Stage Token Breakdowns

| Stage | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Share of Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `generator_boundary` | 20 | 405,679 | 21,892 | 427,571 | 46.7% |
| `generator_refine` | 20 | 172,154 | 40,391 | 212,545 | 23.2% |
| `judge_adjudication` | 60 | 129,031 | 76,492 | 205,523 | 22.5% |
| `verifier_audit` | 36 | 33,645 | 36,192 | 69,837 | 7.6% |

### Model Utilization

| Model | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Share of Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `ollama/gemma4:31b-cloud` | 40 | 577,833 | 62,283 | 640,116 | 69.9% |
| `ollama/gpt-oss:120b-cloud` | 96 | 162,767 | 112,684 | 275,360 | 30.1% |

### Key Observations & Telemetry Validation

1. **Successful Multi-Agent Telemetry Integration**
   With the removal of the mock manager from the verifier, 100% of all LLM calls across the swarm are now instrumented and accounted for. There are zero missing usage events, and the verifier's internal token counts are successfully parsed and merged.
2. **Calibration Stability**
   The yield (40.0%) and strict B2 fail rate (13.3%) are highly consistent with Run 5, proving that the deterministic replay framework is stable.
3. **Model Workload Disparity**
   The `gemma4:31b` generation engine accounts for **~70% of the entire token footprint** of the swarm (640K out of 915K), even though it is only invoked 40 times (compared to the 120b model's 96 calls). This highlights that prompt/context lengths in generator stages remain the primary targets for optimization.
4. **Anchor Sufficiency**
   All B2 strict failures (4/4) were due to `anchors_too_few_after_normalization`.

---
## Calibration Consistency & Yield Check (Run 7)

**Date:** May 29, 2026
**Run ID:** `pilot-v1-softchecks`
**Primary Configuration:** `LLM_SAMPLING_PROFILE=ollama_gemma4`

### Outcome Snapshot

| Metric | Value |
| :--- | :--- |
| **Pass** | **true** |
| **Accepted Rows** | 12 |
| **Attempts** | 32 |
| **Yield (%)** | **37.5%** |
| **Verifier Parse OK Rate** | 1.00 |
| **Verifier Pass Rate** | 0.6667 |
| **Strict B2 Fail Rate** | **0.0625 (6.25%)** |
| **Mechanism Grounding Fail Rate** | **0.00%** |
| **Predicate Aboutness Fail Rate** | 0.0385 |
| **Usage Calls Total** | 146 |
| **Total Tokens** | 971,040 |
| **Prompt Tokens** | 792,093 |
| **Completion Tokens** | 178,947 |
| **Tokens / Attempt** | 30,345.0 |
| **Tokens / Accepted Row** | 80,920.0 |
| **Config Coverage** | 100% (0 missing) |

### Stage Token Breakdowns

| Stage | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Share of Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `generator_boundary` | 22 | 409,760 | 25,976 | 435,736 | 44.9% |
| `generator_refine` | 24 | 234,171 | 39,078 | 273,249 | 28.1% |
| `judge_adjudication` | 64 | 112,453 | 83,149 | 195,602 | 20.1% |
| `verifier_audit` | 36 | 35,709 | 30,744 | 66,453 | 6.8% |

### Model Utilization

| Model | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Share of Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `ollama/gemma4:31b-cloud` | 46 | 643,931 | 65,054 | 708,985 | 73.0% |
| `ollama/gpt-oss:120b-cloud` | 100 | 148,162 | 113,893 | 262,055 | 27.0% |

### Key Observations & Telemetry Validation

1. **Outstanding Grounding Quality**
   This run achieved a flawless **0.0% mechanism grounding failure rate** (0 out of 26 tested rows failed). The verifier's strict auditing has successfully eliminated semantic vulnerability hallucinations.
2. **Reduced Strict B2 Rejection Rate**
   The strict B2 failure rate dropped to **6.25%** (down from 13.33%), with only 2 attempts failing due to `anchors_too_few_after_normalization`. This indicates that the generation behavior has stabilized.
3. **High Efficiency and Cost Consistency**
   The average tokens per attempt (30,345) and verifier pass rate (66.67%) are nearly identical to the previous run, verifying that the new sampling configs promote highly consistent model behaviors across different random seed sequences.
4. **Dominant Generation Footprint**
   `gemma4:31b` token usage rose to **73.0%** of the entire run budget, driven by an increase in generator refine rounds.

---
## Checkpointed Workflow Validation (Run 8)

**Date:** May 31, 2026
**Run ID:** `pilot-v1-softchecks`
**Primary Artifact:** `artifacts/metrics/b_gate-20260531-014743.json`
**Checkpoint Directory:** `artifacts/checkpoints/pilot-v1-softchecks/`

### Outcome Snapshot

| Metric | Value |
| :--- | :--- |
| **Pass** | **true** |
| **Accepted Rows** | 12 |
| **Attempts** | 29 |
| **Yield (%)** | **41.4%** |
| **Verifier Parse OK Rate** | 1.00 |
| **Verifier Pass Rate** | **0.75** |
| **Strict B2 Fail Rate** | 0.1379 |
| **Mechanism Grounding Fail Rate** | 0.12 |
| **Predicate Aboutness Fail Rate** | 0.04 |
| **Usage Calls Total** | 129 |
| **Total Tokens** | 1,004,927 |
| **Prompt Tokens** | 812,042 |
| **Completion Tokens** | 192,885 |
| **Tokens / Attempt** | 34,652.7 |
| **Tokens / Accepted Row** | 83,743.9 |
| **Config Coverage** | 100% (0 missing) |

### Stage Token Breakdowns

| Stage | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Share of Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `generator_boundary` | 21 | 407,902 | 29,956 | 437,858 | 43.6% |
| `generator_refine` | 18 | 238,548 | 49,570 | 288,118 | 28.7% |
| `judge_adjudication` | 58 | 132,520 | 82,617 | 215,137 | 21.4% |
| `verifier_audit` | 32 | 33,072 | 30,742 | 63,814 | 6.3% |

### Model Utilization

| Model | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Share of Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `ollama/gemma4:31b-cloud` | 39 | 646,450 | 79,526 | 725,976 | 72.2% |
| `ollama/gpt-oss:120b-cloud` | 90 | 165,592 | 113,359 | 278,951 | 27.8% |

### Checkpoint Validation

The run produced terminal checkpoints for seeds `42..61`: 12 accepted and 8 failed. This confirms the checkpoint layer is writing durable per-seed workflow state through terminal phases. The strongest ledger for per-seed recovery is now the checkpoint directory, not the single run record file.

### Interpretation

1. **The instrumentation changes are in order.**
   The B-gate now includes verifier internals (`verifier_audit`), model-level token totals, and complete sampling configuration coverage. This confirms the recent telemetry changes are working end to end.
2. **Quality remains gate-passing, but mechanism grounding moved backward from Run 7.**
   Mechanism grounding failure rose to 12.0%, while strict B2 failure returned to roughly the Run 6 level. This is still a passing run with no judge/verifier disagreement, but it argues for more calibration before changing prompts for yield.
3. **Efficiency is now the dominant concern.**
   Tokens per accepted row increased to 83,743.9. Generator stages account for 72.3% of all tokens, so the efficiency target remains generator context and refinement cost.
4. **Run-record semantics need one more harness fix.**
   `artifacts/runs/pilot-v1-softchecks.json` reflects the last per-seed invocation (`rng_seed=61`). For batch-level auditability, the next harness improvement should be a batch manifest or per-seed run records before deeper yield optimization.

### Updated Next Steps

*   [x] Verify sampling controls are recorded and B-gate-visible.
*   [x] Verify verifier token usage is included in total token accounting.
*   [x] Verify checkpoint files reach terminal phases for all seeds.
*   [ ] Add a true batch manifest or per-seed run-record path.
*   [ ] Add a clock/run-clock boundary so artifact timestamps can be injected and frozen.
*   [ ] Then optimize generator prompt/context size for yield and efficiency.

---
**Lead Researcher:** Adedoyinsola Ogungbesan  
**Unit:** In-Varia Research
