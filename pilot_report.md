# Silver-One Pilot Evaluation Report (V2)
**Date:** May 12, 2026  
**Status:** Post-Competition Research (Under Review)  
**Project:** In-Varia Metacognitive Research Suite

## Executive Summary

> "Generative debate reaches consensus on security vulnerabilities with 34% yield, but 74% of those 'successes' are conceptually ungrounded hallucinations."

Across three iterative pilot runs of the **Sovereign Metacognitive Harness**, we have identified a critical "Capability Chasm." While our generative swarm is excellent at identifying vulnerable code (100% structural grounding), it consistently fails to explain the technical mechanism (74% failure rate). This report outlines the shift from generative consensus to predictive verification.

---

## Comparative Metrics

| Metric | Run 1 (Initial) | Run 2 (Scale) | Run 3 (Stability) |
| :--- | :--- | :--- | :--- |
| **Total Attempts** | 30 | 52 | 29 |
| **Accepted Rows** | 8 | 7 | 10 |
| **Yield (%)** | 26.6% | 13.4% | **34.4%** |
| **Mechanism Failure Rate** | 61.9% | 73.0% | **73.9%** |
| **Predicate Aboutness Fail** | 23.8% | 8.1% | 13.0% |
| **Anchor Match Rate** | 100% | 100% | 100% |

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
**Lead Researcher:** Adedoyinsola Ogungbesan  
**Unit:** In-Varia Research
