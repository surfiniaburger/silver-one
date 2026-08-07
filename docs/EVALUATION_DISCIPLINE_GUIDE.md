# Evaluation Discipline & Anti-Gaming Guardrails Guide

> "Automation is merely a multiplier of your existing discipline. Focus on the fundamentals first: keep your steps small, validate continuously, and keep your software always releasable." — Dave Farley (*Modern Software Engineering*)

This document defines the **evaluation discipline, anti-gaming guardrails, and statistical decision standards** enforced across the **BARRED Multi-Agent Vulnerability Swarm** (`silver-one`).

---

## 1. Context & Motivation

As AI agent systems move toward self-improvement (e.g. GEPA prompt adaptation, pre-filter active learning), they incur a critical risk: **self-bias and benchmark gaming**. 

If an agent rates its own generated code highly or mutates prompts to bypass quality checks, shallow un-grounded claims can contaminate the synthetic training dataset. To prevent this, `silver-one` enforces **4 Anti-Gaming Invariants** and strict statistical hypothesis testing before any candidate prompt strategy or model version is accepted.

---

## 2. The 4 Anti-Gaming Quality Invariants

Every batch execution output and candidate prompt strategy is evaluated by `scenarios/debate/offline_b_gate.py` against 4 non-negotiable invariants:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    The 4 Anti-Gaming Invariants                         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                             ▼                             ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
│ 1. Zero Logic Error Rate  │ │ 2. Strict Anchor Grounding│ │ 3. Verifier Parse Success │
│ (accepted_logic_error == 0│ │ (b2_anchor_match >= 0.80) │ │ (verifier_parse_ok >= 0.95│
└───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
                                     │
                                     ▼
                      ┌───────────────────────────┐
                      │ 4. Leak-Proof Partitioning│
                      │ (Scenario-Grouped Folds)  │
                      └───────────────────────────┘
```

### Invariant 1: Zero Logic Errors in Accepted Corpus (`accepted_logic_error_rate == 0.0`)
- **Requirement**: No row containing logic errors (`verifier_logic_error == True`) may enter the accepted corpus.
- **Empirical Baseline**: `0.0000` across all 14 historical benchmark runs.

### Invariant 2: Strict Anchor Grounding (`b2_anchor_match_rate >= 0.80`)
- **Requirement**: Code line anchors extracted by the Pro debater must strictly match actual source code lines in the generated candidate.
- **Threshold**: Minimum $80.0\%$ match rate (`b2_anchor_match_rate >= 0.80`).
- **Empirical Baseline**: `1.0000` ($100\%$) across all passing benchmark runs.

### Invariant 3: Verifier Parse Success (`verifier_parse_ok_rate >= 0.95`)
- **Requirement**: Verifier audit responses must parse cleanly into structured JSON schemas.
- **Threshold**: Minimum $95.0\%$ parse success rate (`verifier_parse_ok_rate >= 0.95`).
- **Empirical Baseline**: 13 out of 14 historical benchmark runs each achieved a per-run parse OK rate of `1.0000` ($100\%$). Run `a` achieved `0.9375` ($93.75\%$) due to LLM output formatting truncation during initial pilot testing.

### Invariant 4: Leak-Proof Scenario-Grouped Partitioning
- **Requirement**: Candidate pre-filter models ($v1.0 \rightarrow v2.0$) must be evaluated using `partition_dataset_by_scenario_stratified` as adopted in [RFC_PRE_FILTER_STRATIFIED_CV.md](RFC_PRE_FILTER_STRATIFIED_CV.md).
- **Threshold**: Scenario IDs use `cve_id` when available and SHA-256 predicate text hashing (`HASH-{sha256[:10]}`) otherwise, guaranteeing **zero scenario-predicate leakage** across train and test folds.

---

## 3. BARRED Swarm Project Evaluation Standard (Silver-One Protocol)

Reporting point estimates (Mean $\pm$ StdDev) across small runs ($N = 3$) quantifies **empirical run-to-run variation** (LLM sampling noise).

To declare **true statistical significance** when comparing candidate configurations against baselines ($N \ge 5$ runs per condition):

1. **Paired Seed Comparison**: Baseline and candidate runs are paired by PRNG seed (`--seed 42, 1337, 2026`).
2. **Statistical Test Selection**:
   - **Welch's Two-Sample t-Test**: Selected when token performance data passes Shapiro-Wilk normality tests ($p > 0.05$).
   - **Mann-Whitney U Rank-Sum Test**: Selected when token distributions exhibit non-normal skewness.
3. **95% Confidence Intervals & Decision Rule**:
   - Calculated as $\text{Mean Delta} \pm 1.96 \times \text{SE}$.
   - A candidate strategy is declared **statistically significant** if and only if:
     - Two-tailed $p$-value satisfies $p < 0.05$ ($\alpha = 0.05$).
     - Bonferroni correction is applied when evaluating across $k$ metric dimensions ($\alpha_{\text{adjusted}} = \alpha / k$).
     - The 95% Confidence Interval for token reduction strictly excludes zero ($0.0$).

---

## 4. Modern Software Engineering Principles (Dave Farley)

All code modifications in `silver-one` strictly adhere to Continuous Delivery fundamentals:

- **Small Incremental Steps**: Break work down into small, independent PRs.
- **Continuous Validation**: Automated CI pipeline (`.github/workflows/farley_ci.yml`) executes both `pytest tests` and `pytest scenarios/debate/test_pre_filter.py` on every pull request.
- **Always Releasable**: Main branch must remain 100% green and releasable at all times.
