# Implementation Plan: Layered High-Precision Pre-Filter (3-Stage Cascade)

This document outlines the design, architecture, and step-by-step PR breakdown for integrating a **3-Stage Layered Acceptance Pre-Filter** into BARRED's debate pipeline.

---

## 1. Executive Summary & Goal

- **Goal**: Reduce token spend per accepted row by **50%** (bringing spend down from ~80k–100k tokens/row to ~35k–50k tokens/row).
- **Core Strategy**: Intercept unviable candidate seeds and doomed refinement loops **before** making expensive LLM API calls.
- **Latency Budget**: $< 10\text{ ms}$ on CPU per seed evaluation ($>80\%$ of negatives rejected in $<1.5\text{ ms}$).

---

## 2. Cascade Architecture & Decision Thresholds

```text
[ Seed Candidate (Predicate + Code) ]
                 │
                 ▼
 ┌───────────────────────────────┐
 │ Stage A: Heuristic Rules      │ ──► Confident? (prob = 0.01 / 0.99) ──► [ Return Decision (<0.1ms) ]
 └───────────────┬───────────────┘
                 │ Uncertain (Rule None)
                 ▼
 ┌───────────────────────────────┐
 │ Stage B: TF-IDF + XGBoost     │ ──► Confident? (prob ≥ 0.995 or ≤ 0.05) ──► [ Return Decision (~1ms) ]
 └───────────────┬───────────────┘
                 │ Ambiguous (0.05 < prob < 0.995)
                 ▼
 ┌───────────────────────────────┐
 │ Stage C: SetFit (Transformer) │ ──► Final Decision (prob ≥ 0.65 Accept, < 0.65 Reject) (~10ms)
 └───────────────────────────────┘
```

- **Stage A (Heuristics, $<0.1\text{ms}$)**: Evaluates regex/syntax patterns. Returns `prob = 0.01` (Reject) for empty/short/malformed predicates or `prob = 0.99` (Accept) for known strong patterns. Abstains (`None`) when ML evaluation is required.
- **Stage B (XGBoost + TF-IDF, $\sim 1\text{ms}$)**: Character n-gram features + histogram gradient boosting. If `prob >= 0.995`, returns `accept=True`. If `prob <= 0.05`, returns `accept=False`.
- **Stage C (SetFit Transformer, $\sim 10\text{ms}$)**: Invoked only when Stage B returns an ambiguous probability ($0.05 < \text{prob} < 0.995$). Returns `accept=True` if `prob >= 0.65`, otherwise `accept=False`.

---

## 3. Pull Request Breakdown

### PR 1: Core Pre-Filter Cascade Module & Unit Tests
- **Objective**: Implement the `BarredPreFilter` evaluation cascade and comprehensive unit test harness.
- **Target Files**:
  - `[NEW]` [`scenarios/debate/pre_filter.py`](../scenarios/debate/pre_filter.py)
  - `[NEW]` [`scenarios/debate/test_pre_filter.py`](../scenarios/debate/test_pre_filter.py)
- **Key Features**:
  - `PreFilterDecision` frozen dataclass (`accept: bool`, `probability: float`, `stage: str`, `elapsed_ms: float`).
  - Stage A: Regex/syntax heuristics for empty/short/ungrounded predicates.
  - Stage B: `TfidfVectorizer` + `XGBClassifier` loader fallback.
  - Stage C: `SetFitModel` loader fallback.
  - Graceful passthrough (`default_pass`) with explicit warning logging when model weights are not yet generated.

---

### PR 2: Offline Dataset Extraction & Model Training Pipeline
- **Objective**: Build the offline dataset generator and trainer script using historical attempt logs.
- **Target Files**:
  - `[NEW]` [`scripts/train_pre_filter.py`](../scripts/train_pre_filter.py)
  - `[NEW]` [`tests/test_train_pre_filter.py`](../tests/test_train_pre_filter.py)
- **Key Features**:
  - Parses `artifacts/attempts/*.jsonl` attempt logs.
  - Extracts text features: `Predicate: <pred> | Code: <code>`.
  - Maps binary label: `1` for `decision == 'accepted'`, `0` for `decision == 'rejected'`.
  - Trains Stage B (`XGBClassifier` with `tree_method="hist"`) and Stage C (`SetFitModel`).
  - Exports persisted model artifacts to `artifacts/models/`.

---

### PR 3: Integration into Batch Runner & Judge Engine
- **Objective**: Intercept bad seeds at Step 0 in `run_batch.py` and block runaway refinement loops in `adk_debate_judge.py`.
- **Target Files**:
  - `[MODIFY]` [`scenarios/debate/run_batch.py`](../scenarios/debate/run_batch.py)
  - `[MODIFY]` [`scenarios/debate/adk_debate_judge.py`](../scenarios/debate/adk_debate_judge.py)
  - `[MODIFY]` [`scenarios/debate/test_adk_debate_judge_refactor.py`](../scenarios/debate/test_adk_debate_judge_refactor.py)
- **Key Features**:
  - `run_batch.py`: Add `--pre-filter / --no-pre-filter` CLI argument. Intercept seeds in `_process_seed()` before sending HTTP JSON-RPC payloads to the green agent server. Record `"skipped_pre_filter"` status in manifest when rejected and log a skipped attempt record (`decision: "skipped_pre_filter"`) to `artifacts/attempts/*.jsonl`.
  - `adk_debate_judge.py`: Check `pre_filter.predict()` in `_run_refinement_iteration()` before generating refinement samples ($i \ge 1$).

---

### PR 4: Benchmarking, Telemetry & Documentation
- **Objective**: Verify token reduction against the 50% target and document benchmark results.
- **Target Files**:
  - `[NEW]` [`reports/pre_filter_benchmark.md`](../reports/pre_filter_benchmark.md)
  - `[MODIFY]` [`README.md`](../README.md)
  - `[MODIFY]` [`docs/EVALUATOR_LATENCY_AND_CACHE_BENCHMARKS.md`](EVALUATOR_LATENCY_AND_CACHE_BENCHMARKS.md)
- **Key Features**:
  - Execute a benchmark sweep (`pilot-v1-calibrated-p`) with pre-filtering enabled.
  - Compare token spend per accepted row against baseline runs (`calibrated-i` and `calibrated-l`).
  - Document P50/P95 latency breakdown of the 3-stage pre-filter cascade.

---

## 4. Verification Plan

### Automated Tests
- Run `uv run python -m pytest` across all unit and integration test modules.
- Verify $<10\text{ms}$ latency threshold for 100 sample pre-filter predictions.

### Manual Verification
- Execute `run_batch.py` on `scenarios/debate/cve_seeds_test.jsonl` with `--pre-filter`.
- Verify in `artifacts/attempts/*.jsonl` and batch manifests that bad seeds are recorded as `"skipped_pre_filter"` without consuming LLM API tokens.
