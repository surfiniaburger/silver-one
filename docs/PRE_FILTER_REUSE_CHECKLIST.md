# Pre-Filter Cascade Implementation & Code Reuse Checklist

This document tracks all existing utilities to be reused, new modules to build, files to modify, and verification steps for the **3-Stage Layered Acceptance Pre-Filter** implementation.

---

## 1. Existing Utilities to Reuse

### From `src/agentbeats/`
- [x] **[`src/agentbeats/tracing.py`](../src/agentbeats/tracing.py)**:
  - Reuse `trace_span()` context manager to track pre-filter latency, decision stage (`heuristic` | `xgboost` | `setfit`), and automatically export telemetry spans to `artifacts/metrics/spans.jsonl`.
- [ ] **[`src/agentbeats/clock.py`](../src/agentbeats/clock.py)** (Planned for PR 3):
  - Reuse `RunClock.from_env().now_iso()` in `run_batch.py` integration to guarantee deterministic ISO timestamps across batch execution and pre-filter logs during record/replay modes.
- [ ] **[`src/agentbeats/checkpoint.py`](../src/agentbeats/checkpoint.py)** (Planned for PR 3):
  - Reuse `save_checkpoint()` and `load_checkpoint()` in `run_batch.py` to persist batch manifests updated with `"skipped_pre_filter"` statuses.
- [ ] **[`src/agentbeats/replay.py`](../src/agentbeats/replay.py)** (Planned for PR 3):
  - Reuse `ReplayManager` to enforce deterministic pre-filter execution during offline replay tests (`mode=replay`).

### From `scripts/`
- [x] **[`scripts/debate_telemetry.py`](../scripts/debate_telemetry.py)**:
  - Reuse to aggregate `spans.jsonl` metrics, compute P50/P95 pre-filter latency distributions, and report token savings.
- [x] **[`scripts/run_b_gate.sh`](../scripts/run_b_gate.sh)**:
  - Reuse to evaluate quality gates (`b2_anchor_match_rate`, `verifier_pass_rate`, `efficiency_tokens_per_accepted_row`) on pre-filtered batch output files.
- [x] **[`scripts/telemetry_utils.py`](../scripts/telemetry_utils.py)**:
  - Reuse string hashing utilities to normalize code tokens before passing inputs to Stage B/C vectorizers.
- [x] **[`scripts/path_utils.py`](../scripts/path_utils.py)**:
  - Reuse for cross-platform model artifact path resolution (`artifacts/models/`).

---

## 2. New Modules to Build

- [x] **[`scenarios/debate/pre_filter.py`](../scenarios/debate/pre_filter.py)** (PR 1):
  - Implement `BarredPreFilter` and `PreFilterDecision` frozen dataclass with documented schema.
  - Implement Stage A (Heuristics), Stage B (XGBoost + TF-IDF), and Stage C (SetFit).
  - Implement `default_pass` fallback with explicit warning logging when model binaries are missing.
- [x] **[`scenarios/debate/test_pre_filter.py`](../scenarios/debate/test_pre_filter.py)** (PR 1):
  - Unit test heuristic rules, model loader fallbacks, and $<10\text{ms}$ CPU wall-clock latency budgets.
- [ ] **[`scripts/train_pre_filter.py`](../scripts/train_pre_filter.py)** (PR 2):
  - Dataset extractor and offline model trainer script.
  - Parses `artifacts/attempts/*.jsonl` logs, fits Stage B (`XGBClassifier`) and Stage C (`SetFitModel`), and saves weights to `artifacts/models/`.
- [ ] **[`tests/test_train_pre_filter.py`](../tests/test_train_pre_filter.py)** (PR 2):
  - Automated tests for dataset extraction and model weight persistence.
- [ ] **[`reports/pre_filter_benchmark.md`](../reports/pre_filter_benchmark.md)** (PR 4):
  - Markdown report documenting pre-filter benchmark sweep results and token reduction percentage.

---

## 3. Core Files to Modify

- [ ] **[`scenarios/debate/run_batch.py`](../scenarios/debate/run_batch.py)** (PR 3):
  - Add `--pre-filter / --no-pre-filter` CLI flag.
  - Intercept seeds in `_process_seed()` before HTTP JSON-RPC calls.
  - Log status `"skipped_pre_filter"` in manifest and write a skipped attempt entry (`decision: "skipped_pre_filter"`) to `artifacts/attempts/*.jsonl`.
- [ ] **[`scenarios/debate/adk_debate_judge.py`](../scenarios/debate/adk_debate_judge.py)** (PR 3):
  - Check `pre_filter.predict()` in `_run_refinement_iteration()` before generating refinement samples ($i \ge 1$) to terminate unfixable attempts early.
- [ ] **[`scenarios/debate/test_adk_debate_judge_refactor.py`](../scenarios/debate/test_adk_debate_judge_refactor.py)** (PR 3):
  - Add unit test coverage for judge refinement interception.
- [ ] **[`README.md`](../README.md)** (PR 4):
  - Document pre-filter usage and CLI flags.
- [ ] **[`docs/EVALUATOR_LATENCY_AND_CACHE_BENCHMARKS.md`](EVALUATOR_LATENCY_AND_CACHE_BENCHMARKS.md)** (PR 4):
  - Update benchmark documentation with pre-filter latency and token spend deltas.

---

## 4. Verification & Quality Assurance

- [ ] **Automated Test Suite**: Run `uv run python -m pytest` to confirm all 197+ unit/integration tests pass.
- [ ] **Latency Assertion**: Verify `<10ms` execution threshold for 100 sample predictions using `trace_span`.
- [ ] **Batch Pre-Filter Verification**: Run `run_batch.py` with `--pre-filter` on `scenarios/debate/cve_seeds_test.jsonl` and confirm unviable seeds are recorded as `"skipped_pre_filter"` with 0 LLM API tokens consumed.
- [ ] **B-Gate Quality Verification**: Run `./scripts/run_b_gate.sh` to confirm B-gate quality checks pass cleanly.
