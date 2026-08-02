# Reproducibility & Data/Model Versioning Guide

> "Changing world values are inputs. Inputs are recorded. Recorded inputs are replayable. Replayable workflows are evolvable."

This document provides a step-by-step guide for reproducing the **BARRED Multi-Agent Vulnerability Dataset Swarm** results, version-controlling machine learning models (`artifacts/models`), executing the 2x2 Evaluation Matrix, and performing rigorous statistical hypothesis testing according to NeurIPS / ICLR empirical standards.

---

## 1. Pillars of Swarm Reproducibility

Following MLOps and Data Version Control (DVC) best practices (Chawla, 2023), reproducing synthetic agent swarm datasets requires versioning three distinct pillars:

```text
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│     1. CODE VERSION     │  │   2. RUN CONFIGURATION  │  │   3. DATA & MODEL STATE │
│   (Git Branch & PRs)    │  │   (Seed, Clock, Profile)│  │   (DVC/Hashes/Cassettes)│
└────────────┬────────────┘  └────────────┬────────────┘  └────────────┬────────────┘
             │                            │                            │
             └────────────────────────────┼────────────────────────────┘
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Deterministic Replay & Audit   │
                         └─────────────────────────────────┘
```

1. **Code Versioning (Git)**:
   - Clean, tagged release commit on `main`.
   - Immutable agent implementation code (`adk_debate_judge.py`, `pre_filter.py`).

2. **Execution & Run Configuration**:
   - Fixed base random seeds (`--seed 42`, `1337`, `2026`).
   - Clock-frozen timestamps (`--clock-now 2026-08-01T04:00:00Z`).
   - Locked LLM sampling profile (`LLM_SAMPLING_PROFILE=ollama_gemma4`, `temperature=1.0`, `top_p=0.95`, `top_k=64`).

3. **Data & Model Versioning**:
   - Canonical seed manifests (`scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl`).
   - Cassette recordings (`artifacts/cassettes/<run-id>.json`).
   - Versioned pre-filter model binaries (`artifacts/models/v1.0_pre_filter/`).

---

## 2. Canonical Seed Manifests & SHA-256 Integrity

Seed datasets reside under version-controlled manifest paths:

- `scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl` (Canonical 10-seed pilot split)
- `scenarios/debate/manifests/cve_seeds_v1_fresh10.jsonl` (Canonical 10-seed fresh out-of-distribution split)
- `scenarios/debate/manifests/cve_seeds_v1_full50.jsonl` (Canonical 50-seed benchmark split)

### Verifying Seed File Hashes
Before executing a benchmark run, verify seed manifest integrity:

```bash
shasum -a 256 scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl
shasum -a 256 scenarios/debate/manifests/cve_seeds_v1_fresh10.jsonl
```

The SHA-256 digest of the exact bytes of the seed file is automatically computed and recorded in `artifacts/runs/<run-id>/batch_manifest.json` under the `seeds_sha256` key alongside `seeds_path`:

```json
{
  "schema_version": 1,
  "run_id": "pilot-v1-seed42",
  "seeds_path": "scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl",
  "seeds_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

---

## 3. The 2x2 Evaluation Matrix (Data Seeds vs PRNG Seeds)

To distinguish between **CVE Prompt Data** (the problem instances) and **PRNG Seeds** (LLM sampling controls), all evaluation runs map to the **2x2 Evaluation Matrix**:

```text
                            Fixed CVE Data Seeds            Fresh/New CVE Data Seeds
                       (cve_seeds_v1_pilot10.jsonl)         (cve_seeds_v1_fresh10.jsonl)
                    ┌────────────────────────────────┬────────────────────────────────┐
Same PRNG Seed      │ 1. Cassette Replay Verification│ 2. Single Out-of-Sample        │
(--seed 42)         │    (Zero-diff validation)      │    Baseline Run                │
                    ├────────────────────────────────┼────────────────────────────────┤
Different PRNG Seeds│ 3. Swarm Stability Benchmark   │ 4. Full Out-of-Distribution    │
(--seed 42,1337,2026│    (Measures LLM sampling)     │    Generalization Test         │
                    └────────────────────────────────┴────────────────────────────────┘
```

### Execution Procedures for All Four Matrix Cells

#### Cell 1: Cassette Replay Verification (Fixed Data + Fixed PRNG)
- **Goal**: Validate zero-network deterministic replay from cached cassettes.
- **Command**:
  ```bash
  uv run python scenarios/debate/run_batch.py \
    --run-id cell1-replay \
    --seed 42 --mode replay \
    --clock-now 2026-08-01T04:00:00Z \
    --seeds scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl \
    --output training_corpus_cell1.jsonl \
    --attempts-out artifacts/attempts/cell1-replay.jsonl
  ```
- **Expected Artifacts**: `training_corpus_cell1.jsonl`, `artifacts/attempts/cell1-replay.jsonl`.

#### Cell 2: Single Out-of-Sample Baseline (Fresh Data + Fixed PRNG)
- **Goal**: Evaluate single-run performance on unseen vulnerability predicates.
- **Command**:
  ```bash
  uv run python scenarios/debate/run_batch.py \
    --run-id cell2-fresh-base \
    --seed 42 --mode record \
    --clock-now 2026-08-01T04:00:00Z \
    --seeds scenarios/debate/manifests/cve_seeds_v1_fresh10.jsonl \
    --output training_corpus_cell2.jsonl \
    --attempts-out artifacts/attempts/cell2-fresh-base.jsonl
  ```
- **Expected Artifacts**: `training_corpus_cell2.jsonl`, `artifacts/attempts/cell2-fresh-base.jsonl`.

#### Cell 3: Swarm Stability Benchmark (Fixed Data + Different PRNG Seeds)
- **Goal**: Measure empirical run-to-run variation across multiple random initializations on the benchmark dataset.
- **Commands**:
  ```bash
  for s in 42 1337 2026; do
    uv run python scenarios/debate/run_batch.py \
      --run-id cell3-stability-seed$s \
      --seed $s --mode record \
      --clock-now 2026-08-01T04:00:00Z \
      --seeds scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl \
      --output training_corpus_cell3_seed$s.jsonl \
      --attempts-out artifacts/attempts/cell3-stability-seed$s.jsonl
  done
  ```
- **Expected Artifacts**: `training_corpus_cell3_seed*.jsonl`, `artifacts/attempts/cell3-stability-seed*.jsonl`.

#### Cell 4: Full Out-of-Distribution Generalization Test (Fresh Data + Different PRNG Seeds)
- **Goal**: Evaluate pre-filter generalization across fresh vulnerability classes under stochastic variation.
- **Commands**:
  ```bash
  for s in 42 1337 2026; do
    uv run python scenarios/debate/run_batch.py \
      --run-id cell4-ood-seed$s \
      --seed $s --mode record \
      --clock-now 2026-08-01T04:00:00Z \
      --seeds scenarios/debate/manifests/cve_seeds_v1_fresh10.jsonl \
      --output training_corpus_cell4_seed$s.jsonl \
      --attempts-out artifacts/attempts/cell4-ood-seed$s.jsonl
  done
  ```
- **Expected Artifacts**: `training_corpus_cell4_seed*.jsonl`, `artifacts/attempts/cell4-ood-seed*.jsonl`.

---

## 4. Empirical Stability & Run-to-Run Variation

When evaluating small sample initializations ($N = 3$ runs), reporting **Mean ± Standard Deviation** quantifies **empirical run-to-run variation** (LLM sampling noise).

### Computing Aggregate Telemetry

```bash
uv run python scripts/debate_telemetry.py \
  --aggregate-runs cell3-stability-seed42 cell3-stability-seed1337 cell3-stability-seed2026 \
  --output-markdown reports/stability_benchmark_report.md
```

This generates a stability report detailing **Mean ± StdDev** for:
- Token efficiency improvement (%)
- Accepted corpus rows
- B-gate pass rate
- Verifier anchor grounding rate

---

## 5. Formal Statistical Significance Standard (NeurIPS / ICLR Guidelines)

To establish **true statistical significance** (beyond run-to-run variation), LLM agent swarm benchmarks must comply with NeurIPS / ICLR statistical evaluation guidelines:

### A. Two-Sample Welch's t-Test & Non-Parametric Mann-Whitney U Test
When comparing a candidate pre-filter configuration against an un-adapted baseline across $N \ge 5$ runs per condition:
- **Welch's Two-Sample t-Test**: Used for continuous token usage metrics where variances between baseline and candidate runs may differ ($s_1^2 \ne s_2^2$).
- **Mann-Whitney U Test**: Non-parametric rank-sum test applied to non-normal LLM token spend distributions.

### B. 95% Confidence Intervals & Significance Decision Rule
- **95% Confidence Interval (CI)**: Reported alongside mean deltas:
  $$\text{Mean Delta} \pm t_{0.025, \text{df}} \times \text{SE}$$
- **Decision Rule**: A performance gain is declared **statistically significant** if and only if:
  1. The two-tailed $p$-value satisfies $p < 0.05$ ($\alpha = 0.05$).
  2. The 95% Confidence Interval for token reduction strictly excludes zero ($0.0$).

---

## 6. Zero-Network Cassette Replay & Fail-Closed Verification

To verify that a recorded run can be 100% reproduced without making network calls or incurring LLM cost:

```bash
uv run python scenarios/debate/run_batch.py \
  --run-id cell1-replay \
  --seed 42 \
  --mode replay \
  --clock-now 2026-08-01T04:00:00Z \
  --seeds scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl \
  --output training_corpus_replay_42.jsonl \
  --attempts-out artifacts/attempts/cell1-replay.jsonl
```

### Fail-Closed Replay Behavior
If any seed in `--mode replay` encounters a cassette cache miss (`OfflineReplayError`), `run_batch.py` **fails closed**: it prints an explicit error and aborts with a non-zero exit code (`sys.exit(1)`), preventing corrupted or incomplete replay runs.

### Validating Replay Artifact Identity
Compare the recorded corpus, attempt logs, and batch manifest against the replayed run:

```bash
diff training_corpus_trip_42.jsonl training_corpus_replay_42.jsonl
diff artifacts/attempts/trip-v1-seed42.jsonl artifacts/attempts/cell1-replay.jsonl
diff artifacts/runs/trip-v1-seed42/batch_manifest.json artifacts/runs/cell1-replay/batch_manifest.json
```

Zero diff output across corpus, attempts, and manifest confirms 100% deterministic reproducibility.
