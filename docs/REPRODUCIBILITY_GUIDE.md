# Reproducibility & Data/Model Versioning Guide

> "Changing world values are inputs. Inputs are recorded. Recorded inputs are replayable. Replayable workflows are evolvable."

This document provides a step-by-step guide for reproducing the **BARRED Multi-Agent Vulnerability Dataset Swarm** results, version-controlling machine learning models (`artifacts/models`), executing the 2x2 Evaluation Matrix, and maintaining the active learning continual retraining flywheel.

---

## 1. Pillars of Swarm Reproducibility

Following MLOps and Data Version Control (DVC) best practices (Chawla, 2023), reproducing synthetic agent swarm datasets requires versioning three distinct pillars:

```
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
- `scenarios/debate/manifests/cve_seeds_v1_full50.jsonl` (Canonical 50-seed benchmark split)

### Verifying Seed File Hashes
Before executing a benchmark run, verify seed manifest integrity:

```bash
shasum -a 256 scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl
```

The SHA-256 digest is automatically recorded in `artifacts/runs/<run-id>/batch_manifest.json`.

---

## 3. The 2x2 Evaluation Matrix (Data Seeds vs PRNG Seeds)

To distinguish between **CVE Prompt Data** (the problem instances) and **PRNG Seeds** (LLM sampling controls), all evaluation runs map to the **2x2 Evaluation Matrix**:

```
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

- **Cell 1 (Replay)**: Verifies deterministic playback from cached cassettes.
- **Cell 3 (Stability)**: Uses the exact same 10 CVE data seeds across 3 random seeds to prove that token savings ($32.5\% \rightarrow 41.0\%$) are statistically stable ($41.0\% \pm 1.2\%$) and not a fluke of sampling.
- **Cell 4 (Generalization)**: Uses 10 fresh, unseen CVE data seeds to prove that Pre-Filter Stage B/C models generalize to new vulnerability classes without overfitting.

---

## 4. Triplicate Empirical Verification Protocol (3-Run Benchmark)

To execute Cell 3 of the matrix and demonstrate statistical significance across random initializations:

### Step 1: Run 3 Independent Random Seed Batches

```bash
# Run 1: Base Seed 42
uv run python scenarios/debate/run_batch.py \
  --run-id trip-v1-seed42 \
  --seed 42 \
  --mode record \
  --clock-now 2026-08-01T04:00:00Z \
  --seeds scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl \
  --output training_corpus_trip_42.jsonl \
  --attempts-out artifacts/attempts/trip-v1-seed42.jsonl

# Run 2: Base Seed 1337
uv run python scenarios/debate/run_batch.py \
  --run-id trip-v1-seed1337 \
  --seed 1337 \
  --mode record \
  --clock-now 2026-08-01T04:00:00Z \
  --seeds scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl \
  --output training_corpus_trip_1337.jsonl \
  --attempts-out artifacts/attempts/trip-v1-seed1337.jsonl

# Run 3: Base Seed 2026
uv run python scenarios/debate/run_batch.py \
  --run-id trip-v1-seed2026 \
  --seed 2026 \
  --mode record \
  --clock-now 2026-08-01T04:00:00Z \
  --seeds scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl \
  --output training_corpus_trip_2026.jsonl \
  --attempts-out artifacts/attempts/trip-v1-seed2026.jsonl
```

### Step 2: Compute Offline B-Gate Quality Metrics

```bash
./scripts/run_b_gate.sh training_corpus_trip_42.jsonl artifacts/attempts/trip-v1-seed42.jsonl artifacts/metrics/b_gate-trip-42.json
./scripts/run_b_gate.sh training_corpus_trip_1337.jsonl artifacts/attempts/trip-v1-seed1337.jsonl artifacts/metrics/b_gate-trip-1337.json
./scripts/run_b_gate.sh training_corpus_trip_2026.jsonl artifacts/attempts/trip-v1-seed2026.jsonl artifacts/metrics/b_gate-trip-2026.json
```

### Step 3: Compute Multi-Run Aggregate Telemetry (Mean ± StdDev)

```bash
uv run python scripts/debate_telemetry.py \
  --aggregate-runs trip-v1-seed42 trip-v1-seed1337 trip-v1-seed2026 \
  --output-markdown reports/triplicate_verification_report.md
```

---

## 5. Continual Pre-Filter Retraining Flywheel ($v1.0 \rightarrow v2.0 \rightarrow v3.0$)

As batch runs complete, every execution appends new labeled attempt traces into `artifacts/attempts/*.jsonl`. This forms an **Active Learning Data Flywheel**:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                        Active Learning Flywheel                         │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  1. RUN BATCH WITH MODEL v1.0 ───────┼────────► 2. LOG NEW ATTEMPTS
  (Filters current run)               │          (artifacts/attempts/*.jsonl)
                                      │                        │
  4. EVALUATE v2.0 vs v1.0 ◄──────────┼────────── 3. RETRAIN MODEL v2.0
  (Measures accuracy gain)            │          (scripts/train_pre_filter.py)
                                      │
```

### Retraining Protocol

#### 1. Versioned Model Checkpoint Directories
Model artifacts are versioned in discrete directories:
- `artifacts/models/v1.0_pre_filter/` (Initial baseline model trained on ~100 attempts)
- `artifacts/models/v2.0_pre_filter/` (Retrained model after accumulating ~400 attempts)
- `artifacts/models/v3.0_pre_filter/` (Retrained model after full 50-CVE batch execution)

#### 2. Executing Leak-Proof Model Retraining
Run `scripts/train_pre_filter.py` over accumulated attempt logs:

```bash
uv run python scripts/train_pre_filter.py \
  --attempts-dir artifacts/attempts \
  --output-dir artifacts/models/v2.0_pre_filter \
  --train-setfit
```

`partition_dataset_by_cve` automatically groups all attempt samples by CVE ID, ensuring strict leak-proof holdout evaluation between training and test splits.

#### 3. Comparing Model Performance (A/B Benchmark)
Compare `holdout_metrics.json` across versioned checkpoints:

```bash
diff artifacts/models/v1.0_pre_filter/holdout_metrics.json artifacts/models/v2.0_pre_filter/holdout_metrics.json
```

As the attempt dataset grows, Stage B (XGBoost) and Stage C (SetFit) achieve higher ROC-AUC and F1 scores, driving pre-filter token efficiency gains from **$41\%$ up toward $60-70\%$** on future runs.

---

## 6. Zero-Network Cassette Replay Verification

To verify that a recorded run can be 100% reproduced without making network calls or incurring LLM cost:

```bash
uv run python scenarios/debate/run_batch.py \
  --run-id trip-v1-seed42 \
  --seed 42 \
  --mode replay \
  --clock-now 2026-08-01T04:00:00Z \
  --seeds scenarios/debate/manifests/cve_seeds_v1_pilot10.jsonl \
  --output training_corpus_replay_42.jsonl \
  --attempts-out artifacts/attempts/trip-v1-replay-42.jsonl
```

### Validating Replay Bit-Identity

```bash
diff training_corpus_trip_42.jsonl training_corpus_replay_42.jsonl
```

Zero diff output confirms 100% deterministic reproducibility.
