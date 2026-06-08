---
title: "silver-one — Our Comeback Story (Finish-Up-A-Thon Submission)"
tags: [github-finish-up-a-thon, agentbeats, dataset, security]
---

# silver-one — Finish-Up-A-Thon Submission

**This is a submission for the [GitHub Finish-Up-A-Thon Challenge](https://dev.to/challenges/github-2026-05-21)**

## What I Built

This project began life as an Agentbeats platform template for the AgentX-AgentBeats competition (initial commits in Nov 2025). At the time it was a straightforward pro/con debater and judge wired to local models (Ollama). Over the last several months I reworked the harness into `silver-one`: a reproducible, auditable pipeline for generating evidence-grounded code-security reasoning data.

![Initial rest point](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/by1d82v0kp8kqg7cwum1.png)


Key ideas implemented:
- BARRED (Boundary Adversarial Reasoning for Reproducible Evaluation and Dataset generation) — an agent debate + verifier architecture that records every changing input for replay and audit.
- Deterministic replay: `LLMCassette`/`ReplayManager` to record prompts, responses, and run state so outputs are reproducible and debuggable.
- Offline B-gate: `offline_b_gate.py` computes quality gates (structural completeness, anchor grounding, predicate aboutness, verifier parse/pass) and produces `artifacts/metrics/*.json`.
- Telemetry and token accounting: per-stage and per-model token totals so we can optimize cost vs quality.

Why this matters: synthetic security corpora are easy to generate but hard to trust. `silver-one` treats generation settings, verifier outcomes, rejected attempts, and run checkpoints as first-class artifacts — enabling training data to be audited, replayed, and improved.

## Demo

![Current Workflow](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/d7p5vnfkypvm86ibt68i.png)

The repo includes scripts and instructions to run the BARRED stack locally (see `scenarios/debate/start_stack.sh` and `scenarios/debate/run_batch.py`). Example B-gate computation:

```bash
# start judge + participants
./scenarios/debate/start_stack.sh

# run a clocked batch and export attempts + corpus
uv run python scenarios/debate/run_batch.py \
  --run-id pilot-v1-calibrated-d \
  --seed 42 \
  --mode record \
  --clock-now 2026-06-07T14:12:00Z \
  --seeds scenarios/debate/cve_seeds_test.jsonl \
  --output training_corpus_calibrated_d.jsonl \
  --attempts-out artifacts/attempts/pilot-v1-calibrated-d.jsonl


# compute B-gate metrics
./scripts/run_b_gate.sh \
  training_corpus_calibrated_d.jsonl \
  artifacts/attempts/pilot-v1-calibrated-d.jsonl \
  artifacts/metrics/b_gate-pilot-v1-calibrated-d.json
```

Screenshots / video: the harness is CLI-first and instrumented; artifacts in `artifacts/` contain the data needed to build a short walkthrough video.

## The Comeback Story

Where it started: the project was a small Agentbeats demo wired to Ollama models (Nov 2025). Over time, the repo grew into a research harness as I discovered that simple debate + label workflows produced many ungrounded labels.

What I changed and finished up:
- Hardened structured-output parsing and repair to avoid malformed JSON rows.
- Added deterministic replay (`ReplayManager`) so the same prompts and responses can be replayed and audited.
- Implemented a verifier agent and wired it into judge gating; verifier reports are now used to improve grounding and reduce hallucinated mechanism claims.
- Built `offline_b_gate.py` to compute reproducible quality metrics and surface failure modes (anchor normalization, mechanism grounding, predicate aboutness).
- Added telemetry (per-stage tokens, per-model usage) so we can optimize generator context size without losing auditability.

Result: the harness now produces a small high-fidelity corpus (examples: `training_corpus_calibrated_a.jsonl`…`_d.jsonl`) with accompanying `artifacts/metrics/b_gate-pilot-v1-calibrated-*.json` showing gate pass and detailed telemetry.

## Inspiration & Context

The redesign was strongly inspired by the Google DeepMind AGI hackathon and the Metacognitive research conversations there: we wanted to observe how models behave under adversarial conditions and make those behaviours auditable. I also leaned on practical tutorials and engineering approaches (e.g., Daily Dose of Data Science, Dave Farley / Modern Software Engineering) to make the system deterministic and "test-easy by design" — small, replayable units, strong structured-output parsing, and clear telemetry.

This work is directly connected to the Metacognitive Coding Safety Benchmark (MCSB) — which was our submission to the Google DeepMind AGI hackathon. MCSB's multi-tier structure (pilot/core/adversarial) and its focus on directional confidence updates shaped our Tiered experiment design and the offline B-gate evaluation. 

### A note on cassettes — why the metaphor matters
Peter Quill's mixtape in Guardians of the Galaxy is a tiny, precious archive that preserves memory, identity, and the reason a song feels like "the best." Our `LLMCassette` serves a similar purpose for `silver-one`: it preserves the prompt, model responses, and run state so the project's decisions remain auditable, replayable, and emotionally intelligible. Treating generation traces as a cassette makes the system kinder to debugging and kinder to future researchers — you can rewind to the exact moment a label was created, listen to the context, and understand why a model thought a particular answer was "the best." 

## Kaggle Task & Repo

We published a companion Kaggle benchmark task for predicate-quality evaluation that lets you test multiple models locally and run remote benchmark jobs from the CLI. Task link:

https://www.kaggle.com/benchmarks/tasks/surfiniaburger/silver-one-predicate-quality

To push and run the task from the repository (example from `kaggle_notebooks/silver_one_predicate_quality_task.py`):

```bash
kaggle b t push silver-one-predicate-quality -f kaggle_notebooks/silver_one_predicate_quality_task.py --wait
kaggle b t run  silver-one-predicate-quality -m gemini-3.5-flash --wait
```

Repository (source): https://github.com/surfiniaburger/silver-one

## My Experience with GitHub Copilot

GitHub Copilot (and iterative local editing) helped speed up refactors and suggested tests and doc improvements during the finish-up. I used the suggestions as a productivity assist rather than an authoritative change — every structural tweak was followed by running the deterministic smoke path and validating metric artifacts.

## Files I relied on to finish this up
- `README.md` — project summary and quick start
- `pilot_report.md` — pilot run analysis, verifier-era notes, and telemetry interpretation
- `scenarios/debate/*` — implementation of BARRED, judge, verifier, data generator, batch runner
- `src/agentbeats/replay.py` and `src/agentbeats/structured_output.py` — deterministic replay and structured-output repair
- `artifacts/metrics/b_gate-pilot-v1-calibrated-*.json` — final calibrated metrics (used to summarize improvements)

## Where to go from here
- Improve anchor normalization to increase accepted yield without raising predicate failures.
- Add CI guardrails to limit `generator_boundary` prompt sizes and prevent runaway token use.
- Expand verifier capabilities toward bit-level data-flow tracing for stronger mechanism grounding.

## References & Related Docs

 - **BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate** — Arnon Mazza*, Elad Levi* (Plurai Inc.), Preprint Jan 21, 2026. See [BARRED/Barred.md](BARRED/Barred.md#L1). Role: scenario specification and debate-based synthetic-data generation algorithm; served as the blueprint for the BARRED scenario, gating rules, and the offline B-gate implementation used in this repo. (*equal contribution)

 - **Pioneer Agent: Continual Improvement of Small Language Models in Production** — Dhruv Atreja, Julia White, Nikhil Nayak, Kelton Zhang, Henrijs Princis, George Hurn-Maloney, Ash Lewis, Urchade Zaratiana (Fastino Labs), arXiv:2604.09791, Apr 10, 2026. See [docs/paper.md](docs/paper.md#L1). Role: engineering systems paper that inspired deterministic replay, telemetry-driven adaptation loops, and the economic-efficiency framing (CVT / Trust Score) used in our evaluation.

These two documents are core references for the project: `BARRED/Barred.md` captures the scenario-level design and gating semantics; `docs/paper.md` provides a broader systems and evaluation perspective that informed metric choices and economic analyses.

---
_Repo snapshot and recent activity:_ initial commits Nov 2025; active development and calibration runs May–June 2026 (see git history). This submission file was generated from the repository sources and commit log.

## Figures — Key Metrics Visuals

The following figures were generated from `artifacts/metrics` and saved in `artifacts/metrics/plots/`.

![Attempts vs Accepted](artifacts/metrics/plots/yield_attempts_accepted.png)
- Attempts vs Accepted rows by run — shows yield and how many attempts were required per accepted corpus row.

![Predicate & B2 Fail Rates](artifacts/metrics/plots/quality_rates.png)
- Predicate-fail and B2 strict-fail by run — highlights quality improvements and failure modes across runs.

![Verifier Rates](artifacts/metrics/plots/verifier_rates.png)
- Verifier called rate and verifier pass rate by run — shows verifier coverage and effectiveness.

![Cost vs Quality](artifacts/metrics/plots/cost_vs_quality.png)
- Tokens per accepted row vs predicate-fail (point size = accepted rows) — visualizes the cost/quality tradeoff.

![Stage Token Breakdown](artifacts/metrics/plots/stage_token_breakdown.png)
- Stacked token usage by stage (`generator_boundary`, `generator_refine`, `judge_adjudication`, `verifier_audit`) — identifies where tokens are spent.

![Model Token Share](artifacts/metrics/plots/model_token_share.png)
- Per-model token share by run — shows which models dominate token costs.

![Metrics Heatmap](artifacts/metrics/plots/metrics_heatmap.png)
- Normalized heatmap of core metrics across runs for quick comparison.

---
Created by repository automation and a quick audit of the git history and pilot reports.
