# silver-one

`silver-one` is an Agentbeats evaluation project focused on deterministic, reproducible multi-agent security debates.

The main implemented scenario is **BARRED** (Boundary Adversarial Reasoning for Reproducible Evaluation and Dataset generation):
- a Green agent (`adk_debate_judge.py`) orchestrates debate rounds,
- two Purple agents (`debater.py`) argue opposite sides,
- an optional Verifier agent (`adk_debate_verifier.py`) audits groundedness,
- outputs are written as training corpus rows and audited with an offline B-gate.

## What This Repo Contains

- Agent runtime primitives under `src/agentbeats`.
- A complete debate scenario under `scenarios/debate`.
- Determinism tooling (record/replay cassettes and run records).
- Batch and Kaggle runners for larger corpus generation.

## Architecture

### Core runtime (`src/agentbeats`)

- `run_scenario.py`: starts participant and green agents from scenario TOML, waits for readiness, optionally launches the client.
- `client_cli.py`: sends `assessment_request` payload to the green agent and streams updates/artifacts.
- `client.py`: A2A messaging helpers.
- `green_executor.py`: base green-agent execution wrapper.
- `models.py`: shared Pydantic request/result models.
- `replay.py`: deterministic run infrastructure (`RunRecord`, `LLMCassette`, `ReplayManager`).
- `structured_output.py`: robust structured-output parser/repair pipeline for JSON responses.

### Debate scenario (`scenarios/debate`)

- `adk_debate_judge.py`: primary BARRED green agent with orchestration, strict gates, and export logic.
- `debater.py`: participant debater agent.
- `adk_debate_verifier.py`: optional verifier used for mechanism/anchor audits.
- `data_generator.py`: boundary-sample generation and refinement for BARRED loops.
- `run_batch.py`: seed-driven batch execution against the running judge.
- `offline_b_gate.py`: offline quality gate and metrics computation over generated corpus.
- `scenario.toml`: minimal debate scenario.
- `barred_test.toml`: BARRED scenario including verifier participant.

## Installation

### Prerequisites

- Python `>=3.11`
- `uv`
- model provider access (for example Ollama local models or remote LiteLLM providers)

### Setup

```bash
git clone https://github.com/surfiniaburger/silver-one.git
cd silver-one
uv sync
cp sample.env .env
```

Configure API/provider environment variables in `.env` as needed.

## Quick Start

Run the default debate scenario:

```bash
uv run agentbeats-run scenarios/debate/scenario.toml
```

Run with logs:

```bash
PYTHONPATH=src uv run agentbeats-run scenarios/debate/scenario.toml --show-logs
```

Serve agents only (no client run):

```bash
uv run agentbeats-run scenarios/debate/scenario.toml --serve-only
```

### Model overrides

The scenario reads model choices from environment variables:

- `JUDGE_MODEL`
- `DEBATER_MODEL`
- `GENERATOR_MODEL`
- `VERIFIER_MODEL`
- `GEPA_MODEL` (used by seed-loader workflows)

### Sampling controls

Tracked LiteLLM calls read and record sampling controls from environment variables:

- `LLM_SAMPLING_PROFILE=ollama_gemma4`
- `LLM_TEMPERATURE`
- `LLM_TOP_P`
- `LLM_TOP_K`
- `LLM_MAX_TOKENS`

The `ollama_gemma4` profile applies Ollama's Gemma 4 sampling recommendation only to tracked models whose name contains `gemma4`:

```bash
temperature=1.0
top_p=0.95
top_k=64
```

These values are written into run records, attempt logs, corpus row metadata, and B-gate metrics so generation settings are treated as control variables rather than hidden confounders.

Example:

```bash
JUDGE_MODEL="ollama/qwen2.5-coder:7b" \
DEBATER_MODEL="ollama/qwen2.5-coder:7b" \
uv run agentbeats-run scenarios/debate/scenario.toml
```

## BARRED Workflow

## Latest Benchmark Snapshot (Run 5)

Source: `artifacts/metrics/b_gate-20260527-043758.json`

| Metric | Value |
| :--- | :--- |
| Pass | `true` |
| Accepted Rows | `13` |
| Attempts | `30` |
| Verifier Parse OK Rate | `1.00` |
| Verifier Pass Rate | `0.6842` |
| Strict B2 Fail Rate | `0.1667` |
| Total Tokens | `846,681` |
| Tokens / Attempt | `28,222.7` |
| Tokens / Accepted Row | `65,129.3` |
| Usage Source Coverage | `provider: 101/101 (100%)` |

Top token sinks by stage:

| Stage | Calls | Prompt Tokens | Completion Tokens | Total Tokens | Share |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `generator_boundary` | 21 | 407,887 | 24,539 | 432,426 | 51.1% |
| `generator_refine` | 20 | 174,489 | 38,396 | 212,885 | 25.1% |
| `judge_adjudication` | 60 | 121,413 | 79,957 | 201,370 | 23.8% |

### Option A: Start the full stack via helper script

```bash
# Terminal 1: Start full stack (judge + debaters + verifier)
./scenarios/debate/start_stack.sh
```

This script:
- kills stale listeners on ports `9009,9018,9019,9020`
- exports default model env vars if not already set
- launches `agentbeats-run scenarios/debate/barred_test.toml --serve-only`

### Option B: Start the full stack directly

```bash
uv run agentbeats-run scenarios/debate/barred_test.toml --serve-only
```

### 2) Run batch generation from seeds

```bash
uv run python scenarios/debate/run_batch.py \
  --seeds scenarios/debate/cve_seeds_50.jsonl \
  --output training_corpus.jsonl \
  --run-id pilot-v1 \
  --seed 42 \
  --mode record
```

### 3) Compute offline B-gate metrics

```bash
uv run python scenarios/debate/offline_b_gate.py \
  --input training_corpus.jsonl \
  --attempts artifacts/attempts/pilot-v1.jsonl \
  --metrics-out artifacts/metrics/b_gate.json
```

### Soft-check run (attempts + B metrics)

Use this exact flow when you want attempts-level soft checks captured and scored:

```bash
# Terminal 1: Start full stack (judge + debaters + verifier)
./scenarios/debate/start_stack.sh

# Re-run a record run to generate attempts with soft_checks
uv run python scenarios/debate/run_batch.py \
  --run-id pilot-v1-softchecks \
  --mode record \
  --seed 42 \
  --seeds scenarios/debate/cve_seeds_test.jsonl \
  --output training_corpus.jsonl \
  --attempts-out artifacts/attempts/pilot-v1-softchecks.jsonl

# Compute B metrics + soft-check rates
./scripts/run_b_gate.sh

or

UV_CACHE_DIR=/tmp/uv-cache uv run python scenarios/debate/offline_b_gate.py \
  --input training_corpus.jsonl \
  --attempts artifacts/attempts/pilot-v1-softchecks.jsonl \
  --metrics-out artifacts/metrics/b_gate.json
```

## Determinism and Replay

The project supports deterministic record/replay for model calls.

- **Record mode**: real provider calls are made, responses are cached.
- **Replay mode**: no cache miss is allowed; missing entries fail fast.

Key artifacts:

- `artifacts/cassettes/<run-id>.json`
- `artifacts/runs/<run-id>.json`
- `artifacts/attempts/<run-id>.jsonl`

Replay example:

```bash
uv run python scenarios/debate/run_batch.py \
  --run-id pilot-v1 \
  --seed 42 \
  --mode replay \
  --seeds scenarios/debate/cve_seeds_50.jsonl \
  --output training_corpus_replay.jsonl
```

## Kaggle Runner

Use the automation helper:

```bash
uv run python kaggle_notebooks/run_barred_kaggle.py \
  --scenario scenarios/debate/barred_test.toml \
  --run-id kaggle-pilot \
  --mode record \
  --seed 42 \
  --seeds scenarios/debate/cve_seeds_50.jsonl \
  --output /kaggle/working/training_corpus.jsonl \
  --attempts-out /kaggle/working/attempts.jsonl \
  --metrics-out /kaggle/working/b_gate.json
```

## Testing

Scenario-level tests currently live under `scenarios/debate`.

```bash
uv run python scenarios/debate/test_structured_output.py
uv run python scenarios/debate/test_offline_b_gate.py
```

If you have `pytest` installed in your environment:

```bash
uv run pytest -q scenarios/debate
```

## Repository Layout

```text
silver-one/
├─ src/agentbeats/
│  ├─ run_scenario.py
│  ├─ client_cli.py
│  ├─ replay.py
│  ├─ structured_output.py
│  └─ ...
├─ scenarios/debate/
│  ├─ adk_debate_judge.py
│  ├─ adk_debate_verifier.py
│  ├─ debater.py
│  ├─ data_generator.py
│  ├─ run_batch.py
│  ├─ offline_b_gate.py
│  ├─ scenario.toml
│  └─ barred_test.toml
├─ kaggle_notebooks/
│  └─ run_barred_kaggle.py
├─ sample.env
├─ pyproject.toml
└─ README.md
```

## Architecture Diagrams and Stress-Test Playbook

Use the Mermaid-based breakdown and critique flows here:
- `docs/ARCHITECTURE_MERMAID.md`

It maps Prompt Engineering, Context Engineering, and Harness Engineering directly to the `silver-one` code paths and includes stress-test/checklist flows for security and quality reviews.

## Contributing

- Keep changes deterministic-friendly (record/replay aware).
- Update scenario docs if you change config fields in TOML or expected output schema.
- Prefer robust structured JSON parsing via `src/agentbeats/structured_output.py`.

See `AGENT.md` for repository-specific coding and review instructions.

## License

MIT License. See `LICENSE`.
