# AGENT.md

Repository-specific instructions for contributors and coding agents.

## Scope

This file defines how to work safely and consistently in `silver-one`.

## Ground Rules

- Do not remove or bypass deterministic record/replay behavior in `src/agentbeats/replay.py`.
- Do not add parallel or hidden model calls that skip `ReplayManager` in judge/generator paths.
- Keep structured JSON parsing centralized in `src/agentbeats/structured_output.py`.
- Preserve scenario compatibility with `scenarios/debate/scenario.toml` and `scenarios/debate/barred_test.toml`.

## Entry Points

- Main runner: `uv run agentbeats-run <scenario.toml>`
- Batch runner: `uv run python scenarios/debate/run_batch.py ...`
- Offline gate: `uv run python scenarios/debate/offline_b_gate.py ...`
- Kaggle helper: `uv run python kaggle_notebooks/run_barred_kaggle.py ...`

## Code Ownership by Area

- `src/agentbeats/*`: shared runtime utilities. Prefer backward-compatible changes.
- `scenarios/debate/adk_debate_judge.py`: orchestration and gating logic. High-impact file.
- `scenarios/debate/data_generator.py`: generation/refinement prompts and schema calls.
- `scenarios/debate/offline_b_gate.py`: quality metrics and release thresholds.

## Required Practices

### Structured output

- Use `call_structured(...)` for schema-constrained model outputs.
- If parsing helpers are needed, import from `agentbeats.structured_output`; do not duplicate.
- Maintain graceful fallback + repair behavior.

### Determinism

- New model calls in BARRED flow must be routed through `ReplayManager.acompletion(...)` unless intentionally exempt (like verifier's bare mode).
- In replay mode, treat cache misses as failures, not silent fallthrough.

### Logging and artifacts

- Keep attempt logs append-only JSONL.
- Do not silently change field names in exported corpus/output objects.
- If schema fields change, update docs and tests in same PR.

## Local Validation Checklist

Before opening a PR:

1. Run a quick scenario smoke test:
   - `uv run agentbeats-run scenarios/debate/scenario.toml --serve-only`
2. Run parser/gate tests:
   - `uv run python scenarios/debate/test_structured_output.py`
   - `uv run python scenarios/debate/test_offline_b_gate.py`
3. If touching batch/determinism code, run one short record/replay cycle.
4. Verify README examples still match real commands and files.

## PR Guidelines

- Keep PRs focused (one logical change per PR).
- Note any env vars introduced or changed.
- Include migration note when changing JSON shape or metric keys.
- Avoid committing generated corpus artifacts unless the PR is explicitly about fixture updates.

## Common Pitfalls

- Duplicating helper logic in scenario files instead of shared module.
- Adding non-deterministic behavior (time/random/network) without recording inputs.
- Breaking replay by modifying call payload hashing semantics without migration strategy.

## When Unsure

If a change may affect evaluation correctness or reproducibility, prefer:

1. conservative behavior,
2. explicit logging,
3. follow-up refactor PR instead of silent broad edits.
