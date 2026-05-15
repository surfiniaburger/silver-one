#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Clean stale listeners from previous runs.
lsof -ti:9009,9018,9019,9020 | xargs kill -9 2>/dev/null || true

JUDGE_MODEL="${JUDGE_MODEL:-ollama/gpt-oss:120b-cloud}"
DEBATER_MODEL="${DEBATER_MODEL:-ollama/gemma4:31b-cloud}"
GENERATOR_MODEL="${GENERATOR_MODEL:-ollama/gemma4:31b-cloud}"
VERIFIER_MODEL="${VERIFIER_MODEL:-ollama/gpt-oss:120b-cloud}"
export JUDGE_MODEL DEBATER_MODEL GENERATOR_MODEL VERIFIER_MODEL

echo "Starting debate stack on ports 9009, 9018, 9019, 9020"
exec uv run agentbeats-run scenarios/debate/barred_test.toml --serve-only
