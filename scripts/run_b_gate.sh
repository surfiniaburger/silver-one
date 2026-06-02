#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_PATH="${1:-training_corpus_clocked.jsonl}"
ATTEMPTS_PATH="${2:-artifacts/attempts/pilot-v1-clocked.jsonl}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_PATH="${3:-artifacts/metrics/b_gate-${STAMP}.json}"

mkdir -p "$(dirname "$OUT_PATH")"

echo "[b-gate] cwd:        $PWD"
echo "[b-gate] input:      $INPUT_PATH"
echo "[b-gate] attempts:   $ATTEMPTS_PATH"
echo "[b-gate] metrics out:$OUT_PATH"

UV_CACHE_DIR=/tmp/uv-cache uv run python scenarios/debate/offline_b_gate.py \
  --input "$INPUT_PATH" \
  --attempts "$ATTEMPTS_PATH" \
  --metrics-out "$OUT_PATH"

echo "[b-gate] done: $OUT_PATH"
