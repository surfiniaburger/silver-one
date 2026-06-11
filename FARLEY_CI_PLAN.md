# Farley CI Plan

This document describes how to record, compare, and gate Farley Score evaluations for the repository using cassettes and CI.

> Note: filename intentionally created as `FARLEY_CI_PLAN.md` per request.

---

## Goals

- Produce replayable Farley evaluation cassettes for PRs and baselines.
- Enable deterministic replay in CI (no LLM calls) when using `--mode replay`.
- Provide a concise Markdown report posted to PRs summarizing regressions.
- Gate merges when test-quality regressions exceed configured thresholds.

---

## Cassette JSON Schema (recommended)

Store structured outputs only (no raw prompts by default). Per-run cassette schema (JSON):

- run_id: string
- timestamp: ISO8601
- git_commit: string
- branch: string
- tool_version: string
- model: string
- evaluator_config: { mode, weights, seed }
- paths_scanned: [string]
- tests: [
  {
    id: string (sha256 of file path + test name),
    file_path: string,
    test_name: string,
    class_name: string | null,
    code_hash: string (sha256, optional),
    code_snippet: string (optional, opt-in),
    farley_breakdown: {
      understandable: { score: number, rationale: string, suggestions: [string] },
      maintainable: {...},
      repeatable: {...},
      atomic: {...},
      necessary: {...},
      granular: {...},
      fast: {...},
      first_tdd: {...},
      summary: string
    },
    farley_index: number
  }
]
- suite_summary: { avg_index: number, count: int, per_property_avg: { U,M,R,A,N,G,F,T } }

Notes:
- Per-test granularity is recommended.
- `code_snippet` should be opt-in (privacy). Always store `code_hash` for uniqueness.
- Keep cassette file compressed if large (e.g., `farley_score-<run_id>.json.gz`).

---

## Example cassette entry (short)

```json
{
  "run_id":"pr-123",
  "timestamp":"2026-06-11T12:00:00Z",
  "git_commit":"abcdef012345",
  "branch":"feature/x",
  "tool_version":"farley-evaluator-1.0",
  "model":"local/litellm:gpt4-lite",
  "paths_scanned":["src/","tests/"],
  "tests":[
    {
      "id":"b2f5ff47436671b6e533d8dc3614845d",
      "file_path":"tests/test_math.py",
      "test_name":"test_add",
      "class_name":null,
      "code_hash":"...",
      "farley_breakdown":{
        "understandable":{"score":9,"rationale":"Clear name","suggestions":[]},
        "maintainable":{"score":8,"rationale":"Behavior-level assertions","suggestions":[]},
        "repeatable":{"score":10,"rationale":"No I/O","suggestions":[]},
        "atomic":{"score":9,"rationale":"Single outcome","suggestions":[]},
        "necessary":{"score":10,"rationale":"Unique check","suggestions":[]},
        "granular":{"score":9,"rationale":"Precise target","suggestions":[]},
        "fast":{"score":10,"rationale":"Instant","suggestions":[]},
        "first_tdd":{"score":8,"rationale":"AAA style","suggestions":[]},
        "summary":"Good unit test."
      },
      "farley_index":9.0
    }
  ],
  "suite_summary":{"avg_index":9.0,"count":1}
}
```

---

## CI Gating Rules (recommended defaults)

- Baseline cassette: `artifacts/cassettes/farley_score-main-latest.json` (update manually or via automation on main).
- PR cassette: `artifacts/cassettes/farley_score-PR-<pr-number>.json`

Fail (exit non-zero) if ANY of the following:
- Suite Farley Index decreases by >= `0.25` (PR_avg - baseline_avg <= -0.25).
- Average `Understandable` OR `Maintainable` decreases by >= `0.5`.
- > `5%` of evaluated tests drop by >= `2.0` points.

Warn (pass but annotate) if:
- Suite delta is between -0.10 and -0.25.
- Property drops are between -0.25 and -0.5.

Notes:
- Thresholds are tunable per repository. For smaller suites, use absolute counts instead of percentages.
- You may treat single-test big drop (>=3 points) as `flag` for human review rather than automatic fail.

---

## CI Flow (high level)

1. Checkout PR branch.
2. (Optional) Start local LLM provider (LiteLLM, Ollama, or container) if using a local model.
3. Run evaluator in record mode to produce PR cassette.
4. Run compare script against baseline cassette.
5. Produce Markdown report and post to PR (optionally via `GITHUB_TOKEN`).
6. Upload cassette & report as workflow artifacts.

Example run command (in CI):

```bash
python3 scripts/farley_score_evaluator.py . --mode record --cassette artifacts/cassettes/farley_score-PR-${{PR_NUMBER}}.json --run-id pr-${{PR_NUMBER}} --model "local/litellm:gpt4-lite"
```

---

## GitHub Actions example (skeleton)

This example shows running a local LiteLLM container and the evaluator. Adapt to your model/service.

```yaml
name: Farley Score
on: [pull_request]

jobs:
  farley-eval:
    runs-on: ubuntu-latest
    env:
      PR_NUMBER: ${{ github.event.number }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Start LiteLLM (optional)
        run: |
          # Example: start a local lite-llm server (adapt to your runtime)
          docker run -d --name litellm -p 8080:8080 your-lite-llm-image:latest

      - name: Install deps
        run: |
          python -m pip install -r requirements.txt || true

      - name: Run Farley evaluator
        run: |
          python3 scripts/farley_score_evaluator.py . --mode record --cassette artifacts/cassettes/farley_score-PR-${PR_NUMBER}.json --run-id pr-${PR_NUMBER} --model "local/litellm:gpt4-lite"

      - name: Upload cassette
        uses: actions/upload-artifact@v4
        with:
          name: farley-cassette-pr-${{ env.PR_NUMBER }}
          path: artifacts/cassettes/farley_score-PR-${{ env.PR_NUMBER }}.json

      - name: Compare to baseline and comment
        run: |
          python3 scripts/farley_compare.py --baseline artifacts/cassettes/farley_score-main-latest.json --pr artifacts/cassettes/farley_score-PR-${PR_NUMBER}.json --out report.md || true
          # Use gh or curl to post report.md as a PR comment if desired.
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

Notes:
- Replace LiteLLM container line with the provider you use.
- If using remote API (e.g., Nebius/OpenAI), ensure secrets are available and `call_structured` / adapter is configured.

---

## Comparison script outline (`scripts/farley_compare.py`)

Responsibilities:
- Load baseline and PR cassettes.
- Compute suite avg delta and per-property averages.
- Create tables: Top regressions by test, top worst tests in PR, per-property diffs.
- Return non-zero exit code if fail thresholds reached.
- Emit `report.md` (Markdown) with the report template below.

Pseudocode summary:

```python
import json
from statistics import mean

base = json.load(open(baseline))
pr = json.load(open(prfile))

# Build maps by test id
# Compute averages and deltas
# Identify regressions
# Write report.md
# Exit with code >0 if fail conditions met
```

I'll include a runnable reference implementation later if you'd like.

---

## Reporter Markdown template

Use the following sections (CI `report.md`) and post to PR:

- Header with PR and verdict (PASS/WARN/FAIL)
- Summary: baseline vs PR indices and delta
- Top 5 regressions table
- Per-property summary
- Attach links to artifacts (cassette JSON)
- Suggested reviewer actions

(Template previously provided in conversation; include verbatim in `report.md` generation.)

---

## Secrets & Environment

- `NEBIUS_API_KEY` (if using Nebius OpenAI-compatible endpoint)
- `LITELLM_API_KEY` or `LITELLM_HOST` (if using hosted LiteLLM)
- `GITHUB_TOKEN` (for posting PR comments)

Security:
- DO NOT write secrets into cassettes. Ensure the evaluator redacts any environment variables when recording prompts or responses.

---

## Replay mode

- To reproduce a run without calling the LLM, support `--mode replay` which causes `call_structured` to return structured responses from the cassette by test `id`.
- This allows reviewers to re-render reports deterministically in CI or locally.

---

## Redaction & Privacy

- Default: store only structured Farley breakdown.
- Optional flag `--include-prompts` to store raw prompts/responses (useful for debugging). Must be disabled by default.
- If raw is enabled, record only if `REDACT_SENSITIVE` is false. Prefer storing `code_hash` instead of full `code_snippet`.

---

## Next steps (recommended order)

1. Confirm decisions: per-test granularity (yes/no), include_prompts default (yes/no), fail thresholds.
2. Implement `scripts/farley_compare.py` (reference implementation).
3. Add GitHub Actions job (copy skeleton above); test with a small PR.
4. Optionally: implement `--include-prompts` opt-in and safe redaction.
5. Add baseline update automation for `main` branch (scheduled job that updates `artifacts/cassettes/farley_score-main-latest.json`).

---

## Appendix: Adapter Notes for common providers

- LiteLLM / local server: start server in CI, set `--model` to provider identifier (e.g., `local/litellm:gpt4-lite`), ensure `call_structured` uses the host/port.
- Nebius / OpenAI-compatible: use OpenAI-compatible client with `base_url` and `api_key` (example provided in conversation). Ensure `call_structured` maps model string to provider call.
- Ollama: if using `ollama` CLI or REST, start or ensure availability in CI and point `--model` accordingly.

---

If you confirm this filename and the default decisions (per-test granularity = yes, include_prompts = opt-in/disabled, fail thresholds as listed), I'll next draft a runnable `scripts/farley_compare.py` and an initial GitHub Actions YAML ready for your repo.
