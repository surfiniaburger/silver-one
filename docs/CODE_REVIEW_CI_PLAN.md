# Code Review CI Plan

Building an in-house LLM-powered code reviewer to replace Gemini Code Assist
(sunset July 17 2026), modelled on the existing Farley Score evaluator pipeline.

---

## Context

The Gemini Code Assist bot proved its value on PR #51 by catching a **semantic
path-resolution bug** that static tools like `ruff` and `bandit` would have
missed.  With the bot being sunset, we can replicate and extend that capability
using the infrastructure we already have.

---

## What We Already Have (the reusable chassis)

The Farley evaluator established a complete, working pattern:

```
git diff → extract units → LLM structured eval → cassette → compare → report.md → PR comment
```

Every piece of this pipeline is reusable:

| Component | Location | Reuse |
|---|---|---|
| Provider-agnostic LLM calls | `scripts/llm_adapter.py` | Verbatim |
| Structured output + cassette | `call_structured` / `ReplayManager` | Verbatim |
| Baseline diffing + gate logic | `scripts/farley_compare.py` | Thin wrapper |
| PR comment posting | `.github/workflows/farley_ci.yml` | Extend existing job |
| Path validation / sanitization | `validate_path` in evaluator | Verbatim |

The only genuinely new piece is a **diff extractor** that reads a `git diff`
and produces function-level review units instead of test-function AST nodes.

---

## The Key Structural Difference

|  | Farley (test review) | Code Review |
|---|---|---|
| **Input unit** | AST-extracted test function | `git diff` hunk + enclosing function |
| **Extraction** | `TestExtractor(ast.NodeVisitor)` | `git diff` + AST line-range filter |
| **Schema** | `FarleyScoreBreakdown` (8 properties) | `CodeReviewBreakdown` (CQI dimensions) |
| **Cassette key** | `sha256(file_path + test_name)` | `sha256(file_path + fn_name + diff_hash)` |
| **Baseline** | Per-test Farley index | Per-function CQI (or per-file aggregate) |
| **Gate logic** | Suite delta, property drops | Severity counts, BLOCK/WARN/OK per unit |

---

## Proposed Review Schema: `CodeReviewBreakdown`

Based on the Code Quality Index (CQI) defined in `CODE_TEST_QUALITY_PLAN.md`:

```python
from typing import Literal, List
from pydantic import BaseModel, Field

class PropertyEvaluation(BaseModel):
    score: int = Field(..., description="Score 0-10, 10 is perfect.")
    rationale: str = Field(..., description="1-2 sentences justifying the score.")
    suggestions: List[str] = Field(default_factory=list)

class CodeReviewBreakdown(BaseModel):
    readability: PropertyEvaluation      # naming, docstrings         weight 1.5
    maintainability: PropertyEvaluation  # coupling, modularization    weight 1.5
    correctness: PropertyEvaluation      # logic errors, edge cases    weight 2.0
    complexity: PropertyEvaluation       # cyclomatic, nesting depth   weight 1.0
    security: PropertyEvaluation         # injection, traversal, leaks weight 2.0
    test_coverage: PropertyEvaluation    # changed code has tests      weight 1.0
    summary: str
    severity: Literal["OK", "WARN", "BLOCK"]  # top-level verdict per unit
```

`security` and `correctness` are weighted 2.0 and can individually trigger a
`BLOCK` verdict regardless of the aggregate CQI score.

### CQI Formula

```
CQI = (1.5·readability + 1.5·maintainability + 2.0·correctness
       + 1.0·complexity + 2.0·security + 1.0·test_coverage) / 9.0
```

---

## Architecture: `scripts/code_review_evaluator.py`

### Step 1 — Extract changed units from diff

```python
import subprocess, ast

def get_diff(base_ref: str = "origin/main") -> str:
    return subprocess.check_output(
        ["git", "diff", base_ref, "--unified=5", "--", "*.py"],
        text=True,
    )

def extract_changed_functions(diff: str) -> list[dict]:
    """
    Parse diff hunks → for each changed file, use AST to find which
    functions/classes contain the changed lines → return those as units.
    Returning the full function body (not just diff lines) gives the LLM
    enough context to reason about correctness and intent.
    """
    ...
```

The key design choice: extract the **enclosing function body**, not just the
raw diff hunk.  Raw diff lines lack the context the LLM needs to catch semantic
bugs (as demonstrated by the Gemini review on PR #51).

### Step 2 — Evaluate each unit

```python
async def evaluate_unit(replay_mgr, model: str, unit: dict) -> CodeReviewBreakdown:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": format_unit(unit)},
    ]
    return await call_structured(
        replay_manager=replay_mgr,
        model=model,
        messages=messages,
        schema_name="CodeReviewBreakdown",
        schema_model=CodeReviewBreakdown,
        stage="code_review",
    )
```

### Step 3 — Write cassette (same schema as Farley)

Unit ID: `sha256(file_path + function_name + diff_hash)` — stable across runs
on the same change, changes when the function body changes.

---

## `scripts/code_review_compare.py`

Simpler than `farley_compare.py` — no numeric baseline needed initially:

- Count `BLOCK` / `WARN` / `OK` units in the PR cassette
- Gate: any `BLOCK` → CI fails; `WARN` count > threshold → warning comment
- Emit `code_review_report.md` in the same Markdown format

Later: persist a baseline cassette on `main` to track CQI trends over time,
exactly like the Farley baseline workflow.

---

## CI Integration

Add a second job to `farley_ci.yml` (reuses the Ollama service container):

```yaml
code-review:
  needs: []          # runs in parallel with farley-eval
  permissions:
    pull-requests: write
    contents: read
  runs-on: ubuntu-latest
  env:
    PR_NUMBER: ${{ github.event.number }}
  services:
    ollama:
      image: ollama/ollama:latest
      ports:
        - 11434:11434
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0    # need full history for git diff

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install deps
      run: python -m pip install -e .

    - name: Pull model
      run: |
        curl -sS -X POST http://localhost:11434/api/pull \
          -H 'Content-Type: application/json' \
          -d '{"name":"qwen3.5:2b"}' || true

    - name: Run code review evaluator
      run: |
        python3 scripts/code_review_evaluator.py \
          --base origin/main \
          --cassette pr-code-review-${{ env.PR_NUMBER }}.json \
          --run-id pr-cr-${{ env.PR_NUMBER }} \
          --model "ollama/qwen3.5:2b"

    - name: Ensure report exists
      if: always()
      run: touch $GITHUB_WORKSPACE/code_review_report.md

    - name: Compare and gate
      run: |
        python3 scripts/code_review_compare.py \
          --pr $GITHUB_WORKSPACE/artifacts/cassettes/pr-code-review-${{ env.PR_NUMBER }}.json \
          --out $GITHUB_WORKSPACE/code_review_report.md || true

    - name: Post code review as PR comment
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        github-token: ${{ secrets.GITHUB_TOKEN }}
        script: |
          const fs = require('fs');
          const marker = '<!-- code-review-report -->';
          let body;
          try {
            const raw = fs.readFileSync('code_review_report.md', 'utf8');
            const maxLen = 60000;
            const content = raw.length > maxLen
              ? raw.slice(0, maxLen) + '\n\n_Report truncated._'
              : raw;
            body = `${marker}\n${content}`;
          } catch (e) {
            body = `${marker}\n⚠️ Code review report could not be read: ${e.message}`;
          }
          // upsert comment (same pattern as Farley job)
          const { data: comments } = await github.rest.issues.listComments({
            owner: context.repo.owner,
            repo: context.repo.repo,
            issue_number: context.issue.number,
          });
          const existing = comments.find(c => c.body && c.body.includes(marker));
          if (existing) {
            await github.rest.issues.updateComment({
              owner: context.repo.owner, repo: context.repo.repo,
              comment_id: existing.id, body,
            });
          } else {
            await github.rest.issues.createComment({
              owner: context.repo.owner, repo: context.repo.repo,
              issue_number: context.issue.number, body,
            });
          }
```

---

## Static Tools vs LLM — Recommended Split

Don't use the LLM for things static tools already catch perfectly:

| Concern | Tool | LLM needed? |
|---|---|---|
| Style / formatting | `ruff format` | No |
| Lint rules | `ruff check` | No |
| Type errors | `mypy` | No |
| Known security patterns (e.g., SQL injection) | `bandit` / `semgrep` | No |
| **Semantic bugs** (wrong algorithm, API misuse) | — | **Yes** |
| **Architectural concerns** (coupling, leaky abstraction) | — | **Yes** |
| **Missing edge cases** | — | **Yes** |
| **Correctness of security logic** (e.g., path traversal) | — | **Yes** |

Run `ruff` and `bandit` as fast, cheap pre-checks.  The LLM job only fires
if those pass, keeping CI costs low.

---

## Gating Rules (recommended defaults)

| Condition | Action |
|---|---|
| Any unit has `severity = BLOCK` | CI fails (exit non-zero) |
| WARN count > 3 | Post comment, CI warns but passes |
| CQI drops > 1.0 vs previous run on same function | Flag for human review |

Start warn-only (same Phase 1 approach as Farley) and tighten after 2–4 PR
cycles of observing false positive rates.

---

## Build Order

1. **`scripts/diff_extractor.py`** — `git diff` → list of `{file, fn_name, body, diff_lines}` units.  Reuse the AST walker from `farley_score_evaluator.py`, just add line-range filtering.
2. **`scripts/code_review_evaluator.py`** — calls `diff_extractor`, sends each unit through `call_structured` with `CodeReviewBreakdown`, writes cassette.
3. **`scripts/code_review_compare.py`** — aggregate BLOCK/WARN/OK, emit `code_review_report.md`.
4. **Extend `farley_ci.yml`** — add the parallel `code-review` job above.
5. **Tune** — after observing real PRs, adjust BLOCK threshold and add a
   baseline cassette on `main` to track CQI trends.

---

## Secrets & Environment

Same as Farley CI:

- `GITHUB_TOKEN` — PR comment posting (already available)
- `NEBIUS_API_KEY` / `LITELLM_HOST` — if switching to a stronger model for
  authoritative nightly runs
- No new secrets needed for the Ollama-based PR job

---

## Token Spend & Cost Governance

CI fires on every PR, potentially multiple times per day.  Without explicit
spend controls, a single large PR or a misconfigured prompt can silently exhaust
a monthly API budget.  The same discipline applied to the debate agent's token
monitoring must apply here.

### What to track

Every cassette entry should include a `usage` sub-object, populated from the
provider's response (all major providers — Ollama, LiteLLM, Nebius,
OpenAI-compatible — return `usage.prompt_tokens`, `usage.completion_tokens`,
`usage.total_tokens` in the chat completion response):

```json
{
  "id": "b2f5ff47...",
  "file_path": "scripts/farley_score_evaluator.py",
  "function_name": "validate_path",
  "usage": {
    "prompt_tokens": 812,
    "completion_tokens": 340,
    "total_tokens": 1152,
    "estimated_cost_usd": 0.0006
  },
  "code_review_breakdown": { ... }
}
```

The `estimated_cost_usd` field is zero for local Ollama runs (compute-time
cost only) but populated for remote APIs using a simple per-model rate table
stored in config.

### Where to capture it

`call_structured` in `llm_adapter.py` already receives the full provider
response.  Add a thin extraction step:

```python
def extract_usage(response) -> dict:
    usage = getattr(response, "usage", None) or {}
    pt = getattr(usage, "prompt_tokens", 0)
    ct = getattr(usage, "completion_tokens", 0)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "estimated_cost_usd": estimate_cost(model_id, pt, ct),
    }
```

Return this alongside the structured breakdown so `ReplayManager` can persist
it in the cassette.  Replayed runs (cassette hits) record `total_tokens: 0`
since no LLM call was made — this makes the replay saving visible in reports.

### Per-PR budget gate

Before evaluating, estimate the total token budget for the PR:

```python
MAX_TOKENS_PER_PR = int(os.getenv("CR_MAX_TOKENS_PER_PR", "50000"))

def estimate_pr_tokens(units: list[dict]) -> int:
    # ~4 chars per token, system prompt ~400 tokens fixed overhead per unit
    return sum(len(u["body"]) // 4 + 400 for u in units)

if estimate_pr_tokens(units) > MAX_TOKENS_PER_PR:
    # Prioritise: evaluate only the top-N most-changed units
    units = sorted(units, key=lambda u: u["lines_changed"], reverse=True)[:MAX_UNITS]
```

`MAX_TOKENS_PER_PR` is an env var so it can be tightened without a code
change.  The default (50 k) covers ~40 medium functions at qwen3.5:2b context.

### Smart replay — the primary cost control

The cassette system already provides the most effective cost control: if a
function's body hash matches a previous cassette entry, **replay the stored
result** at zero token cost.  For a PR that only touches 3 functions in a
100-test codebase, only those 3 units incur LLM calls.

Track the cassette hit rate in the PR report:

```
Units evaluated:   3 (LLM call)
Units replayed:    37 (cassette hit, 0 tokens)
Tokens used:       3 410
Estimated cost:    $0.0021
```

### Run-tier differentiation

| Run type | Model | Max units | Budget gate |
|---|---|---|---|
| PR (warn-only) | `ollama/qwen3.5:2b` (local) | 20 most-changed | 50 k tokens |
| PR (soft gate) | `ollama/qwen3.5:2b` (local) | all changed | 80 k tokens |
| Nightly (authoritative) | Nebius / hosted 70B | full suite | separate budget |

Local Ollama runs have no per-token dollar cost, but still enforce a token
limit to bound **CI wall time** (token count ≈ runtime proxy for local models).
Remote API runs enforce both a token limit and a dollar budget.

### Aggregate telemetry

Append a `ci_spend` record to `artifacts/metrics/token_spend.jsonl` on every
run:

```json
{
  "timestamp": "2026-06-18T16:00:00Z",
  "run_id": "pr-51",
  "pr_number": 51,
  "model": "ollama/qwen3.5:2b",
  "units_evaluated": 3,
  "units_replayed": 37,
  "prompt_tokens": 2436,
  "completion_tokens": 974,
  "total_tokens": 3410,
  "estimated_cost_usd": 0.0,
  "wall_time_s": 42
}
```

Review this file weekly (same cadence as the Farley Index KPI review) to spot
cost trends before they become surprises.  A simple threshold alert:

```python
# In CI — fail the spend-check step if weekly total > budget
WEEKLY_BUDGET_USD = float(os.getenv("CR_WEEKLY_BUDGET_USD", "5.00"))
```

### Secret / rate-limit hygiene

- Store API keys in GitHub Secrets, never in cassettes or logs.
- Add `REDACT_SENSITIVE=true` guard in `call_structured` to strip prompt text
  from cassettes when `--include-prompts` is not explicitly set.
- Set `OLLAMA_MAX_LOADED_MODELS=1` in the service container to prevent
  accidental parallel model loading doubling memory/compute.

---

## Related Documents

- [`FARLEY_CI_PLAN.md`](./FARLEY_CI_PLAN.md) — test quality CI architecture
- [`CODE_TEST_QUALITY_PLAN.md`](./CODE_TEST_QUALITY_PLAN.md) — north star,
  CQI formula, rollout phases, KPIs
