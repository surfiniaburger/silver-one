# Code & Test Quality North Star

Purpose
- Provide a clear, actionable "north star" for maintaining high-quality code and high-quality tests as the codebase grows.
- Ensure every PR improves or at least does not degrade code quality and test quality.

Goals
- Keep code clean, well-typed, and maintainable.
- Ensure tests are meaningful, maintainable, and fast (use Farley Score for test quality).
- Make CI the single source of truth for gate checks and diagnostics.

Principles
- Small, focused changes: PRs should be atomic and limited in scope.
- Measure and act: define metrics for code and tests and track them.
- Fail fast, give guidance: CI should surface problems quickly and provide remediation steps.
- Incremental rollout: warn-first then gate-only when the team is comfortable.

What we check (concrete)
- Code checks
  - Formatting: `black` / `ruff` enforcement.
  - Linting: `ruff`, `flake8` rules (configurable).
  - Types: `mypy` / pydantic runtime checks for data models.
  - Static/Security: `bandit`/`semgrep` for security patterns.
  - Complexity: cyclomatic complexity threshold per file (e.g., <= 12).
  - Documentation: public functions/classes require docstrings.
- Test checks
  - Unit tests pass: `pytest` on targeted tests.
  - Test quality: Farley Score per-test (automated LLM-based structured review).
  - Flakiness signals: detect `time.sleep`, external I/O, randomness without seeding.
  - Tautology detection: mock-only tests or trivial assertions.
  - Performance: flag tests slower than threshold (e.g., > 2s for unit tests).

Quality Indices
- Farley Index (tests): weighted average of the 8 properties (U, M, R, A, N, G, F, T).
  - Use Understandable and Maintainable with higher weight (1.5x) per Farley guidance.
- Code Quality Index (CQI) (suggested): combine the following normalized metrics into a 0-10 index:
  - Readability (naming, docstrings) — weight 1.5
  - Maintainability (modularization, low coupling) — weight 1.5
  - Complexity (cyclomatic) — weight 1.0
  - Test coverage for changed code — weight 1.0
  - Security findings (negated) — weight 0.5

CI Workflow & Gates (recommended rollout)
- Phase 0 (local developer)
  - Run `uv run pytest tests` and `uv run ruff check src tests` locally.
  - Run `python3 scripts/farley_score_evaluator.py . --mode record --cassette artifacts/cassettes/farley_local.json --model "litellm/qwen3.5:2b"` to see test quality locally.
- Phase 1 (PRs, warn-only)
  - CI runs targeted unit tests (`tests/`) and static checks. If failures → FAIL.
  - CI runs Farley evaluator (light local model) but does not block — posts `report.md` and artifacts.
- Phase 2 (soft gates)
  - After several weeks of observing false positives, enable warning thresholds that fail PRs only on severe regressions.
  - Example fail thresholds (tunable): suite Farley delta <= -0.25, U or M drop >= -0.5, at least two tests drop by >=2 and those drops represent >5% of evaluated tests.
- Phase 3 (strict gates + nightly authoritative)
  - Nightly full-suite run using Nebius or self-hosted GPU model produces authoritative baseline.
  - CI blocks merges if soft gate thresholds are exceeded.

PR Checklist (developer responsibilities)
- [ ] Unit tests included for new logic and pass locally.
- [ ] Linted: `uv run ruff check` passes or auto-fixed.
- [ ] Typing: `mypy` checks added types for public APIs.
- [ ] Farley-aware: tests added or changed have Farley-friendly patterns (clear names, single behavior, minimal mocking).
- [ ] Provide short reasoning in PR description about test strategy for complex cases.

Reviewer Checklist (code reviewer responsibilities)
- Verify tests are granular and meaningful; ask for split if multiple behaviors are tested.
- Check for mocking tautologies and prefer behavioral assertions.
- Confirm public APIs have docstrings and types.
- If Farley report flags regressions, confirm whether they are acceptable or require changes.

Automation & Artifacts
- Cassettes: `artifacts/cassettes/farley_score-<run_id>.json` (store structured breakdown per test).
- Reports: `report.md` produced by `scripts/farley_compare.py` on PRs and nightly.
- Metrics: `artifacts/metrics/` contains aggregated trends and plots.

Implementation Tasks (initial)
1. Ensure `ruff`, `black`, `mypy`, and `pytest` are in CI (we have pytest via `uv`).
2. Keep `scripts/farley_score_evaluator.py` and `scripts/llm_adapter.py` in repo and maintain presets.
3. Add `scripts/farley_compare.py` to produce `report.md` and integrate into CI (done).
4. Start PRs with warn-only Farley runs; collect feedback and tune thresholds.
5. Implement nightly authoritative runs (Nebius/self-hosted) and update baseline cassette.

Failure Modes & Mitigations
- LLM false positives: keep Farley warn-only initially and require human review; use structured suggestions rather than hard blocks.
- CI cost: evaluate only changed tests in PRs; run full-suite nightly.
- Secrets & privacy: default to structured outputs only; `--include-prompts` opt-in for debug runs only.

Signals to monitor (KPI)
- Median Farley Index per PR.
- % of PRs with any Farley regression > 2 points.
- Mean CQI over time.
- Test runtime distribution and number of flaky test reruns.

Owner & Governance
- Responsibility: Engineering team owns thresholds and remediation policy.
- Review cadence: weekly review of Farley trends; monthly threshold tuning.

Next steps (immediately actionable)
- Confirm rollout policy: `warn-only` or `block-on-fail` for PRs. (Recommended: `warn-only` initially.)
- Push current branch and observe CI. Tweak workflow guard if you want model calls skipped on PRs.
- After 2-4 PR cycles, review `artifacts/metrics/metrics_comparison.csv` and tune thresholds.


If you want, I will also:
- Add a small `CODE_QUALITY.md` with commands and example outputs.
- Add a GitHub Action step to post `report.md` as a PR comment when Farley warns/fails.
