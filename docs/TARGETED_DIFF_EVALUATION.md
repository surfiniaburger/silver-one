# Proposal: Targeted Diff-Only Evaluation & Virtual Suite Comparison

This document proposes a critical shift in how the **Farley Test Evaluator** measures code quality, aligning it with the scope of the **Code Reviewer** to save runner time, reduce token spend, and produce highly accurate comparative metrics.

---

## 1. The Core Architectural Insight

Currently, we have a misalignment in the scope of our two CI feedback loops:
* **Code Reviewer**: Evaluates *only the diff* (new/modified code units relative to the base branch). It measures **PR-local quality**.
* **Farley Test Evaluator**: Evaluates *the entire test directory* on every push. It measures **repository-wide quality**.

### The Problem with Repository-Wide Test Evaluation in PRs
1. **Token & Time Waste**: Running LLM evaluations on 20+ unchanged test cases over and over on every PR push is extremely wasteful.
2. **Metric Dilution**: If a developer writes a flawless new test, its score is diluted by all the historical tests in the suite. The resulting "PR avg Farley Index" doesn't reflect the work done in the PR.
3. **Incomparable Averages**: Comparing a PR-local average of 2 changed tests directly against a repo-wide baseline average of 20 tests is mathematically invalid because they are drawn from different sets.

---

## 2. Proposed Solution: Targeted Diff-Only Test Evaluation

Instead of modifying prompts or parallelizing VM jobs to scale monolithic runs, we align both evaluators to **evaluate changed code only**.

```
Current PR Push:
  All 20+ tests in tests/ are parsed and evaluated
  ↓
  PR cassette contains all tests, adding ~25,000+ tokens of overhead per CI run

Proposed PR Push:
  Farley Evaluator is called with `--base origin/main`
  ↓
  AST parser extracts only the specific test functions that intersect with the git diff
  ↓
  PR cassette contains only the evaluated changed/new tests (~1,000 - 3,000 tokens)
```

### Steps to Implement Diff Filtering
1. **Add `--base` Option**: Implement a `--base` argument in `farley_score_evaluator.py` (matching `code_review_evaluator.py`).
2. **Changed Line Intersection**:
   * Use the existing `diff_extractor.get_changed_lines(base_ref, project_root)` to get the changed line numbers for each file.
   * Modify the test parser to record `start_line` and `end_line` for each test case.
   * Only evaluate test functions whose line ranges overlap with the modified lines.

---

## 3. How to Compare Averages: The "Virtual PR Suite"

If the PR cassette only contains scores for the 2 modified tests, and the baseline cassette contains all 20 tests, a direct comparison of their raw averages will break. 

To resolve this, `farley_compare.py` will construct a **Virtual PR Suite** for comparison:

```
                  Baseline Suite (from main)
                ┌───────────────────────────┐
                │ Test A (8.0), Test B (6.5)│
                │ Test C (7.0), Test D (5.5)│
                └─────────────┬─────────────┘
                              │
                    PR Evaluated Tests (only changed)
                ┌─────────────▼─────────────┐
                │ Test B (Revised: 8.5)     │
                │ Test E (New: 9.0)         │
                └─────────────┬─────────────┘
                              │
                              ▼
                      Virtual PR Suite
                ┌───────────────────────────┐
                │ Test A (8.0)  - Unchanged │
                │ Test B (8.5)  - OVERWRITTEN│
                │ Test C (7.0)  - Unchanged │
                │ Test D (5.5)  - Unchanged │
                │ Test E (9.0)  - NEW       │
                └───────────────────────────┘
```

### Calculation Flow
1. **Load Baseline**: Load the latest baseline cassette (`farley_score-main-latest.json`) representing the full repository test suite.
2. **Load PR Cassette**: Load the PR-local cassette containing only the test cases evaluated in the PR.
3. **Merge**: Overlay the PR cassette tests onto the baseline tests by test ID.
4. **Compare**:
   * **Baseline Avg**: Average score of the original baseline suite.
   * **PR Avg**: Average score of the merged Virtual PR Suite.
   * **Delta**: `PR Avg - Baseline Avg`.
5. **Verdict**: Apply gating rules to this delta. If a test case score dropped, or the overall virtual suite average decreased below the threshold, fail the gate.

This gives a mathematically sound suite-wide comparison while evaluating only changed tests.

---

## 4. Enabling Specialist Reviewers

By switching to a targeted diff-only strategy, we slash evaluation costs and execution latency. For example:
* **Old Monolithic Run**: 20 tests * 1 system prompt = 20 LLM calls (~25k tokens).
* **New Scoped Run**: 2 tests * 1 system prompt = 2 LLM calls (~2.5k tokens).

This opens up a massive token and time budget to run **specialist reviewers** (such as specialized Compatibility, Security, and Correctness checkers) on just the changed files. 

For instance, we can run:
1. `general_reviewer` (Readability, Maintainability, Granularity)
2. `compatibility_reviewer` (Dedicated checks for OS, pathing, and runner environments)
3. `correctness_reviewer` (Checks for logic errors, edge cases, and assertion robustness)

Since these only run on the 1 or 2 changed tests, the total calls are kept low, yielding high-precision feedback without bloat.


PHASE 1
Fix evaluation methodology

- Diff-only review
- Diff-only Farley
- Virtual Suite
- Compatibility metric

PHASE 2
Validate quality improvements

- Prompt experiments
- Benchmark runs
- Measure precision/recall
- Compare reviewers

PHASE 3
Introduce specialization

- Security reviewer
- Compatibility reviewer
- Correctness reviewer

