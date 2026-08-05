# Review Feedback Patterns

This document records recurring feedback patterns from Gemini Code Assist,
Farley, SonarQube, Observability Engineering (2nd Edition, Ch. 2), and our own PR
iterations. The goal is not to obey every automated suggestion. The goal is to
identify review signals that repeatedly point to real workflow risks, then turn
them into engineering habits.

Use this document as a checklist when changing evaluator, cassette, report, and
structured-output code.

## How To Use This

Treat each review comment as a hypothesis.

Accept or adapt the comment when it:

- protects a real input boundary,
- preserves an existing compatibility contract,
- improves structured-output reliability,
- improves telemetry or recoverability,
- reduces proven fragility in tests or CI,
- simplifies code that Sonar or maintainers repeatedly struggle with.

Reject or defer the comment when it:

- is generic style advice with no clear risk reduction,
- contradicts the purpose of the test,
- assumes external I/O or network calls that are not present,
- optimizes an implementation detail while weakening the domain behavior,
- creates churn in unrelated modules.

## Defensive Input Handling

Automated reviewers consistently flag unchecked assumptions around `None`,
missing keys, and malformed JSON-derived values. These comments are usually
worth taking seriously because Silver-One consumes persisted cassettes, LLM
responses, CI artifacts, and legacy payloads.

Common examples:

- `units is None`
- missing `unit["code"]`
- `lines_changed=None`
- `lines_changed="10"` or another JSON-derived non-integer value
- missing cassette `__metadata__`
- malformed `reviews`
- malformed nested telemetry objects such as `details="..."`,
  `structured_output=null`, or `provider_error="..."`
- legacy review payloads without the current wrapper shape
- optional validation fields set to `null`

Preferred pattern:

- Return an empty result for absent collections when absence is harmless.
- Validate object shape with `isinstance(value, dict)` or
  `isinstance(value, list)` before reading nested fields.
- Use explicit defaults for JSON-derived scalar fields, especially when `None`
  is a plausible serialized value.
- Coerce JSON-derived numeric fields before comparing, sorting, or using them
  in arithmetic. Fall back to a safe default when parsing fails.
- Raise a clear `ValueError` when malformed input means the caller has violated
  the contract.

Example:

```python
def _coerce_lines_changed(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
```

When reviewers include a `References` note beneath a comment, treat it as a
candidate general rule. If the rule repeats across PRs, capture it here rather
than only fixing the local instance.

## Dictionary Shape Validation

Gemini often points out that `dict.get(...)` alone is not enough. This is valid
when the input comes from cassettes, LLM output, GitHub artifacts, or legacy
schema versions.

Preferred pattern:

- Check container type before reading from it.
- Check list element type before processing it.
- Check nested dictionary type at each level before calling `.get(...)` on the
  nested value.
- Keep validation close to the boundary where data enters the workflow.
- Use small helpers when the same shape check appears in several places.

Example:

```python
def get_code_review_coverage(cassette: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cassette, dict):
        return {}
    metadata = cassette.get("__metadata__")
    if not isinstance(metadata, dict):
        return {}
    summary = metadata.get("code_review_usage_summary")
    if not isinstance(summary, dict):
        return {}
    coverage = summary.get("review_coverage")
    return coverage if isinstance(coverage, dict) else {}
```

For nested telemetry, do not rely on `value = parent.get("child") or {}` when
the child may be a truthy non-dictionary such as a string or list. Normalize the
shape before reading deeper:

```python
details = summary.get("details")
if not isinstance(details, dict):
    details = {}
structured = details.get("structured_output")
if not isinstance(structured, dict):
    structured = {}
```

## Compatibility Contracts

Reviewers correctly flagged the case where `filter_units_by_budget` was changed
from greedy selection to "first planned batch" behavior while still being named
and tested as a compatibility wrapper.

Rule:

- If a function exists for legacy callers, preserve its old semantics.
- If old semantics are no longer wanted, rename or remove the function
  explicitly and update callers in the same PR.
- Do not call something a compatibility wrapper unless its behavior is actually
  compatible.

Preferred pattern:

- New behavior gets a new function name.
- Old function keeps old behavior and delegates only when delegation preserves
  semantics exactly.

Example:

```python
def batch_units_by_budget(...):
    """New behavior: split all units into review batches."""


def filter_units_by_budget(...):
    """Legacy behavior: greedily select one bounded subset."""
```

## Prompt And Schema Reliability

The code-review prompt now follows a more explicit contract style inspired by
the debate scenario prompts:

- `<role>`
- `<task>`
- `<context>`
- `<dimensions>`
- `<scoring_rules>`
- `<severity_rules>`
- `<finding_rules>`
- `<constraints>`
- `<output_format>`
- `<examples>`

This pattern matters because smaller local models need more structure than a
frontier model. The prompt should make the schema contract hard to miss.

Preferred pattern:

- Tell the model every required top-level field.
- Tell the model every required nested field.
- Say that scores must be JSON numbers, not strings, percentages, fractions, or
  labels.
- Require non-empty summaries.
- Require `findings: []` when there are no findings.
- Ask structured findings to include a reusable `reference_principle` when a
  finding represents a general engineering pattern, not only a one-off local
  defect.
- Include complete JSON examples for normal and edge cases.
- Test prompt examples by parsing them and validating them against the schema.

Avoid:

- substring-only tests for examples that are meant to be schema contracts,
- prompt examples that are almost JSON but not valid JSON,
- adding hidden reasoning fields unless the schema actually supports them.
- making new finding metadata mandatory for legacy cassettes unless the loader
  supplies a compatibility default.

## Test Design

Farley and Gemini repeatedly prefer tests that express the business rule, not
only the output string. This is especially important for report-rendering tests,
where string assertions are necessary but can become opaque.

Preferred pattern:

- Use domain names in Arrange variables.
- Name the rule in the test name or docstring.
- Assert the behavior that matters, not incidental formatting alone.
- Keep report string assertions precise when the report itself is the contract.
- Avoid tests that depend on incidental constants such as current prompt length.

Example:

```python
def test_write_unified_report_displays_baseline_state(tmp_path):
    """A required missing Farley baseline must fail the gate and explain the missing input."""
    expected_reason = "Farley baseline is required but was not found."
    missing_required_baseline = farley_report_data(
        baseline_exists=False,
        baseline_state="BASELINE_MISSING",
        baseline_reason=expected_reason,
    )
```

## Unified Report Test Fixtures

The unified report is a public markdown contract, so exact string assertions
are appropriate when they protect lane names, metric wording, table columns, or
fallback labels. The brittle part is not the presence of string assertions; it
is repeating setup and expected phrases without naming the report behavior under
test.

Preferred pattern:

- Keep one local helper, such as `render_unified_report(...)`, for constructing
  report fixtures with explicit default cassettes.
- Give that helper a docstring that states it renders the public markdown
  contract and returns the file content for assertions.
- Name important expected report fragments once in the test when a phrase is
  reused or represents a domain rule.
- Use exact table-row assertions for public lane output.
- Use focused substring assertions for detailed sections where the surrounding
  markdown is not the behavior under test.
- Add malformed-input cases to the helper defaults instead of creating separate
  ad hoc report writers.

Avoid:

- mocking `write_unified_report(...)` in tests whose purpose is to verify the
  rendered report,
- extracting all report strings into global constants before they represent a
  stable public contract,
- relying on incidental formatting outside the lane, table, or fallback being
  tested.

## Avoid Incidental Constants

Prompt size, token estimates, and fixture text lengths are easy to bake into
tests by accident. Gemini correctly flagged that tests should not fail merely
because prompt wording changed.

Preferred pattern:

- Derive expected budgets from `estimate_unit_tokens(...)`.
- Use `monkeypatch` for prompt-size-sensitive tests when the prompt content is
  irrelevant to the behavior under test.
- Assert relative behavior rather than exact token constants unless the exact
  constant is the public contract.

## Regex And Parsing

Sonar flagged a reluctant quantifier used to extract prompt examples. The
broader rule is useful: if the content has explicit delimiters or structure,
prefer explicit parsing over clever regex.

Preferred pattern:

- Use explicit start and end tags for prompt blocks.
- Use `json.loads(...)` for JSON examples.
- Use schema validation for schema examples.
- Use structured parsers when the codebase or standard library provides one.

Avoid:

- broad `.*?` regexes over multiline content,
- parsing JSON-like examples with string slicing only,
- tests that only check that a marker exists.

## Complexity And Helper Extraction

Sonar repeatedly pushes on cognitive complexity. Gemini often makes adjacent
comments about nested logic, mixed responsibilities, and hard-to-test branches.

Preferred pattern:

- Extract small helpers for validation, classification, formatting, and
  telemetry aggregation.
- Keep orchestration functions readable by delegating mechanics to named
  helpers.
- Avoid combining parsing, policy, report rendering, and telemetry aggregation
  in one function.
- Avoid nested loops and deep conditional hierarchies (e.g. loops inside loops containing multiple `if` checks). Extract nested loops or nested-object extraction logic into flat helpers to keep cognitive complexity strictly below the 15 threshold allowed by SonarQube (e.g. `_extract_unit_findings(...)` to extract findings from review units).

Good local examples:

- `build_validation_summary(...)`
- `_process_structured_output_telemetry(...)`
- `_recoverable_failure_cqi_label(...)`
- `build_review_coverage(...)`

## Shared Utilities vs. Duplicate Code

Gemini and Sonar flag duplicate helper functions (e.g., `_coerce_int`, `_coerce_float`, or parsing logic) spread across multiple modules. When helpers perform the same type coercion, parsing, or normalization, centralize them to guarantee consistent execution and reduce maintenance overhead.

Preferred pattern:

- Move general-purpose coercion helpers (like converting JSON raw strings/objects to float or int) into shared utility modules such as `scripts/telemetry_utils.py` (e.g. `coerce_int`, `coerce_float`).
- Keep schema-specific parsing and mapping logic (like confidence string/float literals parsing) in the schema boundary module (`scripts/finding_schema.py`) and import/reuse those functions (e.g. `_parse_numeric_confidence`, `_parse_string_confidence`) during report compilation rather than re-implementing them.
- Ensure fallback values for invalid structures are identical across validation and reporting.

## O(1) And Repeated Lookup Feedback

Gemini sometimes flags code that repeatedly scans lists or performs nested
lookups when a set or dictionary would be clearer and faster. This is not always
urgent, but it is often right when the code is on a hot path or used over many
units.

Preferred pattern:

- Build a set for repeated membership checks.
- Build a dictionary index for repeated lookup by ID, file path, or test name.
- Compute expensive derived values once and reuse them.
- Avoid nested loops when the relationship can be represented as a map.

Rule of thumb:

- If the collection is tiny and the code is clearer as a loop, keep the loop.
- If the lookup is repeated or part of CI-scale evaluation, build the index.

## Policy Versus Mechanism

The CQI work exposed a recurring design issue: functions should not both
calculate metrics and decide pipeline policy unless that is their explicit job.

Preferred pattern:

- Calculators return structured results.
- Orchestration decides pass/fail.
- Reports render the structured result and reason.

Example:

```python
class CQIResult(BaseModel):
    valid: bool
    value: Optional[float] = None
    reason: Optional[str] = None
    error_code: Optional[str] = None
```

This makes corrupted or missing data visible instead of silently falling back to
a default score.

## Telemetry And Recoverability

Automated feedback and our own CI runs both point to the same reliability need:
counts are useful, but causes are better.

Preferred pattern:

- Every recoverable failure should have a `type`.
- Every final failure should have a human-readable `message`.
- Structured-output failures should record repair attempts, retries, and final
  failure status.
- Provider failures should be distinct from schema failures.
- Review coverage should state extracted, reviewed, skipped, and batch count.

Examples of useful telemetry:

- `structured_output.repair_attempts`
- `structured_output.validation_retries`
- `structured_output.final_failures`
- `provider_error.failures`
- `review_coverage.total_extracted_units`
- `review_coverage.reviewed_units`
- `review_coverage.skipped_units`
- `review_coverage.batch_count`

## Boundary Decisions

At every workflow boundary, decide which behavior is intended:

- validate,
- normalize,
- repair,
- fail recoverably,
- reject with a clear exception.

Boundary examples:

- LLM response: repair or retry, then fail recoverably.
- Cassette JSON: normalize legacy payloads when safe.
- Compatibility wrapper input: preserve legacy semantics.
- Internal programming error: raise a clear exception.
- Report rendering: degrade gracefully and explain missing data.

## Floating Point Comparisons

Sonar flags direct equality checks on floating point numbers (e.g., `val == 0.0` or `val != 0.0`) due to precision limits in IEEE 754 representations.

Preferred pattern:

- Use `math.isclose` with a defined tolerance when checking for float closeness/equality.

Example:

```python
import math

if not math.isclose(den, 0.0, abs_tol=1e-9):
    val = num / den
```
## Inconsistent Classification and Defaulting

When aggregating metrics or calculating distributions into discrete categories (e.g., confidence distributions, severity counts), fallback logic in loops can silently misclassify invalid or "N/A" values.

Rule:

- Do not use a broad `else` block to assign unrecognized, fallback, or non-applicable values to a valid classification bucket (like counting `"N/A"` or unknown confidence values as `"MEDIUM"`).
- Explicitly check membership in the target categories and exclude or handle unrecognized/fallback values specifically.
- Ensure that the classification logic in metrics aggregation matches the formatting logic in detailed tables/views.

Example:

```python
# Avoid:
for finding in findings:
    conf = finding.get("confidence")
    formatted = _format_confidence(conf)
    if formatted in counts:
        counts[formatted] += 1
    else:
        counts["MEDIUM"] += 1  # Silently counts "N/A", None, or invalid values as "MEDIUM"

# Preferred:
for finding in findings:
    conf = finding.get("confidence")
    formatted = _format_confidence(conf)
    if formatted in counts:
        counts[formatted] += 1
```

## Pinning and Locking CI/CD Dependencies

Floating versions in GitHub Actions workflows (both for actions themselves and for packages installed within them) introduce supply-chain risks and make builds non-deterministic.

Preferred pattern:
- Pin third-party GitHub Actions to a full 40-character commit SHA instead of major version tags (e.g., use `@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86` instead of `@v5`). Always append a comment indicating the human-readable version (e.g., `# v5.4.2`) for readability.
- Explicitly pin the tool versions (such as `version: "0.7.2"` under `astral-sh/setup-uv`) rather than allowing the action to dynamically pull the latest release.
- Enforce lockfile compliance in run steps using the `--locked` (or `--frozen`) flags.

## Restricting Arbitrary Build Execution

When installing dependencies or executing packages, omitting binary-only constraints can cause the package manager to download source distributions and run arbitrary build scripts (`setup.py` / PEP 517 backends) on the host runner.

Preferred pattern:
- In CI workflows, sync dependencies with `uv sync --no-install-project --no-build --locked` to ensure only pre-compiled wheels are fetched.
- Execute project scripts using `uv run --no-sync --no-build --locked <command>` (and set `PYTHONPATH: src` at the job environment level) so no resolution or building occurs during the run phase.

## Bumpy Road Code Smell (Refactoring Sequential Conditionals)

Functions containing multiple sequential blocks of nested conditional checks ("bumpy roads") place a heavy tax on working memory and complicate state tracking.

Preferred pattern:
- Identify distinct logical phases (such as validating compatibility, resolving status, setting score defaults, or list coercion) and extract them into small, single-purpose helper functions (each with a Cyclomatic Complexity <= 2).
- Keep the main orchestrator function flat (CC = 1) by delegating input parsing to these extracted helpers.

## Test Suite Duplication and Complexity

Inline duplication of deep nested mock-data dictionaries (such as cassette outputs or validation reports) increases test suite length beyond quality limits and hurts readability.

Preferred pattern:
- Refactor test files to extract shared mock-data structures into compact builder helper functions (e.g., `_make_validation_field`) or pytest fixtures.
- Keep test cases under the 50-line limit to maintain high comprehensibility.

## Atomic Persistence and Manifest Validation

As established in *Observability Engineering (Ch. 2)*, persistence problems (corrupted datasets, invalid model weights, broken checkpoints) are permanent. Application logic errors can be fixed in post or rolled back, but writing partial or corrupted bits out to disk destroys pipeline integrity.

Preferred pattern:

- **Atomic File Replacement:** Write serialized artifacts to a PID-tagged temporary file (e.g., `target.tmp.[pid]`) and perform an atomic `os.replace` to swap the file into place.
- **Explicit Disk Flushing:** Flush file buffers (`f.flush()`) and invoke `os.fsync(f.fileno())` when appending to line-delimited files (like `.jsonl` attempt logs) to prevent truncated line writes upon unexpected process exit.
- **Artifact Manifest Checksums:** Generate a companion `model_manifest.json` recording SHA-256 digests, file sizes, feature dimensions, and timestamps for persisted binaries. Validate these checksums before loading model weights into memory.

Example:

```python
def _save_artifact_atomic(obj: Any, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    try:
        if joblib is not None:
            joblib.dump(obj, tmp_path)
        else:
            with tmp_path.open("wb") as f:
                pickle.dump(obj, f)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
```

## When Not To Apply Feedback

Some automated comments are useful but generic. Others are simply wrong for the
local context.

Examples we should treat carefully:

- Claims that `tmp_path` harms repeatability. In pytest, `tmp_path` is normally
  the repeatable isolation mechanism.
- Suggestions to mock the report renderer in tests whose purpose is to verify
  report output.
- Generic advice to add type hints or comments where the local code is already
  obvious.
- Suggestions that would preserve style while weakening domain coverage.

The standard is not "Did the bot suggest it?" The standard is "Does this reduce
a real risk in Silver-One's CI feedback loop?"

## Current Checklist

Before merging evaluator, report, or structured-output changes, ask:

- Can JSON-derived `None` values break sorting, scoring, or formatting?
- Are legacy wrappers still behavior-compatible?
- Are prompt examples valid JSON and schema-valid?
- Do tests depend on incidental prompt length or fixture text length?
- Does every recoverable failure include a type and message?
- Does the report say how much of the PR was actually reviewed?
- Are repeated lookups indexed where scale matters?
- Is policy separate from calculation?
- Is any regex parsing something that should be delimiter-based or schema-based?
- Are floating point equality checks avoided by using `math.isclose`?
- Are duplicate utility helpers avoided by centralizing them into shared modules (like `telemetry_utils`) or importing them from the source schema?
- Do report-rendering tests document the helper contract and reserve exact
  string assertions for public markdown behavior?
- Is Cognitive Complexity kept below 15 by extracting nested loops or nested-object extraction steps into clean helper functions?
- Do classification and aggregation counts exclude unrecognized, None, or fallback "N/A" values instead of silently counting them in a default bucket?
- Are third-party GitHub Actions pinned to a full 40-character commit SHA rather than a mutable version tag?
- Are CI package managers configured with `--no-build` and `--locked` (or `--no-sync`) to prevent execution of unverified setup scripts?
- Are sequential conditional blocks ("bumpy roads") refactored into flat helper functions?
- Are test suite functions kept under 50 lines by extracting helper mock-data builders?
- Are persistent artifacts (model weights, JSONL corpora, manifests) written atomically using PID-tagged temporary files (`.tmp.[pid]` -> `os.replace`) and validated against SHA-256 manifest checksums before loading?


