# Review Feedback Patterns

This document records recurring feedback patterns from Gemini Code Assist,
Farley, SonarQube, and our own PR iterations. The goal is not to obey every
automated suggestion. The goal is to identify review signals that repeatedly
point to real workflow risks, then turn them into engineering habits.

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

Good local examples:

- `build_validation_summary(...)`
- `_process_structured_output_telemetry(...)`
- `_recoverable_failure_cqi_label(...)`
- `build_review_coverage(...)`

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
