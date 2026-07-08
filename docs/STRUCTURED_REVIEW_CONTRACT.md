# Structured Review Contract

Silver-One's code review workflow treats LLM output as an engineering contract.
The model may be local, small, or occasionally malformed, but the CI workflow
must still produce deterministic artifacts, telemetry, and gate decisions.

This document defines the cassette and report semantics used by the code review
evaluator, comparator, and unified report.

## Review Unit Shapes

Every entry loaded from a code review cassette is normalized into a
`UnitReviewArtifact`-shaped dictionary before comparison.

### Current Review Unit

Current successful review units contain:

```json
{
  "file_path": "scripts/example.py",
  "name": "changed_function",
  "class_name": null,
  "review": {
    "readability": {"score": 8, "rationale": "...", "suggestions": []},
    "maintainability": {"score": 8, "rationale": "...", "suggestions": []},
    "correctness": {"score": 8, "rationale": "...", "suggestions": []},
    "complexity": {"score": 8, "rationale": "...", "suggestions": []},
    "security": {"score": 8, "rationale": "...", "suggestions": []},
    "test_coverage": {"score": 8, "rationale": "...", "suggestions": []},
    "summary": "...",
    "severity": "OK",
    "findings": []
  },
  "validation": {
    "repaired": false,
    "normalized": false,
    "fields": []
  },
  "raw_response": "{...}",
  "structured_output": {
    "invalid_json_detected": false,
    "repair_attempts": 0,
    "repair_succeeded": false,
    "validation_retries": 0,
    "final_failure": false
  }
}
```

`validation` describes schema-level field repairs and normalizations after a
structured response has been parsed. `structured_output` describes parser,
repair, retry, and final-failure behavior before or during schema validation.

### Legacy Review Unit

Older cassettes may store the review payload directly inside `reviews[]`,
without the wrapper fields:

```json
{
  "readability": {"score": 8, "rationale": "..."},
  "maintainability": {"score": 8, "rationale": "..."},
  "correctness": {"score": 8, "rationale": "..."},
  "complexity": {"score": 8, "rationale": "..."},
  "security": {"score": 8, "rationale": "..."},
  "test_coverage": {"score": 8, "rationale": "..."},
  "summary": "...",
  "severity": "OK",
  "findings": []
}
```

The comparator wraps these entries with default metadata:

- `file_path`: `legacy-cassette`, unless the legacy entry already has metadata
- `name`: `legacy_review_N`, unless the legacy entry already has metadata
- `validation`: `{repaired: false, normalized: false, fields: []}`
- `raw_response`: JSON serialization of the legacy review payload

Legacy metadata keys such as `file_path`, `name`, and `class_name` are preserved
on the wrapper and excluded from the inner `review` payload.

### Recoverable Evaluation Failure

If structured output parsing, repair, and bounded retries still fail, the
evaluator records a recoverable failure unit instead of aborting the workflow:

```json
{
  "file_path": "scripts/example.py",
  "name": "changed_function",
  "review": null,
  "validation": {
    "repaired": false,
    "normalized": false,
    "fields": [
      {
        "field_name": "llm_response",
        "status": "INVALID",
        "raw_value": "...",
        "repaired_value": null,
        "repair_type": "structured_output",
        "repair_reason": "invalid_json: ..."
      }
    ]
  },
  "raw_response": "...",
  "structured_output": {
    "invalid_json_detected": true,
    "repair_attempts": 1,
    "repair_succeeded": false,
    "validation_retries": 2,
    "final_failure": true
  },
  "recoverable_failure": {
    "type": "structured_output",
    "schema_name": "CodeReviewBreakdown",
    "message": "Failed to validate LLM response ..."
  }
}
```

Recoverable failure units must keep their `recoverable_failure` field during
normalization. They are visible in reports, but they are not valid reviews and
therefore do not participate in CQI scoring.

## CQI Semantics

CQI is calculated only for units that contain a real `review` payload with all
required CQI dimensions:

- `readability`
- `maintainability`
- `correctness`
- `complexity`
- `security`
- `test_coverage`

A malformed review payload with missing or invalid CQI scores is an invalid CQI
unit and fails the code review gate. This means the model produced a review-like
object that does not satisfy the scoring contract.

A recoverable evaluation failure is different: no valid review was produced
after repair and retry attempts. The unified report shows it as
`N/A (recoverable failure)` and excludes it from CQI averages. The failure is
measured through structured-output telemetry rather than CQI.

## Validation Summary Semantics

Each evaluated unit contributes to exactly one terminal validation state:

- `valid_units`: no field repair, normalization, or invalid field status
- `repaired_units`: at least one field was repaired
- `normalized_units`: at least one field was normalized and no repairs occurred
- `invalid_units`: an invalid validation field or final structured-output failure

Structured-output counters are recorded independently:

- `invalid_json_detected`
- `repair_attempts`
- `repair_successes`
- `validation_retries`
- `final_failures`

These counters are used to measure harness reliability over time. They should
not be conflated with reviewer quality metrics such as CQI or severity.

## Reporting Rules

The unified report follows these rules:

- Successful current and legacy review units appear with CQI and severity.
- Invalid CQI review units appear as `INVALID (...)` and fail the CQI metric.
- Recoverable evaluation failures appear as `N/A (recoverable failure)`.
- CQI averages are computed from valid CQI review units only.
- Recoverable failures are counted in validation telemetry and excluded from CQI.

This separation lets Silver-One answer two different questions:

- Did the reviewer produce useful engineering feedback?
- Did the structured-output harness keep the workflow reliable?
