# Engineering Feedback Platform Migration

## Phase 2A — Introduce Structured Findings (Minimal Evolution)

## Purpose

This document narrows the scope of the Engineering Feedback Platform migration.

The previous implementation proposal correctly identified the long-term direction but attempted to redesign too many parts of the review pipeline simultaneously.

Silver-One follows an incremental engineering philosophy:

> Measure → Validate → Evolve

Rather than:

> Replace → Hope

This phase focuses only on introducing a structured finding model while preserving existing review behavior.

---

# Primary Goal

Introduce a reusable **Engineering Finding** model without disrupting the current evaluation pipeline.

This phase is **not** about redesigning CQI, changing gating logic, or replacing existing review dimensions.

Those decisions should be informed by benchmark data rather than assumptions.

---

# What We Are NOT Changing

The following components remain unchanged.

* Existing CQI calculation
* Existing review dimensions

  * Readability
  * Maintainability
  * Correctness
  * Complexity
  * Security
  * Test Coverage
* Existing PASS/WARN/BLOCK workflow
* Existing benchmark pipeline
* Existing comparison metrics

This ensures complete backward compatibility.

---

# Target 1 — Introduce EngineeringFinding

## Objective

Create a reusable schema representing a single engineering finding.

The schema should represent structured engineering reasoning, **not** replace existing review outputs.

### Definition of Done

* New `EngineeringFinding` model exists.
* Model is independent of reviewers.
* Model is reusable by future specialist reviewers.
* No existing code paths are broken.

---

# Target 2 — Preserve Existing Review Schema

## Objective

Do **not** replace the current review payload.

Instead, extend it.

Current direction:

```text
Review
```

Desired direction:

```text
Review
├── Existing dimension scores
├── Existing summary
└── Findings
```

Example:

```yaml
review:
  readability:
  maintainability:
  correctness:
  complexity:
  security:
  test_coverage:

summary:

findings:
```

The existing review system remains the source of truth during this phase.

### Definition of Done

* Existing review payload continues to work.
* Existing tests continue to pass.
* Existing reports remain valid.
* Findings become an additional data structure.

---

# Target 3 — Do NOT Redesign CQI

## Objective

The proposal attempted to derive CQI directly from finding counts.

This should **not** happen yet.

Reason:

No benchmark data currently supports weighting findings using arbitrary constants.

For example:

```text
INFO  = -0.5

WARN  = -1.5

BLOCK = -3.5
```

These values are assumptions.

Silver-One should avoid encoding assumptions into quality metrics without evidence.

### Definition of Done

* CQI calculation remains unchanged.
* Existing benchmark comparisons remain valid.
* Findings do not influence scoring yet.

---

# Target 4 — Findings Complement Existing Reviews

Findings should explain engineering decisions.

They should **not** become the scoring mechanism yet.

Desired relationship:

```text
Engineering Findings

↓

Support

↓

Dimension Scores

↓

CQI

↓

Overall Verdict
```

Not:

```text
Engineering Findings

↓

CQI
```

### Definition of Done

* Findings enrich reports.
* Existing quality metrics remain untouched.

---

# Target 5 — Add Engineering Consequence

The proposal contains:

* rationale
* recommendation

It is missing the most valuable field.

Every finding should explicitly answer:

> "What happens if this issue is ignored?"

Add a field similar to:

```yaml
engineering_consequence:
```

Example:

```yaml
finding:
Removing default parameter

engineering_consequence:
Existing callers relying on the default value will raise TypeError.

recommended_action:
Retain the default value or introduce a migration path.
```

This captures the engineering reasoning that differentiates actionable feedback from generic review comments.

### Definition of Done

Every finding explains:

* why the issue exists
* what engineering consequence follows
* how to resolve it

---

# Target 6 — Make Evidence Future-Proof

Current proposal tightly couples evidence to source code.

Example:

```yaml
file
function
start_line
end_line
```

Silver-One already evaluates more than source code.

Future findings may originate from:

* compatibility analysis
* Farley evaluation
* workflow evaluation
* repository metrics
* architecture analysis
* security scans

The evidence model should therefore remain generic enough to evolve.

Avoid locking the schema to only source code functions.

### Definition of Done

Evidence model is extensible without future breaking schema changes.

---

# Target 7 — Keep Confidence Numeric

Current proposal:

```text
LOW

MEDIUM

HIGH
```

Preferred direction:

```text
0.0 → 1.0
```

or

```text
0 → 100
```

Reason:

Numeric confidence enables future work:

* calibration
* agreement analysis
* confidence distributions
* reviewer benchmarking
* confidence-weighted aggregation

### Definition of Done

Confidence is machine-computable rather than categorical.

---

# Target 8 — Defer References

References are valuable.

However:

* documentation links
* standards
* external guidance

can easily become hallucinated by an LLM.

Until references can be validated automatically, they should remain optional or omitted.

Focus instead on:

* evidence
* consequence
* recommendation

### Definition of Done

No requirement exists for LLM-generated references during this phase.

---

# Target 9 — Report Rendering Only

The report compiler should simply display findings.

It should not reinterpret them.

No additional scoring.

No weighting.

No aggregation.

Just rendering.

### Definition of Done

Reports show:

* Finding
* Evidence
* Engineering consequence
* Recommendation

without altering existing report behavior.

---

# Success Criteria

At the end of this phase:

✅ Existing reviewers still function.

✅ Existing CQI still functions.

✅ Existing benchmarks remain valid.

✅ Existing reports remain valid.

✅ Structured findings exist.

✅ Reports can display findings.

Nothing else changes.

---

# Deferred Work (Future Phases)

The following work is intentionally postponed until sufficient benchmark data exists.

* CQI redesign
* Severity weighting
* Confidence weighting
* Finding aggregation
* Cross-reviewer consensus
* Specialist reviewer integration
* Agent-to-agent finding exchange
* Decision synthesis
* Trust metrics based on findings

These belong to later phases after structured findings have been validated in production.

---

# Guiding Principle

This phase is an architectural extension—not a replacement.

The objective is to introduce a structured engineering language that can coexist with the current review system.

Only after collecting sufficient evidence should structured findings become the primary representation for scoring, benchmarking, and agent communication.

Following this incremental approach preserves stability, maintains comparability with historical results, and aligns with Silver-One's philosophy of evolving through measurable feedback rather than speculative redesign.
