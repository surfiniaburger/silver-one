# AI Reviewer Engineering Principles

This document distills two reference ideas that are useful for Silver-One:

- Linus Torvalds' recent comments on AI-assisted maintenance and patch review.
- Dave Farley's coupling guidance from Continuous Delivery / Modern Software
  Engineering.

The goal is not to copy either source directly. The goal is to turn them into
engineering constraints for building a local-LLM code reviewer that is useful in
CI, measurable over time, and safe to evolve.

## 1. What The Linus Article Is Really About

The article is less about whether AI is good or bad, and more about the
standard AI tools must meet before maintainers can trust them.

Key points:

- Maintainers work at the level of intent, risk, and explanation, not just raw
  code diffs.
- AI-generated reports are harmful when they look plausible but require humans
  to spend time proving they are false.
- A useful AI finding should not be thrown over the wall. It should include
  enough context for a human to evaluate it quickly.
- The best AI-assisted workflow is interactive: the human remains responsible
  for judgment, while the tool accelerates discovery, patch sketching, and
  verification.
- Suggested code changes are useful, but only when they explain why the change
  fixes the real bug rather than hiding the immediate symptom.
- The language or tool does not remove logic bugs. Better workflows, tests,
  verification, and review discipline still matter.

For Silver-One, this implies that a review finding should eventually have three
parts:

1. Explanation: what is wrong and why it matters.
2. Evidence: where the behavior appears in the code, tests, or artifacts.
3. Action: a concrete patch direction or suggested code change.

If a finding cannot provide those three parts, it should be treated as lower
confidence or marked as needing human follow-up.

## 2. What The Farley Coupling Article Is Really About

Farley's article frames software quality as the ability to change software
safely. Coupling is the main force that makes change expensive.

Important concepts:

- Coupling is necessary. The discipline is choosing which coupling to accept.
- Strong coupling is acceptable when the dependency is stable.
- Unstable, invisible, accidental, or unintended coupling is the real danger.
- Slow feedback makes coupling more dangerous.
- The worst state is a strongly coupled system with slow feedback.
- CI is a way to survive coupling while the system is still being understood.
- Tests that are hard to write are design feedback, not just test friction.
- Boundaries should validate and translate inputs so instability does not leak
  through the system.

For Silver-One, the unstable boundary is not only Python code. It is the LLM
contract:

- prompt text,
- structured JSON output,
- repair and retry behavior,
- cassette shape,
- report semantics,
- CI gate rules.

Those boundaries must be explicit, validated, and measured.

## 3. How This Maps To Silver-One

Silver-One is not merely "a model that reviews code." The product is the
harness around the model:

- changed-unit extraction,
- batching,
- prompt contract,
- structured-output validation,
- repair and retry policy,
- cassette compatibility,
- telemetry,
- unified reporting,
- final gate decision.

The model supplies judgment. The harness supplies reliability.

That means we should bias toward a thicker harness while using a small local
model. The model should not be trusted to remember every contract rule. The
system should make valid output easier, malformed output recoverable, and
failure modes observable.

## 4. Design Principles For The Reviewer

### Explain Before Suggesting

A suggested code change without an explanation is not enough. A reviewer should
explain:

- the failure mode,
- why the current code is risky,
- what the proposed change preserves,
- what trade-off the fix accepts.

This follows the Linus standard: the maintainer needs the bigger picture, not
only a patch-shaped artifact.

### Prefer Evidence-Carrying Findings

Each finding should ideally include:

- file path,
- function or unit name,
- line reference when available,
- observed condition,
- expected behavior,
- confidence,
- suggested action.

This helps distinguish useful bugs from plausible but false reports.

### Separate Reviewer Quality From Harness Reliability

Farley's coupling lens says we should name the coupling. In Silver-One, reviewer
quality and harness reliability are coupled unless we deliberately separate
them.

Keep separate metrics for:

- review quality: severity, findings, CQI, agreement, false positives.
- structured-output health: first-pass validation, repairs, retries, final
  failures, provider errors.
- coverage: extracted units, reviewed units, skipped units, batch count.

If the report mixes these together, we cannot tell whether the model was weak or
the workflow was unreliable.

### Validate At Boundaries

Every external or unstable boundary should validate and normalize data:

- LLM raw response to schema model,
- schema model to cassette artifact,
- cassette artifact to report input,
- report metrics to gate decision.

This matches Farley's "translate at the edge" guidance. Do not let malformed
provider output or legacy cassette shapes leak deep into reporting logic.

### Keep Compatibility Explicit

Legacy cassette support is a coupling decision. It is acceptable because old CI
artifacts are real consumers of the current report flow.

Rules:

- name legacy adapters clearly,
- keep compatibility tests,
- do not hide schema migrations inside unrelated report logic,
- record defaults applied during migration when they affect metrics.

### Treat Prompt Shape As Infrastructure

The prompt is not just prose. It is part of the interface between the harness
and the model.

Prompt changes should be reviewed like code changes:

- schema examples must parse,
- required fields must be listed,
- scoring constraints must be explicit,
- edge cases should be represented,
- telemetry should show whether the change improved first-pass compliance.

## 5. Current Practical Uses

These ideas are immediately useful in the current work:

- Keep structured-output reliability work ahead of new reviewer features.
- Continue measuring score repairs, empty summaries, retries, and final
  failures.
- Treat Gemini's defensive-programming comments as boundary-hardening signals,
  not just style suggestions.
- Keep batching and coverage telemetry visible so CI reports are honest about
  how much changed code was reviewed.
- Make report output explain when a unit failed because of provider/runtime
  instability rather than reviewer judgment.
- Avoid review comments that only say "this is bad"; prefer comments that
  explain impact and suggest a concrete fix direction.

## 6. Later Uses

These ideas also point to future capabilities:

- Suggested patches attached to findings.
- Finding confidence levels based on available evidence.
- A "needs human verification" category for plausible but unproven issues.
- Benchmarks that measure false positives and false negatives, not only JSON
  validity.
- Prompt A/B testing where first-pass schema compliance and repair rates are
  tracked before merging prompt changes.
- Coupling-focused review dimensions that detect hidden dependencies, long
  parameter lists, brittle tests, duplicated concepts, and unstable boundaries.
- Specialist reviewers that share the same validation, telemetry, cassette, and
  report infrastructure.

## 7. Working Standard

Silver-One should aspire to this standard:

> A review finding is useful when it explains the risk, shows evidence, and
> gives the maintainer a concrete next action without hiding uncertainty.

And the workflow should satisfy this standard:

> A failed review run should explain whether the problem came from the code, the
> model, the provider, the schema contract, or the CI harness.

Those standards keep the project aligned with both sources: useful AI assistance
from the Linus perspective, and fast, decoupled, observable feedback from the
Farley perspective.
