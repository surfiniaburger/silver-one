# BARRED Roadmap & Cost-Reduction Proposal

## Summary

You have strong pipeline reliability but high generation cost per accepted example. This document captures the feedback and recommended roadmap to reduce cost, scale the corpus, and prepare for student training.

## Key Observations

- Total accepted examples across 5 runs: 58 accepted rows (A:13, B:11, C:10, D:13, E:11).
- Zero accepted logic errors across accepted examples — exceptional pipeline reliability.
- Token cost per run ≈ 1M tokens producing 10–13 accepted rows ⇒ ~80k–100k tokens per accepted example.
- Generation discard rate is ~60–70% (start with 30–35 attempts, accept 10–13) — major source of cost.

## Core Recommendation

Phase 1 (Immediate, 2–4 weeks): Focus on cost reduction by predicting acceptance before expensive generation steps.

- Build an Acceptance Predictor (small classifier).
  - Input: `seed`, `predicate`, `anchor` (candidate example metadata or short text features).
  - Output: `Likely Accept` / `Likely Reject`.
  - Goal: Filter out many likely-failing candidates before running full debate/generation.
- Also build an Anchor-Quality Predictor to detect weak anchors early.
- Target: 50% reduction in token cost per accepted row.

Rationale: The pipeline is reliable but not yet scaled. It's cheaper to reject bad attempts before costly debate than to train a student prematurely.

## Phased Roadmap

Phase 1 — Cost Reduction (Now)
- Deliverables:
  - Acceptance Predictor (small classifier)
  - Anchor-quality predictor
  - Instrument token accounting per accepted row
- Target: 50% cost reduction in token spend per accepted row

Phase 2 — Corpus Expansion
- Goal: 500–1000 accepted rows using existing pipeline with improved cost-efficiency.
- Keep current architecture; scale generation once cost per accepted row is reasonable.
- Target split for training later: 600 train / 75 val / 75 test (for ~750 accepted rows)

Phase 3 — Student Pilot
- Train a small student (1B–3B parameters) on structured triples: Anchor / Mechanism / Impact
- Start with a Mechanism Extractor experiment (code + predicate → anchor, mechanism, impact)
- Measure whether learning transfers before building larger models like SecurityDecisionGuard

Phase 4 — Trust-Boundary Corpus
- Expand beyond CVEs to package updates, workflow changes, AI agent configs, and supply-chain events
- Explore Trust Drift metrics (e.g., Verifier Overconfidence Index)

## Experiments to Run Immediately

- Investigate Run E closely (adversarial calibration failure).
  - Why did verifier confidence increase while evidence quality fell?
  - Potential output: `Trust Drift Score` or `Verifier Overconfidence Index`.
- Instrument per-attempt diagnostics and token breakdown to quantify cost sinks.

## Why 750 Accepted Rows?
- Provides sufficient data for train/val/test splits and meaningful learning curves.
- Avoids the temptation to gather massive but lower-quality corpora; quality matters more here.

## Allocation Suggestion (by effort)
- 40% Cost Reduction
- 35% Corpus Expansion
- 15% Student Pilot
- 10% Trust-Boundary Research

## Phase 1 Deliverable Spec: Acceptance Predictor

- Model: small classifier (logistic / small transformer / tuned tree) depending on feature format
- Inputs: short representations of `seed`, `predicate`, and `anchor` (text encodings or handcrafted features)
- Labels: `Accepted` / `Rejected` (from previous runs)
- Evaluation: Precision on predicted-accept class; prioritize precision to avoid false accepts
- Deployment: Use as a pre-filter before expensive debate/generation

## Metrics to Track

- Tokens spent per accepted row (before / after)
- Acceptance precision & recall of the predictor
- Overall accepted yield (absolute accepted rows per unit spend)
- Verifier confidence vs. independent quality measures (trust drift)

## Next Steps (actionable)

1. Export a CSV of past attempts with features: seed, predicate, anchor-length, verifier score, accepted/rejected label.
2. Prototype a small classifier (scikit-learn / tiny transformer) and measure precision for top-k predicted accepts.
3. Integrate predictor as a pre-filter into the generation pipeline and re-run a short experiment to measure token reduction.

---

*Prepared as requested — concise roadmap and immediate experiment plan to reduce cost and scale BARRED.*
