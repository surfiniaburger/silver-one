# Executive Summary — BARRED Next Steps

Verdict
- Pipeline is highly reliable: 58 accepted examples with zero accepted logic errors.
- Bottleneck is economics: ~80k–100k tokens per accepted example.

Immediate Goal (Phase 1)
- Reduce token cost per accepted example by 50% through pre-filtering.
- Build an Acceptance Predictor (small classifier) and an Anchor-quality predictor.

Why this first
- 60–70% of generation effort is discarded; filtering early multiplies cost savings without changing quality.

Targets
- Phase 1 (2–4 weeks): 50% cost reduction.
- Phase 2: 500–1000 accepted rows (stabilize current pipeline).
- Phase 3: Train small student (1B–3B) on Anchor/Mechanism/Impact.

Critical Immediate Action
- Analyze Run E for verifier drift; instrument to detect verifier overconfidence.

One-line ask
- Let me build the Acceptance Predictor and run a short cost-reduction experiment next.
