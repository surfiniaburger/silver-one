# Debate Benchmark Telemetry: `pilot-v1-calibrated-l`

## Executive Summary

| Metric | Value |
| :--- | :--- |
| **Run ID** | `pilot-v1-calibrated-l` |
| **Total Attempts** | 13 |
| **Accepted Rows** | 7 |
| **Yield Pass Rate** | 53.8% |
| **Total Wall-Clock Time** | 0.0s |
| **Total Prompt Tokens** | 359,344 |
| **Total Completion Tokens** | 77,196 |
| **Tokens / Accepted Row** | 62,362.86 |
| **Cache Hit Rate** | 0.0% (0 hits / 58 misses) |

## A/B Comparison: Candidate `pilot-v1-calibrated-l` vs Baseline `pilot-v1-calibrated-i`

| Dimension | Baseline | Candidate | Delta / Speedup |
| :--- | :--- | :--- | :--- |
| **Total Tokens** | 527,435 | 436,540 | `-17.23%` |
| **Prompt Tokens** | 418,245 | 359,344 | `-14.08%` |
| **Tokens / Accepted Row** | 75,347.86 | 62,362.86 | `-17.23%` |
| **Wall-Clock Duration** | 0.0s | 0.0s | `+0.00%` |
| **Cache Hit Rate** | 3.3% | 0.0% | `-3.28%` |

### Stage Latency Speedups

| Stage | Baseline Avg (ms) | Candidate Avg (ms) | Speedup (%) |
| :--- | :--- | :--- | :--- |
| `generator_boundary` | 0.0ms | 0.0ms | `-0.00%` |
| `generator_refine` | 0.0ms | 0.0ms | `-0.00%` |
| `judge_adjudication` | 0.0ms | 0.0ms | `-0.00%` |
| `verifier_audit` | 0.0ms | 0.0ms | `-0.00%` |

## Stage Breakdown

| Stage | Calls | Prompt Tokens | Completion Tokens | Avg Duration | P95 Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `generator_boundary` | 10 | 210,200 | 17,333 | 0.0ms | 0.0ms |
| `judge_adjudication` | 26 | 53,566 | 36,287 | 0.0ms | 0.0ms |
| `verifier_audit` | 16 | 19,489 | 15,894 | 0.0ms | 0.0ms |
| `generator_refine` | 6 | 76,089 | 7,682 | 0.0ms | 0.0ms |

## Quality Gate Signals

- **Verifier Pass Rate:** 87.5%
- **Verifier Parse OK Rate:** 100.0%
- **Anchor Match Rate:** 100.0%
- **B-Gate Pass Status:** `PASS`
