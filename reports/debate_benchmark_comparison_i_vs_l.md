# Debate Benchmark Telemetry: `pilot-v1-calibrated-l`

## Executive Summary

| Metric | Value |
| :--- | :--- |
| **Run ID** | `pilot-v1-calibrated-l` |
| **Total Attempts** | 13 |
| **Accepted Rows** | 7 |
| **Yield Pass Rate** | 53.8% |
| **Total Wall-Clock Time** | 1099.7s |
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
| **Wall-Clock Duration** | 1041.6s | 1099.7s | `+5.58%` |
| **Cache Hit Rate** | 3.3% | 0.0% | `-3.28%` |

### Stage Latency Speedups

| Stage | Baseline Avg (ms) | Candidate Avg (ms) | Speedup (%) |
| :--- | :--- | :--- | :--- |
| `generator_boundary` | 22696.0ms | 39656.1ms | `-74.73%` |
| `generator_refine` | 56509.1ms | 28035.0ms | `+50.39%` |
| `judge_adjudication` | 12138.6ms | 20574.7ms | `-69.50%` |
| `verifier_audit` | 11044.9ms | 18155.6ms | `-64.38%` |

## Stage Breakdown

| Stage | Calls | Prompt Tokens | Completion Tokens | Avg Duration | P95 Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `generator_boundary` | 10 | 210,200 | 17,333 | 39656.1ms | 80186.9ms |
| `judge_adjudication` | 26 | 53,566 | 36,287 | 20574.7ms | 33746.8ms |
| `verifier_audit` | 16 | 19,489 | 15,894 | 18155.6ms | 24498.9ms |
| `generator_refine` | 6 | 76,089 | 7,682 | 28035.0ms | 53754.7ms |

## Quality Gate Signals

- **Verifier Pass Rate:** 87.5%
- **Verifier Parse OK Rate:** 100.0%
- **Anchor Match Rate:** 100.0%
- **B-Gate Pass Status:** `PASS`
