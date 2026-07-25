# Debate Benchmark Telemetry: `pilot-v1-calibrated-j`

## Executive Summary

| Metric | Value |
| :--- | :--- |
| **Run ID** | `pilot-v1-calibrated-j` |
| **Total Attempts** | 18 |
| **Accepted Rows** | 6 |
| **Yield Pass Rate** | 33.3% |
| **Total Wall-Clock Time** | 0.0s |
| **Total Prompt Tokens** | 430,374 |
| **Total Completion Tokens** | 98,634 |
| **Tokens / Accepted Row** | 88,168.0 |
| **Cache Hit Rate** | 1.3% (1 hits / 78 misses) |

## A/B Comparison: Candidate `pilot-v1-calibrated-j` vs Baseline `pilot-v1-calibrated-i`

| Dimension | Baseline | Candidate | Delta / Speedup |
| :--- | :--- | :--- | :--- |
| **Total Tokens** | 527,435 | 529,008 | `+0.30%` |
| **Prompt Tokens** | 418,245 | 430,374 | `+2.90%` |
| **Tokens / Accepted Row** | 75,347.86 | 88,168.00 | `+17.01%` |
| **Wall-Clock Duration** | 0.0s | 0.0s | `+0.00%` |
| **Cache Hit Rate** | 3.3% | 1.3% | `-2.01%` |

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
| `generator_boundary` | 11 | 213,782 | 11,877 | 0.0ms | 0.0ms |
| `judge_adjudication` | 36 | 57,202 | 50,202 | 0.0ms | 0.0ms |
| `verifier_audit` | 18 | 17,134 | 16,666 | 0.0ms | 0.0ms |
| `generator_refine` | 14 | 142,256 | 19,889 | 0.0ms | 0.0ms |

## Quality Gate Signals

- **Verifier Pass Rate:** 66.7%
- **Verifier Parse OK Rate:** 100.0%
- **Anchor Match Rate:** 100.0%
- **B-Gate Pass Status:** `FAIL`
