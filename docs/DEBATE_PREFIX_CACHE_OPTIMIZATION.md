# Debate Prefix Cache & Optimization Architectural Principles

This document defines the 4 core architectural pillars of Silver-One's Debate scenario optimization, KV-cache prefix alignment, telemetry extraction, and cassette governance workflow.

---

## 1. Deterministic System Prompt Prefix Alignment & KV-Cache Reusability

To maximize LLM Inference Engine (Ollama/vLLM/SGLang) KV-cache hit rates:
- **Static Prefix Isolation:** All static system roles, judge rubrics, and JSON schema constraints MUST be strictly housed in `messages[0]` (system prompt).
- **Dynamic Context Decoupling:** Dynamic inputs (CVE seeds, target predicates, code snippets, and debate turn transcripts) MUST be placed in `messages[1]` (user message) or subsequent turn messages.
- **Strict Immutability:** System prompt strings must remain 100% byte-for-byte identical across calls within and across debate runs, enabling multi-token prefix matching and KV-cache reuse.

---

## 2. Model Keep-Alive (`keep_alive: 24h`) & VRAM Retention

To prevent expensive cold-boot latency and model unloading between debate turns and verifier audits:
- All structured output calls (`call_structured`) and standard completions (`_call_llm`) pass `"keep_alive": "24h"` in LiteLLM options.
- Retains model weights and KV-cache state in GPU VRAM throughout long-running 10-seed debate runs and multi-concurrency evaluation batches.

---

## 3. Controlled A/B Benchmark & Telemetry Extractor Workflow

To empirically evaluate latency, cache efficiency, and token usage deltas without subjective guesswork:
- **Baseline Logging:** Establish calibrated baseline datasets (e.g. `pilot-v1-calibrated-i`) recorded under `--mode record`.
- **Candidate Benchmark:** Execute candidate post-caching record runs (e.g. `pilot-v1-calibrated-j`).
- **Telemetry Extraction:** Process execution traces with `scripts/debate_telemetry.py` to extract:
  - Cache hit rates and prefill latency deltas ($\Delta$ seconds/call).
  - Token efficiency and attempt count breakdowns.
  - Stage-by-stage latency distributions (Judge vs Generator vs Verifier).
  - Markdown report generation (`reports/debate_benchmark_comparison_i_vs_j.md`).

---

## 4. Cassette Baseline Lineage & Versioned Record-Mode Governance

To balance cassette replay reproducibility with active prompt evolution:
- **Deterministic Replay Mechanics:** `ReplayManager` computes `Hash(model, messages, kwargs)`. Any structural prompt refactoring causes a cassette miss under `--mode replay`.
- **Versioned Baseline Lineage:** When system prompts are refactored for prefix alignment, old baseline cassettes (e.g. `pilot-v1-calibrated-a` through `i`) are preserved as immutable historical baselines.
- **Controlled New Baselines:** New calibrated record runs (`pilot-v1-calibrated-j`) establish the fresh baseline for post-optimization replay testing.
- **Git Tracking Rules:** Official baseline cassettes (`pilot-v1-calibrated-a`..`g` and CI cassettes `farley_score*`) remain tracked, while local/experimental runs (`artifacts/cassettes/*.json`) are ignored via `.gitignore`.

---

## Execution & Concurrency Verification

- **Automated Test Suite:**
  ```bash
  uv run pytest tests/
  uv run pytest scenarios/debate/ -o asyncio_mode=auto
  ```

- **GPU Concurrency Sweep:**
  Evaluate `--max-concurrency` (1, 2, 4, 8) using `scenarios/debate/run_batch.py` to measure tensor bandwidth scaling and throughput limits on Cloud GPUs.
