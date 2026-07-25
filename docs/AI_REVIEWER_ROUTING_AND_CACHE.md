# AI Reviewer Routing And Prompt Cache Discipline

This document distills the DDODS notes on production LLM routing and KV-cache
economics into design rules for Silver-One's evaluator and future specialist
reviewers.

The main lesson is simple: model routing only saves money when it preserves the
cache shape of the task. A cheaper model with a cold cache can cost more than a
more expensive model with a hot cache.

## Production Routing Shape

A production routing layer is more than "send easy prompts to a cheap model."
The useful shape is:

1. **Guardrail filter**
   Validate safety, policy, and traceability before model selection.

2. **Router model**
   Use a small, cheap classifier to infer task type, domain, and action.

3. **Selection policy**
   Pick the cheapest or fastest acceptable model from the candidates. Keep a
   fallback model available when the winner fails.

4. **Model affinity**
   Pin the selected model to the task/session. Do not switch models mid-task
   unless there is a deliberate fallback or retry policy.

The fourth step is the part that matters most for agent cost. Agents do not make
one call per task. They plan, call tools, read observations, analyze results,
and then answer. Each turn resends accumulated context. If the model changes
inside that sequence, the old model's cache cannot be reused.

## KV Cache Mechanics

Transformer inference has two phases:

- **Prefill:** process the whole input prompt and compute attention state for
  all prompt tokens. This is compute-heavy.
- **Decode:** generate one token at a time while reading previously computed
  key/value tensors. This is more memory-bound.

During prefill, the model computes query, key, and value vectors. The key/value
vectors for earlier tokens do not change after they are computed. Prompt caching
stores those key/value tensors so the next request with the same prefix can skip
recomputing them.

The cache is indexed by the exact token prefix. If the prefix changes, the cache
misses. This includes small changes such as:

- adding a timestamp to the system prompt,
- shuffling tool schema order,
- changing tool definitions mid-session,
- mutating project context in the static prefix,
- switching model providers or model names mid-task.

Prompt caches are model-specific. A cache-hot expensive model can be cheaper
than a cache-cold cheap model when the task has a large repeated prefix.

## Prompt Layout Rule

Keep stable context at the top and dynamic context at the bottom:

1. base system instructions,
2. tool definitions,
3. stable project docs or reviewer rubric,
4. session configuration,
5. user messages, tool outputs, observations, and run-specific data.

The first four layers should stay stable for the task. New facts should be
appended as dynamic messages rather than inserted into or rewriting the prefix.

This matters for Silver-One prompts. A compact embedded rubric is better than a
fake instruction like "read this file" when the model cannot read that file. If
we later load doc-derived rubric text, the loader should inject a bounded,
stable section in a predictable location.

## Cache-Safe Compaction

When context grows too large, do not mutate the static prompt to summarize
state. Fork a compaction call:

1. keep the same system prompt, tools, and prior context,
2. append a new instruction asking for a bounded summary,
3. build a fresh context with the same stable prefix plus the summary.

This preserves cache reuse during the compaction call and avoids contaminating
the static prefix with changing session state.

## Implications For Silver-One

Current Silver-One lanes are conceptually separate:

- Farley test-quality evaluation,
- code-review evaluation,
- unified report aggregation.

They may share one model backend today. That is fine while the workflow is
small. The important rule is to avoid switching models inside a single lane's
task once review context has accumulated.

Future specialist reviewers should be designed as stable lanes, not as casual
mid-task model switches:

- `boundary_contract_reviewer`,
- `correctness_reviewer`,
- `compatibility_reviewer`,
- `security_reviewer`,
- `test_quality_reviewer`,
- `maintainability_coupling_reviewer`.

Each lane should have:

- a stable system prompt,
- a narrow rubric,
- a fixed structured-output schema,
- lane-specific telemetry,
- a pinned model choice for the task/session,
- a clear contribution to the unified report.

This can still run inside one CI job. The boundary is logical first: one lane,
one stable prompt shape, one cache-friendly task/session.

## Small Model (SLM) Serving Economics & Memory Contention

A common architectural trap when moving from large frontier models (GPT-4/Claude) to small task-specific models (SLMs like Qwen 2B or MiniLM) is assuming that cheaper per-call tokens automatically reduce total system costs.

As highlighted in production SLM serving literature (*Why Small Models Alone Don't Reduce Inference Costs* by Avi Chawla), hardware cost in production and CI environments is determined by the GPUs or CPU vCPUs rented over wall-clock time. Standard serving frameworks (`vLLM`, `TEI`, `Ollama`) typically pre-allocate memory buffers per model instance or concurrency slot. When serving multiple model families (e.g. OCR, embedding, reranking, generation) without dynamic load/eviction or unified batching, tool fragmentation forces multiple dedicated instances, recreating hardware over-provisioning.

In constrained local execution environments (such as 2-vCPU CI runners), uncoordinated parallel slot execution introduces two primary bottleneck mechanisms:

1. **Memory Bus & Core Saturation (Hypothesized Mechanism)**: Local LLM token decoding is strictly memory-bandwidth bound. Running multiple concurrent inference slots (e.g. `OLLAMA_NUM_PARALLEL: 2`) on 2 vCPUs causes matrix multiplication threads to compete for shared CPU memory bandwidth and cache lines. Token generation throughput drops significantly (from ~9.8 tokens/sec down to ~3.2 tokens/sec per slot), inflating wall-clock execution time.
2. **Uncoordinated KV-Cache Eviction (Hypothesized Mechanism)**: Splitting inference into uncoordinated parallel slots forces the engine to maintain or switch between distinct memory context buffers. Alternating unit evaluation requests can result in back-and-forth KV-cache buffer evictions, forcing prompt prefill recalculation even when system prompts are deterministic.

As documented in [`docs/EVALUATOR_LATENCY_AND_CACHE_BENCHMARKS.md`](EVALUATOR_LATENCY_AND_CACHE_BENCHMARKS.md), enabling dual slots (`OLLAMA_NUM_PARALLEL: 2`) on a 2-vCPU runner inflated average per-call latency from 59.1s (optimized single-slot) up to 124.8s. Single-slot queueing (`max-concurrency: 1`) maximized memory bus throughput and preserved prompt cache warmth in this benchmark. Higher concurrency on multi-core or GPU configurations requires separate empirical testing.

## Design Rules

- Route at the task or lane level, not every individual LLM call.
- Pin the chosen model for the lane/session.
- Keep prompt prefixes deterministic.
- Do not inject timestamps or mutable state into system prompts.
- Do not add, remove, or reorder tools mid-session.
- Keep doc-derived rubric snippets bounded and stable.
- Append dynamic observations instead of rewriting static context.
- Track cache efficiency when provider telemetry exposes cache reads/writes.
- Treat model switching as a policy decision with known cache cost, not as a
  free optimization.

## Governance Connection

Routing and caching are part of reviewer governance. If a reviewer lane changes
model, prompt, tools, or schema unpredictably, then its cost, reproducibility,
and calibration become harder to audit.

For Silver-One, the reliable path is:

- stable lane prompts,
- explicit schema contracts,
- structured cassettes,
- per-lane telemetry,
- unified aggregation,
- small calibration patches when the reviewer produces noisy or misleading
  feedback.

The goal is not just a cheaper reviewer. The goal is a reviewer whose cost,
behavior, and failure modes are understandable over time.
