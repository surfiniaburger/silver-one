# silver-one Architecture: Prompt, Context, Harness

This document maps the Prompt/Context/Harness model to the actual `silver-one` implementation and provides stress-test and security review flows.

## 1) Concept map in this repo

```mermaid
flowchart LR
    A[Prompt Engineering\nMessage quality] --> G[Gather Phase]
    B[Context Engineering\nWindow curation] --> G
    G --> H[Harness Engineering\nOuter machine]
    H --> I[Act + Verify loop]
```

## 2) Prompt engineering (unit of work: one input)

Repo mapping:
- Prompt assembly and schema constraints: `src/agentbeats/structured_output.py`
- Judge prompt and role constraints: `scenarios/debate/adk_debate_judge.py`
- Verifier prompt and audit constraints: `scenarios/debate/adk_debate_verifier.py`
- Generator prompts for boundary/refinement: `scenarios/debate/data_generator.py`

```mermaid
flowchart LR
    R[Role] --> P[Prompt]
    C[Context snippet] --> P
    I[Instructions] --> P
    E[Examples] --> P
    F[Format/schema] --> P
    P --> L[LLM inference]
    L --> O[Raw output]
    O --> J[Schema parse/repair\ncall_structured]
```

### Prompt stress tests

```mermaid
flowchart TD
    S[Start prompt test] --> A[Inject markdown fences]
    A --> B[Inject leading/trailing junk text]
    B --> C[Inject escaped/backslash edge cases]
    C --> D[Inject nested JSON objects]
    D --> E[Run call_structured parsing candidates]
    E --> F{Schema valid?}
    F -- No --> G[Trigger repair path]
    F -- Yes --> H[Pass]
    G --> I{Repair valid?}
    I -- No --> J[Fail test]
    I -- Yes --> H
```

## 3) Context engineering (unit of work: what stays in the window each step)

Repo mapping:
- Input curation decisions in runtime flow: `scenarios/debate/adk_debate_judge.py`
- Working set evolution across rounds: `run_eval` loop in judge
- Attempt-level telemetry for later curation audits: `artifacts/attempts/*.jsonl`

```mermaid
flowchart LR
    U[User/config/topic] --> K[Curator logic in judge]
    M[Prior outputs\nattempts/transcript] --> K
    T[Tool outputs\nverifier, debaters] --> K
    K --> W[Context window\ncurrent step budget]
    W --> LLM[LLM step]
    LLM --> N[New output]
    N --> K
```

### Context stress tests

```mermaid
flowchart TD
    A[Start context test] --> B[Increase num_rounds]
    B --> C[Add verbose tool outputs]
    C --> D[Add stale prior turns]
    D --> E[Check if key predicate/anchors still present]
    E --> F{Signal preserved?}
    F -- No --> G[Adjust curation/compression rules]
    F -- Yes --> H[Pass]
```

## 4) Harness engineering (unit of work: the machine)

Repo mapping:
- Scenario runner/orchestration: `src/agentbeats/run_scenario.py`
- Client/event streaming: `src/agentbeats/client_cli.py`, `src/agentbeats/client.py`
- Determinism and replay: `src/agentbeats/replay.py`
- Full BARRED judge loop and gating: `scenarios/debate/adk_debate_judge.py`
- Batch and offline gates: `scenarios/debate/run_batch.py`, `scenarios/debate/offline_b_gate.py`

```mermaid
flowchart LR
    G[Gather] --> A[Act]
    A --> V[Verify]
    V --> Q{Pass?}
    Q -- Yes --> R[Export artifact/output]
    Q -- No --> X[Retry with updated context]
    X --> G
```

```mermaid
flowchart TD
    S[Seed input] --> G1[Generator: boundary sample]
    G1 --> D1[Debate rounds pro/con]
    D1 --> J1[Judge structured eval]
    J1 --> V1[Verifier audit optional]
    V1 --> B2[Strict gates\nanchors/evidence/support]
    B2 --> P{Accepted?}
    P -- Yes --> O[Append training corpus]
    P -- No --> R[Refine sample and retry]
    R --> G1
```

## 5) Deterministic record/replay flow

```mermaid
flowchart TD
    A[Model call requested] --> B{Cassette hit?}
    B -- Yes --> C[Return cached response]
    B -- No --> D{Mode replay?}
    D -- Yes --> E[Fail fast: replay cache miss]
    D -- No --> F[Call provider]
    F --> G[Save response in cassette]
    G --> H[Return response]
```

## 6) Security and quality review flow

```mermaid
flowchart TD
    A[PR/change] --> B[Prompt safety review]
    B --> C[Context leak/staleness review]
    C --> D[Harness failure-mode review]
    D --> E[Determinism replay check]
    E --> F[Offline B-gate metrics]
    F --> G{Meets thresholds?}
    G -- No --> H[Block merge + fix]
    G -- Yes --> I[Approve merge]
```

### Security critique checklist

- Prompt layer:
  - Are system instructions robust against untrusted content injection?
  - Are output formats schema-constrained everywhere they should be?
  - Are parser helpers centralized (`structured_output.py`) with no duplicate drift?

- Context layer:
  - Can stale turns/tool noise bury critical predicate/anchor info?
  - Are we recording enough attempt metadata to audit context failures?
  - Do retry loops preserve relevant evidence and drop irrelevant noise?

- Harness layer:
  - Does every critical model call go through deterministic replay paths where required?
  - Do replay cache misses fail loudly?
  - Are verify gates strong enough to prevent low-groundedness outputs from being accepted?

## 7) Suggested stress-test matrix

```mermaid
flowchart LR
    A[Prompt tests] --> D[Acceptance criteria]
    B[Context tests] --> D
    C[Harness tests] --> D
    D --> E[Pass: merge]
    D --> F[Fail: refine + rerun]
```

- Prompt tests:
  - malformed JSON wrappers, nested objects, bad escapes, long preambles.
- Context tests:
  - long transcripts, noisy tool outputs, repeated retries, contradictory prior turns.
- Harness tests:
  - replay mode with missing cassette entries, verifier parse failures, strict gate rejections.

## 8) How to use this doc

1. Pick one layer (Prompt, Context, Harness).
2. Run the stress tests for only that layer.
3. Record failures in attempts/metrics.
4. Patch only the mapped files for that layer.
5. Re-run `offline_b_gate.py` and compare metrics.
