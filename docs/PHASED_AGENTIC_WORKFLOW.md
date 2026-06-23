# Proposal: Phased Evolution of the Agentic Quality Workflow

This document outlines the roadmap for transitioning our static test and code review scripts into a fully agentic workflow, leveraging **Agent Skills**, **Subagent Delegation**, and **Agent-to-Agent (A2A)** handoffs.

---

## The Phased Roadmap

Instead of introducing complex agentic frameworks immediately, we propose a four-phased approach. This ensures we establish solid core quality gates first, package them for agent consumption second, and scale orchestration only when the infrastructure supports it.

```
┌────────────────────────────────────────────────────────┐
│ PHASE 1: Script-Driven Core logic (Current)            │
│ Standard python scripts, local CPU models, diff scopes │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 2: Agent Skills Packaging                        │
│ Package scripts as Skills (.md + tools) for Swarm use  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 3: Subagent & Specialist Delegation              │
│ Parent agent spawns child VMs/contexts for dimensions  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 4: Macro A2A Coordination                        │
│ Cross-agent handoff (e.g., Code Review -> Release)     │
└────────────────────────────────────────────────────────┘
```

---

## Phase 1: Script-Driven Core Logic (Current)

### Focus
Get the diff-only parser, the Farley scoring, and the Virtual Suite comparison working flawlessly.

* **Mechanics**: Plain python scripts (`farley_score_evaluator.py`, `code_review_evaluator.py`) running sequentially in a single process.
* **Why**: Keeps debugging easy. If the core math or diff-parsing is broken, wrapping it in an agent wrapper will only obscure the bugs.
* **Local Run Friendly**: Running on local CPU (Ollama) has zero framework overhead.

---

## Phase 2: Agent Skills Packaging

### Focus
Expose our quality checks as reusable capabilities ("Skills") so that any agent working in this repo can run them on demand.

* **Mechanics**: 
  * Create `skills/farley_evaluation/` and `skills/code_review/` directories.
  * Write `SKILL.md` defining what the skill does (e.g., *"Invoke me when the user wants to audit the Farley TDD index of a pull request"*).
  * Place our scripts and helpers inside the skill folder as its execution tools.
* **Value**: When a developer interacts with a Chat/PR agent (like Claude/Gemini), the agent can dynamically pull the skill into its context window, run the validation tool, and interpret the results conversationally.

---

## Phase 3: Subagent & Specialist Delegation

### Focus
Parallelize and isolate specialized audits (Security, Compatibility, Correctness) using child contexts.

```
                 Parent PR Review Agent
             ┌─────────────────────────────┐
             │ Orchestrates diff filtering │
             └──────────────┬──────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
  Subagent 1: Security   Subagent 2: Compat   Subagent 3: Correct
 ┌────────────────────┐ ┌──────────────────┐ ┌───────────────────┐
 │ Focus: Vulns, leaks│ │ Focus: OS paths, │ │ Focus: Logic,     │
 │                    │ │ encodings        │ │ edge cases        │
 └─────────┬──────────┘ └────────┬─────────┘ └─────────┬─────────┘
           │                     │                     │
           └──────────────────┐  │  ┌──────────────────┘
                              ▼  ▼  ▼
                 Parent PR Review Agent
             ┌─────────────────────────────┐
             │ Consolidates findings & code│
             └─────────────────────────────┘
```

* **Mechanics**:
  * The parent agent filters the diff.
  * It spawns **3 distinct subagents** in parallel, each with a fresh, clean context and a highly specific system prompt (specializing in one domain).
  * Subagents return their focused findings back to the parent context.
* **Prerequisite**: Requires either a cloud LLM provider (where parallel generation is instant and cheap) or optimized local batch processing to avoid sequential CPU queues.

---

## Phase 4: Macro Agent-to-Agent (A2A) Coordination

### Focus
Integrate our review agents with other long-running service agents across the engineering ecosystem.

* **Mechanics**:
  * Each agent publishes an **Agent Card** explaining its inputs and capabilities.
  * Once the **PR Review Agent** passes the PR with a high Farley score, it discovers and triggers the **Release/Merge Agent**:
    > *"I have verified PR #64. The Code Review is OK and the Farley Index improved by +0.3. Here is the signed cassette. Please deploy to staging."*
  * The **Release Agent** deploys the code and returns the staging status to the PR comment.
