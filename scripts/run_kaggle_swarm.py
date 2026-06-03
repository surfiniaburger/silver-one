# %%
"""
workflow

cd /Users/surfiniaburger/Desktop/modular-metacog-swarm-v3/agent_training/silver-one

PYTHONUNBUFFERED=1 \
KAGGLE_SWARM_MAX_SEEDS=3 \
uv run python scripts/run_kaggle_swarm.py


run the metrics


cd /Users/surfiniaburger/Desktop/modular-metacog-swarm-v3/agent_training/silver-one

RUN_ID="$(basename "$(ls -t ../../artifacts/attempts/kaggle_swarm_seeds_*.jsonl | head -1)" .jsonl)"

python3 scenarios/debate/offline_b_gate.py \
  --input "../../artifacts/training_corpus_kaggle_${RUN_ID}.jsonl" \
  --attempts "../../artifacts/attempts/${RUN_ID}.jsonl" \
  --metrics-out "../../artifacts/metrics/${RUN_ID}_b_gate.json"


run a small seed 
  
PYTHONUNBUFFERED=1 \
KAGGLE_SWARM_MAX_SEEDS=1 \
KAGGLE_SWARM_MODELS=google/gemini-3-flash-preview \
uv run python scripts/run_kaggle_swarm.py


"""
# %%

import asyncio
import json
import os
import pathlib
import re
import sys
from typing import List

from pydantic import BaseModel, Field

from agentbeats.replay import ReplayManager
from agentbeats.structured_output import call_structured
from agentbeats.clock import RunClock

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# --- Schemas ---

class PredicateGeneration(BaseModel):
    thinking_process: str = Field(description="Step-by-step analysis of the code.")
    predicate: str = Field(description="A specific vulnerability claim about the code.")
    mechanism: str = Field(description="The technical mechanism of the vulnerability.")

class VerdictResponse(BaseModel):
    verdict: str = Field(description="Exactly '1' if vulnerable, '0' if not.")
    reasoning: str = Field(description="Brief technical justification.")

class AnchorResponse(BaseModel):
    anchors: List[str] = Field(description="List of concrete code identifiers found in the snippet.")
    reasoning: str = Field(description="Why these anchors were chosen.")

# --- Config & Paths ---

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent
CORPUS_PATH = SCRIPT_DIR.parent / "scenarios" / "debate" / "cve_seeds_test.jsonl"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# --- Helpers ---

def load_env():
    """Load .env from root if it exists."""
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

def get_available_models() -> List[str]:
    override = os.getenv("KAGGLE_SWARM_MODELS", "")
    if override.strip():
        return [m.strip() for m in override.split(",") if m.strip()]

    models_str = os.getenv("LLMS_AVAILABLE", "")
    if not models_str:
        return ["google/gemini-3-flash-preview"] # Fallback
    models = [m.strip() for m in models_str.split(",")]
    return models

def _anchor_in_code(anchor: str, code: str) -> bool:
    pattern = r"\b" + re.escape(anchor.strip()) + r"\b"
    return bool(re.search(pattern, code, re.IGNORECASE))


def _write_jsonl(path: pathlib.Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

# --- Runner ---

async def run_eval():
    load_env()
    
    proxy_url = os.getenv("MODEL_PROXY_URL")
    if proxy_url:
        # The ModelProxy implementation shows it appends /openapi
        if not proxy_url.endswith("/openapi"):
            proxy_url = f"{proxy_url.rstrip('/')}/openapi"
    
    proxy_key = os.getenv("MODEL_PROXY_API_KEY")
    
    raw_models = get_available_models()
    # Prefix for LiteLLM to use OpenAI-compatible proxy
    models = [f"openai/{m}" for m in raw_models]
    print(f"Evaluating models: {models}")

    # Initialize ReplayManager
    clock = RunClock()
    run_id = f"kaggle_swarm_seeds_{clock.now_iso().replace(':', '-')}"
    cassette_path = ARTIFACTS_DIR / "cassettes" / f"{run_id}.json"
    replay = ReplayManager.from_config(
        run_id=run_id,
        seed=42,
        cassette_path=str(cassette_path),
        mode="record",
        model_config={m: m for m in models}
    )

    # Artifact Paths
    attempts_path = ARTIFACTS_DIR / "attempts" / f"{run_id}.jsonl"
    corpus_out_path = ARTIFACTS_DIR / f"training_corpus_kaggle_{run_id}.jsonl"
    disagreements_path = ARTIFACTS_DIR / f"kaggle_disagreements_{run_id}.jsonl"
    os.makedirs(attempts_path.parent, exist_ok=True)
    os.makedirs(cassette_path.parent, exist_ok=True)

    # Load Seeds
    seeds = []
    max_seeds = int(os.getenv("KAGGLE_SWARM_MAX_SEEDS", "3"))
    with open(CORPUS_PATH) as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))
            if len(seeds) >= max_seeds:
                break

    results = {}

    # We'll use the default eval model to generate predicates
    gen_raw = os.getenv('LLM_DEFAULT_EVAL', raw_models[0])
    generator_model = f"openai/{gen_raw}"
    print(f"Using {generator_model} to generate predicates from seeds.")

    for i, seed in enumerate(seeds):
        seed_id = f"seed_{i:02d}"
        code = seed["topic"]
        print(f"\n>>> Processing {seed_id} (Code length: {len(code)})")

        # 1. Generate Predicate (The "Pro" move)
        try:
            replay.reset_usage_events()
            gen_data = await call_structured(
                replay_manager=replay,
                model=generator_model,
                messages=[
                    {"role": "system", "content": "You are a security researcher. Identify a potential vulnerability in the code and formulate a specific PREDICATE claim."},
                    {"role": "user", "content": f"CODE:\n{code}"}
                ],
                schema_name="PredicateGeneration",
                schema_model=PredicateGeneration,
                stage=f"gen/{seed_id}/predicate",
                base_url=proxy_url,
                api_key=proxy_key
            )
            predicate = gen_data.predicate
            generator_usage = replay.get_usage_summary()
            print(f"  Generated Predicate: {predicate}")

            # 2. Evaluate with all models
            for model in models:
                replay.reset_usage_events()
                model_slug = model.split("/")[-1]
                if model_slug not in results:
                    results[model_slug] = {
                        "passed_verdict": 0,
                        "anchors_grounded": 0,
                        "accepted_positive": 0,
                        "grounded_disagreement": 0,
                        "total": 0,
                    }
                
                print(f"    Testing Model: {model}")
                
                # Verdict Task
                res_verdict = await call_structured(
                    replay_manager=replay,
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a security auditor. Classify the code as vulnerable (1) or not (0) based on the predicate."},
                        {"role": "user", "content": f"PREDICATE: {predicate}\n\nCODE:\n{code}"}
                    ],
                    schema_name="VerdictResponse",
                    schema_model=VerdictResponse,
                    stage=f"{model_slug}/{seed_id}/verdict",
                    base_url=proxy_url,
                    api_key=proxy_key
                )
                
                # Anchor Task
                res_anchors = await call_structured(
                    replay_manager=replay,
                    model=model,
                    messages=[
                        {"role": "system", "content": "Extract concrete code anchors (identifiers) that support your verdict."},
                        {"role": "user", "content": f"PREDICATE: {predicate}\n\nCODE:\n{code}"}
                    ],
                    schema_name="AnchorResponse",
                    schema_model=AnchorResponse,
                    stage=f"{model_slug}/{seed_id}/anchors",
                    base_url=proxy_url,
                    api_key=proxy_key
                )
                
                grounded = [a for a in res_anchors.anchors if _anchor_in_code(a, code)]
                has_grounding = len(grounded) >= 2
                is_supported = res_verdict.verdict == "1"
                is_positive_corpus_row = has_grounding and is_supported
                is_grounded_disagreement = has_grounding and not is_supported
                
                if is_supported:
                    results[model_slug]["passed_verdict"] += 1
                if has_grounding:
                    results[model_slug]["anchors_grounded"] += 1
                if is_positive_corpus_row:
                    results[model_slug]["accepted_positive"] += 1
                if is_grounded_disagreement:
                    results[model_slug]["grounded_disagreement"] += 1
                
                print(f"      Verdict: {res_verdict.verdict} | Anchors: {len(grounded)} grounded")
                results[model_slug]["total"] += 1

                # --- Write Attempt Row ---
                attempt_row = {
                    "run_id": run_id,
                    "seed": 42 + i,
                    "model": model_slug,
                    "decision": "accepted" if is_positive_corpus_row else "rejected",
                    "reject_reason": None if is_positive_corpus_row else (
                        "grounded_disagreement" if is_grounded_disagreement else "insufficient_grounding"
                    ),
                    "input": code,
                    "predicate": predicate,
                    "verdict": res_verdict.verdict,
                    "reasoning": res_verdict.reasoning,
                    "support_level": "supported" if is_supported else "unsupported",
                    "output": {
                        "predicate": predicate,
                        "anchors": res_anchors.anchors,
                        "counterfactual": "N/A (Kaggle Swarm)",
                        "verifier_report": {
                            "grounded_anchors": grounded,
                            "grounding_pass": has_grounding
                        },
                        "support_level": "supported" if is_supported else "unsupported"
                    },
                    "soft_checks": {
                        "mechanism_grounding": {"pass": has_grounding},
                        "predicate_quality": {"pass": True}
                    },
                    "generator_llm_usage": generator_usage,
                    "llm_usage": replay.get_usage_summary()
                }
                
                _write_jsonl(attempts_path, attempt_row)

                # --- Write supported positives to Training Corpus ---
                if is_positive_corpus_row:
                    corpus_row = {
                        "instruction": f"Analyze this input for the condition: {predicate}",
                        "input": code,
                        "output": attempt_row["output"]
                    }
                    _write_jsonl(corpus_out_path, corpus_row)
                elif is_grounded_disagreement:
                    _write_jsonl(disagreements_path, attempt_row)

        except Exception as e:
            print(f"  Error processing {seed_id}: {e}")

    # Final Summary
    print("\n" + "="*60)
    print("KAGGLE SWARM SEED EVALUATION SUMMARY (Multi-Agent Flow)")
    print("="*60)
    for model_slug, res in results.items():
        vuln_rate = res["passed_verdict"] / res["total"] if res["total"] else 0
        grd_rate = res["anchors_grounded"] / res["total"] if res["total"] else 0
        print(
            f"{model_slug:40} | Vuln Rate: {vuln_rate:6.1%} | "
            f"Grounding: {grd_rate:6.1%} | "
            f"Accepted+: {res['accepted_positive']:2d} | "
            f"Disagree: {res['grounded_disagreement']:2d}"
        )
    print("="*60)

    replay.save_record(ARTIFACTS_DIR / f"{run_id}_record.json")
    print(f"Run record saved to artifacts/{run_id}_record.json")
    print(f"Attempts logged to {attempts_path}")
    print(f"Positive corpus logged to {corpus_out_path}")
    print(f"Grounded disagreements logged to {disagreements_path}")

if __name__ == "__main__":
    asyncio.run(run_eval())
