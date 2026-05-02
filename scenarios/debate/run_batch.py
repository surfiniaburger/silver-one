import json
import asyncio
import sys
import os

# Add src to PYTHONPATH to import agentbeats
sys.path.append(os.path.join(os.getcwd(), "src"))

from agentbeats.client import send_message

async def run_batch():
    seeds_file = "scenarios/debate/cve_seeds_50.jsonl"
    judge_url = "http://127.0.0.1:9009" # Base URL for A2A client
    output_file = "test_corpus_50.jsonl"
    
    if not os.path.exists(seeds_file):
        print(f"Error: {seeds_file} not found.")
        return

    # Load existing results to support resume
    processed_predicates = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # The instruction contains the predicate
                    processed_predicates.add(data.get("instruction", ""))
                except:
                    pass

    with open(seeds_file, "r") as f:
        seeds = [json.loads(line) for line in f]
    
    print(f"Loaded {len(seeds)} seeds. {len(processed_predicates)} already processed.")
    print(f"Target Judge: {judge_url}")
    print(f"Output: {output_file}")
    
    for i, seed in enumerate(seeds):
        # We check if the instruction (Analyze this input for the condition: <predicate>) is already there
        instruction = f"Analyze this input for the condition: {seed['predicate']}"
        if instruction in processed_predicates:
            print(f"Skipping seed {i+1} (already processed).")
            continue

        print(f"\n>>> [{i+1}/{len(seeds)}] Seed Predicate: {seed.get('predicate')[:80]}...")
        
        payload = {
            "participants": {
                "pro_debater": "http://127.0.0.1:9019/",
                "con_debater": "http://127.0.0.1:9018/"
            },
            "config": {
                "topic": seed["topic"],
                "predicate": seed["predicate"],
                "target_verdict": "True",
                "target_dimension": "Security Invariants",
                "num_rounds": 2,
                "max_refinements": 1,
                "output_file": output_file
            }
        }
        
        try:
            # send_message expects a string
            # Increased timeout for the client call explicitly as well
            result = await send_message(json.dumps(payload), judge_url)
            print(f"  Result received. Status: {result.get('status')}")
        except Exception as e:
            print(f"  ERROR: Seed {i+1} failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_batch())


