import asyncio
import json
import os
import sys

# Add src to PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), "src"))

from agentbeats.client import send_message

async def main():
    judge_url = "http://127.0.0.1:9009"
    payload = {
        "participants": {
            "pro_debater": "http://127.0.0.1:9019/",
            "con_debater": "http://127.0.0.1:9018/"
        },
        "config": {
            "topic": "test code",
            "predicate": "test predicate",
            "target_verdict": "True",
            "target_dimension": "Security",
            "num_rounds": 1,
            "max_refinements": 0,
            "output_file": "test_output.jsonl"
        }
    }
    
    print(f"Sending test request to {judge_url}...")
    result = await send_message(json.dumps(payload), judge_url)
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
