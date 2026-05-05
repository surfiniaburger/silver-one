import subprocess
import os
import json
import time

def run_cmd(cmd):
    print(f"Running: {cmd}")
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def test_determinism():
    run_id = "test-det-123"
    seed = 999
    output_file = "test_determinism_output.jsonl"
    cassette_path = f"artifacts/cassettes/{run_id}.json"
    
    # Clean up
    if os.path.exists(output_file): os.remove(output_file)
    if os.path.exists(cassette_path): os.remove(cassette_path)
    
    # 1. Record Mode
    print("--- RECORD MODE ---")
    # We only run 1 seed for the test
    cmd_record = f"uv run python scenarios/debate/run_batch.py --run-id {run_id} --seed {seed} --mode record --seeds scenarios/debate/cve_seeds_test.jsonl --output {output_file}"
    res1 = run_cmd(cmd_record)
    if res1.returncode != 0:
        print(f"Record failed: {res1.stderr}")
        return
        
    with open(output_file, "r") as f:
        content1 = f.read()
    
    # 2. Replay Mode (Offline)
    print("\n--- REPLAY MODE ---")
    # Move the output so we don't skip
    os.remove(output_file)
    
    # We don't actually kill the servers here, but we set mode=replay which enforces no network in the code
    cmd_replay = f"uv run python scenarios/debate/run_batch.py --run-id {run_id} --seed {seed} --mode replay --seeds scenarios/debate/cve_seeds_test.jsonl --output {output_file}"
    res2 = run_cmd(cmd_replay)
    if res2.returncode != 0:
        print(f"Replay failed: {res2.stderr}")
        return

    with open(output_file, "r") as f:
        content2 = f.read()
        
    # 3. Compare
    if content1 == content2:
        print("\n✅ DETERMINISM TEST PASSED: Outputs are byte-identical.")
    else:
        print("\n❌ DETERMINISM TEST FAILED: Outputs differ.")
        # Print diff snippet
        print("Content 1:", content1[:100])
        print("Content 2:", content2[:100])

if __name__ == "__main__":
    test_determinism()
