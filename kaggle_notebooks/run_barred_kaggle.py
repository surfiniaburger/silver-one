import subprocess
import os
import time
import sys

def run_command(cmd, shell=True):
    print(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")

def start_ollama():
    print("Starting Ollama...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(10)
    print("Pulling model...")
    run_command("ollama pull qwen2.5-coder:7b")

def setup_env():
    print("Setting up environment...")
    run_command("pip install uv litellm httpx pydantic uvicorn starlette")
    os.environ["PYTHONPATH"] = f".:{os.environ.get('PYTHONPATH', '')}"

def main():
    # 1. Setup
    setup_env()
    
    # Check if we are inside silver-one or parent
    if os.path.exists("scenarios/debate/barred_test.toml"):
        base_dir = "."
    elif os.path.exists("silver-one/scenarios/debate/barred_test.toml"):
        base_dir = "silver-one"
        os.chdir(base_dir)
        print(f"Changed directory to {base_dir}")
    else:
        # If running on Kaggle, the repo might be cloned into a folder
        print("Searching for scenarios/debate/barred_test.toml...")
        found = False
        for root, dirs, files in os.walk("."):
            if "scenarios/debate/barred_test.toml" in [os.path.join(r, f) for r, ds, fs in [(root, dirs, files)] for f in fs]:
                os.chdir(root.split("scenarios")[0])
                print(f"Found and moved to {os.getcwd()}")
                found = True
                break
        if not found:
            print("Error: Could not find silver-one directory structure.")
            return

    # 2. Start Ollama
    start_ollama()
    
    # 3. Start BARRED Agents
    print("Starting BARRED Agents...")
    env = os.environ.copy()
    model_name = "qwen2.5-coder:7b"
    env["JUDGE_MODEL"] = f"ollama/{model_name}"
    env["DEBATER_MODEL"] = f"ollama/{model_name}"
    
    judge_proc = subprocess.Popen(["uv", "run", "agentbeats-run", "--serve-only", "scenarios/debate/barred_test.toml"], 
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    print("Waiting for agents to initialize (60s)...")
    time.sleep(60)
    
    # 4. Run Batch Pilot
    print("Starting Batch Runner...")
    output_file = "/kaggle/working/test_corpus_50.jsonl"
    run_command(f"uv run python scenarios/debate/run_batch.py")
    
    print(f"Pilot run complete. Results should be in test_corpus_50.jsonl")
    
    judge_proc.terminate()

if __name__ == "__main__":
    main()
