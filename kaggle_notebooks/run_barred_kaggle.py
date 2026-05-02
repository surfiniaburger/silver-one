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
        # We don't exit here because some commands might fail but we want to continue

def start_ollama():
    print("Starting Ollama...")
    # Run ollama serve in the background
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(10) # Give it time to start
    print("Pulling model...")
    run_command("ollama pull qwen2.5-coder:7b")

def setup_env():
    print("Setting up environment...")
    run_command("pip install uv litellm httpx pydantic uvicorn starlette")
    # Add current dir to PYTHONPATH
    os.environ["PYTHONPATH"] = f".:{os.environ.get('PYTHONPATH', '')}"

def main():
    # 1. Setup
    setup_env()
    
    # 2. Start Ollama
    start_ollama()
    
    # 3. Start BARRED Agents
    print("Starting BARRED Agents...")
    env = os.environ.copy()
    env["JUDGE_MODEL"] = "ollama/qwen2.5-coder:7b"
    env["DEBATER_MODEL"] = "ollama/qwen2.5-coder:7b"
    
    # We use subprocess.Popen to run agents in the background
    judge_proc = subprocess.Popen(["uv", "run", "agentbeats-run", "--serve-only", "scenarios/debate/barred_test.toml"], 
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Wait for agents to be ready (approx 1 minute for model loading)
    print("Waiting for agents to initialize...")
    time.sleep(60)
    
    # 4. Run Batch Pilot
    print("Starting Batch Runner...")
    # Point output to /kaggle/working
    output_file = "/kaggle/working/test_corpus_50.jsonl"
    
    # We might need to adjust the seeds path if we are running from a specific directory
    seeds_file = "scenarios/debate/cve_seeds_50.jsonl"
    
    run_command(f"uv run python scenarios/debate/run_batch.py")
    
    print(f"Pilot run complete. Results saved to {output_file}")
    
    # Kill background processes
    judge_proc.terminate()

if __name__ == "__main__":
    main()
