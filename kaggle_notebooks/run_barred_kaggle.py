import subprocess
import os
import time
import sys
import argparse
from pathlib import Path
import shlex


def run_command(cmd):
    print(f"Executing: {cmd}")
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")

def start_ollama(pull_models: list[str]):
    print("Starting Ollama...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(10)
    for m in pull_models:
        print(f"Pulling model: {m}")
        run_command(f"ollama pull {m}")

def setup_env():
    print("Setting up environment...")
    run_command("pip install uv litellm httpx pydantic uvicorn starlette")
    os.environ["PYTHONPATH"] = f".:{os.environ.get('PYTHONPATH', '')}"

def _wait_for_agent_cards(endpoints: list[str], timeout_s: int) -> bool:
    try:
        import httpx
    except Exception:
        print("httpx missing; installing...")
        run_command("pip install httpx")
        import httpx

    start = time.time()
    remaining = set(endpoints)
    while time.time() - start < timeout_s:
        for base in list(remaining):
            url = base.rstrip("/") + "/.well-known/agent-card.json"
            try:
                r = httpx.get(url, timeout=2.0)
                if r.status_code == 200:
                    remaining.discard(base)
            except Exception:
                pass
        if not remaining:
            return True
        print(f"Waiting on agent cards: {sorted(remaining)}")
        time.sleep(1)
    return False

def main():
    parser = argparse.ArgumentParser(description="Kaggle runner for BARRED (record/replay compatible)")
    parser.add_argument("--scenario", default="scenarios/debate/barred_test.toml", help="Scenario TOML path")
    parser.add_argument("--run-id", default=os.getenv("RUN_ID", "kaggle-run"))
    parser.add_argument("--mode", choices=["record", "replay"], default=os.getenv("MODE", "record"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--seeds", default=os.getenv("SEEDS_PATH", "scenarios/debate/cve_seeds_50.jsonl"))
    parser.add_argument("--output", default=os.getenv("OUTPUT_PATH", "/kaggle/working/training_corpus.jsonl"))
    parser.add_argument("--cassette-path", default=os.getenv("CASSETTE_PATH", ""), help="Optional cassette path; defaults to artifacts/cassettes/<run-id>.json inside the judge.")
    parser.add_argument("--record-path", default=os.getenv("RECORD_PATH", ""), help="Optional run record path; defaults to artifacts/runs/<run-id>.json inside the judge.")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "ollama/qwen2.5-coder:7b"))
    parser.add_argument("--debater-model", default=os.getenv("DEBATER_MODEL", "ollama/qwen2.5-coder:7b"))
    parser.add_argument("--generator-model", default=os.getenv("GENERATOR_MODEL", "ollama/qwen2.5-coder:7b"))
    parser.add_argument("--verifier-model", default=os.getenv("VERIFIER_MODEL", "ollama/qwen2.5-coder:7b"))
    parser.add_argument("--attempts-out", default=os.getenv("ATTEMPTS_OUT_PATH", "/kaggle/working/attempts.jsonl"))
    parser.add_argument("--metrics-out", default=os.getenv("METRICS_OUT_PATH", "/kaggle/working/b_gate.json"))
    parser.add_argument("--gepa-model", default=os.getenv("GEPA_MODEL", ""), help="Optional GEPA model for seed loader runs")
    parser.add_argument("--ollama-pull", default=os.getenv("OLLAMA_PULL", "qwen2.5-coder:7b"), help="Comma-separated model names to pull via ollama (no provider prefix)")
    parser.add_argument("--no-ollama", action="store_true", help="Skip starting/pulling ollama (use if models are remote)")
    parser.add_argument("--agents-timeout", type=int, default=int(os.getenv("AGENTS_TIMEOUT", "120")), help="Seconds to wait for agents")
    args = parser.parse_args()

    # 1. Setup
    setup_env()
    
    # Check if we are inside silver-one or parent
    if os.path.exists(args.scenario):
        base_dir = "."
    elif os.path.exists(str(Path("silver-one") / args.scenario)):
        base_dir = "silver-one"
        os.chdir(base_dir)
        print(f"Changed directory to {base_dir}")
    else:
        # If running on Kaggle, the repo might be cloned into a folder
        print(f"Searching for {args.scenario}...")
        found = False
        for root, dirs, files in os.walk("."):
            candidate = os.path.join(root, args.scenario)
            if os.path.exists(candidate):
                os.chdir(root)
                print(f"Found and moved to {os.getcwd()}")
                found = True
                break
        if not found:
            print("Error: Could not find silver-one directory structure.")
            return

    # 2. Start Ollama (optional)
    if not args.no_ollama and any(m.startswith("ollama/") for m in [args.judge_model, args.debater_model, args.generator_model, args.verifier_model] if m):
        pull_models = [m.strip() for m in args.ollama_pull.split(",") if m.strip()]
        start_ollama(pull_models)
    
    # 3. Start BARRED Agents
    print("Starting BARRED Agents...")
    env = os.environ.copy()
    env["JUDGE_MODEL"] = args.judge_model
    env["DEBATER_MODEL"] = args.debater_model
    env["GENERATOR_MODEL"] = args.generator_model
    env["VERIFIER_MODEL"] = args.verifier_model
    if args.gepa_model:
        env["GEPA_MODEL"] = args.gepa_model
    
    judge_proc = subprocess.Popen(["uv", "run", "agentbeats-run", "--serve-only", args.scenario],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    endpoints = ["http://127.0.0.1:9009", "http://127.0.0.1:9018", "http://127.0.0.1:9019", "http://127.0.0.1:9020"]
    print(f"Waiting for agents to initialize ({args.agents_timeout}s)...")
    if not _wait_for_agent_cards(endpoints, args.agents_timeout):
        print("Error: agents did not become ready in time.")
        judge_proc.terminate()
        return
    
    # 4. Run Batch Pilot
    print("Starting Batch Runner...")
    cassette_flag = f"--cassette-path {args.cassette_path}" if args.cassette_path else ""
    record_flag = f"--record-path {args.record_path}" if args.record_path else ""
    run_command(
        " ".join([
            "uv run python scenarios/debate/run_batch.py",
            f"--run-id {args.run_id}",
            f"--mode {args.mode}",
            f"--seed {args.seed}",
            f"--seeds {args.seeds}",
            f"--output {args.output}",
            f"--attempts-out {args.attempts_out}",
            cassette_flag,
            record_flag,
        ]).strip()
    )
    
    print(f"Pilot run complete. Results should be in {args.output}")
    
    # 5. Compute B metrics + soft-check rates
    print("Computing B metrics...")
    os.environ.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    run_command(
        " ".join([
            "uv run python scenarios/debate/offline_b_gate.py",
            f"--input {args.output}",
            f"--attempts {args.attempts_out}",
            f"--metrics-out {args.metrics_out}"
        ]).strip()
    )

    print(f"B metrics complete. Results should be in {args.metrics_out}")

    judge_proc.terminate()

if __name__ == "__main__":
    main()
