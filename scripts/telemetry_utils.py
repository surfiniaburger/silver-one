'''
Shared telemetry and token spend tracking helpers used across CI scripts.
'''

import json
from pathlib import Path
from typing import Any, Dict, Optional


def persist_usage_artifacts(
    replay_mgr: Any,
    *,
    run_id: str,
    model: str,
    cassette_path: Path,
    reviewed_count: int,
    project_root: Path,
    metrics_root: Path,
    reviewed_key: str,
    usage_key: str,
) -> Optional[Dict[str, Any]]:
    '''
    Extracts usage summary from the replay manager, persists the telemetry payload
    to token_spend.jsonl under metrics_root, and writes the usage metadata back to
    the cassette JSON file.
    '''
    if replay_mgr is None or not hasattr(replay_mgr, "get_usage_summary"):
        return None

    usage_summary = replay_mgr.get_usage_summary()
    if not isinstance(usage_summary, dict):
        usage_summary = {}

    try:
        cassette_str = str(cassette_path.relative_to(project_root))
    except ValueError:
        cassette_str = str(cassette_path)

    payload = {
        "run_id": run_id,
        "model": model,
        reviewed_key: reviewed_count,
        "cassette": cassette_str,
        "usage": usage_summary,
    }

    metrics_path = metrics_root / "token_spend.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")

    try:
        cassette_data = {}
        if cassette_path.exists():
            with cassette_path.open("r", encoding="utf-8") as f:
                cassette_data = json.load(f)
        if not isinstance(cassette_data, dict):
            cassette_data = {}

        metadata = cassette_data.get("__metadata__")
        if not isinstance(metadata, dict):
            metadata = {}

        cassette_data["__metadata__"] = {
            **metadata,
            usage_key: payload,
        }
        with cassette_path.open("w", encoding="utf-8") as f:
            json.dump(cassette_data, f, indent=2)
    except Exception as exc:
        print(f"\033[93mWarning: failed to write cassette usage metadata: {exc}\033[0m")

    totals = usage_summary.get("totals")
    if not isinstance(totals, dict):
        totals = {}
    print(
        "\033[94mToken usage: "
        f"{totals.get('total_tokens', 0)} total "
        f"({totals.get('prompt_tokens', 0)} prompt, "
        f"{totals.get('completion_tokens', 0)} completion) across "
        f"{totals.get('calls', 0)} call(s); "
        f"estimated cost ${float(totals.get('cost_usd', 0.0) or 0.0):.6f}"
        "\033[0m"
    )
    print(f"\033[92mSaved token spend metrics to {metrics_path.relative_to(project_root)}\033[0m")
    return payload
