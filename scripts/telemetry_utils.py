'''
Shared telemetry and token spend tracking helpers used across CI scripts.
'''

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from agentbeats.tracing import trace_span
except ImportError:
    import contextlib
    @contextlib.contextmanager
    def trace_span(*a, **kw):
        yield None


def _write_telemetry_payload(metrics_root: Path, payload: Dict[str, Any]) -> Path:
    metrics_path = metrics_root / "token_spend.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    return metrics_path


def _update_cassette_metadata(cassette_path: Path, usage_key: str, payload: Dict[str, Any]) -> None:
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


def _print_validation_summary(val_sum: Dict[str, Any]) -> None:
    details = val_sum.get("details", {})
    structured = details.get("structured_output", {})
    provider = details.get("provider_error", {})
    print(
        "\033[94mValidation Summary: "
        f"{val_sum.get('valid_units', 0)} valid, "
        f"{val_sum.get('repaired_units', 0)} repaired, "
        f"{val_sum.get('normalized_units', 0)} normalized, "
        f"{val_sum.get('invalid_units', 0)} invalid "
        f"({details.get('repaired_confidence_count', 0)} conf repairs, "
        f"{details.get('repaired_score_count', 0)} score repairs, "
        f"{details.get('normalized_path_count', 0)} path normalizations, "
        f"{details.get('normalized_text_count', 0)} text normalizations, "
        f"{structured.get('repair_attempts', 0)} structured repair attempts, "
        f"{structured.get('validation_retries', 0)} structured retries, "
        f"{structured.get('final_failures', 0)} structured final failures, "
        f"{provider.get('failures', 0)} provider failures)"
        "\033[0m"
    )


def _print_review_coverage(cov: Dict[str, Any]) -> None:
    print(
        "\033[94mReview Coverage: "
        f"{cov.get('reviewed_units', 0)}/"
        f"{cov.get('total_extracted_units', 0)} unit(s) reviewed, "
        f"{cov.get('skipped_units', 0)} skipped across "
        f"{cov.get('batch_count', 0)} batch(es) "
        f"(max {cov.get('max_units_per_batch', 0)} unit(s)/batch, "
        f"max {cov.get('max_tokens_per_batch', 0)} tokens/batch)"
        "\033[0m"
    )


def _print_token_and_latency_usage(usage_summary: Dict[str, Any]) -> None:
    totals = usage_summary.get("totals")
    if not isinstance(totals, dict):
        totals = {}
    total_dur = float(totals.get("total_duration_ms") or 0.0)
    avg_dur = float(totals.get("avg_duration_ms") or 0.0)
    max_dur = float(totals.get("max_duration_ms") or 0.0)
    print(
        "\033[94mToken usage & Latency: "
        f"{totals.get('total_tokens', 0)} total tokens "
        f"({totals.get('prompt_tokens', 0)} prompt, "
        f"{totals.get('completion_tokens', 0)} completion) across "
        f"{totals.get('calls', 0)} call(s); "
        f"latency: {total_dur/1000.0:.2f}s total (avg {avg_dur:.1f}ms, max {max_dur:.1f}ms); "
        f"estimated cost ${float(totals.get('cost_usd', 0.0) or 0.0):.6f}"
        "\033[0m"
    )


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
    **kwargs: Any,
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

    validation_summary = kwargs.get("validation_summary")
    review_coverage = kwargs.get("review_coverage")

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
    if validation_summary is not None:
        payload["validation_summary"] = validation_summary
    if review_coverage is not None:
        payload["review_coverage"] = review_coverage

    metrics_path = _write_telemetry_payload(metrics_root, payload)
    _update_cassette_metadata(cassette_path, usage_key, payload)

    if validation_summary is not None:
        _print_validation_summary(validation_summary)

    if review_coverage is not None:
        _print_review_coverage(review_coverage)

    _print_token_and_latency_usage(usage_summary)

    try:
        rel_metrics = metrics_path.relative_to(project_root)
    except ValueError:
        rel_metrics = metrics_path
    print(f"\033[92mSaved token spend metrics to {rel_metrics}\033[0m")
    return payload


def coerce_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    try:
        return int(float(value)) if isinstance(value, str) else int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def coerce_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
