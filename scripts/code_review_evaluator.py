import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Literal

# Enable relative imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ReplayManager = None
LLMCassette = None
try:
    from agentbeats.replay import ReplayManager as RM, LLMCassette as LC
    ReplayManager = RM
    LLMCassette = LC
except ImportError:
    pass

from scripts import llm_adapter
from scripts import diff_extractor
from scripts import telemetry_utils
from scripts.finding_schema import EngineeringFinding

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASSETTE_ROOT = (PROJECT_ROOT / "artifacts" / "cassettes").resolve()
RUN_ROOT = (PROJECT_ROOT / "artifacts" / "runs").resolve()
METRICS_ROOT = (PROJECT_ROOT / "artifacts" / "metrics").resolve()

CASSETTE_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_ROOT.mkdir(parents=True, exist_ok=True)


class PropertyEvaluation(BaseModel):
    score: float = Field(..., description="Score 0-10, 10 is perfect.")
    rationale: str = Field(..., description="1-2 sentences justifying the score.")
    suggestions: List[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, v):
        if v is None:
            raise ValueError("Score cannot be None.")
        if isinstance(v, bool):
            raise ValueError("Score cannot be a boolean value.")
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError("Score must be a numeric value.")
        
        if 0.0 <= val <= 10.0:
            return val
        elif 10.0 < val <= 100.0:
            return val / 10.0
        else:
            raise ValueError("Score is out of bounds [0.0, 10.0].")

    @field_validator("rationale", mode="before")
    @classmethod
    def validate_rationale(cls, v):
        if v is None:
            raise ValueError("rationale cannot be None.")
        if not isinstance(v, str):
            raise ValueError("rationale must be a string.")
        return v.strip()


class CodeReviewBreakdown(BaseModel):
    readability: PropertyEvaluation
    maintainability: PropertyEvaluation
    correctness: PropertyEvaluation
    complexity: PropertyEvaluation
    security: PropertyEvaluation
    test_coverage: PropertyEvaluation
    summary: str = Field("", description="A brief summary of the overall code quality.")
    severity: Literal["OK", "WARN", "BLOCK"] = Field("OK", description="Top-level verdict per unit.")
    findings: List[EngineeringFinding] = Field(
        default_factory=list,
        description="List of structured engineering findings supporting this evaluation."
    )

    @field_validator("summary", mode="before")
    @classmethod
    def validate_summary(cls, v):
        if v is None:
            raise ValueError("summary cannot be None.")
        if not isinstance(v, str):
            raise ValueError("summary must be a string.")
        return v.strip()

    @field_validator("findings", mode="before")
    @classmethod
    def drop_invalid_findings(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("findings must be a list.")

        valid_findings = []
        for finding in v:
            if isinstance(finding, EngineeringFinding):
                valid_findings.append(finding)
            elif isinstance(finding, dict):
                try:
                    valid_findings.append(EngineeringFinding.model_validate(finding))
                except Exception:
                    continue
        return valid_findings


SYSTEM_PROMPT = """You are an elite, senior software engineer and security auditor.
Your task is to review a given python code snippet (function, method, or module) and evaluate it along the following dimensions:
1. Readability: Naming clarity, clarity of intent, presence/absence of docstrings and comments.
2. Maintainability: Level of coupling, separation of concerns, modularity.
3. Correctness: Logic errors, unexpected behaviors, handling of edge cases, proper exception management.
4. Complexity: Nesting depth, branch complexity, cyclomatic complexity.
5. Security: Vulnerabilities, path traversal, injection, resource leakage, unvalidated inputs.
6. Test Coverage: Does the function look like it's designed to be easily testable?

For each dimension, output a score between 0 and 10 (where 10 is perfect quality), a short rationale, and optional list of concrete suggestions.
Finally, determine a top-level severity:
- OK: The code is solid and safe to merge.
- WARN: There are issues that should be improved, but they do not block merging.
- BLOCK: Critical logic errors, major security vulnerabilities, or severe structural issues that must be fixed before merging.

For any issues found (especially for WARN or BLOCK severity), you must also populate the `findings` list with structured engineering findings. Each finding should explicitly specify:
- A concise title summarizing the finding.
- The category (e.g. Readability, Maintainability, Correctness, Complexity, Security, Testability, Null Safety, Performance, API Evolution).
- The severity (INFO, WARN, or BLOCK).
- Evidence pointing to the location_type ('code'), path (file path), and details (like function_name, start_line, end_line).
- The engineering rationale (why the issue exists).
- The engineering consequence (what happens if the issue is ignored).
- An impact evaluation across code quality domains (correctness, compatibility, security, maintainability, performance) with values NONE, LOW, MEDIUM, or HIGH.
- Concrete recommended action to resolve it.
- A numeric confidence score from 0.0 to 1.0.

Provide constructive, specific comments. Cite specific variables, lines, or constructs where appropriate.
"""


def estimate_pr_tokens(units: List[Dict[str, Any]]) -> int:
    # Estimate: ~4 characters per token + 400 fixed overhead tokens per unit (system prompt, overhead)
    return sum(len(unit["code"]) // 4 + 400 for unit in units)


def filter_units_by_budget(
    units: List[Dict[str, Any]], max_tokens: int, max_units: int
) -> Tuple[List[Dict[str, Any]], int]:
    """Sort units by size/lines modified and truncate to fit the token budget and unit limit."""
    sorted_units = sorted(units, key=lambda u: u.get("lines_changed", 0), reverse=True)
    selected: List[Dict[str, Any]] = []
    current_tokens = 0

    for unit in sorted_units:
        if len(selected) >= max_units:
            break
        unit_tokens = len(unit["code"]) // 4 + 400
        if current_tokens + unit_tokens > max_tokens:
            if not selected:  # Always evaluate at least one unit if there are any
                selected.append(unit)
                current_tokens += unit_tokens
            continue
        selected.append(unit)
        current_tokens += unit_tokens

    return selected, current_tokens


def format_unit(unit: Dict[str, Any]) -> str:
    class_info = f" (in Class {unit['class_name']})" if unit.get("class_name") else ""
    return f"""File: {unit['file_path']}
Unit: {unit['name']}{class_info}
Line range: {unit['start_line']} - {unit['end_line']}
Lines changed: {unit['lines_changed']}

Code:
```python
{unit['code']}
```
"""


def _init_replay_manager(
    run_id: str,
    seed: int,
    cassette_path: str,
    mode: str,
    model: str,
) -> Optional[Any]:
    if ReplayManager is None:
        return None
    model_config = {"default": model}
    return ReplayManager.from_config(
        run_id=run_id,
        seed=seed,
        cassette_path=cassette_path,
        mode=mode,
        model_config=model_config,
    )


def _process_field_telemetry(field: Dict[str, Any], summary: Dict[str, Any]) -> None:
    status = field.get("status")
    field_name = field.get("field_name", "")
    if status == "REPAIRED":
        if "confidence" in field_name:
            summary["details"]["repaired_confidence_count"] += 1
        elif "score" in field_name:
            summary["details"]["repaired_score_count"] += 1
        elif field.get("repair_type") == "dropped_invalid_finding":
            summary["details"]["dropped_finding_count"] += 1
        elif field.get("repair_type") == "missing_default":
            summary["details"]["repaired_default_count"] += 1
    elif status == "NORMALIZED":
        if "path" in field_name:
            summary["details"]["normalized_path_count"] += 1
        else:
            summary["details"]["normalized_text_count"] += 1
    elif status == "INVALID":
        summary["details"]["invalid_field_count"] += 1


def _empty_structured_output_summary() -> Dict[str, int]:
    return {
        "invalid_json_detected": 0,
        "repair_attempts": 0,
        "repair_successes": 0,
        "validation_retries": 0,
        "final_failures": 0,
    }


def _process_structured_output_telemetry(unit: Dict[str, Any], summary: Dict[str, Any]) -> bool:
    diagnostics = unit.get("structured_output")
    if not isinstance(diagnostics, dict):
        return False

    structured = summary["details"]["structured_output"]
    if diagnostics.get("invalid_json_detected"):
        structured["invalid_json_detected"] += 1
    structured["repair_attempts"] += int(diagnostics.get("repair_attempts", 0) or 0)
    if diagnostics.get("repair_succeeded"):
        structured["repair_successes"] += 1
    structured["validation_retries"] += int(diagnostics.get("validation_retries", 0) or 0)
    if diagnostics.get("final_failure"):
        structured["final_failures"] += 1
        summary["invalid_units"] += 1
        return True
    return False


def _has_invalid_validation_field(fields: List[Dict[str, Any]]) -> bool:
    return any(field.get("status") == "INVALID" for field in fields)


def _record_validation_terminal_state(validation: Dict[str, Any], fields: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Record one mutually exclusive terminal state for a reviewed unit."""
    if _has_invalid_validation_field(fields):
        summary["invalid_units"] += 1
    elif validation.get("repaired", False):
        summary["repaired_units"] += 1
    elif validation.get("normalized", False):
        summary["normalized_units"] += 1
    else:
        summary["valid_units"] += 1


def build_validation_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate validation telemetry with one terminal state per reviewed unit."""
    summary = {
        "total_units": len(results),
        "valid_units": 0,
        "repaired_units": 0,
        "normalized_units": 0,
        "invalid_units": 0,
        "details": {
            "repaired_confidence_count": 0,
            "repaired_score_count": 0,
            "repaired_default_count": 0,
            "dropped_finding_count": 0,
            "normalized_path_count": 0,
            "normalized_text_count": 0,
            "invalid_field_count": 0,
            "structured_output": _empty_structured_output_summary(),
        }
    }
    for unit in results:
        if _process_structured_output_telemetry(unit, summary):
            continue

        validation = unit.get("validation")
        if not validation:
            summary["valid_units"] += 1
            continue

        fields = validation.get("fields", [])
        _record_validation_terminal_state(validation, fields, summary)
        for field in fields:
            _process_field_telemetry(field, summary)
                
    return summary


def persist_usage_artifacts(
    replay_mgr: Any,
    *,
    run_id: str,
    model: str,
    cassette_path: Path,
    reviewed_count: int,
    validation_summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    return telemetry_utils.persist_usage_artifacts(
        replay_mgr,
        run_id=run_id,
        model=model,
        cassette_path=cassette_path,
        reviewed_count=reviewed_count,
        project_root=PROJECT_ROOT,
        metrics_root=METRICS_ROOT,
        reviewed_key="reviewed_units",
        usage_key="code_review_usage_summary",
        **{"validation_summary": validation_summary}
    )


async def evaluate_units(
    replay_mgr: Any, model: str, units: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    from scripts.finding_schema import UnitReviewArtifact, build_validation_context
    results = []
    print(f"\033[94mEvaluating {len(units)} code units...\033[0m")

    for i, unit in enumerate(units, 1):
        name = unit["name"]
        file_path = unit["file_path"]
        print(f"[{i}/{len(units)}] Reviewing {name} in {file_path}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_unit(unit)},
        ]

        try:
            breakdown, raw_str, structured_diagnostics = await llm_adapter.call_structured_with_raw_and_diagnostics(
                replay_manager=replay_mgr,
                model=model,
                messages=messages,
                schema_name="CodeReviewBreakdown",
                schema_model=CodeReviewBreakdown,
                stage="code_review",
            )
            # Parse raw response as JSON to inspect raw values for validation context
            try:
                raw_json = json.loads(raw_str)
            except Exception:
                raw_json = {}
                
            context = build_validation_context(raw_json, breakdown)
            artifact = UnitReviewArtifact(
                file_path=file_path,
                name=name,
                review=breakdown,
                validation=context,
                raw_response=raw_str
            )
            artifact_payload = artifact.model_dump()
            artifact_payload["structured_output"] = structured_diagnostics
            results.append(artifact_payload)
        except llm_adapter.StructuredOutputError as exc:
            print(f"\033[91mRecoverable structured output failure for {name} in {file_path}: {exc}\033[0m")
            failure_diagnostics = dict(exc.diagnostics)
            failure_diagnostics["final_failure"] = True
            results.append({
                "file_path": file_path,
                "name": name,
                "review": None,
                "validation": {
                    "repaired": False,
                    "normalized": False,
                    "fields": [
                        {
                            "field_name": "llm_response",
                            "status": "INVALID",
                            "raw_value": exc.raw_response,
                            "repaired_value": None,
                            "repair_type": "structured_output",
                            "repair_reason": failure_diagnostics.get("failure_reason"),
                        }
                    ],
                },
                "raw_response": exc.raw_response,
                "structured_output": failure_diagnostics,
                "recoverable_failure": {
                    "type": "structured_output",
                    "schema_name": exc.schema_name,
                    "message": str(exc),
                },
            })
        except Exception as exc:
            print(f"\033[91mError reviewing {name} in {file_path}: {exc}\033[0m")

    return results


def save_review_cassette(cassette_path: Path, results: List[Dict[str, Any]]):
    """Save the reviews separately to the cassette if not using ReplayManager."""
    try:
        data = {}
        if cassette_path.exists():
            try:
                with cassette_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        if not isinstance(data, dict):
            data = {}

        data["reviews"] = results
        with cassette_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\033[92mSaved reviews to cassette at {cassette_path.relative_to(PROJECT_ROOT)}\033[0m")
    except Exception as exc:
        print(f"\033[91mError saving review cassette: {exc}\033[0m")


async def main_async():
    parser = argparse.ArgumentParser(description="MSEC Code Review Evaluator")
    parser.add_argument("--base", type=str, default="origin/main", help="Git base ref to diff against")
    parser.add_argument("--model", type=str, default="ollama/qwen3.5:2b", help="LLM model identifier")
    parser.add_argument("--run-id", type=str, default="code-review-eval", help="Identifier for run metadata")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    parser.add_argument("--mode", choices=["record", "replay"], default="record", help="Cassette mode")
    parser.add_argument(
        "--cassette",
        type=str,
        default="code_review.json",
        help="Cassette filename relative to artifacts/cassettes",
    )
    args = parser.parse_args()

    try:
        cassette_path = diff_extractor.validate_path(args.cassette, CASSETTE_ROOT)
        if cassette_path.suffix.lower() != ".json":
            raise ValueError("Cassette must have a .json extension.")
        safe_run_id = "".join([c if c.isalnum() or c in "-_" else "_" for c in args.run_id])
    except ValueError as e:
        print(f"\033[91mInvalid input: {e}\033[0m", file=sys.stderr)
        sys.exit(1)

    print(f"\033[94mExtracting changed units against {args.base}...\033[0m")
    try:
        all_units = diff_extractor.get_all_changed_units(args.base, PROJECT_ROOT)
    except ValueError as e:
        print(f"\033[91mInvalid base reference: {e}\033[0m", file=sys.stderr)
        sys.exit(1)

    if not all_units:
        print("\033[92mNo Python code modifications found. Skipping code review.\033[0m")
        # Write an empty cassette reviews section so compare tool doesn't fail
        save_review_cassette(cassette_path, [])
        sys.exit(0)

    total_est_tokens = estimate_pr_tokens(all_units)
    max_tokens = int(os.getenv("CR_MAX_TOKENS_PER_PR", "80000"))
    max_units = int(os.getenv("CR_MAX_UNITS", "20"))

    print(f"\033[94mTotal estimated tokens for changed code: {total_est_tokens}\033[0m")
    target_units, _ = filter_units_by_budget(all_units, max_tokens, max_units)

    if len(target_units) < len(all_units):
        print(
            f"\033[93mWarning: Token budget ({max_tokens}) or unit limit ({max_units}) exceeded. "
            f"Truncating evaluation from {len(all_units)} to {len(target_units)} units.\033[0m"
        )

    replay_mgr = _init_replay_manager(
        safe_run_id,
        args.seed,
        str(cassette_path),
        args.mode,
        args.model,
    )

    results = await evaluate_units(replay_mgr, args.model, target_units)

    val_summary = build_validation_summary(results)
    persist_usage_artifacts(
        replay_mgr,
        run_id=safe_run_id,
        model=args.model,
        cassette_path=cassette_path,
        reviewed_count=len(results),
        validation_summary=val_summary,
    )

    save_review_cassette(cassette_path, results)

    # Save run record if recording
    if args.mode == "record" and replay_mgr is not None:
        try:
            run_record_path = RUN_ROOT / safe_run_id / f"{args.seed}.json"
            run_record_path.parent.mkdir(parents=True, exist_ok=True)
            replay_mgr.save_record(str(run_record_path))
            print(f"\033[92mSaved run record to {run_record_path.relative_to(PROJECT_ROOT)}\033[0m")
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main_async())
