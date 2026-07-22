#!/usr/bin/env python3
import os
import re
import sys
import argparse
import asyncio
import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Callable

TEST_PATTERNS = ("**/test_*.py", "**/*_test.py")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_ROOT = (PROJECT_ROOT / "tests").resolve()
CASSETTE_ROOT = (PROJECT_ROOT / "artifacts" / "cassettes").resolve()
RUN_ROOT = (PROJECT_ROOT / "artifacts" / "runs").resolve()
METRICS_ROOT = (PROJECT_ROOT / "artifacts" / "metrics").resolve()

SAFE_RUN_ID = re.compile(r"[^a-zA-Z0-9._-]")

TEST_ROOT.mkdir(parents=True, exist_ok=True)
CASSETTE_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_ROOT.mkdir(parents=True, exist_ok=True)

# Add the project src directory to sys.path to enable local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
# Ensure scripts dir is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# Ensure project root is importable so that diff_extractor can resolve
# its own `from scripts.path_utils import ...` dependency.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ReplayManager = None
try:
    from agentbeats.replay import ReplayManager as RM
    ReplayManager = RM
except Exception:
    pass
from llm_adapter import call_structured
from path_utils import validate_path
from telemetry_utils import persist_usage_artifacts as telemetry_persist_usage_artifacts
try:
    from agentbeats.tracing import trace_span
except Exception:
    import contextlib
    @contextlib.contextmanager
    def trace_span(*a, **kw):
        yield None

def log_info(msg: str) -> None:
    print(f"\033[94m{msg}\033[0m")

def log_success(msg: str) -> None:
    print(f"\033[92m{msg}\033[0m")

def log_warn(msg: str) -> None:
    print(f"\033[93m{msg}\033[0m")

def log_error(msg: str) -> None:
    print(f"\033[91m{msg}\033[0m")

def log_debug(msg: str) -> None:
    print(f"\033[90m{msg}\033[0m")

# diff_extractor is optional — only needed when --base is supplied.
_get_changed_lines: Optional[Callable[[str, Path], Dict[str, List[int]]]] = None
try:
    from diff_extractor import get_changed_lines as _get_changed_lines_impl
    _get_changed_lines = _get_changed_lines_impl
    _DIFF_AVAILABLE = True
except ImportError:
    _DIFF_AVAILABLE = False

try:
    from pydantic import BaseModel, Field

    class PropertyEvaluation(BaseModel):
        score: int = Field(..., description="Score from 0 to 10 (integer) where 10 is perfect.")
        rationale: str = Field(..., description="1-2 sentences justifying the score.")
        suggestions: List[str] = Field(default_factory=list, description="Actionable recommendations to improve this property (empty if score is 10).")

    class FarleyScoreBreakdown(BaseModel):
        understandable: PropertyEvaluation = Field(..., description="Test readability and use of domain-specific language.")
        maintainable: PropertyEvaluation = Field(..., description="Decoupling from implementation details, resilience to refactoring.")
        repeatable: PropertyEvaluation = Field(..., description="Determinism, lack of reliance on shared/external state, networks, or file systems.")
        atomic: PropertyEvaluation = Field(..., description="Testing exactly one requirement or behavior.")
        necessary: PropertyEvaluation = Field(..., description="Unique value added by the test, avoiding redundancy.")
        granular: PropertyEvaluation = Field(..., description="Focused scope and high precision of assertion targets.")
        fast: PropertyEvaluation = Field(..., description="Execution speed and avoidance of unnecessary waits/sleeps.")
        first_tdd: PropertyEvaluation = Field(..., description="Design quality suggesting a behavior-first TDD approach.")
        summary: str = Field(..., description="General summary of the test case quality and style.")
except Exception:
    # Minimal fallback dataclasses so tests can import the module without pydantic
    from dataclasses import dataclass

    def field_stub(*a, **kw):
        return None
    Field = field_stub

    @dataclass
    class PropertyEvaluation:
        score: int
        rationale: str
        suggestions: List[str]

    @dataclass
    class FarleyScoreBreakdown:
        understandable: PropertyEvaluation
        maintainable: PropertyEvaluation
        repeatable: PropertyEvaluation
        atomic: PropertyEvaluation
        necessary: PropertyEvaluation
        granular: PropertyEvaluation
        fast: PropertyEvaluation
        first_tdd: PropertyEvaluation
        summary: str
        @classmethod
        def model_validate_json(cls, raw_json: str):
            payload = json.loads(raw_json)

            def _mk(pe):
                return PropertyEvaluation(score=int(pe.get('score', 0)), rationale=pe.get('rationale', ''), suggestions=pe.get('suggestions', []))

            return cls(
                understandable=_mk(payload['understandable']),
                maintainable=_mk(payload['maintainable']),
                repeatable=_mk(payload['repeatable']),
                atomic=_mk(payload['atomic']),
                necessary=_mk(payload['necessary']),
                granular=_mk(payload['granular']),
                fast=_mk(payload['fast']),
                first_tdd=_mk(payload['first_tdd']),
                summary=payload.get('summary', '')
            )


# --- AST Parser for Extracting Tests ---

class TestExtractor(ast.NodeVisitor):
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.test_cases = []
        self.current_class = None

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        if node.name.startswith("Test") or node.name.endswith("Test"):
            self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        is_test = node.name.startswith("test_")
        if is_test or (self.current_class and node.name.startswith("test_")):
            code_segment = ast.get_source_segment(self.source_code, node)
            self.test_cases.append({
                "name": node.name,
                "code": code_segment or "",
                "class_name": self.current_class,
                # Line-range metadata required for diff-intersection filtering.
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
            })
        self.generic_visit(node)

    # Ensure async test functions are captured as well (async def test_...)
    visit_AsyncFunctionDef = visit_FunctionDef


def sanitize_run_id(run_id: str) -> str:
    return SAFE_RUN_ID.sub("_", run_id)

def extract_tests_from_file(
    filepath: str,
    changed_lines: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Extract individual test functions/methods from a python file using AST.

    Args:
        filepath: Absolute path to the test file.
        changed_lines: When provided (diff-only mode), only test functions whose
            source-line range intersects this set are returned.  Pass ``None``
            to return every test in the file (full-suite mode).
    """

    try:
        safe_path = validate_path(
            filepath,
            TEST_ROOT,
            {".py"},
        )

        with safe_path.open("r", encoding="utf-8") as f:
            content = f.read()

    except Exception as e:
        log_error(f"Error reading file {filepath}: {e}")
        return []

    try:
        tree = ast.parse(content)
        extractor = TestExtractor(content)
        extractor.visit(tree)

        if not extractor.test_cases:
            extractor.test_cases.append(
                {
                    "name": safe_path.name,
                    "code": content,
                    "class_name": None,
                }
            )

        # --- Diff-intersection filter (Seam B) ---
        # When changed_lines is provided, keep only test functions whose line
        # range overlaps with the modified lines.  Test cases that lack line
        # metadata (e.g. the module-level fallback above) are kept conservatively.
        if changed_lines is not None:
            changed_set: Set[int] = set(changed_lines)
            filtered: List[Dict[str, Any]] = []
            for tc in extractor.test_cases:
                sl = tc.get("start_line")
                el = tc.get("end_line")
                # Include if: no line metadata (conservative) OR range overlaps the diff.
                if sl is None or el is None or any(line in changed_set for line in range(sl, el + 1)):
                    filtered.append(tc)
            extractor.test_cases = filtered


        return extractor.test_cases

    except SyntaxError as e:
        log_error(f"Syntax error parsing {safe_path}: {e}")
        return []

    except Exception as e:
        log_error(f"Error parsing AST for {safe_path}: {e}")
        return []






def add_file(path_obj: Path, target_files: List[str], seen: Set[str]) -> None:
    resolved = str(path_obj.resolve())

    if resolved not in seen:
        seen.add(resolved)
        target_files.append(resolved)


def collect_test_files(directory: Path) -> List[Path]:
    files: List[Path] = []

    for pattern in TEST_PATTERNS:
        files.extend(directory.glob(pattern))

    return files


def find_target_files(input_paths: List[str]) -> List[str]:
    paths = input_paths or ["."]
    target_files: List[str] = []
    seen: Set[str] = set()

    for path in paths:
        try:
            # Resolve relative paths against CWD first so that e.g.
            # "tests/test_foo.py" doesn't get double-prefixed as
            # TEST_ROOT/tests/test_foo.py inside validate_path.
            resolved_path = str(Path(path).resolve())
            safe_path = validate_path(resolved_path, TEST_ROOT)
        except ValueError as exc:
            log_error(f"Skipping invalid path '{path}': {exc}")
            continue

        if safe_path.is_file() and safe_path.suffix == ".py":
            add_file(safe_path, target_files, seen)
            continue

        if safe_path.is_dir():
            for file_path in collect_test_files(safe_path):
                add_file(file_path, target_files, seen)

    return target_files


def _init_replay_manager(run_id: str, seed: int, cassette: str, mode: str, model: str, clock_now: str = ""):
    if ReplayManager is None:
        return None
    return ReplayManager.from_config(
        run_id=run_id,
        seed=seed,
        cassette_path=cassette,
        mode=mode,
        model_config={"evaluator": model},
        created_at=clock_now.strip() or None,
    )


def serialize_breakdown(report: Any) -> Dict[str, Any]:
    # Already a plain dict (e.g. cached/mock response) — return as-is.
    if isinstance(report, dict):
        return report

    if hasattr(report, "model_dump"):
        return report.model_dump()

    # Manual serialization for fallback dataclasses
    def _pe_to_dict(pe):
        return {
            "score": pe.score,
            "rationale": pe.rationale,
            "suggestions": pe.suggestions
        }
    return {
        "understandable": _pe_to_dict(report.understandable),
        "maintainable": _pe_to_dict(report.maintainable),
        "repeatable": _pe_to_dict(report.repeatable),
        "atomic": _pe_to_dict(report.atomic),
        "necessary": _pe_to_dict(report.necessary),
        "granular": _pe_to_dict(report.granular),
        "fast": _pe_to_dict(report.fast),
        "first_tdd": _pe_to_dict(report.first_tdd),
        "summary": report.summary
    }


def _build_test_id(rel_filepath: str, class_name: Optional[str], test_name: str) -> str:
    """Construct a unique test ID matching the pytest naming pattern."""
    return f"{rel_filepath}::{class_name}::{test_name}" if class_name else f"{rel_filepath}::{test_name}"


async def _evaluate_single_test(
    replay_mgr: ReplayManager,
    model: str,
    tc: Dict[str, Any],
    filepath: str,
) -> Optional[Dict[str, Any]]:
    """Evaluate one test case and return a result dict, or None on failure."""
    try:
        report = await evaluate_test_case(replay_mgr, model, tc, filepath)
        idx = display_report(filepath, tc["name"], tc.get("class_name"), report)

        rel_filepath = filepath
        try:
            if PROJECT_ROOT:
                rel_filepath = str(Path(filepath).relative_to(PROJECT_ROOT))
        except (ValueError, TypeError):
            pass

        class_name = tc.get("class_name")
        test_name = tc["name"]
        test_id = _build_test_id(rel_filepath, class_name, test_name)

        # Build result entry before touching counters so a serialization failure
        # doesn't leave counters out of sync with results.
        serialized = serialize_breakdown(report)
        return {
            "id": test_id,
            "file_path": rel_filepath,
            "test_name": test_name,
            "class_name": class_name,
            "farley_index": idx,
            "farley_breakdown": serialized,
        }
    except Exception as e:
        log_error(f"Failed to evaluate test case {tc.get('name')}: {e}")
        return None


async def evaluate_files(
    replay_mgr: ReplayManager,
    model: str,
    target_files: List[str],
    changed_lines_by_file: Optional[Dict[str, List[int]]] = None,
) -> Tuple[List[float], int, List[Dict[str, Any]]]:
    """Evaluate test files, optionally restricting to diff-intersecting functions.

    Args:
        changed_lines_by_file: Maps absolute file path → list of changed line
            numbers (diff-only mode).  ``None`` means evaluate every test
            (full-suite mode, backward-compatible default).
    """
    if not target_files:
        return [], 0, []

    all_indices: List[float] = []
    reviewed_count = 0
    results = []
    for filepath in target_files:
        log_debug(f"Parsing test cases from {filepath}...")
        cl = changed_lines_by_file.get(filepath) if changed_lines_by_file is not None else None
        test_cases = extract_tests_from_file(filepath, cl)
        if not test_cases:
            continue
        log_debug(f"Evaluating {len(test_cases)} case(s) in {os.path.basename(filepath)}...")
        for tc in test_cases:
            result = await _evaluate_single_test(replay_mgr, model, tc, filepath)
            if result is not None:
                results.append(result)
                all_indices.append(result["farley_index"])
                reviewed_count += 1
    return all_indices, reviewed_count, results



def save_farley_cassette(cassette_path: Path, results: List[Dict[str, Any]]):
    """Save the test evaluations to the cassette using an atomic write (temp + os.replace)
    so that an interrupted write never leaves a half-corrupt JSON file.
    """
    import tempfile
    try:
        data: Dict[str, Any] = {}
        if cassette_path.exists():
            try:
                with cassette_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        if not isinstance(data, dict):
            data = {}

        data["tests"] = results

        # Atomic write: write to a temp file in the same directory, then
        # os.replace() it over the target — rename is atomic on POSIX.
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(cassette_path.parent),
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        )
        try:
            json.dump(data, tmp, indent=2)
            tmp.close()
            os.replace(tmp.name, str(cassette_path))
        except Exception:
            # Clean up the temp file if the replace fails.
            try:
                os.remove(tmp.name)
            except OSError:
                pass
            raise

        try:
            rel_cassette = cassette_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_cassette = cassette_path
        log_success(f"Saved tests to cassette at {rel_cassette}")
    except Exception as exc:
        log_error(f"Error saving Farley cassette: {exc}")




def persist_usage_artifacts(
    replay_mgr: ReplayManager,
    *,
    run_id: str,
    model: str,
    cassette_path: Path,
    reviewed_count: int,
) -> Optional[Dict[str, Any]]:
    return telemetry_persist_usage_artifacts(
        replay_mgr,
        run_id=run_id,
        model=model,
        cassette_path=cassette_path,
        reviewed_count=reviewed_count,
        project_root=PROJECT_ROOT,
        metrics_root=METRICS_ROOT,
        reviewed_key="reviewed_test_cases",
        usage_key="farley_usage_summary",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MSEC Farley Score Evaluator")
    parser.add_argument("paths", nargs="*", help="Python test files or directories to review")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("FARLEY_EVALUATOR_MODEL", "ollama/gpt-oss:120b-cloud"),
        help="LiteLLM model string",
    )
    parser.add_argument("--run-id", type=str, default="farley-eval", help="Identifier for run metadata")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    parser.add_argument("--mode", choices=["record", "replay"], default="record", help="Cassette mode")
    parser.add_argument(
        "--cassette",
        type=str,
        default="farley_score.json",
        help="Cassette filename relative to artifacts/cassettes",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help=(
            "Git ref to diff against (e.g. 'origin/main'). "
            "When set, only test functions whose source lines intersect "
            "the diff are evaluated (diff-only mode). "
            "Omit to evaluate all tests in the supplied paths (full-suite mode)."
        ),
    )
    parser.add_argument(
        "--clock-now",
        type=str,
        default=os.getenv("RUN_CLOCK_NOW", ""),
        help="Inject a fixed ISO timestamp for run records/cassettes",
    )
    return parser


def _parse_and_validate_args(
    parser: argparse.ArgumentParser,
    args_list: Optional[List[str]] = None,
) -> Tuple[argparse.Namespace, Path, str]:
    """Parse CLI args and validate the cassette path and run-id. Exits on error."""
    args = parser.parse_args(args_list)
    try:
        if not hasattr(args, "cassette") or not args.cassette:
            raise ValueError("Cassette path argument is missing or empty.")
        if not hasattr(args, "run_id") or not args.run_id:
            raise ValueError("Run ID argument is missing or empty.")

        cassette_path = validate_path(args.cassette, CASSETTE_ROOT, {".json"})
        safe_run_id = sanitize_run_id(args.run_id)
    except (ValueError, TypeError) as e:
        log_error(f"Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error during argument validation: {e}")
        sys.exit(1)
    return args, cassette_path, safe_run_id


def _filter_changed_test_files(raw_changed: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Filter a git-diff map to validated test files inside TEST_ROOT."""
    result: Dict[str, List[int]] = {}
    for rel_path, lines in raw_changed.items():
        p = Path(rel_path)
        if not (p.name.startswith("test_") or p.name.endswith("_test.py")):
            continue
        abs_path = PROJECT_ROOT / rel_path
        try:
            safe = validate_path(str(abs_path), TEST_ROOT, {".py"})
            # Verify containment explicitly for absolute paths
            safe.relative_to(TEST_ROOT)
            if safe.exists():
                result[str(safe)] = lines
        except ValueError:
            pass
    return result


def _resolve_target_files(
    args: argparse.Namespace,
    cassette_path: Path,
) -> Tuple[List[str], Optional[Dict[str, List[int]]]]:
    """Return ``(target_files, changed_lines_by_file)`` for the chosen mode.

    Writes an empty cassette and exits with code 0 when the diff produces no
    relevant test changes (so the rest of the pipeline still has a cassette
    to compare against).  Exits with code 1 for hard errors.
    """
    if args.base is None:
        # Full-suite mode — unchanged legacy behaviour.
        try:
            target_files = find_target_files(args.paths)
        except Exception as exc:
            log_error(f"Error finding target files: {exc}")
            sys.exit(1)
        if not target_files:
            log_error("No python test files found to evaluate.")
            sys.exit(1)
        return target_files, None

    # Diff-only mode.
    if not _DIFF_AVAILABLE or _get_changed_lines is None:
        log_error(
            "Error: --base requires diff_extractor to be importable. "
            "Ensure the scripts package is on sys.path."
        )
        sys.exit(1)

    log_info(f"Diff-only mode: extracting changed lines against '{args.base}'...")
    try:
        raw_changed: Dict[str, List[int]] = _get_changed_lines(args.base, PROJECT_ROOT)
    except ValueError as exc:
        log_error(f"Invalid base reference: {exc}")
        sys.exit(1)
    except Exception as exc:
        log_error(f"Error extracting diff: {exc}")
        sys.exit(1)

    test_file_changed_lines = _filter_changed_test_files(raw_changed) if raw_changed else {}

    if not raw_changed or not test_file_changed_lines:
        log_success("No relevant test file changes found in diff. Skipping Farley evaluation.")
        save_farley_cassette(cassette_path, [])
        sys.exit(0)

    target_files = list(test_file_changed_lines.keys())
    log_info(f"Diff-only: {len(target_files)} test file(s) with changes to evaluate.")
    return target_files, test_file_changed_lines


def _save_run_record(replay_mgr: Any, safe_run_id: str, seed: int) -> None:
    """Persist the replay cassette when recording mode is active."""
    if replay_mgr is None:
        return
    try:
        run_record_path = validate_path(
            f"{safe_run_id}/{seed}.json",
            RUN_ROOT,
            {".json"},
        )
        run_record_path.parent.mkdir(parents=True, exist_ok=True)
        replay_mgr.save_record(str(run_record_path))
        log_success(f"Saved run record to {run_record_path.relative_to(PROJECT_ROOT)}")
    except Exception as exc:
        log_warn(f"Warning: could not save run record: {exc}")



def _print_suite_summary(all_indices: List[float], reviewed_count: int) -> None:
    """Print the final suite-level Farley Index summary."""
    if reviewed_count > 0:
        suite_average = sum(all_indices) / len(all_indices)
        avg_color = get_color_for_score(suite_average)
        print("=" * 80)
        print("\033[1mFINAL TEST SUITE SUMMARY\033[0m")
        print(f"Total Reviewed Test Cases: {reviewed_count}")
        print(f"Suite Farley Index Average: {avg_color}{suite_average:.2f}/10\033[0m")
        print("=" * 80)
    else:
        log_warn("No test cases were successfully evaluated.")


async def main_async() -> None:
    args, cassette_path, safe_run_id = _parse_and_validate_args(_build_arg_parser())

    target_files, changed_lines_by_file = _resolve_target_files(args, cassette_path)

    log_info(f"Found {len(target_files)} test file(s) to evaluate.")
    log_info(f"Using model: {args.model} | Mode: {args.mode}\n")

    replay_mgr = _init_replay_manager(
        safe_run_id, args.seed, str(cassette_path), args.mode, args.model, clock_now=getattr(args, "clock_now", "")
    )
    if args.mode == "replay" and replay_mgr is None:
        log_error("Error: Replay mode requested but ReplayManager (agentbeats) is not available.")
        sys.exit(1)

    all_indices, reviewed_count, results = await evaluate_files(
        replay_mgr, args.model, target_files, changed_lines_by_file
    )

    save_farley_cassette(cassette_path, results)
    persist_usage_artifacts(
        replay_mgr,
        run_id=safe_run_id,
        model=args.model,
        cassette_path=cassette_path,
        reviewed_count=reviewed_count,
    )

    if args.mode == "record" and replay_mgr is not None:
        _save_run_record(replay_mgr, safe_run_id, args.seed)

    _print_suite_summary(all_indices, reviewed_count)




# --- LLM Review Orchestrator ---

async def evaluate_test_case(
    replay_manager: ReplayManager,
    model: str,
    test_case: Dict[str, Any],
    filepath: str
) -> FarleyScoreBreakdown:
    """Send test code snippet to the LLM to get a structured Farley breakdown."""
    system_prompt = (
        "You are an expert Software Engineering Coach specializing in Continuous Delivery and Test-Driven Development (TDD).\n"
        "Your task is to analyze the provided test code snippet and evaluate it against Dave Farley's '8 Properties of Good Tests'.\n\n"
        "For each of the following 8 properties, provide a score (integer 0 to 10), a 1-2 sentence rationale, and specific suggestions:\n"
        "1. understandable: Clear intent/business rule, written in ubiquitous language of the domain, avoiding low-level implementation clutter.\n"
        "2. maintainable: Decoupled from system implementation details (tests 'what' not 'how'). Not brittle when system is refactored.\n"
        "3. repeatable: Deterministic, self-contained, no reliance on external shared state, database, network, local filesystem, or timing.\n"
        "4. atomic: Verifies exactly one behavior/rule. If it fails, the cause is instantly obvious.\n"
        "5. necessary: Has a unique purpose; does not duplicate coverage of other tests or test trivial things unnecessarily.\n"
        "6. granular: Targeted and precise in what it isolates and asserts.\n"
        "7. fast: Executes rapidly; avoids artificial sleeps/delays, heavy setups, or unneeded slow I/O operations.\n"
        "8. first_tdd: Demonstrates design properties of a test designed first (specifies behavior, clean separation of arrange-act-assert).\n\n"
        "Make your feedback highly critical, professional, and actionable.\n\n"
        "You must return your evaluation strictly as a valid JSON object matching the FarleyScoreBreakdown schema."
    )

    class_context = f" inside class {test_case['class_name']}" if test_case['class_name'] else ""
    user_prompt = (
        f"File Path: {filepath}\n"
        f"Test Case: {test_case['name']}{class_context}\n\n"
        f"Code:\n```python\n{test_case['code']}\n```"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # call_structured handles schema validation and cassette recording/replay automatically
    with trace_span("farley_eval_unit", stage="farley_evaluation", attributes={"file": filepath, "test": test_case.get("name")}):
        result = await call_structured(
            replay_manager=replay_manager,
            model=model,
            messages=messages,
            schema_name="FarleyScoreBreakdown",
            schema_model=FarleyScoreBreakdown,
            stage="farley_evaluation"
        )
    return result

# --- Reporting / Presentation ---

def get_color_for_score(score: float) -> str:
    """Return ANSI escape code corresponding to the score range."""
    if score >= 8.0:
        return "\033[92m"  # Green
    elif score >= 5.0:
        return "\033[93m"  # Yellow
    else:
        return "\033[91m"  # Red

def print_property_row(name: str, eval_obj: PropertyEvaluation):
    color = get_color_for_score(eval_obj.score)
    reset = "\033[0m"
    print(f"  - {name:<15}: {color}{eval_obj.score:>2}/10{reset} | {eval_obj.rationale}")
    if eval_obj.suggestions:
        for sug in eval_obj.suggestions:
            print(f"      * \033[90mSuggestion: {sug}\033[0m")

def display_report(filepath: str, test_name: str, class_name: Optional[str], report: FarleyScoreBreakdown):
    reset = "\033[0m"
    bold = "\033[1m"
    cyan = "\033[96m"
    
    context = f"{class_name}::" if class_name else ""
    full_name = f"{context}{test_name}"
    
    # Calculate overall index
    scores = [
        report.understandable.score,
        report.maintainable.score,
        report.repeatable.score,
        report.atomic.score,
        report.necessary.score,
        report.granular.score,
        report.fast.score,
        report.first_tdd.score,
    ]
    farley_index = sum(scores) / len(scores)
    index_color = get_color_for_score(farley_index)

    print("=" * 80)
    print(f"{bold}{cyan}TEST REVIEW: {full_name}{reset}")
    print(f"File: {filepath}")
    print(f"Overall Farley Index: {index_color}{farley_index:.1f}/10{reset}")
    print("-" * 80)
    
    print_property_row("Understandable", report.understandable)
    print_property_row("Maintainable", report.maintainable)
    print_property_row("Repeatable", report.repeatable)
    print_property_row("Atomic", report.atomic)
    print_property_row("Necessary", report.necessary)
    print_property_row("Granular", report.granular)
    print_property_row("Fast", report.fast)
    print_property_row("First (TDD)", report.first_tdd)
    
    print("-" * 80)
    print(f"{bold}Summary:{reset} {report.summary}")
    print("=" * 80)
    print()
    return farley_index

# (Main orchestration implemented in the refactored `main_async` above.)

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\033[93mEvaluation aborted by user.\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
