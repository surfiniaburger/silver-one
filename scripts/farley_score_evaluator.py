#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
import ast
import json
import glob
from typing import List, Dict, Any, Optional

# Add the project src directory to sys.path to enable local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
# Ensure scripts dir is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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
                "class_name": self.current_class
            })
        self.generic_visit(node)

    # Ensure async test functions are captured as well (async def test_...)
    visit_AsyncFunctionDef = visit_FunctionDef


def extract_tests_from_file(filepath: str) -> List[Dict[str, Any]]:
    """Extract individual test functions/methods from a python file using AST."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"\033[91mError reading file {filepath}: {e}\033[0m")
        return []

    try:
        tree = ast.parse(content)
        extractor = TestExtractor(content)
        extractor.visit(tree)
        
        # Fallback: if no test cases are found using standard naming, treat the whole file as a single case
        if not extractor.test_cases:
            extractor.test_cases.append({
                "name": os.path.basename(filepath),
                "code": content,
                "class_name": None
            })
        return extractor.test_cases
    except SyntaxError as e:
        print(f"\033[91mSyntax error parsing {filepath}: {e}\033[0m")
        return []
    except Exception as e:
        print(f"\033[91mError parsing AST for {filepath}: {e}\033[0m")
        return []



# --- Helper functions (kept at module level) ---
def find_target_files(input_paths: List[str]) -> List[str]:
    paths = input_paths if input_paths else ["."]
    target_files: List[str] = []
    seen = set()

    def add_file(f):
        abs_f = os.path.abspath(f)
        if abs_f not in seen:
            seen.add(abs_f)
            target_files.append(f)

    for path in paths:
        if os.path.isfile(path) and path.endswith('.py'):
            add_file(path)
        elif os.path.isdir(path):
            for pattern in ["**/test_*.py", "**/*_test.py"]:
                for f in glob.glob(os.path.join(path, pattern), recursive=True):
                    add_file(f)
    return target_files


def _init_replay_manager(run_id: str, seed: int, cassette: str, mode: str, model: str):
    if ReplayManager is None:
        return None
    return ReplayManager.from_config(
        run_id=run_id,
        seed=seed,
        cassette_path=cassette,
        mode=mode,
        model_config={"evaluator": model}
    )


async def evaluate_files(replay_mgr: ReplayManager, model: str, target_files: List[str]):
    all_indices: List[float] = []
    reviewed_count = 0
    for filepath in target_files:
        print(f"\033[90mParsing test cases from {filepath}...\033[0m")
        test_cases = extract_tests_from_file(filepath)
        if not test_cases:
            continue
        print(f"\033[90mEvaluating {len(test_cases)} case(s) in {os.path.basename(filepath)}...\033[0m")
        for tc in test_cases:
            try:
                report = await evaluate_test_case(replay_mgr, model, tc, filepath)
                idx = display_report(filepath, tc["name"], tc.get("class_name"), report)
                all_indices.append(idx)
                reviewed_count += 1
            except Exception as e:
                print(f"\033[91mFailed to evaluate test case {tc.get('name')}: {e}\033[0m")
    return all_indices, reviewed_count


async def main_async():
    parser = argparse.ArgumentParser(description="MSEC Farley Score Evaluator")
    parser.add_argument("paths", nargs="*", help="Python test files or directories to review")
    parser.add_argument("--model", type=str, default=os.getenv("FARLEY_EVALUATOR_MODEL", "ollama/gpt-oss:120b-cloud"), help="LiteLLM model string")
    parser.add_argument("--run-id", type=str, default="farley-eval", help="Identifier for run metadata")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    parser.add_argument("--mode", choices=["record", "replay"], default="record", help="Cassette mode")
    parser.add_argument("--cassette", type=str, default="artifacts/cassettes/farley_score.json", help="Path to replay cassette JSON")
    args = parser.parse_args()

    target_files = find_target_files(args.paths)
    if not target_files:
        print("\033[91mNo python test files found to evaluate.\033[0m")
        sys.exit(1)

    print(f"\033[94mFound {len(target_files)} test file(s) to evaluate.\033[0m")
    print(f"\033[94mUsing model: {args.model} | Mode: {args.mode}\033[0m\n")

    replay_mgr = _init_replay_manager(args.run_id, args.seed, args.cassette, args.mode, args.model)
    if args.mode == "replay" and replay_mgr is None:
        print("\033[91mError: Replay mode requested but ReplayManager (agentbeats) is not available.\033[0m")
        sys.exit(1)

    all_indices, reviewed_count = await evaluate_files(replay_mgr, args.model, target_files)

    # Save replay cassette if recording
    if args.mode == "record" and replay_mgr is not None:
        os.makedirs(os.path.dirname(args.cassette), exist_ok=True)
        run_record_path = f"artifacts/runs/{args.run_id}/{args.seed}.json"
        try:
            replay_mgr.save_record(run_record_path)
            print(f"\033[92mSaved run record to {run_record_path}\033[0m")
        except Exception:
            pass

    if reviewed_count > 0:
        suite_average = sum(all_indices) / len(all_indices)
        avg_color = get_color_for_score(suite_average)
        print("=" * 80)
        print("\033[1mFINAL TEST SUITE SUMMARY\033[0m")
        print(f"Total Reviewed Test Cases: {reviewed_count}")
        print(f"Suite Farley Index Average: {avg_color}{suite_average:.2f}/10\033[0m")
        print("=" * 80)
    else:
        print("\033[93mNo test cases were successfully evaluated.\033[0m")

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
        "Make your feedback highly critical, professional, and actionable."
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
