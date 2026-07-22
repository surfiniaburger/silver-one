import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.farley_score_evaluator import _init_replay_manager as farley_init_rm, _build_arg_parser as farley_parser
from scripts.code_review_evaluator import _init_replay_manager as cr_init_rm


def test_farley_arg_parser_clock_now():
    parser = farley_parser()
    args = parser.parse_args(["--clock-now", "2026-06-01T12:00:00Z"])
    assert args.clock_now == "2026-06-01T12:00:00Z"


def test_farley_init_replay_manager_clock_now(tmp_path):
    cassette_file = tmp_path / "test_cassette.json"
    rm = farley_init_rm(
        run_id="test-run",
        seed=42,
        cassette=str(cassette_file),
        mode="record",
        model="ollama/qwen3.5:2b",
        clock_now="2026-06-01T12:00:00Z",
    )
    if rm is not None:
        assert rm.run_record.created_at == "2026-06-01T12:00:00Z"


def test_code_review_init_replay_manager_clock_now(tmp_path):
    cassette_file = tmp_path / "test_cr_cassette.json"
    rm = cr_init_rm(
        run_id="test-cr-run",
        seed=42,
        cassette_path=str(cassette_file),
        mode="record",
        model="ollama/qwen3.5:2b",
        clock_now="2026-06-01T15:30:00Z",
    )
    if rm is not None:
        assert rm.run_record.created_at == "2026-06-01T15:30:00Z"
