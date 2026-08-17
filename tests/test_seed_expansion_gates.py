import json
import pytest
from unittest.mock import AsyncMock, patch

from agentbeats.replay import ReplayManager
from scenarios.debate.cve_seed_loader import (
    CVESeedLoader,
    SeedExpansionConfig,
    _predicate_quality,
    _is_generic_anchor,
)

SAMPLE_VALID_CODE = """
int unimac_mdio_read(struct mii_bus *bus, int phy_id, int reg) {
    u32 cmd = MDIO_RD | (phy_id << MDIO_PMD_SHIFT) | (reg << MDIO_REG_SHIFT);
    unimac_mdio_writel(bus->priv, cmd, MDIO_CMD);
    return cmd;
}

int unimac_mdio_write(struct mii_bus *bus, int phy_id, int reg, u16 val) {
    u32 cmd = MDIO_WR | (phy_id << MDIO_PMD_SHIFT) | (reg << MDIO_REG_SHIFT) | val;
    unimac_mdio_writel(bus->priv, cmd, MDIO_CMD);
    return 0;
}
"""

SAMPLE_CODE_DUPLICATE_COMMENTS = (
    "// Broadcom UniMAC MDIO bus controller driver\n"
    "/* Copyright Broadcom */\n"
    + SAMPLE_VALID_CODE
)


def _init_test_loader(run_id: str, eval_path: str, src_path: str, mode: str = "record") -> CVESeedLoader:
    """Initialize a deterministic CVESeedLoader instance for testing."""
    replay_mgr = ReplayManager.from_config(run_id, 42, "artifacts/cassettes/test.json", mode)
    return CVESeedLoader(eval_path, src_path, replay_mgr)


def _write_csv_rows(file_path, rows: list[str]) -> None:
    """Write header and candidate code rows into a test CSV file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("code,language,safety\n")
        for r in rows:
            f.write(f'"{r}",c,vulnerable\n')


def _generate_synthetic_candidates(count: int) -> list[str]:
    """Generate n distinct C function snippets for stop rule validation."""
    return [
        f"int fn_{i}(int a) {{ int b_{i}[10]; return b_{i}[a]; /* ovfl */ }}\nint cl_{i}(int b) {{ return fn_{i}(b); }}"
        for i in range(count)
    ]


def test_3tier_deduplication(tmp_path):
    eval_path = tmp_path / "eval.csv"
    src_path = tmp_path / "src.csv"
    _write_csv_rows(eval_path, [])
    _write_csv_rows(src_path, [])
    loader = _init_test_loader("test-dedup", str(eval_path), str(src_path))
    
    # 1. Register base snippet
    assert not loader.is_duplicate(SAMPLE_VALID_CODE)
    loader.register_code(SAMPLE_VALID_CODE)
    
    # 2. Exact match check
    assert loader.is_duplicate(SAMPLE_VALID_CODE)
    
    # 3. Normalized match (comments & whitespace stripped)
    assert loader.is_duplicate(SAMPLE_CODE_DUPLICATE_COMMENTS)
    
    # 4. Completely different code is NOT duplicate
    diff_code = "int calculate_sum(int a, int b) { return a + b; }"
    assert not loader.is_duplicate(diff_code)


def test_untrusted_predicate_quality_gate():
    # Placeholder / hedging rejection
    res_placeholder = _predicate_quality("Vulnerability suspected.", SAMPLE_VALID_CODE)
    assert not res_placeholder["pass"]
    assert "predicate_too_short" in res_placeholder["reasons"] or "predicate_contains_hedging_or_placeholder" in res_placeholder["reasons"]

    # Missing vulnerability class rejection
    res_no_vuln = _predicate_quality("The function unimac_mdio_read contains an issue with phy_id and reg.", SAMPLE_VALID_CODE)
    assert not res_no_vuln["pass"]
    assert "missing_vulnerability_class" in res_no_vuln["reasons"]

    # Missing code symbols
    res_no_sym = _predicate_quality("The system has an out-of-bounds register write vulnerability in some external hardware component.", SAMPLE_VALID_CODE)
    assert not res_no_sym["pass"]
    assert "missing_code_symbol_or_grounded_term" in res_no_sym["reasons"]

    # Valid specific predicate
    valid_pred = "The driver is vulnerable to out-of-bounds MDIO register access because it does not validate phy_id and reg arguments in unimac_mdio_read()."
    res_valid = _predicate_quality(valid_pred, SAMPLE_VALID_CODE)
    assert res_valid["pass"]


def test_anchor_extraction_and_generic_filtering(tmp_path):
    eval_path = tmp_path / "eval.csv"
    src_path = tmp_path / "src.csv"
    _write_csv_rows(eval_path, [])
    _write_csv_rows(src_path, [])
    loader = _init_test_loader("test-anchors", str(eval_path), str(src_path))

    # Generic anchors must be detected
    assert _is_generic_anchor("missing bounds check")
    assert _is_generic_anchor("no validation")
    assert _is_generic_anchor("memory safety")
    assert not _is_generic_anchor("unimac_mdio_writel(bus->priv, cmd, MDIO_CMD)")

    # Extract anchors from gepa_info
    gepa_info = {
        "predicate": "The driver has an out-of-bounds write in `unimac_mdio_read`.",
        "evidence_hooks": [
            "unimac_mdio_read(): `cmd = MDIO_RD | (phy_id << MDIO_PMD_SHIFT)`",
            "unimac_mdio_writel(bus->priv, cmd, MDIO_CMD)",
            "missing validation", # Should be ignored as generic
        ]
    }
    anchors = loader.extract_anchors(gepa_info, SAMPLE_VALID_CODE)
    assert len(anchors) >= 2
    assert "unimac_mdio_writel(bus->priv, cmd, MDIO_CMD)" in anchors
    assert "missing validation" not in anchors
    assert all(not _is_generic_anchor(a) for a in anchors)


def test_existing_seeds_prefix_and_dedup_loading(tmp_path):
    exist_path = tmp_path / "exist.jsonl"
    eval_path = tmp_path / "eval.csv"
    src_path = tmp_path / "src.csv"
    _write_csv_rows(eval_path, [])
    _write_csv_rows(src_path, [])
    
    # Write existing seeds
    seed_record = {
        "topic": SAMPLE_VALID_CODE,
        "predicate": "The driver is vulnerable to out-of-bounds MDIO register access in unimac_mdio_read.",
        "gepa_info": {"evidence_hooks": ["unimac_mdio_read"]},
        "language": "c",
        "original_safety": "vulnerable",
        "anchors": ["unimac_mdio_read", "unimac_mdio_writel"],
    }
    with open(exist_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(seed_record) + "\n")

    loader = _init_test_loader("test-existing", str(eval_path), str(src_path))
    loaded = loader.load_existing_seeds(str(exist_path))
    
    assert len(loaded) == 1
    assert loader.is_duplicate(SAMPLE_VALID_CODE)
    assert loader.is_duplicate(SAMPLE_CODE_DUPLICATE_COMMENTS)


def test_eval_exclusion_isolation_from_shingles(tmp_path):
    eval_path = tmp_path / "eval.csv"
    src_path = tmp_path / "src.csv"
    _write_csv_rows(eval_path, [SAMPLE_VALID_CODE])
    _write_csv_rows(src_path, [])

    loader = _init_test_loader("test-eval-shingles", str(eval_path), str(src_path))
    loader.load_eval_exclusion_set()

    # Eval snippet is recognized as duplicate via hash sets
    assert loader.is_duplicate(SAMPLE_VALID_CODE)
    assert loader.is_duplicate(SAMPLE_CODE_DUPLICATE_COMMENTS)

    # Eval exclusion sets must not pollute the candidate fuzzy shingle index
    assert len(loader.eval_exact_hashes) == 1
    assert len(loader.eval_norm_hashes) == 1
    assert len(loader.used_shingles) == 0


def test_seed_expansion_config_validation():
    # Valid constructions
    cfg1 = SeedExpansionConfig.from_args(target_total=100)
    assert cfg1.target_total == 100

    cfg2 = SeedExpansionConfig.from_args(50)
    assert cfg2.target_total == 50

    cfg3 = SeedExpansionConfig.from_args(cfg1, max_calls=500)
    assert cfg3.target_total == 100
    assert cfg3.max_calls == 500

    # Unsupported type must raise TypeError
    with pytest.raises(TypeError, match="Unsupported config type"):
        SeedExpansionConfig.from_args(["invalid", "type"])


@pytest.mark.asyncio
async def test_expand_seeds_stop_rules_and_telemetry(tmp_path):
    src_path = tmp_path / "src.csv"
    eval_path = tmp_path / "eval.csv"
    exist_path = tmp_path / "exist.jsonl"
    out_path = tmp_path / "out.jsonl"
    telem_path = tmp_path / "telem.json"
    attempts_path = tmp_path / "attempts.jsonl"

    # Write 1 existing seed
    seed_record = {
        "topic": SAMPLE_VALID_CODE,
        "predicate": "The driver is vulnerable to out-of-bounds MDIO register access in unimac_mdio_read.",
        "gepa_info": {"evidence_hooks": ["unimac_mdio_read", "unimac_mdio_writel"]},
        "language": "c",
        "original_safety": "vulnerable",
        "anchors": ["unimac_mdio_read", "unimac_mdio_writel"],
    }
    with open(exist_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(seed_record) + "\n")

    code_cand_1 = "int foo(int x) { int arr[10]; return arr[x]; /* buffer overflow */ }\nint bar(int y) { return foo(y); }"
    code_cand_2 = "int baz(int a) { int buffer[5]; return buffer[a]; /* out-of-bounds read */ }\nint qux(int b) { return baz(b); }"
    _write_csv_rows(src_path, [code_cand_1, code_cand_2])
    _write_csv_rows(eval_path, [])

    loader = _init_test_loader("test-expand", str(eval_path), str(src_path))

    mock_gepa = {
        "predicate": "The code is vulnerable to a buffer overflow in foo when indexing arr.",
        "evidence_hooks": ["int arr[10]", "return arr[x]"],
        "uncertainty": "Low",
        "proof_requirements": "Provide x >= 10",
    }
    with patch.object(loader, "gepa_explain_with_retry", new=AsyncMock(return_value=mock_gepa)):
        seeds = await loader.expand_seeds(
            target_total=2, # Initial 1 + 1 new = 2
            existing_seeds_path=str(exist_path),
            output_path=str(out_path),
            telemetry_path=str(telem_path),
            attempts_path=str(attempts_path),
            min_marginal_yield=0.10,
            window_size=5,
        )

    assert len(seeds) == 2
    with open(out_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["topic"] == SAMPLE_VALID_CODE
    assert lines[1]["topic"] == code_cand_1

    with open(telem_path, "r", encoding="utf-8") as f:
        telemetry = json.load(f)
    assert telemetry["target_total"] == 2
    assert telemetry["initial_verified_count"] == 1
    assert telemetry["final_accepted_count"] == 2
    assert telemetry["new_accepted_count"] == 1
    assert telemetry["stop_reason"] == "target_total_reached"

    # Verify attempt JSONL logging
    with open(attempts_path, "r", encoding="utf-8") as f:
        attempt_lines = [json.loads(line) for line in f if line.strip()]
    assert len(attempt_lines) >= 1
    assert attempt_lines[0]["valid"] is True


@pytest.mark.asyncio
async def test_stop_rule_on_marginal_yield_decay(tmp_path):
    src_path = tmp_path / "src.csv"
    eval_path = tmp_path / "eval.csv"
    out_path = tmp_path / "out.jsonl"
    telem_path = tmp_path / "telem.json"

    _write_csv_rows(src_path, _generate_synthetic_candidates(10))
    _write_csv_rows(eval_path, [])
    loader = _init_test_loader("test-yield-decay", str(eval_path), str(src_path))

    mock_bad_gepa = {
        "predicate": "Vulnerability suspected.",
        "evidence_hooks": [],
    }
    with patch.object(loader, "gepa_explain_with_retry", new=AsyncMock(return_value=mock_bad_gepa)):
        seeds = await loader.expand_seeds(
            target_total=10,
            output_path=str(out_path),
            telemetry_path=str(telem_path),
            min_marginal_yield=0.30,
            window_size=3, # Will check after 3 consecutive failures
        )

    assert len(seeds) == 0
    with open(telem_path, "r", encoding="utf-8") as f:
        telemetry = json.load(f)
    assert telemetry["stop_reason"] == "marginal_yield_decay_threshold_reached"
    assert telemetry["final_accepted_count"] == 0
    assert telemetry["total_calls"] == 3


@pytest.mark.asyncio
async def test_stop_rule_on_max_calls_budget(tmp_path):
    src_path = tmp_path / "src.csv"
    eval_path = tmp_path / "eval.csv"
    out_path = tmp_path / "out.jsonl"
    telem_path = tmp_path / "telem.json"

    _write_csv_rows(src_path, _generate_synthetic_candidates(10))
    _write_csv_rows(eval_path, [])
    loader = _init_test_loader("test-max-calls", str(eval_path), str(src_path))

    mock_gepa = {
        "predicate": "The code is vulnerable to a buffer overflow in func_0 when indexing buf_0.",
        "evidence_hooks": ["int buf_0[10]", "return buf_0[a]"],
    }
    with patch.object(loader, "gepa_explain_with_retry", new=AsyncMock(return_value=mock_gepa)):
        seeds = await loader.expand_seeds(
            target_total=10,
            output_path=str(out_path),
            telemetry_path=str(telem_path),
            max_calls=2, # Budget limit is 2
            window_size=10,
        )

    with open(telem_path, "r", encoding="utf-8") as f:
        telemetry = json.load(f)
    assert telemetry["stop_reason"] == "max_calls_budget_reached"
    assert telemetry["total_calls"] == 2
    assert len(seeds) == telemetry["final_accepted_count"]
    assert len(seeds) <= 2


@pytest.mark.asyncio
async def test_replay_mode_cache_miss_raises(tmp_path):
    eval_path = tmp_path / "eval.csv"
    src_path = tmp_path / "src.csv"
    _write_csv_rows(eval_path, [])
    _write_csv_rows(src_path, [])

    loader = _init_test_loader("test-replay-miss", str(eval_path), str(src_path), mode="replay")
    with patch.object(loader.replay_manager, "acompletion", side_effect=RuntimeError("Replay cache miss for key")):
        with pytest.raises(RuntimeError, match="Replay cache miss"):
            await loader.gepa_explain(SAMPLE_VALID_CODE, "c")


@pytest.mark.asyncio
async def test_legacy_get_seeds_compatibility(tmp_path):
    src_path = tmp_path / "src.csv"
    eval_path = tmp_path / "eval.csv"

    code_c = "int test_func(int a) { return a * 2; /* safe computation */ }\nint test_call(int b) { return test_func(b); }"
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(f"code,language,safety\n\"{code_c}\",c,safe\n")
    _write_csv_rows(eval_path, [])

    loader = _init_test_loader("test-legacy", str(eval_path), str(src_path))

    mock_gepa = {
        "predicate": "The function test_func is verified safe.",
        "evidence_hooks": ["return a * 2"],
    }
    with patch.object(loader, "gepa_explain_with_retry", new=AsyncMock(return_value=mock_gepa)):
        seeds = await loader.get_seeds(n=1, target_lang="c")

    assert len(seeds) == 1
    assert seeds[0]["topic"] == code_c
    assert seeds[0]["predicate"] == "The function test_func is verified safe."
    assert seeds[0]["original_safety"] == "safe"
