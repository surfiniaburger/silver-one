import os
import json
import tempfile
import pytest
from unittest.mock import AsyncMock, patch

from agentbeats.replay import ReplayManager
from scenarios.debate.cve_seed_loader import (
    CVESeedLoader,
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

SAMPLE_CODE_DUPLICATE_COMMENTS = """
// Broadcom UniMAC MDIO bus controller driver
/* Copyright Broadcom */
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


def test_3tier_deduplication():
    with tempfile.NamedTemporaryFile("w", delete=False) as eval_f, \
         tempfile.NamedTemporaryFile("w", delete=False) as src_f:
        eval_f.write("code,language,safety\n")
        eval_f.flush()
        src_f.write("code,language,safety\n")
        src_f.flush()
        
        replay_mgr = ReplayManager.from_config("test-dedup", 42, "artifacts/cassettes/test.json", "record")
        loader = CVESeedLoader(eval_f.name, src_f.name, replay_mgr)
        
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


def test_anchor_extraction_and_generic_filtering():
    with tempfile.NamedTemporaryFile("w", delete=False) as eval_f, \
         tempfile.NamedTemporaryFile("w", delete=False) as src_f:
        replay_mgr = ReplayManager.from_config("test-anchors", 42, "artifacts/cassettes/test.json", "record")
        loader = CVESeedLoader(eval_f.name, src_f.name, replay_mgr)

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


def test_existing_seeds_prefix_and_dedup_loading():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as exist_f, \
         tempfile.NamedTemporaryFile("w", delete=False) as eval_f, \
         tempfile.NamedTemporaryFile("w", delete=False) as src_f:
        
        # Write existing seeds
        seed_record = {
            "topic": SAMPLE_VALID_CODE,
            "predicate": "The driver is vulnerable to out-of-bounds MDIO register access in unimac_mdio_read.",
            "gepa_info": {"evidence_hooks": ["unimac_mdio_read"]},
            "language": "c",
            "original_safety": "vulnerable",
            "anchors": ["unimac_mdio_read", "unimac_mdio_writel"],
        }
        exist_f.write(json.dumps(seed_record) + "\n")
        exist_f.flush()

        replay_mgr = ReplayManager.from_config("test-existing", 42, "artifacts/cassettes/test.json", "record")
        loader = CVESeedLoader(eval_f.name, src_f.name, replay_mgr)
        loaded = loader.load_existing_seeds(exist_f.name)
        
        assert len(loaded) == 1
        assert loader.is_duplicate(SAMPLE_VALID_CODE)
        assert loader.is_duplicate(SAMPLE_CODE_DUPLICATE_COMMENTS)


@pytest.mark.asyncio
async def test_expand_seeds_stop_rules_and_telemetry():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as src_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as eval_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as exist_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as out_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as telem_f:

        # Write 1 existing seed
        seed_record = {
            "topic": SAMPLE_VALID_CODE,
            "predicate": "The driver is vulnerable to out-of-bounds MDIO register access in unimac_mdio_read.",
            "gepa_info": {"evidence_hooks": ["unimac_mdio_read", "unimac_mdio_writel"]},
            "language": "c",
            "original_safety": "vulnerable",
            "anchors": ["unimac_mdio_read", "unimac_mdio_writel"],
        }
        exist_f.write(json.dumps(seed_record) + "\n")
        exist_f.flush()

        # Write source CSV with multiple candidates
        src_f.write("code,language,safety\n")
        code_cand_1 = "int foo(int x) { int arr[10]; return arr[x]; /* buffer overflow */ }\nint bar(int y) { return foo(y); }"
        code_cand_2 = "int baz(int a) { int buffer[5]; return buffer[a]; /* out-of-bounds read */ }\nint qux(int b) { return baz(b); }"
        src_f.write(f'"{code_cand_1}",c,vulnerable\n')
        src_f.write(f'"{code_cand_2}",c,vulnerable\n')
        src_f.flush()

        eval_f.write("code,language,safety\n")
        eval_f.flush()

        replay_mgr = ReplayManager.from_config("test-expand", 42, "artifacts/cassettes/test.json", "record")
        loader = CVESeedLoader(eval_f.name, src_f.name, replay_mgr)

        # Mock gepa_explain_with_retry
        mock_gepa = {
            "predicate": "The code is vulnerable to a buffer overflow in foo when indexing arr.",
            "evidence_hooks": ["int arr[10]", "return arr[x]"],
            "uncertainty": "Low",
            "proof_requirements": "Provide x >= 10",
        }
        with patch.object(loader, "gepa_explain_with_retry", new=AsyncMock(return_value=mock_gepa)):
            seeds = await loader.expand_seeds(
                target_total=2, # Initial 1 + 1 new = 2
                existing_seeds_path=exist_f.name,
                output_path=out_f.name,
                telemetry_path=telem_f.name,
                min_marginal_yield=0.10,
                window_size=5,
            )

        assert len(seeds) == 2
        # Check output file has exactly 2 records
        with open(out_f.name, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["topic"] == SAMPLE_VALID_CODE
        assert lines[1]["topic"] == code_cand_1

        # Check telemetry file
        with open(telem_f.name, "r") as f:
            telemetry = json.load(f)
        assert telemetry["target_total"] == 2
        assert telemetry["initial_verified_count"] == 1
        assert telemetry["final_accepted_count"] == 2
        assert telemetry["new_accepted_count"] == 1
        assert telemetry["stop_reason"] == "target_total_reached"


@pytest.mark.asyncio
async def test_stop_rule_on_marginal_yield_decay():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as src_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as eval_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as out_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as telem_f:

        src_f.write("code,language,safety\n")
        # Write 10 valid candidates
        for i in range(10):
            code_c = f"int func_{i}(int a) {{ int buf_{i}[10]; return buf_{i}[a]; /* buffer overflow */ }}\nint call_{i}(int b) {{ return func_{i}(b); }}"
            src_f.write(f'"{code_c}",c,vulnerable\n')
        src_f.flush()
        eval_f.write("code,language,safety\n")
        eval_f.flush()

        replay_mgr = ReplayManager.from_config("test-yield-decay", 42, "artifacts/cassettes/test.json", "record")
        loader = CVESeedLoader(eval_f.name, src_f.name, replay_mgr)

        # Mock GEPA returning poor quality / rejected predicates
        mock_bad_gepa = {
            "predicate": "Vulnerability suspected.",
            "evidence_hooks": [],
        }
        with patch.object(loader, "gepa_explain_with_retry", new=AsyncMock(return_value=mock_bad_gepa)):
            seeds = await loader.expand_seeds(
                target_total=10,
                output_path=out_f.name,
                telemetry_path=telem_f.name,
                min_marginal_yield=0.30,
                window_size=3, # Will check after 3 consecutive failures
            )

        # Should halt early on marginal yield drop
        assert len(seeds) == 0
        with open(telem_f.name, "r") as f:
            telemetry = json.load(f)
        assert telemetry["stop_reason"] == "marginal_yield_decay_threshold_reached"
        assert telemetry["final_accepted_count"] == 0
        assert telemetry["total_calls"] == 3


@pytest.mark.asyncio
async def test_stop_rule_on_max_calls_budget():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as src_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as eval_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as out_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as telem_f:

        src_f.write("code,language,safety\n")
        for i in range(10):
            code_c = f"int func_{i}(int a) {{ int buf_{i}[10]; return buf_{i}[a]; /* buffer overflow */ }}\nint call_{i}(int b) {{ return func_{i}(b); }}"
            src_f.write(f'"{code_c}",c,vulnerable\n')
        src_f.flush()
        eval_f.write("code,language,safety\n")
        eval_f.flush()

        replay_mgr = ReplayManager.from_config("test-max-calls", 42, "artifacts/cassettes/test.json", "record")
        loader = CVESeedLoader(eval_f.name, src_f.name, replay_mgr)

        mock_gepa = {
            "predicate": "The code is vulnerable to a buffer overflow in func_0 when indexing buf_0.",
            "evidence_hooks": ["int buf_0[10]", "return buf_0[a]"],
        }
        with patch.object(loader, "gepa_explain_with_retry", new=AsyncMock(return_value=mock_gepa)):
            seeds = await loader.expand_seeds(
                target_total=10,
                output_path=out_f.name,
                telemetry_path=telem_f.name,
                max_calls=2, # Budget limit is 2
                window_size=10,
            )

        with open(telem_f.name, "r") as f:
            telemetry = json.load(f)
        assert telemetry["stop_reason"] == "max_calls_budget_reached"
        assert telemetry["total_calls"] == 2


@pytest.mark.asyncio
async def test_legacy_get_seeds_compatibility():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as src_f, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as eval_f:

        src_f.write("code,language,safety\n")
        code_c = "int test_func(int a) { return a * 2; /* safe computation */ }\nint test_call(int b) { return test_func(b); }"
        src_f.write(f'"{code_c}",c,safe\n')
        src_f.flush()
        eval_f.write("code,language,safety\n")
        eval_f.flush()

        replay_mgr = ReplayManager.from_config("test-legacy", 42, "artifacts/cassettes/test.json", "record")
        loader = CVESeedLoader(eval_f.name, src_f.name, replay_mgr)

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
