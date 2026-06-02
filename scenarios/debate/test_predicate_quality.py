from adk_debate_judge import _predicate_quality


VALID_CODE = """
int unimac_mdio_read(struct mii_bus *bus, int phy_id, int reg) {
    u32 cmd = MDIO_RD | (phy_id << MDIO_PMD_SHIFT) | (reg << MDIO_REG_SHIFT);
    unimac_mdio_writel(bus->priv, cmd, MDIO_CMD);
    return cmd;
}

int parse_string(const char *ptr, int len) {
    char *out = malloc(len + 1);
    return parse_hex4(ptr);
}

int get_max_column_count(int max_width, int item_extra, int item_min_size) {
    return max_width / (item_extra + item_min_size);
}
"""


def test_rejects_placeholder_predicate():
    result = _predicate_quality("Vulnerability suspected.", VALID_CODE)
    assert result["pass"] is False
    assert "predicate_too_short" in result["reasons"]
    assert "predicate_contains_hedging_or_placeholder" in result["reasons"]


def test_accepts_unicode_hyphen_vulnerability_classes():
    predicates = [
        (
            "The driver is vulnerable to out‑of‑bounds MDIO register access because it does not "
            "validate the phy_id and reg arguments in unimac_mdio_read() and unimac_mdio_write()."
        ),
        (
            "The code is vulnerable to a heap buffer overflow in parse_string when processing "
            "Unicode escape sequences because the allocated buffer size does not account for the "
            "possible expansion to up to 4 UTF‑8 bytes per code point."
        ),
        (
            "The code is vulnerable to a division‑by‑zero denial‑of‑service in the function "
            "get_max_column_count when item_extra + item_min_size equals 0."
        ),
    ]
    for predicate in predicates:
        result = _predicate_quality(predicate, VALID_CODE)
        assert result["pass"] is True, result


if __name__ == "__main__":
    test_rejects_placeholder_predicate()
    test_accepts_unicode_hyphen_vulnerability_classes()
    print("ok")
