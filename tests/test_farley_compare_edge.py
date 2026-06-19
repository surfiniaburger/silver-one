import pytest
from scripts.farley_compare import (
    compute_suite_summary,
    get_usage_totals,
    top_regressions,
)


def test_top_regressions_handles_missing_ids():
    base = {'tests': [
        {'id': 't1', 'farley_index': 8.0},
        {'farley_index': 5.0},  # missing id
    ]}
    pr = {'tests': [
        {'id': 't1', 'farley_index': 6.0},
        {'farley_index': 4.0},  # missing id
    ]}
    regs = top_regressions(base, pr, top_n=10)
    # should not raise and should include regression for t1
    assert any(r[1].get('id') == 't1' for r in regs)

def test_compute_suite_summary_empty():
    summary = compute_suite_summary({'tests': []})
    assert summary['avg_index'] == pytest.approx(0.0)
    assert summary['count'] == 0


def test_get_usage_totals_reads_farley_metadata():
    cassette = {
        "__metadata__": {
            "farley_usage_summary": {
                "usage": {
                    "totals": {
                        "calls": 2,
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                        "cost_usd": 0.01,
                    }
                }
            }
        }
    }

    totals = get_usage_totals(cassette)

    assert totals["calls"] == 2
    assert totals["total_tokens"] == 120
