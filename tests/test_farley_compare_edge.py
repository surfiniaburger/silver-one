from scripts.farley_compare import top_regressions, compute_suite_summary


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
    assert summary['avg_index'] == 0.0
    assert summary['count'] == 0
