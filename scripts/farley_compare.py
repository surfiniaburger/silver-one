#!/usr/bin/env python3
import json
import sys
import argparse
from statistics import mean

FAIL_DELTA = 0.25
FAIL_PROP = 0.5
FAIL_PERCENT_TESTS = 0.05


def load_cassette(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_suite_summary(cassette):
    tests = cassette.get('tests', [])
    if not tests:
        return {'avg_index': 0.0, 'count': 0, 'per_property': {}}
    indices = [t.get('farley_index', 0.0) for t in tests]
    per_prop = {}
    # aggregate per-property averages
    props = ['understandable','maintainable','repeatable','atomic','necessary','granular','fast','first_tdd']
    for p in props:
        vals = []
        for t in tests:
            fb = t.get('farley_breakdown', {})
            v = fb.get(p, {}).get('score')
            if v is not None:
                vals.append(v)
        per_prop[p] = mean(vals) if vals else 0.0
    return {'avg_index': mean(indices), 'count': len(indices), 'per_property': per_prop}


def top_regressions(base, pr, top_n=5):
    base_map = {t.get('id'): t for t in base.get('tests', []) if t.get('id') is not None}
    reg = []
    for t in pr.get('tests', []):
        tid = t.get('id')
        b = base_map.get(tid)
        if b:
            delta = t.get('farley_index',0.0) - b.get('farley_index',0.0)
            if delta < 0:
                reg.append((delta, b, t))
    reg.sort(key=lambda x: x[0])
    return reg[:top_n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--pr', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    base = load_cassette(args.baseline)
    pr = load_cassette(args.pr)

    bsum = compute_suite_summary(base)
    psum = compute_suite_summary(pr)

    delta = psum['avg_index'] - bsum['avg_index']

    # identify tests dropping by >=2
    base_map = {t.get('id'): t for t in base.get('tests', []) if t.get('id') is not None}
    drops = 0
    total = len(pr.get('tests', []))
    biggest = []
    for t in pr.get('tests', []):
        tid = t.get('id')
        b = base_map.get(tid)
        if b:
            d = b.get('farley_index',0.0) - t.get('farley_index',0.0)
            if d >= 2.0:
                drops += 1
                biggest.append((d, b, t))
    pct = (drops / total) if total else 0.0

    verdict = 'PASS'
    exit_code = 0
    reasons = []
    if delta <= -FAIL_DELTA:
        verdict = 'FAIL'
        exit_code = 2
        reasons.append(f'Suite Farley Index decreased by {abs(delta):.2f} >= {FAIL_DELTA}')
    if psum['per_property'].get('understandable',0.0) - bsum['per_property'].get('understandable',0.0) <= -FAIL_PROP:
        verdict = 'FAIL'
        exit_code = 2
        reasons.append('Understandable dropped too much')
    if psum['per_property'].get('maintainable',0.0) - bsum['per_property'].get('maintainable',0.0) <= -FAIL_PROP:
        verdict = 'FAIL'
        exit_code = 2
        reasons.append('Maintainable dropped too much')
    if pct > FAIL_PERCENT_TESTS:
        verdict = 'FAIL'
        exit_code = 2
        reasons.append(f'{pct*100:.1f}% of tests dropped by >=2 points')

    # write report
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(f'# Farley Compare Report\n\n')
        f.write(f'**Baseline avg**: {bsum["avg_index"]:.2f}\n')
        f.write(f'**PR avg**: {psum["avg_index"]:.2f}\n')
        f.write(f'**Delta**: {delta:+.2f}\n\n')
        f.write(f'**Verdict**: {verdict}\n')
        if reasons:
            f.write('\n**Reasons**:\n')
            for r in reasons:
                f.write(f'- {r}\n')
        f.write('\n## Top regressions\n')
        regs = top_regressions(base, pr, top_n=10)
        if not regs:
            f.write('No regressions found.\n')
        else:
            f.write('| Delta | File | Test | Base | PR |\n')
            f.write('|---|---|---|---:|---:|\n')
            for d,bp,tp in regs:
                f.write(f'| {d:.2f} | {bp.get("file_path")} | {bp.get("test_name")} | {bp.get("farley_index",0.0):.2f} | {tp.get("farley_index",0.0):.2f} |\n')

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
