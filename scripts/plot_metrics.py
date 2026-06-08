#!/usr/bin/env python3
"""Generate plots from artifacts/metrics CSV and JSON files.

Produces PNGs in artifacts/metrics/plots/
"""
import json
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "artifacts" / "metrics"
OUT_DIR = METRICS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv():
    csv_path = METRICS_DIR / "metrics_comparison.csv"
    return pd.read_csv(csv_path)


def load_json_runs():
    runs = {}
    for p in METRICS_DIR.glob("b_gate-pilot-v1-calibrated-*.json"):
        key = p.stem.split("-")[-1].upper()
        with open(p) as f:
            runs[key] = json.load(f)
    return runs


def plot_yield(df):
    ax = df.set_index('run')[['attempts','accepted']].plot(kind='bar')
    ax.set_ylabel('Count')
    ax.set_title('Attempts vs Accepted Rows by Run')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'yield_attempts_accepted.png')
    plt.close()


def plot_quality(df):
    ax = df.set_index('run')[['predicate_fail','b2_strict_fail']].plot(kind='bar')
    ax.set_ylabel('Rate')
    ax.set_title('Predicate-fail and B2 strict fail by Run')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'quality_rates.png')
    plt.close()


def plot_verifier(df):
    ax = df.set_index('run')[['verifier_called_rate','verifier_pass']].plot(kind='bar')
    ax.set_ylabel('Rate')
    ax.set_title('Verifier Called Rate and Verifier Pass by Run')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'verifier_rates.png')
    plt.close()


def plot_cost_quality(df):
    fig, ax = plt.subplots()
    sc = ax.scatter(df['tokens_per_accepted_row'], df['predicate_fail'], s=df['accepted']*20)
    for i, r in df.iterrows():
        ax.annotate(r['run'], (r['tokens_per_accepted_row'], r['predicate_fail']))
    ax.set_xlabel('Tokens per accepted row')
    ax.set_ylabel('Predicate fail rate')
    ax.set_title('Cost vs Predicate-fail (point size = accepted rows)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'cost_vs_quality.png')
    plt.close()


def plot_stage_breakdown(runs_json):
    # Build DataFrame of stage totals per run
    rows = []
    for run, data in runs_json.items():
        stages = data.get('usage_by_stage_totals', {})
        row = {'run': run}
        for s, v in stages.items():
            row[s + '_tokens'] = v.get('total_tokens', 0)
        rows.append(row)
    sdf = pd.DataFrame(rows).set_index('run').fillna(0)
    if sdf.shape[1] == 0:
        return
    sdf.plot(kind='bar', stacked=True)
    plt.ylabel('Tokens')
    plt.title('Token Budget by Stage (stacked)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'stage_token_breakdown.png')
    plt.close()


def plot_model_share(runs_json):
    rows = []
    for run, data in runs_json.items():
        models = data.get('usage_by_model_totals', {})
        row = {'run': run}
        for m, v in models.items():
            row[m] = v.get('total_tokens', 0)
        rows.append(row)
    mdf = pd.DataFrame(rows).set_index('run').fillna(0)
    if mdf.shape[1] == 0:
        return
    mdf.plot(kind='bar', stacked=True)
    plt.ylabel('Tokens')
    plt.title('Per-model Token Share by Run')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'model_token_share.png')
    plt.close()


def plot_heatmap(df):
    cols = ['predicate_fail','b2_strict_fail','tokens_per_accepted_row','verifier_called_rate','verifier_pass']
    present = [c for c in cols if c in df.columns]
    h = df.set_index('run')[present]
    # normalize for heatmap
    hn = (h - h.min()) / (h.max() - h.min())
    plt.figure(figsize=(6, max(2, len(h)*0.6)))
    sns.heatmap(hn, annot=h.round(3), cmap='vlag', cbar_kws={'label': 'normalized'})
    plt.title('Normalized Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'metrics_heatmap.png')
    plt.close()


def main():
    df = load_csv()
    runs_json = load_json_runs()
    plot_yield(df)
    plot_quality(df)
    plot_verifier(df)
    plot_cost_quality(df)
    plot_stage_breakdown(runs_json)
    plot_model_share(runs_json)
    plot_heatmap(df)


if __name__ == '__main__':
    main()
