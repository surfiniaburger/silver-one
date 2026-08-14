"""
Evaluation Script for Graph Data-Flow Pre-Filter (PR 3 Benchmark).
Executes 5-Fold Stratified Scenario-Grouped CV across 5 random seeds on the deduplicated dataset.
Computes ROC-AUC, PR-AUC, Hodges-Lehmann CIs, and graph parser error bucket diagnostics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from scenarios.debate.graph_dataflow import evaluate_graph_reachability
from scenarios.debate.graph_extractor import extract_flow_graph_snapshot
from scenarios.debate.pre_filter import (
    CODE_DELIMITER,
    _parse_attempt_record,
    partition_dataset_by_scenario_stratified,
)
from scripts.train_pre_filter import (
    _classify_graph_extraction_bucket,
    _compute_graph_fold_diagnostics,
    extract_dataset_from_attempts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _extract_code_text(text: str) -> str:
    if CODE_DELIMITER in text:
        return text.split(CODE_DELIMITER, 1)[1]
    return text


def evaluate_graph_on_sample(text: str, idx: int) -> float:
    code_text = _extract_code_text(text)
    snapshot = extract_flow_graph_snapshot(
        code_text=code_text,
        scenario_id=f"eval_scenario_{idx}",
        snapshot_id=f"eval_snap_{idx}",
        version=1,
        created_at=0.0,
    )
    return evaluate_graph_reachability(snapshot)


def compute_hodges_lehmann_ci(differences: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Computes the Hodges-Lehmann median difference estimate and 95% distribution-free CI."""
    if len(differences) == 0:
        return 0.0, 0.0, 0.0
    pairwise_means = [(differences[i] + differences[j]) / 2.0 for i in range(len(differences)) for j in range(i, len(differences))]
    hl_estimate = float(np.median(pairwise_means))
    lower_ci = float(np.percentile(pairwise_means, (alpha / 2.0) * 100))
    upper_ci = float(np.percentile(pairwise_means, (1.0 - alpha / 2.0) * 100))
    return hl_estimate, lower_ci, upper_ci


DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _evaluate_single_seed(
    texts: List[str],
    labels: List[int],
    scenario_ids: List[str],
    seed: int,
    n_splits: int,
) -> Dict[str, Any]:
    folds = partition_dataset_by_scenario_stratified(texts, labels, scenario_ids, n_splits=n_splits, seed=seed)
    oof_y_true: List[int] = []
    oof_graph_probs: List[float] = []
    oof_graph_preds: List[int] = []
    oof_texts: List[str] = []

    for fold in folds:
        for idx, (txt, lbl) in enumerate(zip(fold["test_texts"], fold["test_labels"], strict=True)):
            risk_score = evaluate_graph_on_sample(txt, idx)
            pred = 1 if risk_score >= 0.10 else 0
            oof_y_true.append(lbl)
            oof_graph_probs.append(risk_score)
            oof_graph_preds.append(pred)
            oof_texts.append(txt)

    return _compute_seed_metrics(oof_texts, oof_y_true, oof_graph_probs, oof_graph_preds, seed)


def _compute_seed_metrics(
    oof_texts: List[str],
    oof_y_true: List[int],
    oof_graph_probs: List[float],
    oof_graph_preds: List[int],
    seed: int,
) -> Dict[str, Any]:
    y_true = np.array(oof_y_true)
    probs = np.array(oof_graph_probs)
    preds = np.array(oof_graph_preds)

    try:
        roc_auc = float(roc_auc_score(y_true, probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except Exception:
        pr_auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    diagnostics = _compute_graph_fold_diagnostics(oof_texts, oof_y_true, oof_graph_preds)

    return {
        "seed": seed,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "diagnostics": diagnostics,
    }


def run_graph_cv_evaluation(
    attempts_dir: Path,
    output_path: Path,
    n_splits: int = 5,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Runs 5-fold Stratified Scenario-Grouped CV across seeds for Graph Data-Flow Pre-Filter."""
    active_seeds = seeds if seeds is not None else DEFAULT_SEEDS
    texts, labels, scenario_ids = extract_dataset_from_attempts(
        attempts_dir,
        dedup_near_duplicates=True,
        similarity_threshold=0.85,
    )

    logger.info("Dataset loaded: %d deduplicated samples across %d unique scenarios", len(texts), len(set(scenario_ids)))

    seed_results = [
        _evaluate_single_seed(texts, labels, scenario_ids, seed, n_splits)
        for seed in active_seeds
    ]

    roc_aucs = np.array([r["roc_auc"] for r in seed_results])
    pr_aucs = np.array([r["pr_auc"] for r in seed_results])

    hl_roc, hl_roc_low, hl_roc_high = compute_hodges_lehmann_ci(roc_aucs)
    hl_pr, hl_pr_low, hl_pr_high = compute_hodges_lehmann_ci(pr_aucs)

    report = {
        "dataset_summary": {
            "total_samples": len(texts),
            "unique_scenarios": len(set(scenario_ids)),
            "positive_samples": int(sum(labels)),
            "negative_samples": int(len(labels) - sum(labels)),
        },
        "evaluation_protocol": {
            "n_splits": n_splits,
            "seeds": active_seeds,
            "partition_strategy": "Stratified Scenario-Grouped K-Fold",
        },
        "graph_pre_filter_metrics": {
            "mean_roc_auc": round(float(np.mean(roc_aucs)), 4),
            "std_roc_auc": round(float(np.std(roc_aucs)), 4),
            "hl_roc_auc_estimate": round(hl_roc, 4),
            "hl_roc_auc_95_ci": [round(hl_roc_low, 4), round(hl_roc_high, 4)],
            "mean_pr_auc": round(float(np.mean(pr_aucs)), 4),
            "std_pr_auc": round(float(np.std(pr_aucs)), 4),
            "hl_pr_auc_estimate": round(hl_pr, 4),
            "hl_pr_auc_95_ci": [round(hl_pr_low, 4), round(hl_pr_high, 4)],
        },
        "seed_breakdown": seed_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation report saved to '%s'", output_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Graph Data-Flow Pre-Filter 5-Fold CV.")
    parser.add_argument("--attempts-dir", type=str, default="artifacts/attempts", help="Directory containing attempt log files.")
    parser.add_argument("--output-file", type=str, default="artifacts/metrics/graph_pre_filter_cv_report.json", help="Path to save evaluation JSON report.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of CV folds.")
    args = parser.parse_args()

    run_graph_cv_evaluation(
        attempts_dir=Path(args.attempts_dir),
        output_path=Path(args.output_file),
        n_splits=args.k_folds,
    )


if __name__ == "__main__":
    main()
