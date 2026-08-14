"""
Evaluation Script for Graph Data-Flow Pre-Filter (PR 3 Benchmark).
Executes 5-Fold Stratified Scenario-Grouped CV across 5 random seeds on the deduplicated dataset.
Computes ROC-AUC, PR-AUC, seed-wise percentile CIs, and graph parser error bucket diagnostics.

Report Schema Overview (schema_version: "1.0"):
- schema_version: "1.0"
- dataset_summary: total_samples, unique_scenarios, positive_samples, negative_samples
- evaluation_protocol: requested_n_splits, effective_n_splits, seeds, partition_strategy
- graph_pre_filter_metrics: mean_roc_auc, std_roc_auc, mean_pr_auc, std_pr_auc, roc_auc_95_percentile_ci, pr_auc_95_percentile_ci
- seed_breakdown: per-seed metrics, confusion matrices, and graph parser diagnostics
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
    _validate_safe_path,
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


def compute_seed_percentile_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Computes the median estimate and 95% distribution-free percentile CI across seeds."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    median_val = float(np.median(values))
    lower_ci = float(np.percentile(values, (alpha / 2.0) * 100))
    upper_ci = float(np.percentile(values, (1.0 - alpha / 2.0) * 100))
    return median_val, lower_ci, upper_ci


DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _evaluate_single_seed(
    texts: List[str],
    labels: List[int],
    scenario_ids: List[str],
    seed: int,
    n_splits: int,
) -> Tuple[Dict[str, Any], int]:
    folds = partition_dataset_by_scenario_stratified(texts, labels, scenario_ids, n_splits=n_splits, seed=seed)
    effective_n_splits = len(folds)
    oof_y_true: List[int] = []
    oof_graph_probs: List[float] = []
    oof_graph_preds: List[int] = []
    oof_texts: List[str] = []

    for fold in folds:
        for idx, (txt, lbl) in enumerate(zip(fold["test_texts"], fold["test_labels"], strict=True)):
            risk_score = evaluate_graph_on_sample(txt, idx)
            accepted_probability = 1.0 - risk_score
            pred = 1 if accepted_probability >= 0.90 else 0

            oof_y_true.append(lbl)
            oof_graph_probs.append(accepted_probability)
            oof_graph_preds.append(pred)
            oof_texts.append(txt)

    seed_metrics = _compute_seed_metrics(oof_texts, oof_y_true, oof_graph_probs, oof_graph_preds, seed)
    return seed_metrics, effective_n_splits


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

    if len(np.unique(y_true)) > 1:
        try:
            roc_auc: Optional[float] = float(roc_auc_score(y_true, probs))
        except ValueError:
            roc_auc = None
        try:
            pr_auc: Optional[float] = float(average_precision_score(y_true, probs))
        except ValueError:
            pr_auc = None
    else:
        roc_auc = None
        pr_auc = None

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    diagnostics = _compute_graph_fold_diagnostics(oof_texts, oof_y_true, oof_graph_preds)

    return {
        "seed": seed,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
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
    safe_output_path = _validate_safe_path(output_path)
    active_seeds = seeds if seeds is not None else DEFAULT_SEEDS

    texts, labels, scenario_ids = extract_dataset_from_attempts(
        attempts_dir,
        dedup_near_duplicates=True,
        similarity_threshold=0.85,
    )

    logger.info("Dataset loaded: %d deduplicated samples across %d unique scenarios", len(texts), len(set(scenario_ids)))

    seed_results = []
    effective_splits = n_splits

    for seed in active_seeds:
        seed_res, eff_splits = _evaluate_single_seed(texts, labels, scenario_ids, seed, n_splits)
        seed_results.append(seed_res)
        effective_splits = eff_splits

    valid_roc_aucs = np.array([r["roc_auc"] for r in seed_results if r["roc_auc"] is not None])
    valid_pr_aucs = np.array([r["pr_auc"] for r in seed_results if r["pr_auc"] is not None])

    med_roc, roc_low, roc_high = compute_seed_percentile_ci(valid_roc_aucs)
    med_pr, pr_low, pr_high = compute_seed_percentile_ci(valid_pr_aucs)

    report = {
        "schema_version": "1.0",
        "dataset_summary": {
            "total_samples": len(texts),
            "unique_scenarios": len(set(scenario_ids)),
            "positive_samples": int(sum(labels)),
            "negative_samples": int(len(labels) - sum(labels)),
        },
        "evaluation_protocol": {
            "requested_n_splits": n_splits,
            "effective_n_splits": effective_splits,
            "seeds": active_seeds,
            "partition_strategy": "Stratified Scenario-Grouped K-Fold",
        },
        "graph_pre_filter_metrics": {
            "mean_roc_auc": round(float(np.mean(valid_roc_aucs)), 4) if len(valid_roc_aucs) > 0 else None,
            "std_roc_auc": round(float(np.std(valid_roc_aucs)), 4) if len(valid_roc_aucs) > 0 else None,
            "median_roc_auc_estimate": round(med_roc, 4),
            "roc_auc_95_percentile_ci": [round(roc_low, 4), round(roc_high, 4)],
            "mean_pr_auc": round(float(np.mean(valid_pr_aucs)), 4) if len(valid_pr_aucs) > 0 else None,
            "std_pr_auc": round(float(np.std(valid_pr_aucs)), 4) if len(valid_pr_aucs) > 0 else None,
            "median_pr_auc_estimate": round(med_pr, 4),
            "pr_auc_95_percentile_ci": [round(pr_low, 4), round(pr_high, 4)],
        },
        "seed_breakdown": seed_results,
    }

    safe_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(safe_output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation report saved to '%s'", safe_output_path)
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
