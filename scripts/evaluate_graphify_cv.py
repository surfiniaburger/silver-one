"""
Comparative 5-Fold Stratified Scenario-Grouped CV Benchmark.
Compares the Original Regex/C Extractor vs. the Graphify-Inspired Tree-Sitter Extractor.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from scenarios.debate.pre_filter import partition_dataset_by_scenario_stratified
from scripts.train_pre_filter import extract_dataset_from_attempts
from scenarios.debate.graph_extractor import extract_flow_graph_snapshot as orig_extract
from scenarios.debate.graphify_flow_extractor import extract_graphify_flow_snapshot as graphify_extract
from scenarios.debate.graph_dataflow import evaluate_graph_reachability

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_graphify_cv")

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


@dataclass(frozen=True)
class EvaluationDataset:
    """Encapsulates test corpus texts, ground-truth binary labels, and scenario groupings."""
    texts: List[str]
    labels: List[int]
    scenario_ids: List[str]

    @property
    def total_samples(self) -> int:
        return len(self.texts)

    @property
    def positive_count(self) -> int:
        return sum(1 for label in self.labels if label == 1)

    @property
    def negative_count(self) -> int:
        return sum(1 for label in self.labels if label == 0)

    @property
    def unique_scenarios(self) -> int:
        return len(set(self.scenario_ids))


def extract_code_from_sample_text(combined_text: str) -> str:
    """Extracts the code snippet portion from a formatted candidate attempt text."""
    if "CODE_DELIMITER:" in combined_text:
        return combined_text.split("CODE_DELIMITER:", 1)[1].strip()
    return combined_text.strip()


def evaluate_extractor_on_sample(
    extractor_fn: Callable[..., Any], text: str, idx: int
) -> Tuple[bool, bool, float]:
    """
    Evaluates a code sample through an extractor and returns completeness,
    signature presence, and acceptance probability.
    """
    code = extract_code_from_sample_text(text)
    snap = extractor_fn(code, f"cv_s_{idx}", f"snap_{idx}", 1, 0.0)
    is_complete = snap.is_complete and snap.parse_error is None
    has_signatures = bool(snap.signatures)

    if not is_complete:
        return False, False, 0.0

    if not has_signatures:
        return True, False, 0.05

    risk = evaluate_graph_reachability(snap)
    prob = 0.95 if risk >= 0.10 else 0.05
    return True, True, prob


def _evaluate_fold_samples(
    extractor_fn: Callable[..., Any], fold: Dict[str, Any]
) -> Tuple[List[int], List[float], List[int], int, int]:
    """Evaluates test samples in a single fold without deep nesting."""
    y_true: List[int] = []
    probs: List[float] = []
    preds: List[int] = []
    parse_count = 0
    sig_count = 0

    test_texts = fold["test_texts"]
    test_labels = fold["test_labels"]

    for idx, (txt, lbl) in enumerate(zip(test_texts, test_labels, strict=True)):
        complete, has_sigs, prob = evaluate_extractor_on_sample(extractor_fn, txt, idx)
        if complete:
            parse_count += 1
        if has_sigs:
            sig_count += 1

        y_true.append(lbl)
        probs.append(prob)
        preds.append(1 if prob >= 0.50 else 0)

    return y_true, probs, preds, parse_count, sig_count


def _compute_metrics_for_seed(
    oof_true: List[int],
    oof_probs: List[float],
    oof_preds: List[int],
    seed: int,
    total_samples: int,
    parse_complete_count: int,
    signatures_count: int,
) -> Dict[str, Any]:
    """Calculates classification and parser coverage metrics for a single seed run."""
    evaluated = len(oof_true)
    if evaluated != total_samples:
        logger.warning("Fold test sets covered %d of %d samples for seed %d.", evaluated, total_samples, seed)

    y_true_arr = np.array(oof_true)
    probs_arr = np.array(oof_probs)
    preds_arr = np.array(oof_preds)

    fallback_metrics_used = False

    try:
        roc_auc = float(roc_auc_score(y_true_arr, probs_arr))
    except ValueError as err:
        logger.warning("ROC-AUC calculation failed on seed %d (error: %s). Using fallback 0.5.", seed, err)
        roc_auc = 0.5
        fallback_metrics_used = True

    try:
        pr_auc = float(average_precision_score(y_true_arr, probs_arr))
    except ValueError as err:
        mean_pos = float(np.mean(y_true_arr)) if len(y_true_arr) > 0 else 0.0
        logger.warning("PR-AUC calculation failed on seed %d (error: %s). Using fallback %f.", seed, err, mean_pos)
        pr_auc = mean_pos
        fallback_metrics_used = True

    cm = confusion_matrix(y_true_arr, preds_arr, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    return {
        "seed": seed,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "fallback_metrics_used": fallback_metrics_used,
        "parse_coverage": round(parse_complete_count / max(evaluated, 1), 4),
        "signatures_coverage": round(signatures_count / max(evaluated, 1), 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _evaluate_single_seed(
    extractor_fn: Callable[..., Any],
    dataset: EvaluationDataset,
    seed: int,
) -> Dict[str, Any]:
    """Executes a 5-fold cross validation pass for a single seed."""
    folds = partition_dataset_by_scenario_stratified(
        dataset.texts, dataset.labels, dataset.scenario_ids, n_splits=5, seed=seed
    )
    oof_true: List[int] = []
    oof_probs: List[float] = []
    oof_preds: List[int] = []
    total_parse_complete = 0
    total_signatures = 0

    for fold in folds:
        y_true, probs, preds, parse_cnt, sig_cnt = _evaluate_fold_samples(extractor_fn, fold)
        oof_true.extend(y_true)
        oof_probs.extend(probs)
        oof_preds.extend(preds)
        total_parse_complete += parse_cnt
        total_signatures += sig_cnt

    return _compute_metrics_for_seed(
        oof_true=oof_true,
        oof_probs=oof_probs,
        oof_preds=oof_preds,
        seed=seed,
        total_samples=dataset.total_samples,
        parse_complete_count=total_parse_complete,
        signatures_count=total_signatures,
    )


def run_5fold_cv_for_extractor(
    extractor_fn: Callable[..., Any],
    dataset: EvaluationDataset,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Executes 5-fold Stratified Scenario-Grouped CV across seeds for an extractor.
    """
    if dataset.total_samples == 0:
        logger.warning("Empty dataset passed to run_5fold_cv_for_extractor.")
        return {
            "extractor_symbol": f"{extractor_fn.__module__}.{extractor_fn.__qualname__}",
            "mean_roc_auc": 0.5,
            "mean_pr_auc": 0.0,
            "mean_parse_coverage": 0.0,
            "mean_signatures_coverage": 0.0,
            "seed_breakdown": [],
        }

    eval_seeds = seeds if seeds is not None else DEFAULT_SEEDS
    seed_results = [_evaluate_single_seed(extractor_fn, dataset, s) for s in eval_seeds]

    mean_roc = float(np.mean([r["roc_auc"] for r in seed_results]))
    mean_pr = float(np.mean([r["pr_auc"] for r in seed_results]))
    mean_cov = float(np.mean([r["parse_coverage"] for r in seed_results]))
    mean_sig_cov = float(np.mean([r["signatures_coverage"] for r in seed_results]))

    return {
        "extractor_symbol": f"{extractor_fn.__module__}.{extractor_fn.__qualname__}",
        "mean_roc_auc": round(mean_roc, 4),
        "mean_pr_auc": round(mean_pr, 4),
        "mean_parse_coverage": round(mean_cov, 4),
        "mean_signatures_coverage": round(mean_sig_cov, 4),
        "seed_breakdown": seed_results,
    }


def main():
    attempts_dir = Path("artifacts/attempts")
    texts, labels, scenario_ids = extract_dataset_from_attempts(
        attempts_dir=attempts_dir,
        dedup_near_duplicates=True,
        similarity_threshold=0.85,
    )
    if not texts:
        logger.error("No attempts extracted from %s. Exiting.", attempts_dir)
        return

    dataset = EvaluationDataset(texts=texts, labels=labels, scenario_ids=scenario_ids)
    logger.info("Extracted %d deduplicated samples across %d scenarios.", dataset.total_samples, dataset.unique_scenarios)

    logger.info("Running 5-Fold CV for Original Extractor...")
    orig_results = run_5fold_cv_for_extractor(orig_extract, dataset)

    logger.info("Running 5-Fold CV for Graphify-Inspired Extractor...")
    graphify_results = run_5fold_cv_for_extractor(graphify_extract, dataset)

    comparison_report = {
        "seeds": DEFAULT_SEEDS,
        "dataset_summary": {
            "total_samples": dataset.total_samples,
            "unique_scenarios": dataset.unique_scenarios,
            "positive_samples": dataset.positive_count,
            "negative_samples": dataset.negative_count,
        },
        "original_extractor": orig_results,
        "graphify_extractor": graphify_results,
    }

    out_path = Path("artifacts/metrics/graphify_cv_comparison_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")
    logger.info("Comparison report written to %s", out_path)

    print("\n" + "="*80)
    print("5-FOLD STRATIFIED SCENARIO-GROUPED CV COMPARISON REPORT")
    print("="*80)
    print(f"{'Metric':<30} | {'Original Extractor':<22} | {'Graphify Extractor':<22}")
    print("-" * 80)
    print(f"{'Parser Coverage':<30} | {orig_results['mean_parse_coverage']*100:<21.1f}% | {graphify_results['mean_parse_coverage']*100:<21.1f}%")
    print(f"{'Signature Extraction Rate':<30} | {orig_results['mean_signatures_coverage']*100:<21.1f}% | {graphify_results['mean_signatures_coverage']*100:<21.1f}%")
    print(f"{'Mean ROC-AUC':<30} | {orig_results['mean_roc_auc']:<22.4f} | {graphify_results['mean_roc_auc']:<22.4f}")
    print(f"{'Mean PR-AUC':<30} | {orig_results['mean_pr_auc']:<22.4f} | {graphify_results['mean_pr_auc']:<22.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
