# Formal Hypotheses & Statistical Protocol: Graph Data-Flow Pre-Filter

- **Document Version:** 1.0.0  
- **Date:** August 2026  
- **Target Component:** `scenarios/debate/pre_filter.py` & `scripts/train_pre_filter.py`  
- **Baseline Metric Reference:** `artifacts/models_near_dedup_test/cv_holdout_metrics.json` ($n=247$ deduplicated out-of-fold samples across 51 scenario groups)

---

## 1. Executive Context & Empirical Baseline

Diagnostic evaluation of Stage B (TF-IDF + XGBoost) and Stage C (SetFit embeddings) on unseen scenario-grouped holdouts ($n=247$) established:

- **Pooled Out-of-Fold Balanced Accuracy:** `0.4692 ± 0.059` (statistically indistinguishable from random chance `0.5000`).
- **Pooled Out-of-Fold ROC-AUC:** `0.4052` (within $1.6\sigma$ of chance).
- **Pooled Out-of-Fold PR-AUC:** `0.2864` (against random baseline `0.3117`).
- **Sensitivity / TPR:** `0.2208` (misses 77.9% of positive candidates at standard $0.50$ threshold).
- **False Negative Count:** $60$ out of $77$ positive candidates falsely rejected.
- **False Positive Count:** $48$ out of $170$ negative candidates misclassified due to surface security keywords (`malloc`, `memcpy`, `overflow`).

---

## 2. Formal Hypotheses Specification

### 2.1 Null Hypothesis ($H_0$)
> **$H_0$**: Replacing surface text features (TF-IDF n-grams and text embeddings) with deterministic semantic data-flow reachability signatures yields **no statistically significant increase** in 5-fold Stratified Scenario-Grouped out-of-fold ROC-AUC ($p \ge \alpha_{\text{adjusted}}$ via the Holm Step-Down procedure) across $N \ge 5$ paired random seed runs on 100% unseen scenario holdouts ($n=247$).

### 2.2 Alternative Hypothesis ($H_1$)
> **$H_1$**: Deterministic semantic data-flow reachability signatures achieve a **statistically significant ROC-AUC gain** ($\Delta \text{ROC-AUC} \ge +0.25$, targeting $\text{ROC-AUC} \ge 0.70$) with $p < \alpha_{\text{adjusted}}$, and the two-tailed **Hodges-Lehmann 95% Confidence Interval** for the performance delta strictly excludes $0.0$.

---

## 3. Failure Bucket Disaggregation Matrix

To prove or disprove $H_1$, candidate graph models will be evaluated against four explicit failure buckets disaggregated from the baseline $n=247$ holdout dataset:

| Failure Bucket ID | Sample Size ($n=247$) | Baseline Behavior (Text Model) | Target Behavior (Graph Signature Model) | Primary Metrics |
| :--- | :---: | :--- | :--- | :--- |
| **Bucket FP-Keyword** | $48$ | Falsely accepts rejected code because surface security terms (`malloc`, `overflow`) are present. | Identifies absence of untrusted source-to-sink reachability path $\rightarrow$ **Rejects**. | False Positive Rate ($\text{FPR} \le 0.15$) |
| **Bucket FN-NoKeyword** | $60$ | Falsely rejects valid code because surface security keywords are absent. | Identifies structural reachability path from untrusted input to sink $\rightarrow$ **Passes**. | Sensitivity / TPR ($\text{TPR} \ge 0.85$) |
| **Bucket TN-Junk** | $122$ | Correctly rejected syntax junk or incomplete fragments. | Fast-rejected via Stage A `NEGATIVE_RULES` + AST parse failure. | Specificity ($\text{TNR} \ge 0.85$) |
| **Bucket TP-Clean** | $17$ | Correctly passed valid security attempts. | Verified via data-flow reachability path $\rightarrow$ **Passes**. | Precision / PPV ($\text{PPV} \ge 0.75$) |

---

## 4. Statistical Testing Protocol & Acceptance Bounds

Following [EVALUATION_DISCIPLINE_GUIDE.md](file:///Users/surfiniaburger/Desktop/modular-metacog-swarm-v3/agent_training/silver-one/docs/EVALUATION_DISCIPLINE_GUIDE.md):

1. **Paired-Seed Evaluation**: $N=5$ paired random seeds (`--seed 42, 1337, 2026, 7, 99`).
2. **Multiplicity Adjustment**: **Holm Step-Down Procedure** across macro balanced accuracy, ROC-AUC, and PR-AUC.
3. **Non-Parametric CI Estimation**: **Hodges-Lehmann 95% Confidence Intervals** computed over paired out-of-fold differences.
4. **Hard Acceptance Floor**:
   - `accepted_logic_error_rate == 0.0`
   - Out-of-fold $\text{ROC-AUC} \ge 0.7000$
   - Out-of-fold $\text{PR-AUC} \ge 0.6000$
   - Sensitivity (TPR) at advisory threshold ($0.10$) $\ge 0.9000$.
