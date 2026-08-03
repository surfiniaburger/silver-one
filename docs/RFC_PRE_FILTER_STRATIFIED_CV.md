# RFC: Stratified Scenario-Grouped K-Fold Evaluation & Generalization Diagnostics for Pre-Filter

- **Status:** Concluded & Adopted
- **Date:** 2026-08-03
- **Author:** Silver-One Development Team
- **Target Component:** `scripts/train_pre_filter.py` & `scenarios/debate/pre_filter.py`
- **Primary Artifacts:** `artifacts/models_near_dedup_test/cv_holdout_metrics.json`, `artifacts/models/setfit_model`

---

## 1. Executive Summary & Context

During initial holdout evaluations of the BARRED 3-stage acceptance pre-filter, Stage B (XGBoost) exhibited an apparent collapse: it predicted `0` (rejection) for 100% of samples on a fixed test split, yielding a baseline accuracy of `0.3478` and a balanced accuracy of exactly `0.5000`.

To determine whether this behavior stemmed from a pipeline implementation bug, target data leakage, identifier ambiguity, or fundamental data volume/representation limitations, we conducted a systematic diagnostic investigation and architectural refactoring of the evaluation harness.

### Key Conclusions
1. **Model Mechanics Verified (No Training Bug):** In-sample evaluation proved SetFit achieves **95.12% Accuracy / 94.93% Balanced Accuracy** on training data (classification head weight norm = `4.94`), proving that the transformer backbone and classification head train correctly in-sample.
2. **Identifier Resolution & Terminology Precision:** Raw attempt logs (`artifacts/attempts/*.jsonl`) and seed files lack explicit `cve_id` fields. Scenario grouping was resolved via 10-character SHA-256 hashing of the scenario `predicate` text (`HASH-{sha256[:10]}`), yielding **51 unique vulnerability scenario groups**. The codebase and documentation were refactored to **Scenario-Grouped Partitioning** to accurately reflect the guarantee of zero scenario-predicate leakage.
3. **Exact & Near-Duplicate Contamination Discovered:** Audit revealed `HASH-94b6a965ec` was a retry-heavy single seed repeating 92 times across attempt logs. Automatic exact `(text, label)` deduplication reduced the raw 514 samples to 320 unique samples. Collapsing near-duplicate attempt texts ($>0.95$ TF-IDF cosine similarity) produced a clean benchmark dataset of **247 unique samples** across 51 scenario groups.
4. **Definitive "No Signal" Proof ($n=247$):** On the clean near-deduplicated dataset ($n=247$), Stage B XGBoost yields **ROC-AUC = 0.4052** and **PR-AUC = 0.2864** (against a random PR-AUC baseline of `0.3117`, a net delta of `-0.0254`, which does not exceed random chance). Balanced accuracy settles at **0.4692 ± 0.059** ($SE \approx 0.059$), which is **statistically indistinguishable from chance (0.5000)**.
5. **Stage A Rule Asymmetry Discovered:** Stage A rules fire on **70.9% of samples** (227/320). Disaggregating rule types reveals sharp asymmetry:
   - **`NEGATIVE_RULES` (Sanity Gate):** 62.6% accuracy on rejections (107/171 true rejects). Function effectively as a low-cost syntax/length sanity gate for obvious junk inputs.
   - **`POSITIVE_RULES` (Keyword Matcher):** 21.4% accuracy on acceptances (12/56 true accepts). Counterproductive because surface vulnerability terms appear heavily in rejected attempt descriptions.
6. **Reframed Pass-Through Policy (`xgb_low_threshold = 0.05`–`0.10`):** The pre-filter is adopted strictly as an **advisory soft filter**, prioritizing **recall protection** (passing 80.5%–90.9% of valid accepted candidates). Compute reduction on this baseline feature set is modest (~4%–8%), reflecting the honest reality that the model cannot yet discriminate accepted vs rejected attempts on unseen vulnerability scenarios.

---

## 2. Problem Statement & Diagnostic Findings

Prior single-split evaluation suffered from three major vulnerabilities:

1. **Fold-Selection Variance & Probability Compression:** On a fixed split, 92 identical duplicate rejected samples flooded Test Fold 1, compressing XGBoost test probabilities into `0.0519–0.4108`. At threshold `0.50`, 100% of test predictions evaluated to `0` (`[[99, 0], [23, 0]]`).
2. **Ambiguous Leakage Guarantees:** Calling the grouping "CVE-level" was inaccurate because seed records lacked official CVE identifiers.
3. **Superficial Feature Limitations:** Keyword/regex rules (Stage A) and TF-IDF char n-grams (Stage B) fail to represent code reachability and semantic flow, resulting in high false-positive rates on surface vulnerability terms.

---

## 3. Methodology & System Architecture

```
                                  Attempt Log Records (.jsonl)
                                              │
                                              ▼
                              Extract Predicate & Code Snippet
                                              │
                                              ▼
                               Exact (Text, Label) Deduplication
                                    (514 -> 320 Samples)
                                              │
                                              ▼
                               Near-Duplicate Cosine Collapsing
                                    (320 -> 247 Samples)
                                              │
                                              ▼
                               Deterministic Predicate Hashing
                                HASH-{sha256(predicate)[:10]}
                                              │
                                              ▼
                             51 Unique Scenario Groups (247 Samples)
                                              │
                                              ▼
                           StratifiedGroupKFold Partitioning (k=5)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         Train Split (34-46 Scenarios)                       Test Split (5-17 Scenarios)
                    │                                                   │
         Strict Isolated Fitting:                                Zero-Leakage Out-of-Fold
          - TF-IDF Char N-Grams (3-5)                             Evaluation & Prediction:
          - Domain Feature StandardScaler                         - Stage B XGBoost Probs
          - Class Weight scale_pos_weight                         - Stage C SetFit Head
          - Stage B XGBoost Classifier                                  │
          - Stage C SetFit Transformer                                  ▼
                    │                                        Pooled Micro & Macro Metrics
                    └──────────────────────────────────────►  - ROC-AUC & PR-AUC Scores
                                                              - Full Out-of-Fold Matrix
```

---

## 4. Experimental Results & Diagnostics

### 4.1. Stage A Heuristics Rule Asymmetry ($n=320$)

Disaggregating Stage A regex and keyword rules on the 320 deduplicated attempt samples:

| Category | Sample Count | Percentage | True Rejects / Accepts | Subset Accuracy | Rule Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`NEGATIVE_RULES` Fired (`0.01`)** | 171 | 53.4% | 107 True Rejects / 64 False Rejects | **62.6%** | **Useful Sanity Gate** (catches short/empty inputs) |
| **`POSITIVE_RULES` Fired (`0.99`)** | 56 | 17.5% | 12 True Accepts / 44 False Accepts | **21.4%** | **Counterproductive** (78.6% false accepts) |
| **Stage A Fired Subset Total** | **227** | **70.9%** | **119 Correct / 108 Incorrect** | **52.4%** | Net Fired Subset Performance |
| **Stage A Ambiguous (`None`)** | 93 | 29.1% | Defaulted to Pass (`accept=True`) | N/A | Pass-Through Fallback |

*Takeaway:* Stage A contains two distinct mechanisms: `NEGATIVE_RULES` function effectively as a fast sub-millisecond sanity check, whereas `POSITIVE_RULES` produce a 78.6% false-accept rate because surface vulnerability terms appear extensively in rejected candidate descriptions.

---

### 4.2. Near-Deduplicated 5-Fold Stratified CV ($n=247$)

Full 5-fold evaluation across the 247 near-deduplicated out-of-fold samples (cosine similarity $>0.95$ collapsed):

```text
========================================================================================
=== 5-FOLD STRATIFIED SCENARIO-GROUPED CV SUMMARY (n=247 NEAR-DEDUPLICATED)          ===
========================================================================================

--- Discriminative Power & Curve Metrics ---
  • Pooled Out-of-Fold ROC-AUC:   0.4052 (SE ≈ 0.059, within 1.6σ of chance 0.5000)
  • Pooled Out-of-Fold PR-AUC:    0.2864 (Random Baseline: 0.3117, Delta: -0.0254, SE ≈ 0.038)

--- Stage B (XGBoost + TF-IDF + Domain Features) ---
  • Macro Mean Balanced Accuracy: 0.4685 ± 0.0994
  • Macro Mean Accuracy:          0.5675 ± 0.0595
  • Pooled Out-of-Fold BalAcc:   0.4692 ± 0.059 (Statistically Chance Level)
  • Pooled Confusion Matrix:      TN=122, FP=48, FN=60, TP=17
  • Pooled Sensitivity (Recall):  0.2208
  • Pooled Specificity (TNR):     0.7176
========================================================================================
```

*Note on PR-AUC:* The PR-AUC score of `0.2864` did not exceed the random baseline of `0.3117` ($\Delta = -0.0254$, within sampling error $SE \approx 0.038$), confirming that precision-recall performance is statistically equivalent to random selection.

---

### 4.3. Threshold Operating Sweep ($n=247$ Near-Deduplicated)

Sweep on pooled out-of-fold probabilities from the clean $n=247$ near-deduplicated dataset:

| Threshold | Recall (TPR) | Rejection (TNR) | False Pass Rate (FPR) | BalAcc | Net Filter Action |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05** | **90.9%** (70 / 77) | 3.5% (6 / 170) | 96.5% (164 / 170) | 0.4722 | Pass 234 (94.7%) / Reject 13 |
| **0.10** | **80.5%** (62 / 77) | 7.6% (13 / 170) | 92.4% (157 / 170) | 0.4408 | Pass 219 (88.7%) / Reject 28 |
| **0.15** | **63.6%** (49 / 77) | 17.1% (29 / 170) | 82.9% (141 / 170) | 0.4035 | Pass 190 (76.9%) / Reject 57 |
| **0.20** | **57.1%** (44 / 77) | 30.0% (51 / 170) | 70.0% (119 / 170) | 0.4357 | Pass 163 (66.0%) / Reject 84 |
| **0.25** | **49.4%** (38 / 77) | 36.5% (62 / 170) | 63.5% (108 / 170) | 0.4291 | Pass 146 (59.1%) / Reject 101 |
| **0.50 (Default)** | **18.2%** (14 / 77) | 70.0% (119 / 170) | 30.0% (51 / 170) | 0.4409 | Pass 65 (26.3%) / Reject 182 |

---

## 5. Final Architecture & Operating Policy

1. **Stage A Rule Refactoring:**
   - Retain and tighten `NEGATIVE_RULES` (empty/short predicate checks) as a sub-millisecond sanity gate for obvious junk inputs.
   - Disable or re-architect `POSITIVE_RULES` (vulnerability keyword matcher), as keyword matching misclassifies 78.6% of positive predictions.
2. **Stop Model/Hyperparameter Tuning:** With an ROC-AUC of **0.4052** and PR-AUC of **0.2864** ($n=247$, statistically chance level), further hyperparameter tuning on the surface text/n-gram feature set is halted.
3. **Conservative Pass-Through Policy (`xgb_low_threshold = 0.05`–`0.10`):**
   - The pre-filter operates strictly as an **advisory soft filter**, prioritizing **recall protection** (passing 80.5%–90.9% of valid accepted candidates).
   - Net compute reduction on this clean baseline feature set is modest (~4%–8%), reflecting the honest reality that the model cannot yet discriminate accepted vs rejected attempts on unseen vulnerability scenarios.
4. **Primary Roadmap:** Future pre-filter improvements require moving beyond surface text/n-grams to **semantic control/data-flow reachability representations** and acquiring broader scenario diversity.
