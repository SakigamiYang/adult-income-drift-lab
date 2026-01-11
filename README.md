# Adult Income Drift Lab

**Model Stability and Data Shift Analysis on the UCI Adult Income Dataset**

---

## Overview

This project studies **model stability under data distribution shift** using the UCI Adult Income dataset.
Rather than focusing solely on predictive performance under i.i.d. assumptions, we analyze how **covariate shift and
subpopulation shift** affect both ranking performance and probability calibration of different machine learning models.

The project combines:

- Statistical distribution shift analysis (KS test, PSI, Wasserstein distance)
- Supervised learning (logistic regression and tree-based models)
- Performance degradation and calibration analysis under shift

---

## Dataset

- **Source:** UCI Adult Income Dataset (via OpenML, ID 1590)
- **Task:** Binary classification — income `>50K`
- **Cleaned size:** 45,222 samples
- **Features:** Demographic and socioeconomic attributes (age, education, occupation, marital status, etc.)

Missing values (`?`) are explicitly normalized and rows with missing entries are removed for reproducibility.

---

## Experimental Design

We construct three train–test scenarios:

### A. Random Split (Baseline)

- Stratified random split
- Train and test share the same distribution
- Serves as an i.i.d. reference

### B. Age-Based Subpopulation Shift

- Train: `age ≤ 35`
- Test: `age > 35`
- Simulates a realistic demographic shift

### C. Label Prior Shift (Auxiliary)

- Synthetic resampling to alter positive class prevalence
- Used for methodological comparison (not the main focus)

---

## Data Drift Quantification

Before training any model, we quantify feature-level distribution shift between training and test sets.

### Metrics

- **Numeric features:**

    - Kolmogorov–Smirnov statistic
    - Wasserstein distance
- **Categorical features:**

    - Population Stability Index (PSI) with epsilon smoothing

### Key Findings (Age Shift)

- Although the split is defined by `age`, the strongest drift occurs in **socioeconomic proxy variables**:

    - `marital-status` (PSI ≈ 1.21)
    - `relationship` (PSI ≈ 0.84)
    - `workclass`, `education`, `occupation`
- This indicates a **multi-dimensional covariate shift**, not a single-feature drift.

---

## Models Evaluated

Two complementary models are used:

### Logistic Regression

- StandardScaler for numeric features
- One-hot encoding for categorical features
- Serves as a linear, interpretable baseline

### Tree-Based Model

- `HistGradientBoostingClassifier`
- Captures non-linear feature interactions
- Used to assess robustness under shift

---

## Model Stability Results

### Performance Comparison

| Split       | Model    | ROC-AUC | PR-AUC | Brier |
|-------------|----------|---------|--------|-------|
| A_random    | Logistic | 0.906   | 0.770  | 0.104 |
| A_random    | Tree     | 0.925   | 0.825  | 0.092 |
| B_age_shift | Logistic | 0.818   | 0.706  | 0.218 |
| B_age_shift | Tree     | 0.890   | 0.836  | 0.130 |

### Observations

- **Logistic regression is highly sensitive to subpopulation shift**:

    - Large drops in ROC-AUC and PR-AUC
    - Brier score nearly doubles, indicating severe calibration degradation
- **Tree-based models are more robust**:

    - Ranking performance degrades less under shift
    - Calibration still deteriorates, but remains substantially better than the linear model

---

## Visual Analysis

The following plots are automatically generated:

- ROC curves (Random vs Age Shift)
- Precision–Recall curves
- Calibration (reliability) curves

These figures clearly illustrate that:

- Strong feature-level drift does not always imply proportional ranking degradation
- Probability estimates are significantly more sensitive to distribution shift than ranking metrics

---

## Key Takeaways

1. **Subpopulation shift induces multi-dimensional covariate drift**, especially in correlated socioeconomic features.
2. **Linear models suffer substantial performance and calibration degradation under covariate shift**, even with large
   training samples.
3. **Tree-based models exhibit stronger robustness**, but are not immune and still require monitoring or recalibration.

---

## Reproducibility

All experiments are fully reproducible.

```bash
# create environment
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# environments
export PYTHONPATH="./src:${PYTHONPATH}"

# run stability analysis
python -m src.run_stability

# generate figures
python -m src.plot_metrics
```

---

## Project Structure

```
adult-income-drift-lab/
├── src/
│   ├── data.py
│   ├── split.py
│   ├── drift.py
│   ├── train.py
│   ├── run_stability.py
│   └── plot_metrics.py
├── reports/
│   ├── model_stability_comparison.csv
│   └── figures/
├── docs/
│   ├── methodology.md
│   └── results.md
└── README.md
```

---

## Possible Extensions

- Explicit drift monitoring thresholds for production systems
- Post-shift recalibration (Platt scaling / isotonic regression)
- Decision-aware evaluation under cost-sensitive settings
