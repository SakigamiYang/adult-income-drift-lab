# Results

This document summarizes the empirical findings of the data drift and model stability experiments.  
All results are reported on the cleaned UCI Adult Income dataset under different distribution shift scenarios.

---

## 1. Data Drift Results

### 1.1 Age-Based Subpopulation Shift

The age-based split (`age ≤ 35` for training, `age > 35` for testing) induces a strong and structured distribution shift.

#### Categorical Features (PSI)

The most affected categorical features are:

- **marital-status** (PSI ≈ 1.21)
- **relationship** (PSI ≈ 0.84)
- **workclass**, **education**, **occupation** (moderate PSI)

These features are not independent of age and represent socioeconomic factors that evolve across demographic groups.

#### Numeric Features (KS / Wasserstein)

Among numeric features, the strongest drift is observed in:

- **age** (by construction; KS = 1.0)
- **hours-per-week**
- **education-num**
- **capital-gain / capital-loss**

These results indicate that the shift is **multi-dimensional**, affecting both demographic and economic attributes rather than a single variable.

---

## 2. Model Performance Under Distribution Shift

Two models are evaluated:

- Logistic Regression (linear baseline)
- Tree-based model (HistGradientBoosting)

Performance is compared between the random split (i.i.d.) and the age-based shift.

### 2.1 Overall Performance Metrics

| Split        | Model     | ROC-AUC | PR-AUC | Brier |
|-------------|-----------|---------|--------|--------|
| A_random    | Logistic  | 0.906 | 0.770 | 0.104 |
| A_random    | Tree      | 0.925 | 0.825 | 0.092 |
| B_age_shift | Logistic  | 0.818 | 0.706 | 0.218 |
| B_age_shift | Tree      | 0.890 | 0.836 | 0.130 |

---

## 3. Key Observations

### 3.1 Logistic Regression Is Highly Sensitive to Shift

Under the age-based shift:

- ROC-AUC drops by nearly **0.09**
- PR-AUC decreases noticeably
- **Brier score more than doubles**

This indicates that the linear model suffers not only from reduced ranking performance but also from **severe probability miscalibration**.

---

### 3.2 Tree-Based Model Shows Higher Robustness

The tree-based model exhibits:

- Smaller degradation in ROC-AUC
- Stable or slightly improved PR-AUC
- Moderate increase in Brier score

This suggests that non-linear feature interactions help absorb part of the covariate shift, although calibration still degrades.

---

## 4. Ranking vs Calibration

Visual diagnostics (ROC, Precision–Recall, and calibration curves) highlight an important distinction:

- Ranking metrics (ROC-AUC, PR-AUC) degrade moderately under shift
- **Calibration degrades more strongly**, especially for logistic regression

This implies that models may still rank instances reasonably well while producing unreliable probability estimates.

---

## 5. Relationship Between Drift and Performance

Although several features exhibit strong distribution drift, the magnitude of drift alone does not fully explain performance degradation.

In particular:

- Some features with high PSI or KS statistics do not lead to proportional drops in ranking performance
- Model architecture plays a significant role in robustness

This confirms that **feature-level drift is a necessary but insufficient indicator of model instability**.

---

## 6. Practical Implications

From an applied perspective:

- Drift detection should precede model retraining decisions
- Linear models require closer monitoring and recalibration under subpopulation shifts
- Tree-based models offer improved robustness but are not immune to calibration issues

---

## 7. Summary

The experiments demonstrate that:

1. Realistic demographic shifts induce complex, correlated feature drift
2. Logistic regression is particularly vulnerable to such shifts
3. Tree-based models are more stable but still affected
4. Calibration metrics provide critical information beyond ranking performance

These results highlight the importance of **joint drift detection and stability evaluation** in applied machine learning systems.
