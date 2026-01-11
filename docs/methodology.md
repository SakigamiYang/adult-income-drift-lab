# Methodology

This document describes the methodological choices behind the data drift and model stability analysis conducted in this project.  
The focus is on **quantifying distribution shift**, **evaluating model robustness**, and **separating ranking performance from probability calibration**.

---

## 1. Problem Setting

Most supervised learning models are trained under the assumption that training and test data are independently and identically distributed (i.i.d.).  
In real-world applications, this assumption is frequently violated due to **covariate shift**, **subpopulation shift**, or **label prior shift**.

The goal of this project is not to maximize predictive accuracy under i.i.d. conditions, but to analyze:

- How feature distributions change under realistic shifts
- How such shifts affect different classes of models
- Which evaluation metrics are most sensitive to distributional changes

---

## 2. Dataset and Preprocessing

### 2.1 Dataset

The experiments use the **UCI Adult Income Dataset** (via OpenML, ID 1590), a widely studied benchmark for binary classification.

- Target variable: income `>50K`
- Features: demographic and socioeconomic attributes
- Cleaned dataset size: 45,222 samples

### 2.2 Missing Value Handling

The dataset contains missing values encoded as the string `"?"`.  
The preprocessing pipeline applies the following explicit and reproducible policy:

1. Normalize `"?"` to `NaN`
2. Drop rows containing missing values

This choice prioritizes transparency and reproducibility over aggressive imputation.

---

## 3. Distribution Shift Scenarios

Three train–test split scenarios are constructed to isolate different types of distribution shift.

### 3.1 Random Split (Baseline)

A stratified random split serves as an i.i.d. reference scenario.

- Purpose: establish upper-bound performance
- Assumption: identical train and test distributions

### 3.2 Age-Based Subpopulation Shift

To simulate a realistic demographic shift:

- Training set: `age ≤ 35`
- Test set: `age > 35`

Although the split is defined by a single variable (`age`), this induces correlated shifts in multiple socioeconomic features such as marital status, education, and working hours.

This scenario is the primary focus of the analysis.

### 3.3 Label Prior Shift (Auxiliary)

A synthetic label prior shift is created by resampling positives and negatives to alter class prevalence between train and test sets.

This scenario is included for methodological completeness but is not the main emphasis of the project.

---

## 4. Data Drift Quantification

Before training any predictive model, feature-level distribution shift is quantified to avoid conflating **data drift** with **model failure**.

### 4.1 Numeric Features

For numeric variables, two complementary metrics are used:

#### Kolmogorov–Smirnov Test

- Measures the maximum difference between empirical cumulative distribution functions
- Sensitive to any distributional change
- Provides a statistical hypothesis test (p-value)

#### Wasserstein Distance

- Measures the minimum “cost” of transforming one distribution into another
- Provides a scale-aware notion of distributional difference
- Less sensitive to sample size than KS statistics

### 4.2 Categorical Features

For categorical variables, **Population Stability Index (PSI)** is used.

\[
\mathrm{PSI} = \sum_i (a_i - e_i)\,\log\left(\frac{a_i}{e_i}\right)
\]

where:
- \( e_i \) is the proportion of category \( i \) in the training set
- \( a_i \) is the proportion of category \( i \) in the test set

To ensure numerical stability, **epsilon smoothing** is applied to avoid zero probabilities.

### 4.3 Interpretation

Feature-level drift metrics are **descriptive, not predictive**.  
Large drift does not automatically imply performance degradation; instead, it motivates further model-level analysis.

---

## 5. Models and Training Strategy

Two complementary model families are evaluated to assess robustness under shift.

### 5.1 Logistic Regression

Logistic regression serves as a linear, interpretable baseline.

- Numeric features: standardized
- Categorical features: one-hot encoded
- Optimization: maximum likelihood

This model represents systems relying on global linear coefficients.

### 5.2 Tree-Based Model

A histogram-based gradient boosting classifier is used to capture non-linear interactions.

- Automatically models feature interactions
- Less sensitive to monotonic feature transformations
- Often empirically more robust under covariate shift

---

## 6. Evaluation Metrics

Three evaluation metrics are used to capture different aspects of model behavior.

### 6.1 ROC-AUC

- Measures ranking quality independent of classification threshold
- Insensitive to class imbalance
- Often overly optimistic under distribution shift

### 6.2 Precision–Recall AUC

- Focuses on performance for the positive class
- More informative under class imbalance
- Still primarily reflects ranking performance

### 6.3 Brier Score

\[
\mathrm{Brier} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2
\]

- Measures accuracy of predicted probabilities
- Sensitive to miscalibration
- Particularly important for decision-making systems

---

## 7. Stability and Calibration Analysis

Model stability is evaluated by comparing performance under:

- Random split (i.i.d.)
- Age-based subpopulation shift

In addition to scalar metrics, the following visual diagnostics are used:

- ROC curves
- Precision–Recall curves
- Calibration (reliability) curves

This allows separation of:
- Ranking degradation
- Probability calibration degradation

---

## 8. Methodological Limitations

- The analysis focuses on a single dataset and a single dominant shift scenario
- No causal interpretation is claimed
- Recalibration techniques are not applied, only diagnosed

These limitations are intentional to keep the analysis focused and interpretable.

---

## 9. Summary

This methodology emphasizes a **sequential analysis pipeline**:

1. Define realistic shift scenarios
2. Quantify feature-level distribution drift
3. Evaluate model stability under shift
4. Distinguish ranking performance from probability calibration

Such a workflow is essential for deploying machine learning systems in non-stationary environments.
