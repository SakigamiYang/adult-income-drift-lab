# coding: utf-8
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

NUMERIC_DTYPES = ("int64", "float64")


def compute_psi(
        expected: pd.Series,
        actual: pd.Series,
        eps: float = 1e-6,
) -> float:
    """
    Numerically stable PSI for categorical variables.

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)
    where e_i and a_i are proportions in expected/train and actual/test.

    We apply epsilon-smoothing to avoid log(0) and division by zero.
    """
    exp_dist = expected.value_counts(normalize=True)
    act_dist = actual.value_counts(normalize=True)

    all_bins = exp_dist.index.union(act_dist.index)

    # Reindex to common bins and apply epsilon smoothing
    e = exp_dist.reindex(all_bins, fill_value=0.0).astype(float).to_numpy()
    a = act_dist.reindex(all_bins, fill_value=0.0).astype(float).to_numpy()

    e = np.clip(e, eps, 1.0)
    a = np.clip(a, eps, 1.0)

    psi = np.sum((a - e) * np.log(a / e))
    return float(psi)


def drift_numeric(train: pd.Series, test: pd.Series) -> Dict[str, float]:
    ks_stat, ks_p = ks_2samp(train, test)
    wd = wasserstein_distance(train, test)
    return {
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "wasserstein": float(wd),
    }


def drift_categorical(train: pd.Series, test: pd.Series) -> Dict[str, float]:
    psi = compute_psi(train, test)
    return {"psi": psi}


def compute_drift_table(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str = "target",
) -> pd.DataFrame:
    rows: List[Dict] = []

    for col in df_train.columns:
        if col == target_col:
            continue

        if df_train[col].dtype.name in NUMERIC_DTYPES:
            stats = drift_numeric(df_train[col], df_test[col])
            row = {
                "feature": col,
                "type": "numeric",
                **stats,
                "psi": np.nan,
            }
        else:
            stats = drift_categorical(df_train[col], df_test[col])
            row = {
                "feature": col,
                "type": "categorical",
                "ks_stat": np.nan,
                "ks_pvalue": np.nan,
                "wasserstein": np.nan,
                **stats,
            }

        rows.append(row)

    out = pd.DataFrame(rows)

    # Rank within type for readability
    out["rank_numeric"] = np.where(
        out["type"].eq("numeric"),
        out["ks_stat"].rank(ascending=False, method="min"),
        np.nan,
    )
    out["rank_categorical"] = np.where(
        out["type"].eq("categorical"),
        out["psi"].rank(ascending=False, method="min"),
        np.nan,
    )

    # Default display ordering: categorical by PSI, numeric by KS
    out = out.sort_values(
        by=["type", "psi", "ks_stat"],
        ascending=[True, False, False],
        na_position="last",
    )

    return out
