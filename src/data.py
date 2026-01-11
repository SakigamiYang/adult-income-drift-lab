# coding: utf-8
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


@dataclass(frozen=True)
class DatasetInfo:
    n_rows_raw: int
    n_rows_clean: int
    n_cols: int
    positive_rate: float
    missing_cells: int


def load_adult_income(random_state: int = 42) -> Tuple[pd.DataFrame, DatasetInfo]:
    """
    Load Adult Income dataset from OpenML (id=1590), clean missing values,
    and return a DataFrame with binary target column `target`.
    """
    bunch = fetch_openml(data_id=1590, as_frame=True)
    df = bunch.frame.copy()

    if "class" not in df.columns:
        raise ValueError("Expected column 'class' as target.")

    n_rows_raw = len(df)

    # Normalize '?' to NaN
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("?", np.nan)

    missing_cells = int(df.isna().sum().sum())

    # Drop rows with any missing values
    df = df.dropna(axis=0).reset_index(drop=True)

    # Build binary target
    y = df["class"].astype(str).str.strip()
    y = y.replace({">50K.": ">50K", "<=50K.": "<=50K"})
    df = df.drop(columns=["class"])
    df["target"] = (y == ">50K").astype(int)

    info = DatasetInfo(
        n_rows_raw=n_rows_raw,
        n_rows_clean=len(df),
        n_cols=df.shape[1],
        positive_rate=float(df["target"].mean()),
        missing_cells=missing_cells,
    )
    return df, info


if __name__ == "__main__":
    df, info = load_adult_income()

    print("=== Adult Income Dataset (OpenML 1590) ===")
    print(f"Raw rows:        {info.n_rows_raw}")
    print(f"Missing cells:   {info.missing_cells}")
    print(f"Clean rows:      {info.n_rows_clean}")
    print(f"Columns (incl.): {info.n_cols}")
    print(f"Positive rate:   {info.positive_rate:.4f}")
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
