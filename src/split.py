# coding: utf-8
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import load_adult_income


@dataclass(frozen=True)
class SplitSummary:
    n_train: int
    n_test: int
    y_rate_train: float
    y_rate_test: float


def summarize(train: pd.DataFrame, test: pd.DataFrame) -> SplitSummary:
    return SplitSummary(
        n_train=len(train),
        n_test=len(test),
        y_rate_train=float(train["target"].mean()),
        y_rate_test=float(test["target"].mean()),
    )


# A: Random split (baseline, no shift)
def split_random(
        df: pd.DataFrame,
        test_size: float = 0.25,
        random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# B: Age-based subpopulation shift
def split_age_shift(
        df: pd.DataFrame,
        age_threshold: int = 35,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "age" not in df.columns:
        raise ValueError("Column 'age' not found.")
    tr = df[df["age"] <= age_threshold].copy()
    te = df[df["age"] > age_threshold].copy()
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# C: Label prior shift (synthetic)
def split_label_prior_shift(
        df: pd.DataFrame,
        train_pos_rate: float = 0.15,
        test_pos_rate: float = 0.35,
        n_train: int = 20000,
        n_test: int = 8000,
        random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]

    def sample(n: int, pos_rate: float) -> pd.DataFrame:
        n_pos = int(round(n * pos_rate))
        n_neg = n - n_pos
        pos_idx = rng.choice(pos.index, size=n_pos, replace=False)
        neg_idx = rng.choice(neg.index, size=n_neg, replace=False)
        out = pd.concat([df.loc[pos_idx], df.loc[neg_idx]])
        return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return sample(n_train, train_pos_rate), sample(n_test, test_pos_rate)


def build_splits(df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    return {
        "A_random": split_random(df),
        "B_age_shift": split_age_shift(df, age_threshold=35),
        "C_label_prior_shift": split_label_prior_shift(df),
    }


if __name__ == "__main__":
    df, _ = load_adult_income()
    splits = build_splits(df)

    print("=== Split summaries ===")
    for name, (tr, te) in splits.items():
        s = summarize(tr, te)
        print(
            f"{name:18s} | "
            f"train={s.n_train:6d} y_rate={s.y_rate_train:.4f} | "
            f"test={s.n_test:6d} y_rate={s.y_rate_test:.4f}"
        )
