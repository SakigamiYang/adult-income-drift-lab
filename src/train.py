# coding: utf-8
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ArrayLike = Union[np.ndarray]


@dataclass(frozen=True)
class EvalResult:
    roc_auc: float
    pr_auc: float
    brier: float


def split_xy(df: pd.DataFrame, target_col: str = "target") -> Tuple[pd.DataFrame, np.ndarray]:
    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col])
    return X, y


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray) -> EvalResult:
    return EvalResult(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return pre


def fit_logistic(train: pd.DataFrame) -> Pipeline:
    X_train, y_train = split_xy(train)

    pre = build_preprocessor(X_train)
    model = Pipeline(
        steps=[
            ("pre", pre),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def predict_proba_logistic(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    X, _ = split_xy(df)
    return model.predict_proba(X)[:, 1]


def fit_tree(train: pd.DataFrame) -> HistGradientBoostingClassifier:
    """
    HistGradientBoosting can handle categorical features if they are pandas 'category' dtype
    and categorical_features='from_dtype' is enabled.
    """
    X_train, y_train = split_xy(train)

    X_train = X_train.copy()
    for col in X_train.columns:
        if X_train[col].dtype.name not in ("int64", "float64"):
            X_train[col] = X_train[col].astype("category")

    model = HistGradientBoostingClassifier(
        max_depth=6,
        random_state=42,
        categorical_features="from_dtype",
    )
    model.fit(X_train, y_train)
    return model


def predict_proba_tree(model: HistGradientBoostingClassifier, df: pd.DataFrame) -> np.ndarray:
    X, _ = split_xy(df)

    X = X.copy()
    for col in X.columns:
        if X[col].dtype.name not in ("int64", "float64"):
            X[col] = X[col].astype("category")

    return model.predict_proba(X)[:, 1]


def evaluate_model_on_split(
        model_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Convenience wrapper returning a flat dict for CSV/reporting.
    """
    if model_name == "logistic":
        model = fit_logistic(train_df)
        y_prob = predict_proba_logistic(model, test_df)
    elif model_name == "tree":
        model = fit_tree(train_df)
        y_prob = predict_proba_tree(model, test_df)
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    _, y_test = split_xy(test_df)
    res = evaluate_binary(y_test, y_prob)

    return {
        "roc_auc": res.roc_auc,
        "pr_auc": res.pr_auc,
        "brier": res.brier,
    }
