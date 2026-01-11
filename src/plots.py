# coding: utf-8
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from data import load_adult_income
from split import build_splits
from train import (
    fit_logistic,
    fit_tree,
    predict_proba_logistic,
    predict_proba_tree,
    split_xy,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def _ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc(model_name: str) -> None:
    _ensure_dir()
    df, _ = load_adult_income()
    splits = build_splits(df)

    plt.figure()

    for split_name in ["A_random", "B_age_shift"]:
        train_df, test_df = splits[split_name]
        _, y_test = split_xy(test_df)

        if model_name == "logistic":
            model = fit_logistic(train_df)
            y_prob = predict_proba_logistic(model, test_df)
        else:
            model = fit_tree(train_df)
            y_prob = predict_proba_tree(model, test_df)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=split_name)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_name})")
    plt.legend()

    out = FIG_DIR / f"roc_{model_name}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def plot_pr(model_name: str) -> None:
    _ensure_dir()
    df, _ = load_adult_income()
    splits = build_splits(df)

    plt.figure()

    for split_name in ["A_random", "B_age_shift"]:
        train_df, test_df = splits[split_name]
        _, y_test = split_xy(test_df)

        if model_name == "logistic":
            model = fit_logistic(train_df)
            y_prob = predict_proba_logistic(model, test_df)
        else:
            model = fit_tree(train_df)
            y_prob = predict_proba_tree(model, test_df)

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(rec, prec, label=split_name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve ({model_name})")
    plt.legend()

    out = FIG_DIR / f"pr_{model_name}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def plot_calibration(model_name: str, n_bins: int = 10) -> None:
    _ensure_dir()
    df, _ = load_adult_income()
    splits = build_splits(df)

    plt.figure()

    for split_name in ["A_random", "B_age_shift"]:
        train_df, test_df = splits[split_name]
        _, y_test = split_xy(test_df)

        if model_name == "logistic":
            model = fit_logistic(train_df)
            y_prob = predict_proba_logistic(model, test_df)
        else:
            model = fit_tree(train_df)
            y_prob = predict_proba_tree(model, test_df)

        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=split_name)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve ({model_name})")
    plt.legend()

    out = FIG_DIR / f"calibration_{model_name}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    for model_name in ["logistic", "tree"]:
        plot_roc(model_name)
        plot_pr(model_name)
        plot_calibration(model_name)

    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
