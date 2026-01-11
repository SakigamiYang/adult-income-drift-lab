# coding: utf-8
from pathlib import Path

import pandas as pd

from data import load_adult_income
from split import build_splits
from train import evaluate_model_on_split

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)

    df, _ = load_adult_income()
    splits = build_splits(df)

    rows = []
    for split_name in ["A_random", "B_age_shift"]:
        train_df, test_df = splits[split_name]

        for model_name in ["logistic", "tree"]:
            metrics = evaluate_model_on_split(model_name, train_df, test_df)
            rows.append({"split": split_name, "model": model_name, **metrics})

    out = pd.DataFrame(rows)
    out_path = REPORT_DIR / "model_stability_comparison.csv"
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
