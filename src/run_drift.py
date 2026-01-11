# coding: utf-8
from pathlib import Path

from data import load_adult_income
from drift import compute_drift_table
from split import build_splits

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)

    df, _ = load_adult_income()
    splits = build_splits(df)

    train, test = splits["B_age_shift"]

    drift_df = compute_drift_table(train, test)
    out_path = REPORT_DIR / "drift_overview_age_shift.csv"
    drift_df.to_csv(out_path, index=False)

    print("Top 10 drifted features (Age Shift):")
    print(drift_df.head(10).to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
