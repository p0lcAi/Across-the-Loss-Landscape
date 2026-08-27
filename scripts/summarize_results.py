from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize replicated experiment runs")
    parser.add_argument("output_root", type=str)
    args = parser.parse_args()

    root = Path(args.output_root)
    input_path = root / "all_runs_summary.csv"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No rows found in {input_path}")

    group_cols = [
        "run_group",
        "criterion",
        "growth_enabled",
        "growth_mode",
        "stages",
        "patience",
        "dataset",
        "arch",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "seed"]

    mean_df = df.groupby(group_cols, as_index=False)[numeric_cols].mean()
    std_df = df.groupby(group_cols, as_index=False)[numeric_cols].std(ddof=1)

    mean_df.to_csv(root / "grouped_summary_mean.csv", index=False)
    std_df.to_csv(root / "grouped_summary_std.csv", index=False)

    merged = mean_df.copy()
    for col in numeric_cols:
        merged[f"{col}_std"] = std_df[col].to_numpy()
    merged.to_csv(root / "grouped_summary_mean_std.csv", index=False)

    print(f"Wrote summaries to {root}")


if __name__ == "__main__":
    main()
