from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 12,
    "figure.figsize": (6, 4),
    "axes.titlesize": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
    "pdf.compression": 9,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage{amsmath}",
    "pgf.rcfonts": False,
})

ROOT = Path("outputs")
OUTDIR = Path("figures/posthoc")
OUTDIR.mkdir(parents=True, exist_ok=True)

ALLOWED_STAGES = {3, 5, 10}

# Some keywords we want to exclude.
EXCLUDE_KEYWORDS = [
    "aggressive",
    "conservative",
    "s20",
    "schedule",
    "smoke",
    "debug",
    "patience",
    "tmp",
]


def load_summary_json(run_dir: Path) -> Dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "summary" in obj and isinstance(obj["summary"], dict):
        return obj["summary"]
    return obj


def collect_transition_files(root: Path) -> List[Path]:
    return sorted(root.rglob("stage_transitions.csv"))


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den
    return out.replace([float("inf"), float("-inf")], pd.NA)


def load_all_transitions(root: Path) -> pd.DataFrame:
    paths = collect_transition_files(root)
    all_dfs = []

    if len(paths) == 0:
        raise FileNotFoundError(f"No stage_transitions.csv found under {root.resolve()}")

    for csv_path in paths:
        run_dir = csv_path.parent

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path} (read error: {e})")
            continue

        if len(df) == 0:
            print(f"Skipping {csv_path} (empty)")
            continue

        meta = load_summary_json(run_dir)

        df = df.copy()
        df["run_dir"] = str(run_dir)
        df["run_name"] = meta.get("run_name", run_dir.name)
        df["criterion"] = meta.get("criterion", "unknown")
        df["stages"] = meta.get("stages", pd.NA)
        df["patience"] = meta.get("patience", pd.NA)
        df["seed"] = meta.get("seed", pd.NA)
        df["growth_enabled"] = meta.get("growth_enabled", True)

        if "new_lambda_max" in df.columns and "active_lambda_max" in df.columns:
            df["lambda_ratio_new_active"] = safe_div(df["new_lambda_max"], df["active_lambda_max"])

        if "new_trace" in df.columns and "active_trace" in df.columns:
            df["trace_ratio_new_active"] = safe_div(df["new_trace"], df["active_trace"])

        if "new_trace_per_param" in df.columns and "active_trace_per_param" in df.columns:
            df["trace_per_param_ratio_new_active"] = safe_div(
                df["new_trace_per_param"], df["active_trace_per_param"]
            )

        all_dfs.append(df)

    if len(all_dfs) == 0:
        raise RuntimeError("Found transition CSVs, but none could be loaded as non-empty DataFrames.")

    return pd.concat(all_dfs, ignore_index=True)


def debug_print_unique_values(df: pd.DataFrame) -> None:
    print("\n=== DEBUG: unique run_dir values ===")
    for x in sorted(df["run_dir"].astype(str).unique()):
        print(x)

    print("\n=== DEBUG: unique run_name values ===")
    for x in sorted(df["run_name"].astype(str).unique()):
        print(x)

    print("\n=== DEBUG: unique criterion values ===")
    print(sorted(df["criterion"].astype(str).unique()))

    print("\n=== DEBUG: unique stages values ===")
    vals = pd.to_numeric(df["stages"], errors="coerce").dropna().unique().tolist()
    print(sorted(vals))

    if "patience" in df.columns:
        print("\n=== DEBUG: unique patience values ===")
        pvals = pd.to_numeric(df["patience"], errors="coerce").dropna().unique().tolist()
        print(sorted(pvals))


def filter_to_main_ecml_experiments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep criterions
    df = df[df["criterion"].isin(["val_acc", "train_loss"])].copy()

    # Keep stages
    df["stages"] = pd.to_numeric(df["stages"], errors="coerce")
    df = df[df["stages"].isin(list(ALLOWED_STAGES))].copy()

    # Exclude some auxiliary experiments
    pattern = "|".join(EXCLUDE_KEYWORDS)
    df = df[
        ~df["run_dir"].astype(str).str.contains(pattern, case=False, na=False)
    ].copy()
    df = df[
        ~df["run_name"].astype(str).str.contains(pattern, case=False, na=False)
    ].copy()

    return df


def print_spearman(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        print(f"{title}: not enough data")
        return
    rho = sub[x].corr(sub[y], method="spearman")
    print(f"{title}: Spearman rho = {rho:.4f} (n={len(sub)})")


def print_groupwise_spearman(df: pd.DataFrame, x: str, y: str, group_col: str) -> None:
    print(f"\nGroupwise Spearman for {x} vs {y} by {group_col}:")
    for g, subdf in df.groupby(group_col):
        sub = subdf[[x, y]].dropna()
        if len(sub) < 3:
            print(f"  {g}: not enough data")
            continue
        rho = sub[x].corr(sub[y], method="spearman")
        print(f"  {g}: rho = {rho:.4f} (n={len(sub)})")


def scatter_by_criterion_and_stages(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    markers = {3: "o", 5: "s", 10: "^"}
    colors = {"val_acc": "tab:blue", "train_loss": "tab:orange"}

    plotted_labels = set()

    for _, row in df[[x, y, "criterion", "stages"]].dropna().iterrows():
        crit = str(row["criterion"])
        stg = int(row["stages"])

        if stg not in ALLOWED_STAGES:
            continue

        label = f"{crit}, $S={stg}$"

        ax.scatter(
            row[x],
            row[y],
            marker=markers[stg],
            color=colors.get(crit, "gray"),
            alpha=0.8,
            s=70,
            label=(label if label not in plotted_labels else None),
        )
        plotted_labels.add(label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def add_spearman_text(ax, df: pd.DataFrame, x: str, y: str) -> None:
    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        txt = "Spearman: n/a"
    else:
        rho = sub[x].corr(sub[y], method="spearman")
        txt = rf"Spearman $\rho={rho:.2f}$"

    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
    )


def merge_legends_from_axes(axes):
    handles_all = []
    labels_all = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles_all.extend(h)
        labels_all.extend(l)

    label_to_handle = {}
    for h, l in zip(handles_all, labels_all):
        if l not in label_to_handle:
            label_to_handle[l] = h

    return list(label_to_handle.values()), list(label_to_handle.keys())


def main() -> None:
    df = load_all_transitions(ROOT)

    for col in ["stages", "patience", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    debug_print_unique_values(df)

    df = filter_to_main_ecml_experiments(df)

    if len(df) == 0:
        raise RuntimeError(
            "Filtering removed all transitions. Regarde la sortie DEBUG pour adapter EXCLUDE_KEYWORDS si besoin."
        )

    print("\n=== After filtering ===")
    print(
        df[["run_name", "run_dir", "criterion", "stages", "patience"]]
        .drop_duplicates()
        .sort_values(["criterion", "stages", "run_name"])
        .to_string(index=False)
    )
    print(f"\nRemaining transitions: {len(df)}")

    plot_df = df[df["stages"].isin(list(ALLOWED_STAGES))].copy()
    plot_df.to_csv(OUTDIR / "all_stage_transitions_concat_ecml_main.csv", index=False)

    print("\n=== Global Spearman correlations ===")
    print_spearman(plot_df, "barrier", "endpoint_abs_gap", "barrier vs endpoint_abs_gap")
    print_spearman(plot_df, "leakage", "endpoint_abs_gap", "leakage vs endpoint_abs_gap")
    print_spearman(plot_df, "barrier", "endpoint_degradation", "barrier vs endpoint_degradation")
    print_spearman(plot_df, "lambda_ratio_new_active", "barrier", "lambda_ratio_new_active vs barrier")
    print_spearman(plot_df, "lambda_ratio_new_active", "retained", "lambda_ratio_new_active vs retained")

    if "trace_ratio_new_active" in plot_df.columns:
        print_spearman(plot_df, "trace_ratio_new_active", "barrier", "trace_ratio_new_active vs barrier")
        print_spearman(plot_df, "trace_ratio_new_active", "retained", "trace_ratio_new_active vs retained")

    print_groupwise_spearman(plot_df, "barrier", "endpoint_abs_gap", "criterion")
    print_groupwise_spearman(plot_df, "leakage", "endpoint_abs_gap", "criterion")
    print_groupwise_spearman(plot_df, "lambda_ratio_new_active", "barrier", "criterion")
    print_groupwise_spearman(plot_df, "lambda_ratio_new_active", "retained", "criterion")

    # Figure 1: transition quality
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    scatter_by_criterion_and_stages(
        axes[0],
        plot_df,
        x="barrier",
        y="endpoint_abs_gap",
        xlabel="Barrier",
        ylabel="Absolute endpoint gap",
        title="(a) Barrier vs absolute endpoint gap",
    )
    add_spearman_text(axes[0], plot_df, "barrier", "endpoint_abs_gap")

    scatter_by_criterion_and_stages(
        axes[1],
        plot_df,
        x="leakage",
        y="endpoint_abs_gap",
        xlabel="Leakage",
        ylabel="Absolute endpoint gap",
        title="(b) Leakage vs absolute endpoint gap",
    )
    add_spearman_text(axes[1], plot_df, "leakage", "endpoint_abs_gap")

    handles, labels = merge_legends_from_axes(axes)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(OUTDIR / "posthoc_transition_quality_ecml_main.png")
    fig.savefig(OUTDIR / "posthoc_transition_quality_ecml_main.pdf")
    plt.close(fig)

    # Figure 2: relative curvature and stability
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    scatter_by_criterion_and_stages(
        axes[0],
        plot_df,
        x="lambda_ratio_new_active",
        y="barrier",
        xlabel=r"$\lambda_{\max}^{\mathrm{new}} / \lambda_{\max}^{\mathrm{act}}$",
        ylabel="Barrier",
        title="(a) Curvature ratio vs barrier",
    )
    add_spearman_text(axes[0], plot_df, "lambda_ratio_new_active", "barrier")

    scatter_by_criterion_and_stages(
        axes[1],
        plot_df,
        x="lambda_ratio_new_active",
        y="retained",
        xlabel=r"$\lambda_{\max}^{\mathrm{new}} / \lambda_{\max}^{\mathrm{act}}$",
        ylabel="Retention",
        title="(b) Curvature ratio vs retention",
    )
    add_spearman_text(axes[1], plot_df, "lambda_ratio_new_active", "retained")

    handles, labels = merge_legends_from_axes(axes)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(OUTDIR / "posthoc_curvature_transition_ecml_main.png")
    fig.savefig(OUTDIR / "posthoc_curvature_transition_ecml_main.pdf")
    plt.close(fig)

    print(f"\nSaved outputs to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()