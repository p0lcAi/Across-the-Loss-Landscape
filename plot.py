from __future__ import annotations

from pathlib import Path

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
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "legend.frameon": False,
    "legend.loc": "upper right",
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.5,
    "legend.columnspacing": 1.5,
    "legend.borderaxespad": 0.5,
    "legend.borderpad": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "pdf.compression": 9,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage{amsmath}",
    "pgf.rcfonts": False,
})

# Paths
ROOT = Path("outputs")

BASELINE_CSV = ROOT / "ecml_resnet18_baseline_only" / "grouped_summary_mean.csv"
VALACC_CSV = ROOT / "ecml_resnet18_growth_valacc_stages" / "grouped_summary_mean.csv"
TRAINLOSS_CSV = ROOT / "ecml_resnet18_growth_trainloss_stages" / "grouped_summary_mean.csv"
PATIENCE_CSV = ROOT / "ecml_resnet18_growth_trainloss_patience" / "grouped_summary_mean.csv"

OUTDIR = Path("figures/resnet")
OUTDIR.mkdir(parents=True, exist_ok=True)


# Helpers
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def add_curvature_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lambda_ratio_new_active"] = (
        df["transition_new_lambda_max_mean"] / df["transition_active_lambda_max_mean"]
    )
    df["trace_ratio_new_active"] = (
        df["transition_new_trace_mean"] / df["transition_active_trace_mean"]
    )
    return df


def sort_by_stages(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("stages").reset_index(drop=True)


def convert_accuracy_columns_to_percent(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["final_test_acc", "final_val_acc"]:
        if col in df.columns:
            df[col] = 100.0 * df[col]
    return df


# Load data
baseline_df = convert_accuracy_columns_to_percent(load_csv(BASELINE_CSV))
val_df = convert_accuracy_columns_to_percent(add_curvature_ratios(sort_by_stages(load_csv(VALACC_CSV))))
train_df = convert_accuracy_columns_to_percent(add_curvature_ratios(sort_by_stages(load_csv(TRAINLOSS_CSV))))
pat_df = convert_accuracy_columns_to_percent(add_curvature_ratios(load_csv(PATIENCE_CSV).sort_values("patience").reset_index(drop=True)))

baseline_test_acc = float(baseline_df["final_test_acc"].iloc[0])

print("Baseline test acc (%):", baseline_test_acc)
print(val_df[["run_group", "stages", "final_test_acc", "transition_barrier_mean", "retention_mean", "lambda_ratio_new_active"]])
print(train_df[["run_group", "stages", "final_test_acc", "transition_barrier_mean", "retention_mean", "lambda_ratio_new_active"]])


# Figure 1: final test accuracy vs number of stages
plt.figure(figsize=(5.5, 4.0))
plt.plot(val_df["stages"], val_df["final_test_acc"], marker="o", label="val_acc trigger")
plt.plot(train_df["stages"], train_df["final_test_acc"], marker="s", label="train_loss trigger")
plt.axhline(y=baseline_test_acc, linestyle="--", label="baseline")

plt.xlabel("Number of stages $S$")
plt.ylabel("Final test accuracy (\%)")
plt.xticks(val_df["stages"].tolist())
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUTDIR / "resnet_testacc_vs_stages.png", dpi=300, bbox_inches="tight")
plt.savefig(OUTDIR / "resnet_testacc_vs_stages.pdf", bbox_inches="tight")
plt.close()


# Figure 2: transition geometry vs number of stages
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

# (a) barrier
axes[0].plot(val_df["stages"], val_df["transition_barrier_mean"], marker="o", label="val_acc")
axes[0].plot(train_df["stages"], train_df["transition_barrier_mean"], marker="s", label="train_loss")
axes[0].set_xlabel("Number of stages $S$")
axes[0].set_ylabel("Mean barrier")
axes[0].set_xticks(val_df["stages"].tolist())
axes[0].set_title("(a) Barrier")

# (b) retention
axes[1].plot(val_df["stages"], val_df["retention_mean"], marker="o", label="val_acc")
axes[1].plot(train_df["stages"], train_df["retention_mean"], marker="s", label="train_loss")
axes[1].set_xlabel("Number of stages $S$")
axes[1].set_ylabel("Mean retention")
axes[1].set_xticks(val_df["stages"].tolist())
axes[1].set_ylim(0.0, 1.05)
axes[1].set_title("(b) Retention")

# (c) curvature ratio
axes[2].plot(val_df["stages"], val_df["lambda_ratio_new_active"], marker="o", label="val_acc")
axes[2].plot(train_df["stages"], train_df["lambda_ratio_new_active"], marker="s", label="train_loss")
axes[2].set_xlabel("Number of stages $S$")
axes[2].set_ylabel(r"$\lambda_{\max}^{\mathrm{new}} / \lambda_{\max}^{\mathrm{act}}$")
axes[2].set_xticks(val_df["stages"].tolist())
axes[2].set_title("(c) Curvature ratio")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
fig.savefig(OUTDIR / "resnet_geometry_vs_stages.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTDIR / "resnet_geometry_vs_stages.pdf", bbox_inches="tight")
plt.close(fig)


# Figure 3: patience ablation for S=10 train_loss
plt.figure(figsize=(8.5, 3.8))

plt.subplot(1, 3, 1)
plt.plot(pat_df["patience"], pat_df["final_test_acc"], marker="o")
plt.xlabel("Patience")
plt.ylabel("Final test acc. (\%)")
plt.title("(a) Test accuracy")

plt.subplot(1, 3, 2)
plt.plot(pat_df["patience"], pat_df["transition_barrier_mean"], marker="o")
plt.xlabel("Patience")
plt.ylabel("Mean barrier")
plt.title("(b) Barrier")

plt.subplot(1, 3, 3)
plt.plot(pat_df["patience"], pat_df["retention_mean"], marker="o")
plt.xlabel("Patience")
plt.ylabel("Mean retention")
plt.ylim(0.0, 1.05)
plt.title("(c) Retention")

plt.tight_layout()
plt.savefig(OUTDIR / "resnet_trainloss_patience_ablation.png", dpi=300, bbox_inches="tight")
plt.savefig(OUTDIR / "resnet_trainloss_patience_ablation.pdf", bbox_inches="tight")
plt.close()

print(f"Saved figures to: {OUTDIR.resolve()}")