import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- IEEE/paper-style settings ----------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# single-column IEEE-like figure size
FIGSIZE = (3.5, 2.6)

input_dir = Path("results")
out_dir = Path("results/analysis")
out_dir.mkdir(parents=True, exist_ok=True)

excel_files = sorted(input_dir.glob("*crash*uavs.xlsx"))

rows = []

metric_cols = [
    "total_acc_revenue_when_arrives_to_depot",
    "total_backedup_revenue_when_arrives_to_depot",
    "total_acc_revenue_when_it_crashes",
    "total_backedup_revenue_when_it_crashes",
]

for excel_path in excel_files:
    m = re.match(r"(\d+(?:\.\d+)?)crash(\d+)uavs\.xlsx$", excel_path.name)
    if not m:
        continue

    cb = float(m.group(1))
    total_uavs = int(m.group(2))

    metrics = pd.read_excel(excel_path, sheet_name="UAV Metrics")
    metrics = metrics.copy()
    metrics.columns = [str(c).strip() for c in metrics.columns]

    missing = [c for c in metric_cols if c not in metrics.columns]
    if missing:
        print(f"Skipping {excel_path.name} because columns are missing: {missing}")
        continue

    for c in metric_cols:
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce").fillna(0)

    rows.append({
        "cb": cb,
        "total_uavs": total_uavs,
        "total_acc_revenue_when_arrives_to_depot":
            metrics["total_acc_revenue_when_arrives_to_depot"].sum(),
        "total_backedup_revenue_when_arrives_to_depot":
            metrics["total_backedup_revenue_when_arrives_to_depot"].sum(),
        "total_acc_revenue_when_it_crashes":
            metrics["total_acc_revenue_when_it_crashes"].sum(),
        "total_backedup_revenue_when_it_crashes":
            metrics["total_backedup_revenue_when_it_crashes"].sum(),
    })

plot_df = pd.DataFrame(rows).sort_values(["cb", "total_uavs"]).reset_index(drop=True)

# muted academic colors
COLOR_ACC_DEPOT = "#4C78A8"   # blue
COLOR_BACK_DEPOT = "#F58518"  # orange
COLOR_ACC_CRASH = "#54A24B"   # green
COLOR_BACK_CRASH = "#E45756"  # red

for cb_value in sorted(plot_df["cb"].dropna().unique()):
    cb_df = plot_df[plot_df["cb"] == cb_value].copy()

    cb_label = str(cb_value)
    x_labels = cb_df["total_uavs"].astype(int).tolist()
    x = np.arange(len(x_labels))
    width = 0.36

    # -------- Figure 1: returns to depot --------
    acc_depot = cb_df["total_acc_revenue_when_arrives_to_depot"].to_numpy()
    backup_depot = cb_df["total_backedup_revenue_when_arrives_to_depot"].to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.bar(
        x - width / 2,
        acc_depot,
        width,
        label="Accumulated revenue",
        color=COLOR_ACC_DEPOT,
        # edgecolor="black",
        linewidth=0.5
    )
    ax.bar(
        x + width / 2,
        backup_depot,
        width,
        label="Backed-up revenue",
        color=COLOR_BACK_DEPOT,
        # edgecolor="black",
        linewidth=0.5
    )

    ax.set_xlabel("Number of UAVs")
    ax.set_ylabel("Revenue")
    # ax.set_title(f"When returns to depot (cb={cb_value})", pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="best", frameon=True, edgecolor="black", fancybox=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.6)
    fig.savefig(out_dir / f"{cb_label}crash_fig1.png", bbox_inches="tight")
    plt.close(fig)

    # -------- Figure 2: crashes --------
    acc_crash = cb_df["total_acc_revenue_when_it_crashes"].to_numpy()
    backup_crash = cb_df["total_backedup_revenue_when_it_crashes"].to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.bar(
        x - width / 2,
        acc_crash,
        width,
        label="Accumulated revenue",
        color=COLOR_ACC_CRASH,
        # edgecolor="black",
        linewidth=0.5
    )
    ax.bar(
        x + width / 2,
        backup_crash,
        width,
        label="Backed-up revenue",
        color=COLOR_BACK_CRASH,
        # edgecolor="black",
        linewidth=0.5
    )

    ax.set_xlabel("Number of UAVs")
    ax.set_ylabel("Revenue")
    # ax.set_title(f"When it crashes (cb={cb_value})", pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="best", frameon=True, edgecolor="black", fancybox=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.6)
    fig.savefig(out_dir / f"{cb_label}crash_fig2.png", bbox_inches="tight")
    plt.close(fig)

print("Done.")
print(plot_df)