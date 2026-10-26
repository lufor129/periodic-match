"""讀 scenes.csv，輸出摘要表（summary.md）與圖（cell_sweep.png、axis_sweep.png）。

用法：uv run python scripts/analyze.py results/quick
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_benchmark import AXIS_LEVELS

# 方法 → 固定顏色（dataviz 預設 categorical palette 的前 5 個位置）與 marker
STYLE = {
    "ncc": ("#2a78d6", "o", "ZNCC (Template Matching)"),
    "lattice": ("#008300", "P", "Lattice-first (FFT + GPA)"),
    "dinov3-s": ("#eb6834", "s", "DINOv3-S (feature ZNCC)"),
    "dinov3-s-cos": ("#1baf7a", "v", "DINOv3-S (naive cosine)"),
    "dinov3-l": ("#eda100", "D", "DINOv3-L (feature ZNCC)"),
    "dinov3-l-sat": ("#e87ba4", "^", "DINOv3-L-SAT (feature ZNCC)"),
}
METRICS = [
    ("phase_dprime", "Phase 鑑別 d′（越高越好）", True),
    ("auroc_bg", "Instance vs 背景 AUROC", False),
    ("bacc_global", "固定閾值 balanced acc.", False),
    ("recall", "偵測 recall", False),
    ("auroc_anomaly", "Anomalous Instance AUROC", False),
    ("loc_err_px", "定位誤差 px（越低越好）", True),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": ["Heiti TC", "Arial Unicode MS", "DejaVu Sans"],
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def _line(ax, x, y, method):
    color, marker, _ = STYLE[method]
    ax.plot(x, y, color=color, marker=marker, lw=2, ms=6, markeredgecolor="#fcfcfb", markeredgewidth=1)


def _legend(fig, methods):
    handles = [plt.Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], lw=2, ms=6, label=STYLE[m][2])
               for m in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(methods), frameon=False, fontsize=9)


def plot_cell(df, out):
    d = df[df.sweep == "cell"]
    methods = [m for m in STYLE if m in d.method.unique()]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (col, title, log) in zip(axes.flat, METRICS):
        for m in methods:
            g = d[d.method == m].groupby("pitch")[col].mean()
            _line(ax, g.index, g.values, m)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(d.pitch.unique()), [str(p) for p in sorted(d.pitch.unique())])
        ax.minorticks_off()
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("Unit Cell pitch (px)", fontsize=9)
    _legend(fig, methods)
    fig.suptitle("Unit Cell 尺寸掃描（中等隨機擾動，方形＋六角平均）", y=0.995, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_axis(df, out):
    d = df[df.sweep == "axis"]
    methods = [m for m in STYLE if m in d.method.unique()]
    rows = [m for m in METRICS if m[0] in ("phase_dprime", "auroc_bg", "bacc_global", "recall")]
    axes_names = list(AXIS_LEVELS)
    fig, axes = plt.subplots(len(rows), len(axes_names), figsize=(2.3 * len(axes_names), 2.3 * len(rows)),
                             sharey="row")
    for c, axis in enumerate(axes_names):
        for r, (col, title, log) in enumerate(rows):
            ax = axes[r, c]
            for m in methods:
                g = d[(d.method == m) & (d.axis == axis)].groupby("level_idx")[col].mean()
                _line(ax, g.index, g.values, m)
            ax.set_xticks(range(len(AXIS_LEVELS[axis])), [f"{v:g}" for v in AXIS_LEVELS[axis]], fontsize=8)
            if log:
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(axis, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(title, fontsize=8)
    _legend(fig, methods)
    fig.suptitle(f"單一擾動加重（pitch {d.pitch.iloc[0]}）", y=0.995, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def summary_table(df):
    cols = ["phase_acc", "phase_dprime", "auroc_bg", "auroc_wrong", "auroc_anomaly", "loc_err_px",
            "bacc_global", "thr_drift", "precision", "recall", "seconds"]
    t = df.groupby(["sweep", "method"])[cols].mean().round(3)
    return t


def main():
    root = Path(sys.argv[1])
    df = pd.read_csv(root / "scenes.csv")
    plot_cell(df, root / "cell_sweep.png")
    plot_axis(df, root / "axis_sweep.png")
    t = summary_table(df)
    (root / "summary.md").write_text(t.to_markdown())
    print(t.to_string())


if __name__ == "__main__":
    main()
