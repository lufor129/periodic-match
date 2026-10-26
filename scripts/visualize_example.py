"""單一場景的視覺比較：Search Image（GT 框）、Reference、各方法分數圖（同一區域），
以及 Lattice-first 直接輸出的 Instance Grid。

用法：uv run python scripts/visualize_example.py results/quick/example.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import INK, STYLE
from periodic_match.synth import Design, Imaging, make_scene
from run_benchmark import make_matchers


def main():
    out = sys.argv[1]
    methods = sys.argv[2].split(",") if len(sys.argv) > 2 else ["ncc", "lattice", "dinov3-l-sat"]
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    design = Design("square", 48, (0.3, 0.6))
    imaging = Imaging(scale=1.12, rotation_deg=3.1, elastic=0.05, gamma=1.4, contrast=0.85, gradient=0.15,
                      noise=0.06, defect_rate=0.03)
    scene = make_scene(design, imaging, Imaging(gamma=0.85, noise=0.02), seed=7)
    y0, y1, x0, x1 = 300, 900, 300, 900

    matchers = make_matchers(methods, device)
    n_grid = sum(hasattr(m, "instances") for m in matchers)
    n = 2 + len(methods) + n_grid
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.6))
    ax = axes[0]
    ax.imshow(scene.search[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
    best = np.bincount(scene.tiling).argmax()
    for (bx0, by0, bx1, by1), an, t in zip(scene.boxes, scene.anomalous, scene.tiling):
        if t == best:
            ax.add_patch(plt.Rectangle((bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0, fill=False, lw=1.2,
                                       ec="#e34948" if an else "#1baf7a"))
    ax.set_xlim(0, x1 - x0)
    ax.set_ylim(y1 - y0, 0)
    ax.set_title("Search Image（綠：Instance／紅：Anomalous）", fontsize=9, color=INK)
    axes[1].imshow(scene.reference, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reference（3×3 Unit Cell，另一台機台）", fontsize=9, color=INK)

    gt = scene.centers
    sel = (gt[:, 0] > x0) & (gt[:, 0] < x1) & (gt[:, 1] > y0) & (gt[:, 1] < y1)
    panels = iter(axes[2:])
    for m in matchers:
        ax = next(panels)
        sm = m.score_map(scene.search, scene.reference, min(design.cell))
        crop = sm.score[y0:y1, x0:x1]
        ax.imshow(crop, cmap="viridis")
        ax.scatter(gt[sel, 0] - x0, gt[sel, 1] - y0, s=6, c="#fcfcfb", marker="+", lw=0.6)
        ax.set_title(f"{STYLE[m.name][2]}\n（白十字＝所有正確 Phase 位置）", fontsize=9, color=INK)
        if hasattr(m, "instances"):
            ax = next(panels)
            inst = m.instances()
            ax.imshow(scene.search[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
            for (bx0, by0, bx1, by1), an in zip(inst.boxes, inst.anomalous):
                ax.add_patch(plt.Rectangle((bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0, fill=False, lw=1.2,
                                           ec="#e34948" if an else "#1baf7a"))
            ax.set_xlim(0, x1 - x0)
            ax.set_ylim(y1 - y0, 0)
            ax.set_title("Lattice-first 輸出的 Instance Grid\n（綠：Instance／紅：標記為 Anomalous）", fontsize=9, color=INK)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
