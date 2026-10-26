"""在真實影像上跑 Lattice-first：每張 Search Image 輸出 Instance 表、疊圖與診斷圖。

Reference 必須剛好裁成 REF_CELLS（預設 3×3）個 Unit Cell，裁切邊界即定義 Phase。

用法：
  uv run python scripts/run_on_image.py data/ref.tif data/search_*.tif --out out/run1
  uv run python scripts/run_on_image.py data/ref.tif data/a.tif --out out/a --amp-frac 0.4 --z-thr 4

每張 Search Image 輸出到 <out>/<檔名>/：
  instances.csv    cx, cy, x0, y0, x1, y1, score, anomalous, region（像素座標）
  overlay.png      Instance Grid（綠：Instance／紅：Anomalous）
  diagnostics.png  Periodic Region、自相似度、Instance 分數分佈（圖內文字用英文，公司 Linux 機器常缺中文字型）
  summary.json     估計的尺度、角度、極性、諧波、Instance 數、耗時
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from periodic_match.lattice import REF_CELLS, LatticeMatcher


def load_gray(path):
    """任意 bit depth 的灰階／彩色影像 → float32 [0, 1]。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [0.1, 99.9])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def draw_boxes(ax, img, inst):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    for (x0, y0, x1, y1), an in zip(inst.boxes, inst.anomalous):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.8,
                                   ec="#e34948" if an else "#1baf7a"))
    ax.set_xticks([])
    ax.set_yticks([])


def run_one(matcher, search, ref, out):
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    r = matcher.analyze(search, ref)
    seconds = time.perf_counter() - t
    inst = r["instances"]

    pd.DataFrame({
        "cx": inst.centers[:, 0], "cy": inst.centers[:, 1],
        "x0": inst.boxes[:, 0], "y0": inst.boxes[:, 1], "x1": inst.boxes[:, 2], "y1": inst.boxes[:, 3],
        "score": inst.score, "anomalous": inst.anomalous, "region": inst.region,
    }).to_csv(out / "instances.csv", index=False)

    h, w = search.shape
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    draw_boxes(ax, search, inst)
    ax.set_title(f"{len(inst.score)} Instances, {int(inst.anomalous.sum())} Anomalous", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "overlay.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(search, cmap="gray", vmin=0, vmax=1)
    axes[0].imshow(np.ma.masked_where(r["periodic"], r["periodic"]), cmap="autumn", alpha=0.45)
    axes[0].set_title("Periodic Region (red = non-periodic)", fontsize=10)
    im = axes[1].imshow(r["amp"], cmap="viridis")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title("lattice self-similarity", fontsize=10)
    sc = inst.score[np.isfinite(inst.score)]
    axes[2].hist(sc, bins=40, color="#2a78d6")
    if inst.anomalous.any():
        axes[2].axvline(np.nanmax(inst.score[inst.anomalous]), color="#e34948", ls="--", lw=1)
    axes[2].set_title("Instance score (min of 9 cell ZNCC)", fontsize=10)
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=110)
    plt.close(fig)

    summary = dict(
        search_shape=[h, w], ref_shape=list(ref.shape),
        cell_px_ref=[ref.shape[1] / REF_CELLS[0], ref.shape[0] / REF_CELLS[1]],
        scale=r["scale"], angle_deg=r["angle_deg"], polarity=r["polarity"],
        harmonics=[list(map(int, k)) for k in r["harmonics"]],
        periodic_fraction=float(r["periodic"].mean()),
        n_instances=int(len(inst.score)), n_anomalous=int(inst.anomalous.sum()),
        score_median=float(np.nanmedian(inst.score)) if len(inst.score) else None,
        seconds=round(seconds, 2),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("search", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--sigma-cells", type=float, default=1.0)
    p.add_argument("--amp-frac", type=float, default=0.5)
    p.add_argument("--min-region-cells", type=float, default=8.0)
    p.add_argument("--z-thr", type=float, default=3.5)
    a = p.parse_args()

    matcher = LatticeMatcher(a.sigma_cells, a.amp_frac, a.min_region_cells, a.z_thr)
    ref = load_gray(a.reference)
    rows = []
    for s in a.search:
        summary = run_one(matcher, load_gray(s), ref, Path(a.out) / Path(s).stem)
        rows.append(dict(search=Path(s).name, **{k: v for k, v in summary.items() if not isinstance(v, list)}))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(Path(a.out) / "runs.csv", index=False)


if __name__ == "__main__":
    main()
