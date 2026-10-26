"""Template Matching (ZNCC) vs Lattice-first vs DINOv3 特徵比對：合成週期 contact hole array benchmark。

兩組實驗：
  cell  — Unit Cell 尺寸 16–128 px × 方形/六角，中等隨機擾動
  axis  — 固定 pitch 48，逐一加重單一擾動（灰階、雜訊、模糊、尺度、旋轉、形變、defect，
          以及對 ZNCC 不利的強 gamma、對比反轉、微粒遮擋）

用法：
  uv run python scripts/run_benchmark.py --quick
  uv run python scripts/run_benchmark.py --seeds 3 --device cuda --methods ncc,lattice,dinov3-l-sat
"""
import argparse
import time
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from periodic_match.lattice import LatticeMatcher
from periodic_match.matchers import DinoMatcher, NCCMatcher
from periodic_match.metrics import (balanced_acc, best_threshold, detect, detection_pr, find_peaks,
                                    instance_anomaly_auroc, scene_scores, summarize)
from periodic_match.synth import Design, Imaging, make_scene

PITCHES = [16, 24, 32, 48, 64, 96, 128]
AXIS_PITCH = 48
AXIS_LEVELS = {
    "photometric": [0, 1, 2, 3],
    "noise": [0.02, 0.06, 0.12, 0.20],
    "blur": [0.5, 1.5, 3.0, 5.0],
    "scale": [1.0, 1.12, 0.83, 1.27],
    "rotation": [0.0, 1.3, 3.1, 4.9],
    "elastic": [0.0, 0.03, 0.08, 0.15],
    "defect": [0.0, 0.02, 0.05, 0.10],
    "gamma": [1.0, 2.0, 3.5, 0.3],
    "invert": [0, 1],
    "occlusion": [0.0, 0.01, 0.03, 0.06],
}
DINO_MODELS = {
    "dinov3-s": dict(model_id="facebook/dinov3-vits16-pretrain-lvd1689m"),
    "dinov3-s-cos": dict(model_id="facebook/dinov3-vits16-pretrain-lvd1689m", mode="cos", ref_context=False),
    "dinov3-l": dict(model_id="facebook/dinov3-vitl16-pretrain-lvd1689m"),
    "dinov3-l-sat": dict(model_id="facebook/dinov3-vitl16-pretrain-sat493m"),
}


def ref_imaging(rng):
    """Reference 來自「另一台機台」：灰階特性略有差異。"""
    return Imaging(gamma=float(np.exp(rng.uniform(-0.18, 0.18))), contrast=rng.uniform(0.85, 1.15),
                   noise=0.02, blur=0.5)


def moderate_imaging(rng):
    return Imaging(scale=rng.uniform(0.8, 1.3), rotation_deg=rng.uniform(-5, 5), elastic=0.03,
                   gamma=float(np.exp(rng.uniform(-0.34, 0.34))), contrast=rng.uniform(0.8, 1.2),
                   brightness=rng.uniform(-0.04, 0.04), gradient=0.1, blur=0.6, noise=0.04, defect_rate=0.02)


def axis_imaging(axis, level, rng):
    base = Imaging(defect_rate=0.01)
    if axis == "photometric":
        m = level / 3
        sign = rng.choice([-1, 1])
        return replace(base, gamma=float(np.exp(sign * m * np.log(1.8))), contrast=1 - 0.4 * m * rng.choice([-1, 1]),
                       brightness=0.08 * m * rng.choice([-1, 1]), gradient=0.25 * m)
    key = {"rotation": "rotation_deg", "defect": "defect_rate"}.get(axis, axis)
    return replace(base, **{key: bool(level) if axis == "invert" else level})


def build_scenes(seeds, quick):
    specs = []
    for seed in range(seeds):
        for lattice in ["square", "hex"]:
            for pitch in PITCHES:
                specs.append(dict(sweep="cell", axis="moderate", level=pitch, lattice=lattice, pitch=pitch, seed=seed))
        for lattice in (["square"] if quick else ["square", "hex"]):
            for axis, levels in AXIS_LEVELS.items():
                for li, level in enumerate(levels):
                    specs.append(dict(sweep="axis", axis=axis, level=level, level_idx=li, lattice=lattice,
                                      pitch=AXIS_PITCH, seed=seed))
    return specs


def realize(spec):
    key = zlib.crc32(f'{spec["sweep"]}|{spec["axis"]}|{spec["level"]}|{spec["lattice"]}|{spec["pitch"]}'.encode())
    rng = np.random.default_rng([spec["seed"], key])
    design = Design(spec["lattice"], spec["pitch"], tuple(rng.uniform(0, 1, 2)))
    imaging = moderate_imaging(rng) if spec["sweep"] == "cell" else axis_imaging(spec["axis"], spec["level"], rng)
    return make_scene(design, imaging, ref_imaging(rng), seed=int(rng.integers(2**31)))


def make_matchers(names, device):
    out = []
    for n in names:
        if n == "ncc":
            out.append(NCCMatcher())
        elif n == "lattice":
            out.append(LatticeMatcher())
        else:
            out.append(DinoMatcher(device=device, name=n, **DINO_MODELS[n]))
    return out


def threshold_analysis(df, raws):
    """每個 (sweep, method)：一個全域閾值套用到所有場景，看準確率與偵測結果；以及各場景最佳閾值的漂移。"""
    rows = []
    for (sweep, method), g in df.groupby(["sweep", "method"]):
        keys = list(g["key"])
        pos = np.concatenate([raws[k]["true"] for k in keys])
        neg = np.concatenate([np.concatenate([raws[k]["wrong"], raws[k]["bg"]]) for k in keys])
        thr, _ = best_threshold(pos, neg)
        per_t = []
        for k in keys:
            r = raws[k]
            n = np.concatenate([r["wrong"], r["bg"]])
            t_k, bacc_k = best_threshold(r["true"], n)
            per_t.append(t_k)
            rows.append(dict(key=k, global_thr=thr, bacc_global=balanced_acc(r["true"], n, thr),
                             bacc_oracle=bacc_k, scene_thr=t_k))
        spread = np.nanmedian(pos) - np.nanmedian(neg)
        df.loc[g.index, "thr_drift"] = np.std(per_t) / (abs(spread) + 1e-9)
    return df.merge(pd.DataFrame(rows), on="key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="ncc,lattice", help="另有 dinov3-s,dinov3-s-cos,dinov3-l,dinov3-l-sat（需 --extra dino）")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="1 seed，axis 實驗只跑方形")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 個場景（測試用）")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        try:   # torch 只在 --extra dino 時安裝；只跑 ncc,lattice 不需要
            import torch
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    seeds = 1 if args.quick else args.seeds
    out = Path(args.out or f"results/{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    specs = build_scenes(seeds, args.quick)[: args.limit]
    matchers = make_matchers(args.methods.split(","), device)
    rows, raws = [], {}
    t0 = time.time()
    for si, spec in enumerate(specs):
        scene = realize(spec)
        for m in matchers:
            t = time.time()
            sm = m.score_map(scene.search, scene.reference, min(scene.design.cell))
            dt = time.time() - t
            raw = scene_scores(sm, scene, np.random.default_rng(si))
            key = f"{si}:{m.name}"
            raw["peaks"] = find_peaks(sm, scene)
            raw["scene"] = scene.__class__(**{**scene.__dict__, "search": None, "reference": None,
                                              "periodic_mask": None})  # 偵測只需 GT
            row = summarize(raw)
            if hasattr(m, "instances"):  # 直接輸出 Instance Grid：不需閾值與 NMS
                inst = m.instances()
                raw["instances"] = inst.centers
                row["auroc_anomaly"] = instance_anomaly_auroc(inst.centers, inst.score, scene)
            raws[key] = raw
            rows.append(dict(key=key, scene=si, method=m.name, seconds=dt, n_max=scene.n_max,
                             **spec, **{f"im_{k}": v for k, v in asdict(scene.imaging).items()}, **row))
        print(f"[{si + 1}/{len(specs)}] {spec['sweep']}/{spec['axis']}={spec['level']} {spec['lattice']} "
              f"pitch={spec['pitch']} ({time.time() - t0:.0f}s)", flush=True)
        if si % 10 == 9 or si == len(specs) - 1:
            pd.DataFrame(rows).to_csv(out / "scenes_partial.csv", index=False)

    df = threshold_analysis(pd.DataFrame(rows), raws)
    det = []
    for k in df["key"]:
        r = raws[k]
        if "instances" in r:
            centers = r["instances"]
        else:
            _, centers = detect(r["peaks"], r["scene"], float(df.loc[df.key == k, "global_thr"].iloc[0]))
        det.append(dict(key=k, **detection_pr(centers, r["scene"])))
    df = df.merge(pd.DataFrame(det), on="key")
    df.to_csv(out / "scenes.csv", index=False)
    print(f"wrote {out / 'scenes.csv'}")


if __name__ == "__main__":
    main()
