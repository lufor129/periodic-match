"""從分數圖與 ground truth 計算鑑別力指標與偵測結果。"""
import numpy as np
from scipy.ndimage import maximum_filter


def auroc(pos, neg):
    """P(pos > neg)，以秩計算（ties 算一半）。"""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # 平均 tie 的秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _window(score, cx, cy, hx, hy):
    h, w = score.shape
    x0, x1 = int(max(0, np.floor(cx - hx))), int(min(w, np.ceil(cx + hx) + 1))
    y0, y1 = int(max(0, np.floor(cy - hy))), int(min(h, np.ceil(cy + hy) + 1))
    return score[y0:y1, x0:x1], x0, y0


def _subpixel(score, x, y):
    """拋物線擬合的次像素峰值位置。"""
    h, w = score.shape

    def fit(a, b, c):
        d = a - 2 * b + c
        return 0.0 if not np.isfinite(d) or d >= 0 else 0.5 * (a - c) / d

    dx = fit(score[y, x - 1], score[y, x], score[y, x + 1]) if 0 < x < w - 1 else 0.0
    dy = fit(score[y - 1, x], score[y, x], score[y + 1, x]) if 0 < y < h - 1 else 0.0
    return x + np.clip(dx, -1, 1), y + np.clip(dy, -1, 1)


def scene_scores(sm, scene, rng, n_bg=1500):
    """每個合法 Instance 的正確 Phase 分數、錯誤 Phase 最高分、定位誤差，與背景分數。"""
    score = sm.score
    cw, ch = scene.cell_px
    cmin = min(cw, ch)
    r_tol = max(1.5, 0.08 * cmin)
    r_phase = 0.25 * cmin
    hx, hy = 0.45 * cw, 0.45 * ch

    true_s, wrong_s, loc_err, phase_ok = [], [], [], []
    for cx, cy in scene.centers:
        win, x0, y0 = _window(score, cx, cy, hx, hy)
        yy, xx = np.mgrid[y0:y0 + win.shape[0], x0:x0 + win.shape[1]]
        dist = np.hypot(xx - cx, yy - cy)
        w = np.where(np.isfinite(win), win, -np.inf)
        near = w[dist <= r_tol]
        far = w[dist > r_phase]
        true_s.append(near.max() if near.size else np.nan)
        wrong_s.append(far.max() if far.size else np.nan)
        iy, ix = np.unravel_index(np.argmax(w), w.shape)
        px, py = _subpixel(score, x0 + ix, y0 + iy)
        e = np.hypot(px - cx, py - cy)
        phase_ok.append(e < r_phase)
        loc_err.append(e)
    true_s, wrong_s = np.array(true_s), np.array(wrong_s)

    # 背景：Instance 框有一半以上落在非 Periodic Region 的位置
    tw, th = 3 * cw, 3 * ch
    H, W = score.shape
    nonper = (~scene.periodic_mask).astype(np.float32)
    integ = np.pad(nonper.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    step = max(2, int(0.5 * cmin))
    ys, xs = np.mgrid[int(th / 2):int(H - th / 2):step, int(tw / 2):int(W - tw / 2):step]
    ys, xs = ys.ravel(), xs.ravel()
    y0, y1 = (ys - th / 2).astype(int), (ys + th / 2).astype(int)
    x0, x1 = (xs - tw / 2).astype(int), (xs + tw / 2).astype(int)
    frac = (integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]) / ((y1 - y0) * (x1 - x0))
    sel = np.flatnonzero(frac >= 0.5)
    if len(sel) > n_bg:
        sel = rng.choice(sel, n_bg, replace=False)
    bg = []
    for k in sel:
        win, *_ = _window(score, xs[k], ys[k], hx, hy)
        bg.append(np.nanmax(win) if np.isfinite(win).any() else np.nan)

    return dict(true=true_s, wrong=wrong_s, bg=np.array(bg), loc_err=np.array(loc_err),
                phase_ok=np.array(phase_ok), anomalous=scene.anomalous)


def summarize(raw):
    """單一場景的閾值無關指標。"""
    t, w, bg, an = raw["true"], raw["wrong"], raw["bg"], raw["anomalous"]
    paired = np.isfinite(t) & np.isfinite(w)
    margin = t[paired] - w[paired]
    sd = np.sqrt(0.5 * (np.nanvar(t) + np.nanvar(w))) + 1e-9
    ok = raw["phase_ok"]
    return dict(
        n_instances=len(t),
        phase_acc=float((margin > 0).mean()) if paired.any() else np.nan,
        phase_dprime=float((np.nanmean(t) - np.nanmean(w)) / sd),
        auroc_bg=auroc(t, bg),
        auroc_wrong=auroc(t, w),
        auroc_anomaly=auroc(t[~an], t[an]) if an.any() and (~an).any() else np.nan,
        loc_err_px=float(np.median(raw["loc_err"][ok])) if ok.any() else np.nan,
    )


def best_threshold(pos, neg):
    """最大化 balanced accuracy 的閾值。"""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    cand = np.unique(np.concatenate([pos, neg]))
    if len(cand) > 2000:
        cand = np.quantile(cand, np.linspace(0, 1, 2000))
    tpr = (pos[None] >= cand[:, None]).mean(1)
    tnr = (neg[None] < cand[:, None]).mean(1)
    k = np.argmax(tpr + tnr)
    return float(cand[k]), float(0.5 * (tpr[k] + tnr[k]))


def balanced_acc(pos, neg, thr):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    return 0.5 * ((pos >= thr).mean() + (neg < thr).mean())


def find_peaks(sm, scene):
    """分數圖的局部極大（依分數遞減）：x, y, score, scale。"""
    score = np.where(np.isfinite(sm.score), sm.score, -np.inf)
    size = max(3, int(0.5 * min(scene.cell_px)))
    peaks = (score == maximum_filter(score, size=size)) & np.isfinite(score)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-score[ys, xs])
    ys, xs = ys[order], xs[order]
    return dict(x=xs, y=ys, score=score[ys, xs], scale=sm.scale[ys, xs])


def detect(peaks, scene, thr, max_overlap=0.05):
    """傳統流程：局部極大 → 閾值 → 依分數貪婪挑選不重疊的框。"""
    sel = peaks["score"] >= thr
    xs, ys, s = peaks["x"][sel], peaks["y"][sel], peaks["scale"][sel]
    cw, ch = scene.design.cell  # Reference 像素；框大小用每個峰的最佳尺度
    bw, bh = 3 * cw * s, 3 * ch * s
    boxes = np.stack([xs - bw / 2, ys - bh / 2, xs + bw / 2, ys + bh / 2], 1)
    keep = []
    for k, b in enumerate(boxes):
        if keep:
            kb = boxes[keep]
            iw = np.clip(np.minimum(kb[:, 2], b[2]) - np.maximum(kb[:, 0], b[0]), 0, None)
            ih = np.clip(np.minimum(kb[:, 3], b[3]) - np.maximum(kb[:, 1], b[1]), 0, None)
            if (iw * ih / ((b[2] - b[0]) * (b[3] - b[1]))).max() > max_overlap:
                continue
        keep.append(k)
    return boxes[keep], np.stack([xs[keep], ys[keep]], 1).astype(float)


def match_instances(pred_centers, scene):
    """預測中心一對一配對到合法 Instance（9 種切法皆可，容許 0.25 Cell）；回傳 (pred_idx, gt_idx)。"""
    if len(pred_centers) == 0 or len(scene.centers) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    tol = 0.25 * min(scene.cell_px)
    d = np.hypot(pred_centers[:, None, 0] - scene.centers[None, :, 0],
                 pred_centers[:, None, 1] - scene.centers[None, :, 1])
    used, pi, gi = set(), [], []
    for i in np.argsort(d.min(1)):
        j = int(np.argmin(d[i]))
        if d[i, j] < tol and j not in used:
            used.add(j)
            pi.append(i)
            gi.append(j)
    return np.array(pi, int), np.array(gi, int)


def detection_pr(pred_centers, scene):
    """recall 以最多 Instance 的切法為分母。"""
    if len(pred_centers) == 0:
        return dict(n_pred=0, tp=0, precision=np.nan, recall=0.0 if scene.n_max else np.nan)
    tp = len(match_instances(pred_centers, scene)[0])
    return dict(n_pred=len(pred_centers), tp=tp, precision=tp / len(pred_centers),
                recall=min(1.0, tp / scene.n_max) if scene.n_max else np.nan)


def instance_anomaly_auroc(pred_centers, pred_scores, scene):
    """偵測到的 Instance 中，正常 vs Anomalous 的分數 AUROC。"""
    pi, gi = match_instances(pred_centers, scene)
    an = scene.anomalous[gi]
    return auroc(pred_scores[pi][~an], pred_scores[pi][an]) if an.any() and (~an).any() else np.nan
