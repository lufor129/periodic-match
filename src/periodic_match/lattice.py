"""Lattice-first：先從 Search Image 估出晶格，再用 Reference 決定 Phase，直接產生 Instance Grid。

  1. 幾何：Reference 的 Fourier 係數 C_h 預測 Search 功率譜上的峰值位置，搜尋 (尺度, 角度)
     使峰值能量最大。功率譜不含相位，對灰階、對比反轉、平移都不敏感。
  2. 晶格節點：GPA（geometric phase analysis）。兩個基本諧波的局部相位與 Reference 相位比對，
     以 Newton 迭代求出每個節點的位置，節點因此跟隨局部漂移與形變。
  3. Periodic Region：影像與「自己平移一個晶格向量」的局部 ZNCC（兩個方向取小）高於門檻；
     對單調灰階變化與對比反轉不變。小於數個 Cell 的空洞（微粒等）補回。
  4. Instance Grid：每個 Periodic Region 各自挑 Instance 數最多的切法，不需要分數閾值與 NMS。
  5. Anomalous Instance：每個 Unit Cell 單獨做 ZNCC，Instance 分數取 9 個 Cell 的最小值。

座標慣例與 synth 相同：q = B x + δ(x) 是以 Unit Cell 為單位的理想座標，q 為整數處是 Instance 中心候選。
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage, optimize

from .matchers import ScoreMap, place

REF_CELLS = (3, 3)
HARMONICS = [(m, n) for m in range(-3, 4) for n in range(-3, 4) if (m, n) > (0, 0)]  # 半平面


@dataclass
class Instances:
    centers: np.ndarray    # (N,2) x,y
    boxes: np.ndarray      # (N,4)
    score: np.ndarray      # (N,) 9 個 Cell ZNCC 的最小值
    anomalous: np.ndarray  # (N,) robust z-score < -z_thr
    region: np.ndarray     # (N,) Periodic Region 編號


def ref_coefficients(ref, cell):
    """C_h：Reference 以中心為原點、Unit Cell 為單位的 Fourier 係數（Hann 窗）；以及窗內 rms。"""
    h, w = ref.shape
    y, x = np.mgrid[0:h, 0:w]
    qx, qy = (x - (w - 1) / 2) / cell[0], (y - (h - 1) / 2) / cell[1]
    win = np.outer(np.hanning(h), np.hanning(w))
    r = ref - (ref * win).sum() / win.sum()
    C = {k: (r * win * np.exp(-2j * np.pi * (k[0] * qx + k[1] * qy))).sum() / win.sum() for k in HARMONICS}
    return C, float(np.sqrt((win * r ** 2).sum() / win.sum()))


def recip(s, th, cell):
    """q = B x 的線性部分：x = s R(θ) u（u 為 Reference 像素）→ q = diag(1/cell) R(-θ) x / s。"""
    c, si = np.cos(th), np.sin(th)
    return np.diag([1 / cell[0], 1 / cell[1]]) @ np.array([[c, si], [-si, c]]) / s


def geometry_objective(hp, C, cell):
    """對數功率譜 P 與目標函數 score(s, θ)：以 |C_h| 加權、取樣 P 在預測峰值位置的值。"""
    h, w = hp.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.abs(np.fft.fftshift(np.fft.fft2(hp * win))) ** 2
    P = cv2.GaussianBlur(np.log1p(F / np.median(F)).astype(np.float32), (0, 0), 1.0)
    amp = np.array([abs(C[k]) for k in HARMONICS])
    keep = amp > 0.1 * amp.max()
    hk = np.array(HARMONICS, float)[keep]
    wt = amp[keep] / amp[keep].sum()
    kx, ky = hk[:, 0] / cell[0], hk[:, 1] / cell[1]

    def score(s, th):
        s, th = np.asarray(s)[..., None], np.asarray(th)[..., None]
        gx = (np.cos(th) * kx - np.sin(th) * ky) / s
        gy = (np.sin(th) * kx + np.cos(th) * ky) / s
        v = ndimage.map_coordinates(P, [h // 2 + gy * h, w // 2 + gx * w], order=1, mode="constant")
        return (v * wt).sum(-1)

    return P, score


def fit_geometry(hp, C, cell, s_range=(0.7, 1.4), th_range_deg=7.0):
    """在對數功率譜上，以 |C_h| 加權的預測峰值能量最大化求 (s, θ)：格點搜尋後 Nelder-Mead 精修。"""
    _, score = geometry_objective(hp, C, cell)
    S, T = np.meshgrid(np.arange(s_range[0], s_range[1], 0.004), np.deg2rad(np.arange(-th_range_deg, th_range_deg, 0.2)))
    k = np.argmax(score(S, T))
    res = optimize.minimize(lambda p: -score(p[0], p[1]), [S.flat[k], T.flat[k]], method="Nelder-Mead",
                            options=dict(xatol=1e-5, fatol=1e-6))
    return float(res.x[0]), float(res.x[1])


def pick_harmonics(C):
    """兩個最低階、不共線且振幅夠大的諧波，決定節點晶格 Λ = {q : h_k·q ∈ Z}。"""
    amp = {k: abs(v) for k, v in C.items()}
    cand = sorted([k for k in HARMONICS if amp[k] >= 0.25 * max(amp.values())],
                  key=lambda k: (k[0] ** 2 + k[1] ** 2, -amp[k]))
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            if a[0] * b[1] - a[1] * b[0] != 0:
                return a, b
    raise ValueError("Reference 沒有兩個不共線的諧波")


def _gpa(img, g, sigma):
    """H(x) = G_σ * (I(x) e^{-2πi g·x})。"""
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]
    ph = 2 * np.pi * (g[0] * x + g[1] * y)
    re = cv2.GaussianBlur((img * np.cos(ph)).astype(np.float32), (0, 0), sigma)
    im = cv2.GaussianBlur((-img * np.sin(ph)).astype(np.float32), (0, 0), sigma)
    return re, im


def _self_similarity(img, shifts, sigma):
    """各平移向量 a 下 I(x - a/2) 與 I(x + a/2) 的局部（高斯窗）ZNCC，取最小值。"""
    out = None
    for a in shifts:
        def sh(d):
            m = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
            return cv2.warpAffine(img, m, img.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REFLECT_101)
        u, v = sh(-a / 2), sh(a / 2)
        blur = lambda z: cv2.GaussianBlur(z, (0, 0), sigma)
        mu, mv = blur(u), blur(v)
        cov = blur(u * v) - mu * mv
        var = (blur(u * u) - mu ** 2) * (blur(v * v) - mv ** 2)
        r = cov / np.sqrt(np.maximum(var, 1e-12))
        out = r if out is None else np.minimum(out, r)
    return out


def _sample(a, X, Y):
    return ndimage.map_coordinates(a, [Y, X], order=1, mode="nearest")


def _warp_template(img, s, th, out_wh):
    """以中心對中心、x' = s R(θ) x 把 img 映成 out_wh 大小的模板。"""
    h, w = img.shape
    A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = np.array([(out_wh[0] - 1) / 2, (out_wh[1] - 1) / 2]) - A @ np.array([(w - 1) / 2, (h - 1) / 2])
    return cv2.warpAffine(img, np.hstack([A, t[:, None]]).astype(np.float32), out_wh, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def _zncc_map(search, tmpl):
    th, tw = tmpl.shape
    r = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
    m = place(r, 1.0, ((th - 1) / 2, (tw - 1) / 2), search.shape)
    m[m < -1.01] = np.nan
    return m


def _local_max(m, X, Y, r):
    """每個點 (X,Y) 在 ±r 像素窗內的最大值與其次像素位置。"""
    h, w = m.shape
    mm = np.where(np.isfinite(m), m, -np.inf)
    xi = np.clip(np.round(X), 0, w - 1).astype(int)
    yi = np.clip(np.round(Y), 0, h - 1).astype(int)
    best = np.full(X.shape, -np.inf)
    bx, by = xi.copy(), yi.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x, y = np.clip(xi + dx, 0, w - 1), np.clip(yi + dy, 0, h - 1)
            v = mm[y, x]
            better = v > best
            best[better], bx[better], by[better] = v[better], x[better], y[better]

    def fit(a, b, c):
        with np.errstate(invalid="ignore"):
            d = a - 2 * b + c
            ok = np.isfinite(d) & (d < 0)
            return np.where(ok, 0.5 * (a - c) / np.where(ok, d, -1), 0.0)

    xl, xr = np.clip(bx - 1, 0, w - 1), np.clip(bx + 1, 0, w - 1)
    yu, yd = np.clip(by - 1, 0, h - 1), np.clip(by + 1, 0, h - 1)
    sx = np.clip(fit(mm[by, xl], best, mm[by, xr]), -1, 1)
    sy = np.clip(fit(mm[yu, bx], best, mm[yd, bx]), -1, 1)
    return best, bx + sx, by + sy


def _normalized_smooth(values, weight, sigma):
    """在節點索引格上做 normalized convolution，用好節點的值補壞節點。"""
    num = ndimage.gaussian_filter(np.where(weight, values, 0.0), sigma, mode="constant")
    den = ndimage.gaussian_filter(weight.astype(float), sigma, mode="constant")
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0), den > 1e-3


# ---------------------------------------------------------------- 各步驟（analyze 依序呼叫；notebook 可逐步呼叫）

HYPOTHESES = [(f1, f2, sign) for f1, f2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)] for sign in (1.0, -1.0)]


@dataclass
class Frame:
    """估出的晶格幾何。q = B x + δ(x)；節點晶格索引 n = Hm q；g_k = G[k]（cycles/px）。"""
    cell: tuple          # Reference 上的 Unit Cell (w, h)，像素
    s: float
    th: float            # 弧度
    hs: tuple            # 用來定位節點的兩個諧波
    B: np.ndarray
    Binv: np.ndarray
    Hm: np.ndarray
    Hinv: np.ndarray
    G: np.ndarray
    Ginv: np.ndarray

    @property
    def cell_s(self):
        """Search Image 上 Unit Cell 的短邊（像素）。"""
        return self.s * min(self.cell)


def highpass(search, cell):
    img = search.astype(np.float32)
    return img - cv2.GaussianBlur(img, (0, 0), 3 * max(cell))


def make_frame(cell, s, th, C):
    B = recip(s, th, cell)
    hs = pick_harmonics(C)
    Hm = np.array(hs, float)
    G = Hm @ B
    return Frame(cell, s, th, hs, B, np.linalg.inv(B), Hm, np.linalg.inv(Hm), G, np.linalg.inv(G))


def periodic_regions(hp, fr, sigma, amp_frac=0.5, min_region_cells=8.0):
    """自相似度 → Periodic Region（含補洞）、區域編號、檢查框用的內縮版本。"""
    amp = _self_similarity(hp, [fr.Binv[:, 0], fr.Binv[:, 1]], sigma)
    periodic = amp > amp_frac * np.percentile(amp, 90)
    holes, n = ndimage.label(~periodic)
    sizes = ndimage.sum(np.ones_like(amp), holes, np.arange(1, n + 1))
    periodic |= np.isin(holes, 1 + np.flatnonzero(sizes < min_region_cells * fr.s ** 2 * np.prod(fr.cell)))
    regions, _ = ndimage.label(periodic)
    # 自相似度在邊界外溢約 0.3–0.5 個 Cell：檢查框是否完整落在區域內時用內縮的版本
    k = max(1, int(round(0.3 * fr.cell_s)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    inner = np.where(cv2.erode(periodic.astype(np.uint8), kernel) > 0, regions, 0)
    return amp, periodic, regions, inner


def node_index_grid(fr, shape):
    """涵蓋整張圖的節點索引 (N1, N2) 與對應理想座標 Q = Hm⁻¹ n。"""
    H_img, W_img = shape
    corners = np.array([[0, 0], [W_img, 0], [0, H_img], [W_img, H_img]], float)
    nc = (fr.Hm @ fr.B @ corners.T).T
    n1 = np.arange(np.floor(nc[:, 0].min()) - 2, np.ceil(nc[:, 0].max()) + 3)
    n2 = np.arange(np.floor(nc[:, 1].min()) - 2, np.ceil(nc[:, 1].max()) + 3)
    N1, N2 = np.meshgrid(n1, n2, indexing="ij")
    return N1, N2, np.einsum("ij,jab->iab", fr.Hinv, np.stack([N1, N2]))


def total_phase(gpa, fr, X, Y, phi):
    """2π h_k·q(x) 的殘差（wrap 到 ±π）；節點處為 0。"""
    f = []
    for k, (re, im) in enumerate(gpa):
        z = (_sample(re, X, Y) + 1j * _sample(im, X, Y)) * np.exp(-1j * phi[k])
        f.append(np.angle(z * np.exp(2j * np.pi * (fr.G[k, 0] * X + fr.G[k, 1] * Y))))
    return np.stack(f)


def newton_nodes(gpa, fr, X, Y, phi, iters=8, history=None):
    """Newton 迭代把每個預測點推到最近的「相位為 0」處；history 為 list 時記錄每步位置。"""
    for _ in range(iters):
        d = -np.einsum("ij,jab->iab", fr.Ginv, total_phase(gpa, fr, X, Y, phi)) / (2 * np.pi)
        norm = np.hypot(d[0], d[1])
        d *= np.minimum(1.0, 0.3 * fr.cell_s / (norm + 1e-9))
        X, Y = X + d[0], Y + d[1]
        if history is not None:
            history.append((X, Y))
    return X, Y


def global_offset(gpa, fr, periodic, phi):
    """全域平移 δ0：h_k·δ0 = arg(H_k e^{-iφ_k}) 在 Periodic Region 上的圓形平均。"""
    yy, xx = np.nonzero(periodic)
    t = []
    for k, (re, im) in enumerate(gpa):
        z = (re + 1j * im)[periodic] * np.exp(-1j * phi[k])
        z = z * np.exp(2j * np.pi * (fr.G[k, 0] * xx + fr.G[k, 1] * yy))
        t.append(np.angle(z.sum()) / (2 * np.pi))
    return fr.Hinv @ np.array(t)


def solve_nodes(gpa, fr, periodic, phi, N1, N2, Q):
    """預測 → Newton → 由中心往外重新擬合 affine（半徑 6、15、全部）→ 壞節點用鄰近漂移補。"""
    H_img, W_img = periodic.shape
    d0 = global_offset(gpa, fr, periodic, phi)
    X, Y = np.einsum("ij,jab->iab", fr.Binv, Q - d0[:, None, None])
    X0, Y0 = X, Y
    # affine：x = coef^T [n1, n2, 1]；擬合失敗時沿用預測
    coef = np.vstack([(fr.Binv @ fr.Hinv).T, -(fr.Binv @ d0)])

    def solved_good(X, Y):
        Xs, Ys = newton_nodes(gpa, fr, X, Y, phi)
        inside = (Xs >= 0) & (Xs <= W_img - 1) & (Ys >= 0) & (Ys <= H_img - 1)
        good = inside & periodic[np.clip(np.round(Ys), 0, H_img - 1).astype(int),
                                 np.clip(np.round(Xs), 0, W_img - 1).astype(int)]
        return Xs, Ys, good

    Xs, Ys, good = solved_good(X, Y)
    dist = np.where(good, (Xs - W_img / 2) ** 2 + (Ys - H_img / 2) ** 2, np.inf)
    ci, cj = np.unravel_index(np.argmin(dist), X.shape)
    rr = np.hypot(N1 - N1[ci, cj], N2 - N2[ci, cj])
    for radius in (6, 15, np.inf):
        sel = good & (rr <= radius)
        if sel.sum() < 6:
            break
        A = np.stack([N1[sel], N2[sel], np.ones(sel.sum())], 1)
        coef, *_ = np.linalg.lstsq(A, np.stack([Xs[sel], Ys[sel]], 1), rcond=None)
        X = coef[0, 0] * N1 + coef[1, 0] * N2 + coef[2, 0]
        Y = coef[0, 1] * N1 + coef[1, 1] * N2 + coef[2, 1]
        Xs, Ys, good = solved_good(X, Y)
    # 好節點用 GPA 解；其餘（微粒、邊界）用鄰近好節點的漂移補
    dx, _ = _normalized_smooth(Xs - X, good, 2.0)
    dy, _ = _normalized_smooth(Ys - Y, good, 2.0)
    Xf, Yf = np.where(good, Xs, X + dx), np.where(good, Ys, Y + dy)
    return dict(d0=d0, X0=X0, Y0=Y0, Xa=X, Ya=Y, Xs=Xs, Ys=Ys, good=good, Xf=Xf, Yf=Yf, coef=coef)


def templates(img, ref, fr):
    """估計幾何下的 Cell 模板（1.5 Cell 見方）與 Instance 模板的 ZNCC 圖。"""
    s, th, cell = fr.s, fr.th, fr.cell
    big = np.tile(ref, (3, 3))
    cell_tmpl = _warp_template(big, s, th, (int(round(1.5 * cell[0] * s)), int(round(1.5 * cell[1] * s))))
    inst_tmpl = _warp_template(big, s, th, (int(round(ref.shape[1] * s)), int(round(ref.shape[0] * s))))
    return cell_tmpl, inst_tmpl, _zncc_map(img, cell_tmpl), _zncc_map(img, inst_tmpl)


def score_hypotheses(gpa, fr, C, periodic, cell_map, N1, N2, Q):
    """諧波符號（4 種）× 模板極性（2 種）：每種假設的節點與 Cell ZNCC 中位數。

    非線性灰階（強 gamma）可能讓個別諧波係數變號而影像本身不反轉；對比反轉則讓模板變號。
    """
    r_ref = max(1, int(round(0.1 * fr.cell_s)))
    out, nodes = [], {}
    for f1, f2, sign in HYPOTHESES:
        if (f1, f2) not in nodes:
            phi = [np.angle(f * C[k]) for f, k in zip((f1, f2), fr.hs)]
            nodes[(f1, f2)] = solve_nodes(gpa, fr, periodic, phi, N1, N2, Q)
        nd = nodes[(f1, f2)]
        cs, _, _ = _local_max(sign * cell_map, nd["Xf"], nd["Yf"], r_ref)
        good = nd["good"] & np.isfinite(cs)
        med = np.median(cs[good]) if good.any() else -np.inf
        out.append(dict(flips=(f1, f2), sign=sign, median=med, nodes=nd, cell_score=cs))
    return out


def instance_grid(fr, nd, cs, regions, inner, N1, N2):
    """候選 Instance（9 個 Cell 同區域、框完整在內縮區域內）與每區域的最佳切法。"""
    H_img, W_img = regions.shape
    Xf, Yf, coef = nd["Xf"], nd["Yf"], nd["coef"]
    inside = (Xf >= 0) & (Xf <= W_img - 1) & (Yf >= 0) & (Yf <= H_img - 1)
    node_region = np.where(inside, regions[np.clip(np.round(Yf), 0, H_img - 1).astype(int),
                                           np.clip(np.round(Xf), 0, W_img - 1).astype(int)], 0)
    M = coef[:2].T @ fr.Hm                  # dx/dq（像素 / Unit Cell）
    offs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    dn = [tuple((fr.Hm @ np.array(o, float)).round().astype(int)) for o in offs]
    g = np.linspace(-1.5, 1.5, 5)
    box_pts = np.array([(a, b) for a in g for b in g]) @ M.T
    n1n, n2n = Xf.shape
    cand = []
    for a in range(n1n):
        for b in range(n2n):
            rg = node_region[a, b]
            if rg == 0:
                continue
            idx = [(a + d1, b + d2) for d1, d2 in dn]
            if any(not (0 <= i < n1n and 0 <= j < n2n) or node_region[i, j] != rg for i, j in idx):
                continue
            px, py = Xf[a, b] + box_pts[:, 0], Yf[a, b] + box_pts[:, 1]
            if (px < 0).any() or (py < 0).any() or (px > W_img - 1).any() or (py > H_img - 1).any():
                continue
            if (inner[np.round(py).astype(int), np.round(px).astype(int)] != rg).any():
                continue
            # 影像邊緣的 Cell 放不下 Cell 模板（無分數）：只取有分數的 Cell
            vals = [cs[i, j] for i, j in idx if np.isfinite(cs[i, j])]
            cand.append((a, b, rg, min(vals) if vals else np.nan))
    if not cand:
        e = np.zeros(0, int)
        return dict(a=e, b=e, region=e, score=np.zeros(0), cls=e, keep=np.zeros(0, bool), M=M)
    a, b, rg, sc = map(np.array, zip(*cand))
    q = np.einsum("ij,jk->ik", fr.Hinv, np.stack([N1[a, b], N2[a, b]]))
    key = np.round(2 * np.mod(q, 3)).astype(int)    # 切法 = q mod 3（六角另含半格平移）
    cls = key[0] * 6 + key[1]
    keep = np.zeros(len(a), bool)
    for r in np.unique(rg):                          # 每個 Periodic Region 各取 Instance 最多的切法
        m = rg == r
        keep[m] = cls[m] == np.bincount(cls[m]).argmax()
    return dict(a=a, b=b, region=rg, score=sc, cls=cls, keep=keep, M=M)


def finalize(grid, nd, fr, inst_map, sign, z_thr):
    """保留的 Instance：中心以 Instance ZNCC 微調、框、Anomalous 標記（robust z-score）。"""
    keep = grid["keep"]
    if not keep.any():
        e = np.zeros((0, 2))
        return Instances(e, np.zeros((0, 4)), np.zeros(0), np.zeros(0, bool), np.zeros(0, int))
    a, b, sc, rg = grid["a"][keep], grid["b"][keep], grid["score"][keep], grid["region"][keep]
    _, ix, iy = _local_max(sign * inst_map, nd["Xf"][a, b], nd["Yf"][a, b], max(1, int(round(0.1 * fr.cell_s))))
    half = np.abs(np.array([[-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [1.5, 1.5]]) @ grid["M"].T).max(0)
    boxes = np.stack([ix - half[0], iy - half[1], ix + half[0], iy + half[1]], 1)
    med = np.nanmedian(sc)
    mad = 1.4826 * np.nanmedian(np.abs(sc - med)) + 1e-6
    return Instances(np.stack([ix, iy], 1), boxes, sc, (sc - med) / mad < -z_thr, rg)


class LatticeMatcher:
    """Lattice-first Instance Grid；score_map 為估計幾何下的單一模板 ZNCC（含極性），供共用指標使用。"""

    def __init__(self, sigma_cells=1.0, amp_frac=0.5, min_region_cells=8.0, z_thr=3.5, name="lattice"):
        self.sigma_cells, self.amp_frac = sigma_cells, amp_frac
        self.min_region_cells, self.z_thr = min_region_cells, z_thr
        self.name = name
        self.last = None

    def score_map(self, search, ref, cell_px=None):
        self.last = self.analyze(search, ref)
        return self.last["score_map"]

    def instances(self):
        return self.last["instances"]

    def analyze(self, search, ref):
        search, ref = search.astype(np.float32), ref.astype(np.float32)   # cv2.matchTemplate 只收 8U/32F
        cell = (ref.shape[1] / REF_CELLS[0], ref.shape[0] / REF_CELLS[1])
        hp = highpass(search, cell)
        C, _ = ref_coefficients(ref, cell)
        fr = make_frame(cell, *fit_geometry(hp, C, cell), C)
        sigma = self.sigma_cells * fr.cell_s
        gpa = [_gpa(hp, fr.G[k], sigma) for k in range(2)]
        amp, periodic, regions, inner = periodic_regions(hp, fr, sigma, self.amp_frac, self.min_region_cells)
        N1, N2, Q = node_index_grid(fr, search.shape)
        _, _, cell_map, inst_map = templates(search, ref, fr)
        hyps = score_hypotheses(gpa, fr, C, periodic, cell_map, N1, N2, Q)
        best = max(hyps, key=lambda h: h["median"])    # 同分時取先出現者
        grid = instance_grid(fr, best["nodes"], best["cell_score"], regions, inner, N1, N2)
        inst = finalize(grid, best["nodes"], fr, inst_map, best["sign"], self.z_thr)
        shape = search.shape
        sm = ScoreMap(best["sign"] * inst_map, np.full(shape, fr.s, np.float32), np.full(shape, np.rad2deg(fr.th), np.float32))
        return dict(score_map=sm, instances=inst, scale=fr.s, angle_deg=float(np.rad2deg(fr.th)),
                    polarity=best["sign"], harmonics=fr.hs, periodic=periodic, amp=amp)
