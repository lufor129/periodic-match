"""SEM 風格 contact hole array 的合成場景與 ground truth。

座標慣例：像素索引即座標（pixel center 在整數位置）。
「理想座標」u 是 Reference 倍率下、未旋轉未形變的設計座標；Search Image 透過
u = c + off + R(-θ)(x - c)/s + d(x) 由像素座標 x 反推理想座標。
"""
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

SQRT3 = float(np.sqrt(3.0))
REF_CELLS = (3, 3)  # Reference 含 3×3 個 Unit Cell


@dataclass(frozen=True)
class Design:
    """一種 contact hole array 設計，定義 Unit Cell 與孔在 Cell 內的位置。"""
    lattice: str                 # "square" | "hex"（hex 以 pitch × pitch√3 的置中矩形 Cell 表示）
    pitch: float                 # 孔距，Reference 像素
    hole_offset: tuple = (0.5, 0.5)  # 孔在 Unit Cell 內的位置（比例），決定 Phase
    radius_ratio: float = 0.3

    @property
    def cell(self):
        return (self.pitch, self.pitch) if self.lattice == "square" else (self.pitch, self.pitch * SQRT3)

    def sites(self):
        ox, oy = self.hole_offset
        if self.lattice == "square":
            return [(ox, oy)]
        return [(ox, oy), ((ox + 0.5) % 1.0, (oy + 0.5) % 1.0)]


@dataclass(frozen=True)
class Imaging:
    """成像條件：幾何（相對 Reference）、灰階、雜訊與製程變異。"""
    scale: float = 1.0
    rotation_deg: float = 0.0
    elastic: float = 0.0         # 局部位移幅度（pitch 比例）
    gamma: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0
    gradient: float = 0.0        # 低頻亮度漂移幅度（charging）
    blur: float = 0.5            # 高斯模糊 σ（像素）
    noise: float = 0.02          # 雜訊 σ
    defect_rate: float = 0.0     # 每個孔成為 defect 的機率
    cd_jitter: float = 0.03      # 孔半徑相對變異
    invert: bool = False         # 對比反轉（如偵測器 / 成像模式不同）
    occlusion: float = 0.0       # 污染微粒覆蓋的面積比例（被遮到的 Instance 為 Anomalous）


@dataclass
class Scene:
    search: np.ndarray           # float32 [0,1]
    reference: np.ndarray        # float32 [0,1]
    periodic_mask: np.ndarray    # bool，Periodic Region
    centers: np.ndarray          # (N,2) 所有合法 Instance 中心 (x,y)，涵蓋 9 種切法
    boxes: np.ndarray            # (N,4) x0,y0,x1,y1
    anomalous: np.ndarray        # (N,) bool
    tiling: np.ndarray           # (N,) 0..8，屬於哪一種切法
    n_max: int                   # 最多 Instance 的切法所含數量
    design: Design
    imaging: Imaging
    info: dict = field(default_factory=dict)

    @property
    def cell_px(self):
        """Search Image 上 Unit Cell 的 (w, h) 像素。"""
        cw, ch = self.design.cell
        return cw * self.imaging.scale, ch * self.imaging.scale


# ---------------------------------------------------------------- 孔陣列渲染

class _Holes:
    """每個孔的 CD 變異、位移與 defect，依 (site, j, i) 索引。"""

    def __init__(self, design, imaging, irange, jrange, rng, allow_defects=True):
        self.i0, self.j0 = irange[0], jrange[0]
        n_s, nj, ni = len(design.sites()), jrange[1] - jrange[0] + 1, irange[1] - irange[0] + 1
        shape = (n_s, nj, ni)
        r0 = design.radius_ratio * design.pitch
        self.rscale = 1.0 + rng.normal(0, imaging.cd_jitter, shape)
        self.shift = rng.normal(0, 0.02 * r0, shape + (2,))
        self.missing = np.zeros(shape, bool)
        self.defect = np.zeros(shape, bool)
        if allow_defects and imaging.defect_rate > 0:
            self.defect = rng.random(shape) < imaging.defect_rate
            kind = rng.integers(0, 4, shape)
            self.missing = self.defect & (kind == 0)
            self.rscale = np.where(self.defect & (kind == 1), 1.5, self.rscale)
            self.rscale = np.where(self.defect & (kind == 2), 0.55, self.rscale)
            ang = rng.uniform(0, 2 * np.pi, shape)
            moved = self.defect & (kind == 3)
            self.shift[moved] = 0.4 * r0 * np.stack([np.cos(ang), np.sin(ang)], -1)[moved]


def _signed_distance(design, holes, u, v):
    """理想座標下到最近孔邊緣的有號距離（孔內為負）。"""
    cw, ch = design.cell
    r0 = design.radius_ratio * design.pitch
    best = np.full(u.shape, np.inf, np.float32)
    for k, (fx, fy) in enumerate(design.sites()):
        i = np.round((u - fx * cw) / cw).astype(np.int64)
        j = np.round((v - fy * ch) / ch).astype(np.int64)
        ii = np.clip(i - holes.i0, 0, holes.rscale.shape[2] - 1)
        jj = np.clip(j - holes.j0, 0, holes.rscale.shape[1] - 1)
        cx = i * cw + fx * cw + holes.shift[k, jj, ii, 0]
        cy = j * ch + fy * ch + holes.shift[k, jj, ii, 1]
        d = np.hypot(u - cx, v - cy) - r0 * holes.rscale[k, jj, ii]
        d = np.where(holes.missing[k, jj, ii], np.inf, d)
        best = np.minimum(best, d)
    return best


def _sem_profile(sd, pitch, rng_grain):
    """SEM 風格：背景中灰、孔內暗、孔緣亮（edge effect）。"""
    w = max(0.8, 0.05 * pitch)
    inside = 1.0 / (1.0 + np.exp(np.clip(sd / (0.5 * w), -50, 50)))
    rim = np.exp(-(sd / w) ** 2)
    return (0.35 * (1 - inside) + 0.10 * inside + 0.45 * rim + rng_grain).astype(np.float32)


def _smooth_field(shape, corr_px, rng, n=1):
    """相關長度約 corr_px 的平滑隨機場，最大絕對值正規化為 1。"""
    h, w = shape
    gh, gw = max(2, int(np.ceil(h / corr_px)) + 2), max(2, int(np.ceil(w / corr_px)) + 2)
    out = []
    for _ in range(n):
        g = rng.normal(size=(gh, gw)).astype(np.float32)
        f = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        out.append(f / (np.abs(f).max() + 1e-6))
    return out


def _grain(shape, rng, amp=0.03):
    g = cv2.GaussianBlur(rng.normal(size=shape).astype(np.float32), (0, 0), 1.5)
    return amp * g / (g.std() + 1e-6)


class _Warp:
    """Search 像素座標 ↔ 理想座標的映射（尺度、旋轉、彈性形變）。"""

    def __init__(self, shape, imaging, design, offset, rng):
        self.h, self.w = shape
        self.c = np.array([(self.w - 1) / 2, (self.h - 1) / 2])
        self.s, self.th = imaging.scale, np.deg2rad(imaging.rotation_deg)
        self.off = np.asarray(offset, float)
        amp = imaging.elastic * design.pitch
        if amp > 0:
            du, dv = _smooth_field(shape, 5 * design.pitch * self.s, rng, n=2)
            self.d = (amp * du, amp * dv)
        else:
            self.d = None

    def _disp(self, x, y):
        if self.d is None:
            return 0.0, 0.0
        coords = np.stack([np.clip(y, 0, self.h - 1), np.clip(x, 0, self.w - 1)])
        return (map_coordinates(self.d[0], coords, order=1),
                map_coordinates(self.d[1], coords, order=1))

    def inverse(self, x, y):
        """像素 → 理想座標。"""
        cos, sin = np.cos(self.th), np.sin(self.th)
        xp, yp = (x - self.c[0]) / self.s, (y - self.c[1]) / self.s
        u = cos * xp + sin * yp + self.c[0] + self.off[0]
        v = -sin * xp + cos * yp + self.c[1] + self.off[1]
        if self.d is not None:
            if np.ndim(x) == 2 and np.shape(x) == (self.h, self.w):
                u, v = u + self.d[0], v + self.d[1]
            else:
                du, dv = self._disp(x, y)
                u, v = u + du, v + dv
        return u, v

    def forward(self, u, v, iters=6):
        """理想座標 → 像素（形變部分以不動點迭代求解）。"""
        cos, sin = np.cos(self.th), np.sin(self.th)
        u, v = np.asarray(u, float), np.asarray(v, float)
        du = dv = np.zeros_like(u)
        for _ in range(iters if self.d is not None else 1):
            a = u - self.c[0] - self.off[0] - du
            b = v - self.c[1] - self.off[1] - dv
            x = self.s * (cos * a - sin * b) + self.c[0]
            y = self.s * (sin * a + cos * b) + self.c[1]
            if self.d is not None:
                du, dv = self._disp(x, y)
        return x, y


def _render_holes(design, imaging, warp, rng, allow_defects):
    yy, xx = np.mgrid[0:warp.h, 0:warp.w].astype(np.float64)
    u, v = warp.inverse(xx, yy)
    cw, ch = design.cell
    irange = (int(np.floor(u.min() / cw)) - 2, int(np.ceil(u.max() / cw)) + 2)
    jrange = (int(np.floor(v.min() / ch)) - 2, int(np.ceil(v.max() / ch)) + 2)
    holes = _Holes(design, imaging, irange, jrange, rng, allow_defects)
    img = _sem_profile(_signed_distance(design, holes, u, v), design.pitch, _grain(u.shape, rng))
    return img, holes, irange, jrange


def _apply_imaging(img, im, rng):
    """模糊 → 亮度漂移 → 對比/亮度 → gamma → 雜訊。"""
    if im.blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), im.blur)
    if im.gradient > 0:
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
        a = rng.uniform(-1, 1, 2)
        plane = a[0] * (xx - 0.5) + a[1] * (yy - 0.5)
        blob = _smooth_field(img.shape, max(h, w) / 2, rng)[0]
        img = img + im.gradient * (plane / (np.abs(plane).max() + 1e-6) + blob) / 2
    img = (img - 0.35) * im.contrast + 0.35 + im.brightness
    if im.invert:
        img = 1.0 - img
    img = np.clip(img, 0, 1) ** im.gamma
    img = img + im.noise * (0.5 + img) * rng.normal(size=img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- 非 Periodic Region

def _distractor_and_lines(shape, design, imaging, warp, rng):
    """非 Periodic Region 內容：近似但不同設計的孔陣列，以及隨機繞線。"""
    other = replace(design, lattice="square", pitch=design.pitch * 1.35, radius_ratio=0.22)
    distractor, *_ = _render_holes(other, replace(imaging, defect_rate=0), warp, rng, False)

    lines = np.full(shape, 0.35, np.float32) + _grain(shape, rng)
    h, w = shape
    s = imaging.scale
    lw = max(2, int(0.25 * design.pitch * s))
    for _ in range(int(h * w / (design.pitch * s) ** 2 / 6)):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        if rng.random() < 0.7:
            cv2.rectangle(lines, (x0, y0), (x0 + int(rng.integers(2, 12) * design.pitch * s), y0 + lw), 0.75, -1)
        else:
            cv2.rectangle(lines, (x0, y0), (x0 + lw, y0 + int(rng.integers(1, 5) * design.pitch * s)), 0.75, -1)
    return distractor, cv2.GaussianBlur(lines, (0, 0), 0.8)


def _particles(canvas, cell, coverage, rng):
    """在 canvas 上畫隨機微粒（橢圓，0.5–2 個 Cell 大小），直到覆蓋率達 coverage；回傳覆蓋 mask。"""
    h, w = canvas.shape
    mask = np.zeros(canvas.shape, np.uint8)
    while mask.mean() < coverage:
        axes = (int(rng.uniform(0.25, 1.0) * cell), int(rng.uniform(0.25, 1.0) * cell))
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        m = np.zeros_like(mask)
        cv2.ellipse(m, center, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
        canvas[:] = canvas * (1 - soft) + float(rng.uniform(0.05, 0.9)) * soft
        mask |= m
    return mask.astype(bool)


def _random_rects(shape, rng, n=2):
    h, w = shape
    rects = []
    for _ in range(n):
        rw, rh = int(rng.uniform(0.2, 0.35) * w), int(rng.uniform(0.2, 0.35) * h)
        rects.append((int(rng.integers(0, w - rw)), int(rng.integers(0, h - rh)), rw, rh))
    return rects


# ---------------------------------------------------------------- Scene

def search_size(pitch):
    return int(np.clip(30 * pitch, 768, 2048))


def make_reference(design, imaging, rng):
    """以「另一台機台」條件拍攝、從 Unit Cell 邊界起裁的 3×3 Cell Reference。"""
    cw, ch = design.cell
    nx, ny = REF_CELLS
    w, h = int(round(nx * cw)), int(round(ny * ch))
    i0, j0 = rng.integers(0, 50, 2)
    # 無旋轉縮放時 u = x + off；像素 x 涵蓋 [x, x+1)，故陣列中心 = Reference 中心
    flat = replace(imaging, scale=1.0, rotation_deg=0.0, elastic=0.0)
    warp = _Warp((h, w), flat, design, (i0 * cw + 0.5, j0 * ch + 0.5), rng)
    img, *_ = _render_holes(design, flat, warp, rng, False)
    return _apply_imaging(img, imaging, rng)


def make_scene(design, imaging, ref_imaging, seed):
    rng = np.random.default_rng(seed)
    size = search_size(design.pitch)
    shape = (size, size)
    cw, ch = design.cell
    warp = _Warp(shape, imaging, design, rng.uniform(0, 1000, 2), rng)

    holes_img, holes, irange, jrange = _render_holes(design, imaging, warp, rng, True)
    distractor, lines = _distractor_and_lines(shape, design, imaging, warp, rng)
    mask = np.ones(shape, bool)
    canvas = holes_img.copy()
    for k, (x, y, w, h) in enumerate(_random_rects(shape, rng)):
        canvas[y:y + h, x:x + w] = (distractor if k == 0 else lines)[y:y + h, x:x + w]
        mask[y:y + h, x:x + w] = False
    occluded = (_particles(canvas, design.pitch * imaging.scale, imaging.occlusion, rng) if imaging.occlusion > 0
                else np.zeros(shape, bool))
    search = _apply_imaging(canvas, imaging, rng)
    reference = make_reference(design, ref_imaging, rng)

    # 所有合法 Instance：3×3 Cell 完整落在影像與 Periodic Region 內。
    # 六角陣列平移 (½,½) 個矩形 Cell 後內容完全相同，也是同一 Phase 的合法 Instance。
    nx, ny = REF_CELLS
    ii, jj = np.meshgrid(np.arange(irange[0], irange[1] - nx + 1), np.arange(jrange[0], jrange[1] - ny + 1))
    shifts = [0.0, 0.5] if design.lattice == "hex" else [0.0]
    half = np.concatenate([np.full(ii.size, k) for k in range(len(shifts))])
    ii = np.concatenate([ii.ravel() + d for d in shifts])
    jj = np.concatenate([jj.ravel() + d for d in shifts])
    g = np.linspace(0, 1, 5)
    gu, gv = np.meshgrid(g, g)
    pu = (ii[:, None] + nx * gu.ravel()[None]) * cw
    pv = (jj[:, None] + ny * gv.ravel()[None]) * ch
    px, py = warp.forward(pu.ravel(), pv.ravel())
    px, py = px.reshape(pu.shape), py.reshape(pv.shape)
    inside = (px >= 0).all(1) & (py >= 0).all(1) & (px <= size - 1).all(1) & (py <= size - 1).all(1)
    xi, yi = np.clip(np.round(px), 0, size - 1).astype(int), np.clip(np.round(py), 0, size - 1).astype(int)
    valid = inside & mask[yi, xi].all(1)
    ii, jj, half, px, py = ii[valid], jj[valid], half[valid], px[valid], py[valid]
    cx, cy = warp.forward((ii + nx / 2) * cw, (jj + ny / 2) * ch)
    boxes = np.stack([px.min(1), py.min(1), px.max(1), py.max(1)], 1)

    # Anomalous：框內（理想座標）有 defect 孔，或被微粒遮到
    k, dj, di = np.nonzero(holes.defect)
    sites = np.array(design.sites())
    du = (di + holes.i0 + sites[k, 0]) * cw + holes.shift[k, dj, di, 0]
    dv = (dj + holes.j0 + sites[k, 1]) * ch + holes.shift[k, dj, di, 1]
    anomalous = ((du[None] >= ii[:, None] * cw) & (du[None] < (ii[:, None] + nx) * cw)
                 & (dv[None] >= jj[:, None] * ch) & (dv[None] < (jj[:, None] + ny) * ch)).any(1)
    if occluded.any():
        integ = np.pad(occluded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        b = np.clip(np.round(boxes), 0, size - 1).astype(int)
        hit = integ[b[:, 3] + 1, b[:, 2] + 1] - integ[b[:, 1], b[:, 2] + 1] - integ[b[:, 3] + 1, b[:, 0]] + integ[b[:, 1], b[:, 0]]
        anomalous |= hit > 0
    tiling = ((np.floor(ii).astype(int) % nx) * ny + np.floor(jj).astype(int) % ny) + nx * ny * half
    n_max = int(np.bincount(tiling).max()) if len(tiling) else 0

    return Scene(search, reference, mask, np.stack([cx, cy], 1), boxes, anomalous, tiling, n_max,
                 design, imaging, {"seed": seed})
