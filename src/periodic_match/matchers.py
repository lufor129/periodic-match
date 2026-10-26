"""相似度圖：OpenCV ZNCC 與 DINOv3 特徵空間 correlation。

兩者共用同一組 Reference 尺度／角度搜尋，並輸出「以 Reference 中心為錨點」、
與 Search Image 同尺寸的分數圖（取所有尺度／角度的最大值），以便公平比較。
"""
import json
from dataclasses import dataclass

import cv2
import numpy as np

SCALES = np.round(np.arange(0.80, 1.3001, 0.05), 3)
ANGLES = np.arange(-5.0, 5.001, 2.5)
INVALID = -10.0


@dataclass
class ScoreMap:
    score: np.ndarray   # (H,W)，NaN = 無法放置 Reference
    scale: np.ndarray   # (H,W) 最佳尺度
    angle: np.ndarray   # (H,W) 最佳角度


def template_set(ref, scales=SCALES, angles=ANGLES):
    for s in scales:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        r = cv2.resize(ref, None, fx=s, fy=s, interpolation=interp)
        h, w = r.shape
        for a in angles:
            if a == 0:
                yield s, a, r
                continue
            m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), a, 1.0)
            yield s, a, cv2.warpAffine(r, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def place(m, stride, origin, shape):
    """m[i,j] 對應影像座標 (origin_y + i*stride, origin_x + j*stride)，雙線性內插到整張影像。"""
    a = np.float32([[stride, 0, origin[1]], [0, stride, origin[0]]])
    out = cv2.warpAffine(m.astype(np.float32), a, (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=INVALID)
    return out


class _MaxAccumulator:
    def __init__(self, shape):
        self.score = np.full(shape, INVALID, np.float32)
        self.scale = np.zeros(shape, np.float32)
        self.angle = np.zeros(shape, np.float32)

    def update(self, m, s, a):
        better = m > self.score
        self.score[better], self.scale[better], self.angle[better] = m[better], s, a

    def result(self):
        score = self.score.copy()
        score[score < -1.01] = np.nan
        return ScoreMap(score, self.scale, self.angle)


class NCCMatcher:
    """cv2.TM_CCOEFF_NORMED，多尺度／角度搜尋。"""
    name = "ncc"

    def score_map(self, search, ref, cell_px=None):
        acc = _MaxAccumulator(search.shape)
        for s, a, t in template_set(ref):
            th, tw = t.shape
            r = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
            acc.update(place(r, 1.0, ((th - 1) / 2, (tw - 1) / 2), search.shape), s, a)
        return acc.result()


class DinoMatcher:
    """DINOv3 patch 特徵；Reference token grid 當 kernel，於 Search 特徵圖上做特徵空間比對。

    - 輸入先放大，使一個 Unit Cell 至少涵蓋 min_patches_per_cell 個 patch。
    - Reference 是整數個 Unit Cell，可無縫 tile 成 3×3 提供 context，再取中央 token，
      避免 Reference 與 Search 的 attention context 不對稱。
    - mode="zncc"：每個窗口減去局部平均特徵後正規化（多通道 ZNCC）；
      mode="cos"：全圖平均中心化後的平均 cosine。
    """

    def __init__(self, model_id, device, min_patches_per_cell=4.0, tile=2048, margin=256,
                 mode="zncc", ref_context=True, name=None):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        cfg = json.load(open(hf_hub_download(model_id, "preprocessor_config.json")))
        self.mean = torch.tensor(cfg["image_mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["image_std"], device=device).view(1, 3, 1, 1)
        self.patch = self.model.config.patch_size
        self.n_skip = 1 + getattr(self.model.config, "num_register_tokens", 0)
        self.min_ppc, self.tile, self.margin = min_patches_per_cell, tile, margin
        self.mode, self.ref_context = mode, ref_context
        self.name = name or model_id.split("/")[-1]

    def _forward(self, img):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)[None, None].expand(1, 3, -1, -1)
        x = (x - self.mean) / self.std
        h, w = img.shape[0] // self.patch, img.shape[1] // self.patch
        with torch.inference_mode():
            tok = self.model(pixel_values=x).last_hidden_state[0, self.n_skip:]
        return tok.T.reshape(-1, h, w).float()

    def features(self, img):
        """img 尺寸須為 patch 的倍數；大圖切塊（含重疊邊界）以控制 attention 記憶體。"""
        p, H, W = self.patch, img.shape[0], img.shape[1]
        if H * W <= self.tile ** 2:
            return self._forward(img)
        core, m = self.tile - 2 * self.margin, self.margin
        out = None
        for y in range(0, H, core):
            for x in range(0, W, core):
                y0, x0 = max(0, y - m), max(0, x - m)
                y1, x1 = min(H, y + core + m), min(W, x + core + m)
                f = self._forward(img[y0:y1, x0:x1])
                if out is None:
                    out = self.torch.zeros((f.shape[0], H // p, W // p), device=self.device)
                cy1, cx1 = min(H, y + core), min(W, x + core)
                out[:, y // p:cy1 // p, x // p:cx1 // p] = f[:, (y - y0) // p:(cy1 - y0) // p, (x - x0) // p:(cx1 - x0) // p]
        return out

    def _ref_features(self, ref, fy, fx):
        """每組 (尺度, 角度) 的 Reference token grid，(C, th, tw)。"""
        p = self.patch
        big = np.tile(ref, (3, 3)) if self.ref_context else ref
        k = 3 if self.ref_context else 1
        for s, a, t in template_set(big):
            th = max(1, round(ref.shape[0] * s * fy / p))
            tw = max(1, round(ref.shape[1] * s * fx / p))
            f = self.features(cv2.resize(t, (k * tw * p, k * th * p), interpolation=cv2.INTER_CUBIC))
            c0 = (k - 1) // 2
            yield s, a, f[:, c0 * th:(c0 + 1) * th, c0 * tw:(c0 + 1) * tw]

    def score_map(self, search, ref, cell_px):
        torch, F, p = self.torch, self.torch.nn.functional, self.patch
        f = max(1.0, self.min_ppc * p / cell_px)
        H, W = search.shape
        Hu, Wu = int(round(H * f)), int(round(W * f))
        fy, fx = Hu / H, Wu / W
        up = cv2.resize(search, (Wu, Hu), interpolation=cv2.INTER_CUBIC)
        up = up[: Hu // p * p, : Wu // p * p]
        fs = self.features(up)
        mu = fs.mean((1, 2), keepdim=True)
        fs = F.normalize(fs - mu, dim=0)
        sq = (fs ** 2).sum(0, keepdim=True)

        acc = _MaxAccumulator(search.shape)
        for s, a, fr in self._ref_features(ref, fy, fx):
            th, tw = fr.shape[1:]
            fr = F.normalize(fr - mu, dim=0)
            if self.mode == "zncc":
                frc = fr - fr.mean((1, 2), keepdim=True)
                n = th * tw
                num = F.conv2d(fs[None], frc[None])[0, 0]
                s1 = F.avg_pool2d(fs[None], (th, tw), stride=1)[0] * n
                s2 = F.avg_pool2d(sq[None], (th, tw), stride=1)[0, 0] * n
                var = (s2 - (s1 ** 2).sum(0) / n).clamp_min(1e-6)
                r = num / (var.sqrt() * frc.norm() + 1e-6)
            else:
                r = F.conv2d(fs[None], fr[None])[0, 0] / (th * tw)
            r = r.cpu().numpy()
            # token (i,j) 的 kernel 中心在放大影像 (i*p + th*p/2 - 0.5)，換回原影像座標
            origin = ((th * p / 2) / fy - 0.5, (tw * p / 2) / fx - 0.5)
            acc.update(place(r, p / fy, origin, search.shape), s, a)
        return acc.result()
