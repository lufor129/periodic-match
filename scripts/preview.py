"""把影像轉成看得清楚的 8-bit PNG：全圖縮圖與原解析度裁圖（docs/method-selection.md 步驟 1）。

用法：
  uv run python scripts/preview.py data/A.tif out/select/A
  uv run python scripts/preview.py data/A.tif out/select/A --at 1200,300 --at 400,900
  uv run python scripts/preview.py data/ref_A.tif out/select/A/ref

輸出到 <out>/：
  overview.png       全圖縮放到長邊 1500 px；小圖（如 Reference）以最近鄰放大
  crop_<位置>.png    512 px 見方、原解析度：center 與四個象限中心；--at x,y 另加 crop_x<x>_y<y>.png
                     影像短邊不到 512 px 時不輸出裁圖
"""
import argparse
from pathlib import Path

import cv2

from run_on_image import load_gray

CROP = 512
OVERVIEW = 1500


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("out")
    p.add_argument("--at", action="append", default=[], metavar="X,Y", help="另加裁圖中心（原圖像素座標）")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    u8 = (load_gray(a.image) * 255).astype("uint8")
    h, w = u8.shape
    f = OVERVIEW / max(h, w)
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_NEAREST
    cv2.imwrite(str(out / "overview.png"), cv2.resize(u8, None, fx=f, fy=f, interpolation=interp))
    print(f"{w}x{h} px；overview 座標 / {f:.3f} = 原圖座標")
    if min(h, w) < CROP:
        return
    centers = {"center": (w // 2, h // 2), "top_left": (w // 4, h // 4), "top_right": (3 * w // 4, h // 4),
               "bottom_left": (w // 4, 3 * h // 4), "bottom_right": (3 * w // 4, 3 * h // 4)}
    for s in a.at:
        x, y = map(int, s.split(","))
        centers[f"x{x}_y{y}"] = (x, y)
    for name, (x, y) in centers.items():
        x0 = min(max(0, x - CROP // 2), w - CROP)
        y0 = min(max(0, y - CROP // 2), h - CROP)
        cv2.imwrite(str(out / f"crop_{name}.png"), u8[y0:y0 + CROP, x0:x0 + CROP])


if __name__ == "__main__":
    main()
