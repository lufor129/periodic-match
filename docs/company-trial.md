# 公司真實影像試跑 Lattice-first

目標：在公司的 SEM/TEM 影像上試跑 Lattice-first，並和現有的 Template Matching 正面比較。產出一份試跑報告 `out/report.md`，讓使用者判斷這個方法值不值得往下做。

用詞以 `GLOSSARY.md` 為準（Unit Cell、Reference、Search Image、Instance、Phase、Instance Grid、Periodic Region、Anomalous Instance）。報告和程式註解都用這套詞。

## 資料邊界

公司影像、裁出的 Reference，以及由它們產生的所有輸出，只放在 `data/` 和 `out/` 底下。這兩個目錄、`*.tif` 和 `*.log` 都已列在 `.gitignore`，所以 commit 只會包含程式碼和文件。

- 每次 commit 前先跑 `git status`，確認清單裡沒有任何影像、CSV 或 `out/` 底下的檔案。
- 要帶回外部的只有聚合數字（張數、P/R、耗時、參數），以及不含影像的文字描述。
- 截圖、疊圖、座標表都留在公司機器上。

## 方法摘要

先從 Search Image 估出晶格，再用 Reference 決定 Phase，直接排出 Instance Grid。因此流程裡沒有分數閾值，也沒有 NMS。各步驟依序是：

1. Fourier 幾何估計 (s, θ)。
2. GPA 求晶格節點。
3. 晶格自相似度判斷 Periodic Region。
4. 每個區域取 Instance 數最多的排法。
5. 逐 Cell 做 ZNCC，取最小值當 Instance 分數，用 robust z-score 標記 Anomalous。

細節在 `src/periodic_match/lattice.py` 檔頭。要逐步看中間結果，用 `notebooks/lattice_first_walkthrough.ipynb`（kernel 選 `Python (periodic-match)`，第一次使用要先執行 `uv run python -m ipykernel install --prefix .venv --name periodic-match --display-name "Python (periodic-match)"`）。下面每個診斷步驟都標了 notebook 的對應章節。

## 步驟

### 1. 環境

```
uv sync --no-dev          # 只裝 numpy/scipy/opencv/pandas/matplotlib，不裝 torch
uv sync                   # 加上 jupyter，要開 notebook 時用
```

- 沒有 uv 時，可改用 `python -m venv .venv && .venv/bin/pip install -e .`。
- 只有要比較 DINOv3 時才需要 `--extra dino`，這次試跑用不到。

**完成條件**：下列煙霧測試有輸出 `scenes.csv`，而且 lattice 那一列的 `precision` 接近 1：

```
uv run python scripts/run_benchmark.py --quick --methods ncc,lattice --limit 2 --out out/smoke
```

### 2. 準備 Reference

Reference 必須**剛好**包含 3×3 個 Unit Cell。程式用 `ref 寬 / 3` 和 `ref 高 / 3` 當作 Cell 大小，裁切的邊界就定義了 Phase。

- 公司現有 Template Matching 用的模板如果不是 3×3 Cell，就從同一張圖重新裁一張 3×3 的。
  - 裁切位置要和現有模板一致，這樣兩個方法框出的才是同一個 Phase。
- 如果結構本來就需要其他 Cell 數，改 `lattice.py` 的 `REF_CELLS` 即可，這是唯一需要改的地方。
- Reference 裡的 Cell 必須是二維週期。line/space 這類一維週期結構只有一個方向的諧波，會丟出「Reference 沒有兩個不共線的諧波」。遇到這種情況就記為「方法不適用」，不要硬調參數。

**完成條件**：`out/<run>/<image>/summary.json` 裡的 `cell_px_ref`，和人工量 Reference 得到的 pitch 一致，誤差在 2% 以內。

### 3. 選試跑集

先挑 5–10 組（Reference, Search Image），盡量涵蓋以下每一類各至少一組：

- 不同機台或倍率
- 有 defect 或微粒的影像
- 週期區域只佔一部分的影像（例如邊緣或 pad 旁）
- 現有 Template Matching 已知失敗的案例

最後一類最重要，這次試跑的價值就在這裡。在 `data/manifest.csv` 記下每組的類別和已知問題。

### 4. 執行

```
uv run python scripts/run_on_image.py data/ref_A.png data/A_*.tif --out out/A
```

每張 Search Image 會在 `out/A/<檔名>/` 底下產生以下檔案，`out/A/runs.csv` 則彙總整批：

| 檔案 | 內容 |
|---|---|
| `summary.json` | s、θ、polarity、Periodic Region 比例、Instance 數、耗時 |
| `overlay.png` | Instance Grid，綠框為 Instance，紅框為 Anomalous |
| `diagnostics.png` | Periodic Region、自相似度、分數直方圖 |
| `instances.csv` | 每個 Instance 的中心、框、分數、是否 Anomalous、所屬區域 |

輸入可以是任意 bit depth 的影像，程式內會做 percentile 正規化。

### 5. 逐張診斷

每張圖依下面的順序檢查。在第一個出錯的地方停下，查「故障對照」修正，再從頭跑一次。後面的步驟都依賴前面的結果，所以順序不能跳。

1. **幾何**：`summary.json` 的 `scale` 是 Search Image 的 Cell 像素大小除以 Reference 的 Cell 像素大小，應接近兩張影像的倍率比（Search Image 的 px/nm 除以 Reference 的 px/nm），`angle_deg` 應接近實際旋轉角。（notebook §2）
2. **Periodic Region**：在 `diagnostics.png` 中，紅色區塊應該只覆蓋真正沒有結構的地方。（notebook §4）
3. **Instance Grid**：在 `overlay.png` 中檢查四件事：
   - 框和結構對齊；
   - Phase 和 Reference 一致；
   - 每塊週期區域都排滿；
   - 邊界上不完整的 Instance 沒有被框出來。

   （notebook §5–§8）
4. **Anomalous**：紅框應落在真正有 defect 或微粒的位置。（notebook §9）

**完成條件**：每張圖都有一行判定，判定值是「正確」、「幾何錯」、「區域錯」、「排列錯」、「Anomalous 錯」或「方法不適用」之一，並附上已嘗試的參數。

### 6. 和 Template Matching 比較

用公司現有流程，在同一批圖上產出 TM 的框。兩邊的框以中心距離小於 0.25 Cell、一對一配對（作法同 `metrics.match_instances`）。

- 只有一方有的框，就是**分歧框**。逐個看 overlay，判定該算哪一方的錯。
- 沒有人工標註 GT 時，分歧框就是唯一需要人看的地方。把分歧框的裁圖整理成一頁給使用者確認，不要自己猜。

**完成條件**：每組都有下列數字：

- 兩方各自的框數
- 配對數
- 各自獨有的框數
- 人判定後兩方各自的 P/R（或「待人確認」）
- 兩方的耗時

### 7. 報告

寫 `out/report.md`，內容包含：

- 每組一列的總表，欄位是步驟 6 的數字加上步驟 5 的判定；
- 失敗案例分類，以及最後採用的參數；
- 和 benchmark 已知弱點（見下）對照，標出哪些是新發現。

要帶出公司的版本只保留聚合數字和文字描述。

## 故障對照

| 症狀 | 原因 | 處理 |
|---|---|---|
| `scale` 或 `angle_deg` 明顯錯，Instance Grid 整片歪掉 | 倍率比超出 0.7–1.4，或旋轉超過 ±7°（`fit_geometry` 的搜尋範圍） | 先把 Reference resize 到大致相同倍率；或放寬 `fit_geometry` 的 `s_range` 和 `th_range_deg` |
| 丟出「沒有兩個不共線的諧波」 | Reference 不是二維週期，或裁成的不是 3×3 Cell | 重裁 Reference；如果是一維結構，記為「方法不適用」 |
| 週期區域邊緣的 Instance 漏框 | Periodic Region 太緊 | 調低 `--amp-frac`，例如 0.4 |
| 框跑進非週期區 | Periodic Region 太鬆 | 調高 `--amp-frac`，例如 0.6 |
| 小顆微粒把區域切碎 | 洞沒有被補起來 | 調高 `--min-region-cells`。注意：如果太大，真正的小塊非週期區也會被補成週期區 |
| 整片框偏移整數個 Cell | 每塊區域都取 Instance 數最多的排法（見 GLOSSARY 的 Instance Grid），這是設計行為 | 只有排法與 Reference 的 Phase 不一致時才算錯 |
| `polarity` 為 -1 | 兩台機台的對比相反 | 屬正常，方法本來就處理這種情況 |
| Anomalous 太多或太少 | 標記用的是同一張圖內的相對分數（robust z-score） | 調整 `--z-thr`。沒有 defect 的圖也可能標出少數紅框，報告裡要註明 |
| 強烈漂移或形變處框對不準 | GPA 平滑尺度太大 | 調低 `--sigma-cells`，例如 0.7。但雜訊會變大 |
| 大圖很慢或記憶體不足 | 耗時與像素數成正比 | Cell 大於約 32 px 時，把兩張圖等比例縮小再跑，最後把座標乘回去 |

調參時一次只改一個參數，並把每次嘗試記進報告。

## Benchmark 已知弱點

以下都來自合成的 contact hole 陣列 benchmark，真實影像還沒有驗證過。

- **微粒遮擋**：遮擋面積達 6% 時，recall 掉到 0.65，比 ZNCC 的 0.70 還低。
- **六角陣列**：recall 0.83，比方形陣列的 0.89 低。
- **影像邊緣**：落在邊緣、不完整 Cell 上的 defect 偵測不到。
- **相對強項**：對比反轉、強雜訊、強 gamma。這幾類情況 ZNCC 會大幅失效，Lattice-first 仍能維持。如果公司 Template Matching 的失敗案例正好屬於這幾類，要在報告中特別標出。
