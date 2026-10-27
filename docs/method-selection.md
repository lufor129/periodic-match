# 選方法：Template Matching 或 Lattice-first

目標：看過一張 Search Image（有 Reference 時一併看），判定該用 Template Matching（ZNCC）還是 Lattice-first，並寫成一份判定。

用詞以 `GLOSSARY.md` 為準。表中的 benchmark 數字來自合成 contact hole 陣列（`results/final/scenes.csv`），真實影像還沒驗證過，只用來說明每條規則的由來。

## 1. 轉成可看的圖

公司影像常是 16-bit TIFF。先用 `scripts/preview.py` 轉成 8-bit PNG，輸出放在 `out/select/<圖名>/`：

```
uv run python scripts/preview.py data/A.tif out/select/A
uv run python scripts/preview.py data/ref_A.tif out/select/A/ref     # 有 Reference 時
```

- Search Image 會產生 `overview.png`（長邊 1500 px）和 5 張 512 px 原解析度裁圖：中心和四個象限。
- Reference 只產生放大後的 `overview.png`。
- 要看其他位置（污染、亮度不均、可疑結構）時，加上 `--at x,y`（原圖像素座標；換算方式見指令輸出）再跑一次。

**完成條件**：看過 Search Image 的 `overview.png` 和 5 張裁圖；有 Reference 時，也看過它的 `overview.png`。

## 2. 逐項檢查

每一項的判讀是「成立」、「不成立」或「無法判斷」。判讀以看圖為準，觀察只寫圖裡實際看得到的東西，看不清楚就填「無法判斷」。需要精確數字的情況，交給步驟 4 的試跑。

### 前提：Lattice-first 能不能用

| 項目 | 看哪裡 | 成立的樣子 |
|---|---|---|
| P1 二維週期 | 裁圖 | 同一個結構沿兩個不平行的方向重複，例如孔或柱的方形、六角陣列。只沿一個方向重複（line/space）、隨機繞線、找不到重複單元，都算不成立 |
| P2 找的是重複的 Instance | Reference、任務說明 | Reference 是陣列的一小塊，數得出整數個 Cell。Reference 是 alignment mark、pad 角這類在圖中只出現一次或幾次的目標，算不成立 |

P1 或 P2 不成立時，其餘各列填「—」，直接跳到步驟 5，結論由規則 1 決定。

### 驗證範圍：benchmark 有沒有涵蓋

| 項目 | 看哪裡 | 成立的樣子 |
|---|---|---|
| R1 週期區佔比 | overview | 和 Reference 同一種結構的區域，佔全圖約一半以上。孔距不同的另一塊陣列、繞線區都算非週期區；benchmark 每個場景都有這兩種區塊，週期區約佔 75–90% |
| R2 結構類型 | 裁圖 | Unit Cell 是單一形狀的孔或柱，方形或六角排列。Unit Cell 本身由多種形狀組成（例如同一個 Cell 裡有孔也有線）、TEM 原子晶格，都算不成立 |

### Lattice-first 的優勢

S4、S6 要用到 Cell 大小：

- **Search Image**：在一張裁圖裡沿一排找相隔最遠、都完整的兩個孔，Cell 大小 = 兩孔中心距離 ÷ 兩孔之間的 Cell 數。
- **Reference**：剛好 3×3 Cell 時，Cell 大小 = 寬 ÷ 3；不是的話，用同樣的方法量。

依據欄是 benchmark 的 recall，寫成「ZNCC 對 Lattice-first」。

| 項目 | 看哪裡 | 成立的樣子 | 依據 |
|---|---|---|---|
| S1 對比反轉 | Search 與 Reference 並排 | 同一結構在兩張圖亮暗相反，例如孔在一張是暗的、另一張是亮的 | 0.00 對 0.95 |
| S2 強雜訊 | 裁圖 | 孔的輪廓被顆粒雜訊蓋過，要看縮圖才看得出排列 | 0.33 對 0.91 |
| S3 灰階差異 | Search 與 Reference、overview | 亮暗反轉以外的灰階差異：兩張圖灰階特性明顯不同（一張發灰、一張高對比），或同一張圖有大範圍亮度不均（charging） | 0.77 對 0.90 |
| S4 旋轉或倍率差 | overview、兩邊的 Cell 大小 | 格線相對 Reference 的格線明顯傾斜：沿一排孔橫看整張 overview，上下偏移超過圖寬約 1/10（約 5° 以上）。或兩邊 Cell 大小差 10% 以上。2–3° 的傾斜肉眼看不出，對 ZNCC 的影響也小，算不成立 | 旋轉 4.9°：0.71 對 0.91 |
| S5 漂移或形變 | overview | 同一塊陣列裡，格線彎曲或間距逐漸改變 | 0.75 對 0.90 |
| S6 小 Cell | Search 的 Cell 大小 | ≤ 48 px | 16 px：0.72 對 0.96，耗時也較短 |

### Lattice-first 的弱點

| 項目 | 看哪裡 | 成立的樣子 | 依據 |
|---|---|---|---|
| W1 微粒遮擋 | overview | 污染或微粒蓋住週期區的面積約 5% 以上 | 遮擋 6%：recall 0.70 對 0.65 |
| W2 defect 密且重 Anomalous | 裁圖、任務說明 | 約 1/10 以上的 Cell 有缺孔、孔過大過小或位移，而且任務重點是標出 Anomalous Instance。defect 比這少就是不成立；defect 夠多但任務沒說明重點時，填「無法判斷」 | Anomalous AUROC 0.98 對 0.88 |

### 前處理：寫進判定的「前處理」欄，不影響結論

結論是 Template Matching 時，前處理寫「無」。其餘情況逐條檢查：

- **Reference 不是剛好 3×3 Unit Cell**：寫「重裁 3×3 Reference」（裁法見 `docs/company-trial.md` 步驟 2）。
- **倍率比（Search Cell ÷ Reference Cell）在 0.7–1.4 之外**：寫「Reference resize 約 <比值> 倍」。
- **S4 的傾斜成立**：寫「確認旋轉在 ±7° 內，超過時放寬 `fit_geometry` 的 `th_range_deg`」。
- **Search 的 Cell ≥ 96 px**：寫「兩張圖等比例縮小到 Cell ≈ 48 px」。這個尺寸以上 recall 不再領先 ZNCC，耗時卻是 2–3 倍。

**完成條件**：P1–P2、R1–R2、S1–S6、W1–W2 每一項都有觀察和判讀（P 不成立時其餘填「—」），前處理四條都檢查過。

## 3. 套用決策規則

依序套用，第一條符合的規則決定結論：

1. P1 或 P2 不成立 → **Template Matching**。
2. P1、P2、R1、R2 有任一項不是「成立」 → **試跑**（步驟 4）。
3. S1 或 S2 成立 → **Lattice-first**。這兩種情況 ZNCC 幾乎完全失效，W1、W2 的代價相對小。
4. W1 或 W2 成立：S3–S6 也有成立 → **試跑**；否則 → **Template Matching**。
5. 其餘 → **Lattice-first**。benchmark 上即使沒有任何擾動，Lattice-first 的 precision 和 recall 都高於 ZNCC。

## 4. 試跑（只在規則 2 或 4 指向試跑時做）

試跑需要一張剛好 3×3 Unit Cell 的 Reference。手上沒有時，結論寫「不確定」，依據寫「需要 3×3 Reference 才能試跑」，然後跳到步驟 5。

```
uv run python scripts/run_on_image.py data/ref_A.tif data/A.tif --out out/select/A/run
```

依序檢查 `out/select/A/run/A/` 的輸出，第一個失敗就停：

1. **諧波**：指令丟出「Reference 沒有兩個不共線的諧波」 → **Template Matching**（P1 不成立）。
2. **幾何**：`summary.json` 的 `scale` 接近「Search Cell ÷ Reference Cell」，`angle_deg` 和 overview 看到的傾斜方向、大小一致（看不出傾斜時應在 ±3° 內）。不符合，而且倍率或旋轉超出前處理的範圍時，照前處理處理後重跑；在範圍內仍不符合 → **Template Matching**。
3. **Periodic Region**：`diagnostics.png` 的紅色（非週期）只蓋在沒有 Reference 結構的地方，`periodic_fraction` 和 R1 看到的佔比相近。
4. **Instance Grid**：`overlay.png` 的框對齊結構，每塊週期區內部排滿。區域邊界內約 1–2 個 Cell 寬的帶狀區沒有框是正常的，因為 Instance 的 3×3 Cell 要完整落在區域內；要找的是區域內部的大片空缺，特別是微粒附近。

四項都通過 → **Lattice-first**。第 3 或 4 項失敗 → **Template Matching**，依據寫出失敗項；`docs/company-trial.md` 的故障對照有對應調參時，一併寫出。

**完成條件**：四項都有結果，或已在第一個失敗項停下並記下原因。

## 5. 寫判定

寫到 `out/select/<圖名>/verdict.md`，並在回覆中貼出同樣內容。觀察欄用文字描述位置（例如「左上約 1/4」），這樣判定可以帶出公司。

```markdown
## 判定：<圖名>

- **結論**：Lattice-first ｜ Template Matching ｜ 不確定
- **信心**：高 ｜ 中 ｜ 低 ｜ —（結論為不確定時）
- **決定規則**：規則 <編號>（<觸發的項目>）；試跑時寫「試跑：通過」或「試跑第 <N> 項失敗：<原因>」
- **前處理**：無 ｜ <步驟 2 前處理的動作>

| 項目 | 觀察 | 判讀 |
|---|---|---|
| P1 二維週期 | | |
| P2 找的是重複的 Instance | | |
| R1 週期區佔比 | | |
| R2 結構類型 | | |
| S1 對比反轉 | | |
| S2 強雜訊 | | |
| S3 灰階差異 | | |
| S4 旋轉或倍率差 | | |
| S5 漂移或形變 | | |
| S6 小 Cell | | |
| W1 微粒遮擋 | | |
| W2 defect 密且重 Anomalous | | |

**依據**：一到兩句，說明決定規則為什麼適用這張圖。
```

信心：

- **高**：由規則 1、3 或試跑決定。
- **中**：由規則 4、5 決定。
- **降一級**：某項填了「無法判斷」，而把它改成「成立」或「不成立」會換成另一條規則決定結論。

一次判斷多張圖時，每張各寫一份，另在 `out/select/verdicts.md` 寫一張彙總表，欄位是圖名、結論、信心、決定規則。

**完成條件**：`verdict.md` 的每個欄位、表格的每一列都已填寫，結論和步驟 3、4 的規則一致。
