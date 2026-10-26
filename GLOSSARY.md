# Image Matching

在半導體 SEM/TEM 週期性灰階影像上，以小 FoV 參考圖在大 FoV 影像中框出所有對應區域。

## Language

**Unit Cell**:
影像中最小的重複結構單元（週期的基本單位）。
_Avoid_: cell, pattern unit

**Reference**:
小 FoV 的參考影像，內含數個 Unit Cell，定義要尋找的外觀與大小。
_Avoid_: template, query, patch

**Search Image**:
要在其中尋找 Instance 的大 FoV 影像，可能與 Reference 來自不同倍率或機台。
_Avoid_: target, big image, source

**Instance**:
Search Image 中與 Reference 涵蓋相同數量與排列之 Unit Cell 的區域，以一個 BBox 表示；同一張 Search Image 的 Instance 彼此不重疊。
_Avoid_: match, hit, detection

**Phase**:
Reference 邊界相對於 Unit Cell 晶格的偏移；所有 Instance 須與 Reference 的 Phase 一致。若結構本身在某個非整數 Unit Cell 的平移下不變（例如六角陣列的置中平移），平移後的位置也屬同一 Phase。
_Avoid_: offset, alignment

**Instance Grid**:
一塊 Periodic Region 內依 Reference 的 Phase 排列、彼此不重疊的 Instance 格狀排列；相鄰關係固定，但每個 Instance 可局部跟隨實際結構（drift、形變）。每塊 Periodic Region 各自擁有一個 Instance Grid。整個格子平移整數個 Unit Cell 仍屬同一 Phase，此時取 Instance 數量最多的排法。
_Avoid_: tiling, mesh

**Periodic Region**:
Search Image 中由 Reference 所含結構週期性重複構成的區域；Instance 只存在於 Periodic Region 內且須完整落於影像內。
_Avoid_: valid area, ROI

**Anomalous Instance**:
位置符合 Instance Grid、但外觀與 Reference 差異明顯（defect、製程變異、微粒遮擋）的 Instance；仍需輸出並標記。
_Avoid_: defect instance, outlier, bad match
