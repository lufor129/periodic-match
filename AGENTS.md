# periodic-match

週期性 SEM/TEM 影像的 Instance 偵測：Lattice-first（`src/periodic_match/lattice.py`）對照 Template Matching（ZNCC）與 DINOv3。

- 用詞以 `GLOSSARY.md` 為準。
- 公司影像與其輸出只放 `data/`、`out/`（已 gitignore）；commit 前用 `git status` 確認沒有影像或座標表。
- 在公司真實影像上試跑或評估 Lattice-first：照 `docs/company-trial.md` 做。
- 判斷一張影像該用 Template Matching 還是 Lattice-first：照 `docs/method-selection.md` 做。
- 改 `lattice.py` 後，用 `uv run python scripts/run_benchmark.py --quick --methods ncc,lattice` 確認結果和 `results/final/summary.md` 相近（final 是 3 seeds 完整版）。
