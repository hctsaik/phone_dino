# Paired Current/Golden 主體內部比較

更新日期：2026-08-03

## 現行流程

現行 `PM-ABC-001` 使用 PhoneDINO artifact schema `1.6`、wire schema `1.2`，PhoneCV profile schema `1.3`。流程固定如下：

1. 先完成 Current 到 target canonical coordinate space 的對位與品質 gate。
2. 對位合格後，使用 artifact 綁定的 MobileSAM repository、weights、Inspection ROI box prompt 分割 Current。
3. 將 Golden 與 Current 遮罩限制在相同 ROI，檢查 IoU、面積差與可用內部面積。
4. 建立 `interior = erode(golden_mask ∩ current_mask, 8 px)`。
5. Current 與 Golden 在 `interior` 外都填入相同的中性灰 `[127, 127, 127]`。
6. DINO global score、patch heatmap、connected components 與 crop verifier 全部只使用這組 paired-interior scorer inputs。
7. Golden／Current 的輪廓差異走獨立 boundary geometry evidence，不進入主要 DINO 候選。

因此，原本 Golden mask 向外擴張 24 px 所造成的背景滲入已從現行 schema-1.6 路徑移除；`supportPaddingPx=0`、`boundaryBandPx=0`，真正用於 DINO 的範圍還會再以 8 px 內縮共同遮罩。

## Fail-closed 順序

- 對位不合格：不執行 Current MobileSAM，也不執行 DINO，回傳 `RECAPTURE_REQUIRED + NOT_RUN`。
- Current MobileSAM 失敗：不執行 DINO，回傳 `CURRENT_SUBJECT_SEGMENTATION_FAILED`。
- mask IoU `< 0.85`、面積差 `> 0.15` 或 interior ratio `< 0.70`：不執行 DINO，回傳 paired-subject gate 原因。
- scorer/profile/model/artifact 任一 pin 不一致：readiness 或 request fail closed。

測試會記錄 segmenter 與 embedder 呼叫次數，確認上述 gate 不是只隱藏輸出，而是真的在 DINO inference 前停止。

## 邊界證據

`boundaryDifferenceEvidence` 單獨提供：

- Golden／Current mask IoU
- 面積差比例
- Golden 缺少於 Current 的區域（protruding）
- Current 缺少於 Golden 的區域（missing）
- 雙向輪廓平均、p95、最大位移（px）
- 邊界差異 mask 與 connected-component regions

主要 `spatialDifferenceEvidence.generationMethod` 是 `PAIRED_INTERIOR_ROI_TILED_PATCH_DISTANCE`，其 regions 只允許 `SUBJECT_INTERIOR`。PhoneCV 將 boundary evidence 另區顯示，不計入主要 DINO 候選數。

## 現行 pins

- Artifact：`engineering-real-dino-artifact-v12.json`
- Artifact package digest：`sha256:ee97ce5acfe46a4af902e957be970c289713aaa705cd6d05e22ce32d88b03e1c`
- Analyzer runtime：`sha256:072b92ffc0de212ab60e3887cc46dc553de95fe33bea1e33f77ea49aaed2bdb3`
- Scorer input contract：`sha256:93e19e1f6cd0ec25c4011e7218eea8aa24e12cbcef9419646d5afe5abd936b66`
- Recipe analysis profile：`sha256:4eb9b0823328728bbbb1898070903d989d5541c8deee9bfaab69bc2af2853def`
- MobileSAM repository：`sha256:d64ade5205e0d8d8ce9b958d750d1d712596da02fa2d13c8b0a1c0a89157f5c5`
- MobileSAM weights：`sha256:6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`

## SAM 3 的位置

這一版不需要先導入 SAM 3。眼前誤差主要來自 Golden-only mask、24 px 外擴和邊界混入 scorer，而不是 MobileSAM 完全無法辨識主體。先以 paired Current/Golden MobileSAM、共同內縮與 fail-closed gate 建立實拍基線；若之後仍有穩定的漏切／錯切，再用同一批 canonical 影像離線比較 MobileSAM 與 SAM 3，通過準確率、延遲、記憶體與模型 provenance gate 後才升級。

## 2026-08-03 實拍證據

- PhoneCV request `66e0bc09-46ed-4edd-bde6-368dc0116321` 使用同一張先前產生 7 個混合候選的手機 JPEG。
- Current／Golden mask IoU 為 `0.924344`，8 px 內縮後 interior ratio 為 `0.798207`。
- DINO 產生 25 個 raw components，21 個小區域被抑制，最後保留 4 個 `SUBJECT_INTERIOR`；沒有任何 boundary region 進入 DINO 候選。
- Boundary geometry 另外回傳 5 個 missing／protruding regions，p95 contour distance `9.55 px`，供人工核對但不宣稱缺陷。
