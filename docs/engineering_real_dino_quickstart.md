# Engineering Real DINO 實作與操作紀錄

更新日期：2026-08-03

## 已接通的範圍

目前 `phone_cv -> phone_dino` 已使用本機真正的 DINOv2 ViT-S/14 權重執行分析，不再使用固定分數的 fixture：

- 模型輸出：384 維 CLS embedding。
- 空間差異：在 immutable Inspection ROI 上執行重疊 tiles，每個 tile 使用 16 x 16 patch token cosine distance。
- 全域距離與空間差異使用完全相同、固定順序的 ROI／subject-support tiles；支援外像素中性化，不會影響 scorer input 或距離。
- Golden artifact compiler 以真正的 MobileSAM ViT-T 和 ROI box prompt 產生 hash-bound subject mask；MobileSAM 不在 runtime 或 Current capture 上重新執行。
- Runtime 從 Golden subject mask 派生 support／boundary masks，並在 DINO 推論前把 support 外 Current 背景替換成 Golden 背景。
- Evidence：真正的 `SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE` full-canonical heatmap、raw threshold mask、retained candidate mask 與候選 regions。
- Evidence 在編碼前套用 immutable Golden support 與 Inspection ROI；兩者之外的 heatmap 與 mask 像素為零。
- ChArUco 可見時仍優先用於 plane normalization，但工程模式不強制它出現在畫面中。
- 黑色低紋理設備可在明確開啟的 engineering flag 下，以 Golden 綁定的 subject contour、coarse affine 與 ECC refinement 對位；response 明確回報 `SUBJECT_CONTOUR_ECC_AFFINE`，PhoneCV 將它標為 `LIMITED`。
- `phone_cv` 僅顯示 observation，不將結果轉成 PASS、FAIL 或機台放行。

這是「真實模型的工程分析」，但 `productionAuthorized` 固定為 `false`。SAM subject mask 只是背景抑制的空間依據，DINO heatmap／candidate mask 只是相對差異，兩者都不是 defect proof。輪廓 anchor、門檻與真實手機／光照資料尚未完成 blind qualification，因此不能冒充正式 production defect decision。完整設計見 [Golden 主體分割與背景抑制設計](golden_subject_segmentation_design.md)。

## 目前 immutable pins

| 項目 | 值 |
|---|---|
| Active alignment template | `PM-ABC-001` v7 / `8c6fb84f-7912-44b8-a29c-e8c1bfbd8ef5` |
| Compiled Golden ID | `GOLDEN-ACTIVE-V10-8C6FB84F` |
| Golden SHA-256 | `9fc72bdca285d6237af2b97400ec2469d662e7583ceb598ca59b177e663d2f67` |
| Model | DINOv2 ViT-S/14 |
| Weights SHA-256 | `sha256:b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| Model repository digest | `sha256:6f2d411cf095064c503259f7539f399ef6929059d58ca86230792ace634cd063` |
| Phone Dino release | `0.7.1` |
| Analyzer runtime digest | `sha256:ee726620f17c7fc8e730f3ba240f4687f2af89c3e5b1beb46bb8223952aec723` |
| Artifact schema | `1.8` |
| Artifact | `engineering-real-dino-artifact-v17.json` |
| Artifact digest | `sha256:7d1256e66a6be99c564b648d4b88dfc026e3215f3215862ff4b97bd12d8542ef` |
| Artifact ROI digest | `sha256:49e89b6adbc8202c2b79575e6e61b0b45601666e0fe0ee277691055ee6f67514` |
| Scorer input contract digest | `sha256:93e19e1f6cd0ec25c4011e7218eea8aa24e12cbcef9419646d5afe5abd936b66` |
| Recipe analysis profile digest | `sha256:4eb9b0823328728bbbb1898070903d989d5541c8deee9bfaab69bc2af2853def` |
| Golden scorer input SHA-256 | `sha256:7e9901b329018a81adf27fd3da1e528876796637ff71744dbfea15c8b06cf884` |
| Subject method | `MOBILE_SAM_VIT_T_BOX_PROMPT` |
| Subject mask SHA-256 | `sha256:073bfe5037a19b13cc072c617f7a2bd648e9be78d9b91b05bde400256db0905f` |
| MobileSAM repository digest | `sha256:d64ade5205e0d8d8ce9b958d750d1d712596da02fa2d13c8b0a1c0a89157f5c5` |
| MobileSAM weights SHA-256 | `sha256:6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` |

Artifact 與 provenance 位於 `runtime/engineering-real-dino/`。`engineering-real-dino-summary.json` 是供人閱讀的摘要，artifact JSON 與 `.evidence.json` 才是服務驗證的內容。

## 啟動

先在兩個 PowerShell session 設定相同的 `PHONE_DINO_SERVICE_TOKEN`（由本機 secret store 提供，不得寫入 repository），再啟動 DINO 服務：

```powershell
Set-Location C:\code\claude\phone_dino
.\start-engineering-real-dino.ps1
```

另一個 PowerShell 啟動 `phone_cv`：

```powershell
Set-Location C:\code\claude\phone_cv
.\start-engineering-phonecv.ps1
```

驗證兩個服務：

```powershell
Invoke-RestMethod http://127.0.0.1:8082/readyz
Invoke-RestMethod http://127.0.0.1:4174/api/v1/recipes/PM-ABC-001/engineering-dino-readiness
```

兩者都必須回報 `analysisMode: ENGINEERING_REAL_DINO` 與 ready。PhoneDino readiness 的 subject metadata 必須是 `MOBILE_SAM_VIT_T_BOX_PROMPT` 與上述 mask SHA，capabilities 必須包含 `GOLDEN_DIMENSION_BASELINE_V1`，並回報相同的 runtime、ROI、scorer-input 與 recipe-profile digests；PhoneCV profile 必須 pin 相同值。若任一 pin mismatch，代表兩個服務使用不同 artifact/profile；不要略過驗證，應以相同 v17 artifact 與 profile 重啟。PhoneDino 只有在模型與 runtime MobileSAM 載入、immutable Golden ROI patch cache 完整預算後才會 ready；任一 identity 不符都會 fail closed。

## 每次比對的輸出

每次 `POST /api/v1/recipes/PM-ABC-001/engineering-dino-analyses` 都建立一個獨立資料夾：

```text
phone_cv/runtime/private-blobs/engineering-dino-comparisons/
  PM-ABC-001/
    <request-id>/
      golden.jpg
      capture.jpg
      canonical-golden.png
      canonical-current.png
      request-manifest.json
      difference-heatmap.png
      candidate-mask.png
      raw-threshold-mask.png
      subject-mask.png
      support-mask.png
      boundary-mask.png
      result.json
```

資料夾只存在 Server private storage。Browser response 不回傳 `resultFolder`、blob key 或 filesystem path，而是使用 authenticated evidence URLs。`result.json` 不重複內嵌 base64 圖片。`raw-threshold-mask.png` 是 component filtering 前的候選；`candidate-mask.png` 只包含 retained candidates，並必須和 result 中 regions／candidate-filter counts 一致。

## 實際 E2E 證據

目前已用實拍 JPEG 完成 v10 HTTP 跨服務分析；v9、v8、v6、v4 僅保留為歷史基線：

| Request | Global cosine distance | Regions | Alignment | Comparison | Evidence |
|---|---:|---:|---|---|---|
| `efe4499b-3aca-4350-9a25-1680704bde3b`（v10） | `0.071138` | 6 | `SUBJECT_CONTOUR_ECC_AFFINE` | `LIMITED` | `SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE` |
| `86fd1766-0131-4c75-8cc9-6c48e4c4372c`（v9） | `0.071889` | 6 | `SUBJECT_CONTOUR_ECC_AFFINE` | `LIMITED` | `SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE` |
| `ba4b5412-26a4-4ec1-a59c-33b32688bb9d`（v8） | `0.071889` | 6 | `SUBJECT_CONTOUR_ECC_AFFINE` | `LIMITED` | `SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE` |
| `d4a22c15-0c72-4875-ad27-43e2392510c5`（v6） | `0.159948` | 8 | `CONTOUR_ANCHOR_AFFINE` | `LIMITED` | `SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE` |
| `3160072f-0054-43b3-92d6-08061c2c4c63`（v4 歷史基線） | `0.336849` | 7 | `CONTOUR_ANCHOR_AFFINE` | `LIMITED` | `ROI_TILED_PATCH_DISTANCE` |

v10 Server-private folder：`phone_cv/runtime/private-blobs/engineering-dino-comparisons/PM-ABC-001/efe4499b-3aca-4350-9a25-1680704bde3b`。

v10 第一次 process request 在未改動 60 秒 timeout 的條件下於 `7.076 s` 完成，且不需 retry。v9 的首次 request 曾超過 60 秒；修正後 startup 預算 Golden tiles，global scorer 產生的 Current patch embeddings 也直接由 spatial evidence 共用。

這次 v10 E2E 的 identity 稽核值：

- `scoringScope=INSPECTION_ROI_ONLY`，`evidenceCoordinateSpace=TARGET_CANONICAL_IMAGE`。
- Target canonical SHA：`4203b032d894dc3efb9e70713d102190adb569c03ca418ffc2b391e67d9e6207`。
- Current scorer input SHA：`f5f1c9b5b8da19d62ce9ab9d88d8ad86e8ae05cb4b038ff0214c7dcc90ad037d`；nearest normal scorer input SHA：`28c5c08aae1da0eacce5d3abf49af6eecaa680152831548d1d8ef5358bbe4e66`。
- 4 個 ordered scorer tile digests 與 analyzed region 均已持久化；canonical、heatmap、candidate、raw threshold、support、boundary evidence endpoints 全部回 200。

前一個 v8 E2E 的幾何／候選稽核值仍可用作歷史比較：

- Analysis ID：`1cb39dafc5575e3da9c43f6a1ac3fd204a21e2cdbc2607dfe42849fda40842fc`。
- Alignment：1,669 inliers、inlier ratio `0.858090`、coverage `0.836591`、held-out residual `5.730000 px`，transform bounds 通過。
- Subject mask pin：`sha256:073bfe5037a19b13cc072c617f7a2bd648e9be78d9b91b05bde400256db0905f`。
- Candidate filter：raw `33`、retained `6`、small-region suppressed `27`、limit suppressed `0`、verifier suppressed `0`；五項完整對帳且最終 regions 為 6。
- 六個候選都完成 `DINO_CROP_COSINE_V1` 二階段驗證；D001/D002 為 `HIGH`，D003–D006 為 `REVIEW`。`SHADOW` 不會 suppress 候選。
- 10/10 authenticated evidence assets 均已寫入 private folder；`result.json` 不含 base64 image persistence。
- 唯一的 `LIMITED` reason 是 `ENGINEERING_SUBJECT_ALIGNMENT_UNQUALIFIED`；subject segmentation 與 DINO spatial evidence 本身皆為 `AVAILABLE`。

v8、v6 與 v4 folders、舊 artifact digests 只保留供歷史稽核，不可拿來代表現行 schema-1.5 E2E。

分數不是缺陷門檻。下一個 production qualification 階段仍需收集同一 Recipe 的正常重拍、已知異常、角度、距離與光照資料，凍結 threshold 後做 blind evaluation；LightGlue 仍只能 feature flag／shadow benchmark。
