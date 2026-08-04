# Target-relative alignment 設計修正

> 日期：2026-08-01  
> 原因：現場不保證 ChArUco board 與受檢設備維持固定相對位置。

## 決策

ChArUco 只負責相機姿態、尺度與 board plane 的正規化，不再被當作設備位置基準。DINO 只能分析經過獨立 target localization/alignment 的 canonical target ROI。

```text
Raw Still
  -> decode / Still Gate
  -> ChArUco plane normalization
  -> target localization and coarse alignment
  -> bounded target-relative residual alignment
  -> Alignment Gate
  -> canonical target ROI
  -> DINO / Golden observation
```

任何階段缺少可信證據時都回 `RECAPTURE_REQUIRED + NOT_RUN`，不得執行 DINO，也不得產生可被 `phone_cv` 投影為 PASS 的 observation。

## Artifact 1.1

Production artifact 必須 immutable pin：

- target reference image 與 digest；
- target canonical width/height；
- fit `alignmentMask`、獨立 `heldOutMask` 與 `inspectionMask`；三者必須非空、範圍合法且兩兩互斥；
- alignment method/version；
- minimum match/inlier count、inlier ratio、spatial coverage；
- fit 與 held-out P95 reprojection error、translation、rotation、scale/shear 等 correction bounds；
- secondary-transform support 上限，用於拒絕 duplicate/ambiguous target；
- Golden embeddings，以及建立 Golden 時所用的相同 target-relative pipeline 版本。

舊版 board-relative artifact 不可自動升級，也不可通過 production readiness；必須重新編譯、重新驗證並取得新的 `artifactPackageDigest`。

現行 schema 1.5 在此幾何契約之上再綁定 target-canonical Inspection ROI scorer contract 與四角色 RecipeAnalysisProfile。AlignmentTemplate 決定如何對位，TargetReference 決定 canonical target，NormalReferenceSet 決定距離基準，DisplayReference 只供 UI 視覺比較；四者不可再由單一「Golden」名稱隱式混用。全域與 spatial DINO evidence 必須使用相同 ROI tiles，並保存 scorer input identity。

## Runtime contract

`captureAssessment=ACCEPTED` 時，`normalization.alignment` 必須存在：

```json
{
  "state": "ALIGNED",
  "method": "TARGET_AFFINE",
  "targetRelative": true,
  "inlierCount": 24,
  "inlierRatio": 0.8,
  "reprojectionErrorPx": 0.9,
  "coverageRatio": 0.65,
  "transformWithinBounds": true,
  "inspectionMaskApplied": true
}
```

允許的方法為 `TARGET_HOMOGRAPHY`、`TARGET_AFFINE`、`LIGHTGLUE_RESIDUAL`，以及明確限於工程模式的 `CONTOUR_ANCHOR_AFFINE`、`SUBJECT_CONTOUR_ECC_AFFINE` 或 `SIMULATION_FIXTURE`。Production 不得把後三者宣稱為 qualified target alignment；PhoneCV 必須將 contour fallback 標為 `LIMITED`。

`canonicalSha256` 必須代表 target-aligned ROI，而不是 board canonical canvas。`phone_cv` 必須嚴格驗證 alignment 欄位、數值範圍與狀態一致性後，才可套用 DecisionPolicy。

## LightGlue 邊界

LightGlue 是 pluggable residual aligner，不是 unrestricted image warp：

- keypoints/correspondences 只能來自 immutable `alignmentMask`；
- `inspectionMask`、可能缺件區、反光區、board 與可動零件不得參與 matching；
- transformation 必須通過 correction bounds 與 spatial coverage gate；
- repeated texture、clustered matches、secondary coherent transform、held-out parallax/non-planar residual 或 ambiguous target 一律 fail closed；
- blind benchmark 必須證明不會把缺件、位移或形變對齊消除，才可將 `LIGHTGLUE_RESIDUAL` 從 shadow 提升為 production method。

在 LightGlue 通過上述門檻前，可使用相同 contract 的 deterministic、offline target aligner 建立 pipeline 與測試，但不得據此宣稱 physical CV 已驗證。

## Acceptance tests

- board-only translation/rotation 不得改變 target canonical ROI。
- target 相對 board 的允許範圍位移/旋轉/尺度仍可正確對位。
- target absent、duplicate、partial、超出 correction bounds 必須 `RECAPTURE_REQUIRED + NOT_RUN`。
- correspondences 只落在 inspection/defect mask 或集中在小區域時必須拒絕。
- 缺件或被移除的 component 不得藉由 alignment 被補償或隱藏。
- board 與 target 不同深度造成 parallax 時必須拒絕單一平面 transform。
- NaN、Infinity、負數、越界 metrics 或缺少 alignment proof 必須被 producer/consumer contract 拒絕。
- Golden compiler 與 runtime 必須使用相同 target-relative normalization，並記錄 canonical/mask/alignment digests。

## Release boundary

完成程式與 synthetic regression 只代表 engineering integration 可接受。Physical CV、LightGlue production enablement 與 Controlled Pilot 仍需真實設備、手機矩陣、光照、Golden/blind dataset 與 zero unsafe-accept evidence。

## 2026-08-01 實作與驗收

- 已完成 strict artifact schema 1.1；舊 1.0 artifact 不自動遷移並在 readiness fail closed。
- 已完成 ChArUco plane normalizer、可插拔 `TargetAligner`，以及 bounded ORB/RANSAC affine baseline。
- 已實作 translation、rotation、scale、coverage、inlier、P95 reprojection、secondary ambiguity 與 held-out/parallax gates。
- 已完成 producer/consumer strict alignment contract；DINO 在 alignment 未通過時不會執行。
- `phone_dino` production-vision tests 32/32 通過；`phone_cv` tests 53/53 通過。
- engineering E2E 與 `simulation:false` production contract E2E 各 1/1 通過。
- Multi-agent 獨立終審：`ACCEPT`，0 blocker、0 major；此結論只適用 engineering increment，不代表 physical CV/Pilot 核准。

## 2026-08-03 Golden subject gate 補充

Target alignment 與 subject segmentation 是兩個不同問題。前者把 Current 安全地映射到 Golden canonical coordinates；後者只定義該 canonical Golden 中的設備空間。Artifact schema 1.5 保留此 gate，並在 alignment 成功後才套用 hash-bound Golden subject/support/boundary masks，在 DINO 推論前以 Golden pixels 替換 support 外的 Current 背景。

`SUBJECT_CONTOUR_ECC_AFFINE` 的 held-out residual 只允許使用貼近 immutable Golden subject-mask boundary 的 Golden Canny 強邊。`alignmentBandPx` 定義的是 Current 邊緣可被搜尋的寬帶，不代表該寬帶內的螢幕、桌面、線材或其他背景強邊可以成為 Golden 驗證樣本。這個區分避免背景變化被誤報成 subject pose error，同時保留既有 correlation、residual、coverage 與 transform bounds fail-closed gates。

MobileSAM 只在 Golden artifact compiler 執行，不參與 Current 對位，也不允許用每張 Current 的 segmentation 改寫 inspection support。若 alignment gate 失敗，流程仍是 `RECAPTURE_REQUIRED + NOT_RUN`，不可因為 SAM 有輸出而繞過。LightGlue 的狀態也不變：僅能 feature flag／shadow benchmark。

完整 scorer contract 見 [Golden 主體分割與背景抑制設計](golden_subject_segmentation_design.md)；現行 runtime/artifact pin 為 PhoneDINO 0.7.2、schema 1.8、v18，Golden ID 仍保留 V10 lineage。

## 2026-08-03 schema 1.5／v10 驗證補充

現行工程 v15 artifact（沿用 V10 Golden lineage）使用分層策略：ORB/RANSAC 優先；失敗時才啟用 rotated subject contour、coarse affine 與 ECC dual-hypothesis refinement。實拍 request `efe4499b-3aca-4350-9a25-1680704bde3b` 是 v10 的歷史 E2E 證據，成功回報 `SUBJECT_CONTOUR_ECC_AFFINE`、1,651 inliers，並輸出 `INSPECTION_ROI_ONLY` 分數與 target-canonical evidence identity。方法仍為 engineering-only，所以 comparison 正確維持 `LIMITED`。

Artifact schema 1.5 將 AlignmentTemplate、TargetReference、NormalReferenceSet、DisplayReference 四種角色及其 recipe analysis profile 一併 digest-bind。PhoneCV profile schema 1.2 在送出 analyzer schema-1.1 request 前驗證這些 pins，回應也再次驗證 ROI、scorer contract、profile 與 evidence tile identity；任一不符即 fail closed。
