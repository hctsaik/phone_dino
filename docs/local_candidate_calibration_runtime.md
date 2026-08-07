# 局部候選校正與結構確認

> 目前 active engineering pins 已升級為 PhoneDINO `0.7.6`、artifact schema
> `1.8`、wire schema `1.5`、PhoneCV profile `1.6`、
> `engineering-real-dino-artifact-v21.json`。候選確認方法仍為
> `DINO_CROP_COSINE_LOCAL_STRUCTURE_V2`；新增尺寸能力詳見
> [Calibration + Segmentation physical dimension runtime](physical_dimension_measurement.md)。

更新日期：2026-08-04

## 執行順序

PhoneDINO 先完成 target canonical 對位、Current MobileSAM 分割與 paired
interior 建立，才對每個 `SUBJECT_INTERIOR` 候選執行下列第二階段：

1. 以候選周圍的共同 interior 為局部對位 context，並排除候選本身及
   5 px 外圈，避免真實變化拉動對位。
2. 使用 `GRADIENT_ECC_TRANSLATION_V1` 估計局部 X/Y 平移。相關係數必須
   至少 0.45，且各軸位移不得超過 6 px。
3. 對位合格後，以 `OPENCV_LAB_CONTEXT_MEDIAN_MAD_V1` 將 Current 的局部
   光度映射到 Golden context。
4. 在候選框內計算正規化 Lab 變化面積與 Canny 邊緣變化面積。
5. Lab delta 大於 0.12 的面積比例至少 0.30，或邊緣變化面積比例至少
   0.15，才回報 `structureConfirmation=CONFIRMED`。

局部對位不合格時不產生外觀／邊緣比例，也不能確認候選。此閘門只影響
候選呈現與工程優先順序，不是 defect、PASS/FAIL 或製造動作判定。

## 版本與 pins

- PhoneDINO：`0.6.3`
- Artifact schema：`1.7`
- Wire schema：`1.3`
- PhoneCV profile schema：`1.4`
- Artifact：`engineering-real-dino-artifact-v13.json`
- Artifact digest：`sha256:752bc2b95003f3edd40680f7ef0bd78b2d78635e1f3ffa3093d768ccabf1596c`
- Runtime digest：`sha256:96b782ab77b44e63e1396c48394acf8857e490e623c4643a475d54cad409633c`

## D-003／D-004 回放

PhoneCV request `9c5ff663-0bc0-4908-940c-1a6826f4f610` 使用原負案例 JPEG
通過真實 8082 HTTP 路徑：

- D-001：DINO 0.445、局部位移 (-0.49, -1.60) px、Lab 變化面積
  33.2%，`CONFIRMED`。
- D-002：DINO 0.314、局部位移 (2.84, -4.15) px、Lab 變化面積
  34.0%，`CONFIRMED`。
- D-003：DINO 0.280，但局部 X 位移 -11.50 px 超界，
  `LOCAL_ALIGNMENT_UNQUALIFIED`，只進工程明細。
- D-004：DINO 0.226、Lab 變化面積 0.5%、邊緣變化面積 5.2%，
  `UNCONFIRMED`，不進主要畫面。
- D-005：局部結構雖確認，但 DINO 僅 0.127，仍不進主要畫面。

最終主要人工複核位置為 D-001 與 D-002。完整五筆 evidence 仍保留，因
artifact 目前是 `SHADOW + ENGINEERING_AUTO`，尚未宣稱 Recipe 生產核准。
