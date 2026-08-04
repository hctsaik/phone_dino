# Phone Dino / Phone CV 開發交接紀錄（2026-08-03）

## 0. 續作完成狀態（同日）

本文件原本記錄關機時的未完成快照；以下工作已在續作中全部完成。現行版本為 Phone Dino `0.6.1`、artifact schema `1.5`、immutable v10，並已由 Phone CV profile schema `1.2` 完整綁定：

- Artifact：`runtime/engineering-real-dino/engineering-real-dino-artifact-v10.json`
- Artifact digest：`sha256:a0c4b6b4dad09130c55b596094619a9246c113d559dd13420603fd503945c529`
- Analyzer runtime digest：`sha256:3549081e824b0f42a0b78af0bc9dbe1cc62e44901556a129e526c4118ba9bf7b`
- Inspection ROI digest：`sha256:49e89b6adbc8202c2b79575e6e61b0b45601666e0fe0ee277691055ee6f67514`
- Scorer input contract digest：`sha256:6524dc72c89a725696fc230597c094ed410b379ed5c7ada304bd37c1550ab7bb`
- Recipe analysis profile digest：`sha256:8e9adf26e02b477493d43f4b07e4e496af865868d4d72457c14fd23398325d86`
- Golden scorer input SHA-256：`sha256:28c5c08aae1da0eacce5d3abf49af6eecaa680152831548d1d8ef5358bbe4e66`

實拍 HTTP E2E 已成功：request `efe4499b-3aca-4350-9a25-1680704bde3b`、analysis `60390926769ba72b7179aa8dc017dfda13c24117255ac88de07f33b4b702f069`。這是 v10 process 啟動後第一筆實拍 request，在沿用 60 秒 timeout 下於 `7.076 s` 完成且不需 retry；v9 冷啟動基線曾超過 60 秒。結果為 `analysisState=RUN`、ROI-only global distance `0.0711379130`、`SUBJECT_CONTOUR_ECC_AFFINE`、1,651 inliers，且 `SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE` 為 `AVAILABLE`。四個 scorer tiles 的 Current patch embeddings 由 global 與 spatial evidence 共用，immutable Golden tile embeddings 則在 startup、ready 之前預算；34 個 raw components 經篩選後保留 6 個，六個皆為 `EVALUATED`。

驗證結果：Phone Dino `86 passed`；Phone CV Vitest `109 passed`、2 個 environment-gated tests 在一般 suite skip；相同兩項已由 `npm run test:pilot-infra` 使用真實 PostgreSQL 17 與 private S3-compatible MinIO 各 `1 passed`；Phone CV typecheck／production build 通過。既有 Playwright 基線為 `25 passed`、1 個 intentional viewport skip。現行結論仍是 engineering `LIMITED`／`productionAuthorized=false`，不可宣稱 production-qualified。

為使真實 DB 與 v10 artifact 的 source pins 一致，Phone CV active alignment-template pointer 維持已確認的 v7 template `8c6fb84f-7912-44b8-a29c-e8c1bfbd8ef5`；舊 active-pointer 狀態已備份於 `runtime/phone-cv-phase0.before-v7-reactivation-20260803.sqlite`（SHA-256 `b46b1089620a6386623b385e6dbc968d52ee77da1be6b99aab0415e7a423a614`），原 template 與影像資料均未刪除。

本輪使用者指定的剩餘項目 1（首次分析 timeout）與 4（正式 PostgreSQL／Pilot infrastructure integration）均已完成。尚未完成的是 production qualification：收集具多樣性的正常／已知異常實體資料、盲測、核准與正式 pin promotion；這不是本輪工程修正的遺漏項目。

## 1. 本次目標與目前狀態

使用者指定先完成：

1. 正式分層對位引擎。
2. 候選區第二階段判斷（原建議清單第 4 項）。

以下段落保留截至關機前的歷史狀態；當時尚未完成的是「最後一版對位修正的測試、重新編譯 artifact、真實照片 E2E、完整回歸及文件收尾」。這些項目目前已依第 0 節完成，但仍不可宣稱 production-qualified。

不要 reset、checkout 或清除兩個 repository。兩邊在本工作開始前就已有大量使用者修改及其他並行開發內容；所有 dirty files 都必須保留。

## 2. 已完成內容

### 2.1 分層對位

Phone Dino 已實作以下順序：

1. ORB / RANSAC target alignment 優先。
2. ORB 失敗後，engineering flag 開啟時才使用 Golden subject fallback。
3. fallback method 為 `SUBJECT_CONTOUR_ECC_AFFINE`。
4. LightGlue 仍只允許 feature flag / shadow benchmark，未參與正式判定。
5. ChArUco 不列為必要條件。

subject fallback 已包含：

- Golden subject mask boundary band。
- fit / held-out checker split。
- ECC correlation、殘餘平移、旋轉、scale、shear 上限。
- Golden strong-edge held-out residual 與 coverage 驗證。
- response 明確回報實際 alignment method，不冒充 ORB 或 ChArUco。

### 2.2 候選區第二階段判斷

Phone Dino 已加入 `DINO_CROP_COSINE_V1`：

- 每個第一階段 spatial candidate 都取 Current / Golden 對應 crop。
- 使用真實 DINO embedding cosine distance 做第二階段排序。
- priority 為 `HIGH`、`REVIEW`、`LOW`。
- 每個結果都帶 `CANDIDATE_VERIFICATION_NOT_DEFECT_PROOF`。
- `SHADOW` 模式只排序，不可抑制候選。
- 只有 artifact policy 為 `APPROVED + GATE` 才能抑制 `LOW`。
- candidate filter 新增 `suppressedByVerifierCount`，並有 exact accounting validation。

目前工程 artifact / profile 設定為：

- method：`DINO_CROP_COSINE_V1`
- mode：`SHADOW`
- approval：`ENGINEERING_AUTO`

因此它尚未用來做正式 defect decision，也不會偷偷刪除候選。

### 2.3 Artifact / contract

已完成 schema 1.4 的：

- `SubjectAlignmentContract`
- `CandidateVerificationPolicy`
- `ProductionArtifactV14`
- compiler / loader / readiness validation
- non-engineering approval gate

Phone Dino runtime version 已升為 `0.5.0`，目前 runtime digest：

`sha256:ef484fa7ac831d70f24939b5d27c6638034e2110ef566e4881d156a7d41f737c`

注意：工作樹內另有尚在整合中的 schema 1.5（ROI-only scorer input 與四種 reference role）修改。現在的 `scripts/build_engineering_real_artifact.py` 已會產出 schema 1.5，但目前啟動中的 v7 仍是 schema 1.4。下次不可直接假設 build script 還會產生 1.4；必須先把 schema 1.5 的 Phone Dino / Phone CV pins 與 tests 一起審核，或明確保留一條 1.4 compiler path。

### 2.4 Phone CV 串接與 UI

`C:\code\claude\phone_cv` 已加入：

- alignment method `SUBJECT_CONTOUR_ECC_AFFINE` contract validation。
- per-region candidate verification strict validation。
- SHADOW 不可 suppress、GATE 不可保留 LOW 的 invariants。
- readiness 的 subject alignment / candidate verification metadata 與 profile pins。
- `ENGINEERING_SUBJECT_ALIGNMENT_UNQUALIFIED` LIMITED reason。
- Review UI 顯示二階段 priority、crop distance、SHADOW disclaimer。
- 候選統計顯示 verifier suppression count。

已建立 / 更新 OpenSpec：

- `C:\code\claude\phone_cv\docs\openspec\subject-alignment-candidate-verification.md`
- `C:\code\claude\phone_cv\docs\openspec\controlled-pilot-golden-pipeline.md`

Phone CV 已通過的最近驗證：

- focused unit / integration：21 tests passed。
- `npm run build`：通過（typecheck、Vite build、artifact assertion）。

這些結果是在最後一版旋轉輪廓修正之前，但 Phone CV contract 本身沒有被該修正改動。

## 3. Active artifact 與服務設定

目前啟動腳本與 Phone CV profile 仍指向 immutable v7：

- artifact：`runtime/engineering-real-dino/engineering-real-dino-artifact-v7.json`
- artifact digest：`sha256:4a334eb8bdeb639793df5c0d10ee1ad33e95a32a5c62209615afd675daa6a4e3`
- schema：1.4
- Golden ID：`GOLDEN-ACTIVE-V7-8C6FB84F`
- Golden source SHA：`sha256:9fc72bdca285d6237af2b97400ec2469d662e7583ceb598ca59b177e663d2f67`
- subject mask SHA：`sha256:073bfe5037a19b13cc072c617f7a2bd648e9be78d9b91b05bde400256db0905f`
- DINO repository digest：`sha256:6f2d411cf095064c503259f7539f399ef6929059d58ca86230792ace634cd063`

設定位置：

- `C:\code\claude\phone_dino\start-engineering-real-dino.ps1`
- `C:\code\claude\phone_cv\config\engineering-dino-pm-abc-001.json`

原服務埠：

- Phone Dino：`127.0.0.1:8082`
- Phone CV server：`127.0.0.1:4174`
- Phone CV Vite：`127.0.0.1:4173`

關機後服務自然停止；下次需重新啟動。

## 4. 最後一次真實 E2E 結果（尚未通過）

測試照片：

`C:\code\claude\phone_cv\runtime\private-blobs\captures\fff56d2d-14da-4f20-ae86-b975408779ad\1\3405ff87f6a4...jpg`

使用 template id：

`8c6fb84f-7912-44b8-a29c-e8c1bfbd8ef5`

E2E request：

- request ID：`9d5d7c0e-ce5c-4cb2-9590-c6fd17f98014`
- analysis ID：`963ed5d...`
- response backup：`C:\code\claude\phone_cv\runtime\engineering-dino-v7-e2e-response.json`

HTTP 為 200，但結果為 `RECAPTURE_REQUIRED / NOT_RUN`：

- `TARGET_MATCHES_INSUFFICIENT`
- `SUBJECT_ALIGNMENT_HELD_OUT_RESIDUAL_HIGH`

alignment observation：

- method：`SUBJECT_CONTOUR_ECC_AFFINE`
- ECC ratio：`0.85795`
- held-out residual：`10.3786 px`
- coverage：`0.75767`
- transform bounds：true

結論：ORB 正確先失敗，subject fallback 有執行，但上方新增的藍色物體使舊的 axis-aligned contour bootstrap / ECC 驗證不夠穩定，因此尚未進入 DINO candidate verification 的真實 E2E 階段。

## 5. 關機前最後修改（尚未跑測試）

主要檔案：

`C:\code\claude\phone_dino\src\phone_dino\production.py`

已將 dark body bootstrap 改為：

- 使用 `cv2.minAreaRect`，保留 subject center、短邊、長邊、rotation。
- 以 Golden aspect similarity 為主要 scoring，避免上方新增物體改寫外框。
- 用旋轉矩形直接建立 Current-to-Golden coarse affine。
- 移除不穩定的 row / column dark occupancy trimming。

synthetic debug 中，7° rotation + 1.05 scale + 位移 + 上方新增藍色方塊時，新 coarse transform 與真實 inverse transform 的四角誤差已約 1 px。

隨後又加入：

- `_held_out_subject_edge_metrics`
- Golden strong edges（Canny 100 / 200）驗證。
- coarse contour hypothesis 與 ECC-refined hypothesis 的 held-out 雙假設選擇。
- ECC 只有在 held-out geometry 不比 coarse 差時才被採用。
- ECC 不收斂時，如果 coarse hypothesis 獨立通過，仍可安全採用 coarse；不會因 optional refinement 失敗而否決正確姿態。

最後這一版 patch 套用後尚未執行 pytest。關機後首先要驗證它。

另外修正了：

`C:\code\claude\phone_dino\src\phone_dino\contracts.py`

其中 Pydantic forward annotation 從錯誤的 quoted union 改成可解析的 `BboxNormalized | None`。這是 schema 1.5 並行修改暴露出的 collection error。

## 6. 下一次接續步驟（依序執行）

### Step 1：先測最後一版對位

```powershell
Set-Location C:\code\claude\phone_dino
.\.venv\Scripts\python.exe -m pytest tests\test_target_alignment.py -q
```

關機前上一次（雙假設 patch 之前）結果為 8 passed / 1 failed；唯一失敗為新增物體 synthetic case 的 held-out residual。最後 patch 的目的就是修正它。

若仍失敗，不可只放寬 threshold。先輸出並比較：

- detected rotated rectangle
- coarse transform / ECC residual transform
- coarse vs refined correlation
- coarse vs refined held-out residual / coverage
- 是否選到正確 hypothesis

### Step 2：跑 Phone Dino focused regression

```powershell
Set-Location C:\code\claude\phone_dino
.\.venv\Scripts\python.exe -m pytest `
  tests\test_target_alignment.py `
  tests\test_artifacts.py `
  tests\test_compiler.py `
  tests\test_spatial_evidence.py `
  tests\test_api.py -q
```

特別注意目前 artifact code 同時包含 V14 與 V15；`isinstance(artifact, ProductionArtifactV14)` 對 V15 也會是 true，但 readiness metadata、compiler evidence、Phone CV pins 必須逐項審核。

### Step 3：決定下一個 artifact 是 1.4 或 1.5

長期建議：完成 schema 1.5 integration 後，建立新的 immutable v8，不要覆寫 v7。

目前 build script 的必要來源：

- Golden：`runtime\engineering-real-dino\active-golden-v7.jpg`
- DINO repo：`runtime\models\dinov2`
- DINO weights：`runtime\models\dinov2_vits14_pretrain.pth`
- MobileSAM repo：`runtime\engineering-real-dino\mobile-sam-repository`
- MobileSAM weights：`runtime\engineering-real-dino\mobile-sam-repository\weights\mobile_sam.pt`
- ROI：`0.3001481282913154 0.20195141655449603 0.32007408713980406 0.5381111082430239`

建議命令骨架：

```powershell
Set-Location C:\code\claude\phone_dino
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe scripts\build_engineering_real_artifact.py `
  --golden runtime\engineering-real-dino\active-golden-v7.jpg `
  --output-dir runtime\engineering-real-dino `
  --golden-id GOLDEN-ACTIVE-V8-8C6FB84F `
  --recipe-id PM-ABC-001 `
  --machine-id MC-07 `
  --board-id CB-001 `
  --golden-set-version sha256:cfd0b95937bd1c53d1bb3689d074a53956fbd3e76df9a8863ef0fde60b396f20 `
  --normalization-version sha256:16d2b7f0e9588e1a8e5974f4a60c1aa712934153350cde860a1ae1548ad2c831 `
  --analyzer-model-version sha256:0c8a202ec5b6ca0e0ecd0fc6a47b83b7da99c42170cf783118dfbda566a99580 `
  --decision-policy-version sha256:65922a40be56be04dcd5f6f1729fa7f3fb67a512578a7d6d6759307929c9f287 `
  --board-installation-version sha256:6990c3291474d54663af4cfba944c14fef9c6fad6b289f8c8ebe084c95a047bc `
  --model-repository runtime\models\dinov2 `
  --model-weights runtime\models\dinov2_vits14_pretrain.pth `
  --segmenter-repository runtime\engineering-real-dino\mobile-sam-repository `
  --segmenter-weights runtime\engineering-real-dino\mobile-sam-repository\weights\mobile_sam.pt `
  --roi 0.3001481282913154 0.20195141655449603 0.32007408713980406 0.5381111082430239 `
  --artifact-name engineering-real-dino-artifact-v8.json
```

命令執行前先確認 `scripts/build_engineering_real_artifact.py` 的 `schemaVersion`。若為 1.5，Phone CV 必須能驗證 `recipeAnalysisProfileDigest`、ROI-only scorer input digest、reference roles；否則不要切換 active artifact。

### Step 4：更新雙邊 pin，再重啟

新 artifact 編譯完成後，從 summary / evidence 取得新 digest，更新：

1. `C:\code\claude\phone_dino\start-engineering-real-dino.ps1`
2. `C:\code\claude\phone_cv\config\engineering-dino-pm-abc-001.json`

不可只改檔名而漏改 digest。

重啟：

```powershell
# PowerShell 1
Set-Location C:\code\claude\phone_dino
.\start-engineering-real-dino.ps1

# PowerShell 2
Set-Location C:\code\claude\phone_cv
.\runtime\start-phonecv-dino-dev.ps1

# PowerShell 3
Set-Location C:\code\claude\phone_cv
.\runtime\start-phonecv-web.ps1
```

readiness：

```powershell
Invoke-RestMethod http://127.0.0.1:8082/readyz
Invoke-RestMethod http://127.0.0.1:4174/api/v1/recipes/PM-ABC-001/engineering-dino-readiness
```

### Step 5：重跑真實照片 E2E

使用最新 capture JPEG 與 template id `8c6fb84f-7912-44b8-a29c-e8c1bfbd8ef5`。

必須確認：

- `captureAssessment.state=ACCEPTED`
- `analysis.state=RUN`
- ORB 先執行；只有 ORB 不足才回報 `SUBJECT_CONTOUR_ECC_AFFINE`
- subject fallback 未核准時，Phone CV decision 為 `LIMITED`，reason 為 `ENGINEERING_SUBJECT_ALIGNMENT_UNQUALIFIED`
- `spatialDifferenceEvidence.state=AVAILABLE`
- 每個 subject-gated region 都有 `verification`
- verification method 為 `DINO_CROP_COSINE_V1`
- mode 為 `SHADOW`
- `suppressedByVerifierCount=0`
- priority / crop distance 能在 UI 顯示
- 不把 priority 說成 defect proof

每次比較的 private folder 應位於：

`C:\code\claude\phone_cv\runtime\private-blobs\engineering-dino-comparisons\PM-ABC-001\{requestId}`

要保留 response JSON、rectified images、overlay、DINO map、candidate mask、comparison summary，以及 subject masks。

### Step 6：完整回歸

Phone Dino：

```powershell
Set-Location C:\code\claude\phone_dino
.\.venv\Scripts\python.exe -m pytest -q
```

Phone CV：

```powershell
Set-Location C:\code\claude\phone_cv
npm test
npm run build
npx playwright test tests\e2e\engineering-result.spec.ts
```

若 Playwright 需要先建立 e2e build / fixture server，依 `package.json` 現有 script 啟動，不要臨時改掉 test contract。

### Step 7：文件收尾

真實 E2E 通過後更新：

Phone Dino：

- `docs\target_relative_alignment_design.md`
- `docs\real_production_artifact_runbook.md`
- `docs\engineering_real_dino_quickstart.md`
- `docs\golden_subject_segmentation_design.md`
- `docs\real_photo_comparison_findings.md`
- `README.md`

Phone CV：

- `docs\openspec\subject-alignment-candidate-verification.md`
- `docs\openspec\controlled-pilot-golden-pipeline.md`
- reviewer UI / E2E user guide（若畫面已改）

文件必須記錄 final artifact digest、request ID、alignment metrics、candidate verification mode、private result folder 與完整 test counts。

## 7. 尚未完成的明確清單

- [x] 執行最後一版 rotated contour + coarse/ECC dual hypothesis unit test。
- [x] 修正任何剩餘 alignment false reject，但不可只放寬 threshold。
- [x] 審核 schema 1.5 並行修改是否已能端到端使用。
- [x] 跑 Phone Dino focused regression。
- [x] 建立 immutable v8 artifact；不可覆寫 v7。
- [x] 更新 Phone Dino / Phone CV artifact pins 與 digest。
- [x] 重啟並確認雙邊 readiness metadata 完整一致。
- [x] 使用最新真實照片跑 E2E，確定對位成功。
- [x] 確定真實 DINO spatial evidence 已執行。
- [x] 確定所有候選都有二階段 DINO crop verification。
- [x] 確定 SHADOW 不 suppress，UI 只顯示 review priority。
- [x] 確定每次比較都建立獨立 folder 並保存所有 evidence。
- [x] 跑 Phone Dino 全測試。
- [x] 跑 Phone CV 全測試、build、Playwright。
- [x] 更新最終實作與真實照片結果文件。

## 8. 安全與長期設計原則

- ChArUco 可用但不必要。
- ORB 是第一層；subject contour/ECC 是 fallback。
- LightGlue 只能 shadow benchmark，未經 qualification 不可正式 gate。
- SAM / MobileSAM 只界定 Golden subject scope，不判斷 defect。
- DINO crop verifier 目前只做候選排序，不是 defect classifier。
- 對位品質與 anomaly score 必須分開；對位失敗不可硬跑 defect decision。
- 新增物體可以改變 subject silhouette；held-out validation 必須容忍局部 anomaly，但仍要拒絕主體缺失或錯物。
- 所有 policy、mask、model、ROI、Golden、runtime 與 artifact 必須 digest-bound。
- 每次結果需可追溯、可重現、存入獨立 folder。
- Engineering result 必須明確標示 LIMITED / not defect proof，不可冒充 production verdict。
