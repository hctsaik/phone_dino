# Real PM Recipe production artifact runbook

> Schema 1.6 changes MobileSAM from a compile-only dependency to a compile-and-runtime dependency so each aligned Current image can be segmented. See [Paired Current/Golden 主體內部比較](paired_current_subject_runtime.md). The schema-1.5 compile-only statements below describe the legacy compatibility path.

This is the handoff from `phone_cv` to local `phone_dino`. The downloaded DINOv2 model is not enough by itself; every digest below must describe the same Recipe state. ChArUco is preferred when visible, but it is not mandatory: target-only alignment may be enabled when the target reference, alignment regions, held-out regions, and transform gates pass.

## 1. Freeze the source evidence in phone_cv

For the target Recipe, ensure all of these are active before compiling:

- active BoardInstallation in state `ACTIVE`;
- active Golden set containing only approved `GOLDEN_GOOD` references;
- active Alignment Template with `inspectionRoi.status = CONFIRMED`;
- canonical target reference and alignment/held-out/inspection regions;
- approved model, normalization, decision, and blind-evaluation evidence IDs.

Record the exact `recipeId`, `machineId`, `boardId`, Golden-set digest, BoardInstallation digest, and every Golden source SHA-256. Do not copy these from the synthetic suite.

## 2. Build the artifact spec

Copy the values from `phone_cv` into a schema-1.5 build spec. The important fields are `goldenSources`, `goldenSetVersion`, `boardInstallationVersion`, `targetAlignment`, `inspectionRoi`, `subjectSegmentation`, `scorerInputContract`, `recipeAnalysisProfile`, and the DINO/MobileSAM model pins. Golden paths are resolved relative to the spec file. Older schemas remain readable only for their original compatibility scope and cannot claim the complete schema-1.5 scorer/profile identity.

Compute the local model pins:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$repoVersion = .venv\Scripts\phone-dino-digest-repository.exe .\runtime\models\dinov2
$weightsHash = (Get-FileHash .\runtime\models\dinov2_vits14_pretrain.pth -Algorithm SHA256).Hash.ToLower()
$subjectRepoVersion = .venv\Scripts\phone-dino-digest-repository.exe .\runtime\engineering-real-dino\mobile-sam-repository
$subjectWeightsHash = (Get-FileHash .\runtime\engineering-real-dino\mobile-sam-repository\weights\mobile_sam.pt -Algorithm SHA256).Hash.ToLower()
```

Compile offline:

```powershell
.venv\Scripts\phone-dino-compile-artifact.exe `
  .\PM-ABC-001-build-spec.json `
  .\PM-ABC-001-production-artifact.json `
  --model-repository .\runtime\models\dinov2 `
  --model-weights .\runtime\models\dinov2_vits14_pretrain.pth `
  --segmenter-repository .\runtime\engineering-real-dino\mobile-sam-repository `
  --segmenter-weights .\runtime\engineering-real-dino\mobile-sam-repository\weights\mobile_sam.pt `
  --segmenter-device cpu `
  --allow-target-only-alignment `
  --evidence-output .\PM-ABC-001-production-artifact.evidence.json
```

MobileSAM runs only in this offline compile step. It generates one canonical binary subject mask for each Golden and binds it to that Golden image plus the MobileSAM repository/weights digests. The runtime consumes those masks and does not load MobileSAM or segment Current captures. The compiler must succeed using real target-alignment evidence; ChArUco evidence is used when the board is visible but is not mandatory when target-only alignment is explicitly enabled and passes all gates. A synthetic artifact is only an engineering smoke test.

## 3. Verify production readiness

Configure `phone_dino` with:

```powershell
$env:PHONE_DINO_ENABLE_ENGINEERING_FIXTURES = "false"
$env:PHONE_DINO_ARTIFACT_MANIFEST = (Resolve-Path .\PM-ABC-001-production-artifact.json).Path
$env:PHONE_DINO_ARTIFACT_PACKAGE_DIGEST = "sha256:<digest of that exact JSON>"
$env:PHONE_DINO_MODEL_REPO = (Resolve-Path .\runtime\models\dinov2).Path
$env:PHONE_DINO_MODEL_WEIGHTS = (Resolve-Path .\runtime\models\dinov2_vits14_pretrain.pth).Path
$env:PHONE_DINO_MODEL_REPOSITORY_VERSION = $repoVersion
$env:PHONE_DINO_ALLOW_TARGET_ONLY_ALIGNMENT = "true"
```

`GET /readyz` must return `simulation: false`, `status: ready`, and subject-segmentation, ROI, scorer-input, and recipe-profile metadata matching the approved schema-1.5 artifact. Any missing model, artifact, digest, target reference, ROI contract, Golden subject mask, scorer contract, or profile pin is a fail-closed condition.

## 4. Activate through phone_cv

Upload the exact artifact bytes to `POST /api/v1/recipes/{recipeId}/production-artifact` as an admin with:

- `x-production-attestation: true`;
- `x-model-approval-id`;
- `x-normalization-approval-id`;
- `x-blind-evaluation-evidence-id`.

The server compares Recipe, active Golden, BoardInstallation, ROI, model, normalization, and decision pins. It must return `ARTIFACT_EXECUTION_BUNDLE_MISMATCH` rather than activating if any value differs.

Only after activation should `phone_cv` point `PHONE_DINO_URL` at this non-simulation service and restart. Recheck both `/api/v1/health/ready` and `/readyz`, then run one real-device capture and confirm the observation reports `simulation: false`, `analysis.state: RUN`, target-relative alignment evidence, `subjectSegmentationEvidence.state=AVAILABLE`, and `generationMethod=SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE`. PhoneCV must independently verify every evidence PNG digest and canonical dimension.

## Current state

The repository now contains the schema-1.8 engineering v18 artifact for `PM-ABC-001`, bound to Phone Dino 0.7.2 runtime digest `sha256:34d3750b4ea54cd6a00d92fea0d12e0a4c1b3ff177e0517878c531dcd73c0a2e` and artifact digest `sha256:ddfb5213d1efbce6aefee4d85efdf16273299a6f7fe4ed95b300eefbbcd7b637`. It retains the real DINOv2/MobileSAM evidence pipeline and adds fail-closed ChArUco multi-profile selection for Golden and Current dimensions. QR remains a deliberate User lookup aid and is not a measurement input. Subject-mask, ROI, scorer-input, and recipe-profile digests remain pinned across both services. The compiled Golden ID intentionally retains the V10 lineage; the active runtime/artifact identity is v18. It remains `ENGINEERING_AUTO` and `productionAuthorized=false`; this is not a signed production artifact.

## Automated snapshot from the current local phone_cv database

The active Recipe is `PM-ABC-001` / `MC-07` / `CB-001`.

- Golden set digest: `sha256:acb0762dd935d23502b66ead3e086e889265e02ed91a22cc471379dfb93087f6`
- Board installation digest: `sha256:6990c3291474d54663af4cfba944c14fef9c6fad6b289f8c8ebe084c95a047bc`
- Active alignment template: `8c6fb84f-7912-44b8-a29c-e8c1bfbd8ef5` (v7, confirmed ROI)
- Active Golden references currently contain the same source SHA (`sha256:9a24e6f6289d60152c2427ce35dc25e12e1ed1eb9627e2951a5e14d1c4d63f15`), so they are not a valid diverse physical Golden set for production.
- The stored Golden image has zero detectable `DICT_4X4_50` markers. This is acceptable only when target-only alignment is explicitly enabled and the target reference/held-out gates can align it independently; the v15 engineering path (using the same V10 Golden lineage) uses bounded `SUBJECT_CONTOUR_ECC_AFFINE` fallback and therefore remains `LIMITED`, not production-qualified.

The next real-world action is therefore a controlled dataset with sufficiently textured target references in the confirmed ROI, multiple normal recaptures, known abnormal cases, and angle/distance/lighting variation. A ChArUco board may be outside the frame; if so, the target reference and held-out alignment evidence must pass independently. After blind evaluation and approval, recompile a new immutable artifact with `approvalState=APPROVED`, then promote the PhoneDino artifact pin and PhoneCV profile pin together.

For alignment regression triage, replay real captures and deterministic generated controls without changing the artifact or thresholds:

```powershell
.\.venv\Scripts\python.exe scripts\replay_subject_alignment_cases.py `
  --artifact runtime\engineering-real-dino\engineering-real-dino-artifact-v15.json `
  --generated `
  --case recent-pass=C:\path\to\capture-pass.jpg `
  --case recent-fail=C:\path\to\capture-fail.jpg
```

The JSON-lines report separates alignment state, method, correlation/inlier ratio, held-out/reprojection residual, coverage, and reason codes. Generated inspection changes are expected to remain alignable so downstream anomaly observation can see them; missing targets must remain rejected. The generated perspective case is observation-only and requires Recipe-specific real-data calibration before an acceptance expectation is pinned.
