# Real PM Recipe production artifact runbook

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

Copy the values from `phone_cv` into a schema-1.1 build spec (or schema 1.2 after the consumer contract is enabled). The important fields are `goldenSources`, `goldenSetVersion`, `boardInstallationVersion`, `targetAlignment`, `inspectionRegions`, and the model pins. Golden paths are resolved relative to the spec file.

Compute the local model pins:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$repoVersion = .venv\Scripts\phone-dino-digest-repository.exe .\runtime\models\dinov2
$weightsHash = (Get-FileHash .\runtime\models\dinov2_vits14_pretrain.pth -Algorithm SHA256).Hash.ToLower()
```

Compile offline:

```powershell
.venv\Scripts\phone-dino-compile-artifact.exe `
  .\PM-ABC-001-build-spec.json `
  .\PM-ABC-001-production-artifact.json `
  --model-repository .\runtime\models\dinov2 `
  --model-weights .\runtime\models\dinov2_vits14_pretrain.pth `
  --allow-target-only-alignment `
  --evidence-output .\PM-ABC-001-production-artifact.evidence.json
```

The compiler must succeed using the real ChArUco captures. A synthetic artifact is only an engineering smoke test.

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

`GET /readyz` must return `simulation: false` and `status: ready`. Any missing model, artifact, digest, target reference, or ROI contract is a fail-closed condition.

## 4. Activate through phone_cv

Upload the exact artifact bytes to `POST /api/v1/recipes/{recipeId}/production-artifact` as an admin with:

- `x-production-attestation: true`;
- `x-model-approval-id`;
- `x-normalization-approval-id`;
- `x-blind-evaluation-evidence-id`.

The server compares Recipe, active Golden, BoardInstallation, ROI, model, normalization, and decision pins. It must return `ARTIFACT_EXECUTION_BUNDLE_MISMATCH` rather than activating if any value differs.

Only after activation should `phone_cv` point `PHONE_DINO_URL` at this non-simulation service and restart. Recheck both `/api/v1/health/ready` and `/readyz`, then run one real-device capture and confirm the observation reports `simulation: false`, `analysis.state: RUN`, and target-relative alignment evidence.

## Current state

The repository contains a verified synthetic artifact and local model smoke test, but not a real PM Recipe artifact. The remaining work is evidence collection and pin matching, not another model download.

## Automated snapshot from the current local phone_cv database

The active Recipe is `PM-ABC-001` / `MC-07` / `CB-001`.

- Golden set digest: `sha256:acb0762dd935d23502b66ead3e086e889265e02ed91a22cc471379dfb93087f6`
- Board installation digest: `sha256:6990c3291474d54663af4cfba944c14fef9c6fad6b289f8c8ebe084c95a047bc`
- Active alignment template: `bcf8a00a-2edd-4e53-8167-c5b2c0cd3be6` (v6)
- Active Golden references currently contain the same source SHA (`sha256:9a24e6f6289d60152c2427ce35dc25e12e1ed1eb9627e2951a5e14d1c4d63f15`), so they are not a valid diverse physical Golden set for production.
- The stored Golden image has zero detectable `DICT_4X4_50` markers. This is acceptable only when target-only alignment is explicitly enabled and the target reference/held-out gates can align it independently; the current stored template does not yet provide that verified canonical target contract.

The next real-world action is therefore one controlled Golden capture with a sufficiently textured target in the confirmed ROI. A ChArUco board may be outside the frame; if so, the target reference and held-out alignment evidence must be generated and pass independently. Once that evidence is stored, the remaining spec, compiler, digest, and readiness steps are automatable.
