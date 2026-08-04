# phone_dino

## Physical dimensions (artifact 1.8 / wire 1.4)

The engineering runtime can now combine this capture's ChArUco plane scale with
the Current MobileSAM mask to report rotated-rectangle `lengthMm`, `widthMm`,
mask `areaMm2`, and a conservative uncertainty estimate. The target alignment
transform is inverted before measurement; target-canonical pixels are never
treated as physical pixels directly. Missing calibration, unqualified
alignment/segmentation, invalid contour geometry, or excessive uncertainty
returns explicit `UNAVAILABLE` evidence without any mm values. See
[Calibration + Segmentation physical dimension runtime](docs/physical_dimension_measurement.md).

The active engineering pins are PhoneDINO `0.7.2`, artifact schema `1.8`, wire
schema `1.4`, and PhoneCV profile schema `1.5`. Physical dimensions remain
engineering evidence until a locked known-size real-device dataset is approved.

## Paired Current/Golden subject runtime (schema 1.8)

The active engineering artifact is schema `1.8` and the analyzer wire response is schema `1.4`. After target-canonical alignment passes, runtime MobileSAM segments the Current image with the same pinned repository, weights, ROI box prompt, and canonical coordinate space used for Golden. It then computes `interior = erode(golden_mask & current_mask, 8 px)`. Both Current and Golden DINO inputs use the same neutral RGB value outside that interior, so background and boundary pixels cannot contribute to the primary DINO score or candidates.

Alignment failure stops before MobileSAM and DINO. Current-mask or paired-interior quality failure stops before DINO and returns `RECAPTURE_REQUIRED + NOT_RUN`. Boundary geometry is reported separately as mask IoU, area delta, missing/protruding regions, and contour distance; it is never emitted as a primary `SUBJECT_BOUNDARY` DINO candidate. See [Paired Current/Golden subject runtime](docs/paired_current_subject_runtime.md).

`phone_dino` is the internal, fail-closed image observation service used by `phone_cv`. It never returns a manufacturing action or a PASS/FAIL decision. `phone_cv` owns its versioned decision policy.

The service includes strict multipart contracts, service-token authentication, content and bundle digest verification, bounded JPEG/PNG decoding, structured observations, and an explicitly enabled engineering fixture analyzer. The production path loads one immutable, content-addressed artifact, applies EXIF-safe Still Gate, uses ChArUco only for camera/plane normalization when it is available, then independently aligns the inspected target before a local-only DINOv2 ViT-S/14 comparison. LightGlue remains outside the controlled-pilot baseline; the offline baseline is bounded ORB/RANSAC affine alignment behind a pluggable target-aligner interface. Beyond the global embedding distance, an optional patch-level comparison can localize *where* a capture differs from Golden as a heatmap, binary mask, and bounding-box regions (see "Patch-level spatial difference evidence" below) — honestly reported as `UNAVAILABLE` rather than fabricated whenever the pinned artifact or Golden doesn't actually support it. Artifact schema 1.5 additionally binds a Golden subject mask compiled offline by MobileSAM, an inspection-ROI-only scorer input contract, and four hash-bound Golden reference roles; see [Golden 主體分割與背景抑制設計](docs/golden_subject_segmentation_design.md).

## Local engineering mode

要直接執行目前已綁定 `PM-ABC-001` Active Golden 的真實 DINOv2 工程流程，請參考 [Engineering Real DINO 實作與操作紀錄](docs/engineering_real_dino_quickstart.md)。以下 fixture 模式只用於 deterministic contract 測試。

現行工程 pin 是 PhoneDINO `0.7.2`、artifact schema `1.8` 的 `engineering-real-dino-artifact-v18.json`，package digest `sha256:ddfb5213d1efbce6aefee4d85efdf16273299a6f7fe4ed95b300eefbbcd7b637`。Runtime 與 PhoneCV 會同時驗證 analyzer runtime digest `sha256:34d3750b4ea54cd6a00d92fea0d12e0a4c1b3ff177e0517878c531dcd73c0a2e`、paired-current subject contract、ROI-only scorer contract、Golden dimension baseline capability 和四角色 RecipeAnalysisProfile；尺寸校正會以 ChArUco 幾何辨識小／大板，QR 不參與量測；任一 identity 不符皆 fail closed。

Use Python 3.11 or 3.12:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
$env:PHONE_DINO_SERVICE_TOKEN = "replace-for-local-use"
$env:PHONE_DINO_ENABLE_ENGINEERING_FIXTURES = "true"
$env:PHONE_DINO_FIXTURE_DIR = "C:\path\to\fixtures"
.venv\Scripts\uvicorn phone_dino.app:app --host 127.0.0.1 --port 8080
```

Fixtures are selected only by the SHA-256 of the actual uploaded bytes. For image hash `<hash>`, create `<PHONE_DINO_FIXTURE_DIR>/<hash>.json`:

```json
{
  "outcome": "review",
  "globalDistance": 0.42,
  "nearestGoldenDistance": 0.40,
  "uncertainty": 0.25,
  "reasonCodes": []
}
```

`outcome` may be `pass`, `fail`, `review`, `recapture`, or `system-error`. These labels only select deterministic engineering behavior: for pass/fail/review the service returns a distance and `simulation: true`, leaving policy classification to `phone_cv`. Recapture returns `analysis.state=NOT_RUN`; system-error returns HTTP 503.

For the checked-out `phone_cv` E2E images, prepare all five records reproducibly:

```powershell
.venv\Scripts\phone-dino-prepare-fixtures C:\code\claude\phone_cv\public\fixtures .\.fixtures
$env:PHONE_DINO_FIXTURE_DIR = (Resolve-Path .\.fixtures)
```

## Contract and safety behavior

- `GET /healthz` is liveness only.
- `GET /readyz` is ready in engineering mode only when the token and fixture directory exist.
- `POST /internal/v1/analyze` requires `Authorization: Bearer ...`, a bounded `application/json` file part named `manifest`, and an `image` file part. Expired deadlines are rejected before decode.
- The manifest is strict: unknown fields, unknown enum values, non-finite values, and unsupported schema versions are rejected.
- `rawSha256` is checked against actual bytes. `executionBundleDigest` is SHA-256 of canonical JSON (sorted keys, UTF-8, no whitespace) for `executionBundle`.
- `artifactPackageDigest` is a separate content address for the immutable vision artifact package selected by the Recipe. It must not be copied from, or treated as an alias of, `executionBundleDigest`; both identities are echoed independently in `resolvedVersions`.
- Upload bytes, width, height, and decoded pixel count are bounded by `PHONE_DINO_MAX_IMAGE_*` settings.
- Client-provided paths, URLs, masks, model choices, thresholds, and decisions are not accepted.

Production mode requires `PHONE_DINO_ARTIFACT_MANIFEST` and `PHONE_DINO_ARTIFACT_PACKAGE_DIGEST` (the digest of that exact file). The configured package digest is the service-side pin and must equal the independent `artifactPackageDigest` sent by `phone_cv`. Artifact schema 1.5 strictly binds Recipe, Golden set, normalization pipeline, analyzer model, board installation, reviewed model-weights digest, ChArUco geometry, Still Gate limits, a hash-bound target reference, stable alignment regions, excluded inspection regions, transform gates, a separately digested inspection-ROI contract, immutable Golden embeddings, one hash-bound Golden subject mask per embedding, the scorer input contract, and the recipe analysis profile. Every request must match those pins. Schema 1.2–1.4 remain readable for their original compatibility scope, but they cannot claim the complete schema-1.5 scorer/profile identity. A mismatch, tamper, missing dependency, invalid alignment, invalid embedding, invalid subject mask, or scorer/profile pin mismatch fails closed.

The ChArUco board is never treated as the target position. Its homography produces a plane-normalized camera view only. A separate target aligner locates that target relative to the Golden reference, excludes every `inspectionRegion` from reference feature extraction, checks inlier count/distribution, reprojection error, scale, rotation and shear, and emits the final canonical target ROI. DINO receives only that final ROI. Missing or unsafe target evidence returns `RECAPTURE_REQUIRED` with `analysis.state=NOT_RUN`; the response's `normalization.alignment` contains quantitative evidence and `canonicalSha256` exists only for the final target ROI.

Set `PHONE_DINO_MODEL_REPO` to an approved local DINOv2 checkout and `PHONE_DINO_MODEL_WEIGHTS` to the approved local weights file. The adapter uses `torch.hub` with `source="local"` and never downloads at runtime. Vision libraries remain an explicit installation extra because PyTorch builds are platform-specific:

```powershell
.venv\Scripts\python -m pip install -e ".[vision]"
.venv\Scripts\phone-dino-production-preflight
```

MobileSAM is compiler-only for legacy schema-1.5 artifacts. Schema 1.6 also requires the pinned MobileSAM repository and weights at runtime to segment each aligned Current image. Install `.[segmentation]` on both the artifact-build host and the schema-1.6 analyzer host, and set `PHONE_DINO_SUBJECT_SEGMENTER_REPO`, `PHONE_DINO_SUBJECT_SEGMENTER_WEIGHTS`, and `PHONE_DINO_SUBJECT_SEGMENTER_DEVICE`.

Preflight refuses engineering-fixture mode and exits non-zero unless the full artifact/model/dependency chain is ready. Model and artifact presence are necessary but do not constitute physical qualification: release still requires a locked real-device target-alignment dataset (with ChArUco evidence when the board is visible), approved weights/repository provenance, blind evaluation, and signed operational evidence. ChArUco is not mandatory when the independently pinned target-only alignment gates pass.

## Compile an approved production artifact

Compilation is offline and refuses to overwrite outputs. First content-address the reviewed local DINOv2
source tree (VCS/cache metadata is excluded; symlinks are rejected), then place that digest in the build
spec as `modelRepositoryVersion`:

```powershell
$repositoryVersion = .venv\Scripts\phone-dino-digest-repository C:\approved\dinov2
.venv\Scripts\phone-dino-compile-artifact `
  C:\approved\PM-ABC-001-build-spec.json `
  C:\approved\PM-ABC-001-production-artifact.json `
  --model-repository C:\approved\dinov2 `
  --model-weights C:\approved\dinov2_vits14.pth `
  --segmenter-repository C:\approved\MobileSAM `
  --segmenter-weights C:\approved\mobile_sam.pt
```

The build spec uses the artifact fields below, but replaces `goldenEmbeddings` with `goldenSources`:

```json
"goldenSources": [
  {"id": "GOLDEN-001", "path": "golden/GOLDEN-001.jpg", "sourceSha256": "sha256:<64 hex>"}
]
```

Paths are resolved relative to the build-spec file. The compiler verifies source and weights hashes,
repository-tree digest, image bounds, ChArUco/Still Gate, target-relative alignment, finite equal-dimension embeddings, Golden subject-mask quality/bounds/hash, and writes
canonical JSON plus a separate `.evidence.json` report. A hash mismatch or recapture-required Golden
produces no artifact. Preserve both outputs with the blind-evaluation evidence.

The production artifact schema is strict. The compact example below shows the legacy 1.1 alignment fields (digests and embeddings abbreviated here only; real files require complete SHA-256 values). A current subject-gated build uses schema 1.5, includes `inspectionRoi` and `subjectSegmentation`, and also binds the scorer input and recipe analysis profile documented in [Golden 主體分割與背景抑制設計](docs/golden_subject_segmentation_design.md).

```json
{
  "schemaVersion": "1.1",
  "recipeId": "PM-ABC-001",
  "machineId": "MC-07",
  "boardId": "CB-001",
  "goldenSetVersion": "sha256:<64 hex>",
  "normalizationPipelineVersion": "sha256:<64 hex>",
  "analyzerModelVersion": "sha256:<64 hex>",
  "decisionPolicyVersion": "sha256:<64 hex>",
  "analyzerRuntimeVersion": "sha256:<64 hex>",
  "modelRepositoryVersion": "sha256:<64 hex>",
  "boardInstallationVersion": "sha256:<64 hex>",
  "modelWeightsSha256": "sha256:<64 hex>",
  "board": {"squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0, "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896},
  "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
  "targetAlignment": {
    "method": "TARGET_AFFINE",
    "referenceImageBase64": "<base64 PNG/JPEG target reference>",
    "referenceImageSha256": "sha256:<64 hex>",
    "canonicalWidth": 640,
    "canonicalHeight": 480,
    "alignmentRegions": [{"x": 0, "y": 0, "width": 640, "height": 120}],
    "heldOutRegions": [{"x": 0, "y": 400, "width": 640, "height": 80}],
    "inspectionRegions": [{"x": 180, "y": 140, "width": 280, "height": 220}],
    "minMatches": 20,
    "minInliers": 15,
    "minInlierRatio": 0.5,
    "minCoverageRatio": 0.12,
    "maxReprojectionErrorPx": 2.5,
    "minScale": 0.8,
    "maxScale": 1.2,
    "maxRotationDegrees": 15.0,
    "maxShear": 0.03,
    "maxTranslationPx": 300.0,
    "maxSecondaryInlierRatio": 0.35,
    "minHeldOutMatches": 12,
    "maxHeldOutReprojectionErrorPx": 3.0
  },
  "goldenEmbeddings": [{"id": "GOLDEN-001", "sourceSha256": "sha256:<64 hex>", "values": [0.1, 0.2]}],
  "spatialDifferencePolicy": {"anomalyDistanceThreshold": 0.35, "minRegionAreaRatio": 0.01, "maxRegions": 8}
}
```

Schema `1.0` production artifacts are intentionally rejected. They encoded board-canonical normalization without the independently bound target reference and safety gates. Recompile from a reviewed `1.1` build spec; there is no automatic migration because inventing masks or geometry limits would be unsafe.

## Patch-level spatial difference evidence (map / mask / regions)

`analysis.globalDistance` alone cannot say *where* a capture differs from Golden. When the compiler's embedder supports it, each Golden embedding carries `patchValues`/`patchGridHeight`/`patchGridWidth` plus the digest-bound canonical Golden PNG — the DINOv2 patch-token grid (not just the CLS token) and exact comparison image captured at compile time. At analyze time, if the artifact additionally pins a `spatialDifferencePolicy` and the nearest Golden has patch features, the response's `analysis.spatialDifferenceEvidence` is:

```json
{
  "state": "AVAILABLE",
  "disclaimerCode": "DIFFERENCE_NOT_DEFECT_PROOF",
  "generationMethod": "PATCH_DISTANCE",
  "evidenceRegionNormalized": {"x": 0.17, "y": 0.06, "width": 0.66, "height": 0.88},
  "regions": [{"id": "D-001", "bboxNormalized": {"x": 0.2, "y": 0.2, "width": 0.1, "height": 0.1}, "peakScore": 0.7, "meanScore": 0.5}],
  "mapPngBase64": "<base64 PNG heatmap>",
  "maskPngBase64": "<base64 PNG binary mask>",
  "mapSha256": "<64 hex>",
  "maskSha256": "<64 hex>"
}
```

- The map/mask are rendered at the exact pixel footprint DINO actually analyzed (post `Resize`+`CenterCrop`, always 224x224), not resampled across the full canonical ROI. `evidenceRegionNormalized` locates that analyzed rectangle within canonical-ROI normalized coordinates so a consumer never overlays evidence onto pixels the model never saw.
- Reported `regions` are filtered to those overlapping at least one `targetAlignment.inspectionRegion` — a difference driven by a stable alignment/held-out landmark is not evidence about the inspected equipment and is dropped before `maxRegions` is applied.
- Artifact-1.6/schema-1.2 analysis uses `PAIRED_INTERIOR_ROI_TILED_PATCH_DISTANCE`: Current MobileSAM runs only after canonical alignment passes; the eroded intersection of Golden and Current masks is the sole DINO support; both images are filled with the same neutral gray outside it. Every DINO region is `SUBJECT_INTERIOR`. Boundary changes are a separate geometry evidence channel and do not consume the DINO candidate limit.
- Legacy artifact-1.5/schema-1.1 analysis uses `SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE` with the immutable Golden-only support mask and may classify candidates as `SUBJECT_INTERIOR` or `SUBJECT_BOUNDARY`.
- Legacy explicit `ENGINEERING_REAL_DINO` mode uses `ROI_TILED_PATCH_DISTANCE`: overlapping immutable-ROI tiles are independently inferred for the exact Current and nearest hash-bound Golden, projected into a full-canonical map, averaged at overlaps, and zeroed outside ROI before encoding. Golden tile tokens are cached by canonical digest. Both engineering paths remain `productionAuthorized=false` until separately qualified.
- Whenever real evidence cannot be produced — no `spatialDifferencePolicy` pinned, the nearest Golden lacks patch features, a patch-grid size mismatch, or the embedder doesn't support patches at all — the response is honest by construction instead of fabricated:

```json
{"state": "UNAVAILABLE", "disclaimerCode": "DIFFERENCE_NOT_DEFECT_PROOF", "reasonCode": "GOLDEN_PATCH_FEATURES_UNAVAILABLE"}
```

`compile_artifact` automatically records patch features per Golden whenever the configured embedder exposes `embed_with_patches` (the real `DinoV2Embedder` always does); nothing else needs to change in the build spec besides adding `spatialDifferencePolicy` if you want spatial evidence enabled for that Recipe.

## Performance: warm-up, readiness caching, timeout, device

- `readiness()` re-verifies the artifact/weights/model-repository digests, which is expensive (hashing the full ~90MB weights file and the vendored model-repository tree). A **positive** result is cached for the process lifetime — those are supposed to be immutable, read-only mounts in production — so this cost is paid once, not on every request or `/readyz` poll. A negative result is never cached, so a not-yet-mounted artifact keeps being retried.
- The FastAPI app has a `lifespan` startup hook that loads DINOv2 and, for schema 1.6, warms the pinned MobileSAM runtime before readiness becomes available. Legacy immutable Golden tile caches are not reused as paired-interior scores because every schema-1.6 request has a Current-specific common mask.
- ROI-only scoring keeps the Current tile patch embeddings generated for the global score and reuses the exact same tensors for spatial evidence. The first v10 real HTTP request completed in `7.076 s` under the unchanged `60 s` engineering timeout; v9's first request exceeded `60 s` because it repeated Current tile inference and populated the Golden cache on demand.
- `PHONE_DINO_ANALYSIS_TIMEOUT_SECONDS` (default `8.0`) bounds how long `/internal/v1/analyze` will wait for the synchronous analysis to finish, further bounded by the request's own `deadline`. A timeout returns `504` with `detail=ANALYSIS_TIMEOUT`. Because analysis runs in a thread pool, a timeout stops the *client* from waiting, not the in-flight CPU work; a hard-kill worker process is a later increment, not Phase 0.
- `PHONE_DINO_DEVICE` (default `cpu`) selects the torch device for the embedder (`cpu` or `cuda`).

## Deterministic synthetic full-scene suite

The vision extra provides an engineering-only generator with an actual OpenCV ChArUco board and an independently transformed inspected target:

```powershell
.venv\Scripts\python -m pip install -e ".[synthetic]"
.venv\Scripts\phone-dino-generate-synthetic-suite C:\temp\phone-dino-suite --seed 20260802
```

The output directory must be empty. The command writes `suite.manifest.json`, one target reference and 16 bounded PNG/JPEG scenes. The suite covers board-only, target-only and combined motion; camera perspective; an inspection-only missing component; JPEG/shadow; target absence, duplication and partial visibility; held-out parallax/local warp; blur, exposure, glare, board occlusion and an out-of-bounds target transform.

The manifest is `phone_dino.synthetic-suite/1.0` and is explicitly marked `evidenceClass: SYNTHETIC_ENGINEERING_ONLY`. It pins the seed, generator source/package/Python/OpenCV/NumPy versions, encoded file hashes, independent board/target/camera matrices, projected corners, exact inspection/defect/occlusion support, requested conditions, measurements recomputed from encoded bytes, expected safe state, and complete schema-1.1 `artifactInputs`. `verify_suite_manifest` recomputes hashes, transform composition, projected geometry and CV measurements. Synthetic output is for regression and cross-service E2E only; it is never physical qualification, blind evidence or production approval input.

For cross-repository HTTP tests, `tests.synthetic_contract_app:app` uses the production decoder, ChArUco plane normalizer, target aligner and contract checks while replacing only DINO with a deterministic three-dimensional embedder. It additionally requires `PHONE_DINO_ENABLE_SYNTHETIC_CONTRACT_APP=true`, refuses fixture mode, and retains all ordinary production artifact, model repository, weights, digest and service-token pins.

At runtime, set `PHONE_DINO_MODEL_REPOSITORY_VERSION` to the same repository-tree digest. Readiness
recomputes the local tree, checks the configured pin, checks the artifact pin, verifies the model weights,
and also requires `analyzerRuntimeVersion` to match the running service.

Run tests with:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,synthetic]"
.venv\Scripts\pytest
```

CI must install both `dev` and `synthetic`; the synthetic tests deliberately skip when OpenCV is unavailable, while the cross-repository synthetic E2E fails if it cannot find an OpenCV-enabled Python. A normal unit-test pass with those skips is not synthetic-vision evidence.
