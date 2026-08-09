# Frozen physical-alignment replay cohort

`phone-dino.physical-alignment-cohort/1.0` is an external-only evidence
format for replaying already captured stills through the same
`OpenCvCharucoNormalizer` path used by the service. It is useful for target
alignment regression triage and for recording native JPEG coding facts before
an augmentation recipe is proposed.

It is not a production qualification, a defect-data study, a threshold
selection mechanism, or a PASS/FAIL decision. Every cohort and report must
declare `authoritative: false` and `productionAuthorized: false`.

## External package layout

Keep the manifest, raw stills, saved requests/responses, pinned artifact, and
saved `/readyz` JSON beneath one non-reparse external directory. Do not place
the cohort or its output in this repository. All referenced paths are relative
to the manifest directory; absolute paths, drive-relative paths, traversal,
symbolic links, junctions, and reparse points are rejected.

```text
C:\external\PM-ABC-001-alignment-r1\
  cohort.json
  artifact\artifact.json
  readyz.json
  raw\capture-001.jpg
  requests\capture-001.json
  responses\capture-001.json
```

The manifest has this closed shape (the SHA-256 values are illustrative):

```json
{
  "schemaVersion": "phone-dino.physical-alignment-cohort/1.0",
  "cohortSha256": "sha256:<canonical JSON without cohortSha256>",
  "authoritative": false,
  "productionAuthorized": false,
  "purpose": "PHYSICAL_ALIGNMENT_REPLAY_ONLY",
  "artifact": {
    "relativePath": "artifact/artifact.json",
    "artifactPackageDigest": "sha256:<artifact bytes>",
    "schemaVersion": "1.9",
    "recipeId": "PM-ABC-001",
    "machineId": "MC-07",
    "boardId": "CB-001"
  },
  "readyzEvidence": {"relativePath": "readyz.json", "sha256": "sha256:<bytes>"},
  "replayPolicy": {
    "allowTargetOnlyAlignment": true,
    "allowContourAnchorAlignment": true
  },
  "cases": [
    {
      "caseId": "capture-001",
      "partition": "DEVELOPMENT",
      "acquisitionGroupId": "session-2026-08-09-a",
      "intendedAlignmentState": "ALIGNED",
      "captureStrata": {
        "device": "device-class", "camera": "rear-wide", "lens": "native",
        "lighting": "bench-a", "distance": "fixed-300mm", "view": "top"
      },
      "raw": {"relativePath": "raw/capture-001.jpg", "sha256": "sha256:<bytes>"},
      "request": {"relativePath": "requests/capture-001.json", "sha256": "sha256:<bytes>"},
      "response": {"relativePath": "responses/capture-001.json", "sha256": "sha256:<bytes>"},
      "opaqueNativeAttestationDigest": "sha256:<optional opaque attestation>"
    }
  ]
}
```

`cohortSha256` is canonical JSON SHA-256 using sorted keys, compact
separators, ASCII escaping, and the `cohortSha256` field omitted. The raw
manifest byte digest is recorded separately in the replay report.

The saved `/readyz` file must be the service response from the same deployed
artifact. In addition to normal readiness fields, current PhoneDINO emits a
non-secret binding like this:

```json
{
  "status": "ready",
  "simulation": true,
  "analysisMode": "ENGINEERING_REAL_DINO",
  "supportedSchemas": ["1.0", "1.1", "1.2", "1.3", "1.4"],
  "replayProvenance": {
    "artifactPackageDigest": "sha256:<artifact bytes>",
    "analyzerRuntimeVersion": "sha256:<runtime>",
    "allowTargetOnlyAlignment": true,
    "allowContourAnchorAlignment": true,
    "maxImageBytes": 12582912,
    "maxImagePixels": 24000000,
    "maxImageWidth": 8000,
    "maxImageHeight": 8000
  }
}
```

## Required validation

Before replay, the tool recomputes every declared file digest and validates:

- the artifact bytes and schema (1.4 through 1.9), Recipe/machine/board IDs;
- request raw/artifact/execution-bundle identity and response request/raw,
  resolved model/runtime/normalization/profile/scorer identity;
- the historical analysis ID derived from the pinned artifact runtime;
- the saved readiness snapshot is `status: ready`,
  `analysisMode: ENGINEERING_REAL_DINO`, and has the same `simulation` value
  as every request. Its service-emitted `replayProvenance` must bind the
  artifact digest, analyzer runtime digest, target-only/contour fallback
  flags, and image byte/dimension limits. Each saved request wire schema must
  be listed in `supportedSchemas`. V19 additionally requires saved support
  for wire 1.6 and a `VERIFIED` saved reference-board observation bound to the
  artifact QR digest;
- one unique raw still per case and no acquisition group crossing
  `DEVELOPMENT` and `HELD_OUT`. A saved `sessionId` or `correlationId` may
  belong to only one group/partition, so one real capture session cannot be
  relabelled across the split.

The tool parses saved observations to validate their identity but never copies
embedded image/mask/base64 fields to its report.

Raw JPEGs are reopened with the runtime sequence
`PIL.Image.open(bytes) → ImageOps.exif_transpose → RGB`. The report records
only safe coding facts: oriented dimensions, JPEG component/sampling factors,
progressive flag, and a digest of quantization tables. It deliberately omits
EXIF, raw paths, raw bytes, and image payloads.

For V19 the replay also reruns the same-still QR/ChArUco reference-board gate
before normalisation. This is a visual replay only; it does not authenticate
PhoneCV lifecycle records, signatures, or authorization.

Use opaque, non-personal identifiers for `caseId`, `acquisitionGroupId`, and
capture strata. The schema accepts only letters, digits, `.`, `_`, and `-` for
those reportable values.

## Replay command

```powershell
.\.venv\Scripts\python.exe scripts\replay_subject_alignment_cases.py `
  --cohort C:\external\PM-ABC-001-alignment-r1\cohort.json `
  --output C:\external\PM-ABC-001-alignment-r1\reports\alignment-replay.json
```

The output path must not exist and must be external to the Git worktree. The
report is self-digested and is written only after all inputs validate. A replay
expectation mismatch returns exit code 2 but retains the non-authoritative
report for engineering review.

When every request is an engineering simulation, the report is labelled
`ENGINEERING_REPLAY_ONLY`. A non-simulation capture remains
`UNAUTHORIZED_CAPTURE_REPLAY_ONLY` unless separate physical authorization and
qualification evidence has been obtained. The report also records the runtime
that performed the replay. A historical artifact whose runtime digest differs
from the local replay runtime is explicitly `CROSS_RUNTIME_REPLAY_ONLY`, not
evidence that the current service would reproduce the historical result.

`opaqueNativeAttestationDigest`, when supplied, is merely an opaque
cross-reference. The tool validates its SHA-256 syntax but does not verify a
signature or treat it as a native-device attestation.

## Legacy local triage

The older form remains available for quick diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts\replay_subject_alignment_cases.py `
  --artifact runtime\engineering-real-dino\engineering-real-dino-artifact-v15.json `
  --generated --case local-photo=C:\temp\photo.jpg
```

Its JSON-lines output is explicitly `UNVERIFIED_AD_HOC`; it does not bind a
cohort, artifact digest, request/response, or readiness snapshot and must not
be used as physical evidence.
