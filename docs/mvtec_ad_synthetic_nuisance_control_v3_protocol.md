# V3 paired generic-capture nuisance-control protocol

## Purpose and boundary

V3 is an offline MVTec engineering audit that compares deterministic V2
synthetic stimuli with deterministic R3 generic-capture derivatives of the
same raw query parent. It is intended to distinguish a response to an added
synthetic stimulus from a response to ordinary bounded capture nuisance.

It is not a real-defect evaluation. Its report always declares:

- `evidenceClass: SYNTHETIC_ENGINEERING_ONLY`
- `realAnomalyPerformance: NOT_ESTIMATED`
- `realPrecisionRecall: NOT_ESTIMATED`
- non-authoritative, non-production, no comparison/promotion status

The report must not be used for model, algorithm, hyperparameter, threshold,
package, or production selection; physical qualification; or a claim about
real anomaly performance.

## Immutable input boundary

The audit accepts only the sealed successor-V2 `FIT` raw-normal partition. It
recreates the predeclared source-digest / case-id parent split per category:

| Parent role | Count per category | Total |
| --- | ---: | ---: |
| Prototype | 6 | 18 |
| Raw calibration | 2 | 6 |
| Query parent | 4 | 12 |

It never opens image bytes from threshold tuning, selection, confirmation,
reserve, blind, anomaly, or mask partitions. It does not accept or read any
V1/V2 response report.

The raw calibration maximum fixes one threshold per category before either
child package loader can access image bytes.

### Permanent quarantine of `fresh_normal_holdout_v1`

The previously proposed `fresh_normal_holdout_v1` V3 input is permanently
ineligible. An erroneous recursive byte-hash preflight made the exact image
read set unprovable, so it cannot support a claim that its held-out boundary
remained untouched. This is not a retryable receipt state and it has no
release or override path.

The external JSON-only incident ledger record
`incident_ledger\fresh_normal_holdout_v1_quarantine.json` is deliberately
outside the quarantined cohort root and binds both the
frozen manifest file digest
`sha256:0034e045001787a6ce35042701cb470a97c03ff72117311ff7525fd5d9106b18`
and declared digest
`sha256:51a359f5d579a99321dc33687fecc6d9a8db92fb7f921960bbb6898c23e2e74e`.
Its file digest is
`sha256:f8797a33eca14339cefa05c3c0aa94827617e2d717e1c21ee8287479baf7781d`
and its independently retained declared incident pin is
`sha256:be690e112ae28f04a69db572a7b9931d862fcac1da9653e92a99ad5995fbf2d4`.

The V3 direct API and CLI require the incident path plus that independent pin,
validate the incident and parent manifest through JSON-only paths before any
output slot, receipt, model work, FIT loader, query image, or child image can
be opened, and compile the known cohort identity into the guard. Missing,
replaced, or self-consistent forged incident records therefore fail closed and
cannot re-enable this cohort. No future controlled cohort may use this V3
implementation: it is hard-bound to the old successor evidence chain. Any
recovery requires a separately designed and preregistered V4 protocol.

## Paired packages

The audit requires these existing immutable packages, both bound to the same
sealed successor contract and FIT identity:

| Package | Child scope opened by V3 | Per query parent | Total |
| --- | ---: | ---: | ---: |
| R3 generic-capture package | `registration`, `illumination`, `sensor_transport` | 3 controls | 36 |
| V2-r2 synthetic-stimulus package | 3 families × 3 fixed levels | 9 stimuli | 108 |

The full manifests, recipes, declared self-digests, contract bindings, and
model identity are preflighted. V3 opens only the 36 control child images that
belong to the fixed 12 query parents; all other R3 child images remain out of
scope. The V2 package is required to cover every query parent / family / level
combination.

Preflight must address only the explicitly supplied paths. It must never
recursively enumerate or byte-hash an external cohort root to discover an
artifact: hashing an image is itself image-byte access and invalidates any
claim that a supposedly held-out partition remained untouched.

## One-time registry receipt

`--registry-root` is an external persistent directory, not a Git directory.
After raw FIT and model preflight, but before either child loader, V3 computes a
deterministic identity from the sealed contract, parent split, model identity,
and immutable package/recipe identities. It atomically creates an exclusive
receipt named from that identity.

An already-present receipt rejects the invocation before camera-control or
synthetic child bytes are loaded. This intentionally prevents retrying the
same package/model/contract identity after inspecting a prior attempt.

## Report scope

The report records raw query, generic-capture control, and synthetic-stimulus
above-threshold response counts/rates plus score deltas. Its principal
contrast unit is the raw query parent (`n=12`), not its dependent rendered
children: per parent, it compares the mean of three generic-capture
child-minus-parent deltas with the mean of nine synthetic-stimulus
child-minus-parent deltas.

It intentionally contains no confusion matrix or numeric precision, recall,
F1, AUROC, AP, confidence interval, ranking, or real-defect performance
metric. The required `realPrecisionRecall` field is the literal status
`NOT_ESTIMATED`, not a measured metric.

## Historical invocation shape (not an authorization)

V3 has no valid future invocation. The syntax below records the mandatory
guard arguments for the historic chain and demonstrates why the old cohort
rejects; it must not be repointed at a new cohort. A recovery experiment needs
a separately designed V4 entry point and protocol.

The historic command required a new external snapshot from independently
approved source pins and preservation of the printed `snapshotManifestSha256`
outside that directory. V3 does not calculate or trust a replacement manifest
digest supplied by the snapshot itself.

```powershell
.venv\Scripts\python.exe tools\materialize_sealed_dino_snapshot.py `
  --expected-repository-sha256 <approved-source-repository-sha256> `
  --expected-weights-sha256 <approved-source-weights-sha256> `
  --output <new-external-sealed-dino-snapshot>
```

```powershell
.venv\Scripts\python.exe tools\run_mvtec_ad_synthetic_nuisance_control_v3.py `
  --sealed-model-snapshot <external-sealed-dino-snapshot> `
  --expected-sealed-model-snapshot-manifest-sha256 <approved-snapshot-manifest-sha256> `
  --parent-holdout <external>\normal_holdout.json `
  --cohort-quarantine-incident <external>\incident_ledger\fresh_normal_holdout_v1_quarantine.json `
  --expected-cohort-quarantine-incident-sha256 sha256:be690e112ae28f04a69db572a7b9931d862fcac1da9653e92a99ad5995fbf2d4 `
  --parent-selection-contract <external>\selection_protocol_v2\fresh_normal_selection_contract.json `
  --plan <external>\successor_protocol_v2\fresh_normal_successor_plan.json `
  --envelope <external>\successor_protocol_v2\fresh_normal_successor_envelope.json `
  --source-root <external>\source_bytes `
  --stimulus-augmentation-manifest <external>\synthetic_stress_v2_r2\augmentation_manifest.json `
  --capture-control-augmentation-manifest <external>\successor_v2_fit_camera_r3\augmentation_manifest.json `
  --registry-root <external>\synthetic_nuisance_control_v3_registry `
  --output <external>\synthetic_nuisance_control_v3_report.json
```

Both the output path and receipt identity are new-only. Use newly generated,
independently predeclared inputs for a distinct future audit; do not delete or
reuse a receipt to repeat the same identity.

Do not invoke this command with the quarantined `fresh_normal_holdout_v1`
manifest shown here: it must reject before it can open any FIT/query bytes.

The snapshot manifest pin is a separate trust anchor: a self-consistent slot
replacement is rejected unless it matches that externally retained digest.
Activation rejects any preloaded `dinov2` modules and centralized Python
bytecode caches, and it serializes process-global import state. On Windows,
same-privilege filesystem replacement races cannot be eliminated by this
Python layer; store the external snapshot in an ACL-protected/read-only
directory and do not permit competing writers.
