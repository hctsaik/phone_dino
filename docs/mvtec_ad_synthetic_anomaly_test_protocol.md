# Synthetic-only MVTec augmentation test

## Scope

This is an offline engineering test for deterministic, programmatically
rendered image stimuli. It is intentionally separate from the frozen MVTec
normal-selection and reserve-successor protocols.

Every generated child and every reported metric is labelled `SYNTHETIC_ONLY`.
The test does **not** estimate real-defect precision, real-defect recall,
real-world anomaly performance, device performance, calibration validity, or
physical qualification. It cannot change a normal-selection lock, a threshold,
or a production configuration.

## Allowed inputs

The generator and evaluator may request only the successor V2 `FIT` partition
through the phase-safe loader. The closed normal-only JSON chain may be
validated before that request, but no image bytes from `THRESHOLD_TUNING`,
`NORMAL_SELECTION`, `NORMAL_CONFIRMATION`, `RESERVE_UNTOUCHED`, blind,
anomaly, or mask sources may be opened.

The 12 raw FIT parents in each category are deterministically ranked by their
immutable source digest and allocated as follows:

| Use | Parents/category | Permitted role |
| --- | ---: | --- |
| Prototype | 6 | Raw normal feature bank only |
| Calibration | 2 | Raw normal threshold only |
| Query | 4 | Raw nominal negatives and synthetic-positive parents |

Parents and source groups must be disjoint across these three uses. The split
is fixed before a synthetic label is consulted.

## Synthetic renderings

The closed recipe emits one deterministic PNG child per query parent for each
of these stimulus families:

- `LOCAL_SCRATCH`
- `LOCAL_SPOT`
- `LOCAL_OCCLUSION`

They are deliberately simple procedural overlays, not reconstructions of
observed defects. Each child is bound to its FIT parent digest, a named
SHA-256 substream, parameters, recipe digest, output digest, and a
byte-for-byte renderer check. The package is external-only and new-only.

## Fixed scoring rule

The evaluator builds a raw-normal prototype bank from the six prototype
parents and computes each category's threshold before it scores any query:

```text
threshold(category) = max(raw calibration-normal scores)
predicted synthetic-positive iff score > threshold(category)
```

Raw query parents are the synthetic-negative examples. Rendered children are
the synthetic-positive examples. The report may state only
`syntheticPrecision`, `syntheticRecall`, `syntheticF1`, and their underlying
synthetic TP/FP/FN/TN counts. It must also state:

```text
metricScope: SYNTHETIC_RENDERING_DISCRIMINATION_ONLY
realAnomalyPerformance: NOT_ESTIMATED
forbiddenUses: MODEL_SELECTION, THRESHOLD_SELECTION,
               PRODUCTION_VALIDATION, PHYSICAL_QUALIFICATION
```

The samples are paired by raw parent and are small. Therefore even a perfect
synthetic score demonstrates only that the frozen detector responds to these
particular rendered patterns.

## Commands

Create a new external synthetic package first:

```powershell
.venv\Scripts\python.exe tools\generate_mvtec_ad_synthetic_anomaly_augmentations.py `
  --parent-holdout <external-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
  --plan <external-successor-plan.json> `
  --envelope <external-successor-envelope.json> `
  --source-root <external-source_bytes> `
  --recipe tools\mvtec_ad_synthetic_anomaly_recipe_v1.json `
  --output <new-external-synthetic-package>
```

Then run a new external report:

```powershell
.venv\Scripts\python.exe tools\run_mvtec_ad_synthetic_anomaly_test.py `
  --parent-holdout <external-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
  --plan <external-successor-plan.json> `
  --envelope <external-successor-envelope.json> `
  --source-root <external-source_bytes> `
  --augmentation-manifest <external-synthetic-package/augmentation_manifest.json> `
  --output <new-external-synthetic-report.json>
```

Both commands reject repository-local outputs, existing outputs, unsafe paths,
links, reparse points, altered bytes, and inputs outside the FIT partition.
