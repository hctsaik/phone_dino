# Fresh FIT-only camera augmentation protocol

This is a new, external-only augmentation envelope for the fresh normal MVTec
holdout. It is separate from the historical V3--V5 camera packages, feature
caches, reports, and selector contracts. Its output is non-authoritative,
non-production offline research material; it does not establish device
calibration, physical qualification, or anomaly-detection performance.

## Input boundary

The generator accepts only:

- a frozen external `phone-dino.mvtec-ad-normal-holdout/1.0` manifest;
- its external normal-byte root; and
- the closed repository recipe
  `tools/mvtec_ad_fresh_fit_camera_recipe_v1.json`.

It asks the phase-safe holdout reader for `FIT` only. That reader validates the
closed normal-holdout JSON and rehashes/decodes requested FIT parents, without
opening the source pool, historical-use ledger, allocation plan, or public
`samples.json` inventory. It therefore never opens tuning, selection,
confirmation, reserve, blind, anomaly, or mask bytes during generation.

`THRESHOLD_TUNING`, `NORMAL_SELECTION`, and `NORMAL_CONFIRMATION` remain raw
normal originals. Synthetic images may enter the prototype bank with FIT
parents only; they must not set a threshold or be used to select/confirm a
candidate.

## V1 generic prior

The closed V1 recipe uses conservative, normal-label-preserving transforms:

- sub-degree perspective/scale/translation variation with reflected borders;
- bounded exposure, gamma, red/blue gain, directional shading, and vignette;
- bounded off-axis low-frequency lens shading;
- low-amplitude signal-dependent plus read-noise simulation; and
- explicit non-progressive JPEG 4:2:0 quality 95 output.

It intentionally excludes crop, flip, blur, glare, occlusion, hot pixels,
radial distortion, and synthetic defects. Those effects would need a separate
capture-reject stress protocol and cannot be mixed into normal FIT, threshold,
selection, or confirmation inputs.

Every effect samples from a named SHA-256 substream derived from the recipe
digest, parent case/digest, and variant ID. Adding a future effect in a new
schema does not reorder the existing geometry, photometry, lens-shading, or
noise draws. V1 fixes both the recipe fields and the expected Q95 JPEG
quantization-table digest; a different image transform or coding profile needs
a new recipe/schema rather than an edited V1 file.

## Package validation

The generated external package binds the frozen holdout file and declared
digest, development and FIT-parent identities, recipe bytes, generator/tool/
holdout-module source hashes, runtime versions, and every child record.

For each child, the validator verifies:

- its parent is a currently validated `FIT` / `NOMINAL` / `good` record;
- exact parent case, source digest, group, category, variant coverage, named
  parameters, and deterministic path;
- output SHA-256 and complete JPEG component/subsampling/progressive/
  quantization-table attestation; and
- exact deterministic re-rendered JPEG bytes from the FIT parent and frozen
  parameters.

All package outputs must be new external paths. Existing files, paths in the
Git worktree, path traversal, symbolic links, and Windows reparse points are
rejected. A failed output directory remains burned rather than being reused.

## Command

Use a fresh non-existent external output directory. Four variants per FIT
parent produce 576 derivatives for the current 144-parent fresh cohort.

```powershell
.venv\Scripts\python.exe tools\generate_mvtec_ad_fresh_fit_augmentations.py `
  --holdout C:\code\claude\_media_out_of_repo\mvtec_ad\fresh_normal_holdout_v1\normal_holdout.json `
  --source-root C:\code\claude\_media_out_of_repo\mvtec_ad\fresh_normal_holdout_v1\source_bytes `
  --recipe tools\mvtec_ad_fresh_fit_camera_recipe_v1.json `
  --variants-per-parent 4 `
  --output C:\code\claude\_media_out_of_repo\mvtec_ad\fresh_normal_holdout_v1\fresh_fit_camera_v1_r4
```

Generation alone is not a configuration decision. Before reading
`NORMAL_SELECTION`, a later evaluator must freeze its candidate configurations,
augmentation package identity, gates, and selection contract. It may then
consume selection once, lock one candidate without automatic promotion, and
use raw `NORMAL_CONFIRMATION` only for a one-time observation.
