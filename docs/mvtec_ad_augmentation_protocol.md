# MVTec AD normal-capture augmentation protocol

This protocol creates a deterministic, non-commercial research-only camera and
lighting augmentation package from the frozen MVTec AD smoke manifest. It is
not a PhoneDINO runtime feature, qualification dataset, production threshold,
or equipment decision input.

## Boundary and selection rule

Only `NOMINAL` source records with roles `FIT` or `THRESHOLD_TUNING` may be
augmented. `BLIND` records, anomaly-labelled records, and records with an
official MVTec mask are rejected as augmentation parents. Blind source images
remain original and are reporting-only.

This rule is deliberate: the frozen subset has no labelled anomaly development
split. Choosing camera-recipe parameters with blind AUROC, per-defect misses,
or the official blind masks would contaminate the only held-out anomaly set.
An iteration may compare normal-score stability, threshold inflation, resource
use, and elapsed time on the FIT/tuning normal sources only. It must lock the
recipe before making a blind observation report.

## Generic camera envelope

The committed [v1 recipe](../tools/mvtec_ad_camera_lighting_recipe_v1.json) is
a bounded generic simulation, not a calibration to any phone or fixture:

- residual rotation at most ±1°, scale ±1.5%, translation ±0.5% of an edge,
  and corner displacement ±0.5% of the shorter edge;
- exposure ±0.35 EV, gamma ±0.08, red/blue white balance ±8%, low-frequency
  shading up to 15%, and vignetting up to 10%;
- sensor read-noise standard deviation at most 1.5 DN and JPEG quality 90–98.

It deliberately excludes crop, flip, cutout, synthetic defects, strong glare,
and blur. Those changes can destroy a normal label or model conditions that a
real PhoneDINO capture path must instead reject or independently qualify.

[v2](../tools/mvtec_ad_camera_lighting_recipe_v2.json) is a deliberately
narrower envelope. It may be selected only from normal FIT/tuning robustness
and threshold-inflation results; it must not be selected from blind AUROC,
defect labels, or MVTec masks.

## Generate a package

The destination must be a new or empty directory outside this Git worktree.
Do not put MVTec sources, generated images, feature caches, or benchmark
reports in the repository.

```powershell
.venv\Scripts\python tools\generate_mvtec_ad_normal_augmentations.py `
  C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\subset_manifest.json `
  C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\camera_capture_v1 `
  --variants-per-parent 1
```

The package contains JPEG derivatives and `augmentation_manifest.json`. Each
derivative binds its parent case/source digest, recipe digest, variant ID,
derived seed, exact sampled parameters, output path, and output SHA-256. The
seed is derived from the recipe digest, parent case ID, parent source digest,
and variant ID, so adding records or changing enumeration order cannot alter
an existing derivative.

The manifest declares all of the following:

- `authoritative: false` and `productionAuthorized: false`;
- `purpose: OFFLINE_MVTEC_RESEARCH_ONLY`;
- `blindPolicy: BLIND_ORIGINAL_ONLY`; and
- the raw frozen-manifest file digest plus its declared manifest identity.

Any score runner must verify these bindings before consuming the derived
normal FIT/tuning images. It must verify every generated image SHA-256 and
must never add a generated blind image to its data flow.
