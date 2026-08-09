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

[v3](../tools/mvtec_ad_camera_lighting_recipe_v3.json) is a controlled A/B
extension of v2. It adds only a bounded, non-brightening off-axis lens-shading
term. Its `samplingSeedAnchor` is v2's recipe digest, so every pre-existing
v2 random draw remains byte-for-byte identical for the same parent and variant;
only the three new lens-shading parameters are sampled afterwards. This keeps
the normal-only comparison attributable to the lens-shading change rather than
to a wholesale reseeding of geometry and photometry.

V3 deliberately does not add blur, glare, radial distortion, hot pixels, or
stronger noise. Those effects need real device/fixture evidence before they
can be treated as normal-label-preserving simulation. The current JPEG output
for v1--v3 is 4:4:4.

[v4](../tools/mvtec_ad_camera_lighting_recipe_v4.json) is the separate,
single-factor mobile-JPEG experiment: it retains v3's geometry, photometry,
off-axis lens shading, quality range, and v2-based `samplingSeedAnchor`, but
uses only JPEG 4:2:0 chroma subsampling. It adds no RNG draw or reordering, so
every v4 parent/variant retains exactly the same sampled v3 parameters; the
encoded chroma sampling is the only intended visual difference. It is still a
generic research simulation, not proof of a particular phone encoder, device
calibration, physical qualification, or production authorization.

[v5](../tools/mvtec_ad_camera_lighting_recipe_v5.json) is a separate,
single-factor V4-compatible coding-profile probe. It retains the exact V4
geometry, photometry, off-axis lens shading, `samplingSeedAnchor`, and
95--98 JPEG-quality sampling stream. It adds no RNG draw or reorder: every
sampled parameter, including `parameters.jpegQuality`, remains identical to
V4 for the same parent and variant. Only the quality passed to Pillow is
overridden to Q95, with 4:2:0 sampling, non-progressive coding, and the locked
Q95 quantization-table digest
`sha256:f67e35fd0dcd2fd9f999077e2aae8560e6327a8477c45427f6ea2e0a224cd187`.
Each record separately reports the sampled quality and
`outputJpegQuality: 95`, so the output profile cannot be confused with the
preserved sampling stream. This is a bounded research probe motivated by an
engineering JPEG-header observation, not a device calibration, native-capture
attestation, physical qualification, or production authorization.

V3-R4 is a seed-replication study, not a new visual effect: it uses the
unchanged v3 recipe with `--variants-per-parent 4`. Its formal normal-only
selection contract must bind that count. Every normal FIT and tuning parent
must then have exactly variants `1` through `4`, and each variant is subject
to its own paired-score P95/max gate. This prevents one deterministic draw
from being hidden in a pooled robustness summary. It still must not read or
score the frozen blind set.

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
seed is derived from the recipe digest (or an explicitly recorded controlled
seed anchor), parent case ID, parent source digest, and variant ID, so adding
records or changing enumeration order cannot alter an existing derivative.

The manifest declares all of the following:

- `authoritative: false` and `productionAuthorized: false`;
- `purpose: OFFLINE_MVTEC_RESEARCH_ONLY`;
- `blindPolicy: BLIND_ORIGINAL_ONLY`; and
- the raw frozen-manifest file digest plus its declared manifest identity.

The current generator emits augmentation-manifest schema `1.3`. It embeds the
parsed recipe and records generator module/entrypoint hashes, Git revision and
worktree state, plus Python, Pillow, OpenCV, and NumPy versions. Every record
also attests `outputEncoding` (JPEG/RGB, chroma subsampling, and the three
decoded sampling factors). It also binds component IDs, quantization-table
selectors, non-progressive encoding, and a SHA-256 of the two 64-value JPEG
quantization tables. Those tables are independently rederived with the exact
effective output JPEG quality and explicit Pillow save arguments. Generation
reopens each written JPEG to verify that attestation; before scoring, the loader
verifies it again after checking the file digest. This prevents a manifest
whose hashes were recomputed from silently relabelling a 4:4:4 or other JPEG
as v4/v5 4:2:0, lowering or changing its locked output JPEG quality, or
changing it to progressive coding.
It is a coding-profile check, not a replacement for an externally signed
package or deterministic re-rendering when an adversary can replace pixels
while preserving all JPEG header and quantization fields.

Recipe schema `1.0` is retained only for the historical v1--v3 4:4:4 recipes.
The v4 recipe uses schema `1.1`, whose closed allowlist permits exactly the
locked v3 geometry, photometry, quality range, off-axis lens values, and
v2-based seed anchor plus `jpegSubsampling: "4:2:0"`. Schema `1.1` is reserved
for this exact V4 experiment, so it cannot hide a geometry, photometry,
quantization, progressive, or other visual-effect change behind the one-factor
claim. The loader rederives every parameter set from its parent and variant and
requires exactly every eligible normal parent times the declared variant count.
V5 uses recipe schema `1.2`, which is reserved for that same V4 baseline plus
only `jpegQualityOutputOverride: 95` and the fixed Q95 quantization-table
digest. Augmentation schema `1.3` adds the mandatory, independently rederived
`outputJpegQuality` record field. Historical V4-R4 schema-`1.2` packages are
locked historical evidence: this current loader intentionally does not consume
them because they predate that record contract and have a pinned generator
module digest. Inspect them only with their pinned `de55a73` worktree, or
regenerate a fresh independent envelope; never silently upgrade or compare
them with V5 in one selection universe.

Any score runner must verify these bindings before consuming the derived
normal FIT/tuning images. It must verify every generated image SHA-256 and
must never add a generated blind image to its data flow.
