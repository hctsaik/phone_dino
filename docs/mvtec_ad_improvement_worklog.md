# MVTec AD research improvement work log

**Status:** implementation, one historical locked-configuration blind
observation, and one later normal-only configuration lock completed on
2026-08-09. This log is for non-commercial research only. None of the work
below changes the PhoneDINO service, artifact schema, QR/ChArUco process,
physical measurement, or any equipment decision.

## Fixed inputs

- Dataset root (outside Git):
  `C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1`
- Frozen manifest:
  `subset_manifest.json`
- Manifest digest:
  `sha256:cfcffd2326278cda20d803f72366c276014334df333d15fcd45911c92704ae27`
- Categories: `capsule`, `metal_nut`, `tile`.
- Per category: 48 normal `FIT`, 16 normal `THRESHOLD_TUNING`, 16 normal
  `BLIND`, and 32 anomalous `BLIND` images.  The 96 available official masks
  are retained for a later pixel-level evaluation.

All 336 images and 96 masks were SHA-256 verified against the manifest.

## Completed baseline

The global DINOv2 nearest-normal cosine baseline was run on CPU and recorded
in `dinov2_global_knn_report.json` next to the manifest.  It scores an image
as its closest normal FIT image and derives each category's threshold only
from normal `THRESHOLD_TUNING` images.

| Category | Blind image AUROC | Anomalies above tuning threshold | Finding |
| --- | ---: | ---: | --- |
| capsule | 0.9355 | 24 / 32 | 8 misses: faulty_imprint (4), poke (3), scratch (1) |
| metal_nut | 0.8984 | 22 / 32 | 10 misses: bent (5), color (3), scratch (2) |
| tile | 1.0000 | 32 / 32 | No anomaly misses in this small split |

The global baseline therefore remains a research comparator, not an accepted
algorithm or a valid operational threshold.

## Implemented research iteration tooling

The original `tools/run_mvtec_ad_smoke.py` accepts:

```powershell
--algorithm patch-knn --max-prototypes 1024 --top-k-patches 5
```

The new `patch-knn` method reuses the existing local `DinoV2Embedder`
`embed_with_patches` interface (16 x 16 patch tokens).  For each category it:

1. pools normal FIT patch tokens;
2. takes a deterministic, evenly spaced, bounded memory bank of at most 1,024
   prototype patches;
3. finds every query patch's closest normal patch by cosine distance; and
4. scores the mean of the five largest nearest-normal distances.

It is deliberately labelled a *PatchCore-style research baseline*, not a
PatchCore implementation. Its score threshold still comes only from normal
`THRESHOLD_TUNING` records. The produced JSON remains `authoritative: false`.

The follow-on tools are now available:

- `tools/generate_mvtec_ad_normal_augmentations.py` generates deterministic,
  generic camera/lighting JPEG derivatives only for normal `FIT` and
  `THRESHOLD_TUNING` inputs. It rejects blind, anomalous, and mask-bearing
  parents; each output binds parent/source/recipe/seed/parameter/output
  digests, and must be outside Git.
- `tools/run_mvtec_ad_iteration.py` batches feature extraction, has a
  content-bound external cache, scores exact bounded-memory patch k-NN blocks,
  records provenance/timing, preserves a 16 x 16 distance grid, and adds
  research-only pixel AUROC/AUPRO after score fixing.
- `--normal-only` never loads, scores, or reports `BLIND` records. It exposes
  separate original and derivative tuning distributions for normal-only
  iteration.
- `tools/compare_mvtec_ad_reports.py` strictly joins report scores by immutable
  `caseId` plus `sourceSha256`. It has no winner/ranking output and explicitly
  forbids blind-based model selection.

See [augmentation protocol](mvtec_ad_augmentation_protocol.md) and
[iteration protocol](mvtec_ad_iteration_protocol.md).

## Validation completed

```powershell
py -3.11 -m pytest tests\test_mvtec_ad_smoke.py
```

Result: `4 passed in 1.70s`.

The focused tests cover AUROC ties, threshold/report separation, deterministic
memory-bank selection, and the top-k patch-distance scorer.

## Reconciled baseline and locked observation

The prior work log said the 1,024-prototype run was terminated before a report.
The external data directory nevertheless contained a structurally complete
legacy report. Because that report lacked elapsed-time/cache/source-revision
provenance, it was not relied on as the verification record. It was reproduced
using a fresh output path with the new tool:

```powershell
.venv\Scripts\python tools\run_mvtec_ad_iteration.py `
  C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\subset_manifest.json `
  --algorithm patch-knn --max-prototypes 1024 `
  --batch-size 4 --prototype-block-size 256 `
  --feature-cache C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\feature_cache_iteration_v1 `
  --output C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\reports_iteration_v1\patch_clean_iteration.json
```

It reproduced the legacy image-level values (within float32 precision) and
recorded 16 x 16 localization grids. The cold-cache phases were feature
inference 34.01 s, input verification 11.55 s, patch scoring 0.88 s, pixel
metrics 3.81 s, total 54.44 s. This confirms that the historical delay was
feature extraction rather than the bounded nearest-neighbour calculation.

The v1 generic capture envelope caused normal-only threshold inflation with
1,024 prototypes (capsule 0.4901, metal_nut 0.4987, tile 0.3052), so it was not
used as the next configuration. That decision used normal FIT/tuning data only.
The narrower v2 recipe with 2,048 prototypes was then evaluated using
`--normal-only`: `BLIND` count was zero, cache/input digests were recorded, and
the combined original-plus-derivative normal maxima were capsule 0.3016,
metal_nut 0.2733, tile 0.2447. Its bounded scoring phase was 1.00 s. The
configuration was locked before the following blind observation; these
numbers do not establish a production threshold.

| Category | Locked image AUROC | Anomalies above normal-only threshold | Pixel metric contribution |
| --- | ---: | ---: | --- |
| capsule | 0.7988 | 20 / 32 | Reported only after lock |
| metal_nut | 0.9746 | 30 / 32 | Reported only after lock |
| tile | 1.0000 | 32 / 32 | Reported only after lock |

The locked report has 192 original score records, 96 normal calibration
records (including derivatives), `blindAugmentedCount: 0`, 96 frozen official
masks, pixel AUROC 0.8880, and AUPRO@30% FPR 0.7356. Its 384 normal feature
cache hits and 144 blind misses are recorded in the external report. The
comparison against the clean 1,024-prototype report is external at
`patch_clean1024_vs_locked_v2_2048.json`; it is an observational delta report,
not a candidate winner.

## V3 off-axis normal-only configuration lock (2026-08-09)

The schema-1.1 V3 camera envelope was generated outside Git at
`camera_capture_v3_off_axis\augmentation_manifest.json`. It contains 192
deterministic normal-only derivatives (one each for every FIT/tuning normal),
with the narrow V2 envelope preserved through its sampling anchor and one
additional bounded, non-brightening off-axis lens-shading perturbation. Its
manifest digest is
`sha256:da5bc83d10ad6857af0d9c27e9fb391b63a4bdbda524e405084702b3718bea79`;
the recipe digest is
`sha256:5cf5b83ac58251b350d1fb2c01c6ac07e6fe624bf5e47ae0f5f6da7c8ee887d4`.
Its manifest explicitly declares `BLIND_ORIGINAL_ONLY`.

Two schema-1.2 `--normal-only` reports were then run from exactly this normal
evaluation envelope: `patch_v3_1024_normal_only.json` (cold cache, 103.78 s)
and `patch_v3_2048_normal_only.json` (reused cache, 27.90 s). Both have zero
blind/anomaly feature inputs, zero blind score records, no pixel metrics, and
the same score-free feature/calibration/original-tuning membership identities.
The 1024 report digest is
`sha256:8db20fa0e855df272124633cf4990a1b89523e7bee1f911b94400ff5bef2f2a8`;
the 2048 report digest is
`sha256:ea58486207c8dec5ae520efd4f2aa250c072593b100344bb31b9513c485d73a6`.

Before the 2048 run, the external contract
`selection_v3\v3_normal_only_contract.json` locked both candidate
configurations, the feature/calibration membership, the model/tool/recipe
digests, and the per-category normal gates. Its digest is
`sha256:224fba9878e087b36e553c69e27c42703da021929aee0f058f7b7f86d05af507`.
It permits no threshold or original-tuning P95 increase versus the 1024
reference, caps the augmentation P95-minus-original and paired delta P95/max
at 0.05, and prioritizes the smallest worst paired augmentation delta P95.

| Candidate | Worst paired augmentation ΔP95 | Worst threshold | Mean threshold | Selector result |
| --- | ---: | ---: | ---: | --- |
| V3 1024 | 0.02746 | 0.51143 | 0.42507 | Locked |
| V3 2048 | 0.03338 | 0.30447 | 0.27817 | Eligible, not selected |

The JSON-only selector wrote
`selection_v3\v3_normal_selection.json` with digest
`sha256:1adb386b097bce7b4699d2f288d7f19ecffe78d5a8fe83c568cfd05d1f266c31`.
Both candidates passed every frozen gate. V3 1024 was selected because its
worst paired normal perturbation ΔP95 was lower; the fixed lexicographic
objective deliberately ranks that robustness measure before lower thresholds.
This is a normal-only research configuration lock, not evidence of better
anomaly detection. It did not access, score, or report the current blind set.

## Physical readiness audit (2026-08-09)

The offline research result was not used as a substitute for a physical-device
qualification. A read-only check of the canonical PhoneCV topology found the
engineering Web (`4173`), API (`4174`), real PhoneDINO analyzer (`8082`),
recipe binding, and Tailscale HTTPS path reachable. The fixture-only port
`8080` was not used.

The overall engineering gate remains **not ready**: the canonical
`test-engineering-services.ps1` exits non-zero because both local and remote
Capture Health endpoints are `DEGRADED`. Their external probe is
`NOT_CONFIGURED`; the approved `CAPTURE_HEALTH_PROBE_TOKEN` and a registered
readiness-watchdog task are absent. This is intentionally not auto-remediated:
it requires an approved machine secret and an authorized external state change.

No admissible physical qualification cohort is present. Capture Health records
zero independent captures and zero traceable reference masters; its stored
results lack usable capture-source provenance and geometry-calibration
eligibility. The current engineering readiness reports
`ENGINEERING_REAL_DINO`, but also `simulation: true` and
`productionAuthorized: false`, so it must not be presented as production or
physical qualification. The external MVTec subset and its derivatives remain
offline research only.

Before a physical bake-off or qualification can begin, freeze the deployed
source/profile/artifact digests with a real-capture manifest, then provide the
controlled normal/anomaly/capture-reject cohorts, QR + ChArUco traceable
reference-board evidence, native-still attestation, and the defined MSA and
feature-ladder captures. An independently held blind partition must remain
untouched until configuration is locked.

## Next boundary

The offline software deliverables for this MVTec iteration are complete. Do
not tune V2/2,048 from the historical locked blind AUROC, defect deltas, or
masks. The later V3 1024 lock must likewise not trigger another use of the
current blind set: a new blind observation requires a newly frozen held-out
manifest. Any further configuration development must use only normal
FIT/tuning data or a separately frozen normal development envelope. Physical-
device work remains separate and requires controlled captures, QR + ChArUco
reference-board gates, context anchors, native capture attestation, fault
injection, and MSA. MVTec images cannot prove those requirements.

## Working-tree boundary

Research source changes are committed independently. Pre-existing unrelated
changes in `pyproject.toml`, `docs/offline_model_bakeoff.md`,
`src/phone_dino/bakeoff.py`, and `tests/test_bakeoff.py` remain untouched.
