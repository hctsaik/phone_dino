# MVTec AD research improvement work log

**Status:** implementation, one historical locked-configuration blind
observation, and three later normal-only configuration locks completed on
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

## V3-R4 four-variant normal-only configuration lock (2026-08-09)

The V3 recipe was held unchanged and regenerated as a deterministic
four-seed replication package at
`camera_capture_v3_r4\augmentation_manifest.json`. It has 768 normal-only
derivatives: 576 FIT and 192 threshold-tuning images (256 per category), with
four variants for every normal parent and `BLIND_ORIGINAL_ONLY`. The package
manifest digest is
`sha256:d7ad6525cfce72cb2776dd2b287c926652f90aa5ae23a029f8e5a3c8b44c5a11`;
the unchanged V3 recipe digest is
`sha256:5cf5b83ac58251b350d1fb2c01c6ac07e6fe624bf5e47ae0f5f6da7c8ee887d4`.

The schema-1.4 reports bind `variantId` in feature membership and every
tuning score. The schema-1.2 contract fixes four variants per parent and a
0.05 P95/max paired-delta cap for every variant, in addition to the existing
aggregate normal-only gates. It was frozen before the 2,048-prototype run at
`selection_v3_r4\v3_r4_normal_only_contract.json`, digest
`sha256:c514f6e6100dbd97dd0a12e8c3fb8a2b97287118623df83dcf6877aaba083b5a`.

Both reports have 960 normal feature inputs, 240 tuning calibration scores,
zero blind/anomaly feature inputs, `blindReporting.state: NOT_RUN`, and no
pixel metrics. The 1,024 reference had 960 cache misses and took 223.26 s;
the 2,048 candidate reused all 960 verified entries and took 59.21 s.

| Candidate | Report digest | Worst paired delta P95 | Worst per-variant delta P95/max | Worst threshold | Mean threshold | Selector result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V3-R4 1024 | `sha256:e2116c21a9c35de2e4b52c968487daac0005bfbdeba946ea35f1f3c321e0644e` | 0.03686 | 0.04015 | 0.32436 | 0.29960 | Eligible |
| V3-R4 2048 | `sha256:06c55c1a87ed820b192e8921e2dae71e8c9ed3b0aa7ac3231cd30746efa50dbd` | 0.02717 | 0.03305 | 0.26777 | 0.25927 | Locked |

The JSON-only selector wrote
`selection_v3_r4\v3_r4_normal_selection.json`, digest
`sha256:ef4e1d38a9b75c8724d2dcf74dd5de041a1ffb03b489e43fb2a3279a60307566`,
and selected `v3-r4-2048`. Both candidates passed every aggregate and
per-variant gate. This is a normal-robustness configuration lock only; it does
not prove anomaly detection, create a production threshold, qualify a physical
device, or authorize another use of the current blind set.

## V4 JPEG 4:2:0 four-variant normal-only configuration lock (2026-08-09)

V4 is a deliberately single-factor extension of V3: it keeps the exact V3
geometry, photometry, off-axis lens shading, JPEG quality range, and V2-based
sampling anchor, while changing only the encoded JPEG chroma sampling to
4:2:0. The committed implementation `de55a73` uses a closed V4 recipe,
reopens every JPEG to attest RGB components, sampling factors, non-progressive
coding, and the quantization tables derived from its sampled quality. It also
rejects any nominal-but-non-`good` FIT/tuning parent before augmentation or
iteration.

The fresh external package at
`camera_capture_v4_jpeg420_r4\augmentation_manifest.json` was generated from
a clean detached `de55a73` worktree. It contains 768 normal-only derivatives
(576 FIT, 192 threshold-tuning; 256 per category) with exactly four variants
per parent, `BLIND_ORIGINAL_ONLY`, no masks, and no blind inputs. Its declared
manifest digest is
`sha256:f40cafb298bd2ba5295780124f2995ec12dda5aeb5acdf303049756ea7323dc3`
(file digest
`sha256:12821667ab5f0851d6fb275cb9a21887cfedc081a385269199611b0e9a7ba2e6`);
the locked V4 recipe digest is
`sha256:5a812712609b48bd63321403eaadc4737ff1a0732a8177d35b169b0ef20a451d`.

The V4 reference report was normal-only from a cold cache (960 misses,
286.68 s); the predeclared 2,048-prototype candidate reused all 960 verified
entries (62.42 s). Both reports have 960 normal feature inputs, 240 tuning
calibration scores, 48 original-only reported tuning scores,
`blindReporting.state: NOT_RUN`, zero blind/anomaly feature inputs, and no
pixel metrics. Before running the candidate, the external schema-1.2 contract
`selection_v4_jpeg420_r4\v4_r4_normal_only_contract.json` fixed the two exact
candidate configurations and the four per-variant P95/max gates. Its declared
digest is
`sha256:eaf688b6ea960a20a8dd2dc9529bb53bbecf2f693652aace8aee900bb2be08ad`.

| Candidate | Report file digest | Worst paired delta P95 | Worst per-variant delta P95/max | Worst threshold | Mean threshold | Selector result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V4-R4 1024 | `sha256:ec87a3bb7e2110ae9ac2282f172f48782084271ef49a51d5b4ca04fa9dbc22c2` | 0.03380 | 0.04057 | 0.32273 | 0.29833 | Eligible |
| V4-R4 2048 | `sha256:fa78513e4bb3b9dd9ceec721d722f1d4096e75f2e52f9c28ffb7727894bd72de` | 0.02482 | 0.03323 | 0.26674 | 0.25937 | Locked |

The JSON-only selection result at
`selection_v4_jpeg420_r4\v4_r4_normal_selection.json` has file digest
`sha256:1c99fc61841a2f087191d7df746093102eb4162f24c618c95a471eb5346355fc`
and selected `v4-r4-2048`; both candidates passed every frozen gate. This is
only a V4-envelope normal-robustness configuration lock. V3-R4 and V4-R4 use
different derivative bytes and frozen augmentation identities, so the strict
selector intentionally does not compare or promote one envelope over the
other. No frozen blind image, anomaly label, defect result, or mask was read
for this decision.

## V5 fixed-Q95 implementation boundary (2026-08-09)

The next research implementation is intentionally narrower than adding a new
lighting, blur, glare, or noise effect. The schema-1.2 V5 recipe preserves the
full V4 parameter stream: its V2 sampling anchor, 95--98 sampled JPEG quality,
geometry, photometry, off-axis lens shading, and RNG call order are unchanged.
The sole output difference is a fixed JPEG Q95 / 4:2:0 coding profile. Every
V5 record preserves the sampled `parameters.jpegQuality` for reproducibility,
then separately records `outputJpegQuality: 95`; RGB component layout,
sampling factors, non-progressive coding, and the locked Q95 quantization-table
digest are reopened and checked both during generation and before scoring.

The Q95 table digest
`sha256:f67e35fd0dcd2fd9f999077e2aae8560e6327a8477c45427f6ea2e0a224cd187`
is an explicit fail-closed encoder requirement. A Pillow/libjpeg change that
emits different Q95 tables will stop package generation/loading rather than
quietly changing the study. This is only a generic research coding-profile
probe; an engineering JPEG-header observation does not calibrate a phone or
qualify a physical capture path.

The augmentation manifest format is now `1.3` because it adds the mandatory
`outputJpegQuality` record binding. Existing V4-R4 schema-`1.2` artifacts are
locked historical evidence and intentionally cannot be consumed by the new
loader: they retain their `de55a73` generator identity and must be inspected
with that pinned worktree, or regenerated as a new envelope. They must not be
silently upgraded or ranked against V5.

The V5-R4 package was generated from clean detached `a252de7` at
`camera_capture_v5_jpeg420_q95_r4_retry1\augmentation_manifest.json`. It has
768 derivatives (576 FIT and 192 threshold-tuning) with exactly four variants
per good nominal parent, no masks, no blind records, and the following frozen
identities: declared manifest
`sha256:c5de5539313a75eb7db238a548aff35010bf0f2f347851d3beff43ba99b9596d`,
file `sha256:fbe5e5c3a1c0f301e1539c48b1f6368143b6a6cad03e918cf950c19692d167dc`,
and recipe
`sha256:00a970a63994e25d43adb7cf76fc66f1ffb9dbcab7dba26ebdf5fd89b84831dd`.
The loader revalidated all 768 images before either scoring run; every output
was Q95 / 4:2:0 with the locked table digest.

The 1,024-prototype reference used a fresh V5 cache (960 misses, 311.15 s) and
the predeclared 2,048 candidate reused its 960 verified entries (56.39 s).
Both are normal-only: 960 feature inputs, 240 tuning calibration scores, 48
original-only reported tuning scores, `blindReporting.state: NOT_RUN`, zero
blind/anomaly feature inputs, and no pixel metrics. Before the candidate ran,
the external contract
`selection_v5_jpeg420_q95_r4\v5_r4_normal_only_contract.json` bound its exact
membership, feature identity, V5 manifest/recipe, two configurations, and
four per-variant `0.04` P95/max gates. Its declared digest is
`sha256:8e27685cc1ab3deb62bd6b11e021c367fef882b060581d74c178ba7f802b33ea`
(file `sha256:05aab19db6b54288a988df409fa1c55fe2eac0cc92287658d54931aaca103a92`).

| Candidate | Report file digest | Worst paired delta P95 | Worst per-variant delta max | Worst threshold | Mean threshold | Selector result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| V5-R4 1024 | `sha256:bc3ee27fb541c4dff7069e46f4033d81f6364f0b24e1b934b8225eca7601f171` | 0.03209 | 0.03940 | 0.32250 | 0.29853 | Locked |
| V5-R4 2048 | `sha256:4ca3dfc657628ed0d95556f2ce6ec4e3be118e32841f5bbf91a959640730cf82` | 0.02556 | 0.03516 | 0.27138 | 0.25931 | Rejected gate |

The JSON-only result at
`selection_v5_jpeg420_q95_r4\v5_r4_normal_selection.json` has file digest
`sha256:b667244c4ed24e230d3f6d1fe7f5741c5382f2334d6763c227e241d802536d25`
and selected `v5-r4-1024`. The 2048 candidate was rejected—not promoted—because
its tile threshold increased by `0.000540912151337` above the frozen zero
increase limit. That rejection is retained instead of loosening the contract
after seeing the data. This is a V5-envelope normal-robustness lock only. No
V3/V4/V5 cross-envelope selection, new blind report, or anomaly/mask access is
authorized.

## Fresh normal holdout cohort V1 (2026-08-09)

The existing subset's normal images had already appeared in V3--V5 research,
so they were not relabelled as a new holdout. A separate external-only cohort
was instead acquired from the pinned Voxel51 MVTec mirror revision
`30a183a3b96e3aef953f230784b123b719b09d97`. The acquisition mechanically
filtered only `train` / `good` metadata records with no mask field for
capsule, metal_nut, and tile. It made per-file requests only: each redirect's
revision, raw LFS digest, and byte count were checked before download, then
the downloaded bytes and image decoder were checked again. No category archive,
test image, anomaly image, or ground-truth mask was downloaded.

The resulting external package is
`fresh_normal_holdout_v1` with the following frozen artifacts:

| Artifact | Declared digest | File digest |
| --- | --- | --- |
| Historical normal-usage ledger | `sha256:38bedae2c856bdbb73d16863152bd9b5581b99dc74157648fdeb1cb8be430c10` | `sha256:fffa4b335044ecb10e749d67f195de727a639c73b3d8752d518f4ef9c084c3fc` |
| Fresh normal candidates | `sha256:4ccabfe0deac5c090868541447a1d9c100ca23df78c3cf6f64e9c82ba9aa2b94` | `sha256:a36c23d722a7b1bc9c909654e434392cccd587a0044a80d2e67c2ad17ac7a5e6` |
| Rehashed normal source pool | `sha256:7fa8a459864f56b8053437dadeaa1bbbc860441e0356f60063784d195e19bd68` | `sha256:ceaf874d67eea5d13ab9e10bec5ace38aea084820a2f58511a4670baf72379e6` |
| Predeclared holdout plan | `sha256:1acd2578ff95f055120274a651722a15ee2c9fccb7b999c4f92198505e0b5e24` | `sha256:5e8121d1d5b14b74687c8e975583a03e9965ea93ed38c01c5cf743c94eca5e53` |
| Allocated normal holdout | `sha256:51a359f5d579a99321dc33687fecc6d9a8db92fb7f921960bbb6898c23e2e74e` | `sha256:0034e045001787a6ce35042701cb470a97c03ff72117311ff7525fd5d9106b18` |

The held-out manifest revalidated all 477 source bytes and contains 144 FIT,
48 threshold-tuning, 96 normal-selection, 96 normal-confirmation, and 93
untouched reserve records. Per category, its plan reserves 48/16/32/32
FIT/tuning/selection/confirmation inputs, then freezes the remaining
27/28/38 sources as reserve. The source metadata has no capture/session
information, so its grouping claim is deliberately only `EXACT_CONTENT_ONLY`;
it prevents identical-byte reuse but does not claim physical acquisition-group
independence.

This cohort is non-authoritative and non-production. No DINO candidate, score,
threshold, blind report, anomaly label, or mask was used to choose its
partition. The next standalone evaluator may tune only on its development
partition, then make one observation-only normal-confirmation measurement; it
must not use confirmation data to select the next augmentation.

## Phase-safe normal evaluator boundary (2026-08-09)

The fresh cohort now has a dedicated phase-safe reader. It validates the
closed normal-holdout manifest and opens only the requested normal image
partitions. It never opens the source pool, historical ledger, allocation plan,
or pinned public `samples.json` inventory; the latter includes public test and
anomaly metadata and remains restricted to the acquisition/freeze audit.

The real external cohort was revalidated through this reader for `FIT` plus
`THRESHOLD_TUNING`: 144 FIT and 48 tuning originals were rehashed and decoded
under holdout manifest
`sha256:51a359f5d579a99321dc33687fecc6d9a8db92fb7f921960bbb6898c23e2e74e`.
No selection, confirmation, reserve, blind, anomaly, or mask input was opened.

The initial development augmentation will be a new standalone fresh-cohort
tool, not an adapter to the locked V3--V5 generator. It will derive bounded
camera/coding variants from FIT parents only; tuning, selection, and
confirmation remain raw normal originals. This keeps the new evidence chain
separate from historical recipe/module/cache identities.

## Fresh FIT camera augmentation V1 implementation (2026-08-09)

The standalone V1 generator and validator are now implemented, with a closed
generic-prior recipe. Its named SHA-256 substreams independently sample narrow
geometry, photometry, low-frequency lens shading, and conservative
signal-dependent/read noise; output is pinned to non-progressive JPEG 4:2:0
Q95 including its full quantization-table digest. It intentionally excludes
blur, glare, occlusion, crop, synthetic defects, and capture-reject effects.

The package validator rehashes FIT parents and validates every child against
its exact parent/variant parameters, JPEG coding header, source/tool hashes,
and deterministic byte-for-byte re-render. Its focused test fixture proves
that corrupt tuning, selection, confirmation, and reserve files are not
opened. The implementation is ready to generate a new external V1/R4 package;
no external derivative has been created at this point.

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

Do not tune V2/2,048 from the historical locked blind AUROC, defect deltas, or
masks. The later V3 and V3-R4 locks must likewise not trigger another use of
the current blind set: a new blind observation requires a newly frozen held-
out manifest. Any further configuration development must use only normal
FIT/tuning data or a separately frozen normal development envelope. Physical-
device work remains separate and requires controlled captures, QR + ChArUco
reference-board gates, context anchors, native capture attestation, fault
injection, and MSA. MVTec images cannot prove those requirements.

## Working-tree boundary

Research source changes are committed independently. Pre-existing unrelated
changes in `pyproject.toml`, `docs/offline_model_bakeoff.md`,
`src/phone_dino/bakeoff.py`, and `tests/test_bakeoff.py` remain untouched.
