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

## Fresh FIT camera augmentation V1/R4 package (2026-08-09)

The new external-only package
`fresh_normal_holdout_v1\fresh_fit_camera_v1_r4` has now been generated from
the frozen fresh holdout. It contains exactly 576 children: 4 variants for each
of 144 FIT parents, or 192 children per category (`capsule`, `metal_nut`, and
`tile`). Every record is `FIT` parent / `NOMINAL` / `good`; raw tuning,
selection, confirmation, reserve, blind, anomaly, and mask inputs were not
opened.

| Binding | Digest |
| --- | --- |
| Package declared digest | `sha256:aa7d976a430c01b20afa2dbd0a3af94bca2cd7adad42273df069749bb15e80ba` |
| Package file digest | `sha256:234c1a2f9a2d79127217bfab999b0ef5a48abf6fd1124365329003a5b67f5895` |
| Fresh holdout declared digest | `sha256:51a359f5d579a99321dc33687fecc6d9a8db92fb7f921960bbb6898c23e2e74e` |
| Fresh development identity | `sha256:1c19aca07d0efe4552921f1afc830962bbdff9629b91359c259bd23b14176a18` |
| FIT-parent identity | `sha256:8f57a1e69098313776a70fa70fbb941bcbb61185490193618a6f809d72c16418` |
| Closed recipe file | `sha256:d0c0575bff44feafeb8364fc32e5eebe30335291d0aff7056abb0e25b22869a7` |

The post-generation validator rehashed all FIT parents and package children,
verified each complete 4:2:0/Q95 JPEG coding profile, regenerated every child
from its parent/parameters, and matched all 576 byte strings exactly. This is
only reproducibility evidence for a generic normal FIT augmentation package;
it is not a threshold change, candidate selection, anomaly-detection result,
or production/physical qualification.

## Fresh normal development evaluator implementation (2026-08-09)

A standalone fresh-holdout development evaluator is now implemented. It shares
only the local DINO model/preprocessing semantics with PhoneDINO; it does not
import the legacy V3--V5 runner, cache, selector, report schema, or blind-set
logic. Its closed candidate configuration binds the patch-kNN prototype limit,
top-K, block size, batch size, and deterministic prototype-selection policy.

It accepts only raw FIT, validated FIT-only derivatives, and raw
threshold-tuning images. Per category it builds a bounded patch prototype bank,
locks the threshold to the maximum raw tuning score, and records normal-only
scores/provenance. The development entry point has no selection or confirmation
partition parameter, so it cannot open those bytes. The first predeclared
comparison will be otherwise identical 1,024- and 2,048-prototype candidates;
a selection contract will be frozen only after both development reports exist.

## Fresh normal development observations (2026-08-09)

Both predeclared candidates have completed a real local-DINO development run.
Each report passed its self-digest check, used the same feature-extractor,
feature-input, and raw-tuning identities, and contains exactly 768 normal
feature inputs: 144 raw FIT parents, 576 validated FIT-only children, and 48
raw threshold-tuning images. Its evidence records zero blind or anomaly inputs;
no selection, confirmation, reserve, mask, or anomaly bytes were opened.

| Candidate | Report file digest | Report declared digest |
| --- | --- | --- |
| `fresh-fit-v1-r4-patch-1024` | `sha256:4878f1b075769820ba21b4f2c7765209adcb7cf78a8d449f3fe8582a57e70748` | `sha256:80a3642b6ab0d687cfb894e66303f7a2bbbd6b8ea1f4b921449f72b08871e457` |
| `fresh-fit-v1-r4-patch-2048` | `sha256:0040b962b64a3c5517107103ba38285ac540c8d51091e974a1010cf905ea67e3` | `sha256:1b29f5f74da80be2fe12effef597e77a850e305e694eec60477cd60b843293d4` |

| Candidate | capsule raw-tuning max | metal_nut raw-tuning max | tile raw-tuning max |
| --- | ---: | ---: | ---: |
| 1,024 prototypes | 0.341442 | 0.344680 | 0.241221 |
| 2,048 prototypes | 0.240708 | 0.274566 | 0.234979 |

These maxima are locked raw-normal calibration observations, not defect
thresholds or production settings. The shared feature-input identity is
`sha256:a50609605afb27e1bb90da3d9e9a762ac9e93e03c9b1029dd5808c1366bb3dc2`,
the shared raw-tuning identity is
`sha256:834dcb781b41302cd673ad4e495f154a2b697ea122aeefa0e13eeb9f17742190`,
and the shared feature-extractor identity is
`sha256:0808d305d3d8fa500274574fac0001d2ec6ad7e17645be68bcf128f6a6b82bde`.
The next step is to freeze an explicit normal-selection contract before
opening the untouched `NORMAL_SELECTION` partition.

## Fresh normal selection contract frozen (2026-08-09)

The first fresh normal-selection contract is now frozen externally at
`fresh_normal_holdout_v1\selection_protocol_v1\fresh_normal_selection_contract.json`.
Its file was created by the JSON-only contract tool before any
`NORMAL_SELECTION` or `NORMAL_CONFIRMATION` image was opened.

| Binding | Digest |
| --- | --- |
| Contract file digest | `sha256:98f03fffd3a0bb8cf5c37a82a59fef3bda6e344558fb0b686a3bed30e99547d2` |
| Contract declared digest | `sha256:cb0123fd4dec344bacedc23f0bf9a64a7e8deeeb12ea4b11daf90348c2171d95` |
| Candidate-universe identity | `sha256:8b9a723373653252dd6fb635a5a22c76899d2677df9438dd2fb7d5487cf2d6eb` |
| Raw normal-selection membership identity | `sha256:e05fb962f6b0b2678e8eae7bbdb73449b4f9f190221f54052e9e4dd9a65640d4` |
| Raw normal-confirmation membership identity | `sha256:04f42f3f85459428e1c70ceb983fb586077a1b6d87437831748fe8f89e7611bd` |

It binds exactly the 1,024- and 2,048-prototype development reports, their
closed raw-tuning thresholds, their shared feature extractor, one FIT-only R4
package, and the complete path-free held-out normal membership. The explicit
per-category gates allow at most 12.5% above the frozen raw-tuning threshold,
P95 excess of 0.05, and maximum excess of 0.10. The objective is fixed before
selection: minimize worst and mean above-threshold rates, then worst and mean
P95 excess, then candidate ID. The next action is a separate one-time claim
and receipt; that action, not contract creation, will authorize the first read
of raw normal selection inputs.

### Contract registry hardening (before any claim or query read)

The initial non-consuming contract used a sibling claim-slot design. A security
review found that copying its bytes to another external contract directory
could create a second tool-mediated slot. No claim, receipt, selection query,
or confirmation query was created from that contract. The contract/claim
schema was therefore advanced to `1.1`: it now binds a cohort-wide external
`partition_access` registry derived from the frozen holdout file and partition
identity. A copied contract resolves to the same claim/receipt slots. The
initial schema-1.0 external contract is historical and intentionally rejected
by the current tool; a new schema-1.1 contract will be frozen before any held-
out normal image is opened.

The replacement schema-1.1 contract was then frozen at
`fresh_normal_holdout_v1\selection_protocol_v2\fresh_normal_selection_contract.json`,
still before any claim/receipt/query read. Its file digest is
`sha256:8d9aa3d04b1c2825e983cde22e177bf1bf771625fa7ca9af6ba956350af0509d`
and its declared digest is
`sha256:dcbf4ddd4ba9c1b3bb62b1f3af3bff301989dad7564c5ab0915fbccc28e33416`.
Its fixed selection slot is derived under the frozen cohort's
`partition_access` root with key
`sha256:ade1705aa03fdd84090208ac8d32cdcf317005a9c33867400bf89a0cf8527f03`;
the confirmation key is
`sha256:9d7c4d0d512dc0e37cf6008f8d2389c5cc845b7d5487cd01857a534a8b43c110`.

## Fresh normal selection observation and lock (2026-08-09)

The schema-1.1 contract's fixed selection claim, receipt, aggregate
observation, and JSON-only lock have now completed. FIT-only preflight
succeeded before the receipt was atomically committed; the observation then
used 720 prototype inputs (144 raw FIT plus 576 R4 FIT derivatives) and 96
raw `NORMAL_SELECTION` queries. Evidence records zero blind, anomaly, and mask
inputs and no persistent query cache.

| Artifact | File digest | Declared digest |
| --- | --- | --- |
| Selection claim | `sha256:71809f0b0bc42933a7121ed6734d961300d32d9a0a2f687fb91dfaf677c1172b` | `sha256:0868b52454ad7035316823a6226f88d554cec804b8db8821edb5f260583e4a9c` |
| Selection observation | `sha256:9d7196c882512af0fc350c35aa2040974f6c3fea3436f59c66de82be146b6cde` | `sha256:02bf7f6583f75d884cd477dd09d58a0e8a03e834bd5889416f2e7b40186690da` |
| Selection lock | `sha256:e21006612fbfc24f4921f453a0292fe63b8dd96a2eeeadf71c38fb42546e4b14` | `sha256:cf1ba55690aa8ab6220e5dff384d16b024d680ff54a53720e7f772f898ab01a7` |

The frozen gates were above-threshold rate <= 0.125, P95 excess <= 0.05, and
maximum excess <= 0.10, independently applied per category. The normal-only
selection metrics were:

| Candidate | Category | Above rate | P95 excess | Maximum excess | Gate outcome |
| --- | --- | ---: | ---: | ---: | --- |
| 1,024 prototypes | capsule | 0.00000 | -0.02878 | -0.01774 | pass |
| 1,024 prototypes | metal_nut | 0.06250 | 0.00550 | 0.01302 | pass |
| 1,024 prototypes | tile | 0.15625 | 0.06509 | 0.07573 | fail (rate and P95) |
| 2,048 prototypes | capsule | 0.00000 | -0.02153 | -0.01454 | pass |
| 2,048 prototypes | metal_nut | 0.03125 | -0.00557 | 0.00688 | pass |
| 2,048 prototypes | tile | 0.12500 | 0.07634 | 0.08838 | fail (P95) |

The lock therefore records `NO_ELIGIBLE_CONFIGURATION`, with no selected
candidate, no production promotion, and no automatic confirmation. A separate
confirmation-claim command was intentionally refused by the JSON-only lock;
no confirmation claim, receipt, or confirmation image observation exists.
These results describe normal robustness under this generic augmentation
envelope only. They do not establish anomaly-detection improvement or a
production threshold.

## Reserve-successor V2 pre-registration (2026-08-09)

The V1 selection partition is permanently consumed and cannot be retuned. No
new independently auditable source cohort is locally available: the remaining
parent source directory has no read audit or exclusive access control. The
next research stage is therefore explicitly limited to a
`TOOL_MEDIATED_UNCONSUMED_ONLY` successor envelope, not presented as an
independent or human-unseen validation cohort.

Only the 93 parent `RESERVE_UNTOUCHED` raw-normal records may be delegated to
the successor's development and selection phases. The 96 parent
`NORMAL_CONFIRMATION` records remain unopened and may only be used once after
an eligible successor lock. The frozen reserve allocation is 12 FIT, 4 raw
tuning, and 8 raw selection records per category, leaving 3 / 4 / 14 reserve
records unopened for capsule / metal_nut / tile.

The pre-registered V2 candidate universe has one raw-FIT baseline and three
generic camera-simulation candidates. It uses a true nested 1,024/2,048-patch
prototype selector and a limited top-K ablation (K=3 or 5). V2 separates
registration, illumination, and sensor/transport perturbations into three
deterministic FIT-only derivatives rather than combining every effect in every
variant. It retains fixed RGB JPEG 4:2:0 Q95, is explicitly generic rather
than device-calibrated, and excludes crop, flip, glare, occlusion, synthetic
defects, and capture-reject simulation.

The small-sample screen is frozen before selection: at most one of eight raw
selection scores per category may exceed the four-score raw-tuning maximum,
and both the P95 and maximum excess must be at most 0.05. No candidate can
change a threshold, gate, recipe, model, or confirmation state automatically.
The full boundary and required safety tests are in
[`mvtec_ad_reserve_successor_v2_protocol.md`](mvtec_ad_reserve_successor_v2_protocol.md).

### Successor seal and envelope frozen

The reserve-only foundation is now frozen externally after revalidating the
complete V1 JSON chain (holdout, schema-1.1 contract/claim/receipt/
observation/lock). These commands accept no source-image root and created no
image observation. Their digests are:

| Artifact | File digest | Declared digest |
| --- | --- | --- |
| Successor reserve seal | `sha256:207655336e21fdc67998fae420b5c1822d8eac6a2bef6896390634936a1aa44e` | `sha256:96198334e9a7fcfa356eea5f7ba8b9fe68d1b21eece5d0489d5d2f8bce56fbc0` |
| Successor allocation plan | `sha256:0c021d32b1b336111c338b6b1986bdee3792c88da599cfc9144621eabd954952` | `sha256:fe423b9fc7025ba6ca997314712390772dd131a34ed11dd60d5247aa7905cd2d` |
| Successor envelope | `sha256:a185b2f8c3ae92ea2680de11ab9eeed1142da55830fcc4473fb001dd47028787` | `sha256:8313a27a25b563ec6e875dacd2f87a0dbfd7bc6636a3b6ff10214fe6cb28398e` |

The envelope binds 36 FIT, 12 raw tuning, 24 raw selection, and 21 still
unopened successor-reserve records. Its partition identities are respectively
`sha256:0f0df94572a3de59221c504b56a620c3aae0c84b55583bdf89c6181083290bfc`,
`sha256:07b1d0f68940bba2eb65799d7a927fa2c7bf7407c73bb472cf542fb1592fa675`,
`sha256:0dc2f4d969bf6e1df4d0bf7bc567e483a160e6fb65c95e2274757dd9c9df6017`,
and `sha256:8e629392d77724ea56e4136b3c4c064675b1b98304b58bf8ed93e4d68c30c433`.
The parent confirmation partition remains explicitly
`PRESERVED_NOT_DELEGATED_NOT_OPENED` with 96 records. This remains an
exploratory, not-independent parent-reserve derivation.

### V2 FIT-only R3 package generated and re-rendered

The sealed successor FIT partition has now been consumed for the pre-registered
V2 augmentation package only. It contains 36 raw FIT parents and 108
deterministic derivatives: one each of `registration`, `illumination`, and
`sensor_transport` per parent. The generator and validator opened no successor
tuning, selection, remaining-reserve, or parent-confirmation image bytes.

| Artifact | File digest | Declared digest |
| --- | --- | --- |
| V2 R3 augmentation manifest | `sha256:db5a773fa5e837854aaec834c73bfd33c7586314ef78f923dd0324f83c7632d1` | `sha256:bbc62770511f9066a8fc1bb9d9da065666ad21ed3519294db9e5e09072c43e43` |
| Closed V2 recipe | `sha256:aad6e5fe929a1e75ee9fc874ce1c1708870ba8b9dbf1314637152bc3be74591f` | `sha256:aad6e5fe929a1e75ee9fc874ce1c1708870ba8b9dbf1314637152bc3be74591f` |

The package is bound to successor FIT identity
`sha256:0f0df94572a3de59221c504b56a620c3aae0c84b55583bdf89c6181083290bfc`
and the frozen successor envelope. It has fixed RGB JPEG 4:2:0 Q95 encoding
and passed a full byte-for-byte deterministic re-render validation of all 108
children. Its generation recorded commit `afc858e`; the repository worktree
was not clean solely because the separately preserved bakeoff files and
`pyproject.toml` changes remain user-owned and unstaged.

### V2 development reports frozen and audited

All four pre-registered V2 candidates completed in one in-memory DINO feature
run. The independent report audit recomputed every report/self digest,
calibration maximum, and category summary; it also checked the exact parent
chain, V2 R3 package, current feature-extractor identity, and zero
blind/anomaly/mask input counts. No successor selection, remaining-reserve, or
parent-confirmation image was opened.

| Candidate | File digest | Declared report digest | FIT prototype policy |
| --- | --- | --- | --- |
| `reserve-v2-raw-p2048-k5` | `sha256:7133e8447cd8b84cd296bbe8a7e46c8cd665d01388bfc3fdb42bb6922e8eb09d` | `sha256:285768da0f1b23fb152288001ece5d638ca6896dd009793763720dbafd6c1312` | 36 raw FIT feature inputs |
| `reserve-v2-r3-p1024-k3` | `sha256:d23c9e241e2ef8f3ca8af3aaab368c25a722b94d7e80bee3057b8fe71e27a0bf` | `sha256:a97a8a049463eea537b65d7a0d5d6e2076fdbd1d393c52bf73744746ce39de64` | 36 raw FIT + 108 R3 feature inputs |
| `reserve-v2-r3-p2048-k3` | `sha256:436a81d66ae5edd7b335e56a946d97b7909c7cd6c5fa4d4e7f7054a806eba709` | `sha256:17ab746b1e5a5c06ad66c8a9697e501895f915385b34e2999a5809a0b9962210` | 36 raw FIT + 108 R3 feature inputs |
| `reserve-v2-r3-p2048-k5` | `sha256:b2b108b032b1fc524bc54be92a052b9c692683e3cad7a78eb59f814334a3c92e` | `sha256:d3b6fd751e9e9780226f690acc529b2a9fe00862b5cae1b136970aadd6b464dc` | 36 raw FIT + 108 R3 feature inputs |

Every candidate used 12 raw tuning scores (4/category), never augmented for
calibration. The resulting per-category raw-tuning maxima were:

| Candidate | capsule | metal_nut | tile |
| --- | ---: | ---: | ---: |
| raw p2048 / K5 | 0.142542 | 0.182903 | 0.191811 |
| R3 p1024 / K3 | 0.194577 | 0.263123 | 0.242084 |
| R3 p2048 / K3 | 0.162358 | 0.237567 | 0.211751 |
| R3 p2048 / K5 | 0.149514 | 0.199732 | 0.194456 |

The reports record a non-clean Git worktree because unrelated user-owned
files were preserved. Their validated module/model/preprocessing hashes, not
that worktree flag, are the binding provenance.

### V2 pushed-Git evidence anchor and one-time selection lock

Before selection, the four frozen development reports were projected into the
closed [`mvtec_ad_successor_v2_development_evidence_ledger.json`](mvtec_ad_successor_v2_development_evidence_ledger.json)
and committed/pushed in `9506923`. The contract resolver fetches only the
canonical `origin/master` ref into a disposable bare repository, verifies a
regular Git blob and its ancestry, and reads raw blob bytes rather than the
checkout. This prevents local report/ledger/contract substitution under the
documented `PUSHED_GIT_AUDIT_ONLY` boundary.

| Evidence anchor | Value |
| --- | --- |
| Git commit | `9506923ee2ef90a5b69481bb6b38ec2d9b3cb2b5` |
| Git blob | `5cb438690edd5f3d5adffbbaf501acd2ff0b892f` |
| Raw blob SHA-256 | `sha256:6462347e4573b9454f1b38f2dfc835c493faaa510075cb411fbecfb67056864a` |
| Ledger declared digest | `sha256:d4b864ef3e7b0b1a41929f9e4e0ad9079b2fa8926f13c6a4d4640d276dc7d382` |
| Contract projection digest | `sha256:954873a0c68ce5dc9ce73f74aace241ee856660199c54882137fe4278cebe266` |

This is audit provenance, not a signed attestation: a party able to rewrite
the configured remote ref or the executing tool can replace the baseline. A
FIT/tuning-only deterministic replay would be required to independently prove
that scores were honestly computed before the ledger freeze.

The JSON-only contract and claim then froze one global selection slot. The
observer revalidated FIT/R3 inputs and the feature extractor, wrote and fsynced
the receipt, and only then opened the 24 raw successor selection images once
for all four candidates. It reported no blind, anomaly, mask, parent
confirmation, or remaining-reserve inputs, and retained no query cache.

| Artifact | File digest | Declared digest |
| --- | --- | --- |
| V2 selection contract | `sha256:2e5df713f349353cda1490de07968c4e4f33e5933cedf998d1e33f34242003cf` | `sha256:51160e906cde46b7a092cc7aae4d838eddd59d6e162dfd504291fdf6fd7edffb` |
| V2 selection claim | `sha256:dbd35e9ec883a6a549ce81add7e3efc3488337fea368cfa2285a0c7e96fb366d` | `sha256:ad1cb7dcefbdd1ea4663943ab849a576f8b0e48ecdad98d53741fa04df8fb0e9` |
| V2 selection receipt | `sha256:cb760b317a890e7a08f4ee19d4596bc5a1359a35e64892a2a81393a1b4a4c244` | `sha256:b2e4ebbdc6560b9bfd7fb36713ce34d228605f67bd0507826766e4147380d345` |
| V2 selection observation | `sha256:1018a4bca88bf9778ea66f6dbf85304b195052dbc147a2e8251aa775fbc1d6db` | `sha256:4b615e83afaa5b85882941b44e0e063d8ad72f313ee04f9d6ed657aa15a5184b` |
| V2 selection lock | `sha256:4fb93dbcd19729f9fd96fd6c14a97e78fbf1b55947faed627b7c3e04b0396347` | `sha256:518ac9c0124a14d30e2c5a39dc36a5134efe68df475be3956fbef051af48853c` |

The JSON-only lock is `NO_ELIGIBLE_CONFIGURATION`. All candidates failed the
pre-registered normal-selection gates, chiefly through the capsule tail; the
result is not an anomaly-detection conclusion.

| Candidate | Gate-relevant outcome |
| --- | --- |
| `reserve-v2-raw-p2048-k5` | capsule 4/8 above threshold, P95/max excess `0.071562`; metal_nut 2/8 and tile 2/8 above threshold |
| `reserve-v2-r3-p1024-k3` | capsule 4/8 above threshold, P95/max excess `0.104019` |
| `reserve-v2-r3-p2048-k3` | capsule 4/8 above threshold, P95/max excess `0.106630`; tile 2/8 above threshold |
| `reserve-v2-r3-p2048-k5` | capsule 4/8 above threshold, P95/max excess `0.072861`; tile 2/8 above threshold |

Accordingly no confirmation claim was created, no parent confirmation image
was opened, and the 21 remaining successor-reserve records remain untouched.
Do not loosen these gates retrospectively or automatically consume either
partition. Any future iteration needs a newly pre-registered envelope and a
clearer independence/evidence boundary.

### Synthetic-only augmentation harness and test (2026-08-12)

The separate [`synthetic-only protocol`](mvtec_ad_synthetic_anomaly_test_protocol.md)
was added as an engineering test harness, not a successor-selection retry. It
uses only normal bytes allocated to the sealed successor envelope's `FIT`
partition; it does not open `THRESHOLD_TUNING`, `NORMAL_SELECTION`,
`NORMAL_CONFIRMATION`, remaining reserve, blind, true-anomaly, or mask image
bytes. It does not create a claim,
receipt, observation, or lock in the V1/V2 selection registries and does not
change the `NO_ELIGIBLE_CONFIGURATION` result above.

Per category, the 12 FIT parents were deterministically split by immutable
source identity into 6 raw prototype parents, 2 raw calibration parents, and
4 query parents. Only the 12 query parents received three deterministic PNG
stimuli (`LOCAL_SCRATCH`, `LOCAL_SPOT`, `LOCAL_OCCLUSION`), for 36 children
total. All package children were byte-hashed and deterministically re-rendered
before the test read them.

| Artifact | File digest | Declared digest |
| --- | --- | --- |
| Synthetic-only package manifest | `sha256:789a9cd9289dd6d53573de99c61a0afe59c48fc69a1f6546e88a8c71d2ded4b7` | `sha256:a5d87ce8a1adb293a40d4f58556e2a2dbb2c31bebb3aac4acf183802aad3c5f6` |
| Closed synthetic recipe | `sha256:935588502d295c939b5e042f99b4e276f83e5d2bbe7136a968f5d68bffe7252a` | same file digest |
| Synthetic-only test report (`r2`) | `sha256:462b4599309c95456991e41ff19cc8fd24dcf54a94ae4c420eda639fb95dbde8` | `sha256:692292841d74e924d46f9bb618c850a9e7ec5a7824ff5e755d7d8b5bcd98d77d` |

The fixed threshold for each category was the maximum score of its two raw
calibration parents. The query result contains 12 raw-normal synthetic
negatives and 36 rendered synthetic positives. Its **synthetic-only** outcome
was:

| Category | Synthetic TP / FP / FN / TN | Synthetic precision | Synthetic recall |
| --- | --- | ---: | ---: |
| capsule | 12 / 1 / 0 / 3 | 0.9231 | 1.0000 |
| metal_nut | 12 / 1 / 0 / 3 | 0.9231 | 1.0000 |
| tile | 12 / 2 / 0 / 2 | 0.8571 | 1.0000 |
| aggregate | 36 / 4 / 0 / 8 | 0.9000 | 1.0000 |

This is strictly `SYNTHETIC_RENDERING_DISCRIMINATION_ONLY`; its report states
`realAnomalyPerformance: NOT_ESTIMATED` and forbids model selection, threshold
selection, production validation, and physical qualification. The samples are
small and paired by parent, so the displayed precision/recall says only that
this DINO setup responds to these particular programmatic overlays. It is not
a real-defect or physical-device precision/recall result.

An earlier `r1` external diagnostic report included host-suspend time in a
wall-clock field and is deliberately not used for this worklog entry. The
recorded `r2` report uses process CPU time (66.30 seconds total) and completed
with the same source/model identity and fixed test rule.

### Post-V1 synthetic-stimulus stress V2 (2026-08-12)

The separate [`synthetic-stress V2 protocol`](mvtec_ad_synthetic_stress_v2_protocol.md)
extends the engineering harness without reopening or modifying the V1 result.
It uses only normal parents allocated to the sealed successor envelope's `FIT`
partition, has no V1-report input, and opens no `THRESHOLD_TUNING`,
`NORMAL_SELECTION`, `NORMAL_CONFIRMATION`, reserve, blind, true-anomaly, or
mask image bytes. It creates no selection-registry artifact and cannot change
the locked `NO_ELIGIBLE_CONFIGURATION` outcome.

The same immutable 6 / 2 / 4 FIT split per category provided raw prototypes,
raw calibration parents, and raw query parents. Before the V2 package was
loaded, the test froze each category threshold as the maximum of its two raw
calibration scores. Only the 12 query parents were rendered as the fixed
3-family x 3-level matrix (`LOCAL_SCRATCH`, `LOCAL_SPOT`, `LOCAL_OCCLUSION` x
`SUBTLE`, `MODERATE`, `PRONOUNCED`), yielding 108 byte-hashed, deterministic
PNG stimuli. Package validation re-rendered every child before scoring.

| Artifact | File digest | Declared digest |
| --- | --- | --- |
| Synthetic-stress V2 package manifest | `sha256:f79f13e26fd7cafba1a17756530622103d6d6e07d0d32a476ddc59d814e20016` | `sha256:9e2f6db244e76ed4bff43c6771c541cd0ad11ca8b8209fbeb5c8c9e31e8fd73f` |
| Closed synthetic-stress V2 recipe | `sha256:3d258a09d12d5510b16c43cdbe36bc2c65be3f1558662cb21b45504c753bc72a` | same file digest |
| Synthetic-stress V2 response report | `sha256:2d580769eddd0db2d8044049a9a71b95a7551d807c4487d25ee83e9d685672fd` | `sha256:9a98570d6977a27c6909b3496c3e9282a91df1d0dcec1209fa5a8530760aaea1` |

This is a response-only observation, not a classifier evaluation. Of 12 raw
query normals, 4 were above their frozen raw-calibration threshold (33.33%).
Of 108 synthetic stimuli, 97 were above it (89.81%); 104 / 108 paired stimulus
scores increased relative to their raw parent, with mean child-minus-parent
score `0.261049`. Per category the synthetic-stimulus response was 32 / 36
for capsule, 30 / 36 for metal_nut, and 35 / 36 for tile. By render level it
was 27 / 36 subtle, 35 / 36 moderate, and 35 / 36 pronounced.

The r1 report is explicitly `SYNTHETIC_STIMULUS_RESPONSE_ONLY` with
`realAnomalyPerformance: NOT_ESTIMATED`; it deliberately contains no
TP/FP/FN/TN, precision, recall, F1, AUROC, AP, V1-versus-V2 comparison, or
promotion result. A later audit found that r1 did not serialize the protocol's
additional `realPrecisionRecall: NOT_ESTIMATED` and
`evidenceClass: SYNTHETIC_ENGINEERING_ONLY` fields. r1 is therefore retained
only as a schema-2.0 historical engineering observation, not the current
contract-conformant record; it will not be overwritten. The hardened r2
reissue will add those fields and a strict JSON-only verifier. Both r1 and r2
forbid model, algorithm, hyperparameter, threshold, or package selection;
production validation; and physical qualification. They show only this frozen
DINO configuration's response to paired programmatic overlays, not observed
physical defects.

### Hardened V2-r2 reissue and V3 disposition (2026-08-12)

V2-r1 was retained unchanged as historical schema-2.0 evidence. A separately
generated V2-r2 package and schema-2.1 response-only report then re-ran the
same fixed stimulus design with a sealed, cache-free DINO snapshot, an
externally retained snapshot-manifest pin, strict score/summary reconciliation,
and metadata-only package binding. It remains an engineering response
observation, not an independent confirmation or a real-defect evaluation.

| Artifact | File digest | Declared digest |
| --- | --- | --- |
| V2-r2 synthetic-stress package manifest | `sha256:0d4c3d0825fc4f9b25e921c47e0bf33a3fb5615896c2ad8ff511317d64e56511` | `sha256:3fb95c6f104ae8a7bc14f1f6574725ac11753372b0b5e0833b37024deb71557c` |
| V2-r2 response report | `sha256:f745e449f2e770dc68c9424dddff8a5ab8d9a7bfcce805dcd7b1c1ca4ed14625` | `sha256:40f860d82853585d5380e19508f870da89c8890280e81bf307dcc95200bee435` |
| Sealed DINO snapshot manifest | `sha256:7f4579534a9c30263212c05b29ffc7f9c180e65495d5202573b95058d11892b2` | `sha256:d3b8614f2ddc97cf507c29518537232bfa3ad302f058be0d862ba71edee53dea` |

The sealed snapshot used repository source digest
`sha256:e3ba222bcc948f73a43e1f37320f64c209f5b1f05e682b8a1f4b7a184bfc4015`,
weights digest
`sha256:b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`,
and a report-bound snapshot-guard digest
`sha256:696718900f71fd9319c1c1b4a74d02a5916616b3f94e219a50d94e9e1d030bb7`.
The legacy evaluator's model-directory digest is a different named algorithm
and is deliberately not treated as equal to the sealed repository digest.

The r2 response was 4 / 12 raw query normals and 97 / 108 synthetic stimuli
above their frozen raw-calibration thresholds. The paired child-minus-parent
mean was `0.261049` (104 / 108 positive deltas). The report explicitly records
`realAnomalyPerformance: NOT_ESTIMATED`,
`realPrecisionRecall: NOT_ESTIMATED`, and
`evidenceClass: SYNTHETIC_ENGINEERING_ONLY`; none of these values is precision,
recall, or real anomaly performance.

The planned V3 nuisance-control observation was **not run**. During a
preflight, an erroneous recursive byte-hash command was issued against the
external cohort root. Its output cannot establish the exact set of files it
read, so the strict untouched-image boundary for that cohort can no longer be
asserted. No V3 receipt, report, or selection artifact was created, and this
cohort will not be used to claim an independent V3 control result. A future
strict V3 observation requires a newly controlled source cohort; the protocol
now explicitly forbids recursive cohort discovery or hashing during preflight.

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
