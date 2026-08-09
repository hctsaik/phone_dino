# Fresh normal holdout protocol

This protocol creates a new, external-only MVTec AD normal cohort for research
configuration work. It does not authorize production use, physical
qualification, defect-detection claims, or a blind-data promotion.

The existing local subset cannot supply this cohort. Its FIT, tuning, and
historical blind-normal records have already been observed during V3--V5
research. Relabelling any of them as a holdout would be retrospective reuse,
not independent evidence.

## Boundaries

The implementation is deliberately separate from the legacy augmentation,
iteration, and normal-candidate selector tools. It does not modify or consume
their feature caches, model outputs, blind scores, masks, or DINO runtime.
This preserves the locked V3--V5 provenance contracts.

After this cohort is frozen, phase-specific tooling must use
`load_evaluation_safe_normal_holdout_inputs()` rather than the full
freeze-audit loader. The phase-safe reader validates the closed
`normal_holdout.json` contract and opens only the requested normal images; it
does not open the source pool, history ledger, plan, or public `samples.json`
inventory. The latter contains public test/anomaly metadata and is permitted
only during acquisition/freeze audit, never during FIT augmentation, tuning,
selection, or confirmation.

The only public source accepted by the current schema is the pinned Voxel51
MVTec mirror revision `30a183a3b96e3aef953f230784b123b719b09d97`, using its
`samples.json` inventory with raw SHA-256
`sha256:dbbbb94cee2ddec28c1eef318733d07df4d59b9cc066e62e6aeef386c1db281d`.
The source-acquisition stage mechanically selects only records with:

- category `capsule`, `metal_nut`, or `tile`;
- `split == train`;
- defect label `good`; and
- no defect-mask field.

It must transfer files file-by-file from the pinned revision, never as an
archive. Before following
the download redirect it validates `X-Repo-Commit`, `X-Linked-ETag`, and
`X-Linked-Size`; it then streams the file and verifies the same raw SHA-256 and
byte count. It must never download an official category archive, because those
archives bundle test and ground-truth data.

The public metadata has no capture/session grouping. Therefore this protocol
uses `EXACT_CONTENT_ONLY`: each `sourceGroupId` is the exact raw-content digest.
It prevents byte-identical leakage, but does not claim that near-duplicate or
same-session leakage has been ruled out.

## Artifacts and order

All artifacts and image bytes must live outside the Git working tree. Every
output path is new-only; a partially failed path is not reused.

1. Create a `phone-dino.mvtec-ad-historical-normal-usage-ledger/1.0` from one
   or more existing `phone-dino.mvtec-ad-iteration-report/1.4` reports whose
   `blindReporting.state` is `NOT_RUN`. The parser consumes only their closed
   normal feature membership and rejects blind/anomaly inputs.
2. Use the file-level acquisition command to acquire and emit a closed
   `phone-dino.mvtec-ad-normal-source-candidates/1.0` artifact. It contains
   only the selected normal records, their pinned remote identities, and
   `EXACT_CONTENT_ONLY` groups.
3. Freeze a `phone-dino.mvtec-ad-normal-source-pool/1.0`. This rehashes and
   decodes each local image and rechecks it against the pinned metadata.
4. Before running any new DINO candidate, create a closed
   `phone-dino.mvtec-ad-normal-holdout-plan/1.0`. It predeclares exact group
   counts in five partitions: `FIT`, `THRESHOLD_TUNING`, `NORMAL_SELECTION`,
   `NORMAL_CONFIRMATION`, and `RESERVE_UNTOUCHED`. Every eligible group must
   be assigned exactly once.
5. Allocate the corresponding `phone-dino.mvtec-ad-normal-holdout/1.0`
   manifest. Its deterministic SHA-256 ranking is bound to the frozen pool,
   historical ledger, and plan. A historical source-hash match removes the
   entire group before allocation.

The current recommended public-data plan is 48 FIT, 16 threshold-tuning, 32
normal-selection, and 32 normal-confirmation groups per category, with all
remaining eligible groups frozen as reserve. The exact reserve counts are part
of the plan, never silently sampled.

## Later evaluator rules

The cohort-freezing implementation intentionally does not run DINO. A later
standalone evaluator may:

- fit prototypes from `FIT` only;
- lock thresholds from `THRESHOLD_TUNING` only;
- use `NORMAL_SELECTION` only under a predeclared normal-robustness contract;
- emit exactly one observation-only report for `NORMAL_CONFIRMATION`; and
- leave `RESERVE_UNTOUCHED` untouched.

Neither normal selection nor confirmation may select a winner, regenerate an
augmentation, alter a threshold, or open blind/anomaly data. Any further
iteration needs a newly frozen normal cohort or an explicitly separate
development envelope.

The first fresh augmentation package is likewise separate from V3--V5. It may
derive bounded camera/coding variants from `FIT` only. Raw
`THRESHOLD_TUNING`, `NORMAL_SELECTION`, and `NORMAL_CONFIRMATION` inputs must
remain unmodified normal images: synthetic variants must not influence a
threshold, candidate selection, or confirmation observation.

The exact FIT-only transform, package-validation, and command boundary are
specified in [the fresh FIT-only camera augmentation protocol](mvtec_ad_fresh_fit_augmentation_protocol.md).

## Commands

The commands below are examples; replace every external path with a fresh one.

```powershell
python tools/build_mvtec_ad_historical_normal_usage_ledger.py `
  --report C:\external\v5\patch_v5_r4_1024_normal_only.json `
  --output C:\external\holdout\historical_normal_usage_ledger.json

python tools/acquire_mvtec_ad_fresh_normal_sources.py `
  --source-metadata C:\external\source_metadata\samples.json `
  --ledger C:\external\holdout\historical_normal_usage_ledger.json `
  --source-root C:\external\holdout\source_bytes `
  --candidate-output C:\external\holdout\normal_source_candidates.json

python tools/freeze_mvtec_ad_normal_source_pool.py `
  --candidates C:\external\holdout\normal_source_candidates.json `
  --source-root C:\external\holdout\source_bytes `
  --source-metadata C:\external\source_metadata\samples.json `
  --output C:\external\holdout\normal_source_pool.json

python tools/create_mvtec_ad_normal_holdout_plan.py `
  --pool C:\external\holdout\normal_source_pool.json `
  --ledger C:\external\holdout\historical_normal_usage_ledger.json `
  --source-root C:\external\holdout\source_bytes `
  --source-metadata C:\external\source_metadata\samples.json `
  --quota capsule=48,16,32,32,27 `
  --quota metal_nut=48,16,32,32,28 `
  --quota tile=48,16,32,32,38 `
  --output C:\external\holdout\normal_holdout_plan.json

python tools/build_mvtec_ad_normal_holdout.py `
  --pool C:\external\holdout\normal_source_pool.json `
  --ledger C:\external\holdout\historical_normal_usage_ledger.json `
  --plan C:\external\holdout\normal_holdout_plan.json `
  --source-root C:\external\holdout\source_bytes `
  --source-metadata C:\external\source_metadata\samples.json `
  --output C:\external\holdout\normal_holdout.json
```

`acquire_mvtec_ad_fresh_normal_sources.py` is the only stage allowed to access
the public dataset inventory or network. The remaining commands consume only
frozen, normal-only external artifacts.
