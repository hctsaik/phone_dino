# MVTec AD offline iteration and comparison protocol

The MVTec tools in this document are offline, non-commercial research only.
They do not connect to PhoneDINO's browser-facing path, service API, artifact
schema, QR/ChArUco process, physical measurement, or equipment decision.

## Safe iteration loop

1. Generate normal-only camera/lighting derivatives with the augmentation
   protocol. Only FIT and tuning normal records participate.
2. Run a candidate on the frozen originals plus those normal derivatives. The
   FIT memory bank and tuning threshold may include the derivatives; blind
   images must remain original.
3. Compare normal robustness, threshold inflation, cache behavior, and elapsed
   time with `--normal-only`. Lock the candidate configuration and all
   source/recipe digests.
4. Produce one observational blind report after that lock. Do not use its AUROC, masks, or
   per-defect deltas to iterate the same frozen blind set.
5. If new anomaly-guided work is needed, create a newly frozen development or
   blind manifest. Never silently reuse the current blind labels as tuning.

The tool records `NORMAL_ONLY_ITERATION_THEN_BLIND_REPORTING_ONLY` and an
explicit `blindAugmentedCount: 0` in every augmented report.

## Efficient patch iteration

`run_mvtec_ad_iteration.py` loads the same local DINOv2 weights and uses the
same `Resize(256) + CenterCrop(224)` preprocessing as the production embedder,
but its batching and cache exist only in the offline research tool. It streams
one feature array per input into an optional cache outside Git, processes
exact nearest-normal cosine distances in bounded prototype blocks, and records
batch/cache/timing/runtime provenance.

It retains the 16 × 16 nearest-normal patch-distance grid for blind samples.
After all image scores and thresholds are fixed, it transforms frozen official
masks through that exact DINO resize/crop geometry and reports research-only
pixel AUROC and AUPRO at 30% FPR. This localization output is not a PhoneDINO
ROI, defect proof, or production capability claim.

Example patch run using one normal derivative per FIT/tuning input:

```powershell
.venv\Scripts\python tools\run_mvtec_ad_iteration.py `
  C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\subset_manifest.json `
  --algorithm patch-knn `
  --augmentation-manifest C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\camera_capture_v1\augmentation_manifest.json `
  --feature-cache C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\feature_cache_v1 `
  --output C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\reports\patch_capture_v1.json
```

For a tuning-only iteration, add `--normal-only`. It neither loads nor scores
blind images and writes `blindReporting.state: NOT_RUN`; its per-category
`normalScoreMedian`, `normalScoreP95`, and `normalScoreMax` make threshold
inflation measurable without blind-label leakage. Augmented runs also split
the original versus derivative tuning distributions, so an apparent change
cannot be hidden by mixing two different normal populations.

Use a fresh `--output` path for every run. The command refuses output and
feature-cache locations inside this Git worktree, and never overwrites an
existing report.

## Strict comparison

To compare two fixed reports, join their identical original score membership
by both `caseId` and `sourceSha256`:

```powershell
.venv\Scripts\python tools\compare_mvtec_ad_reports.py `
  C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\reports\global_clean.json `
  C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\reports\patch_capture_v1.json `
  --output C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\reports\global_vs_patch_capture_v1.json
```

The comparison refuses different frozen manifest identities, score-case sets,
or source digests. It emits per-case, per-category, and blind per-defect
deltas, but deliberately has no winner/ranking field and declares
`COMPARISON_ONLY_NO_BLIND_BASED_MODEL_SELECTION`.
