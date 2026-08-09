# Fresh normal-holdout evaluator protocol

This evaluator is a new offline research path for the fresh normal holdout. It
does not import the historical V3--V5 iteration runner, cache, selector, or
report contracts. It has no PhoneDINO runtime, production threshold,
physical-qualification, PASS/FAIL, or equipment-control role.

## Development observation

Each development report evaluates exactly one predeclared patch-kNN candidate.
It opens only:

- raw `FIT` normal parents;
- their validated fresh FIT-only augmentation children; and
- raw `THRESHOLD_TUNING` normal images.

Per category, raw and augmented FIT patch tokens form the prototype bank. A
deterministic evenly spaced subset is selected when the bank exceeds the
candidate's `maxPrototypePatches`. Only the raw tuning scores set the fixed
threshold, using their maximum. No tuning derivative is accepted.

The candidate configuration is closed and records every score-affecting knob:
algorithm ID, stable memory-bank policy, prototype limit, top-K patch score,
prototype block size, and inference batch size. The evaluator captures local
model repository/weights and source identities before loading DINO, after
loading, and after feature extraction; any identity change fails the run.

It writes a new external
`phone-dino.mvtec-ad-normal-holdout-development-report/1.0` report containing
normal-only membership, thresholds, tuning scores, normal score summaries, and
source/runtime provenance. It does not contain query image paths, feature
arrays, patch grids, masks, anomaly labels, AUROC, pixel localization, or a
winner.

## Command

The following is a development-only example for one candidate. The output must
not already exist and must stay outside the Git worktree.

```powershell
.venv\Scripts\python.exe tools\run_mvtec_ad_normal_holdout_development.py `
  --holdout C:\code\claude\_media_out_of_repo\mvtec_ad\fresh_normal_holdout_v1\normal_holdout.json `
  --source-root C:\code\claude\_media_out_of_repo\mvtec_ad\fresh_normal_holdout_v1\source_bytes `
  --augmentation-manifest C:\code\claude\_media_out_of_repo\mvtec_ad\fresh_normal_holdout_v1\fresh_fit_camera_v1_r4\augmentation_manifest.json `
  --recipe tools\mvtec_ad_fresh_fit_camera_recipe_v1.json `
  --candidate-id fresh-fit-v1-r4-patch-1024 `
  --max-prototype-patches 1024 `
  --top-k-most-anomalous-patches 5 `
  --prototype-block-size 256 `
  --batch-size 4 `
  --output C:\code\claude\_media_out_of_repo\mvtec_ad\fresh_normal_holdout_v1\development_reports\fresh-fit-v1-r4-patch-1024.json
```

The only initially predeclared comparison is the same configuration with
`maxPrototypePatches` set to `2048`. Both development reports must exist and
be validated before a selection contract can be frozen. Their tuning results
may guide development analysis, but they must not cause any read of
`NORMAL_SELECTION` or `NORMAL_CONFIRMATION`.

## Next immutable boundary

Before any selection image is opened, a separate contract will bind the fresh
holdout identity, one augmentation package, both candidate configuration/report
identities, model identity, explicit gates, and lexicographic objective. The
selection evaluator will atomically burn a consumption receipt before opening
raw `NORMAL_SELECTION`; it will score all frozen candidates in one observation.
The JSON-only selector may lock a research configuration but cannot promote it
or start confirmation. Confirmation likewise requires an explicit one-time
claim and only observes raw `NORMAL_CONFIRMATION`.

The contract, claim, receipt, and confirmation boundary are specified in
[the fresh normal selection and confirmation protocol](mvtec_ad_fresh_normal_selection_protocol.md).
