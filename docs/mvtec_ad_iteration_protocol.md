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
but its batching and cache exist only in the offline research tool. It stages
new feature arrays for an optional cache outside Git, verifies extractor
provenance after inference, then publishes only verified entries. It processes
exact nearest-normal cosine distances in bounded prototype blocks and records
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

## Normal-only configuration lock

Iteration-report schema `1.4` adds `normalOnlyEvidence` and an exact candidate
configuration. The evidence contains score-free, stable-sorted feature,
calibration, and original-tuning membership lists plus independently checked
digests. A normal-only report must have zero blind and anomaly feature inputs,
only `THRESHOLD_TUNING`/`NOMINAL` score records, and no pixel metrics. Reports
that do not meet those conditions cannot enter a formal candidate comparison.

Every 1.4 report also emits a canonical `featureExtractor` object and its
digest. It pins the local DINO repository content tree, model weights,
entrypoint, exact preprocessing configuration, runner/production/loader/
augmentation-source modules, Python/Torch/Torchvision/Pillow/NumPy versions,
thread count, and relevant Torch backend settings. The runner rehashes this
identity after model loading and after feature extraction; a change fails the
run before a report can be written or a newly extracted cache entry can be
published. The report separates cache validation and cache-write timing.

Feature-cache schema `1.1` keys entries with that extractor identity and
stores every feature as a content-addressed `.npy` plus strict metadata. On a
cache hit it verifies the metadata, input identity, fixed DINO feature shape
and `float32` dtype, file digest, and finite values. A legacy or incomplete
cache entry is a miss, never an accepted feature. This makes a local model
repository, preprocessing, dependency, or feature-array change invalidate the
cache instead of silently mixing results between configurations.

For a multi-seed augmentation run, report identity and every tuning score
carry `variantId` (`null` for an original; a positive integer for a derived
image). The external contract fixes `augmentationVariantsPerParent`; the
selector requires every FIT and tuning parent × variant pair, validates its
parent/source/role/category/recipe linkage, and applies explicit P95/max
paired-score gates per variant as well as the aggregate gates.

After a reference normal-only report exists, freeze an external selection
contract (schema `1.2`) before running the remaining predeclared candidates.
Each candidate
entry binds its exact algorithm configuration (including patch-memory and
top-k settings), not just a friendly ID. The contract also binds the
source-manifest identity, model-repository/extractor/preprocessing/tool
digests, exact normal
feature/calibration/tuning membership, normal augmentation envelope,
reference-report digest, explicit per-category normal-robustness caps, and a
fixed lexicographic objective.

For a patch candidate, a frozen contract entry has this shape (all digests and
the contract self-digest must be concrete values in the external file):

```json
{
  "id": "v3-2048",
  "candidateConfiguration": {
    "algorithmId": "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1",
    "batchSize": 4,
    "memoryBankSelection": "DETERMINISTIC_EVENLY_SPACED_PATCH_SUBSET_AFTER_STABLE_PARENT_SORT",
    "maxPrototypePatches": 2048,
    "topKMostAnomalousPatches": 5,
    "prototypeBlockSize": 256
  }
}
```

```powershell
.venv\Scripts\python tools\select_mvtec_ad_normal_candidate.py `
  --contract C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\selection\v3_contract.json `
  --candidate v3-1024=C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\reports\v3_1024_normal_only.json `
  --candidate v3-2048=C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\reports\v3_2048_normal_only.json `
  --output C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\selection\v3_normal_selection.json
```

The selector is JSON-only: it refuses reports inside Git, never opens images,
masks, model weights, or PhoneDINO endpoints, and never starts a blind run. It
rejects any candidate with blind/anomaly score evidence, mismatched input,
calibration, candidate-configuration, or tool identity, missing
derivative-parent coverage, inconsistent normal summaries, out-of-range cosine
scores, duplicate report reuse, or a failed declared gate. It hashes the exact
JSON bytes it validates and emits either a research configuration lock or
`NO_ELIGIBLE_CONFIGURATION`; neither result proves anomaly performance or
authorizes production/physical use.

Do not compare an unaugmented report against an augmented report through this
selector: their normal input identities differ. Legacy `1.0`/`1.1`/`1.2`/`1.3`
iteration reports lack either the required normal-only evidence, full feature
extractor provenance, or formal variant membership. Preserve them as
historical audit artifacts, but regenerate the external package and rerun only
`--normal-only` after the current tool revision instead of retrofitting an old
report.

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
