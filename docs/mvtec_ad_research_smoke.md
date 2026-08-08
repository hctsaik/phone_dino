# MVTec AD research smoke benchmark

This optional, offline benchmark measures DINOv2 nearest-normal baselines
against a frozen subset of the MVTec AD data. It is strictly non-commercial
research work under MVTec AD's CC BY-NC-SA 4.0 licence.

It does **not** qualify PhoneDINO for production, a physical device, QR/ChArUco
co-location, metric measurement, replay prevention, or an equipment action.

## Dataset subset

The local subset is intentionally outside Git. `subset_manifest.json` records
the MVTec source revision, selected input and mask hashes, and the fixed split:

- `capsule`, `metal_nut`, `tile`;
- 48 normal `FIT` and 16 normal `THRESHOLD_TUNING` images per category;
- 16 normal and 32 anomalous `BLIND` images per category, with available
  official pixel masks.

The threshold is calculated only from the normal tuning split. Blind labels
are only used after scoring to report image AUROC and threshold exceedance
rates. This smoke set is deliberately small: it is a reproducible engineering
signal, not a published benchmark or a source of production thresholds.

## Run

```powershell
py -3.11 tools/run_mvtec_ad_smoke.py `
  C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\subset_manifest.json
```

The default is a whole-image (global) nearest-normal cosine-distance
baseline. To make small local changes visible, run the bounded patch-token
baseline on the exact same manifest and compare its report to the global one:

```powershell
py -3.11 tools/run_mvtec_ad_smoke.py `
  C:\code\claude\_media_out_of_repo\mvtec_ad\subset_v1\subset_manifest.json `
  --algorithm patch-knn `
  --max-prototypes 1024 `
  --top-k-patches 5
```

`patch-knn` pools normal FIT patch tokens per category, selects a deterministic
bounded memory bank, finds each query patch's closest normal patch, and scores
the mean of the five largest distances. It is a PatchCore-style research
baseline, **not** a PatchCore implementation or a production decision policy.
Its threshold remains the maximum score from normal `THRESHOLD_TUNING` images;
blind labels remain reporting-only.

The tool uses the locally pinned DINOv2 repository and checkpoint under
`runtime/models`; it neither downloads model weights nor calls the PhoneDINO
service. Its JSON report has `authoritative: false` by design.
