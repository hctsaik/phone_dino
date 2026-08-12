# Recovery V4 source foundation

Recovery V4 replaces the exposed `fresh_normal_holdout_v1` lineage with a
separate, predeclared source foundation. It is offline MVTec research tooling
only. It does not make a production decision, establish real-world precision
or recall, or establish a human-unseen/ACL-isolated boundary.

The code enforces tool-mediated containment. A genuinely independent
same-privilege boundary still requires an externally provisioned broker and
access-control/audit policy.

## Closed allowlist

The tracked machine-readable policy is
[`tools/mvtec_ad_recovery_v4_source_allowlist.json`](../tools/mvtec_ad_recovery_v4_source_allowlist.json).
It contains exactly 32 `train/good`, unmasked image paths for each of
`bottle`, `cable`, and `hazelnut` (96 total), at MVTec mirror revision
`30a183a3b96e3aef953f230784b123b719b09d97`.

For each eligible metadata row, the rank input is UTF-8 bytes of exactly:

```text
PHONE_DINO_STRICT_V3_FRESH_COHORT_V1|30a183a3b96e3aef953f230784b123b719b09d97|{category}|{sourceRemotePath}
```

The policy stores `metadataRankSha256` for every record. Rows sort by
`(rankDigestHex, sourceRemotePath)`, then use ranks 1–32. Per category the
fixed roles are 6 `PROTOTYPE`, 2 `RAW_CALIBRATION`, 4 `QUERY`, and 20
`RESERVE`.

## Non-overlap trust anchor

Before planning, freeze one external JSON-only non-overlap ledger. It binds:

- the trusted historical usage ledger raw SHA-256,
  `sha256:fffa4b335044ecb10e749d67f195de727a639c73b3d8752d518f4ef9c084c3fc`;
- its declared digest,
  `sha256:38bedae2c856bdbb73d16863152bd9b5581b99dc74157648fdeb1cb8be430c10`;
- its exact sorted 960-source-hash identity,
  `sha256:08f6a802f4c9cb09a0f579c74a187c33f5bb6b6a2430cbcceee9b67415a37160`;
- the known V3-exposed cohort incident, manifest identities, all 477
  quarantined remote-path/hash records, and the canonical quarantined root.

The ledger fails closed unless `excludedSourceSha256` is exactly the union of
those historical and quarantined source hashes. It is not enough for that set
to be a subset. All output paths for the ledger, plan, source root, and source
manifest must be new external paths and must not be inside, or be an ancestor
of, the quarantined root.

## Phases

`freeze-non-overlap-ledger` reads only three explicitly named JSON files. It
does not receive a source root or image path.

`create-source-plan` is the only Recovery V4 stage allowed to read the pinned
`samples.json`. It validates the tracked allowlist against metadata only and
does not make HTTP requests, decode an image, or enumerate a directory.

`acquire` never reads `samples.json` and never scans a source pool. For each
explicit plan record it performs a no-redirect HTTPS resolve, requires the
pinned `X-Repo-Commit`, raw-content `X-Linked-ETag` SHA-256 and
`X-Linked-Size`, then streams only that file into a new external root. It uses
new-only outputs, link/reparse checks, content verification, and a durable
hard-link promotion check before emitting an acquired-source manifest.

Example commands (all angle-bracket paths are external, new paths):

```powershell
.venv\Scripts\python.exe tools\prepare_mvtec_ad_recovery_v4_sources.py freeze-non-overlap-ledger `
  --historical-usage-ledger <historical_normal_usage_ledger.json> `
  --quarantine-incident <cohort_quarantine_incident_v1.json> `
  --quarantined-cohort-manifest <fresh_normal_holdout_v1\normal_holdout.json> `
  --output <recovery_v4_registry\non_overlap_ledger.json>

.venv\Scripts\python.exe tools\prepare_mvtec_ad_recovery_v4_sources.py create-source-plan `
  --source-metadata <source_metadata\samples.json> `
  --non-overlap-ledger <recovery_v4_registry\non_overlap_ledger.json> `
  --output <recovery_v4_registry\source_acquisition_plan.json>

.venv\Scripts\python.exe tools\prepare_mvtec_ad_recovery_v4_sources.py acquire `
  --plan <recovery_v4_registry\source_acquisition_plan.json> `
  --non-overlap-ledger <recovery_v4_registry\non_overlap_ledger.json> `
  --source-root <new_recovery_v4_source_vault> `
  --source-manifest-output <recovery_v4_registry\acquired_source_manifest.json>
```

The last phase intentionally has not been run by this source-foundation
change. Any later evaluation must use the acquired manifest rather than a
directory discovery operation.
