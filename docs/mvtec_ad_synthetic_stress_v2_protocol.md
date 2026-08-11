# Post-V1 synthetic stimulus stress test

## Scope

This is a separate, post-V1 offline `SYNTHETIC_ONLY` engineering stress test.
It expands the number and intensity of deterministic programmatic stimuli; it
does not reopen or revise the V1 result, select a model, tune a threshold, or
establish real defect-detection performance.

The V1 synthetic report is historical knowledge, so V2 is explicitly
exploratory and not an independent confirmation. It must not be ranked against
V1 or used to claim improvement.

## Revision and model provenance

The already-recorded V2-r1 response report has schema `2.0` and remains a
historical engineering observation. It is not overwritten: its report did not
serialize the later-required `realPrecisionRecall` and `evidenceClass` scope
fields. The hardened reissue uses a newly generated package and a schema-`2.1`
response-only report, with those fields and strict detail-to-summary
reconciliation.

Every schema-`2.1` report must carry sealed DINO snapshot provenance. The
external snapshot manifest SHA-256 is an independently retained pin, not a
value derived by the observation command. Its repository-content digest
(`SEALED_REPOSITORY_CONTENT_SHA256_EXCLUDING_GIT_AND_PYTHON_BYTECODE_V1`) is
deliberately a different algorithm from the legacy evaluator's
`modelRepositorySha256`; both values are bound but are not required to be
equal.

## Closed input boundary

Both the generator and evaluator request only the successor V2 `FIT`
partition through the phase-safe loader. The closed normal-only metadata chain
may be revalidated, but no image bytes from raw tuning, normal selection,
normal confirmation, reserve, blind, true anomaly, or mask sources may be
opened.

For each category, the 12 FIT parents are deterministically assigned from
their immutable identities:

| Role | Parents/category | Use |
| --- | ---: | --- |
| Prototype | 6 | Raw-normal feature bank only |
| Calibration | 2 | Raw-normal fixed threshold only |
| Query | 4 | Raw nominal diagnostics and synthetic-stimulus parents |

No parent or source group can appear in more than one role.

## Pre-registered stimulus matrix

Only query parents receive synthetic children. Every query parent gets the
complete fixed matrix below, encoded as deterministic RGB PNG:

| Programmatic family | Render intensity levels |
| --- | --- |
| `LOCAL_SCRATCH` | `SUBTLE`, `MODERATE`, `PRONOUNCED` |
| `LOCAL_SPOT` | `SUBTLE`, `MODERATE`, `PRONOUNCED` |
| `LOCAL_OCCLUSION` | `SUBTLE`, `MODERATE`, `PRONOUNCED` |

That is 9 children per query parent and exactly 108 children across the three
categories. They are synthetic visual stimuli, not reconstructions of
observed defects or physical failure modes. The immutable V2 recipe fixes the
parameter ranges, named SHA-256 substreams, output encoding, and renderer;
the validator re-renders every child and verifies its bytes and digest.

## Fixed response observation

For each category, the detector uses raw prototype parents and fixes its
threshold before it loads or scores the V2 package:

```text
threshold(category) = max(raw calibration-normal scores)
stimulus response = score > threshold(category)
```

The V2 report intentionally does not contain TP/FP/FN/TN, precision, recall,
F1, AUROC, AP, confidence intervals, rankings, or V1-versus-V2 deltas.
Changing the number of rendered variants would alter those class-prevalence
metrics mechanically. Instead it records only:

- raw query-normal over-threshold count and rate;
- per category/family/intensity stimulus over-threshold count and rate; and
- paired child-minus-parent score-delta summaries.

The report must declare:

```text
metricScope: SYNTHETIC_STIMULUS_RESPONSE_ONLY
realAnomalyPerformance: NOT_ESTIMATED
realPrecisionRecall: NOT_ESTIMATED
evidenceClass: SYNTHETIC_ENGINEERING_ONLY
```

It also forbids model, algorithm, hyperparameter, or threshold selection,
production validation, and physical qualification.

## Commands

Before any DINO feature extraction, materialize a new external snapshot from
the independently approved source pins. The materializer prints
`snapshotManifestSha256`; record that value outside the snapshot directory as
the approved pin for the observation command. Do not derive it from the
directory immediately before running the observation.

```powershell
.venv\Scripts\python.exe tools\materialize_sealed_dino_snapshot.py `
  --expected-repository-sha256 <approved-source-repository-sha256> `
  --expected-weights-sha256 <approved-source-weights-sha256> `
  --output <new-external-sealed-dino-snapshot>
```

First create a new external package; the output directory must not already
exist:

```powershell
.venv\Scripts\python.exe tools\generate_mvtec_ad_synthetic_anomaly_stress_v2.py `
  --parent-holdout <external-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
  --plan <external-successor-plan.json> `
  --envelope <external-successor-envelope.json> `
  --source-root <external-source_bytes> `
  --recipe tools\mvtec_ad_synthetic_anomaly_stress_recipe_v2.json `
  --output <new-external-synthetic-stress-v2-package>
```

Then write one new external response-only report:

```powershell
.venv\Scripts\python.exe tools\run_mvtec_ad_synthetic_stress_v2.py `
  --sealed-model-snapshot <external-sealed-dino-snapshot> `
  --expected-sealed-model-snapshot-manifest-sha256 <approved-snapshot-manifest-sha256> `
  --parent-holdout <external-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
  --plan <external-successor-plan.json> `
  --envelope <external-successor-envelope.json> `
  --source-root <external-source_bytes> `
  --augmentation-manifest <external-v2-package/augmentation_manifest.json> `
  --output <new-external-synthetic-stress-v2-report.json>
```

Outputs are external and new-only. The tools reject repository-local paths,
existing slots, links/reparse points, changed bytes, incomplete matrices, and
any input outside the FIT-only boundary.

The snapshot manifest pin is an external trust anchor: a self-consistent
replacement snapshot with a newly calculated internal manifest digest is
rejected. The process rejects preloaded `dinov2` modules and centralized
Python bytecode caches; it serializes snapshot activation because Python's
import state is process-global. Windows cannot provide a fully atomic
no-follow traversal of mutable parent directories, so the external snapshot
directory must be ACL-protected/read-only to other writers. A same-privilege
actor able to race filesystem replacement remains outside this tool's
guarantee.
