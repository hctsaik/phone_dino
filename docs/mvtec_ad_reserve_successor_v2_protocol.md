# Reserve successor V2 normal-only protocol

## Status and scope

This is a pre-registered, offline research protocol for a small successor
normal-robustness screen. It follows the fresh V1 normal-selection lock of
2026-08-09, which recorded `NO_ELIGIBLE_CONFIGURATION`. It does not change
that lock, reopen its `NORMAL_SELECTION` images, relax its gates, report an
anomaly result, alter a production threshold, or authorize physical-device
use.

The source bytes come from the parent cohort's `RESERVE_UNTOUCHED` partition.
They have not been consumed by the tool-mediated V1 workflow, but the source
directory has no read audit or exclusive access control. Consequently every
artifact from this protocol must use the label
`TOOL_MEDIATED_UNCONSUMED_ONLY`; it must never be described as an independent,
fresh, or human-unseen cohort.

The parent cohort's `NORMAL_CONFIRMATION` partition remains reserved for one
final normal-only observation after, and only after, this successor locks an
eligible research configuration. It must not be used for development,
threshold tuning, candidate selection, or augmentation.

## Parent-source boundary

The successor seal must validate the parent holdout, schema-1.1 selection
contract, claim, receipt, observation, and `NO_ELIGIBLE_CONFIGURATION` lock as
closed JSON. It must permanently exclude all parent `FIT`,
`THRESHOLD_TUNING`, and `NORMAL_SELECTION` source digests. It may delegate
only the parent `RESERVE_UNTOUCHED` identities to this successor and must
retain every parent `NORMAL_CONFIRMATION` identity for the bridge-confirmation
step.

No phase may read a public MVTec inventory, source pool, historical ledger,
blind image, anomaly image, defect label, or mask. A phase-safe loader may
validate closed normal-only metadata, then hash/decode only the image bytes in
its explicitly requested successor partition.

All delegation, claim, receipt, observation, lock, and confirmation artifacts
have fixed external slots under the parent cohort's `partition_access`
registry. They are created with new-only atomic writes. A receipt is written
before the first query image is opened; a failed run therefore burns that
tool-mediated partition rather than allowing a retry.

Create the three JSON-only foundation artifacts in this order. None of these
commands accepts a source-image root or opens image bytes:

```powershell
.venv\Scripts\python.exe tools\create_mvtec_ad_fresh_normal_successor_seal.py `
  --parent-holdout <external-v1-normal_holdout.json> `
  --parent-selection-contract <external-v1-schema-1.1-contract.json>

.venv\Scripts\python.exe tools\create_mvtec_ad_fresh_normal_successor_plan.py `
  --parent-holdout <external-v1-normal_holdout.json> `
  --parent-selection-contract <external-v1-schema-1.1-contract.json> `
  --output <new-external-successor-plan.json>

.venv\Scripts\python.exe tools\create_mvtec_ad_fresh_normal_successor_envelope.py `
  --parent-holdout <external-v1-normal_holdout.json> `
  --parent-selection-contract <external-v1-schema-1.1-contract.json> `
  --plan <external-successor-plan.json> `
  --output <new-external-successor-envelope.json>
```

## Successor reserve allocation

The parent reserve has 93 raw normal images. The successor plan consumes only
the following deterministic allocation; the remaining successor reserve stays
unopened.

| Category | FIT | Raw tuning | Raw selection | Successor reserve |
| --- | ---: | ---: | ---: | ---: |
| capsule | 12 | 4 | 8 | 3 |
| metal_nut | 12 | 4 | 8 | 4 |
| tile | 12 | 4 | 8 | 14 |

There is deliberately no V2 confirmation partition. If an eligible candidate
is locked, the parent `NORMAL_CONFIRMATION` partition is the sole final
normal-only confirmation input. If no candidate is eligible, neither it nor
the remaining reserve may be opened automatically.

This small allocation is an exploratory screen. Four raw tuning images per
category cannot establish a population coverage guarantee; all output must
state that limitation.

## Generic camera-augmentation V2

V2 retains raw FIT originals and creates exactly three deterministic,
component-separated FIT derivatives per parent. It is a generic camera prior,
not a calibration of a real device. Each profile uses an independent named
SHA-256 substream and fixed RGB JPEG 4:2:0 Q95 encoding. Q95 is retained from
the only available engineering still's observed coding profile; it is not
claimed to represent a deployed camera.

| Variant | Allowed generic effect | Explicit exclusions |
| --- | --- | --- |
| `registration` | micro rotation, scale, translation, corner jitter, and at most 0.20-pixel blur | crop, flip, perspective replacement |
| `illumination` | narrow exposure, gamma, white-balance, directional shading, vignette, and lens-shading shifts | glare, shadows that obscure content, colour replacement |
| `sensor_transport` | narrow read/shot noise, 0.99--1.00 down/up sampling, and at most 0.20-pixel blur | hot-pixel injection, synthetic defects, capture rejection |

All ranges must be closed by the V2 recipe and deterministic re-rendered by
the validator. No derivative may be made from tuning, selection, confirmation,
reserve, blind, anomaly, or mask inputs.

## Candidate universe

Before any successor selection image is opened, freeze exactly these
candidates. The prototype selector is
`DETERMINISTIC_STRATIFIED_HASH_RANKED_PATCH_PREFIX_V2`: it visits stable
parent/variant strata round-robin and uses a stable patch hash within each
stratum. Its 1,024-patch bank is therefore a true prefix of its 2,048-patch
bank.

| Candidate ID | Prototype inputs | Max prototypes/category | top-K patch score |
| --- | --- | ---: | ---: |
| `reserve-v2-raw-p2048-k5` | raw FIT originals only | 2,048 | 5 |
| `reserve-v2-r3-p1024-k3` | raw FIT + V2 R3 derivatives | 1,024 | 3 |
| `reserve-v2-r3-p2048-k3` | raw FIT + V2 R3 derivatives | 2,048 | 3 |
| `reserve-v2-r3-p2048-k5` | raw FIT + V2 R3 derivatives | 2,048 | 5 |

All candidates use the same frozen DINO/preprocessing identity,
`prototypeBlockSize=256`, `batchSize=4`, and exact blocked cosine scoring.
The raw candidate makes the augmentation effect observable; it is not a
replacement for the V2 real-world-simulation candidates.

## Threshold, screen, and lock

For each candidate/category, the threshold is the maximum of the four frozen
raw tuning scores. The selection partition has eight raw normal inputs per
category. Because its P95 order statistic equals its maximum, V2 requires all
of the following per category:

- at most one selection score above the frozen threshold (12.5%);
- P95 score excess at most `0.05`;
- maximum score excess at most `0.05`.

The 12.5% count cap preserves the V1 rate budget; the maximum-excess cap is
stricter than V1. These are a deliberately small-sample screen, not a
confidence statement about normal-population coverage. Candidate ranking, if
more than one candidate meets every gate, is fixed lexicographically by
maximum category excess, maximum above-threshold rate, mean P95 excess, then
candidate ID. A JSON-only lock may emit only
`RESEARCH_CONFIGURATION_LOCKED` or `NO_ELIGIBLE_CONFIGURATION`; it must never
adjust a gate, threshold, recipe, or model.

## Final confirmation boundary

Only a `RESEARCH_CONFIGURATION_LOCKED` result enables an explicit bridge
confirmation claim. The claim binds the selected V2 candidate and the parent
confirmation identity, then creates a parent-registry receipt before opening
any parent confirmation image. The resulting report is
`OBSERVATION_ONLY_NO_CONFIGURATION_CHANGE_OR_PROMOTION`: it cannot select a
new winner, recalibrate a threshold, produce a defect claim, or change a
production setting.

## Required validation

- parent-lock, digest, path-copy, duplicate-claim, symbolic-link/reparse, and
  fixed-slot tampering tests;
- proof that successor source digests are disjoint from parent FIT/tuning/
  selection and from parent confirmation;
- phase-isolation spies proving that development opens only successor FIT and
  tuning, while selection opens only successor FIT derivatives and selection;
- deterministic V2 re-render, JPEG header/profile, source-digest, and
  augmentation-parent coverage tests;
- true nested prototype-prefix and top-K scorer tests;
- no-eligible regression proving no bridge confirmation or remaining-reserve
  image is opened.
