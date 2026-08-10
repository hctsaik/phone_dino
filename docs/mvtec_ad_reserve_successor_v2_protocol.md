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
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json>

.venv\Scripts\python.exe tools\create_mvtec_ad_fresh_normal_successor_plan.py `
  --parent-holdout <external-v1-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
  --output <new-external-successor-plan.json>

.venv\Scripts\python.exe tools\create_mvtec_ad_fresh_normal_successor_envelope.py `
  --parent-holdout <external-v1-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
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

## Preselection development-evidence ledger

Before a selection contract can be frozen, create a JSON-only review draft of
the exact closed candidate-binding projection. It includes the four distinct
report identities and thresholds, common feature extractor, V2 R3 package,
FIT-only raw/R3 prototype commitments, candidate universe, and this selection
protocol module hash. This step validates closed JSON reports and provenance;
it does not open selection or other source image bytes.

Use the V2 parent contract at
`selection_protocol_v2/fresh_normal_selection_contract.json` (schema 1.1),
not the old `selection_protocol_v1` artifact:

```powershell
.venv\Scripts\python.exe tools\freeze_mvtec_ad_successor_v2_development_evidence_ledger.py `
  --parent-holdout <external-v1-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
  --plan <external-successor-plan.json> `
  --envelope <external-successor-envelope.json> `
  --development-report <external-raw-p2048-k5.json> `
  --development-report <external-r3-p1024-k3.json> `
  --development-report <external-r3-p2048-k3.json> `
  --development-report <external-r3-p2048-k5.json> `
  --augmentation-manifest <external-r3-manifest.json> `
  --recipe tools\mvtec_ad_successor_fit_camera_recipe_v2.json `
  --output <external-review-only-development-evidence-ledger.json>
```

Review the draft, then add its canonical JSON through a reviewed patch at the
fixed repository path
`docs/mvtec_ad_successor_v2_development_evidence_ledger.json`. The freeze CLI
must never write that tracked path. Commit and push the ledger together with
the selection code before any contract command is attempted.

The contract and every consuming entry point use the fixed
`PUSHED_GIT_AUDIT_ONLY` boundary: a temporary bare repository fetches only
`https://github.com/hctsaik/phone_dino.git` at `refs/heads/master`, requires
the recorded commit to be an ancestor of that fetched tip, resolves the fixed
path as a regular `100644` blob, and parses the raw `git cat-file blob` bytes.
It binds the commit OID, blob OID, raw blob SHA-256, ledger digest, and
projection digest. The source checkout's ledger bytes and index state are not
authority; the builder compares its raw `HEAD:path` blob with the independently
fetched raw blob, so CRLF/filter state and unrelated dirty files cannot alter
the ledger.

This is a pushed-Git audit anchor, not a signed attestation: a party able to
push or rewrite the required ref can replace the baseline. Replaying the
FIT/tuning evidence is a stronger control and is outside this protocol step.

## One-time selection execution

After all four immutable V2 development reports exist, freeze the contract and
claim without opening source image bytes. The contract binds the parent
holdout/selection chain, sealed plan and envelope, the V2 R3 manifest and
recipe, each candidate's distinct raw/R3 feature membership, every threshold,
and the common feature-extractor identity. It also JSON-derives the FIT-only
raw/R3 prototype commitments (36/144 inputs); the observer must match those
commitments before it can create the selection receipt.

Claim creation, observation, lock creation, and validated-lock loading each
resolve the recorded pushed ledger again and require the contract's full
candidate-binding projection and `selectionProtocolModuleSha256` to match
before a slot is read or written or any source input can be opened.

Development execution environment, entrypoint provenance, and timings remain
informational audit metadata. The selection decision binds the frozen
`featureExtractor` identity and hashes, not mutable environment or timing
values.

```powershell
.venv\Scripts\python.exe tools\create_mvtec_ad_successor_v2_selection_contract.py `
  --parent-holdout <external-v1-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
  --plan <external-successor-plan.json> `
  --envelope <external-successor-envelope.json> `
  --development-report <external-raw-p2048-k5.json> `
  --development-report <external-r3-p1024-k3.json> `
  --development-report <external-r3-p2048-k3.json> `
  --development-report <external-r3-p2048-k5.json> `
  --augmentation-manifest <external-r3-manifest.json> `
  --recipe tools\mvtec_ad_successor_fit_camera_recipe_v2.json `
  --development-evidence-ledger docs\mvtec_ad_successor_v2_development_evidence_ledger.json `
  --output <new-external-successor-v2-selection-contract.json>

.venv\Scripts\python.exe tools\create_mvtec_ad_successor_v2_selection_claim.py `
  --parent-holdout <external-v1-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
  --plan <external-successor-plan.json> `
  --envelope <external-successor-envelope.json> `
  --contract <external-successor-v2-selection-contract.json>
```

The observer validates/re-renders only successor FIT/R3 prototype inputs,
durably reserves the parent-registry receipt slot with an exclusive create and
`fsync`, and only then opens the 24 raw successor `NORMAL_SELECTION` images
once for all four candidates. It never opens parent confirmation, remaining
reserve, tuning, historical, blind, anomaly, or mask image bytes. Slots are
derived from the parent `partition_access` root, parent-holdout digests, and
the successor selection identity, so copying a contract cannot create another
tool-mediated attempt.

```powershell
.venv\Scripts\python.exe tools\run_mvtec_ad_successor_v2_selection_observation.py `
  --parent-holdout <external-v1-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
  --plan <external-successor-plan.json> `
  --envelope <external-successor-envelope.json> `
  --contract <external-successor-v2-selection-contract.json> `
  --augmentation-manifest <external-r3-manifest.json> `
  --recipe tools\mvtec_ad_successor_fit_camera_recipe_v2.json `
  --source-root <external-parent-source-bytes>

.venv\Scripts\python.exe tools\create_mvtec_ad_successor_v2_selection_lock.py `
  --parent-holdout <external-v1-normal_holdout.json> `
  --parent-selection-contract <external-selection_protocol_v2/fresh_normal_selection_contract.json> `
  --plan <external-successor-plan.json> `
  --envelope <external-successor-envelope.json> `
  --contract <external-successor-v2-selection-contract.json>
```

The lock is JSON-only and never creates a confirmation claim. In particular,
`NO_ELIGIBLE_CONFIGURATION` opens neither parent confirmation nor remaining
reserve images.

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
