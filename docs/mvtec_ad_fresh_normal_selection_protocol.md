# Fresh normal selection and confirmation protocol

This protocol consumes the untouched normal-only partitions of a frozen fresh
MVTec cohort. It is an offline research safeguard, not evidence of defect
detection, a production threshold change, physical qualification, or a
PASS/FAIL decision.

It is deliberately separate from the historical V3--V5 selection tooling. The
fresh FIT augmentation package and both development reports are already frozen
before this protocol starts; their source modules are not modified afterwards.

## State sequence

```text
development reports
  -> immutable selection contract (JSON only)
  -> one-time selection claim (JSON only)
  -> one-time selection consumption receipt
  -> one aggregate raw-normal selection observation
  -> JSON-only research configuration lock
  -> explicit one-time confirmation claim
  -> one-time confirmation receipt
  -> raw-normal confirmation observation
```

The contract and claim do not open any image bytes. The selection receipt is
atomically written before the first `NORMAL_SELECTION` image is decoded;
therefore a crash after receipt creation burns that partition for this
tool-mediated protocol. The same applies to confirmation. This prevents an
accidental retry from repeatedly inspecting the same held-out normal inputs.
It does not prevent a person with direct filesystem access from copying or
opening those files outside the tool; that stronger boundary requires separate
ACL or service-account controls.

Consumption slots are not derived from the contract's directory. They are
derived from the frozen holdout manifest's external `partition_access` registry
and the manifest file digest, declared digest, and partition identity. Thus a
copy of the same valid contract in another directory reaches the same claim,
receipt, observation, and lock slots rather than authorizing a second
tool-mediated read.

## Frozen selection inputs

A selection contract binds all of the following with both raw-file and declared
SHA-256 digests where applicable:

- fresh holdout manifest and its development, selection, and confirmation
  identities;
- a single validated FIT-only augmentation package and its recipe;
- the closed membership of raw `NORMAL_SELECTION` and raw
  `NORMAL_CONFIRMATION` inputs, without image paths;
- every candidate's closed configuration, development report, raw-tuning
  thresholds, feature-input identities, and full feature-extractor identity;
- explicit per-category gates and a fixed lexicographic objective.

The contract cannot silently choose a candidate, recipe, threshold, model,
preprocessing setup, or gate. It rejects duplicate keys, non-finite JSON,
unknown fields, repository-contained inputs/outputs, symbolic links, and
Windows reparse points.

Create the contract with both already-frozen development reports. The gate
values are required command inputs; the following values are the predeclared
normal-robustness budget for the first fresh cohort (at most 4 of 32 raw
normal observations per category above threshold, a P95 excess of at most
0.05, and a maximum excess of at most 0.10):

```powershell
.venv\Scripts\python.exe tools\create_mvtec_ad_fresh_normal_selection_contract.py `
  --holdout <external-normal_holdout.json> `
  --augmentation-manifest <external-fresh-fit-augmentation_manifest.json> `
  --development-report <external-patch-1024-development.json> `
  --development-report <external-patch-2048-development.json> `
  --max-above-threshold-rate 0.125 `
  --max-p95-score-minus-threshold 0.05 `
  --max-maximum-score-minus-threshold 0.10 `
  --output <new-external-selection-contract.json>

.venv\Scripts\python.exe tools\create_mvtec_ad_fresh_normal_selection_claim.py `
  --contract <external-selection-contract.json>
```

The claim command is intentionally separate from contract creation. Its
success alone does not read selection images; the later observer writes the
receipt immediately before it does so.

## Selection observation

Before consuming selection, the evaluator validates the frozen JSON bindings,
loads only raw `FIT` plus the verified FIT derivatives to form prototypes, and
checks the current feature-extractor identity against every development report.
It then creates the receipt and opens only raw `NORMAL_SELECTION` normals.
It never reopens tuning, confirmation, reserve, pool, ledger, public dataset
metadata, blind, anomaly, or mask data.

For each category and candidate, the raw-tuning threshold is copied from the
validated development report. The observation records only source identities
and normal scores. Its gates are calculated as:

```text
aboveRate = count(selection score > frozen raw-tuning threshold) / count
p95Excess = P95(selection scores) - frozen threshold
maxExcess = max(selection scores) - frozen threshold
```

The JSON-only lock recomputes every score summary and gate from the observation
records. It can emit `RESEARCH_CONFIGURATION_LOCKED` or
`NO_ELIGIBLE_CONFIGURATION`, but it never promotes a configuration, changes a
runtime setting, or launches confirmation.

## Confirmation observation

Confirmation is a separate explicit action after a locked research
configuration. Its claim binds the lock, candidate, development report,
frozen threshold, and `NORMAL_CONFIRMATION` identity. After a new atomic
receipt, the evaluator opens only raw `FIT`, validated FIT derivatives, and
raw `NORMAL_CONFIRMATION` normals. It does not revisit selection or tuning.

The result is permanently labelled
`OBSERVATION_ONLY_NO_CONFIGURATION_CHANGE_OR_PROMOTION`. It has no winner,
no follow-on selector, and no authority beyond reporting the bounded
normal-input observation.
