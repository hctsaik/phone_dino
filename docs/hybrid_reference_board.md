# Hybrid QR + ChArUco Reference Board (engineering contract)

## Status

Implemented as an opt-in, engineering-only `artifact/1.9` and `wire/1.6`
contract. It does not authorise equipment actions, production release, or
measurement decisions.

## Non-negotiable separation of responsibilities

* A QR code is an authenticated **candidate identity**. It may identify a
  tag/template/installation chosen by PhoneCV, but never supplies scale.
* ChArUco is the sole source for millimetres and plane geometry. Physical
  dimensions must still report `CHARUCO_CORNERS` (or be unavailable).
* The analyser verifies both items again from the same authoritative uploaded
  still. It maps the decoded QR quadrilateral through the ChArUco plane and
  compares it with the immutable QR symbol bounds pinned in the artifact.
* PhoneCV owns QR signature verification, lifecycle/revocation, operator
  authorization and future station-context policy. PhoneDINO only returns
  image observation evidence; it receives only hashes, never signing keys or
  the QR payload itself.

## P0 flow

1. PhoneCV may decode a preview QR to guide an operator, but treats it only as
   a candidate and resolves it server-side.
2. PhoneCV pins a `referenceBoard` binding in the engineering profile and
   passes it in the wire request. The binding includes hashes of the tag,
   board serial, manifest, active installation and expected QR payload.
3. PhoneDINO verifies the binding against immutable `referenceBoard` artifact
   policy and, before normalisation or embedding, validates in the raw still:
   QR decode/payload hash, QR image size, ChArUco support, and QR-to-ChArUco
   co-location residual.
4. Any failure produces `RECAPTURE_REQUIRED` and `NOT_RUN`; the embedder is
   never called. Target-only alignment cannot bypass this gate.
5. The returned `referenceBoardEvidence` records only hashed identity and
   visual evidence/reason codes. PhoneCV persists it beside the raw hash.

## Deferred P1 controls

QR plus ChArUco does not establish that a whole board has not been copied,
moved, or replayed. A formal deployment must additionally require a current
station context anchor (or controlled fixture/NFC/BLE factor), protected native
capture + nonce/attestation, lifecycle/revocation checks, dual commissioning,
and clone monitoring. Until those controls and hardware/MSA validation exist,
this remains engineering observation evidence.

## Test/fault-injection minimum

The contract tests cover missing QR, payload mismatch, undersized QR,
insufficient ChArUco support, QR moved away from its ChArUco-pinned symbol
bounds, request/artifact binding mismatch, and the assertion that a rejection
precedes any embedding. Device/print/light/blur/skew bench, replay and context
anchor tests remain required before a pilot can be expanded.
