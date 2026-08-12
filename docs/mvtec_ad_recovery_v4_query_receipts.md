# Recovery-V4 external query receipts

This is an offline-research safeguard for the fresh recovery-V4 cohort. It is
not a production authorization, a PASS/FAIL decision, a threshold change, or
a real-anomaly performance claim.

## Query-set binding

Before any `QUERY` source bytes are opened, recovery-V4 builds this exact
canonical JSON binding:

```json
{
  "schemaVersion": "phone-dino.recovery-v4-query-set/1.0",
  "purpose": "ONE_TIME_RAW_QUERY_CONSUMPTION",
  "cohortIdentitySha256": "sha256:...",
  "queryRecords": [
    {
      "caseId": "...",
      "category": "...",
      "sourceGroupId": "...",
      "sourceRemotePath": "...",
      "sourceSha256": "sha256:..."
    }
  ],
  "querySetSha256": "sha256:..."
}
```

`queryRecords` are sorted by `caseId`; the digest covers every other field.
The schema rejects model, recipe, feature-extractor, threshold, and runtime
fields. Consequently, changing an execution configuration cannot create a
new entitlement to read the same raw query inputs.

## Authority contract

The runtime request is limited to the pre-provisioned identity:

```json
{
  "schemaVersion": "phone-dino.recovery-v4-query-receipt-request/1.0",
  "authorityKeyId": "ed25519:...",
  "policy": "CONSUME_ONCE",
  "querySetId": "recovery-v4:...",
  "querySetSha256": "sha256:..."
}
```

The external authority must atomically append a receipt under a unique
`querySetSha256` constraint. A duplicate, timeout, malformed response, or
crash after the append burns the query slot and fails before any query image
is opened. Receipts must be retained in append-only storage with a sequence
and previous-entry hash for later operator audit.

A production receipt has schema
`phone-dino.recovery-v4-query-receipt/1.0`, an `Ed25519` signature, the exact
query-set id/digest, a server-issued UUID, UTC time, sequence, and previous
entry digest. The signed bytes are the fixed domain separator
`phone-dino.recovery-v4-query-receipt/1.0\0` followed by canonical unsigned
receipt JSON.

The authority constructs that unsigned payload, signs those bytes with its
private key, then appends the base64 signature. The local client verifies the
same bytes using the separately pinned public key.

## Provisioning and current dependency boundary

Production configuration pins an HTTPS/mTLS authority endpoint, Ed25519 key
id, 32-byte base64 public key and its SHA-256 digest, client mTLS identity
reference, and the exact query-set id/digest. It must be operator-signed or
otherwise deployed through a trusted configuration channel; remote key
discovery and trust-on-first-use are forbidden.

The local runtime must provide an approved Ed25519 verification provider. The
current project environment has no approved Python Ed25519 provider, so the
foundation intentionally supplies only an explicit fail-closed verifier. It
does not implement cryptography itself and does not accept a fallback
signature check. A recovery invocation therefore cannot proceed until an
operator provisions an approved verifier implementation and its dependency.

`InMemoryTestQueryReceiptAuthority` is a separate `TEST_FAKE` protocol. It
cannot emit a production receipt and is rejected before production query
access. The production HTTPS/mTLS client is transport-injected; this module
does not make network calls itself.

## Current checkpoint: do not consume queries

This is a design and test foundation only. No Recovery-V4 cohort, source
vault, query receipt, augmentation, or observation has been created.

Before a caller may use a production query-access path, the receipt foundation
must be hardened and independently re-reviewed for three known issues:

1. A public `authority_kind` string is not an authority assurance; an
   arbitrary injected object must not be able to masquerade as the production
   HTTPS/mTLS authority.
2. A signed receipt needs a fresh request nonce (and matching signed echo), or
   an equivalent independently durable replay guard, so a cached valid receipt
   cannot admit a second query action.
3. Ed25519 verifier availability must be preflighted before any consume call;
   an unavailable verifier must not burn a query set.

The current virtual environment has no approved Ed25519 verifier package and
no provisioned external append-only authority. Until those items exist and the
three issues above are fixed, every Recovery-V4 query attempt must fail before
opening a `QUERY` image byte. A local filesystem lock or test fake is not a
substitute for this authority.
