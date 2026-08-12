"""Focused fail-closed tests for the recovery-V4 receipt authority boundary."""

from __future__ import annotations

import base64
import hashlib

import pytest

from phone_dino import mvtec_recovery_v4_query_receipts as receipts


def _sha(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _query_record(case_id: str, category: str, path: str) -> dict[str, str]:
    return {
        "caseId": case_id,
        "category": category,
        "sourceGroupId": f"CONTENT_SHA256:{_sha(path)[7:]}",
        "sourceRemotePath": path,
        "sourceSha256": _sha(path),
    }


def _query_set() -> dict[str, object]:
    return receipts.build_query_set_binding(
        cohort_identity_sha256=_sha("cohort"),
        query_records=[
            _query_record("mvtec-ad/cable/query/002", "cable", "data/data_34/002.png"),
            _query_record("mvtec-ad/bottle/query/001", "bottle", "data/data_47/001.png"),
        ],
    )


def _production_config(query_set: dict[str, object]) -> dict[str, str]:
    public_key = bytes(range(32))
    return {
        "schemaVersion": receipts.RECOVERY_V4_PRODUCTION_AUTHORITY_CONFIG_SCHEMA,
        "authorityKind": receipts.RECOVERY_V4_PRODUCTION_AUTHORITY_KIND,
        "authorityUrl": "https://receipt-authority.example/v1/query-receipts:consume",
        "authorityKeyId": "ed25519:recovery-v4-test-pinned-key",
        "ed25519PublicKeyBase64": base64.b64encode(public_key).decode("ascii"),
        "ed25519PublicKeySha256": f"sha256:{hashlib.sha256(public_key).hexdigest()}",
        "clientMtlsIdentityReference": "mtls:recovery-v4-test-client",
        "querySetId": "recovery-v4:fixed-query-set",
        "querySetSha256": str(query_set["querySetSha256"]),
    }


def _structurally_signed_receipt(
    query_set: dict[str, object], config: dict[str, str], **overrides: object
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schemaVersion": receipts.RECOVERY_V4_SIGNED_RECEIPT_SCHEMA,
        "receiptId": "00000000-0000-4000-8000-000000000001",
        "issuedAt": "2026-08-12T00:00:00Z",
        "sequence": 1,
        "policy": receipts.RECOVERY_V4_CONSUME_ONCE_POLICY,
        "querySetId": config["querySetId"],
        "querySetSha256": query_set["querySetSha256"],
        "previousEntrySha256": _sha("authority-genesis"),
        "authorityKeyId": config["authorityKeyId"],
        "signatureAlgorithm": receipts.RECOVERY_V4_ED25519_ALGORITHM,
        "signature": base64.b64encode(bytes(64)).decode("ascii"),
    }
    receipt.update(overrides)
    return receipt


class _StaticProductionAuthority:
    authority_kind = receipts.RECOVERY_V4_PRODUCTION_AUTHORITY_KIND

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls = 0

    def consume_once(self, request: object) -> dict[str, object]:
        assert isinstance(request, dict)
        self.calls += 1
        return self.response


class _AlreadyConsumedProductionAuthority:
    authority_kind = receipts.RECOVERY_V4_PRODUCTION_AUTHORITY_KIND

    def __init__(self) -> None:
        self.calls = 0

    def consume_once(self, request: object) -> dict[str, object]:
        assert isinstance(request, dict)
        self.calls += 1
        raise receipts.QuerySetAlreadyConsumed("already consumed")


class _NeverCalledMtlsTransport:
    def __init__(self) -> None:
        self.calls = 0

    def post_json(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        self.calls += 1
        raise AssertionError("tests must not call the production HTTPS/mTLS transport")


def test_query_set_is_deterministic_and_rejects_model_or_recipe_fields() -> None:
    query_set = _query_set()

    assert [record["caseId"] for record in query_set["queryRecords"]] == [
        "mvtec-ad/bottle/query/001",
        "mvtec-ad/cable/query/002",
    ]
    assert receipts.validate_query_set_binding(query_set) == query_set

    injected = dict(query_set)
    injected["modelIdentitySha256"] = _sha("model")
    with pytest.raises(receipts.RecoveryV4QueryReceiptError, match="unsupported modelIdentitySha256"):
        receipts.validate_query_set_binding(injected)

    injected_record = dict(query_set)
    injected_record["queryRecords"] = [dict(record) for record in query_set["queryRecords"]]
    injected_record["queryRecords"][0]["recipeIdentitySha256"] = _sha("recipe")
    with pytest.raises(receipts.RecoveryV4QueryReceiptError, match="unsupported recipeIdentitySha256"):
        receipts.validate_query_set_binding(injected_record)


def test_query_set_requires_sorted_records_even_when_digest_is_recomputed() -> None:
    query_set = _query_set()
    unsorted = dict(query_set)
    unsorted["queryRecords"] = list(reversed(query_set["queryRecords"]))
    unsigned = dict(unsorted)
    unsigned.pop("querySetSha256")
    unsorted["querySetSha256"] = receipts.canonical_json_sha256(unsigned)

    with pytest.raises(receipts.RecoveryV4QueryReceiptError, match="sorted by caseId"):
        receipts.validate_query_set_binding(unsorted)


def test_pinned_production_config_rejects_test_fake_and_bad_public_key_digest() -> None:
    query_set = _query_set()
    config = _production_config(query_set)
    assert receipts.validate_production_authority_config(config, query_set=query_set)["querySetSha256"] == query_set[
        "querySetSha256"
    ]

    fake = dict(config)
    fake["authorityKind"] = receipts.RECOVERY_V4_TEST_AUTHORITY_KIND
    with pytest.raises(receipts.RecoveryV4QueryReceiptError, match="test or unknown authority"):
        receipts.validate_production_authority_config(fake, query_set=query_set)

    bad_key = dict(config)
    bad_key["ed25519PublicKeySha256"] = _sha("different-key")
    with pytest.raises(receipts.RecoveryV4QueryReceiptError, match="public key digest"):
        receipts.validate_production_authority_config(bad_key, query_set=query_set)


def test_production_https_mtls_client_validates_at_construction_without_network_call() -> None:
    query_set = _query_set()
    config = _production_config(query_set)
    transport = _NeverCalledMtlsTransport()

    authority = receipts.ProductionHttpsMtlsQueryReceiptAuthority(
        production_config=config,
        query_set=query_set,
        transport=transport,
    )

    assert authority.authority_kind == receipts.RECOVERY_V4_PRODUCTION_AUTHORITY_KIND
    assert transport.calls == 0


def test_test_fake_is_explicit_and_cannot_be_used_for_production_query_access() -> None:
    query_set = _query_set()
    config = _production_config(query_set)
    fake = receipts.InMemoryTestQueryReceiptAuthority()
    query_calls: list[str] = []

    with pytest.raises(receipts.RecoveryV4QueryReceiptError, match="test or unknown authority"):
        receipts.consume_verified_receipt_before_query(
            authority=fake,
            verifier=None,
            query_set=query_set,
            production_config=config,
            query_action=lambda: query_calls.append("opened"),
        )

    assert fake.calls == 0
    assert query_calls == []

    request = receipts.build_receipt_request(query_set=query_set, production_config=config)
    test_receipt = fake.consume_once(request)
    assert receipts.validate_test_receipt(test_receipt)["authorityKind"] == receipts.RECOVERY_V4_TEST_AUTHORITY_KIND
    with pytest.raises(receipts.QuerySetAlreadyConsumed):
        fake.consume_once(request)


@pytest.mark.parametrize(
    "verifier",
    [None, receipts.UnavailableEd25519ReceiptVerifier()],
)
def test_missing_or_unavailable_ed25519_verifier_fails_before_query_action(
    verifier: receipts.VerifiedEd25519ReceiptVerifier | None,
) -> None:
    query_set = _query_set()
    config = _production_config(query_set)
    authority = _StaticProductionAuthority(_structurally_signed_receipt(query_set, config))
    query_calls: list[str] = []

    with pytest.raises(receipts.Ed25519VerifierUnavailable, match="approved Ed25519"):
        receipts.consume_verified_receipt_before_query(
            authority=authority,
            verifier=verifier,
            query_set=query_set,
            production_config=config,
            query_action=lambda: query_calls.append("opened"),
        )

    assert authority.calls == 1
    assert query_calls == []


def test_duplicate_or_mismatched_receipt_fails_before_query_action() -> None:
    query_set = _query_set()
    config = _production_config(query_set)
    query_calls: list[str] = []

    duplicate = _AlreadyConsumedProductionAuthority()
    with pytest.raises(receipts.QuerySetAlreadyConsumed):
        receipts.consume_verified_receipt_before_query(
            authority=duplicate,
            verifier=receipts.UnavailableEd25519ReceiptVerifier(),
            query_set=query_set,
            production_config=config,
            query_action=lambda: query_calls.append("opened"),
        )
    assert duplicate.calls == 1
    assert query_calls == []

    mismatched = _StaticProductionAuthority(
        _structurally_signed_receipt(query_set, config, querySetSha256=_sha("other-query-set"))
    )
    with pytest.raises(receipts.RecoveryV4QueryReceiptError, match="query set differs"):
        receipts.consume_verified_receipt_before_query(
            authority=mismatched,
            verifier=receipts.UnavailableEd25519ReceiptVerifier(),
            query_set=query_set,
            production_config=config,
            query_action=lambda: query_calls.append("opened"),
        )
    assert mismatched.calls == 1
    assert query_calls == []


def test_receipt_signing_bytes_exclude_signature_and_strict_json_rejects_duplicates() -> None:
    query_set = _query_set()
    config = _production_config(query_set)
    receipt = _structurally_signed_receipt(query_set, config)
    changed_signature = dict(receipt)
    changed_signature["signature"] = base64.b64encode(bytes([1]) * 64).decode("ascii")

    unsigned = dict(receipt)
    unsigned.pop("signature")
    assert receipts.receipt_signing_bytes(receipt) == receipts.receipt_signing_bytes(changed_signature)
    assert receipts.receipt_signing_bytes(receipt) == receipts.receipt_signing_bytes_from_unsigned(unsigned)
    with pytest.raises(receipts.RecoveryV4QueryReceiptError, match="duplicate JSON key"):
        receipts.parse_strict_json_bytes(b'{"key": 1, "key": 2}', description="fixture")
