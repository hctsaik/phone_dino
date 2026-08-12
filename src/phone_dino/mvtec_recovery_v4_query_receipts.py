"""Fail-closed recovery-V4 query-receipt authority boundary.

This module deliberately owns only the deterministic query-set binding,
authority configuration validation, and pre-query control flow.  It contains
no HTTP implementation and no cryptographic fallback.  A production caller
must inject an approved Ed25519 verifier; when one is unavailable, execution
fails before a query action can open image bytes.

The query-set digest intentionally excludes model, recipe, extractor,
threshold, and runtime fields.  A query set therefore remains consumed even
if a caller changes those execution details.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, TypeVar
from urllib.parse import urlsplit


RECOVERY_V4_QUERY_SET_SCHEMA = "phone-dino.recovery-v4-query-set/1.0"
RECOVERY_V4_RECEIPT_REQUEST_SCHEMA = "phone-dino.recovery-v4-query-receipt-request/1.0"
RECOVERY_V4_SIGNED_RECEIPT_SCHEMA = "phone-dino.recovery-v4-query-receipt/1.0"
RECOVERY_V4_PRODUCTION_AUTHORITY_CONFIG_SCHEMA = (
    "phone-dino.recovery-v4-production-query-receipt-authority-config/1.0"
)
RECOVERY_V4_TEST_RECEIPT_SCHEMA = "phone-dino.recovery-v4-test-query-receipt/1.0"

RECOVERY_V4_QUERY_SET_PURPOSE = "ONE_TIME_RAW_QUERY_CONSUMPTION"
RECOVERY_V4_CONSUME_ONCE_POLICY = "CONSUME_ONCE"
RECOVERY_V4_PRODUCTION_AUTHORITY_KIND = "PRODUCTION_HTTPS_MTLS"
RECOVERY_V4_TEST_AUTHORITY_KIND = "TEST_FAKE"
RECOVERY_V4_ED25519_ALGORITHM = "Ed25519"
RECOVERY_V4_RECEIPT_SIGNATURE_DOMAIN = b"phone-dino.recovery-v4-query-receipt/1.0\0"

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_KEY_ID_PATTERN = re.compile(r"^ed25519:[A-Za-z0-9._:-]{1,120}$")
_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9._:/-]{1,240}$")

_QUERY_RECORD_FIELDS = {
    "caseId",
    "category",
    "sourceGroupId",
    "sourceRemotePath",
    "sourceSha256",
}
_QUERY_SET_FIELDS = {
    "schemaVersion",
    "purpose",
    "cohortIdentitySha256",
    "queryRecords",
    "querySetSha256",
}
_AUTHORITY_CONFIG_FIELDS = {
    "schemaVersion",
    "authorityKind",
    "authorityUrl",
    "authorityKeyId",
    "ed25519PublicKeyBase64",
    "ed25519PublicKeySha256",
    "clientMtlsIdentityReference",
    "querySetId",
    "querySetSha256",
}
_RECEIPT_REQUEST_FIELDS = {
    "schemaVersion",
    "authorityKeyId",
    "policy",
    "querySetId",
    "querySetSha256",
}
_SIGNED_RECEIPT_FIELDS = {
    "schemaVersion",
    "receiptId",
    "issuedAt",
    "sequence",
    "policy",
    "querySetId",
    "querySetSha256",
    "previousEntrySha256",
    "authorityKeyId",
    "signatureAlgorithm",
    "signature",
}
_UNSIGNED_RECEIPT_FIELDS = _SIGNED_RECEIPT_FIELDS - {"signature"}
_TEST_RECEIPT_FIELDS = {
    "schemaVersion",
    "authorityKind",
    "querySetId",
    "querySetSha256",
    "testReceiptId",
}


class RecoveryV4QueryReceiptError(ValueError):
    """Raised when the recovery-V4 query-consumption boundary is unsafe."""


class QuerySetAlreadyConsumed(RecoveryV4QueryReceiptError):
    """Raised by an authority when a one-time query set already has a receipt."""


class Ed25519VerifierUnavailable(RecoveryV4QueryReceiptError):
    """Raised when the runtime has no approved Ed25519 verification provider."""


def canonical_json_bytes(document: object) -> bytes:
    """Return finite, deterministic JSON bytes for cryptographic bindings."""

    try:
        return json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecoveryV4QueryReceiptError("document is not finite canonical JSON") from error


def canonical_json_sha256(document: object) -> str:
    """Return a ``sha256:`` digest over :func:`canonical_json_bytes`."""

    return f"sha256:{hashlib.sha256(canonical_json_bytes(document)).hexdigest()}"


def parse_strict_json_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    """Parse a JSON object while rejecting duplicate keys and non-finite values."""

    if not isinstance(raw, bytes):
        raise RecoveryV4QueryReceiptError(f"{description} must be bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RecoveryV4QueryReceiptError(f"{description} has duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise RecoveryV4QueryReceiptError(f"{description} has non-finite JSON value {value}")

    try:
        document = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryV4QueryReceiptError(f"unable to parse {description}") from error
    if not isinstance(document, dict):
        raise RecoveryV4QueryReceiptError(f"{description} must be a JSON object")
    return document


def _require_exact_fields(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecoveryV4QueryReceiptError(f"{name} must be an object")
    missing = fields.difference(value)
    unknown = set(value).difference(fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unsupported {', '.join(sorted(unknown))}")
        raise RecoveryV4QueryReceiptError(f"{name} has {'; '.join(details)} fields")
    return value


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryV4QueryReceiptError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise RecoveryV4QueryReceiptError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RecoveryV4QueryReceiptError(f"{name} must be a positive integer")
    return value


def _require_safe_reference(value: object, *, name: str) -> str:
    text = _require_string(value, name=name)
    if not _REFERENCE_PATTERN.fullmatch(text):
        raise RecoveryV4QueryReceiptError(f"{name} has unsupported characters")
    return text


def _require_remote_path(value: object, *, name: str) -> str:
    path = _require_string(value, name=name)
    if "\\" in path or path.startswith("/") or not path.endswith(".png"):
        raise RecoveryV4QueryReceiptError(f"{name} must be a relative PNG path")
    parts = path.split("/")
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise RecoveryV4QueryReceiptError(f"{name} is unsafe")
    return path


def _normalize_query_record(value: object) -> dict[str, str]:
    record = _require_exact_fields(value, name="recovery-V4 query record", fields=_QUERY_RECORD_FIELDS)
    return {
        "caseId": _require_safe_reference(record.get("caseId"), name="query record caseId"),
        "category": _require_safe_reference(record.get("category"), name="query record category"),
        "sourceGroupId": _require_safe_reference(record.get("sourceGroupId"), name="query record sourceGroupId"),
        "sourceRemotePath": _require_remote_path(record.get("sourceRemotePath"), name="query record sourceRemotePath"),
        "sourceSha256": _require_sha256(record.get("sourceSha256"), name="query record sourceSha256"),
    }


def _normalize_query_records(value: object, *, require_sorted: bool) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise RecoveryV4QueryReceiptError("queryRecords must be a non-empty list")
    normalized = [_normalize_query_record(item) for item in value]
    case_ids = [item["caseId"] for item in normalized]
    source_hashes = [item["sourceSha256"] for item in normalized]
    source_paths = [item["sourceRemotePath"] for item in normalized]
    if len(case_ids) != len(set(case_ids)):
        raise RecoveryV4QueryReceiptError("queryRecords has duplicate caseId")
    if len(source_hashes) != len(set(source_hashes)):
        raise RecoveryV4QueryReceiptError("queryRecords has duplicate sourceSha256")
    if len(source_paths) != len(set(source_paths)):
        raise RecoveryV4QueryReceiptError("queryRecords has duplicate sourceRemotePath")
    ordered = sorted(normalized, key=lambda item: item["caseId"])
    if require_sorted and normalized != ordered:
        raise RecoveryV4QueryReceiptError("queryRecords must be sorted by caseId")
    return ordered


def build_query_set_binding(
    *, cohort_identity_sha256: str, query_records: Sequence[Mapping[str, object]]
) -> dict[str, Any]:
    """Build a model/recipe-independent immutable raw-query binding."""

    normalized_records = _normalize_query_records(list(query_records), require_sorted=False)
    document: dict[str, Any] = {
        "schemaVersion": RECOVERY_V4_QUERY_SET_SCHEMA,
        "purpose": RECOVERY_V4_QUERY_SET_PURPOSE,
        "cohortIdentitySha256": _require_sha256(cohort_identity_sha256, name="cohortIdentitySha256"),
        "queryRecords": normalized_records,
    }
    document["querySetSha256"] = canonical_json_sha256(document)
    return document


def validate_query_set_binding(value: object) -> dict[str, Any]:
    """Validate an exact query-set binding and its self-digest."""

    document = _require_exact_fields(value, name="recovery-V4 query set", fields=_QUERY_SET_FIELDS)
    if document.get("schemaVersion") != RECOVERY_V4_QUERY_SET_SCHEMA:
        raise RecoveryV4QueryReceiptError("query set schemaVersion is unsupported")
    if document.get("purpose") != RECOVERY_V4_QUERY_SET_PURPOSE:
        raise RecoveryV4QueryReceiptError("query set purpose is unsafe")
    normalized = {
        "schemaVersion": RECOVERY_V4_QUERY_SET_SCHEMA,
        "purpose": RECOVERY_V4_QUERY_SET_PURPOSE,
        "cohortIdentitySha256": _require_sha256(document.get("cohortIdentitySha256"), name="cohortIdentitySha256"),
        "queryRecords": _normalize_query_records(document.get("queryRecords"), require_sorted=True),
    }
    expected_digest = canonical_json_sha256(normalized)
    if _require_sha256(document.get("querySetSha256"), name="querySetSha256") != expected_digest:
        raise RecoveryV4QueryReceiptError("query set digest does not match canonical binding")
    normalized["querySetSha256"] = expected_digest
    return normalized


def _decode_base64(value: object, *, name: str, expected_bytes: int) -> bytes:
    text = _require_string(value, name=name)
    try:
        raw = base64.b64decode(text.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as error:
        raise RecoveryV4QueryReceiptError(f"{name} is not strict base64") from error
    if len(raw) != expected_bytes:
        raise RecoveryV4QueryReceiptError(f"{name} has an unsafe byte length")
    return raw


def _validate_production_url(value: object) -> str:
    url = _require_string(value, name="authorityUrl")
    try:
        parsed = urlsplit(url)
    except ValueError as error:
        raise RecoveryV4QueryReceiptError("authorityUrl is invalid") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RecoveryV4QueryReceiptError("authorityUrl must be a plain HTTPS authority endpoint")
    return url


def validate_production_authority_config(
    value: object, *, query_set: Mapping[str, object]
) -> dict[str, Any]:
    """Validate a pinned production authority config against an exact query set."""

    binding = validate_query_set_binding(dict(query_set))
    config = _require_exact_fields(
        value,
        name="recovery-V4 production authority config",
        fields=_AUTHORITY_CONFIG_FIELDS,
    )
    if config.get("schemaVersion") != RECOVERY_V4_PRODUCTION_AUTHORITY_CONFIG_SCHEMA:
        raise RecoveryV4QueryReceiptError("production authority config schemaVersion is unsupported")
    if config.get("authorityKind") != RECOVERY_V4_PRODUCTION_AUTHORITY_KIND:
        raise RecoveryV4QueryReceiptError("test or unknown authority is forbidden for production")
    key_id = _require_string(config.get("authorityKeyId"), name="authorityKeyId")
    if not _KEY_ID_PATTERN.fullmatch(key_id):
        raise RecoveryV4QueryReceiptError("authorityKeyId is not an Ed25519 key identifier")
    public_key = _decode_base64(
        config.get("ed25519PublicKeyBase64"),
        name="ed25519PublicKeyBase64",
        expected_bytes=32,
    )
    public_key_sha256 = _require_sha256(
        config.get("ed25519PublicKeySha256"), name="ed25519PublicKeySha256"
    )
    expected_key_digest = f"sha256:{hashlib.sha256(public_key).hexdigest()}"
    if public_key_sha256 != expected_key_digest:
        raise RecoveryV4QueryReceiptError("pinned Ed25519 public key digest does not match")
    query_set_id = _require_safe_reference(config.get("querySetId"), name="querySetId")
    if not query_set_id.startswith("recovery-v4:"):
        raise RecoveryV4QueryReceiptError("querySetId must be a recovery-V4 identifier")
    if _require_sha256(config.get("querySetSha256"), name="config querySetSha256") != binding["querySetSha256"]:
        raise RecoveryV4QueryReceiptError("production authority config is bound to a different query set")
    return {
        "schemaVersion": RECOVERY_V4_PRODUCTION_AUTHORITY_CONFIG_SCHEMA,
        "authorityKind": RECOVERY_V4_PRODUCTION_AUTHORITY_KIND,
        "authorityUrl": _validate_production_url(config.get("authorityUrl")),
        "authorityKeyId": key_id,
        "ed25519PublicKeyBase64": config["ed25519PublicKeyBase64"],
        "ed25519PublicKeySha256": public_key_sha256,
        "clientMtlsIdentityReference": _require_safe_reference(
            config.get("clientMtlsIdentityReference"), name="clientMtlsIdentityReference"
        ),
        "querySetId": query_set_id,
        "querySetSha256": binding["querySetSha256"],
    }


def build_receipt_request(
    *, query_set: Mapping[str, object], production_config: Mapping[str, object]
) -> dict[str, str]:
    """Create the only model/recipe-independent runtime authority request."""

    binding = validate_query_set_binding(dict(query_set))
    config = validate_production_authority_config(production_config, query_set=binding)
    return {
        "schemaVersion": RECOVERY_V4_RECEIPT_REQUEST_SCHEMA,
        "authorityKeyId": config["authorityKeyId"],
        "policy": RECOVERY_V4_CONSUME_ONCE_POLICY,
        "querySetId": config["querySetId"],
        "querySetSha256": binding["querySetSha256"],
    }


def validate_receipt_request(value: object) -> dict[str, str]:
    """Validate a request without consulting a model, recipe, or source image."""

    request = _require_exact_fields(value, name="query receipt request", fields=_RECEIPT_REQUEST_FIELDS)
    if request.get("schemaVersion") != RECOVERY_V4_RECEIPT_REQUEST_SCHEMA:
        raise RecoveryV4QueryReceiptError("query receipt request schemaVersion is unsupported")
    key_id = _require_string(request.get("authorityKeyId"), name="request authorityKeyId")
    if not _KEY_ID_PATTERN.fullmatch(key_id):
        raise RecoveryV4QueryReceiptError("request authorityKeyId is unsafe")
    if request.get("policy") != RECOVERY_V4_CONSUME_ONCE_POLICY:
        raise RecoveryV4QueryReceiptError("query receipt request policy is unsafe")
    query_set_id = _require_safe_reference(request.get("querySetId"), name="request querySetId")
    if not query_set_id.startswith("recovery-v4:"):
        raise RecoveryV4QueryReceiptError("request querySetId is unsafe")
    return {
        "schemaVersion": RECOVERY_V4_RECEIPT_REQUEST_SCHEMA,
        "authorityKeyId": key_id,
        "policy": RECOVERY_V4_CONSUME_ONCE_POLICY,
        "querySetId": query_set_id,
        "querySetSha256": _require_sha256(request.get("querySetSha256"), name="request querySetSha256"),
    }


def _validate_receipt_timestamp(value: object) -> str:
    timestamp = _require_string(value, name="receipt issuedAt")
    if not timestamp.endswith("Z"):
        raise RecoveryV4QueryReceiptError("receipt issuedAt must be UTC RFC3339 with Z")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise RecoveryV4QueryReceiptError("receipt issuedAt is invalid") from error
    if parsed.tzinfo is None:
        raise RecoveryV4QueryReceiptError("receipt issuedAt must include UTC")
    return timestamp


def _normalize_unsigned_receipt(value: object) -> dict[str, Any]:
    receipt = _require_exact_fields(value, name="unsigned query receipt", fields=_UNSIGNED_RECEIPT_FIELDS)
    if receipt.get("schemaVersion") != RECOVERY_V4_SIGNED_RECEIPT_SCHEMA:
        raise RecoveryV4QueryReceiptError("unsigned query receipt schemaVersion is unsupported")
    try:
        receipt_id = str(uuid.UUID(_require_string(receipt.get("receiptId"), name="receiptId")))
    except ValueError as error:
        raise RecoveryV4QueryReceiptError("receiptId must be a UUID") from error
    if receipt_id != receipt.get("receiptId"):
        raise RecoveryV4QueryReceiptError("receiptId must use canonical UUID text")
    key_id = _require_string(receipt.get("authorityKeyId"), name="receipt authorityKeyId")
    if not _KEY_ID_PATTERN.fullmatch(key_id):
        raise RecoveryV4QueryReceiptError("receipt authorityKeyId is unsafe")
    if receipt.get("policy") != RECOVERY_V4_CONSUME_ONCE_POLICY:
        raise RecoveryV4QueryReceiptError("receipt policy is unsafe")
    if receipt.get("signatureAlgorithm") != RECOVERY_V4_ED25519_ALGORITHM:
        raise RecoveryV4QueryReceiptError("receipt signatureAlgorithm must be Ed25519")
    return {
        "schemaVersion": RECOVERY_V4_SIGNED_RECEIPT_SCHEMA,
        "receiptId": receipt_id,
        "issuedAt": _validate_receipt_timestamp(receipt.get("issuedAt")),
        "sequence": _require_positive_int(receipt.get("sequence"), name="receipt sequence"),
        "policy": RECOVERY_V4_CONSUME_ONCE_POLICY,
        "querySetId": _require_safe_reference(receipt.get("querySetId"), name="receipt querySetId"),
        "querySetSha256": _require_sha256(receipt.get("querySetSha256"), name="receipt querySetSha256"),
        "previousEntrySha256": _require_sha256(
            receipt.get("previousEntrySha256"), name="receipt previousEntrySha256"
        ),
        "authorityKeyId": key_id,
        "signatureAlgorithm": RECOVERY_V4_ED25519_ALGORITHM,
    }


def _validate_signed_receipt_shape(value: object) -> tuple[dict[str, Any], bytes]:
    receipt = _require_exact_fields(value, name="signed query receipt", fields=_SIGNED_RECEIPT_FIELDS)
    unsigned = _normalize_unsigned_receipt({key: value for key, value in receipt.items() if key != "signature"})
    signature_text = _require_string(receipt.get("signature"), name="receipt signature")
    signature = _decode_base64(signature_text, name="receipt signature", expected_bytes=64)
    return {**unsigned, "signature": signature_text}, signature


def receipt_signing_bytes_from_unsigned(unsigned_receipt: Mapping[str, object]) -> bytes:
    """Return signing bytes before a production authority adds its signature."""

    normalized = _normalize_unsigned_receipt(dict(unsigned_receipt))
    return RECOVERY_V4_RECEIPT_SIGNATURE_DOMAIN + canonical_json_bytes(normalized)


def receipt_signing_bytes(receipt: Mapping[str, object]) -> bytes:
    """Return the domain-separated canonical bytes an Ed25519 provider signs."""

    normalized, _ = _validate_signed_receipt_shape(dict(receipt))
    unsigned = dict(normalized)
    unsigned.pop("signature")
    return receipt_signing_bytes_from_unsigned(unsigned)


class VerifiedEd25519ReceiptVerifier(Protocol):
    """Deployment-provided verifier; it must raise on every verification failure."""

    def verify_ed25519_receipt(
        self,
        *,
        signing_bytes: bytes,
        signature: bytes,
        authority_key_id: str,
        pinned_public_key: bytes,
    ) -> None:
        """Verify exactly one receipt signature or raise an exception."""


class QueryReceiptAuthority(Protocol):
    """Authority boundary with an atomic one-time consume operation."""

    authority_kind: str

    def consume_once(self, request: Mapping[str, str]) -> Mapping[str, object]:
        """Append and return a receipt, or raise :class:`QuerySetAlreadyConsumed`."""


class HttpsMtlsJsonTransport(Protocol):
    """Deployment-owned HTTPS/mTLS transport; intentionally not implemented here."""

    def post_json(
        self,
        *,
        url: str,
        document: Mapping[str, str],
        client_mtls_identity_reference: str,
    ) -> Mapping[str, object]:
        """Submit a JSON document using the provisioned client identity."""


class UnavailableEd25519ReceiptVerifier:
    """Explicit fail-closed verifier used when no approved crypto provider is installed."""

    def verify_ed25519_receipt(
        self,
        *,
        signing_bytes: bytes,
        signature: bytes,
        authority_key_id: str,
        pinned_public_key: bytes,
    ) -> None:
        del signing_bytes, signature, authority_key_id, pinned_public_key
        raise Ed25519VerifierUnavailable(
            "an approved Ed25519 verification provider must be injected before recovery-V4 query access"
        )


class ProductionHttpsMtlsQueryReceiptAuthority:
    """A transport-injected production authority client with no fallback network stack."""

    authority_kind = RECOVERY_V4_PRODUCTION_AUTHORITY_KIND

    def __init__(
        self,
        *,
        production_config: Mapping[str, object],
        query_set: Mapping[str, object],
        transport: HttpsMtlsJsonTransport,
    ) -> None:
        if not isinstance(production_config, Mapping):
            raise RecoveryV4QueryReceiptError("production_config must be a mapping")
        if not callable(getattr(transport, "post_json", None)):
            raise RecoveryV4QueryReceiptError("production authority transport lacks post_json")
        self._config = validate_production_authority_config(production_config, query_set=query_set)
        self._transport = transport

    def consume_once(self, request: Mapping[str, str]) -> Mapping[str, object]:
        normalized = validate_receipt_request(dict(request))
        if normalized["authorityKeyId"] != self._config.get("authorityKeyId"):
            raise RecoveryV4QueryReceiptError("authority request key id differs from production config")
        if normalized["querySetId"] != self._config.get("querySetId"):
            raise RecoveryV4QueryReceiptError("authority request query set id differs from production config")
        if normalized["querySetSha256"] != self._config.get("querySetSha256"):
            raise RecoveryV4QueryReceiptError("authority request query set differs from production config")
        return self._transport.post_json(
            url=_validate_production_url(self._config.get("authorityUrl")),
            document=normalized,
            client_mtls_identity_reference=_require_safe_reference(
                self._config.get("clientMtlsIdentityReference"), name="clientMtlsIdentityReference"
            ),
        )


class InMemoryTestQueryReceiptAuthority:
    """Test-only fake that cannot emit a production Ed25519 receipt."""

    authority_kind = RECOVERY_V4_TEST_AUTHORITY_KIND

    def __init__(self) -> None:
        self._consumed_query_sets: set[str] = set()
        self.calls = 0

    def consume_once(self, request: Mapping[str, str]) -> Mapping[str, object]:
        normalized = validate_receipt_request(dict(request))
        self.calls += 1
        key = normalized["querySetSha256"]
        if key in self._consumed_query_sets:
            raise QuerySetAlreadyConsumed("test fake query set was already consumed")
        self._consumed_query_sets.add(key)
        return {
            "schemaVersion": RECOVERY_V4_TEST_RECEIPT_SCHEMA,
            "authorityKind": RECOVERY_V4_TEST_AUTHORITY_KIND,
            "querySetId": normalized["querySetId"],
            "querySetSha256": normalized["querySetSha256"],
            "testReceiptId": f"test:{len(self._consumed_query_sets)}",
        }


def validate_test_receipt(value: object) -> dict[str, str]:
    """Validate a fake receipt only for isolated tests, never for production use."""

    receipt = _require_exact_fields(value, name="test fake receipt", fields=_TEST_RECEIPT_FIELDS)
    if receipt.get("schemaVersion") != RECOVERY_V4_TEST_RECEIPT_SCHEMA:
        raise RecoveryV4QueryReceiptError("test fake receipt schemaVersion is unsupported")
    if receipt.get("authorityKind") != RECOVERY_V4_TEST_AUTHORITY_KIND:
        raise RecoveryV4QueryReceiptError("test fake receipt authority kind is unsafe")
    return {
        "schemaVersion": RECOVERY_V4_TEST_RECEIPT_SCHEMA,
        "authorityKind": RECOVERY_V4_TEST_AUTHORITY_KIND,
        "querySetId": _require_safe_reference(receipt.get("querySetId"), name="test receipt querySetId"),
        "querySetSha256": _require_sha256(receipt.get("querySetSha256"), name="test receipt querySetSha256"),
        "testReceiptId": _require_safe_reference(receipt.get("testReceiptId"), name="test receipt id"),
    }


def verify_signed_receipt(
    *,
    receipt: Mapping[str, object],
    query_set: Mapping[str, object],
    production_config: Mapping[str, object],
    verifier: VerifiedEd25519ReceiptVerifier | None,
) -> dict[str, Any]:
    """Validate receipt binding and invoke an injected real Ed25519 verifier."""

    binding = validate_query_set_binding(dict(query_set))
    config = validate_production_authority_config(production_config, query_set=binding)
    normalized, signature = _validate_signed_receipt_shape(dict(receipt))
    if normalized["authorityKeyId"] != config["authorityKeyId"]:
        raise RecoveryV4QueryReceiptError("receipt authority key id differs from pinned config")
    if normalized["querySetId"] != config["querySetId"]:
        raise RecoveryV4QueryReceiptError("receipt query set id differs from pinned config")
    if normalized["querySetSha256"] != binding["querySetSha256"]:
        raise RecoveryV4QueryReceiptError("receipt query set differs from requested raw query inputs")
    if verifier is None:
        raise Ed25519VerifierUnavailable("an approved Ed25519 verifier was not supplied")
    public_key = _decode_base64(
        config["ed25519PublicKeyBase64"], name="ed25519PublicKeyBase64", expected_bytes=32
    )
    verifier.verify_ed25519_receipt(
        signing_bytes=receipt_signing_bytes(normalized),
        signature=signature,
        authority_key_id=config["authorityKeyId"],
        pinned_public_key=public_key,
    )
    return normalized


_QueryResult = TypeVar("_QueryResult")


def consume_verified_receipt_before_query(
    *,
    authority: QueryReceiptAuthority,
    verifier: VerifiedEd25519ReceiptVerifier | None,
    query_set: Mapping[str, object],
    production_config: Mapping[str, object],
    query_action: Callable[[], _QueryResult],
) -> _QueryResult:
    """Consume and verify a receipt before the supplied query action can run."""

    if not callable(query_action):
        raise RecoveryV4QueryReceiptError("query_action must be callable")
    binding = validate_query_set_binding(dict(query_set))
    config = validate_production_authority_config(production_config, query_set=binding)
    if getattr(authority, "authority_kind", None) != config["authorityKind"]:
        raise RecoveryV4QueryReceiptError("test or unknown authority is forbidden before production query access")
    request = build_receipt_request(query_set=binding, production_config=config)
    receipt = authority.consume_once(request)
    verify_signed_receipt(
        receipt=receipt,
        query_set=binding,
        production_config=config,
        verifier=verifier,
    )
    return query_action()
