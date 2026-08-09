"""One-time, reserve-only successor cohorts for failed normal selection.

This module is deliberately separate from the original holdout, augmentation,
evaluator, and selection implementations.  It can only consume the closed
normal-only JSON evidence from a completed parent cohort whose selection lock
is ``NO_ELIGIBLE_CONFIGURATION``.  It never reads the parent source pool,
usage ledger, public MVTec inventory, anomaly labels, or parent images while
sealing and allocating the successor.

The successor is explicitly exploratory and not an independent dataset: it is
a deterministic re-partition of the parent cohort's still-unopened reserve.
The parent confirmation partition remains excluded and untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from phone_dino import mvtec_fresh_normal_observation as observation
from phone_dino import mvtec_fresh_normal_selection as selection
from phone_dino import mvtec_normal_holdout as holdout


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

FRESH_NORMAL_SUCCESSOR_SEAL_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-successor-seal/1.0"
FRESH_NORMAL_SUCCESSOR_PLAN_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-successor-plan/1.0"
FRESH_NORMAL_SUCCESSOR_ENVELOPE_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-successor-envelope/1.0"

FRESH_NORMAL_SUCCESSOR_SEAL_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_SUCCESSOR_SEAL"
FRESH_NORMAL_SUCCESSOR_PLAN_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_SUCCESSOR_PLAN"
FRESH_NORMAL_SUCCESSOR_ENVELOPE_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_SUCCESSOR_ENVELOPE"

FRESH_NORMAL_SUCCESSOR_SEAL_PHASE = "SUCCESSOR_RESERVE_SEAL"
FRESH_NORMAL_SUCCESSOR_PLAN_PHASE = "SUCCESSOR_ALLOCATION_PLAN"
FRESH_NORMAL_SUCCESSOR_ENVELOPE_PHASE = "SUCCESSOR_ENVELOPE"

FRESH_NORMAL_SUCCESSOR_BLIND_POLICY = "NO_BLIND_OR_ANOMALY_DATA"
FRESH_NORMAL_SUCCESSOR_DELEGATION_POLICY = "TOOL_MEDIATED_UNCONSUMED_ONLY"
FRESH_NORMAL_SUCCESSOR_RESULT_LABEL = "EXPLORATORY_NOT_INDEPENDENT"
FRESH_NORMAL_SUCCESSOR_INDEPENDENCE_LABEL = "NOT_INDEPENDENT_PARENT_RESERVE_DERIVATION"
FRESH_NORMAL_SUCCESSOR_ALLOCATION_ALGORITHM = "SHA256_RANKED_RESERVE_ONLY_V1"
FRESH_NORMAL_SUCCESSOR_PARTITION_ACCESS_DIRECTORY = "partition_access"

PARENT_HISTORICAL_PARTITIONS = ("FIT", "THRESHOLD_TUNING", "NORMAL_SELECTION")
PARENT_RESERVE_PARTITION = "RESERVE_UNTOUCHED"
PARENT_CONFIRMATION_PARTITION = "NORMAL_CONFIRMATION"
SUCCESSOR_PARTITIONS = ("FIT", "THRESHOLD_TUNING", "NORMAL_SELECTION", "RESERVE_UNTOUCHED")

# These quotas consume every one of the 93 still-unopened reserve images.  They
# are intentionally fixed: this foundation does not permit post-selection
# tuning of the successor partitioning scheme.
SUCCESSOR_CATEGORY_QUOTAS = (
    {
        "category": "capsule",
        "fitCount": 12,
        "thresholdTuningCount": 4,
        "normalSelectionCount": 8,
        "reserveUntouchedCount": 3,
    },
    {
        "category": "metal_nut",
        "fitCount": 12,
        "thresholdTuningCount": 4,
        "normalSelectionCount": 8,
        "reserveUntouchedCount": 4,
    },
    {
        "category": "tile",
        "fitCount": 12,
        "thresholdTuningCount": 4,
        "normalSelectionCount": 8,
        "reserveUntouchedCount": 14,
    },
)

SEAL_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "resultLabel",
    "delegationPolicy",
    "parentEvidence",
    "historicalParentSources",
    "historicalParentSourceIdentitySha256",
    "preservedParentNormalConfirmation",
    "delegatedReserveInputs",
    "delegatedReserveInputIdentitySha256",
    "successorSealSha256",
}
PLAN_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "resultLabel",
    "delegationPolicy",
    "sealFileSha256",
    "sealDeclaredSha256",
    "allocationAlgorithm",
    "allocationSeedSha256",
    "categoryQuotas",
    "successorPlanSha256",
}
ENVELOPE_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "resultLabel",
    "independenceLabel",
    "delegationPolicy",
    "sealFileSha256",
    "sealDeclaredSha256",
    "planFileSha256",
    "planDeclaredSha256",
    "parentEvidence",
    "historicalExclusion",
    "preservedParentNormalConfirmation",
    "allocation",
    "records",
    "successorPartitionIdentities",
    "successorEnvelopeSha256",
}

PARENT_EVIDENCE_FIELDS = {
    "holdoutManifestFileSha256",
    "holdoutManifestDeclaredSha256",
    "selectionContractFileSha256",
    "selectionContractDeclaredSha256",
    "selectionClaimFileSha256",
    "selectionClaimDeclaredSha256",
    "selectionReceiptFileSha256",
    "selectionReceiptDeclaredSha256",
    "selectionObservationFileSha256",
    "selectionObservationDeclaredSha256",
    "selectionLockFileSha256",
    "selectionLockDeclaredSha256",
    "selectionLockState",
    "parentReserveUntouchedIdentitySha256",
    "parentNormalConfirmationIdentitySha256",
}
HISTORICAL_SOURCE_FIELDS = {"caseId", "category", "partition", "sourceSha256", "sourceGroupId"}
RESERVE_INPUT_FIELDS = {
    "caseId",
    "category",
    "relativePath",
    "sourceSha256",
    "sourceGroupId",
    "acquisitionStratum",
    "expectedRemoteSha256",
    "expectedRemoteBytes",
    "kind",
    "defect",
    "parentPartition",
}
PRESERVED_CONFIRMATION_FIELDS = {
    "partition",
    "recordCount",
    "normalConfirmationIdentitySha256",
    "sourceIdentitySha256",
    "accessPolicy",
}
PLAN_QUOTA_FIELDS = {
    "category",
    "fitCount",
    "thresholdTuningCount",
    "normalSelectionCount",
    "reserveUntouchedCount",
}
ENVELOPE_RECORD_FIELDS = RESERVE_INPUT_FIELDS.difference({"parentPartition"}).union({"partition"})
HISTORICAL_EXCLUSION_FIELDS = {"policy", "sourceCount", "sourceIdentitySha256", "partitions"}
ALLOCATION_FIELDS = {"algorithm", "seedSha256", "categoryQuotas"}


class FreshNormalSuccessorError(ValueError):
    """Raised when a successor-cohort artifact is unsafe or inconsistent."""


@dataclass(frozen=True)
class _ParentContext:
    holdout_path: Path
    holdout_document: dict[str, Any]
    holdout_file_sha256: str
    contract: dict[str, Any]
    contract_file_sha256: str
    claim: dict[str, Any]
    claim_file_sha256: str
    selection_observation: dict[str, Any]
    selection_observation_file_sha256: str
    selection_lock: dict[str, Any]
    selection_lock_file_sha256: str
    historical_sources: tuple[dict[str, Any], ...]
    reserve_inputs: tuple[dict[str, Any], ...]
    preserved_confirmation: dict[str, Any]


def canonical_json_sha256(document: Any) -> str:
    """Hash JSON canonically after rejecting unsupported/non-finite values."""

    _validate_json_value(document, name="canonical JSON value")
    payload = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _validate_json_value(value: Any, *, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FreshNormalSuccessorError(f"{name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FreshNormalSuccessorError(f"{name} contains a non-string JSON key")
            _validate_json_value(item, name=f"{name}.{key}")
        return
    raise FreshNormalSuccessorError(f"{name} contains a value that is not JSON-compatible")


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return canonical_json_sha256(unsigned)


def _require_exact_fields(document: object, *, name: str, required: set[str]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise FreshNormalSuccessorError(f"{name} must be an object")
    missing = required.difference(document)
    if missing:
        raise FreshNormalSuccessorError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    unknown = set(document).difference(required)
    if unknown:
        raise FreshNormalSuccessorError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")
    return document


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreshNormalSuccessorError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise FreshNormalSuccessorError(f"{name} must be a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise FreshNormalSuccessorError(f"{name} must be a SHA-256 digest") from error
    return digest


def _require_positive_int(value: object, *, name: str, maximum: int = 10_000_000) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= maximum:
        raise FreshNormalSuccessorError(f"{name} must be a positive integer no larger than {maximum}")
    return value


def _require_nonnegative_int(value: object, *, name: str, maximum: int = 10_000_000) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
        raise FreshNormalSuccessorError(f"{name} must be a non-negative integer no larger than {maximum}")
    return value


def _is_under(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError:
        return False
    if stat.S_ISLNK(status.st_mode):
        return True
    attributes = getattr(status, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    return bool(attributes & reparse)


def _reject_links_on_existing_path(path: Path, *, description: str) -> None:
    current = path
    while True:
        if current.exists() or current.is_symlink():
            if _is_link_or_reparse_point(current):
                raise FreshNormalSuccessorError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _require_external_file(path: Path, *, description: str, repository_root: Path) -> None:
    if not isinstance(path, Path):
        raise FreshNormalSuccessorError(f"{description} path must be a Path")
    _reject_links_on_existing_path(path, description=description)
    if not path.is_file() or _is_link_or_reparse_point(path):
        raise FreshNormalSuccessorError(f"{description} must be a regular non-link file")
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise FreshNormalSuccessorError(f"{description} must stay outside the Git working tree")


def _require_external_directory(path: Path, *, description: str, repository_root: Path) -> None:
    if not isinstance(path, Path):
        raise FreshNormalSuccessorError(f"{description} path must be a Path")
    _reject_links_on_existing_path(path, description=description)
    if not path.is_dir() or _is_link_or_reparse_point(path):
        raise FreshNormalSuccessorError(f"{description} must be a directory without links or reparse points")
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise FreshNormalSuccessorError(f"{description} must stay outside the Git working tree")


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError as error:
        raise FreshNormalSuccessorError("unable to stat immutable JSON input") from error
    return (status.st_dev, status.st_ino, status.st_mode, status.st_size, status.st_mtime_ns)


def _parse_json_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreshNormalSuccessorError(f"{description} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise FreshNormalSuccessorError(f"{description} contains a non-finite JSON value: {value}")

    try:
        document = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshNormalSuccessorError(f"unable to read {description}") from error
    _validate_json_value(document, name=description)
    if not isinstance(document, dict):
        raise FreshNormalSuccessorError(f"{description} must be a JSON object")
    return document


def _read_external_json(
    path: Path,
    *,
    description: str,
    repository_root: Path,
) -> tuple[dict[str, Any], str]:
    _require_external_file(path, description=description, repository_root=repository_root)
    before = _stat_signature(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FreshNormalSuccessorError(f"unable to read {description}") from error
    _require_external_file(path, description=description, repository_root=repository_root)
    if before != _stat_signature(path):
        raise FreshNormalSuccessorError(f"{description} changed while it was read")
    return _parse_json_bytes(raw, description=description), f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _prepare_external_output(path: Path, *, description: str, repository_root: Path) -> None:
    if not isinstance(path, Path):
        raise FreshNormalSuccessorError(f"{description} path must be a Path")
    if path.exists() or path.is_symlink():
        raise FreshNormalSuccessorError(f"{description} already exists; the immutable slot is already consumed")
    _reject_links_on_existing_path(path.parent, description=description)
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise FreshNormalSuccessorError(f"{description} must stay outside the Git working tree")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FreshNormalSuccessorError(f"unable to create {description} parent directory") from error
    _reject_links_on_existing_path(path.parent, description=description)


def _write_external_json(path: Path, document: dict[str, Any], *, description: str, repository_root: Path) -> None:
    _prepare_external_output(path, description=description, repository_root=repository_root)
    payload = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise FreshNormalSuccessorError(f"unable to write immutable {description}") from error


def _safe_relative_path(value: object, *, name: str) -> PurePosixPath:
    text = _require_string(value, name=name)
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise FreshNormalSuccessorError(f"{name} must be a safe relative POSIX path")
    if "\\" in text:
        raise FreshNormalSuccessorError(f"{name} must use POSIX separators")
    return candidate


def _safe_file_under(root: Path, relative: PurePosixPath, *, description: str, repository_root: Path) -> Path:
    _require_external_directory(root, description=f"{description} root", repository_root=repository_root)
    candidate = root.joinpath(*relative.parts)
    _reject_links_on_existing_path(candidate, description=description)
    if not candidate.is_file() or _is_link_or_reparse_point(candidate):
        raise FreshNormalSuccessorError(f"{description} must be a regular non-link file")
    if not _is_under(root, candidate):
        raise FreshNormalSuccessorError(f"{description} escapes its declared root")
    return candidate


def _verify_decodable_image(path: Path, *, description: str) -> None:
    try:
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            source.convert("RGB").load()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise FreshNormalSuccessorError(f"{description} is not a decodable image") from error


def _validate_static_scope(
    document: dict[str, Any],
    *,
    schema: str,
    purpose: str,
    phase: str,
    description: str,
) -> None:
    if document.get("schemaVersion") != schema:
        raise FreshNormalSuccessorError(f"{description} schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise FreshNormalSuccessorError(f"{description} must be non-authoritative and non-production")
    if document.get("purpose") != purpose or document.get("phase") != phase:
        raise FreshNormalSuccessorError(f"{description} purpose or phase is unsafe")
    if document.get("blindPolicy") != FRESH_NORMAL_SUCCESSOR_BLIND_POLICY:
        raise FreshNormalSuccessorError(f"{description} blind policy is unsafe")
    if document.get("resultLabel") != FRESH_NORMAL_SUCCESSOR_RESULT_LABEL:
        raise FreshNormalSuccessorError(f"{description} must remain exploratory and not independent")
    if document.get("delegationPolicy") != FRESH_NORMAL_SUCCESSOR_DELEGATION_POLICY:
        raise FreshNormalSuccessorError(f"{description} delegation policy is unsafe")


def _validate_parent_evidence(value: object) -> dict[str, Any]:
    document = _require_exact_fields(value, name="successor parent evidence", required=PARENT_EVIDENCE_FIELDS)
    for field in PARENT_EVIDENCE_FIELDS.difference({"selectionLockState"}):
        _require_sha256(document.get(field), name=f"successor parent evidence {field}")
    if document.get("selectionLockState") != "NO_ELIGIBLE_CONFIGURATION":
        raise FreshNormalSuccessorError("successor parent evidence requires NO_ELIGIBLE_CONFIGURATION")
    return dict(document)


def _validate_historical_sources(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FreshNormalSuccessorError("successor historical parent sources must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_sources: set[str] = set()
    partitions: set[str] = set()
    for raw in value:
        item = _require_exact_fields(raw, name="successor historical parent source", required=HISTORICAL_SOURCE_FIELDS)
        case_id = _require_string(item.get("caseId"), name="successor historical parent source caseId")
        source_sha256 = _require_sha256(item.get("sourceSha256"), name="successor historical parent source sourceSha256")
        partition = item.get("partition")
        if partition not in PARENT_HISTORICAL_PARTITIONS:
            raise FreshNormalSuccessorError("successor historical parent source partition is unsupported")
        if case_id in seen_case_ids or source_sha256 in seen_sources:
            raise FreshNormalSuccessorError("successor historical parent sources are duplicated")
        seen_case_ids.add(case_id)
        seen_sources.add(source_sha256)
        partitions.add(str(partition))
        result.append({
            "caseId": case_id,
            "category": _require_string(item.get("category"), name="successor historical parent source category"),
            "partition": str(partition),
            "sourceSha256": source_sha256,
            "sourceGroupId": _require_string(item.get("sourceGroupId"), name="successor historical parent source sourceGroupId"),
        })
    if partitions != set(PARENT_HISTORICAL_PARTITIONS):
        raise FreshNormalSuccessorError("successor historical parent sources must cover FIT, tuning, and selection")
    if result != sorted(result, key=lambda item: item["caseId"]):
        raise FreshNormalSuccessorError("successor historical parent sources must be sorted by caseId")
    return result


def _validate_reserve_inputs(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FreshNormalSuccessorError("successor delegated reserve inputs must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_paths: set[str] = set()
    for raw in value:
        item = _require_exact_fields(raw, name="successor delegated reserve input", required=RESERVE_INPUT_FIELDS)
        case_id = _require_string(item.get("caseId"), name="successor delegated reserve input caseId")
        source_sha256 = _require_sha256(item.get("sourceSha256"), name="successor delegated reserve input sourceSha256")
        relative_path = _safe_relative_path(item.get("relativePath"), name="successor delegated reserve input relativePath")
        if item.get("parentPartition") != PARENT_RESERVE_PARTITION:
            raise FreshNormalSuccessorError("successor delegated reserve input must come from parent reserve")
        if item.get("acquisitionStratum") != "OFFICIAL_MVTEC_TRAIN_GOOD":
            raise FreshNormalSuccessorError("successor delegated reserve input acquisition stratum is unsupported")
        if item.get("kind") != "NOMINAL" or item.get("defect") != "good":
            raise FreshNormalSuccessorError("successor delegated reserve input must be nominal good only")
        remote_sha256 = _require_sha256(item.get("expectedRemoteSha256"), name="successor delegated reserve input expectedRemoteSha256")
        if remote_sha256 != source_sha256:
            raise FreshNormalSuccessorError("successor delegated reserve input source digest is inconsistent")
        if case_id in seen_case_ids or source_sha256 in seen_sources or relative_path.as_posix() in seen_paths:
            raise FreshNormalSuccessorError("successor delegated reserve inputs are duplicated")
        seen_case_ids.add(case_id)
        seen_sources.add(source_sha256)
        seen_paths.add(relative_path.as_posix())
        result.append({
            "caseId": case_id,
            "category": _require_string(item.get("category"), name="successor delegated reserve input category"),
            "relativePath": relative_path.as_posix(),
            "sourceSha256": source_sha256,
            "sourceGroupId": _require_string(item.get("sourceGroupId"), name="successor delegated reserve input sourceGroupId"),
            "acquisitionStratum": "OFFICIAL_MVTEC_TRAIN_GOOD",
            "expectedRemoteSha256": remote_sha256,
            "expectedRemoteBytes": _require_positive_int(item.get("expectedRemoteBytes"), name="successor delegated reserve input expectedRemoteBytes"),
            "kind": "NOMINAL",
            "defect": "good",
            "parentPartition": PARENT_RESERVE_PARTITION,
        })
    if result != sorted(result, key=lambda item: item["caseId"]):
        raise FreshNormalSuccessorError("successor delegated reserve inputs must be sorted by caseId")
    expected_counts = {
        quota["category"]: sum(int(quota[key]) for key in ("fitCount", "thresholdTuningCount", "normalSelectionCount", "reserveUntouchedCount"))
        for quota in SUCCESSOR_CATEGORY_QUOTAS
    }
    actual_counts: dict[str, int] = {}
    for item in result:
        actual_counts[item["category"]] = actual_counts.get(item["category"], 0) + 1
    if actual_counts != expected_counts or len(result) != 93:
        raise FreshNormalSuccessorError("successor delegated reserve inputs must contain the fixed 93-image category allocation")
    return result


def _validate_preserved_confirmation(value: object, *, parent_evidence: dict[str, Any]) -> dict[str, Any]:
    document = _require_exact_fields(value, name="preserved parent confirmation", required=PRESERVED_CONFIRMATION_FIELDS)
    if document.get("partition") != PARENT_CONFIRMATION_PARTITION:
        raise FreshNormalSuccessorError("preserved parent confirmation partition is unsafe")
    if _require_nonnegative_int(document.get("recordCount"), name="preserved parent confirmation recordCount") != 96:
        raise FreshNormalSuccessorError("preserved parent confirmation must retain all 96 inputs")
    if _require_sha256(document.get("normalConfirmationIdentitySha256"), name="preserved parent confirmation identity") != parent_evidence["parentNormalConfirmationIdentitySha256"]:
        raise FreshNormalSuccessorError("preserved parent confirmation identity does not match parent evidence")
    _require_sha256(document.get("sourceIdentitySha256"), name="preserved parent confirmation source identity")
    if document.get("accessPolicy") != "PRESERVED_NOT_DELEGATED_NOT_OPENED":
        raise FreshNormalSuccessorError("preserved parent confirmation access policy is unsafe")
    return dict(document)


def _validate_seal_document(document: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal successor seal", required=SEAL_FIELDS)
    _validate_static_scope(
        document,
        schema=FRESH_NORMAL_SUCCESSOR_SEAL_SCHEMA,
        purpose=FRESH_NORMAL_SUCCESSOR_SEAL_PURPOSE,
        phase=FRESH_NORMAL_SUCCESSOR_SEAL_PHASE,
        description="fresh normal successor seal",
    )
    parent_evidence = _validate_parent_evidence(document.get("parentEvidence"))
    historical_sources = _validate_historical_sources(document.get("historicalParentSources"))
    historical_identity = _require_sha256(
        document.get("historicalParentSourceIdentitySha256"),
        name="successor historical parent source identity",
    )
    if historical_identity != canonical_json_sha256(historical_sources):
        raise FreshNormalSuccessorError("successor historical parent source identity does not match")
    reserve_inputs = _validate_reserve_inputs(document.get("delegatedReserveInputs"))
    reserve_identity = _require_sha256(
        document.get("delegatedReserveInputIdentitySha256"),
        name="successor delegated reserve input identity",
    )
    if reserve_identity != canonical_json_sha256(reserve_inputs):
        raise FreshNormalSuccessorError("successor delegated reserve input identity does not match")
    preserved_confirmation = _validate_preserved_confirmation(
        document.get("preservedParentNormalConfirmation"),
        parent_evidence=parent_evidence,
    )
    historical_hashes = {item["sourceSha256"] for item in historical_sources}
    reserve_hashes = {item["sourceSha256"] for item in reserve_inputs}
    if historical_hashes.intersection(reserve_hashes):
        raise FreshNormalSuccessorError("successor reserve overlaps parent historical sources")
    if document.get("successorSealSha256") != _document_digest(document, "successorSealSha256"):
        raise FreshNormalSuccessorError("fresh normal successor seal digest does not match")
    return {
        **document,
        "parentEvidence": parent_evidence,
        "historicalParentSources": historical_sources,
        "delegatedReserveInputs": reserve_inputs,
        "preservedParentNormalConfirmation": preserved_confirmation,
    }


def _validate_plan_quotas(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FreshNormalSuccessorError("successor category quotas must be a non-empty list")
    result: list[dict[str, Any]] = []
    for raw in value:
        item = _require_exact_fields(raw, name="successor category quota", required=PLAN_QUOTA_FIELDS)
        result.append({
            "category": _require_string(item.get("category"), name="successor category quota category"),
            "fitCount": _require_positive_int(item.get("fitCount"), name="successor category quota fitCount"),
            "thresholdTuningCount": _require_positive_int(item.get("thresholdTuningCount"), name="successor category quota thresholdTuningCount"),
            "normalSelectionCount": _require_positive_int(item.get("normalSelectionCount"), name="successor category quota normalSelectionCount"),
            "reserveUntouchedCount": _require_positive_int(item.get("reserveUntouchedCount"), name="successor category quota reserveUntouchedCount"),
        })
    if result != list(SUCCESSOR_CATEGORY_QUOTAS):
        raise FreshNormalSuccessorError("successor category quotas are not the fixed reserve-only allocation")
    return result


def _allocation_seed(seal: dict[str, Any]) -> str:
    return canonical_json_sha256({
        "schemaVersion": FRESH_NORMAL_SUCCESSOR_PLAN_SCHEMA,
        "delegatedReserveInputIdentitySha256": seal["delegatedReserveInputIdentitySha256"],
        "parentReserveUntouchedIdentitySha256": seal["parentEvidence"]["parentReserveUntouchedIdentitySha256"],
        "successorSealSha256": seal["successorSealSha256"],
    })


def _build_successor_plan(seal: dict[str, Any], *, seal_file_sha256: str) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_SUCCESSOR_PLAN_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_SUCCESSOR_PLAN_PURPOSE,
        "phase": FRESH_NORMAL_SUCCESSOR_PLAN_PHASE,
        "blindPolicy": FRESH_NORMAL_SUCCESSOR_BLIND_POLICY,
        "resultLabel": FRESH_NORMAL_SUCCESSOR_RESULT_LABEL,
        "delegationPolicy": FRESH_NORMAL_SUCCESSOR_DELEGATION_POLICY,
        "sealFileSha256": seal_file_sha256,
        "sealDeclaredSha256": seal["successorSealSha256"],
        "allocationAlgorithm": FRESH_NORMAL_SUCCESSOR_ALLOCATION_ALGORITHM,
        "allocationSeedSha256": _allocation_seed(seal),
        "categoryQuotas": [dict(quota) for quota in SUCCESSOR_CATEGORY_QUOTAS],
    }
    document["successorPlanSha256"] = _document_digest(document, "successorPlanSha256")
    return document


def _validate_plan_document(
    document: dict[str, Any],
    *,
    seal: dict[str, Any] | None = None,
    seal_file_sha256: str | None = None,
) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal successor plan", required=PLAN_FIELDS)
    _validate_static_scope(
        document,
        schema=FRESH_NORMAL_SUCCESSOR_PLAN_SCHEMA,
        purpose=FRESH_NORMAL_SUCCESSOR_PLAN_PURPOSE,
        phase=FRESH_NORMAL_SUCCESSOR_PLAN_PHASE,
        description="fresh normal successor plan",
    )
    _require_sha256(document.get("sealFileSha256"), name="successor plan sealFileSha256")
    _require_sha256(document.get("sealDeclaredSha256"), name="successor plan sealDeclaredSha256")
    if document.get("allocationAlgorithm") != FRESH_NORMAL_SUCCESSOR_ALLOCATION_ALGORITHM:
        raise FreshNormalSuccessorError("successor plan allocation algorithm is unsupported")
    _require_sha256(document.get("allocationSeedSha256"), name="successor plan allocationSeedSha256")
    _validate_plan_quotas(document.get("categoryQuotas"))
    if document.get("successorPlanSha256") != _document_digest(document, "successorPlanSha256"):
        raise FreshNormalSuccessorError("fresh normal successor plan digest does not match")
    result = dict(document)
    if seal is not None:
        if seal_file_sha256 is None:
            raise FreshNormalSuccessorError("successor plan validation requires the seal file digest")
        expected = _build_successor_plan(seal, seal_file_sha256=seal_file_sha256)
        if result != expected:
            raise FreshNormalSuccessorError("successor plan does not bind the fixed successor seal")
    return result


def _rank_reserve_input(record: dict[str, Any], *, seed: str) -> str:
    material = "\0".join((
        FRESH_NORMAL_SUCCESSOR_ENVELOPE_SCHEMA,
        FRESH_NORMAL_SUCCESSOR_ALLOCATION_ALGORITHM,
        seed,
        str(record["category"]),
        str(record["sourceGroupId"]),
        str(record["sourceSha256"]),
    )).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _quota_partition_counts(quota: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    return (
        ("FIT", int(quota["fitCount"])),
        ("THRESHOLD_TUNING", int(quota["thresholdTuningCount"])),
        ("NORMAL_SELECTION", int(quota["normalSelectionCount"])),
        ("RESERVE_UNTOUCHED", int(quota["reserveUntouchedCount"])),
    )


def _allocate_successor_records(seal: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    reserve_by_category: dict[str, list[dict[str, Any]]] = {}
    for record in seal["delegatedReserveInputs"]:
        reserve_by_category.setdefault(str(record["category"]), []).append(record)
    assigned: list[dict[str, Any]] = []
    for quota in SUCCESSOR_CATEGORY_QUOTAS:
        category = str(quota["category"])
        ranked = sorted(
            reserve_by_category.get(category, []),
            key=lambda record: (_rank_reserve_input(record, seed=str(plan["allocationSeedSha256"])), record["caseId"]),
        )
        cursor = 0
        for partition, count in _quota_partition_counts(quota):
            selected = ranked[cursor:cursor + count]
            if len(selected) != count:
                raise FreshNormalSuccessorError("successor plan cannot satisfy its fixed reserve-only quota")
            cursor += count
            for record in selected:
                assigned.append({
                    "caseId": record["caseId"],
                    "category": record["category"],
                    "relativePath": record["relativePath"],
                    "sourceSha256": record["sourceSha256"],
                    "sourceGroupId": record["sourceGroupId"],
                    "acquisitionStratum": record["acquisitionStratum"],
                    "expectedRemoteSha256": record["expectedRemoteSha256"],
                    "expectedRemoteBytes": record["expectedRemoteBytes"],
                    "kind": record["kind"],
                    "defect": record["defect"],
                    "partition": partition,
                })
        if cursor != len(ranked):
            raise FreshNormalSuccessorError("successor reserve allocation did not consume every delegated input")
    result = sorted(assigned, key=lambda record: record["caseId"])
    if len(result) != 93 or len({record["sourceSha256"] for record in result}) != len(result):
        raise FreshNormalSuccessorError("successor reserve allocation is not one-to-one")
    return result


def _partition_identity(records: Iterable[dict[str, Any]], partition: str) -> str:
    selected = [record for record in records if record["partition"] == partition]
    return canonical_json_sha256(selected)


def _build_successor_envelope(
    seal: dict[str, Any],
    *,
    seal_file_sha256: str,
    plan: dict[str, Any],
    plan_file_sha256: str,
) -> dict[str, Any]:
    records = _allocate_successor_records(seal, plan)
    historical_exclusion = {
        "policy": "PARENT_FIT_TUNING_SELECTION_HISTORICAL_NEVER_CANDIDATES",
        "sourceCount": len(seal["historicalParentSources"]),
        "sourceIdentitySha256": seal["historicalParentSourceIdentitySha256"],
        "partitions": list(PARENT_HISTORICAL_PARTITIONS),
    }
    document: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_SUCCESSOR_ENVELOPE_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_SUCCESSOR_ENVELOPE_PURPOSE,
        "phase": FRESH_NORMAL_SUCCESSOR_ENVELOPE_PHASE,
        "blindPolicy": FRESH_NORMAL_SUCCESSOR_BLIND_POLICY,
        "resultLabel": FRESH_NORMAL_SUCCESSOR_RESULT_LABEL,
        "independenceLabel": FRESH_NORMAL_SUCCESSOR_INDEPENDENCE_LABEL,
        "delegationPolicy": FRESH_NORMAL_SUCCESSOR_DELEGATION_POLICY,
        "sealFileSha256": seal_file_sha256,
        "sealDeclaredSha256": seal["successorSealSha256"],
        "planFileSha256": plan_file_sha256,
        "planDeclaredSha256": plan["successorPlanSha256"],
        "parentEvidence": dict(seal["parentEvidence"]),
        "historicalExclusion": historical_exclusion,
        "preservedParentNormalConfirmation": dict(seal["preservedParentNormalConfirmation"]),
        "allocation": {
            "algorithm": FRESH_NORMAL_SUCCESSOR_ALLOCATION_ALGORITHM,
            "seedSha256": plan["allocationSeedSha256"],
            "categoryQuotas": [dict(quota) for quota in SUCCESSOR_CATEGORY_QUOTAS],
        },
        "records": records,
        "successorPartitionIdentities": {
            partition: _partition_identity(records, partition)
            for partition in SUCCESSOR_PARTITIONS
        },
    }
    document["successorEnvelopeSha256"] = _document_digest(document, "successorEnvelopeSha256")
    return document


def _validate_envelope_records(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 93:
        raise FreshNormalSuccessorError("successor envelope must contain exactly 93 allocated reserve records")
    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_paths: set[str] = set()
    for raw in value:
        item = _require_exact_fields(raw, name="successor envelope record", required=ENVELOPE_RECORD_FIELDS)
        case_id = _require_string(item.get("caseId"), name="successor envelope record caseId")
        source_sha256 = _require_sha256(item.get("sourceSha256"), name="successor envelope record sourceSha256")
        relative_path = _safe_relative_path(item.get("relativePath"), name="successor envelope record relativePath")
        partition = item.get("partition")
        if partition not in SUCCESSOR_PARTITIONS:
            raise FreshNormalSuccessorError("successor envelope record partition is unsupported")
        if item.get("acquisitionStratum") != "OFFICIAL_MVTEC_TRAIN_GOOD":
            raise FreshNormalSuccessorError("successor envelope record acquisition stratum is unsupported")
        if item.get("kind") != "NOMINAL" or item.get("defect") != "good":
            raise FreshNormalSuccessorError("successor envelope record must be nominal good only")
        remote_sha256 = _require_sha256(item.get("expectedRemoteSha256"), name="successor envelope record expectedRemoteSha256")
        if remote_sha256 != source_sha256:
            raise FreshNormalSuccessorError("successor envelope record source digest is inconsistent")
        if case_id in seen_case_ids or source_sha256 in seen_sources or relative_path.as_posix() in seen_paths:
            raise FreshNormalSuccessorError("successor envelope records are duplicated")
        seen_case_ids.add(case_id)
        seen_sources.add(source_sha256)
        seen_paths.add(relative_path.as_posix())
        result.append({
            "caseId": case_id,
            "category": _require_string(item.get("category"), name="successor envelope record category"),
            "relativePath": relative_path.as_posix(),
            "sourceSha256": source_sha256,
            "sourceGroupId": _require_string(item.get("sourceGroupId"), name="successor envelope record sourceGroupId"),
            "acquisitionStratum": "OFFICIAL_MVTEC_TRAIN_GOOD",
            "expectedRemoteSha256": remote_sha256,
            "expectedRemoteBytes": _require_positive_int(item.get("expectedRemoteBytes"), name="successor envelope record expectedRemoteBytes"),
            "kind": "NOMINAL",
            "defect": "good",
            "partition": str(partition),
        })
    if result != sorted(result, key=lambda item: item["caseId"]):
        raise FreshNormalSuccessorError("successor envelope records must be sorted by caseId")
    return result


def _validate_envelope_document(document: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal successor envelope", required=ENVELOPE_FIELDS)
    _validate_static_scope(
        document,
        schema=FRESH_NORMAL_SUCCESSOR_ENVELOPE_SCHEMA,
        purpose=FRESH_NORMAL_SUCCESSOR_ENVELOPE_PURPOSE,
        phase=FRESH_NORMAL_SUCCESSOR_ENVELOPE_PHASE,
        description="fresh normal successor envelope",
    )
    if document.get("independenceLabel") != FRESH_NORMAL_SUCCESSOR_INDEPENDENCE_LABEL:
        raise FreshNormalSuccessorError("successor envelope independence label is unsafe")
    for field in ("sealFileSha256", "sealDeclaredSha256", "planFileSha256", "planDeclaredSha256"):
        _require_sha256(document.get(field), name=f"successor envelope {field}")
    parent_evidence = _validate_parent_evidence(document.get("parentEvidence"))
    historical_exclusion = _require_exact_fields(
        document.get("historicalExclusion"),
        name="successor historical exclusion",
        required=HISTORICAL_EXCLUSION_FIELDS,
    )
    if historical_exclusion.get("policy") != "PARENT_FIT_TUNING_SELECTION_HISTORICAL_NEVER_CANDIDATES":
        raise FreshNormalSuccessorError("successor historical exclusion policy is unsafe")
    _require_positive_int(historical_exclusion.get("sourceCount"), name="successor historical exclusion sourceCount")
    _require_sha256(historical_exclusion.get("sourceIdentitySha256"), name="successor historical exclusion sourceIdentitySha256")
    if historical_exclusion.get("partitions") != list(PARENT_HISTORICAL_PARTITIONS):
        raise FreshNormalSuccessorError("successor historical exclusion partitions are unsafe")
    preserved_confirmation = _validate_preserved_confirmation(
        document.get("preservedParentNormalConfirmation"),
        parent_evidence=parent_evidence,
    )
    allocation = _require_exact_fields(document.get("allocation"), name="successor allocation", required=ALLOCATION_FIELDS)
    if allocation.get("algorithm") != FRESH_NORMAL_SUCCESSOR_ALLOCATION_ALGORITHM:
        raise FreshNormalSuccessorError("successor allocation algorithm is unsupported")
    _require_sha256(allocation.get("seedSha256"), name="successor allocation seedSha256")
    _validate_plan_quotas(allocation.get("categoryQuotas"))
    records = _validate_envelope_records(document.get("records"))
    expected_counts = {
        partition: {
            quota["category"]: int(dict(_quota_partition_counts(quota))[partition])
            for quota in SUCCESSOR_CATEGORY_QUOTAS
        }
        for partition in SUCCESSOR_PARTITIONS
    }
    actual_counts: dict[str, dict[str, int]] = {partition: {} for partition in SUCCESSOR_PARTITIONS}
    for record in records:
        by_category = actual_counts[record["partition"]]
        by_category[record["category"]] = by_category.get(record["category"], 0) + 1
    if actual_counts != expected_counts:
        raise FreshNormalSuccessorError("successor envelope records do not match the fixed category quotas")
    partition_identities = _require_exact_fields(
        document.get("successorPartitionIdentities"),
        name="successor partition identities",
        required=set(SUCCESSOR_PARTITIONS),
    )
    for partition in SUCCESSOR_PARTITIONS:
        actual = _require_sha256(partition_identities.get(partition), name=f"successor {partition} identity")
        if actual != _partition_identity(records, partition):
            raise FreshNormalSuccessorError("successor partition identity does not match records")
    if document.get("successorEnvelopeSha256") != _document_digest(document, "successorEnvelopeSha256"):
        raise FreshNormalSuccessorError("fresh normal successor envelope digest does not match")
    return {
        **document,
        "parentEvidence": parent_evidence,
        "historicalExclusion": dict(historical_exclusion),
        "preservedParentNormalConfirmation": preserved_confirmation,
        "allocation": dict(allocation),
        "records": records,
        "successorPartitionIdentities": dict(partition_identities),
    }


def _reserve_input_from_parent(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "relativePath": record["relativePath"],
        "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"],
        "acquisitionStratum": record["acquisitionStratum"],
        "expectedRemoteSha256": record["expectedRemoteSha256"],
        "expectedRemoteBytes": record["expectedRemoteBytes"],
        "kind": record["kind"],
        "defect": record["defect"],
        "parentPartition": PARENT_RESERVE_PARTITION,
    }


def _historical_source_from_parent(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "partition": record["partition"],
        "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"],
    }


def _confirmation_source_identity(records: list[dict[str, Any]]) -> str:
    return canonical_json_sha256([
        {
            "caseId": record["caseId"],
            "category": record["category"],
            "sourceSha256": record["sourceSha256"],
            "sourceGroupId": record["sourceGroupId"],
        }
        for record in sorted(records, key=lambda record: record["caseId"])
    ])


def _classify_parent_records(
    records: list[dict[str, Any]],
    *,
    holdout_document: dict[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    historical_records = [record for record in records if record["partition"] in PARENT_HISTORICAL_PARTITIONS]
    reserve_records = [record for record in records if record["partition"] == PARENT_RESERVE_PARTITION]
    confirmation_records = [record for record in records if record["partition"] == PARENT_CONFIRMATION_PARTITION]
    if len(reserve_records) != 93 or len(confirmation_records) != 96:
        raise FreshNormalSuccessorError("parent holdout does not expose the fixed unconsumed reserve/confirmation counts")
    if not historical_records or {record["partition"] for record in historical_records} != set(PARENT_HISTORICAL_PARTITIONS):
        raise FreshNormalSuccessorError("parent holdout does not expose all historical FIT/tuning/selection partitions")
    historical_sources = tuple(sorted((_historical_source_from_parent(record) for record in historical_records), key=lambda item: item["caseId"]))
    reserve_inputs = tuple(sorted((_reserve_input_from_parent(record) for record in reserve_records), key=lambda item: item["caseId"]))
    historical_hashes = {record["sourceSha256"] for record in historical_sources}
    reserve_hashes = {record["sourceSha256"] for record in reserve_inputs}
    confirmation_hashes = {record["sourceSha256"] for record in confirmation_records}
    if len(historical_hashes) != len(historical_sources) or len(reserve_hashes) != len(reserve_inputs):
        raise FreshNormalSuccessorError("parent holdout source hashes are not one-to-one")
    if historical_hashes.intersection(reserve_hashes) or historical_hashes.intersection(confirmation_hashes) or reserve_hashes.intersection(confirmation_hashes):
        raise FreshNormalSuccessorError("parent holdout partitions share source hashes")
    preserved_confirmation = {
        "partition": PARENT_CONFIRMATION_PARTITION,
        "recordCount": len(confirmation_records),
        "normalConfirmationIdentitySha256": holdout_document["normalConfirmationIdentitySha256"],
        "sourceIdentitySha256": _confirmation_source_identity(confirmation_records),
        "accessPolicy": "PRESERVED_NOT_DELEGATED_NOT_OPENED",
    }
    # Validate here as well so malformed parent inputs are rejected before a
    # permanent slot is consumed.
    _validate_historical_sources(list(historical_sources))
    _validate_reserve_inputs(list(reserve_inputs))
    return historical_sources, reserve_inputs, preserved_confirmation


def _parent_evidence_from_context(context: _ParentContext) -> dict[str, Any]:
    return {
        "holdoutManifestFileSha256": context.holdout_file_sha256,
        "holdoutManifestDeclaredSha256": context.holdout_document["normalHoldoutManifestSha256"],
        "selectionContractFileSha256": context.contract_file_sha256,
        "selectionContractDeclaredSha256": context.contract["contractSha256"],
        "selectionClaimFileSha256": context.claim_file_sha256,
        "selectionClaimDeclaredSha256": context.claim["claimSha256"],
        "selectionReceiptFileSha256": context.selection_observation["selectionReceiptFileSha256"],
        "selectionReceiptDeclaredSha256": context.selection_observation["selectionReceiptDeclaredSha256"],
        "selectionObservationFileSha256": context.selection_observation_file_sha256,
        "selectionObservationDeclaredSha256": context.selection_observation["selectionObservationSha256"],
        "selectionLockFileSha256": context.selection_lock_file_sha256,
        "selectionLockDeclaredSha256": context.selection_lock["selectionLockSha256"],
        "selectionLockState": "NO_ELIGIBLE_CONFIGURATION",
        "parentReserveUntouchedIdentitySha256": context.holdout_document["reserveUntouchedIdentitySha256"],
        "parentNormalConfirmationIdentitySha256": context.holdout_document["normalConfirmationIdentitySha256"],
    }


def _assert_parent_context_is_no_eligible(context: _ParentContext) -> None:
    decision = context.selection_lock.get("decision")
    if not isinstance(decision, dict) or decision.get("state") != "NO_ELIGIBLE_CONFIGURATION" or decision.get("selectedCandidateId") is not None:
        raise FreshNormalSuccessorError("successor sealing requires a parent NO_ELIGIBLE_CONFIGURATION lock")


def _validate_contract_against_parent(
    contract: dict[str, Any],
    *,
    holdout_document: dict[str, Any],
    holdout_file_sha256: str,
) -> None:
    binding = contract.get("holdout")
    if not isinstance(binding, dict):
        raise FreshNormalSuccessorError("parent selection contract holdout binding is missing")
    expected = {
        "manifestFileSha256": holdout_file_sha256,
        "manifestDeclaredSha256": holdout_document["normalHoldoutManifestSha256"],
        "developmentIdentitySha256": holdout_document["developmentIdentitySha256"],
        "normalSelectionIdentitySha256": holdout_document["normalSelectionIdentitySha256"],
        "normalConfirmationIdentitySha256": holdout_document["normalConfirmationIdentitySha256"],
    }
    for field, expected_value in expected.items():
        if binding.get(field) != expected_value:
            raise FreshNormalSuccessorError("parent selection contract does not bind the closed parent holdout")


def _assert_confirmation_slots_unconsumed(contract: dict[str, Any], *, repository_root: Path) -> None:
    # These are existence checks only.  They intentionally do not read a
    # confirmation artifact and never open a confirmation image.
    paths = (
        selection.fresh_normal_consumption_path(
            contract,
            partition=PARENT_CONFIRMATION_PARTITION,
            artifact="claim",
            repository_root=repository_root,
        ),
        observation.fresh_confirmation_receipt_path(contract, repository_root=repository_root),
        observation.fresh_confirmation_observation_path(contract, repository_root=repository_root),
    )
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FreshNormalSuccessorError("parent NORMAL_CONFIRMATION is no longer unconsumed")


def _load_parent_context(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    *,
    repository_root: Path,
) -> _ParentContext:
    """Load the parent evidence through JSON-only validation paths.

    This function deliberately contains no source-root argument.  Calling it
    cannot open a parent image, including a confirmation image.
    """

    holdout_document, holdout_file_sha256 = _read_external_json(
        parent_holdout_path,
        description="parent fresh normal holdout manifest",
        repository_root=repository_root,
    )
    try:
        records = holdout._validate_closed_normal_holdout_document(holdout_document)
    except holdout.NormalHoldoutError as error:
        raise FreshNormalSuccessorError("parent fresh normal holdout manifest is unsafe") from error
    try:
        contract, contract_file_sha256 = selection.load_validated_fresh_selection_contract(
            parent_selection_contract_path,
            repository_root=repository_root,
        )
        claim, claim_file_sha256 = selection.load_validated_fresh_selection_claim_for_contract(
            parent_selection_contract_path,
            repository_root=repository_root,
        )
        selection_observation, selection_observation_file_sha256 = observation.load_validated_fresh_selection_observation_for_contract(
            parent_selection_contract_path,
            repository_root=repository_root,
        )
        selection_lock, selection_lock_file_sha256 = observation.load_validated_fresh_selection_lock_for_contract(
            parent_selection_contract_path,
            repository_root=repository_root,
        )
    except (selection.FreshNormalSelectionError, observation.FreshNormalObservationError) as error:
        raise FreshNormalSuccessorError("parent selection JSON evidence is unsafe") from error
    _validate_contract_against_parent(
        contract,
        holdout_document=holdout_document,
        holdout_file_sha256=holdout_file_sha256,
    )
    decision = selection_lock.get("decision")
    if not isinstance(decision, dict) or decision.get("state") != "NO_ELIGIBLE_CONFIGURATION" or decision.get("selectedCandidateId") is not None:
        raise FreshNormalSuccessorError("successor sealing requires a parent NO_ELIGIBLE_CONFIGURATION lock")
    _assert_confirmation_slots_unconsumed(contract, repository_root=repository_root)
    historical_sources, reserve_inputs, preserved_confirmation = _classify_parent_records(
        records,
        holdout_document=holdout_document,
    )
    return _ParentContext(
        holdout_path=parent_holdout_path,
        holdout_document=holdout_document,
        holdout_file_sha256=holdout_file_sha256,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
        selection_observation=selection_observation,
        selection_observation_file_sha256=selection_observation_file_sha256,
        selection_lock=selection_lock,
        selection_lock_file_sha256=selection_lock_file_sha256,
        historical_sources=historical_sources,
        reserve_inputs=reserve_inputs,
        preserved_confirmation=preserved_confirmation,
    )


def _successor_seal_slot_key(holdout_document: dict[str, Any], *, holdout_file_sha256: str) -> str:
    return canonical_json_sha256({
        "schemaVersion": FRESH_NORMAL_SUCCESSOR_SEAL_SCHEMA,
        "holdoutManifestFileSha256": holdout_file_sha256,
        "holdoutManifestDeclaredSha256": holdout_document["normalHoldoutManifestSha256"],
        "parentReserveUntouchedIdentitySha256": holdout_document["reserveUntouchedIdentitySha256"],
    })


def _successor_seal_path_for_context(context: _ParentContext, *, repository_root: Path) -> Path:
    root = context.holdout_path.parent / FRESH_NORMAL_SUCCESSOR_PARTITION_ACCESS_DIRECTORY
    if (root.exists() or root.is_symlink()) and _is_link_or_reparse_point(root):
        raise FreshNormalSuccessorError("successor partition access directory contains a symbolic link or reparse point")
    _reject_links_on_existing_path(root.parent, description="successor partition access directory")
    if _is_under(repository_root, root) or _is_under(root, repository_root):
        raise FreshNormalSuccessorError("successor partition access directory must stay outside the Git working tree")
    slot_key = _successor_seal_slot_key(context.holdout_document, holdout_file_sha256=context.holdout_file_sha256)
    return root / f"successor-reserve--{slot_key[7:]}.seal.json"


def fresh_normal_successor_seal_path(
    parent_holdout_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Derive the sole external seal slot for a parent reserve partition."""

    document, file_sha256 = _read_external_json(
        parent_holdout_path,
        description="parent fresh normal holdout manifest",
        repository_root=repository_root,
    )
    try:
        holdout._validate_closed_normal_holdout_document(document)
    except holdout.NormalHoldoutError as error:
        raise FreshNormalSuccessorError("parent fresh normal holdout manifest is unsafe") from error
    root = parent_holdout_path.parent / FRESH_NORMAL_SUCCESSOR_PARTITION_ACCESS_DIRECTORY
    if (root.exists() or root.is_symlink()) and _is_link_or_reparse_point(root):
        raise FreshNormalSuccessorError("successor partition access directory contains a symbolic link or reparse point")
    _reject_links_on_existing_path(root.parent, description="successor partition access directory")
    if _is_under(repository_root, root) or _is_under(root, repository_root):
        raise FreshNormalSuccessorError("successor partition access directory must stay outside the Git working tree")
    slot_key = _successor_seal_slot_key(document, holdout_file_sha256=file_sha256)
    return root / f"successor-reserve--{slot_key[7:]}.seal.json"


def _build_successor_seal(context: _ParentContext) -> dict[str, Any]:
    _assert_parent_context_is_no_eligible(context)
    historical_sources = [dict(item) for item in context.historical_sources]
    reserve_inputs = [dict(item) for item in context.reserve_inputs]
    document: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_SUCCESSOR_SEAL_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_SUCCESSOR_SEAL_PURPOSE,
        "phase": FRESH_NORMAL_SUCCESSOR_SEAL_PHASE,
        "blindPolicy": FRESH_NORMAL_SUCCESSOR_BLIND_POLICY,
        "resultLabel": FRESH_NORMAL_SUCCESSOR_RESULT_LABEL,
        "delegationPolicy": FRESH_NORMAL_SUCCESSOR_DELEGATION_POLICY,
        "parentEvidence": _parent_evidence_from_context(context),
        "historicalParentSources": historical_sources,
        "historicalParentSourceIdentitySha256": canonical_json_sha256(historical_sources),
        "preservedParentNormalConfirmation": dict(context.preserved_confirmation),
        "delegatedReserveInputs": reserve_inputs,
        "delegatedReserveInputIdentitySha256": canonical_json_sha256(reserve_inputs),
    }
    document["successorSealSha256"] = _document_digest(document, "successorSealSha256")
    return document


def create_fresh_normal_successor_seal(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Permanently delegate only a failed parent's unopened reserve inputs."""

    context = _load_parent_context(
        parent_holdout_path,
        parent_selection_contract_path,
        repository_root=repository_root,
    )
    path = _successor_seal_path_for_context(context, repository_root=repository_root)
    document = _build_successor_seal(context)
    _validate_seal_document(document)
    _write_external_json(
        path,
        document,
        description="fresh normal successor seal",
        repository_root=repository_root,
    )
    return document


def _load_successor_seal_for_parent(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    *,
    repository_root: Path,
) -> tuple[_ParentContext, dict[str, Any], str]:
    context = _load_parent_context(
        parent_holdout_path,
        parent_selection_contract_path,
        repository_root=repository_root,
    )
    path = _successor_seal_path_for_context(context, repository_root=repository_root)
    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal successor seal",
        repository_root=repository_root,
    )
    parsed = _validate_seal_document(document)
    if parsed != _build_successor_seal(context):
        raise FreshNormalSuccessorError("fresh normal successor seal does not bind the closed parent evidence")
    return context, parsed, file_sha256


def load_validated_fresh_normal_successor_seal(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Revalidate a successor seal against the full JSON-only parent chain."""

    _context, document, file_sha256 = _load_successor_seal_for_parent(
        parent_holdout_path,
        parent_selection_contract_path,
        repository_root=repository_root,
    )
    return document, file_sha256


def create_fresh_normal_successor_plan(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    output_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Write an immutable deterministic allocation plan for a sealed reserve."""

    _context, seal, seal_file_sha256 = _load_successor_seal_for_parent(
        parent_holdout_path,
        parent_selection_contract_path,
        repository_root=repository_root,
    )
    document = _build_successor_plan(seal, seal_file_sha256=seal_file_sha256)
    _validate_plan_document(document, seal=seal, seal_file_sha256=seal_file_sha256)
    _write_external_json(
        output_path,
        document,
        description="fresh normal successor plan",
        repository_root=repository_root,
    )
    return document


def _load_successor_plan_for_parent(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    *,
    repository_root: Path,
) -> tuple[_ParentContext, dict[str, Any], str, dict[str, Any], str]:
    context, seal, seal_file_sha256 = _load_successor_seal_for_parent(
        parent_holdout_path,
        parent_selection_contract_path,
        repository_root=repository_root,
    )
    document, file_sha256 = _read_external_json(
        plan_path,
        description="fresh normal successor plan",
        repository_root=repository_root,
    )
    parsed = _validate_plan_document(document, seal=seal, seal_file_sha256=seal_file_sha256)
    return context, seal, seal_file_sha256, parsed, file_sha256


def load_validated_fresh_normal_successor_plan(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Revalidate a plan against its fixed seal and parent JSON evidence."""

    _context, _seal, _seal_file, plan, plan_file = _load_successor_plan_for_parent(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        repository_root=repository_root,
    )
    return plan, plan_file


def create_fresh_normal_successor_envelope(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    output_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Materialize a deterministic, reserve-only successor phase envelope."""

    _context, seal, seal_file_sha256, plan, plan_file_sha256 = _load_successor_plan_for_parent(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        repository_root=repository_root,
    )
    document = _build_successor_envelope(
        seal,
        seal_file_sha256=seal_file_sha256,
        plan=plan,
        plan_file_sha256=plan_file_sha256,
    )
    _validate_envelope_document(document)
    _write_external_json(
        output_path,
        document,
        description="fresh normal successor envelope",
        repository_root=repository_root,
    )
    return document


def load_validated_fresh_normal_successor_envelope(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Revalidate an envelope against the fixed parent and deterministic plan."""

    _context, seal, seal_file_sha256, plan, plan_file_sha256 = _load_successor_plan_for_parent(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        repository_root=repository_root,
    )
    document, file_sha256 = _read_external_json(
        envelope_path,
        description="fresh normal successor envelope",
        repository_root=repository_root,
    )
    parsed = _validate_envelope_document(document)
    expected = _build_successor_envelope(
        seal,
        seal_file_sha256=seal_file_sha256,
        plan=plan,
        plan_file_sha256=plan_file_sha256,
    )
    if parsed != expected:
        raise FreshNormalSuccessorError("fresh normal successor envelope does not bind the sealed reserve")
    return parsed, file_sha256


def load_successor_safe_normal_inputs(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    partitions: object,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Open and verify only the requested successor partitions.

    The complete parent/seal/plan/envelope JSON chain is revalidated before a
    source image is opened.  It then validates/decode-checks bytes only for
    the requested successor records; it never touches parent confirmation or
    historical partition files.
    """

    if isinstance(partitions, str):
        raise FreshNormalSuccessorError("successor evaluation partitions must be a non-string collection")
    try:
        requested = set(partitions)
    except TypeError as error:
        raise FreshNormalSuccessorError("successor evaluation partitions must be a collection") from error
    if not requested or not requested.issubset(set(SUCCESSOR_PARTITIONS)) or any(not isinstance(item, str) for item in requested):
        raise FreshNormalSuccessorError("successor evaluation partitions are unsupported")
    envelope, file_sha256 = load_validated_fresh_normal_successor_envelope(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        repository_root=repository_root,
    )
    _require_external_directory(source_root, description="successor source root", repository_root=repository_root)
    selected: list[dict[str, Any]] = []
    for record in envelope["records"]:
        if record["partition"] not in requested:
            continue
        relative = _safe_relative_path(record["relativePath"], name="successor envelope record relativePath")
        path = _safe_file_under(
            source_root,
            relative,
            description="successor evaluation image",
            repository_root=repository_root,
        )
        if sha256_file(path) != record["sourceSha256"] or path.stat().st_size != record["expectedRemoteBytes"]:
            raise FreshNormalSuccessorError("successor evaluation image bytes do not match the sealed reserve")
        _verify_decodable_image(path, description="successor evaluation image")
        if sha256_file(path) != record["sourceSha256"]:
            raise FreshNormalSuccessorError("successor evaluation image changed while it was decoded")
        selected.append({**record, "imagePath": path})
    if not selected:
        raise FreshNormalSuccessorError("successor envelope has no records in the requested partitions")
    return envelope, file_sha256, selected
