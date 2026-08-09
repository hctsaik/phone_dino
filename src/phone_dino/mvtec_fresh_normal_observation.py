"""One-time fresh-normal selection and confirmation observations.

This module is deliberately separate from the historical V3--V5 tooling.  It
consumes only the normal-only partitions frozen by
``mvtec_fresh_normal_selection``: verified FIT parents plus their validated
FIT-only derivatives form the prototypes; a receipt is durably created before
the first held-out query image is opened.  The JSON-only lock deliberately
does not invoke this image-reading path.

The fixed sibling artifacts make a crash after a receipt a consumed attempt
for this tool-mediated protocol.  They are research observations only: no
function in this module emits a production decision, PASS/FAIL, or equipment
action.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from PIL import Image, ImageOps

from phone_dino import mvtec_fresh_fit_augmentation as augmentation
from phone_dino import mvtec_fresh_normal_selection as selection
from phone_dino import mvtec_normal_holdout as holdout
from phone_dino import mvtec_normal_holdout_evaluator as evaluator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

FRESH_NORMAL_SELECTION_RECEIPT_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-selection-receipt/1.0"
FRESH_NORMAL_SELECTION_OBSERVATION_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-selection-observation/1.0"
FRESH_NORMAL_SELECTION_LOCK_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-selection-lock/1.0"
FRESH_NORMAL_CONFIRMATION_CLAIM_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-confirmation-claim/1.0"
FRESH_NORMAL_CONFIRMATION_RECEIPT_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-confirmation-receipt/1.0"
FRESH_NORMAL_CONFIRMATION_OBSERVATION_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-confirmation-observation/1.0"

FRESH_NORMAL_SELECTION_RECEIPT_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_SELECTION_RECEIPT"
FRESH_NORMAL_SELECTION_OBSERVATION_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_SELECTION_OBSERVATION"
FRESH_NORMAL_SELECTION_LOCK_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_SELECTION_LOCK"
FRESH_NORMAL_CONFIRMATION_CLAIM_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_CONFIRMATION_CLAIM"
FRESH_NORMAL_CONFIRMATION_RECEIPT_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_CONFIRMATION_RECEIPT"
FRESH_NORMAL_CONFIRMATION_OBSERVATION_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_CONFIRMATION_OBSERVATION"

FRESH_NORMAL_SELECTION_RECEIPT_PHASE = "NORMAL_SELECTION_RECEIPT"
FRESH_NORMAL_SELECTION_OBSERVATION_PHASE = "NORMAL_SELECTION_OBSERVATION"
FRESH_NORMAL_SELECTION_LOCK_PHASE = "NORMAL_SELECTION_LOCK"
FRESH_NORMAL_CONFIRMATION_CLAIM_PHASE = "NORMAL_CONFIRMATION_CLAIM"
FRESH_NORMAL_CONFIRMATION_RECEIPT_PHASE = "NORMAL_CONFIRMATION_RECEIPT"
FRESH_NORMAL_CONFIRMATION_OBSERVATION_PHASE = "NORMAL_CONFIRMATION_OBSERVATION"

FRESH_NORMAL_BLIND_POLICY = selection.FRESH_NORMAL_SELECTION_BLIND_POLICY
FRESH_NORMAL_SELECTION_RECEIPT_SLOT = "FRESH_NORMAL_SELECTION_RECEIPT_V1"
FRESH_NORMAL_CONFIRMATION_CLAIM_SLOT = "FRESH_NORMAL_CONFIRMATION_CONSUMPTION_V1"
FRESH_NORMAL_CONFIRMATION_RECEIPT_SLOT = "FRESH_NORMAL_CONFIRMATION_RECEIPT_V1"

FRESH_NORMAL_CONFIRMATION_INPUT = {
    "partition": "NORMAL_CONFIRMATION",
    "kind": "NOMINAL",
    "defect": "good",
    "rawOnly": True,
    "oneTimeClaimRequired": True,
}
FRESH_NORMAL_CONFIRMATION_RESULT_SCOPE = "OBSERVATION_ONLY_NO_CONFIGURATION_CHANGE_OR_PROMOTION"
NO_PERSISTENT_QUERY_CACHE = "NO_PERSISTENT_QUERY_CACHE"

SELECTION_RECEIPT_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "selectionInput",
    "receiptSlot",
    "contractFileSha256",
    "contractDeclaredSha256",
    "selectionClaimFileSha256",
    "selectionClaimDeclaredSha256",
    "holdout",
    "augmentation",
    "featureExtractorIdentitySha256",
    "candidateUniverseIdentitySha256",
    "fitFeatureInputIdentitySha256",
    "fitFeatureInputCount",
    "fitOriginalInputCount",
    "fitAugmentedInputCount",
    "selectionReceiptSha256",
}
SELECTION_OBSERVATION_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "selectionInput",
    "contractFileSha256",
    "contractDeclaredSha256",
    "selectionClaimFileSha256",
    "selectionClaimDeclaredSha256",
    "selectionReceiptFileSha256",
    "selectionReceiptDeclaredSha256",
    "holdout",
    "augmentation",
    "featureExtractor",
    "featureExtractorIdentitySha256",
    "normalSelectionInputs",
    "normalSelectionInputIdentitySha256",
    "candidateObservations",
    "normalOnlyEvidence",
    "execution",
    "selectionObservationSha256",
}
SELECTION_CANDIDATE_OBSERVATION_FIELDS = {
    "candidateId",
    "candidateConfiguration",
    "candidateConfigurationSha256",
    "thresholds",
    "thresholdsIdentitySha256",
    "selectionScores",
    "categoryMetrics",
}
SELECTION_LOCK_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "contractFileSha256",
    "contractDeclaredSha256",
    "selectionClaimFileSha256",
    "selectionClaimDeclaredSha256",
    "selectionReceiptFileSha256",
    "selectionReceiptDeclaredSha256",
    "selectionObservationFileSha256",
    "selectionObservationDeclaredSha256",
    "candidateEvaluations",
    "decision",
    "selectionLockSha256",
}
LOCK_CANDIDATE_EVALUATION_FIELDS = {
    "candidateId",
    "categoryMetrics",
    "gatePassed",
    "gateRejectionReasons",
    "objectiveValues",
}
LOCK_DECISION_FIELDS = {
    "state",
    "selectedCandidateId",
    "resultScope",
    "automaticProductionPromotion",
    "automaticConfirmation",
}
CONFIRMATION_CLAIM_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "confirmationInput",
    "claimSlot",
    "contractFileSha256",
    "contractDeclaredSha256",
    "selectionLockFileSha256",
    "selectionLockDeclaredSha256",
    "holdout",
    "augmentation",
    "featureExtractorIdentitySha256",
    "selectedCandidateId",
    "candidateConfiguration",
    "candidateConfigurationSha256",
    "thresholds",
    "thresholdsIdentitySha256",
    "confirmationClaimSha256",
}
CONFIRMATION_RECEIPT_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "confirmationInput",
    "receiptSlot",
    "contractFileSha256",
    "contractDeclaredSha256",
    "selectionLockFileSha256",
    "selectionLockDeclaredSha256",
    "confirmationClaimFileSha256",
    "confirmationClaimDeclaredSha256",
    "holdout",
    "augmentation",
    "featureExtractorIdentitySha256",
    "fitFeatureInputIdentitySha256",
    "fitFeatureInputCount",
    "fitOriginalInputCount",
    "fitAugmentedInputCount",
    "confirmationReceiptSha256",
}
CONFIRMATION_OBSERVATION_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "confirmationInput",
    "resultScope",
    "contractFileSha256",
    "contractDeclaredSha256",
    "selectionLockFileSha256",
    "selectionLockDeclaredSha256",
    "confirmationClaimFileSha256",
    "confirmationClaimDeclaredSha256",
    "confirmationReceiptFileSha256",
    "confirmationReceiptDeclaredSha256",
    "holdout",
    "augmentation",
    "featureExtractor",
    "featureExtractorIdentitySha256",
    "selectedCandidateId",
    "candidateConfiguration",
    "candidateConfigurationSha256",
    "thresholds",
    "thresholdsIdentitySha256",
    "normalConfirmationInputs",
    "normalConfirmationInputIdentitySha256",
    "confirmationScores",
    "categoryMetrics",
    "normalOnlyEvidence",
    "execution",
    "confirmationObservationSha256",
}
CATEGORY_METRIC_FIELDS = {
    "caseCount",
    "aboveThresholdCount",
    "aboveThresholdRate",
    "p95Score",
    "p95ScoreMinusThreshold",
    "maximumScore",
    "maximumScoreMinusThreshold",
}
NORMAL_ONLY_EVIDENCE_FIELDS = {
    "prototypeInputCount",
    "prototypeInputPartitions",
    "prototypeInputKinds",
    "fitOriginalInputCount",
    "fitAugmentedInputCount",
    "queryInputCount",
    "queryInputPartitions",
    "queryInputKinds",
    "blindInputCount",
    "anomalyInputCount",
    "maskInputCount",
    "persistentQueryCache",
}
EXECUTION_FIELDS = {
    "observerModuleSha256",
    "queryFeatureCachePolicy",
    "phaseTimingsSeconds",
    "python",
    "platform",
}
TIMING_FIELDS = {
    "inputPreflightSeconds",
    "prototypeFeatureSeconds",
    "receiptCommitSeconds",
    "queryFeatureSeconds",
    "scoringSeconds",
    "totalElapsedSeconds",
}


class FreshNormalObservationError(ValueError):
    """Raised when a fresh normal observation artifact is unsafe or invalid."""


def _wrap_selection_error(callable_: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return callable_(*args, **kwargs)
    except (selection.FreshNormalSelectionError, holdout.NormalHoldoutError, augmentation.FreshFitAugmentationError) as error:
        raise FreshNormalObservationError(str(error)) from error


def _canonical_json_sha256(document: Any) -> str:
    try:
        return selection.canonical_json_sha256(document)
    except selection.FreshNormalSelectionError as error:
        raise FreshNormalObservationError(str(error)) from error


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return _canonical_json_sha256(unsigned)


def _require_exact_fields(document: object, *, name: str, required: set[str]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise FreshNormalObservationError(f"{name} must be an object")
    missing = required.difference(document)
    if missing:
        raise FreshNormalObservationError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    unknown = set(document).difference(required)
    if unknown:
        raise FreshNormalObservationError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")
    return document


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FreshNormalObservationError(f"{name} must be an object")
    return value


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreshNormalObservationError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise FreshNormalObservationError(f"{name} must be a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise FreshNormalObservationError(f"{name} must be a SHA-256 digest") from error
    return digest


def _require_positive_int(value: object, *, name: str, maximum: int = 10_000_000) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= maximum:
        raise FreshNormalObservationError(f"{name} must be a positive integer no larger than {maximum}")
    return value


def _require_nonnegative_int(value: object, *, name: str, maximum: int = 10_000_000) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
        raise FreshNormalObservationError(f"{name} must be a non-negative integer no larger than {maximum}")
    return value


def _require_finite_number(value: object, *, name: str, minimum: float = 0.0, maximum: float = 2.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FreshNormalObservationError(f"{name} must be a finite number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise FreshNormalObservationError(f"{name} must be between {minimum} and {maximum}")
    return result


def _same_number(left: float, right: object) -> bool:
    return isinstance(right, (int, float)) and not isinstance(right, bool) and math.isclose(
        left, float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _validate_static_scope(
    document: dict[str, Any],
    *,
    schema: str,
    purpose: str,
    phase: str,
    description: str,
) -> None:
    if document.get("schemaVersion") != schema:
        raise FreshNormalObservationError(f"{description} schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise FreshNormalObservationError(f"{description} must be non-authoritative and non-production")
    if document.get("purpose") != purpose or document.get("phase") != phase:
        raise FreshNormalObservationError(f"{description} purpose or phase is unsafe")
    if document.get("blindPolicy") != FRESH_NORMAL_BLIND_POLICY:
        raise FreshNormalObservationError(f"{description} blind policy is unsafe")


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
                raise FreshNormalObservationError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _is_under(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _require_external_file(path: Path, *, description: str, repository_root: Path) -> None:
    if not isinstance(path, Path):
        raise FreshNormalObservationError(f"{description} path must be a Path")
    _reject_links_on_existing_path(path, description=description)
    if not path.is_file() or _is_link_or_reparse_point(path):
        raise FreshNormalObservationError(f"{description} must be a regular non-link file")
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise FreshNormalObservationError(f"{description} must stay outside the Git working tree")


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError as error:
        raise FreshNormalObservationError("unable to stat immutable input") from error
    return (status.st_dev, status.st_ino, status.st_mode, status.st_size, status.st_mtime_ns)


def _parse_json_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreshNormalObservationError(f"{description} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise FreshNormalObservationError(f"{description} contains a non-finite JSON value: {value}")

    try:
        document = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshNormalObservationError(f"unable to read {description}") from error
    try:
        selection._validate_json_value(document, name=description)
    except selection.FreshNormalSelectionError as error:
        raise FreshNormalObservationError(str(error)) from error
    if not isinstance(document, dict):
        raise FreshNormalObservationError(f"{description} must be a JSON object")
    return document


def _read_external_json(path: Path, *, description: str, repository_root: Path) -> tuple[dict[str, Any], str]:
    _require_external_file(path, description=description, repository_root=repository_root)
    before = _stat_signature(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FreshNormalObservationError(f"unable to read {description}") from error
    _require_external_file(path, description=description, repository_root=repository_root)
    if before != _stat_signature(path):
        raise FreshNormalObservationError(f"{description} changed while it was read")
    return _parse_json_bytes(raw, description=description), f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _prepare_fixed_external_output(path: Path, *, description: str, repository_root: Path) -> None:
    if not isinstance(path, Path):
        raise FreshNormalObservationError(f"{description} path must be a Path")
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise FreshNormalObservationError(f"{description} must stay outside the Git working tree")
    if path.exists() or path.is_symlink():
        raise FreshNormalObservationError(f"{description} already exists; the fixed slot is already consumed")
    _reject_links_on_existing_path(path.parent, description=description)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FreshNormalObservationError(f"unable to create {description} parent directory") from error
    _reject_links_on_existing_path(path.parent, description=description)


def _file_identity_from_status(status: os.stat_result) -> tuple[int, int, int, int]:
    return (status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode), status.st_size)


def _assert_written_slot_identity(
    path: Path,
    *,
    file_identity: tuple[int, int, int, int],
    description: str,
) -> None:
    """Ensure the path still names the exact non-link file just fsynced."""

    _reject_links_on_existing_path(path, description=description)
    if _is_link_or_reparse_point(path):
        raise FreshNormalObservationError(f"{description} became a symbolic link or reparse point while it was written")
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as error:
        raise FreshNormalObservationError(f"unable to stat durable {description}") from error
    if not stat.S_ISREG(current.st_mode) or _file_identity_from_status(current) != file_identity:
        raise FreshNormalObservationError(f"{description} path identity changed while it was written")


def _write_fixed_external_json_fsync(
    path: Path,
    document: dict[str, Any],
    *,
    description: str,
    repository_root: Path,
) -> None:
    """Atomically reserve a one-time JSON slot and durably write its payload.

    ``O_EXCL`` is the atomic consumption operation.  The content is fully
    serialized before the slot is created and ``fsync`` completes before a
    caller is allowed to open a held-out query image.
    """

    _prepare_fixed_external_output(path, description=description, repository_root=repository_root)
    payload = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    _reject_links_on_existing_path(path.parent, description=description)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise FreshNormalObservationError(f"{description} already exists; the fixed slot is already consumed") from error
    except OSError as error:
        raise FreshNormalObservationError(f"unable to atomically create {description}") from error
    written_identity: tuple[int, int, int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            written_identity = _file_identity_from_status(os.fstat(stream.fileno()))
    except OSError as error:
        raise FreshNormalObservationError(f"unable to durably write {description}") from error
    if written_identity is None:  # pragma: no cover - defensive fdopen invariant
        raise FreshNormalObservationError(f"unable to retain durable {description} identity")
    _assert_written_slot_identity(path, file_identity=written_identity, description=description)


def _consumption_artifact_path(
    contract: dict[str, Any],
    *,
    partition: str,
    artifact: str,
    repository_root: Path,
) -> Path:
    """Return a cohort-wide artifact path, never a contract-sibling slot.

    The selection-contract module owns the registry derivation, so copying a
    valid contract to a new directory cannot create another held-out lineage.
    """

    try:
        return selection.fresh_normal_consumption_path(
            contract,
            partition=partition,
            artifact=artifact,
            repository_root=repository_root,
        )
    except selection.FreshNormalSelectionError as error:
        raise FreshNormalObservationError(str(error)) from error


def fresh_selection_receipt_path(
    contract: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    return _consumption_artifact_path(
        contract,
        partition="NORMAL_SELECTION",
        artifact="receipt",
        repository_root=repository_root,
    )


def fresh_selection_observation_path(
    contract: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    return _consumption_artifact_path(
        contract,
        partition="NORMAL_SELECTION",
        artifact="observation",
        repository_root=repository_root,
    )


def fresh_selection_lock_path(
    contract: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    return _consumption_artifact_path(
        contract,
        partition="NORMAL_SELECTION",
        artifact="lock",
        repository_root=repository_root,
    )


def fresh_confirmation_claim_path(
    contract: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    return _consumption_artifact_path(
        contract,
        partition="NORMAL_CONFIRMATION",
        artifact="claim",
        repository_root=repository_root,
    )


def fresh_confirmation_receipt_path(
    contract: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    return _consumption_artifact_path(
        contract,
        partition="NORMAL_CONFIRMATION",
        artifact="receipt",
        repository_root=repository_root,
    )


def fresh_confirmation_observation_path(
    contract: dict[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    return _consumption_artifact_path(
        contract,
        partition="NORMAL_CONFIRMATION",
        artifact="observation",
        repository_root=repository_root,
    )


def _selection_input() -> dict[str, Any]:
    return dict(selection.FRESH_NORMAL_SELECTION_INPUT)


def _confirmation_input() -> dict[str, Any]:
    return dict(FRESH_NORMAL_CONFIRMATION_INPUT)


def _validate_selection_input(value: object) -> dict[str, Any]:
    document = _require_exact_fields(value, name="selectionInput", required=set(selection.FRESH_NORMAL_SELECTION_INPUT))
    if document != selection.FRESH_NORMAL_SELECTION_INPUT:
        raise FreshNormalObservationError("selectionInput must require raw NORMAL_SELECTION nominal-good data")
    return dict(document)


def _validate_confirmation_input(value: object) -> dict[str, Any]:
    document = _require_exact_fields(value, name="confirmationInput", required=set(FRESH_NORMAL_CONFIRMATION_INPUT))
    if document != FRESH_NORMAL_CONFIRMATION_INPUT:
        raise FreshNormalObservationError("confirmationInput must require raw NORMAL_CONFIRMATION nominal-good data")
    return dict(document)


def _validate_contract_and_claim(
    contract_path: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    """Load the pre-existing JSON-only selection boundary without image I/O."""

    try:
        contract, contract_file_sha256 = selection.load_validated_fresh_selection_contract(
            contract_path,
            repository_root=repository_root,
        )
        claim, claim_file_sha256 = selection.load_validated_fresh_selection_claim_for_contract(
            contract_path,
            repository_root=repository_root,
        )
    except selection.FreshNormalSelectionError as error:
        raise FreshNormalObservationError(str(error)) from error
    return contract, contract_file_sha256, claim, claim_file_sha256


def _require_contract_binding(
    contract: dict[str, Any],
    *,
    holdout_document: dict[str, Any],
    holdout_file_sha256: str,
    augmentation_document: dict[str, Any],
    augmentation_file_sha256: str,
) -> None:
    expected_holdout = {
        "manifestFileSha256": holdout_file_sha256,
        "manifestDeclaredSha256": holdout_document.get("normalHoldoutManifestSha256"),
        "developmentIdentitySha256": holdout_document.get("developmentIdentitySha256"),
        "normalSelectionIdentitySha256": holdout_document.get("normalSelectionIdentitySha256"),
        "normalConfirmationIdentitySha256": holdout_document.get("normalConfirmationIdentitySha256"),
    }
    for field, expected in expected_holdout.items():
        if contract["holdout"].get(field) != expected:
            raise FreshNormalObservationError(f"frozen contract {field} does not match the normal holdout")
    expected_augmentation = {
        "manifestFileSha256": augmentation_file_sha256,
        "manifestDeclaredSha256": augmentation_document.get("augmentationManifestSha256"),
        "developmentIdentitySha256": holdout_document.get("developmentIdentitySha256"),
        "fitParentIdentitySha256": augmentation_document.get("fitParentIdentitySha256"),
        "recipeFileSha256": augmentation_document.get("recipeFileSha256"),
        "variantsPerParent": augmentation_document.get("variantsPerParent"),
    }
    for field, expected in expected_augmentation.items():
        if contract["augmentation"].get(field) != expected:
            raise FreshNormalObservationError(f"frozen contract augmentation {field} does not match the validated package")


def _safe_relative_path(value: object, *, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise FreshNormalObservationError(f"{name} must be a non-empty safe relative path")
    if "\\" in value:
        raise FreshNormalObservationError(f"{name} must use a POSIX relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or not parsed.parts or any(part in {"", ".", ".."} for part in parsed.parts):
        raise FreshNormalObservationError(f"{name} must be a safe relative path")
    return parsed


def _safe_file_under(root: Path, relative: PurePosixPath, *, description: str, repository_root: Path) -> Path:
    if not isinstance(root, Path):
        raise FreshNormalObservationError(f"{description} root must be a Path")
    if _is_under(repository_root, root) or _is_under(root, repository_root):
        raise FreshNormalObservationError(f"{description} must stay outside the Git working tree")
    _reject_links_on_existing_path(root, description=description)
    if not root.is_dir() or _is_link_or_reparse_point(root):
        raise FreshNormalObservationError(f"{description} root must be a regular non-link directory")
    candidate = root.joinpath(*relative.parts)
    _reject_links_on_existing_path(candidate, description=description)
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError) as error:
        raise FreshNormalObservationError(f"{description} escapes its external root") from error
    if not candidate.is_file() or _is_link_or_reparse_point(candidate):
        raise FreshNormalObservationError(f"{description} must be a regular non-link file")
    return candidate


def _feature_input_identity_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "partition": record["partition"],
        "kind": record["kind"],
        "defect": record["defect"],
        "sourceSha256": record["sourceSha256"],
        "isAugmentation": bool(record.get("isAugmentation", False)),
        "variantId": record.get("variantId"),
        "parentCaseId": record.get("parentCaseId"),
        "parentSourceSha256": record.get("parentSourceSha256"),
        "augmentationManifestSha256": record.get("augmentationManifestSha256"),
    }


def _feature_input_identity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_feature_input_identity_record(record) for record in sorted(records, key=lambda item: str(item["caseId"]))]


def _normal_partition_input_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "partition": record["partition"],
        "kind": record["kind"],
        "defect": record["defect"],
        "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"],
        "acquisitionStratum": record["acquisitionStratum"],
        "expectedRemoteSha256": record["expectedRemoteSha256"],
        "expectedRemoteBytes": record["expectedRemoteBytes"],
    }


def _adapt_fit_original(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("partition") != "FIT" or record.get("kind") != "NOMINAL" or record.get("defect") != "good":
        raise FreshNormalObservationError("prototype input must be a raw FIT nominal-good image")
    if not isinstance(record.get("imagePath"), Path):
        raise FreshNormalObservationError("validated FIT input has no safe image path")
    return {
        **record,
        "isAugmentation": False,
        "variantId": None,
        "parentCaseId": None,
        "parentSourceSha256": None,
        "augmentationManifestSha256": None,
    }


def _adapt_fit_augmentation(
    record: dict[str, Any],
    *,
    augmentation_root: Path,
    augmentation_manifest_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    if (
        record.get("parentPartition") != "FIT"
        or record.get("kind") != "NOMINAL"
        or record.get("defect") != "good"
    ):
        raise FreshNormalObservationError("fresh FIT derivative has an unsafe input scope")
    relative = _safe_relative_path(record.get("relativePath"), name="fresh FIT derivative relativePath")
    image_path = _safe_file_under(
        augmentation_root,
        relative,
        description="fresh FIT derivative image",
        repository_root=repository_root,
    )
    return {
        "caseId": _require_string(record.get("caseId"), name="fresh FIT derivative caseId"),
        "category": _require_string(record.get("category"), name="fresh FIT derivative category"),
        "partition": "FIT",
        "kind": "NOMINAL",
        "defect": "good",
        "sourceSha256": _require_sha256(record.get("sourceSha256"), name="fresh FIT derivative sourceSha256"),
        "sourceGroupId": _require_string(record.get("sourceGroupId"), name="fresh FIT derivative sourceGroupId"),
        "imagePath": image_path,
        "isAugmentation": True,
        "variantId": _require_positive_int(record.get("variantId"), name="fresh FIT derivative variantId", maximum=8),
        "parentCaseId": _require_string(record.get("parentCaseId"), name="fresh FIT derivative parentCaseId"),
        "parentSourceSha256": _require_sha256(
            record.get("parentSourceSha256"), name="fresh FIT derivative parentSourceSha256"
        ),
        "augmentationManifestSha256": augmentation_manifest_sha256,
    }


def _load_fit_prototype_inputs(
    contract: dict[str, Any],
    holdout_manifest_path: Path,
    augmentation_manifest_path: Path,
    recipe_path: Path,
    *,
    source_root: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Open only FIT raw parents and verified FIT derivatives during preflight."""

    try:
        holdout_document, holdout_file_sha256, raw_fit = holdout.load_evaluation_safe_normal_holdout_inputs(
            holdout_manifest_path,
            source_root=source_root,
            partitions={"FIT"},
        )
        augmentation_document, augmentation_records = augmentation.load_validated_fresh_fit_augmentations(
            augmentation_manifest_path,
            holdout_manifest_path,
            source_root=source_root,
            recipe_path=recipe_path,
            repository_root=repository_root,
        )
    except (holdout.NormalHoldoutError, augmentation.FreshFitAugmentationError) as error:
        raise FreshNormalObservationError(str(error)) from error
    augmentation_file_sha256 = selection.sha256_file(augmentation_manifest_path)
    _require_contract_binding(
        contract,
        holdout_document=holdout_document,
        holdout_file_sha256=holdout_file_sha256,
        augmentation_document=augmentation_document,
        augmentation_file_sha256=augmentation_file_sha256,
    )
    originals = [_adapt_fit_original(record) for record in raw_fit]
    derivatives = [
        _adapt_fit_augmentation(
            record,
            augmentation_root=augmentation_manifest_path.parent,
            augmentation_manifest_sha256=str(augmentation_document["augmentationManifestSha256"]),
            repository_root=repository_root,
        )
        for record in augmentation_records
    ]
    combined = sorted(originals + derivatives, key=lambda record: str(record["caseId"]))
    case_ids = [str(record["caseId"]) for record in combined]
    if len(case_ids) != len(set(case_ids)):
        raise FreshNormalObservationError("FIT prototype inputs have duplicated caseIds")
    if any(record["partition"] != "FIT" for record in combined):
        raise FreshNormalObservationError("prototype preflight attempted to include a non-FIT input")
    feature_identity = _feature_input_identity(combined)
    shared_report_identity = {str(candidate["featureInputIdentitySha256"]) for candidate in contract["candidateReports"]}
    # The development feature identity also includes raw threshold tuning.  The
    # selection observer intentionally does not recompute it from images; it
    # validates FIT package bindings and separately records this FIT-only ID.
    if len(shared_report_identity) != 1:
        raise FreshNormalObservationError("frozen candidates do not share one development feature identity")
    return (
        holdout_document,
        holdout_file_sha256,
        augmentation_document,
        augmentation_file_sha256,
        combined,
        feature_identity,
    )


def _load_rgb_and_verify(record: dict[str, Any]) -> Image.Image:
    """Rehash/decode a phase-approved input while rejecting path substitution."""

    source_path = record.get("imagePath")
    if not isinstance(source_path, Path) or not source_path.is_file() or _is_link_or_reparse_point(source_path):
        raise FreshNormalObservationError("phase-approved image input is not a regular non-link file")
    _reject_links_on_existing_path(source_path, description="phase-approved image input")
    before = _stat_signature(source_path)
    expected_sha256 = _require_sha256(record.get("sourceSha256"), name="phase-approved image sourceSha256")
    if selection.sha256_file(source_path) != expected_sha256:
        raise FreshNormalObservationError("phase-approved image digest does not match its frozen record")
    try:
        with Image.open(source_path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, SyntaxError, ValueError) as error:
        raise FreshNormalObservationError("unable to decode a phase-approved image") from error
    _reject_links_on_existing_path(source_path, description="phase-approved image input")
    if before != _stat_signature(source_path) or selection.sha256_file(source_path) != expected_sha256:
        raise FreshNormalObservationError("phase-approved image changed while it was decoded")
    return image


def _extract_patch_features(
    records: list[dict[str, Any]],
    *,
    embedder: Any,
    batch_size: int,
    timings: dict[str, float],
) -> dict[str, object]:
    """Extract in bounded batches and retain no image or on-disk query cache."""

    if not records:
        raise FreshNormalObservationError("patch extraction requires at least one record")
    if len({str(record.get("caseId")) for record in records}) != len(records):
        raise FreshNormalObservationError("patch extraction input caseId is duplicated")
    result: dict[str, object] = {}
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        verification_started = time.perf_counter()
        images = [_load_rgb_and_verify(record) for record in batch]
        timings["inputVerificationSeconds"] = timings.get("inputVerificationSeconds", 0.0) + (
            time.perf_counter() - verification_started
        )
        inference_started = time.perf_counter()
        extracted = embedder.extract_patches(images)
        timings["featureInferenceSeconds"] = timings.get("featureInferenceSeconds", 0.0) + (
            time.perf_counter() - inference_started
        )
        if not isinstance(extracted, list) or len(extracted) != len(batch):
            raise FreshNormalObservationError("DINO patch extraction returned an unexpected batch size")
        for record, values in zip(batch, extracted, strict=True):
            case_id = str(record["caseId"])
            if case_id in result:
                raise FreshNormalObservationError("DINO patch extraction duplicated a caseId")
            result[case_id] = values
    return result


def _p95(values: list[float]) -> float:
    if not values:
        raise FreshNormalObservationError("P95 requires at least one score")
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)])


def _validate_patch_matrix(values: object, *, name: str) -> tuple[int, int]:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - required runtime dependency
        raise RuntimeError("fresh normal observations require numpy") from error
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0 or not np.all(np.isfinite(array)):
        raise FreshNormalObservationError(f"{name} patch matrix is empty or non-finite")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-12):
        raise FreshNormalObservationError(f"{name} patch matrix contains a zero-norm row")
    return int(array.shape[0]), int(array.shape[1])


def _build_prototypes(
    fit_records: list[dict[str, Any]],
    features: dict[str, object],
    candidate_configuration: dict[str, Any],
) -> dict[str, object]:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - required runtime dependency
        raise RuntimeError("fresh normal observations require numpy") from error
    categories = sorted({str(record["category"]) for record in fit_records})
    if not categories:
        raise FreshNormalObservationError("prototype inputs have no categories")
    prototypes: dict[str, object] = {}
    expected_width: int | None = None
    for category in categories:
        records = [record for record in fit_records if record["category"] == category]
        if not records:
            raise FreshNormalObservationError("prototype category has no FIT inputs")
        matrices: list[object] = []
        for record in records:
            case_id = str(record["caseId"])
            if case_id not in features:
                raise FreshNormalObservationError("FIT prototype feature is missing")
            _, width = _validate_patch_matrix(features[case_id], name=f"FIT {case_id}")
            if expected_width is None:
                expected_width = width
            elif width != expected_width:
                raise FreshNormalObservationError("FIT patch feature dimensions do not match")
            matrices.append(np.asarray(features[case_id], dtype=np.float32))
        all_patches = np.concatenate(matrices, axis=0)
        indices = evaluator.deterministic_prototype_indices(
            int(all_patches.shape[0]),
            int(candidate_configuration["maxPrototypePatches"]),
        )
        prototypes[category] = all_patches[indices]
    return prototypes


def _score_query_records(
    query_records: list[dict[str, Any]],
    query_features: dict[str, object],
    prototypes: dict[str, object],
    candidate_configuration: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    partition: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - required runtime dependency
        raise RuntimeError("fresh normal observations require numpy") from error
    if not isinstance(thresholds, dict):
        raise FreshNormalObservationError("candidate thresholds are missing from the frozen observation request")
    scores: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    categories = sorted({str(record["category"]) for record in query_records})
    if set(categories) != set(prototypes) or set(categories) != set(thresholds):
        raise FreshNormalObservationError("query categories, prototypes, and frozen thresholds do not match")
    for category in categories:
        category_records = sorted(
            (record for record in query_records if record["category"] == category),
            key=lambda item: str(item["caseId"]),
        )
        matrices: list[object] = []
        for record in category_records:
            case_id = str(record["caseId"])
            if case_id not in query_features:
                raise FreshNormalObservationError("query patch feature is missing")
            _validate_patch_matrix(query_features[case_id], name=f"query {case_id}")
            matrices.append(np.asarray(query_features[case_id], dtype=np.float32))
        query_patches = np.asarray(matrices, dtype=np.float32)
        components = evaluator.patch_knn_scores_blocked(
            query_patches,
            prototypes[category],
            top_k=int(candidate_configuration["topKMostAnomalousPatches"]),
            prototype_block_size=int(candidate_configuration["prototypeBlockSize"]),
        )
        if len(components) != len(category_records):
            raise FreshNormalObservationError("patch scoring returned an unexpected query score count")
        category_scores: list[float] = []
        for record, component in zip(category_records, components, strict=True):
            values = {
                field: _require_finite_number(component.get(field), name=f"query {field}")
                for field in ("score", "maxPatchDistance", "meanNearestPatchDistance")
            }
            if not values["meanNearestPatchDistance"] <= values["score"] <= values["maxPatchDistance"]:
                raise FreshNormalObservationError("patch score components are inconsistent")
            category_scores.append(values["score"])
            scores.append({
                "caseId": record["caseId"],
                "category": category,
                "partition": partition,
                "kind": "NOMINAL",
                "defect": "good",
                "sourceSha256": record["sourceSha256"],
                **values,
            })
        metrics[category] = _derive_category_metrics(category_scores, float(thresholds[category]))
    scores.sort(key=lambda record: str(record["caseId"]))
    return scores, metrics


def _derive_category_metrics(scores: list[float], threshold: float) -> dict[str, Any]:
    if not scores:
        raise FreshNormalObservationError("normal observation category has no query scores")
    threshold = _require_finite_number(threshold, name="frozen raw-tuning threshold")
    checked = [_require_finite_number(score, name="normal query score") for score in scores]
    maximum = max(checked)
    p95 = _p95(checked)
    above = sum(score > threshold for score in checked)
    return {
        "caseCount": len(checked),
        "aboveThresholdCount": above,
        "aboveThresholdRate": above / len(checked),
        "p95Score": p95,
        "p95ScoreMinusThreshold": p95 - threshold,
        "maximumScore": maximum,
        "maximumScoreMinusThreshold": maximum - threshold,
    }


def _build_normal_only_evidence(
    fit_records: list[dict[str, Any]],
    query_records: list[dict[str, Any]],
    *,
    partition: str,
) -> dict[str, Any]:
    evidence = {
        "prototypeInputCount": len(fit_records),
        "prototypeInputPartitions": ["FIT"],
        "prototypeInputKinds": ["NOMINAL"],
        "fitOriginalInputCount": sum(not record["isAugmentation"] for record in fit_records),
        "fitAugmentedInputCount": sum(bool(record["isAugmentation"]) for record in fit_records),
        "queryInputCount": len(query_records),
        "queryInputPartitions": [partition],
        "queryInputKinds": ["NOMINAL"],
        "blindInputCount": 0,
        "anomalyInputCount": 0,
        "maskInputCount": 0,
        "persistentQueryCache": False,
    }
    _require_exact_fields(evidence, name="normalOnlyEvidence", required=NORMAL_ONLY_EVIDENCE_FIELDS)
    if not evidence["fitOriginalInputCount"] or not evidence["fitAugmentedInputCount"] or not evidence["queryInputCount"]:
        raise FreshNormalObservationError("normal observation evidence has empty required normal-only inputs")
    return evidence


def _execution_metadata(timings: dict[str, float]) -> dict[str, Any]:
    for field in TIMING_FIELDS:
        _require_finite_number(timings.get(field), name=f"observation timing {field}", minimum=0.0, maximum=31_536_000.0)
    return {
        "observerModuleSha256": selection.sha256_file(Path(__file__)),
        "queryFeatureCachePolicy": NO_PERSISTENT_QUERY_CACHE,
        "phaseTimingsSeconds": {field: round(float(timings[field]), 6) for field in sorted(TIMING_FIELDS)},
        "python": sys.version,
        "platform": platform.platform(),
    }


def _validate_fit_evidence(
    *,
    document: dict[str, Any],
    contract: dict[str, Any],
    prefix: str,
) -> None:
    feature_identity = _require_sha256(
        document.get("fitFeatureInputIdentitySha256"), name=f"{prefix} fitFeatureInputIdentitySha256"
    )
    if not feature_identity:
        raise FreshNormalObservationError(f"{prefix} FIT feature identity is missing")
    total = _require_positive_int(document.get("fitFeatureInputCount"), name=f"{prefix} fitFeatureInputCount")
    originals = _require_positive_int(document.get("fitOriginalInputCount"), name=f"{prefix} fitOriginalInputCount")
    derivatives = _require_positive_int(document.get("fitAugmentedInputCount"), name=f"{prefix} fitAugmentedInputCount")
    if total != originals + derivatives:
        raise FreshNormalObservationError(f"{prefix} FIT feature counts do not add up")
    variants = _require_positive_int(contract["augmentation"].get("variantsPerParent"), name="contract variantsPerParent", maximum=8)
    if derivatives != originals * variants:
        raise FreshNormalObservationError(f"{prefix} FIT derivative count does not match the frozen package coverage")


def _validate_selection_receipt_document(
    document: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_file_sha256: str,
    claim: dict[str, Any],
    claim_file_sha256: str,
) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal selection receipt", required=SELECTION_RECEIPT_FIELDS)
    _validate_static_scope(
        document,
        schema=FRESH_NORMAL_SELECTION_RECEIPT_SCHEMA,
        purpose=FRESH_NORMAL_SELECTION_RECEIPT_PURPOSE,
        phase=FRESH_NORMAL_SELECTION_RECEIPT_PHASE,
        description="fresh normal selection receipt",
    )
    _validate_selection_input(document.get("selectionInput"))
    if document.get("receiptSlot") != FRESH_NORMAL_SELECTION_RECEIPT_SLOT:
        raise FreshNormalObservationError("fresh normal selection receipt uses an unsupported slot")
    expected = {
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionClaimFileSha256": claim_file_sha256,
        "selectionClaimDeclaredSha256": claim["claimSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "candidateUniverseIdentitySha256": contract["candidateUniverseIdentitySha256"],
    }
    for field, expected_value in expected.items():
        if document.get(field) != expected_value:
            raise FreshNormalObservationError(f"fresh normal selection receipt {field} does not bind the frozen contract")
    _validate_fit_evidence(document=document, contract=contract, prefix="fresh normal selection receipt")
    if document.get("selectionReceiptSha256") != _document_digest(document, "selectionReceiptSha256"):
        raise FreshNormalObservationError("fresh normal selection receipt digest does not match")
    return document


def _validate_confirmation_receipt_document(
    document: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_file_sha256: str,
    lock: dict[str, Any],
    lock_file_sha256: str,
    claim: dict[str, Any],
    claim_file_sha256: str,
) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal confirmation receipt", required=CONFIRMATION_RECEIPT_FIELDS)
    _validate_static_scope(
        document,
        schema=FRESH_NORMAL_CONFIRMATION_RECEIPT_SCHEMA,
        purpose=FRESH_NORMAL_CONFIRMATION_RECEIPT_PURPOSE,
        phase=FRESH_NORMAL_CONFIRMATION_RECEIPT_PHASE,
        description="fresh normal confirmation receipt",
    )
    _validate_confirmation_input(document.get("confirmationInput"))
    if document.get("receiptSlot") != FRESH_NORMAL_CONFIRMATION_RECEIPT_SLOT:
        raise FreshNormalObservationError("fresh normal confirmation receipt uses an unsupported slot")
    expected = {
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionLockFileSha256": lock_file_sha256,
        "selectionLockDeclaredSha256": lock["selectionLockSha256"],
        "confirmationClaimFileSha256": claim_file_sha256,
        "confirmationClaimDeclaredSha256": claim["confirmationClaimSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
    }
    for field, expected_value in expected.items():
        if document.get(field) != expected_value:
            raise FreshNormalObservationError(f"fresh normal confirmation receipt {field} does not bind the frozen claim")
    _validate_fit_evidence(document=document, contract=contract, prefix="fresh normal confirmation receipt")
    if document.get("confirmationReceiptSha256") != _document_digest(document, "confirmationReceiptSha256"):
        raise FreshNormalObservationError("fresh normal confirmation receipt digest does not match")
    return document


def _validate_score_records(
    value: object,
    *,
    expected_inputs: list[dict[str, Any]],
    partition: str,
    name: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FreshNormalObservationError(f"{name} must be a non-empty list")
    expected_by_case = {str(record["caseId"]): record for record in expected_inputs}
    if len(expected_by_case) != len(expected_inputs):
        raise FreshNormalObservationError(f"{name} expected input membership is duplicated")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        record = _require_exact_fields(item, name=f"{name} record", required=evaluator.CALIBRATION_SCORE_FIELDS)
        case_id = _require_string(record.get("caseId"), name=f"{name}.caseId")
        expected = expected_by_case.get(case_id)
        if expected is None or case_id in seen:
            raise FreshNormalObservationError(f"{name} has an unexpected or duplicated caseId")
        seen.add(case_id)
        for field in ("category", "partition", "kind", "defect", "sourceSha256"):
            if record.get(field) != expected[field]:
                raise FreshNormalObservationError(f"{name} record does not match its frozen input membership")
        if record.get("partition") != partition or record.get("kind") != "NOMINAL" or record.get("defect") != "good":
            raise FreshNormalObservationError(f"{name} record uses an unsafe query scope")
        values = {
            field: _require_finite_number(record.get(field), name=f"{name}.{field}")
            for field in ("score", "maxPatchDistance", "meanNearestPatchDistance")
        }
        if not values["meanNearestPatchDistance"] <= values["score"] <= values["maxPatchDistance"]:
            raise FreshNormalObservationError(f"{name} score components are inconsistent")
        result.append({
            "caseId": case_id,
            "category": str(record["category"]),
            "partition": partition,
            "kind": "NOMINAL",
            "defect": "good",
            "sourceSha256": str(record["sourceSha256"]),
            **values,
        })
    if set(expected_by_case) != seen or [record["caseId"] for record in result] != sorted(seen):
        raise FreshNormalObservationError(f"{name} must cover every frozen query input once in caseId order")
    return result


def _validate_thresholds(value: object, *, expected: dict[str, Any], name: str) -> dict[str, float]:
    document = _require_mapping(value, name=name)
    if list(document) != sorted(document) or set(document) != set(expected):
        raise FreshNormalObservationError(f"{name} categories do not match the frozen candidate")
    result = {
        category: _require_finite_number(document[category], name=f"{name}.{category}")
        for category in document
    }
    if any(not _same_number(float(expected[category]), result[category]) for category in result):
        raise FreshNormalObservationError(f"{name} values do not match the frozen raw-tuning thresholds")
    return result


def _validate_category_metrics(
    value: object,
    *,
    scores: list[dict[str, Any]],
    thresholds: dict[str, float],
    name: str,
) -> dict[str, dict[str, Any]]:
    document = _require_mapping(value, name=name)
    if list(document) != sorted(document) or set(document) != set(thresholds):
        raise FreshNormalObservationError(f"{name} categories do not match frozen thresholds")
    by_category: dict[str, list[float]] = {category: [] for category in thresholds}
    for record in scores:
        category = str(record["category"])
        if category not in by_category:
            raise FreshNormalObservationError(f"{name} score has an unknown category")
        by_category[category].append(float(record["score"]))
    result: dict[str, dict[str, Any]] = {}
    for category in sorted(thresholds):
        supplied = _require_exact_fields(document.get(category), name=f"{name}.{category}", required=CATEGORY_METRIC_FIELDS)
        expected = _derive_category_metrics(by_category[category], thresholds[category])
        parsed = {
            "caseCount": _require_positive_int(supplied.get("caseCount"), name=f"{name}.{category}.caseCount"),
            "aboveThresholdCount": _require_nonnegative_int(
                supplied.get("aboveThresholdCount"), name=f"{name}.{category}.aboveThresholdCount"
            ),
            "aboveThresholdRate": _require_finite_number(
                supplied.get("aboveThresholdRate"), name=f"{name}.{category}.aboveThresholdRate", minimum=0.0, maximum=1.0
            ),
            "p95Score": _require_finite_number(supplied.get("p95Score"), name=f"{name}.{category}.p95Score"),
            "p95ScoreMinusThreshold": _require_finite_number(
                supplied.get("p95ScoreMinusThreshold"), name=f"{name}.{category}.p95ScoreMinusThreshold", minimum=-2.0
            ),
            "maximumScore": _require_finite_number(supplied.get("maximumScore"), name=f"{name}.{category}.maximumScore"),
            "maximumScoreMinusThreshold": _require_finite_number(
                supplied.get("maximumScoreMinusThreshold"), name=f"{name}.{category}.maximumScoreMinusThreshold", minimum=-2.0
            ),
        }
        if parsed["aboveThresholdCount"] > parsed["caseCount"]:
            raise FreshNormalObservationError(f"{name}.{category} above-threshold count exceeds case count")
        for field, expected_value in expected.items():
            if isinstance(expected_value, int):
                matches = parsed[field] == expected_value
            else:
                matches = _same_number(float(expected_value), parsed[field])
            if not matches:
                raise FreshNormalObservationError(f"{name}.{category}.{field} does not match its raw query scores")
        result[category] = parsed
    return result


def _validate_normal_only_evidence(
    value: object,
    *,
    receipt: dict[str, Any],
    query_inputs: list[dict[str, Any]],
    partition: str,
) -> dict[str, Any]:
    document = _require_exact_fields(value, name="normalOnlyEvidence", required=NORMAL_ONLY_EVIDENCE_FIELDS)
    expected = {
        "prototypeInputCount": receipt["fitFeatureInputCount"],
        "prototypeInputPartitions": ["FIT"],
        "prototypeInputKinds": ["NOMINAL"],
        "fitOriginalInputCount": receipt["fitOriginalInputCount"],
        "fitAugmentedInputCount": receipt["fitAugmentedInputCount"],
        "queryInputCount": len(query_inputs),
        "queryInputPartitions": [partition],
        "queryInputKinds": ["NOMINAL"],
        "blindInputCount": 0,
        "anomalyInputCount": 0,
        "maskInputCount": 0,
        "persistentQueryCache": False,
    }
    if document != expected:
        raise FreshNormalObservationError("normalOnlyEvidence is inconsistent with the frozen normal-only inputs")
    return dict(document)


def _validate_execution(value: object, *, name: str) -> dict[str, Any]:
    document = _require_exact_fields(value, name=name, required=EXECUTION_FIELDS)
    _require_sha256(document.get("observerModuleSha256"), name=f"{name}.observerModuleSha256")
    if document.get("queryFeatureCachePolicy") != NO_PERSISTENT_QUERY_CACHE:
        raise FreshNormalObservationError(f"{name} must prohibit a persistent query cache")
    timings = _require_exact_fields(document.get("phaseTimingsSeconds"), name=f"{name}.phaseTimingsSeconds", required=TIMING_FIELDS)
    for field, value_ in timings.items():
        _require_finite_number(value_, name=f"{name}.phaseTimingsSeconds.{field}", minimum=0.0, maximum=31_536_000.0)
    _require_string(document.get("python"), name=f"{name}.python")
    _require_string(document.get("platform"), name=f"{name}.platform")
    return document


def _candidate_by_id(contract: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in contract["candidateReports"]:
        if candidate.get("candidateId") == candidate_id:
            return candidate
    raise FreshNormalObservationError("candidate is not part of the frozen selection universe")


def _validate_candidate_observation(
    value: object,
    *,
    candidate: dict[str, Any],
    query_inputs: list[dict[str, Any]],
    partition: str,
) -> dict[str, Any]:
    document = _require_exact_fields(
        value,
        name="fresh normal candidate observation",
        required=SELECTION_CANDIDATE_OBSERVATION_FIELDS,
    )
    candidate_id = _require_string(document.get("candidateId"), name="candidate observation candidateId")
    if candidate_id != candidate["candidateId"]:
        raise FreshNormalObservationError("candidate observation candidateId does not match the frozen contract")
    try:
        configuration = evaluator.validate_candidate_configuration(document.get("candidateConfiguration"))
    except evaluator.NormalHoldoutEvaluatorError as error:
        raise FreshNormalObservationError(str(error)) from error
    if configuration != candidate["candidateConfiguration"]:
        raise FreshNormalObservationError("candidate observation configuration does not match the frozen contract")
    if document.get("candidateConfigurationSha256") != candidate["candidateConfigurationSha256"]:
        raise FreshNormalObservationError("candidate observation configuration digest does not match the frozen contract")
    thresholds = _validate_thresholds(
        document.get("thresholds"),
        expected=candidate["thresholds"],
        name="candidate observation thresholds",
    )
    if document.get("thresholdsIdentitySha256") != candidate["thresholdsIdentitySha256"]:
        raise FreshNormalObservationError("candidate observation threshold digest does not match the frozen contract")
    scores = _validate_score_records(
        document.get("selectionScores"),
        expected_inputs=query_inputs,
        partition=partition,
        name="candidate observation selectionScores",
    )
    metrics = _validate_category_metrics(
        document.get("categoryMetrics"),
        scores=scores,
        thresholds=thresholds,
        name="candidate observation categoryMetrics",
    )
    return {
        "candidateId": candidate_id,
        "candidateConfiguration": configuration,
        "candidateConfigurationSha256": candidate["candidateConfigurationSha256"],
        "thresholds": thresholds,
        "thresholdsIdentitySha256": candidate["thresholdsIdentitySha256"],
        "selectionScores": scores,
        "categoryMetrics": metrics,
    }


def _validate_selection_observation_document(
    document: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_file_sha256: str,
    claim: dict[str, Any],
    claim_file_sha256: str,
    receipt: dict[str, Any],
    receipt_file_sha256: str,
) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal selection observation", required=SELECTION_OBSERVATION_FIELDS)
    _validate_static_scope(
        document,
        schema=FRESH_NORMAL_SELECTION_OBSERVATION_SCHEMA,
        purpose=FRESH_NORMAL_SELECTION_OBSERVATION_PURPOSE,
        phase=FRESH_NORMAL_SELECTION_OBSERVATION_PHASE,
        description="fresh normal selection observation",
    )
    _validate_selection_input(document.get("selectionInput"))
    expected = {
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionClaimFileSha256": claim_file_sha256,
        "selectionClaimDeclaredSha256": claim["claimSha256"],
        "selectionReceiptFileSha256": receipt_file_sha256,
        "selectionReceiptDeclaredSha256": receipt["selectionReceiptSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractor": contract["featureExtractor"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "normalSelectionInputs": contract["normalSelectionInputs"],
        "normalSelectionInputIdentitySha256": contract["normalSelectionInputIdentitySha256"],
    }
    for field, expected_value in expected.items():
        if document.get(field) != expected_value:
            raise FreshNormalObservationError(f"fresh normal selection observation {field} does not bind the frozen input")
    raw_candidates = document.get("candidateObservations")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise FreshNormalObservationError("fresh normal selection observation candidateObservations must be non-empty")
    expected_candidate_ids = [str(candidate["candidateId"]) for candidate in contract["candidateReports"]]
    supplied_candidate_ids = [
        _require_string(_require_mapping(item, name="candidate observation").get("candidateId"), name="candidate observation candidateId")
        for item in raw_candidates
    ]
    if supplied_candidate_ids != expected_candidate_ids or len(set(supplied_candidate_ids)) != len(supplied_candidate_ids):
        raise FreshNormalObservationError("selection observation must score every frozen candidate once in candidateId order")
    parsed_candidates = [
        _validate_candidate_observation(
            item,
            candidate=_candidate_by_id(contract, candidate_id),
            query_inputs=contract["normalSelectionInputs"],
            partition="NORMAL_SELECTION",
        )
        for candidate_id, item in zip(expected_candidate_ids, raw_candidates, strict=True)
    ]
    _validate_normal_only_evidence(
        document.get("normalOnlyEvidence"),
        receipt=receipt,
        query_inputs=contract["normalSelectionInputs"],
        partition="NORMAL_SELECTION",
    )
    _validate_execution(document.get("execution"), name="fresh normal selection observation execution")
    if document.get("selectionObservationSha256") != _document_digest(document, "selectionObservationSha256"):
        raise FreshNormalObservationError("fresh normal selection observation digest does not match")
    return {**document, "candidateObservations": parsed_candidates}


def _load_selection_receipt_for_contract(
    contract: dict[str, Any],
    *,
    contract_file_sha256: str,
    claim: dict[str, Any],
    claim_file_sha256: str,
    repository_root: Path,
) -> tuple[dict[str, Any], str]:
    path = fresh_selection_receipt_path(contract, repository_root=repository_root)
    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal selection receipt",
        repository_root=repository_root,
    )
    return (
        _validate_selection_receipt_document(
            document,
            contract=contract,
            contract_file_sha256=contract_file_sha256,
            claim=claim,
            claim_file_sha256=claim_file_sha256,
        ),
        file_sha256,
    )


def load_validated_fresh_selection_observation_for_contract(
    contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Load the one aggregate selection observation without opening an image."""

    contract, contract_file_sha256, claim, claim_file_sha256 = _validate_contract_and_claim(
        contract_path,
        repository_root=repository_root,
    )
    receipt, receipt_file_sha256 = _load_selection_receipt_for_contract(
        contract,
        contract_file_sha256=contract_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
        repository_root=repository_root,
    )
    path = fresh_selection_observation_path(contract, repository_root=repository_root)
    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal selection observation",
        repository_root=repository_root,
    )
    return (
        _validate_selection_observation_document(
            document,
            contract=contract,
            contract_file_sha256=contract_file_sha256,
            claim=claim,
            claim_file_sha256=claim_file_sha256,
            receipt=receipt,
            receipt_file_sha256=receipt_file_sha256,
        ),
        file_sha256,
    )


def _require_empty_slot(path: Path, *, description: str, repository_root: Path) -> None:
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise FreshNormalObservationError(f"{description} must stay outside the Git working tree")
    _reject_links_on_existing_path(path.parent, description=description)
    if path.exists() or path.is_symlink():
        raise FreshNormalObservationError(f"{description} already exists; this one-time phase is already consumed")


def _assert_feature_extractor_identity(
    contract: dict[str, Any],
    *,
    model_repo: Path,
    model_weights: Path,
    device: str,
    identity_factory: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    if not isinstance(identity, dict):
        raise FreshNormalObservationError("feature extractor identity factory must return an object")
    if identity != contract["featureExtractor"]:
        raise FreshNormalObservationError("current feature extractor does not match the frozen development reports")
    if _canonical_json_sha256(identity) != contract["featureExtractorIdentitySha256"]:
        raise FreshNormalObservationError("current feature extractor identity digest does not match the frozen contract")
    return identity


def _build_candidate_prototypes(
    contract: dict[str, Any],
    fit_records: list[dict[str, Any]],
    *,
    model_repo: Path,
    model_weights: Path,
    device: str,
    embedder_factory: Callable[..., Any],
    identity_factory: Callable[..., dict[str, Any]],
    timings: dict[str, float],
    candidate_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, object]], Any]:
    if device != "cpu":
        raise FreshNormalObservationError("fresh normal observations support CPU only")
    identity = _assert_feature_extractor_identity(
        contract,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        identity_factory=identity_factory,
    )
    try:
        embedder = embedder_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    except Exception as error:
        raise FreshNormalObservationError("unable to load the frozen DINO feature extractor") from error
    loaded_identity = _assert_feature_extractor_identity(
        contract,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        identity_factory=identity_factory,
    )
    if loaded_identity != identity:
        raise FreshNormalObservationError("feature extractor changed while DINO was loading")
    configurations: dict[str, dict[str, Any]] = {}
    for candidate in contract["candidateReports"]:
        candidate_id = str(candidate["candidateId"])
        if candidate_ids is not None and candidate_id not in candidate_ids:
            continue
        try:
            configuration = evaluator.validate_candidate_configuration(candidate["candidateConfiguration"])
        except evaluator.NormalHoldoutEvaluatorError as error:
            raise FreshNormalObservationError(str(error)) from error
        configurations[candidate_id] = configuration
    if not configurations:
        raise FreshNormalObservationError("prototype preflight has no frozen candidate configurations")
    batch_sizes = {int(configuration["batchSize"]) for configuration in configurations.values()}
    if len(batch_sizes) != 1:
        raise FreshNormalObservationError(
            "all frozen candidates must share one inference batchSize so NORMAL_SELECTION is decoded/inferred once"
        )
    batch_size = next(iter(batch_sizes))
    prototypes: dict[str, dict[str, object]] = {}
    feature_timing: dict[str, float] = {}
    features = _extract_patch_features(
        fit_records,
        embedder=embedder,
        batch_size=batch_size,
        timings=feature_timing,
    )
    timings["prototypeFeatureSeconds"] += sum(feature_timing.values())
    for candidate_id, configuration in configurations.items():
        prototypes[candidate_id] = _build_prototypes(fit_records, features, configuration)
    del features
    completed_identity = _assert_feature_extractor_identity(
        contract,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        identity_factory=identity_factory,
    )
    if completed_identity != identity:
        raise FreshNormalObservationError("feature extractor changed while FIT prototypes were extracted")
    return configurations, prototypes, embedder


def _load_raw_query_inputs(
    contract: dict[str, Any],
    holdout_manifest_path: Path,
    *,
    source_root: Path,
    partition: str,
) -> list[dict[str, Any]]:
    """Open exactly the named raw held-out partition after its receipt exists."""

    expected_inputs_name = {
        "NORMAL_SELECTION": "normalSelectionInputs",
        "NORMAL_CONFIRMATION": "normalConfirmationInputs",
    }.get(partition)
    if expected_inputs_name is None:
        raise FreshNormalObservationError("raw held-out query partition is unsupported")
    try:
        document, file_sha256, records = holdout.load_evaluation_safe_normal_holdout_inputs(
            holdout_manifest_path,
            source_root=source_root,
            partitions={partition},
        )
    except holdout.NormalHoldoutError as error:
        raise FreshNormalObservationError(str(error)) from error
    expected_holdout = contract["holdout"]
    if (
        file_sha256 != expected_holdout["manifestFileSha256"]
        or document.get("normalHoldoutManifestSha256") != expected_holdout["manifestDeclaredSha256"]
        or document.get("developmentIdentitySha256") != expected_holdout["developmentIdentitySha256"]
    ):
        raise FreshNormalObservationError("raw query holdout does not match the frozen contract")
    partition_identity_name = {
        "NORMAL_SELECTION": "normalSelectionIdentitySha256",
        "NORMAL_CONFIRMATION": "normalConfirmationIdentitySha256",
    }[partition]
    if document.get(partition_identity_name) != expected_holdout[partition_identity_name]:
        raise FreshNormalObservationError("raw query partition identity does not match the frozen contract")
    adapted: list[dict[str, Any]] = []
    for record in records:
        if record.get("partition") != partition or record.get("kind") != "NOMINAL" or record.get("defect") != "good":
            raise FreshNormalObservationError("raw held-out query loader returned an unsafe record")
        if not isinstance(record.get("imagePath"), Path):
            raise FreshNormalObservationError("raw held-out query loader returned no image path")
        adapted.append({
            **record,
            "isAugmentation": False,
            "variantId": None,
            "parentCaseId": None,
            "parentSourceSha256": None,
            "augmentationManifestSha256": None,
        })
    adapted.sort(key=lambda record: str(record["caseId"]))
    membership = [_normal_partition_input_record(record) for record in adapted]
    if membership != contract[expected_inputs_name]:
        raise FreshNormalObservationError("raw held-out query membership does not match the frozen contract")
    return adapted


def _make_selection_receipt(
    *,
    contract: dict[str, Any],
    contract_file_sha256: str,
    claim: dict[str, Any],
    claim_file_sha256: str,
    fit_feature_identity: list[dict[str, Any]],
    fit_records: list[dict[str, Any]],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_SELECTION_RECEIPT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_SELECTION_RECEIPT_PURPOSE,
        "phase": FRESH_NORMAL_SELECTION_RECEIPT_PHASE,
        "blindPolicy": FRESH_NORMAL_BLIND_POLICY,
        "selectionInput": _selection_input(),
        "receiptSlot": FRESH_NORMAL_SELECTION_RECEIPT_SLOT,
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionClaimFileSha256": claim_file_sha256,
        "selectionClaimDeclaredSha256": claim["claimSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "candidateUniverseIdentitySha256": contract["candidateUniverseIdentitySha256"],
        "fitFeatureInputIdentitySha256": _canonical_json_sha256(fit_feature_identity),
        "fitFeatureInputCount": len(fit_records),
        "fitOriginalInputCount": sum(not record["isAugmentation"] for record in fit_records),
        "fitAugmentedInputCount": sum(bool(record["isAugmentation"]) for record in fit_records),
    }
    document["selectionReceiptSha256"] = _document_digest(document, "selectionReceiptSha256")
    _validate_selection_receipt_document(
        document,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
    )
    return document


def run_fresh_normal_selection_observation(
    contract_path: Path,
    holdout_manifest_path: Path,
    augmentation_manifest_path: Path,
    recipe_path: Path,
    *,
    source_root: Path,
    model_repo: Path,
    model_weights: Path,
    device: str = "cpu",
    repository_root: Path = REPOSITORY_ROOT,
    embedder_factory: Callable[..., Any] = evaluator.FreshHoldoutBatchEmbedder,
    identity_factory: Callable[..., dict[str, Any]] = evaluator._feature_extractor_identity,
) -> dict[str, Any]:
    """Consume raw ``NORMAL_SELECTION`` once and score every frozen candidate.

    The receipt's durable ``O_EXCL``/``fsync`` commit happens after only FIT
    inputs have been validated and embedded, and immediately before this
    function calls the phase-safe loader for ``NORMAL_SELECTION``.
    """

    started = time.perf_counter()
    timings = {field: 0.0 for field in TIMING_FIELDS}
    contract, contract_file_sha256, claim, claim_file_sha256 = _validate_contract_and_claim(
        contract_path,
        repository_root=repository_root,
    )
    receipt_path = fresh_selection_receipt_path(contract, repository_root=repository_root)
    observation_path = fresh_selection_observation_path(contract, repository_root=repository_root)
    _require_empty_slot(receipt_path, description="fresh normal selection receipt", repository_root=repository_root)
    _require_empty_slot(observation_path, description="fresh normal selection observation", repository_root=repository_root)
    preflight_started = time.perf_counter()
    (
        _holdout_document,
        _holdout_file_sha256,
        _augmentation_document,
        _augmentation_file_sha256,
        fit_records,
        fit_feature_identity,
    ) = _load_fit_prototype_inputs(
        contract,
        holdout_manifest_path,
        augmentation_manifest_path,
        recipe_path,
        source_root=source_root,
        repository_root=repository_root,
    )
    timings["inputPreflightSeconds"] += time.perf_counter() - preflight_started
    configurations, prototypes, query_embedder = _build_candidate_prototypes(
        contract,
        fit_records,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        embedder_factory=embedder_factory,
        identity_factory=identity_factory,
        timings=timings,
    )
    # Re-read the claim/contract immediately before burning the held-out slot.
    current_contract, current_contract_sha256, current_claim, current_claim_sha256 = _validate_contract_and_claim(
        contract_path,
        repository_root=repository_root,
    )
    if (
        current_contract != contract
        or current_contract_sha256 != contract_file_sha256
        or current_claim != claim
        or current_claim_sha256 != claim_file_sha256
    ):
        raise FreshNormalObservationError("selection contract or claim changed during FIT preflight")
    _assert_feature_extractor_identity(
        contract,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        identity_factory=identity_factory,
    )
    receipt = _make_selection_receipt(
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
        fit_feature_identity=fit_feature_identity,
        fit_records=fit_records,
    )
    receipt_started = time.perf_counter()
    _write_fixed_external_json_fsync(
        receipt_path,
        receipt,
        description="fresh normal selection receipt",
        repository_root=repository_root,
    )
    timings["receiptCommitSeconds"] += time.perf_counter() - receipt_started

    candidate_observations: list[dict[str, Any]] = []
    batch_sizes = {int(configuration["batchSize"]) for configuration in configurations.values()}
    if len(batch_sizes) != 1:
        raise FreshNormalObservationError(
            "all frozen candidates must share one inference batchSize so NORMAL_SELECTION is decoded/inferred once"
        )
    query_batch_size = next(iter(batch_sizes))
    # Reuse the identity-checked extractor that built FIT prototypes.  Query
    # tensors are retained in memory only.
    if _assert_feature_extractor_identity(
        contract,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        identity_factory=identity_factory,
    ) != contract["featureExtractor"]:
        raise FreshNormalObservationError("feature extractor changed before held-out selection scoring")
    # No call above this point receives NORMAL_SELECTION source records.
    query_records = _load_raw_query_inputs(
        contract,
        holdout_manifest_path,
        source_root=source_root,
        partition="NORMAL_SELECTION",
    )
    observations_by_id: dict[str, dict[str, Any]] = {}
    feature_timing: dict[str, float] = {}
    query_features = _extract_patch_features(
        query_records,
        embedder=query_embedder,
        batch_size=query_batch_size,
        timings=feature_timing,
    )
    timings["queryFeatureSeconds"] += sum(feature_timing.values())
    for candidate_id, configuration in configurations.items():
        candidate = _candidate_by_id(contract, candidate_id)
        scoring_started = time.perf_counter()
        scores, metrics = _score_query_records(
            query_records,
            query_features,
            prototypes[candidate_id],
            configuration,
            candidate["thresholds"],
            partition="NORMAL_SELECTION",
        )
        timings["scoringSeconds"] += time.perf_counter() - scoring_started
        observations_by_id[candidate_id] = {
            "candidateId": candidate_id,
            "candidateConfiguration": configuration,
            "candidateConfigurationSha256": candidate["candidateConfigurationSha256"],
            "thresholds": candidate["thresholds"],
            "thresholdsIdentitySha256": candidate["thresholdsIdentitySha256"],
            "selectionScores": scores,
            "categoryMetrics": metrics,
        }
    del query_features
    for candidate in contract["candidateReports"]:
        candidate_id = str(candidate["candidateId"])
        if candidate_id not in observations_by_id:
            raise FreshNormalObservationError("not every frozen candidate was scored in the aggregate observation")
        candidate_observations.append(observations_by_id[candidate_id])
    completed_identity = _assert_feature_extractor_identity(
        contract,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        identity_factory=identity_factory,
    )
    if completed_identity != contract["featureExtractor"]:
        raise FreshNormalObservationError("feature extractor changed while held-out selection features were extracted")
    receipt_file_sha256 = selection.sha256_file(receipt_path)
    timings["totalElapsedSeconds"] = time.perf_counter() - started
    observation: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_SELECTION_OBSERVATION_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_SELECTION_OBSERVATION_PURPOSE,
        "phase": FRESH_NORMAL_SELECTION_OBSERVATION_PHASE,
        "blindPolicy": FRESH_NORMAL_BLIND_POLICY,
        "selectionInput": _selection_input(),
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionClaimFileSha256": claim_file_sha256,
        "selectionClaimDeclaredSha256": claim["claimSha256"],
        "selectionReceiptFileSha256": receipt_file_sha256,
        "selectionReceiptDeclaredSha256": receipt["selectionReceiptSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractor": contract["featureExtractor"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "normalSelectionInputs": contract["normalSelectionInputs"],
        "normalSelectionInputIdentitySha256": contract["normalSelectionInputIdentitySha256"],
        "candidateObservations": candidate_observations,
        "normalOnlyEvidence": _build_normal_only_evidence(
            fit_records,
            query_records,
            partition="NORMAL_SELECTION",
        ),
        "execution": _execution_metadata(timings),
    }
    observation["selectionObservationSha256"] = _document_digest(observation, "selectionObservationSha256")
    _validate_selection_observation_document(
        observation,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
        receipt=receipt,
        receipt_file_sha256=receipt_file_sha256,
    )
    _write_fixed_external_json_fsync(
        observation_path,
        observation,
        description="fresh normal selection observation",
        repository_root=repository_root,
    )
    return observation


def _gate_rejection_reasons(metrics: dict[str, dict[str, Any]], gates: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    maximums = {
        "aboveThresholdRate": _require_finite_number(
            gates.get("maxAboveThresholdRate"), name="selection gate maxAboveThresholdRate", minimum=0.0, maximum=1.0
        ),
        "p95ScoreMinusThreshold": _require_finite_number(
            gates.get("maxP95ScoreMinusThreshold"), name="selection gate maxP95ScoreMinusThreshold", minimum=0.0
        ),
        "maximumScoreMinusThreshold": _require_finite_number(
            gates.get("maxMaximumScoreMinusThreshold"), name="selection gate maxMaximumScoreMinusThreshold", minimum=0.0
        ),
    }
    gate_names = {
        "aboveThresholdRate": "maxAboveThresholdRate",
        "p95ScoreMinusThreshold": "maxP95ScoreMinusThreshold",
        "maximumScoreMinusThreshold": "maxMaximumScoreMinusThreshold",
    }
    for category in sorted(metrics):
        for metric_name, maximum in maximums.items():
            value = float(metrics[category][metric_name])
            if value > maximum:
                reasons.append(
                    f"{category}.{gate_names[metric_name]}={value:.12g} exceeds {maximum:.12g}"
                )
    return reasons


def _objective_values(candidate_id: str, metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise FreshNormalObservationError("selection objective requires category metrics")
    return {
        "worstAboveThresholdRate": max(float(metric["aboveThresholdRate"]) for metric in metrics.values()),
        "meanAboveThresholdRate": sum(float(metric["aboveThresholdRate"]) for metric in metrics.values()) / len(metrics),
        "worstP95ScoreMinusThreshold": max(float(metric["p95ScoreMinusThreshold"]) for metric in metrics.values()),
        "meanP95ScoreMinusThreshold": sum(
            float(metric["p95ScoreMinusThreshold"]) for metric in metrics.values()
        ) / len(metrics),
        "candidateId": candidate_id,
    }


def _objective_key(values: dict[str, Any]) -> tuple[float, float, float, float, str]:
    required = {
        "worstAboveThresholdRate",
        "meanAboveThresholdRate",
        "worstP95ScoreMinusThreshold",
        "meanP95ScoreMinusThreshold",
        "candidateId",
    }
    _require_exact_fields(values, name="selection objective values", required=required)
    return (
        _require_finite_number(values["worstAboveThresholdRate"], name="selection objective worstAboveThresholdRate", minimum=0.0, maximum=1.0),
        _require_finite_number(values["meanAboveThresholdRate"], name="selection objective meanAboveThresholdRate", minimum=0.0, maximum=1.0),
        _require_finite_number(values["worstP95ScoreMinusThreshold"], name="selection objective worstP95ScoreMinusThreshold", minimum=-2.0),
        _require_finite_number(values["meanP95ScoreMinusThreshold"], name="selection objective meanP95ScoreMinusThreshold", minimum=-2.0),
        _require_string(values["candidateId"], name="selection objective candidateId"),
    )


def _selection_lock_decision(candidate_evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in candidate_evaluations if item["gatePassed"]]
    winner = min(eligible, key=lambda item: _objective_key(item["objectiveValues"])) if eligible else None
    return {
        "state": "RESEARCH_CONFIGURATION_LOCKED" if winner is not None else "NO_ELIGIBLE_CONFIGURATION",
        "selectedCandidateId": None if winner is None else winner["candidateId"],
        "resultScope": "OFFLINE_RESEARCH_CONFIGURATION_LOCK_ONLY",
        "automaticProductionPromotion": False,
        "automaticConfirmation": False,
    }


def _build_selection_lock_evaluations(
    contract: dict[str, Any],
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {str(item["candidateId"]): item for item in observation["candidateObservations"]}
    result: list[dict[str, Any]] = []
    for candidate in contract["candidateReports"]:
        candidate_id = str(candidate["candidateId"])
        observed = by_id.get(candidate_id)
        if observed is None:
            raise FreshNormalObservationError("selection observation omitted a frozen candidate")
        metrics = observed["categoryMetrics"]
        reasons = _gate_rejection_reasons(metrics, contract["selectionGates"])
        gate_passed = not reasons
        result.append({
            "candidateId": candidate_id,
            "categoryMetrics": metrics,
            "gatePassed": gate_passed,
            "gateRejectionReasons": reasons,
            "objectiveValues": _objective_values(candidate_id, metrics) if gate_passed else None,
        })
    return result


def _validate_selection_lock_document(
    document: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_file_sha256: str,
    claim: dict[str, Any],
    claim_file_sha256: str,
    receipt: dict[str, Any],
    receipt_file_sha256: str,
    observation: dict[str, Any],
    observation_file_sha256: str,
) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal selection lock", required=SELECTION_LOCK_FIELDS)
    _validate_static_scope(
        document,
        schema=FRESH_NORMAL_SELECTION_LOCK_SCHEMA,
        purpose=FRESH_NORMAL_SELECTION_LOCK_PURPOSE,
        phase=FRESH_NORMAL_SELECTION_LOCK_PHASE,
        description="fresh normal selection lock",
    )
    expected = {
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionClaimFileSha256": claim_file_sha256,
        "selectionClaimDeclaredSha256": claim["claimSha256"],
        "selectionReceiptFileSha256": receipt_file_sha256,
        "selectionReceiptDeclaredSha256": receipt["selectionReceiptSha256"],
        "selectionObservationFileSha256": observation_file_sha256,
        "selectionObservationDeclaredSha256": observation["selectionObservationSha256"],
    }
    for field, expected_value in expected.items():
        if document.get(field) != expected_value:
            raise FreshNormalObservationError(f"fresh normal selection lock {field} does not bind the frozen JSON evidence")
    supplied = document.get("candidateEvaluations")
    if not isinstance(supplied, list) or not supplied:
        raise FreshNormalObservationError("fresh normal selection lock candidateEvaluations must be non-empty")
    expected_evaluations = _build_selection_lock_evaluations(contract, observation)
    if len(supplied) != len(expected_evaluations):
        raise FreshNormalObservationError("fresh normal selection lock does not cover every candidate")
    parsed_evaluations: list[dict[str, Any]] = []
    for raw, expected_evaluation in zip(supplied, expected_evaluations, strict=True):
        item = _require_exact_fields(raw, name="fresh normal selection lock candidate evaluation", required=LOCK_CANDIDATE_EVALUATION_FIELDS)
        if item.get("candidateId") != expected_evaluation["candidateId"]:
            raise FreshNormalObservationError("fresh normal selection lock candidate order is unsafe")
        if item.get("categoryMetrics") != expected_evaluation["categoryMetrics"]:
            raise FreshNormalObservationError("fresh normal selection lock metrics do not match the observation")
        if item.get("gatePassed") is not expected_evaluation["gatePassed"]:
            raise FreshNormalObservationError("fresh normal selection lock gate outcome does not match the frozen gates")
        if item.get("gateRejectionReasons") != expected_evaluation["gateRejectionReasons"]:
            raise FreshNormalObservationError("fresh normal selection lock gate reasons do not match the frozen gates")
        if item.get("objectiveValues") != expected_evaluation["objectiveValues"]:
            raise FreshNormalObservationError("fresh normal selection lock objective does not match the frozen observation")
        parsed_evaluations.append(expected_evaluation)
    decision = _require_exact_fields(document.get("decision"), name="fresh normal selection lock decision", required=LOCK_DECISION_FIELDS)
    expected_decision = _selection_lock_decision(parsed_evaluations)
    if decision != expected_decision:
        raise FreshNormalObservationError("fresh normal selection lock decision does not match recomputed gates/objective")
    if document.get("selectionLockSha256") != _document_digest(document, "selectionLockSha256"):
        raise FreshNormalObservationError("fresh normal selection lock digest does not match")
    return {**document, "candidateEvaluations": parsed_evaluations, "decision": expected_decision}


def load_validated_fresh_selection_lock_for_contract(
    contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Load/revalidate a selection lock from JSON-only artifacts only."""

    contract, contract_file_sha256, claim, claim_file_sha256 = _validate_contract_and_claim(
        contract_path,
        repository_root=repository_root,
    )
    receipt, receipt_file_sha256 = _load_selection_receipt_for_contract(
        contract,
        contract_file_sha256=contract_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
        repository_root=repository_root,
    )
    observation, observation_file_sha256 = load_validated_fresh_selection_observation_for_contract(
        contract_path,
        repository_root=repository_root,
    )
    path = fresh_selection_lock_path(contract, repository_root=repository_root)
    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal selection lock",
        repository_root=repository_root,
    )
    return (
        _validate_selection_lock_document(
            document,
            contract=contract,
            contract_file_sha256=contract_file_sha256,
            claim=claim,
            claim_file_sha256=claim_file_sha256,
            receipt=receipt,
            receipt_file_sha256=receipt_file_sha256,
            observation=observation,
            observation_file_sha256=observation_file_sha256,
        ),
        file_sha256,
    )


def create_fresh_normal_selection_lock(
    contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Create the JSON-only research lock; it never starts confirmation."""

    contract, contract_file_sha256, claim, claim_file_sha256 = _validate_contract_and_claim(
        contract_path,
        repository_root=repository_root,
    )
    receipt, receipt_file_sha256 = _load_selection_receipt_for_contract(
        contract,
        contract_file_sha256=contract_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
        repository_root=repository_root,
    )
    observation, observation_file_sha256 = load_validated_fresh_selection_observation_for_contract(
        contract_path,
        repository_root=repository_root,
    )
    lock_path = fresh_selection_lock_path(contract, repository_root=repository_root)
    _require_empty_slot(lock_path, description="fresh normal selection lock", repository_root=repository_root)
    evaluations = _build_selection_lock_evaluations(contract, observation)
    lock: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_SELECTION_LOCK_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_SELECTION_LOCK_PURPOSE,
        "phase": FRESH_NORMAL_SELECTION_LOCK_PHASE,
        "blindPolicy": FRESH_NORMAL_BLIND_POLICY,
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionClaimFileSha256": claim_file_sha256,
        "selectionClaimDeclaredSha256": claim["claimSha256"],
        "selectionReceiptFileSha256": receipt_file_sha256,
        "selectionReceiptDeclaredSha256": receipt["selectionReceiptSha256"],
        "selectionObservationFileSha256": observation_file_sha256,
        "selectionObservationDeclaredSha256": observation["selectionObservationSha256"],
        "candidateEvaluations": evaluations,
        "decision": _selection_lock_decision(evaluations),
    }
    lock["selectionLockSha256"] = _document_digest(lock, "selectionLockSha256")
    _validate_selection_lock_document(
        lock,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
        receipt=receipt,
        receipt_file_sha256=receipt_file_sha256,
        observation=observation,
        observation_file_sha256=observation_file_sha256,
    )
    _write_fixed_external_json_fsync(
        lock_path,
        lock,
        description="fresh normal selection lock",
        repository_root=repository_root,
    )
    return lock


def _selected_candidate_from_lock(contract: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    decision = _require_exact_fields(lock.get("decision"), name="fresh normal selection lock decision", required=LOCK_DECISION_FIELDS)
    if decision.get("state") != "RESEARCH_CONFIGURATION_LOCKED":
        raise FreshNormalObservationError("confirmation requires a locked eligible research configuration")
    candidate_id = _require_string(decision.get("selectedCandidateId"), name="selection lock selectedCandidateId")
    candidate = _candidate_by_id(contract, candidate_id)
    selected_evaluation = next(
        (item for item in lock["candidateEvaluations"] if item["candidateId"] == candidate_id),
        None,
    )
    if selected_evaluation is None or selected_evaluation.get("gatePassed") is not True:
        raise FreshNormalObservationError("selection lock selected candidate is not an eligible frozen candidate")
    return candidate


def _validate_confirmation_claim_document(
    document: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_file_sha256: str,
    lock: dict[str, Any],
    lock_file_sha256: str,
) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal confirmation claim", required=CONFIRMATION_CLAIM_FIELDS)
    _validate_static_scope(
        document,
        schema=FRESH_NORMAL_CONFIRMATION_CLAIM_SCHEMA,
        purpose=FRESH_NORMAL_CONFIRMATION_CLAIM_PURPOSE,
        phase=FRESH_NORMAL_CONFIRMATION_CLAIM_PHASE,
        description="fresh normal confirmation claim",
    )
    _validate_confirmation_input(document.get("confirmationInput"))
    if document.get("claimSlot") != FRESH_NORMAL_CONFIRMATION_CLAIM_SLOT:
        raise FreshNormalObservationError("fresh normal confirmation claim uses an unsupported slot")
    candidate = _selected_candidate_from_lock(contract, lock)
    expected = {
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionLockFileSha256": lock_file_sha256,
        "selectionLockDeclaredSha256": lock["selectionLockSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "selectedCandidateId": candidate["candidateId"],
        "candidateConfiguration": candidate["candidateConfiguration"],
        "candidateConfigurationSha256": candidate["candidateConfigurationSha256"],
        "thresholds": candidate["thresholds"],
        "thresholdsIdentitySha256": candidate["thresholdsIdentitySha256"],
    }
    for field, expected_value in expected.items():
        if document.get(field) != expected_value:
            raise FreshNormalObservationError(f"fresh normal confirmation claim {field} does not bind the research lock")
    if document.get("confirmationClaimSha256") != _document_digest(document, "confirmationClaimSha256"):
        raise FreshNormalObservationError("fresh normal confirmation claim digest does not match")
    return document


def _load_confirmation_claim_for_contract(
    contract_path: Path,
    *,
    repository_root: Path,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
]:
    contract, contract_file_sha256, _selection_claim, _selection_claim_file_sha256 = _validate_contract_and_claim(
        contract_path,
        repository_root=repository_root,
    )
    lock, lock_file_sha256 = load_validated_fresh_selection_lock_for_contract(
        contract_path,
        repository_root=repository_root,
    )
    path = fresh_confirmation_claim_path(contract, repository_root=repository_root)
    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal confirmation claim",
        repository_root=repository_root,
    )
    claim = _validate_confirmation_claim_document(
        document,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        lock=lock,
        lock_file_sha256=lock_file_sha256,
    )
    return contract, contract_file_sha256, lock, lock_file_sha256, claim, file_sha256


def create_fresh_normal_confirmation_claim(
    contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Explicitly consume the one confirmation claim slot after a JSON lock.

    This command only reads JSON.  It does not open a confirmation, selection,
    tuning, FIT, or augmentation image.
    """

    contract, contract_file_sha256, _selection_claim, _selection_claim_file_sha256 = _validate_contract_and_claim(
        contract_path,
        repository_root=repository_root,
    )
    lock, lock_file_sha256 = load_validated_fresh_selection_lock_for_contract(
        contract_path,
        repository_root=repository_root,
    )
    candidate = _selected_candidate_from_lock(contract, lock)
    claim_path = fresh_confirmation_claim_path(contract, repository_root=repository_root)
    _require_empty_slot(claim_path, description="fresh normal confirmation claim", repository_root=repository_root)
    claim: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_CONFIRMATION_CLAIM_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_CONFIRMATION_CLAIM_PURPOSE,
        "phase": FRESH_NORMAL_CONFIRMATION_CLAIM_PHASE,
        "blindPolicy": FRESH_NORMAL_BLIND_POLICY,
        "confirmationInput": _confirmation_input(),
        "claimSlot": FRESH_NORMAL_CONFIRMATION_CLAIM_SLOT,
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionLockFileSha256": lock_file_sha256,
        "selectionLockDeclaredSha256": lock["selectionLockSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "selectedCandidateId": candidate["candidateId"],
        "candidateConfiguration": candidate["candidateConfiguration"],
        "candidateConfigurationSha256": candidate["candidateConfigurationSha256"],
        "thresholds": candidate["thresholds"],
        "thresholdsIdentitySha256": candidate["thresholdsIdentitySha256"],
    }
    claim["confirmationClaimSha256"] = _document_digest(claim, "confirmationClaimSha256")
    _validate_confirmation_claim_document(
        claim,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        lock=lock,
        lock_file_sha256=lock_file_sha256,
    )
    _write_fixed_external_json_fsync(
        claim_path,
        claim,
        description="fresh normal confirmation claim",
        repository_root=repository_root,
    )
    return claim


def _make_confirmation_receipt(
    *,
    contract: dict[str, Any],
    contract_file_sha256: str,
    lock: dict[str, Any],
    lock_file_sha256: str,
    claim: dict[str, Any],
    claim_file_sha256: str,
    fit_feature_identity: list[dict[str, Any]],
    fit_records: list[dict[str, Any]],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_CONFIRMATION_RECEIPT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_CONFIRMATION_RECEIPT_PURPOSE,
        "phase": FRESH_NORMAL_CONFIRMATION_RECEIPT_PHASE,
        "blindPolicy": FRESH_NORMAL_BLIND_POLICY,
        "confirmationInput": _confirmation_input(),
        "receiptSlot": FRESH_NORMAL_CONFIRMATION_RECEIPT_SLOT,
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionLockFileSha256": lock_file_sha256,
        "selectionLockDeclaredSha256": lock["selectionLockSha256"],
        "confirmationClaimFileSha256": claim_file_sha256,
        "confirmationClaimDeclaredSha256": claim["confirmationClaimSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "fitFeatureInputIdentitySha256": _canonical_json_sha256(fit_feature_identity),
        "fitFeatureInputCount": len(fit_records),
        "fitOriginalInputCount": sum(not record["isAugmentation"] for record in fit_records),
        "fitAugmentedInputCount": sum(bool(record["isAugmentation"]) for record in fit_records),
    }
    document["confirmationReceiptSha256"] = _document_digest(document, "confirmationReceiptSha256")
    _validate_confirmation_receipt_document(
        document,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        lock=lock,
        lock_file_sha256=lock_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
    )
    return document


def _load_confirmation_receipt_for_contract(
    contract: dict[str, Any],
    *,
    contract_file_sha256: str,
    lock: dict[str, Any],
    lock_file_sha256: str,
    claim: dict[str, Any],
    claim_file_sha256: str,
    repository_root: Path,
) -> tuple[dict[str, Any], str]:
    path = fresh_confirmation_receipt_path(contract, repository_root=repository_root)
    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal confirmation receipt",
        repository_root=repository_root,
    )
    return (
        _validate_confirmation_receipt_document(
            document,
            contract=contract,
            contract_file_sha256=contract_file_sha256,
            lock=lock,
            lock_file_sha256=lock_file_sha256,
            claim=claim,
            claim_file_sha256=claim_file_sha256,
        ),
        file_sha256,
    )


def _validate_confirmation_observation_document(
    document: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_file_sha256: str,
    lock: dict[str, Any],
    lock_file_sha256: str,
    claim: dict[str, Any],
    claim_file_sha256: str,
    receipt: dict[str, Any],
    receipt_file_sha256: str,
) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal confirmation observation", required=CONFIRMATION_OBSERVATION_FIELDS)
    _validate_static_scope(
        document,
        schema=FRESH_NORMAL_CONFIRMATION_OBSERVATION_SCHEMA,
        purpose=FRESH_NORMAL_CONFIRMATION_OBSERVATION_PURPOSE,
        phase=FRESH_NORMAL_CONFIRMATION_OBSERVATION_PHASE,
        description="fresh normal confirmation observation",
    )
    _validate_confirmation_input(document.get("confirmationInput"))
    if document.get("resultScope") != FRESH_NORMAL_CONFIRMATION_RESULT_SCOPE:
        raise FreshNormalObservationError("fresh normal confirmation observation result scope is unsafe")
    candidate = _selected_candidate_from_lock(contract, lock)
    expected = {
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionLockFileSha256": lock_file_sha256,
        "selectionLockDeclaredSha256": lock["selectionLockSha256"],
        "confirmationClaimFileSha256": claim_file_sha256,
        "confirmationClaimDeclaredSha256": claim["confirmationClaimSha256"],
        "confirmationReceiptFileSha256": receipt_file_sha256,
        "confirmationReceiptDeclaredSha256": receipt["confirmationReceiptSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractor": contract["featureExtractor"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "selectedCandidateId": candidate["candidateId"],
        "candidateConfiguration": candidate["candidateConfiguration"],
        "candidateConfigurationSha256": candidate["candidateConfigurationSha256"],
        "thresholds": candidate["thresholds"],
        "thresholdsIdentitySha256": candidate["thresholdsIdentitySha256"],
        "normalConfirmationInputs": contract["normalConfirmationInputs"],
        "normalConfirmationInputIdentitySha256": contract["normalConfirmationInputIdentitySha256"],
    }
    for field, expected_value in expected.items():
        if document.get(field) != expected_value:
            raise FreshNormalObservationError(f"fresh normal confirmation observation {field} does not bind the frozen input")
    scores = _validate_score_records(
        document.get("confirmationScores"),
        expected_inputs=contract["normalConfirmationInputs"],
        partition="NORMAL_CONFIRMATION",
        name="fresh normal confirmation observation confirmationScores",
    )
    thresholds = _validate_thresholds(
        document.get("thresholds"),
        expected=candidate["thresholds"],
        name="fresh normal confirmation observation thresholds",
    )
    metrics = _validate_category_metrics(
        document.get("categoryMetrics"),
        scores=scores,
        thresholds=thresholds,
        name="fresh normal confirmation observation categoryMetrics",
    )
    _validate_normal_only_evidence(
        document.get("normalOnlyEvidence"),
        receipt=receipt,
        query_inputs=contract["normalConfirmationInputs"],
        partition="NORMAL_CONFIRMATION",
    )
    _validate_execution(document.get("execution"), name="fresh normal confirmation observation execution")
    if document.get("confirmationObservationSha256") != _document_digest(document, "confirmationObservationSha256"):
        raise FreshNormalObservationError("fresh normal confirmation observation digest does not match")
    return {**document, "confirmationScores": scores, "categoryMetrics": metrics}


def load_validated_fresh_confirmation_observation_for_contract(
    contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Load one confirmation observation without opening any image bytes."""

    contract, contract_file_sha256, lock, lock_file_sha256, claim, claim_file_sha256 = _load_confirmation_claim_for_contract(
        contract_path,
        repository_root=repository_root,
    )
    receipt, receipt_file_sha256 = _load_confirmation_receipt_for_contract(
        contract,
        contract_file_sha256=contract_file_sha256,
        lock=lock,
        lock_file_sha256=lock_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
        repository_root=repository_root,
    )
    path = fresh_confirmation_observation_path(contract, repository_root=repository_root)
    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal confirmation observation",
        repository_root=repository_root,
    )
    return (
        _validate_confirmation_observation_document(
            document,
            contract=contract,
            contract_file_sha256=contract_file_sha256,
            lock=lock,
            lock_file_sha256=lock_file_sha256,
            claim=claim,
            claim_file_sha256=claim_file_sha256,
            receipt=receipt,
            receipt_file_sha256=receipt_file_sha256,
        ),
        file_sha256,
    )


def run_fresh_normal_confirmation_observation(
    contract_path: Path,
    holdout_manifest_path: Path,
    augmentation_manifest_path: Path,
    recipe_path: Path,
    *,
    source_root: Path,
    model_repo: Path,
    model_weights: Path,
    device: str = "cpu",
    repository_root: Path = REPOSITORY_ROOT,
    embedder_factory: Callable[..., Any] = evaluator.FreshHoldoutBatchEmbedder,
    identity_factory: Callable[..., dict[str, Any]] = evaluator._feature_extractor_identity,
) -> dict[str, Any]:
    """Consume raw ``NORMAL_CONFIRMATION`` once, without changing a setting.

    The explicit confirmation claim must already exist.  This function writes
    the receipt after FIT-only preflight and immediately before the first raw
    confirmation image is decoded.  It does not open selection or tuning.
    """

    started = time.perf_counter()
    timings = {field: 0.0 for field in TIMING_FIELDS}
    contract, contract_file_sha256, lock, lock_file_sha256, claim, claim_file_sha256 = _load_confirmation_claim_for_contract(
        contract_path,
        repository_root=repository_root,
    )
    candidate = _selected_candidate_from_lock(contract, lock)
    candidate_id = str(candidate["candidateId"])
    receipt_path = fresh_confirmation_receipt_path(contract, repository_root=repository_root)
    observation_path = fresh_confirmation_observation_path(contract, repository_root=repository_root)
    _require_empty_slot(receipt_path, description="fresh normal confirmation receipt", repository_root=repository_root)
    _require_empty_slot(observation_path, description="fresh normal confirmation observation", repository_root=repository_root)
    preflight_started = time.perf_counter()
    (
        _holdout_document,
        _holdout_file_sha256,
        _augmentation_document,
        _augmentation_file_sha256,
        fit_records,
        fit_feature_identity,
    ) = _load_fit_prototype_inputs(
        contract,
        holdout_manifest_path,
        augmentation_manifest_path,
        recipe_path,
        source_root=source_root,
        repository_root=repository_root,
    )
    timings["inputPreflightSeconds"] += time.perf_counter() - preflight_started
    configurations, prototypes, query_embedder = _build_candidate_prototypes(
        contract,
        fit_records,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        embedder_factory=embedder_factory,
        identity_factory=identity_factory,
        timings=timings,
        candidate_ids={candidate_id},
    )
    configuration = configurations.get(candidate_id)
    if configuration is None or configuration != candidate["candidateConfiguration"]:
        raise FreshNormalObservationError("selected confirmation candidate configuration is inconsistent")
    # Ensure every JSON predecessor remains byte-identical before the receipt
    # burns confirmation.  This revalidation has no image I/O.
    current_contract, current_contract_sha256, current_lock, current_lock_sha256, current_claim, current_claim_sha256 = (
        _load_confirmation_claim_for_contract(contract_path, repository_root=repository_root)
    )
    if (
        current_contract != contract
        or current_contract_sha256 != contract_file_sha256
        or current_lock != lock
        or current_lock_sha256 != lock_file_sha256
        or current_claim != claim
        or current_claim_sha256 != claim_file_sha256
    ):
        raise FreshNormalObservationError("confirmation JSON boundary changed during FIT preflight")
    _assert_feature_extractor_identity(
        contract,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        identity_factory=identity_factory,
    )
    receipt = _make_confirmation_receipt(
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        lock=lock,
        lock_file_sha256=lock_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
        fit_feature_identity=fit_feature_identity,
        fit_records=fit_records,
    )
    receipt_started = time.perf_counter()
    _write_fixed_external_json_fsync(
        receipt_path,
        receipt,
        description="fresh normal confirmation receipt",
        repository_root=repository_root,
    )
    timings["receiptCommitSeconds"] += time.perf_counter() - receipt_started
    # Reuse the identity-checked extractor that built the selected FIT bank.
    if _assert_feature_extractor_identity(
        contract,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        identity_factory=identity_factory,
    ) != contract["featureExtractor"]:
        raise FreshNormalObservationError("feature extractor changed before held-out confirmation scoring")
    # No call above this point receives NORMAL_CONFIRMATION source records.
    query_records = _load_raw_query_inputs(
        contract,
        holdout_manifest_path,
        source_root=source_root,
        partition="NORMAL_CONFIRMATION",
    )
    feature_timing: dict[str, float] = {}
    query_features = _extract_patch_features(
        query_records,
        embedder=query_embedder,
        batch_size=int(configuration["batchSize"]),
        timings=feature_timing,
    )
    timings["queryFeatureSeconds"] += sum(feature_timing.values())
    scoring_started = time.perf_counter()
    scores, metrics = _score_query_records(
        query_records,
        query_features,
        prototypes[candidate_id],
        configuration,
        candidate["thresholds"],
        partition="NORMAL_CONFIRMATION",
    )
    timings["scoringSeconds"] += time.perf_counter() - scoring_started
    del query_features
    completed_identity = _assert_feature_extractor_identity(
        contract,
        model_repo=model_repo,
        model_weights=model_weights,
        device=device,
        identity_factory=identity_factory,
    )
    if completed_identity != contract["featureExtractor"]:
        raise FreshNormalObservationError("feature extractor changed while held-out confirmation features were extracted")
    receipt_file_sha256 = selection.sha256_file(receipt_path)
    timings["totalElapsedSeconds"] = time.perf_counter() - started
    observation: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_CONFIRMATION_OBSERVATION_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_CONFIRMATION_OBSERVATION_PURPOSE,
        "phase": FRESH_NORMAL_CONFIRMATION_OBSERVATION_PHASE,
        "blindPolicy": FRESH_NORMAL_BLIND_POLICY,
        "confirmationInput": _confirmation_input(),
        "resultScope": FRESH_NORMAL_CONFIRMATION_RESULT_SCOPE,
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "selectionLockFileSha256": lock_file_sha256,
        "selectionLockDeclaredSha256": lock["selectionLockSha256"],
        "confirmationClaimFileSha256": claim_file_sha256,
        "confirmationClaimDeclaredSha256": claim["confirmationClaimSha256"],
        "confirmationReceiptFileSha256": receipt_file_sha256,
        "confirmationReceiptDeclaredSha256": receipt["confirmationReceiptSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractor": contract["featureExtractor"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "selectedCandidateId": candidate_id,
        "candidateConfiguration": configuration,
        "candidateConfigurationSha256": candidate["candidateConfigurationSha256"],
        "thresholds": candidate["thresholds"],
        "thresholdsIdentitySha256": candidate["thresholdsIdentitySha256"],
        "normalConfirmationInputs": contract["normalConfirmationInputs"],
        "normalConfirmationInputIdentitySha256": contract["normalConfirmationInputIdentitySha256"],
        "confirmationScores": scores,
        "categoryMetrics": metrics,
        "normalOnlyEvidence": _build_normal_only_evidence(
            fit_records,
            query_records,
            partition="NORMAL_CONFIRMATION",
        ),
        "execution": _execution_metadata(timings),
    }
    observation["confirmationObservationSha256"] = _document_digest(observation, "confirmationObservationSha256")
    _validate_confirmation_observation_document(
        observation,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        lock=lock,
        lock_file_sha256=lock_file_sha256,
        claim=claim,
        claim_file_sha256=claim_file_sha256,
        receipt=receipt,
        receipt_file_sha256=receipt_file_sha256,
    )
    _write_fixed_external_json_fsync(
        observation_path,
        observation,
        description="fresh normal confirmation observation",
        repository_root=repository_root,
    )
    return observation
