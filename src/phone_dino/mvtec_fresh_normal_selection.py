"""Fresh, JSON-only normal-selection contract and one-time claim primitives.

This module deliberately does not load an image, invoke DINO, or reuse the
historical V3--V5 selector.  It freezes the exact normal-only development
reports before a future observer is allowed to consume ``NORMAL_SELECTION``.
The claim lives in one fixed sibling slot and is created with ``xb`` so a
second selection attempt cannot silently move to a new path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Sequence

from phone_dino import mvtec_fresh_fit_augmentation as augmentation
from phone_dino import mvtec_normal_holdout as holdout
from phone_dino import mvtec_normal_holdout_evaluator as evaluator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FRESH_NORMAL_SELECTION_CONTRACT_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-selection-contract/1.0"
FRESH_NORMAL_SELECTION_CLAIM_SCHEMA = "phone-dino.mvtec-ad-fresh-normal-selection-claim/1.0"
FRESH_NORMAL_SELECTION_CONTRACT_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_SELECTION_CONTRACT"
FRESH_NORMAL_SELECTION_CLAIM_PURPOSE = "OFFLINE_MVTEC_FRESH_NORMAL_SELECTION_CLAIM"
FRESH_NORMAL_SELECTION_CONTRACT_PHASE = "NORMAL_SELECTION_CONTRACT"
FRESH_NORMAL_SELECTION_CLAIM_PHASE = "NORMAL_SELECTION_CLAIM"
FRESH_NORMAL_SELECTION_BLIND_POLICY = "NO_BLIND_OR_ANOMALY_DATA"
FRESH_NORMAL_SELECTION_CLAIM_SLOT = "FRESH_NORMAL_SELECTION_CONSUMPTION_V1"
FRESH_NORMAL_SELECTION_CLAIM_FILENAME = "fresh_normal_selection_claim.json"

FRESH_NORMAL_SELECTION_INPUT = {
    "partition": "NORMAL_SELECTION",
    "kind": "NOMINAL",
    "defect": "good",
    "rawOnly": True,
    "oneTimeClaimRequired": True,
}
FRESH_NORMAL_SELECTION_DECISION_POLICY = {
    "automaticProductionPromotion": False,
    "automaticConfirmation": False,
    "resultScope": "OFFLINE_RESEARCH_CONFIGURATION_LOCK_ONLY",
}
FRESH_NORMAL_SELECTION_OBJECTIVE = {
    "algorithm": "LEXICOGRAPHIC_MINIMIZE_NORMAL_SELECTION_EXCESS_V1",
    "terms": [
        "worstAboveThresholdRate",
        "meanAboveThresholdRate",
        "worstP95ScoreMinusThreshold",
        "meanP95ScoreMinusThreshold",
        "candidateIdAscending",
    ],
}

CONTRACT_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "selectionInput",
    "decisionPolicy",
    "holdout",
    "augmentation",
    "normalSelectionInputs",
    "normalSelectionInputIdentitySha256",
    "normalConfirmationInputs",
    "normalConfirmationInputIdentitySha256",
    "featureExtractor",
    "featureExtractorIdentitySha256",
    "candidateReports",
    "candidateUniverseIdentitySha256",
    "selectionGates",
    "selectionObjective",
    "contractSha256",
}
CLAIM_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "selectionInput",
    "claimSlot",
    "contractFileSha256",
    "contractDeclaredSha256",
    "holdout",
    "augmentation",
    "featureExtractorIdentitySha256",
    "candidateUniverseIdentitySha256",
    "claimSha256",
}
HOLDOUT_BINDING_FIELDS = {
    "manifestFileSha256",
    "manifestDeclaredSha256",
    "developmentIdentitySha256",
    "normalSelectionIdentitySha256",
    "normalConfirmationIdentitySha256",
    "normalSelectionRecordCount",
    "normalSelectionCategoryCounts",
}
AUGMENTATION_BINDING_FIELDS = {
    "manifestFileSha256",
    "manifestDeclaredSha256",
    "developmentIdentitySha256",
    "fitParentIdentitySha256",
    "recipeFileSha256",
    "variantsPerParent",
}
CANDIDATE_REPORT_BINDING_FIELDS = {
    "candidateId",
    "developmentReportFileSha256",
    "developmentReportDeclaredSha256",
    "candidateConfiguration",
    "candidateConfigurationSha256",
    "featureExtractorIdentitySha256",
    "featureInputIdentitySha256",
    "calibrationInputIdentitySha256",
    "thresholds",
    "thresholdsIdentitySha256",
}
SELECTION_GATE_FIELDS = {
    "maxAboveThresholdRate",
    "maxP95ScoreMinusThreshold",
    "maxMaximumScoreMinusThreshold",
}
NORMAL_PARTITION_INPUT_FIELDS = {
    "caseId",
    "category",
    "partition",
    "kind",
    "defect",
    "sourceSha256",
    "sourceGroupId",
    "acquisitionStratum",
    "expectedRemoteSha256",
    "expectedRemoteBytes",
}


class FreshNormalSelectionError(ValueError):
    """Raised when a fresh normal-selection artifact is unsafe or inconsistent."""


def canonical_json_sha256(document: Any) -> str:
    """Return the protocol's canonical SHA-256 after rejecting non-finite values."""

    _validate_json_value(document, name="canonical JSON value")
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


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
            raise FreshNormalSelectionError(f"{name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(child, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise FreshNormalSelectionError(f"{name} contains a non-string JSON key")
            _validate_json_value(child, name=f"{name}.{key}")
        return
    raise FreshNormalSelectionError(f"{name} contains a value that is not JSON-compatible")


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return canonical_json_sha256(unsigned)


def _require_exact_fields(document: dict[str, Any], *, name: str, required: set[str]) -> None:
    missing = required.difference(document)
    if missing:
        raise FreshNormalSelectionError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    unknown = set(document).difference(required)
    if unknown:
        raise FreshNormalSelectionError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FreshNormalSelectionError(f"{name} must be an object")
    return value


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreshNormalSelectionError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise FreshNormalSelectionError(f"{name} must be a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise FreshNormalSelectionError(f"{name} must be a SHA-256 digest") from error
    return digest


def _require_positive_int(value: object, *, name: str, maximum: int = 1_000_000) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= maximum:
        raise FreshNormalSelectionError(f"{name} must be a positive integer no larger than {maximum}")
    return value


def _require_nonnegative_int(value: object, *, name: str, maximum: int = 1_000_000) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
        raise FreshNormalSelectionError(f"{name} must be a non-negative integer no larger than {maximum}")
    return value


def _require_finite_number(value: object, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FreshNormalSelectionError(f"{name} must be a finite number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise FreshNormalSelectionError(f"{name} must be between {minimum} and {maximum}")
    return result


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
                raise FreshNormalSelectionError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _require_external_file(path: Path, *, description: str, repository_root: Path) -> None:
    if not isinstance(path, Path):
        raise FreshNormalSelectionError(f"{description} path must be a Path")
    _reject_links_on_existing_path(path, description=description)
    if not path.is_file() or _is_link_or_reparse_point(path):
        raise FreshNormalSelectionError(f"{description} must be a regular non-link file")
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise FreshNormalSelectionError(f"{description} must stay outside the Git working tree")


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError as error:
        raise FreshNormalSelectionError("unable to stat immutable JSON input") from error
    return (status.st_dev, status.st_ino, status.st_mode, status.st_size, status.st_mtime_ns)


def _parse_json_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreshNormalSelectionError(f"{description} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise FreshNormalSelectionError(f"{description} contains a non-finite JSON value: {value}")

    try:
        document = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshNormalSelectionError(f"unable to read {description}") from error
    _validate_json_value(document, name=description)
    if not isinstance(document, dict):
        raise FreshNormalSelectionError(f"{description} must be a JSON object")
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
        raise FreshNormalSelectionError(f"unable to read {description}") from error
    _require_external_file(path, description=description, repository_root=repository_root)
    if before != _stat_signature(path):
        raise FreshNormalSelectionError(f"{description} changed while it was read")
    return _parse_json_bytes(raw, description=description), f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _prepare_external_output(path: Path, *, description: str, repository_root: Path) -> None:
    if not isinstance(path, Path):
        raise FreshNormalSelectionError(f"{description} path must be a Path")
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise FreshNormalSelectionError(f"{description} must stay outside the Git working tree")
    if path.exists() or path.is_symlink():
        raise FreshNormalSelectionError(f"{description} already exists; the immutable slot is already consumed")
    _reject_links_on_existing_path(path.parent, description=description)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FreshNormalSelectionError(f"unable to create {description} parent directory") from error
    _reject_links_on_existing_path(path.parent, description=description)


def _write_external_json(path: Path, document: dict[str, Any], *, description: str, repository_root: Path) -> None:
    _prepare_external_output(path, description=description, repository_root=repository_root)
    payload = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise FreshNormalSelectionError(f"unable to write immutable {description}") from error


def _require_static_scope(
    document: dict[str, Any],
    *,
    schema: str,
    purpose: str,
    phase: str,
    description: str,
) -> None:
    if document.get("schemaVersion") != schema:
        raise FreshNormalSelectionError(f"{description} schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise FreshNormalSelectionError(f"{description} must be non-authoritative and non-production")
    if document.get("purpose") != purpose or document.get("phase") != phase:
        raise FreshNormalSelectionError(f"{description} purpose or phase is unsafe")
    if document.get("blindPolicy") != FRESH_NORMAL_SELECTION_BLIND_POLICY:
        raise FreshNormalSelectionError(f"{description} blind policy is unsafe")


def _validate_selection_input(value: object) -> dict[str, Any]:
    document = _require_mapping(value, name="selectionInput")
    _require_exact_fields(document, name="selectionInput", required=set(FRESH_NORMAL_SELECTION_INPUT))
    if document != FRESH_NORMAL_SELECTION_INPUT:
        raise FreshNormalSelectionError("selectionInput must require raw NORMAL_SELECTION nominal-good data")
    return dict(document)


def _validate_decision_policy(value: object) -> dict[str, Any]:
    document = _require_mapping(value, name="decisionPolicy")
    _require_exact_fields(document, name="decisionPolicy", required=set(FRESH_NORMAL_SELECTION_DECISION_POLICY))
    if document != FRESH_NORMAL_SELECTION_DECISION_POLICY:
        raise FreshNormalSelectionError("decisionPolicy must prohibit automatic promotion and confirmation")
    return dict(document)


def _validate_selection_gates(value: object) -> dict[str, float]:
    document = _require_mapping(value, name="selectionGates")
    _require_exact_fields(document, name="selectionGates", required=SELECTION_GATE_FIELDS)
    return {
        "maxAboveThresholdRate": _require_finite_number(
            document.get("maxAboveThresholdRate"),
            name="selectionGates.maxAboveThresholdRate",
            minimum=0.0,
            maximum=1.0,
        ),
        "maxP95ScoreMinusThreshold": _require_finite_number(
            document.get("maxP95ScoreMinusThreshold"),
            name="selectionGates.maxP95ScoreMinusThreshold",
            minimum=0.0,
            maximum=2.0,
        ),
        "maxMaximumScoreMinusThreshold": _require_finite_number(
            document.get("maxMaximumScoreMinusThreshold"),
            name="selectionGates.maxMaximumScoreMinusThreshold",
            minimum=0.0,
            maximum=2.0,
        ),
    }


def _validate_selection_objective(value: object) -> dict[str, Any]:
    document = _require_mapping(value, name="selectionObjective")
    _require_exact_fields(document, name="selectionObjective", required=set(FRESH_NORMAL_SELECTION_OBJECTIVE))
    if document != FRESH_NORMAL_SELECTION_OBJECTIVE:
        raise FreshNormalSelectionError("selectionObjective must use the fixed lexicographic normal-only objective")
    return {"algorithm": str(document["algorithm"]), "terms": list(document["terms"])}


def _validate_holdout_binding(value: object) -> dict[str, Any]:
    document = _require_mapping(value, name="holdout binding")
    _require_exact_fields(document, name="holdout binding", required=HOLDOUT_BINDING_FIELDS)
    result = {
        name: _require_sha256(document.get(name), name=f"holdout.{name}")
        for name in (
            "manifestFileSha256",
            "manifestDeclaredSha256",
            "developmentIdentitySha256",
            "normalSelectionIdentitySha256",
            "normalConfirmationIdentitySha256",
        )
    }
    result["normalSelectionRecordCount"] = _require_positive_int(
        document.get("normalSelectionRecordCount"), name="holdout.normalSelectionRecordCount"
    )
    counts = _require_mapping(document.get("normalSelectionCategoryCounts"), name="holdout.normalSelectionCategoryCounts")
    if not counts:
        raise FreshNormalSelectionError("holdout.normalSelectionCategoryCounts must not be empty")
    if list(counts) != sorted(counts):
        raise FreshNormalSelectionError("holdout.normalSelectionCategoryCounts must be sorted")
    parsed_counts: dict[str, int] = {}
    for category, count in counts.items():
        _require_string(category, name="holdout normal-selection category")
        parsed_counts[category] = _require_positive_int(count, name=f"holdout normal-selection count for {category}")
    if sum(parsed_counts.values()) != result["normalSelectionRecordCount"]:
        raise FreshNormalSelectionError("holdout normal-selection category counts do not add up")
    result["normalSelectionCategoryCounts"] = parsed_counts
    return result


def _holdout_partition_input_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return image-path-free membership evidence for a holdout partition."""

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


def _validate_normal_partition_inputs(value: object, *, partition: str, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FreshNormalSelectionError(f"{name} must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_sources: set[str] = set()
    for item in value:
        document = _require_mapping(item, name=f"{name} record")
        _require_exact_fields(document, name=f"{name} record", required=NORMAL_PARTITION_INPUT_FIELDS)
        case_id = _require_string(document.get("caseId"), name=f"{name}.caseId")
        source_sha256 = _require_sha256(document.get("sourceSha256"), name=f"{name}.sourceSha256")
        if case_id in seen_case_ids or source_sha256 in seen_sources:
            raise FreshNormalSelectionError(f"{name} has a duplicate caseId or source digest")
        seen_case_ids.add(case_id)
        seen_sources.add(source_sha256)
        if document.get("partition") != partition:
            raise FreshNormalSelectionError(f"{name} has an unexpected partition")
        if document.get("kind") != "NOMINAL" or document.get("defect") != "good":
            raise FreshNormalSelectionError(f"{name} must contain nominal-good inputs only")
        if document.get("sourceGroupId") != f"CONTENT_SHA256:{source_sha256[7:]}":
            raise FreshNormalSelectionError(f"{name} sourceGroupId is inconsistent")
        if document.get("acquisitionStratum") != "OFFICIAL_MVTEC_TRAIN_GOOD":
            raise FreshNormalSelectionError(f"{name} acquisition stratum is unsupported")
        expected_remote = _require_sha256(document.get("expectedRemoteSha256"), name=f"{name}.expectedRemoteSha256")
        if expected_remote != source_sha256:
            raise FreshNormalSelectionError(f"{name} remote and source digests do not match")
        result.append({
            "caseId": case_id,
            "category": _require_string(document.get("category"), name=f"{name}.category"),
            "partition": partition,
            "kind": "NOMINAL",
            "defect": "good",
            "sourceSha256": source_sha256,
            "sourceGroupId": document["sourceGroupId"],
            "acquisitionStratum": document["acquisitionStratum"],
            "expectedRemoteSha256": expected_remote,
            "expectedRemoteBytes": _require_positive_int(
                document.get("expectedRemoteBytes"), name=f"{name}.expectedRemoteBytes", maximum=1_000_000_000
            ),
        })
    if [entry["caseId"] for entry in result] != sorted(entry["caseId"] for entry in result):
        raise FreshNormalSelectionError(f"{name} must be sorted by caseId")
    return result


def _validate_augmentation_binding(value: object) -> dict[str, Any]:
    document = _require_mapping(value, name="augmentation binding")
    _require_exact_fields(document, name="augmentation binding", required=AUGMENTATION_BINDING_FIELDS)
    result = {
        name: _require_sha256(document.get(name), name=f"augmentation.{name}")
        for name in (
            "manifestFileSha256",
            "manifestDeclaredSha256",
            "developmentIdentitySha256",
            "fitParentIdentitySha256",
            "recipeFileSha256",
        )
    }
    result["variantsPerParent"] = _require_positive_int(
        document.get("variantsPerParent"), name="augmentation.variantsPerParent", maximum=8
    )
    return result


def _validate_feature_input_records(value: object, *, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FreshNormalSelectionError(f"{name} must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for item in value:
        document = _require_mapping(item, name=f"{name} record")
        _require_exact_fields(document, name=f"{name} record", required=evaluator.FEATURE_INPUT_FIELDS)
        case_id = _require_string(document.get("caseId"), name=f"{name}.caseId")
        if case_id in seen_case_ids:
            raise FreshNormalSelectionError(f"{name} has a duplicate caseId")
        seen_case_ids.add(case_id)
        partition = document.get("partition")
        if partition not in {"FIT", "THRESHOLD_TUNING"}:
            raise FreshNormalSelectionError(f"{name} contains a held-out or unsupported partition")
        if document.get("kind") != "NOMINAL" or document.get("defect") != "good":
            raise FreshNormalSelectionError(f"{name} must contain nominal-good data only")
        is_augmentation = document.get("isAugmentation")
        if not isinstance(is_augmentation, bool):
            raise FreshNormalSelectionError(f"{name}.isAugmentation must be a boolean")
        entry = {
            "caseId": case_id,
            "category": _require_string(document.get("category"), name=f"{name}.category"),
            "partition": partition,
            "kind": "NOMINAL",
            "defect": "good",
            "sourceSha256": _require_sha256(document.get("sourceSha256"), name=f"{name}.sourceSha256"),
            "isAugmentation": is_augmentation,
            "variantId": document.get("variantId"),
            "parentCaseId": document.get("parentCaseId"),
            "parentSourceSha256": document.get("parentSourceSha256"),
            "augmentationManifestSha256": document.get("augmentationManifestSha256"),
        }
        if is_augmentation:
            if partition != "FIT":
                raise FreshNormalSelectionError(f"{name} augmentation must be FIT-only")
            entry["variantId"] = _require_positive_int(entry["variantId"], name=f"{name}.variantId", maximum=8)
            entry["parentCaseId"] = _require_string(entry["parentCaseId"], name=f"{name}.parentCaseId")
            entry["parentSourceSha256"] = _require_sha256(
                entry["parentSourceSha256"], name=f"{name}.parentSourceSha256"
            )
            entry["augmentationManifestSha256"] = _require_sha256(
                entry["augmentationManifestSha256"], name=f"{name}.augmentationManifestSha256"
            )
        elif any(entry[field] is not None for field in ("variantId", "parentCaseId", "parentSourceSha256", "augmentationManifestSha256")):
            raise FreshNormalSelectionError(f"{name} original record has augmentation-only fields")
        result.append(entry)
    if [entry["caseId"] for entry in result] != sorted(entry["caseId"] for entry in result):
        raise FreshNormalSelectionError(f"{name} must be sorted by caseId")
    return result


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


def _same_number(left: float, right: object) -> bool:
    return isinstance(right, (int, float)) and not isinstance(right, bool) and math.isclose(
        left, float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _validate_development_report_document(document: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(document, name="development report", required=evaluator.DEVELOPMENT_REPORT_FIELDS)
    if document.get("schemaVersion") != evaluator.DEVELOPMENT_REPORT_SCHEMA:
        raise FreshNormalSelectionError("development report schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise FreshNormalSelectionError("development report must be non-authoritative and non-production")
    if document.get("purpose") != evaluator.DEVELOPMENT_REPORT_PURPOSE or document.get("phase") != evaluator.DEVELOPMENT_PHASE:
        raise FreshNormalSelectionError("development report purpose or phase is unsafe")
    if document.get("blindPolicy") != evaluator.NORMAL_HOLDOUT_BLIND_POLICY:
        raise FreshNormalSelectionError("development report blind policy is unsafe")
    if document.get("inputPolicy") != "FIT_PLUS_RAW_THRESHOLD_TUNING_ONLY":
        raise FreshNormalSelectionError("development report input policy is unsafe")
    if document.get("developmentReportSha256") != _document_digest(document, "developmentReportSha256"):
        raise FreshNormalSelectionError("development report digest does not match")
    for field in (
        "holdoutManifestFileSha256",
        "holdoutManifestDeclaredSha256",
        "developmentIdentitySha256",
        "augmentationManifestFileSha256",
        "augmentationManifestDeclaredSha256",
        "fitParentIdentitySha256",
        "augmentationRecipeFileSha256",
        "candidateConfigurationSha256",
        "featureExtractorIdentitySha256",
        "featureInputIdentitySha256",
        "calibrationInputIdentitySha256",
    ):
        _require_sha256(document.get(field), name=f"development report {field}")
    try:
        configuration = evaluator.validate_candidate_configuration(document.get("candidateConfiguration"))
    except evaluator.NormalHoldoutEvaluatorError as error:
        raise FreshNormalSelectionError(str(error)) from error
    if document.get("candidateConfigurationSha256") != canonical_json_sha256(configuration):
        raise FreshNormalSelectionError("development report candidate configuration digest does not match")
    extractor = _require_mapping(document.get("featureExtractor"), name="development report featureExtractor")
    if not extractor:
        raise FreshNormalSelectionError("development report featureExtractor must not be empty")
    _validate_json_value(extractor, name="development report featureExtractor")
    if document.get("featureExtractorIdentitySha256") != canonical_json_sha256(extractor):
        raise FreshNormalSelectionError("development report feature extractor digest does not match")
    feature_inputs = _validate_feature_input_records(document.get("featureInputs"), name="development featureInputs")
    if document.get("featureInputIdentitySha256") != canonical_json_sha256(feature_inputs):
        raise FreshNormalSelectionError("development report feature input digest does not match")
    calibration_inputs = _validate_feature_input_records(document.get("calibrationInputs"), name="development calibrationInputs")
    expected_calibration_inputs = [entry for entry in feature_inputs if entry["partition"] == "THRESHOLD_TUNING"]
    if calibration_inputs != expected_calibration_inputs:
        raise FreshNormalSelectionError("development calibration inputs must be exactly raw threshold tuning inputs")
    if any(entry["isAugmentation"] for entry in calibration_inputs):
        raise FreshNormalSelectionError("development calibration inputs must remain raw")
    if document.get("calibrationInputIdentitySha256") != canonical_json_sha256(calibration_inputs):
        raise FreshNormalSelectionError("development report calibration input digest does not match")

    scores_raw = document.get("calibrationScores")
    if not isinstance(scores_raw, list) or not scores_raw:
        raise FreshNormalSelectionError("development calibrationScores must be a non-empty list")
    calibration_by_case = {entry["caseId"]: entry for entry in calibration_inputs}
    scores: list[dict[str, Any]] = []
    seen_scores: set[str] = set()
    for item in scores_raw:
        score = _require_mapping(item, name="development calibration score")
        _require_exact_fields(score, name="development calibration score", required=evaluator.CALIBRATION_SCORE_FIELDS)
        case_id = _require_string(score.get("caseId"), name="development calibration score caseId")
        expected = calibration_by_case.get(case_id)
        if expected is None or case_id in seen_scores:
            raise FreshNormalSelectionError("development calibration score membership is invalid")
        seen_scores.add(case_id)
        for field in ("category", "partition", "kind", "defect", "sourceSha256"):
            if score.get(field) != expected[field]:
                raise FreshNormalSelectionError("development calibration score does not match its raw tuning input")
        values = {
            field: _require_finite_number(score.get(field), name=f"development calibration score {field}", minimum=0.0, maximum=2.0)
            for field in ("score", "maxPatchDistance", "meanNearestPatchDistance")
        }
        if not values["meanNearestPatchDistance"] <= values["score"] <= values["maxPatchDistance"]:
            raise FreshNormalSelectionError("development calibration score components are inconsistent")
        scores.append({**{field: score[field] for field in evaluator.CALIBRATION_SCORE_FIELDS if field not in values}, **values})
    if len(scores) != len(calibration_inputs) or [score["caseId"] for score in scores] != sorted(score["caseId"] for score in scores):
        raise FreshNormalSelectionError("development calibration scores must cover raw tuning inputs once in caseId order")

    thresholds = _require_mapping(document.get("thresholds"), name="development thresholds")
    categories = _require_mapping(document.get("categories"), name="development categories")
    expected_categories = sorted({entry["category"] for entry in calibration_inputs})
    if list(thresholds) != expected_categories or list(categories) != expected_categories:
        raise FreshNormalSelectionError("development categories must exactly match raw tuning categories in sorted order")
    scores_by_category: dict[str, list[float]] = {category: [] for category in expected_categories}
    for score in scores:
        scores_by_category[str(score["category"])].append(float(score["score"]))
    for category in expected_categories:
        values = scores_by_category[category]
        threshold = _require_finite_number(thresholds.get(category), name=f"threshold for {category}", minimum=0.0, maximum=2.0)
        report = _require_mapping(categories.get(category), name=f"category report for {category}")
        _require_exact_fields(report, name=f"category report for {category}", required=evaluator.CATEGORY_REPORT_FIELDS)
        fit_original_count = sum(
            entry["category"] == category and entry["partition"] == "FIT" and not entry["isAugmentation"]
            for entry in feature_inputs
        )
        fit_augmented_count = sum(
            entry["category"] == category and entry["partition"] == "FIT" and entry["isAugmentation"]
            for entry in feature_inputs
        )
        expected_counts = {
            "fitOriginalCount": fit_original_count,
            "fitAugmentedCount": fit_augmented_count,
            "tuningOriginalCount": len(values),
        }
        for field, expected_count in expected_counts.items():
            if report.get(field) != expected_count:
                raise FreshNormalSelectionError(f"category report {field} does not match feature membership")
        patch_count = _require_positive_int(report.get("prototypePatchCount"), name=f"category {category} prototypePatchCount", maximum=65_536)
        fit_patch_count = _require_positive_int(report.get("fitPatchCount"), name=f"category {category} fitPatchCount", maximum=10_000_000)
        if patch_count > fit_patch_count:
            raise FreshNormalSelectionError("category prototypePatchCount exceeds fitPatchCount")
        _require_positive_int(report.get("patchGridHeight"), name=f"category {category} patchGridHeight", maximum=1024)
        _require_positive_int(report.get("patchGridWidth"), name=f"category {category} patchGridWidth", maximum=1024)
        median = sorted(values)[len(values) // 2]
        p95 = _p95(values)
        maximum = max(values)
        for field, expected_value in {
            "thresholdFromRawTuning": maximum,
            "tuningScoreMedian": median,
            "tuningScoreP95": p95,
            "tuningScoreMax": maximum,
        }.items():
            actual = _require_finite_number(report.get(field), name=f"category {category} {field}", minimum=0.0, maximum=2.0)
            if not _same_number(expected_value, actual):
                raise FreshNormalSelectionError(f"category report {field} does not match raw tuning scores")
        if not _same_number(threshold, maximum):
            raise FreshNormalSelectionError("development threshold must be the raw tuning maximum")

    evidence = _require_mapping(document.get("normalOnlyEvidence"), name="development normalOnlyEvidence")
    _require_exact_fields(evidence, name="development normalOnlyEvidence", required=evaluator.NORMAL_ONLY_EVIDENCE_FIELDS)
    expected_evidence = {
        "featureInputCount": len(feature_inputs),
        "featureInputPartitions": ["FIT", "THRESHOLD_TUNING"],
        "featureInputKinds": ["NOMINAL"],
        "fitOriginalFeatureInputCount": sum(not entry["isAugmentation"] and entry["partition"] == "FIT" for entry in feature_inputs),
        "fitAugmentedFeatureInputCount": sum(entry["isAugmentation"] and entry["partition"] == "FIT" for entry in feature_inputs),
        "tuningFeatureInputCount": len(calibration_inputs),
        "blindFeatureInputCount": 0,
        "anomalyFeatureInputCount": 0,
        "calibrationInputCount": len(calibration_inputs),
        "calibrationInputPartitions": ["THRESHOLD_TUNING"],
        "calibrationInputKinds": ["NOMINAL"],
    }
    if evidence != expected_evidence:
        raise FreshNormalSelectionError("development normal-only evidence is inconsistent")
    execution = _require_mapping(document.get("execution"), name="development execution")
    _require_exact_fields(execution, name="development execution", required=evaluator.EXECUTION_FIELDS)
    for field in ("evaluatorModuleSha256", "evaluatorEntrypointSha256"):
        _require_sha256(execution.get(field), name=f"development execution {field}")
    timings = _require_mapping(execution.get("phaseTimingsSeconds"), name="development execution phaseTimingsSeconds")
    if set(timings) != {
        "inputAssemblySeconds",
        "provenanceSeconds",
        "inputVerificationSeconds",
        "featureInferenceSeconds",
        "scoringSeconds",
        "totalElapsedSeconds",
    }:
        raise FreshNormalSelectionError("development execution timing fields are unsupported")
    for name, value in timings.items():
        _require_finite_number(value, name=f"development timing {name}", minimum=0.0, maximum=31_536_000.0)
    for field in ("python", "platform", "numpyVersion", "torchVersion"):
        _require_string(execution.get(field), name=f"development execution {field}")
    _require_positive_int(execution.get("torchThreadCount"), name="development execution torchThreadCount", maximum=1_000_000)
    if execution.get("gitRevision") is not None:
        _require_string(execution.get("gitRevision"), name="development execution gitRevision")
    if execution.get("gitWorktreeClean") is not None and not isinstance(execution.get("gitWorktreeClean"), bool):
        raise FreshNormalSelectionError("development execution gitWorktreeClean must be a boolean or null")
    return document


def load_validated_fresh_development_report(
    path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Load one external development report without reading any image byte."""

    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal development report",
        repository_root=repository_root,
    )
    return _validate_development_report_document(document), file_sha256


def _validate_holdout_for_contract(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal holdout manifest",
        repository_root=repository_root,
    )
    try:
        records = holdout._validate_closed_normal_holdout_document(document)
    except holdout.NormalHoldoutError as error:
        raise FreshNormalSelectionError(str(error)) from error
    selection_records = [record for record in records if record["partition"] == "NORMAL_SELECTION"]
    confirmation_records = [record for record in records if record["partition"] == "NORMAL_CONFIRMATION"]
    if not selection_records:
        raise FreshNormalSelectionError("fresh normal holdout has no NORMAL_SELECTION records")
    if not confirmation_records:
        raise FreshNormalSelectionError("fresh normal holdout has no NORMAL_CONFIRMATION records")
    counts: dict[str, int] = {}
    for record in selection_records:
        counts[str(record["category"])] = counts.get(str(record["category"]), 0) + 1
    binding = {
        "manifestFileSha256": file_sha256,
        "manifestDeclaredSha256": document["normalHoldoutManifestSha256"],
        "developmentIdentitySha256": document["developmentIdentitySha256"],
        "normalSelectionIdentitySha256": document["normalSelectionIdentitySha256"],
        "normalConfirmationIdentitySha256": document["normalConfirmationIdentitySha256"],
        "normalSelectionRecordCount": len(selection_records),
        "normalSelectionCategoryCounts": {category: counts[category] for category in sorted(counts)},
    }
    selection_inputs = [_holdout_partition_input_record(record) for record in selection_records]
    confirmation_inputs = [_holdout_partition_input_record(record) for record in confirmation_records]
    return (
        document,
        file_sha256,
        _validate_holdout_binding(binding),
        _validate_normal_partition_inputs(
            selection_inputs,
            partition="NORMAL_SELECTION",
            name="normalSelectionInputs",
        ),
        _validate_normal_partition_inputs(
            confirmation_inputs,
            partition="NORMAL_CONFIRMATION",
            name="normalConfirmationInputs",
        ),
        records,
    )


def _validate_augmentation_for_contract(
    path: Path,
    *,
    holdout_document: dict[str, Any],
    holdout_file_sha256: str,
    holdout_records: list[dict[str, Any]],
    repository_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    document, file_sha256 = _read_external_json(
        path,
        description="fresh FIT augmentation manifest",
        repository_root=repository_root,
    )
    _require_exact_fields(document, name="fresh FIT augmentation manifest", required=augmentation.AUGMENTATION_MANIFEST_FIELDS)
    if document.get("schemaVersion") != augmentation.FRESH_FIT_AUGMENTATION_SCHEMA:
        raise FreshNormalSelectionError("fresh FIT augmentation manifest schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise FreshNormalSelectionError("fresh FIT augmentation manifest must be non-authoritative and non-production")
    if (
        document.get("purpose") != augmentation.FRESH_FIT_AUGMENTATION_PURPOSE
        or document.get("inputPolicy") != augmentation.FRESH_FIT_INPUT_POLICY
        or document.get("blindPolicy") != augmentation.FRESH_FIT_BLIND_POLICY
        or document.get("parentPartition") != "FIT"
    ):
        raise FreshNormalSelectionError("fresh FIT augmentation manifest scope is unsafe")
    if document.get("augmentationManifestSha256") != _document_digest(document, "augmentationManifestSha256"):
        raise FreshNormalSelectionError("fresh FIT augmentation manifest digest does not match")
    bindings = {
        "holdoutManifestFileSha256": holdout_file_sha256,
        "holdoutManifestDeclaredSha256": holdout_document["normalHoldoutManifestSha256"],
        "developmentIdentitySha256": holdout_document["developmentIdentitySha256"],
    }
    for field, expected in bindings.items():
        if document.get(field) != expected:
            raise FreshNormalSelectionError(f"fresh FIT augmentation manifest {field} does not match the holdout")
    for field in (
        "holdoutManifestFileSha256",
        "holdoutManifestDeclaredSha256",
        "developmentIdentitySha256",
        "fitParentIdentitySha256",
        "recipeFileSha256",
        "augmentationManifestSha256",
    ):
        _require_sha256(document.get(field), name=f"fresh FIT augmentation manifest {field}")
    variants = _require_positive_int(document.get("variantsPerParent"), name="fresh FIT augmentation variantsPerParent", maximum=8)
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise FreshNormalSelectionError("fresh FIT augmentation manifest records must be a non-empty list")
    fit_parents = sorted(
        (record for record in holdout_records if record["partition"] == "FIT"),
        key=lambda record: str(record["caseId"]),
    )
    if not fit_parents:
        raise FreshNormalSelectionError("fresh normal holdout has no FIT parents")
    expected_fit_identity = canonical_json_sha256([
        {
            "caseId": parent["caseId"],
            "category": parent["category"],
            "sourceSha256": parent["sourceSha256"],
            "sourceGroupId": parent["sourceGroupId"],
            "partition": parent["partition"],
        }
        for parent in fit_parents
    ])
    if document.get("fitParentIdentitySha256") != expected_fit_identity:
        raise FreshNormalSelectionError("fresh FIT augmentation manifest fitParentIdentitySha256 does not match holdout FIT parents")
    parents_by_case_id = {str(parent["caseId"]): parent for parent in fit_parents}
    expected_pairs = {
        (str(parent["caseId"]), variant_id)
        for parent in fit_parents
        for variant_id in range(1, variants + 1)
    }
    seen_pairs: set[tuple[str, int]] = set()
    case_ids: list[str] = []
    for record in records:
        item = _require_mapping(record, name="fresh FIT augmentation record")
        _require_exact_fields(item, name="fresh FIT augmentation record", required=augmentation.AUGMENTATION_RECORD_FIELDS)
        if item.get("parentPartition") != "FIT" or item.get("kind") != "NOMINAL" or item.get("defect") != "good":
            raise FreshNormalSelectionError("fresh FIT augmentation record scope is unsafe")
        case_ids.append(_require_string(item.get("caseId"), name="fresh FIT augmentation record caseId"))
        _require_sha256(item.get("sourceSha256"), name="fresh FIT augmentation record sourceSha256")
        parent_case_id = _require_string(item.get("parentCaseId"), name="fresh FIT augmentation record parentCaseId")
        parent = parents_by_case_id.get(parent_case_id)
        if parent is None:
            raise FreshNormalSelectionError("fresh FIT augmentation record refers to an unknown FIT parent")
        if (
            item.get("parentSourceSha256") != parent["sourceSha256"]
            or item.get("sourceGroupId") != parent["sourceGroupId"]
            or item.get("category") != parent["category"]
        ):
            raise FreshNormalSelectionError("fresh FIT augmentation record parent binding does not match the holdout")
        variant_id = _require_positive_int(item.get("variantId"), name="fresh FIT augmentation record variantId", maximum=8)
        pair = (parent_case_id, variant_id)
        if pair in seen_pairs:
            raise FreshNormalSelectionError("fresh FIT augmentation record duplicates a parent/variant pair")
        seen_pairs.add(pair)
        expected_case_id, _ = augmentation._expected_child_identity(
            parent,
            recipe_sha256=document["recipeFileSha256"],
            variant_id=variant_id,
        )
        if item.get("caseId") != expected_case_id:
            raise FreshNormalSelectionError("fresh FIT augmentation record caseId is not deterministically bound to its parent")
    if len(case_ids) != len(set(case_ids)) or case_ids != sorted(case_ids):
        raise FreshNormalSelectionError("fresh FIT augmentation records must be unique and sorted")
    if seen_pairs != expected_pairs:
        raise FreshNormalSelectionError("fresh FIT augmentation records do not cover every FIT parent and declared variant")
    binding = {
        "manifestFileSha256": file_sha256,
        "manifestDeclaredSha256": document["augmentationManifestSha256"],
        "developmentIdentitySha256": document["developmentIdentitySha256"],
        "fitParentIdentitySha256": document["fitParentIdentitySha256"],
        "recipeFileSha256": document["recipeFileSha256"],
        "variantsPerParent": variants,
    }
    return document, file_sha256, _validate_augmentation_binding(binding)


def _expected_development_feature_inputs(
    holdout_records: list[dict[str, Any]],
    augmentation_document: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild the evaluator's image-path-free feature membership from JSON."""

    result: list[dict[str, Any]] = []
    for record in holdout_records:
        if record["partition"] not in {"FIT", "THRESHOLD_TUNING"}:
            continue
        result.append({
            "caseId": record["caseId"],
            "category": record["category"],
            "partition": record["partition"],
            "kind": record["kind"],
            "defect": record["defect"],
            "sourceSha256": record["sourceSha256"],
            "isAugmentation": False,
            "variantId": None,
            "parentCaseId": None,
            "parentSourceSha256": None,
            "augmentationManifestSha256": None,
        })
    for record in augmentation_document["records"]:
        result.append({
            "caseId": record["caseId"],
            "category": record["category"],
            "partition": "FIT",
            "kind": record["kind"],
            "defect": record["defect"],
            "sourceSha256": record["sourceSha256"],
            "isAugmentation": True,
            "variantId": record["variantId"],
            "parentCaseId": record["parentCaseId"],
            "parentSourceSha256": record["parentSourceSha256"],
            "augmentationManifestSha256": augmentation_document["augmentationManifestSha256"],
        })
    result.sort(key=lambda item: str(item["caseId"]))
    if len({item["caseId"] for item in result}) != len(result):
        raise FreshNormalSelectionError("frozen development feature membership has duplicate caseIds")
    return result


def _candidate_binding(report: dict[str, Any], *, file_sha256: str) -> dict[str, Any]:
    configuration = dict(report["candidateConfiguration"])
    return {
        "candidateId": configuration["id"],
        "developmentReportFileSha256": file_sha256,
        "developmentReportDeclaredSha256": report["developmentReportSha256"],
        "candidateConfiguration": configuration,
        "candidateConfigurationSha256": report["candidateConfigurationSha256"],
        "featureExtractorIdentitySha256": report["featureExtractorIdentitySha256"],
        "featureInputIdentitySha256": report["featureInputIdentitySha256"],
        "calibrationInputIdentitySha256": report["calibrationInputIdentitySha256"],
        "thresholds": dict(report["thresholds"]),
        "thresholdsIdentitySha256": canonical_json_sha256(report["thresholds"]),
    }


def _validate_candidate_binding(value: object) -> dict[str, Any]:
    document = _require_mapping(value, name="candidate report binding")
    _require_exact_fields(document, name="candidate report binding", required=CANDIDATE_REPORT_BINDING_FIELDS)
    candidate_id = _require_string(document.get("candidateId"), name="candidate report binding candidateId")
    try:
        configuration = evaluator.validate_candidate_configuration(document.get("candidateConfiguration"))
    except evaluator.NormalHoldoutEvaluatorError as error:
        raise FreshNormalSelectionError(str(error)) from error
    if configuration["id"] != candidate_id:
        raise FreshNormalSelectionError("candidate report binding candidateId does not match its configuration")
    if document.get("candidateConfigurationSha256") != canonical_json_sha256(configuration):
        raise FreshNormalSelectionError("candidate report binding configuration digest does not match")
    result = {
        "candidateId": candidate_id,
        "candidateConfiguration": configuration,
        "candidateConfigurationSha256": document["candidateConfigurationSha256"],
    }
    for field in (
        "developmentReportFileSha256",
        "developmentReportDeclaredSha256",
        "featureExtractorIdentitySha256",
        "featureInputIdentitySha256",
        "calibrationInputIdentitySha256",
    ):
        result[field] = _require_sha256(document.get(field), name=f"candidate report binding {field}")
    thresholds = _require_mapping(document.get("thresholds"), name="candidate report binding thresholds")
    if not thresholds or list(thresholds) != sorted(thresholds):
        raise FreshNormalSelectionError("candidate report binding thresholds must be a non-empty sorted mapping")
    parsed_thresholds: dict[str, float] = {}
    for category, value in thresholds.items():
        _require_string(category, name="candidate report binding threshold category")
        parsed_thresholds[category] = _require_finite_number(
            value,
            name=f"candidate report binding threshold for {category}",
            minimum=0.0,
            maximum=2.0,
        )
    result["thresholds"] = parsed_thresholds
    result["thresholdsIdentitySha256"] = _require_sha256(
        document.get("thresholdsIdentitySha256"), name="candidate report binding thresholdsIdentitySha256"
    )
    if result["thresholdsIdentitySha256"] != canonical_json_sha256(parsed_thresholds):
        raise FreshNormalSelectionError("candidate report binding thresholds digest does not match")
    return result


def _validate_contract_document(document: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal selection contract", required=CONTRACT_FIELDS)
    _require_static_scope(
        document,
        schema=FRESH_NORMAL_SELECTION_CONTRACT_SCHEMA,
        purpose=FRESH_NORMAL_SELECTION_CONTRACT_PURPOSE,
        phase=FRESH_NORMAL_SELECTION_CONTRACT_PHASE,
        description="fresh normal selection contract",
    )
    _validate_selection_input(document.get("selectionInput"))
    _validate_decision_policy(document.get("decisionPolicy"))
    holdout_binding = _validate_holdout_binding(document.get("holdout"))
    augmentation_binding = _validate_augmentation_binding(document.get("augmentation"))
    if augmentation_binding["developmentIdentitySha256"] != holdout_binding["developmentIdentitySha256"]:
        raise FreshNormalSelectionError("selection contract augmentation is not bound to the development holdout")
    selection_inputs = _validate_normal_partition_inputs(
        document.get("normalSelectionInputs"),
        partition="NORMAL_SELECTION",
        name="normalSelectionInputs",
    )
    if len(selection_inputs) != holdout_binding["normalSelectionRecordCount"]:
        raise FreshNormalSelectionError("selection contract normalSelectionInputs count does not match the holdout binding")
    if document.get("normalSelectionInputIdentitySha256") != canonical_json_sha256(selection_inputs):
        raise FreshNormalSelectionError("selection contract normal selection input identity does not match")
    confirmation_inputs = _validate_normal_partition_inputs(
        document.get("normalConfirmationInputs"),
        partition="NORMAL_CONFIRMATION",
        name="normalConfirmationInputs",
    )
    if document.get("normalConfirmationInputIdentitySha256") != canonical_json_sha256(confirmation_inputs):
        raise FreshNormalSelectionError("selection contract normal confirmation input identity does not match")
    extractor = _require_mapping(document.get("featureExtractor"), name="selection contract featureExtractor")
    if not extractor:
        raise FreshNormalSelectionError("selection contract featureExtractor must not be empty")
    _validate_json_value(extractor, name="selection contract featureExtractor")
    extractor_sha256 = _require_sha256(
        document.get("featureExtractorIdentitySha256"), name="selection contract featureExtractorIdentitySha256"
    )
    if extractor_sha256 != canonical_json_sha256(extractor):
        raise FreshNormalSelectionError("selection contract feature extractor digest does not match")
    raw_candidates = document.get("candidateReports")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise FreshNormalSelectionError("selection contract candidateReports must be a non-empty list")
    candidates = [_validate_candidate_binding(value) for value in raw_candidates]
    candidate_ids = [str(candidate["candidateId"]) for candidate in candidates]
    if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise FreshNormalSelectionError("selection contract candidateReports must have unique sorted IDs")
    if len({candidate["developmentReportFileSha256"] for candidate in candidates}) != len(candidates):
        raise FreshNormalSelectionError("selection contract cannot reuse a development report file")
    if any(candidate["featureExtractorIdentitySha256"] != extractor_sha256 for candidate in candidates):
        raise FreshNormalSelectionError("selection contract candidate reports use different feature extractors")
    if len({candidate["featureInputIdentitySha256"] for candidate in candidates}) != 1:
        raise FreshNormalSelectionError("selection contract candidate reports use different feature inputs")
    if len({candidate["calibrationInputIdentitySha256"] for candidate in candidates}) != 1:
        raise FreshNormalSelectionError("selection contract candidate reports use different calibration inputs")
    if document.get("candidateUniverseIdentitySha256") != canonical_json_sha256(candidates):
        raise FreshNormalSelectionError("selection contract candidate universe digest does not match")
    _validate_selection_gates(document.get("selectionGates"))
    _validate_selection_objective(document.get("selectionObjective"))
    if document.get("contractSha256") != _document_digest(document, "contractSha256"):
        raise FreshNormalSelectionError("selection contract digest does not match")
    return document


def load_validated_fresh_selection_contract(
    path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Load a self-contained, external selection contract without opening images."""

    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal selection contract",
        repository_root=repository_root,
    )
    return _validate_contract_document(document), file_sha256


def create_fresh_normal_selection_contract(
    holdout_manifest_path: Path,
    augmentation_manifest_path: Path,
    development_report_paths: Sequence[Path],
    output_path: Path,
    *,
    selection_gates: dict[str, Any],
    selection_objective: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Freeze JSON-only development evidence before normal selection is consumed.

    No source root or image path is accepted: this boundary intentionally
    cannot open FIT, tuning, selection, confirmation, reserve, blind, anomaly,
    or mask bytes.
    """

    if isinstance(development_report_paths, (str, bytes)) or not isinstance(development_report_paths, Sequence):
        raise FreshNormalSelectionError("development_report_paths must be a non-string sequence")
    if not development_report_paths:
        raise FreshNormalSelectionError("at least one development report is required")
    (
        holdout_document,
        holdout_file_sha256,
        holdout_binding,
        selection_inputs,
        confirmation_inputs,
        holdout_records,
    ) = _validate_holdout_for_contract(
        holdout_manifest_path,
        repository_root=repository_root,
    )
    augmentation_document, augmentation_file_sha256, augmentation_binding = _validate_augmentation_for_contract(
        augmentation_manifest_path,
        holdout_document=holdout_document,
        holdout_file_sha256=holdout_file_sha256,
        holdout_records=holdout_records,
        repository_root=repository_root,
    )
    expected_feature_inputs = _expected_development_feature_inputs(holdout_records, augmentation_document)
    expected_calibration_inputs = [
        item for item in expected_feature_inputs if item["partition"] == "THRESHOLD_TUNING"
    ]
    reports: list[dict[str, Any]] = []
    for report_path in development_report_paths:
        report, report_file_sha256 = load_validated_fresh_development_report(
            report_path,
            repository_root=repository_root,
        )
        expected = {
            "holdoutManifestFileSha256": holdout_binding["manifestFileSha256"],
            "holdoutManifestDeclaredSha256": holdout_binding["manifestDeclaredSha256"],
            "developmentIdentitySha256": holdout_binding["developmentIdentitySha256"],
            "augmentationManifestFileSha256": augmentation_file_sha256,
            "augmentationManifestDeclaredSha256": augmentation_document["augmentationManifestSha256"],
            "fitParentIdentitySha256": augmentation_binding["fitParentIdentitySha256"],
            "augmentationRecipeFileSha256": augmentation_binding["recipeFileSha256"],
        }
        for field, expected_value in expected.items():
            if report.get(field) != expected_value:
                raise FreshNormalSelectionError(f"development report {field} does not match frozen selection inputs")
        if report.get("featureInputs") != expected_feature_inputs:
            raise FreshNormalSelectionError("development report feature inputs do not match the frozen FIT/tuning package")
        if report.get("calibrationInputs") != expected_calibration_inputs:
            raise FreshNormalSelectionError("development report calibration inputs do not match raw threshold tuning")
        reports.append(_candidate_binding(report, file_sha256=report_file_sha256))
    reports.sort(key=lambda candidate: str(candidate["candidateId"]))
    if len({candidate["candidateId"] for candidate in reports}) != len(reports):
        raise FreshNormalSelectionError("development reports contain duplicate candidate IDs")
    if len({candidate["developmentReportFileSha256"] for candidate in reports}) != len(reports):
        raise FreshNormalSelectionError("development reports reuse the same external file")
    first_report, _ = load_validated_fresh_development_report(development_report_paths[0], repository_root=repository_root)
    extractor = first_report["featureExtractor"]
    extractor_sha256 = first_report["featureExtractorIdentitySha256"]
    if any(candidate["featureExtractorIdentitySha256"] != extractor_sha256 for candidate in reports):
        raise FreshNormalSelectionError("development reports do not share one feature extractor identity")
    feature_input_identities = {candidate["featureInputIdentitySha256"] for candidate in reports}
    calibration_input_identities = {candidate["calibrationInputIdentitySha256"] for candidate in reports}
    if len(feature_input_identities) != 1 or len(calibration_input_identities) != 1:
        raise FreshNormalSelectionError("development reports do not share one frozen development input set")
    gates = _validate_selection_gates(selection_gates)
    objective = _validate_selection_objective(selection_objective)
    contract: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_SELECTION_CONTRACT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_SELECTION_CONTRACT_PURPOSE,
        "phase": FRESH_NORMAL_SELECTION_CONTRACT_PHASE,
        "blindPolicy": FRESH_NORMAL_SELECTION_BLIND_POLICY,
        "selectionInput": dict(FRESH_NORMAL_SELECTION_INPUT),
        "decisionPolicy": dict(FRESH_NORMAL_SELECTION_DECISION_POLICY),
        "holdout": holdout_binding,
        "augmentation": augmentation_binding,
        "normalSelectionInputs": selection_inputs,
        "normalSelectionInputIdentitySha256": canonical_json_sha256(selection_inputs),
        "normalConfirmationInputs": confirmation_inputs,
        "normalConfirmationInputIdentitySha256": canonical_json_sha256(confirmation_inputs),
        "featureExtractor": extractor,
        "featureExtractorIdentitySha256": extractor_sha256,
        "candidateReports": reports,
        "candidateUniverseIdentitySha256": canonical_json_sha256(reports),
        "selectionGates": gates,
        "selectionObjective": objective,
    }
    contract["contractSha256"] = _document_digest(contract, "contractSha256")
    _validate_contract_document(contract)
    _write_external_json(
        output_path,
        contract,
        description="fresh normal selection contract",
        repository_root=repository_root,
    )
    return contract


def fresh_selection_claim_path(contract_path: Path) -> Path:
    """Return the sole claim slot associated with an immutable contract path."""

    if not isinstance(contract_path, Path):
        raise FreshNormalSelectionError("contract path must be a Path")
    return contract_path.parent / FRESH_NORMAL_SELECTION_CLAIM_FILENAME


def _validate_claim_document(document: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(document, name="fresh normal selection claim", required=CLAIM_FIELDS)
    _require_static_scope(
        document,
        schema=FRESH_NORMAL_SELECTION_CLAIM_SCHEMA,
        purpose=FRESH_NORMAL_SELECTION_CLAIM_PURPOSE,
        phase=FRESH_NORMAL_SELECTION_CLAIM_PHASE,
        description="fresh normal selection claim",
    )
    _validate_selection_input(document.get("selectionInput"))
    if document.get("claimSlot") != FRESH_NORMAL_SELECTION_CLAIM_SLOT:
        raise FreshNormalSelectionError("fresh normal selection claim uses an unsupported slot")
    _require_sha256(document.get("contractFileSha256"), name="selection claim contractFileSha256")
    _require_sha256(document.get("contractDeclaredSha256"), name="selection claim contractDeclaredSha256")
    _validate_holdout_binding(document.get("holdout"))
    _validate_augmentation_binding(document.get("augmentation"))
    _require_sha256(
        document.get("featureExtractorIdentitySha256"), name="selection claim featureExtractorIdentitySha256"
    )
    _require_sha256(
        document.get("candidateUniverseIdentitySha256"), name="selection claim candidateUniverseIdentitySha256"
    )
    if document.get("claimSha256") != _document_digest(document, "claimSha256"):
        raise FreshNormalSelectionError("selection claim digest does not match")
    return document


def load_validated_fresh_selection_claim(
    path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Load a self-contained selection claim without opening a selection image."""

    document, file_sha256 = _read_external_json(
        path,
        description="fresh normal selection claim",
        repository_root=repository_root,
    )
    return _validate_claim_document(document), file_sha256


def validate_fresh_selection_claim_binding(
    contract: dict[str, Any],
    contract_file_sha256: str,
    claim: dict[str, Any],
    *,
    contract_path: Path | None = None,
    claim_path: Path | None = None,
) -> None:
    """Require a claim to bind exactly one validated contract and its fixed slot."""

    _validate_contract_document(contract)
    _validate_claim_document(claim)
    _require_sha256(contract_file_sha256, name="contract file digest")
    expected = {
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "candidateUniverseIdentitySha256": contract["candidateUniverseIdentitySha256"],
    }
    for field, value in expected.items():
        if claim.get(field) != value:
            raise FreshNormalSelectionError(f"selection claim {field} does not bind the selection contract")
    if (contract_path is None) != (claim_path is None):
        raise FreshNormalSelectionError("contract_path and claim_path must be supplied together")
    if contract_path is not None and claim_path is not None:
        if claim_path.resolve() != fresh_selection_claim_path(contract_path).resolve():
            raise FreshNormalSelectionError("selection claim is not stored in the contract's fixed claim slot")


def create_fresh_normal_selection_claim(
    contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Atomically consume the fixed selection-claim slot for a valid contract.

    This does not observe selection data.  A later observer must validate this
    claim and write its own receipt before opening raw ``NORMAL_SELECTION``.
    """

    contract, contract_file_sha256 = load_validated_fresh_selection_contract(
        contract_path,
        repository_root=repository_root,
    )
    claim_path = fresh_selection_claim_path(contract_path)
    claim: dict[str, Any] = {
        "schemaVersion": FRESH_NORMAL_SELECTION_CLAIM_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_NORMAL_SELECTION_CLAIM_PURPOSE,
        "phase": FRESH_NORMAL_SELECTION_CLAIM_PHASE,
        "blindPolicy": FRESH_NORMAL_SELECTION_BLIND_POLICY,
        "selectionInput": dict(FRESH_NORMAL_SELECTION_INPUT),
        "claimSlot": FRESH_NORMAL_SELECTION_CLAIM_SLOT,
        "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "holdout": contract["holdout"],
        "augmentation": contract["augmentation"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "candidateUniverseIdentitySha256": contract["candidateUniverseIdentitySha256"],
    }
    claim["claimSha256"] = _document_digest(claim, "claimSha256")
    _validate_claim_document(claim)
    _write_external_json(
        claim_path,
        claim,
        description="fresh normal selection claim",
        repository_root=repository_root,
    )
    return claim


def load_validated_fresh_selection_claim_for_contract(
    contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Load the unique sibling claim and verify its contract/slot binding."""

    contract, contract_file_sha256 = load_validated_fresh_selection_contract(
        contract_path,
        repository_root=repository_root,
    )
    claim_path = fresh_selection_claim_path(contract_path)
    claim, claim_file_sha256 = load_validated_fresh_selection_claim(
        claim_path,
        repository_root=repository_root,
    )
    validate_fresh_selection_claim_binding(
        contract,
        contract_file_sha256,
        claim,
        contract_path=contract_path,
        claim_path=claim_path,
    )
    return claim, claim_file_sha256
