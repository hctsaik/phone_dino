"""External-only normal-source and holdout manifests for MVTec research.

This module deliberately does not import the legacy smoke-manifest runner or
its augmentation packages.  It builds a fresh normal-only source pool and a
deterministic parent-level holdout partition from sources that have not
appeared in a historical normal-usage ledger.  It is offline research tooling,
not a PhoneDINO runtime, production, or physical-qualification component.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NORMAL_SOURCE_CANDIDATES_SCHEMA = "phone-dino.mvtec-ad-normal-source-candidates/1.0"
NORMAL_SOURCE_POOL_SCHEMA = "phone-dino.mvtec-ad-normal-source-pool/1.0"
HISTORICAL_NORMAL_USAGE_LEDGER_SCHEMA = "phone-dino.mvtec-ad-historical-normal-usage-ledger/1.0"
NORMAL_HOLDOUT_PLAN_SCHEMA = "phone-dino.mvtec-ad-normal-holdout-plan/1.0"
NORMAL_HOLDOUT_SCHEMA = "phone-dino.mvtec-ad-normal-holdout/1.0"
SOURCE_CANDIDATES_PURPOSE = "OFFLINE_MVTEC_NORMAL_SOURCE_POOL_CANDIDATES"
SOURCE_POOL_PURPOSE = "OFFLINE_MVTEC_NORMAL_SOURCE_POOL"
HISTORICAL_LEDGER_PURPOSE = "OFFLINE_MVTEC_HISTORICAL_NORMAL_USAGE_EXCLUSION"
HOLDOUT_PURPOSE = "OFFLINE_MVTEC_NORMAL_HOLDOUT_CONFIGURATION_SELECTION"
HOLDOUT_BLIND_POLICY = "NO_BLIND_OR_ANOMALY_DATA"
HISTORICAL_LEDGER_SCOPE = "ALL_HISTORICAL_NORMAL_FEATURE_INPUT_SOURCE_SHA256"
PARTITION_ALGORITHM = "SHA256_RANKED_ACQUISITION_GROUP_EXACT_QUOTA_V1"
HISTORY_EXCLUSION_ALGORITHM = "EXCLUDE_ENTIRE_SOURCE_GROUP_ON_HISTORICAL_SOURCE_SHA256_MATCH_V1"
ITERATION_REPORT_SCHEMA = "phone-dino.mvtec-ad-iteration-report/1.4"
ITERATION_REPORT_PURPOSE = "NORMAL_ONLY_ITERATION_THEN_BLIND_REPORTING_ONLY"
PINNED_MVTEC_DATASET_ID = "MVTec AD"
PINNED_MVTEC_OFFICIAL_SOURCE_URI = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
PINNED_MVTEC_MIRROR_SOURCE_URI = "https://huggingface.co/datasets/Voxel51/mvtec-ad"
PINNED_MVTEC_MIRROR_REVISION = "30a183a3b96e3aef953f230784b123b719b09d97"
PINNED_MVTEC_SAMPLES_SHA256 = "sha256:dbbbb94cee2ddec28c1eef318733d07df4d59b9cc066e62e6aeef386c1db281d"
PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1 = "d5bc5b023effedf41acf0a02832f30fa76fe0709"
PINNED_MVTEC_LICENSE_NOTICE = "CC BY-NC-SA 4.0"
HOLDOUT_PARTITIONS = (
    "FIT",
    "THRESHOLD_TUNING",
    "NORMAL_SELECTION",
    "NORMAL_CONFIRMATION",
    "RESERVE_UNTOUCHED",
)
ASSIGNABLE_HOLDOUT_PARTITIONS = HOLDOUT_PARTITIONS[:-1]
PLAN_QUOTA_FIELDS = {
    "category",
    "fitGroupCount",
    "thresholdTuningGroupCount",
    "normalSelectionGroupCount",
    "normalConfirmationGroupCount",
    "reserveUntouchedGroupCount",
}
HOLDOUT_RECORD_FIELDS = {
    "caseId",
    "category",
    "relativePath",
    "sourceSha256",
    "sourceGroupId",
    "acquisitionStratum",
    "sourceRemotePath",
    "expectedRemoteSha256",
    "expectedRemoteBytes",
    "kind",
    "defect",
    "partition",
}
HISTORY_EXCLUSION_FIELDS = {
    "algorithm",
    "matchedHistoricalSourceCount",
    "excludedSourceGroupCount",
    "eligibleSourceCount",
    "eligibleSourceIdentitySha256",
}
NORMAL_HOLDOUT_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "blindPolicy",
    "sourcePoolFileSha256",
    "sourcePoolDeclaredSha256",
    "historicalLedgerFileSha256",
    "historicalLedgerDeclaredSha256",
    "planFileSha256",
    "planDeclaredSha256",
    "historyExclusion",
    "records",
    "developmentIdentitySha256",
    "normalSelectionIdentitySha256",
    "normalConfirmationIdentitySha256",
    "reserveUntouchedIdentitySha256",
    "normalHoldoutManifestSha256",
}
FEATURE_INPUT_FIELDS = {
    "caseId",
    "category",
    "role",
    "kind",
    "sourceSha256",
    "isAugmentation",
    "variantId",
    "parentCaseId",
    "parentSourceSha256",
    "augmentationRecipeSha256",
}
NORMAL_ONLY_EVIDENCE_FIELDS = {
    "featureInputCount",
    "featureInputRoles",
    "featureInputKinds",
    "blindFeatureInputCount",
    "anomalyFeatureInputCount",
    "normalInputRecordCount",
    "featureInputs",
    "featureInputIdentitySha256",
    "normalInputIdentitySha256",
    "reportedScoreCount",
    "reportedScoreRoles",
    "reportedScoreKinds",
    "calibrationScoreCount",
    "calibrationScoreRoles",
    "calibrationScoreKinds",
    "calibrationInputs",
    "calibrationInputIdentitySha256",
    "originalTuningInputCount",
    "originalTuningInputs",
    "originalTuningInputIdentitySha256",
}
ITERATION_REPORT_FIELDS = {
    "algorithm",
    "augmentation",
    "authoritative",
    "blindReporting",
    "calibrationScores",
    "candidateConfiguration",
    "candidateConfigurationSha256",
    "categories",
    "disclaimer",
    "execution",
    "featureExtractor",
    "featureExtractorIdentitySha256",
    "inputManifest",
    "inputManifestDeclaredSha256",
    "inputManifestFileSha256",
    "normalOnlyEvidence",
    "pixelLocalization",
    "productionAuthorized",
    "schemaVersion",
    "scores",
    "selectionProtocol",
}
BLIND_REPORTING_FIELDS = {"state", "blindSourcePolicy", "reason"}
NORMAL_SCORE_BASE_FIELDS = {
    "caseId",
    "category",
    "defect",
    "isAugmentation",
    "kind",
    "role",
    "score",
    "sourceSha256",
    "variantId",
}
NORMAL_SCORE_AUGMENTATION_FIELDS = {"parentCaseId", "parentSourceSha256", "augmentationRecipeSha256"}


class NormalHoldoutError(ValueError):
    """Raised when an external normal-holdout artifact violates its protocol."""


def canonical_json_sha256(document: Any) -> str:
    encoded = json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require_exact_fields(document: dict[str, Any], *, name: str, required: set[str]) -> None:
    missing = required.difference(document)
    if missing:
        raise NormalHoldoutError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    unknown = set(document).difference(required)
    if unknown:
        raise NormalHoldoutError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")


def _require_string(document: dict[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value.strip():
        raise NormalHoldoutError(f"{name} must be a non-empty string")
    return value


def _require_sha256(document: dict[str, Any], name: str) -> str:
    value = _require_string(document, name)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise NormalHoldoutError(f"{name} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise NormalHoldoutError(f"{name} must be a sha256 digest") from error
    return value


def _require_nonnegative_int(document: dict[str, Any], name: str) -> int:
    value = document.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise NormalHoldoutError(f"{name} must be a non-negative integer")
    return value


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return canonical_json_sha256(unsigned)


def _is_under(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
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
                raise NormalHoldoutError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _safe_relative_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise NormalHoldoutError(f"{name} must be a non-empty safe relative path")
    if "\x00" in value:
        raise NormalHoldoutError(f"{name} must be a safe relative path")
    windows = PureWindowsPath(value)
    path = Path(value)
    if windows.is_absolute() or windows.drive or windows.root or path.is_absolute() or path.root or not path.parts:
        raise NormalHoldoutError(f"{name} must be a safe relative path")
    if any(
        part in {"", ".", ".."}
        or ":" in part
        or part != part.rstrip(" .")
        or PureWindowsPath(part).is_reserved()
        for part in path.parts
    ):
        raise NormalHoldoutError(f"{name} must be a safe relative path")
    return path


def _safe_file_under(root: Path, relative: Path, *, description: str) -> Path:
    _reject_links_on_existing_path(root, description=description)
    candidate = root.joinpath(*relative.parts)
    _reject_links_on_existing_path(candidate, description=description)
    if not candidate.is_file() or not _is_under(root, candidate) or _is_under(REPOSITORY_ROOT, candidate):
        raise NormalHoldoutError(f"{description} is missing or escapes its external root")
    return candidate


def _require_decodable_image(path: Path, *, description: str) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise NormalHoldoutError(f"{description} is not a decodable image") from error


def _require_external_input_file(path: Path, *, description: str) -> None:
    if _is_under(REPOSITORY_ROOT, path):
        raise NormalHoldoutError(f"{description} must stay outside the Git working tree")
    _reject_links_on_existing_path(path, description=description)
    if not path.is_file():
        raise NormalHoldoutError(f"{description} is missing")


def _require_external_source_root(path: Path) -> None:
    if _is_under(REPOSITORY_ROOT, path) or _is_under(path, REPOSITORY_ROOT):
        raise NormalHoldoutError("normal source root must stay outside the Git working tree")
    _reject_links_on_existing_path(path, description="normal source root")
    if not path.is_dir():
        raise NormalHoldoutError("normal source root is missing")


def _prepare_external_output(path: Path, *, description: str) -> None:
    if _is_under(REPOSITORY_ROOT, path):
        raise NormalHoldoutError(f"{description} must stay outside the Git working tree")
    if path.exists():
        raise NormalHoldoutError(f"{description} already exists; choose a fresh immutable path")
    _reject_links_on_existing_path(path.parent, description=description)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_links_on_existing_path(path.parent, description=description)


def _parse_json_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    try:
        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise NormalHoldoutError(f"{description} contains a duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_nonfinite(value: str) -> Any:
            raise NormalHoldoutError(f"{description} contains a non-finite JSON value: {value}")

        document = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NormalHoldoutError(f"unable to read {description}") from error
    if not isinstance(document, dict):
        raise NormalHoldoutError(f"{description} must be a JSON object")
    return document


def _read_json_bytes(path: Path, *, description: str) -> tuple[dict[str, Any], str, bytes]:
    _require_external_input_file(path, description=description)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise NormalHoldoutError(f"unable to read {description}") from error
    document = _parse_json_bytes(raw, description=description)
    return document, f"sha256:{hashlib.sha256(raw).hexdigest()}", raw


def _read_json(path: Path, *, description: str) -> tuple[dict[str, Any], str]:
    document, file_sha256, _ = _read_json_bytes(path, description=description)
    return document, file_sha256


def _write_json(path: Path, document: dict[str, Any], *, description: str) -> None:
    _prepare_external_output(path, description=description)
    payload = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)


def _validate_origin(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NormalHoldoutError("origin must be an object")
    _require_exact_fields(
        value,
        name="origin",
        required={
            "datasetId", "officialSourceUri", "mirrorSourceUri", "mirrorRevision",
            "sourceMetadataFileSha256", "sourceMetadataGitBlobSha1", "sourceCriterion", "licenseNotice",
            "priorSubsetSourceIdentitySha256",
        },
    )
    expected = {
        "datasetId": PINNED_MVTEC_DATASET_ID,
        "officialSourceUri": PINNED_MVTEC_OFFICIAL_SOURCE_URI,
        "mirrorSourceUri": PINNED_MVTEC_MIRROR_SOURCE_URI,
        "mirrorRevision": PINNED_MVTEC_MIRROR_REVISION,
        "sourceMetadataFileSha256": PINNED_MVTEC_SAMPLES_SHA256,
        "sourceMetadataGitBlobSha1": PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1,
        "licenseNotice": PINNED_MVTEC_LICENSE_NOTICE,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise NormalHoldoutError(f"origin.{name} does not match the pinned MVTec source")
    if len(str(value["sourceMetadataGitBlobSha1"])) != 40:
        raise NormalHoldoutError("sourceMetadataGitBlobSha1 must be a SHA-1 digest")
    try:
        int(str(value["sourceMetadataGitBlobSha1"]), 16)
    except ValueError as error:
        raise NormalHoldoutError("sourceMetadataGitBlobSha1 must be a SHA-1 digest") from error
    _require_sha256(value, "priorSubsetSourceIdentitySha256")
    if value.get("sourceCriterion") != "OFFICIAL_TRAIN_GOOD_ONLY":
        raise NormalHoldoutError("origin.sourceCriterion must be OFFICIAL_TRAIN_GOOD_ONLY")
    return dict(value)


def _load_pinned_source_metadata(path: Path, *, origin: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read the pinned public inventory solely to attest selected train/good paths."""

    document, file_sha256, raw = _read_json_bytes(path, description="pinned MVTec source metadata")
    if file_sha256 != origin["sourceMetadataFileSha256"]:
        raise NormalHoldoutError("pinned MVTec source metadata digest does not match origin")
    blob_prefix = f"blob {len(raw)}\0".encode("ascii")
    if hashlib.sha1(blob_prefix + raw).hexdigest() != origin["sourceMetadataGitBlobSha1"]:
        raise NormalHoldoutError("pinned MVTec source metadata Git blob digest does not match origin")
    _require_exact_fields(document, name="pinned MVTec source metadata", required={"samples"})
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise NormalHoldoutError("pinned MVTec source metadata has no samples")
    inventory: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise NormalHoldoutError("pinned MVTec source metadata contains a non-object sample")
        remote_path = _safe_relative_path(sample.get("filepath"), name="metadata filepath").as_posix()
        if remote_path in inventory:
            raise NormalHoldoutError("pinned MVTec source metadata has a duplicate filepath")
        category = sample.get("category")
        defect = sample.get("defect")
        if not isinstance(category, dict) or not isinstance(defect, dict):
            raise NormalHoldoutError("pinned MVTec source metadata sample lacks labels")
        category_label = category.get("label")
        defect_label = defect.get("label")
        split = sample.get("split")
        if not isinstance(category_label, str) or not isinstance(defect_label, str) or not isinstance(split, str):
            raise NormalHoldoutError("pinned MVTec source metadata sample labels are invalid")
        inventory[remote_path] = {
            "category": category_label,
            "defect": defect_label,
            "split": split,
            "hasMask": "defect_mask" in sample,
        }
    return inventory


def _validate_record_metadata(record: dict[str, Any], *, inventory: dict[str, dict[str, Any]]) -> None:
    metadata = inventory.get(record["sourceRemotePath"])
    if metadata is None:
        raise NormalHoldoutError("normal source candidate remote path is absent from pinned source metadata")
    if (
        metadata["category"] != record["category"]
        or metadata["split"] != "train"
        or metadata["defect"] != "good"
        or metadata["hasMask"]
    ):
        raise NormalHoldoutError("normal source candidate is not a pinned train/good image without a mask")


def _validate_candidate_record(
    value: object,
    *,
    root: Path,
    grouping_strength: str,
    seen_case_ids: set[str],
    seen_paths: set[str],
    seen_remote_paths: set[str],
    seen_sources: set[str],
    groups: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NormalHoldoutError("normal source candidate record must be an object")
    _require_exact_fields(
        value,
        name="normal source candidate record",
        required={
            "caseId", "category", "relativePath", "sourceGroupId", "acquisitionStratum", "sourceRemotePath",
            "expectedRemoteSha256", "expectedRemoteBytes",
        },
    )
    case_id = _require_string(value, "caseId")
    if case_id in seen_case_ids:
        raise NormalHoldoutError("normal source candidate caseId is duplicated")
    seen_case_ids.add(case_id)
    category = _require_string(value, "category")
    source_group_id = _require_string(value, "sourceGroupId")
    acquisition_stratum = _require_string(value, "acquisitionStratum")
    if acquisition_stratum != "OFFICIAL_MVTEC_TRAIN_GOOD":
        raise NormalHoldoutError("normal source candidate acquisitionStratum is unsupported")
    existing_group = groups.setdefault(source_group_id, (category, acquisition_stratum))
    if existing_group != (category, acquisition_stratum):
        raise NormalHoldoutError("normal source candidate sourceGroupId crosses category or acquisition stratum")
    relative = _safe_relative_path(value.get("relativePath"), name="relativePath")
    if relative.as_posix() in seen_paths:
        raise NormalHoldoutError("normal source candidate relativePath is duplicated")
    seen_paths.add(relative.as_posix())
    source_path = _safe_file_under(root, relative, description="normal source candidate image")
    _require_decodable_image(source_path, description="normal source candidate image")
    source_remote_path = _safe_relative_path(value.get("sourceRemotePath"), name="sourceRemotePath").as_posix()
    if source_remote_path in seen_remote_paths:
        raise NormalHoldoutError("normal source candidate sourceRemotePath is duplicated")
    seen_remote_paths.add(source_remote_path)
    expected_remote_sha256 = _require_sha256(value, "expectedRemoteSha256")
    expected_remote_bytes = _require_nonnegative_int(value, "expectedRemoteBytes")
    if expected_remote_bytes == 0:
        raise NormalHoldoutError("expectedRemoteBytes must be positive")
    source_sha256 = sha256_file(source_path)
    if source_sha256 != expected_remote_sha256 or source_path.stat().st_size != expected_remote_bytes:
        raise NormalHoldoutError("normal source candidate bytes do not match pinned remote identity")
    if source_sha256 in seen_sources:
        raise NormalHoldoutError("normal source candidate sourceSha256 is duplicated")
    seen_sources.add(source_sha256)
    if grouping_strength == "EXACT_CONTENT_ONLY" and source_group_id != f"CONTENT_SHA256:{source_sha256[7:]}":
        raise NormalHoldoutError("EXACT_CONTENT_ONLY candidate group must bind the exact source digest")
    if case_id != f"mvtec-ad/{category}/train-good/{source_sha256[7:]}":
        raise NormalHoldoutError("normal source candidate caseId must bind category and exact source digest")
    return {
        "caseId": case_id,
        "category": category,
        "relativePath": relative.as_posix(),
        "sourceGroupId": source_group_id,
        "acquisitionStratum": acquisition_stratum,
        "sourceRemotePath": source_remote_path,
        "expectedRemoteSha256": expected_remote_sha256,
        "expectedRemoteBytes": expected_remote_bytes,
        "sourceSha256": source_sha256,
    }


def load_normal_source_candidates(
    path: Path, *, source_root: Path, source_metadata_path: Path
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Validate a closed normal-only candidate list and rehash every image."""

    document, file_sha256 = _read_json(path, description="normal source candidates")
    _require_exact_fields(
        document,
        name="normal source candidates",
        required={
            "schemaVersion", "authoritative", "productionAuthorized", "purpose", "origin", "groupingStrength",
            "records", "candidateManifestSha256",
        },
    )
    if document.get("schemaVersion") != NORMAL_SOURCE_CANDIDATES_SCHEMA:
        raise NormalHoldoutError("unsupported normal source candidates schema")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise NormalHoldoutError("normal source candidates must be non-authoritative and non-production")
    if document.get("purpose") != SOURCE_CANDIDATES_PURPOSE:
        raise NormalHoldoutError("normal source candidates have an unsafe purpose")
    if document.get("candidateManifestSha256") != _document_digest(document, "candidateManifestSha256"):
        raise NormalHoldoutError("normal source candidates digest does not match")
    if document.get("groupingStrength") != "EXACT_CONTENT_ONLY":
        raise NormalHoldoutError("MVTec source candidates must use EXACT_CONTENT_ONLY grouping")
    origin = _validate_origin(document.get("origin"))
    inventory = _load_pinned_source_metadata(source_metadata_path, origin=origin)
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise NormalHoldoutError("normal source candidates have no records")
    _require_external_source_root(source_root)
    root = source_root
    seen_case_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_remote_paths: set[str] = set()
    seen_sources: set[str] = set()
    groups: dict[str, tuple[str, str]] = {}
    records = [
        _validate_candidate_record(
            value,
            root=root,
            grouping_strength=str(document["groupingStrength"]),
            seen_case_ids=seen_case_ids,
            seen_paths=seen_paths,
            seen_remote_paths=seen_remote_paths,
            seen_sources=seen_sources,
            groups=groups,
        )
        for value in raw_records
    ]
    if [record["caseId"] for record in records] != sorted(record["caseId"] for record in records):
        raise NormalHoldoutError("normal source candidate records must be sorted by caseId")
    for record in records:
        _validate_record_metadata(record, inventory=inventory)
    return document, file_sha256, records


def _normal_source_pool_identity(records: list[dict[str, Any]], origin: dict[str, Any], grouping_strength: str) -> str:
    return canonical_json_sha256({
        "origin": origin,
        "groupingStrength": grouping_strength,
        "records": records,
    })


def freeze_normal_source_pool(
    candidate_path: Path, output_path: Path, *, source_root: Path, source_metadata_path: Path
) -> dict[str, Any]:
    """Freeze a rehashed, normal-only image pool for a fresh research cohort."""

    candidates, candidate_file_sha256, records = load_normal_source_candidates(
        candidate_path,
        source_root=source_root,
        source_metadata_path=source_metadata_path,
    )
    origin = _validate_origin(candidates.get("origin"))
    grouping_strength = _require_string(candidates, "groupingStrength")
    pool_records = [
        {
            "caseId": record["caseId"],
            "category": record["category"],
            "relativePath": record["relativePath"],
            "sourceSha256": record["sourceSha256"],
            "sourceGroupId": record["sourceGroupId"],
            "acquisitionStratum": record["acquisitionStratum"],
            "sourceRemotePath": record["sourceRemotePath"],
            "expectedRemoteSha256": record["expectedRemoteSha256"],
            "expectedRemoteBytes": record["expectedRemoteBytes"],
            "kind": "NOMINAL",
            "defect": "good",
        }
        for record in records
    ]
    document: dict[str, Any] = {
        "schemaVersion": NORMAL_SOURCE_POOL_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": SOURCE_POOL_PURPOSE,
        "candidateManifestFileSha256": candidate_file_sha256,
        "candidateManifestDeclaredSha256": candidates["candidateManifestSha256"],
        "origin": origin,
        "groupingStrength": grouping_strength,
        "records": pool_records,
        "normalSourcePoolIdentitySha256": _normal_source_pool_identity(pool_records, origin, grouping_strength),
    }
    document["normalSourcePoolSha256"] = _document_digest(document, "normalSourcePoolSha256")
    _write_json(output_path, document, description="normal source pool output")
    return document


def _validate_pool_record(
    value: object,
    *,
    root: Path,
    grouping_strength: str,
    seen_case_ids: set[str],
    seen_paths: set[str],
    seen_remote_paths: set[str],
    seen_sources: set[str],
    groups: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NormalHoldoutError("normal source pool record must be an object")
    _require_exact_fields(
        value,
        name="normal source pool record",
        required={
            "caseId", "category", "relativePath", "sourceSha256", "sourceGroupId", "acquisitionStratum",
            "sourceRemotePath", "expectedRemoteSha256", "expectedRemoteBytes", "kind", "defect",
        },
    )
    case_id = _require_string(value, "caseId")
    if case_id in seen_case_ids:
        raise NormalHoldoutError("normal source pool caseId is duplicated")
    seen_case_ids.add(case_id)
    source_sha256 = _require_sha256(value, "sourceSha256")
    if source_sha256 in seen_sources:
        raise NormalHoldoutError("normal source pool sourceSha256 is duplicated")
    seen_sources.add(source_sha256)
    category = _require_string(value, "category")
    group = _require_string(value, "sourceGroupId")
    acquisition_stratum = _require_string(value, "acquisitionStratum")
    if acquisition_stratum != "OFFICIAL_MVTEC_TRAIN_GOOD":
        raise NormalHoldoutError("normal source pool acquisitionStratum is unsupported")
    existing_group = groups.setdefault(group, (category, acquisition_stratum))
    if existing_group != (category, acquisition_stratum):
        raise NormalHoldoutError("normal source pool sourceGroupId crosses category or acquisition stratum")
    if value.get("kind") != "NOMINAL" or value.get("defect") != "good":
        raise NormalHoldoutError("normal source pool records must be good nominal only")
    relative = _safe_relative_path(value.get("relativePath"), name="relativePath")
    if relative.as_posix() in seen_paths:
        raise NormalHoldoutError("normal source pool relativePath is duplicated")
    seen_paths.add(relative.as_posix())
    source_path = _safe_file_under(root, relative, description="normal source pool image")
    _require_decodable_image(source_path, description="normal source pool image")
    source_remote_path = _safe_relative_path(value.get("sourceRemotePath"), name="sourceRemotePath").as_posix()
    if source_remote_path in seen_remote_paths:
        raise NormalHoldoutError("normal source pool sourceRemotePath is duplicated")
    seen_remote_paths.add(source_remote_path)
    expected_remote_sha256 = _require_sha256(value, "expectedRemoteSha256")
    expected_remote_bytes = _require_nonnegative_int(value, "expectedRemoteBytes")
    if expected_remote_bytes == 0:
        raise NormalHoldoutError("expectedRemoteBytes must be positive")
    if source_sha256 != expected_remote_sha256:
        raise NormalHoldoutError("normal source pool source digest does not match pinned remote identity")
    if sha256_file(source_path) != source_sha256 or source_path.stat().st_size != expected_remote_bytes:
        raise NormalHoldoutError("normal source pool image digest does not match")
    if grouping_strength == "EXACT_CONTENT_ONLY" and group != f"CONTENT_SHA256:{source_sha256[7:]}":
        raise NormalHoldoutError("EXACT_CONTENT_ONLY pool group must bind the exact source digest")
    if case_id != f"mvtec-ad/{category}/train-good/{source_sha256[7:]}":
        raise NormalHoldoutError("normal source pool caseId must bind category and exact source digest")
    return {
        "caseId": case_id,
        "category": category,
        "relativePath": relative.as_posix(),
        "sourceSha256": source_sha256,
        "sourceGroupId": group,
        "acquisitionStratum": acquisition_stratum,
        "sourceRemotePath": source_remote_path,
        "expectedRemoteSha256": expected_remote_sha256,
        "expectedRemoteBytes": expected_remote_bytes,
        "kind": "NOMINAL",
        "defect": "good",
    }


def load_normal_source_pool(
    path: Path, *, source_root: Path, source_metadata_path: Path
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Load a source pool, checking its closed schema and every source byte."""

    document, file_sha256 = _read_json(path, description="normal source pool")
    _require_exact_fields(
        document,
        name="normal source pool",
        required={
            "schemaVersion", "authoritative", "productionAuthorized", "purpose", "candidateManifestFileSha256",
            "candidateManifestDeclaredSha256", "origin", "groupingStrength", "records",
            "normalSourcePoolIdentitySha256", "normalSourcePoolSha256",
        },
    )
    if document.get("schemaVersion") != NORMAL_SOURCE_POOL_SCHEMA:
        raise NormalHoldoutError("unsupported normal source pool schema")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise NormalHoldoutError("normal source pool must be non-authoritative and non-production")
    if document.get("purpose") != SOURCE_POOL_PURPOSE:
        raise NormalHoldoutError("normal source pool has an unsafe purpose")
    if document.get("normalSourcePoolSha256") != _document_digest(document, "normalSourcePoolSha256"):
        raise NormalHoldoutError("normal source pool digest does not match")
    _require_sha256(document, "candidateManifestFileSha256")
    _require_sha256(document, "candidateManifestDeclaredSha256")
    origin = _validate_origin(document.get("origin"))
    inventory = _load_pinned_source_metadata(source_metadata_path, origin=origin)
    grouping_strength = document.get("groupingStrength")
    if grouping_strength != "EXACT_CONTENT_ONLY":
        raise NormalHoldoutError("MVTec source pool must use EXACT_CONTENT_ONLY grouping")
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise NormalHoldoutError("normal source pool has no records")
    _require_external_source_root(source_root)
    root = source_root
    seen_case_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_remote_paths: set[str] = set()
    seen_sources: set[str] = set()
    groups: dict[str, tuple[str, str]] = {}
    records = [
        _validate_pool_record(
            value,
            root=root,
            grouping_strength=str(grouping_strength),
            seen_case_ids=seen_case_ids,
            seen_paths=seen_paths,
            seen_remote_paths=seen_remote_paths,
            seen_sources=seen_sources,
            groups=groups,
        )
        for value in raw_records
    ]
    if [record["caseId"] for record in records] != sorted(record["caseId"] for record in records):
        raise NormalHoldoutError("normal source pool records must be sorted by caseId")
    if document.get("normalSourcePoolIdentitySha256") != _normal_source_pool_identity(records, origin, str(grouping_strength)):
        raise NormalHoldoutError("normal source pool identity digest does not match")
    for record in records:
        _validate_record_metadata(record, inventory=inventory)
    return document, file_sha256, records


def _validate_optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise NormalHoldoutError(f"{name} must be a string or null")
    return value


def _validate_feature_input(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NormalHoldoutError("normal-only feature input must be an object")
    _require_exact_fields(value, name="normal-only feature input", required=FEATURE_INPUT_FIELDS)
    case_id = _require_string(value, "caseId")
    category = _require_string(value, "category")
    role = value.get("role")
    if role not in {"FIT", "THRESHOLD_TUNING"}:
        raise NormalHoldoutError("normal-only feature input role is unsafe")
    if value.get("kind") != "NOMINAL":
        raise NormalHoldoutError("normal-only feature input kind is unsafe")
    source_sha256 = _require_sha256(value, "sourceSha256")
    is_augmentation = value.get("isAugmentation")
    if not isinstance(is_augmentation, bool):
        raise NormalHoldoutError("normal-only feature input isAugmentation must be boolean")
    variant_id = value.get("variantId")
    parent_case_id = _validate_optional_string(value.get("parentCaseId"), name="parentCaseId")
    parent_source_sha256 = value.get("parentSourceSha256")
    augmentation_recipe_sha256 = value.get("augmentationRecipeSha256")
    if is_augmentation:
        if not isinstance(variant_id, int) or isinstance(variant_id, bool) or variant_id < 1:
            raise NormalHoldoutError("augmented normal-only feature input variantId is invalid")
        if parent_case_id is None or parent_source_sha256 is None or augmentation_recipe_sha256 is None:
            raise NormalHoldoutError("augmented normal-only feature input lacks parent provenance")
        if not isinstance(parent_source_sha256, str):
            raise NormalHoldoutError("augmented normal-only parentSourceSha256 is invalid")
        if not isinstance(augmentation_recipe_sha256, str):
            raise NormalHoldoutError("augmented normal-only augmentationRecipeSha256 is invalid")
        parent_source_sha256 = _require_sha256({"value": parent_source_sha256}, "value")
        augmentation_recipe_sha256 = _require_sha256({"value": augmentation_recipe_sha256}, "value")
    else:
        if variant_id is not None or parent_case_id is not None or parent_source_sha256 is not None or augmentation_recipe_sha256 is not None:
            raise NormalHoldoutError("original normal-only feature input carries augmentation provenance")
    return {
        "caseId": case_id,
        "category": category,
        "role": role,
        "kind": "NOMINAL",
        "sourceSha256": source_sha256,
        "isAugmentation": is_augmentation,
        "variantId": variant_id,
        "parentCaseId": parent_case_id,
        "parentSourceSha256": parent_source_sha256,
        "augmentationRecipeSha256": augmentation_recipe_sha256,
    }


def _validate_normal_score(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NormalHoldoutError("normal-only score must be an object")
    is_augmentation = value.get("isAugmentation")
    expected_fields = NORMAL_SCORE_BASE_FIELDS | (NORMAL_SCORE_AUGMENTATION_FIELDS if is_augmentation is True else set())
    _require_exact_fields(value, name="normal-only score", required=expected_fields)
    if value.get("role") != "THRESHOLD_TUNING" or value.get("kind") != "NOMINAL" or value.get("defect") != "good":
        raise NormalHoldoutError("normal-only score role, kind, or defect is unsafe")
    score = value.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
        raise NormalHoldoutError("normal-only score must be finite")
    if not isinstance(is_augmentation, bool):
        raise NormalHoldoutError("normal-only score isAugmentation must be boolean")
    variant_id = value.get("variantId")
    if is_augmentation:
        if not isinstance(variant_id, int) or isinstance(variant_id, bool) or variant_id < 1:
            raise NormalHoldoutError("augmented normal-only score variantId is invalid")
        _require_string(value, "parentCaseId")
        _require_sha256(value, "parentSourceSha256")
        _require_sha256(value, "augmentationRecipeSha256")
    elif variant_id is not None:
        raise NormalHoldoutError("original normal-only score carries a variantId")
    return {
        "caseId": _require_string(value, "caseId"),
        "category": _require_string(value, "category"),
        "defect": "good",
        "isAugmentation": is_augmentation,
        "kind": "NOMINAL",
        "role": "THRESHOLD_TUNING",
        "score": float(score),
        "sourceSha256": _require_sha256(value, "sourceSha256"),
        "variantId": variant_id,
    }


def _validate_normal_only_report(path: Path) -> tuple[dict[str, Any], str]:
    """Return the validated feature envelope of one existing normal-only report.

    This intentionally consumes the JSON report only.  It never opens an
    input manifest, image, augmentation, mask, cache, model, or service.
    """

    document, file_sha256 = _read_json(path, description="historical normal-only iteration report")
    _require_exact_fields(document, name="historical normal-only iteration report", required=ITERATION_REPORT_FIELDS)
    if document.get("schemaVersion") != ITERATION_REPORT_SCHEMA:
        raise NormalHoldoutError("historical report schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise NormalHoldoutError("historical report must be non-authoritative and non-production")
    if document.get("selectionProtocol") != ITERATION_REPORT_PURPOSE:
        raise NormalHoldoutError("historical report is not a normal-only iteration")
    blind_reporting = document.get("blindReporting")
    if not isinstance(blind_reporting, dict):
        raise NormalHoldoutError("historical report blindReporting is invalid")
    _require_exact_fields(blind_reporting, name="historical report blindReporting", required=BLIND_REPORTING_FIELDS)
    if blind_reporting != {
        "state": "NOT_RUN",
        "blindSourcePolicy": "ORIGINAL_ONLY",
        "reason": "NORMAL_ONLY_ITERATION",
    }:
        raise NormalHoldoutError("historical report contains blind-reporting evidence")
    if document.get("pixelLocalization") is not None:
        raise NormalHoldoutError("historical report contains pixel-localization evidence")
    evidence = document.get("normalOnlyEvidence")
    if not isinstance(evidence, dict):
        raise NormalHoldoutError("historical report normalOnlyEvidence is invalid")
    _require_exact_fields(evidence, name="historical report normalOnlyEvidence", required=NORMAL_ONLY_EVIDENCE_FIELDS)
    for count_name in (
        "featureInputCount", "blindFeatureInputCount", "anomalyFeatureInputCount", "normalInputRecordCount",
        "reportedScoreCount", "calibrationScoreCount", "originalTuningInputCount",
    ):
        _require_nonnegative_int(evidence, count_name)
    if evidence.get("blindFeatureInputCount") != 0 or evidence.get("anomalyFeatureInputCount") != 0:
        raise NormalHoldoutError("historical report normal-only evidence includes blind or anomaly inputs")
    raw_inputs = evidence.get("featureInputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise NormalHoldoutError("historical report has no normal-only feature inputs")
    feature_inputs = [_validate_feature_input(value) for value in raw_inputs]
    if [value["caseId"] for value in feature_inputs] != sorted(value["caseId"] for value in feature_inputs):
        raise NormalHoldoutError("historical report feature inputs are not sorted by caseId")
    if len({value["caseId"] for value in feature_inputs}) != len(feature_inputs):
        raise NormalHoldoutError("historical report feature input caseId is duplicated")
    if evidence.get("featureInputCount") != len(feature_inputs) or evidence.get("normalInputRecordCount") != len(feature_inputs):
        raise NormalHoldoutError("historical report feature input counts do not match")
    if evidence.get("featureInputRoles") != sorted({value["role"] for value in feature_inputs}):
        raise NormalHoldoutError("historical report feature input roles do not match")
    if evidence.get("featureInputKinds") != ["NOMINAL"]:
        raise NormalHoldoutError("historical report feature input kinds do not match")
    feature_identity = canonical_json_sha256(feature_inputs)
    if evidence.get("featureInputIdentitySha256") != feature_identity or evidence.get("normalInputIdentitySha256") != feature_identity:
        raise NormalHoldoutError("historical report feature input identity does not match")
    originals = {
        (value["caseId"], value["category"], value["role"], value["sourceSha256"])
        for value in feature_inputs
        if not value["isAugmentation"]
    }
    for value in feature_inputs:
        if not value["isAugmentation"]:
            continue
        parent = (
            value["parentCaseId"],
            value["category"],
            value["role"],
            value["parentSourceSha256"],
        )
        if parent not in originals:
            raise NormalHoldoutError("historical report augmented input parent is not an original normal feature input")
    raw_calibration_inputs = evidence.get("calibrationInputs")
    if not isinstance(raw_calibration_inputs, list):
        raise NormalHoldoutError("historical report calibration inputs are invalid")
    calibration_inputs = [_validate_feature_input(value) for value in raw_calibration_inputs]
    expected_calibration_inputs = [value for value in feature_inputs if value["role"] == "THRESHOLD_TUNING"]
    if calibration_inputs != expected_calibration_inputs:
        raise NormalHoldoutError("historical report calibration inputs do not match the normal feature envelope")
    if (
        evidence.get("calibrationScoreCount") != len(calibration_inputs)
        or evidence.get("calibrationInputIdentitySha256") != canonical_json_sha256(calibration_inputs)
    ):
        raise NormalHoldoutError("historical report calibration input identity does not match")
    raw_original_tuning_inputs = evidence.get("originalTuningInputs")
    if not isinstance(raw_original_tuning_inputs, list):
        raise NormalHoldoutError("historical report original tuning inputs are invalid")
    original_tuning_inputs = [_validate_feature_input(value) for value in raw_original_tuning_inputs]
    expected_original_tuning_inputs = [value for value in calibration_inputs if not value["isAugmentation"]]
    if original_tuning_inputs != expected_original_tuning_inputs:
        raise NormalHoldoutError("historical report original tuning inputs do not match")
    if (
        evidence.get("originalTuningInputCount") != len(original_tuning_inputs)
        or evidence.get("originalTuningInputIdentitySha256") != canonical_json_sha256(original_tuning_inputs)
    ):
        raise NormalHoldoutError("historical report original tuning input identity does not match")
    for count_name, roles_name, kinds_name in (
        ("reportedScoreCount", "reportedScoreRoles", "reportedScoreKinds"),
        ("calibrationScoreCount", "calibrationScoreRoles", "calibrationScoreKinds"),
    ):
        if not isinstance(evidence.get(roles_name), list) or not isinstance(evidence.get(kinds_name), list):
            raise NormalHoldoutError("historical report score evidence is invalid")
        if evidence[roles_name] != ["THRESHOLD_TUNING"] or evidence[kinds_name] != ["NOMINAL"]:
            raise NormalHoldoutError("historical report score evidence is unsafe")
    raw_scores = document.get("scores")
    raw_calibration_scores = document.get("calibrationScores")
    if not isinstance(raw_scores, list) or not isinstance(raw_calibration_scores, list):
        raise NormalHoldoutError("historical report scores are invalid")
    scores = [_validate_normal_score(value) for value in raw_scores]
    calibration_scores = [_validate_normal_score(value) for value in raw_calibration_scores]
    if len(scores) != evidence["reportedScoreCount"] or len(calibration_scores) != evidence["calibrationScoreCount"]:
        raise NormalHoldoutError("historical report score counts do not match")
    return {
        "reportFileSha256": file_sha256,
        "featureInputIdentitySha256": feature_identity,
        "normalInputIdentitySha256": feature_identity,
        "normalSourceSha256": sorted({
            source
            for value in feature_inputs
            for source in (value["sourceSha256"], value["parentSourceSha256"])
            if source is not None
        }),
    }, file_sha256


def build_historical_normal_usage_ledger(report_paths: list[Path], output_path: Path) -> dict[str, Any]:
    """Freeze an exclusion list from already-normal-only iteration reports."""

    if not report_paths:
        raise NormalHoldoutError("at least one historical normal-only report is required")
    validated = [_validate_normal_only_report(path)[0] for path in report_paths]
    report_hashes = [value["reportFileSha256"] for value in validated]
    if len(set(report_hashes)) != len(report_hashes):
        raise NormalHoldoutError("historical normal-only report bytes are duplicated")
    evidence = [
        {
            "reportFileSha256": value["reportFileSha256"],
            "featureInputIdentitySha256": value["featureInputIdentitySha256"],
            "normalInputIdentitySha256": value["normalInputIdentitySha256"],
        }
        for value in sorted(validated, key=lambda item: item["reportFileSha256"])
    ]
    document: dict[str, Any] = {
        "schemaVersion": HISTORICAL_NORMAL_USAGE_LEDGER_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": HISTORICAL_LEDGER_PURPOSE,
        "exclusionScope": HISTORICAL_LEDGER_SCOPE,
        "evidence": evidence,
        "normalSourceSha256": sorted({source for value in validated for source in value["normalSourceSha256"]}),
    }
    document["historicalNormalUsageLedgerSha256"] = _document_digest(document, "historicalNormalUsageLedgerSha256")
    _write_json(output_path, document, description="historical normal usage ledger output")
    return document


def load_historical_normal_usage_ledger(path: Path) -> tuple[dict[str, Any], str, set[str]]:
    document, file_sha256 = _read_json(path, description="historical normal usage ledger")
    _require_exact_fields(
        document,
        name="historical normal usage ledger",
        required={
            "schemaVersion", "authoritative", "productionAuthorized", "purpose", "exclusionScope", "evidence",
            "normalSourceSha256", "historicalNormalUsageLedgerSha256",
        },
    )
    if document.get("schemaVersion") != HISTORICAL_NORMAL_USAGE_LEDGER_SCHEMA:
        raise NormalHoldoutError("historical normal usage ledger schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise NormalHoldoutError("historical normal usage ledger must be non-authoritative and non-production")
    if document.get("purpose") != HISTORICAL_LEDGER_PURPOSE or document.get("exclusionScope") != HISTORICAL_LEDGER_SCOPE:
        raise NormalHoldoutError("historical normal usage ledger purpose is unsafe")
    if document.get("historicalNormalUsageLedgerSha256") != _document_digest(document, "historicalNormalUsageLedgerSha256"):
        raise NormalHoldoutError("historical normal usage ledger digest does not match")
    evidence = document.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise NormalHoldoutError("historical normal usage ledger has no evidence")
    normalized_evidence: list[dict[str, str]] = []
    seen_reports: set[str] = set()
    for value in evidence:
        if not isinstance(value, dict):
            raise NormalHoldoutError("historical normal usage ledger evidence is invalid")
        _require_exact_fields(
            value,
            name="historical normal usage ledger evidence",
            required={"reportFileSha256", "featureInputIdentitySha256", "normalInputIdentitySha256"},
        )
        normalized = {
            "reportFileSha256": _require_sha256(value, "reportFileSha256"),
            "featureInputIdentitySha256": _require_sha256(value, "featureInputIdentitySha256"),
            "normalInputIdentitySha256": _require_sha256(value, "normalInputIdentitySha256"),
        }
        if normalized["reportFileSha256"] in seen_reports:
            raise NormalHoldoutError("historical normal usage ledger report digest is duplicated")
        seen_reports.add(normalized["reportFileSha256"])
        normalized_evidence.append(normalized)
    if normalized_evidence != sorted(normalized_evidence, key=lambda value: value["reportFileSha256"]):
        raise NormalHoldoutError("historical normal usage ledger evidence is not sorted")
    source_values = document.get("normalSourceSha256")
    if not isinstance(source_values, list) or not source_values:
        raise NormalHoldoutError("historical normal usage ledger has no normal source hashes")
    source_hashes = [_require_sha256({"value": value}, "value") for value in source_values]
    if source_hashes != sorted(source_hashes) or len(set(source_hashes)) != len(source_hashes):
        raise NormalHoldoutError("historical normal usage ledger source hashes are not sorted and unique")
    return document, file_sha256, set(source_hashes)


def historical_normal_usage_identity(ledger: dict[str, Any], historical_hashes: set[str]) -> str:
    """Bind a source candidate pool to the exact ledger known at acquisition."""

    return canonical_json_sha256({
        "historicalNormalUsageLedgerSha256": ledger["historicalNormalUsageLedgerSha256"],
        "normalSourceSha256": sorted(historical_hashes),
    })


def _require_pool_ledger_binding(pool: dict[str, Any], ledger: dict[str, Any], historical_hashes: set[str]) -> None:
    origin = pool.get("origin")
    if not isinstance(origin, dict):
        raise NormalHoldoutError("normal source pool origin is invalid")
    if origin.get("priorSubsetSourceIdentitySha256") != historical_normal_usage_identity(ledger, historical_hashes):
        raise NormalHoldoutError("normal source pool is not bound to the current historical usage ledger")


def _group_pool_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["sourceGroupId"])].append(record)
    for group_records in groups.values():
        group_records.sort(key=lambda record: record["caseId"])
    return dict(groups)


def _history_exclusion(
    records: list[dict[str, Any]], historical_hashes: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups = _group_pool_records(records)
    eligible_groups: dict[str, list[dict[str, Any]]] = {}
    excluded_groups = 0
    matched_sources = 0
    for group_id, group_records in groups.items():
        matches = [record for record in group_records if record["sourceSha256"] in historical_hashes]
        matched_sources += len(matches)
        if matches:
            excluded_groups += 1
            continue
        eligible_groups[group_id] = group_records
    eligible_records = [record for group in eligible_groups.values() for record in group]
    eligible_identity = canonical_json_sha256([
        _pool_record_identity(record) for record in sorted(eligible_records, key=lambda record: record["caseId"])
    ])
    return eligible_groups, {
        "algorithm": HISTORY_EXCLUSION_ALGORITHM,
        "matchedHistoricalSourceCount": matched_sources,
        "excludedSourceGroupCount": excluded_groups,
        "eligibleSourceCount": len(eligible_records),
        "eligibleSourceIdentitySha256": eligible_identity,
    }


def _pool_record_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"],
        "acquisitionStratum": record["acquisitionStratum"],
        "sourceRemotePath": record["sourceRemotePath"],
        "expectedRemoteSha256": record["expectedRemoteSha256"],
        "expectedRemoteBytes": record["expectedRemoteBytes"],
        "kind": record["kind"],
        "defect": record["defect"],
    }


def _partition_seed(pool: dict[str, Any], ledger: dict[str, Any]) -> str:
    return canonical_json_sha256({
        "schemaVersion": NORMAL_HOLDOUT_PLAN_SCHEMA,
        "partitionAlgorithm": PARTITION_ALGORITHM,
        "normalSourcePoolIdentitySha256": pool["normalSourcePoolIdentitySha256"],
        "historicalNormalUsageLedgerSha256": ledger["historicalNormalUsageLedgerSha256"],
    })


def _quota_value(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise NormalHoldoutError(f"{name} must be a positive integer")
    return value


def _normalize_category_quotas(
    category_group_quotas: dict[str, dict[str, int]], *, eligible_groups: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    categories = {group[0]["category"] for group in eligible_groups.values()}
    if set(category_group_quotas) != categories:
        raise NormalHoldoutError("normal holdout plan categories must exactly match eligible source categories")
    available_by_category: dict[str, int] = defaultdict(int)
    for group_records in eligible_groups.values():
        available_by_category[group_records[0]["category"]] += 1
    quotas: list[dict[str, Any]] = []
    for category in sorted(categories):
        supplied = category_group_quotas.get(category)
        if not isinstance(supplied, dict):
            raise NormalHoldoutError("normal holdout category quotas must be objects")
        _require_exact_fields(supplied, name="normal holdout category quota", required=PLAN_QUOTA_FIELDS - {"category"})
        quota = {
            "category": category,
            "fitGroupCount": _quota_value(supplied.get("fitGroupCount"), name="fitGroupCount"),
            "thresholdTuningGroupCount": _quota_value(
                supplied.get("thresholdTuningGroupCount"), name="thresholdTuningGroupCount"
            ),
            "normalSelectionGroupCount": _quota_value(
                supplied.get("normalSelectionGroupCount"), name="normalSelectionGroupCount"
            ),
            "normalConfirmationGroupCount": _quota_value(
                supplied.get("normalConfirmationGroupCount"), name="normalConfirmationGroupCount"
            ),
            "reserveUntouchedGroupCount": _quota_value(
                supplied.get("reserveUntouchedGroupCount"), name="reserveUntouchedGroupCount"
            ),
        }
        if sum(value for name, value in quota.items() if name != "category") != available_by_category[category]:
            raise NormalHoldoutError("normal holdout category quotas must consume every eligible source group exactly once")
        quotas.append(quota)
    return quotas


def create_normal_holdout_plan(
    pool_path: Path,
    ledger_path: Path,
    output_path: Path,
    *,
    source_root: Path,
    source_metadata_path: Path,
    category_group_quotas: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Predeclare a group-exact partition before any new candidate is scored."""

    pool, _, records = load_normal_source_pool(
        pool_path,
        source_root=source_root,
        source_metadata_path=source_metadata_path,
    )
    ledger, _, historical_hashes = load_historical_normal_usage_ledger(ledger_path)
    _require_pool_ledger_binding(pool, ledger, historical_hashes)
    eligible_groups, _ = _history_exclusion(records, historical_hashes)
    quotas = _normalize_category_quotas(category_group_quotas, eligible_groups=eligible_groups)
    document: dict[str, Any] = {
        "schemaVersion": NORMAL_HOLDOUT_PLAN_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": HOLDOUT_PURPOSE,
        "partitionAlgorithm": PARTITION_ALGORITHM,
        "partitionSeedSha256": _partition_seed(pool, ledger),
        "sourcePoolDeclaredSha256": pool["normalSourcePoolSha256"],
        "historicalLedgerDeclaredSha256": ledger["historicalNormalUsageLedgerSha256"],
        "categoryQuotas": quotas,
    }
    document["normalHoldoutPlanSha256"] = _document_digest(document, "normalHoldoutPlanSha256")
    _write_json(output_path, document, description="normal holdout plan output")
    return document


def _validate_plan_document(
    path: Path,
    *,
    pool: dict[str, Any],
    ledger: dict[str, Any],
    records: list[dict[str, Any]],
    historical_hashes: set[str],
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    _require_pool_ledger_binding(pool, ledger, historical_hashes)
    document, file_sha256 = _read_json(path, description="normal holdout plan")
    _require_exact_fields(
        document,
        name="normal holdout plan",
        required={
            "schemaVersion", "authoritative", "productionAuthorized", "purpose", "partitionAlgorithm",
            "partitionSeedSha256", "sourcePoolDeclaredSha256", "historicalLedgerDeclaredSha256", "categoryQuotas",
            "normalHoldoutPlanSha256",
        },
    )
    if document.get("schemaVersion") != NORMAL_HOLDOUT_PLAN_SCHEMA:
        raise NormalHoldoutError("normal holdout plan schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise NormalHoldoutError("normal holdout plan must be non-authoritative and non-production")
    if document.get("purpose") != HOLDOUT_PURPOSE or document.get("partitionAlgorithm") != PARTITION_ALGORITHM:
        raise NormalHoldoutError("normal holdout plan purpose is unsafe")
    if document.get("normalHoldoutPlanSha256") != _document_digest(document, "normalHoldoutPlanSha256"):
        raise NormalHoldoutError("normal holdout plan digest does not match")
    if document.get("sourcePoolDeclaredSha256") != pool["normalSourcePoolSha256"]:
        raise NormalHoldoutError("normal holdout plan source pool binding does not match")
    if document.get("historicalLedgerDeclaredSha256") != ledger["historicalNormalUsageLedgerSha256"]:
        raise NormalHoldoutError("normal holdout plan historical ledger binding does not match")
    if document.get("partitionSeedSha256") != _partition_seed(pool, ledger):
        raise NormalHoldoutError("normal holdout plan partition seed does not match")
    raw_quotas = document.get("categoryQuotas")
    if not isinstance(raw_quotas, list) or not raw_quotas:
        raise NormalHoldoutError("normal holdout plan has no category quotas")
    supplied: dict[str, dict[str, int]] = {}
    for raw_quota in raw_quotas:
        if not isinstance(raw_quota, dict):
            raise NormalHoldoutError("normal holdout plan category quota is invalid")
        _require_exact_fields(raw_quota, name="normal holdout category quota", required=PLAN_QUOTA_FIELDS)
        category = _require_string(raw_quota, "category")
        if category in supplied:
            raise NormalHoldoutError("normal holdout plan category quota is duplicated")
        supplied[category] = {name: raw_quota[name] for name in PLAN_QUOTA_FIELDS - {"category"}}
    eligible_groups, _ = _history_exclusion(records, historical_hashes)
    quotas = _normalize_category_quotas(supplied, eligible_groups=eligible_groups)
    if raw_quotas != quotas:
        raise NormalHoldoutError("normal holdout plan category quotas are not sorted or canonical")
    return document, file_sha256, quotas


def _group_rank(group_id: str, group_records: list[dict[str, Any]], *, partition_seed_sha256: str) -> str:
    category = group_records[0]["category"]
    material = "\0".join((
        NORMAL_HOLDOUT_SCHEMA,
        PARTITION_ALGORITHM,
        partition_seed_sha256,
        category,
        group_id,
        *sorted(record["sourceSha256"] for record in group_records),
    )).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _partition_quota_map(quota: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        ("FIT", int(quota["fitGroupCount"])),
        ("THRESHOLD_TUNING", int(quota["thresholdTuningGroupCount"])),
        ("NORMAL_SELECTION", int(quota["normalSelectionGroupCount"])),
        ("NORMAL_CONFIRMATION", int(quota["normalConfirmationGroupCount"])),
        ("RESERVE_UNTOUCHED", int(quota["reserveUntouchedGroupCount"])),
    ]


def _holdout_record(record: dict[str, Any], *, partition: str) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "relativePath": record["relativePath"],
        "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"],
        "acquisitionStratum": record["acquisitionStratum"],
        "sourceRemotePath": record["sourceRemotePath"],
        "expectedRemoteSha256": record["expectedRemoteSha256"],
        "expectedRemoteBytes": record["expectedRemoteBytes"],
        "kind": "NOMINAL",
        "defect": "good",
        "partition": partition,
    }


def _allocate_holdout_records(
    eligible_groups: dict[str, list[dict[str, Any]]],
    quotas: list[dict[str, Any]],
    *,
    partition_seed_sha256: str,
) -> list[dict[str, Any]]:
    groups_by_category: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for group_id, group_records in eligible_groups.items():
        groups_by_category[group_records[0]["category"]].append((group_id, group_records))
    assigned: list[dict[str, Any]] = []
    for quota in quotas:
        category = str(quota["category"])
        ranked = sorted(
            groups_by_category[category],
            key=lambda item: (_group_rank(item[0], item[1], partition_seed_sha256=partition_seed_sha256), item[0]),
        )
        cursor = 0
        for partition, group_count in _partition_quota_map(quota):
            selected = ranked[cursor:cursor + group_count]
            if len(selected) != group_count:
                raise NormalHoldoutError("normal holdout plan cannot satisfy its group quota")
            cursor += group_count
            for _, group_records in selected:
                assigned.extend(_holdout_record(record, partition=partition) for record in group_records)
        if cursor != len(ranked):
            raise NormalHoldoutError("normal holdout allocation did not consume every eligible group")
    result = sorted(assigned, key=lambda record: record["caseId"])
    if len({record["caseId"] for record in result}) != len(result):
        raise NormalHoldoutError("normal holdout allocation duplicates a caseId")
    return result


def _holdout_partition_identity(records: list[dict[str, Any]], partitions: set[str]) -> str:
    return canonical_json_sha256([
        {
            **_pool_record_identity(record),
            "partition": record["partition"],
        }
        for record in sorted(records, key=lambda record: record["caseId"])
        if record["partition"] in partitions
    ])


def _validate_history_exclusion(value: object) -> None:
    if not isinstance(value, dict):
        raise NormalHoldoutError("normal holdout manifest historyExclusion must be an object")
    _require_exact_fields(
        value,
        name="normal holdout manifest historyExclusion",
        required=HISTORY_EXCLUSION_FIELDS,
    )
    if value.get("algorithm") != HISTORY_EXCLUSION_ALGORITHM:
        raise NormalHoldoutError("normal holdout manifest history exclusion algorithm is unsupported")
    for name in ("matchedHistoricalSourceCount", "excludedSourceGroupCount", "eligibleSourceCount"):
        _require_nonnegative_int(value, name)
    _require_sha256(value, "eligibleSourceIdentitySha256")


def _validate_closed_holdout_record(
    value: object,
    *,
    seen_case_ids: set[str],
    seen_paths: set[str],
    seen_sources: set[str],
    seen_remote_paths: set[str],
    groups: dict[str, tuple[str, str, str]],
) -> dict[str, Any]:
    """Validate one record without opening its source image or source metadata."""

    if not isinstance(value, dict):
        raise NormalHoldoutError("normal holdout manifest record must be an object")
    _require_exact_fields(value, name="normal holdout manifest record", required=HOLDOUT_RECORD_FIELDS)
    case_id = _require_string(value, "caseId")
    if case_id in seen_case_ids:
        raise NormalHoldoutError("normal holdout manifest caseId is duplicated")
    seen_case_ids.add(case_id)
    category = _require_string(value, "category")
    relative_path = _safe_relative_path(value.get("relativePath"), name="normal holdout relativePath")
    if relative_path.as_posix() in seen_paths:
        raise NormalHoldoutError("normal holdout manifest relativePath is duplicated")
    seen_paths.add(relative_path.as_posix())
    source_sha256 = _require_sha256(value, "sourceSha256")
    if source_sha256 in seen_sources:
        raise NormalHoldoutError("normal holdout manifest sourceSha256 is duplicated")
    seen_sources.add(source_sha256)
    source_group_id = _require_string(value, "sourceGroupId")
    acquisition_stratum = _require_string(value, "acquisitionStratum")
    if acquisition_stratum != "OFFICIAL_MVTEC_TRAIN_GOOD":
        raise NormalHoldoutError("normal holdout manifest acquisitionStratum is unsupported")
    partition = value.get("partition")
    if partition not in HOLDOUT_PARTITIONS:
        raise NormalHoldoutError("normal holdout manifest partition is unsupported")
    existing_group = groups.setdefault(source_group_id, (category, acquisition_stratum, str(partition)))
    if existing_group != (category, acquisition_stratum, partition):
        raise NormalHoldoutError("normal holdout manifest sourceGroupId crosses category, stratum, or partition")
    source_remote_path = _safe_relative_path(value.get("sourceRemotePath"), name="normal holdout sourceRemotePath").as_posix()
    if source_remote_path in seen_remote_paths:
        raise NormalHoldoutError("normal holdout manifest sourceRemotePath is duplicated")
    seen_remote_paths.add(source_remote_path)
    expected_remote_sha256 = _require_sha256(value, "expectedRemoteSha256")
    expected_remote_bytes = _require_nonnegative_int(value, "expectedRemoteBytes")
    if expected_remote_bytes == 0:
        raise NormalHoldoutError("normal holdout manifest expectedRemoteBytes must be positive")
    if value.get("kind") != "NOMINAL" or value.get("defect") != "good":
        raise NormalHoldoutError("normal holdout manifest must contain nominal good records only")
    if expected_remote_sha256 != source_sha256:
        raise NormalHoldoutError("normal holdout manifest remote and source digests do not match")
    if source_group_id != f"CONTENT_SHA256:{source_sha256[7:]}":
        raise NormalHoldoutError("normal holdout manifest exact-content sourceGroupId is invalid")
    if case_id != f"mvtec-ad/{category}/train-good/{source_sha256[7:]}":
        raise NormalHoldoutError("normal holdout manifest caseId does not bind the normal source digest")
    return dict(value)


def _validate_closed_normal_holdout_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the normal-only, self-contained portion of a frozen cohort.

    This deliberately does not open the pool, ledger, plan, or public source
    inventory.  It is the boundary used by later normal-only evaluation code,
    which must never parse public metadata that also lists test/anomaly rows.
    """

    _require_exact_fields(document, name="normal holdout manifest", required=NORMAL_HOLDOUT_FIELDS)
    if document.get("schemaVersion") != NORMAL_HOLDOUT_SCHEMA:
        raise NormalHoldoutError("normal holdout manifest schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise NormalHoldoutError("normal holdout manifest must be non-authoritative and non-production")
    if document.get("purpose") != HOLDOUT_PURPOSE or document.get("blindPolicy") != HOLDOUT_BLIND_POLICY:
        raise NormalHoldoutError("normal holdout manifest purpose is unsafe")
    if document.get("normalHoldoutManifestSha256") != _document_digest(document, "normalHoldoutManifestSha256"):
        raise NormalHoldoutError("normal holdout manifest digest does not match")
    for name in (
        "sourcePoolFileSha256",
        "sourcePoolDeclaredSha256",
        "historicalLedgerFileSha256",
        "historicalLedgerDeclaredSha256",
        "planFileSha256",
        "planDeclaredSha256",
        "developmentIdentitySha256",
        "normalSelectionIdentitySha256",
        "normalConfirmationIdentitySha256",
        "reserveUntouchedIdentitySha256",
    ):
        _require_sha256(document, name)
    _validate_history_exclusion(document.get("historyExclusion"))
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise NormalHoldoutError("normal holdout manifest has no records")
    seen_case_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_sources: set[str] = set()
    seen_remote_paths: set[str] = set()
    groups: dict[str, tuple[str, str, str]] = {}
    records = [
        _validate_closed_holdout_record(
            value,
            seen_case_ids=seen_case_ids,
            seen_paths=seen_paths,
            seen_sources=seen_sources,
            seen_remote_paths=seen_remote_paths,
            groups=groups,
        )
        for value in raw_records
    ]
    if [record["caseId"] for record in records] != sorted(record["caseId"] for record in records):
        raise NormalHoldoutError("normal holdout manifest records must be sorted by caseId")
    identity_expectations = {
        "developmentIdentitySha256": _holdout_partition_identity(records, {"FIT", "THRESHOLD_TUNING"}),
        "normalSelectionIdentitySha256": _holdout_partition_identity(records, {"NORMAL_SELECTION"}),
        "normalConfirmationIdentitySha256": _holdout_partition_identity(records, {"NORMAL_CONFIRMATION"}),
        "reserveUntouchedIdentitySha256": _holdout_partition_identity(records, {"RESERVE_UNTOUCHED"}),
    }
    for name, expected in identity_expectations.items():
        if document.get(name) != expected:
            raise NormalHoldoutError(f"normal holdout manifest {name} does not match")
    return records


def load_evaluation_safe_normal_holdout_inputs(
    manifest_path: Path,
    *,
    source_root: Path,
    partitions: object,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Open only requested normal partitions from a closed holdout manifest.

    Unlike :func:`load_validated_normal_holdout_manifest`, this phase-safe
    reader never opens the source pool, usage ledger, plan, or MVTec
    ``samples.json`` inventory.  It validates the holdout's closed normal-only
    fields, then hashes and decodes only the requested source images.
    """

    if isinstance(partitions, str):
        raise NormalHoldoutError("normal holdout evaluation partitions must be a non-string collection")
    try:
        requested_partitions = set(partitions)
    except TypeError as error:
        raise NormalHoldoutError("normal holdout evaluation partitions must be a collection") from error
    if not requested_partitions or not requested_partitions.issubset(set(HOLDOUT_PARTITIONS)):
        raise NormalHoldoutError("normal holdout evaluation partitions are unsupported")
    if any(not isinstance(partition, str) for partition in requested_partitions):
        raise NormalHoldoutError("normal holdout evaluation partitions are unsupported")
    document, manifest_file_sha256 = _read_json(manifest_path, description="normal holdout manifest")
    records = _validate_closed_normal_holdout_document(document)
    _require_external_source_root(source_root)
    selected: list[dict[str, Any]] = []
    for record in records:
        if record["partition"] not in requested_partitions:
            continue
        relative_path = _safe_relative_path(record["relativePath"], name="normal holdout relativePath")
        source_path = _safe_file_under(source_root, relative_path, description="normal holdout evaluation image")
        if sha256_file(source_path) != record["sourceSha256"] or source_path.stat().st_size != record["expectedRemoteBytes"]:
            raise NormalHoldoutError("normal holdout evaluation image bytes do not match the frozen source")
        _require_decodable_image(source_path, description="normal holdout evaluation image")
        if sha256_file(source_path) != record["sourceSha256"]:
            raise NormalHoldoutError("normal holdout evaluation image changed while it was being decoded")
        selected.append({**record, "imagePath": source_path})
    if not selected:
        raise NormalHoldoutError("normal holdout has no records in the requested evaluation partitions")
    return document, manifest_file_sha256, selected


def build_normal_holdout_manifest(
    pool_path: Path,
    ledger_path: Path,
    plan_path: Path,
    output_path: Path,
    *,
    source_root: Path,
    source_metadata_path: Path,
) -> dict[str, Any]:
    """Allocate a predeclared normal-only development/holdout cohort exactly once."""

    pool, pool_file_sha256, records = load_normal_source_pool(
        pool_path,
        source_root=source_root,
        source_metadata_path=source_metadata_path,
    )
    ledger, ledger_file_sha256, historical_hashes = load_historical_normal_usage_ledger(ledger_path)
    plan, plan_file_sha256, quotas = _validate_plan_document(
        plan_path,
        pool=pool,
        ledger=ledger,
        records=records,
        historical_hashes=historical_hashes,
    )
    eligible_groups, history_exclusion = _history_exclusion(records, historical_hashes)
    allocated = _allocate_holdout_records(
        eligible_groups,
        quotas,
        partition_seed_sha256=str(plan["partitionSeedSha256"]),
    )
    document: dict[str, Any] = {
        "schemaVersion": NORMAL_HOLDOUT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": HOLDOUT_PURPOSE,
        "blindPolicy": HOLDOUT_BLIND_POLICY,
        "sourcePoolFileSha256": pool_file_sha256,
        "sourcePoolDeclaredSha256": pool["normalSourcePoolSha256"],
        "historicalLedgerFileSha256": ledger_file_sha256,
        "historicalLedgerDeclaredSha256": ledger["historicalNormalUsageLedgerSha256"],
        "planFileSha256": plan_file_sha256,
        "planDeclaredSha256": plan["normalHoldoutPlanSha256"],
        "historyExclusion": history_exclusion,
        "records": allocated,
        "developmentIdentitySha256": _holdout_partition_identity(allocated, {"FIT", "THRESHOLD_TUNING"}),
        "normalSelectionIdentitySha256": _holdout_partition_identity(allocated, {"NORMAL_SELECTION"}),
        "normalConfirmationIdentitySha256": _holdout_partition_identity(allocated, {"NORMAL_CONFIRMATION"}),
        "reserveUntouchedIdentitySha256": _holdout_partition_identity(allocated, {"RESERVE_UNTOUCHED"}),
    }
    document["normalHoldoutManifestSha256"] = _document_digest(document, "normalHoldoutManifestSha256")
    _write_json(output_path, document, description="normal holdout manifest output")
    return document


def load_validated_normal_holdout_manifest(
    manifest_path: Path,
    pool_path: Path,
    ledger_path: Path,
    plan_path: Path,
    *,
    source_root: Path,
    source_metadata_path: Path,
) -> tuple[dict[str, Any], str]:
    """Revalidate a cohort byte-for-byte against its source pool, ledger, and plan."""

    pool, pool_file_sha256, records = load_normal_source_pool(
        pool_path,
        source_root=source_root,
        source_metadata_path=source_metadata_path,
    )
    ledger, ledger_file_sha256, historical_hashes = load_historical_normal_usage_ledger(ledger_path)
    plan, plan_file_sha256, quotas = _validate_plan_document(
        plan_path,
        pool=pool,
        ledger=ledger,
        records=records,
        historical_hashes=historical_hashes,
    )
    document, manifest_file_sha256 = _read_json(manifest_path, description="normal holdout manifest")
    closed_records = _validate_closed_normal_holdout_document(document)
    bindings = {
        "sourcePoolFileSha256": pool_file_sha256,
        "sourcePoolDeclaredSha256": pool["normalSourcePoolSha256"],
        "historicalLedgerFileSha256": ledger_file_sha256,
        "historicalLedgerDeclaredSha256": ledger["historicalNormalUsageLedgerSha256"],
        "planFileSha256": plan_file_sha256,
        "planDeclaredSha256": plan["normalHoldoutPlanSha256"],
    }
    for name, expected in bindings.items():
        if document.get(name) != expected:
            raise NormalHoldoutError(f"normal holdout manifest {name} binding does not match")
    eligible_groups, history_exclusion = _history_exclusion(records, historical_hashes)
    if document.get("historyExclusion") != history_exclusion:
        raise NormalHoldoutError("normal holdout manifest history exclusion does not match")
    expected_records = _allocate_holdout_records(
        eligible_groups,
        quotas,
        partition_seed_sha256=str(plan["partitionSeedSha256"]),
    )
    if closed_records != expected_records:
        raise NormalHoldoutError("normal holdout manifest records do not match its frozen allocation")
    return document, manifest_file_sha256
