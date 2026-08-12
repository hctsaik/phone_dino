"""Closed, file-level source acquisition for the quarantined-cohort recovery.

Recovery V4 deliberately has a different source lineage from the exposed
``fresh_normal_holdout_v1`` cohort.  This module has three narrow phases:

* freeze a JSON-only historical/quarantine non-overlap ledger;
* validate the tracked, exact 96-record train/good allowlist against the
  pinned ``samples.json`` inventory and write an external acquisition plan;
* resolve and download only those predeclared files over pinned HTTPS.

The planning and ledger phases never open image bytes.  Acquisition never
enumerates a source tree or public inventory: it receives only the closed
plan, resolves the exact remote paths named there, and writes a new external
source root.  This is offline research infrastructure, not a PhoneDINO
runtime, qualification decision, or production control.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECOVERY_V4_ALLOWLIST_POLICY_PATH = REPOSITORY_ROOT / "tools" / "mvtec_ad_recovery_v4_source_allowlist.json"

RECOVERY_V4_ALLOWLIST_POLICY_SCHEMA = "phone-dino.mvtec-ad-recovery-v4-source-allowlist-policy/1.0"
RECOVERY_V4_NON_OVERLAP_LEDGER_SCHEMA = "phone-dino.mvtec-ad-recovery-v4-non-overlap-metadata-ledger/1.0"
RECOVERY_V4_SOURCE_ACQUISITION_PLAN_SCHEMA = "phone-dino.mvtec-ad-recovery-v4-source-acquisition-plan/1.0"
RECOVERY_V4_SOURCE_MANIFEST_SCHEMA = "phone-dino.mvtec-ad-recovery-v4-acquired-source-manifest/1.0"

ALLOWLIST_POLICY_PURPOSE = "OFFLINE_MVTEC_RECOVERY_V4_EXACT_TRAIN_GOOD_ALLOWLIST"
NON_OVERLAP_LEDGER_PURPOSE = "OFFLINE_MVTEC_RECOVERY_V4_HISTORICAL_AND_QUARANTINE_EXCLUSION"
SOURCE_ACQUISITION_PLAN_PURPOSE = "OFFLINE_MVTEC_RECOVERY_V4_PREDECLARED_FILE_LEVEL_ACQUISITION"
SOURCE_MANIFEST_PURPOSE = "OFFLINE_MVTEC_RECOVERY_V4_ACQUIRED_TRAIN_GOOD_SOURCES"

PINNED_MVTEC_DATASET_ID = "MVTec AD"
PINNED_MVTEC_MIRROR_SOURCE_URI = "https://huggingface.co/datasets/Voxel51/mvtec-ad"
PINNED_MVTEC_MIRROR_REVISION = "30a183a3b96e3aef953f230784b123b719b09d97"
PINNED_MVTEC_SAMPLES_SHA256 = "sha256:dbbbb94cee2ddec28c1eef318733d07df4d59b9cc066e62e6aeef386c1db281d"
PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1 = "d5bc5b023effedf41acf0a02832f30fa76fe0709"

RECOVERY_V4_ALLOCATION_SEED = "PHONE_DINO_STRICT_V3_FRESH_COHORT_V1"
RECOVERY_V4_RANKING_ALGORITHM = "SHA256_SEED_REVISION_CATEGORY_REMOTE_PATH_V1"
RECOVERY_V4_RANKING_INPUT_ENCODING = "UTF8_PIPE_DELIMITED_V1"
RECOVERY_V4_CATEGORIES = ("bottle", "cable", "hazelnut")
RECOVERY_V4_EXPECTED_TRAIN_GOOD_COUNTS = {"bottle": 209, "cable": 224, "hazelnut": 391}
RECOVERY_V4_ROLE_COUNTS = {"PROTOTYPE": 6, "RAW_CALIBRATION": 2, "QUERY": 4, "RESERVE": 20}
RECOVERY_V4_RECORDS_PER_CATEGORY = 32

# These are independent, compiled pins for the cohort exposed by the failed
# V3 preflight.  Keeping the pins here means a replacement non-overlap ledger
# cannot drop the known incident or silently narrow its source exclusion.
KNOWN_QUARANTINED_COHORT_MANIFEST_FILE_SHA256 = (
    "sha256:0034e045001787a6ce35042701cb470a97c03ff72117311ff7525fd5d9106b18"
)
KNOWN_QUARANTINED_COHORT_MANIFEST_DECLARED_SHA256 = (
    "sha256:51a359f5d579a99321dc33687fecc6d9a8db92fb7f921960bbb6898c23e2e74e"
)
KNOWN_QUARANTINED_COHORT_ROOT_TEXT = r"c:\code\claude\_media_out_of_repo\mvtec_ad\fresh_normal_holdout_v1"
KNOWN_QUARANTINE_INCIDENT_DECLARED_SHA256 = (
    "sha256:be690e112ae28f04a69db572a7b9931d862fcac1da9653e92a99ad5995fbf2d4"
)
KNOWN_QUARANTINED_COHORT_SOURCE_RECORD_COUNT = 477
KNOWN_QUARANTINED_COHORT_SOURCE_IDENTITY_SHA256 = (
    "sha256:8f4f999d9e74665434e164efb3b95028cd056d8be016ff91416d004e460631b2"
)
KNOWN_HISTORICAL_USAGE_LEDGER_FILE_SHA256 = (
    "sha256:fffa4b335044ecb10e749d67f195de727a639c73b3d8752d518f4ef9c084c3fc"
)
KNOWN_HISTORICAL_USAGE_LEDGER_DECLARED_SHA256 = (
    "sha256:38bedae2c856bdbb73d16863152bd9b5581b99dc74157648fdeb1cb8be430c10"
)
KNOWN_HISTORICAL_USAGE_SOURCE_HASH_COUNT = 960
KNOWN_HISTORICAL_USAGE_SOURCE_HASH_IDENTITY_SHA256 = (
    "sha256:08f6a802f4c9cb09a0f579c74a187c33f5bb6b6a2430cbcceee9b67415a37160"
)

_HISTORICAL_LEDGER_SCHEMA = "phone-dino.mvtec-ad-historical-normal-usage-ledger/1.0"
_HISTORICAL_LEDGER_PURPOSE = "OFFLINE_MVTEC_HISTORICAL_NORMAL_USAGE_EXCLUSION"
_HISTORICAL_LEDGER_SCOPE = "ALL_HISTORICAL_NORMAL_FEATURE_INPUT_SOURCE_SHA256"
_QUARANTINE_INCIDENT_SCHEMA = "phone-dino.mvtec-ad-cohort-quarantine-incident/1.0"
_QUARANTINE_INCIDENT_PURPOSE = "OFFLINE_MVTEC_COHORT_QUARANTINE"
_QUARANTINE_INCIDENT_STATUS = "QUARANTINED"
_QUARANTINE_INCIDENT_SCOPE = "NORMAL_HOLDOUT_MANIFEST_IDENTITY"
_QUARANTINE_INCIDENT_REASON = "UNBOUNDED_RECURSIVE_BYTE_HASH_EXPOSURE"
_NORMAL_HOLDOUT_SCHEMA = "phone-dino.mvtec-ad-normal-holdout/1.0"

_POLICY_FIELDS = {
    "allocationSeed",
    "authoritative",
    "categoryEligibleTrainGoodCounts",
    "datasetId",
    "mirrorRevision",
    "mirrorSourceUri",
    "policyPurpose",
    "productionAuthorized",
    "rankingAlgorithm",
    "rankingInputEncoding",
    "records",
    "recoveryV4SourceAllowlistPolicySha256",
    "roleCountsPerCategory",
    "schemaVersion",
    "sourceMetadataFileSha256",
    "sourceMetadataGitBlobSha1",
}
_POLICY_RECORD_FIELDS = {"category", "metadataRankSha256", "rank", "role", "sourceRemotePath"}
_HISTORICAL_LEDGER_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "exclusionScope",
    "evidence",
    "normalSourceSha256",
    "historicalNormalUsageLedgerSha256",
}
_QUARANTINE_INCIDENT_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "quarantineStatus",
    "scope",
    "reason",
    "incidentId",
    "cohortManifestSchemaVersion",
    "cohortManifestFileSha256",
    "cohortManifestDeclaredSha256",
    "cohortQuarantineIncidentSha256",
}
_NON_OVERLAP_LEDGER_FIELDS = {
    "authoritative",
    "excludedRemotePaths",
    "excludedSourceSha256",
    "historicalNormalUsageLedgerDeclaredSha256",
    "historicalNormalUsageLedgerFileSha256",
    "historicalSourceHashCount",
    "historicalSourceHashIdentitySha256",
    "historicalSourceSha256",
    "knownQuarantinedCohort",
    "productionAuthorized",
    "purpose",
    "recoveryV4NonOverlapMetadataLedgerSha256",
    "schemaVersion",
}
_KNOWN_QUARANTINED_COHORT_FIELDS = {
    "cohortManifestDeclaredSha256",
    "cohortManifestFileSha256",
    "cohortSourceRecordCount",
    "cohortSourceRecordIdentitySha256",
    "cohortSourceRecords",
    "quarantinedCohortRoot",
    "quarantinedCohortRootIdentitySha256",
    "quarantineIncidentDeclaredSha256",
    "quarantineIncidentFileSha256",
}
_QUARANTINED_SOURCE_RECORD_FIELDS = {"sourceRemotePath", "sourceSha256"}
_PLAN_FIELDS = {
    "allowlistPolicyDeclaredSha256",
    "allowlistPolicyFileSha256",
    "authoritative",
    "nonOverlapLedgerDeclaredSha256",
    "nonOverlapLedgerFileSha256",
    "productionAuthorized",
    "purpose",
    "records",
    "recoveryV4SourceAcquisitionPlanSha256",
    "schemaVersion",
    "sourceMetadataFileSha256",
    "sourceMetadataGitBlobSha1",
}
_MANIFEST_FIELDS = {
    "allowlistPolicyDeclaredSha256",
    "allowlistPolicyFileSha256",
    "authoritative",
    "nonOverlapLedgerDeclaredSha256",
    "nonOverlapLedgerFileSha256",
    "productionAuthorized",
    "purpose",
    "records",
    "recoveryV4AcquiredSourceManifestSha256",
    "sourceAcquisitionPlanDeclaredSha256",
    "sourceAcquisitionPlanFileSha256",
    "sourceRecordIdentitySha256",
    "schemaVersion",
}
_SOURCE_MANIFEST_RECORD_FIELDS = {
    "category",
    "expectedRemoteBytes",
    "expectedRemoteSha256",
    "relativePath",
    "role",
    "sourceRemotePath",
    "sourceSha256",
    "sourceSourceRank",
}


class RecoveryV4SourceError(ValueError):
    """Raised when Recovery V4 source provenance or isolation is unsafe."""


@dataclass(frozen=True)
class RecoveryV4ValidatedSourcePlan:
    """One coherent policy/ledger/plan snapshot for a single acquisition run."""

    plan: dict[str, Any]
    plan_file_sha256: str
    records: tuple[dict[str, Any], ...]
    excluded_remote_paths: frozenset[str]
    excluded_source_hashes: frozenset[str]
    allowlist_policy: dict[str, Any]
    allowlist_policy_file_sha256: str
    non_overlap_ledger: dict[str, Any]
    non_overlap_ledger_file_sha256: str
    quarantined_cohort_root: Path


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        fp: Any,
        code: int,
        message: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def canonical_json_sha256(document: object) -> str:
    """Return the canonical SHA-256 digest used by Recovery V4 JSON records."""

    try:
        payload = json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecoveryV4SourceError("document is not finite JSON") from error
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return canonical_json_sha256(unsigned)


def _require_exact_fields(document: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise RecoveryV4SourceError(f"{name} must be a JSON object")
    missing = fields.difference(document)
    unknown = set(document).difference(fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unsupported {', '.join(sorted(unknown))}")
        raise RecoveryV4SourceError(f"{name} has {'; '.join(details)} fields")
    return document


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryV4SourceError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise RecoveryV4SourceError(f"{name} must be a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise RecoveryV4SourceError(f"{name} must be a SHA-256 digest") from error
    return digest


def _require_positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RecoveryV4SourceError(f"{name} must be a positive integer")
    return value


def _is_under(root: Path, candidate: Path) -> bool:
    try:
        root_text = os.path.normcase(os.path.abspath(str(root)))
        candidate_text = os.path.normcase(os.path.abspath(str(candidate)))
        return os.path.commonpath((root_text, candidate_text)) == root_text
    except (OSError, ValueError):
        return False


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to inspect {path}") from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _reject_links_on_existing_path(path: Path, *, description: str) -> None:
    """Reject any existing symlink/reparse component without enumerating it."""

    current = path
    while True:
        try:
            exists = current.exists() or current.is_symlink()
        except OSError as error:
            raise RecoveryV4SourceError(f"unable to inspect {description}") from error
        if exists and _is_link_or_reparse_point(current):
            raise RecoveryV4SourceError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _require_external_regular_file(path: Path, *, description: str) -> None:
    if not isinstance(path, Path):
        raise RecoveryV4SourceError(f"{description} path must be a Path")
    if _is_under(REPOSITORY_ROOT, path) or _is_under(path, REPOSITORY_ROOT):
        raise RecoveryV4SourceError(f"{description} must stay outside the Git working tree")
    _reject_links_on_existing_path(path, description=description)
    try:
        if not path.is_file() or _is_link_or_reparse_point(path):
            raise RecoveryV4SourceError(f"{description} must be a regular non-link file")
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to inspect {description}") from error


def _require_repository_regular_file(path: Path, *, description: str) -> None:
    if not _is_under(REPOSITORY_ROOT, path):
        raise RecoveryV4SourceError(f"{description} must stay in the Git working tree")
    _reject_links_on_existing_path(path, description=description)
    try:
        if not path.is_file() or _is_link_or_reparse_point(path):
            raise RecoveryV4SourceError(f"{description} must be a regular non-link file")
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to inspect {description}") from error


def _read_json_file(path: Path, *, description: str, external: bool) -> tuple[dict[str, Any], str, bytes]:
    if external:
        _require_external_regular_file(path, description=description)
    else:
        _require_repository_regular_file(path, description=description)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(str(path), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RecoveryV4SourceError(f"{description} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = path.stat(follow_symlinks=False)
    except RecoveryV4SourceError:
        raise
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to read {description}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not _same_file_identity(before, after) or not _same_file_identity(before, path_after):
        raise RecoveryV4SourceError(f"{description} changed while it was read")
    _reject_links_on_existing_path(path, description=description)
    raw = b"".join(chunks)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RecoveryV4SourceError(f"{description} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise RecoveryV4SourceError(f"{description} contains a non-finite JSON value: {value}")

    try:
        document = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryV4SourceError(f"unable to parse {description} as JSON") from error
    if not isinstance(document, dict):
        raise RecoveryV4SourceError(f"{description} must be a JSON object")
    return document, f"sha256:{hashlib.sha256(raw).hexdigest()}", raw


def _assert_new_external_file_target(path: Path, *, description: str) -> None:
    if not isinstance(path, Path):
        raise RecoveryV4SourceError(f"{description} path must be a Path")
    if _is_under(REPOSITORY_ROOT, path) or _is_under(path, REPOSITORY_ROOT):
        raise RecoveryV4SourceError(f"{description} must stay outside the Git working tree")
    try:
        exists = path.exists() or path.is_symlink()
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to inspect {description}") from error
    if exists:
        raise RecoveryV4SourceError(f"{description} already exists; choose a new immutable path")
    _reject_links_on_existing_path(path.parent, description=description)


def _prepare_new_external_file(path: Path, *, description: str) -> None:
    _assert_new_external_file_target(path, description=description)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to create {description} parent") from error
    _reject_links_on_existing_path(path.parent, description=description)
    if path.exists() or path.is_symlink():
        raise RecoveryV4SourceError(f"{description} appeared while its parent was prepared")


def _write_new_external_json(path: Path, document: dict[str, Any], *, description: str) -> None:
    _prepare_new_external_file(path, description=description)
    payload = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(str(path), flags, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to write {description}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _assert_new_external_directory_target(path: Path, *, description: str) -> None:
    if not isinstance(path, Path):
        raise RecoveryV4SourceError(f"{description} path must be a Path")
    if _is_under(REPOSITORY_ROOT, path) or _is_under(path, REPOSITORY_ROOT):
        raise RecoveryV4SourceError(f"{description} must stay outside the Git working tree")
    try:
        exists = path.exists() or path.is_symlink()
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to inspect {description}") from error
    if exists:
        raise RecoveryV4SourceError(f"{description} already exists; choose a new external directory")
    _reject_links_on_existing_path(path.parent, description=description)


def _prepare_new_external_directory(path: Path, *, description: str) -> None:
    _assert_new_external_directory_target(path, description=description)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to create {description}") from error
    _reject_links_on_existing_path(path, description=description)
    try:
        if not path.is_dir() or _is_link_or_reparse_point(path):
            raise RecoveryV4SourceError(f"{description} is not a regular directory")
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to inspect {description}") from error


def _safe_remote_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise RecoveryV4SourceError(f"{name} must be a safe POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RecoveryV4SourceError(f"{name} must be a safe POSIX relative path")
    if path.as_posix() != value or not value.endswith(".png"):
        raise RecoveryV4SourceError(f"{name} must be a canonical PNG relative path")
    return value


def _role_for_rank(rank: int) -> str:
    if 1 <= rank <= RECOVERY_V4_ROLE_COUNTS["PROTOTYPE"]:
        return "PROTOTYPE"
    if rank <= RECOVERY_V4_ROLE_COUNTS["PROTOTYPE"] + RECOVERY_V4_ROLE_COUNTS["RAW_CALIBRATION"]:
        return "RAW_CALIBRATION"
    if rank <= (
        RECOVERY_V4_ROLE_COUNTS["PROTOTYPE"]
        + RECOVERY_V4_ROLE_COUNTS["RAW_CALIBRATION"]
        + RECOVERY_V4_ROLE_COUNTS["QUERY"]
    ):
        return "QUERY"
    if rank <= RECOVERY_V4_RECORDS_PER_CATEGORY:
        return "RESERVE"
    raise RecoveryV4SourceError("Recovery V4 source rank is outside the fixed 32-record allocation")


def _rank_remote_path(category: str, remote_path: str) -> str:
    material = "|".join((
        RECOVERY_V4_ALLOCATION_SEED,
        PINNED_MVTEC_MIRROR_REVISION,
        category,
        remote_path,
    )).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _validate_policy_record(value: object) -> dict[str, Any]:
    record = _require_exact_fields(value, name="Recovery V4 allowlist record", fields=_POLICY_RECORD_FIELDS)
    category = _require_string(record.get("category"), name="Recovery V4 allowlist category")
    if category not in RECOVERY_V4_CATEGORIES:
        raise RecoveryV4SourceError("Recovery V4 allowlist category is unsupported")
    rank = _require_positive_int(record.get("rank"), name="Recovery V4 allowlist rank")
    if rank > RECOVERY_V4_RECORDS_PER_CATEGORY:
        raise RecoveryV4SourceError("Recovery V4 allowlist rank is outside the fixed top 32")
    role = _require_string(record.get("role"), name="Recovery V4 allowlist role")
    if role != _role_for_rank(rank):
        raise RecoveryV4SourceError("Recovery V4 allowlist role does not match its fixed rank allocation")
    remote_path = _safe_remote_path(record.get("sourceRemotePath"), name="Recovery V4 allowlist sourceRemotePath")
    metadata_rank_sha256 = _require_sha256(
        record.get("metadataRankSha256"), name="Recovery V4 allowlist metadataRankSha256"
    )
    if metadata_rank_sha256 != f"sha256:{_rank_remote_path(category, remote_path)}":
        raise RecoveryV4SourceError("Recovery V4 allowlist metadata rank digest does not match the canonical input")
    return {
        "category": category,
        "metadataRankSha256": metadata_rank_sha256,
        "rank": rank,
        "role": role,
        "sourceRemotePath": remote_path,
    }


def load_recovery_v4_allowlist_policy() -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Load the tracked allowlist policy without consulting public metadata."""

    document, file_sha256, _raw = _read_json_file(
        RECOVERY_V4_ALLOWLIST_POLICY_PATH,
        description="tracked Recovery V4 source allowlist policy",
        external=False,
    )
    _require_exact_fields(document, name="Recovery V4 source allowlist policy", fields=_POLICY_FIELDS)
    fixed_values = {
        "schemaVersion": RECOVERY_V4_ALLOWLIST_POLICY_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "policyPurpose": ALLOWLIST_POLICY_PURPOSE,
        "datasetId": PINNED_MVTEC_DATASET_ID,
        "mirrorSourceUri": PINNED_MVTEC_MIRROR_SOURCE_URI,
        "mirrorRevision": PINNED_MVTEC_MIRROR_REVISION,
        "sourceMetadataFileSha256": PINNED_MVTEC_SAMPLES_SHA256,
        "sourceMetadataGitBlobSha1": PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1,
        "allocationSeed": RECOVERY_V4_ALLOCATION_SEED,
        "rankingAlgorithm": RECOVERY_V4_RANKING_ALGORITHM,
        "rankingInputEncoding": RECOVERY_V4_RANKING_INPUT_ENCODING,
        "categoryEligibleTrainGoodCounts": RECOVERY_V4_EXPECTED_TRAIN_GOOD_COUNTS,
        "roleCountsPerCategory": RECOVERY_V4_ROLE_COUNTS,
    }
    for field, expected in fixed_values.items():
        if document.get(field) != expected:
            raise RecoveryV4SourceError(f"Recovery V4 source allowlist policy {field} is unsafe")
    declared = _require_sha256(
        document.get("recoveryV4SourceAllowlistPolicySha256"),
        name="Recovery V4 source allowlist policy declared digest",
    )
    if not hmac.compare_digest(declared, _document_digest(document, "recoveryV4SourceAllowlistPolicySha256")):
        raise RecoveryV4SourceError("Recovery V4 source allowlist policy declared digest does not match")
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise RecoveryV4SourceError("Recovery V4 source allowlist policy records are invalid")
    records = [_validate_policy_record(value) for value in raw_records]
    expected_order: list[dict[str, Any]] = []
    for category in RECOVERY_V4_CATEGORIES:
        category_records = [record for record in records if record["category"] == category]
        if len(category_records) != RECOVERY_V4_RECORDS_PER_CATEGORY:
            raise RecoveryV4SourceError("Recovery V4 source allowlist must contain exactly 32 records per category")
        if [record["rank"] for record in category_records] != list(range(1, RECOVERY_V4_RECORDS_PER_CATEGORY + 1)):
            raise RecoveryV4SourceError("Recovery V4 source allowlist ranks must be canonical")
        expected_order.extend(category_records)
    if records != expected_order:
        raise RecoveryV4SourceError("Recovery V4 source allowlist records must be ordered by category and rank")
    remote_paths = [record["sourceRemotePath"] for record in records]
    if len(set(remote_paths)) != len(remote_paths):
        raise RecoveryV4SourceError("Recovery V4 source allowlist remote paths are duplicated")
    return document, file_sha256, records


def _load_pinned_source_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Read only pinned JSON metadata, never any source image or directory tree."""

    document, file_sha256, raw = _read_json_file(path, description="pinned Recovery V4 source metadata", external=True)
    if not hmac.compare_digest(file_sha256, PINNED_MVTEC_SAMPLES_SHA256):
        raise RecoveryV4SourceError("pinned Recovery V4 source metadata digest does not match")
    blob_prefix = f"blob {len(raw)}\0".encode("ascii")
    if hashlib.sha1(blob_prefix + raw).hexdigest() != PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1:
        raise RecoveryV4SourceError("pinned Recovery V4 source metadata Git blob digest does not match")
    _require_exact_fields(document, name="pinned Recovery V4 source metadata", fields={"samples"})
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RecoveryV4SourceError("pinned Recovery V4 source metadata samples are invalid")
    inventory: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise RecoveryV4SourceError("pinned Recovery V4 source metadata contains a non-object sample")
        remote_path = _safe_remote_path(sample.get("filepath"), name="pinned source metadata filepath")
        if remote_path in inventory:
            raise RecoveryV4SourceError("pinned Recovery V4 source metadata has a duplicate filepath")
        category = sample.get("category")
        defect = sample.get("defect")
        if not isinstance(category, dict) or not isinstance(defect, dict):
            raise RecoveryV4SourceError("pinned Recovery V4 source metadata labels are invalid")
        category_label = category.get("label")
        defect_label = defect.get("label")
        split = sample.get("split")
        if not isinstance(category_label, str) or not isinstance(defect_label, str) or not isinstance(split, str):
            raise RecoveryV4SourceError("pinned Recovery V4 source metadata labels are invalid")
        inventory[remote_path] = {
            "category": category_label,
            "defect": defect_label,
            "split": split,
            "hasMask": "defect_mask" in sample,
        }
    return inventory


def _quarantined_source_record_identity(records: list[dict[str, str]]) -> str:
    return canonical_json_sha256(records)


def _canonical_absolute_path_text(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _quarantined_cohort_root_identity(root_text: str) -> str:
    return canonical_json_sha256({
        "cohortManifestFileSha256": KNOWN_QUARANTINED_COHORT_MANIFEST_FILE_SHA256,
        "quarantinedCohortRoot": root_text,
    })


def _derive_quarantined_cohort_root(manifest_path: Path) -> tuple[Path, str]:
    """Derive the sole forbidden root from the named known cohort manifest."""

    if manifest_path.name != "normal_holdout.json":
        raise RecoveryV4SourceError("quarantined cohort manifest must retain its canonical normal_holdout.json name")
    root = manifest_path.parent
    _reject_links_on_existing_path(root, description="quarantined cohort root")
    try:
        if not root.is_dir() or _is_link_or_reparse_point(root):
            raise RecoveryV4SourceError("quarantined cohort root must be a regular directory")
    except OSError as error:
        raise RecoveryV4SourceError("unable to inspect quarantined cohort root") from error
    root_text = _canonical_absolute_path_text(root)
    if not os.path.isabs(root_text):
        raise RecoveryV4SourceError("quarantined cohort root must be an absolute path")
    if not hmac.compare_digest(root_text, KNOWN_QUARANTINED_COHORT_ROOT_TEXT):
        raise RecoveryV4SourceError("quarantined cohort manifest is not located under the compiled known exposed root")
    if _is_under(REPOSITORY_ROOT, Path(root_text)) or _is_under(Path(root_text), REPOSITORY_ROOT):
        raise RecoveryV4SourceError("quarantined cohort root must stay outside the Git working tree")
    return root, root_text


def _historical_source_hash_identity(source_hashes: list[str]) -> str:
    return canonical_json_sha256(source_hashes)


def _validate_historical_usage_ledger(document: dict[str, Any], *, file_sha256: str) -> set[str]:
    _require_exact_fields(document, name="historical normal usage ledger", fields=_HISTORICAL_LEDGER_FIELDS)
    if (
        document.get("schemaVersion") != _HISTORICAL_LEDGER_SCHEMA
        or document.get("authoritative") is not False
        or document.get("productionAuthorized") is not False
        or document.get("purpose") != _HISTORICAL_LEDGER_PURPOSE
        or document.get("exclusionScope") != _HISTORICAL_LEDGER_SCOPE
    ):
        raise RecoveryV4SourceError("historical normal usage ledger is unsafe")
    declared = _require_sha256(
        document.get("historicalNormalUsageLedgerSha256"),
        name="historical normal usage ledger declared digest",
    )
    if not hmac.compare_digest(declared, _document_digest(document, "historicalNormalUsageLedgerSha256")):
        raise RecoveryV4SourceError("historical normal usage ledger declared digest does not match")
    if not hmac.compare_digest(file_sha256, KNOWN_HISTORICAL_USAGE_LEDGER_FILE_SHA256):
        raise RecoveryV4SourceError("historical normal usage ledger does not match the trusted raw-file pin")
    if not hmac.compare_digest(declared, KNOWN_HISTORICAL_USAGE_LEDGER_DECLARED_SHA256):
        raise RecoveryV4SourceError("historical normal usage ledger does not match the trusted declared pin")
    evidence = document.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise RecoveryV4SourceError("historical normal usage ledger evidence is invalid")
    source_values = document.get("normalSourceSha256")
    if not isinstance(source_values, list) or not source_values:
        raise RecoveryV4SourceError("historical normal usage ledger source hashes are invalid")
    source_hashes = [_require_sha256(value, name="historical normal usage source digest") for value in source_values]
    if source_hashes != sorted(source_hashes) or len(source_hashes) != len(set(source_hashes)):
        raise RecoveryV4SourceError("historical normal usage ledger source hashes must be sorted and unique")
    if len(source_hashes) != KNOWN_HISTORICAL_USAGE_SOURCE_HASH_COUNT or not hmac.compare_digest(
        _historical_source_hash_identity(source_hashes), KNOWN_HISTORICAL_USAGE_SOURCE_HASH_IDENTITY_SHA256
    ):
        raise RecoveryV4SourceError("historical normal usage ledger source hashes do not match the trusted exact identity")
    return set(source_hashes)


def _validate_quarantine_incident(document: dict[str, Any]) -> None:
    _require_exact_fields(document, name="known cohort quarantine incident", fields=_QUARANTINE_INCIDENT_FIELDS)
    expected = {
        "schemaVersion": _QUARANTINE_INCIDENT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": _QUARANTINE_INCIDENT_PURPOSE,
        "quarantineStatus": _QUARANTINE_INCIDENT_STATUS,
        "scope": _QUARANTINE_INCIDENT_SCOPE,
        "reason": _QUARANTINE_INCIDENT_REASON,
        "cohortManifestSchemaVersion": _NORMAL_HOLDOUT_SCHEMA,
        "cohortManifestFileSha256": KNOWN_QUARANTINED_COHORT_MANIFEST_FILE_SHA256,
        "cohortManifestDeclaredSha256": KNOWN_QUARANTINED_COHORT_MANIFEST_DECLARED_SHA256,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise RecoveryV4SourceError("known cohort quarantine incident does not bind the exposed cohort")
    declared = _require_sha256(
        document.get("cohortQuarantineIncidentSha256"),
        name="known cohort quarantine incident declared digest",
    )
    if not hmac.compare_digest(declared, _document_digest(document, "cohortQuarantineIncidentSha256")):
        raise RecoveryV4SourceError("known cohort quarantine incident declared digest does not match")
    if not hmac.compare_digest(declared, KNOWN_QUARANTINE_INCIDENT_DECLARED_SHA256):
        raise RecoveryV4SourceError("known cohort quarantine incident does not match its independent pin")


def _extract_known_quarantined_source_records(
    document: dict[str, Any],
    *,
    manifest_file_sha256: str,
) -> list[dict[str, str]]:
    """Extract only path/hash metadata after verifying the compiled manifest pin."""

    if not hmac.compare_digest(manifest_file_sha256, KNOWN_QUARANTINED_COHORT_MANIFEST_FILE_SHA256):
        raise RecoveryV4SourceError("quarantined cohort manifest does not match the known exposed bytes")
    if document.get("schemaVersion") != _NORMAL_HOLDOUT_SCHEMA:
        raise RecoveryV4SourceError("quarantined cohort manifest schema is unsafe")
    declared = _require_sha256(
        document.get("normalHoldoutManifestSha256"),
        name="quarantined cohort manifest declared digest",
    )
    if not hmac.compare_digest(declared, KNOWN_QUARANTINED_COHORT_MANIFEST_DECLARED_SHA256):
        raise RecoveryV4SourceError("quarantined cohort manifest does not match the known exposed identity")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise RecoveryV4SourceError("quarantined cohort manifest records are invalid")
    normalized: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise RecoveryV4SourceError("quarantined cohort manifest record is invalid")
        remote_path = _safe_remote_path(raw_record.get("sourceRemotePath"), name="quarantined cohort sourceRemotePath")
        source_hash = _require_sha256(raw_record.get("sourceSha256"), name="quarantined cohort sourceSha256")
        if remote_path in seen_paths or source_hash in seen_hashes:
            raise RecoveryV4SourceError("quarantined cohort source metadata is duplicated")
        seen_paths.add(remote_path)
        seen_hashes.add(source_hash)
        normalized.append({"sourceRemotePath": remote_path, "sourceSha256": source_hash})
    normalized.sort(key=lambda record: (record["sourceRemotePath"], record["sourceSha256"]))
    if len(normalized) != KNOWN_QUARANTINED_COHORT_SOURCE_RECORD_COUNT:
        raise RecoveryV4SourceError("quarantined cohort source record count is not the known exposed count")
    if not hmac.compare_digest(
        _quarantined_source_record_identity(normalized), KNOWN_QUARANTINED_COHORT_SOURCE_IDENTITY_SHA256
    ):
        raise RecoveryV4SourceError("quarantined cohort source metadata does not match the known exposed cohort")
    return normalized


def freeze_recovery_v4_non_overlap_metadata_ledger(
    historical_usage_ledger_path: Path,
    quarantine_incident_path: Path,
    quarantined_cohort_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze historical and known-quarantine exclusions without opening images.

    All three inputs are individually named JSON files.  This function never
    receives a source root and does not enumerate any directory.
    """

    historical, historical_file_sha256, _ = _read_json_file(
        historical_usage_ledger_path,
        description="Recovery V4 historical normal usage ledger",
        external=True,
    )
    historical_source_hashes = _validate_historical_usage_ledger(
        historical,
        file_sha256=historical_file_sha256,
    )
    incident, incident_file_sha256, _ = _read_json_file(
        quarantine_incident_path,
        description="Recovery V4 known cohort quarantine incident",
        external=True,
    )
    _validate_quarantine_incident(incident)
    cohort, cohort_file_sha256, _ = _read_json_file(
        quarantined_cohort_manifest_path,
        description="Recovery V4 quarantined cohort manifest",
        external=True,
    )
    cohort_root, cohort_root_text = _derive_quarantined_cohort_root(quarantined_cohort_manifest_path)
    _assert_outside_known_quarantined_cohort_root(
        output_path,
        quarantined_root=cohort_root,
        description="Recovery V4 non-overlap metadata ledger output",
    )
    cohort_source_records = _extract_known_quarantined_source_records(cohort, manifest_file_sha256=cohort_file_sha256)
    cohort_source_hashes = {record["sourceSha256"] for record in cohort_source_records}
    cohort_remote_paths = [record["sourceRemotePath"] for record in cohort_source_records]
    document: dict[str, Any] = {
        "schemaVersion": RECOVERY_V4_NON_OVERLAP_LEDGER_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": NON_OVERLAP_LEDGER_PURPOSE,
        "historicalNormalUsageLedgerFileSha256": historical_file_sha256,
        "historicalNormalUsageLedgerDeclaredSha256": historical["historicalNormalUsageLedgerSha256"],
        "historicalSourceHashCount": len(historical_source_hashes),
        "historicalSourceHashIdentitySha256": _historical_source_hash_identity(sorted(historical_source_hashes)),
        "historicalSourceSha256": sorted(historical_source_hashes),
        "knownQuarantinedCohort": {
            "quarantineIncidentFileSha256": incident_file_sha256,
            "quarantineIncidentDeclaredSha256": incident["cohortQuarantineIncidentSha256"],
            "cohortManifestFileSha256": cohort_file_sha256,
            "cohortManifestDeclaredSha256": cohort["normalHoldoutManifestSha256"],
            "cohortSourceRecordCount": len(cohort_source_records),
            "cohortSourceRecordIdentitySha256": _quarantined_source_record_identity(cohort_source_records),
            "cohortSourceRecords": cohort_source_records,
            "quarantinedCohortRoot": cohort_root_text,
            "quarantinedCohortRootIdentitySha256": _quarantined_cohort_root_identity(cohort_root_text),
        },
        "excludedRemotePaths": cohort_remote_paths,
        "excludedSourceSha256": sorted(historical_source_hashes | cohort_source_hashes),
    }
    document["recoveryV4NonOverlapMetadataLedgerSha256"] = _document_digest(
        document, "recoveryV4NonOverlapMetadataLedgerSha256"
    )
    _write_new_external_json(output_path, document, description="Recovery V4 non-overlap metadata ledger output")
    return document


def _validate_quarantined_cohort_ledger_entry(value: object) -> tuple[set[str], set[str], Path]:
    entry = _require_exact_fields(value, name="Recovery V4 known quarantined cohort ledger entry", fields=_KNOWN_QUARANTINED_COHORT_FIELDS)
    expected = {
        "quarantineIncidentDeclaredSha256": KNOWN_QUARANTINE_INCIDENT_DECLARED_SHA256,
        "cohortManifestFileSha256": KNOWN_QUARANTINED_COHORT_MANIFEST_FILE_SHA256,
        "cohortManifestDeclaredSha256": KNOWN_QUARANTINED_COHORT_MANIFEST_DECLARED_SHA256,
        "cohortSourceRecordCount": KNOWN_QUARANTINED_COHORT_SOURCE_RECORD_COUNT,
        "cohortSourceRecordIdentitySha256": KNOWN_QUARANTINED_COHORT_SOURCE_IDENTITY_SHA256,
    }
    for field, expected_value in expected.items():
        if entry.get(field) != expected_value:
            raise RecoveryV4SourceError("Recovery V4 non-overlap ledger omits or changes a known quarantined cohort identity")
    _require_sha256(entry.get("quarantineIncidentFileSha256"), name="Recovery V4 quarantine incident file digest")
    root_text = _require_string(entry.get("quarantinedCohortRoot"), name="Recovery V4 quarantined cohort root")
    if not os.path.isabs(root_text) or root_text != _canonical_absolute_path_text(Path(root_text)):
        raise RecoveryV4SourceError("Recovery V4 quarantined cohort root is not a canonical absolute path")
    if not hmac.compare_digest(root_text, KNOWN_QUARANTINED_COHORT_ROOT_TEXT):
        raise RecoveryV4SourceError("Recovery V4 non-overlap ledger quarantined root does not match the compiled known root")
    root_identity = _require_sha256(
        entry.get("quarantinedCohortRootIdentitySha256"),
        name="Recovery V4 quarantined cohort root identity",
    )
    if not hmac.compare_digest(root_identity, _quarantined_cohort_root_identity(root_text)):
        raise RecoveryV4SourceError("Recovery V4 quarantined cohort root identity does not match")
    root = Path(root_text)
    if _is_under(REPOSITORY_ROOT, root) or _is_under(root, REPOSITORY_ROOT):
        raise RecoveryV4SourceError("Recovery V4 quarantined cohort root is unsafe")
    records = entry.get("cohortSourceRecords")
    if not isinstance(records, list):
        raise RecoveryV4SourceError("Recovery V4 quarantined cohort source records are invalid")
    normalized: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for raw_record in records:
        record = _require_exact_fields(raw_record, name="Recovery V4 quarantined cohort source record", fields=_QUARANTINED_SOURCE_RECORD_FIELDS)
        remote_path = _safe_remote_path(record.get("sourceRemotePath"), name="Recovery V4 quarantined cohort sourceRemotePath")
        source_hash = _require_sha256(record.get("sourceSha256"), name="Recovery V4 quarantined cohort sourceSha256")
        if remote_path in seen_paths or source_hash in seen_hashes:
            raise RecoveryV4SourceError("Recovery V4 quarantined cohort source records are duplicated")
        seen_paths.add(remote_path)
        seen_hashes.add(source_hash)
        normalized.append({"sourceRemotePath": remote_path, "sourceSha256": source_hash})
    if normalized != sorted(normalized, key=lambda record: (record["sourceRemotePath"], record["sourceSha256"])):
        raise RecoveryV4SourceError("Recovery V4 quarantined cohort source records are not canonical")
    if len(normalized) != KNOWN_QUARANTINED_COHORT_SOURCE_RECORD_COUNT or not hmac.compare_digest(
        _quarantined_source_record_identity(normalized), KNOWN_QUARANTINED_COHORT_SOURCE_IDENTITY_SHA256
    ):
        raise RecoveryV4SourceError("Recovery V4 quarantined cohort source records do not match the known exposed cohort")
    return seen_paths, seen_hashes, root


def load_recovery_v4_non_overlap_metadata_ledger(path: Path) -> tuple[dict[str, Any], str, set[str], set[str], Path]:
    """Load a closed exclusion ledger without opening images or source roots."""

    document, file_sha256, _ = _read_json_file(path, description="Recovery V4 non-overlap metadata ledger", external=True)
    _require_exact_fields(document, name="Recovery V4 non-overlap metadata ledger", fields=_NON_OVERLAP_LEDGER_FIELDS)
    if (
        document.get("schemaVersion") != RECOVERY_V4_NON_OVERLAP_LEDGER_SCHEMA
        or document.get("authoritative") is not False
        or document.get("productionAuthorized") is not False
        or document.get("purpose") != NON_OVERLAP_LEDGER_PURPOSE
    ):
        raise RecoveryV4SourceError("Recovery V4 non-overlap metadata ledger is unsafe")
    declared = _require_sha256(
        document.get("recoveryV4NonOverlapMetadataLedgerSha256"),
        name="Recovery V4 non-overlap metadata ledger declared digest",
    )
    if not hmac.compare_digest(declared, _document_digest(document, "recoveryV4NonOverlapMetadataLedgerSha256")):
        raise RecoveryV4SourceError("Recovery V4 non-overlap metadata ledger declared digest does not match")
    historical_file_sha256 = _require_sha256(
        document.get("historicalNormalUsageLedgerFileSha256"),
        name="Recovery V4 historical usage ledger file digest",
    )
    historical_declared_sha256 = _require_sha256(
        document.get("historicalNormalUsageLedgerDeclaredSha256"),
        name="Recovery V4 historical usage ledger declared digest",
    )
    if not hmac.compare_digest(historical_file_sha256, KNOWN_HISTORICAL_USAGE_LEDGER_FILE_SHA256) or not hmac.compare_digest(
        historical_declared_sha256, KNOWN_HISTORICAL_USAGE_LEDGER_DECLARED_SHA256
    ):
        raise RecoveryV4SourceError("Recovery V4 non-overlap ledger does not bind the trusted historical ledger")
    historical_hash_count = _require_positive_int(
        document.get("historicalSourceHashCount"), name="Recovery V4 historical source hash count"
    )
    historical_hash_identity = _require_sha256(
        document.get("historicalSourceHashIdentitySha256"), name="Recovery V4 historical source hash identity"
    )
    raw_historical_hashes = document.get("historicalSourceSha256")
    if not isinstance(raw_historical_hashes, list):
        raise RecoveryV4SourceError("Recovery V4 historical source hashes are invalid")
    historical_hashes = [_require_sha256(value, name="Recovery V4 historical source digest") for value in raw_historical_hashes]
    if historical_hashes != sorted(historical_hashes) or len(historical_hashes) != len(set(historical_hashes)):
        raise RecoveryV4SourceError("Recovery V4 historical source hashes must be sorted and unique")
    if (
        historical_hash_count != KNOWN_HISTORICAL_USAGE_SOURCE_HASH_COUNT
        or len(historical_hashes) != historical_hash_count
        or not hmac.compare_digest(historical_hash_identity, KNOWN_HISTORICAL_USAGE_SOURCE_HASH_IDENTITY_SHA256)
        or not hmac.compare_digest(_historical_source_hash_identity(historical_hashes), historical_hash_identity)
    ):
        raise RecoveryV4SourceError("Recovery V4 non-overlap ledger historical source hashes do not match the trusted identity")
    quarantined_paths, quarantined_hashes, quarantined_root = _validate_quarantined_cohort_ledger_entry(
        document.get("knownQuarantinedCohort")
    )
    raw_paths = document.get("excludedRemotePaths")
    raw_hashes = document.get("excludedSourceSha256")
    if not isinstance(raw_paths, list) or not isinstance(raw_hashes, list):
        raise RecoveryV4SourceError("Recovery V4 non-overlap exclusions are invalid")
    paths = [_safe_remote_path(value, name="Recovery V4 excluded remote path") for value in raw_paths]
    hashes = [_require_sha256(value, name="Recovery V4 excluded source digest") for value in raw_hashes]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RecoveryV4SourceError("Recovery V4 excluded remote paths must be sorted and unique")
    if hashes != sorted(hashes) or len(hashes) != len(set(hashes)):
        raise RecoveryV4SourceError("Recovery V4 excluded source digests must be sorted and unique")
    if set(paths) != quarantined_paths:
        raise RecoveryV4SourceError("Recovery V4 non-overlap remote-path exclusions do not exactly match the quarantined cohort")
    expected_hashes = set(historical_hashes) | quarantined_hashes
    if set(hashes) != expected_hashes:
        raise RecoveryV4SourceError("Recovery V4 non-overlap source exclusions do not exactly match historical/quarantined union")
    return document, file_sha256, set(paths), set(hashes), quarantined_root


def _assert_outside_known_quarantined_cohort_root(
    candidate: Path,
    *,
    quarantined_root: Path,
    description: str,
) -> None:
    """Reject a new Recovery V4 output that overlaps the actual exposed root.

    The forbidden root is never caller-configurable: it is recovered only
    from the validated, closed non-overlap ledger.  This is a tool-mediated
    containment guard, not a substitute for an externally provisioned
    same-privilege ACL/broker boundary.
    """

    _reject_links_on_existing_path(quarantined_root, description="known quarantined cohort root")
    try:
        if not quarantined_root.is_dir() or _is_link_or_reparse_point(quarantined_root):
            raise RecoveryV4SourceError("known quarantined cohort root is no longer a regular directory")
    except OSError as error:
        raise RecoveryV4SourceError("unable to inspect known quarantined cohort root") from error
    _reject_links_on_existing_path(candidate.parent, description=description)
    if _is_under(quarantined_root, candidate) or _is_under(candidate, quarantined_root):
        raise RecoveryV4SourceError(f"{description} overlaps the known quarantined cohort root")


def _validate_policy_against_pinned_metadata(
    policy_records: list[dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    *,
    excluded_remote_paths: set[str],
) -> None:
    """Prove the tracked table is exactly the seeded top-32 normal-only table."""

    for category in RECOVERY_V4_CATEGORIES:
        candidates = [
            remote_path
            for remote_path, metadata in inventory.items()
            if metadata["category"] == category
            and metadata["split"] == "train"
            and metadata["defect"] == "good"
            and not metadata["hasMask"]
        ]
        if len(candidates) != RECOVERY_V4_EXPECTED_TRAIN_GOOD_COUNTS[category]:
            raise RecoveryV4SourceError("pinned metadata train/good count does not match the Recovery V4 policy")
        expected = [
            (rank_digest, remote_path)
            for rank_digest, remote_path in sorted(
                (_rank_remote_path(category, remote_path), remote_path) for remote_path in candidates
            )[:RECOVERY_V4_RECORDS_PER_CATEGORY]
        ]
        records = [record for record in policy_records if record["category"] == category]
        if [
            (record["metadataRankSha256"][7:], record["sourceRemotePath"])
            for record in records
        ] != expected:
            raise RecoveryV4SourceError("Recovery V4 tracked allowlist is not the fixed seeded top-32 selection")
        for record in records:
            metadata = inventory.get(record["sourceRemotePath"])
            if metadata is None or (
                metadata["category"] != category
                or metadata["split"] != "train"
                or metadata["defect"] != "good"
                or metadata["hasMask"]
            ):
                raise RecoveryV4SourceError("Recovery V4 allowlist selects a non-train/good or masked source")
            if record["sourceRemotePath"] in excluded_remote_paths:
                raise RecoveryV4SourceError("Recovery V4 allowlist overlaps the historical/quarantined remote-path exclusion")


def create_recovery_v4_source_acquisition_plan(
    source_metadata_path: Path,
    non_overlap_ledger_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write one external metadata-only plan for the exact tracked allowlist.

    This is the only Recovery V4 phase allowed to read ``samples.json``.  It
    does not receive a source root, invoke HTTP, decode images, or enumerate a
    filesystem directory.
    """

    policy, policy_file_sha256, policy_records = load_recovery_v4_allowlist_policy()
    ledger, ledger_file_sha256, excluded_remote_paths, _excluded_source_hashes, quarantined_root = load_recovery_v4_non_overlap_metadata_ledger(
        non_overlap_ledger_path
    )
    _assert_outside_known_quarantined_cohort_root(
        output_path,
        quarantined_root=quarantined_root,
        description="Recovery V4 source acquisition plan output",
    )
    inventory = _load_pinned_source_metadata(source_metadata_path)
    _validate_policy_against_pinned_metadata(
        policy_records,
        inventory,
        excluded_remote_paths=excluded_remote_paths,
    )
    document: dict[str, Any] = {
        "schemaVersion": RECOVERY_V4_SOURCE_ACQUISITION_PLAN_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": SOURCE_ACQUISITION_PLAN_PURPOSE,
        "sourceMetadataFileSha256": PINNED_MVTEC_SAMPLES_SHA256,
        "sourceMetadataGitBlobSha1": PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1,
        "allowlistPolicyFileSha256": policy_file_sha256,
        "allowlistPolicyDeclaredSha256": policy["recoveryV4SourceAllowlistPolicySha256"],
        "nonOverlapLedgerFileSha256": ledger_file_sha256,
        "nonOverlapLedgerDeclaredSha256": ledger["recoveryV4NonOverlapMetadataLedgerSha256"],
        "records": policy_records,
    }
    document["recoveryV4SourceAcquisitionPlanSha256"] = _document_digest(
        document, "recoveryV4SourceAcquisitionPlanSha256"
    )
    _write_new_external_json(output_path, document, description="Recovery V4 source acquisition plan output")
    return document


def _validate_plan_document(
    document: dict[str, Any],
    *,
    policy: dict[str, Any],
    policy_file_sha256: str,
    policy_records: list[dict[str, Any]],
    ledger: dict[str, Any],
    ledger_file_sha256: str,
) -> list[dict[str, Any]]:
    _require_exact_fields(document, name="Recovery V4 source acquisition plan", fields=_PLAN_FIELDS)
    if (
        document.get("schemaVersion") != RECOVERY_V4_SOURCE_ACQUISITION_PLAN_SCHEMA
        or document.get("authoritative") is not False
        or document.get("productionAuthorized") is not False
        or document.get("purpose") != SOURCE_ACQUISITION_PLAN_PURPOSE
    ):
        raise RecoveryV4SourceError("Recovery V4 source acquisition plan is unsafe")
    declared = _require_sha256(
        document.get("recoveryV4SourceAcquisitionPlanSha256"),
        name="Recovery V4 source acquisition plan declared digest",
    )
    if not hmac.compare_digest(declared, _document_digest(document, "recoveryV4SourceAcquisitionPlanSha256")):
        raise RecoveryV4SourceError("Recovery V4 source acquisition plan declared digest does not match")
    expected_bindings = {
        "sourceMetadataFileSha256": PINNED_MVTEC_SAMPLES_SHA256,
        "sourceMetadataGitBlobSha1": PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1,
        "allowlistPolicyFileSha256": policy_file_sha256,
        "allowlistPolicyDeclaredSha256": policy["recoveryV4SourceAllowlistPolicySha256"],
        "nonOverlapLedgerFileSha256": ledger_file_sha256,
        "nonOverlapLedgerDeclaredSha256": ledger["recoveryV4NonOverlapMetadataLedgerSha256"],
    }
    for field, expected in expected_bindings.items():
        if document.get(field) != expected:
            raise RecoveryV4SourceError(f"Recovery V4 source acquisition plan {field} binding does not match")
    records = document.get("records")
    if records != policy_records:
        raise RecoveryV4SourceError("Recovery V4 source acquisition plan records do not match the tracked allowlist")
    return [dict(record) for record in policy_records]


def load_recovery_v4_source_acquisition_plan(
    plan_path: Path,
    *,
    non_overlap_ledger_path: Path,
) -> RecoveryV4ValidatedSourcePlan:
    """Validate a closed source plan without reading public metadata or images."""

    policy, policy_file_sha256, policy_records = load_recovery_v4_allowlist_policy()
    ledger, ledger_file_sha256, excluded_paths, excluded_hashes, quarantined_root = load_recovery_v4_non_overlap_metadata_ledger(
        non_overlap_ledger_path
    )
    plan, plan_file_sha256, _ = _read_json_file(
        plan_path,
        description="Recovery V4 source acquisition plan",
        external=True,
    )
    records = _validate_plan_document(
        plan,
        policy=policy,
        policy_file_sha256=policy_file_sha256,
        policy_records=policy_records,
        ledger=ledger,
        ledger_file_sha256=ledger_file_sha256,
    )
    return RecoveryV4ValidatedSourcePlan(
        plan=plan,
        plan_file_sha256=plan_file_sha256,
        records=tuple(records),
        excluded_remote_paths=frozenset(excluded_paths),
        excluded_source_hashes=frozenset(excluded_hashes),
        allowlist_policy=policy,
        allowlist_policy_file_sha256=policy_file_sha256,
        non_overlap_ledger=ledger,
        non_overlap_ledger_file_sha256=ledger_file_sha256,
        quarantined_cohort_root=quarantined_root,
    )


def _header(headers: Any, name: str) -> str:
    value = headers.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RecoveryV4SourceError(f"Recovery V4 remote redirect is missing {name}")
    return value.strip()


def _approved_hf_https_url(url: str, *, description: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        raise RecoveryV4SourceError(f"{description} is not HTTPS")
    if not (host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co")):
        raise RecoveryV4SourceError(f"{description} host is not an approved Hugging Face host")
    return url


def _resolve_remote_identity(remote_path: str, *, timeout_seconds: float) -> tuple[str, int, str]:
    """Resolve one explicit allowlisted path and pin its content identity."""

    safe_path = _safe_remote_path(remote_path, name="Recovery V4 resolved sourceRemotePath")
    resolve_url = (
        f"{PINNED_MVTEC_MIRROR_SOURCE_URI}/resolve/{PINNED_MVTEC_MIRROR_REVISION}/"
        f"{quote(safe_path, safe='/')}?download=true"
    )
    request = Request(resolve_url, headers={"User-Agent": "phone-dino-recovery-v4/1.0"})
    opener = build_opener(_NoRedirect())
    try:
        response = opener.open(request, timeout=timeout_seconds)
    except HTTPError as error:
        if error.code not in {301, 302, 303, 307, 308}:
            raise RecoveryV4SourceError(f"Recovery V4 remote resolve failed with HTTP {error.code}") from error
        response = error
    except URLError as error:
        raise RecoveryV4SourceError("Recovery V4 remote resolve failed") from error
    try:
        if getattr(response, "code", getattr(response, "status", None)) not in {301, 302, 303, 307, 308}:
            raise RecoveryV4SourceError("Recovery V4 remote resolve did not return a redirect")
        if _header(response.headers, "X-Repo-Commit") != PINNED_MVTEC_MIRROR_REVISION:
            raise RecoveryV4SourceError("Recovery V4 remote redirect revision does not match the pinned revision")
        etag = _header(response.headers, "X-Linked-ETag").strip('"')
        if len(etag) != 64:
            raise RecoveryV4SourceError("Recovery V4 remote redirect raw digest is invalid")
        try:
            int(etag, 16)
        except ValueError as error:
            raise RecoveryV4SourceError("Recovery V4 remote redirect raw digest is invalid") from error
        try:
            byte_count = int(_header(response.headers, "X-Linked-Size"))
        except ValueError as error:
            raise RecoveryV4SourceError("Recovery V4 remote redirect raw size is invalid") from error
        if byte_count <= 0:
            raise RecoveryV4SourceError("Recovery V4 remote redirect raw size is invalid")
        location = _approved_hf_https_url(
            urljoin(resolve_url, _header(response.headers, "Location")),
            description="Recovery V4 remote redirect location",
        )
        return f"sha256:{etag}", byte_count, location
    finally:
        response.close()


def _regular_nonlink_file_identity(path: Path, *, description: str) -> os.stat_result:
    _reject_links_on_existing_path(path, description=description)
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to inspect {description}") from error
    if not stat.S_ISREG(metadata.st_mode) or _is_link_or_reparse_point(path):
        raise RecoveryV4SourceError(f"{description} must be a regular non-link file")
    return metadata


def _sha256_file_with_identity(path: Path, *, description: str) -> tuple[str, os.stat_result]:
    _regular_nonlink_file_identity(path, description=description)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(str(path), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RecoveryV4SourceError(f"{description} must be a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = path.stat(follow_symlinks=False)
    except RecoveryV4SourceError:
        raise
    except OSError as error:
        raise RecoveryV4SourceError(f"unable to hash {description}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not _same_file_identity(before, after) or not _same_file_identity(before, path_after):
        raise RecoveryV4SourceError(f"{description} changed while it was hashed")
    _reject_links_on_existing_path(path, description=description)
    return f"sha256:{digest.hexdigest()}", before


def _fsync_parent_directory(path: Path, *, description: str) -> None:
    """Durably sync parent metadata where the platform permits it."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(str(path), flags)
        os.fsync(descriptor)
    except OSError as error:
        # Windows commonly does not permit opening/fsyncing a directory.  The
        # file itself has already been fsynced, and unsupported directory sync
        # must not turn a verified new-only artifact into an overwrite retry.
        if error.errno not in {errno.EACCES, errno.EINVAL, errno.EISDIR, errno.ENOTSUP, errno.EPERM}:
            raise RecoveryV4SourceError(f"unable to fsync {description} parent directory") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_decodable_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise RecoveryV4SourceError("acquired Recovery V4 source is not a decodable image") from error


def _stream_verified_image(
    location: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    timeout_seconds: float,
) -> None:
    """Write one verified file under a new root; never overwrite a source."""

    if destination.exists() or destination.is_symlink():
        raise RecoveryV4SourceError("acquired Recovery V4 source destination already exists")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RecoveryV4SourceError("unable to create acquired Recovery V4 source parent") from error
    _reject_links_on_existing_path(destination.parent, description="acquired Recovery V4 source destination")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise RecoveryV4SourceError("acquired Recovery V4 source partial destination already exists")
    _approved_hf_https_url(location, description="Recovery V4 source content location")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    partial_identity: os.stat_result | None = None
    digest = hashlib.sha256()
    byte_count = 0
    request = Request(location, headers={"User-Agent": "phone-dino-recovery-v4/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            _approved_hf_https_url(response.geturl(), description="Recovery V4 source content final location")
            descriptor = os.open(str(partial), flags, 0o600)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                offset = 0
                while offset < len(chunk):
                    written = os.write(descriptor, chunk[offset:])
                    if written <= 0:
                        raise OSError("short write")
                    offset += written
            os.fsync(descriptor)
            partial_identity = os.fstat(descriptor)
    except (OSError, URLError) as error:
        raise RecoveryV4SourceError("Recovery V4 source download failed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    actual_sha256 = f"sha256:{digest.hexdigest()}"
    if actual_sha256 != expected_sha256 or byte_count != expected_bytes:
        raise RecoveryV4SourceError("Recovery V4 source content does not match the pinned redirect identity")
    if partial_identity is None:
        raise RecoveryV4SourceError("Recovery V4 source partial identity was not captured")
    path_identity = _regular_nonlink_file_identity(partial, description="acquired Recovery V4 source partial")
    if not _same_file_identity(partial_identity, path_identity):
        raise RecoveryV4SourceError("Recovery V4 source partial changed before image verification")
    _verify_decodable_image(partial)
    verified_sha256, verified_identity = _sha256_file_with_identity(
        partial,
        description="acquired Recovery V4 source partial",
    )
    if verified_sha256 != expected_sha256 or not _same_file_identity(partial_identity, verified_identity):
        raise RecoveryV4SourceError("Recovery V4 source changed while it was verified")
    _reject_links_on_existing_path(destination.parent, description="acquired Recovery V4 source destination")
    if destination.exists() or destination.is_symlink():
        raise RecoveryV4SourceError("acquired Recovery V4 source destination appeared before promotion")
    promotion_identity = _regular_nonlink_file_identity(
        partial,
        description="acquired Recovery V4 source partial",
    )
    if not _same_file_identity(verified_identity, promotion_identity):
        raise RecoveryV4SourceError("Recovery V4 source partial changed before promotion")
    try:
        # A hard link is an atomic no-overwrite promotion within this new
        # source root.  It is intentionally not os.replace().
        os.link(str(partial), str(destination))
    except OSError as error:
        raise RecoveryV4SourceError("unable to promote acquired Recovery V4 source without overwrite") from error
    destination_identity = _regular_nonlink_file_identity(
        destination,
        description="promoted Recovery V4 source destination",
    )
    if not _same_file_identity(promotion_identity, destination_identity):
        raise RecoveryV4SourceError("promoted Recovery V4 source destination identity is unsafe")
    destination_sha256, _ = _sha256_file_with_identity(
        destination,
        description="promoted Recovery V4 source destination",
    )
    if destination_sha256 != expected_sha256 or destination.stat(follow_symlinks=False).st_size != expected_bytes:
        raise RecoveryV4SourceError("promoted Recovery V4 source destination digest is unsafe")
    try:
        os.unlink(partial)
    except OSError as error:
        raise RecoveryV4SourceError("unable to remove promoted Recovery V4 source partial") from error
    _fsync_parent_directory(destination.parent, description="promoted Recovery V4 source")


def _source_record_identity(records: list[dict[str, Any]]) -> str:
    return canonical_json_sha256(records)


def acquire_recovery_v4_sources(
    plan_path: Path,
    non_overlap_ledger_path: Path,
    source_root: Path,
    source_manifest_output_path: Path,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Acquire exactly the 96 predeclared paths and freeze their identities.

    The function does not receive or read ``samples.json``.  It resolves all
    allowlisted redirect identities before creating the new source root, so a
    historical hash collision fails before any source bytes are downloaded.
    """

    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise RecoveryV4SourceError("Recovery V4 source timeout_seconds must be positive")
    # Cheap new-only/reparse preflight precedes all reads and network work.
    # It is repeated after loading the closed ledger to narrow the race before
    # the source root is actually created.
    _assert_new_external_directory_target(source_root, description="Recovery V4 acquired source root")
    _assert_new_external_file_target(
        source_manifest_output_path,
        description="Recovery V4 acquired source manifest output",
    )
    validated_plan = load_recovery_v4_source_acquisition_plan(
        plan_path,
        non_overlap_ledger_path=non_overlap_ledger_path,
    )
    _assert_outside_known_quarantined_cohort_root(
        source_root,
        quarantined_root=validated_plan.quarantined_cohort_root,
        description="Recovery V4 acquired source root",
    )
    _assert_outside_known_quarantined_cohort_root(
        source_manifest_output_path,
        quarantined_root=validated_plan.quarantined_cohort_root,
        description="Recovery V4 acquired source manifest output",
    )
    _assert_new_external_directory_target(source_root, description="Recovery V4 acquired source root")
    _assert_new_external_file_target(
        source_manifest_output_path,
        description="Recovery V4 acquired source manifest output",
    )
    resolved: list[dict[str, Any]] = []
    resolved_hashes: set[str] = set()
    for record in validated_plan.records:
        expected_sha256, expected_bytes, location = _resolve_remote_identity(
            record["sourceRemotePath"], timeout_seconds=float(timeout_seconds)
        )
        if expected_sha256 in validated_plan.excluded_source_hashes:
            raise RecoveryV4SourceError("Recovery V4 allowlist source overlaps historical/quarantined source digest exclusion")
        if expected_sha256 in resolved_hashes:
            raise RecoveryV4SourceError("Recovery V4 allowlist resolved to duplicate source content")
        resolved_hashes.add(expected_sha256)
        source_hex = expected_sha256[7:]
        resolved.append({
            **record,
            "expectedRemoteSha256": expected_sha256,
            "expectedRemoteBytes": expected_bytes,
            "relativePath": f"images/{record['category']}/{source_hex}.png",
            "_location": location,
        })
    _prepare_new_external_directory(source_root, description="Recovery V4 acquired source root")
    manifest_records: list[dict[str, Any]] = []
    for record in resolved:
        relative = PurePosixPath(record["relativePath"])
        destination = source_root.joinpath(*relative.parts)
        _stream_verified_image(
            record["_location"],
            destination,
            expected_sha256=record["expectedRemoteSha256"],
            expected_bytes=record["expectedRemoteBytes"],
            timeout_seconds=float(timeout_seconds),
        )
        manifest_records.append({
            "category": record["category"],
            "role": record["role"],
            "sourceSourceRank": record["rank"],
            "sourceRemotePath": record["sourceRemotePath"],
            "relativePath": record["relativePath"],
            "expectedRemoteSha256": record["expectedRemoteSha256"],
            "expectedRemoteBytes": record["expectedRemoteBytes"],
            "sourceSha256": record["expectedRemoteSha256"],
        })
    manifest_records.sort(key=lambda record: (RECOVERY_V4_CATEGORIES.index(record["category"]), record["sourceSourceRank"]))
    document: dict[str, Any] = {
        "schemaVersion": RECOVERY_V4_SOURCE_MANIFEST_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": SOURCE_MANIFEST_PURPOSE,
        "sourceAcquisitionPlanFileSha256": validated_plan.plan_file_sha256,
        "sourceAcquisitionPlanDeclaredSha256": validated_plan.plan["recoveryV4SourceAcquisitionPlanSha256"],
        "allowlistPolicyFileSha256": validated_plan.allowlist_policy_file_sha256,
        "allowlistPolicyDeclaredSha256": validated_plan.allowlist_policy["recoveryV4SourceAllowlistPolicySha256"],
        "nonOverlapLedgerFileSha256": validated_plan.non_overlap_ledger_file_sha256,
        "nonOverlapLedgerDeclaredSha256": validated_plan.non_overlap_ledger["recoveryV4NonOverlapMetadataLedgerSha256"],
        "records": manifest_records,
        "sourceRecordIdentitySha256": _source_record_identity(manifest_records),
    }
    document["recoveryV4AcquiredSourceManifestSha256"] = _document_digest(
        document, "recoveryV4AcquiredSourceManifestSha256"
    )
    _write_new_external_json(source_manifest_output_path, document, description="Recovery V4 acquired source manifest output")
    return document


def load_recovery_v4_acquired_source_manifest(
    manifest_path: Path,
    *,
    plan_path: Path,
    non_overlap_ledger_path: Path,
) -> tuple[dict[str, Any], str]:
    """Validate an acquired-source manifest without opening source bytes."""

    validated_plan = load_recovery_v4_source_acquisition_plan(
        plan_path,
        non_overlap_ledger_path=non_overlap_ledger_path,
    )
    document, file_sha256, _ = _read_json_file(
        manifest_path,
        description="Recovery V4 acquired source manifest",
        external=True,
    )
    _require_exact_fields(document, name="Recovery V4 acquired source manifest", fields=_MANIFEST_FIELDS)
    if (
        document.get("schemaVersion") != RECOVERY_V4_SOURCE_MANIFEST_SCHEMA
        or document.get("authoritative") is not False
        or document.get("productionAuthorized") is not False
        or document.get("purpose") != SOURCE_MANIFEST_PURPOSE
    ):
        raise RecoveryV4SourceError("Recovery V4 acquired source manifest is unsafe")
    declared = _require_sha256(
        document.get("recoveryV4AcquiredSourceManifestSha256"),
        name="Recovery V4 acquired source manifest declared digest",
    )
    if not hmac.compare_digest(declared, _document_digest(document, "recoveryV4AcquiredSourceManifestSha256")):
        raise RecoveryV4SourceError("Recovery V4 acquired source manifest declared digest does not match")
    expected_bindings = {
        "sourceAcquisitionPlanFileSha256": validated_plan.plan_file_sha256,
        "sourceAcquisitionPlanDeclaredSha256": validated_plan.plan["recoveryV4SourceAcquisitionPlanSha256"],
        "allowlistPolicyFileSha256": validated_plan.allowlist_policy_file_sha256,
        "allowlistPolicyDeclaredSha256": validated_plan.allowlist_policy["recoveryV4SourceAllowlistPolicySha256"],
        "nonOverlapLedgerFileSha256": validated_plan.non_overlap_ledger_file_sha256,
        "nonOverlapLedgerDeclaredSha256": validated_plan.non_overlap_ledger["recoveryV4NonOverlapMetadataLedgerSha256"],
    }
    for field, expected in expected_bindings.items():
        if document.get(field) != expected:
            raise RecoveryV4SourceError(f"Recovery V4 acquired source manifest {field} binding does not match")
    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise RecoveryV4SourceError("Recovery V4 acquired source manifest records are invalid")
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for raw_record, plan_record in zip(raw_records, validated_plan.records, strict=True):
        record = _require_exact_fields(raw_record, name="Recovery V4 acquired source manifest record", fields=_SOURCE_MANIFEST_RECORD_FIELDS)
        category = _require_string(record.get("category"), name="Recovery V4 acquired source category")
        role = _require_string(record.get("role"), name="Recovery V4 acquired source role")
        rank = _require_positive_int(record.get("sourceSourceRank"), name="Recovery V4 acquired source rank")
        remote_path = _safe_remote_path(record.get("sourceRemotePath"), name="Recovery V4 acquired sourceRemotePath")
        relative_path = _safe_remote_path(record.get("relativePath"), name="Recovery V4 acquired relativePath")
        expected_hash = _require_sha256(record.get("expectedRemoteSha256"), name="Recovery V4 acquired expected digest")
        source_hash = _require_sha256(record.get("sourceSha256"), name="Recovery V4 acquired source digest")
        expected_bytes = _require_positive_int(record.get("expectedRemoteBytes"), name="Recovery V4 acquired expected bytes")
        if expected_hash != source_hash or source_hash in validated_plan.excluded_source_hashes:
            raise RecoveryV4SourceError("Recovery V4 acquired source digest is unsafe")
        if remote_path in seen_paths or source_hash in seen_hashes:
            raise RecoveryV4SourceError("Recovery V4 acquired source manifest is duplicated")
        seen_paths.add(remote_path)
        seen_hashes.add(source_hash)
        expected_relative_path = f"images/{category}/{source_hash[7:]}.png"
        if (
            category != plan_record["category"]
            or role != plan_record["role"]
            or rank != plan_record["rank"]
            or remote_path != plan_record["sourceRemotePath"]
            or relative_path != expected_relative_path
        ):
            raise RecoveryV4SourceError("Recovery V4 acquired source manifest record does not match the closed plan")
        normalized.append({
            "category": category,
            "role": role,
            "sourceSourceRank": rank,
            "sourceRemotePath": remote_path,
            "relativePath": relative_path,
            "expectedRemoteSha256": expected_hash,
            "expectedRemoteBytes": expected_bytes,
            "sourceSha256": source_hash,
        })
    if len(normalized) != len(validated_plan.records):
        raise RecoveryV4SourceError("Recovery V4 acquired source manifest record count does not match the closed plan")
    if normalized != raw_records:
        raise RecoveryV4SourceError("Recovery V4 acquired source manifest records must be canonical")
    if document.get("sourceRecordIdentitySha256") != _source_record_identity(normalized):
        raise RecoveryV4SourceError("Recovery V4 acquired source manifest source identity does not match")
    return document, file_sha256
