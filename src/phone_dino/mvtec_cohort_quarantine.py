"""Fail-closed quarantine guard for externally exposed MVTec cohorts.

The guard is intentionally JSON-only.  It validates a pinned immutable
incident record and the closed normal-holdout manifest *before* a caller may
ask a phase-safe reader for FIT (or any other) image bytes.  It is not a
mechanism for clearing a quarantine: the known exposed cohort remains pinned
in this module so a missing, replaced, or self-consistent forged incident
record cannot silently re-enable it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from phone_dino import mvtec_normal_holdout as holdout


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

COHORT_QUARANTINE_INCIDENT_SCHEMA = "phone-dino.mvtec-ad-cohort-quarantine-incident/1.0"
COHORT_QUARANTINE_INCIDENT_PURPOSE = "OFFLINE_MVTEC_COHORT_QUARANTINE"
COHORT_QUARANTINE_STATUS = "QUARANTINED"
COHORT_QUARANTINE_SCOPE = "NORMAL_HOLDOUT_MANIFEST_IDENTITY"
COHORT_QUARANTINE_REASON = "UNBOUNDED_RECURSIVE_BYTE_HASH_EXPOSURE"

# These values identify the exact frozen fresh_normal_holdout_v1 manifest that
# was exposed during the abandoned V3 preflight.  They are intentionally
# compiled into the guard as well as recorded externally.  A caller cannot
# remove or substitute the external record to make this cohort eligible again.
KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_FILE_SHA256 = (
    "sha256:0034e045001787a6ce35042701cb470a97c03ff72117311ff7525fd5d9106b18"
)
KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_DECLARED_SHA256 = (
    "sha256:51a359f5d579a99321dc33687fecc6d9a8db92fb7f921960bbb6898c23e2e74e"
)
KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_ID = "fresh_normal_holdout_v1-2026-08-12-recursive-byte-hash-exposure"

INCIDENT_FIELDS = {
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


class CohortQuarantineError(ValueError):
    """Raised when an incident is unsafe or a cohort is quarantined."""


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CohortQuarantineError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise CohortQuarantineError(f"{name} must be a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise CohortQuarantineError(f"{name} must be a SHA-256 digest") from error
    return digest


def _validate_json_value(value: object, *, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CohortQuarantineError(f"{name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CohortQuarantineError(f"{name} contains a non-string JSON key")
            _validate_json_value(item, name=f"{name}.{key}")
        return
    raise CohortQuarantineError(f"{name} contains a value that is not JSON-compatible")


def canonical_json_sha256(document: object) -> str:
    """Return a finite, deterministic digest for JSON-only incident evidence."""

    _validate_json_value(document, name="canonical JSON value")
    payload = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return canonical_json_sha256(unsigned)


def known_fresh_normal_holdout_v1_incident_document() -> dict[str, Any]:
    """Return the sole immutable incident record accepted for this quarantine."""

    document: dict[str, Any] = {
        "schemaVersion": COHORT_QUARANTINE_INCIDENT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": COHORT_QUARANTINE_INCIDENT_PURPOSE,
        "quarantineStatus": COHORT_QUARANTINE_STATUS,
        "scope": COHORT_QUARANTINE_SCOPE,
        "reason": COHORT_QUARANTINE_REASON,
        "incidentId": KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_ID,
        "cohortManifestSchemaVersion": holdout.NORMAL_HOLDOUT_SCHEMA,
        "cohortManifestFileSha256": KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_FILE_SHA256,
        "cohortManifestDeclaredSha256": KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_DECLARED_SHA256,
    }
    document["cohortQuarantineIncidentSha256"] = _document_digest(document, "cohortQuarantineIncidentSha256")
    return document


# This pin is deliberately separate from the self-digest in an external file.
# Keeping it literal means an edited record template cannot silently redefine
# the approved incident identity; validation below also recomputes the record
# self-digest.
KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_DECLARED_SHA256 = (
    "sha256:be690e112ae28f04a69db572a7b9931d862fcac1da9653e92a99ad5995fbf2d4"
)


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CohortQuarantineError(f"unable to inspect {path}") from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(reparse_flag and attributes & reparse_flag)


def _reject_links_on_existing_path(path: Path, *, description: str) -> None:
    current = path
    while True:
        try:
            exists = current.exists() or current.is_symlink()
        except OSError as error:
            raise CohortQuarantineError(f"unable to inspect {description}") from error
        if exists and _is_link_or_reparse_point(current):
            raise CohortQuarantineError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _is_under(root: Path, candidate: Path) -> bool:
    try:
        root_text = os.path.normcase(os.path.abspath(str(root)))
        candidate_text = os.path.normcase(os.path.abspath(str(candidate)))
        return os.path.commonpath((root_text, candidate_text)) == root_text
    except (OSError, ValueError):
        return False


def _require_external_regular_file(path: Path, *, description: str, repository_root: Path) -> None:
    if not isinstance(path, Path):
        raise CohortQuarantineError(f"{description} path must be a Path")
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise CohortQuarantineError(f"{description} must stay outside the Git working tree")
    _reject_links_on_existing_path(path, description=description)
    try:
        if not path.is_file() or _is_link_or_reparse_point(path):
            raise CohortQuarantineError(f"{description} must be a regular non-link file")
    except OSError as error:
        raise CohortQuarantineError(f"unable to inspect {description}") from error


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _read_external_json(path: Path, *, description: str, repository_root: Path) -> tuple[dict[str, Any], str]:
    """Read one external JSON file while rejecting links and obvious swaps."""

    _require_external_regular_file(path, description=description, repository_root=repository_root)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(str(path), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CohortQuarantineError(f"{description} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = path.stat(follow_symlinks=False)
    except CohortQuarantineError:
        raise
    except OSError as error:
        raise CohortQuarantineError(f"unable to read {description}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not _same_file_identity(before, after) or not _same_file_identity(before, path_after):
        raise CohortQuarantineError(f"{description} changed while it was read")
    _reject_links_on_existing_path(path, description=description)
    raw = b"".join(chunks)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise CohortQuarantineError(f"{description} contains a duplicate JSON key: {key}")
            document[key] = value
        return document

    def reject_nonfinite(value: str) -> Any:
        raise CohortQuarantineError(f"{description} contains a non-finite JSON value: {value}")

    try:
        document = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CohortQuarantineError(f"unable to parse {description} as JSON") from error
    if not isinstance(document, dict):
        raise CohortQuarantineError(f"{description} must be a JSON object")
    return document, f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _require_exact_fields(document: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise CohortQuarantineError(f"{name} must be an object")
    missing = fields.difference(document)
    unknown = set(document).difference(fields)
    if missing or unknown:
        detail: list[str] = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unsupported {', '.join(sorted(unknown))}")
        raise CohortQuarantineError(f"{name} has {'; '.join(detail)} fields")
    return document


def _validate_known_incident(document: dict[str, Any]) -> None:
    _require_exact_fields(document, name="cohort quarantine incident", fields=INCIDENT_FIELDS)
    if document.get("schemaVersion") != COHORT_QUARANTINE_INCIDENT_SCHEMA:
        raise CohortQuarantineError("cohort quarantine incident schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise CohortQuarantineError("cohort quarantine incident must be non-authoritative and non-production")
    if document.get("purpose") != COHORT_QUARANTINE_INCIDENT_PURPOSE:
        raise CohortQuarantineError("cohort quarantine incident purpose is unsafe")
    if document.get("quarantineStatus") != COHORT_QUARANTINE_STATUS:
        raise CohortQuarantineError("cohort quarantine incident must remain quarantined")
    if document.get("scope") != COHORT_QUARANTINE_SCOPE:
        raise CohortQuarantineError("cohort quarantine incident scope is unsafe")
    if document.get("reason") != COHORT_QUARANTINE_REASON:
        raise CohortQuarantineError("cohort quarantine incident reason is unsafe")
    if document.get("incidentId") != KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_ID:
        raise CohortQuarantineError("cohort quarantine incident identity is unsafe")
    if document.get("cohortManifestSchemaVersion") != holdout.NORMAL_HOLDOUT_SCHEMA:
        raise CohortQuarantineError("cohort quarantine incident target schema is unsupported")
    for field, expected in (
        ("cohortManifestFileSha256", KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_FILE_SHA256),
        ("cohortManifestDeclaredSha256", KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_DECLARED_SHA256),
    ):
        value = _require_sha256(document.get(field), name=f"cohort quarantine incident {field}")
        if not hmac.compare_digest(value, expected):
            raise CohortQuarantineError("cohort quarantine incident does not bind the known exposed cohort")
    declared = _require_sha256(
        document.get("cohortQuarantineIncidentSha256"), name="cohort quarantine incident declared digest"
    )
    if not hmac.compare_digest(declared, _document_digest(document, "cohortQuarantineIncidentSha256")):
        raise CohortQuarantineError("cohort quarantine incident declared digest does not match")
    if not hmac.compare_digest(declared, KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_DECLARED_SHA256):
        raise CohortQuarantineError("cohort quarantine incident does not match the immutable known incident")


def _load_normal_holdout_identity(
    path: Path, *, repository_root: Path
) -> tuple[str, str]:
    """Validate a closed holdout manifest without opening any source image."""

    document, file_sha256 = _read_external_json(
        path,
        description="V3 parent normal holdout manifest",
        repository_root=repository_root,
    )
    try:
        holdout._validate_closed_normal_holdout_document(document)
    except holdout.NormalHoldoutError as error:
        raise CohortQuarantineError("V3 parent normal holdout manifest is unsafe") from error
    declared = _require_sha256(
        document.get("normalHoldoutManifestSha256"), name="V3 parent normal holdout manifest declared digest"
    )
    return file_sha256, declared


def assert_v3_parent_holdout_not_quarantined(
    parent_holdout_path: Path,
    *,
    quarantine_incident_path: Path,
    expected_quarantine_incident_sha256: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[str, str]:
    """Fail closed before V3 opens FIT/query bytes from a parent cohort.

    The independently supplied pin and the built-in incident pin must both
    match.  Thus missing, swapped, or self-consistent forged external incident
    files stop the run; they cannot be interpreted as an implicit release.
    """

    if not isinstance(parent_holdout_path, Path):
        raise CohortQuarantineError("V3 parent normal holdout path must be a Path")
    if not isinstance(quarantine_incident_path, Path):
        raise CohortQuarantineError("V3 cohort quarantine incident path must be a Path")
    # Keep the durable incident evidence outside the cohort directory that it
    # quarantines.  The built-in pin already prevents a release-by-rewrite,
    # but this avoids treating a cohort-local sidecar as an authoritative
    # ledger in the first place.
    if _is_under(parent_holdout_path.parent, quarantine_incident_path):
        raise CohortQuarantineError("V3 cohort quarantine incident must stay outside the parent cohort root")
    expected = _require_sha256(
        expected_quarantine_incident_sha256,
        name="expected cohort quarantine incident digest",
    )
    if not hmac.compare_digest(expected, KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_DECLARED_SHA256):
        raise CohortQuarantineError("expected cohort quarantine incident digest is not the approved immutable pin")
    incident, _incident_file_sha256 = _read_external_json(
        quarantine_incident_path,
        description="V3 cohort quarantine incident",
        repository_root=repository_root,
    )
    _validate_known_incident(incident)
    manifest_file_sha256, manifest_declared_sha256 = _load_normal_holdout_identity(
        parent_holdout_path,
        repository_root=repository_root,
    )
    # Either identity is enough to denote the same frozen logical cohort.  In
    # particular, a whitespace-only copy cannot evade the declared identity.
    if (
        hmac.compare_digest(manifest_file_sha256, KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_FILE_SHA256)
        or hmac.compare_digest(manifest_declared_sha256, KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_DECLARED_SHA256)
    ):
        raise CohortQuarantineError(
            "V3 parent normal holdout is permanently quarantined after unbounded recursive byte-hash exposure"
        )
    return manifest_file_sha256, manifest_declared_sha256
