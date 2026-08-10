"""One-time selection for the reserve-successor V2 normal-only screen.

The first fresh cohort has already consumed its selection partition and locked
``NO_ELIGIBLE_CONFIGURATION``.  This module is intentionally a new protocol:
it consumes only a sealed reserve-derived successor envelope and never opens a
parent confirmation image or any remaining successor reserve image.  The
contract, claim, and lock are JSON-only.  The only query phase is one aggregate
``NORMAL_SELECTION`` observation, protected by an fsynced O_EXCL receipt.

This is offline research evidence only.  It does not calibrate a device,
promote a model, produce a defect result, or initiate confirmation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from phone_dino import mvtec_normal_successor as successor
from phone_dino import mvtec_fresh_normal_selection as parent_protocol
from phone_dino import mvtec_successor_evaluator_v2 as evaluator
from phone_dino import mvtec_successor_fit_augmentation_v2 as augmentation


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SUCCESSOR_V2_SELECTION_CONTRACT_SCHEMA = "phone-dino.mvtec-ad-reserve-successor-v2-selection-contract/1.0"
SUCCESSOR_V2_SELECTION_CLAIM_SCHEMA = "phone-dino.mvtec-ad-reserve-successor-v2-selection-claim/1.0"
SUCCESSOR_V2_SELECTION_RECEIPT_SCHEMA = "phone-dino.mvtec-ad-reserve-successor-v2-selection-receipt/1.0"
SUCCESSOR_V2_SELECTION_OBSERVATION_SCHEMA = "phone-dino.mvtec-ad-reserve-successor-v2-selection-observation/1.0"
SUCCESSOR_V2_SELECTION_LOCK_SCHEMA = "phone-dino.mvtec-ad-reserve-successor-v2-selection-lock/1.0"
SUCCESSOR_V2_DEVELOPMENT_EVIDENCE_LEDGER_SCHEMA = "phone-dino.mvtec-ad-reserve-successor-v2-development-evidence-ledger/1.0"

CONTRACT_PURPOSE = "OFFLINE_MVTEC_RESERVE_SUCCESSOR_V2_SELECTION_CONTRACT"
CLAIM_PURPOSE = "OFFLINE_MVTEC_RESERVE_SUCCESSOR_V2_SELECTION_CLAIM"
RECEIPT_PURPOSE = "OFFLINE_MVTEC_RESERVE_SUCCESSOR_V2_SELECTION_RECEIPT"
OBSERVATION_PURPOSE = "OFFLINE_MVTEC_RESERVE_SUCCESSOR_V2_SELECTION_OBSERVATION"
LOCK_PURPOSE = "OFFLINE_MVTEC_RESERVE_SUCCESSOR_V2_SELECTION_LOCK"
DEVELOPMENT_EVIDENCE_LEDGER_PURPOSE = "OFFLINE_MVTEC_RESERVE_SUCCESSOR_V2_PRESELECTION_DEVELOPMENT_EVIDENCE"

CONTRACT_PHASE = "SUCCESSOR_V2_SELECTION_CONTRACT"
CLAIM_PHASE = "SUCCESSOR_V2_SELECTION_CLAIM"
RECEIPT_PHASE = "SUCCESSOR_V2_SELECTION_RECEIPT"
OBSERVATION_PHASE = "SUCCESSOR_V2_SELECTION_OBSERVATION"
LOCK_PHASE = "SUCCESSOR_V2_SELECTION_LOCK"
DEVELOPMENT_EVIDENCE_LEDGER_PHASE = "SUCCESSOR_V2_PRESELECTION_DEVELOPMENT_EVIDENCE"

BLIND_POLICY = successor.FRESH_NORMAL_SUCCESSOR_BLIND_POLICY
DELEGATION_POLICY = successor.FRESH_NORMAL_SUCCESSOR_DELEGATION_POLICY
RESULT_LABEL = successor.FRESH_NORMAL_SUCCESSOR_RESULT_LABEL
INDEPENDENCE_LABEL = successor.FRESH_NORMAL_SUCCESSOR_INDEPENDENCE_LABEL
NO_PERSISTENT_QUERY_CACHE = "NO_PERSISTENT_QUERY_CACHE"
CONSUMPTION_REGISTRY_SCHEMA = "phone-dino.mvtec-ad-reserve-successor-v2-selection-consumption-registry/1.0"
DEVELOPMENT_EVIDENCE_LEDGER_SCOPE = "PRESELECTION_FROZEN_SUCCESSOR_V2_CONTRACT_BINDINGS"
PUSHED_GIT_AUDIT_ONLY = "PUSHED_GIT_AUDIT_ONLY"
CANONICAL_GIT_AUDIT_REMOTE_URL = "https://github.com/hctsaik/phone_dino.git"
REQUIRED_GIT_AUDIT_REMOTE_REF = "refs/heads/master"
CANONICAL_GIT_AUDIT_OBJECT_FORMAT = "sha1"
DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH = "docs/mvtec_ad_successor_v2_development_evidence_ledger.json"

SELECTION_INPUT = {
    "partition": "NORMAL_SELECTION",
    "kind": "NOMINAL",
    "defect": "good",
    "rawOnly": True,
    "oneTimeClaimRequired": True,
}
SELECTION_GATES = {
    "maxAboveThresholdCount": 1,
    "maxP95ScoreMinusThreshold": 0.05,
    "maxMaximumScoreMinusThreshold": 0.05,
}
SELECTION_OBJECTIVE = {
    "algorithm": "LEXICOGRAPHIC_MINIMIZE_SUCCESSOR_V2_NORMAL_SELECTION_EXCESS_V1",
    "terms": [
        "worstMaximumScoreMinusThreshold",
        "worstAboveThresholdRate",
        "meanP95ScoreMinusThreshold",
        "candidateIdAscending",
    ],
}

CONTRACT_FIELDS = {
    "schemaVersion", "authoritative", "productionAuthorized", "purpose", "phase", "blindPolicy",
    "resultLabel", "independenceLabel", "delegationPolicy", "selectionInput", "parentEvidence",
    "successorPlanFileSha256", "successorPlanDeclaredSha256", "successorEnvelopeFileSha256",
    "successorEnvelopeDeclaredSha256", "successorEnvelopeSelectionIdentitySha256", "successorSelectionInputs", "successorSelectionInputIdentitySha256",
    "augmentation", "prototypeInputIdentities", "prototypeInputCounts", "developmentEvidenceLedger", "selectionProtocolModuleSha256", "consumptionRegistry", "featureExtractor", "featureExtractorIdentitySha256", "candidateReports",
    "candidateUniverseIdentitySha256", "selectionGates", "selectionObjective", "contractSha256",
}
CLAIM_FIELDS = {
    "schemaVersion", "authoritative", "productionAuthorized", "purpose", "phase", "blindPolicy",
    "resultLabel", "delegationPolicy", "selectionInput", "claimSlot", "contractFileSha256",
    "contractDeclaredSha256", "parentHoldoutFileSha256", "successorEnvelopeDeclaredSha256",
    "successorSelectionInputIdentitySha256", "candidateUniverseIdentitySha256", "claimSha256",
}
RECEIPT_FIELDS = {
    "schemaVersion", "authoritative", "productionAuthorized", "purpose", "phase", "blindPolicy",
    "resultLabel", "delegationPolicy", "selectionInput", "receiptSlot", "contractFileSha256",
    "contractDeclaredSha256", "selectionClaimFileSha256", "selectionClaimDeclaredSha256",
    "successorEnvelopeDeclaredSha256", "successorSelectionInputIdentitySha256",
    "candidateUniverseIdentitySha256", "featureExtractorIdentitySha256", "prototypeInputIdentities",
    "prototypeInputCounts", "selectionReceiptSha256",
}
OBSERVATION_FIELDS = {
    "schemaVersion", "authoritative", "productionAuthorized", "purpose", "phase", "blindPolicy",
    "resultLabel", "independenceLabel", "delegationPolicy", "selectionInput", "contractFileSha256",
    "contractDeclaredSha256", "selectionClaimFileSha256", "selectionClaimDeclaredSha256",
    "selectionReceiptFileSha256", "selectionReceiptDeclaredSha256", "parentEvidence",
    "successorEnvelopeFileSha256", "successorEnvelopeDeclaredSha256", "augmentation",
    "featureExtractor", "featureExtractorIdentitySha256", "successorSelectionInputs",
    "successorSelectionInputIdentitySha256", "candidateObservations", "normalOnlyEvidence", "execution",
    "selectionObservationSha256",
}
CANDIDATE_BINDING_FIELDS = {
    "candidateId", "developmentReportFileSha256", "developmentReportDeclaredSha256", "candidateConfiguration",
    "candidateConfigurationSha256", "prototypeInputPolicy", "featureInputIdentitySha256",
    "calibrationInputIdentitySha256", "featureExtractorIdentitySha256", "thresholds", "thresholdsIdentitySha256", "augmentationManifestFileSha256",
    "augmentationManifestDeclaredSha256", "augmentationRecipeFileSha256",
}
DEVELOPMENT_EVIDENCE_LEDGER_FIELDS = {
    "schemaVersion", "authoritative", "productionAuthorized", "purpose", "phase", "blindPolicy", "resultLabel",
    "independenceLabel", "delegationPolicy", "selectionInput", "ledgerScope", "trustMode",
    "contractBindingProjection", "contractBindingProjectionSha256", "developmentEvidenceLedgerSha256",
}
DEVELOPMENT_EVIDENCE_PROJECTION_FIELDS = {
    "augmentation", "prototypeInputIdentities", "prototypeInputCounts", "featureExtractor", "featureExtractorIdentitySha256",
    "selectionProtocolModuleSha256", "candidateBindings", "candidateUniverseIdentitySha256",
}
DEVELOPMENT_EVIDENCE_GIT_FIELDS = {
    "mode", "canonicalRemoteUrl", "requiredRemoteRef", "gitObjectFormat", "gitCommitOid", "repositoryPath",
    "gitBlobOid", "ledgerBlobSha256", "ledgerDeclaredSha256", "ledgerProjectionSha256",
}
CANDIDATE_OBSERVATION_FIELDS = {
    "candidateId", "candidateConfiguration", "candidateConfigurationSha256", "thresholds",
    "thresholdsIdentitySha256", "selectionScores", "categoryMetrics",
}
SELECTION_SCORE_FIELDS = {
    "caseId", "category", "partition", "kind", "defect", "sourceSha256", "score",
    "maxPatchDistance", "meanNearestPatchDistance",
}
CATEGORY_METRIC_FIELDS = {
    "queryCount", "aboveThresholdCount", "aboveThresholdRate", "p95Score", "maximumScore",
    "p95ScoreMinusThreshold", "maximumScoreMinusThreshold",
}
LOCK_FIELDS = {
    "schemaVersion", "authoritative", "productionAuthorized", "purpose", "phase", "blindPolicy",
    "resultLabel", "delegationPolicy", "contractFileSha256", "contractDeclaredSha256", "selectionProtocolModuleSha256",
    "selectionClaimFileSha256", "selectionClaimDeclaredSha256", "selectionReceiptFileSha256",
    "selectionReceiptDeclaredSha256", "selectionObservationFileSha256", "selectionObservationDeclaredSha256",
    "candidateEvaluations", "decision", "selectionLockSha256",
}
LOCK_CANDIDATE_FIELDS = {
    "candidateId", "categoryMetrics", "gatePassed", "gateRejectionReasons", "objectiveValues",
}
LOCK_DECISION_FIELDS = {
    "state", "selectedCandidateId", "resultScope", "automaticProductionPromotion", "automaticConfirmation",
}
NORMAL_INPUT_FIELDS = {
    "caseId", "category", "partition", "kind", "defect", "sourceSha256", "sourceGroupId",
    "acquisitionStratum", "expectedRemoteSha256", "expectedRemoteBytes",
}
CONSUMPTION_REGISTRY_FIELDS = {"schemaVersion", "root", "selectionSlotKey"}
AUGMENTATION_BINDING_FIELDS = {
    "manifestFileSha256", "manifestDeclaredSha256", "recipeFileSha256", "successorFitIdentitySha256", "variantsPerParent",
}
PROTOTYPE_POLICY_KEYS = {evaluator.RAW_FIT_ONLY, evaluator.RAW_FIT_PLUS_AUGMENTATION_R3}
PROTOTYPE_INPUT_COUNTS = {
    evaluator.RAW_FIT_ONLY: 36,
    evaluator.RAW_FIT_PLUS_AUGMENTATION_R3: 144,
}
NORMAL_ONLY_EVIDENCE_FIELDS = {
    "prototypeInputPartitions", "queryInputCount", "queryInputPartitions", "queryInputKinds", "blindInputCount",
    "anomalyInputCount", "maskInputCount", "parentConfirmationInputCount", "remainingReserveInputCount",
    "persistentQueryCache", "queryCachePolicy",
}
OBSERVATION_EXECUTION_FIELDS = {"selectionModuleSha256", "phaseTimingsSeconds"}
OBSERVATION_TIMING_FIELDS = {"preflightSeconds", "receiptCommitSeconds", "queryFeatureSeconds", "scoringSeconds", "totalElapsedSeconds"}


class SuccessorV2SelectionError(ValueError):
    """Raised for an unsafe reserve-successor V2 selection artifact or phase."""


def canonical_json_sha256(value: Any) -> str:
    try:
        return successor.canonical_json_sha256(value)
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return canonical_json_sha256(unsigned)


def _require_exact_fields(value: object, *, name: str, required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SuccessorV2SelectionError(f"{name} must be an object")
    missing = required.difference(value)
    unknown = set(value).difference(required)
    if missing:
        raise SuccessorV2SelectionError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise SuccessorV2SelectionError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")
    return value


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SuccessorV2SelectionError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise SuccessorV2SelectionError(f"{name} must be a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise SuccessorV2SelectionError(f"{name} must be a SHA-256 digest") from error
    return digest


def _require_finite(value: object, *, name: str, minimum: float = -2.0, maximum: float = 2.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SuccessorV2SelectionError(f"{name} must be a finite number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise SuccessorV2SelectionError(f"{name} must be between {minimum} and {maximum}")
    return result


def _same_number(left: float, right: object) -> bool:
    return isinstance(right, (int, float)) and not isinstance(right, bool) and math.isclose(
        left, float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _read_json(path: Path, *, description: str, repository_root: Path) -> tuple[dict[str, Any], str]:
    try:
        return successor._read_external_json(path, description=description, repository_root=repository_root)
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error


def _sanitized_git_environment(*, global_config_path: Path | None = None) -> dict[str, str]:
    """Return a non-interactive Git environment without caller-controlled Git state.

    The pushed-ledger resolver has a fixed remote URL, but Git's ``url.*.insteadOf``
    setting can rewrite that URL before fetch.  Inherited ``GIT_CONFIG_*`` variables
    can inject such a setting even when the user has no local Git configuration.
    Start from the ambient OS environment only for normal process/network support
    and strip every Git override case-insensitively.  Local, read-only Git checks
    retain normal repository/global configuration because they inspect only named
    HEAD objects, never a remote URL or checkout bytes.  The resolver supplies an
    empty, temporary global config file; that remote-facing path also disables
    system configuration and never depends on a mutable home directory.
    """

    environment = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
    if global_config_path is not None:
        environment.update({
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": str(global_config_path),
            "GIT_ATTR_NOSYSTEM": "1",
        })
    return environment


def _git_process(
    arguments: list[str], *, repository_root: Path, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run Git without a shell and without inherited ``GIT_*`` overrides.

    Ordinary local configuration can remain for raw local-HEAD inspection;
    callers operating on the temporary remote-audit repository pass its fully
    isolated environment explicitly.
    """

    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=45,
            env=environment if environment is not None else _sanitized_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SuccessorV2SelectionError("unable to run the required Git audit command") from error


def _git_text(
    arguments: list[str], *, repository_root: Path, description: str, environment: dict[str, str] | None = None
) -> str:
    result = _git_process(arguments, repository_root=repository_root, environment=environment)
    if result.returncode != 0:
        raise SuccessorV2SelectionError(f"Git audit could not resolve {description}")
    try:
        value = result.stdout.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise SuccessorV2SelectionError(f"Git audit returned a non-text {description}") from error
    if not value or "\n" in value:
        raise SuccessorV2SelectionError(f"Git audit returned an invalid {description}")
    return value


def _require_git_oid(value: object, *, object_format: str, name: str) -> str:
    if object_format not in {"sha1", "sha256"}:
        raise SuccessorV2SelectionError(f"{name} Git object format is unsupported")
    oid = _require_string(value, name=name)
    expected_length = 40 if object_format == "sha1" else 64
    if len(oid) != expected_length:
        raise SuccessorV2SelectionError(f"{name} must be a {object_format} Git object ID")
    try:
        int(oid, 16)
    except ValueError as error:
        raise SuccessorV2SelectionError(f"{name} must be a {object_format} Git object ID") from error
    return oid.lower()


def _git_object_format(*, repository_root: Path, environment: dict[str, str] | None = None) -> str:
    value = _git_text(
        ["rev-parse", "--show-object-format"], repository_root=repository_root, description="object format",
        environment=environment,
    )
    if value not in {"sha1", "sha256"}:
        raise SuccessorV2SelectionError("Git audit repository uses an unsupported object format")
    return value


def _ledger_repository_path(path: Path, *, repository_root: Path) -> str:
    """Require the one reviewed ledger path without resolving link ancestry first."""

    if not isinstance(path, Path):
        raise SuccessorV2SelectionError("development evidence ledger path must be a Path")
    _reject_links(path, description="development evidence ledger")
    _reject_links(repository_root, description="development evidence repository")
    if not path.exists() or not path.is_file():
        raise SuccessorV2SelectionError("development evidence ledger source file is missing")
    try:
        relative = path.resolve(strict=True).relative_to(repository_root.resolve(strict=True)).as_posix()
    except (OSError, RuntimeError, ValueError) as error:
        raise SuccessorV2SelectionError("development evidence ledger must be under the Git repository") from error
    if relative != DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH:
        raise SuccessorV2SelectionError("development evidence ledger must use the fixed reviewed repository path")
    return relative


def _git_blob_from_commit(
    *, repository_root: Path, commit_oid: str, repository_path: str, object_format: str,
    environment: dict[str, str] | None = None,
) -> tuple[str, bytes]:
    tree = _git_process(
        ["ls-tree", "-z", commit_oid, "--", repository_path], repository_root=repository_root, environment=environment
    )
    if tree.returncode != 0:
        raise SuccessorV2SelectionError("Git audit could not resolve the ledger blob from its recorded commit")
    entries = [entry for entry in tree.stdout.split(b"\0") if entry]
    if len(entries) != 1 or b"\t" not in entries[0]:
        raise SuccessorV2SelectionError("recorded Git commit does not contain exactly one ledger entry")
    header, actual_path = entries[0].split(b"\t", 1)
    fields = header.split(b" ")
    if len(fields) != 3:
        raise SuccessorV2SelectionError("recorded Git ledger tree entry is malformed")
    mode, kind, blob_bytes = fields
    try:
        resolved_path = actual_path.decode("utf-8")
        blob_oid = blob_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise SuccessorV2SelectionError("recorded Git ledger tree entry is not text-safe") from error
    if resolved_path != repository_path or kind != b"blob" or mode != b"100644":
        raise SuccessorV2SelectionError("recorded Git ledger path must resolve to a regular blob")
    blob_oid = _require_git_oid(blob_oid, object_format=object_format, name="development evidence ledger blob")
    blob = _git_process(["cat-file", "blob", blob_oid], repository_root=repository_root, environment=environment)
    if blob.returncode != 0:
        raise SuccessorV2SelectionError("Git audit could not load the recorded ledger blob")
    return blob_oid, bytes(blob.stdout)


def _parse_ledger_blob(raw: bytes) -> dict[str, Any]:
    try:
        # Reuse the successor protocol's strict byte parser: duplicate keys and
        # NaN/Infinity are rejected before a ledger can become an authority.
        return successor._parse_json_bytes(raw, description="recorded development evidence ledger blob")
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error


def _resolve_pushed_git_ledger_blob(
    *, commit_oid: str, repository_path: str, canonical_remote_url: str, required_remote_ref: str, object_format: str
) -> tuple[str, bytes]:
    """Fetch one public ref into a disposable bare repository and read its raw blob.

    The resolver never fetches into the user's worktree, never checks out, and
    deliberately reads Git's object bytes rather than CRLF-filtered file bytes.
    """

    with tempfile.TemporaryDirectory(prefix="phone-dino-git-audit-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        bare_repository = temporary_root / "audit.git"
        empty_global_config = temporary_root / "empty-global.gitconfig"
        try:
            empty_global_config.write_bytes(b"")
        except OSError as error:
            raise SuccessorV2SelectionError("unable to prepare the isolated Git audit configuration") from error
        audit_environment = _sanitized_git_environment(global_config_path=empty_global_config)
        init_arguments = ["git", "init", "--bare", f"--object-format={object_format}"]
        init_arguments.append(str(bare_repository))
        try:
            initialized = subprocess.run(
                init_arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=45,
                env=audit_environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SuccessorV2SelectionError("unable to initialize the temporary Git audit repository") from error
        if initialized.returncode != 0:
            raise SuccessorV2SelectionError("unable to initialize the temporary Git audit repository")
        if _git_object_format(repository_root=bare_repository, environment=audit_environment) != object_format:
            raise SuccessorV2SelectionError("temporary Git audit repository has the wrong object format")
        fetched_ref = "refs/phone-dino-audit/fetched-required-ref"
        fetch = _git_process(
            ["fetch", "--no-tags", "--quiet", canonical_remote_url, f"{required_remote_ref}:{fetched_ref}"],
            repository_root=bare_repository, environment=audit_environment,
        )
        if fetch.returncode != 0:
            raise SuccessorV2SelectionError("required pushed Git audit ref is unreachable")
        fetched_tip = _require_git_oid(
            _git_text(
                ["rev-parse", "--verify", fetched_ref], repository_root=bare_repository,
                description="fetched audit ref", environment=audit_environment,
            ),
            object_format=object_format, name="fetched Git audit ref",
        )
        commit_exists = _git_process(
            ["cat-file", "-e", f"{commit_oid}^{{commit}}"], repository_root=bare_repository, environment=audit_environment
        )
        if commit_exists.returncode != 0:
            raise SuccessorV2SelectionError("recorded development evidence ledger commit is not reachable from the required pushed Git ref")
        ancestor = _git_process(
            ["merge-base", "--is-ancestor", commit_oid, fetched_tip], repository_root=bare_repository,
            environment=audit_environment,
        )
        if ancestor.returncode == 1:
            raise SuccessorV2SelectionError("recorded development evidence ledger commit is not reachable from the required pushed Git ref")
        if ancestor.returncode != 0:
            raise SuccessorV2SelectionError("Git audit could not verify the recorded ledger commit ancestry")
        return _git_blob_from_commit(
            repository_root=bare_repository, commit_oid=commit_oid, repository_path=repository_path, object_format=object_format,
            environment=audit_environment,
        )


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
    return bool(getattr(status, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400))


def _reject_links(path: Path, *, description: str) -> None:
    current = path
    while True:
        if current.exists() or current.is_symlink():
            if _is_link_or_reparse_point(current):
                raise SuccessorV2SelectionError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _prepare_slot(path: Path, *, description: str, repository_root: Path) -> None:
    # Reject existing link/reparse ancestry *before* resolve-based containment
    # checks.  This avoids silently resolving an attacker-controlled junction.
    _reject_links(path.parent, description=description)
    if path.exists() or path.is_symlink():
        if _is_link_or_reparse_point(path):
            raise SuccessorV2SelectionError(f"{description} contains a symbolic link or reparse point")
        raise SuccessorV2SelectionError(f"{description} already exists; the fixed slot is already consumed")
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise SuccessorV2SelectionError(f"{description} must stay outside the Git working tree")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_links(path.parent, description=description)


def _write_slot(path: Path, document: dict[str, Any], *, description: str, repository_root: Path) -> None:
    """O_EXCL + fsync + identity recheck before a held-out query may open."""

    _prepare_slot(path, description=description, repository_root=repository_root)
    payload = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    _reject_links(path.parent, description=description)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise SuccessorV2SelectionError(f"{description} already exists; the fixed slot is already consumed") from error
    except OSError as error:
        raise SuccessorV2SelectionError(f"unable to atomically create {description}") from error
    identity: tuple[int, int, int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            status = os.fstat(stream.fileno())
            identity = (status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode), status.st_size)
    except OSError as error:
        raise SuccessorV2SelectionError(f"unable to durably write {description}") from error
    _reject_links(path, description=description)
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as error:
        raise SuccessorV2SelectionError(f"unable to stat durable {description}") from error
    current_identity = (current.st_dev, current.st_ino, stat.S_IFMT(current.st_mode), current.st_size)
    if identity is None or not stat.S_ISREG(current.st_mode) or current_identity != identity:
        raise SuccessorV2SelectionError(f"{description} path identity changed while it was written")


def _static_scope(document: dict[str, Any], *, schema: str, purpose: str, phase: str, name: str) -> None:
    if document.get("schemaVersion") != schema:
        raise SuccessorV2SelectionError(f"{name} schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise SuccessorV2SelectionError(f"{name} must be non-authoritative and non-production")
    if document.get("purpose") != purpose or document.get("phase") != phase:
        raise SuccessorV2SelectionError(f"{name} purpose or phase is unsafe")
    if document.get("blindPolicy") != BLIND_POLICY or document.get("delegationPolicy") != DELEGATION_POLICY:
        raise SuccessorV2SelectionError(f"{name} scope is unsafe")
    if document.get("resultLabel") != RESULT_LABEL:
        raise SuccessorV2SelectionError(f"{name} result label is unsafe")


def _normal_input(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record[field] for field in NORMAL_INPUT_FIELDS}


def _validate_normal_inputs(value: object, *, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SuccessorV2SelectionError(f"{name} must be a non-empty list")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        record = _require_exact_fields(item, name=name, required=NORMAL_INPUT_FIELDS)
        case_id = _require_string(record.get("caseId"), name=f"{name}.caseId")
        if case_id in seen:
            raise SuccessorV2SelectionError(f"{name} caseId is duplicated")
        seen.add(case_id)
        if record.get("partition") != "NORMAL_SELECTION" or record.get("kind") != "NOMINAL" or record.get("defect") != "good":
            raise SuccessorV2SelectionError(f"{name} is not raw nominal selection input")
        for field in ("sourceSha256", "expectedRemoteSha256"):
            _require_sha256(record.get(field), name=f"{name}.{field}")
        if not isinstance(record.get("expectedRemoteBytes"), int) or record["expectedRemoteBytes"] <= 0:
            raise SuccessorV2SelectionError(f"{name}.expectedRemoteBytes must be positive")
        parsed.append(dict(record))
    if [item["caseId"] for item in parsed] != sorted(item["caseId"] for item in parsed):
        raise SuccessorV2SelectionError(f"{name} must be sorted by caseId")
    counts = {category: sum(item["category"] == category for item in parsed) for category in evaluator.SUCCESSOR_V2_CATEGORIES}
    if counts != {category: 8 for category in evaluator.SUCCESSOR_V2_CATEGORIES}:
        raise SuccessorV2SelectionError(f"{name} must contain eight inputs for each fixed successor category")
    return parsed


def _load_parent_chain(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], str]:
    try:
        return successor.load_validated_fresh_normal_successor_envelope(
            parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, repository_root=repository_root
        )
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error


def _selection_inputs_from_envelope(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    records = [_normal_input(record) for record in envelope["records"] if record.get("partition") == "NORMAL_SELECTION"]
    records.sort(key=lambda item: item["caseId"])
    return _validate_normal_inputs(records, name="successorSelectionInputs")


def _load_r3_manifest_json(
    augmentation_manifest_path: Path,
    recipe_path: Path,
    *,
    envelope: dict[str, Any],
    envelope_file_sha256: str,
    repository_root: Path,
) -> tuple[dict[str, Any], str, str]:
    """JSON-only binding validation; re-render happens in observer preflight."""

    document, file_sha256 = _read_json(
        augmentation_manifest_path, description="successor V2 R3 augmentation manifest", repository_root=repository_root
    )
    _require_exact_fields(document, name="successor V2 R3 augmentation manifest", required=augmentation.MANIFEST_FIELDS)
    if document.get("schemaVersion") != augmentation.SUCCESSOR_FIT_AUGMENTATION_V2_SCHEMA:
        raise SuccessorV2SelectionError("successor V2 R3 augmentation manifest schema is unsupported")
    if document.get("augmentationManifestSha256") != _document_digest(document, "augmentationManifestSha256"):
        raise SuccessorV2SelectionError("successor V2 R3 augmentation manifest digest does not match")
    expected = {
        "successorEnvelopeFileSha256": envelope_file_sha256,
        "successorEnvelopeDeclaredSha256": envelope["successorEnvelopeSha256"],
        "successorPlanFileSha256": envelope["planFileSha256"],
        "successorPlanDeclaredSha256": envelope["planDeclaredSha256"],
        "successorFitIdentitySha256": envelope["successorPartitionIdentities"]["FIT"],
    }
    for field, expected_value in expected.items():
        if document.get(field) != expected_value:
            raise SuccessorV2SelectionError(f"successor V2 R3 augmentation manifest {field} does not bind the closed chain")
    if document.get("variantsPerParent") != 3:
        raise SuccessorV2SelectionError("successor V2 R3 augmentation manifest must contain exactly R3 variants")
    try:
        _recipe, recipe_sha256 = augmentation.load_successor_fit_camera_recipe_v2(recipe_path)
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error
    if document.get("recipeFileSha256") != recipe_sha256:
        raise SuccessorV2SelectionError("successor V2 R3 recipe does not match its manifest")
    _validated_r3_records(document, envelope=envelope)
    return document, file_sha256, recipe_sha256


def _identity_from_envelope_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"], "category": record["category"], "partition": record["partition"],
        "kind": record["kind"], "defect": record["defect"], "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"], "isAugmentation": False, "variantId": None,
        "component": None, "parentCaseId": None, "parentSourceSha256": None,
        "augmentationManifestSha256": None,
    }


def _identity_from_augmentation_record(record: dict[str, Any], *, manifest_declared: str) -> dict[str, Any]:
    return {
        "caseId": record["caseId"], "category": record["category"], "partition": "FIT", "kind": record["kind"],
        "defect": record["defect"], "sourceSha256": record["sourceSha256"], "sourceGroupId": record["sourceGroupId"],
        "isAugmentation": True, "variantId": record["variantId"], "component": record["component"],
        "parentCaseId": record["parentCaseId"], "parentSourceSha256": record["parentSourceSha256"],
        "augmentationManifestSha256": manifest_declared,
    }


def _validated_r3_records(manifest: dict[str, Any], *, envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate R3 child membership entirely from closed JSON, without bytes."""

    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != 108:
        raise SuccessorV2SelectionError("successor V2 R3 manifest must contain 108 FIT derivatives")
    parents = {
        str(record["caseId"]): record
        for record in envelope.get("records", [])
        if isinstance(record, dict) and record.get("partition") == "FIT"
    }
    if len(parents) != 36:
        raise SuccessorV2SelectionError("sealed successor envelope must expose exactly 36 FIT parents")
    components_by_variant = {1: "registration", 2: "illumination", 3: "sensor_transport"}
    child_cases: set[str] = set()
    child_sources: set[str] = set()
    children_by_parent: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in parents}
    parent_sources = {str(record["sourceSha256"]) for record in parents.values()}
    parsed: list[dict[str, Any]] = []
    for item in raw_records:
        record = _require_exact_fields(item, name="successor V2 R3 record", required=augmentation.RECORD_FIELDS)
        case_id = _require_string(record.get("caseId"), name="successor V2 R3 record caseId")
        source_sha256 = _require_sha256(record.get("sourceSha256"), name="successor V2 R3 record sourceSha256")
        if case_id in child_cases or source_sha256 in child_sources or source_sha256 in parent_sources:
            raise SuccessorV2SelectionError("successor V2 R3 manifest child case/source identity is duplicated or collides with FIT")
        child_cases.add(case_id)
        child_sources.add(source_sha256)
        parent_case_id = _require_string(record.get("parentCaseId"), name="successor V2 R3 record parentCaseId")
        parent = parents.get(parent_case_id)
        if parent is None:
            raise SuccessorV2SelectionError("successor V2 R3 manifest child has an unknown FIT parent")
        if (
            record.get("parentPartition") != "FIT"
            or record.get("kind") != "NOMINAL"
            or record.get("defect") != "good"
            or record.get("parentSourceSha256") != parent.get("sourceSha256")
            or record.get("category") != parent.get("category")
            or record.get("sourceGroupId") != parent.get("sourceGroupId")
        ):
            raise SuccessorV2SelectionError("successor V2 R3 manifest child does not match its FIT parent")
        variant_id = record.get("variantId")
        if not isinstance(variant_id, int) or isinstance(variant_id, bool) or variant_id not in components_by_variant:
            raise SuccessorV2SelectionError("successor V2 R3 manifest variantId is unsupported")
        if record.get("component") != components_by_variant[variant_id]:
            raise SuccessorV2SelectionError("successor V2 R3 manifest component does not match variantId")
        _require_string(record.get("relativePath"), name="successor V2 R3 record relativePath")
        children_by_parent[parent_case_id].append(dict(record))
        parsed.append(dict(record))
    for parent_case_id, children in children_by_parent.items():
        if len(children) != 3:
            raise SuccessorV2SelectionError("successor V2 R3 manifest does not cover every FIT parent exactly three times")
        if {int(item["variantId"]) for item in children} != {1, 2, 3} or {str(item["component"]) for item in children} != set(components_by_variant.values()):
            raise SuccessorV2SelectionError("successor V2 R3 manifest parent components are incomplete or duplicated")
    if [str(item["caseId"]) for item in raw_records] != sorted(str(item["caseId"]) for item in raw_records):
        raise SuccessorV2SelectionError("successor V2 R3 manifest records must be sorted by caseId")
    return parsed


def _expected_feature_inputs(
    envelope: dict[str, Any], *, r3_manifest: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    originals = [
        _identity_from_envelope_record(record)
        for record in envelope["records"]
        if record["partition"] in {"FIT", "THRESHOLD_TUNING"}
    ]
    raw = sorted(originals, key=lambda record: str(record["caseId"]))
    if r3_manifest is None:
        return raw, raw
    declared = _require_sha256(r3_manifest.get("augmentationManifestSha256"), name="R3 manifest declared digest")
    records = _validated_r3_records(r3_manifest, envelope=envelope)
    augmented: list[dict[str, Any]] = []
    for item in records:
        augmented.append(_identity_from_augmentation_record(item, manifest_declared=declared))
    r3 = sorted(originals + augmented, key=lambda record: str(record["caseId"]))
    return raw, r3


def _prototype_input_binding_from_identities(
    raw_inputs: list[dict[str, Any]], r3_inputs: list[dict[str, Any]], *, name: str
) -> tuple[dict[str, str], dict[str, int]]:
    """Derive the FIT-only raw/R3 prototype commitment without image bytes."""

    raw_fit = [record for record in raw_inputs if record.get("partition") == "FIT"]
    r3_fit = [record for record in r3_inputs if record.get("partition") == "FIT"]
    counts = {
        evaluator.RAW_FIT_ONLY: len(raw_fit),
        evaluator.RAW_FIT_PLUS_AUGMENTATION_R3: len(r3_fit),
    }
    if counts != PROTOTYPE_INPUT_COUNTS:
        raise SuccessorV2SelectionError(f"{name} must contain exactly raw=36 and R3=144 FIT prototype records")
    return {
        evaluator.RAW_FIT_ONLY: canonical_json_sha256(raw_fit),
        evaluator.RAW_FIT_PLUS_AUGMENTATION_R3: canonical_json_sha256(r3_fit),
    }, counts


def _validate_prototype_input_binding(
    identities: object, counts: object, *, name: str
) -> tuple[dict[str, str], dict[str, int]]:
    """Parse the fixed two-policy prototype commitment shared by contract/receipt."""

    if not isinstance(identities, dict) or set(identities) != PROTOTYPE_POLICY_KEYS:
        raise SuccessorV2SelectionError(f"{name} prototype identity map is unsafe")
    if not isinstance(counts, dict) or set(counts) != PROTOTYPE_POLICY_KEYS:
        raise SuccessorV2SelectionError(f"{name} prototype count map is unsafe")
    parsed_identities = {
        policy: _require_sha256(identities.get(policy), name=f"{name} prototype identity {policy}")
        for policy in sorted(PROTOTYPE_POLICY_KEYS)
    }
    parsed_counts: dict[str, int] = {}
    for policy, expected_count in PROTOTYPE_INPUT_COUNTS.items():
        value = counts.get(policy)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected_count:
            raise SuccessorV2SelectionError(f"{name} must record fixed raw=36 and R3=144 prototype counts")
        parsed_counts[policy] = value
    return parsed_identities, parsed_counts


def _validate_report(
    report: dict[str, Any],
    *,
    report_file_sha256: str,
    envelope: dict[str, Any],
    envelope_file_sha256: str,
    r3_manifest: dict[str, Any],
    r3_manifest_file_sha256: str,
    r3_recipe_sha256: str,
) -> dict[str, Any]:
    _require_exact_fields(report, name="successor V2 development report", required=evaluator.DEVELOPMENT_REPORT_FIELDS)
    if report.get("schemaVersion") != evaluator.SUCCESSOR_V2_DEVELOPMENT_REPORT_SCHEMA:
        raise SuccessorV2SelectionError("successor V2 development report schema is unsupported")
    if report.get("authoritative") is not False or report.get("productionAuthorized") is not False:
        raise SuccessorV2SelectionError("successor V2 development report must be non-authoritative and non-production")
    if (
        report.get("purpose") != evaluator.SUCCESSOR_V2_DEVELOPMENT_REPORT_PURPOSE
        or report.get("phase") != evaluator.SUCCESSOR_V2_DEVELOPMENT_PHASE
        or report.get("blindPolicy") != evaluator.SUCCESSOR_V2_BLIND_POLICY
        or report.get("resultLabel") != evaluator.SUCCESSOR_V2_RESULT_LABEL
        or report.get("delegationPolicy") != evaluator.SUCCESSOR_V2_DELEGATION_POLICY
        or report.get("independenceLabel") != successor.FRESH_NORMAL_SUCCESSOR_INDEPENDENCE_LABEL
    ):
        raise SuccessorV2SelectionError("successor V2 development report scope is unsafe")
    if report.get("developmentReportSha256") != _document_digest(report, "developmentReportSha256"):
        raise SuccessorV2SelectionError("successor V2 development report digest does not match")
    configuration = evaluator.validate_candidate_configuration(report.get("candidateConfiguration"))
    if report.get("candidateConfigurationSha256") != canonical_json_sha256(configuration):
        raise SuccessorV2SelectionError("successor V2 development report configuration digest does not match")
    parent = envelope.get("parentEvidence")
    if not isinstance(parent, dict):
        raise SuccessorV2SelectionError("successor envelope parent evidence is missing")
    expected_chain = {
        "parentHoldoutFileSha256": parent["holdoutManifestFileSha256"],
        "parentHoldoutDeclaredSha256": parent["holdoutManifestDeclaredSha256"],
        "parentSelectionContractFileSha256": parent["selectionContractFileSha256"],
        "parentSelectionContractDeclaredSha256": parent["selectionContractDeclaredSha256"],
        "parentNormalConfirmationIdentitySha256": parent["parentNormalConfirmationIdentitySha256"],
        "successorSealFileSha256": envelope["sealFileSha256"],
        "successorSealDeclaredSha256": envelope["sealDeclaredSha256"],
        "successorPlanFileSha256": envelope["planFileSha256"],
        "successorPlanDeclaredSha256": envelope["planDeclaredSha256"],
        "successorEnvelopeFileSha256": envelope_file_sha256,
        "successorEnvelopeDeclaredSha256": envelope["successorEnvelopeSha256"],
        "successorFitIdentitySha256": envelope["successorPartitionIdentities"]["FIT"],
        "successorThresholdTuningIdentitySha256": envelope["successorPartitionIdentities"]["THRESHOLD_TUNING"],
    }
    for field, expected in expected_chain.items():
        if report.get(field) != expected:
            raise SuccessorV2SelectionError(f"successor V2 development report {field} does not bind the closed chain")
    policy = configuration["prototypeInputPolicy"]
    expected_input_policy = (
        "SUCCESSOR_RAW_FIT_PLUS_RAW_THRESHOLD_TUNING_ONLY"
        if policy == evaluator.RAW_FIT_ONLY
        else "SUCCESSOR_RAW_FIT_PLUS_AUGMENTATION_R3_PLUS_RAW_THRESHOLD_TUNING_ONLY"
    )
    if report.get("inputPolicy") != expected_input_policy:
        raise SuccessorV2SelectionError("successor V2 development report prototype input policy is unsafe")
    raw_inputs, r3_inputs = _expected_feature_inputs(envelope, r3_manifest=r3_manifest)
    expected_inputs = raw_inputs if policy == evaluator.RAW_FIT_ONLY else r3_inputs
    supplied_inputs = report.get("featureInputs")
    if supplied_inputs != expected_inputs or report.get("featureInputIdentitySha256") != canonical_json_sha256(expected_inputs):
        raise SuccessorV2SelectionError("successor V2 development report feature membership does not match its raw/R3 policy")
    calibration = [record for record in expected_inputs if record["partition"] == "THRESHOLD_TUNING"]
    if report.get("calibrationInputs") != calibration or report.get("calibrationInputIdentitySha256") != canonical_json_sha256(calibration):
        raise SuccessorV2SelectionError("successor V2 development report calibration membership does not match raw tuning")
    # Thresholds are not user-selected knobs.  Reconstruct them from every
    # raw-tuning score and independently validate all category summaries.
    raw_scores = report.get("calibrationScores")
    if not isinstance(raw_scores, list) or len(raw_scores) != len(calibration):
        raise SuccessorV2SelectionError("successor V2 development report calibration scores do not cover raw tuning")
    calibration_by_case = {record["caseId"]: record for record in calibration}
    seen_cases: set[str] = set()
    scores_by_category: dict[str, list[float]] = {category: [] for category in evaluator.SUCCESSOR_V2_CATEGORIES}
    for score_value in raw_scores:
        score = _require_exact_fields(score_value, name="successor V2 calibration score", required=evaluator.CALIBRATION_SCORE_FIELDS)
        case_id = _require_string(score.get("caseId"), name="successor V2 calibration score caseId")
        expected_input = calibration_by_case.get(case_id)
        if expected_input is None or case_id in seen_cases:
            raise SuccessorV2SelectionError("successor V2 calibration score membership is unsafe")
        seen_cases.add(case_id)
        for field in ("category", "partition", "kind", "defect", "sourceSha256"):
            if score.get(field) != expected_input[field]:
                raise SuccessorV2SelectionError("successor V2 calibration score does not match its raw tuning input")
        components = {
            field: _require_finite(score.get(field), name=f"successor V2 calibration score {field}", minimum=0.0)
            for field in ("score", "maxPatchDistance", "meanNearestPatchDistance")
        }
        if not components["meanNearestPatchDistance"] <= components["score"] <= components["maxPatchDistance"]:
            raise SuccessorV2SelectionError("successor V2 calibration score components are inconsistent")
        scores_by_category[str(score["category"])].append(components["score"])
    if [str(item.get("caseId")) for item in raw_scores] != sorted(calibration_by_case) or seen_cases != set(calibration_by_case):
        raise SuccessorV2SelectionError("successor V2 calibration scores must cover raw tuning once in caseId order")
    if policy == evaluator.RAW_FIT_ONLY:
        if any(report.get(field) is not None for field in (
            "augmentationManifestFileSha256", "augmentationManifestDeclaredSha256", "augmentationRecipeFileSha256",
            "augmentationParentFitIdentitySha256",
        )):
            raise SuccessorV2SelectionError("raw-FIT report must not bind an augmentation package")
    else:
        expected_aug = {
            "augmentationManifestFileSha256": r3_manifest_file_sha256,
            "augmentationManifestDeclaredSha256": r3_manifest["augmentationManifestSha256"],
            "augmentationRecipeFileSha256": r3_recipe_sha256,
            "augmentationParentFitIdentitySha256": envelope["successorPartitionIdentities"]["FIT"],
        }
        for field, expected in expected_aug.items():
            if report.get(field) != expected:
                raise SuccessorV2SelectionError(f"R3 development report {field} does not bind the closed R3 package")
    extractor = report.get("featureExtractor")
    if not isinstance(extractor, dict) or not extractor:
        raise SuccessorV2SelectionError("successor V2 development report feature extractor is unsafe")
    if report.get("featureExtractorIdentitySha256") != canonical_json_sha256(extractor):
        raise SuccessorV2SelectionError("successor V2 development report feature extractor digest does not match")
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(evaluator.SUCCESSOR_V2_CATEGORIES):
        raise SuccessorV2SelectionError("successor V2 development report thresholds do not cover fixed categories")
    parsed_thresholds = {category: _require_finite(thresholds[category], name=f"threshold {category}", minimum=0.0) for category in sorted(thresholds)}
    categories = report.get("categories")
    if not isinstance(categories, dict) or set(categories) != set(evaluator.SUCCESSOR_V2_CATEGORIES):
        raise SuccessorV2SelectionError("successor V2 development report categories are unsafe")
    for category in evaluator.SUCCESSOR_V2_CATEGORIES:
        report_category = categories[category]
        _require_exact_fields(report_category, name=f"successor V2 category {category}", required=evaluator.CATEGORY_REPORT_FIELDS)
        values = scores_by_category[category]
        if len(values) != 4:
            raise SuccessorV2SelectionError("successor V2 calibration scores must contain four raw tuning observations per category")
        expected_threshold = max(values)
        if not _same_number(expected_threshold, parsed_thresholds[category]):
            raise SuccessorV2SelectionError("successor V2 threshold must remain the raw-tuning maximum")
        raw_fit_count = sum(
            record["category"] == category and record["partition"] == "FIT" and not record["isAugmentation"]
            for record in expected_inputs
        )
        augmented_fit_count = sum(
            record["category"] == category and record["partition"] == "FIT" and record["isAugmentation"]
            for record in expected_inputs
        )
        expected_counts = {
            "fitOriginalCount": raw_fit_count,
            "fitAugmentedCount": augmented_fit_count,
            "tuningOriginalCount": len(values),
        }
        for field, expected in expected_counts.items():
            if report_category.get(field) != expected:
                raise SuccessorV2SelectionError(f"successor V2 category {field} does not match frozen feature membership")
        for field in ("prototypePatchCount", "fitPatchCount", "patchGridHeight", "patchGridWidth"):
            value = report_category.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SuccessorV2SelectionError(f"successor V2 category {field} must be positive")
        if report_category["prototypePatchCount"] > report_category["fitPatchCount"]:
            raise SuccessorV2SelectionError("successor V2 prototype patch count exceeds FIT patch count")
        ordered = sorted(values)
        expected_summary = {
            "thresholdFromRawTuning": expected_threshold,
            "tuningScoreMedian": ordered[len(ordered) // 2],
            "tuningScoreP95": _p95(values),
            "tuningScoreMax": expected_threshold,
        }
        for field, expected in expected_summary.items():
            actual = _require_finite(report_category.get(field), name=f"successor V2 category {field}", minimum=0.0)
            if not _same_number(expected, actual):
                raise SuccessorV2SelectionError(f"successor V2 category {field} does not match calibration scores")
    try:
        expected_evidence = evaluator._build_normal_only_evidence(expected_inputs, calibration)
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error
    if report.get("normalOnlyEvidence") != expected_evidence:
        raise SuccessorV2SelectionError("successor V2 development report normal-only evidence is inconsistent")
    execution = _require_exact_fields(report.get("execution"), name="successor V2 development execution", required=evaluator.EXECUTION_FIELDS)
    for field in ("evaluatorModuleSha256", "evaluatorEntrypointSha256"):
        _require_sha256(execution.get(field), name=f"successor V2 development execution {field}")
    if extractor.get("evaluatorModuleSha256") != execution["evaluatorModuleSha256"]:
        raise SuccessorV2SelectionError("successor V2 execution evaluator module does not match the frozen feature extractor")
    timings = execution.get("phaseTimingsSeconds")
    timing_fields = {
        "inputAssemblySeconds", "provenanceSeconds", "inputVerificationSeconds", "featureInferenceSeconds",
        "scoringSeconds", "totalElapsedSeconds",
    }
    if not isinstance(timings, dict) or set(timings) != timing_fields:
        raise SuccessorV2SelectionError("successor V2 development execution timings are unsupported")
    for field, value in timings.items():
        _require_finite(value, name=f"successor V2 development timing {field}", minimum=0.0, maximum=31_536_000.0)
    for field in ("python", "platform", "numpyVersion", "torchVersion"):
        _require_string(execution.get(field), name=f"successor V2 development execution {field}")
    thread_count = execution.get("torchThreadCount")
    if not isinstance(thread_count, int) or isinstance(thread_count, bool) or thread_count <= 0:
        raise SuccessorV2SelectionError("successor V2 development execution torchThreadCount must be positive")
    if execution.get("gitRevision") is not None:
        _require_string(execution.get("gitRevision"), name="successor V2 development execution gitRevision")
    if execution.get("gitWorktreeClean") is not None and not isinstance(execution.get("gitWorktreeClean"), bool):
        raise SuccessorV2SelectionError("successor V2 development execution gitWorktreeClean must be boolean or null")
    return {
        "candidateId": configuration["id"], "developmentReportFileSha256": report_file_sha256,
        "developmentReportDeclaredSha256": report["developmentReportSha256"], "candidateConfiguration": configuration,
        "candidateConfigurationSha256": report["candidateConfigurationSha256"], "prototypeInputPolicy": policy,
        "featureInputIdentitySha256": report["featureInputIdentitySha256"],
        "calibrationInputIdentitySha256": report["calibrationInputIdentitySha256"], "thresholds": parsed_thresholds,
        "thresholdsIdentitySha256": canonical_json_sha256(parsed_thresholds),
        "augmentationManifestFileSha256": report["augmentationManifestFileSha256"],
        "augmentationManifestDeclaredSha256": report["augmentationManifestDeclaredSha256"],
        "augmentationRecipeFileSha256": report["augmentationRecipeFileSha256"],
        "featureExtractor": extractor, "featureExtractorIdentitySha256": report["featureExtractorIdentitySha256"],
    }


def _candidate_binding_for_contract(binding: dict[str, Any]) -> dict[str, Any]:
    return {field: binding[field] for field in CANDIDATE_BINDING_FIELDS}


def _validate_contract_binding_projection(value: object, *, name: str) -> dict[str, Any]:
    """Validate the closed projection that a Git-ledger and contract share."""

    projection = _require_exact_fields(value, name=name, required=DEVELOPMENT_EVIDENCE_PROJECTION_FIELDS)
    augmentation_binding = _require_exact_fields(
        projection.get("augmentation"), name=f"{name} augmentation", required=AUGMENTATION_BINDING_FIELDS
    )
    for field in ("manifestFileSha256", "manifestDeclaredSha256", "recipeFileSha256", "successorFitIdentitySha256"):
        _require_sha256(augmentation_binding.get(field), name=f"{name} augmentation {field}")
    if augmentation_binding.get("variantsPerParent") != 3:
        raise SuccessorV2SelectionError(f"{name} requires exactly three R3 variants per FIT parent")
    prototype_identities, prototype_counts = _validate_prototype_input_binding(
        projection.get("prototypeInputIdentities"), projection.get("prototypeInputCounts"), name=name
    )
    extractor = projection.get("featureExtractor")
    if not isinstance(extractor, dict) or not extractor:
        raise SuccessorV2SelectionError(f"{name} feature extractor is unsafe")
    extractor_identity = _require_sha256(projection.get("featureExtractorIdentitySha256"), name=f"{name} feature extractor identity")
    if extractor_identity != canonical_json_sha256(extractor):
        raise SuccessorV2SelectionError(f"{name} feature extractor digest does not match")
    selection_module_sha = _require_sha256(
        projection.get("selectionProtocolModuleSha256"), name=f"{name} selection protocol module digest"
    )
    reports = projection.get("candidateBindings")
    if not isinstance(reports, list) or len(reports) != len(evaluator.PRE_REGISTERED_CANDIDATES):
        raise SuccessorV2SelectionError(f"{name} must bind every pre-registered candidate")
    parsed_reports: list[dict[str, Any]] = []
    expected_ids = [item["id"] for item in evaluator.PRE_REGISTERED_CANDIDATES]
    for item, expected_id in zip(reports, expected_ids, strict=True):
        binding = _require_exact_fields(item, name=f"{name} candidate binding", required=CANDIDATE_BINDING_FIELDS)
        configuration = evaluator.validate_candidate_configuration(binding.get("candidateConfiguration"))
        if binding.get("candidateId") != expected_id or configuration["id"] != expected_id:
            raise SuccessorV2SelectionError(f"{name} candidate universe order is unsafe")
        if binding.get("candidateConfigurationSha256") != canonical_json_sha256(configuration):
            raise SuccessorV2SelectionError(f"{name} candidate configuration digest does not match")
        for field in (
            "developmentReportFileSha256", "developmentReportDeclaredSha256", "featureInputIdentitySha256",
            "calibrationInputIdentitySha256", "thresholdsIdentitySha256",
        ):
            _require_sha256(binding.get(field), name=f"{name} candidate {field}")
        if binding.get("featureExtractorIdentitySha256") != extractor_identity:
            raise SuccessorV2SelectionError(f"{name} candidate does not share the frozen feature extractor")
        thresholds = binding.get("thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != set(evaluator.SUCCESSOR_V2_CATEGORIES):
            raise SuccessorV2SelectionError(f"{name} candidate thresholds are unsafe")
        parsed_thresholds = {
            category: _require_finite(thresholds[category], name=f"{name} threshold {category}", minimum=0.0)
            for category in sorted(thresholds)
        }
        if binding.get("thresholdsIdentitySha256") != canonical_json_sha256(parsed_thresholds):
            raise SuccessorV2SelectionError(f"{name} candidate thresholds digest does not match")
        policy = binding.get("prototypeInputPolicy")
        if policy != configuration["prototypeInputPolicy"]:
            raise SuccessorV2SelectionError(f"{name} candidate prototype policy does not match configuration")
        expected_augmentation = {
            "augmentationManifestFileSha256": augmentation_binding["manifestFileSha256"],
            "augmentationManifestDeclaredSha256": augmentation_binding["manifestDeclaredSha256"],
            "augmentationRecipeFileSha256": augmentation_binding["recipeFileSha256"],
        }
        if policy == evaluator.RAW_FIT_ONLY:
            if any(binding.get(field) is not None for field in expected_augmentation):
                raise SuccessorV2SelectionError(f"{name} raw candidate must not bind the R3 augmentation package")
        elif policy == evaluator.RAW_FIT_PLUS_AUGMENTATION_R3:
            for field, expected in expected_augmentation.items():
                if binding.get(field) != expected:
                    raise SuccessorV2SelectionError(f"{name} R3 candidate does not bind the exact augmentation package")
        else:
            raise SuccessorV2SelectionError(f"{name} candidate prototype policy is unsupported")
        parsed_reports.append({**dict(binding), "candidateConfiguration": configuration, "thresholds": parsed_thresholds})
    if (
        len({item["developmentReportFileSha256"] for item in parsed_reports}) != len(parsed_reports)
        or len({item["developmentReportDeclaredSha256"] for item in parsed_reports}) != len(parsed_reports)
    ):
        raise SuccessorV2SelectionError(f"{name} candidate report digests must be unique")
    expected_universe = canonical_json_sha256([_candidate_binding_for_contract(item) for item in parsed_reports])
    if _require_sha256(projection.get("candidateUniverseIdentitySha256"), name=f"{name} candidate universe identity") != expected_universe:
        raise SuccessorV2SelectionError(f"{name} candidate universe digest does not match")
    return {
        "augmentation": dict(augmentation_binding), "prototypeInputIdentities": prototype_identities,
        "prototypeInputCounts": prototype_counts, "featureExtractor": extractor,
        "featureExtractorIdentitySha256": extractor_identity, "selectionProtocolModuleSha256": selection_module_sha,
        "candidateBindings": [_candidate_binding_for_contract(item) for item in parsed_reports],
        "candidateUniverseIdentitySha256": expected_universe,
    }


def _development_evidence_ledger_document(projection: dict[str, Any]) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schemaVersion": SUCCESSOR_V2_DEVELOPMENT_EVIDENCE_LEDGER_SCHEMA,
        "authoritative": False, "productionAuthorized": False,
        "purpose": DEVELOPMENT_EVIDENCE_LEDGER_PURPOSE, "phase": DEVELOPMENT_EVIDENCE_LEDGER_PHASE,
        "blindPolicy": BLIND_POLICY, "resultLabel": RESULT_LABEL, "independenceLabel": INDEPENDENCE_LABEL,
        "delegationPolicy": DELEGATION_POLICY, "selectionInput": dict(SELECTION_INPUT),
        "ledgerScope": DEVELOPMENT_EVIDENCE_LEDGER_SCOPE, "trustMode": PUSHED_GIT_AUDIT_ONLY,
        "contractBindingProjection": projection,
        "contractBindingProjectionSha256": canonical_json_sha256(projection),
    }
    document["developmentEvidenceLedgerSha256"] = _document_digest(document, "developmentEvidenceLedgerSha256")
    return document


def _validate_development_evidence_ledger(document: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(document, name="successor V2 development evidence ledger", required=DEVELOPMENT_EVIDENCE_LEDGER_FIELDS)
    _static_scope(
        document, schema=SUCCESSOR_V2_DEVELOPMENT_EVIDENCE_LEDGER_SCHEMA,
        purpose=DEVELOPMENT_EVIDENCE_LEDGER_PURPOSE, phase=DEVELOPMENT_EVIDENCE_LEDGER_PHASE,
        name="successor V2 development evidence ledger",
    )
    if (
        document.get("independenceLabel") != INDEPENDENCE_LABEL or document.get("selectionInput") != SELECTION_INPUT
        or document.get("ledgerScope") != DEVELOPMENT_EVIDENCE_LEDGER_SCOPE or document.get("trustMode") != PUSHED_GIT_AUDIT_ONLY
    ):
        raise SuccessorV2SelectionError("successor V2 development evidence ledger scope is unsafe")
    projection = _validate_contract_binding_projection(
        document.get("contractBindingProjection"), name="successor V2 development evidence ledger projection"
    )
    if document.get("contractBindingProjectionSha256") != canonical_json_sha256(projection):
        raise SuccessorV2SelectionError("successor V2 development evidence ledger projection digest does not match")
    if document.get("developmentEvidenceLedgerSha256") != _document_digest(document, "developmentEvidenceLedgerSha256"):
        raise SuccessorV2SelectionError("successor V2 development evidence ledger digest does not match")
    return {**document, "contractBindingProjection": projection}


def _validate_development_evidence_anchor(value: object) -> dict[str, str]:
    anchor = _require_exact_fields(value, name="successor V2 development evidence Git anchor", required=DEVELOPMENT_EVIDENCE_GIT_FIELDS)
    if (
        anchor.get("mode") != PUSHED_GIT_AUDIT_ONLY
        or anchor.get("canonicalRemoteUrl") != CANONICAL_GIT_AUDIT_REMOTE_URL
        or anchor.get("requiredRemoteRef") != REQUIRED_GIT_AUDIT_REMOTE_REF
        or anchor.get("repositoryPath") != DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH
    ):
        raise SuccessorV2SelectionError("successor V2 development evidence Git anchor is not the canonical pushed-audit boundary")
    object_format = _require_string(anchor.get("gitObjectFormat"), name="development evidence Git object format")
    if object_format != CANONICAL_GIT_AUDIT_OBJECT_FORMAT:
        raise SuccessorV2SelectionError("successor V2 development evidence Git object format is not the canonical audit format")
    commit_oid = _require_git_oid(anchor.get("gitCommitOid"), object_format=object_format, name="development evidence Git commit")
    blob_oid = _require_git_oid(anchor.get("gitBlobOid"), object_format=object_format, name="development evidence Git blob")
    result = {
        "mode": PUSHED_GIT_AUDIT_ONLY, "canonicalRemoteUrl": CANONICAL_GIT_AUDIT_REMOTE_URL,
        "requiredRemoteRef": REQUIRED_GIT_AUDIT_REMOTE_REF, "gitObjectFormat": object_format,
        "gitCommitOid": commit_oid, "repositoryPath": DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH,
        "gitBlobOid": blob_oid,
    }
    for field in ("ledgerBlobSha256", "ledgerDeclaredSha256", "ledgerProjectionSha256"):
        result[field] = _require_sha256(anchor.get(field), name=f"development evidence {field}")
    return result


def _load_recorded_development_evidence_ledger(
    anchor_value: object, *, repository_root: Path = REPOSITORY_ROOT
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve a ledger exclusively from a pushed commit/blob, never the worktree."""

    anchor = _validate_development_evidence_anchor(anchor_value)
    blob_oid, raw_blob = _resolve_pushed_git_ledger_blob(
        commit_oid=anchor["gitCommitOid"], repository_path=anchor["repositoryPath"],
        canonical_remote_url=anchor["canonicalRemoteUrl"], required_remote_ref=anchor["requiredRemoteRef"],
        object_format=anchor["gitObjectFormat"],
    )
    if blob_oid != anchor["gitBlobOid"]:
        raise SuccessorV2SelectionError("recorded development evidence Git blob no longer matches its anchor")
    blob_sha = f"sha256:{hashlib.sha256(raw_blob).hexdigest()}"
    if blob_sha != anchor["ledgerBlobSha256"]:
        raise SuccessorV2SelectionError("recorded development evidence raw Git blob digest does not match")
    ledger = _validate_development_evidence_ledger(_parse_ledger_blob(raw_blob))
    if (
        ledger["developmentEvidenceLedgerSha256"] != anchor["ledgerDeclaredSha256"]
        or ledger["contractBindingProjectionSha256"] != anchor["ledgerProjectionSha256"]
    ):
        raise SuccessorV2SelectionError("recorded development evidence ledger declared/projection digests do not match its anchor")
    return ledger, anchor


def _load_head_development_evidence_ledger(
    ledger_path: Path, *, repository_root: Path = REPOSITORY_ROOT
) -> tuple[dict[str, Any], dict[str, str]]:
    """Accept a ledger for contract freeze only from HEAD's raw, pushed Git blob.

    The supplied worktree pathname fixes the reviewed repository path, but its
    bytes (and index state) are deliberately not a source of authority.  This
    avoids line-ending/filter ambiguity and permits unrelated or staged working
    changes: the local HEAD blob must exactly equal the independently fetched,
    isolated remote blob before it is parsed.
    """

    repository_path = _ledger_repository_path(ledger_path, repository_root=repository_root)
    object_format = _git_object_format(repository_root=repository_root)
    if object_format != CANONICAL_GIT_AUDIT_OBJECT_FORMAT:
        raise SuccessorV2SelectionError("development evidence ledger repository has the wrong Git object format")
    head = _require_git_oid(
        _git_text(["rev-parse", "--verify", "HEAD"], repository_root=repository_root, description="HEAD"),
        object_format=object_format, name="development evidence HEAD commit",
    )
    local_blob_oid, local_raw_blob = _git_blob_from_commit(
        repository_root=repository_root, commit_oid=head, repository_path=repository_path, object_format=object_format
    )
    blob_oid, raw_blob = _resolve_pushed_git_ledger_blob(
        commit_oid=head, repository_path=repository_path, canonical_remote_url=CANONICAL_GIT_AUDIT_REMOTE_URL,
        required_remote_ref=REQUIRED_GIT_AUDIT_REMOTE_REF, object_format=object_format,
    )
    if blob_oid != local_blob_oid or raw_blob != local_raw_blob:
        raise SuccessorV2SelectionError("pushed Git ledger blob does not exactly match the local HEAD raw ledger blob")
    ledger = _validate_development_evidence_ledger(_parse_ledger_blob(raw_blob))
    anchor = {
        "mode": PUSHED_GIT_AUDIT_ONLY, "canonicalRemoteUrl": CANONICAL_GIT_AUDIT_REMOTE_URL,
        "requiredRemoteRef": REQUIRED_GIT_AUDIT_REMOTE_REF, "gitObjectFormat": object_format,
        "gitCommitOid": head, "repositoryPath": repository_path, "gitBlobOid": blob_oid,
        "ledgerBlobSha256": f"sha256:{hashlib.sha256(raw_blob).hexdigest()}",
        "ledgerDeclaredSha256": ledger["developmentEvidenceLedgerSha256"],
        "ledgerProjectionSha256": ledger["contractBindingProjectionSha256"],
    }
    return ledger, anchor


def _validate_contract_development_evidence_ledger(
    contract: dict[str, Any], projection: dict[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    ledger, _anchor = _load_recorded_development_evidence_ledger(
        contract.get("developmentEvidenceLedger"), repository_root=repository_root
    )
    if ledger["contractBindingProjection"] != projection:
        raise SuccessorV2SelectionError("successor V2 selection contract does not match the immutable development evidence ledger projection")
    return ledger


def _validate_contract(document: dict[str, Any], *, repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    _require_exact_fields(document, name="successor V2 selection contract", required=CONTRACT_FIELDS)
    _static_scope(document, schema=SUCCESSOR_V2_SELECTION_CONTRACT_SCHEMA, purpose=CONTRACT_PURPOSE, phase=CONTRACT_PHASE, name="successor V2 selection contract")
    if document.get("independenceLabel") != INDEPENDENCE_LABEL or document.get("selectionInput") != SELECTION_INPUT:
        raise SuccessorV2SelectionError("successor V2 selection contract scope is unsafe")
    _require_sha256(document.get("successorEnvelopeSelectionIdentitySha256"), name="successor envelope selection identity")
    registry = _validate_consumption_registry(document.get("consumptionRegistry"), contract=document, repository_root=repository_root)
    inputs = _validate_normal_inputs(document.get("successorSelectionInputs"), name="successorSelectionInputs")
    if document.get("successorSelectionInputIdentitySha256") != canonical_json_sha256(inputs):
        raise SuccessorV2SelectionError("successor V2 selection contract selection input digest does not match")
    prototype_identities, prototype_counts = _validate_prototype_input_binding(
        document.get("prototypeInputIdentities"), document.get("prototypeInputCounts"),
        name="successor V2 selection contract",
    )
    augmentation_binding = _require_exact_fields(
        document.get("augmentation"), name="successor V2 R3 augmentation binding", required=AUGMENTATION_BINDING_FIELDS
    )
    for field in ("manifestFileSha256", "manifestDeclaredSha256", "recipeFileSha256", "successorFitIdentitySha256"):
        _require_sha256(augmentation_binding.get(field), name=f"successor V2 augmentation {field}")
    if augmentation_binding.get("variantsPerParent") != 3:
        raise SuccessorV2SelectionError("successor V2 contract requires exactly three R3 variants per FIT parent")
    extractor = document.get("featureExtractor")
    if not isinstance(extractor, dict) or not extractor or document.get("featureExtractorIdentitySha256") != canonical_json_sha256(extractor):
        raise SuccessorV2SelectionError("successor V2 selection contract feature extractor is unsafe")
    reports = document.get("candidateReports")
    if not isinstance(reports, list) or len(reports) != len(evaluator.PRE_REGISTERED_CANDIDATES):
        raise SuccessorV2SelectionError("successor V2 selection contract must bind every pre-registered candidate")
    parsed_reports: list[dict[str, Any]] = []
    expected_ids = [item["id"] for item in evaluator.PRE_REGISTERED_CANDIDATES]
    for item, expected_id in zip(reports, expected_ids, strict=True):
        binding = _require_exact_fields(item, name="successor V2 candidate report", required=CANDIDATE_BINDING_FIELDS)
        configuration = evaluator.validate_candidate_configuration(binding.get("candidateConfiguration"))
        if binding.get("candidateId") != expected_id or configuration["id"] != expected_id:
            raise SuccessorV2SelectionError("successor V2 candidate universe order is unsafe")
        if binding.get("candidateConfigurationSha256") != canonical_json_sha256(configuration):
            raise SuccessorV2SelectionError("successor V2 candidate configuration digest does not match")
        for field in ("developmentReportFileSha256", "developmentReportDeclaredSha256", "featureInputIdentitySha256", "calibrationInputIdentitySha256", "thresholdsIdentitySha256"):
            _require_sha256(binding.get(field), name=f"successor V2 candidate {field}")
        if binding.get("featureExtractorIdentitySha256") != document["featureExtractorIdentitySha256"]:
            raise SuccessorV2SelectionError("successor V2 candidate report does not share the frozen feature extractor")
        thresholds = binding.get("thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != set(evaluator.SUCCESSOR_V2_CATEGORIES):
            raise SuccessorV2SelectionError("successor V2 candidate thresholds are unsafe")
        parsed_thresholds = {category: _require_finite(thresholds[category], name=f"successor V2 threshold {category}", minimum=0.0) for category in sorted(thresholds)}
        if binding.get("thresholdsIdentitySha256") != canonical_json_sha256(parsed_thresholds):
            raise SuccessorV2SelectionError("successor V2 candidate thresholds digest does not match")
        policy = binding.get("prototypeInputPolicy")
        if policy != configuration["prototypeInputPolicy"]:
            raise SuccessorV2SelectionError("successor V2 candidate prototype policy does not match configuration")
        expected_augmentation = {
            "augmentationManifestFileSha256": augmentation_binding["manifestFileSha256"],
            "augmentationManifestDeclaredSha256": augmentation_binding["manifestDeclaredSha256"],
            "augmentationRecipeFileSha256": augmentation_binding["recipeFileSha256"],
        }
        if policy == evaluator.RAW_FIT_ONLY:
            if any(binding.get(field) is not None for field in expected_augmentation):
                raise SuccessorV2SelectionError("raw successor V2 candidate must not bind the R3 augmentation package")
        elif policy == evaluator.RAW_FIT_PLUS_AUGMENTATION_R3:
            for field, expected in expected_augmentation.items():
                if binding.get(field) != expected:
                    raise SuccessorV2SelectionError("R3 successor V2 candidate does not bind the exact contract augmentation package")
        else:  # evaluator.validate_candidate_configuration already makes this defensive branch unreachable.
            raise SuccessorV2SelectionError("successor V2 candidate prototype policy is unsupported")
        parsed_reports.append({**dict(binding), "candidateConfiguration": configuration, "thresholds": parsed_thresholds})
    if len({item["developmentReportFileSha256"] for item in parsed_reports}) != len(parsed_reports) or len({item["developmentReportDeclaredSha256"] for item in parsed_reports}) != len(parsed_reports):
        raise SuccessorV2SelectionError("successor V2 candidate report digests must be unique")
    if document.get("candidateUniverseIdentitySha256") != canonical_json_sha256([_candidate_binding_for_contract(item) for item in parsed_reports]):
        raise SuccessorV2SelectionError("successor V2 candidate universe digest does not match")
    selection_protocol_module_sha = _require_sha256(
        document.get("selectionProtocolModuleSha256"), name="successor V2 selection protocol module digest"
    )
    if selection_protocol_module_sha != sha256_file(Path(__file__)):
        raise SuccessorV2SelectionError("current successor V2 selection protocol module does not match the frozen development evidence")
    projection = _validate_contract_binding_projection({
        "augmentation": augmentation_binding,
        "prototypeInputIdentities": prototype_identities,
        "prototypeInputCounts": prototype_counts,
        "featureExtractor": extractor,
        "featureExtractorIdentitySha256": document["featureExtractorIdentitySha256"],
        "selectionProtocolModuleSha256": selection_protocol_module_sha,
        "candidateBindings": [_candidate_binding_for_contract(item) for item in parsed_reports],
        "candidateUniverseIdentitySha256": document["candidateUniverseIdentitySha256"],
    }, name="successor V2 selection contract development binding projection")
    _validate_contract_development_evidence_ledger(document, projection, repository_root=repository_root)
    if document.get("selectionGates") != SELECTION_GATES or document.get("selectionObjective") != SELECTION_OBJECTIVE:
        raise SuccessorV2SelectionError("successor V2 selection gates or objective differ from the pre-registered protocol")
    if document.get("contractSha256") != _document_digest(document, "contractSha256"):
        raise SuccessorV2SelectionError("successor V2 selection contract digest does not match")
    return {
        **document,
        "augmentation": dict(augmentation_binding),
        "consumptionRegistry": registry,
        "successorSelectionInputs": inputs,
        "prototypeInputIdentities": prototype_identities,
        "prototypeInputCounts": prototype_counts,
        "developmentEvidenceLedger": _validate_development_evidence_anchor(document["developmentEvidenceLedger"]),
        "selectionProtocolModuleSha256": selection_protocol_module_sha,
        "candidateReports": parsed_reports,
    }


def load_validated_successor_v2_selection_contract(
    contract_path: Path, *, repository_root: Path = REPOSITORY_ROOT
) -> tuple[dict[str, Any], str]:
    document, file_sha256 = _read_json(contract_path, description="successor V2 selection contract", repository_root=repository_root)
    return _validate_contract(document, repository_root=repository_root), file_sha256


def _load_closed_development_evidence(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    development_report_paths: list[Path],
    augmentation_manifest_path: Path,
    recipe_path: Path,
    *,
    expected_projection: dict[str, Any] | None,
    repository_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str, str, list[dict[str, Any]], dict[str, Any]]:
    """Load only sealed JSON/report evidence; never selection source bytes."""

    if not isinstance(development_report_paths, list) or len(development_report_paths) != len(evaluator.PRE_REGISTERED_CANDIDATES):
        raise SuccessorV2SelectionError("provide exactly four successor V2 development reports")
    envelope, envelope_file_sha256 = _load_parent_chain(
        parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, repository_root=repository_root
    )
    r3_manifest, r3_manifest_file_sha256, r3_recipe_sha256 = _load_r3_manifest_json(
        augmentation_manifest_path, recipe_path, envelope=envelope, envelope_file_sha256=envelope_file_sha256,
        repository_root=repository_root,
    )
    expected_bindings = None if expected_projection is None else expected_projection["candidateBindings"]
    bindings: list[dict[str, Any]] = []
    for index, report_path in enumerate(development_report_paths):
        report, report_file_sha256 = _read_json(report_path, description="successor V2 development report", repository_root=repository_root)
        if expected_bindings is not None:
            expected = expected_bindings[index]
            configuration = report.get("candidateConfiguration")
            if (
                report_file_sha256 != expected["developmentReportFileSha256"]
                or report.get("developmentReportSha256") != expected["developmentReportDeclaredSha256"]
                or not isinstance(configuration, dict)
                or configuration.get("id") != expected["candidateId"]
            ):
                raise SuccessorV2SelectionError("supplied development report does not match the immutable ledger before semantic validation")
        binding = _validate_report(
            report, report_file_sha256=report_file_sha256, envelope=envelope, envelope_file_sha256=envelope_file_sha256,
            r3_manifest=r3_manifest, r3_manifest_file_sha256=r3_manifest_file_sha256, r3_recipe_sha256=r3_recipe_sha256,
        )
        if expected_bindings is not None and _candidate_binding_for_contract(binding) != expected_bindings[index]:
            raise SuccessorV2SelectionError("semantically validated development report does not match the immutable ledger projection")
        bindings.append(binding)
    expected_ids = [item["id"] for item in evaluator.PRE_REGISTERED_CANDIDATES]
    if [item["candidateId"] for item in bindings] != expected_ids:
        raise SuccessorV2SelectionError("development reports must be supplied in the pre-registered candidate order")
    extractor = bindings[0]["featureExtractor"]
    extractor_digest = bindings[0]["featureExtractorIdentitySha256"]
    if any(item["featureExtractor"] != extractor or item["featureExtractorIdentitySha256"] != extractor_digest for item in bindings):
        raise SuccessorV2SelectionError("all successor V2 reports must share one feature extractor identity")
    raw_feature_inputs, r3_feature_inputs = _expected_feature_inputs(envelope, r3_manifest=r3_manifest)
    prototype_identities, prototype_counts = _prototype_input_binding_from_identities(
        raw_feature_inputs, r3_feature_inputs, name="successor V2 development evidence"
    )
    projection = _validate_contract_binding_projection({
        "augmentation": {
            "manifestFileSha256": r3_manifest_file_sha256,
            "manifestDeclaredSha256": r3_manifest["augmentationManifestSha256"],
            "recipeFileSha256": r3_recipe_sha256,
            "successorFitIdentitySha256": envelope["successorPartitionIdentities"]["FIT"],
            "variantsPerParent": 3,
        },
        "prototypeInputIdentities": prototype_identities,
        "prototypeInputCounts": prototype_counts,
        "featureExtractor": extractor, "featureExtractorIdentitySha256": extractor_digest,
        "selectionProtocolModuleSha256": sha256_file(Path(__file__)),
        "candidateBindings": [_candidate_binding_for_contract(item) for item in bindings],
        "candidateUniverseIdentitySha256": canonical_json_sha256([_candidate_binding_for_contract(item) for item in bindings]),
    }, name="successor V2 closed development evidence projection")
    if expected_projection is not None and projection != expected_projection:
        raise SuccessorV2SelectionError("closed development evidence projection does not match the immutable ledger")
    return envelope, envelope_file_sha256, r3_manifest, r3_manifest_file_sha256, r3_recipe_sha256, bindings, projection


def freeze_successor_v2_development_evidence_ledger(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    development_report_paths: list[Path],
    augmentation_manifest_path: Path,
    recipe_path: Path,
    output_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Create an external review artifact before it is patched into tracked Git history."""

    _envelope, _envelope_file, _r3, _r3_file, _recipe_file, _bindings, projection = _load_closed_development_evidence(
        parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, development_report_paths,
        augmentation_manifest_path, recipe_path, expected_projection=None, repository_root=repository_root,
    )
    document = _development_evidence_ledger_document(projection)
    _validate_development_evidence_ledger(document)
    _write_slot(output_path, document, description="successor V2 development evidence ledger", repository_root=repository_root)
    return document


def create_successor_v2_selection_contract(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    development_report_paths: list[Path],
    augmentation_manifest_path: Path,
    recipe_path: Path,
    development_evidence_ledger_path: Path,
    output_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Freeze every V2 report before the successor selection partition is read."""

    ledger, ledger_anchor = _load_head_development_evidence_ledger(
        development_evidence_ledger_path, repository_root=repository_root
    )
    envelope, envelope_file_sha256, _r3_manifest, _r3_file, _recipe_file, _bindings, projection = _load_closed_development_evidence(
        parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, development_report_paths,
        augmentation_manifest_path, recipe_path, expected_projection=ledger["contractBindingProjection"],
        repository_root=repository_root,
    )
    registry = _consumption_registry_for_parent_contract(
        parent_selection_contract_path,
        parent_evidence=envelope["parentEvidence"],
        successor_selection_identity=envelope["successorPartitionIdentities"]["NORMAL_SELECTION"],
        repository_root=repository_root,
    )
    inputs = _selection_inputs_from_envelope(envelope)
    document: dict[str, Any] = {
        "schemaVersion": SUCCESSOR_V2_SELECTION_CONTRACT_SCHEMA, "authoritative": False, "productionAuthorized": False,
        "purpose": CONTRACT_PURPOSE, "phase": CONTRACT_PHASE, "blindPolicy": BLIND_POLICY, "resultLabel": RESULT_LABEL,
        "independenceLabel": INDEPENDENCE_LABEL, "delegationPolicy": DELEGATION_POLICY, "selectionInput": dict(SELECTION_INPUT),
        "parentEvidence": envelope["parentEvidence"], "successorPlanFileSha256": envelope["planFileSha256"],
        "successorPlanDeclaredSha256": envelope["planDeclaredSha256"], "successorEnvelopeFileSha256": envelope_file_sha256,
        "successorEnvelopeDeclaredSha256": envelope["successorEnvelopeSha256"],
        "successorEnvelopeSelectionIdentitySha256": envelope["successorPartitionIdentities"]["NORMAL_SELECTION"],
        "successorSelectionInputs": inputs,
        "successorSelectionInputIdentitySha256": canonical_json_sha256(inputs),
        "prototypeInputIdentities": projection["prototypeInputIdentities"], "prototypeInputCounts": projection["prototypeInputCounts"],
        "developmentEvidenceLedger": ledger_anchor, "selectionProtocolModuleSha256": projection["selectionProtocolModuleSha256"],
        "augmentation": projection["augmentation"],
        "consumptionRegistry": registry,
        "featureExtractor": projection["featureExtractor"], "featureExtractorIdentitySha256": projection["featureExtractorIdentitySha256"],
        "candidateReports": projection["candidateBindings"],
        "candidateUniverseIdentitySha256": projection["candidateUniverseIdentitySha256"],
        "selectionGates": dict(SELECTION_GATES), "selectionObjective": dict(SELECTION_OBJECTIVE),
    }
    document["contractSha256"] = _document_digest(document, "contractSha256")
    _validate_contract(document, repository_root=repository_root)
    _write_slot(output_path, document, description="successor V2 selection contract", repository_root=repository_root)
    return document


def _canonical_registry_root(root_value: object, *, repository_root: Path) -> Path:
    """Validate an external registry root without resolving link ancestry first."""

    root_text = _require_string(root_value, name="successor V2 consumption registry root")
    root = Path(root_text)
    if not root.is_absolute():
        raise SuccessorV2SelectionError("successor V2 consumption registry root must be absolute")
    _reject_links(root.parent, description="successor V2 partition access directory")
    if (root.exists() or root.is_symlink()) and _is_link_or_reparse_point(root):
        raise SuccessorV2SelectionError("successor V2 partition access directory contains a symbolic link or reparse point")
    if root.exists() and not root.is_dir():
        raise SuccessorV2SelectionError("successor V2 partition access directory must be a directory when it exists")
    resolved = root.resolve()
    # The ancestry was checked before resolve, and a second check catches a
    # race that turns a previously ordinary directory into a reparse point.
    _reject_links(resolved.parent, description="successor V2 partition access directory")
    if (resolved.exists() or resolved.is_symlink()) and _is_link_or_reparse_point(resolved):
        raise SuccessorV2SelectionError("successor V2 partition access directory contains a symbolic link or reparse point")
    if _is_under(repository_root, resolved) or _is_under(resolved, repository_root):
        raise SuccessorV2SelectionError("successor V2 partition access directory must stay outside the Git working tree")
    return resolved


def _consumption_registry_for_parent_contract(
    parent_selection_contract_path: Path,
    *,
    parent_evidence: dict[str, Any],
    successor_selection_identity: str,
    repository_root: Path,
) -> dict[str, Any]:
    """Anchor V2 slots to V1's immutable registry, never a caller's parent path.

    The schema-1.1 V1 contract already freezes the canonical registry root.
    Reusing it means a byte-for-byte copied parent package cannot relocate the
    V2 receipt/observation slots merely by changing a path argument.
    """

    try:
        parent_contract, parent_contract_file_sha256 = parent_protocol.load_validated_fresh_selection_contract(
            parent_selection_contract_path, repository_root=repository_root
        )
    except ValueError as error:
        raise SuccessorV2SelectionError("parent selection contract is unsafe") from error
    if (
        parent_contract_file_sha256 != parent_evidence.get("selectionContractFileSha256")
        or parent_contract.get("contractSha256") != parent_evidence.get("selectionContractDeclaredSha256")
    ):
        raise SuccessorV2SelectionError("parent selection contract does not match the sealed successor chain")
    parent_registry = parent_contract.get("consumptionRegistry")
    if not isinstance(parent_registry, dict):
        raise SuccessorV2SelectionError("parent selection contract consumption registry is missing")
    root = _canonical_registry_root(parent_registry.get("root"), repository_root=repository_root)
    selection_identity = _require_sha256(successor_selection_identity, name="successor envelope selection identity")
    slot_key = canonical_json_sha256({
        "schemaVersion": CONSUMPTION_REGISTRY_SCHEMA,
        "parentPartitionAccessRoot": str(root),
        "parentHoldoutFileSha256": parent_evidence["holdoutManifestFileSha256"],
        "parentHoldoutDeclaredSha256": parent_evidence["holdoutManifestDeclaredSha256"],
        "successorEnvelopeSelectionIdentitySha256": selection_identity,
    })
    return {"schemaVersion": CONSUMPTION_REGISTRY_SCHEMA, "root": str(root), "selectionSlotKey": slot_key}


def _validate_contract_against_parent_registry(
    parent_selection_contract_path: Path,
    contract: dict[str, Any],
    *,
    contract_file_sha256: str,
    repository_root: Path,
) -> None:
    """Reject a self-digested V2 contract that attempts to relocate its slot.

    The V2 contract's own digest protects accidental corruption but cannot be
    treated as authority for a registry root.  This comparison anchors every
    consuming phase to the canonical root already frozen in the parent V1
    schema-1.1 contract.
    """

    _require_sha256(contract_file_sha256, name="successor V2 selection contract file digest")
    expected = _consumption_registry_for_parent_contract(
        parent_selection_contract_path,
        parent_evidence=contract["parentEvidence"],
        successor_selection_identity=contract["successorEnvelopeSelectionIdentitySha256"],
        repository_root=repository_root,
    )
    if contract.get("consumptionRegistry") != expected:
        raise SuccessorV2SelectionError("successor V2 consumption registry does not match the canonical parent V1 registry")


def _validate_consumption_registry(value: object, *, contract: dict[str, Any], repository_root: Path) -> dict[str, Any]:
    registry = _require_exact_fields(value, name="successor V2 consumption registry", required=CONSUMPTION_REGISTRY_FIELDS)
    if registry.get("schemaVersion") != CONSUMPTION_REGISTRY_SCHEMA:
        raise SuccessorV2SelectionError("successor V2 consumption registry schema is unsupported")
    root = _canonical_registry_root(registry.get("root"), repository_root=repository_root)
    if registry.get("root") != str(root):
        raise SuccessorV2SelectionError("successor V2 consumption registry root must be canonical")
    parent = contract["parentEvidence"]
    expected_key = canonical_json_sha256({
        "schemaVersion": CONSUMPTION_REGISTRY_SCHEMA,
        "parentPartitionAccessRoot": str(root),
        "parentHoldoutFileSha256": parent["holdoutManifestFileSha256"],
        "parentHoldoutDeclaredSha256": parent["holdoutManifestDeclaredSha256"],
        "successorEnvelopeSelectionIdentitySha256": contract["successorEnvelopeSelectionIdentitySha256"],
    })
    if _require_sha256(registry.get("selectionSlotKey"), name="successor V2 selection slot key") != expected_key:
        raise SuccessorV2SelectionError("successor V2 selection slot key does not bind the parent registry and envelope")
    return {"schemaVersion": CONSUMPTION_REGISTRY_SCHEMA, "root": str(root), "selectionSlotKey": expected_key}


def successor_v2_selection_path(
    parent_holdout_or_contract: Path | dict[str, Any], contract: dict[str, Any] | None = None,
    *, artifact: str, repository_root: Path = REPOSITORY_ROOT
) -> Path:
    """Return the V2 global slot using *only* the contract-bound V1 registry.

    ``parent_holdout_or_contract`` accepts the former ``(parent, contract)``
    calling convention for compatibility, but the supplied parent path is not
    used to derive the result.  New callers should pass the contract alone.
    """

    bound_contract = parent_holdout_or_contract if contract is None else contract
    if not isinstance(bound_contract, dict):
        raise SuccessorV2SelectionError("successor V2 selection path requires a contract")
    _validate_contract(bound_contract, repository_root=repository_root)
    if artifact not in {"claim", "receipt", "observation", "lock"}:
        raise SuccessorV2SelectionError("successor V2 selection artifact is unsupported")
    registry = _validate_consumption_registry(bound_contract["consumptionRegistry"], contract=bound_contract, repository_root=repository_root)
    return Path(registry["root"]) / f"successor-v2-selection--{registry['selectionSlotKey'][7:]}.{artifact}.json"


def _claim_document(contract: dict[str, Any], *, contract_file_sha256: str) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schemaVersion": SUCCESSOR_V2_SELECTION_CLAIM_SCHEMA, "authoritative": False, "productionAuthorized": False,
        "purpose": CLAIM_PURPOSE, "phase": CLAIM_PHASE, "blindPolicy": BLIND_POLICY, "resultLabel": RESULT_LABEL,
        "delegationPolicy": DELEGATION_POLICY, "selectionInput": dict(SELECTION_INPUT),
        "claimSlot": "SUCCESSOR_V2_SELECTION_CONSUMPTION_V1", "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"],
        "parentHoldoutFileSha256": contract["parentEvidence"]["holdoutManifestFileSha256"],
        "successorEnvelopeDeclaredSha256": contract["successorEnvelopeDeclaredSha256"],
        "successorSelectionInputIdentitySha256": contract["successorSelectionInputIdentitySha256"],
        "candidateUniverseIdentitySha256": contract["candidateUniverseIdentitySha256"],
    }
    document["claimSha256"] = _document_digest(document, "claimSha256")
    return document


def _validate_claim(document: dict[str, Any], *, contract: dict[str, Any], contract_file_sha256: str) -> dict[str, Any]:
    _require_exact_fields(document, name="successor V2 selection claim", required=CLAIM_FIELDS)
    _static_scope(document, schema=SUCCESSOR_V2_SELECTION_CLAIM_SCHEMA, purpose=CLAIM_PURPOSE, phase=CLAIM_PHASE, name="successor V2 selection claim")
    expected = _claim_document(contract, contract_file_sha256=contract_file_sha256)
    if document != expected:
        raise SuccessorV2SelectionError("successor V2 selection claim does not bind the frozen contract")
    return document


def create_successor_v2_selection_claim(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    contract, contract_file_sha256 = load_validated_successor_v2_selection_contract(contract_path, repository_root=repository_root)
    _validate_contract_chain(
        parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, contract,
        contract_file_sha256=contract_file_sha256, repository_root=repository_root,
    )
    claim = _claim_document(contract, contract_file_sha256=contract_file_sha256)
    _write_slot(successor_v2_selection_path(contract, artifact="claim", repository_root=repository_root), claim,
                description="successor V2 selection claim", repository_root=repository_root)
    return claim


def _load_claim(
    parent_selection_contract_path: Path,
    contract: dict[str, Any],
    *,
    contract_file_sha256: str,
    repository_root: Path,
) -> tuple[dict[str, Any], str]:
    _validate_contract_against_parent_registry(
        parent_selection_contract_path, contract, contract_file_sha256=contract_file_sha256, repository_root=repository_root
    )
    document, file_sha256 = _read_json(
        successor_v2_selection_path(contract, artifact="claim", repository_root=repository_root),
        description="successor V2 selection claim", repository_root=repository_root,
    )
    return _validate_claim(document, contract=contract, contract_file_sha256=contract_file_sha256), file_sha256


def _validate_contract_chain(
    parent_holdout_path: Path, parent_selection_contract_path: Path, plan_path: Path, envelope_path: Path,
    contract: dict[str, Any], *, contract_file_sha256: str, repository_root: Path,
) -> tuple[dict[str, Any], str]:
    # Repeat this parent-V1 binding immediately before the phase-safe chain
    # validation so a caller cannot redirect a self-digested V2 contract to a
    # second registry root between JSON-only claim and query preflight.
    _validate_contract_against_parent_registry(
        parent_selection_contract_path,
        contract,
        contract_file_sha256=contract_file_sha256,
        repository_root=repository_root,
    )
    envelope, envelope_file_sha256 = _load_parent_chain(parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, repository_root=repository_root)
    expected = {
        "parentEvidence": envelope["parentEvidence"], "successorPlanFileSha256": envelope["planFileSha256"],
        "successorPlanDeclaredSha256": envelope["planDeclaredSha256"], "successorEnvelopeFileSha256": envelope_file_sha256,
        "successorEnvelopeDeclaredSha256": envelope["successorEnvelopeSha256"],
        "successorEnvelopeSelectionIdentitySha256": envelope["successorPartitionIdentities"]["NORMAL_SELECTION"],
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise SuccessorV2SelectionError("successor V2 selection contract no longer binds the parent/plan/envelope chain")
    if contract["successorSelectionInputs"] != _selection_inputs_from_envelope(envelope):
        raise SuccessorV2SelectionError("successor V2 selection contract selection membership differs from the envelope")
    return envelope, envelope_file_sha256


def _assert_identity(
    contract: dict[str, Any], *, model_repo: Path, model_weights: Path, device: str,
    identity_factory: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    if not isinstance(identity, dict) or identity != contract["featureExtractor"]:
        raise SuccessorV2SelectionError("current feature extractor does not match frozen successor V2 development reports")
    if canonical_json_sha256(identity) != contract["featureExtractorIdentitySha256"]:
        raise SuccessorV2SelectionError("current feature extractor identity digest does not match the contract")
    return identity


def _load_prototype_records(
    parent_holdout_path: Path, parent_selection_contract_path: Path, plan_path: Path, envelope_path: Path,
    contract: dict[str, Any], *, source_root: Path, recipe_path: Path, repository_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    """Perform FIT/R3-only preflight; no tuning/selection/reserve bytes are requested."""

    try:
        envelope, envelope_file_sha256, raw_records = successor.load_successor_safe_normal_inputs(
            parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, source_root=source_root,
            partitions={"FIT"}, repository_root=repository_root,
        )
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error
    raw = sorted([evaluator._adapt_successor_original_record(record) for record in raw_records], key=lambda record: str(record["caseId"]))
    manifest_path = Path(contract["augmentation"].get("manifestPath", ""))
    # A caller supplies the manifest path to the observer; it is checked by the outer function before this helper.
    _ = manifest_path
    return raw, [], envelope, envelope_file_sha256


def _build_prototypes(
    records: list[dict[str, Any]], features: dict[str, object], configuration: dict[str, Any]
) -> dict[str, object]:
    import numpy as np
    result: dict[str, object] = {}
    for category in evaluator.SUCCESSOR_V2_CATEGORIES:
        category_records = [record for record in records if record["category"] == category and record["partition"] == "FIT"]
        matrices = [np.asarray(features[str(record["caseId"])], dtype=np.float32) for record in category_records]
        if not matrices:
            raise SuccessorV2SelectionError("successor V2 prototype preflight has no FIT records")
        patch_counts = [int(matrix.shape[0]) if matrix.ndim == 2 else 0 for matrix in matrices]
        try:
            indices = evaluator.deterministic_stratified_hash_ranked_patch_indices(
                category_records, patch_counts, int(configuration["maxPrototypePatches"])
            )
        except ValueError as error:
            raise SuccessorV2SelectionError(str(error)) from error
        result[category] = np.concatenate(matrices, axis=0)[indices]
    return result


def _adapt_selection_source(record: dict[str, Any]) -> dict[str, Any]:
    """Adapt an already phase-safely loaded successor selection original."""

    if record.get("partition") != "NORMAL_SELECTION" or record.get("kind") != "NOMINAL" or record.get("defect") != "good":
        raise SuccessorV2SelectionError("selection observer received an unsafe successor query record")
    if not isinstance(record.get("imagePath"), Path):
        raise SuccessorV2SelectionError("selection observer received no phase-safe query image path")
    for field in ("caseId", "category", "sourceGroupId"):
        _require_string(record.get(field), name=f"successor selection {field}")
    _require_sha256(record.get("sourceSha256"), name="successor selection sourceSha256")
    return {
        **record,
        "isAugmentation": False,
        "variantId": None,
        "component": None,
        "parentCaseId": None,
        "parentSourceSha256": None,
        "augmentationManifestSha256": None,
    }


def _p95(values: list[float]) -> float:
    if not values:
        raise SuccessorV2SelectionError("P95 requires values")
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


def _score_candidate(
    candidate: dict[str, Any], query_records: list[dict[str, Any]], query_features: dict[str, object], prototypes: dict[str, object]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    import numpy as np
    scores: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    configuration = candidate["candidateConfiguration"]
    for category in evaluator.SUCCESSOR_V2_CATEGORIES:
        category_records = [record for record in query_records if record["category"] == category]
        query = np.asarray([query_features[str(record["caseId"])] for record in category_records], dtype=np.float32)
        try:
            values = evaluator.patch_knn_scores_blocked(
                query, prototypes[category], top_k=int(configuration["topKMostAnomalousPatches"]),
                prototype_block_size=int(configuration["prototypeBlockSize"]),
            )
        except ValueError as error:
            raise SuccessorV2SelectionError(str(error)) from error
        threshold = float(candidate["thresholds"][category])
        category_scores: list[float] = []
        for record, component in zip(category_records, values, strict=True):
            score = _require_finite(component["score"], name="successor V2 selection score", minimum=0.0)
            category_scores.append(score)
            scores.append({
                "caseId": record["caseId"], "category": category, "partition": "NORMAL_SELECTION",
                "kind": "NOMINAL", "defect": "good", "sourceSha256": record["sourceSha256"], "score": score,
                "maxPatchDistance": _require_finite(component["maxPatchDistance"], name="maxPatchDistance", minimum=0.0),
                "meanNearestPatchDistance": _require_finite(component["meanNearestPatchDistance"], name="meanNearestPatchDistance", minimum=0.0),
            })
        above = sum(value > threshold for value in category_scores)
        metrics[category] = {
            "queryCount": len(category_scores), "aboveThresholdCount": above, "aboveThresholdRate": above / len(category_scores),
            "p95Score": _p95(category_scores), "maximumScore": max(category_scores),
            "p95ScoreMinusThreshold": _p95(category_scores) - threshold,
            "maximumScoreMinusThreshold": max(category_scores) - threshold,
        }
    scores.sort(key=lambda item: item["caseId"])
    return scores, metrics


def _receipt(
    contract: dict[str, Any], *, contract_file_sha256: str, claim: dict[str, Any], claim_file_sha256: str,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schemaVersion": SUCCESSOR_V2_SELECTION_RECEIPT_SCHEMA, "authoritative": False, "productionAuthorized": False,
        "purpose": RECEIPT_PURPOSE, "phase": RECEIPT_PHASE, "blindPolicy": BLIND_POLICY, "resultLabel": RESULT_LABEL,
        "delegationPolicy": DELEGATION_POLICY, "selectionInput": dict(SELECTION_INPUT),
        "receiptSlot": "SUCCESSOR_V2_SELECTION_RECEIPT_V1", "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"], "selectionClaimFileSha256": claim_file_sha256,
        "selectionClaimDeclaredSha256": claim["claimSha256"], "successorEnvelopeDeclaredSha256": contract["successorEnvelopeDeclaredSha256"],
        "successorSelectionInputIdentitySha256": contract["successorSelectionInputIdentitySha256"],
        "candidateUniverseIdentitySha256": contract["candidateUniverseIdentitySha256"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "prototypeInputIdentities": contract["prototypeInputIdentities"],
        "prototypeInputCounts": contract["prototypeInputCounts"],
    }
    document["selectionReceiptSha256"] = _document_digest(document, "selectionReceiptSha256")
    return document


def _validate_receipt(document: dict[str, Any], *, contract: dict[str, Any], contract_file_sha256: str, claim: dict[str, Any], claim_file_sha256: str) -> dict[str, Any]:
    _require_exact_fields(document, name="successor V2 selection receipt", required=RECEIPT_FIELDS)
    _static_scope(document, schema=SUCCESSOR_V2_SELECTION_RECEIPT_SCHEMA, purpose=RECEIPT_PURPOSE, phase=RECEIPT_PHASE, name="successor V2 selection receipt")
    base = {
        "selectionInput": SELECTION_INPUT, "contractFileSha256": contract_file_sha256, "contractDeclaredSha256": contract["contractSha256"],
        "selectionClaimFileSha256": claim_file_sha256, "selectionClaimDeclaredSha256": claim["claimSha256"],
        "successorEnvelopeDeclaredSha256": contract["successorEnvelopeDeclaredSha256"],
        "successorSelectionInputIdentitySha256": contract["successorSelectionInputIdentitySha256"],
        "candidateUniverseIdentitySha256": contract["candidateUniverseIdentitySha256"], "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
    }
    for field, expected in base.items():
        if document.get(field) != expected:
            raise SuccessorV2SelectionError("successor V2 selection receipt does not bind frozen JSON evidence")
    if document.get("receiptSlot") != "SUCCESSOR_V2_SELECTION_RECEIPT_V1":
        raise SuccessorV2SelectionError("successor V2 selection receipt slot is unsafe")
    contract_identities, contract_counts = _validate_prototype_input_binding(
        contract.get("prototypeInputIdentities"), contract.get("prototypeInputCounts"),
        name="successor V2 selection contract",
    )
    receipt_identities, receipt_counts = _validate_prototype_input_binding(
        document.get("prototypeInputIdentities"), document.get("prototypeInputCounts"),
        name="successor V2 selection receipt",
    )
    if receipt_identities != contract_identities or receipt_counts != contract_counts:
        raise SuccessorV2SelectionError("successor V2 selection receipt prototype maps do not match the frozen contract")
    if document.get("selectionReceiptSha256") != _document_digest(document, "selectionReceiptSha256"):
        raise SuccessorV2SelectionError("successor V2 selection receipt digest does not match")
    return document


def run_successor_v2_selection_observation(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    contract_path: Path,
    augmentation_manifest_path: Path,
    recipe_path: Path,
    *,
    source_root: Path,
    model_repo: Path,
    model_weights: Path,
    device: str = "cpu",
    repository_root: Path = REPOSITORY_ROOT,
    embedder_factory: Callable[..., Any] = evaluator.SuccessorV2BatchEmbedder,
    identity_factory: Callable[..., dict[str, Any]] = evaluator._feature_extractor_identity,
) -> dict[str, Any]:
    """Consume the 24 successor selection normals exactly once for all four candidates."""

    if device != "cpu":
        raise SuccessorV2SelectionError("successor V2 selection supports CPU only")
    started = time.perf_counter()
    timings = {name: 0.0 for name in ("preflightSeconds", "receiptCommitSeconds", "queryFeatureSeconds", "scoringSeconds", "totalElapsedSeconds")}
    contract, contract_file_sha256 = load_validated_successor_v2_selection_contract(contract_path, repository_root=repository_root)
    claim, claim_file_sha256 = _load_claim(parent_selection_contract_path, contract, contract_file_sha256=contract_file_sha256, repository_root=repository_root)
    envelope, envelope_file_sha256 = _validate_contract_chain(
        parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, contract,
        contract_file_sha256=contract_file_sha256, repository_root=repository_root
    )
    receipt_path = successor_v2_selection_path(contract, artifact="receipt", repository_root=repository_root)
    observation_path = successor_v2_selection_path(contract, artifact="observation", repository_root=repository_root)
    _prepare_slot(receipt_path, description="successor V2 selection receipt", repository_root=repository_root)
    _prepare_slot(observation_path, description="successor V2 selection observation", repository_root=repository_root)
    preflight_started = time.perf_counter()
    # FIT is the sole original image partition permitted before receipt.
    try:
        _fit_envelope, _fit_file_sha256, raw_fit_sources = successor.load_successor_safe_normal_inputs(
            parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, source_root=source_root,
            partitions={"FIT"}, repository_root=repository_root,
        )
        r3_document, r3_file_sha256, r3_records = augmentation.load_validated_successor_fit_augmentations_with_file_sha256(
            augmentation_manifest_path, parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path,
            source_root=source_root, recipe_path=recipe_path, repository_root=repository_root,
        )
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error
    if r3_file_sha256 != contract["augmentation"]["manifestFileSha256"] or r3_document.get("augmentationManifestSha256") != contract["augmentation"]["manifestDeclaredSha256"]:
        raise SuccessorV2SelectionError("validated R3 augmentation package does not match the frozen contract")
    if sha256_file(recipe_path) != contract["augmentation"]["recipeFileSha256"]:
        raise SuccessorV2SelectionError("validated R3 recipe does not match the frozen contract")
    raw_fit = sorted([evaluator._adapt_successor_original_record(record) for record in raw_fit_sources], key=lambda record: str(record["caseId"]))
    r3_fit = sorted(raw_fit + [
        evaluator._adapt_augmentation_record(record, augmentation_root=augmentation_manifest_path.parent,
                                               augmentation_manifest_sha256=r3_document["augmentationManifestSha256"], repository_root=repository_root)
        for record in r3_records
    ], key=lambda record: str(record["caseId"]))
    raw_identity = evaluator._feature_input_identity(raw_fit)
    r3_identity = evaluator._feature_input_identity(r3_fit)
    prototype_identities, prototype_counts = _prototype_input_binding_from_identities(
        raw_identity, r3_identity, name="successor V2 FIT/R3 preflight"
    )
    contract_identities, contract_counts = _validate_prototype_input_binding(
        contract["prototypeInputIdentities"], contract["prototypeInputCounts"],
        name="successor V2 selection contract",
    )
    if prototype_identities != contract_identities or prototype_counts != contract_counts:
        raise SuccessorV2SelectionError("successor V2 FIT/R3 preflight prototype membership does not match the frozen contract")
    candidates = contract["candidateReports"]
    identity = _assert_identity(contract, model_repo=model_repo, model_weights=model_weights, device=device, identity_factory=identity_factory)
    try:
        embedder = embedder_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    except Exception as error:
        raise SuccessorV2SelectionError("unable to load frozen successor V2 feature extractor") from error
    if _assert_identity(contract, model_repo=model_repo, model_weights=model_weights, device=device, identity_factory=identity_factory) != identity:
        raise SuccessorV2SelectionError("feature extractor changed while DINO loaded")
    all_fit = sorted({record["caseId"]: record for record in r3_fit}.values(), key=lambda record: str(record["caseId"]))
    feature_timings = {"inputVerificationSeconds": 0.0, "featureInferenceSeconds": 0.0}
    try:
        fit_features = evaluator._extract_patch_features(all_fit, embedder=embedder, batch_size=4, timings=feature_timings)
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error
    prototypes = {
        candidate["candidateId"]: _build_prototypes(
            raw_fit if candidate["prototypeInputPolicy"] == evaluator.RAW_FIT_ONLY else r3_fit,
            fit_features, candidate["candidateConfiguration"],
        )
        for candidate in candidates
    }
    timings["preflightSeconds"] = time.perf_counter() - preflight_started
    current_contract, current_contract_file = load_validated_successor_v2_selection_contract(contract_path, repository_root=repository_root)
    current_claim, current_claim_file = _load_claim(parent_selection_contract_path, current_contract, contract_file_sha256=current_contract_file, repository_root=repository_root)
    if current_contract != contract or current_contract_file != contract_file_sha256 or current_claim != claim or current_claim_file != claim_file_sha256:
        raise SuccessorV2SelectionError("selection contract or claim changed during FIT/R3 preflight")
    _assert_identity(contract, model_repo=model_repo, model_weights=model_weights, device=device, identity_factory=identity_factory)
    receipt = _receipt(
        contract, contract_file_sha256=contract_file_sha256, claim=claim, claim_file_sha256=claim_file_sha256,
    )
    _validate_receipt(receipt, contract=contract, contract_file_sha256=contract_file_sha256, claim=claim, claim_file_sha256=claim_file_sha256)
    receipt_started = time.perf_counter()
    _write_slot(receipt_path, receipt, description="successor V2 selection receipt", repository_root=repository_root)
    timings["receiptCommitSeconds"] = time.perf_counter() - receipt_started
    # This is the first call that can receive NORMAL_SELECTION image paths.
    try:
        _query_envelope, query_envelope_sha, query_sources = successor.load_successor_safe_normal_inputs(
            parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, source_root=source_root,
            partitions={"NORMAL_SELECTION"}, repository_root=repository_root,
        )
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error
    if query_envelope_sha != envelope_file_sha256:
        raise SuccessorV2SelectionError("successor envelope changed after the selection receipt was committed")
    query_records = sorted([_adapt_selection_source(record) for record in query_sources], key=lambda record: str(record["caseId"]))
    actual_inputs = [_normal_input(record) for record in query_records]
    if actual_inputs != contract["successorSelectionInputs"]:
        raise SuccessorV2SelectionError("successor selection query membership does not match frozen contract")
    query_timings = {"inputVerificationSeconds": 0.0, "featureInferenceSeconds": 0.0}
    feature_started = time.perf_counter()
    try:
        query_features = evaluator._extract_patch_features(query_records, embedder=embedder, batch_size=4, timings=query_timings)
    except ValueError as error:
        raise SuccessorV2SelectionError(str(error)) from error
    timings["queryFeatureSeconds"] = time.perf_counter() - feature_started
    candidate_observations: list[dict[str, Any]] = []
    for candidate in candidates:
        scoring_started = time.perf_counter()
        scores, metrics = _score_candidate(candidate, query_records, query_features, prototypes[candidate["candidateId"]])
        timings["scoringSeconds"] += time.perf_counter() - scoring_started
        candidate_observations.append({
            "candidateId": candidate["candidateId"], "candidateConfiguration": candidate["candidateConfiguration"],
            "candidateConfigurationSha256": candidate["candidateConfigurationSha256"], "thresholds": candidate["thresholds"],
            "thresholdsIdentitySha256": candidate["thresholdsIdentitySha256"], "selectionScores": scores, "categoryMetrics": metrics,
        })
    if _assert_identity(contract, model_repo=model_repo, model_weights=model_weights, device=device, identity_factory=identity_factory) != identity:
        raise SuccessorV2SelectionError("feature extractor changed while successor selection features were extracted")
    timings["totalElapsedSeconds"] = time.perf_counter() - started
    receipt_file_sha256 = sha256_file(receipt_path)
    observation: dict[str, Any] = {
        "schemaVersion": SUCCESSOR_V2_SELECTION_OBSERVATION_SCHEMA, "authoritative": False, "productionAuthorized": False,
        "purpose": OBSERVATION_PURPOSE, "phase": OBSERVATION_PHASE, "blindPolicy": BLIND_POLICY, "resultLabel": RESULT_LABEL,
        "independenceLabel": INDEPENDENCE_LABEL, "delegationPolicy": DELEGATION_POLICY, "selectionInput": dict(SELECTION_INPUT),
        "contractFileSha256": contract_file_sha256, "contractDeclaredSha256": contract["contractSha256"],
        "selectionClaimFileSha256": claim_file_sha256, "selectionClaimDeclaredSha256": claim["claimSha256"],
        "selectionReceiptFileSha256": receipt_file_sha256, "selectionReceiptDeclaredSha256": receipt["selectionReceiptSha256"],
        "parentEvidence": contract["parentEvidence"], "successorEnvelopeFileSha256": envelope_file_sha256,
        "successorEnvelopeDeclaredSha256": contract["successorEnvelopeDeclaredSha256"], "augmentation": contract["augmentation"],
        "featureExtractor": contract["featureExtractor"], "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "successorSelectionInputs": contract["successorSelectionInputs"],
        "successorSelectionInputIdentitySha256": contract["successorSelectionInputIdentitySha256"],
        "candidateObservations": candidate_observations,
        "normalOnlyEvidence": {
            "prototypeInputPartitions": ["FIT"], "queryInputCount": len(query_records),
            "queryInputPartitions": ["NORMAL_SELECTION"], "queryInputKinds": ["NOMINAL"],
            "blindInputCount": 0, "anomalyInputCount": 0, "maskInputCount": 0,
            "parentConfirmationInputCount": 0, "remainingReserveInputCount": 0,
            "persistentQueryCache": False, "queryCachePolicy": NO_PERSISTENT_QUERY_CACHE,
        },
        "execution": {"selectionModuleSha256": sha256_file(Path(__file__)), "phaseTimingsSeconds": timings},
    }
    observation["selectionObservationSha256"] = _document_digest(observation, "selectionObservationSha256")
    _validate_observation(observation, contract=contract, contract_file_sha256=contract_file_sha256, claim=claim,
                          claim_file_sha256=claim_file_sha256, receipt=receipt, receipt_file_sha256=receipt_file_sha256)
    _write_slot(observation_path, observation, description="successor V2 selection observation", repository_root=repository_root)
    return observation


def _validate_observation(
    document: dict[str, Any], *, contract: dict[str, Any], contract_file_sha256: str,
    claim: dict[str, Any], claim_file_sha256: str, receipt: dict[str, Any], receipt_file_sha256: str,
) -> dict[str, Any]:
    _require_exact_fields(document, name="successor V2 selection observation", required=OBSERVATION_FIELDS)
    _static_scope(document, schema=SUCCESSOR_V2_SELECTION_OBSERVATION_SCHEMA, purpose=OBSERVATION_PURPOSE,
                  phase=OBSERVATION_PHASE, name="successor V2 selection observation")
    if document.get("independenceLabel") != INDEPENDENCE_LABEL or document.get("selectionInput") != SELECTION_INPUT:
        raise SuccessorV2SelectionError("successor V2 selection observation scope is unsafe")
    expected = {
        "contractFileSha256": contract_file_sha256, "contractDeclaredSha256": contract["contractSha256"],
        "selectionClaimFileSha256": claim_file_sha256, "selectionClaimDeclaredSha256": claim["claimSha256"],
        "selectionReceiptFileSha256": receipt_file_sha256, "selectionReceiptDeclaredSha256": receipt["selectionReceiptSha256"],
        "parentEvidence": contract["parentEvidence"], "successorEnvelopeFileSha256": contract["successorEnvelopeFileSha256"],
        "successorEnvelopeDeclaredSha256": contract["successorEnvelopeDeclaredSha256"], "augmentation": contract["augmentation"],
        "featureExtractor": contract["featureExtractor"], "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "successorSelectionInputs": contract["successorSelectionInputs"],
        "successorSelectionInputIdentitySha256": contract["successorSelectionInputIdentitySha256"],
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise SuccessorV2SelectionError("successor V2 selection observation does not bind frozen evidence")
    observations = document.get("candidateObservations")
    if not isinstance(observations, list) or len(observations) != len(contract["candidateReports"]):
        raise SuccessorV2SelectionError("successor V2 selection observation does not cover every candidate")
    for observed, candidate in zip(observations, contract["candidateReports"], strict=True):
        _require_exact_fields(observed, name="successor V2 candidate observation", required=CANDIDATE_OBSERVATION_FIELDS)
        for field in ("candidateId", "candidateConfiguration", "candidateConfigurationSha256", "thresholds", "thresholdsIdentitySha256"):
            if observed.get(field) != candidate[field]:
                raise SuccessorV2SelectionError("successor V2 candidate observation does not bind frozen candidate")
        scores = observed.get("selectionScores")
        if not isinstance(scores, list) or len(scores) != 24:
            raise SuccessorV2SelectionError("successor V2 candidate observation must contain 24 selection scores")
        cases = []
        category_values: dict[str, list[float]] = {category: [] for category in evaluator.SUCCESSOR_V2_CATEGORIES}
        for score in scores:
            _require_exact_fields(score, name="successor V2 selection score", required=SELECTION_SCORE_FIELDS)
            case = _require_string(score.get("caseId"), name="selection score caseId")
            cases.append(case)
            expected_input = next((item for item in contract["successorSelectionInputs"] if item["caseId"] == case), None)
            if expected_input is None or any(score.get(field) != expected_input[field] for field in ("category", "partition", "kind", "defect", "sourceSha256")):
                raise SuccessorV2SelectionError("successor V2 selection score membership is unsafe")
            value = _require_finite(score.get("score"), name="selection score", minimum=0.0)
            maximum_distance = _require_finite(score.get("maxPatchDistance"), name="selection maxPatchDistance", minimum=0.0)
            mean_distance = _require_finite(score.get("meanNearestPatchDistance"), name="selection meanNearestPatchDistance", minimum=0.0)
            if not mean_distance <= value <= maximum_distance:
                raise SuccessorV2SelectionError("successor V2 selection score components are inconsistent")
            category_values[str(score["category"])].append(value)
        if cases != sorted(cases) or set(cases) != {item["caseId"] for item in contract["successorSelectionInputs"]}:
            raise SuccessorV2SelectionError("successor V2 selection scores do not cover frozen inputs once")
        metrics = observed.get("categoryMetrics")
        if not isinstance(metrics, dict) or set(metrics) != set(evaluator.SUCCESSOR_V2_CATEGORIES):
            raise SuccessorV2SelectionError("successor V2 category metrics are unsafe")
        for category, values in category_values.items():
            metric = _require_exact_fields(metrics[category], name="successor V2 category metric", required=CATEGORY_METRIC_FIELDS)
            threshold = float(candidate["thresholds"][category])
            expected_metric = {
                "queryCount": len(values), "aboveThresholdCount": sum(value > threshold for value in values),
                "aboveThresholdRate": sum(value > threshold for value in values) / len(values),
                "p95Score": _p95(values), "maximumScore": max(values), "p95ScoreMinusThreshold": _p95(values) - threshold,
                "maximumScoreMinusThreshold": max(values) - threshold,
            }
            if metric != expected_metric:
                raise SuccessorV2SelectionError("successor V2 category metrics do not recompute from scores")
    evidence = document.get("normalOnlyEvidence")
    _require_exact_fields(evidence, name="successor V2 normal-only evidence", required=NORMAL_ONLY_EVIDENCE_FIELDS)
    expected_evidence = {
        "prototypeInputPartitions": ["FIT"], "queryInputCount": 24,
        "queryInputPartitions": ["NORMAL_SELECTION"], "queryInputKinds": ["NOMINAL"],
        "blindInputCount": 0, "anomalyInputCount": 0, "maskInputCount": 0,
        "parentConfirmationInputCount": 0, "remainingReserveInputCount": 0,
        "persistentQueryCache": False, "queryCachePolicy": NO_PERSISTENT_QUERY_CACHE,
    }
    if evidence != expected_evidence:
        raise SuccessorV2SelectionError("successor V2 normal-only evidence is unsafe")
    execution = _require_exact_fields(document.get("execution"), name="successor V2 selection execution", required=OBSERVATION_EXECUTION_FIELDS)
    if _require_sha256(execution.get("selectionModuleSha256"), name="successor V2 selection execution module digest") != contract["selectionProtocolModuleSha256"]:
        raise SuccessorV2SelectionError("successor V2 selection observation execution module does not match the frozen protocol module")
    timings = execution.get("phaseTimingsSeconds")
    if not isinstance(timings, dict) or set(timings) != OBSERVATION_TIMING_FIELDS:
        raise SuccessorV2SelectionError("successor V2 selection execution timings are unsafe")
    for field, value in timings.items():
        _require_finite(value, name=f"successor V2 selection execution timing {field}", minimum=0.0, maximum=31_536_000.0)
    if document.get("selectionObservationSha256") != _document_digest(document, "selectionObservationSha256"):
        raise SuccessorV2SelectionError("successor V2 selection observation digest does not match")
    return document


def _load_receipt_and_observation(
    parent_holdout_path: Path, contract: dict[str, Any], *, contract_file_sha256: str, claim: dict[str, Any], claim_file_sha256: str,
    repository_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    receipt_path = successor_v2_selection_path(contract, artifact="receipt", repository_root=repository_root)
    receipt, receipt_file_sha256 = _read_json(receipt_path, description="successor V2 selection receipt", repository_root=repository_root)
    receipt = _validate_receipt(receipt, contract=contract, contract_file_sha256=contract_file_sha256, claim=claim, claim_file_sha256=claim_file_sha256)
    observation_path = successor_v2_selection_path(contract, artifact="observation", repository_root=repository_root)
    observation, observation_file_sha256 = _read_json(observation_path, description="successor V2 selection observation", repository_root=repository_root)
    observation = _validate_observation(observation, contract=contract, contract_file_sha256=contract_file_sha256, claim=claim,
                                        claim_file_sha256=claim_file_sha256, receipt=receipt, receipt_file_sha256=receipt_file_sha256)
    return receipt, receipt_file_sha256, observation, observation_file_sha256


def _rejection_reasons(metrics: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for category in evaluator.SUCCESSOR_V2_CATEGORIES:
        metric = metrics[category]
        if metric["aboveThresholdCount"] > SELECTION_GATES["maxAboveThresholdCount"]:
            reasons.append(f"{category}.maxAboveThresholdCount={metric['aboveThresholdCount']} exceeds 1")
        if metric["p95ScoreMinusThreshold"] > SELECTION_GATES["maxP95ScoreMinusThreshold"]:
            reasons.append(f"{category}.maxP95ScoreMinusThreshold={metric['p95ScoreMinusThreshold']:.12g} exceeds 0.05")
        if metric["maximumScoreMinusThreshold"] > SELECTION_GATES["maxMaximumScoreMinusThreshold"]:
            reasons.append(f"{category}.maxMaximumScoreMinusThreshold={metric['maximumScoreMinusThreshold']:.12g} exceeds 0.05")
    return reasons


def _objective(candidate_id: str, metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "worstMaximumScoreMinusThreshold": max(float(item["maximumScoreMinusThreshold"]) for item in metrics.values()),
        "worstAboveThresholdRate": max(float(item["aboveThresholdRate"]) for item in metrics.values()),
        "meanP95ScoreMinusThreshold": sum(float(item["p95ScoreMinusThreshold"]) for item in metrics.values()) / len(metrics),
        "candidateId": candidate_id,
    }


def _lock_evaluations(contract: dict[str, Any], observation: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {item["candidateId"]: item for item in observation["candidateObservations"]}
    result: list[dict[str, Any]] = []
    for candidate in contract["candidateReports"]:
        observed = by_id.get(candidate["candidateId"])
        if observed is None:
            raise SuccessorV2SelectionError("selection observation omitted a frozen successor V2 candidate")
        reasons = _rejection_reasons(observed["categoryMetrics"])
        result.append({
            "candidateId": candidate["candidateId"], "categoryMetrics": observed["categoryMetrics"], "gatePassed": not reasons,
            "gateRejectionReasons": reasons, "objectiveValues": None if reasons else _objective(candidate["candidateId"], observed["categoryMetrics"]),
        })
    return result


def _decision(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in evaluations if item["gatePassed"]]
    winner = min(eligible, key=lambda item: (
        item["objectiveValues"]["worstMaximumScoreMinusThreshold"], item["objectiveValues"]["worstAboveThresholdRate"],
        item["objectiveValues"]["meanP95ScoreMinusThreshold"], item["objectiveValues"]["candidateId"],
    )) if eligible else None
    return {
        "state": "RESEARCH_CONFIGURATION_LOCKED" if winner else "NO_ELIGIBLE_CONFIGURATION",
        "selectedCandidateId": None if winner is None else winner["candidateId"],
        "resultScope": "OFFLINE_RESEARCH_CONFIGURATION_LOCK_ONLY", "automaticProductionPromotion": False,
        "automaticConfirmation": False,
    }


def create_successor_v2_selection_lock(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Create the JSON-only lock.  It never creates a confirmation claim."""

    contract, contract_file_sha256 = load_validated_successor_v2_selection_contract(contract_path, repository_root=repository_root)
    _validate_contract_chain(
        parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, contract,
        contract_file_sha256=contract_file_sha256, repository_root=repository_root,
    )
    claim, claim_file_sha256 = _load_claim(parent_selection_contract_path, contract, contract_file_sha256=contract_file_sha256, repository_root=repository_root)
    receipt, receipt_file_sha256, observation, observation_file_sha256 = _load_receipt_and_observation(
        parent_holdout_path, contract, contract_file_sha256=contract_file_sha256, claim=claim, claim_file_sha256=claim_file_sha256,
        repository_root=repository_root,
    )
    evaluations = _lock_evaluations(contract, observation)
    document: dict[str, Any] = {
        "schemaVersion": SUCCESSOR_V2_SELECTION_LOCK_SCHEMA, "authoritative": False, "productionAuthorized": False,
        "purpose": LOCK_PURPOSE, "phase": LOCK_PHASE, "blindPolicy": BLIND_POLICY, "resultLabel": RESULT_LABEL,
        "delegationPolicy": DELEGATION_POLICY, "contractFileSha256": contract_file_sha256,
        "contractDeclaredSha256": contract["contractSha256"], "selectionProtocolModuleSha256": contract["selectionProtocolModuleSha256"], "selectionClaimFileSha256": claim_file_sha256,
        "selectionClaimDeclaredSha256": claim["claimSha256"], "selectionReceiptFileSha256": receipt_file_sha256,
        "selectionReceiptDeclaredSha256": receipt["selectionReceiptSha256"], "selectionObservationFileSha256": observation_file_sha256,
        "selectionObservationDeclaredSha256": observation["selectionObservationSha256"], "candidateEvaluations": evaluations,
        "decision": _decision(evaluations),
    }
    document["selectionLockSha256"] = _document_digest(document, "selectionLockSha256")
    _validate_lock(document, contract=contract, contract_file_sha256=contract_file_sha256, claim=claim, claim_file_sha256=claim_file_sha256,
                   receipt=receipt, receipt_file_sha256=receipt_file_sha256, observation=observation, observation_file_sha256=observation_file_sha256)
    _write_slot(successor_v2_selection_path(contract, artifact="lock", repository_root=repository_root), document,
                description="successor V2 selection lock", repository_root=repository_root)
    return document


def _validate_lock(
    document: dict[str, Any], *, contract: dict[str, Any], contract_file_sha256: str, claim: dict[str, Any],
    claim_file_sha256: str, receipt: dict[str, Any], receipt_file_sha256: str, observation: dict[str, Any], observation_file_sha256: str,
) -> dict[str, Any]:
    _require_exact_fields(document, name="successor V2 selection lock", required=LOCK_FIELDS)
    _static_scope(document, schema=SUCCESSOR_V2_SELECTION_LOCK_SCHEMA, purpose=LOCK_PURPOSE, phase=LOCK_PHASE, name="successor V2 selection lock")
    expected = {
        "contractFileSha256": contract_file_sha256, "contractDeclaredSha256": contract["contractSha256"],
        "selectionProtocolModuleSha256": contract["selectionProtocolModuleSha256"],
        "selectionClaimFileSha256": claim_file_sha256, "selectionClaimDeclaredSha256": claim["claimSha256"],
        "selectionReceiptFileSha256": receipt_file_sha256, "selectionReceiptDeclaredSha256": receipt["selectionReceiptSha256"],
        "selectionObservationFileSha256": observation_file_sha256, "selectionObservationDeclaredSha256": observation["selectionObservationSha256"],
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise SuccessorV2SelectionError("successor V2 selection lock does not bind JSON evidence")
    evaluations = _lock_evaluations(contract, observation)
    if document.get("candidateEvaluations") != evaluations or document.get("decision") != _decision(evaluations):
        raise SuccessorV2SelectionError("successor V2 selection lock does not recompute frozen gates/objective")
    if document.get("selectionLockSha256") != _document_digest(document, "selectionLockSha256"):
        raise SuccessorV2SelectionError("successor V2 selection lock digest does not match")
    return document


def load_validated_successor_v2_selection_lock(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    contract_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    contract, contract_file_sha256 = load_validated_successor_v2_selection_contract(contract_path, repository_root=repository_root)
    _validate_contract_chain(
        parent_holdout_path, parent_selection_contract_path, plan_path, envelope_path, contract,
        contract_file_sha256=contract_file_sha256, repository_root=repository_root,
    )
    claim, claim_file_sha256 = _load_claim(parent_selection_contract_path, contract, contract_file_sha256=contract_file_sha256, repository_root=repository_root)
    receipt, receipt_file_sha256, observation, observation_file_sha256 = _load_receipt_and_observation(
        parent_holdout_path, contract, contract_file_sha256=contract_file_sha256, claim=claim, claim_file_sha256=claim_file_sha256,
        repository_root=repository_root,
    )
    document, file_sha256 = _read_json(
        successor_v2_selection_path(contract, artifact="lock", repository_root=repository_root),
        description="successor V2 selection lock", repository_root=repository_root,
    )
    return _validate_lock(document, contract=contract, contract_file_sha256=contract_file_sha256, claim=claim,
                          claim_file_sha256=claim_file_sha256, receipt=receipt, receipt_file_sha256=receipt_file_sha256,
                          observation=observation, observation_file_sha256=observation_file_sha256), file_sha256
