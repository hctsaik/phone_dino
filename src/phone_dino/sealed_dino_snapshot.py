"""Create and activate cache-free, hash-pinned local DINO snapshots.

This module is deliberately standalone.  It does not change the production
loader or any frozen research evaluator; a caller must opt in by materialising
and then activating a snapshot explicitly.  A snapshot is an immutable-slot
directory containing a cache-free copy of a local DINO repository, a copied
weights file, and a self-authenticating manifest.

The source repository digest intentionally follows :func:`security.digest_directory`:
``.git`` and Python bytecode caches are excluded.  They are never removed from
the source tree.  They are, however, forbidden in a completed snapshot.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import importlib.machinery
import json
import os
import stat
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, Callable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SEALED_DINO_SNAPSHOT_SCHEMA = "phone-dino.sealed-local-dino-snapshot/1.0"
SEALED_DINO_SNAPSHOT_PURPOSE = "SEALED_LOCAL_DINOV2_SNAPSHOT"
SNAPSHOT_MANIFEST_NAME = "sealed_dino_snapshot.json"
SNAPSHOT_REPOSITORY_DIRECTORY = "repository"
SNAPSHOT_WEIGHTS_FILENAME = "model_weights.pth"
SNAPSHOT_ENTRYPOINT = "dinov2_vits14"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_FIELD = "sealedDinoSnapshot"
SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_SCHEMA = "phone-dino.sealed-local-dino-feature-extractor-provenance/1.1"
SEALED_DINO_REPOSITORY_DIGEST_ALGORITHM = "SEALED_REPOSITORY_CONTENT_SHA256_EXCLUDING_GIT_AND_PYTHON_BYTECODE_V1"
SEALED_DINO_WEIGHTS_DIGEST_ALGORITHM = "REGULAR_FILE_BYTES_SHA256_V1"
SEALED_DINO_SNAPSHOT_GUARD_MODULE_DIGEST_ALGORITHM = "REGULAR_FILE_BYTES_SHA256_V1"

# ``sys.path``, ``sys.modules``, and ``sys.dont_write_bytecode`` are process
# globals.  Keep one activation in control of them until it has verified and
# removed the DINO imports it created.  This is deliberately reentrant so a
# caller can compose helpers in the same thread without an artificial deadlock.
_DINO_ACTIVATION_LOCK = RLock()

_MANIFEST_FIELDS = {
    "schemaVersion",
    "purpose",
    "repositoryRelativePath",
    "weightsRelativePath",
    "expectedRepositorySha256",
    "expectedWeightsSha256",
    "snapshotRepositorySha256",
    "snapshotWeightsSha256",
    "repositoryFiles",
    "snapshotManifestSha256",
}
_MANIFEST_FILE_FIELDS = {"relativePath", "sha256", "byteCount"}
_CACHE_DIRECTORY_NAMES = frozenset({".git", "__pycache__"})
_TORCH_HUB_DYNAMIC_NAMESPACE_DIRECTORIES = {
    "dinov2.hub.cell_dino": PurePosixPath("dinov2/hub/cell_dino"),
    "dinov2.hub.xray_dino": PurePosixPath("dinov2/hub/xray_dino"),
}


class SealedDinoSnapshotError(ValueError):
    """Raised when a source, snapshot, or DINO activation is not safe to use."""


@dataclass(frozen=True)
class SealedDinoSnapshot:
    """A fully validated local snapshot suitable for an explicit activation."""

    root: Path
    repository: Path
    weights: Path
    manifest_path: Path
    repository_sha256: str
    weights_sha256: str
    manifest_sha256: str

    def activate(
        self,
        *,
        expected_manifest_sha256: str,
        repository_root: Path = REPOSITORY_ROOT,
    ) -> "SealedDinoSnapshotActivation":
        """Return a context which validates DINO imports around model use."""

        return activate_sealed_dino_snapshot(
            self,
            expected_manifest_sha256=expected_manifest_sha256,
            repository_root=repository_root,
        )


@dataclass(frozen=True)
class _RepositoryFile:
    relative_path: PurePosixPath
    source_path: Path
    sha256: str
    byte_count: int
    signature: tuple[int, int, int, int, int]


def canonical_json_sha256(document: Any) -> str:
    """Return a stable SHA-256 for a JSON-compatible value."""

    _validate_json_value(document, name="canonical JSON value")
    payload = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _prefixed_digest(payload)


def sha256_file(path: Path) -> str:
    """Hash one regular, non-link file without following a reparse point."""

    digest, _byte_count, _signature = _digest_regular_file(path, description="file")
    return digest


def _snapshot_guard_module_sha256() -> str:
    """Digest this guard's loaded source through the regular-file verifier."""

    return sha256_file(Path(__file__))


def sealed_repository_sha256(repository: Path) -> str:
    """Hash included repository content using the existing DINO source policy.

    ``.git``, ``__pycache__``, and ``*.pyc`` entries are excluded rather than
    deleted.  Every included directory and file must be a non-link regular
    filesystem object.
    """

    _records, digest = _scan_repository(repository, description="model repository", exclude_cache_entries=True)
    return digest


def materialize_sealed_dino_snapshot(
    source_repository: Path,
    source_weights: Path,
    output_directory: Path,
    *,
    expected_repository_sha256: str,
    expected_weights_sha256: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> SealedDinoSnapshot:
    """Create one new external snapshot from pinned local repository/weights.

    The output directory is a new-only immutable slot.  It is intentionally not
    cleaned on a failure: an interrupted or invalid slot has no manifest and
    cannot be activated, while preserving it avoids silently reusing a path.
    """

    expected_repository_sha256 = _require_sha256(expected_repository_sha256, name="expected repository SHA-256")
    expected_weights_sha256 = _require_sha256(expected_weights_sha256, name="expected weights SHA-256")
    worktree = _require_worktree_root(repository_root)
    # Source bytes are intentionally allowed to live in the checked-out
    # worktree (the normal local engineering layout does exactly that).  They
    # remain no-follow inputs and are re-hashed before/after copy; only the
    # completed snapshot slot itself must stay external and new-only.
    source_repository = _require_safe_source_directory(
        source_repository,
        description="source model repository",
    )
    source_weights = _require_safe_source_file(
        source_weights,
        description="source model weights",
    )
    output_directory = _require_new_external_output_directory(
        output_directory,
        description="sealed snapshot output",
        repository_root=worktree,
    )
    _reject_path_overlap(source_repository, output_directory, description="source model repository and sealed snapshot output")
    _reject_path_overlap(source_weights, output_directory, description="source model weights and sealed snapshot output")

    source_files, source_repository_digest = _scan_repository(
        source_repository,
        description="source model repository",
        exclude_cache_entries=True,
    )
    _require_dino_source_layout(source_repository, source_files)
    if source_repository_digest != expected_repository_sha256:
        raise SealedDinoSnapshotError("source model repository digest does not match its expected SHA-256")
    source_weights_digest, _weight_size, source_weights_signature = _digest_regular_file(
        source_weights,
        description="source model weights",
    )
    if source_weights_digest != expected_weights_sha256:
        raise SealedDinoSnapshotError("source model weights digest does not match its expected SHA-256")

    _create_new_directory(output_directory, description="sealed snapshot output")
    snapshot_repository = output_directory / SNAPSHOT_REPOSITORY_DIRECTORY
    _create_new_directory(snapshot_repository, description="sealed snapshot repository")
    for record in source_files:
        destination = snapshot_repository.joinpath(*record.relative_path.parts)
        _copy_regular_file_exclusive(
            record.source_path,
            destination,
            expected_digest=record.sha256,
            expected_byte_count=record.byte_count,
            expected_signature=record.signature,
            description=f"source repository file {record.relative_path.as_posix()}",
        )

    snapshot_weights = output_directory / SNAPSHOT_WEIGHTS_FILENAME
    _copy_regular_file_exclusive(
        source_weights,
        snapshot_weights,
        expected_digest=source_weights_digest,
        expected_byte_count=_weight_size,
        expected_signature=source_weights_signature,
        description="source model weights",
    )

    # Re-scan the source after copy so a newly added included file cannot be
    # omitted from a snapshot whose source claimed to remain pinned.
    current_source_files, current_source_digest = _scan_repository(
        source_repository,
        description="source model repository",
        exclude_cache_entries=True,
    )
    if current_source_digest != source_repository_digest or _records_identity(current_source_files) != _records_identity(source_files):
        raise SealedDinoSnapshotError("source model repository changed while the sealed snapshot was materialized")
    current_weights_digest, _current_weight_size, current_weights_signature = _digest_regular_file(
        source_weights,
        description="source model weights",
    )
    if current_weights_digest != source_weights_digest or current_weights_signature != source_weights_signature:
        raise SealedDinoSnapshotError("source model weights changed while the sealed snapshot was materialized")

    snapshot_files, snapshot_repository_digest = _scan_repository(
        snapshot_repository,
        description="sealed snapshot repository",
        exclude_cache_entries=False,
    )
    if snapshot_repository_digest != expected_repository_sha256 or _records_identity(snapshot_files) != _records_identity(source_files):
        raise SealedDinoSnapshotError("sealed snapshot repository digest does not match the pinned source")
    snapshot_weights_digest, _snapshot_weight_size, _snapshot_weight_signature = _digest_regular_file(
        snapshot_weights,
        description="sealed snapshot weights",
    )
    if snapshot_weights_digest != expected_weights_sha256:
        raise SealedDinoSnapshotError("sealed snapshot weights digest does not match the pinned source")

    manifest = {
        "schemaVersion": SEALED_DINO_SNAPSHOT_SCHEMA,
        "purpose": SEALED_DINO_SNAPSHOT_PURPOSE,
        "repositoryRelativePath": SNAPSHOT_REPOSITORY_DIRECTORY,
        "weightsRelativePath": SNAPSHOT_WEIGHTS_FILENAME,
        "expectedRepositorySha256": expected_repository_sha256,
        "expectedWeightsSha256": expected_weights_sha256,
        "snapshotRepositorySha256": snapshot_repository_digest,
        "snapshotWeightsSha256": snapshot_weights_digest,
        "repositoryFiles": _manifest_records(snapshot_files),
    }
    manifest["snapshotManifestSha256"] = _document_digest(manifest, field="snapshotManifestSha256")
    manifest_path = output_directory / SNAPSHOT_MANIFEST_NAME
    _write_json_exclusive(manifest_path, manifest, description="sealed snapshot manifest")
    return load_sealed_dino_snapshot(
        output_directory,
        expected_manifest_sha256=manifest["snapshotManifestSha256"],
        repository_root=worktree,
    )


def load_sealed_dino_snapshot(
    snapshot_directory: Path,
    *,
    expected_manifest_sha256: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> SealedDinoSnapshot:
    """Validate a complete snapshot against an independently supplied pin."""

    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        name="expected sealed snapshot manifest SHA-256",
    )
    worktree = _require_worktree_root(repository_root)
    root = _require_external_directory(
        snapshot_directory,
        description="sealed snapshot directory",
        repository_root=worktree,
    )
    _require_exact_snapshot_root(root)
    manifest_path = root / SNAPSHOT_MANIFEST_NAME
    manifest = _read_json_regular(manifest_path, description="sealed snapshot manifest")
    _validate_manifest(manifest)
    declared_manifest_digest = _require_sha256(manifest["snapshotManifestSha256"], name="snapshot manifest self digest")
    if declared_manifest_digest != _document_digest(manifest, field="snapshotManifestSha256"):
        raise SealedDinoSnapshotError("sealed snapshot manifest self digest does not match")
    if not hmac.compare_digest(declared_manifest_digest, expected_manifest_sha256):
        raise SealedDinoSnapshotError("sealed snapshot manifest does not match the approved SHA-256 pin")

    repository = root / SNAPSHOT_REPOSITORY_DIRECTORY
    weights = root / SNAPSHOT_WEIGHTS_FILENAME
    if manifest["repositoryRelativePath"] != SNAPSHOT_REPOSITORY_DIRECTORY:
        raise SealedDinoSnapshotError("sealed snapshot manifest repository path is unsafe")
    if manifest["weightsRelativePath"] != SNAPSHOT_WEIGHTS_FILENAME:
        raise SealedDinoSnapshotError("sealed snapshot manifest weights path is unsafe")
    _require_directory(repository, description="sealed snapshot repository")
    _require_regular_file(weights, description="sealed snapshot weights")

    expected_files = _manifest_to_records(manifest["repositoryFiles"])
    actual_files, actual_repository_digest = _scan_repository(
        repository,
        description="sealed snapshot repository",
        exclude_cache_entries=False,
    )
    if _records_identity(actual_files) != _records_identity(expected_files):
        raise SealedDinoSnapshotError("sealed snapshot repository files do not match its manifest")
    expected_repository = _require_sha256(manifest["expectedRepositorySha256"], name="expected repository SHA-256")
    snapshot_repository = _require_sha256(manifest["snapshotRepositorySha256"], name="snapshot repository SHA-256")
    if actual_repository_digest != expected_repository or actual_repository_digest != snapshot_repository:
        raise SealedDinoSnapshotError("sealed snapshot repository digest does not match its pins")
    actual_weights_digest, _weight_size, _weight_signature = _digest_regular_file(weights, description="sealed snapshot weights")
    expected_weights = _require_sha256(manifest["expectedWeightsSha256"], name="expected weights SHA-256")
    snapshot_weights = _require_sha256(manifest["snapshotWeightsSha256"], name="snapshot weights SHA-256")
    if actual_weights_digest != expected_weights or actual_weights_digest != snapshot_weights:
        raise SealedDinoSnapshotError("sealed snapshot weights digest does not match its pins")
    _require_dino_source_layout(repository, actual_files)
    return SealedDinoSnapshot(
        root=root,
        repository=repository,
        weights=weights,
        manifest_path=manifest_path,
        repository_sha256=actual_repository_digest,
        weights_sha256=actual_weights_digest,
        manifest_sha256=declared_manifest_digest,
    )


def activate_sealed_dino_snapshot(
    snapshot: SealedDinoSnapshot | Path,
    *,
    expected_manifest_sha256: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> "SealedDinoSnapshotActivation":
    """Return a fail-closed activation context for local DINO imports."""

    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        name="expected sealed snapshot manifest SHA-256",
    )
    if isinstance(snapshot, SealedDinoSnapshot):
        root = snapshot.root
        object_manifest_sha256 = _require_sha256(
            snapshot.manifest_sha256,
            name="sealed snapshot object manifest SHA-256",
        )
        if not hmac.compare_digest(object_manifest_sha256, expected_manifest_sha256):
            raise SealedDinoSnapshotError("sealed snapshot object does not match the supplied manifest SHA-256 pin")
    else:
        root = snapshot
    if not isinstance(root, Path):
        raise SealedDinoSnapshotError("sealed snapshot must be a SealedDinoSnapshot or Path")
    return SealedDinoSnapshotActivation(
        root,
        expected_manifest_sha256=expected_manifest_sha256,
        repository_root=repository_root,
    )


class SealedDinoSnapshotActivation(AbstractContextManager["SealedDinoSnapshotActivation"]):
    """Temporarily prioritise one validated DINO source tree on ``sys.path``.

    The context refuses every preloaded ``dinov2`` module rather than trusting
    mutable module metadata.  It verifies ordinary DINO source origins and
    digests plus the two pinned PEP 420 torch-hub namespaces after each hub
    load and on exit, then removes only the verified modules it loaded so a
    later activation starts from a clean import state.
    ``sys.dont_write_bytecode`` is set for the active period and centralized
    bytecode caches are rejected before any snapshot import.
    """

    def __init__(
        self,
        snapshot_directory: Path,
        *,
        expected_manifest_sha256: str,
        repository_root: Path,
    ) -> None:
        self._snapshot_directory = snapshot_directory
        self._expected_manifest_sha256 = expected_manifest_sha256
        self._repository_root = repository_root
        self._snapshot: SealedDinoSnapshot | None = None
        self._previous_dont_write_bytecode: bool | None = None
        self._inserted_path: str | None = None
        self._active = False
        self._lock_acquired = False
        self._snapshot_guard_module_sha256: str | None = None

    @property
    def snapshot(self) -> SealedDinoSnapshot:
        if self._snapshot is None:
            raise SealedDinoSnapshotError("sealed DINO snapshot activation is not active")
        return self._snapshot

    @property
    def snapshot_guard_module_sha256(self) -> str:
        """Return the guard source digest captured for this activation."""

        if self._snapshot_guard_module_sha256 is None:
            raise SealedDinoSnapshotError("sealed DINO snapshot activation is not active")
        return self._snapshot_guard_module_sha256

    def __enter__(self) -> "SealedDinoSnapshotActivation":
        if self._active or self._lock_acquired:
            raise SealedDinoSnapshotError("sealed DINO snapshot activation is already active")
        _DINO_ACTIVATION_LOCK.acquire()
        self._lock_acquired = True
        try:
            _require_no_external_pycache_prefix()
            _reject_preloaded_dino_modules()
            self._snapshot_guard_module_sha256 = _snapshot_guard_module_sha256()
            snapshot = load_sealed_dino_snapshot(
                self._snapshot_directory,
                expected_manifest_sha256=self._expected_manifest_sha256,
                repository_root=self._repository_root,
            )
            self._snapshot = snapshot
            self._previous_dont_write_bytecode = sys.dont_write_bytecode
            self._inserted_path = str(snapshot.repository)
            sys.dont_write_bytecode = True
            sys.path.insert(0, self._inserted_path)
            self._active = True
            self.verify_integrity()
        except BaseException:
            self._restore_process_state()
            raise
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None:
        cleanup_error: BaseException | None = None
        try:
            if self._active:
                try:
                    self.verify_integrity()
                    _remove_verified_loaded_dino_modules(self.snapshot)
                except BaseException as error:
                    cleanup_error = error
        finally:
            self._restore_process_state()
        # Never hide the caller's operational exception behind a cleanup
        # failure.  A failed cleanup deliberately leaves DINO modules in the
        # import cache, so the next activation refuses that preloaded state.
        if exc_type is not None and cleanup_error is not None:
            if isinstance(exc_value, BaseException):
                raise BaseExceptionGroup(
                    "sealed DINO activation body and integrity cleanup both failed",
                    [exc_value, cleanup_error],
                )
            raise cleanup_error
        if exc_type is not None:
            return None
        if cleanup_error is not None:
            raise cleanup_error
        return None

    def verify_integrity(self) -> None:
        """Revalidate files and every loaded ``dinov2*`` module origin."""

        if not self._active or self._snapshot is None:
            raise SealedDinoSnapshotError("sealed DINO snapshot activation is not active")
        _require_no_external_pycache_prefix()
        if not hmac.compare_digest(self.snapshot_guard_module_sha256, _snapshot_guard_module_sha256()):
            raise SealedDinoSnapshotError("sealed DINO snapshot guard module changed during activation")
        current = load_sealed_dino_snapshot(
            self._snapshot.root,
            expected_manifest_sha256=self._expected_manifest_sha256,
            repository_root=self._repository_root,
        )
        if current != self._snapshot:
            raise SealedDinoSnapshotError("sealed DINO snapshot changed during activation")
        _verify_loaded_dino_modules(current)

    def load_torch_hub_model(
        self,
        *,
        torch_module: Any | None = None,
        entrypoint: str = SNAPSHOT_ENTRYPOINT,
    ) -> object:
        """Load a local DINO hub entrypoint from this exact snapshot path."""

        if not isinstance(entrypoint, str) or entrypoint != SNAPSHOT_ENTRYPOINT:
            raise SealedDinoSnapshotError("sealed DINO snapshot only permits the pinned dinov2_vits14 entrypoint")
        self.verify_integrity()
        torch = torch_module if torch_module is not None else importlib.import_module("torch")
        hub = getattr(torch, "hub", None)
        loader = getattr(hub, "load", None)
        if not callable(loader):
            raise SealedDinoSnapshotError("torch.hub.load is not available")
        try:
            model = loader(str(self.snapshot.repository), entrypoint, source="local", pretrained=False)
        except Exception as error:
            raise SealedDinoSnapshotError("unable to load DINO from the sealed local snapshot") from error
        self.verify_integrity()
        return model

    def _restore_process_state(self) -> None:
        inserted_path = self._inserted_path
        if inserted_path is not None:
            try:
                sys.path.remove(inserted_path)
            except ValueError:
                pass
        if self._previous_dont_write_bytecode is not None:
            sys.dont_write_bytecode = self._previous_dont_write_bytecode
        self._active = False
        self._inserted_path = None
        self._previous_dont_write_bytecode = None
        self._snapshot = None
        self._snapshot_guard_module_sha256 = None
        if self._lock_acquired:
            self._lock_acquired = False
            _DINO_ACTIVATION_LOCK.release()


def sealed_snapshot_identity_factory(
    base_identity_factory: Callable[..., dict[str, Any]],
    activation: SealedDinoSnapshotActivation,
) -> Callable[..., dict[str, Any]]:
    """Wrap feature provenance with the active sealed-snapshot identity.

    The returned factory is intended for an evaluator's existing
    ``identity_factory`` seam.  It only accepts the active snapshot's exact
    repository and weights paths, revalidates the snapshot before and after
    the base identity is assembled, and adds the immutable manifest/schema
    binding.  This keeps the caller's model loader and its evidence record on
    the same sealed bytes without changing frozen evaluator modules.
    """

    if not callable(base_identity_factory):
        raise SealedDinoSnapshotError("base feature extractor identity factory must be callable")
    if not isinstance(activation, SealedDinoSnapshotActivation):
        raise SealedDinoSnapshotError("sealed feature extractor identity factory requires an active snapshot activation")

    def bound_identity_factory(*, model_repo: Path, model_weights: Path, device: str) -> dict[str, Any]:
        active_snapshot = activation.snapshot
        if model_repo != active_snapshot.repository or model_weights != active_snapshot.weights:
            raise SealedDinoSnapshotError("feature extractor must use the active sealed snapshot repository and weights")
        activation.verify_integrity()
        identity = base_identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
        if not isinstance(identity, dict):
            raise SealedDinoSnapshotError("base feature extractor identity factory must return an object")
        if SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_FIELD in identity:
            raise SealedDinoSnapshotError("base feature extractor identity reserves the sealed snapshot provenance field")
        activation.verify_integrity()
        bound = dict(identity)
        bound[SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_FIELD] = {
            "schemaVersion": SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_SCHEMA,
            "snapshotSchemaVersion": SEALED_DINO_SNAPSHOT_SCHEMA,
            "snapshotManifestSha256": active_snapshot.manifest_sha256,
            "snapshotRepositorySha256": active_snapshot.repository_sha256,
            "repositoryDigestAlgorithm": SEALED_DINO_REPOSITORY_DIGEST_ALGORITHM,
            "snapshotWeightsSha256": active_snapshot.weights_sha256,
            "weightsDigestAlgorithm": SEALED_DINO_WEIGHTS_DIGEST_ALGORITHM,
            "snapshotGuardModuleSha256": activation.snapshot_guard_module_sha256,
            "snapshotGuardModuleDigestAlgorithm": SEALED_DINO_SNAPSHOT_GUARD_MODULE_DIGEST_ALGORITHM,
        }
        return bound

    return bound_identity_factory


def load_sealed_dinov2_vits14(
    snapshot: SealedDinoSnapshot | Path,
    *,
    torch_module: Any | None = None,
    expected_manifest_sha256: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> object:
    """Load DINOv2 ViT-S/14 and its snapshot weights through the sealed context."""

    with activate_sealed_dino_snapshot(
        snapshot,
        expected_manifest_sha256=expected_manifest_sha256,
        repository_root=repository_root,
    ) as activation:
        torch = torch_module if torch_module is not None else importlib.import_module("torch")
        model = activation.load_torch_hub_model(torch_module=torch)
        loader = getattr(torch, "load", None)
        if not callable(loader):
            raise SealedDinoSnapshotError("torch.load is not available")
        try:
            checkpoint = loader(str(activation.snapshot.weights), map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            load_state_dict = getattr(model, "load_state_dict", None)
            evaluate = getattr(model, "eval", None)
            if not callable(load_state_dict) or not callable(evaluate):
                raise TypeError("DINO hub model does not implement the expected torch model interface")
            load_state_dict(state_dict, strict=True)
            evaluate()
        except SealedDinoSnapshotError:
            raise
        except Exception as error:
            raise SealedDinoSnapshotError("unable to load sealed DINO snapshot weights") from error
        activation.verify_integrity()
        return model


def _require_worktree_root(path: Path) -> Path:
    if not isinstance(path, Path):
        raise SealedDinoSnapshotError("repository root must be a Path")
    _reject_links_on_existing_path(path, description="repository root")
    if not path.is_dir() or _is_link_or_reparse_point(path):
        raise SealedDinoSnapshotError("repository root must be a regular directory")
    return path.resolve()


def _require_external_directory(path: Path, *, description: str, repository_root: Path) -> Path:
    if not isinstance(path, Path):
        raise SealedDinoSnapshotError(f"{description} must be a Path")
    _reject_links_on_existing_path(path, description=description)
    if not path.is_dir() or _is_link_or_reparse_point(path):
        raise SealedDinoSnapshotError(f"{description} must be a regular non-link directory")
    resolved = path.resolve()
    _reject_worktree_overlap(resolved, repository_root, description=description)
    return resolved


def _require_external_file(path: Path, *, description: str, repository_root: Path) -> Path:
    if not isinstance(path, Path):
        raise SealedDinoSnapshotError(f"{description} must be a Path")
    _reject_links_on_existing_path(path, description=description)
    if not path.is_file() or _is_link_or_reparse_point(path):
        raise SealedDinoSnapshotError(f"{description} must be a regular non-link file")
    resolved = path.resolve()
    _reject_worktree_overlap(resolved, repository_root, description=description)
    return resolved


def _require_safe_source_directory(path: Path, *, description: str) -> Path:
    """Return a non-link source directory regardless of worktree location."""

    if not isinstance(path, Path):
        raise SealedDinoSnapshotError(f"{description} must be a Path")
    _require_directory(path, description=description)
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as error:
        raise SealedDinoSnapshotError(f"unable to resolve {description}") from error
    _require_directory(resolved, description=description)
    return resolved


def _require_safe_source_file(path: Path, *, description: str) -> Path:
    """Return a regular, no-follow source file regardless of worktree location."""

    if not isinstance(path, Path):
        raise SealedDinoSnapshotError(f"{description} must be a Path")
    _require_regular_file(path, description=description)
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as error:
        raise SealedDinoSnapshotError(f"unable to resolve {description}") from error
    _require_regular_file(resolved, description=description)
    return resolved


def _require_new_external_output_directory(path: Path, *, description: str, repository_root: Path) -> Path:
    if not isinstance(path, Path):
        raise SealedDinoSnapshotError(f"{description} must be a Path")
    if path.exists() or path.is_symlink():
        raise SealedDinoSnapshotError(f"{description} already exists; the immutable output slot is consumed")
    _reject_links_on_existing_path(path.parent, description=description)
    resolved = path.resolve(strict=False)
    _reject_worktree_overlap(resolved, repository_root, description=description)
    return resolved


def _reject_worktree_overlap(candidate: Path, repository_root: Path, *, description: str) -> None:
    if _paths_overlap(candidate, repository_root):
        raise SealedDinoSnapshotError(f"{description} must stay outside the Git working tree")


def _reject_path_overlap(left: Path, right: Path, *, description: str) -> None:
    if _paths_overlap(left, right):
        raise SealedDinoSnapshotError(f"{description} must not contain one another")


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return _stat_is_link_or_reparse_point(status)


def _stat_is_link_or_reparse_point(status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    return bool(attributes & reparse_flag)


def _reject_links_on_existing_path(path: Path, *, description: str) -> None:
    current = path.absolute()
    while True:
        if current.exists() or current.is_symlink():
            if _is_link_or_reparse_point(current):
                raise SealedDinoSnapshotError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _require_directory(path: Path, *, description: str) -> None:
    _reject_links_on_existing_path(path, description=description)
    if not path.is_dir() or _is_link_or_reparse_point(path):
        raise SealedDinoSnapshotError(f"{description} must be a regular non-link directory")


def _require_regular_file(path: Path, *, description: str) -> tuple[int, int, int, int, int]:
    _reject_links_on_existing_path(path, description=description)
    try:
        status = path.stat(follow_symlinks=False)
    except OSError as error:
        raise SealedDinoSnapshotError(f"unable to stat {description}") from error
    if _stat_is_link_or_reparse_point(status) or not stat.S_ISREG(status.st_mode):
        raise SealedDinoSnapshotError(f"{description} must be a regular non-link file")
    return _stat_signature(status)


def _stat_signature(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (status.st_dev, status.st_ino, status.st_mode, status.st_size, status.st_mtime_ns)


def _scan_repository(
    repository: Path,
    *,
    description: str,
    exclude_cache_entries: bool,
) -> tuple[tuple[_RepositoryFile, ...], str]:
    _require_directory(repository, description=description)
    records: list[_RepositoryFile] = []

    def walk(directory: Path, relative_directory: PurePosixPath | None) -> None:
        try:
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
        except OSError as error:
            raise SealedDinoSnapshotError(f"unable to scan {description}") from error
        for entry in entries:
            relative = PurePosixPath(entry.name) if relative_directory is None else relative_directory / entry.name
            if _is_excluded_cache_path(relative):
                if exclude_cache_entries:
                    continue
                raise SealedDinoSnapshotError(f"{description} contains a forbidden cache or VCS entry: {relative.as_posix()}")
            path = Path(entry.path)
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise SealedDinoSnapshotError(f"unable to stat {description} entry {relative.as_posix()}") from error
            if _stat_is_link_or_reparse_point(status):
                raise SealedDinoSnapshotError(f"{description} contains a symbolic link or reparse point: {relative.as_posix()}")
            if stat.S_ISDIR(status.st_mode):
                walk(path, relative)
                continue
            if not stat.S_ISREG(status.st_mode):
                raise SealedDinoSnapshotError(f"{description} contains a non-regular entry: {relative.as_posix()}")
            # On Windows ``DirEntry.stat(follow_symlinks=False)`` may expose
            # zero device/inode fields even for an ordinary file.  Re-stat the
            # exact path with the same no-follow policy before using it as the
            # immutable-copy signature.
            signature = _require_regular_file(path, description=f"{description} file {relative.as_posix()}")
            digest, byte_count, signature = _digest_regular_file(
                path,
                description=f"{description} file {relative.as_posix()}",
                expected_signature=signature,
            )
            records.append(_RepositoryFile(relative, path, digest, byte_count, signature))

    walk(repository, None)
    if not records:
        raise SealedDinoSnapshotError(f"{description} has no included source files")
    records.sort(key=lambda record: record.relative_path.as_posix())
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        _update_digest_from_regular_file(
            digest,
            record.source_path,
            description=f"{description} file {record.relative_path.as_posix()}",
            expected_signature=record.signature,
        )
        digest.update(b"\0")
    return tuple(records), f"sha256:{digest.hexdigest()}"


def _is_excluded_cache_path(relative: PurePosixPath) -> bool:
    return any(part in _CACHE_DIRECTORY_NAMES for part in relative.parts) or relative.name.lower().endswith(".pyc")


def _require_dino_source_layout(repository: Path, records: tuple[_RepositoryFile, ...]) -> None:
    paths = {record.relative_path.as_posix() for record in records}
    if "hubconf.py" not in paths:
        raise SealedDinoSnapshotError("model repository must include a regular hubconf.py")
    if not any(path == "dinov2/__init__.py" for path in paths):
        raise SealedDinoSnapshotError("model repository must include a regular dinov2 package")
    _require_regular_file(repository / "hubconf.py", description="model repository hubconf.py")
    _require_regular_file(repository / "dinov2" / "__init__.py", description="model repository dinov2 package")


def _digest_regular_file(
    path: Path,
    *,
    description: str,
    expected_signature: tuple[int, int, int, int, int] | None = None,
) -> tuple[str, int, tuple[int, int, int, int, int]]:
    digest = hashlib.sha256()
    byte_count, signature = _update_digest_from_regular_file(
        digest,
        path,
        description=description,
        expected_signature=expected_signature,
    )
    return f"sha256:{digest.hexdigest()}", byte_count, signature


def _update_digest_from_regular_file(
    digest: "hashlib._Hash",
    path: Path,
    *,
    description: str,
    expected_signature: tuple[int, int, int, int, int] | None = None,
) -> tuple[int, tuple[int, int, int, int, int]]:
    before = _require_regular_file(path, description=description)
    if expected_signature is not None and before != expected_signature:
        raise SealedDinoSnapshotError(f"{description} changed before it was read")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SealedDinoSnapshotError(f"unable to open {description}") from error
    byte_count = 0
    try:
        descriptor_status = os.fstat(descriptor)
        if _stat_is_link_or_reparse_point(descriptor_status) or not stat.S_ISREG(descriptor_status.st_mode):
            raise SealedDinoSnapshotError(f"{description} is not a regular non-link file")
        if _stat_signature(descriptor_status) != before:
            raise SealedDinoSnapshotError(f"{description} changed while it was opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after_descriptor = _stat_signature(os.fstat(descriptor))
    except OSError as error:
        raise SealedDinoSnapshotError(f"unable to read {description}") from error
    finally:
        os.close(descriptor)
    after = _require_regular_file(path, description=description)
    if after != before or after_descriptor != before or byte_count != before[3]:
        raise SealedDinoSnapshotError(f"{description} changed while it was read")
    return byte_count, before


def _copy_regular_file_exclusive(
    source: Path,
    destination: Path,
    *,
    expected_digest: str,
    expected_byte_count: int,
    expected_signature: tuple[int, int, int, int, int],
    description: str,
) -> None:
    _require_sha256(expected_digest, name=f"{description} expected SHA-256")
    _ensure_new_output_parent(destination.parent, description=f"{description} output parent")
    if destination.exists() or destination.is_symlink():
        raise SealedDinoSnapshotError(f"{description} output already exists")
    source_before = _require_regular_file(source, description=description)
    if source_before != expected_signature:
        raise SealedDinoSnapshotError(f"{description} changed before it was copied")
    source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as error:
        raise SealedDinoSnapshotError(f"unable to open {description} for copy") from error
    try:
        source_descriptor_status = os.fstat(source_descriptor)
        if _stat_signature(source_descriptor_status) != source_before or not stat.S_ISREG(source_descriptor_status.st_mode):
            raise SealedDinoSnapshotError(f"{description} changed while it was opened for copy")
        try:
            destination_descriptor = os.open(destination, destination_flags, 0o600)
        except FileExistsError as error:
            raise SealedDinoSnapshotError(f"{description} output already exists") from error
        except OSError as error:
            raise SealedDinoSnapshotError(f"unable to create exclusive output for {description}") from error
        digest = hashlib.sha256()
        byte_count = 0
        try:
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
                _write_all(destination_descriptor, chunk, description=description)
            os.fsync(destination_descriptor)
        except OSError as error:
            raise SealedDinoSnapshotError(f"unable to copy {description}") from error
        finally:
            os.close(destination_descriptor)
        copied_digest = f"sha256:{digest.hexdigest()}"
        source_after_descriptor = _stat_signature(os.fstat(source_descriptor))
    finally:
        os.close(source_descriptor)
    source_after = _require_regular_file(source, description=description)
    if (
        source_after != source_before
        or source_after_descriptor != source_before
        or byte_count != expected_byte_count
        or copied_digest != expected_digest
    ):
        raise SealedDinoSnapshotError(f"{description} changed while it was copied")
    copied_digest_check, copied_size, _copied_signature = _digest_regular_file(destination, description=f"copied {description}")
    if copied_digest_check != expected_digest or copied_size != expected_byte_count:
        raise SealedDinoSnapshotError(f"copied {description} digest does not match its source pin")


def _write_all(descriptor: int, payload: bytes, *, description: str) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError(f"short write while writing {description}")
        offset += written


def _create_new_directory(path: Path, *, description: str) -> None:
    _ensure_new_output_parent(path.parent, description=f"{description} parent")
    if path.exists() or path.is_symlink():
        raise SealedDinoSnapshotError(f"{description} already exists")
    try:
        os.mkdir(path)
    except FileExistsError as error:
        raise SealedDinoSnapshotError(f"{description} already exists") from error
    except OSError as error:
        raise SealedDinoSnapshotError(f"unable to create {description}") from error
    _require_directory(path, description=description)


def _ensure_new_output_parent(path: Path, *, description: str) -> None:
    missing: list[Path] = []
    current = path.absolute()
    while not current.exists() and not current.is_symlink():
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise SealedDinoSnapshotError(f"unable to find existing parent for {description}")
        current = parent
    _reject_links_on_existing_path(current, description=description)
    if not current.is_dir() or _is_link_or_reparse_point(current):
        raise SealedDinoSnapshotError(f"{description} has no regular directory parent")
    for directory in reversed(missing):
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass
        except OSError as error:
            raise SealedDinoSnapshotError(f"unable to create {description}") from error
        _require_directory(directory, description=description)
    _reject_links_on_existing_path(path, description=description)
    if not path.is_dir() or _is_link_or_reparse_point(path):
        raise SealedDinoSnapshotError(f"{description} must be a regular directory")


def _manifest_records(records: tuple[_RepositoryFile, ...]) -> list[dict[str, Any]]:
    return [
        {
            "relativePath": record.relative_path.as_posix(),
            "sha256": record.sha256,
            "byteCount": record.byte_count,
        }
        for record in records
    ]


def _records_identity(records: tuple[_RepositoryFile, ...]) -> tuple[tuple[str, str, int], ...]:
    return tuple((record.relative_path.as_posix(), record.sha256, record.byte_count) for record in records)


def _write_json_exclusive(path: Path, document: dict[str, Any], *, description: str) -> None:
    _validate_json_value(document, name=description)
    payload = (json.dumps(document, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _ensure_new_output_parent(path.parent, description=f"{description} parent")
    if path.exists() or path.is_symlink():
        raise SealedDinoSnapshotError(f"{description} already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise SealedDinoSnapshotError(f"{description} already exists") from error
    except OSError as error:
        raise SealedDinoSnapshotError(f"unable to create {description}") from error
    try:
        _write_all(descriptor, payload, description=description)
        os.fsync(descriptor)
    except OSError as error:
        raise SealedDinoSnapshotError(f"unable to write {description}") from error
    finally:
        os.close(descriptor)
    _digest_regular_file(path, description=description)


def _read_json_regular(path: Path, *, description: str) -> dict[str, Any]:
    raw = _read_regular_bytes(path, description=description, maximum_bytes=MAX_MANIFEST_BYTES)
    if not raw:
        raise SealedDinoSnapshotError(f"{description} is empty")

    return _parse_json_bytes(raw, description=description)


def _read_regular_bytes(path: Path, *, description: str, maximum_bytes: int) -> bytes:
    before = _require_regular_file(path, description=description)
    if before[3] > maximum_bytes:
        raise SealedDinoSnapshotError(f"{description} exceeds its maximum safe size")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SealedDinoSnapshotError(f"unable to open {description}") from error
    chunks: list[bytes] = []
    byte_count = 0
    try:
        descriptor_status = os.fstat(descriptor)
        if _stat_signature(descriptor_status) != before or not stat.S_ISREG(descriptor_status.st_mode):
            raise SealedDinoSnapshotError(f"{description} changed while it was opened")
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes - byte_count + 1))
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise SealedDinoSnapshotError(f"{description} exceeds its maximum safe size")
            chunks.append(chunk)
        descriptor_after = _stat_signature(os.fstat(descriptor))
    except OSError as error:
        raise SealedDinoSnapshotError(f"unable to read {description}") from error
    finally:
        os.close(descriptor)
    if _require_regular_file(path, description=description) != before or descriptor_after != before or byte_count != before[3]:
        raise SealedDinoSnapshotError(f"{description} changed while it was read")
    return b"".join(chunks)


def _parse_json_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SealedDinoSnapshotError(f"{description} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise SealedDinoSnapshotError(f"{description} contains a non-finite JSON value: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedDinoSnapshotError(f"unable to parse {description}") from error
    _validate_json_value(value, name=description)
    if not isinstance(value, dict):
        raise SealedDinoSnapshotError(f"{description} must be a JSON object")
    return value


def _validate_manifest(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise SealedDinoSnapshotError("sealed snapshot manifest must be an object")
    missing = _MANIFEST_FIELDS.difference(manifest)
    unknown = set(manifest).difference(_MANIFEST_FIELDS)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unsupported: {', '.join(sorted(unknown))}")
        raise SealedDinoSnapshotError(f"sealed snapshot manifest fields are unsafe ({'; '.join(detail)})")
    if manifest.get("schemaVersion") != SEALED_DINO_SNAPSHOT_SCHEMA:
        raise SealedDinoSnapshotError("sealed snapshot manifest schema is unsupported")
    if manifest.get("purpose") != SEALED_DINO_SNAPSHOT_PURPOSE:
        raise SealedDinoSnapshotError("sealed snapshot manifest purpose is unsafe")
    for name in (
        "repositoryRelativePath",
        "weightsRelativePath",
        "expectedRepositorySha256",
        "expectedWeightsSha256",
        "snapshotRepositorySha256",
        "snapshotWeightsSha256",
        "snapshotManifestSha256",
    ):
        if not isinstance(manifest.get(name), str) or not str(manifest[name]).strip():
            raise SealedDinoSnapshotError(f"sealed snapshot manifest {name} must be a non-empty string")
    _manifest_to_records(manifest.get("repositoryFiles"))


def _manifest_to_records(value: object) -> tuple[_RepositoryFile, ...]:
    if not isinstance(value, list) or not value:
        raise SealedDinoSnapshotError("sealed snapshot manifest repositoryFiles must be a non-empty list")
    records: list[_RepositoryFile] = []
    seen: set[str] = set()
    previous: str | None = None
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise SealedDinoSnapshotError(f"sealed snapshot manifest repositoryFiles[{index}] must be an object")
        if set(item) != _MANIFEST_FILE_FIELDS:
            raise SealedDinoSnapshotError(f"sealed snapshot manifest repositoryFiles[{index}] has unsupported fields")
        relative = _safe_relative_path(item.get("relativePath"), name=f"repositoryFiles[{index}].relativePath")
        if _is_excluded_cache_path(relative):
            raise SealedDinoSnapshotError("sealed snapshot manifest must not contain cache or VCS paths")
        serialized = relative.as_posix()
        if serialized in seen or (previous is not None and serialized <= previous):
            raise SealedDinoSnapshotError("sealed snapshot manifest repositoryFiles must be unique and sorted")
        seen.add(serialized)
        previous = serialized
        digest = _require_sha256(item.get("sha256"), name=f"repositoryFiles[{index}].sha256")
        byte_count = item.get("byteCount")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise SealedDinoSnapshotError(f"repositoryFiles[{index}].byteCount must be a non-negative integer")
        records.append(_RepositoryFile(relative, Path(), digest, byte_count, (0, 0, 0, 0, 0)))
    return tuple(records)


def _safe_relative_path(value: object, *, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SealedDinoSnapshotError(f"{name} must be a non-empty POSIX relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise SealedDinoSnapshotError(f"{name} must be a safe relative path")
    return relative


def _require_exact_snapshot_root(root: Path) -> None:
    _require_directory(root, description="sealed snapshot directory")
    try:
        with os.scandir(root) as scan:
            entries = {entry.name: entry for entry in scan}
    except OSError as error:
        raise SealedDinoSnapshotError("unable to scan sealed snapshot directory") from error
    expected = {SNAPSHOT_REPOSITORY_DIRECTORY, SNAPSHOT_WEIGHTS_FILENAME, SNAPSHOT_MANIFEST_NAME}
    if set(entries) != expected:
        raise SealedDinoSnapshotError("sealed snapshot directory has missing or unsupported entries")
    for name, entry in entries.items():
        try:
            status = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise SealedDinoSnapshotError(f"unable to stat sealed snapshot entry {name}") from error
        if _stat_is_link_or_reparse_point(status):
            raise SealedDinoSnapshotError(f"sealed snapshot contains a symbolic link or reparse point: {name}")
    if not stat.S_ISDIR(entries[SNAPSHOT_REPOSITORY_DIRECTORY].stat(follow_symlinks=False).st_mode):
        raise SealedDinoSnapshotError("sealed snapshot repository entry must be a directory")
    for name in (SNAPSHOT_WEIGHTS_FILENAME, SNAPSHOT_MANIFEST_NAME):
        if not stat.S_ISREG(entries[name].stat(follow_symlinks=False).st_mode):
            raise SealedDinoSnapshotError(f"sealed snapshot {name} must be a regular file")


def _document_digest(document: Mapping[str, Any], *, field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return canonical_json_sha256(unsigned)


def _prefixed_digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise SealedDinoSnapshotError(f"{name} must be a SHA-256 digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise SealedDinoSnapshotError(f"{name} must be a SHA-256 digest") from error
    return value


def _validate_json_value(value: Any, *, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise SealedDinoSnapshotError(f"{name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SealedDinoSnapshotError(f"{name} contains a non-string JSON key")
            _validate_json_value(item, name=f"{name}.{key}")
        return
    raise SealedDinoSnapshotError(f"{name} contains a value that is not JSON-compatible")


def _loaded_dino_module_items() -> tuple[tuple[str, Any], ...]:
    """Return a stable view of all process-global DINO import-cache entries."""

    return tuple(
        (name, module)
        for name, module in sys.modules.items()
        if name == "dinov2" or name.startswith("dinov2.")
    )


def _reject_preloaded_dino_modules() -> None:
    """Require a clean DINO import cache before any sealed activation."""

    names = sorted(name for name, _module in _loaded_dino_module_items())
    if names:
        raise SealedDinoSnapshotError(
            "sealed DINO activation refuses preloaded dinov2 modules: " + ", ".join(names)
        )


def _require_no_external_pycache_prefix() -> None:
    """Reject cache-prefix imports whose bytecode lies outside the snapshot."""

    if getattr(sys, "pycache_prefix", None) is not None or os.environ.get("PYTHONPYCACHEPREFIX"):
        raise SealedDinoSnapshotError(
            "sealed DINO activation requires no centralized Python bytecode cache prefix"
        )


def _remove_verified_loaded_dino_modules(snapshot: SealedDinoSnapshot) -> None:
    """Discard only DINO modules whose sealed origin was just revalidated.

    Entry rejects a non-empty DINO cache, so every module visible here was
    loaded during this context.  If a concurrent mutation replaces a module
    after verification, its object identity differs and it is deliberately
    left in place; the next activation then fails closed as preloaded state.
    """

    verified = _loaded_dino_module_items()
    _verify_loaded_dino_modules(snapshot)
    for name, module in verified:
        if sys.modules.get(name) is module:
            del sys.modules[name]


def _verify_loaded_dino_modules(snapshot: SealedDinoSnapshot) -> None:
    manifest = _read_json_regular(snapshot.manifest_path, description="sealed snapshot manifest")
    _validate_manifest(manifest)
    allowed = {record.relative_path.as_posix(): record.sha256 for record in _manifest_to_records(manifest["repositoryFiles"])}
    for name, module in _loaded_dino_module_items():
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            _verify_torch_hub_dynamic_namespace_module(
                name,
                module,
                snapshot=snapshot,
                allowed=allowed,
            )
            continue
        _verify_source_backed_dino_module(name, module, snapshot=snapshot, allowed=allowed)


def _verify_source_backed_dino_module(
    name: str,
    module: Any,
    *,
    snapshot: SealedDinoSnapshot,
    allowed: Mapping[str, str],
) -> None:
    """Prove one ordinary source-backed DINO module comes from the snapshot."""

    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str) or not module_path:
        raise SealedDinoSnapshotError(f"preloaded {name} has no verifiable source origin")
    candidate = Path(module_path)
    _reject_links_on_existing_path(candidate, description=f"preloaded {name}")
    if not candidate.is_file() or _is_link_or_reparse_point(candidate):
        raise SealedDinoSnapshotError(f"preloaded {name} is not a regular source file")
    try:
        resolved = candidate.resolve()
        relative = resolved.relative_to(snapshot.repository).as_posix()
    except (OSError, RuntimeError, ValueError) as error:
        raise SealedDinoSnapshotError(f"preloaded {name} originates outside the sealed snapshot") from error
    expected_digest = allowed.get(relative)
    if expected_digest is None:
        raise SealedDinoSnapshotError(f"preloaded {name} originates from an unmanifested snapshot file")
    actual_digest = sha256_file(resolved)
    if actual_digest != expected_digest:
        raise SealedDinoSnapshotError(f"preloaded {name} source digest does not match the sealed snapshot")


def _verify_torch_hub_dynamic_namespace_module(
    name: str,
    module: Any,
    *,
    snapshot: SealedDinoSnapshot,
    allowed: Mapping[str, str],
) -> None:
    """Accept only the two PEP 420 namespaces imported by this pinned hubconf.

    ``torch.hub`` imports the repository's ``hubconf.py`` under an anonymous
    module and Python creates namespace packages for the cache-free Cell-DINO
    and XRay-DINO directories.  Those packages intentionally have no
    ``__file__``.  Activation starts from an empty ``dinov2`` import cache, so
    this narrow structural proof binds the post-entry modules to the already
    revalidated sealed repository without trusting mutable module metadata.
    """

    expected_relative = _TORCH_HUB_DYNAMIC_NAMESPACE_DIRECTORIES.get(name)
    if expected_relative is None:
        raise SealedDinoSnapshotError(f"preloaded {name} has no verifiable source origin")
    if getattr(module, "__file__", None) is not None:
        raise SealedDinoSnapshotError(f"preloaded {name} is not a torch hub namespace package")
    spec = getattr(module, "__spec__", None)
    if not isinstance(spec, importlib.machinery.ModuleSpec):
        raise SealedDinoSnapshotError(f"preloaded {name} has no verifiable namespace specification")
    loader = spec.loader
    if not isinstance(loader, importlib.machinery.NamespaceLoader):
        raise SealedDinoSnapshotError(f"preloaded {name} does not use the Python namespace loader")
    locations = spec.submodule_search_locations
    if (
        spec.name != name
        or spec.origin is not None
        or spec.has_location
        or locations is None
        or getattr(module, "__package__", None) != name
        or getattr(module, "__loader__", None) is not loader
        or getattr(module, "__path__", None) is not locations
        or getattr(module, "__cached__", None) is not None
        or getattr(loader, "_path", None) is not locations
    ):
        raise SealedDinoSnapshotError(f"preloaded {name} has an unsafe namespace specification")
    if isinstance(locations, (str, bytes)):
        raise SealedDinoSnapshotError(f"preloaded {name} has an unsafe namespace search path")
    try:
        location_values = tuple(locations)
    except TypeError as error:
        raise SealedDinoSnapshotError(f"preloaded {name} has an unsafe namespace search path") from error
    if len(location_values) != 1 or not isinstance(location_values[0], str) or not location_values[0]:
        raise SealedDinoSnapshotError(f"preloaded {name} has an unsafe namespace search path")
    expected_directory = snapshot.repository.joinpath(*expected_relative.parts)
    _require_directory(expected_directory, description=f"preloaded {name} expected namespace directory")
    expected_resolved = expected_directory.resolve()
    candidate = Path(location_values[0])
    _reject_links_on_existing_path(candidate, description=f"preloaded {name} namespace directory")
    if not candidate.is_dir() or _is_link_or_reparse_point(candidate):
        raise SealedDinoSnapshotError(f"preloaded {name} namespace directory is not a regular directory")
    try:
        if candidate.resolve() != expected_resolved:
            raise SealedDinoSnapshotError(f"preloaded {name} namespace directory is not the sealed torch hub directory")
    except (OSError, RuntimeError) as error:
        raise SealedDinoSnapshotError(f"preloaded {name} namespace directory cannot be resolved safely") from error
    namespace_prefix = expected_relative.as_posix() + "/"
    if expected_relative.joinpath("__init__.py").as_posix() in allowed or not any(
        relative.startswith(namespace_prefix) for relative in allowed
    ):
        raise SealedDinoSnapshotError(f"preloaded {name} is not backed by a sealed namespace source directory")
    parent_name = name.rpartition(".")[0]
    parent = sys.modules.get(parent_name)
    if parent is None:
        raise SealedDinoSnapshotError(f"preloaded {name} has no source-backed parent package")
    _verify_source_backed_dino_module(parent_name, parent, snapshot=snapshot, allowed=allowed)
