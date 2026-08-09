"""Closed-envelope development evaluator for the reserve-successor V2 screen.

This module is deliberately independent of the frozen V1 evaluator.  It can
open only the successor ``FIT`` and ``THRESHOLD_TUNING`` image partitions via
the successor phase-safe loader.  The raw-FIT baseline never opens an
augmentation package; R3 candidates validate and use only FIT-derived images.
No selection, confirmation, reserve, blind, anomaly, or mask image is an
accepted input to this development entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Lock
from typing import Any, Callable

from PIL import Image, ImageOps, __version__ as PILLOW_VERSION

from phone_dino.engines import LocalDinoV2Adapter
from phone_dino.mvtec_normal_successor import (
    FRESH_NORMAL_SUCCESSOR_BLIND_POLICY,
    FRESH_NORMAL_SUCCESSOR_DELEGATION_POLICY,
    FRESH_NORMAL_SUCCESSOR_INDEPENDENCE_LABEL,
    FRESH_NORMAL_SUCCESSOR_RESULT_LABEL,
    FreshNormalSuccessorError,
    canonical_json_sha256,
    load_successor_safe_normal_inputs,
)
from phone_dino.production import DINO_INPUT_SIZE, DINO_RESIZE_SHORT_EDGE


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SUCCESSOR_V2_DEVELOPMENT_REPORT_SCHEMA = "phone-dino.mvtec-ad-successor-v2-development-report/1.0"
SUCCESSOR_V2_DEVELOPMENT_REPORT_PURPOSE = "OFFLINE_MVTEC_RESERVE_SUCCESSOR_V2_DEVELOPMENT"
SUCCESSOR_V2_DEVELOPMENT_PHASE = "SUCCESSOR_V2_DEVELOPMENT"
SUCCESSOR_V2_FEATURE_EXTRACTOR_SCHEMA = "phone-dino.mvtec-ad-successor-v2-feature-extractor/1.0"
SUCCESSOR_V2_PREPROCESSING_ID = "DINOV2_RESIZE_SHORT_EDGE_256_CENTER_CROP_224_IMAGENET_NORMALIZE_V1"
SUCCESSOR_V2_ALGORITHM_ID = "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V2"
SUCCESSOR_V2_PROTOTYPE_SELECTION = "DETERMINISTIC_STRATIFIED_HASH_RANKED_PATCH_PREFIX_V2"
RAW_FIT_ONLY = "RAW_FIT_ONLY"
RAW_FIT_PLUS_AUGMENTATION_R3 = "RAW_FIT_PLUS_AUGMENTATION_R3"
SUCCESSOR_V2_AUGMENTATION_VARIANTS = 3
SUCCESSOR_V2_AUGMENTATION_COMPONENTS = frozenset({"registration", "illumination", "sensor_transport"})
SUCCESSOR_V2_RESULT_LABEL = FRESH_NORMAL_SUCCESSOR_RESULT_LABEL
SUCCESSOR_V2_DELEGATION_POLICY = FRESH_NORMAL_SUCCESSOR_DELEGATION_POLICY
SUCCESSOR_V2_BLIND_POLICY = FRESH_NORMAL_SUCCESSOR_BLIND_POLICY
SUCCESSOR_V2_CATEGORIES = ("capsule", "metal_nut", "tile")

DEVELOPMENT_REPORT_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "resultLabel",
    "delegationPolicy",
    "independenceLabel",
    "inputPolicy",
    "parentHoldoutFileSha256",
    "parentHoldoutDeclaredSha256",
    "parentSelectionContractFileSha256",
    "parentSelectionContractDeclaredSha256",
    "parentNormalConfirmationIdentitySha256",
    "successorSealFileSha256",
    "successorSealDeclaredSha256",
    "successorPlanFileSha256",
    "successorPlanDeclaredSha256",
    "successorEnvelopeFileSha256",
    "successorEnvelopeDeclaredSha256",
    "successorFitIdentitySha256",
    "successorThresholdTuningIdentitySha256",
    "augmentationManifestFileSha256",
    "augmentationManifestDeclaredSha256",
    "augmentationRecipeFileSha256",
    "augmentationParentFitIdentitySha256",
    "candidateConfiguration",
    "candidateConfigurationSha256",
    "featureExtractor",
    "featureExtractorIdentitySha256",
    "featureInputs",
    "featureInputIdentitySha256",
    "calibrationInputs",
    "calibrationInputIdentitySha256",
    "thresholds",
    "categories",
    "calibrationScores",
    "normalOnlyEvidence",
    "execution",
    "developmentReportSha256",
}

CANDIDATE_CONFIGURATION_FIELDS = {
    "id",
    "algorithmId",
    "memoryBankSelection",
    "prototypeInputPolicy",
    "maxPrototypePatches",
    "topKMostAnomalousPatches",
    "prototypeBlockSize",
    "batchSize",
}

FEATURE_INPUT_FIELDS = {
    "caseId",
    "category",
    "partition",
    "kind",
    "defect",
    "sourceSha256",
    "sourceGroupId",
    "isAugmentation",
    "variantId",
    "component",
    "parentCaseId",
    "parentSourceSha256",
    "augmentationManifestSha256",
}

CALIBRATION_SCORE_FIELDS = {
    "caseId",
    "category",
    "partition",
    "kind",
    "defect",
    "sourceSha256",
    "score",
    "maxPatchDistance",
    "meanNearestPatchDistance",
}

CATEGORY_REPORT_FIELDS = {
    "fitOriginalCount",
    "fitAugmentedCount",
    "tuningOriginalCount",
    "prototypePatchCount",
    "fitPatchCount",
    "patchGridHeight",
    "patchGridWidth",
    "thresholdFromRawTuning",
    "tuningScoreMedian",
    "tuningScoreP95",
    "tuningScoreMax",
}

NORMAL_ONLY_EVIDENCE_FIELDS = {
    "featureInputCount",
    "featureInputPartitions",
    "featureInputKinds",
    "fitOriginalFeatureInputCount",
    "fitAugmentedFeatureInputCount",
    "tuningFeatureInputCount",
    "blindFeatureInputCount",
    "anomalyFeatureInputCount",
    "maskFeatureInputCount",
    "calibrationInputCount",
    "calibrationInputPartitions",
    "calibrationInputKinds",
    "blindCalibrationInputCount",
    "anomalyCalibrationInputCount",
    "maskCalibrationInputCount",
}

EXECUTION_FIELDS = {
    "evaluatorModuleSha256",
    "evaluatorEntrypointSha256",
    "phaseTimingsSeconds",
    "python",
    "platform",
    "numpyVersion",
    "torchVersion",
    "torchThreadCount",
    "gitRevision",
    "gitWorktreeClean",
}

CACHE_IGNORED_DIRECTORY_NAMES = frozenset({".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"})

PRE_REGISTERED_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "reserve-v2-raw-p2048-k5",
        "algorithmId": SUCCESSOR_V2_ALGORITHM_ID,
        "memoryBankSelection": SUCCESSOR_V2_PROTOTYPE_SELECTION,
        "prototypeInputPolicy": RAW_FIT_ONLY,
        "maxPrototypePatches": 2048,
        "topKMostAnomalousPatches": 5,
        "prototypeBlockSize": 256,
        "batchSize": 4,
    },
    {
        "id": "reserve-v2-r3-p1024-k3",
        "algorithmId": SUCCESSOR_V2_ALGORITHM_ID,
        "memoryBankSelection": SUCCESSOR_V2_PROTOTYPE_SELECTION,
        "prototypeInputPolicy": RAW_FIT_PLUS_AUGMENTATION_R3,
        "maxPrototypePatches": 1024,
        "topKMostAnomalousPatches": 3,
        "prototypeBlockSize": 256,
        "batchSize": 4,
    },
    {
        "id": "reserve-v2-r3-p2048-k3",
        "algorithmId": SUCCESSOR_V2_ALGORITHM_ID,
        "memoryBankSelection": SUCCESSOR_V2_PROTOTYPE_SELECTION,
        "prototypeInputPolicy": RAW_FIT_PLUS_AUGMENTATION_R3,
        "maxPrototypePatches": 2048,
        "topKMostAnomalousPatches": 3,
        "prototypeBlockSize": 256,
        "batchSize": 4,
    },
    {
        "id": "reserve-v2-r3-p2048-k5",
        "algorithmId": SUCCESSOR_V2_ALGORITHM_ID,
        "memoryBankSelection": SUCCESSOR_V2_PROTOTYPE_SELECTION,
        "prototypeInputPolicy": RAW_FIT_PLUS_AUGMENTATION_R3,
        "maxPrototypePatches": 2048,
        "topKMostAnomalousPatches": 5,
        "prototypeBlockSize": 256,
        "batchSize": 4,
    },
)


class SuccessorV2EvaluatorError(ValueError):
    """Raised when a reserve-successor V2 development run is unsafe."""


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


def _require_exact_fields(document: object, *, name: str, required: set[str]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise SuccessorV2EvaluatorError(f"{name} must be an object")
    missing = required.difference(document)
    if missing:
        raise SuccessorV2EvaluatorError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    unknown = set(document).difference(required)
    if unknown:
        raise SuccessorV2EvaluatorError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")
    return document


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SuccessorV2EvaluatorError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise SuccessorV2EvaluatorError(f"{name} must be a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise SuccessorV2EvaluatorError(f"{name} must be a SHA-256 digest") from error
    return digest


def _require_positive_int(value: object, *, name: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= maximum:
        raise SuccessorV2EvaluatorError(f"{name} must be a positive integer no larger than {maximum}")
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
                raise SuccessorV2EvaluatorError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _canonical_external_output_slot(path: Path, *, repository_root: Path) -> Path:
    """Validate an unwritten external report slot and return its canonical path.

    Reparse-point inspection deliberately happens before any containment check
    calls ``Path.resolve()``.  Batch callers use this no-write preflight to
    reject duplicate aliases before the first immutable report is created.
    """

    if not isinstance(path, Path):
        raise SuccessorV2EvaluatorError("development report output path must be a Path")
    _reject_links_on_existing_path(path, description="development report output")
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise SuccessorV2EvaluatorError("development report output must stay outside the Git working tree")
    if path.exists() or path.is_symlink():
        raise SuccessorV2EvaluatorError("development report output already exists; choose a new immutable path")
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise SuccessorV2EvaluatorError("unable to canonicalize development report output path") from error


def _require_external_output(path: Path, *, repository_root: Path) -> Path:
    """Prepare one prevalidated external report slot without allowing aliases."""

    canonical_path = _canonical_external_output_slot(path, repository_root=repository_root)
    try:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SuccessorV2EvaluatorError("unable to create development report output directory") from error
    # Re-check after creating the directory so a concurrent reparse/output
    # swap is detected before the report is opened with exclusive creation.
    return _canonical_external_output_slot(canonical_path, repository_root=repository_root)


def _immutable_file_sha256(path: Path, *, description: str) -> str:
    if not isinstance(path, Path) or not path.is_file() or _is_link_or_reparse_point(path):
        raise SuccessorV2EvaluatorError(f"{description} must be a regular non-link file")
    return sha256_file(path)


def sha256_directory(root: Path) -> str:
    """Hash a model source tree while rejecting links and generated caches."""

    if not isinstance(root, Path) or not root.is_dir() or _is_link_or_reparse_point(root):
        raise SuccessorV2EvaluatorError("model repository must be a non-link directory")
    entries: list[tuple[str, Path]] = []

    def visit(directory: Path) -> None:
        try:
            children = list(os.scandir(directory))
        except OSError as error:
            raise SuccessorV2EvaluatorError("unable to enumerate model repository") from error
        for child in children:
            candidate = Path(child.path)
            relative = candidate.relative_to(root)
            if _is_link_or_reparse_point(candidate):
                raise SuccessorV2EvaluatorError(
                    f"model repository contains a symbolic link or reparse point: {relative.as_posix()}"
                )
            if child.is_dir(follow_symlinks=False):
                if any(part in CACHE_IGNORED_DIRECTORY_NAMES for part in relative.parts):
                    continue
                entries.append(("D", candidate))
                visit(candidate)
            elif child.is_file(follow_symlinks=False):
                if any(part in CACHE_IGNORED_DIRECTORY_NAMES for part in relative.parts) or candidate.suffix == ".pyc":
                    continue
                entries.append(("F", candidate))
            else:
                raise SuccessorV2EvaluatorError("model repository contains an unsupported entry")

    visit(root)
    if not any(kind == "F" for kind, _ in entries):
        raise SuccessorV2EvaluatorError("model repository has no files to hash")
    digest = hashlib.sha256()
    for kind, candidate in sorted(entries, key=lambda item: (item[0], item[1].relative_to(root).as_posix())):
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(candidate.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        if kind == "F":
            digest.update(_immutable_file_sha256(candidate, description="model repository source").encode("ascii"))
            digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def pre_registered_candidate_configuration(identifier: str) -> dict[str, Any]:
    """Return a copy of exactly one V2 pre-registered candidate configuration."""

    for configuration in PRE_REGISTERED_CANDIDATES:
        if configuration["id"] == identifier:
            return dict(configuration)
    raise SuccessorV2EvaluatorError("candidate id is not in the pre-registered successor V2 universe")


def validate_candidate_configuration(value: object) -> dict[str, Any]:
    """Reject all knobs outside the immutable successor V2 candidate universe."""

    configuration = _require_exact_fields(value, name="candidateConfiguration", required=CANDIDATE_CONFIGURATION_FIELDS)
    identifier = _require_string(configuration.get("id"), name="candidateConfiguration.id")
    expected = pre_registered_candidate_configuration(identifier)
    if configuration != expected:
        raise SuccessorV2EvaluatorError("candidateConfiguration does not match its pre-registered V2 definition")
    return dict(expected)


def _feature_extractor_identity(*, model_repo: Path, model_weights: Path, device: str) -> dict[str, Any]:
    import numpy as np
    import torch
    import torchvision

    mkldnn_backend = getattr(torch.backends, "mkldnn", None)
    source_files = {
        "evaluatorModuleSha256": Path(__file__),
        "successorModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "mvtec_normal_successor.py",
        "successorAugmentationModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "mvtec_successor_fit_augmentation_v2.py",
        "productionModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "production.py",
        "enginesModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "engines.py",
    }
    return {
        "schemaVersion": SUCCESSOR_V2_FEATURE_EXTRACTOR_SCHEMA,
        "modelWeightsSha256": _immutable_file_sha256(model_weights, description="model weights"),
        "modelRepositorySha256": sha256_directory(model_repo),
        "preprocessingId": SUCCESSOR_V2_PREPROCESSING_ID,
        "preprocessing": {
            "colorSpace": "RGB",
            "resizeShortEdge": DINO_RESIZE_SHORT_EDGE,
            "centerCropWidth": DINO_INPUT_SIZE,
            "centerCropHeight": DINO_INPUT_SIZE,
            "resizeAntialias": True,
            "normalizeMean": [0.485, 0.456, 0.406],
            "normalizeStd": [0.229, 0.224, 0.225],
        },
        "modelEntrypoint": "dinov2_vits14",
        "device": device,
        **{name: _immutable_file_sha256(path, description="feature extractor source") for name, path in source_files.items()},
        "pythonVersion": platform.python_version(),
        "numpyVersion": np.__version__,
        "torchVersion": torch.__version__,
        "torchvisionVersion": torchvision.__version__,
        "pillowVersion": PILLOW_VERSION,
        "torchThreadCount": torch.get_num_threads(),
        "torchBackend": {
            "deterministicAlgorithmsEnabled": bool(torch.are_deterministic_algorithms_enabled()),
            "mkldnnAvailable": bool(mkldnn_backend and mkldnn_backend.is_available()),
            "mkldnnEnabled": bool(mkldnn_backend and mkldnn_backend.enabled),
        },
    }


class SuccessorV2BatchEmbedder:
    """Batch-only local DINO wrapper used by the V2 offline evaluator."""

    def __init__(self, *, model_repo: Path, model_weights: Path, device: str) -> None:
        adapter = LocalDinoV2Adapter(repository=model_repo, weights=model_weights)
        ready, reason = adapter.readiness()
        if not ready:
            raise RuntimeError(reason)
        self._device = device
        self._model = adapter.smoke_load().to(device)
        self._lock = Lock()

    def _transform(self, image: Image.Image) -> object:
        import numpy as np
        import torch
        from torchvision.transforms import v2

        pil = Image.fromarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        return v2.Compose([
            v2.Resize(DINO_RESIZE_SHORT_EDGE, antialias=True),
            v2.CenterCrop(DINO_INPUT_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])(pil).unsqueeze(0).to(self._device)

    def extract_patches(self, images: list[Image.Image]) -> list[object]:
        import numpy as np
        import torch

        if not images:
            return []
        tensors = [self._transform(image) for image in images]
        with self._lock:
            batch = torch.cat(tensors, dim=0)
            with torch.inference_mode():
                patches = self._model.forward_features(batch)["x_norm_patchtokens"].detach().cpu().numpy()
        return [np.asarray(value, dtype=np.float32) for value in patches]


def _safe_relative_path(value: object, *, name: str) -> PurePosixPath:
    text = _require_string(value, name=name)
    candidate = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        "\x00" in text
        or candidate.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or windows.root
        or not candidate.parts
        or any(
            part in {"", ".", ".."}
            or ":" in part
            or part != part.rstrip(" .")
            or PureWindowsPath(part).is_reserved()
            for part in candidate.parts
        )
    ):
        raise SuccessorV2EvaluatorError(f"{name} must be a safe relative POSIX path")
    if "\\" in text:
        raise SuccessorV2EvaluatorError(f"{name} must use POSIX separators")
    return candidate


def _safe_file_under(root: Path, relative: PurePosixPath, *, description: str, repository_root: Path) -> Path:
    if not root.is_dir() or _is_link_or_reparse_point(root):
        raise SuccessorV2EvaluatorError(f"{description} root must be a non-link directory")
    if _is_under(repository_root, root) or _is_under(root, repository_root):
        raise SuccessorV2EvaluatorError(f"{description} root must stay outside the Git working tree")
    candidate = root.joinpath(*relative.parts)
    _reject_links_on_existing_path(candidate, description=description)
    if not candidate.is_file() or _is_link_or_reparse_point(candidate) or not _is_under(root, candidate):
        raise SuccessorV2EvaluatorError(f"{description} is missing, unsafe, or escapes its package")
    return candidate


def _load_rgb_and_verify(record: dict[str, Any]) -> Image.Image:
    source_path = record.get("imagePath")
    if not isinstance(source_path, Path) or not source_path.is_file():
        raise SuccessorV2EvaluatorError("feature input is not a regular file")
    _reject_links_on_existing_path(source_path, description="feature input")
    if sha256_file(source_path) != _require_sha256(record.get("sourceSha256"), name="feature input sourceSha256"):
        raise SuccessorV2EvaluatorError("feature input digest does not match its frozen record")
    try:
        with Image.open(source_path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, SyntaxError, ValueError) as error:
        raise SuccessorV2EvaluatorError("unable to decode a feature input image") from error
    _reject_links_on_existing_path(source_path, description="feature input")
    if sha256_file(source_path) != _require_sha256(record.get("sourceSha256"), name="feature input sourceSha256"):
        raise SuccessorV2EvaluatorError("feature input changed while it was decoded")
    return image


def _extract_patch_features(
    records: list[dict[str, Any]],
    *,
    embedder: Any,
    batch_size: int,
    timings: dict[str, float],
    feature_cache: dict[tuple[str, str], object] | None = None,
    feature_cache_identity: str | None = None,
) -> dict[str, object]:
    if not records:
        raise SuccessorV2EvaluatorError("patch extraction requires at least one record")
    features: dict[str, object] = {}
    images: list[tuple[dict[str, Any], Image.Image, tuple[str, str] | None]] = []
    verification_started = time.perf_counter()
    for record in records:
        case_id = _require_string(record.get("caseId"), name="feature input caseId")
        if case_id in features or any(str(existing[0].get("caseId")) == case_id for existing in images):
            raise SuccessorV2EvaluatorError("feature input caseId is duplicated")
        source_sha256 = _require_sha256(record.get("sourceSha256"), name="feature input sourceSha256")
        cache_key = None if feature_cache is None or feature_cache_identity is None else (feature_cache_identity, source_sha256)
        image = _load_rgb_and_verify(record)
        if cache_key is not None and cache_key in feature_cache:
            features[case_id] = feature_cache[cache_key]
            continue
        images.append((record, image, cache_key))
    timings["inputVerificationSeconds"] += time.perf_counter() - verification_started
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inference_started = time.perf_counter()
        extracted = embedder.extract_patches([image for _, image, _ in batch])
        timings["featureInferenceSeconds"] += time.perf_counter() - inference_started
        if len(extracted) != len(batch):
            raise SuccessorV2EvaluatorError("DINO patch extraction returned an unexpected batch size")
        for (record, _, cache_key), values in zip(batch, extracted, strict=True):
            features[str(record["caseId"])] = values
            if cache_key is not None and feature_cache is not None:
                feature_cache[cache_key] = values
    return features


def _normalized_rows(values: object) -> object:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise SuccessorV2EvaluatorError("patch feature matrix is empty or invalid")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.all(np.isfinite(array)) or np.any(norms <= 1e-12):
        raise SuccessorV2EvaluatorError("patch feature matrix contains non-finite or zero-norm values")
    return array / norms


def _stratum_identifier(record: dict[str, Any]) -> tuple[str, int, str]:
    """Return a stable raw/variant stratum key for a FIT feature record."""

    parent_case_id = record.get("parentCaseId")
    if parent_case_id is None:
        parent_case_id = record.get("caseId")
    parent = _require_string(parent_case_id, name="prototype parent caseId")
    if bool(record.get("isAugmentation")):
        variant = _require_positive_int(record.get("variantId"), name="prototype variantId", maximum=SUCCESSOR_V2_AUGMENTATION_VARIANTS)
    else:
        variant = 0
    case_id = _require_string(record.get("caseId"), name="prototype caseId")
    return parent, variant, case_id


def deterministic_stratified_hash_ranked_patch_indices(
    records: list[dict[str, Any]],
    patch_counts: list[int],
    maximum: int,
) -> list[int]:
    """Select a deterministic round-robin prefix over stable parent/variant strata.

    Every stratum ranks its own patch indices with SHA-256.  The global order
    is then generated round-robin over lexicographically stable strata.  A
    shorter requested bank is therefore exactly a prefix of every longer bank
    from the same records, which is required by the 1,024/2,048 V2 ablation.
    """

    maximum = _require_positive_int(maximum, name="maxPrototypePatches", maximum=65_536)
    if not records or len(records) != len(patch_counts):
        raise SuccessorV2EvaluatorError("prototype records and patch counts must be non-empty and aligned")
    start = 0
    strata: dict[tuple[str, int, str], list[int]] = {}
    for record, count in zip(records, patch_counts, strict=True):
        patch_count = _require_positive_int(count, name="prototype patch count", maximum=65_536)
        if record.get("partition") != "FIT":
            raise SuccessorV2EvaluatorError("prototype selection accepts FIT records only")
        key = _stratum_identifier(record)
        if key in strata:
            raise SuccessorV2EvaluatorError("prototype parent/variant stratum is duplicated")
        case_id = key[2]
        source_sha256 = _require_sha256(record.get("sourceSha256"), name="prototype sourceSha256")
        ranked_local = sorted(
            range(patch_count),
            key=lambda patch_index: (
                hashlib.sha256(
                    "\0".join((
                        SUCCESSOR_V2_PROTOTYPE_SELECTION,
                        case_id,
                        source_sha256,
                        str(patch_index),
                    )).encode("utf-8")
                ).hexdigest(),
                patch_index,
            ),
        )
        strata[key] = [start + patch_index for patch_index in ranked_local]
        start += patch_count
    selected: list[int] = []
    while len(selected) < min(maximum, start):
        progressed = False
        for key in sorted(strata):
            queue = strata[key]
            if queue:
                selected.append(queue.pop(0))
                progressed = True
                if len(selected) == min(maximum, start):
                    break
        if not progressed:
            break
    if len(selected) != min(maximum, start) or len(selected) != len(set(selected)):
        raise SuccessorV2EvaluatorError("deterministic prototype selection did not produce a unique prefix")
    return selected


def patch_knn_scores_blocked(
    query_patches: object,
    prototype_patches: object,
    *,
    top_k: int,
    prototype_block_size: int,
) -> list[dict[str, float]]:
    """Score query patches with exact blocked nearest-cosine-distance scoring."""

    import numpy as np

    query = np.asarray(query_patches, dtype=np.float32)
    prototypes = np.asarray(prototype_patches, dtype=np.float32)
    if query.ndim != 3 or prototypes.ndim != 2 or query.shape[2] != prototypes.shape[1]:
        raise SuccessorV2EvaluatorError("patch query/prototype dimensions do not match")
    if query.shape[0] == 0 or query.shape[1] == 0 or prototypes.shape[0] == 0:
        raise SuccessorV2EvaluatorError("patch query/prototype matrix is empty")
    if top_k <= 0 or top_k > query.shape[1] or prototype_block_size <= 0:
        raise SuccessorV2EvaluatorError("patch scoring parameters are invalid")
    normalized_query = _normalized_rows(query.reshape(-1, query.shape[2]))
    normalized_prototypes = _normalized_rows(prototypes)
    nearest = np.full(normalized_query.shape[0], np.inf, dtype=np.float32)
    for start in range(0, normalized_prototypes.shape[0], prototype_block_size):
        block = normalized_prototypes[start:start + prototype_block_size]
        distances = np.clip(1.0 - normalized_query @ block.T, 0.0, 2.0)
        nearest = np.minimum(nearest, np.min(distances, axis=1))
    grids = nearest.reshape(query.shape[0], query.shape[1])
    largest = np.partition(grids, -top_k, axis=1)[:, -top_k:]
    return [
        {
            "score": float(np.mean(largest[index])),
            "maxPatchDistance": float(np.max(grids[index])),
            "meanNearestPatchDistance": float(np.mean(grids[index])),
        }
        for index in range(query.shape[0])
    ]


def _feature_input_identity_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "partition": record["partition"],
        "kind": record["kind"],
        "defect": record["defect"],
        "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"],
        "isAugmentation": bool(record.get("isAugmentation", False)),
        "variantId": record.get("variantId"),
        "component": record.get("component"),
        "parentCaseId": record.get("parentCaseId"),
        "parentSourceSha256": record.get("parentSourceSha256"),
        "augmentationManifestSha256": record.get("augmentationManifestSha256"),
    }


def _feature_input_identity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_feature_input_identity_record(record) for record in sorted(records, key=lambda item: str(item["caseId"]))]


def _adapt_successor_original_record(record: dict[str, Any]) -> dict[str, Any]:
    partition = record.get("partition")
    if partition not in {"FIT", "THRESHOLD_TUNING"}:
        raise SuccessorV2EvaluatorError("development evaluator received an unsupported successor partition")
    if record.get("kind") != "NOMINAL" or record.get("defect") != "good":
        raise SuccessorV2EvaluatorError("development evaluator requires nominal-good successor originals")
    for name in ("caseId", "category", "sourceGroupId"):
        _require_string(record.get(name), name=f"successor original {name}")
    _require_sha256(record.get("sourceSha256"), name="successor original sourceSha256")
    if not isinstance(record.get("imagePath"), Path):
        raise SuccessorV2EvaluatorError("successor original was not phase-safely loaded")
    return {
        **record,
        "isAugmentation": False,
        "variantId": None,
        "component": None,
        "parentCaseId": None,
        "parentSourceSha256": None,
        "augmentationManifestSha256": None,
    }


def _adapt_augmentation_record(
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
        raise SuccessorV2EvaluatorError("successor V2 augmentation record has an unsafe input scope")
    for name in ("caseId", "parentCaseId", "parentSourceSha256", "sourceGroupId", "category", "sourceSha256", "component"):
        if name.endswith("Sha256"):
            _require_sha256(record.get(name), name=f"successor V2 augmentation {name}")
        else:
            _require_string(record.get(name), name=f"successor V2 augmentation {name}")
    variant_id = _require_positive_int(
        record.get("variantId"), name="successor V2 augmentation variantId", maximum=SUCCESSOR_V2_AUGMENTATION_VARIANTS
    )
    if record.get("component") not in SUCCESSOR_V2_AUGMENTATION_COMPONENTS:
        raise SuccessorV2EvaluatorError("successor V2 augmentation component is unsupported")
    relative = _safe_relative_path(record.get("relativePath"), name="successor V2 augmentation relativePath")
    image_path = _safe_file_under(
        augmentation_root,
        relative,
        description="successor V2 augmentation output",
        repository_root=repository_root,
    )
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "partition": "FIT",
        "kind": "NOMINAL",
        "defect": "good",
        "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"],
        "imagePath": image_path,
        "isAugmentation": True,
        "variantId": variant_id,
        "component": record["component"],
        "parentCaseId": record["parentCaseId"],
        "parentSourceSha256": record["parentSourceSha256"],
        "augmentationManifestSha256": augmentation_manifest_sha256,
    }


def _load_validated_successor_fit_augmentations(
    augmentation_manifest_path: Path,
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    recipe_path: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Load the optional V2 augmentation package lazily to preserve raw isolation."""

    try:
        from phone_dino.mvtec_successor_fit_augmentation_v2 import (  # type: ignore[import-not-found]
            load_validated_successor_fit_augmentations_with_file_sha256,
        )
    except ImportError as error:  # pragma: no cover - only possible in a partial install
        raise SuccessorV2EvaluatorError("successor V2 augmentation support is not installed") from error
    try:
        return load_validated_successor_fit_augmentations_with_file_sha256(
            augmentation_manifest_path,
            parent_holdout_path,
            parent_selection_contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=recipe_path,
            repository_root=repository_root,
        )
    except ValueError as error:
        raise SuccessorV2EvaluatorError(str(error)) from error


def _validate_augmentation_binding(
    augmentation: dict[str, Any],
    *,
    envelope: dict[str, Any],
    envelope_file_sha256: str,
    plan_path: Path,
    repository_root: Path,
) -> tuple[str, str, str]:
    """Validate report-critical bindings without relying on augmentation internals."""

    required = {
        "augmentationManifestSha256",
        "successorFitIdentitySha256",
        "successorEnvelopeFileSha256",
        "successorEnvelopeDeclaredSha256",
        "successorPlanFileSha256",
        "successorPlanDeclaredSha256",
        "recipeFileSha256",
        "variantsPerParent",
    }
    if not isinstance(augmentation, dict):
        raise SuccessorV2EvaluatorError("successor V2 augmentation manifest must be an object")
    parsed = augmentation
    missing = required.difference(parsed)
    if missing:
        raise SuccessorV2EvaluatorError(
            f"successor V2 augmentation manifest is missing required binding fields: {', '.join(sorted(missing))}"
        )
    manifest_declared = _require_sha256(parsed.get("augmentationManifestSha256"), name="augmentationManifestSha256")
    if _require_sha256(parsed.get("successorFitIdentitySha256"), name="successorFitIdentitySha256") != envelope[
        "successorPartitionIdentities"
    ]["FIT"]:
        raise SuccessorV2EvaluatorError("successor V2 augmentation is not bound to the successor FIT identity")
    if _require_sha256(parsed.get("successorEnvelopeFileSha256"), name="successorEnvelopeFileSha256") != envelope_file_sha256:
        raise SuccessorV2EvaluatorError("successor V2 augmentation is not bound to this successor envelope file")
    if _require_sha256(parsed.get("successorEnvelopeDeclaredSha256"), name="successorEnvelopeDeclaredSha256") != envelope[
        "successorEnvelopeSha256"
    ]:
        raise SuccessorV2EvaluatorError("successor V2 augmentation is not bound to this successor envelope")
    if _require_sha256(parsed.get("successorPlanFileSha256"), name="successorPlanFileSha256") != sha256_file(plan_path):
        raise SuccessorV2EvaluatorError("successor V2 augmentation is not bound to this successor plan file")
    if _require_sha256(parsed.get("successorPlanDeclaredSha256"), name="successorPlanDeclaredSha256") != envelope[
        "planDeclaredSha256"
    ]:
        raise SuccessorV2EvaluatorError("successor V2 augmentation is not bound to this successor plan")
    recipe_file_sha256 = _require_sha256(parsed.get("recipeFileSha256"), name="recipeFileSha256")
    if _require_positive_int(parsed.get("variantsPerParent"), name="variantsPerParent", maximum=8) != SUCCESSOR_V2_AUGMENTATION_VARIANTS:
        raise SuccessorV2EvaluatorError("successor V2 augmentation must contain exactly R3 derivatives")
    return manifest_declared, recipe_file_sha256, _require_sha256(
        parsed.get("successorFitIdentitySha256"), name="successorFitIdentitySha256"
    )


def _development_inputs(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    configuration: dict[str, Any],
    augmentation_manifest_path: Path | None,
    recipe_path: Path | None,
    repository_root: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, Any] | None, str | None]:
    """Read only successor FIT/raw-tuning, then optional FIT-only R3 children."""

    try:
        envelope, envelope_file_sha256, originals = load_successor_safe_normal_inputs(
            parent_holdout_path,
            parent_selection_contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            partitions={"FIT", "THRESHOLD_TUNING"},
            repository_root=repository_root,
        )
    except FreshNormalSuccessorError as error:
        raise SuccessorV2EvaluatorError(str(error)) from error
    original_records = [_adapt_successor_original_record(record) for record in originals]
    if {record["partition"] for record in original_records} != {"FIT", "THRESHOLD_TUNING"}:
        raise SuccessorV2EvaluatorError("successor development requires both FIT and raw threshold-tuning inputs")
    if configuration["prototypeInputPolicy"] == RAW_FIT_ONLY:
        if augmentation_manifest_path is not None or recipe_path is not None:
            raise SuccessorV2EvaluatorError("raw-FIT baseline must not accept an augmentation package or recipe")
        combined = sorted(original_records, key=lambda record: str(record["caseId"]))
        case_ids = [str(record["caseId"]) for record in combined]
        if len(case_ids) != len(set(case_ids)):
            raise SuccessorV2EvaluatorError("successor V2 development feature input caseId is duplicated")
        return envelope, envelope_file_sha256, combined, None, None
    if configuration["prototypeInputPolicy"] != RAW_FIT_PLUS_AUGMENTATION_R3:
        raise SuccessorV2EvaluatorError("successor V2 prototype input policy is unsupported")
    if augmentation_manifest_path is None or recipe_path is None:
        raise SuccessorV2EvaluatorError("R3 candidate requires an augmentation manifest and recipe")
    augmentation, augmentation_file_sha256, augmentation_records = _load_validated_successor_fit_augmentations(
        augmentation_manifest_path,
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=recipe_path,
        repository_root=repository_root,
    )
    if not isinstance(augmentation, dict) or not isinstance(augmentation_records, list):
        raise SuccessorV2EvaluatorError("successor V2 augmentation loader returned an unsafe value")
    augmentation_file_sha256 = _require_sha256(
        augmentation_file_sha256, name="validated successor V2 augmentation manifest file digest"
    )
    if _immutable_file_sha256(augmentation_manifest_path, description="successor V2 augmentation manifest") != augmentation_file_sha256:
        raise SuccessorV2EvaluatorError("successor V2 augmentation manifest changed after validation")
    manifest_declared, _recipe_file_sha256, _fit_identity = _validate_augmentation_binding(
        augmentation,
        envelope=envelope,
        envelope_file_sha256=envelope_file_sha256,
        plan_path=plan_path,
        repository_root=repository_root,
    )
    if recipe_path is None or _immutable_file_sha256(recipe_path, description="successor V2 augmentation recipe") != _recipe_file_sha256:
        raise SuccessorV2EvaluatorError("successor V2 augmentation recipe file does not match its manifest")
    augmented = [
        _adapt_augmentation_record(
            record,
            augmentation_root=augmentation_manifest_path.parent,
            augmentation_manifest_sha256=manifest_declared,
            repository_root=repository_root,
        )
        for record in augmentation_records
    ]
    raw_fit = [record for record in original_records if record["partition"] == "FIT"]
    expected_by_parent = {str(record["caseId"]): record for record in raw_fit}
    seen_augmented_case_ids: set[str] = set()
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for record in augmented:
        case_id = str(record["caseId"])
        if case_id in seen_augmented_case_ids:
            raise SuccessorV2EvaluatorError("successor V2 augmentation caseId is duplicated")
        seen_augmented_case_ids.add(case_id)
        parent = expected_by_parent.get(str(record["parentCaseId"]))
        if parent is None or parent["sourceSha256"] != record["parentSourceSha256"] or parent["category"] != record["category"]:
            raise SuccessorV2EvaluatorError("successor V2 augmentation parent does not match successor FIT")
        by_parent.setdefault(str(record["parentCaseId"]), []).append(record)
    if set(by_parent) != set(expected_by_parent):
        raise SuccessorV2EvaluatorError("successor V2 augmentation does not cover every successor FIT parent")
    for records in by_parent.values():
        variants = sorted(int(record["variantId"]) for record in records)
        components = {str(record["component"]) for record in records}
        if variants != [1, 2, 3] or components != SUCCESSOR_V2_AUGMENTATION_COMPONENTS:
            raise SuccessorV2EvaluatorError("successor V2 augmentation must provide one distinct R3 child per FIT parent")
    combined = sorted(original_records + augmented, key=lambda record: str(record["caseId"]))
    case_ids = [str(record["caseId"]) for record in combined]
    if len(case_ids) != len(set(case_ids)):
        raise SuccessorV2EvaluatorError("successor V2 development feature input caseId is duplicated")
    return envelope, envelope_file_sha256, combined, augmentation, augmentation_file_sha256


def _p95(values: list[float]) -> float:
    if not values:
        raise SuccessorV2EvaluatorError("P95 requires at least one score")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _normal_summary(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise SuccessorV2EvaluatorError("normal score summary requires at least one score")
    ordered = sorted(values)
    return ordered[len(ordered) // 2], _p95(ordered), ordered[-1]


def _build_normal_only_evidence(feature_records: list[dict[str, Any]], calibration_records: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = {
        "featureInputCount": len(feature_records),
        "featureInputPartitions": sorted({str(record["partition"]) for record in feature_records}),
        "featureInputKinds": sorted({str(record["kind"]) for record in feature_records}),
        "fitOriginalFeatureInputCount": sum(
            record["partition"] == "FIT" and not record["isAugmentation"] for record in feature_records
        ),
        "fitAugmentedFeatureInputCount": sum(
            record["partition"] == "FIT" and record["isAugmentation"] for record in feature_records
        ),
        "tuningFeatureInputCount": sum(record["partition"] == "THRESHOLD_TUNING" for record in feature_records),
        "blindFeatureInputCount": 0,
        "anomalyFeatureInputCount": 0,
        "maskFeatureInputCount": 0,
        "calibrationInputCount": len(calibration_records),
        "calibrationInputPartitions": sorted({str(record["partition"]) for record in calibration_records}),
        "calibrationInputKinds": sorted({str(record["kind"]) for record in calibration_records}),
        "blindCalibrationInputCount": 0,
        "anomalyCalibrationInputCount": 0,
        "maskCalibrationInputCount": 0,
    }
    _require_exact_fields(evidence, name="normalOnlyEvidence", required=NORMAL_ONLY_EVIDENCE_FIELDS)
    if (
        evidence["featureInputPartitions"] != ["FIT", "THRESHOLD_TUNING"]
        or evidence["featureInputKinds"] != ["NOMINAL"]
        or evidence["calibrationInputPartitions"] != ["THRESHOLD_TUNING"]
        or evidence["calibrationInputKinds"] != ["NOMINAL"]
    ):
        raise SuccessorV2EvaluatorError("successor V2 normal-only evidence is inconsistent")
    return evidence


def _execution_metadata(timings: dict[str, float], *, repository_root: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        worktree_clean = not subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - environment-dependent Git availability
        revision = None
        worktree_clean = None
    metadata = {
        "evaluatorModuleSha256": sha256_file(Path(__file__)),
        "evaluatorEntrypointSha256": sha256_file(
            repository_root / "tools" / "run_mvtec_ad_successor_v2_development.py"
        ),
        "phaseTimingsSeconds": {name: round(value, 6) for name, value in timings.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "numpyVersion": np.__version__,
        "torchVersion": torch.__version__,
        "torchThreadCount": torch.get_num_threads(),
        "gitRevision": revision,
        "gitWorktreeClean": worktree_clean,
    }
    _require_exact_fields(metadata, name="development execution", required=EXECUTION_FIELDS)
    return metadata


def _required_envelope_digest(envelope: dict[str, Any], name: str) -> str:
    return _require_sha256(envelope.get(name), name=f"successor envelope {name}")


def run_successor_v2_development_evaluation(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    output_path: Path,
    *,
    source_root: Path,
    model_repo: Path,
    model_weights: Path,
    device: str,
    candidate_configuration: dict[str, Any],
    augmentation_manifest_path: Path | None = None,
    recipe_path: Path | None = None,
    repository_root: Path = REPOSITORY_ROOT,
    embedder_factory: Callable[..., Any] = SuccessorV2BatchEmbedder,
    identity_factory: Callable[..., dict[str, Any]] = _feature_extractor_identity,
    in_memory_feature_cache: dict[tuple[str, str], object] | None = None,
) -> dict[str, Any]:
    """Run one immutable V2 normal-only development observation.

    The function accepts no selection, confirmation, blind, anomaly, or mask
    path.  It makes the fixed raw tuning maximum a threshold and writes an
    external, new-only research report; it does not select or promote a model.
    """

    configuration = validate_candidate_configuration(candidate_configuration)
    if device != "cpu":
        raise SuccessorV2EvaluatorError("successor V2 development supports CPU only")
    output_path = _require_external_output(output_path, repository_root=repository_root)
    started = time.perf_counter()
    timings = {
        "inputAssemblySeconds": 0.0,
        "provenanceSeconds": 0.0,
        "inputVerificationSeconds": 0.0,
        "featureInferenceSeconds": 0.0,
        "scoringSeconds": 0.0,
        "totalElapsedSeconds": 0.0,
    }
    input_started = time.perf_counter()
    envelope, envelope_file_sha256, feature_records, augmentation, augmentation_file_sha256 = _development_inputs(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        configuration=configuration,
        augmentation_manifest_path=augmentation_manifest_path,
        recipe_path=recipe_path,
        repository_root=repository_root,
    )
    timings["inputAssemblySeconds"] += time.perf_counter() - input_started
    provenance_started = time.perf_counter()
    extractor_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    extractor_identity_sha256 = canonical_json_sha256(extractor_identity)
    timings["provenanceSeconds"] += time.perf_counter() - provenance_started
    embedder = embedder_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    provenance_started = time.perf_counter()
    loaded_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    timings["provenanceSeconds"] += time.perf_counter() - provenance_started
    if loaded_identity != extractor_identity:
        raise SuccessorV2EvaluatorError("feature extractor inputs changed while DINO loaded")

    categories = sorted({str(record["category"]) for record in feature_records})
    if tuple(categories) != SUCCESSOR_V2_CATEGORIES:
        raise SuccessorV2EvaluatorError("successor V2 development evaluator requires the fixed capsule, metal_nut, and tile categories")
    calibration_scores: list[dict[str, Any]] = []
    category_reports: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for category in categories:
        category_inputs = [record for record in feature_records if record["category"] == category]
        fit = sorted((record for record in category_inputs if record["partition"] == "FIT"), key=lambda item: str(item["caseId"]))
        tuning = sorted(
            (record for record in category_inputs if record["partition"] == "THRESHOLD_TUNING"),
            key=lambda item: str(item["caseId"]),
        )
        raw_fit = [record for record in fit if not record["isAugmentation"]]
        augmented_fit = [record for record in fit if record["isAugmentation"]]
        if len(raw_fit) != 12 or len(tuning) != 4:
            raise SuccessorV2EvaluatorError("each successor category requires exactly 12 raw FIT and 4 raw tuning inputs")
        if any(record["isAugmentation"] for record in tuning):
            raise SuccessorV2EvaluatorError("successor V2 threshold tuning must remain raw originals")
        if configuration["prototypeInputPolicy"] == RAW_FIT_ONLY and augmented_fit:
            raise SuccessorV2EvaluatorError("raw-FIT candidate must not receive augmented prototype inputs")
        if configuration["prototypeInputPolicy"] == RAW_FIT_PLUS_AUGMENTATION_R3 and len(augmented_fit) != 36:
            raise SuccessorV2EvaluatorError("R3 candidate requires exactly 36 FIT derivatives per category")
        scoring_started = time.perf_counter()
        features = _extract_patch_features(
            fit + tuning,
            embedder=embedder,
            batch_size=int(configuration["batchSize"]),
            timings=timings,
            feature_cache=in_memory_feature_cache,
            feature_cache_identity=extractor_identity_sha256,
        )
        import numpy as np

        fit_matrices = [np.asarray(features[str(record["caseId"])], dtype=np.float32) for record in fit]
        patch_counts = [int(matrix.shape[0]) if matrix.ndim == 2 else 0 for matrix in fit_matrices]
        prototype_indices = deterministic_stratified_hash_ranked_patch_indices(
            fit,
            patch_counts,
            int(configuration["maxPrototypePatches"]),
        )
        fit_patches = np.concatenate(fit_matrices, axis=0)
        prototypes = fit_patches[prototype_indices]
        query_patches = np.asarray([features[str(record["caseId"])] for record in tuning], dtype=np.float32)
        components = patch_knn_scores_blocked(
            query_patches,
            prototypes,
            top_k=int(configuration["topKMostAnomalousPatches"]),
            prototype_block_size=int(configuration["prototypeBlockSize"]),
        )
        timings["scoringSeconds"] += time.perf_counter() - scoring_started
        patch_count = query_patches.shape[1]
        grid_side = int(math.isqrt(int(patch_count)))
        if grid_side * grid_side != patch_count:
            raise SuccessorV2EvaluatorError("DINO patch grid is not square")
        category_scores: list[float] = []
        for record, component in zip(tuning, components, strict=True):
            score = float(component["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 2.0:
                raise SuccessorV2EvaluatorError("DINO patch score is outside cosine-distance bounds")
            category_scores.append(score)
            score_record = {
                "caseId": record["caseId"],
                "category": category,
                "partition": "THRESHOLD_TUNING",
                "kind": "NOMINAL",
                "defect": "good",
                "sourceSha256": record["sourceSha256"],
                "score": score,
                "maxPatchDistance": float(component["maxPatchDistance"]),
                "meanNearestPatchDistance": float(component["meanNearestPatchDistance"]),
            }
            _require_exact_fields(score_record, name="successor V2 calibration score", required=CALIBRATION_SCORE_FIELDS)
            calibration_scores.append(score_record)
        median, p95, maximum = _normal_summary(category_scores)
        thresholds[category] = maximum
        category_report = {
            "fitOriginalCount": len(raw_fit),
            "fitAugmentedCount": len(augmented_fit),
            "tuningOriginalCount": len(tuning),
            "prototypePatchCount": int(len(prototypes)),
            "fitPatchCount": int(len(fit_patches)),
            "patchGridHeight": grid_side,
            "patchGridWidth": grid_side,
            "thresholdFromRawTuning": maximum,
            "tuningScoreMedian": median,
            "tuningScoreP95": p95,
            "tuningScoreMax": maximum,
        }
        _require_exact_fields(category_report, name="successor V2 category report", required=CATEGORY_REPORT_FIELDS)
        category_reports[category] = category_report

    provenance_started = time.perf_counter()
    completed_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    timings["provenanceSeconds"] += time.perf_counter() - provenance_started
    if completed_identity != extractor_identity:
        raise SuccessorV2EvaluatorError("feature extractor inputs changed while DINO features were extracted")
    feature_inputs = _feature_input_identity(feature_records)
    calibration_records = [record for record in feature_records if record["partition"] == "THRESHOLD_TUNING"]
    calibration_inputs = _feature_input_identity(calibration_records)
    calibration_scores.sort(key=lambda record: str(record["caseId"]))
    timings["totalElapsedSeconds"] = time.perf_counter() - started
    parent_evidence = envelope.get("parentEvidence")
    if not isinstance(parent_evidence, dict):
        raise SuccessorV2EvaluatorError("successor envelope parent evidence is missing")
    augmentation_declared = None if augmentation is None else _require_sha256(
        augmentation.get("augmentationManifestSha256"), name="augmentationManifestSha256"
    )
    augmentation_recipe = None if augmentation is None else _require_sha256(
        augmentation.get("recipeFileSha256"), name="recipeFileSha256"
    )
    augmentation_fit_identity = None if augmentation is None else _require_sha256(
        augmentation.get("successorFitIdentitySha256"), name="successorFitIdentitySha256"
    )
    report: dict[str, Any] = {
        "schemaVersion": SUCCESSOR_V2_DEVELOPMENT_REPORT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": SUCCESSOR_V2_DEVELOPMENT_REPORT_PURPOSE,
        "phase": SUCCESSOR_V2_DEVELOPMENT_PHASE,
        "blindPolicy": SUCCESSOR_V2_BLIND_POLICY,
        "resultLabel": SUCCESSOR_V2_RESULT_LABEL,
        "delegationPolicy": SUCCESSOR_V2_DELEGATION_POLICY,
        "independenceLabel": FRESH_NORMAL_SUCCESSOR_INDEPENDENCE_LABEL,
        "inputPolicy": (
            "SUCCESSOR_RAW_FIT_PLUS_RAW_THRESHOLD_TUNING_ONLY"
            if configuration["prototypeInputPolicy"] == RAW_FIT_ONLY
            else "SUCCESSOR_RAW_FIT_PLUS_AUGMENTATION_R3_PLUS_RAW_THRESHOLD_TUNING_ONLY"
        ),
        "parentHoldoutFileSha256": _require_sha256(
            parent_evidence.get("holdoutManifestFileSha256"), name="parent holdout file digest"
        ),
        "parentHoldoutDeclaredSha256": _require_sha256(
            parent_evidence.get("holdoutManifestDeclaredSha256"), name="parent holdout declared digest"
        ),
        "parentSelectionContractFileSha256": _require_sha256(
            parent_evidence.get("selectionContractFileSha256"), name="parent selection contract file digest"
        ),
        "parentSelectionContractDeclaredSha256": _require_sha256(
            parent_evidence.get("selectionContractDeclaredSha256"), name="parent selection contract declared digest"
        ),
        "parentNormalConfirmationIdentitySha256": _require_sha256(
            parent_evidence.get("parentNormalConfirmationIdentitySha256"), name="parent normal confirmation identity"
        ),
        "successorSealFileSha256": _required_envelope_digest(envelope, "sealFileSha256"),
        "successorSealDeclaredSha256": _required_envelope_digest(envelope, "sealDeclaredSha256"),
        "successorPlanFileSha256": _required_envelope_digest(envelope, "planFileSha256"),
        "successorPlanDeclaredSha256": _required_envelope_digest(envelope, "planDeclaredSha256"),
        "successorEnvelopeFileSha256": envelope_file_sha256,
        "successorEnvelopeDeclaredSha256": _required_envelope_digest(envelope, "successorEnvelopeSha256"),
        "successorFitIdentitySha256": _require_sha256(
            envelope.get("successorPartitionIdentities", {}).get("FIT"), name="successor FIT identity"
        ),
        "successorThresholdTuningIdentitySha256": _require_sha256(
            envelope.get("successorPartitionIdentities", {}).get("THRESHOLD_TUNING"), name="successor tuning identity"
        ),
        "augmentationManifestFileSha256": augmentation_file_sha256,
        "augmentationManifestDeclaredSha256": augmentation_declared,
        "augmentationRecipeFileSha256": augmentation_recipe,
        "augmentationParentFitIdentitySha256": augmentation_fit_identity,
        "candidateConfiguration": configuration,
        "candidateConfigurationSha256": canonical_json_sha256(configuration),
        "featureExtractor": extractor_identity,
        "featureExtractorIdentitySha256": extractor_identity_sha256,
        "featureInputs": feature_inputs,
        "featureInputIdentitySha256": canonical_json_sha256(feature_inputs),
        "calibrationInputs": calibration_inputs,
        "calibrationInputIdentitySha256": canonical_json_sha256(calibration_inputs),
        "thresholds": {category: thresholds[category] for category in sorted(thresholds)},
        "categories": {category: category_reports[category] for category in sorted(category_reports)},
        "calibrationScores": calibration_scores,
        "normalOnlyEvidence": _build_normal_only_evidence(feature_records, calibration_records),
        "execution": _execution_metadata(timings, repository_root=repository_root),
    }
    report["developmentReportSha256"] = _document_digest(report, "developmentReportSha256")
    _require_exact_fields(report, name="successor V2 development report", required=DEVELOPMENT_REPORT_FIELDS)
    try:
        with output_path.open("xb") as stream:
            stream.write((json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise SuccessorV2EvaluatorError("unable to write immutable successor V2 development report") from error
    return report


def run_successor_v2_development_evaluations(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    output_paths: dict[str, Path],
    *,
    source_root: Path,
    model_repo: Path,
    model_weights: Path,
    device: str,
    candidate_configurations: list[dict[str, Any]],
    augmentation_manifest_path: Path | None = None,
    recipe_path: Path | None = None,
    repository_root: Path = REPOSITORY_ROOT,
    embedder_factory: Callable[..., Any] = SuccessorV2BatchEmbedder,
    identity_factory: Callable[..., dict[str, Any]] = _feature_extractor_identity,
) -> dict[str, dict[str, Any]]:
    """Evaluate several frozen V2 candidates without repeating DINO inference.

    This is an in-process convenience API, not a combined report: every
    candidate still receives its own external, immutable report.  Source image
    bytes are re-verified for every report, while feature values are cached
    only in memory and keyed by the full extractor identity plus source digest.
    A pinned identity factory rejects a changing model/source tree before any
    cached model or feature can be reused.
    """

    if device != "cpu":
        raise SuccessorV2EvaluatorError("successor V2 development supports CPU only")
    if not isinstance(candidate_configurations, list) or not candidate_configurations:
        raise SuccessorV2EvaluatorError("candidate_configurations must be a non-empty list")
    configurations = [validate_candidate_configuration(configuration) for configuration in candidate_configurations]
    identifiers = [str(configuration["id"]) for configuration in configurations]
    if len(identifiers) != len(set(identifiers)):
        raise SuccessorV2EvaluatorError("candidate_configurations contains a duplicate candidate id")
    if not isinstance(output_paths, dict) or set(output_paths) != set(identifiers):
        raise SuccessorV2EvaluatorError("output_paths must map exactly one external output to every candidate id")
    canonical_output_paths: dict[str, Path] = {}
    output_owner_by_key: dict[str, str] = {}
    for identifier in identifiers:
        canonical_path = _canonical_external_output_slot(output_paths[identifier], repository_root=repository_root)
        key = os.path.normcase(str(canonical_path))
        owner = output_owner_by_key.get(key)
        if owner is not None:
            raise SuccessorV2EvaluatorError(
                f"output_paths maps candidates {owner!r} and {identifier!r} to the same canonical report slot"
            )
        output_owner_by_key[key] = identifier
        canonical_output_paths[identifier] = canonical_path
    needs_augmentation = any(
        configuration["prototypeInputPolicy"] == RAW_FIT_PLUS_AUGMENTATION_R3 for configuration in configurations
    )
    if needs_augmentation != (augmentation_manifest_path is not None and recipe_path is not None):
        raise SuccessorV2EvaluatorError(
            "R3 candidates require both an augmentation manifest and recipe; a raw-only batch accepts neither"
        )
    # Prepare every output directory before evaluating or writing the first
    # candidate report.  A bad later slot cannot leave an earlier report as a
    # misleading partial multi-candidate batch.
    for identifier in identifiers:
        canonical_output_paths[identifier] = _require_external_output(
            canonical_output_paths[identifier], repository_root=repository_root
        )
    pinned_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    if not isinstance(pinned_identity, dict):
        raise SuccessorV2EvaluatorError("feature extractor identity factory must return an object")

    def pinned_identity_factory(*, model_repo: Path, model_weights: Path, device: str) -> dict[str, Any]:
        current = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
        if current != pinned_identity:
            raise SuccessorV2EvaluatorError("feature extractor inputs changed during multi-candidate evaluation")
        return dict(pinned_identity)

    shared_embedder: Any | None = None

    def shared_embedder_factory(*, model_repo: Path, model_weights: Path, device: str) -> Any:
        nonlocal shared_embedder
        if shared_embedder is None:
            shared_embedder = embedder_factory(model_repo=model_repo, model_weights=model_weights, device=device)
        return shared_embedder

    feature_cache: dict[tuple[str, str], object] = {}
    reports: dict[str, dict[str, Any]] = {}
    for configuration in configurations:
        uses_augmentation = configuration["prototypeInputPolicy"] == RAW_FIT_PLUS_AUGMENTATION_R3
        identifier = str(configuration["id"])
        reports[identifier] = run_successor_v2_development_evaluation(
            parent_holdout_path,
            parent_selection_contract_path,
            plan_path,
            envelope_path,
            canonical_output_paths[identifier],
            source_root=source_root,
            model_repo=model_repo,
            model_weights=model_weights,
            device=device,
            candidate_configuration=configuration,
            augmentation_manifest_path=augmentation_manifest_path if uses_augmentation else None,
            recipe_path=recipe_path if uses_augmentation else None,
            repository_root=repository_root,
            embedder_factory=shared_embedder_factory,
            identity_factory=pinned_identity_factory,
            in_memory_feature_cache=feature_cache,
        )
    return reports


# A short alias makes the module convenient to use without accidentally
# importing the frozen V1 evaluator's similarly named entry point.
run_development_evaluation = run_successor_v2_development_evaluation
