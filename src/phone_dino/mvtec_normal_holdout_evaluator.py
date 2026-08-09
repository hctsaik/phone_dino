"""Standalone normal-only evaluator for the fresh MVTec holdout.

The evaluator is intentionally separate from the V3--V5 iteration runner. It
uses the same local DINOv2 preprocessing/model API, but has its own schema,
feature provenance, and input boundary: FIT parents plus FIT-only derivatives
build prototypes, while raw threshold-tuning images alone set thresholds.
Selection and confirmation partitions are not accepted by this module's
development entry point.
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
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from PIL import Image, ImageOps, __version__ as PILLOW_VERSION

from phone_dino.engines import LocalDinoV2Adapter
from phone_dino.mvtec_fresh_fit_augmentation import (
    FreshFitAugmentationError,
    load_validated_fresh_fit_augmentations,
)
from phone_dino.mvtec_normal_holdout import (
    NORMAL_HOLDOUT_SCHEMA,
    NormalHoldoutError,
    canonical_json_sha256,
    load_evaluation_safe_normal_holdout_inputs,
)
from phone_dino.production import DINO_INPUT_SIZE, DINO_RESIZE_SHORT_EDGE


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_REPORT_SCHEMA = "phone-dino.mvtec-ad-normal-holdout-development-report/1.0"
DEVELOPMENT_REPORT_PURPOSE = "OFFLINE_MVTEC_NORMAL_HOLDOUT_DEVELOPMENT"
DEVELOPMENT_PHASE = "DEVELOPMENT"
NORMAL_HOLDOUT_BLIND_POLICY = "NO_BLIND_OR_ANOMALY_DATA"
FEATURE_EXTRACTOR_SCHEMA = "phone-dino.mvtec-ad-normal-holdout-feature-extractor/1.0"
PREPROCESSING_ID = "DINOV2_RESIZE_SHORT_EDGE_256_CENTER_CROP_224_IMAGENET_NORMALIZE_V1"
ALGORITHM_ID = "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1"
PROTOTYPE_SELECTION = "DETERMINISTIC_EVENLY_SPACED_PATCH_SUBSET_AFTER_CASEID_SORT"
DEVELOPMENT_REPORT_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "phase",
    "blindPolicy",
    "inputPolicy",
    "holdoutManifestFileSha256",
    "holdoutManifestDeclaredSha256",
    "developmentIdentitySha256",
    "augmentationManifestFileSha256",
    "augmentationManifestDeclaredSha256",
    "fitParentIdentitySha256",
    "augmentationRecipeFileSha256",
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
    "isAugmentation",
    "variantId",
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
    "calibrationInputCount",
    "calibrationInputPartitions",
    "calibrationInputKinds",
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


class NormalHoldoutEvaluatorError(ValueError):
    """Raised when a fresh normal-holdout development run is unsafe."""


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


def _require_exact_fields(document: dict[str, Any], *, name: str, required: set[str]) -> None:
    missing = required.difference(document)
    if missing:
        raise NormalHoldoutEvaluatorError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    unknown = set(document).difference(required)
    if unknown:
        raise NormalHoldoutEvaluatorError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")


def _require_string(document: dict[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value.strip():
        raise NormalHoldoutEvaluatorError(f"{name} must be a non-empty string")
    return value


def _require_sha256(document: dict[str, Any], name: str) -> str:
    value = _require_string(document, name)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise NormalHoldoutEvaluatorError(f"{name} must be a SHA-256 digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise NormalHoldoutEvaluatorError(f"{name} must be a SHA-256 digest") from error
    return value


def _require_positive_int(value: object, *, name: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > maximum:
        raise NormalHoldoutEvaluatorError(f"{name} must be a positive integer no larger than {maximum}")
    return value


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
                raise NormalHoldoutEvaluatorError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _require_external_output(path: Path, *, repository_root: Path) -> None:
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise NormalHoldoutEvaluatorError("development report output must stay outside the Git working tree")
    if path.exists():
        raise NormalHoldoutEvaluatorError("development report output already exists; choose a fresh immutable path")
    _reject_links_on_existing_path(path.parent, description="development report output")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_links_on_existing_path(path.parent, description="development report output")


def _immutable_file_sha256(path: Path, *, description: str) -> str:
    if not path.is_file() or _is_link_or_reparse_point(path):
        raise NormalHoldoutEvaluatorError(f"{description} must be a regular non-link file")
    return sha256_file(path)


def sha256_directory(root: Path) -> str:
    """Hash the local model source tree while rejecting links/reparse points."""

    if not root.is_dir() or _is_link_or_reparse_point(root):
        raise NormalHoldoutEvaluatorError("model repository must be a non-link directory")
    entries: list[tuple[str, Path]] = []

    def visit(directory: Path) -> None:
        try:
            children = list(os.scandir(directory))
        except OSError as error:
            raise NormalHoldoutEvaluatorError("unable to enumerate model repository") from error
        for child in children:
            candidate = Path(child.path)
            relative = candidate.relative_to(root)
            if _is_link_or_reparse_point(candidate):
                raise NormalHoldoutEvaluatorError(
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
                raise NormalHoldoutEvaluatorError("model repository contains an unsupported entry")

    visit(root)
    if not any(kind == "F" for kind, _ in entries):
        raise NormalHoldoutEvaluatorError("model repository has no files to hash")
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


def validate_candidate_configuration(value: object) -> dict[str, Any]:
    """Validate every score-affecting patch-kNN candidate knob."""

    if not isinstance(value, dict):
        raise NormalHoldoutEvaluatorError("candidateConfiguration must be an object")
    _require_exact_fields(value, name="candidateConfiguration", required=CANDIDATE_CONFIGURATION_FIELDS)
    identifier = _require_string(value, "id")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in identifier):
        raise NormalHoldoutEvaluatorError("candidateConfiguration.id must use lowercase letters, digits, or hyphens")
    if value.get("algorithmId") != ALGORITHM_ID or value.get("memoryBankSelection") != PROTOTYPE_SELECTION:
        raise NormalHoldoutEvaluatorError("candidateConfiguration algorithm is unsupported")
    maximum = _require_positive_int(value.get("maxPrototypePatches"), name="maxPrototypePatches", maximum=65_536)
    top_k = _require_positive_int(value.get("topKMostAnomalousPatches"), name="topKMostAnomalousPatches", maximum=256)
    if top_k > 256:
        raise NormalHoldoutEvaluatorError("topKMostAnomalousPatches exceeds the DINO patch grid")
    _require_positive_int(value.get("prototypeBlockSize"), name="prototypeBlockSize", maximum=65_536)
    _require_positive_int(value.get("batchSize"), name="batchSize", maximum=8)
    if maximum < top_k:
        raise NormalHoldoutEvaluatorError("maxPrototypePatches must be at least topKMostAnomalousPatches")
    return dict(value)


def _feature_extractor_identity(*, model_repo: Path, model_weights: Path, device: str) -> dict[str, Any]:
    import numpy as np
    import torch
    import torchvision

    mkldnn_backend = getattr(torch.backends, "mkldnn", None)
    source_files = {
        "evaluatorModuleSha256": Path(__file__),
        "productionModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "production.py",
        "enginesModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "engines.py",
    }
    return {
        "schemaVersion": FEATURE_EXTRACTOR_SCHEMA,
        "modelWeightsSha256": _immutable_file_sha256(model_weights, description="model weights"),
        "modelRepositorySha256": sha256_directory(model_repo),
        "preprocessingId": PREPROCESSING_ID,
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


class FreshHoldoutBatchEmbedder:
    """Batch-only benchmark wrapper around the production DINO preprocessing/model."""

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
            model = self._model
            batch = torch.cat(tensors, dim=0)
            with torch.inference_mode():
                patches = model.forward_features(batch)["x_norm_patchtokens"].detach().cpu().numpy()
        return [np.asarray(value, dtype=np.float32) for value in patches]


def _load_rgb_and_verify(record: dict[str, Any]) -> Image.Image:
    source_path = record.get("imagePath")
    if not isinstance(source_path, Path) or not source_path.is_file():
        raise NormalHoldoutEvaluatorError("feature input is not a regular file")
    _reject_links_on_existing_path(source_path, description="feature input")
    if sha256_file(source_path) != record.get("sourceSha256"):
        raise NormalHoldoutEvaluatorError("feature input digest does not match its frozen record")
    try:
        with Image.open(source_path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, SyntaxError, ValueError) as error:
        raise NormalHoldoutEvaluatorError("unable to decode a feature input image") from error
    _reject_links_on_existing_path(source_path, description="feature input")
    if sha256_file(source_path) != record.get("sourceSha256"):
        raise NormalHoldoutEvaluatorError("feature input changed while it was decoded")
    return image


def _extract_patch_features(
    records: list[dict[str, Any]],
    *,
    embedder: Any,
    batch_size: int,
    timings: dict[str, float],
) -> dict[str, object]:
    if not records:
        raise NormalHoldoutEvaluatorError("patch extraction requires at least one record")
    features: dict[str, object] = {}
    images: list[tuple[dict[str, Any], Image.Image]] = []
    verification_started = time.perf_counter()
    for record in records:
        case_id = str(record.get("caseId"))
        if case_id in features or any(str(existing[0].get("caseId")) == case_id for existing in images):
            raise NormalHoldoutEvaluatorError("feature input caseId is duplicated")
        images.append((record, _load_rgb_and_verify(record)))
    timings["inputVerificationSeconds"] += time.perf_counter() - verification_started
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inference_started = time.perf_counter()
        extracted = embedder.extract_patches([image for _, image in batch])
        timings["featureInferenceSeconds"] += time.perf_counter() - inference_started
        if len(extracted) != len(batch):
            raise NormalHoldoutEvaluatorError("DINO patch extraction returned an unexpected batch size")
        for (record, _), values in zip(batch, extracted, strict=True):
            features[str(record["caseId"])] = values
    return features


def _normalized_rows(values: object) -> object:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise NormalHoldoutEvaluatorError("patch feature matrix is empty or invalid")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.all(np.isfinite(array)) or np.any(norms <= 1e-12):
        raise NormalHoldoutEvaluatorError("patch feature matrix contains non-finite or zero-norm values")
    return array / norms


def deterministic_prototype_indices(total: int, maximum: int) -> list[int]:
    if total <= 0 or maximum <= 0:
        raise NormalHoldoutEvaluatorError("prototype count must be positive")
    if total <= maximum:
        return list(range(total))
    return [((2 * index + 1) * total) // (2 * maximum) for index in range(maximum)]


def patch_knn_scores_blocked(
    query_patches: object,
    prototype_patches: object,
    *,
    top_k: int,
    prototype_block_size: int,
) -> list[dict[str, float]]:
    import numpy as np

    query = np.asarray(query_patches, dtype=np.float32)
    prototypes = np.asarray(prototype_patches, dtype=np.float32)
    if query.ndim != 3 or prototypes.ndim != 2 or query.shape[2] != prototypes.shape[1]:
        raise NormalHoldoutEvaluatorError("patch query/prototype dimensions do not match")
    if query.shape[0] == 0 or query.shape[1] == 0 or prototypes.shape[0] == 0:
        raise NormalHoldoutEvaluatorError("patch query/prototype matrix is empty")
    if top_k <= 0 or top_k > query.shape[1] or prototype_block_size <= 0:
        raise NormalHoldoutEvaluatorError("patch scoring parameters are invalid")
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
        "isAugmentation": bool(record.get("isAugmentation", False)),
        "variantId": record.get("variantId"),
        "parentCaseId": record.get("parentCaseId"),
        "parentSourceSha256": record.get("parentSourceSha256"),
        "augmentationManifestSha256": record.get("augmentationManifestSha256"),
    }


def _feature_input_identity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_feature_input_identity_record(record) for record in sorted(records, key=lambda item: str(item["caseId"]))]


def _adapt_original_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("partition") not in {"FIT", "THRESHOLD_TUNING"}:
        raise NormalHoldoutEvaluatorError("development evaluator received an unsupported original partition")
    if record.get("kind") != "NOMINAL" or record.get("defect") != "good":
        raise NormalHoldoutEvaluatorError("development evaluator requires nominal good originals")
    return {
        **record,
        "isAugmentation": False,
        "variantId": None,
        "parentCaseId": None,
        "parentSourceSha256": None,
        "augmentationManifestSha256": None,
    }


def _adapt_augmentation_record(
    record: dict[str, Any],
    *,
    augmentation_root: Path,
    augmentation_manifest_sha256: str,
) -> dict[str, Any]:
    if (
        record.get("parentPartition") != "FIT"
        or record.get("kind") != "NOMINAL"
        or record.get("defect") != "good"
    ):
        raise NormalHoldoutEvaluatorError("fresh FIT augmentation record has an unsafe input scope")
    relative_path = record.get("relativePath")
    if not isinstance(relative_path, str) or not relative_path:
        raise NormalHoldoutEvaluatorError("fresh FIT augmentation record has no safe output path")
    image_path = augmentation_root / Path(relative_path)
    if not image_path.is_file() or not _is_under(augmentation_root, image_path) or _is_link_or_reparse_point(image_path):
        raise NormalHoldoutEvaluatorError("fresh FIT augmentation output is missing or escapes its package")
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
        "variantId": record["variantId"],
        "parentCaseId": record["parentCaseId"],
        "parentSourceSha256": record["parentSourceSha256"],
        "augmentationManifestSha256": augmentation_manifest_sha256,
    }


def _development_inputs(
    holdout_manifest_path: Path,
    *,
    source_root: Path,
    augmentation_manifest_path: Path,
    recipe_path: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, Any], str]:
    """Load only FIT/raw-tuning and validated FIT children for development."""

    try:
        holdout, holdout_file_sha256, original_records = load_evaluation_safe_normal_holdout_inputs(
            holdout_manifest_path,
            source_root=source_root,
            partitions={"FIT", "THRESHOLD_TUNING"},
        )
    except NormalHoldoutError as error:
        raise NormalHoldoutEvaluatorError(str(error)) from error
    if holdout.get("schemaVersion") != NORMAL_HOLDOUT_SCHEMA:
        raise NormalHoldoutEvaluatorError("development evaluator requires a normal-holdout manifest")
    try:
        augmentation, augmentation_records = load_validated_fresh_fit_augmentations(
            augmentation_manifest_path,
            holdout_manifest_path,
            source_root=source_root,
            recipe_path=recipe_path,
        )
    except FreshFitAugmentationError as error:
        raise NormalHoldoutEvaluatorError(str(error)) from error
    augmentation_file_sha256 = sha256_file(augmentation_manifest_path)
    if (
        augmentation.get("holdoutManifestFileSha256") != holdout_file_sha256
        or augmentation.get("holdoutManifestDeclaredSha256") != holdout.get("normalHoldoutManifestSha256")
        or augmentation.get("developmentIdentitySha256") != holdout.get("developmentIdentitySha256")
    ):
        raise NormalHoldoutEvaluatorError("fresh FIT augmentation is not bound to this development holdout")
    originals = [_adapt_original_record(record) for record in original_records]
    augmented = [
        _adapt_augmentation_record(
            record,
            augmentation_root=augmentation_manifest_path.parent,
            augmentation_manifest_sha256=augmentation["augmentationManifestSha256"],
        )
        for record in augmentation_records
    ]
    combined = sorted(originals + augmented, key=lambda record: str(record["caseId"]))
    case_ids = [str(record["caseId"]) for record in combined]
    if len(case_ids) != len(set(case_ids)):
        raise NormalHoldoutEvaluatorError("development feature input caseId is duplicated")
    if any(record["partition"] not in {"FIT", "THRESHOLD_TUNING"} for record in combined):
        raise NormalHoldoutEvaluatorError("development evaluator attempted to load a held-out partition")
    return holdout, holdout_file_sha256, combined, augmentation, augmentation_file_sha256


def _p95(values: list[float]) -> float:
    if not values:
        raise NormalHoldoutEvaluatorError("P95 requires at least one score")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _normal_summary(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise NormalHoldoutEvaluatorError("normal score summary requires at least one score")
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
        "calibrationInputCount": len(calibration_records),
        "calibrationInputPartitions": sorted({str(record["partition"]) for record in calibration_records}),
        "calibrationInputKinds": sorted({str(record["kind"]) for record in calibration_records}),
    }
    _require_exact_fields(evidence, name="normalOnlyEvidence", required=NORMAL_ONLY_EVIDENCE_FIELDS)
    if (
        evidence["featureInputPartitions"] != ["FIT", "THRESHOLD_TUNING"]
        or evidence["featureInputKinds"] != ["NOMINAL"]
        or evidence["calibrationInputPartitions"] != ["THRESHOLD_TUNING"]
        or evidence["calibrationInputKinds"] != ["NOMINAL"]
    ):
        raise NormalHoldoutEvaluatorError("development normal-only evidence is inconsistent")
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
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - Git availability is environment-specific
        revision = None
        worktree_clean = None
    return {
        "evaluatorModuleSha256": sha256_file(Path(__file__)),
        "evaluatorEntrypointSha256": sha256_file(
            repository_root / "tools" / "run_mvtec_ad_normal_holdout_development.py"
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


def run_development_evaluation(
    holdout_manifest_path: Path,
    augmentation_manifest_path: Path,
    recipe_path: Path,
    output_path: Path,
    *,
    source_root: Path,
    model_repo: Path,
    model_weights: Path,
    device: str,
    candidate_configuration: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
    embedder_factory: Callable[..., Any] = FreshHoldoutBatchEmbedder,
    identity_factory: Callable[..., dict[str, Any]] = _feature_extractor_identity,
) -> dict[str, Any]:
    """Run a development-only raw-tuning observation for one frozen candidate.

    This function deliberately has no selection/confirmation partition argument.
    It cannot consume those partitions, and it writes no candidate lock or
    production decision.
    """

    configuration = validate_candidate_configuration(candidate_configuration)
    if device != "cpu":
        raise NormalHoldoutEvaluatorError("fresh normal-holdout development supports CPU only")
    _require_external_output(output_path, repository_root=repository_root)
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
    holdout, holdout_file_sha256, feature_records, augmentation, augmentation_file_sha256 = _development_inputs(
        holdout_manifest_path,
        source_root=source_root,
        augmentation_manifest_path=augmentation_manifest_path,
        recipe_path=recipe_path,
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
        raise NormalHoldoutEvaluatorError("feature extractor inputs changed while DINO loaded")

    categories = sorted({str(record["category"]) for record in feature_records})
    if not categories:
        raise NormalHoldoutEvaluatorError("development evaluator has no categories")
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
        if not fit or not tuning:
            raise NormalHoldoutEvaluatorError("each development category needs FIT and raw threshold-tuning inputs")
        if any(record["isAugmentation"] for record in tuning):
            raise NormalHoldoutEvaluatorError("threshold tuning must remain raw originals")
        scoring_started = time.perf_counter()
        features = _extract_patch_features(
            fit + tuning,
            embedder=embedder,
            batch_size=int(configuration["batchSize"]),
            timings=timings,
        )
        import numpy as np

        fit_patches = np.concatenate([np.asarray(features[str(record["caseId"])], dtype=np.float32) for record in fit], axis=0)
        prototype_indices = deterministic_prototype_indices(
            len(fit_patches),
            int(configuration["maxPrototypePatches"]),
        )
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
            raise NormalHoldoutEvaluatorError("DINO patch grid is not square")
        category_scores: list[float] = []
        for record, component in zip(tuning, components, strict=True):
            score = float(component["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 2.0:
                raise NormalHoldoutEvaluatorError("DINO patch score is outside cosine-distance bounds")
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
            _require_exact_fields(score_record, name="development calibration score", required=CALIBRATION_SCORE_FIELDS)
            calibration_scores.append(score_record)
        median, p95, maximum = _normal_summary(category_scores)
        thresholds[category] = maximum
        category_report = {
            "fitOriginalCount": sum(not record["isAugmentation"] for record in fit),
            "fitAugmentedCount": sum(record["isAugmentation"] for record in fit),
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
        _require_exact_fields(category_report, name="development category report", required=CATEGORY_REPORT_FIELDS)
        category_reports[category] = category_report

    provenance_started = time.perf_counter()
    completed_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    timings["provenanceSeconds"] += time.perf_counter() - provenance_started
    if completed_identity != extractor_identity:
        raise NormalHoldoutEvaluatorError("feature extractor inputs changed while DINO features were extracted")
    feature_inputs = _feature_input_identity(feature_records)
    calibration_input_records = [record for record in feature_records if record["partition"] == "THRESHOLD_TUNING"]
    calibration_inputs = _feature_input_identity(calibration_input_records)
    calibration_scores.sort(key=lambda record: str(record["caseId"]))
    timings["totalElapsedSeconds"] = time.perf_counter() - started
    report: dict[str, Any] = {
        "schemaVersion": DEVELOPMENT_REPORT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": DEVELOPMENT_REPORT_PURPOSE,
        "phase": DEVELOPMENT_PHASE,
        "blindPolicy": NORMAL_HOLDOUT_BLIND_POLICY,
        "inputPolicy": "FIT_PLUS_RAW_THRESHOLD_TUNING_ONLY",
        "holdoutManifestFileSha256": holdout_file_sha256,
        "holdoutManifestDeclaredSha256": holdout["normalHoldoutManifestSha256"],
        "developmentIdentitySha256": holdout["developmentIdentitySha256"],
        "augmentationManifestFileSha256": augmentation_file_sha256,
        "augmentationManifestDeclaredSha256": augmentation["augmentationManifestSha256"],
        "fitParentIdentitySha256": augmentation["fitParentIdentitySha256"],
        "augmentationRecipeFileSha256": augmentation["recipeFileSha256"],
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
        "normalOnlyEvidence": _build_normal_only_evidence(feature_records, calibration_input_records),
        "execution": _execution_metadata(timings, repository_root=repository_root),
    }
    report["developmentReportSha256"] = _document_digest(report, "developmentReportSha256")
    _require_exact_fields(report, name="development report", required=DEVELOPMENT_REPORT_FIELDS)
    try:
        with output_path.open("xb") as stream:
            stream.write((json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    except OSError as error:
        raise NormalHoldoutEvaluatorError("unable to write immutable development report") from error
    return report
