"""Run a reproducible, offline MVTec AD research iteration.

This tool deliberately keeps model selection separate from the frozen blind
set.  Camera augmentation is accepted only for normal FIT/tuning inputs;
blind inputs remain original and are reported only after calibration is fixed.
It never calls the PhoneDINO service or changes its runtime/artifact contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, __version__ as PILLOW_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.engines import LocalDinoV2Adapter
from phone_dino.mvtec_research import (
    MvtecResearchError,
    canonical_json_sha256,
    load_validated_normal_augmentations,
    sha256_file,
)
from phone_dino.production import DINO_INPUT_SIZE, DINO_RESIZE_SHORT_EDGE, DinoV2Embedder


ITERATION_SCHEMA_VERSION = "phone-dino.mvtec-ad-iteration-report/1.3"
FEATURE_CACHE_SCHEMA_VERSION = "phone-dino.mvtec-ad-feature-cache/1.1"
FEATURE_CACHE_ENTRY_SCHEMA_VERSION = "phone-dino.mvtec-ad-feature-cache-entry/1.0"
FEATURE_EXTRACTOR_SCHEMA_VERSION = "phone-dino.mvtec-ad-feature-extractor/1.0"
PREPROCESSING_ID = "DINOV2_RESIZE_SHORT_EDGE_256_CENTER_CROP_224_IMAGENET_NORMALIZE_V1"
FEATURE_EXTRACTOR_SOURCE_FILES = {
    "iterationToolSha256": Path(__file__),
    "productionModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "production.py",
    "enginesModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "engines.py",
    "mvtecResearchModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "mvtec_research.py",
}
CACHE_IGNORED_DIRECTORY_NAMES = frozenset({".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"})
FEATURE_SHAPES = {"global": (384,), "patch": (256, 384)}
FEATURE_CACHE_METADATA_FIELDS = {
    "schemaVersion",
    "key",
    "featureKind",
    "sourceSha256",
    "augmentationRecipeSha256",
    "featureExtractor",
    "featureExtractorIdentitySha256",
    "shape",
    "dtype",
    "dataFileName",
    "arrayFileSha256",
}


class IterationError(ValueError):
    """Raised when an offline MVTec research iteration is unsafe or invalid."""


def _safe_relative_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise IterationError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise IterationError(f"{name} must be a safe relative path")
    return path


def _require_string(document: dict[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value:
        raise IterationError(f"{name} must be a non-empty string")
    return value


def _require_sha256(document: dict[str, Any], name: str) -> str:
    value = _require_string(document, name)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise IterationError(f"{name} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise IterationError(f"{name} must be a sha256 digest") from error
    return value


def _is_under(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IterationError(f"Unable to read {description}: {path}") from error
    if not isinstance(document, dict):
        raise IterationError(f"{description} must be a JSON object")
    return document


def _is_link_or_reparse_point(path: Path) -> bool:
    """Reject POSIX links and Windows junction/reparse entries without resolving them."""

    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise IterationError(f"unable to stat immutable input: {path}") from error
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)


def _immutable_file_sha256(path: Path, *, description: str) -> str:
    if not path.is_file() or _is_link_or_reparse_point(path):
        raise IterationError(f"{description} must be a regular non-link file: {path}")
    return sha256_file(path)


def sha256_directory(root: Path) -> str:
    """Return a stable content digest for a local model source tree.

    Git metadata and interpreter caches are intentionally excluded: neither is
    executable model source. Symlinks are rejected so a model repository cannot
    silently acquire code outside its declared root.
    """

    if not root.is_dir() or _is_link_or_reparse_point(root):
        raise IterationError(f"model repository must be a non-symlink directory: {root}")
    entries: list[tuple[str, Path]] = []

    def visit(directory: Path) -> None:
        try:
            children = list(os.scandir(directory))
        except OSError as error:
            raise IterationError(f"unable to enumerate model repository: {directory}") from error
        for child in children:
            candidate = Path(child.path)
            relative = candidate.relative_to(root)
            if _is_link_or_reparse_point(candidate):
                raise IterationError(f"model repository contains a link or reparse point: {relative.as_posix()}")
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
                raise IterationError(f"model repository contains an unsupported entry: {relative.as_posix()}")

    visit(root)
    if not any(kind == "F" for kind, _ in entries):
        raise IterationError("model repository has no source files to hash")
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


def feature_extractor_identity(
    *,
    model_repo: Path,
    model_weights_sha256: str,
    device: str,
) -> dict[str, Any]:
    """Bind every local implementation and dependency that produces features."""

    import numpy as np
    import torch
    import torchvision

    mkldnn_backend = getattr(torch.backends, "mkldnn", None)
    source_digests = {
        name: _immutable_file_sha256(path, description="feature extractor source")
        for name, path in FEATURE_EXTRACTOR_SOURCE_FILES.items()
    }
    return {
        "schemaVersion": FEATURE_EXTRACTOR_SCHEMA_VERSION,
        "modelWeightsSha256": model_weights_sha256,
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
        **source_digests,
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


def load_frozen_records(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the immutable MVTec source manifest without accepting unknown roles."""

    manifest = _read_json(manifest_path, description="frozen MVTec manifest")
    if manifest.get("schemaVersion") != "phone-dino.mvtec-ad-smoke/1.0" or manifest.get("authoritative") is not False:
        raise IterationError("expected a non-authoritative phone-dino.mvtec-ad-smoke/1.0 manifest")
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise IterationError("frozen MVTec manifest has no records")
    root = manifest_path.parent
    records: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise IterationError("frozen MVTec record is not an object")
        record = dict(raw)
        case_id = _require_string(record, "caseId")
        if case_id in seen_case_ids:
            raise IterationError("frozen MVTec caseId is duplicated")
        seen_case_ids.add(case_id)
        role = _require_string(record, "role")
        kind = _require_string(record, "kind")
        if role not in {"FIT", "THRESHOLD_TUNING", "BLIND"} or kind not in {"NOMINAL", "ANOMALY"}:
            raise IterationError("frozen MVTec role or kind is unsupported")
        if role in {"FIT", "THRESHOLD_TUNING"} and kind != "NOMINAL":
            raise IterationError("FIT and THRESHOLD_TUNING records must be nominal")
        record["category"] = _require_string(record, "category")
        record["sourceSha256"] = _require_sha256(record, "sourceSha256")
        relative_path = _safe_relative_path(record.get("relativePath"), name="relativePath")
        image_path = root / relative_path
        if not _is_under(root, image_path) or not image_path.is_file():
            raise IterationError("frozen MVTec image is missing or escapes the dataset root")
        record["imagePath"] = image_path
        record["isAugmentation"] = False
        if kind == "ANOMALY":
            mask_path = _safe_relative_path(record.get("maskRelativePath"), name="maskRelativePath")
            record["maskSha256"] = _require_sha256(record, "maskSha256")
            resolved_mask = root / mask_path
            if not _is_under(root, resolved_mask) or not resolved_mask.is_file():
                raise IterationError("frozen MVTec mask is missing or escapes the dataset root")
            record["maskPath"] = resolved_mask
        records.append(record)
    return manifest, records


def input_sort_key(record: dict[str, Any]) -> tuple[str, int, str]:
    """Stabilize generated and original parents before prototype selection."""

    parent_case_id = str(record.get("parentCaseId", record["caseId"]))
    variant_id = int(record.get("variantId", 0))
    return parent_case_id, variant_id, str(record["caseId"])


def build_iteration_inputs(
    manifest_path: Path,
    augmentation_manifest_path: Path | None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build per-category inputs, retaining original-only blind membership."""

    manifest, originals = load_frozen_records(manifest_path)
    augmented: list[dict[str, Any]] = []
    augmentation_details: dict[str, Any] = {"state": "NONE", "blindAugmentedCount": 0}
    if augmentation_manifest_path is not None:
        try:
            augmentation_document, augmentation_records = load_validated_normal_augmentations(
                augmentation_manifest_path, manifest_path
            )
        except MvtecResearchError as error:
            raise IterationError(str(error)) from error
        for raw in augmentation_records:
            record = dict(raw)
            if record.get("role") not in {"FIT", "THRESHOLD_TUNING"} or record.get("kind") != "NOMINAL":
                raise IterationError("augmentation manifest attempted to add a non-normal blind/development input")
            record["imagePath"] = augmentation_manifest_path.parent / _safe_relative_path(
                record.get("relativePath"), name="augmentation relativePath"
            )
            record["isAugmentation"] = True
            record["augmentationRecipeSha256"] = augmentation_document["recipeSha256"]
            augmented.append(record)
        augmentation_details = {
            "state": "NORMAL_FIT_AND_TUNING_ONLY",
            "blindPolicy": augmentation_document["blindPolicy"],
            "blindAugmentedCount": 0,
            "augmentationManifestPath": str(augmentation_manifest_path),
            "augmentationManifestSha256": augmentation_document["augmentationManifestSha256"],
            "recipeSha256": augmentation_document["recipeSha256"],
            "derivedRecordCount": len(augmented),
        }

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in originals + augmented:
        categories[str(record["category"])].append(record)
    for records in categories.values():
        records.sort(key=input_sort_key)
    return manifest, dict(categories), augmentation_details


def deterministic_prototype_indices(total: int, maximum: int) -> list[int]:
    if total <= 0:
        raise IterationError("Patch memory bank has no normal prototypes")
    if maximum <= 0:
        raise IterationError("max_prototypes must be positive")
    if total <= maximum:
        return list(range(total))
    return [((2 * index + 1) * total) // (2 * maximum) for index in range(maximum)]


def _normalized_rows(values: object) -> object:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise IterationError("feature matrix is empty or invalid")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.all(np.isfinite(array)) or np.any(norms <= 1e-12):
        raise IterationError("feature matrix contains non-finite or zero-norm values")
    return array / norms


def global_knn_scores(query_vectors: object, prototype_vectors: object) -> list[float]:
    import numpy as np

    query = _normalized_rows(query_vectors)
    prototypes = _normalized_rows(prototype_vectors)
    if query.shape[1] != prototypes.shape[1]:
        raise IterationError("global feature dimensions do not match")
    distances = np.clip(1.0 - query @ prototypes.T, 0.0, 2.0)
    return [float(value) for value in np.min(distances, axis=1)]


def patch_knn_scores_blocked(
    query_patches: object,
    prototype_patches: object,
    *,
    top_k: int,
    prototype_block_size: int,
) -> list[dict[str, Any]]:
    """Exact bounded-memory nearest-normal patch scoring for a query batch."""

    import numpy as np

    query = np.asarray(query_patches, dtype=np.float32)
    prototypes = np.asarray(prototype_patches, dtype=np.float32)
    if query.ndim != 3 or prototypes.ndim != 2 or query.shape[2] != prototypes.shape[1]:
        raise IterationError("Patch matrix dimensions do not match")
    if query.shape[0] == 0 or query.shape[1] == 0 or prototypes.shape[0] == 0:
        raise IterationError("Patch matrix is empty")
    if top_k <= 0 or top_k > query.shape[1]:
        raise IterationError("top_k must be between 1 and the query patch count")
    if prototype_block_size <= 0:
        raise IterationError("prototype_block_size must be positive")
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
            "patchDistanceGrid": grids[index],
        }
        for index in range(query.shape[0])
    ]


class FeatureCache:
    """A conservative image-feature cache for repeatable offline iterations."""

    def __init__(
        self,
        root: Path,
        *,
        feature_extractor: dict[str, Any],
        feature_extractor_identity_sha256: str,
    ):
        if _is_under(REPOSITORY_ROOT, root):
            raise IterationError("feature cache must stay outside the Git working tree")
        if canonical_json_sha256(feature_extractor) != feature_extractor_identity_sha256:
            raise IterationError("feature cache extractor identity digest does not match")
        self.root = root
        self.feature_extractor = feature_extractor
        self.feature_extractor_identity_sha256 = feature_extractor_identity_sha256
        self.root.mkdir(parents=True, exist_ok=True)
        if _is_link_or_reparse_point(self.root):
            raise IterationError("feature cache root must not be a link or reparse point")

    def key_for(self, record: dict[str, Any], feature_kind: str) -> str:
        if feature_kind not in FEATURE_SHAPES:
            raise IterationError("feature cache key has an unsupported feature kind")
        return canonical_json_sha256({
            "schemaVersion": FEATURE_CACHE_SCHEMA_VERSION,
            "featureKind": feature_kind,
            "sourceSha256": record["sourceSha256"],
            "augmentationRecipeSha256": record.get("augmentationRecipeSha256"),
            "featureExtractorIdentitySha256": self.feature_extractor_identity_sha256,
        })[7:]

    @staticmethod
    def _validate_key(key: str) -> str:
        if len(key) != 64:
            raise IterationError("feature cache key is invalid")
        try:
            int(key, 16)
        except ValueError as error:
            raise IterationError("feature cache key is invalid") from error
        return key

    def _metadata_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    @staticmethod
    def _array_file_name(key: str, array_file_sha256: str) -> str:
        return f"{key}.{array_file_sha256[7:]}.npy"

    @staticmethod
    def _validate_feature_array(values: object, *, feature_kind: str) -> object:
        import numpy as np

        if feature_kind not in FEATURE_SHAPES:
            raise IterationError("feature cache has an unsupported feature kind")
        array = np.asarray(values)
        expected_shape = FEATURE_SHAPES[feature_kind]
        if array.dtype != np.dtype(np.float32) or tuple(array.shape) != expected_shape:
            raise IterationError(f"feature cache entry has an unexpected {feature_kind} feature shape or dtype")
        if not np.all(np.isfinite(array)):
            raise IterationError("feature cache entry is non-finite")
        return np.ascontiguousarray(array)

    @staticmethod
    def _write_json_fsync(path: Path, document: dict[str, Any]) -> None:
        payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        with path.open("wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())

    def get(self, key: str, *, feature_kind: str, record: dict[str, Any]) -> object | None:
        import numpy as np

        self._validate_key(key)
        metadata_path = self._metadata_path(key)
        if not metadata_path.exists():
            return None
        if not metadata_path.is_file() or _is_link_or_reparse_point(metadata_path):
            raise IterationError("feature cache metadata must be a regular non-link file")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IterationError(f"feature cache metadata is unreadable: {metadata_path}") from error
        if not isinstance(metadata, dict) or set(metadata) != FEATURE_CACHE_METADATA_FIELDS:
            raise IterationError("feature cache metadata has an invalid schema")
        if (
            metadata.get("schemaVersion") != FEATURE_CACHE_ENTRY_SCHEMA_VERSION
            or metadata.get("key") != key
            or metadata.get("featureKind") != feature_kind
            or metadata.get("sourceSha256") != record.get("sourceSha256")
            or metadata.get("augmentationRecipeSha256") != record.get("augmentationRecipeSha256")
            or metadata.get("featureExtractor") != self.feature_extractor
            or metadata.get("featureExtractorIdentitySha256") != self.feature_extractor_identity_sha256
            or metadata.get("dtype") != "float32"
        ):
            raise IterationError("feature cache metadata does not match the requested feature identity")
        shape = metadata.get("shape")
        if not isinstance(shape, list) or any(not isinstance(value, int) or isinstance(value, bool) for value in shape):
            raise IterationError("feature cache metadata shape is invalid")
        expected_shape = list(FEATURE_SHAPES.get(feature_kind, ()))
        if shape != expected_shape:
            raise IterationError("feature cache metadata shape does not match the feature contract")
        array_digest = metadata.get("arrayFileSha256")
        if not isinstance(array_digest, str) or len(array_digest) != 71 or not array_digest.startswith("sha256:"):
            raise IterationError("feature cache metadata array digest is invalid")
        try:
            int(array_digest[7:], 16)
        except ValueError as error:
            raise IterationError("feature cache metadata array digest is invalid") from error
        expected_file_name = self._array_file_name(key, array_digest)
        if metadata.get("dataFileName") != expected_file_name:
            raise IterationError("feature cache metadata data filename is invalid")
        path = self.root / expected_file_name
        if not _is_under(self.root, path):
            raise IterationError("feature cache metadata data path escapes the cache root")
        if not path.is_file() or _is_link_or_reparse_point(path):
            raise IterationError("feature cache metadata references a missing or linked data file")
        maximum_file_size = int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize + 65536
        if path.stat().st_size > maximum_file_size:
            raise IterationError("feature cache entry exceeds the bounded feature size")
        if _immutable_file_sha256(path, description="feature cache data") != array_digest:
            raise IterationError("feature cache data digest does not match metadata")
        try:
            values = np.load(path, allow_pickle=False, max_header_size=65536)
        except (OSError, ValueError) as error:
            raise IterationError(f"feature cache entry is unreadable: {path}") from error
        array = self._validate_feature_array(values, feature_kind=feature_kind)
        return array

    def put(self, key: str, values: object, *, feature_kind: str, record: dict[str, Any]) -> None:
        import numpy as np

        self._validate_key(key)
        value = self._validate_feature_array(values, feature_kind=feature_kind)
        temporary = self.root / f".{key}.{os.getpid()}.{time.time_ns()}.tmp.npy"
        try:
            with temporary.open("wb") as output:
                np.save(output, value, allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
            array_file_sha256 = _immutable_file_sha256(temporary, description="temporary feature cache data")
            data_file_name = self._array_file_name(key, array_file_sha256)
            data_path = self.root / data_file_name
            os.replace(temporary, data_path)
            metadata = {
                "schemaVersion": FEATURE_CACHE_ENTRY_SCHEMA_VERSION,
                "key": key,
                "featureKind": feature_kind,
                "sourceSha256": record["sourceSha256"],
                "augmentationRecipeSha256": record.get("augmentationRecipeSha256"),
                "featureExtractor": self.feature_extractor,
                "featureExtractorIdentitySha256": self.feature_extractor_identity_sha256,
                "shape": list(value.shape),
                "dtype": "float32",
                "dataFileName": data_file_name,
                "arrayFileSha256": array_file_sha256,
            }
            metadata_temporary = self.root / f".{key}.{os.getpid()}.{time.time_ns()}.tmp.json"
            try:
                self._write_json_fsync(metadata_temporary, metadata)
                os.replace(metadata_temporary, self._metadata_path(key))
            finally:
                if metadata_temporary.exists():
                    metadata_temporary.unlink()
        finally:
            if temporary.exists():
                temporary.unlink()


class ResearchBatchEmbedder:
    """Batch-only wrapper that reuses the production embedder preprocessing.

    It intentionally lives in this offline tool, leaving PhoneDINO's public
    production embedding API and runtime behavior unchanged.
    """

    def __init__(self, *, model_repo: Path, model_weights: Path, device: str):
        adapter = LocalDinoV2Adapter(repository=model_repo, weights=model_weights)
        ready, reason = adapter.readiness()
        if not ready:
            raise RuntimeError(reason)
        self._embedder = DinoV2Embedder(adapter, device=device)
        self._embedder.warm_up()

    def extract(self, images: list[Image.Image], *, feature_kind: str) -> list[object]:
        import numpy as np
        import torch

        if not images:
            return []
        tensors = [self._embedder._transform(image) for image in images]  # noqa: SLF001 - benchmark-only reuse
        with self._embedder._lock:  # noqa: SLF001 - keeps the loaded model exclusive
            model = self._embedder._model  # noqa: SLF001
            if model is None:  # pragma: no cover - warm_up above establishes this invariant
                raise RuntimeError("DINO model was not loaded")
            batch = torch.cat(tensors, dim=0)
            with torch.inference_mode():
                if feature_kind == "patch":
                    values = model.forward_features(batch)["x_norm_patchtokens"].detach().cpu().numpy()
                elif feature_kind == "global":
                    values = model(batch).detach().cpu().numpy()
                else:
                    raise IterationError("unknown feature kind")
        return [np.asarray(value, dtype=np.float32) for value in values]


def _load_rgb_and_verify(record: dict[str, Any]) -> Image.Image:
    path = record["imagePath"]
    if not isinstance(path, Path) or sha256_file(path) != record["sourceSha256"]:
        raise IterationError(f"input digest mismatch: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def extract_features(
    records: list[dict[str, Any]],
    *,
    feature_kind: str,
    embedder: ResearchBatchEmbedder,
    cache: FeatureCache | None,
    batch_size: int,
    timings: dict[str, float],
    cache_counts: dict[str, int],
) -> tuple[dict[str, object], list[tuple[str, object, dict[str, Any]]]]:
    """Verify every input, then batch extract only cache misses.

    New cache values are deliberately returned to the caller rather than
    published here.  ``run`` verifies the extractor identity again after all
    inference before promoting these values into a reusable cache.
    """

    if batch_size <= 0:
        raise IterationError("batch_size must be positive")
    features: dict[str, object] = {}
    pending_cache_writes: list[tuple[str, object, dict[str, Any]]] = []
    misses: list[tuple[dict[str, Any], Image.Image, str | None]] = []
    verification_started = time.perf_counter()
    for record in records:
        case_id = str(record["caseId"])
        if case_id in features:
            raise IterationError("feature input caseId is duplicated")
        cache_key = None if cache is None else cache.key_for(record, feature_kind)
        image = _load_rgb_and_verify(record)
        cache_validation_started = time.perf_counter()
        cached = None if cache is None else cache.get(cache_key, feature_kind=feature_kind, record=record)
        timings["cacheValidationSeconds"] += time.perf_counter() - cache_validation_started
        if cached is not None:
            features[case_id] = cached
            cache_counts["hits"] += 1
        else:
            misses.append((record, image, cache_key))
            cache_counts["misses"] += 1
    timings["inputVerificationSeconds"] += time.perf_counter() - verification_started
    for offset in range(0, len(misses), batch_size):
        batch = misses[offset:offset + batch_size]
        inference_started = time.perf_counter()
        extracted = embedder.extract([image for _, image, _ in batch], feature_kind=feature_kind)
        timings["featureInferenceSeconds"] += time.perf_counter() - inference_started
        if len(extracted) != len(batch):  # pragma: no cover - defensive model contract guard
            raise RuntimeError("DINO feature batch returned an unexpected count")
        for (record, _, cache_key), values in zip(batch, extracted, strict=True):
            if cache is not None and cache_key is not None:
                pending_cache_writes.append((cache_key, values, record))
            features[str(record["caseId"])] = values
    return features, pending_cache_writes


def image_auroc(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for sorted_index in range(position, end):
            ranks[order[sorted_index]] = average_rank
        position = end
    positive_rank_sum = sum(rank for label, rank in zip(labels, ranks, strict=True) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def metric_summary(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [record["kind"] == "ANOMALY" for record in records]
    scores = [float(record["score"]) for record in records]
    nominal = [record for record in records if record["kind"] == "NOMINAL"]
    anomalous = [record for record in records if record["kind"] == "ANOMALY"]
    return {
        "blindCases": len(records),
        "blindNominalCases": len(nominal),
        "blindAnomalyCases": len(anomalous),
        "imageAuRoc": image_auroc(labels, scores),
        "thresholdFromNominalTuning": threshold,
        "nominalAboveThresholdRate": (
            sum(record["score"] > threshold for record in nominal) / len(nominal) if nominal else None
        ),
        "anomalyAboveThresholdRate": (
            sum(record["score"] > threshold for record in anomalous) / len(anomalous) if anomalous else None
        ),
    }


def normal_calibration_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize normal-only tuning behavior without inspecting blind labels."""

    scores = sorted(float(record["score"]) for record in records)
    if not scores:
        raise IterationError("normal calibration summary requires scored tuning records")
    def summary(prefix: str, selected: list[float]) -> dict[str, Any]:
        if not selected:
            return {}
        selected = sorted(selected)
        p95_index = max(0, math.ceil(len(selected) * 0.95) - 1)
        return {
            f"{prefix}Cases": len(selected),
            f"{prefix}Median": selected[len(selected) // 2],
            f"{prefix}P95": selected[p95_index],
            f"{prefix}Max": selected[-1],
        }
    result = {
        "normalCalibrationCases": len(scores),
        "normalScoreMedian": scores[len(scores) // 2],
        "normalScoreP95": scores[max(0, math.ceil(len(scores) * 0.95) - 1)],
        "normalScoreMax": scores[-1],
    }
    result |= summary("originalTuningNormalScore", [
        float(record["score"]) for record in records if not record.get("isAugmentation", False)
    ])
    result |= summary("augmentedTuningNormalScore", [
        float(record["score"]) for record in records if record.get("isAugmentation", False)
    ])
    return result


def _mask_for_dino_input(path: Path) -> object:
    """Apply DINO's exact resize/crop geometry to a binary research mask."""

    import numpy as np
    from torchvision.transforms import InterpolationMode, functional as transforms_functional

    with Image.open(path) as opened:
        mask = opened.convert("L")
    resized = transforms_functional.resize(mask, DINO_RESIZE_SHORT_EDGE, interpolation=InterpolationMode.NEAREST)
    cropped = transforms_functional.center_crop(resized, [DINO_INPUT_SIZE, DINO_INPUT_SIZE])
    return np.asarray(cropped, dtype=np.uint8) > 0


def _binary_auroc_from_counts(score_counts: dict[float, list[int]]) -> float | None:
    positives = sum(values[0] for values in score_counts.values())
    negatives = sum(values[1] for values in score_counts.values())
    if not positives or not negatives:
        return None
    negatives_below = 0
    pair_count = 0.0
    for score in sorted(score_counts):
        positive_count, negative_count = score_counts[score]
        pair_count += positive_count * (negatives_below + negative_count / 2.0)
        negatives_below += negative_count
    return pair_count / (positives * negatives)


def _aupro_at_fpr(
    score_counts: dict[float, list[int]],
    component_events: dict[float, list[tuple[int, int]]],
    component_sizes: dict[int, int],
    *,
    max_fpr: float = 0.30,
) -> float | None:
    if not component_sizes:
        return None
    total_normal_pixels = sum(values[1] for values in score_counts.values())
    if not total_normal_pixels:
        return None
    detected = {component_id: 0 for component_id in component_sizes}
    false_positive = 0
    previous_fpr = 0.0
    previous_pro = 0.0
    area = 0.0
    for score in sorted(score_counts, reverse=True):
        _, negative_count = score_counts[score]
        false_positive += negative_count
        for component_id, count in component_events.get(score, []):
            detected[component_id] += count
        current_fpr = false_positive / total_normal_pixels
        current_pro = sum(
            min(1.0, detected[component_id] / component_sizes[component_id])
            for component_id in component_sizes
        ) / len(component_sizes)
        if current_fpr <= max_fpr:
            area += (current_fpr - previous_fpr) * (current_pro + previous_pro) / 2.0
            previous_fpr, previous_pro = current_fpr, current_pro
            continue
        if current_fpr > previous_fpr:
            fraction = (max_fpr - previous_fpr) / (current_fpr - previous_fpr)
            bounded_pro = previous_pro + fraction * (current_pro - previous_pro)
            area += (max_fpr - previous_fpr) * (bounded_pro + previous_pro) / 2.0
        return area / max_fpr
    return area / max_fpr


def calculate_pixel_localization_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score frozen blind masks only after every image-level score is fixed."""

    import cv2
    import numpy as np

    score_counts: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    component_events: dict[float, list[tuple[int, int]]] = defaultdict(list)
    component_sizes: dict[int, int] = {}
    component_offset = 0
    mask_cases = 0
    for record in records:
        localization = record.get("localization")
        if not isinstance(localization, dict):
            raise IterationError("patch localization is missing from a blind record")
        grid = np.asarray(localization["patchDistanceGrid"], dtype=np.float32)
        if grid.ndim != 2 or DINO_INPUT_SIZE % grid.shape[0] or DINO_INPUT_SIZE % grid.shape[1]:
            raise IterationError("patch localization grid does not align to the DINO input")
        patch_height = DINO_INPUT_SIZE // grid.shape[0]
        patch_width = DINO_INPUT_SIZE // grid.shape[1]
        if record["kind"] == "ANOMALY":
            mask_path = record.get("maskPath")
            if not isinstance(mask_path, Path) or sha256_file(mask_path) != record.get("maskSha256"):
                raise IterationError("frozen MVTec mask digest mismatch")
            mask = _mask_for_dino_input(mask_path)
            connected_count, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            for component_id in range(1, connected_count):
                size = int(np.sum(components == component_id))
                if size:
                    component_sizes[component_offset + component_id] = size
            mask_cases += 1
        else:
            mask = np.zeros((DINO_INPUT_SIZE, DINO_INPUT_SIZE), dtype=bool)
            components = np.zeros_like(mask, dtype=np.int32)
            connected_count = 1
        for row in range(grid.shape[0]):
            for column in range(grid.shape[1]):
                score = float(grid[row, column])
                top, left = row * patch_height, column * patch_width
                patch_mask = mask[top:top + patch_height, left:left + patch_width]
                positive_count = int(np.sum(patch_mask))
                pixel_count = patch_mask.size
                score_counts[score][0] += positive_count
                score_counts[score][1] += pixel_count - positive_count
                patch_components = components[top:top + patch_height, left:left + patch_width]
                component_ids, counts = np.unique(patch_components[patch_components > 0], return_counts=True)
                for component_id, count in zip(component_ids, counts, strict=True):
                    component_events[score].append((component_offset + int(component_id), int(count)))
        component_offset += connected_count
    return {
        "state": "AVAILABLE" if mask_cases and component_sizes else "UNAVAILABLE",
        "generationMethod": "FROZEN_BLIND_MASK_POST_HOC_RESEARCH_ONLY",
        "dinoInputWidth": DINO_INPUT_SIZE,
        "dinoInputHeight": DINO_INPUT_SIZE,
        "maskCaseCount": mask_cases,
        "connectedMaskRegions": len(component_sizes),
        "pixelAuRoc": _binary_auroc_from_counts(score_counts),
        "auproAt30Fpr": _aupro_at_fpr(score_counts, component_events, component_sizes),
        "disclaimer": "MVTec mask localization research only; it is not a PhoneDINO inspection ROI, defect proof, or production qualification.",
    }


def _score_record(record: dict[str, Any], components: dict[str, Any] | None = None) -> dict[str, Any]:
    score_record = {
        "caseId": record["caseId"],
        "category": record["category"],
        "role": record["role"],
        "kind": record["kind"],
        "defect": record.get("defect"),
        "sourceSha256": record["sourceSha256"],
        "isAugmentation": bool(record.get("isAugmentation", False)),
        "score": float(record["score"]),
    }
    if record.get("isAugmentation", False):
        score_record["parentCaseId"] = record.get("parentCaseId")
        score_record["parentSourceSha256"] = record.get("parentSourceSha256")
        score_record["augmentationRecipeSha256"] = record.get("augmentationRecipeSha256")
    if components is not None:
        score_record |= {
            "maxPatchDistance": float(components["maxPatchDistance"]),
            "meanNearestPatchDistance": float(components["meanNearestPatchDistance"]),
            "localization": {
                "state": "AVAILABLE",
                "generationMethod": "PATCH_NEAREST_NORMAL_DISTANCE_GRID",
                "gridHeight": int(components["patchDistanceGrid"].shape[0]),
                "gridWidth": int(components["patchDistanceGrid"].shape[1]),
                "dinoInputWidth": DINO_INPUT_SIZE,
                "dinoInputHeight": DINO_INPUT_SIZE,
                "patchDistanceGrid": [[float(value) for value in row] for row in components["patchDistanceGrid"]],
                "disclaimer": "Difference localization is research evidence, not defect proof.",
            },
        }
    return score_record


def _categorize(
    records: list[dict[str, Any]], *, normal_only: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fit = [record for record in records if record["role"] == "FIT" and record["kind"] == "NOMINAL"]
    tuning = [record for record in records if record["role"] == "THRESHOLD_TUNING" and record["kind"] == "NOMINAL"]
    blind = [record for record in records if record["role"] == "BLIND" and not record["isAugmentation"]]
    if not fit or not tuning or (not normal_only and not blind):
        raise IterationError("each category needs normal FIT/tuning and original blind inputs")
    return sorted(fit, key=input_sort_key), sorted(tuning, key=input_sort_key), ([] if normal_only else sorted(blind, key=input_sort_key))


def input_identity_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return the score-free input membership fields bound by a research report."""

    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "role": record["role"],
        "kind": record["kind"],
        "sourceSha256": record["sourceSha256"],
        "isAugmentation": bool(record.get("isAugmentation", False)),
        "parentCaseId": record.get("parentCaseId"),
        "parentSourceSha256": record.get("parentSourceSha256"),
        "augmentationRecipeSha256": record.get("augmentationRecipeSha256"),
    }


def input_identity_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [input_identity_record(record) for record in sorted(records, key=lambda item: str(item["caseId"]))]


def normal_input_identity(records: list[dict[str, Any]]) -> str:
    """Bind the exact normal feature envelope without reading blind labels."""

    identity_records = input_identity_records([
        record for record in records
        if record["role"] in {"FIT", "THRESHOLD_TUNING"} and record["kind"] == "NOMINAL"
    ])
    if not identity_records:
        raise IterationError("normal input identity requires FIT or THRESHOLD_TUNING nominal records")
    return canonical_json_sha256(identity_records)


def normal_only_evidence(
    feature_records: list[dict[str, Any]],
    score_records: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expose auditable membership for a JSON-only normal candidate selector."""

    def roles(records: list[dict[str, Any]]) -> list[str]:
        return sorted({str(record["role"]) for record in records})

    def kinds(records: list[dict[str, Any]]) -> list[str]:
        return sorted({str(record["kind"]) for record in records})

    normal_records = [
        record for record in feature_records
        if record["role"] in {"FIT", "THRESHOLD_TUNING"} and record["kind"] == "NOMINAL"
    ]
    feature_inputs = input_identity_records(feature_records)
    calibration_inputs = input_identity_records(calibration_records)
    original_tuning_inputs = [record for record in calibration_inputs if not record["isAugmentation"]]
    return {
        "featureInputCount": len(feature_records),
        "featureInputRoles": roles(feature_records),
        "featureInputKinds": kinds(feature_records),
        "blindFeatureInputCount": sum(record["role"] == "BLIND" for record in feature_records),
        "anomalyFeatureInputCount": sum(record["kind"] == "ANOMALY" for record in feature_records),
        "normalInputRecordCount": len(normal_records),
        "featureInputs": feature_inputs,
        "featureInputIdentitySha256": canonical_json_sha256(feature_inputs),
        "normalInputIdentitySha256": normal_input_identity(normal_records),
        "reportedScoreCount": len(score_records),
        "reportedScoreRoles": roles(score_records),
        "reportedScoreKinds": kinds(score_records),
        "calibrationScoreCount": len(calibration_records),
        "calibrationScoreRoles": roles(calibration_records),
        "calibrationScoreKinds": kinds(calibration_records),
        "calibrationInputs": calibration_inputs,
        "calibrationInputIdentitySha256": canonical_json_sha256(calibration_inputs),
        "originalTuningInputCount": len(original_tuning_inputs),
        "originalTuningInputs": original_tuning_inputs,
        "originalTuningInputIdentitySha256": canonical_json_sha256(original_tuning_inputs),
    }


def candidate_configuration(algorithm_report: dict[str, Any], *, batch_size: int) -> dict[str, Any]:
    """Expose every score-affecting candidate knob for a frozen selection contract."""

    configuration: dict[str, Any] = {
        "algorithmId": algorithm_report["id"],
        "batchSize": batch_size,
    }
    if algorithm_report["id"] == "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1":
        for name in (
            "memoryBankSelection",
            "maxPrototypePatches",
            "topKMostAnomalousPatches",
            "prototypeBlockSize",
        ):
            configuration[name] = algorithm_report[name]
    return configuration


def run(
    manifest_path: Path,
    output_path: Path,
    *,
    model_repo: Path,
    model_weights: Path,
    device: str,
    algorithm: str,
    max_prototypes: int,
    top_k_patches: int,
    batch_size: int,
    prototype_block_size: int,
    augmentation_manifest_path: Path | None = None,
    feature_cache_path: Path | None = None,
    include_pixel_metrics: bool = True,
    normal_only: bool = False,
) -> dict[str, Any]:
    if _is_under(REPOSITORY_ROOT, output_path):
        raise IterationError("output must stay outside the Git working tree")
    if output_path.exists():
        raise IterationError("output already exists; choose a fresh immutable report path")
    if algorithm not in {"global-knn", "patch-knn"}:
        raise IterationError("algorithm must be global-knn or patch-knn")
    started = time.perf_counter()
    manifest, categories, augmentation = build_iteration_inputs(manifest_path, augmentation_manifest_path)
    timings = {
        "provenanceSeconds": 0.0,
        "inputVerificationSeconds": 0.0,
        "cacheValidationSeconds": 0.0,
        "cacheWriteSeconds": 0.0,
        "featureInferenceSeconds": 0.0,
        "scoringSeconds": 0.0,
    }
    provenance_started = time.perf_counter()
    model_weights_sha256 = _immutable_file_sha256(model_weights, description="model weights")
    extractor_identity = feature_extractor_identity(
        model_repo=model_repo,
        model_weights_sha256=model_weights_sha256,
        device=device,
    )
    extractor_identity_sha256 = canonical_json_sha256(extractor_identity)
    cache = None if feature_cache_path is None else FeatureCache(
        feature_cache_path,
        feature_extractor=extractor_identity,
        feature_extractor_identity_sha256=extractor_identity_sha256,
    )
    timings["provenanceSeconds"] += time.perf_counter() - provenance_started
    embedder = ResearchBatchEmbedder(model_repo=model_repo, model_weights=model_weights, device=device)
    provenance_started = time.perf_counter()
    loaded_identity = feature_extractor_identity(
        model_repo=model_repo,
        model_weights_sha256=_immutable_file_sha256(model_weights, description="model weights"),
        device=device,
    )
    timings["provenanceSeconds"] += time.perf_counter() - provenance_started
    if loaded_identity != extractor_identity:
        raise IterationError("feature extractor inputs changed while the model was loading")
    feature_kind = "global" if algorithm == "global-knn" else "patch"
    all_records_by_case: dict[str, dict[str, Any]] = {}
    for category_records in categories.values():
        for record in category_records:
            if normal_only and (record["role"] not in {"FIT", "THRESHOLD_TUNING"} or record["kind"] != "NOMINAL"):
                continue
            case_id = str(record["caseId"])
            if case_id in all_records_by_case:
                raise IterationError("iteration input caseId is duplicated")
            all_records_by_case[case_id] = record
    feature_records = [all_records_by_case[case_id] for case_id in sorted(all_records_by_case)]
    cache_counts = {"hits": 0, "misses": 0}
    features, pending_cache_writes = extract_features(
        feature_records,
        feature_kind=feature_kind,
        embedder=embedder,
        cache=cache,
        batch_size=batch_size,
        timings=timings,
        cache_counts=cache_counts,
    )
    provenance_started = time.perf_counter()
    completed_identity = feature_extractor_identity(
        model_repo=model_repo,
        model_weights_sha256=_immutable_file_sha256(model_weights, description="model weights"),
        device=device,
    )
    timings["provenanceSeconds"] += time.perf_counter() - provenance_started
    if completed_identity != extractor_identity:
        raise IterationError("feature extractor inputs changed while features were extracted")
    if cache is not None:
        cache_write_started = time.perf_counter()
        for cache_key, values, record in pending_cache_writes:
            cache.put(cache_key, values, feature_kind=feature_kind, record=record)
        timings["cacheWriteSeconds"] += time.perf_counter() - cache_write_started

    category_reports: dict[str, Any] = {}
    score_records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    blind_score_records: list[dict[str, Any]] = []
    for category, category_inputs in sorted(categories.items()):
        fit, tuning, blind = _categorize(category_inputs, normal_only=normal_only)
        scoring_started = time.perf_counter()
        query = tuning + blind
        component_by_case: dict[str, dict[str, Any]] = {}
        if algorithm == "global-knn":
            import numpy as np

            prototypes = np.asarray([features[str(record["caseId"])] for record in fit], dtype=np.float32)
            query_features = np.asarray([features[str(record["caseId"])] for record in query], dtype=np.float32)
            scores = global_knn_scores(query_features, prototypes)
            for record, score in zip(query, scores, strict=True):
                record["score"] = score
        else:
            import numpy as np

            fit_patches = np.concatenate([np.asarray(features[str(record["caseId"])] , dtype=np.float32) for record in fit], axis=0)
            prototype_indices = deterministic_prototype_indices(len(fit_patches), max_prototypes)
            prototypes = fit_patches[prototype_indices]
            query_patches = np.asarray([features[str(record["caseId"])] for record in query], dtype=np.float32)
            components = patch_knn_scores_blocked(
                query_patches, prototypes, top_k=top_k_patches, prototype_block_size=prototype_block_size
            )
            patch_count = query_patches.shape[1]
            side = int(math.isqrt(patch_count))
            if side * side != patch_count:
                raise IterationError("DINO patch count is not a square grid")
            patch_grid_height, patch_grid_width = side, side
            for record, components_for_record in zip(query, components, strict=True):
                components_for_record["patchDistanceGrid"] = components_for_record["patchDistanceGrid"].reshape(side, side)
                record["score"] = components_for_record["score"]
                component_by_case[str(record["caseId"])] = components_for_record
        timings["scoringSeconds"] += time.perf_counter() - scoring_started
        threshold = max(float(record["score"]) for record in tuning)
        category_algorithm: dict[str, Any] = {
            "fitOriginalCount": sum(not record["isAugmentation"] for record in fit),
            "fitAugmentedCount": sum(record["isAugmentation"] for record in fit),
            "tuningOriginalCount": sum(not record["isAugmentation"] for record in tuning),
            "tuningAugmentedCount": sum(record["isAugmentation"] for record in tuning),
        }
        if algorithm == "patch-knn":
            category_algorithm |= {
                "fitPatchCount": int(len(fit_patches)),
                "prototypePatchCount": int(len(prototypes)),
                "patchGridHeight": int(patch_grid_height),
                "patchGridWidth": int(patch_grid_width),
            }
        category_reports[category] = metric_summary(blind, threshold) | normal_calibration_summary(tuning) | category_algorithm
        for record in tuning:
            calibration_records.append(_score_record(
                record,
                None if normal_only else component_by_case.get(str(record["caseId"])),
            ))
            if not record["isAugmentation"]:
                score_records.append(_score_record(
                    record,
                    None if normal_only else component_by_case.get(str(record["caseId"])),
                ))
        for record in blind:
            scored = _score_record(record, component_by_case.get(str(record["caseId"])))
            score_records.append(scored)
            blind_score_records.append({**record, **scored})

    pixel_started = time.perf_counter()
    pixel_metrics = None
    if algorithm == "patch-knn" and include_pixel_metrics and not normal_only:
        pixel_metrics = calculate_pixel_localization_metrics(blind_score_records)
    timings["pixelMetricsSeconds"] = time.perf_counter() - pixel_started
    timings["totalElapsedSeconds"] = time.perf_counter() - started
    algorithm_report: dict[str, Any] = {
        "id": "DINOV2_GLOBAL_NEAREST_NORMAL_COSINE_V1" if algorithm == "global-knn" else "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1",
        "modelRepository": str(model_repo),
        "modelWeights": str(model_weights),
        "modelWeightsSha256": model_weights_sha256,
        "preprocessingId": PREPROCESSING_ID,
        "device": device,
    }
    if algorithm == "patch-knn":
        algorithm_report |= {
            "memoryBankSelection": "DETERMINISTIC_EVENLY_SPACED_PATCH_SUBSET_AFTER_STABLE_PARENT_SORT",
            "maxPrototypePatches": max_prototypes,
            "topKMostAnomalousPatches": top_k_patches,
            "prototypeBlockSize": prototype_block_size,
        }
    algorithm_report["modelRepositorySha256"] = extractor_identity["modelRepositorySha256"]
    evidence = normal_only_evidence(feature_records, score_records, calibration_records)
    configuration = candidate_configuration(algorithm_report, batch_size=batch_size)
    report = {
        "schemaVersion": ITERATION_SCHEMA_VERSION,
        "authoritative": False,
        "productionAuthorized": False,
        "disclaimer": "Offline non-commercial MVTec research only. This cannot select a PhoneDINO production threshold, qualify a physical device, or emit PASS/FAIL/equipment decisions.",
        "selectionProtocol": "NORMAL_ONLY_ITERATION_THEN_BLIND_REPORTING_ONLY",
        "blindReporting": {
            "state": "NOT_RUN" if normal_only else "REPORTED_ONCE_AFTER_CONFIGURATION_LOCK",
            "blindSourcePolicy": "ORIGINAL_ONLY",
            "reason": "NORMAL_ONLY_ITERATION" if normal_only else "FIXED_CONFIGURATION_OBSERVATION",
        },
        "inputManifest": str(manifest_path),
        "inputManifestFileSha256": sha256_file(manifest_path),
        "inputManifestDeclaredSha256": manifest.get("manifestSha256"),
        "augmentation": augmentation,
        "algorithm": algorithm_report,
        "featureExtractor": extractor_identity,
        "featureExtractorIdentitySha256": extractor_identity_sha256,
        "candidateConfiguration": configuration,
        "candidateConfigurationSha256": canonical_json_sha256(configuration),
        "execution": {
            "batchSize": batch_size,
            "featureCache": None if cache is None else str(feature_cache_path),
            "featureCacheSchemaVersion": FEATURE_CACHE_SCHEMA_VERSION,
            "featureCacheHits": cache_counts["hits"],
            "featureCacheMisses": cache_counts["misses"],
            "iterationToolSha256": extractor_identity["iterationToolSha256"],
            "phaseTimingsSeconds": {name: round(value, 6) for name, value in timings.items()},
            "python": sys.version,
            "platform": platform.platform(),
            "numpyVersion": __import__("numpy").__version__,
            "torchVersion": __import__("torch").__version__,
            "torchThreadCount": __import__("torch").get_num_threads(),
        },
        "categories": category_reports,
        "normalOnlyEvidence": evidence,
        "calibrationScores": sorted(calibration_records, key=lambda record: record["caseId"]),
        "scores": sorted(score_records, key=lambda record: record["caseId"]),
        "pixelLocalization": pixel_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one offline, normal-only MVTec AD research iteration")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new immutable report path outside this Git worktree")
    parser.add_argument("--model-repo", type=Path, default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2")
    parser.add_argument("--model-weights", type=Path, default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2_vits14_pretrain.pth")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--algorithm", choices=("global-knn", "patch-knn"), default="patch-knn")
    parser.add_argument("--max-prototypes", type=int, default=1024)
    parser.add_argument("--top-k-patches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prototype-block-size", type=int, default=256)
    parser.add_argument("--augmentation-manifest", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--no-pixel-metrics", action="store_true")
    parser.add_argument("--normal-only", action="store_true", help="do not embed, score, or report the frozen blind set")
    arguments = parser.parse_args()
    if _is_under(REPOSITORY_ROOT, arguments.output):
        parser.error("--output must stay outside the Git working tree")
    report = run(
        arguments.manifest,
        arguments.output,
        model_repo=arguments.model_repo,
        model_weights=arguments.model_weights,
        device=arguments.device,
        algorithm=arguments.algorithm,
        max_prototypes=arguments.max_prototypes,
        top_k_patches=arguments.top_k_patches,
        batch_size=arguments.batch_size,
        prototype_block_size=arguments.prototype_block_size,
        augmentation_manifest_path=arguments.augmentation_manifest,
        feature_cache_path=arguments.feature_cache,
        include_pixel_metrics=not arguments.no_pixel_metrics,
        normal_only=arguments.normal_only,
    )
    print(json.dumps({
        "output": str(arguments.output),
        "categories": report["categories"],
        "blindAugmentedCount": report["augmentation"]["blindAugmentedCount"],
        "elapsedSeconds": report["execution"]["phaseTimingsSeconds"]["totalElapsedSeconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
