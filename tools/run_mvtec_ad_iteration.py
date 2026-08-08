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
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


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


ITERATION_SCHEMA_VERSION = "phone-dino.mvtec-ad-iteration-report/1.0"
FEATURE_CACHE_SCHEMA_VERSION = "phone-dino.mvtec-ad-feature-cache/1.0"
PREPROCESSING_ID = "DINOV2_RESIZE_SHORT_EDGE_256_CENTER_CROP_224_IMAGENET_NORMALIZE_V1"


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

    def __init__(self, root: Path, *, model_weights_sha256: str, source_sha256: str):
        if _is_under(REPOSITORY_ROOT, root):
            raise IterationError("feature cache must stay outside the Git working tree")
        self.root = root
        self.model_weights_sha256 = model_weights_sha256
        self.source_sha256 = source_sha256
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for(self, record: dict[str, Any], feature_kind: str) -> str:
        return canonical_json_sha256({
            "schemaVersion": FEATURE_CACHE_SCHEMA_VERSION,
            "featureKind": feature_kind,
            "sourceSha256": record["sourceSha256"],
            "augmentationRecipeSha256": record.get("augmentationRecipeSha256"),
            "modelWeightsSha256": self.model_weights_sha256,
            "preprocessingId": PREPROCESSING_ID,
            "iterationToolSha256": self.source_sha256,
        })[7:]

    def get(self, key: str) -> object | None:
        import numpy as np

        path = self.root / f"{key}.npy"
        if not path.exists():
            return None
        try:
            values = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise IterationError(f"feature cache entry is unreadable: {path}") from error
        if not np.all(np.isfinite(values)):
            raise IterationError(f"feature cache entry is non-finite: {path}")
        return values.astype(np.float32, copy=False)

    def put(self, key: str, values: object) -> None:
        import numpy as np

        value = np.asarray(values, dtype=np.float32)
        temporary = self.root / f".{key}.{os.getpid()}.tmp.npy"
        np.save(temporary, value, allow_pickle=False)
        os.replace(temporary, self.root / f"{key}.npy")


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
) -> dict[str, object]:
    """Verify every input, then batch extract only cache misses."""

    if batch_size <= 0:
        raise IterationError("batch_size must be positive")
    features: dict[str, object] = {}
    misses: list[tuple[dict[str, Any], Image.Image, str | None]] = []
    verification_started = time.perf_counter()
    for record in records:
        case_id = str(record["caseId"])
        if case_id in features:
            raise IterationError("feature input caseId is duplicated")
        cache_key = None if cache is None else cache.key_for(record, feature_kind)
        image = _load_rgb_and_verify(record)
        cached = None if cache is None else cache.get(cache_key)
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
                cache.put(cache_key, values)
            features[str(record["caseId"])] = values
    return features


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
    model_weights_sha256 = sha256_file(model_weights)
    source_sha256 = sha256_file(Path(__file__))
    cache = None if feature_cache_path is None else FeatureCache(
        feature_cache_path, model_weights_sha256=model_weights_sha256, source_sha256=source_sha256
    )
    embedder = ResearchBatchEmbedder(model_repo=model_repo, model_weights=model_weights, device=device)
    feature_kind = "global" if algorithm == "global-knn" else "patch"
    all_records_by_case: dict[str, dict[str, Any]] = {}
    for category_records in categories.values():
        for record in category_records:
            if normal_only and record["role"] == "BLIND":
                continue
            case_id = str(record["caseId"])
            if case_id in all_records_by_case:
                raise IterationError("iteration input caseId is duplicated")
            all_records_by_case[case_id] = record
    timings = {"inputVerificationSeconds": 0.0, "featureInferenceSeconds": 0.0, "scoringSeconds": 0.0}
    cache_counts = {"hits": 0, "misses": 0}
    features = extract_features(
        [all_records_by_case[case_id] for case_id in sorted(all_records_by_case)],
        feature_kind=feature_kind,
        embedder=embedder,
        cache=cache,
        batch_size=batch_size,
        timings=timings,
        cache_counts=cache_counts,
    )

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
            calibration_records.append(_score_record(record, component_by_case.get(str(record["caseId"]))))
            if not record["isAugmentation"]:
                score_records.append(_score_record(record, component_by_case.get(str(record["caseId"]))))
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
        "execution": {
            "batchSize": batch_size,
            "featureCache": None if cache is None else str(feature_cache_path),
            "featureCacheHits": cache_counts["hits"],
            "featureCacheMisses": cache_counts["misses"],
            "iterationToolSha256": source_sha256,
            "phaseTimingsSeconds": {name: round(value, 6) for name, value in timings.items()},
            "python": sys.version,
            "platform": platform.platform(),
            "numpyVersion": __import__("numpy").__version__,
            "torchVersion": __import__("torch").__version__,
            "torchThreadCount": __import__("torch").get_num_threads(),
        },
        "categories": category_reports,
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
