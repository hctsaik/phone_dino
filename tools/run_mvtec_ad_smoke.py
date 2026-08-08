"""Run non-authoritative DINOv2 anomaly-detection baselines on frozen MVTec AD data.

This is deliberately an offline research tool.  It does not call the PhoneDINO
service, alter a runtime artifact, or emit an equipment decision.  Thresholds
come only from normal ``THRESHOLD_TUNING`` records; blind labels are read only
after all scores are fixed for reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.engines import LocalDinoV2Adapter
from phone_dino.production import DinoV2Embedder


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def cosine_distance(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("DINO embedding norm is zero")
    return max(0.0, min(2.0, 1.0 - numerator / (left_norm * right_norm)))


def image_auroc(labels: list[bool], scores: list[float]) -> float | None:
    """Compute binary AUROC from scores without pulling a benchmark dependency."""

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


def deterministic_prototype_indices(total: int, maximum: int) -> list[int]:
    """Return evenly spaced, deterministic memory-bank positions.

    This is a bounded research baseline rather than a claim of a full PatchCore
    coreset.  It prevents the benchmark's memory use from depending on the
    number of normal FIT captures, while retaining samples across the complete
    (stable, manifest-sorted) patch sequence.
    """

    if total <= 0:
        raise ValueError("Patch memory bank has no normal prototypes")
    if maximum <= 0:
        raise ValueError("max_prototypes must be positive")
    if total <= maximum:
        return list(range(total))
    return [((2 * index + 1) * total) // (2 * maximum) for index in range(maximum)]


def patch_knn_score(query_patches: object, prototype_patches: object, *, top_k: int) -> dict[str, float]:
    """Score an image from its most anomalous DINO patch-token distances.

    Each query patch is compared with every normal prototype patch.  The image
    score is the mean of its largest ``top_k`` nearest-normal distances, which
    makes a local defect visible without letting one noisy patch dominate.
    """

    import numpy as np

    query = np.asarray(query_patches, dtype=np.float32)
    prototypes = np.asarray(prototype_patches, dtype=np.float32)
    if query.ndim != 2 or prototypes.ndim != 2 or query.shape[1] != prototypes.shape[1]:
        raise ValueError("Patch matrix dimensions do not match")
    if query.shape[0] == 0 or prototypes.shape[0] == 0:
        raise ValueError("Patch matrix is empty")
    if top_k <= 0 or top_k > query.shape[0]:
        raise ValueError("top_k must be between 1 and the number of query patches")
    query_norms = np.linalg.norm(query, axis=1, keepdims=True)
    prototype_norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if np.any(query_norms <= 1e-12) or np.any(prototype_norms <= 1e-12):
        raise ValueError("Patch embedding norm is zero")
    similarity = (query / query_norms) @ (prototypes / prototype_norms).T
    nearest = np.min(np.clip(1.0 - similarity, 0.0, 2.0), axis=1)
    largest = np.partition(nearest, -top_k)[-top_k:]
    return {
        "score": float(np.mean(largest)),
        "maxPatchDistance": float(np.max(nearest)),
        "meanNearestPatchDistance": float(np.mean(nearest)),
    }


def run(
    manifest_path: Path,
    output_path: Path,
    *,
    model_repo: Path,
    model_weights: Path,
    device: str,
    algorithm: str = "global-knn",
    max_prototypes: int = 1024,
    top_k_patches: int = 5,
) -> dict[str, Any]:
    from PIL import Image

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != "phone-dino.mvtec-ad-smoke/1.0" or manifest.get("authoritative") is not False:
        raise ValueError("Expected a non-authoritative MVTec AD smoke manifest")
    root = manifest_path.parent
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest has no records")
    adapter = LocalDinoV2Adapter(repository=model_repo, weights=model_weights)
    ready, reason = adapter.readiness()
    if not ready:
        raise RuntimeError(reason)
    embedder = DinoV2Embedder(adapter, device=device)
    embedder.warm_up()

    if algorithm not in {"global-knn", "patch-knn"}:
        raise ValueError("algorithm must be global-knn or patch-knn")
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for original in records:
        if not isinstance(original, dict):
            raise ValueError("Manifest record is not an object")
        record = dict(original)
        image_path = root / str(record["relativePath"])
        if sha256_file(image_path) != record["sourceSha256"]:
            raise ValueError(f"Input digest mismatch: {image_path}")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            if algorithm == "patch-knn":
                patch_embedding = embedder.embed_with_patches(rgb)
                record["patches"] = patch_embedding.patch_grid
                record["patchGridHeight"] = patch_embedding.grid_height
                record["patchGridWidth"] = patch_embedding.grid_width
            else:
                record["embedding"] = embedder.embed(rgb)
        by_category[str(record["category"])].append(record)

    category_reports: dict[str, Any] = {}
    score_records: list[dict[str, Any]] = []
    for category, category_records in sorted(by_category.items()):
        fit_records = [record for record in category_records if record["role"] == "FIT" and record["kind"] == "NOMINAL"]
        tuning = [record for record in category_records if record["role"] == "THRESHOLD_TUNING" and record["kind"] == "NOMINAL"]
        blind = [record for record in category_records if record["role"] == "BLIND"]
        if not fit_records or not tuning or not blind:
            raise ValueError(f"Incomplete split for {category}")
        category_algorithm: dict[str, Any] = {}
        if algorithm == "patch-knn":
            import numpy as np

            grids = {(record["patchGridHeight"], record["patchGridWidth"]) for record in category_records}
            if len(grids) != 1:
                raise ValueError(f"Patch grid mismatch for {category}")
            all_fit_patches = np.concatenate([
                np.asarray(record["patches"], dtype=np.float32) for record in fit_records
            ], axis=0)
            prototype_indices = deterministic_prototype_indices(len(all_fit_patches), max_prototypes)
            prototypes = all_fit_patches[prototype_indices]
            for record in tuning + blind:
                components = patch_knn_score(record["patches"], prototypes, top_k=top_k_patches)
                record.update(components)
            patch_grid_height, patch_grid_width = next(iter(grids))
            category_algorithm = {
                "fitPatchCount": int(len(all_fit_patches)),
                "prototypePatchCount": int(len(prototypes)),
                "patchGridHeight": patch_grid_height,
                "patchGridWidth": patch_grid_width,
            }
        else:
            fit = [record["embedding"] for record in fit_records]
            for record in tuning + blind:
                record["score"] = min(cosine_distance(record["embedding"], prototype) for prototype in fit)
        threshold = max(float(record["score"]) for record in tuning)
        category_reports[category] = metric_summary(blind, threshold) | category_algorithm
        score_records.extend({
            "caseId": record["caseId"],
            "category": category,
            "role": record["role"],
            "kind": record["kind"],
            "defect": record["defect"],
            "sourceSha256": record["sourceSha256"],
            "score": record["score"],
            **({
                "maxPatchDistance": record["maxPatchDistance"],
                "meanNearestPatchDistance": record["meanNearestPatchDistance"],
            } if algorithm == "patch-knn" else {}),
        } for record in tuning + blind)

    algorithm_report = {
        "id": "DINOV2_GLOBAL_NEAREST_NORMAL_COSINE_V1",
        "modelRepository": str(model_repo),
        "modelWeights": str(model_weights),
        "modelWeightsSha256": sha256_file(model_weights),
        "device": device,
    }
    if algorithm == "patch-knn":
        algorithm_report = {
            **algorithm_report,
            "id": "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1",
            "memoryBankSelection": "DETERMINISTIC_EVENLY_SPACED_PATCH_SUBSET",
            "maxPrototypePatches": max_prototypes,
            "topKMostAnomalousPatches": top_k_patches,
        }
    report = {
        "schemaVersion": f"phone-dino.mvtec-ad-dinov2-{algorithm}-report/1.0",
        "authoritative": False,
        "disclaimer": "Offline non-commercial research benchmark only. This is not PhoneDINO runtime qualification, physical-device validation, or a PASS/FAIL/equipment decision.",
        "inputManifest": str(manifest_path),
        "inputManifestSha256": manifest.get("manifestSha256"),
        "algorithm": algorithm_report,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "categories": category_reports,
        "scores": sorted(score_records, key=lambda record: record["caseId"]),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline MVTec AD DINOv2 research baseline")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-repo", type=Path, default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2")
    parser.add_argument("--model-weights", type=Path, default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2_vits14_pretrain.pth")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--algorithm", choices=("global-knn", "patch-knn"), default="global-knn")
    parser.add_argument("--max-prototypes", type=int, default=1024)
    parser.add_argument("--top-k-patches", type=int, default=5)
    arguments = parser.parse_args()
    default_name = "dinov2_global_knn_report.json" if arguments.algorithm == "global-knn" else "dinov2_patch_knn_report.json"
    output = arguments.output or arguments.manifest.parent / default_name
    report = run(
        arguments.manifest, output, model_repo=arguments.model_repo, model_weights=arguments.model_weights,
        device=arguments.device, algorithm=arguments.algorithm, max_prototypes=arguments.max_prototypes,
        top_k_patches=arguments.top_k_patches,
    )
    print(json.dumps({"output": str(output), "categories": report["categories"]}, indent=2))


if __name__ == "__main__":
    main()
