"""Synthetic-only DINO discrimination test for offline MVTec engineering.

This module is deliberately independent of the frozen V1/V2 normal-selection
and successor workflows.  It opens raw source bytes only through the
successor FIT-only loader and accepts rendered query children only through the
separate synthetic-only augmentation validator.  It never accepts a blind,
true-anomaly, mask, selection, confirmation, tuning, or reserve image path.

Its metrics distinguish deterministic rendered stimuli from raw normal query
parents.  They are *not* estimates of real anomaly precision, recall, device
performance, or physical qualification.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from phone_dino import mvtec_normal_successor as successor
from phone_dino import mvtec_successor_evaluator_v2 as knn
from phone_dino import mvtec_synthetic_anomaly_augmentation as augmentation


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_ANOMALY_TEST_V1_SCHEMA = "phone-dino.mvtec-ad-synthetic-only-test-report/1.0"
SYNTHETIC_ANOMALY_TEST_V1_PURPOSE = "OFFLINE_MVTEC_SYNTHETIC_ONLY_AUGMENTATION_AND_TESTING"
SYNTHETIC_ANOMALY_TEST_V1_PHASE = "SYNTHETIC_RENDERING_DISCRIMINATION_TEST"
SYNTHETIC_ANOMALY_TEST_V1_METRIC_SCOPE = "SYNTHETIC_RENDERING_DISCRIMINATION_ONLY"
SYNTHETIC_ANOMALY_TEST_V1_INPUT_POLICY = "SUCCESSOR_FIT_RAW_NORMAL_PARENTS_ONLY"
SYNTHETIC_ANOMALY_TEST_V1_RESULT_LABEL = "SYNTHETIC_ONLY_NOT_REAL_ANOMALY_PERFORMANCE"
SYNTHETIC_ANOMALY_TEST_V1_REAL_PERFORMANCE = "NOT_ESTIMATED"
SYNTHETIC_ANOMALY_TEST_V1_FORBIDDEN_USES = [
    "MODEL_SELECTION",
    "THRESHOLD_SELECTION",
    "PRODUCTION_VALIDATION",
    "PHYSICAL_QUALIFICATION",
]
SYNTHETIC_ANOMALY_TEST_V1_CATEGORIES = ("capsule", "metal_nut", "tile")
SYNTHETIC_ANOMALY_TEST_V1_SPLIT_ALGORITHM = augmentation.SYNTHETIC_ANOMALY_PARENT_SPLIT_ALGORITHM
SYNTHETIC_ANOMALY_TEST_V1_DECISION_RULE = "SCORE_STRICTLY_GREATER_THAN_RAW_CALIBRATION_MAX_V1"
SYNTHETIC_ANOMALY_TEST_V1_ALGORITHM = "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_SYNTHETIC_V1"
SYNTHETIC_ANOMALY_TEST_V1_PROTOTYPE_SELECTION = "DETERMINISTIC_STRATIFIED_HASH_RANKED_PATCH_PREFIX_V2"
SYNTHETIC_ANOMALY_TEST_V1_MAX_PROTOTYPE_PATCHES = 1024
SYNTHETIC_ANOMALY_TEST_V1_TOP_K = 5
SYNTHETIC_ANOMALY_TEST_V1_BLOCK_SIZE = 256
SYNTHETIC_ANOMALY_TEST_V1_BATCH_SIZE = 4
SYNTHETIC_ANOMALY_TEST_V1_PARENT_COUNTS = {
    "SYNTHETIC_PROTOTYPE": 6,
    "SYNTHETIC_CALIBRATION": 2,
    "SYNTHETIC_QUERY": 4,
}

REPORT_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "syntheticOnly",
    "purpose",
    "phase",
    "metricScope",
    "realAnomalyPerformance",
    "forbiddenUses",
    "inputPolicy",
    "blindPolicy",
    "resultLabel",
    "testConfiguration",
    "parentHoldoutManifestFileSha256",
    "parentHoldoutManifestDeclaredSha256",
    "parentSelectionContractFileSha256",
    "parentSelectionContractDeclaredSha256",
    "successorSealFileSha256",
    "successorSealDeclaredSha256",
    "successorPlanFileSha256",
    "successorPlanDeclaredSha256",
    "successorEnvelopeFileSha256",
    "successorEnvelopeDeclaredSha256",
    "successorFitIdentitySha256",
    "parentNormalConfirmationIdentitySha256",
    "augmentationManifestFileSha256",
    "augmentationManifestDeclaredSha256",
    "recipeFileSha256",
    "parentSplit",
    "parentSplitIdentitySha256",
    "featureExtractor",
    "featureExtractorIdentitySha256",
    "thresholds",
    "calibrationScores",
    "queryScores",
    "categories",
    "aggregate",
    "execution",
    "syntheticTestReportSha256",
}

PARENT_SPLIT_RECORD_FIELDS = {"caseId", "category", "sourceSha256", "sourceGroupId", "role"}
CALIBRATION_SCORE_FIELDS = {
    "caseId",
    "category",
    "sourceSha256",
    "score",
    "maxPatchDistance",
    "meanNearestPatchDistance",
}
QUERY_SCORE_FIELDS = {
    "caseId",
    "parentCaseId",
    "category",
    "sourceSha256",
    "inputKind",
    "syntheticDefectFamily",
    "score",
    "maxPatchDistance",
    "meanNearestPatchDistance",
}
METRIC_FIELDS = {
    "syntheticTP",
    "syntheticFP",
    "syntheticFN",
    "syntheticTN",
    "syntheticPrecision",
    "syntheticRecall",
    "syntheticF1",
}


class SyntheticAnomalyTestError(ValueError):
    """Raised when a synthetic-only test would cross its input boundary."""


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without retaining the full file in memory."""

    return augmentation.sha256_file(path)


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SyntheticAnomalyTestError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise SyntheticAnomalyTestError(f"{name} must be a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise SyntheticAnomalyTestError(f"{name} must be a SHA-256 digest") from error
    return digest


def _require_finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SyntheticAnomalyTestError(f"{name} must be finite")
    return float(value)


def _require_exact_fields(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SyntheticAnomalyTestError(f"{name} must be an object")
    missing = fields.difference(value)
    unknown = set(value).difference(fields)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unsupported {', '.join(sorted(unknown))}")
        raise SyntheticAnomalyTestError(f"{name} has {'; '.join(detail)} fields")
    return value


def _document_digest(document: dict[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("syntheticTestReportSha256", None)
    return successor.canonical_json_sha256(unsigned)


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat(follow_symlinks=False)
    return status.st_dev, status.st_ino, status.st_mode, status.st_size


def _write_external_report(path: Path, document: dict[str, Any], *, repository_root: Path) -> Path:
    """Write a new external report slot after V2's reparse-safe preflight."""

    try:
        prepared = knn._require_external_output(path, repository_root=repository_root)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticAnomalyTestError(str(error)) from error
    data = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with prepared.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            fd_signature = os.fstat(stream.fileno())
    except OSError as error:
        raise SyntheticAnomalyTestError("unable to write synthetic-only test report") from error
    try:
        if knn._is_link_or_reparse_point(prepared) or _stat_signature(prepared) != (
            fd_signature.st_dev,
            fd_signature.st_ino,
            fd_signature.st_mode,
            fd_signature.st_size,
        ):
            raise SyntheticAnomalyTestError("synthetic-only test report changed while it was written")
    except OSError as error:
        raise SyntheticAnomalyTestError("unable to verify synthetic-only report output") from error
    return prepared


def _feature_extractor_identity(*, model_repo: Path, model_weights: Path, device: str) -> dict[str, Any]:
    """Bind actual DINO bytes plus every source module used by this harness."""

    if device != "cpu":
        raise SyntheticAnomalyTestError("synthetic-only test supports CPU only")
    try:
        import numpy as np
        import torch
        import torchvision
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SyntheticAnomalyTestError("synthetic-only test requires numpy, torch, and torchvision") from error
    try:
        model_weights_sha256 = knn._immutable_file_sha256(model_weights, description="model weights")
        model_repository_sha256 = knn.sha256_directory(model_repo)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticAnomalyTestError(str(error)) from error
    source_paths = {
        "syntheticTestModuleSha256": Path(__file__),
        "syntheticAugmentationModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "mvtec_synthetic_anomaly_augmentation.py",
        "successorModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "mvtec_normal_successor.py",
        "patchKnnModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "mvtec_successor_evaluator_v2.py",
        "productionModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "production.py",
        "enginesModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "engines.py",
    }
    return {
        "schemaVersion": "phone-dino.mvtec-ad-synthetic-only-feature-extractor/1.0",
        "modelWeightsSha256": model_weights_sha256,
        "modelRepositorySha256": model_repository_sha256,
        "modelEntrypoint": "dinov2_vits14",
        "device": device,
        "preprocessingId": knn.SUCCESSOR_V2_PREPROCESSING_ID,
        "algorithmId": SYNTHETIC_ANOMALY_TEST_V1_ALGORITHM,
        "prototypeSelection": SYNTHETIC_ANOMALY_TEST_V1_PROTOTYPE_SELECTION,
        **{name: sha256_file(path) for name, path in source_paths.items()},
        "pythonVersion": platform.python_version(),
        "numpyVersion": np.__version__,
        "torchVersion": torch.__version__,
        "torchvisionVersion": torchvision.__version__,
    }


def _execution_metadata(timings: dict[str, float], *, repository_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - environment dependent
        revision = None
    return {
        "syntheticTestModuleSha256": sha256_file(Path(__file__)),
        "entrypointSha256": sha256_file(repository_root / "tools" / "run_mvtec_ad_synthetic_anomaly_test.py"),
        "timingBasis": "PROCESS_CPU_TIME_EXCLUDES_SUSPEND",
        "phaseTimingsSeconds": {name: round(value, 6) for name, value in timings.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "gitRevision": revision,
    }


def _fit_parent_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fail closed unless these are exactly raw nominal FIT records."""

    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_groups: set[str] = set()
    for record in records:
        if record.get("partition") != "FIT" or record.get("kind") != "NOMINAL" or record.get("defect") != "good":
            raise SyntheticAnomalyTestError("synthetic-only test may open raw nominal FIT parents only")
        case_id = _require_string(record.get("caseId"), name="FIT parent caseId")
        source = _require_sha256(record.get("sourceSha256"), name="FIT parent sourceSha256")
        group = _require_string(record.get("sourceGroupId"), name="FIT parent sourceGroupId")
        category = _require_string(record.get("category"), name="FIT parent category")
        if category not in SYNTHETIC_ANOMALY_TEST_V1_CATEGORIES:
            raise SyntheticAnomalyTestError("synthetic-only test received an unsupported category")
        if not isinstance(record.get("imagePath"), Path):
            raise SyntheticAnomalyTestError("FIT parent was not loaded through the phase-safe loader")
        if case_id in seen_case_ids or source in seen_sources or group in seen_groups:
            raise SyntheticAnomalyTestError("FIT parents must have unique case, content, and source-group identities")
        seen_case_ids.add(case_id)
        seen_sources.add(source)
        seen_groups.add(group)
        result.append(dict(record))
    if len(result) != 36:
        raise SyntheticAnomalyTestError("synthetic-only test requires exactly 36 successor FIT parents")
    return result


def split_synthetic_fit_parents(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Make the fixed 6/2/4 per-category label-free parent split."""

    parents = _fit_parent_records(records)
    try:
        tagged = augmentation.split_synthetic_anomaly_fit_parents(parents)
    except augmentation.SyntheticAnomalyAugmentationError as error:
        raise SyntheticAnomalyTestError(str(error)) from error
    role_map = {
        "PROTOTYPE": "SYNTHETIC_PROTOTYPE",
        "CALIBRATION": "SYNTHETIC_CALIBRATION",
        "QUERY": "SYNTHETIC_QUERY",
    }
    result = {role: [] for role in SYNTHETIC_ANOMALY_TEST_V1_PARENT_COUNTS}
    for record in tagged:
        tagged_role = record.get("syntheticTestRole")
        role = role_map.get(tagged_role)
        if role is None:
            raise SyntheticAnomalyTestError("synthetic-only parent split emitted an unsupported role")
        result[role].append({**record, "syntheticRole": role})
    for role, expected_per_category in SYNTHETIC_ANOMALY_TEST_V1_PARENT_COUNTS.items():
        for category in SYNTHETIC_ANOMALY_TEST_V1_CATEGORIES:
            if sum(record["category"] == category for record in result[role]) != expected_per_category:
                raise SyntheticAnomalyTestError("synthetic-only parent split count is inconsistent")
    groups_by_role = {role: {record["sourceGroupId"] for record in values} for role, values in result.items()}
    if any(groups_by_role[left].intersection(groups_by_role[right]) for left in groups_by_role for right in groups_by_role if left < right):
        raise SyntheticAnomalyTestError("synthetic-only parent split reuses a source group across roles")
    return {role: sorted(values, key=lambda record: str(record["caseId"])) for role, values in result.items()}


def _parent_split_records(split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records = [
        {
            "caseId": record["caseId"],
            "category": record["category"],
            "sourceSha256": record["sourceSha256"],
            "sourceGroupId": record["sourceGroupId"],
            "role": role,
        }
        for role, values in split.items()
        for record in values
    ]
    if any(set(record) != PARENT_SPLIT_RECORD_FIELDS for record in records):  # pragma: no cover - invariant guard
        raise SyntheticAnomalyTestError("synthetic-only parent split record shape is unsafe")
    return sorted(records, key=lambda record: str(record["caseId"]))


def _adapt_raw(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "partition": "FIT",
        "kind": "NOMINAL",
        "defect": "good",
        "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"],
        "imagePath": record["imagePath"],
        "isAugmentation": False,
        "parentCaseId": None,
        "variantId": None,
        "component": None,
    }


def _adapt_synthetic_query_records(
    records: list[dict[str, Any]], *, query_parents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    parent_by_case = {str(record["caseId"]): record for record in query_parents}
    expected = {
        (str(parent["caseId"]), variant_id)
        for parent in query_parents
        for variant_id in range(1, augmentation.SYNTHETIC_ANOMALY_VARIANTS_PER_PARENT + 1)
    }
    found: set[tuple[str, int]] = set()
    adapted: list[dict[str, Any]] = []
    for record in records:
        parent_case_id = _require_string(record.get("parentCaseId"), name="synthetic query parentCaseId")
        parent = parent_by_case.get(parent_case_id)
        if parent is None:
            continue
        variant_id = record.get("variantId")
        if not isinstance(variant_id, int) or isinstance(variant_id, bool):
            raise SyntheticAnomalyTestError("synthetic query variantId is invalid")
        pair = parent_case_id, variant_id
        if pair in found:
            raise SyntheticAnomalyTestError("synthetic query parent/variant is duplicated")
        found.add(pair)
        if (
            record.get("category") != parent["category"]
            or record.get("sourceGroupId") != parent["sourceGroupId"]
            or record.get("parentSourceSha256") != parent["sourceSha256"]
            or record.get("parentPartition") != "FIT"
            or record.get("syntheticLabel") != "SYNTHETIC_ANOMALY"
            or record.get("syntheticTestRole") != "QUERY"
            or not isinstance(record.get("imagePath"), Path)
        ):
            raise SyntheticAnomalyTestError("synthetic query record does not bind its FIT query parent")
        family = record.get("syntheticDefectFamily")
        if family not in {"LOCAL_SCRATCH", "LOCAL_SPOT", "LOCAL_OCCLUSION"}:
            raise SyntheticAnomalyTestError("synthetic query family is unsupported")
        adapted.append({
            "caseId": _require_string(record.get("caseId"), name="synthetic query caseId"),
            "parentCaseId": parent_case_id,
            "category": parent["category"],
            "sourceSha256": _require_sha256(record.get("sourceSha256"), name="synthetic query sourceSha256"),
            "imagePath": record["imagePath"],
            "syntheticDefectFamily": family,
        })
    if found != expected:
        raise SyntheticAnomalyTestError("synthetic query package does not cover every query parent and family")
    return sorted(adapted, key=lambda record: str(record["caseId"]))


def _extract_features(
    records: list[dict[str, Any]], *, embedder: Any, timings: dict[str, float]
) -> dict[str, object]:
    started = time.process_time()
    try:
        # The frozen helper reports wall-clock sub-timings.  Keep those out of
        # this report so the stated process-CPU timing basis remains true even
        # if a workstation is suspended mid-run.
        result = knn._extract_patch_features(
            records,
            embedder=embedder,
            batch_size=SYNTHETIC_ANOMALY_TEST_V1_BATCH_SIZE,
            timings={"inputVerificationSeconds": 0.0, "featureInferenceSeconds": 0.0},
        )
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticAnomalyTestError(str(error)) from error
    timings["featureProcessingSeconds"] += time.process_time() - started
    return result


def _prototype_bank(records: list[dict[str, Any]], features: dict[str, object], *, category: str) -> object:
    import numpy as np

    selected_records = [record for record in records if record["category"] == category]
    if len(selected_records) != SYNTHETIC_ANOMALY_TEST_V1_PARENT_COUNTS["SYNTHETIC_PROTOTYPE"]:
        raise SyntheticAnomalyTestError("synthetic-only prototype count is inconsistent")
    selected_records.sort(key=lambda record: str(record["caseId"]))
    matrices = [np.asarray(features[str(record["caseId"])], dtype=np.float32) for record in selected_records]
    indices = knn.deterministic_stratified_hash_ranked_patch_indices(
        selected_records,
        [matrix.shape[0] for matrix in matrices],
        SYNTHETIC_ANOMALY_TEST_V1_MAX_PROTOTYPE_PATCHES,
    )
    bank = np.concatenate(matrices, axis=0)
    return bank[indices]


def _score_records(
    records: list[dict[str, Any]],
    features: dict[str, object],
    *,
    prototype_banks: dict[str, object],
) -> list[dict[str, Any]]:
    import numpy as np

    result: list[dict[str, Any]] = []
    for category in SYNTHETIC_ANOMALY_TEST_V1_CATEGORIES:
        category_records = [record for record in records if record["category"] == category]
        if not category_records:
            continue
        query = np.stack([np.asarray(features[str(record["caseId"])], dtype=np.float32) for record in category_records], axis=0)
        try:
            components = knn.patch_knn_scores_blocked(
                query,
                prototype_banks[category],
                top_k=SYNTHETIC_ANOMALY_TEST_V1_TOP_K,
                prototype_block_size=SYNTHETIC_ANOMALY_TEST_V1_BLOCK_SIZE,
            )
        except knn.SuccessorV2EvaluatorError as error:
            raise SyntheticAnomalyTestError(str(error)) from error
        for record, component in zip(category_records, components, strict=True):
            result.append({**record, **component})
    return sorted(result, key=lambda record: str(record["caseId"]))


def _thresholds_from_calibration(scores: list[dict[str, Any]]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for category in SYNTHETIC_ANOMALY_TEST_V1_CATEGORIES:
        values = [_require_finite(record.get("score"), name="raw calibration score") for record in scores if record["category"] == category]
        if len(values) != SYNTHETIC_ANOMALY_TEST_V1_PARENT_COUNTS["SYNTHETIC_CALIBRATION"]:
            raise SyntheticAnomalyTestError("synthetic-only calibration score count is inconsistent")
        thresholds[category] = max(values)
    return thresholds


def synthetic_confusion_metrics(records: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    """Compute fixed-threshold synthetic-only TP/FP/FN/TN without defaults."""

    threshold = _require_finite(threshold, name="synthetic threshold")
    tp = fp = fn = tn = 0
    for record in records:
        label = record.get("syntheticLabel")
        score = _require_finite(record.get("score"), name="synthetic query score")
        if label not in {"SYNTHETIC_ANOMALY", "SYNTHETIC_NOMINAL"}:
            raise SyntheticAnomalyTestError("synthetic query label is unsupported")
        predicted = score > threshold
        if label == "SYNTHETIC_ANOMALY":
            if predicted:
                tp += 1
            else:
                fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    result = {
        "syntheticTP": tp,
        "syntheticFP": fp,
        "syntheticFN": fn,
        "syntheticTN": tn,
        "syntheticPrecision": precision,
        "syntheticRecall": recall,
        "syntheticF1": f1,
    }
    _require_exact_fields(result, name="synthetic metrics", fields=METRIC_FIELDS)
    return result


def _score_output_record(record: dict[str, Any], *, input_kind: str) -> dict[str, Any]:
    family = record.get("syntheticDefectFamily") if input_kind == "SYNTHETIC_ANOMALY" else None
    output = {
        "caseId": record["caseId"],
        "parentCaseId": record["parentCaseId"],
        "category": record["category"],
        "sourceSha256": record["sourceSha256"],
        "inputKind": input_kind,
        "syntheticDefectFamily": family,
        "score": float(record["score"]),
        "maxPatchDistance": float(record["maxPatchDistance"]),
        "meanNearestPatchDistance": float(record["meanNearestPatchDistance"]),
    }
    _require_exact_fields(output, name="synthetic query score", fields=QUERY_SCORE_FIELDS)
    return output


def _category_report(
    category: str,
    *,
    threshold: float,
    calibration_scores: list[dict[str, Any]],
    query_records: list[dict[str, Any]],
    prototype_patch_count: int,
) -> dict[str, Any]:
    category_calibration = [record for record in calibration_scores if record["category"] == category]
    category_query = [record for record in query_records if record["category"] == category]
    metrics = synthetic_confusion_metrics(category_query, threshold=threshold)
    by_family = {
        family: synthetic_confusion_metrics(
            [record for record in category_query if record["syntheticLabel"] == "SYNTHETIC_NOMINAL" or record.get("syntheticDefectFamily") == family],
            threshold=threshold,
        )
        for family in ("LOCAL_SCRATCH", "LOCAL_SPOT", "LOCAL_OCCLUSION")
    }
    return {
        "prototypeParentCount": SYNTHETIC_ANOMALY_TEST_V1_PARENT_COUNTS["SYNTHETIC_PROTOTYPE"],
        "calibrationParentCount": len(category_calibration),
        "queryNominalParentCount": sum(record["syntheticLabel"] == "SYNTHETIC_NOMINAL" for record in category_query),
        "querySyntheticAnomalyCount": sum(record["syntheticLabel"] == "SYNTHETIC_ANOMALY" for record in category_query),
        "prototypePatchCount": prototype_patch_count,
        "thresholdFromRawCalibration": threshold,
        "syntheticMetrics": metrics,
        "syntheticMetricsByFamily": by_family,
    }


def run_synthetic_anomaly_test(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    augmentation_manifest_path: Path,
    output_path: Path,
    *,
    source_root: Path,
    recipe_path: Path,
    model_repo: Path,
    model_weights: Path,
    device: str = "cpu",
    repository_root: Path = REPOSITORY_ROOT,
    embedder_factory: Callable[..., Any] = knn.SuccessorV2BatchEmbedder,
    identity_factory: Callable[..., dict[str, Any]] = _feature_extractor_identity,
) -> dict[str, Any]:
    """Run one external synthetic-only rendering-discrimination test.

    The function never receives a path for any held-out normal, blind, true
    anomaly, or mask source.  A raw FIT-only parent split and its threshold
    are fixed before synthetic query labels are used for metrics.
    """

    if device != "cpu":
        raise SyntheticAnomalyTestError("synthetic-only test supports CPU only")
    try:
        prepared_output = knn._require_external_output(output_path, repository_root=repository_root)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticAnomalyTestError(str(error)) from error
    # This test may run on a workstation which is suspended between tool
    # invocations.  Process CPU time keeps the audit timings meaningful and
    # avoids reporting suspend time as image processing time.
    started = time.process_time()
    timings = {
        "inputAssemblySeconds": 0.0,
        "provenanceSeconds": 0.0,
        "featureProcessingSeconds": 0.0,
        "scoringSeconds": 0.0,
        "totalElapsedSeconds": 0.0,
    }
    input_started = time.process_time()
    try:
        envelope, envelope_file_sha256, raw_parents = successor.load_successor_safe_normal_inputs(
            parent_holdout_path,
            parent_selection_contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            partitions={"FIT"},
            repository_root=repository_root,
        )
    except successor.FreshNormalSuccessorError as error:
        raise SyntheticAnomalyTestError(str(error)) from error
    parents = _fit_parent_records(raw_parents)
    split = split_synthetic_fit_parents(parents)
    augmentation_manifest, augmentation_file_sha256, augmented_records = augmentation.load_validated_synthetic_anomaly_augmentations(
        augmentation_manifest_path,
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=recipe_path,
        repository_root=repository_root,
    )
    synthetic_queries = _adapt_synthetic_query_records(augmented_records, query_parents=split["SYNTHETIC_QUERY"])
    timings["inputAssemblySeconds"] += time.process_time() - input_started

    provenance_started = time.process_time()
    extractor_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    extractor_identity_sha256 = successor.canonical_json_sha256(extractor_identity)
    timings["provenanceSeconds"] += time.process_time() - provenance_started
    embedder = embedder_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    provenance_started = time.process_time()
    loaded_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    timings["provenanceSeconds"] += time.process_time() - provenance_started
    if loaded_identity != extractor_identity:
        raise SyntheticAnomalyTestError("feature extractor changed while DINO loaded")

    raw_feature_records = [_adapt_raw(record) for record in parents]
    raw_features = _extract_features(raw_feature_records, embedder=embedder, timings=timings)
    prototype_records = [_adapt_raw(record) for record in split["SYNTHETIC_PROTOTYPE"]]
    calibration_records = [_adapt_raw(record) for record in split["SYNTHETIC_CALIBRATION"]]
    query_nominal_records = [
        {**_adapt_raw(record), "parentCaseId": record["caseId"], "syntheticLabel": "SYNTHETIC_NOMINAL"}
        for record in split["SYNTHETIC_QUERY"]
    ]
    prototype_banks = {
        category: _prototype_bank(prototype_records, raw_features, category=category)
        for category in SYNTHETIC_ANOMALY_TEST_V1_CATEGORIES
    }
    scoring_started = time.process_time()
    raw_calibration_components = _score_records(calibration_records, raw_features, prototype_banks=prototype_banks)
    thresholds = _thresholds_from_calibration(raw_calibration_components)
    raw_query_components = _score_records(query_nominal_records, raw_features, prototype_banks=prototype_banks)
    synthetic_feature_records = [
        {
            "caseId": record["caseId"],
            "category": record["category"],
            "partition": "SYNTHETIC_QUERY",
            "kind": "SYNTHETIC",
            "defect": "synthetic",
            "sourceSha256": record["sourceSha256"],
            "sourceGroupId": next(
                parent["sourceGroupId"] for parent in split["SYNTHETIC_QUERY"] if parent["caseId"] == record["parentCaseId"]
            ),
            "imagePath": record["imagePath"],
            "parentCaseId": record["parentCaseId"],
            "syntheticDefectFamily": record["syntheticDefectFamily"],
        }
        for record in synthetic_queries
    ]
    synthetic_features = _extract_features(synthetic_feature_records, embedder=embedder, timings=timings)
    synthetic_components = _score_records(synthetic_feature_records, synthetic_features, prototype_banks=prototype_banks)
    timings["scoringSeconds"] += time.process_time() - scoring_started

    calibration_scores = [
        {
            "caseId": record["caseId"],
            "category": record["category"],
            "sourceSha256": record["sourceSha256"],
            "score": float(record["score"]),
            "maxPatchDistance": float(record["maxPatchDistance"]),
            "meanNearestPatchDistance": float(record["meanNearestPatchDistance"]),
        }
        for record in raw_calibration_components
    ]
    if any(set(record) != CALIBRATION_SCORE_FIELDS for record in calibration_scores):  # pragma: no cover - invariant guard
        raise SyntheticAnomalyTestError("raw calibration score shape is unsafe")

    # The threshold and every image score exist at this point.  Only now join
    # known renderer labels to build the synthetic-only confusion matrices.
    query_records: list[dict[str, Any]] = []
    for record in raw_query_components:
        query_records.append({
            **record,
            "parentCaseId": record["caseId"],
            "syntheticLabel": "SYNTHETIC_NOMINAL",
            "syntheticDefectFamily": None,
        })
    family_by_case = {record["caseId"]: record["syntheticDefectFamily"] for record in synthetic_feature_records}
    parent_by_case = {record["caseId"]: record["parentCaseId"] for record in synthetic_feature_records}
    for record in synthetic_components:
        query_records.append({
            **record,
            "parentCaseId": parent_by_case[record["caseId"]],
            "syntheticLabel": "SYNTHETIC_ANOMALY",
            "syntheticDefectFamily": family_by_case[record["caseId"]],
        })
    query_records.sort(key=lambda record: str(record["caseId"]))
    query_scores = [
        _score_output_record(record, input_kind="SYNTHETIC_ANOMALY" if record["syntheticLabel"] == "SYNTHETIC_ANOMALY" else "RAW_QUERY_NOMINAL")
        for record in query_records
    ]
    category_reports = {
        category: _category_report(
            category,
            threshold=thresholds[category],
            calibration_scores=calibration_scores,
            query_records=query_records,
            prototype_patch_count=int(prototype_banks[category].shape[0]),
        )
        for category in SYNTHETIC_ANOMALY_TEST_V1_CATEGORIES
    }
    aggregate = synthetic_confusion_metrics(query_records, threshold=0.0)
    # Aggregate must apply the relevant category threshold, rather than a
    # meaningless shared scalar.  Build it from fixed per-category predictions.
    aggregate_rows = [
        {**record, "score": 1.0 if record["score"] > thresholds[record["category"]] else 0.0}
        for record in query_records
    ]
    aggregate = synthetic_confusion_metrics(aggregate_rows, threshold=0.5)

    provenance_started = time.process_time()
    completed_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    timings["provenanceSeconds"] += time.process_time() - provenance_started
    if completed_identity != extractor_identity:
        raise SyntheticAnomalyTestError("feature extractor changed while synthetic-only test ran")
    timings["totalElapsedSeconds"] = time.process_time() - started
    parent_evidence = envelope.get("parentEvidence")
    if not isinstance(parent_evidence, dict) or not isinstance(envelope.get("successorPartitionIdentities"), dict):
        raise SyntheticAnomalyTestError("successor envelope is missing its closed parent evidence")
    split_records = _parent_split_records(split)
    document: dict[str, Any] = {
        "schemaVersion": SYNTHETIC_ANOMALY_TEST_V1_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "purpose": SYNTHETIC_ANOMALY_TEST_V1_PURPOSE,
        "phase": SYNTHETIC_ANOMALY_TEST_V1_PHASE,
        "metricScope": SYNTHETIC_ANOMALY_TEST_V1_METRIC_SCOPE,
        "realAnomalyPerformance": SYNTHETIC_ANOMALY_TEST_V1_REAL_PERFORMANCE,
        "forbiddenUses": list(SYNTHETIC_ANOMALY_TEST_V1_FORBIDDEN_USES),
        "inputPolicy": SYNTHETIC_ANOMALY_TEST_V1_INPUT_POLICY,
        "blindPolicy": augmentation.SYNTHETIC_ANOMALY_BLIND_POLICY,
        "resultLabel": SYNTHETIC_ANOMALY_TEST_V1_RESULT_LABEL,
        "testConfiguration": {
            "algorithmId": SYNTHETIC_ANOMALY_TEST_V1_ALGORITHM,
            "splitAlgorithm": SYNTHETIC_ANOMALY_TEST_V1_SPLIT_ALGORITHM,
            "parentCountsPerCategory": dict(SYNTHETIC_ANOMALY_TEST_V1_PARENT_COUNTS),
            "maxPrototypePatches": SYNTHETIC_ANOMALY_TEST_V1_MAX_PROTOTYPE_PATCHES,
            "topKMostAnomalousPatches": SYNTHETIC_ANOMALY_TEST_V1_TOP_K,
            "prototypeBlockSize": SYNTHETIC_ANOMALY_TEST_V1_BLOCK_SIZE,
            "batchSize": SYNTHETIC_ANOMALY_TEST_V1_BATCH_SIZE,
            "decisionRule": SYNTHETIC_ANOMALY_TEST_V1_DECISION_RULE,
            "thresholdEstablishedBeforeQueryScoring": True,
            "queryPairing": "RAW_QUERY_PARENT_AND_SYNTHETIC_CHILDREN_ARE_NOT_INDEPENDENT",
        },
        "parentHoldoutManifestFileSha256": parent_evidence.get("holdoutManifestFileSha256"),
        "parentHoldoutManifestDeclaredSha256": parent_evidence.get("holdoutManifestDeclaredSha256"),
        "parentSelectionContractFileSha256": parent_evidence.get("selectionContractFileSha256"),
        "parentSelectionContractDeclaredSha256": parent_evidence.get("selectionContractDeclaredSha256"),
        "successorSealFileSha256": envelope.get("sealFileSha256"),
        "successorSealDeclaredSha256": envelope.get("sealDeclaredSha256"),
        "successorPlanFileSha256": envelope.get("planFileSha256"),
        "successorPlanDeclaredSha256": envelope.get("planDeclaredSha256"),
        "successorEnvelopeFileSha256": envelope_file_sha256,
        "successorEnvelopeDeclaredSha256": envelope.get("successorEnvelopeSha256"),
        "successorFitIdentitySha256": envelope["successorPartitionIdentities"].get("FIT"),
        "parentNormalConfirmationIdentitySha256": parent_evidence.get("parentNormalConfirmationIdentitySha256"),
        "augmentationManifestFileSha256": augmentation_file_sha256,
        "augmentationManifestDeclaredSha256": augmentation_manifest.get("augmentationManifestSha256"),
        "recipeFileSha256": augmentation_manifest.get("recipeFileSha256"),
        "parentSplit": split_records,
        "parentSplitIdentitySha256": successor.canonical_json_sha256(split_records),
        "featureExtractor": extractor_identity,
        "featureExtractorIdentitySha256": extractor_identity_sha256,
        "thresholds": thresholds,
        "calibrationScores": sorted(calibration_scores, key=lambda record: str(record["caseId"])),
        "queryScores": query_scores,
        "categories": category_reports,
        "aggregate": aggregate,
        "execution": _execution_metadata(timings, repository_root=repository_root),
    }
    for name in (
        "parentHoldoutManifestFileSha256", "parentHoldoutManifestDeclaredSha256", "parentSelectionContractFileSha256",
        "parentSelectionContractDeclaredSha256", "successorSealFileSha256", "successorSealDeclaredSha256",
        "successorPlanFileSha256", "successorPlanDeclaredSha256", "successorEnvelopeFileSha256",
        "successorEnvelopeDeclaredSha256", "successorFitIdentitySha256", "parentNormalConfirmationIdentitySha256",
        "augmentationManifestFileSha256", "augmentationManifestDeclaredSha256", "recipeFileSha256",
        "parentSplitIdentitySha256", "featureExtractorIdentitySha256",
    ):
        document[name] = _require_sha256(document[name], name=f"synthetic-only report {name}")
    _require_exact_fields(document, name="synthetic-only test report", fields=REPORT_FIELDS.difference({"syntheticTestReportSha256"}))
    document["syntheticTestReportSha256"] = _document_digest(document)
    _require_exact_fields(document, name="synthetic-only test report", fields=REPORT_FIELDS)
    # Re-use the previously prepared path: it has been checked before any
    # source image is opened and is still guaranteed new-only by ``xb``.
    _write_external_report(prepared_output, document, repository_root=repository_root)
    return document
