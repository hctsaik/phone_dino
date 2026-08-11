"""V2 synthetic-stimulus response evaluator for offline MVTec research.

This harness is deliberately separate from the frozen V1 synthetic report and
the normal-only successor selection workflow.  It opens raw bytes only through
the successor FIT-only loader, fixes the 6/2/4 parent split before package
loading, establishes raw-calibration thresholds before any V2 child is loaded
or scored, and evaluates only deterministic rendered stimuli.

The resulting report is an engineering response trace, not real-anomaly
evidence.  It must not be used to select models, thresholds, algorithms, or
packages, and cannot support production or physical-qualification claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from phone_dino import mvtec_normal_successor as successor
from phone_dino import mvtec_successor_evaluator_v2 as knn
from phone_dino import mvtec_synthetic_anomaly_stress_v2 as augmentation


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_STRESS_V2_REPORT_SCHEMA = "phone-dino.mvtec-ad-synthetic-stimulus-response-report/2.0"
SYNTHETIC_STRESS_V2_PURPOSE = "OFFLINE_MVTEC_SYNTHETIC_STIMULUS_AUGMENTATION_AND_RESPONSE_TESTING"
SYNTHETIC_STRESS_V2_PHASE = "POST_V1_SYNTHETIC_STIMULUS_STRESS_RESPONSE"
SYNTHETIC_STRESS_V2_METRIC_SCOPE = "SYNTHETIC_STIMULUS_RESPONSE_ONLY"
SYNTHETIC_STRESS_V2_INPUT_POLICY = "SUCCESSOR_V2_FIT_RAW_NORMAL_PARENTS_ONLY"
SYNTHETIC_STRESS_V2_BLIND_POLICY = "NO_BLIND_OR_TRUE_ANOMALY_DATA"
SYNTHETIC_STRESS_V2_RESULT_LABEL = "SYNTHETIC_ONLY_NOT_REAL_ANOMALY_PERFORMANCE"
SYNTHETIC_STRESS_V2_REAL_PERFORMANCE = "NOT_ESTIMATED"
SYNTHETIC_STRESS_V2_FORBIDDEN_USES = [
    "MODEL_SELECTION",
    "ALGORITHM_SELECTION",
    "HYPERPARAMETER_SELECTION",
    "THRESHOLD_SELECTION",
    "PACKAGE_COMPARISON_OR_PROMOTION",
    "PRODUCTION_VALIDATION",
    "PHYSICAL_QUALIFICATION",
]
SYNTHETIC_STRESS_V2_CATEGORIES = ("capsule", "metal_nut", "tile")
SYNTHETIC_STRESS_V2_FAMILIES = ("LOCAL_SCRATCH", "LOCAL_SPOT", "LOCAL_OCCLUSION")
SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS = ("SUBTLE", "MODERATE", "PRONOUNCED")
SYNTHETIC_STRESS_V2_PARENT_SPLIT_ALGORITHM = "SOURCE_SHA256_THEN_CASE_ID_PER_CATEGORY_V1"
SYNTHETIC_STRESS_V2_PARENT_COUNTS = {
    "SYNTHETIC_PROTOTYPE": 6,
    "SYNTHETIC_CALIBRATION": 2,
    "SYNTHETIC_QUERY": 4,
}
SYNTHETIC_STRESS_V2_DECISION_RULE = "SCORE_STRICTLY_GREATER_THAN_RAW_CALIBRATION_MAX_V2"
SYNTHETIC_STRESS_V2_ALGORITHM = "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_SYNTHETIC_STRESS_V2"
SYNTHETIC_STRESS_V2_PROTOTYPE_SELECTION = "DETERMINISTIC_STRATIFIED_HASH_RANKED_PATCH_PREFIX_V2"
SYNTHETIC_STRESS_V2_MAX_PROTOTYPE_PATCHES = 1024
SYNTHETIC_STRESS_V2_TOP_K = 5
SYNTHETIC_STRESS_V2_BLOCK_SIZE = 256
SYNTHETIC_STRESS_V2_BATCH_SIZE = 4

REPORT_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "syntheticOnly",
    "postV1Exploratory",
    "comparisonOrPromotionAllowed",
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
    "augmentationSchemaVersion",
    "recipeFileSha256",
    "parentSplit",
    "parentSplitIdentitySha256",
    "featureExtractor",
    "featureExtractorIdentitySha256",
    "thresholds",
    "calibrationScores",
    "rawQueryScores",
    "stimulusScores",
    "categories",
    "aggregate",
    "execution",
    "syntheticStressReportSha256",
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
RAW_QUERY_SCORE_FIELDS = {
    "caseId",
    "parentCaseId",
    "category",
    "sourceSha256",
    "score",
    "maxPatchDistance",
    "meanNearestPatchDistance",
    "aboveRawCalibrationThreshold",
}
STIMULUS_SCORE_FIELDS = {
    "caseId",
    "parentCaseId",
    "category",
    "sourceSha256",
    "syntheticDefectFamily",
    "renderIntensityLevel",
    "score",
    "maxPatchDistance",
    "meanNearestPatchDistance",
    "aboveRawCalibrationThreshold",
}
CHILD_RECORD_FIELDS = {
    "caseId",
    "parentCaseId",
    "parentSourceSha256",
    "sourceGroupId",
    "category",
    "parentPartition",
    "syntheticTestRole",
    "syntheticLabel",
    "syntheticDefectFamily",
    "renderIntensityLevel",
    "variantId",
    "relativePath",
    "sourceSha256",
    "parameters",
    "outputEncoding",
    "imagePath",
}


class SyntheticStressV2Error(ValueError):
    """Raised when the V2 synthetic-stimulus boundary is not closed."""


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without retaining the entire file in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SyntheticStressV2Error(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise SyntheticStressV2Error(f"{name} must be a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise SyntheticStressV2Error(f"{name} must be a SHA-256 digest") from error
    return digest


def _require_finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SyntheticStressV2Error(f"{name} must be finite")
    return float(value)


def _require_exact_fields(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SyntheticStressV2Error(f"{name} must be an object")
    missing = fields.difference(value)
    unknown = set(value).difference(fields)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unsupported {', '.join(sorted(unknown))}")
        raise SyntheticStressV2Error(f"{name} has {'; '.join(detail)} fields")
    return value


def _document_digest(document: dict[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("syntheticStressReportSha256", None)
    return successor.canonical_json_sha256(unsigned)


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat(follow_symlinks=False)
    return status.st_dev, status.st_ino, status.st_mode, status.st_size


def write_response_only_report(path: Path, document: dict[str, Any], *, repository_root: Path) -> Path:
    """Write a new immutable external response-only report slot."""

    try:
        prepared = knn._require_external_output(path, repository_root=repository_root)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticStressV2Error(str(error)) from error
    data = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with prepared.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            fd_signature = os.fstat(stream.fileno())
    except OSError as error:
        raise SyntheticStressV2Error("unable to write synthetic-stimulus response report") from error
    try:
        if knn._is_link_or_reparse_point(prepared) or _stat_signature(prepared) != (
            fd_signature.st_dev,
            fd_signature.st_ino,
            fd_signature.st_mode,
            fd_signature.st_size,
        ):
            raise SyntheticStressV2Error("synthetic-stimulus response report changed while it was written")
    except OSError as error:
        raise SyntheticStressV2Error("unable to verify synthetic-stimulus response report output") from error
    return prepared


def _feature_extractor_identity(*, model_repo: Path, model_weights: Path, device: str) -> dict[str, Any]:
    """Bind model, implementation, and runtime bytes used by this harness."""

    if device != "cpu":
        raise SyntheticStressV2Error("synthetic-stimulus response test supports CPU only")
    try:
        import numpy as np
        import torch
        import torchvision
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SyntheticStressV2Error("synthetic-stimulus response test requires numpy, torch, and torchvision") from error
    try:
        model_weights_sha256 = knn._immutable_file_sha256(model_weights, description="model weights")
        model_repository_sha256 = knn.sha256_directory(model_repo)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticStressV2Error(str(error)) from error
    source_paths = {
        "syntheticStressEvaluatorModuleSha256": Path(__file__),
        "syntheticStressAugmentationModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "mvtec_synthetic_anomaly_stress_v2.py",
        "successorModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "mvtec_normal_successor.py",
        "patchKnnModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "mvtec_successor_evaluator_v2.py",
        "productionModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "production.py",
        "enginesModuleSha256": REPOSITORY_ROOT / "src" / "phone_dino" / "engines.py",
    }
    return {
        "schemaVersion": "phone-dino.mvtec-ad-synthetic-stimulus-feature-extractor/2.0",
        "modelWeightsSha256": model_weights_sha256,
        "modelRepositorySha256": model_repository_sha256,
        "modelEntrypoint": "dinov2_vits14",
        "device": device,
        "preprocessingId": knn.SUCCESSOR_V2_PREPROCESSING_ID,
        "algorithmId": SYNTHETIC_STRESS_V2_ALGORITHM,
        "prototypeSelection": SYNTHETIC_STRESS_V2_PROTOTYPE_SELECTION,
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
        "syntheticStressEvaluatorModuleSha256": sha256_file(Path(__file__)),
        "entrypointSha256": sha256_file(repository_root / "tools" / "run_mvtec_ad_synthetic_stress_v2.py"),
        "timingBasis": "PROCESS_CPU_TIME_EXCLUDES_SUSPEND",
        "phaseTimingsSeconds": {name: round(value, 6) for name, value in timings.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "gitRevision": revision,
    }


def _fit_parent_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fail closed unless every input is a unique raw nominal FIT parent."""

    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_groups: set[str] = set()
    for record in records:
        if record.get("partition") != "FIT" or record.get("kind") != "NOMINAL" or record.get("defect") != "good":
            raise SyntheticStressV2Error("synthetic-stimulus test may open raw nominal FIT parents only")
        case_id = _require_string(record.get("caseId"), name="FIT parent caseId")
        source = _require_sha256(record.get("sourceSha256"), name="FIT parent sourceSha256")
        group = _require_string(record.get("sourceGroupId"), name="FIT parent sourceGroupId")
        category = _require_string(record.get("category"), name="FIT parent category")
        if category not in SYNTHETIC_STRESS_V2_CATEGORIES:
            raise SyntheticStressV2Error("synthetic-stimulus test received an unsupported category")
        if not isinstance(record.get("imagePath"), Path):
            raise SyntheticStressV2Error("FIT parent was not loaded through the phase-safe loader")
        if case_id in seen_case_ids or source in seen_sources or group in seen_groups:
            raise SyntheticStressV2Error("FIT parents must have unique case, content, and source-group identities")
        seen_case_ids.add(case_id)
        seen_sources.add(source)
        seen_groups.add(group)
        result.append(dict(record))
    if len(result) != 36:
        raise SyntheticStressV2Error("synthetic-stimulus test requires exactly 36 successor FIT parents")
    return sorted(result, key=lambda record: str(record["caseId"]))


def load_safe_v2_fit_inputs(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Open raw normal data only through the closed successor FIT boundary."""

    try:
        envelope, envelope_file_sha256, records = successor.load_successor_safe_normal_inputs(
            parent_holdout_path,
            parent_selection_contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            partitions={"FIT"},
            repository_root=repository_root,
        )
    except successor.FreshNormalSuccessorError as error:
        raise SyntheticStressV2Error(str(error)) from error
    return envelope, envelope_file_sha256, _fit_parent_records(records)


def build_fixed_parent_split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Create the deterministic, label-free 6/2/4 parent split per category."""

    parents = _fit_parent_records(records)
    result = {role: [] for role in SYNTHETIC_STRESS_V2_PARENT_COUNTS}
    for category in SYNTHETIC_STRESS_V2_CATEGORIES:
        category_parents = sorted(
            (record for record in parents if record["category"] == category),
            key=lambda record: (str(record["sourceSha256"]), str(record["caseId"])),
        )
        if len(category_parents) != sum(SYNTHETIC_STRESS_V2_PARENT_COUNTS.values()):
            raise SyntheticStressV2Error("synthetic-stimulus parent split needs exactly 12 FIT parents per category")
        boundary = SYNTHETIC_STRESS_V2_PARENT_COUNTS["SYNTHETIC_PROTOTYPE"]
        second_boundary = boundary + SYNTHETIC_STRESS_V2_PARENT_COUNTS["SYNTHETIC_CALIBRATION"]
        for index, record in enumerate(category_parents):
            role = (
                "SYNTHETIC_PROTOTYPE"
                if index < boundary
                else "SYNTHETIC_CALIBRATION"
                if index < second_boundary
                else "SYNTHETIC_QUERY"
            )
            result[role].append({**record, "syntheticStressRole": role})
    groups_by_role = {role: {record["sourceGroupId"] for record in values} for role, values in result.items()}
    if any(
        groups_by_role[left].intersection(groups_by_role[right])
        for left in groups_by_role
        for right in groups_by_role
        if left < right
    ):
        raise SyntheticStressV2Error("synthetic-stimulus split reuses a source group across roles")
    for role, expected_per_category in SYNTHETIC_STRESS_V2_PARENT_COUNTS.items():
        for category in SYNTHETIC_STRESS_V2_CATEGORIES:
            if sum(record["category"] == category for record in result[role]) != expected_per_category:
                raise SyntheticStressV2Error("synthetic-stimulus parent split count is inconsistent")
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
        raise SyntheticStressV2Error("synthetic-stimulus parent split record shape is unsafe")
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


def _extract_features(
    records: list[dict[str, Any]], *, embedder: Any, timings: dict[str, float]
) -> dict[str, object]:
    started = time.process_time()
    try:
        result = knn._extract_patch_features(
            records,
            embedder=embedder,
            batch_size=SYNTHETIC_STRESS_V2_BATCH_SIZE,
            timings={"inputVerificationSeconds": 0.0, "featureInferenceSeconds": 0.0},
        )
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticStressV2Error(str(error)) from error
    timings["featureProcessingSeconds"] += time.process_time() - started
    return result


def _prototype_bank(records: list[dict[str, Any]], features: dict[str, object], *, category: str) -> object:
    import numpy as np

    selected_records = [record for record in records if record["category"] == category]
    if len(selected_records) != SYNTHETIC_STRESS_V2_PARENT_COUNTS["SYNTHETIC_PROTOTYPE"]:
        raise SyntheticStressV2Error("synthetic-stimulus prototype count is inconsistent")
    selected_records.sort(key=lambda record: str(record["caseId"]))
    matrices = [np.asarray(features[str(record["caseId"])], dtype=np.float32) for record in selected_records]
    indices = knn.deterministic_stratified_hash_ranked_patch_indices(
        selected_records,
        [matrix.shape[0] for matrix in matrices],
        SYNTHETIC_STRESS_V2_MAX_PROTOTYPE_PATCHES,
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
    for category in SYNTHETIC_STRESS_V2_CATEGORIES:
        category_records = [record for record in records if record["category"] == category]
        if not category_records:
            continue
        query = np.stack(
            [np.asarray(features[str(record["caseId"])], dtype=np.float32) for record in category_records],
            axis=0,
        )
        try:
            components = knn.patch_knn_scores_blocked(
                query,
                prototype_banks[category],
                top_k=SYNTHETIC_STRESS_V2_TOP_K,
                prototype_block_size=SYNTHETIC_STRESS_V2_BLOCK_SIZE,
            )
        except knn.SuccessorV2EvaluatorError as error:
            raise SyntheticStressV2Error(str(error)) from error
        for record, component in zip(category_records, components, strict=True):
            result.append({**record, **component})
    return sorted(result, key=lambda record: str(record["caseId"]))


def calibrate_raw_thresholds(scores: list[dict[str, Any]]) -> dict[str, float]:
    """Fix one raw-calibration maximum per category before package access."""

    thresholds: dict[str, float] = {}
    for category in SYNTHETIC_STRESS_V2_CATEGORIES:
        values = [
            _require_finite(record.get("score"), name="raw calibration score")
            for record in scores
            if record.get("category") == category
        ]
        if len(values) != SYNTHETIC_STRESS_V2_PARENT_COUNTS["SYNTHETIC_CALIBRATION"]:
            raise SyntheticStressV2Error("synthetic-stimulus calibration score count is inconsistent")
        thresholds[category] = max(values)
    return thresholds


def _require_manifest_scope(manifest: dict[str, Any]) -> None:
    expected = {
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "postV1Exploratory": True,
        "comparisonOrPromotionAllowed": False,
        "parentPartition": "FIT",
        "inputPolicy": SYNTHETIC_STRESS_V2_INPUT_POLICY,
        "blindPolicy": SYNTHETIC_STRESS_V2_BLIND_POLICY,
        "resultLabel": SYNTHETIC_STRESS_V2_RESULT_LABEL,
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise SyntheticStressV2Error("V2 synthetic-stimulus package scope is unsafe")
    if manifest.get("schemaVersion") != "phone-dino.mvtec-ad-synthetic-only-stress-augmentation/2.0":
        raise SyntheticStressV2Error("V2 synthetic-stimulus package schema is unsupported")
    if manifest.get("parentSplitAlgorithm") != SYNTHETIC_STRESS_V2_PARENT_SPLIT_ALGORITHM:
        raise SyntheticStressV2Error("V2 synthetic-stimulus package parent split algorithm is unsupported")
    if manifest.get("parentSplitCountsPerCategory") != {"PROTOTYPE": 6, "CALIBRATION": 2, "QUERY": 4}:
        raise SyntheticStressV2Error("V2 synthetic-stimulus package parent split counts are unsupported")
    if manifest.get("variantsPerParent") != len(SYNTHETIC_STRESS_V2_FAMILIES) * len(SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS):
        raise SyntheticStressV2Error("V2 synthetic-stimulus package variant count is unsupported")
    _require_sha256(manifest.get("augmentationManifestSha256"), name="V2 package manifest declared digest")
    _require_sha256(manifest.get("recipeFileSha256"), name="V2 package recipe digest")


def load_and_validate_v2_package(
    augmentation_manifest_path: Path,
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    recipe_path: Path,
    query_parents: list[dict[str, Any]],
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Load a V2 package only after raw-calibration thresholds are fixed.

    The generator performs byte-level re-render validation.  This evaluator
    additionally binds every returned child to the independently recreated
    fixed query-parent split and requires all family/level combinations.
    """

    try:
        manifest, manifest_file_sha256, records = augmentation.load_validated_synthetic_stress_v2(
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
        raise SyntheticStressV2Error(str(error)) from error
    if not isinstance(manifest, dict) or not isinstance(records, list):
        raise SyntheticStressV2Error("V2 synthetic-stimulus package loader returned an unsafe value")
    _require_manifest_scope(manifest)
    _require_sha256(manifest_file_sha256, name="V2 package manifest file digest")

    parent_by_case = {str(record["caseId"]): record for record in query_parents}
    if len(parent_by_case) != len(query_parents) or len(query_parents) != 12:
        raise SyntheticStressV2Error("V2 synthetic-stimulus query-parent split is inconsistent")
    expected = {
        (str(parent["caseId"]), family, level)
        for parent in query_parents
        for family in SYNTHETIC_STRESS_V2_FAMILIES
        for level in SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS
    }
    seen_case_ids: set[str] = set()
    seen_variants_by_parent: dict[str, set[int]] = {case_id: set() for case_id in parent_by_case}
    found: set[tuple[str, str, str]] = set()
    validated: list[dict[str, Any]] = []
    for raw_record in records:
        record = _require_exact_fields(raw_record, name="V2 synthetic-stimulus child", fields=CHILD_RECORD_FIELDS)
        case_id = _require_string(record.get("caseId"), name="V2 child caseId")
        if case_id in seen_case_ids:
            raise SyntheticStressV2Error("V2 synthetic-stimulus child caseId is duplicated")
        seen_case_ids.add(case_id)
        parent_case_id = _require_string(record.get("parentCaseId"), name="V2 child parentCaseId")
        parent = parent_by_case.get(parent_case_id)
        if parent is None:
            raise SyntheticStressV2Error("V2 synthetic-stimulus child does not belong to a fixed query parent")
        family = record.get("syntheticDefectFamily")
        level = record.get("renderIntensityLevel")
        if family not in SYNTHETIC_STRESS_V2_FAMILIES or level not in SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS:
            raise SyntheticStressV2Error("V2 synthetic-stimulus child family or render level is unsupported")
        key = (parent_case_id, family, level)
        if key in found:
            raise SyntheticStressV2Error("V2 synthetic-stimulus child parent/family/level is duplicated")
        found.add(key)
        if (
            _require_sha256(record.get("parentSourceSha256"), name="V2 child parentSourceSha256") != parent["sourceSha256"]
            or record.get("sourceGroupId") != parent["sourceGroupId"]
            or record.get("category") != parent["category"]
            or record.get("parentPartition") != "FIT"
            or record.get("syntheticTestRole") != "QUERY"
            or record.get("syntheticLabel") != "SYNTHETIC_STIMULUS"
            or not isinstance(record.get("imagePath"), Path)
        ):
            raise SyntheticStressV2Error("V2 synthetic-stimulus child does not bind its FIT query parent")
        _require_sha256(record.get("sourceSha256"), name="V2 child sourceSha256")
        variant_id = record.get("variantId")
        if not isinstance(variant_id, int) or isinstance(variant_id, bool) or variant_id < 1:
            raise SyntheticStressV2Error("V2 synthetic-stimulus child variantId is invalid")
        if variant_id in seen_variants_by_parent[parent_case_id]:
            raise SyntheticStressV2Error("V2 synthetic-stimulus child variantId is duplicated per parent")
        seen_variants_by_parent[parent_case_id].add(variant_id)
        validated.append(dict(record))
    if found != expected:
        raise SyntheticStressV2Error("V2 synthetic-stimulus package does not cover every query parent, family, and render level")
    expected_variant_ids = set(range(1, len(SYNTHETIC_STRESS_V2_FAMILIES) * len(SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS) + 1))
    if any(variant_ids != expected_variant_ids for variant_ids in seen_variants_by_parent.values()):
        raise SyntheticStressV2Error("V2 synthetic-stimulus package variants do not form the fixed 1..9 sequence")
    return manifest, manifest_file_sha256, sorted(validated, key=lambda record: str(record["caseId"]))


def _adapt_stimulus_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "partition": "SYNTHETIC_QUERY",
        "kind": "SYNTHETIC",
        "defect": "synthetic",
        "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"],
        "imagePath": record["imagePath"],
        "isAugmentation": False,
        "parentCaseId": record["parentCaseId"],
        "variantId": record["variantId"],
        "component": record["syntheticDefectFamily"],
        "syntheticDefectFamily": record["syntheticDefectFamily"],
        "renderIntensityLevel": record["renderIntensityLevel"],
    }


def _calibration_output_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {
        "caseId": record["caseId"],
        "category": record["category"],
        "sourceSha256": record["sourceSha256"],
        "score": float(record["score"]),
        "maxPatchDistance": float(record["maxPatchDistance"]),
        "meanNearestPatchDistance": float(record["meanNearestPatchDistance"]),
    }
    _require_exact_fields(output, name="raw calibration score", fields=CALIBRATION_SCORE_FIELDS)
    return output


def evaluate_response_only(
    raw_query_components: list[dict[str, Any]],
    stimulus_components: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind child score responses to their raw query parents without labels."""

    if set(thresholds) != set(SYNTHETIC_STRESS_V2_CATEGORIES):
        raise SyntheticStressV2Error("synthetic-stimulus thresholds do not cover every category")
    raw_by_case: dict[str, dict[str, Any]] = {}
    raw_scores: list[dict[str, Any]] = []
    for record in raw_query_components:
        case_id = _require_string(record.get("caseId"), name="raw query caseId")
        category = _require_string(record.get("category"), name="raw query category")
        if category not in thresholds or case_id in raw_by_case:
            raise SyntheticStressV2Error("raw query score is not a unique configured category record")
        score = _require_finite(record.get("score"), name="raw query score")
        output = {
            "caseId": case_id,
            "parentCaseId": case_id,
            "category": category,
            "sourceSha256": _require_sha256(record.get("sourceSha256"), name="raw query sourceSha256"),
            "score": score,
            "maxPatchDistance": _require_finite(record.get("maxPatchDistance"), name="raw query max patch distance"),
            "meanNearestPatchDistance": _require_finite(record.get("meanNearestPatchDistance"), name="raw query mean patch distance"),
            "aboveRawCalibrationThreshold": score > thresholds[category],
        }
        _require_exact_fields(output, name="raw query score", fields=RAW_QUERY_SCORE_FIELDS)
        raw_by_case[case_id] = output
        raw_scores.append(output)
    expected_raw_count = SYNTHETIC_STRESS_V2_PARENT_COUNTS["SYNTHETIC_QUERY"] * len(SYNTHETIC_STRESS_V2_CATEGORIES)
    if len(raw_scores) != expected_raw_count:
        raise SyntheticStressV2Error("raw query score count is inconsistent")

    stimulus_scores: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for record in stimulus_components:
        case_id = _require_string(record.get("caseId"), name="synthetic-stimulus caseId")
        parent_case_id = _require_string(record.get("parentCaseId"), name="synthetic-stimulus parentCaseId")
        category = _require_string(record.get("category"), name="synthetic-stimulus category")
        parent = raw_by_case.get(parent_case_id)
        if case_id in seen_case_ids or parent is None or parent["category"] != category:
            raise SyntheticStressV2Error("synthetic-stimulus score does not pair to one raw query parent")
        seen_case_ids.add(case_id)
        family = record.get("syntheticDefectFamily")
        level = record.get("renderIntensityLevel")
        if family not in SYNTHETIC_STRESS_V2_FAMILIES or level not in SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS:
            raise SyntheticStressV2Error("synthetic-stimulus score family or render level is unsupported")
        score = _require_finite(record.get("score"), name="synthetic-stimulus score")
        output = {
            "caseId": case_id,
            "parentCaseId": parent_case_id,
            "category": category,
            "sourceSha256": _require_sha256(record.get("sourceSha256"), name="synthetic-stimulus sourceSha256"),
            "syntheticDefectFamily": family,
            "renderIntensityLevel": level,
            "score": score,
            "maxPatchDistance": _require_finite(record.get("maxPatchDistance"), name="synthetic-stimulus max patch distance"),
            "meanNearestPatchDistance": _require_finite(record.get("meanNearestPatchDistance"), name="synthetic-stimulus mean patch distance"),
            "aboveRawCalibrationThreshold": score > thresholds[category],
        }
        _require_exact_fields(output, name="synthetic-stimulus score", fields=STIMULUS_SCORE_FIELDS)
        stimulus_scores.append(output)
        pairs.append({
            "caseId": case_id,
            "parentCaseId": parent_case_id,
            "category": category,
            "syntheticDefectFamily": family,
            "renderIntensityLevel": level,
            "parentScore": parent["score"],
            "childScore": score,
            "childMinusParentScore": score - parent["score"],
        })
    expected_stimuli = expected_raw_count * len(SYNTHETIC_STRESS_V2_FAMILIES) * len(SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS)
    if len(stimulus_scores) != expected_stimuli or len(pairs) != expected_stimuli:
        raise SyntheticStressV2Error("synthetic-stimulus score count is inconsistent")
    return (
        sorted(raw_scores, key=lambda record: str(record["caseId"])),
        sorted(stimulus_scores, key=lambda record: str(record["caseId"])),
        sorted(pairs, key=lambda record: str(record["caseId"])),
    )


def _response_counts(raw_scores: list[dict[str, Any]], stimulus_scores: list[dict[str, Any]]) -> dict[str, int]:
    if not raw_scores or not stimulus_scores:
        raise SyntheticStressV2Error("response counts require raw queries and synthetic stimuli")
    return {
        "rawQueryCount": len(raw_scores),
        "rawQueryAboveThresholdCount": sum(bool(record["aboveRawCalibrationThreshold"]) for record in raw_scores),
        "syntheticStimulusCount": len(stimulus_scores),
        "syntheticStimulusAboveThresholdCount": sum(bool(record["aboveRawCalibrationThreshold"]) for record in stimulus_scores),
    }


def _response_rates(counts: dict[str, int]) -> dict[str, float]:
    raw_count = counts.get("rawQueryCount")
    stimulus_count = counts.get("syntheticStimulusCount")
    if not isinstance(raw_count, int) or raw_count <= 0 or not isinstance(stimulus_count, int) or stimulus_count <= 0:
        raise SyntheticStressV2Error("response rates require positive response counts")
    return {
        "rawQueryAboveThresholdRate": counts["rawQueryAboveThresholdCount"] / raw_count,
        "syntheticStimulusAboveThresholdRate": counts["syntheticStimulusAboveThresholdCount"] / stimulus_count,
    }


def _paired_score_delta_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise SyntheticStressV2Error("paired score-delta summary requires at least one pair")
    values = [_require_finite(pair.get("childMinusParentScore"), name="paired child-minus-parent score") for pair in pairs]
    positive_count = sum(value > 0.0 for value in values)
    zero_count = sum(value == 0.0 for value in values)
    return {
        "pairCount": len(values),
        "meanChildMinusParentScore": statistics.fmean(values),
        "medianChildMinusParentScore": statistics.median(values),
        "minimumChildMinusParentScore": min(values),
        "maximumChildMinusParentScore": max(values),
        "positiveChildMinusParentCount": positive_count,
        "zeroChildMinusParentCount": zero_count,
        "negativeChildMinusParentCount": len(values) - positive_count - zero_count,
        "positiveChildMinusParentRate": positive_count / len(values),
    }


def _condition_report(
    raw_scores: list[dict[str, Any]],
    stimulus_scores: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "responseCounts": _response_counts(raw_scores, stimulus_scores),
        "responseRates": _response_rates(_response_counts(raw_scores, stimulus_scores)),
        "pairedScoreDeltaSummary": _paired_score_delta_summary(pairs),
    }


def _category_report(
    category: str,
    *,
    threshold: float,
    prototype_patch_count: int,
    calibration_scores: list[dict[str, Any]],
    raw_scores: list[dict[str, Any]],
    stimulus_scores: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    category_raw = [record for record in raw_scores if record["category"] == category]
    category_stimuli = [record for record in stimulus_scores if record["category"] == category]
    category_pairs = [record for record in pairs if record["category"] == category]
    by_family = {
        family: _condition_report(
            category_raw,
            [record for record in category_stimuli if record["syntheticDefectFamily"] == family],
            [record for record in category_pairs if record["syntheticDefectFamily"] == family],
        )
        for family in SYNTHETIC_STRESS_V2_FAMILIES
    }
    by_level = {
        level: _condition_report(
            category_raw,
            [record for record in category_stimuli if record["renderIntensityLevel"] == level],
            [record for record in category_pairs if record["renderIntensityLevel"] == level],
        )
        for level in SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS
    }
    by_family_and_level = {
        family: {
            level: _condition_report(
                category_raw,
                [
                    record
                    for record in category_stimuli
                    if record["syntheticDefectFamily"] == family and record["renderIntensityLevel"] == level
                ],
                [
                    record
                    for record in category_pairs
                    if record["syntheticDefectFamily"] == family and record["renderIntensityLevel"] == level
                ],
            )
            for level in SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS
        }
        for family in SYNTHETIC_STRESS_V2_FAMILIES
    }
    return {
        "prototypeParentCount": SYNTHETIC_STRESS_V2_PARENT_COUNTS["SYNTHETIC_PROTOTYPE"],
        "calibrationParentCount": sum(record["category"] == category for record in calibration_scores),
        "rawQueryParentCount": len(category_raw),
        "syntheticStimulusCount": len(category_stimuli),
        "prototypePatchCount": prototype_patch_count,
        "thresholdFromRawCalibration": threshold,
        **_condition_report(category_raw, category_stimuli, category_pairs),
        "bySyntheticDefectFamily": by_family,
        "byRenderIntensityLevel": by_level,
        "bySyntheticDefectFamilyAndRenderIntensityLevel": by_family_and_level,
    }


def run_synthetic_stress_v2(
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
    """Run one immutable V2 synthetic-stimulus response-only report.

    No V1 report path is accepted.  The raw threshold computation concludes
    before the V2 generator package loader is called, and its synthetic child
    records are never treated as real physical observations.
    """

    if device != "cpu":
        raise SyntheticStressV2Error("synthetic-stimulus response test supports CPU only")
    try:
        prepared_output = knn._require_external_output(output_path, repository_root=repository_root)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticStressV2Error(str(error)) from error
    started = time.process_time()
    timings = {
        "inputAssemblySeconds": 0.0,
        "provenanceSeconds": 0.0,
        "featureProcessingSeconds": 0.0,
        "scoringSeconds": 0.0,
        "totalElapsedSeconds": 0.0,
    }

    input_started = time.process_time()
    envelope, envelope_file_sha256, parents = load_safe_v2_fit_inputs(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        repository_root=repository_root,
    )
    split = build_fixed_parent_split(parents)
    timings["inputAssemblySeconds"] += time.process_time() - input_started

    provenance_started = time.process_time()
    extractor_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    if not isinstance(extractor_identity, dict):
        raise SyntheticStressV2Error("feature extractor identity must be an object")
    extractor_identity_sha256 = successor.canonical_json_sha256(extractor_identity)
    timings["provenanceSeconds"] += time.process_time() - provenance_started
    embedder = embedder_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    provenance_started = time.process_time()
    loaded_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    timings["provenanceSeconds"] += time.process_time() - provenance_started
    if loaded_identity != extractor_identity:
        raise SyntheticStressV2Error("feature extractor changed while DINO loaded")

    raw_feature_records = [_adapt_raw(record) for record in parents]
    raw_features = _extract_features(raw_feature_records, embedder=embedder, timings=timings)
    prototype_records = [_adapt_raw(record) for record in split["SYNTHETIC_PROTOTYPE"]]
    calibration_records = [_adapt_raw(record) for record in split["SYNTHETIC_CALIBRATION"]]
    raw_query_records = [_adapt_raw(record) for record in split["SYNTHETIC_QUERY"]]
    prototype_banks = {
        category: _prototype_bank(prototype_records, raw_features, category=category)
        for category in SYNTHETIC_STRESS_V2_CATEGORIES
    }
    scoring_started = time.process_time()
    raw_calibration_components = _score_records(calibration_records, raw_features, prototype_banks=prototype_banks)
    thresholds = calibrate_raw_thresholds(raw_calibration_components)
    raw_query_components = _score_records(raw_query_records, raw_features, prototype_banks=prototype_banks)
    timings["scoringSeconds"] += time.process_time() - scoring_started

    # The V2 package cannot be loaded (and therefore its child bytes cannot be
    # scored) until the raw-only calibration thresholds above are already fixed.
    input_started = time.process_time()
    augmentation_manifest, augmentation_file_sha256, child_records = load_and_validate_v2_package(
        augmentation_manifest_path,
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=recipe_path,
        query_parents=split["SYNTHETIC_QUERY"],
        repository_root=repository_root,
    )
    timings["inputAssemblySeconds"] += time.process_time() - input_started
    stimulus_feature_records = [_adapt_stimulus_record(record) for record in child_records]
    synthetic_features = _extract_features(stimulus_feature_records, embedder=embedder, timings=timings)
    scoring_started = time.process_time()
    stimulus_components = _score_records(stimulus_feature_records, synthetic_features, prototype_banks=prototype_banks)
    timings["scoringSeconds"] += time.process_time() - scoring_started

    calibration_scores = [_calibration_output_record(record) for record in raw_calibration_components]
    if any(set(record) != CALIBRATION_SCORE_FIELDS for record in calibration_scores):  # pragma: no cover - invariant guard
        raise SyntheticStressV2Error("raw calibration score shape is unsafe")
    raw_query_scores, stimulus_scores, paired_scores = evaluate_response_only(
        raw_query_components,
        stimulus_components,
        thresholds=thresholds,
    )

    provenance_started = time.process_time()
    completed_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    timings["provenanceSeconds"] += time.process_time() - provenance_started
    if completed_identity != extractor_identity:
        raise SyntheticStressV2Error("feature extractor changed while synthetic-stimulus response test ran")
    timings["totalElapsedSeconds"] = time.process_time() - started

    parent_evidence = envelope.get("parentEvidence")
    partition_identities = envelope.get("successorPartitionIdentities")
    if not isinstance(parent_evidence, dict) or not isinstance(partition_identities, dict):
        raise SyntheticStressV2Error("successor envelope is missing closed parent evidence")
    split_records = _parent_split_records(split)
    category_reports = {
        category: _category_report(
            category,
            threshold=thresholds[category],
            prototype_patch_count=int(prototype_banks[category].shape[0]),
            calibration_scores=calibration_scores,
            raw_scores=raw_query_scores,
            stimulus_scores=stimulus_scores,
            pairs=paired_scores,
        )
        for category in SYNTHETIC_STRESS_V2_CATEGORIES
    }
    document: dict[str, Any] = {
        "schemaVersion": SYNTHETIC_STRESS_V2_REPORT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "postV1Exploratory": True,
        "comparisonOrPromotionAllowed": False,
        "purpose": SYNTHETIC_STRESS_V2_PURPOSE,
        "phase": SYNTHETIC_STRESS_V2_PHASE,
        "metricScope": SYNTHETIC_STRESS_V2_METRIC_SCOPE,
        "realAnomalyPerformance": SYNTHETIC_STRESS_V2_REAL_PERFORMANCE,
        "forbiddenUses": list(SYNTHETIC_STRESS_V2_FORBIDDEN_USES),
        "inputPolicy": SYNTHETIC_STRESS_V2_INPUT_POLICY,
        "blindPolicy": SYNTHETIC_STRESS_V2_BLIND_POLICY,
        "resultLabel": SYNTHETIC_STRESS_V2_RESULT_LABEL,
        "testConfiguration": {
            "algorithmId": SYNTHETIC_STRESS_V2_ALGORITHM,
            "splitAlgorithm": SYNTHETIC_STRESS_V2_PARENT_SPLIT_ALGORITHM,
            "parentCountsPerCategory": dict(SYNTHETIC_STRESS_V2_PARENT_COUNTS),
            "maxPrototypePatches": SYNTHETIC_STRESS_V2_MAX_PROTOTYPE_PATCHES,
            "topKMostAnomalousPatches": SYNTHETIC_STRESS_V2_TOP_K,
            "prototypeBlockSize": SYNTHETIC_STRESS_V2_BLOCK_SIZE,
            "batchSize": SYNTHETIC_STRESS_V2_BATCH_SIZE,
            "decisionRule": SYNTHETIC_STRESS_V2_DECISION_RULE,
            "rawCalibrationThresholdEstablishedBeforePackageLoad": True,
            "rawCalibrationThresholdEstablishedBeforePackageScoring": True,
            "queryPairing": "RAW_QUERY_PARENT_AND_SYNTHETIC_CHILDREN_ARE_NOT_INDEPENDENT",
            "families": list(SYNTHETIC_STRESS_V2_FAMILIES),
            "renderIntensityLevels": list(SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS),
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
        "successorFitIdentitySha256": partition_identities.get("FIT"),
        "parentNormalConfirmationIdentitySha256": parent_evidence.get("parentNormalConfirmationIdentitySha256"),
        "augmentationManifestFileSha256": augmentation_file_sha256,
        "augmentationManifestDeclaredSha256": augmentation_manifest.get("augmentationManifestSha256"),
        "augmentationSchemaVersion": augmentation_manifest.get("schemaVersion"),
        "recipeFileSha256": augmentation_manifest.get("recipeFileSha256"),
        "parentSplit": split_records,
        "parentSplitIdentitySha256": successor.canonical_json_sha256(split_records),
        "featureExtractor": extractor_identity,
        "featureExtractorIdentitySha256": extractor_identity_sha256,
        "thresholds": thresholds,
        "calibrationScores": sorted(calibration_scores, key=lambda record: str(record["caseId"])),
        "rawQueryScores": raw_query_scores,
        "stimulusScores": stimulus_scores,
        "categories": category_reports,
        "aggregate": _condition_report(raw_query_scores, stimulus_scores, paired_scores),
        "execution": _execution_metadata(timings, repository_root=repository_root),
    }
    for name in (
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
        "parentSplitIdentitySha256",
        "featureExtractorIdentitySha256",
    ):
        document[name] = _require_sha256(document[name], name=f"synthetic-stimulus report {name}")
    if document["augmentationSchemaVersion"] != "phone-dino.mvtec-ad-synthetic-only-stress-augmentation/2.0":
        raise SyntheticStressV2Error("V2 synthetic-stimulus package schema changed while the response test ran")
    _require_exact_fields(document, name="synthetic-stimulus response report", fields=REPORT_FIELDS.difference({"syntheticStressReportSha256"}))
    document["syntheticStressReportSha256"] = _document_digest(document)
    _require_exact_fields(document, name="synthetic-stimulus response report", fields=REPORT_FIELDS)
    # ``prepared_output`` was preflighted before any source image was opened;
    # exclusive write inside ``write_response_only_report`` preserves new-only output.
    write_response_only_report(prepared_output, document, repository_root=repository_root)
    return document
