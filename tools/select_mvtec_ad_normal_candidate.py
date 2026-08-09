"""Lock one offline MVTec candidate using normal-only research evidence.

This tool deliberately consumes JSON reports only. It never opens source images,
derived images, masks, model weights, or a PhoneDINO service endpoint, and it
never starts a blind run. It is not a production or physical qualification
selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_research import canonical_json_sha256, sha256_file


CONTRACT_SCHEMA_VERSION = "phone-dino.mvtec-ad-normal-selection-contract/1.2"
ITERATION_REPORT_SCHEMA_VERSION = "phone-dino.mvtec-ad-iteration-report/1.4"
SELECTION_SCHEMA_VERSION = "phone-dino.mvtec-ad-normal-selection/1.2"
FEATURE_CACHE_SCHEMA_VERSION = "phone-dino.mvtec-ad-feature-cache/1.1"
FEATURE_EXTRACTOR_SCHEMA_VERSION = "phone-dino.mvtec-ad-feature-extractor/1.0"
SELECTION_PURPOSE = "OFFLINE_MVTEC_NORMAL_ONLY_CONFIGURATION_LOCK"
NORMAL_OBJECTIVE = [
    "MINIMIZE_WORST_PAIRED_AUGMENTED_SCORE_DELTA_P95",
    "MINIMIZE_WORST_CATEGORY_THRESHOLD",
    "MINIMIZE_MEAN_CATEGORY_THRESHOLD",
    "CANDIDATE_ID_ASC",
]
GATE_NAMES = (
    "maxThresholdIncreaseVsReference",
    "maxOriginalP95IncreaseVsReference",
    "maxAugmentedP95MinusOriginalP95",
    "maxPairedAugmentedScoreDeltaP95",
    "maxPairedAugmentedScoreDeltaMax",
)
IDENTITY_FIELDS = {
    "caseId",
    "category",
    "role",
    "kind",
    "sourceSha256",
    "isAugmentation",
    "variantId",
    "parentCaseId",
    "parentSourceSha256",
    "augmentationRecipeSha256",
}
NORMAL_ONLY_EVIDENCE_FIELDS = {
    "featureInputCount",
    "featureInputRoles",
    "featureInputKinds",
    "blindFeatureInputCount",
    "anomalyFeatureInputCount",
    "normalInputRecordCount",
    "featureInputs",
    "featureInputIdentitySha256",
    "normalInputIdentitySha256",
    "reportedScoreCount",
    "reportedScoreRoles",
    "reportedScoreKinds",
    "calibrationScoreCount",
    "calibrationScoreRoles",
    "calibrationScoreKinds",
    "calibrationInputs",
    "calibrationInputIdentitySha256",
    "originalTuningInputCount",
    "originalTuningInputs",
    "originalTuningInputIdentitySha256",
}
BLIND_REPORTING_FIELDS = {"state", "blindSourcePolicy", "reason"}
FEATURE_EXTRACTOR_FIELDS = {
    "schemaVersion",
    "modelWeightsSha256",
    "modelRepositorySha256",
    "preprocessingId",
    "preprocessing",
    "modelEntrypoint",
    "device",
    "iterationToolSha256",
    "productionModuleSha256",
    "enginesModuleSha256",
    "mvtecResearchModuleSha256",
    "pythonVersion",
    "numpyVersion",
    "torchVersion",
    "torchvisionVersion",
    "pillowVersion",
    "torchThreadCount",
    "torchBackend",
}
FEATURE_EXTRACTOR_PREPROCESSING_FIELDS = {
    "colorSpace",
    "resizeShortEdge",
    "centerCropWidth",
    "centerCropHeight",
    "resizeAntialias",
    "normalizeMean",
    "normalizeStd",
}
FEATURE_EXTRACTOR_TORCH_BACKEND_FIELDS = {
    "deterministicAlgorithmsEnabled",
    "mkldnnAvailable",
    "mkldnnEnabled",
}
ITERATION_REPORT_FIELDS = {
    "schemaVersion", "authoritative", "productionAuthorized", "disclaimer", "selectionProtocol", "blindReporting",
    "inputManifest", "inputManifestFileSha256", "inputManifestDeclaredSha256", "augmentation", "algorithm",
    "featureExtractor", "featureExtractorIdentitySha256", "candidateConfiguration", "candidateConfigurationSha256",
    "execution", "categories", "normalOnlyEvidence",
    "calibrationScores", "scores", "pixelLocalization",
}
ALGORITHM_BASE_FIELDS = {
    "id", "modelRepository", "modelRepositorySha256", "modelWeights", "modelWeightsSha256", "preprocessingId", "device",
}
PATCH_ALGORITHM_FIELDS = {
    "memoryBankSelection", "maxPrototypePatches", "topKMostAnomalousPatches", "prototypeBlockSize",
}
EXECUTION_FIELDS = {
    "batchSize", "featureCache", "featureCacheHits", "featureCacheMisses", "iterationToolSha256",
    "featureCacheSchemaVersion", "phaseTimingsSeconds", "python", "platform", "numpyVersion", "torchVersion", "torchThreadCount",
}
TIMING_FIELDS = {
    "provenanceSeconds", "inputVerificationSeconds", "cacheValidationSeconds", "cacheWriteSeconds", "featureInferenceSeconds",
    "scoringSeconds", "pixelMetricsSeconds", "totalElapsedSeconds",
}
CATEGORY_COMMON_FIELDS = {
    "blindCases", "blindNominalCases", "blindAnomalyCases", "imageAuRoc", "thresholdFromNominalTuning",
    "nominalAboveThresholdRate", "anomalyAboveThresholdRate", "normalCalibrationCases", "normalScoreMedian",
    "normalScoreP95", "normalScoreMax", "originalTuningNormalScoreCases", "originalTuningNormalScoreMedian",
    "originalTuningNormalScoreP95", "originalTuningNormalScoreMax", "augmentedTuningNormalScoreCases",
    "augmentedTuningNormalScoreMedian", "augmentedTuningNormalScoreP95", "augmentedTuningNormalScoreMax",
    "fitOriginalCount", "fitAugmentedCount", "tuningOriginalCount", "tuningAugmentedCount",
}
CATEGORY_PATCH_FIELDS = {"fitPatchCount", "prototypePatchCount", "patchGridHeight", "patchGridWidth"}


class SelectionError(ValueError):
    """Raised when a selection contract or normal-only report is unsafe."""


def _is_under(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _read_json_with_digest(path: Path, *, description: str) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"unable to read {description}: {path}") from error
    if not isinstance(document, dict):
        raise SelectionError(f"{description} must be a JSON object")
    return document, f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    return _read_json_with_digest(path, description=description)[0]


def _require_mapping(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise SelectionError(f"{name} must be an object")
    return value


def _require_string(document: dict[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value:
        raise SelectionError(f"{name} must be a non-empty string")
    return value


def _require_sha256(document: dict[str, Any], name: str) -> str:
    value = _require_string(document, name)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise SelectionError(f"{name} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise SelectionError(f"{name} must be a sha256 digest") from error
    return value


def _require_finite_number(document: dict[str, Any], name: str) -> float:
    value = document.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SelectionError(f"{name} must be a finite number")
    return float(value)


def _require_nonnegative_int(document: dict[str, Any], name: str) -> int:
    value = document.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SelectionError(f"{name} must be a non-negative integer")
    return value


def _require_exact_fields(document: dict[str, Any], fields: set[str], *, name: str) -> None:
    unknown = set(document) - fields
    missing = fields - set(document)
    if unknown or missing:
        detail = "unknown" if unknown else "missing"
        values = sorted(unknown or missing)
        raise SelectionError(f"{name} has {detail} fields: {', '.join(values)}")


def _unique_string_list(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SelectionError(f"{name} must be a non-empty array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise SelectionError(f"{name} must contain non-empty strings")
        if item in result:
            raise SelectionError(f"{name} contains a duplicate value")
        result.append(item)
    return result


def _expected_augmentation_variant_ids(universe: dict[str, Any]) -> list[int]:
    variants_per_parent = universe.get("augmentationVariantsPerParent")
    if (
        not isinstance(variants_per_parent, int)
        or isinstance(variants_per_parent, bool)
        or not 1 <= variants_per_parent <= 8
    ):
        raise SelectionError("candidateUniverse.augmentationVariantsPerParent must be between 1 and 8")
    return list(range(1, variants_per_parent + 1))


def _per_variant_paired_delta_gates(gate: dict[str, Any], expected_variant_ids: list[int]) -> dict[int, dict[str, float]]:
    raw_gates = gate.get("perVariantPairedDeltaGates")
    if not isinstance(raw_gates, list):
        raise SelectionError("gate.perVariantPairedDeltaGates must be an array")
    parsed: dict[int, dict[str, float]] = {}
    ordered_ids: list[int] = []
    for index, raw_gate in enumerate(raw_gates):
        if not isinstance(raw_gate, dict):
            raise SelectionError(f"gate.perVariantPairedDeltaGates[{index}] must be an object")
        _require_exact_fields(
            raw_gate,
            {"variantId", "maxPairedAugmentedScoreDeltaP95", "maxPairedAugmentedScoreDeltaMax"},
            name=f"gate.perVariantPairedDeltaGates[{index}]",
        )
        variant_id = raw_gate.get("variantId")
        if not isinstance(variant_id, int) or isinstance(variant_id, bool) or variant_id <= 0:
            raise SelectionError("gate.perVariantPairedDeltaGates variantId must be a positive integer")
        if variant_id in parsed:
            raise SelectionError("gate.perVariantPairedDeltaGates contains a duplicate variantId")
        limits = {
            name: _require_finite_number(raw_gate, name)
            for name in ("maxPairedAugmentedScoreDeltaP95", "maxPairedAugmentedScoreDeltaMax")
        }
        if any(value < 0.0 for value in limits.values()):
            raise SelectionError("gate.perVariantPairedDeltaGates limits must be non-negative")
        parsed[variant_id] = limits
        ordered_ids.append(variant_id)
    if ordered_ids != expected_variant_ids:
        raise SelectionError("gate.perVariantPairedDeltaGates must exactly cover sorted augmentation variants")
    return parsed


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return canonical_json_sha256(unsigned)


def _load_contract_with_digest(path: Path) -> tuple[dict[str, Any], str]:
    """Load a predeclared research-only candidate universe and gate."""

    document, file_digest = _read_json_with_digest(path, description="normal selection contract")
    _require_exact_fields(document, {
        "schemaVersion", "authoritative", "productionAuthorized", "purpose",
        "inputManifestDeclaredSha256", "inputManifestFileSha256", "candidates",
        "candidateUniverse", "referenceCandidate", "gate", "selection", "contractSha256",
    }, name="normal selection contract")
    if document.get("schemaVersion") != CONTRACT_SCHEMA_VERSION:
        raise SelectionError("unsupported normal selection contract schema")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise SelectionError("normal selection contract must be non-authoritative and non-production")
    if document.get("purpose") != SELECTION_PURPOSE:
        raise SelectionError("normal selection contract has an unsafe purpose")
    if document.get("contractSha256") != _document_digest(document, "contractSha256"):
        raise SelectionError("normal selection contract digest does not match")
    _require_sha256(document, "inputManifestDeclaredSha256")
    _require_sha256(document, "inputManifestFileSha256")
    raw_candidates = document.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise SelectionError("candidates must be a non-empty array")
    candidate_ids: list[str] = []
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            raise SelectionError("candidates entries must be objects")
        _require_exact_fields(raw_candidate, {"id", "candidateConfiguration"}, name=f"candidates[{index}]")
        candidate_id = _require_string(raw_candidate, "id")
        if candidate_id in candidate_ids:
            raise SelectionError("candidates contains a duplicate id")
        candidate_ids.append(candidate_id)
        configuration = _require_mapping(raw_candidate, "candidateConfiguration")
        if not configuration:
            raise SelectionError("candidateConfiguration must not be empty")
    universe = _require_mapping(document, "candidateUniverse")
    _require_exact_fields(universe, {
        "algorithmId", "modelWeightsSha256", "modelRepositorySha256", "preprocessingId", "device", "iterationToolSha256",
        "featureExtractorIdentitySha256",
        "categories", "normalInputIdentitySha256", "calibrationInputIdentitySha256",
        "originalTuningInputIdentitySha256", "augmentationManifestSha256", "recipeSha256",
        "augmentationVariantsPerParent",
    }, name="candidateUniverse")
    _require_string(universe, "algorithmId")
    _require_sha256(universe, "modelWeightsSha256")
    _require_sha256(universe, "modelRepositorySha256")
    _require_string(universe, "preprocessingId")
    _require_string(universe, "device")
    _require_sha256(universe, "iterationToolSha256")
    _require_sha256(universe, "featureExtractorIdentitySha256")
    categories = _unique_string_list(universe.get("categories"), name="candidateUniverse.categories")
    if categories != sorted(categories):
        raise SelectionError("candidateUniverse.categories must be sorted")
    _require_sha256(universe, "normalInputIdentitySha256")
    _require_sha256(universe, "calibrationInputIdentitySha256")
    _require_sha256(universe, "originalTuningInputIdentitySha256")
    _require_sha256(universe, "augmentationManifestSha256")
    _require_sha256(universe, "recipeSha256")
    expected_variant_ids = _expected_augmentation_variant_ids(universe)
    reference = _require_mapping(document, "referenceCandidate")
    _require_exact_fields(reference, {"id", "reportSha256"}, name="referenceCandidate")
    reference_id = _require_string(reference, "id")
    if reference_id not in candidate_ids:
        raise SelectionError("referenceCandidate.id must appear in candidates")
    _require_sha256(reference, "reportSha256")
    gate = _require_mapping(document, "gate")
    _require_exact_fields(gate, set(GATE_NAMES) | {"perVariantPairedDeltaGates"}, name="gate")
    for name in GATE_NAMES:
        if _require_finite_number(gate, name) < 0.0:
            raise SelectionError(f"gate {name} must be non-negative")
    _per_variant_paired_delta_gates(gate, expected_variant_ids)
    selection = _require_mapping(document, "selection")
    _require_exact_fields(selection, {"objective"}, name="selection")
    if selection.get("objective") != NORMAL_OBJECTIVE:
        raise SelectionError("selection objective must be the fixed normal-only objective")
    return document, file_digest


def load_contract(path: Path) -> dict[str, Any]:
    return _load_contract_with_digest(path)[0]


def _candidate_configurations(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_candidates = contract.get("candidates")
    if not isinstance(raw_candidates, list):  # pragma: no cover - load_contract establishes this invariant
        raise SelectionError("candidates must be an array")
    return {
        str(candidate["id"]): dict(candidate["candidateConfiguration"])
        for candidate in raw_candidates
        if isinstance(candidate, dict)
    }


def _quantile_summary(values: list[float], *, prefix: str) -> dict[str, float | int]:
    if not values:
        raise SelectionError(f"{prefix} normal score population is empty")
    sorted_values = sorted(values)
    return {
        f"{prefix}Cases": len(sorted_values),
        f"{prefix}Median": sorted_values[len(sorted_values) // 2],
        f"{prefix}P95": sorted_values[max(0, math.ceil(len(sorted_values) * 0.95) - 1)],
        f"{prefix}Max": sorted_values[-1],
    }


def _require_close(actual: object, expected: float | int, *, name: str) -> None:
    if isinstance(expected, int):
        if actual != expected:
            raise SelectionError(f"{name} does not match the calibration scores")
        return
    if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(actual):
        raise SelectionError(f"{name} must be a finite number")
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise SelectionError(f"{name} does not match the calibration scores")


def _identity_record(value: object, *, categories: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectionError(f"{name} record must be an object")
    record = dict(value)
    _require_exact_fields(record, IDENTITY_FIELDS, name=name)
    case_id = _require_string(record, "caseId")
    if record.get("category") not in categories:
        raise SelectionError(f"{name} record category is outside the frozen universe")
    if record.get("role") not in {"FIT", "THRESHOLD_TUNING", "BLIND"} or record.get("kind") not in {"NOMINAL", "ANOMALY"}:
        raise SelectionError(f"{name} record role or kind is invalid")
    _require_sha256(record, "sourceSha256")
    if not isinstance(record.get("isAugmentation"), bool):
        raise SelectionError(f"{name} isAugmentation must be a boolean")
    if record["isAugmentation"]:
        variant_id = record.get("variantId")
        if not isinstance(variant_id, int) or isinstance(variant_id, bool) or variant_id <= 0:
            raise SelectionError(f"{name} derived record variantId must be a positive integer")
        _require_string(record, "parentCaseId")
        _require_sha256(record, "parentSourceSha256")
        _require_sha256(record, "augmentationRecipeSha256")
    elif any(record.get(field) is not None for field in (
        "variantId", "parentCaseId", "parentSourceSha256", "augmentationRecipeSha256"
    )):
        raise SelectionError(f"{name} original record has unexpected augmentation parent fields")
    return record | {"caseId": case_id}


def _identity_records(value: object, *, categories: set[str], name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SelectionError(f"{name} must be a non-empty array")
    records = [_identity_record(item, categories=categories, name=name) for item in value]
    case_ids = [record["caseId"] for record in records]
    if case_ids != sorted(case_ids) or len(set(case_ids)) != len(case_ids):
        raise SelectionError(f"{name} caseIds must be unique and sorted")
    return records


def _normal_score_record(value: object, *, categories: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SelectionError(f"{name} record must be an object")
    record = dict(value)
    common_fields = {
        "caseId", "category", "role", "kind", "defect", "sourceSha256", "isAugmentation", "variantId", "score"
    }
    expected_fields = common_fields | ({"parentCaseId", "parentSourceSha256", "augmentationRecipeSha256"} if record.get("isAugmentation") is True else set())
    _require_exact_fields(record, expected_fields, name=name)
    case_id = _require_string(record, "caseId")
    if record.get("category") not in categories:
        raise SelectionError(f"{name} record category is outside the frozen universe")
    if record.get("role") != "THRESHOLD_TUNING" or record.get("kind") != "NOMINAL" or record.get("defect") != "good":
        raise SelectionError(f"{name} contains blind, anomalous, or non-good score data")
    _require_sha256(record, "sourceSha256")
    if not isinstance(record.get("isAugmentation"), bool):
        raise SelectionError(f"{name} isAugmentation must be a boolean")
    score = _require_finite_number(record, "score")
    if not 0.0 <= score <= 2.0:
        raise SelectionError(f"{name} score is outside the cosine-distance range")
    record["score"] = score
    if record["isAugmentation"]:
        variant_id = record.get("variantId")
        if not isinstance(variant_id, int) or isinstance(variant_id, bool) or variant_id <= 0:
            raise SelectionError(f"{name} derived record variantId must be a positive integer")
        _require_string(record, "parentCaseId")
        _require_sha256(record, "parentSourceSha256")
        _require_sha256(record, "augmentationRecipeSha256")
    elif record.get("variantId") is not None:
        raise SelectionError(f"{name} original record has an unexpected variantId")
    return record | {"caseId": case_id}


def _score_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "role": record["role"],
        "kind": record["kind"],
        "sourceSha256": record["sourceSha256"],
        "isAugmentation": record["isAugmentation"],
        "variantId": record["variantId"],
        "parentCaseId": record.get("parentCaseId"),
        "parentSourceSha256": record.get("parentSourceSha256"),
        "augmentationRecipeSha256": record.get("augmentationRecipeSha256"),
    }


def _validate_augmentation_variant_coverage(
    feature_inputs: list[dict[str, Any]],
    universe: dict[str, Any],
) -> list[int]:
    """Require every normal FIT/tuning parent to carry each frozen variant once."""

    expected_variant_ids = _expected_augmentation_variant_ids(universe)
    expected_variant_set = set(expected_variant_ids)
    originals_by_case = {record["caseId"]: record for record in feature_inputs if not record["isAugmentation"]}
    if not originals_by_case:
        raise SelectionError("normal-only feature membership has no original parents")
    variants_by_parent: dict[str, list[int]] = defaultdict(list)
    for record in feature_inputs:
        if not record["isAugmentation"]:
            continue
        parent = originals_by_case.get(record["parentCaseId"])
        if parent is None:
            raise SelectionError("derived normal feature record has no original parent")
        if any(record[name] != parent[name] for name in ("category", "role", "kind")):
            raise SelectionError("derived normal feature record does not match its original parent membership")
        if record["parentSourceSha256"] != parent["sourceSha256"]:
            raise SelectionError("derived normal feature parent digest does not match")
        if record["augmentationRecipeSha256"] != universe["recipeSha256"]:
            raise SelectionError("derived normal feature recipe digest does not match the frozen universe")
        variant_id = record["variantId"]
        if variant_id not in expected_variant_set:
            raise SelectionError("derived normal feature variantId is outside the frozen coverage")
        variants_by_parent[parent["caseId"]].append(variant_id)
    for parent_case_id in sorted(originals_by_case):
        variant_ids = variants_by_parent.get(parent_case_id, [])
        if len(variant_ids) != len(set(variant_ids)):
            raise SelectionError("derived normal feature coverage has a duplicate variantId for one parent")
        if sorted(variant_ids) != expected_variant_ids:
            raise SelectionError("derived normal feature coverage does not match the frozen variant set for one parent")
    return expected_variant_ids


def _validate_normal_evidence(
    report: dict[str, Any],
    universe: dict[str, Any],
    calibration_records: list[dict[str, Any]],
    score_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = _require_mapping(report, "normalOnlyEvidence")
    _require_exact_fields(evidence, NORMAL_ONLY_EVIDENCE_FIELDS, name="normalOnlyEvidence")
    if evidence.get("featureInputRoles") != ["FIT", "THRESHOLD_TUNING"]:
        raise SelectionError("normalOnlyEvidence feature input roles are not normal-only")
    if evidence.get("featureInputKinds") != ["NOMINAL"]:
        raise SelectionError("normalOnlyEvidence feature input kinds are not normal-only")
    if evidence.get("blindFeatureInputCount") != 0 or evidence.get("anomalyFeatureInputCount") != 0:
        raise SelectionError("normalOnlyEvidence includes blind or anomalous feature inputs")
    feature_input_count = _require_nonnegative_int(evidence, "featureInputCount")
    normal_input_count = _require_nonnegative_int(evidence, "normalInputRecordCount")
    if feature_input_count <= 0 or normal_input_count != feature_input_count:
        raise SelectionError("normalOnlyEvidence normal feature input count is invalid")
    category_set = set(_unique_string_list(universe.get("categories"), name="candidateUniverse.categories"))
    feature_inputs = _identity_records(evidence.get("featureInputs"), categories=category_set, name="normalOnlyEvidence.featureInputs")
    if len(feature_inputs) != feature_input_count:
        raise SelectionError("normalOnlyEvidence feature input count does not match featureInputs")
    if any(record["role"] not in {"FIT", "THRESHOLD_TUNING"} or record["kind"] != "NOMINAL" for record in feature_inputs):
        raise SelectionError("normalOnlyEvidence featureInputs contain blind or anomalous membership")
    feature_identity = canonical_json_sha256(feature_inputs)
    if evidence.get("featureInputIdentitySha256") != feature_identity or evidence.get("normalInputIdentitySha256") != feature_identity:
        raise SelectionError("normalOnlyEvidence feature input identity does not match featureInputs")
    if feature_identity != universe["normalInputIdentitySha256"]:
        raise SelectionError("normalOnlyEvidence input identity does not match the frozen universe")
    _validate_augmentation_variant_coverage(feature_inputs, universe)
    if (
        _require_nonnegative_int(evidence, "reportedScoreCount") != len(score_records)
        or _require_nonnegative_int(evidence, "calibrationScoreCount") != len(calibration_records)
    ):
        raise SelectionError("normalOnlyEvidence score counts do not match the report")
    if evidence.get("reportedScoreRoles") != ["THRESHOLD_TUNING"] or evidence.get("reportedScoreKinds") != ["NOMINAL"]:
        raise SelectionError("normalOnlyEvidence reported scores are not normal-only")
    if evidence.get("calibrationScoreRoles") != ["THRESHOLD_TUNING"] or evidence.get("calibrationScoreKinds") != ["NOMINAL"]:
        raise SelectionError("normalOnlyEvidence calibration scores are not normal-only")
    calibration_inputs = _identity_records(
        evidence.get("calibrationInputs"), categories=category_set, name="normalOnlyEvidence.calibrationInputs"
    )
    if len(calibration_inputs) != len(calibration_records):
        raise SelectionError("normalOnlyEvidence calibration input count does not match calibrationScores")
    if any(record["role"] != "THRESHOLD_TUNING" or record["kind"] != "NOMINAL" for record in calibration_inputs):
        raise SelectionError("normalOnlyEvidence calibrationInputs are not normal-only tuning membership")
    expected_calibration_inputs = [
        record for record in feature_inputs if record["role"] == "THRESHOLD_TUNING"
    ]
    if calibration_inputs != expected_calibration_inputs:
        raise SelectionError("normalOnlyEvidence calibration membership does not match tuning feature membership")
    calibration_identity = canonical_json_sha256(calibration_inputs)
    if evidence.get("calibrationInputIdentitySha256") != calibration_identity:
        raise SelectionError("normalOnlyEvidence calibration input identity does not match calibrationInputs")
    if calibration_identity != universe["calibrationInputIdentitySha256"]:
        raise SelectionError("normalOnlyEvidence calibration identity does not match the frozen universe")
    derived_calibration_inputs = sorted((_score_identity(record) for record in calibration_records), key=lambda record: record["caseId"])
    if calibration_inputs != derived_calibration_inputs:
        raise SelectionError("normalOnlyEvidence calibration membership does not match calibrationScores")
    original_tuning_inputs = [record for record in calibration_inputs if not record["isAugmentation"]]
    if _require_nonnegative_int(evidence, "originalTuningInputCount") != len(original_tuning_inputs):
        raise SelectionError("normalOnlyEvidence original tuning input count does not match calibrationInputs")
    declared_originals = _identity_records(
        evidence.get("originalTuningInputs"), categories=category_set, name="normalOnlyEvidence.originalTuningInputs"
    )
    if declared_originals != original_tuning_inputs:
        raise SelectionError("normalOnlyEvidence original tuning membership does not match calibrationInputs")
    original_identity = canonical_json_sha256(original_tuning_inputs)
    if evidence.get("originalTuningInputIdentitySha256") != original_identity:
        raise SelectionError("normalOnlyEvidence original tuning identity does not match originalTuningInputs")
    if original_identity != universe["originalTuningInputIdentitySha256"]:
        raise SelectionError("normalOnlyEvidence original tuning identity does not match the frozen universe")
    return feature_inputs


def _derive_category_metrics(
    category: str,
    records: list[dict[str, Any]],
    category_report: dict[str, Any],
    expected_variant_ids: list[int],
) -> dict[str, Any]:
    originals = [record for record in records if not record["isAugmentation"]]
    augmented = [record for record in records if record["isAugmentation"]]
    if not originals or not augmented:
        raise SelectionError(f"{category} must contain both original and derived tuning normals")
    original_by_case = {record["caseId"]: record for record in originals}
    if len(original_by_case) != len(originals):
        raise SelectionError(f"{category} has duplicate original tuning caseIds")
    deltas: list[float] = []
    deltas_by_variant: dict[int, list[float]] = {variant_id: [] for variant_id in expected_variant_ids}
    covered_pairs: set[tuple[str, int]] = set()
    covered_parent_case_ids: set[str] = set()
    for record in augmented:
        parent = original_by_case.get(record["parentCaseId"])
        if parent is None:
            raise SelectionError(f"{category} derived tuning record has no original tuning parent")
        if record["parentSourceSha256"] != parent["sourceSha256"]:
            raise SelectionError(f"{category} derived tuning parent digest does not match")
        variant_id = record["variantId"]
        if variant_id not in deltas_by_variant:
            raise SelectionError(f"{category} derived tuning variantId is outside the frozen coverage")
        pair = (record["parentCaseId"], variant_id)
        if pair in covered_pairs:
            raise SelectionError(f"{category} derived tuning has a duplicate parent and variantId")
        covered_pairs.add(pair)
        covered_parent_case_ids.add(record["parentCaseId"])
        delta = record["score"] - parent["score"]
        deltas.append(delta)
        deltas_by_variant[variant_id].append(delta)
    if set(original_by_case) != covered_parent_case_ids:
        raise SelectionError(f"{category} original tuning cases are missing derived coverage")
    expected_pairs = {
        (case_id, variant_id)
        for case_id in original_by_case
        for variant_id in expected_variant_ids
    }
    if covered_pairs != expected_pairs:
        raise SelectionError(f"{category} derived tuning coverage does not match the frozen parent and variant set")
    all_scores = [record["score"] for record in records]
    original_summary = _quantile_summary([record["score"] for record in originals], prefix="originalTuningNormalScore")
    augmented_summary = _quantile_summary([record["score"] for record in augmented], prefix="augmentedTuningNormalScore")
    overall_summary = _quantile_summary(all_scores, prefix="normalScore")
    overall_summary["normalCalibrationCases"] = overall_summary.pop("normalScoreCases")
    expected_summary = overall_summary | original_summary | augmented_summary
    for name, expected in expected_summary.items():
        _require_close(category_report.get(name), expected, name=f"categories.{category}.{name}")
    threshold = max(all_scores)
    _require_close(category_report.get("thresholdFromNominalTuning"), threshold, name=f"categories.{category}.thresholdFromNominalTuning")
    for name in ("blindCases", "blindNominalCases", "blindAnomalyCases"):
        if category_report.get(name) != 0:
            raise SelectionError(f"categories.{category}.{name} must be zero in a normal-only report")
    for name in ("imageAuRoc", "nominalAboveThresholdRate", "anomalyAboveThresholdRate"):
        if category_report.get(name) is not None:
            raise SelectionError(f"categories.{category}.{name} must be null in a normal-only report")
    delta_summary = _quantile_summary(deltas, prefix="pairedAugmentedScoreDelta")
    per_variant = []
    for variant_id in expected_variant_ids:
        summary = _quantile_summary(deltas_by_variant[variant_id], prefix="pairedAugmentedScoreDelta")
        per_variant.append({
            "variantId": variant_id,
            "pairedAugmentedScoreDeltaP95": float(summary["pairedAugmentedScoreDeltaP95"]),
            "pairedAugmentedScoreDeltaMax": float(summary["pairedAugmentedScoreDeltaMax"]),
        })
    return {
        "threshold": threshold,
        "originalP95": float(original_summary["originalTuningNormalScoreP95"]),
        "augmentedP95": float(augmented_summary["augmentedTuningNormalScoreP95"]),
        "pairedAugmentedScoreDeltaP95": float(delta_summary["pairedAugmentedScoreDeltaP95"]),
        "pairedAugmentedScoreDeltaMax": float(delta_summary["pairedAugmentedScoreDeltaMax"]),
        "pairedAugmentedScoreDeltaByVariant": per_variant,
    }


def _report_candidate_configuration(algorithm: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    algorithm_id = _require_string(algorithm, "id")
    if algorithm_id not in {"DINOV2_GLOBAL_NEAREST_NORMAL_COSINE_V1", "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1"}:
        raise SelectionError("algorithm.id is not supported by the normal-only selector")
    expected_algorithm_fields = ALGORITHM_BASE_FIELDS | (
        PATCH_ALGORITHM_FIELDS if algorithm_id == "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1" else set()
    )
    _require_exact_fields(algorithm, expected_algorithm_fields, name="algorithm")
    _require_string(algorithm, "modelRepository")
    _require_sha256(algorithm, "modelRepositorySha256")
    _require_string(algorithm, "modelWeights")
    _require_sha256(algorithm, "modelWeightsSha256")
    _require_string(algorithm, "preprocessingId")
    _require_string(algorithm, "device")
    _require_exact_fields(execution, EXECUTION_FIELDS, name="execution")
    batch_size = _require_nonnegative_int(execution, "batchSize")
    if batch_size <= 0:
        raise SelectionError("execution.batchSize must be positive")
    _require_sha256(execution, "iterationToolSha256")
    if execution.get("featureCacheSchemaVersion") != FEATURE_CACHE_SCHEMA_VERSION:
        raise SelectionError("execution feature cache schema is unsupported")
    for name in ("featureCacheHits", "featureCacheMisses", "torchThreadCount"):
        _require_nonnegative_int(execution, name)
    if execution.get("featureCache") is not None and not isinstance(execution.get("featureCache"), str):
        raise SelectionError("execution.featureCache must be a string or null")
    for name in ("python", "platform", "numpyVersion", "torchVersion"):
        _require_string(execution, name)
    timings = _require_mapping(execution, "phaseTimingsSeconds")
    _require_exact_fields(timings, TIMING_FIELDS, name="execution.phaseTimingsSeconds")
    for name in TIMING_FIELDS:
        if _require_finite_number(timings, name) < 0.0:
            raise SelectionError(f"execution.phaseTimingsSeconds.{name} must be non-negative")
    configuration: dict[str, Any] = {"algorithmId": algorithm_id, "batchSize": batch_size}
    if algorithm_id == "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1":
        _require_string(algorithm, "memoryBankSelection")
        for name in ("maxPrototypePatches", "topKMostAnomalousPatches", "prototypeBlockSize"):
            value = _require_nonnegative_int(algorithm, name)
            if value <= 0:
                raise SelectionError(f"algorithm.{name} must be positive")
            configuration[name] = value
        configuration["memoryBankSelection"] = algorithm["memoryBankSelection"]
    return configuration


def _report_feature_extractor_identity(
    report: dict[str, Any],
    algorithm: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    """Validate the exact local implementation and dependency identity behind features."""

    identity = _require_mapping(report, "featureExtractor")
    _require_exact_fields(identity, FEATURE_EXTRACTOR_FIELDS, name="featureExtractor")
    if identity.get("schemaVersion") != FEATURE_EXTRACTOR_SCHEMA_VERSION:
        raise SelectionError("featureExtractor has an unsupported schema")
    for name in (
        "modelWeightsSha256",
        "modelRepositorySha256",
        "iterationToolSha256",
        "productionModuleSha256",
        "enginesModuleSha256",
        "mvtecResearchModuleSha256",
    ):
        _require_sha256(identity, name)
    for name in (
        "preprocessingId",
        "modelEntrypoint",
        "device",
        "pythonVersion",
        "numpyVersion",
        "torchVersion",
        "torchvisionVersion",
        "pillowVersion",
    ):
        _require_string(identity, name)
    if _require_nonnegative_int(identity, "torchThreadCount") <= 0:
        raise SelectionError("featureExtractor torchThreadCount must be positive")
    preprocessing = _require_mapping(identity, "preprocessing")
    _require_exact_fields(preprocessing, FEATURE_EXTRACTOR_PREPROCESSING_FIELDS, name="featureExtractor.preprocessing")
    if preprocessing.get("colorSpace") != "RGB" or preprocessing.get("resizeAntialias") is not True:
        raise SelectionError("featureExtractor preprocessing is not the supported RGB antialiased transform")
    for name in ("resizeShortEdge", "centerCropWidth", "centerCropHeight"):
        if _require_nonnegative_int(preprocessing, name) <= 0:
            raise SelectionError(f"featureExtractor preprocessing {name} must be positive")
    for name in ("normalizeMean", "normalizeStd"):
        values = preprocessing.get(name)
        if not isinstance(values, list) or len(values) != 3:
            raise SelectionError(f"featureExtractor preprocessing {name} must be a three-channel array")
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise SelectionError(f"featureExtractor preprocessing {name} must be finite")
    backend = _require_mapping(identity, "torchBackend")
    _require_exact_fields(backend, FEATURE_EXTRACTOR_TORCH_BACKEND_FIELDS, name="featureExtractor.torchBackend")
    if any(not isinstance(backend.get(name), bool) for name in FEATURE_EXTRACTOR_TORCH_BACKEND_FIELDS):
        raise SelectionError("featureExtractor torch backend flags must be boolean")
    for identity_name, report_name, report_document in (
        ("modelWeightsSha256", "modelWeightsSha256", algorithm),
        ("modelRepositorySha256", "modelRepositorySha256", algorithm),
        ("preprocessingId", "preprocessingId", algorithm),
        ("device", "device", algorithm),
        ("iterationToolSha256", "iterationToolSha256", execution),
    ):
        if identity[identity_name] != report_document[report_name]:
            raise SelectionError(f"featureExtractor {identity_name} does not match the report")
    if report.get("featureExtractorIdentitySha256") != canonical_json_sha256(identity):
        raise SelectionError("featureExtractor identity digest does not match")
    return identity


def validate_candidate_report(
    path: Path,
    contract: dict[str, Any],
    expected_configuration: dict[str, Any],
) -> dict[str, Any]:
    """Validate one report without opening any image or model asset."""

    if _is_under(REPOSITORY_ROOT, path):
        raise SelectionError("candidate report must stay outside the Git working tree")
    report, report_digest = _read_json_with_digest(path, description="normal-only candidate report")
    _require_exact_fields(report, ITERATION_REPORT_FIELDS, name="candidate report")
    universe = _require_mapping(contract, "candidateUniverse")
    if report.get("schemaVersion") != ITERATION_REPORT_SCHEMA_VERSION:
        raise SelectionError("candidate report has an unsupported iteration schema")
    if report.get("authoritative") is not False or report.get("productionAuthorized") is not False:
        raise SelectionError("candidate report must be non-authoritative and non-production")
    if report.get("selectionProtocol") != "NORMAL_ONLY_ITERATION_THEN_BLIND_REPORTING_ONLY":
        raise SelectionError("candidate report has an unsafe selection protocol")
    blind_reporting = _require_mapping(report, "blindReporting")
    _require_exact_fields(blind_reporting, BLIND_REPORTING_FIELDS, name="blindReporting")
    if (
        blind_reporting.get("state") != "NOT_RUN"
        or blind_reporting.get("blindSourcePolicy") != "ORIGINAL_ONLY"
        or blind_reporting.get("reason") != "NORMAL_ONLY_ITERATION"
    ):
        raise SelectionError("candidate report is not a normal-only report")
    if report.get("pixelLocalization") is not None:
        raise SelectionError("candidate report must not contain pixel or mask metrics")
    if report.get("inputManifestDeclaredSha256") != contract["inputManifestDeclaredSha256"]:
        raise SelectionError("candidate report declared manifest identity does not match the contract")
    if report.get("inputManifestFileSha256") != contract["inputManifestFileSha256"]:
        raise SelectionError("candidate report manifest file identity does not match the contract")
    augmentation = _require_mapping(report, "augmentation")
    _require_exact_fields(augmentation, {
        "state", "blindPolicy", "blindAugmentedCount", "augmentationManifestPath",
        "augmentationManifestSha256", "recipeSha256", "variantsPerParent", "derivedRecordCount",
    }, name="augmentation")
    if (
        augmentation.get("state") != "NORMAL_FIT_AND_TUNING_ONLY"
        or augmentation.get("blindPolicy") != "BLIND_ORIGINAL_ONLY"
        or augmentation.get("blindAugmentedCount") != 0
    ):
        raise SelectionError("candidate report augmentation is not normal-only")
    if augmentation.get("augmentationManifestSha256") != universe["augmentationManifestSha256"]:
        raise SelectionError("candidate report augmentation identity does not match the contract")
    if augmentation.get("recipeSha256") != universe["recipeSha256"]:
        raise SelectionError("candidate report recipe identity does not match the contract")
    if (
        not isinstance(augmentation.get("variantsPerParent"), int)
        or isinstance(augmentation.get("variantsPerParent"), bool)
        or not 1 <= augmentation["variantsPerParent"] <= 8
    ):
        raise SelectionError("candidate report augmentation variantsPerParent is invalid")
    if augmentation.get("variantsPerParent") != universe["augmentationVariantsPerParent"]:
        raise SelectionError("candidate report augmentation variant coverage does not match the contract")
    _require_string(augmentation, "augmentationManifestPath")
    if _require_nonnegative_int(augmentation, "derivedRecordCount") <= 0:
        raise SelectionError("candidate report augmentation must contain derived normal records")
    algorithm = _require_mapping(report, "algorithm")
    execution = _require_mapping(report, "execution")
    actual_configuration = _report_candidate_configuration(algorithm, execution)
    _report_feature_extractor_identity(report, algorithm, execution)
    reported_configuration = _require_mapping(report, "candidateConfiguration")
    if reported_configuration != actual_configuration:
        raise SelectionError("candidate report configuration does not match its algorithm and execution fields")
    if report.get("candidateConfigurationSha256") != canonical_json_sha256(actual_configuration):
        raise SelectionError("candidate report configuration digest does not match")
    if actual_configuration != expected_configuration:
        raise SelectionError("candidate report configuration does not match its predeclared candidate")
    for report_name, contract_name in (
        ("id", "algorithmId"),
        ("modelWeightsSha256", "modelWeightsSha256"),
        ("modelRepositorySha256", "modelRepositorySha256"),
        ("preprocessingId", "preprocessingId"),
        ("device", "device"),
    ):
        if algorithm.get(report_name) != universe[contract_name]:
            raise SelectionError(f"candidate report algorithm {report_name} does not match the contract")
    if execution.get("iterationToolSha256") != universe["iterationToolSha256"]:
        raise SelectionError("candidate report iteration tool identity does not match the contract")
    if report.get("featureExtractorIdentitySha256") != universe["featureExtractorIdentitySha256"]:
        raise SelectionError("candidate report feature extractor identity does not match the contract")

    categories = _require_mapping(report, "categories")
    expected_categories = _unique_string_list(universe.get("categories"), name="candidateUniverse.categories")
    if set(categories) != set(expected_categories):
        raise SelectionError("candidate report categories do not match the contract")
    expected_category_fields = CATEGORY_COMMON_FIELDS | (
        CATEGORY_PATCH_FIELDS if actual_configuration["algorithmId"] == "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1" else set()
    )
    for category in expected_categories:
        category_report = _require_mapping(categories, category)
        _require_exact_fields(category_report, expected_category_fields, name=f"categories.{category}")
        for name in (
            "blindCases", "blindNominalCases", "blindAnomalyCases", "normalCalibrationCases",
            "originalTuningNormalScoreCases", "augmentedTuningNormalScoreCases",
            "fitOriginalCount", "fitAugmentedCount", "tuningOriginalCount", "tuningAugmentedCount",
        ):
            _require_nonnegative_int(category_report, name)
        if actual_configuration["algorithmId"] == "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1":
            for name in ("fitPatchCount", "prototypePatchCount", "patchGridHeight", "patchGridWidth"):
                _require_nonnegative_int(category_report, name)
    raw_calibration = report.get("calibrationScores")
    raw_scores = report.get("scores")
    if not isinstance(raw_calibration, list) or not raw_calibration:
        raise SelectionError("candidate report calibrationScores must be a non-empty array")
    if not isinstance(raw_scores, list) or not raw_scores:
        raise SelectionError("candidate report scores must be a non-empty array")
    category_set = set(expected_categories)
    calibration_records = [
        _normal_score_record(value, categories=category_set, name="calibrationScores") for value in raw_calibration
    ]
    score_records = [_normal_score_record(value, categories=category_set, name="scores") for value in raw_scores]
    calibration_by_case = {record["caseId"]: record for record in calibration_records}
    if len(calibration_by_case) != len(calibration_records):
        raise SelectionError("candidate report calibrationScores has duplicate caseIds")
    score_by_case = {record["caseId"]: record for record in score_records}
    if len(score_by_case) != len(score_records):
        raise SelectionError("candidate report scores has duplicate caseIds")
    original_calibration = {case_id: record for case_id, record in calibration_by_case.items() if not record["isAugmentation"]}
    if set(score_by_case) != set(original_calibration):
        raise SelectionError("candidate report scores must contain exactly the original tuning scores")
    for case_id, score_record in score_by_case.items():
        original = original_calibration[case_id]
        for name in ("caseId", "category", "role", "kind", "sourceSha256", "isAugmentation", "score"):
            if score_record.get(name) != original.get(name):
                raise SelectionError("candidate report scores do not match calibrationScores")
    feature_inputs = _validate_normal_evidence(report, universe, calibration_records, score_records)
    derived_feature_count = sum(record["isAugmentation"] for record in feature_inputs)
    if augmentation["derivedRecordCount"] != derived_feature_count:
        raise SelectionError("candidate report derived record count does not match feature membership")
    for category in expected_categories:
        category_report = _require_mapping(categories, category)
        category_inputs = [record for record in feature_inputs if record["category"] == category]
        expected_counts = {
            "fitOriginalCount": sum(record["role"] == "FIT" and not record["isAugmentation"] for record in category_inputs),
            "fitAugmentedCount": sum(record["role"] == "FIT" and record["isAugmentation"] for record in category_inputs),
            "tuningOriginalCount": sum(
                record["role"] == "THRESHOLD_TUNING" and not record["isAugmentation"] for record in category_inputs
            ),
            "tuningAugmentedCount": sum(
                record["role"] == "THRESHOLD_TUNING" and record["isAugmentation"] for record in category_inputs
            ),
        }
        for name, expected in expected_counts.items():
            if category_report[name] != expected:
                raise SelectionError(f"categories.{category}.{name} does not match feature membership")

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in calibration_records:
        by_category[str(record["category"])].append(record)
    expected_variant_ids = _expected_augmentation_variant_ids(universe)
    metrics = {
        category: _derive_category_metrics(
            category,
            by_category[category],
            _require_mapping(categories, category),
            expected_variant_ids,
        )
        for category in expected_categories
    }
    return {
        "reportSha256": report_digest,
        "normalMetricsByCategory": metrics,
    }


def _gate_reasons(
    candidate_metrics: dict[str, dict[str, Any]],
    reference_metrics: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    expected_variant_ids: list[int],
) -> list[str]:
    reasons: list[str] = []
    per_variant_gates = _per_variant_paired_delta_gates(gate, expected_variant_ids)
    for category in sorted(candidate_metrics):
        candidate = candidate_metrics[category]
        reference = reference_metrics[category]
        checks = (
            ("maxThresholdIncreaseVsReference", candidate["threshold"] - reference["threshold"]),
            ("maxOriginalP95IncreaseVsReference", candidate["originalP95"] - reference["originalP95"]),
            ("maxAugmentedP95MinusOriginalP95", candidate["augmentedP95"] - candidate["originalP95"]),
            ("maxPairedAugmentedScoreDeltaP95", candidate["pairedAugmentedScoreDeltaP95"]),
            ("maxPairedAugmentedScoreDeltaMax", candidate["pairedAugmentedScoreDeltaMax"]),
        )
        for name, value in checks:
            maximum = _require_finite_number(gate, name)
            if value > maximum:
                reasons.append(f"{category}.{name}={value:.12g} exceeds {maximum:.12g}")
        per_variant_metrics = candidate["pairedAugmentedScoreDeltaByVariant"]
        if not isinstance(per_variant_metrics, list):  # pragma: no cover - derived internally above
            raise SelectionError("derived per-variant metrics are invalid")
        if [metric.get("variantId") for metric in per_variant_metrics] != expected_variant_ids:
            raise SelectionError("derived per-variant metrics do not match the frozen coverage")
        for metric in per_variant_metrics:
            variant_id = metric["variantId"]
            limits = per_variant_gates[variant_id]
            for metric_name, gate_name in (
                ("pairedAugmentedScoreDeltaP95", "maxPairedAugmentedScoreDeltaP95"),
                ("pairedAugmentedScoreDeltaMax", "maxPairedAugmentedScoreDeltaMax"),
            ):
                value = float(metric[metric_name])
                maximum = limits[gate_name]
                if value > maximum:
                    reasons.append(
                        f"{category}.variant{variant_id}.{gate_name}={value:.12g} exceeds {maximum:.12g}"
                    )
    return reasons


def _objective_values(candidate_id: str, metrics: dict[str, dict[str, Any]]) -> dict[str, float | str]:
    thresholds = [value["threshold"] for value in metrics.values()]
    paired_p95 = [value["pairedAugmentedScoreDeltaP95"] for value in metrics.values()]
    return {
        "worstPairedAugmentedScoreDeltaP95": max(paired_p95),
        "worstCategoryThreshold": max(thresholds),
        "meanCategoryThreshold": sum(thresholds) / len(thresholds),
        "candidateId": candidate_id,
    }


def _objective_key(values: dict[str, float | str]) -> tuple[float, float, float, str]:
    return (
        float(values["worstPairedAugmentedScoreDeltaP95"]),
        float(values["worstCategoryThreshold"]),
        float(values["meanCategoryThreshold"]),
        str(values["candidateId"]),
    )


def run_selection(contract_path: Path, candidate_paths: dict[str, Path], output_path: Path) -> dict[str, Any]:
    """Evaluate a frozen normal-only candidate universe and write an immutable audit."""

    if _is_under(REPOSITORY_ROOT, output_path):
        raise SelectionError("selection output must stay outside the Git working tree")
    if _is_under(REPOSITORY_ROOT, contract_path):
        raise SelectionError("selection contract must stay outside the Git working tree")
    if output_path.exists():
        raise SelectionError("selection output already exists; choose a fresh immutable path")
    contract, contract_file_sha256 = _load_contract_with_digest(contract_path)
    candidate_configurations = _candidate_configurations(contract)
    candidate_ids = list(candidate_configurations)
    if set(candidate_paths) != set(candidate_ids):
        raise SelectionError("--candidate ids must exactly match the frozen candidates")
    resolved_paths = [str(candidate_paths[candidate_id].resolve()) for candidate_id in candidate_ids]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise SelectionError("each predeclared candidate requires a distinct report path")
    evaluations: list[dict[str, Any]] = []
    valid: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        path = candidate_paths[candidate_id]
        try:
            evaluated = validate_candidate_report(path, contract, candidate_configurations[candidate_id])
        except SelectionError as error:
            evaluations.append({
                "id": candidate_id,
                "reportPath": str(path),
                "state": "REJECTED_INVALID_REPORT",
                "rejectionReasons": [str(error)],
            })
            continue
        valid[candidate_id] = evaluated
        evaluations.append({
            "id": candidate_id,
            "reportPath": str(path),
            "reportSha256": evaluated["reportSha256"],
            "state": "VALID_PENDING_GATE",
            "rejectionReasons": [],
            "normalMetricsByCategory": evaluated["normalMetricsByCategory"],
        })

    valid_report_digests = [str(value["reportSha256"]) for value in valid.values()]
    if len(set(valid_report_digests)) != len(valid_report_digests):
        raise SelectionError("distinct candidates must not reuse the same report content digest")

    reference = _require_mapping(contract, "referenceCandidate")
    reference_id = _require_string(reference, "id")
    reference_result = valid.get(reference_id)
    reference_error: str | None = None
    if reference_result is None:
        reference_error = "reference candidate report is invalid"
    elif reference_result["reportSha256"] != reference["reportSha256"]:
        reference_error = "reference candidate report digest does not match the contract"
    eligible: list[dict[str, Any]] = []
    expected_variant_ids = _expected_augmentation_variant_ids(_require_mapping(contract, "candidateUniverse"))
    for evaluation in evaluations:
        candidate_id = evaluation["id"]
        if evaluation["state"] != "VALID_PENDING_GATE":
            continue
        if reference_error is not None:
            evaluation["state"] = "REJECTED_GATE"
            evaluation["rejectionReasons"] = [reference_error]
            continue
        reasons = _gate_reasons(
            valid[candidate_id]["normalMetricsByCategory"],
            reference_result["normalMetricsByCategory"],
            _require_mapping(contract, "gate"),
            expected_variant_ids,
        )
        if reasons:
            evaluation["state"] = "REJECTED_GATE"
            evaluation["rejectionReasons"] = reasons
            continue
        objective_values = _objective_values(candidate_id, valid[candidate_id]["normalMetricsByCategory"])
        evaluation["state"] = "ELIGIBLE"
        evaluation["objectiveValues"] = objective_values
        eligible.append(evaluation)
    selected = min(eligible, key=lambda item: _objective_key(item["objectiveValues"])) if eligible else None
    decision: dict[str, Any] = {
        "state": "RESEARCH_CONFIGURATION_LOCKED" if selected is not None else "NO_ELIGIBLE_CONFIGURATION",
        "selectionUses": "NORMAL_ONLY_NO_BLIND_OR_ANOMALY_INPUT",
        "objective": NORMAL_OBJECTIVE,
        "selectedCandidateId": None if selected is None else selected["id"],
        "disclaimer": "This normal-only research configuration lock does not prove anomaly detection, select a production threshold, qualify a physical device, or emit PASS/FAIL/equipment decisions.",
    }
    output = {
        "schemaVersion": SELECTION_SCHEMA_VERSION,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": SELECTION_PURPOSE,
        "contractPath": str(contract_path),
        "contractFileSha256": contract_file_sha256,
        "contractSha256": contract["contractSha256"],
        "selectorToolSha256": sha256_file(Path(__file__)),
        "candidateEvaluations": evaluations,
        "decision": decision,
    }
    output["selectionSha256"] = _document_digest(output, "selectionSha256")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _parse_candidate(value: str) -> tuple[str, Path]:
    candidate_id, separator, raw_path = value.partition("=")
    if not separator or not candidate_id or not raw_path:
        raise argparse.ArgumentTypeError("--candidate must use <id>=<normal-only-report.json>")
    return candidate_id, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one offline MVTec candidate from normal-only reports")
    parser.add_argument("--contract", type=Path, required=True, help="frozen normal-only selection contract")
    parser.add_argument("--candidate", action="append", type=_parse_candidate, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new immutable output outside this Git worktree")
    arguments = parser.parse_args()
    candidates: dict[str, Path] = {}
    for candidate_id, path in arguments.candidate:
        if candidate_id in candidates:
            parser.error("--candidate ids must be unique")
        candidates[candidate_id] = path
    try:
        output = run_selection(arguments.contract, candidates, arguments.output)
    except SelectionError as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(arguments.output),
        "state": output["decision"]["state"],
        "selectedCandidateId": output["decision"]["selectedCandidateId"],
    }, indent=2))


if __name__ == "__main__":
    main()
