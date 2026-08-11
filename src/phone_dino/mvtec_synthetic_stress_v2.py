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
import unicodedata
from pathlib import Path
from typing import Any, Callable

from phone_dino import mvtec_normal_successor as successor
from phone_dino import mvtec_successor_evaluator_v2 as knn
from phone_dino import mvtec_synthetic_anomaly_stress_v2 as augmentation
from phone_dino import sealed_dino_snapshot as sealed_snapshot


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# The r1 report remains an immutable historical artifact.  r2 makes the
# evidence limits explicit rather than silently treating r1 documents as if
# they had already carried the stronger contract.
SYNTHETIC_STRESS_V2_R1_REPORT_SCHEMA = "phone-dino.mvtec-ad-synthetic-stimulus-response-report/2.0"
SYNTHETIC_STRESS_V2_R2_REPORT_SCHEMA = "phone-dino.mvtec-ad-synthetic-stimulus-response-report/2.1"
SYNTHETIC_STRESS_V2_REPORT_SCHEMA = SYNTHETIC_STRESS_V2_R2_REPORT_SCHEMA
SYNTHETIC_STRESS_V2_HISTORICAL_REPORT_SCHEMAS = frozenset({SYNTHETIC_STRESS_V2_R1_REPORT_SCHEMA})
SYNTHETIC_STRESS_V2_R2_MIGRATION = "R1_2_0_REPORTS_ARE_HISTORICAL_AND_DO_NOT_SATISFY_THE_R2_2_1_CONTRACT"
SYNTHETIC_STRESS_V2_PURPOSE = "OFFLINE_MVTEC_SYNTHETIC_STIMULUS_AUGMENTATION_AND_RESPONSE_TESTING"
SYNTHETIC_STRESS_V2_PHASE = "POST_V1_SYNTHETIC_STIMULUS_STRESS_RESPONSE"
SYNTHETIC_STRESS_V2_METRIC_SCOPE = "SYNTHETIC_STIMULUS_RESPONSE_ONLY"
SYNTHETIC_STRESS_V2_INPUT_POLICY = "SUCCESSOR_V2_FIT_RAW_NORMAL_PARENTS_ONLY"
SYNTHETIC_STRESS_V2_BLIND_POLICY = "NO_BLIND_OR_TRUE_ANOMALY_DATA"
SYNTHETIC_STRESS_V2_RESULT_LABEL = "SYNTHETIC_ONLY_NOT_REAL_ANOMALY_PERFORMANCE"
SYNTHETIC_STRESS_V2_REAL_PERFORMANCE = "NOT_ESTIMATED"
SYNTHETIC_STRESS_V2_REAL_PRECISION_RECALL = "NOT_ESTIMATED"
SYNTHETIC_STRESS_V2_EVIDENCE_CLASS = "SYNTHETIC_ENGINEERING_ONLY"
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
    "realPrecisionRecall",
    "evidenceClass",
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
TEST_CONFIGURATION_FIELDS = {
    "algorithmId",
    "splitAlgorithm",
    "parentCountsPerCategory",
    "maxPrototypePatches",
    "topKMostAnomalousPatches",
    "prototypeBlockSize",
    "batchSize",
    "decisionRule",
    "rawCalibrationThresholdEstablishedBeforePackageLoad",
    "rawCalibrationThresholdEstablishedBeforePackageScoring",
    "queryPairing",
    "families",
    "renderIntensityLevels",
}
RESPONSE_COUNT_FIELDS = {
    "rawQueryCount",
    "rawQueryAboveThresholdCount",
    "syntheticStimulusCount",
    "syntheticStimulusAboveThresholdCount",
}
RESPONSE_RATE_FIELDS = {
    "rawQueryAboveThresholdRate",
    "syntheticStimulusAboveThresholdRate",
}
PAIRED_SCORE_DELTA_FIELDS = {
    "pairCount",
    "meanChildMinusParentScore",
    "medianChildMinusParentScore",
    "minimumChildMinusParentScore",
    "maximumChildMinusParentScore",
    "positiveChildMinusParentCount",
    "zeroChildMinusParentCount",
    "negativeChildMinusParentCount",
    "positiveChildMinusParentRate",
}
CONDITION_REPORT_FIELDS = {"responseCounts", "responseRates", "pairedScoreDeltaSummary"}
CATEGORY_REPORT_FIELDS = {
    "prototypeParentCount",
    "calibrationParentCount",
    "rawQueryParentCount",
    "syntheticStimulusCount",
    "prototypePatchCount",
    "thresholdFromRawCalibration",
    "responseCounts",
    "responseRates",
    "pairedScoreDeltaSummary",
    "bySyntheticDefectFamily",
    "byRenderIntensityLevel",
    "bySyntheticDefectFamilyAndRenderIntensityLevel",
}
EXECUTION_FIELDS = {
    "syntheticStressEvaluatorModuleSha256",
    "entrypointSha256",
    "timingBasis",
    "phaseTimingsSeconds",
    "python",
    "platform",
    "gitRevision",
}
FEATURE_EXTRACTOR_FIELDS = {
    "schemaVersion",
    "modelWeightsSha256",
    "modelRepositorySha256",
    "modelEntrypoint",
    "device",
    "preprocessingId",
    "algorithmId",
    "prototypeSelection",
    "syntheticStressEvaluatorModuleSha256",
    "syntheticStressAugmentationModuleSha256",
    "successorModuleSha256",
    "patchKnnModuleSha256",
    "productionModuleSha256",
    "enginesModuleSha256",
    "pythonVersion",
    "numpyVersion",
    "torchVersion",
    "torchvisionVersion",
}
SEALED_DINO_SNAPSHOT_PROVENANCE_FIELDS = {
    "schemaVersion",
    "snapshotSchemaVersion",
    "repositoryDigestAlgorithm",
    "weightsDigestAlgorithm",
    "snapshotManifestSha256",
    "snapshotRepositorySha256",
    "snapshotWeightsSha256",
    "snapshotGuardModuleSha256",
    "snapshotGuardModuleDigestAlgorithm",
}
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
MAX_REPORT_JSON_BYTES = 16 * 1024 * 1024
MAX_REPORT_JSON_DEPTH = 64
FORBIDDEN_CLASSIFICATION_FIELD_NAMES = {
    "syntheticTP",
    "syntheticFP",
    "syntheticFN",
    "syntheticTN",
    "syntheticPrecision",
    "syntheticRecall",
    "syntheticF1",
    "precision",
    "recall",
    "f1",
    "auroc",
    "averagePrecision",
}
FORBIDDEN_CLASSIFICATION_FIELD_NAME_NORMALIZATIONS = {
    "synthetictp",
    "syntheticfp",
    "syntheticfn",
    "synthetictn",
    "syntheticprecision",
    "syntheticrecall",
    "syntheticf1",
    "precision",
    "recall",
    "f1",
    "auroc",
    "rocauc",
    "auc",
    "averageprecision",
    "ap",
    "truepositive",
    "falsepositive",
    "truenegative",
    "falsenegative",
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
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SyntheticStressV2Error(f"{name} must be finite")
    try:
        coerced = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise SyntheticStressV2Error(f"{name} must be a representable finite number") from error
    if not math.isfinite(coerced):
        raise SyntheticStressV2Error(f"{name} must be finite")
    return coerced


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


def _directory_identity(path: Path) -> tuple[int, int, int]:
    status = path.stat(follow_symlinks=False)
    return status.st_dev, status.st_ino, status.st_mode


def _fd_signature(fd: int) -> tuple[int, int, int, int]:
    status = os.fstat(fd)
    return status.st_dev, status.st_ino, status.st_mode, status.st_size


def _serialize_response_report(document: dict[str, Any]) -> bytes:
    try:
        return (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise SyntheticStressV2Error("unable to serialize synthetic-stimulus response report as finite JSON") from error


def _validate_json_value(value: object, *, name: str, depth: int = 0) -> None:
    if depth > MAX_REPORT_JSON_DEPTH:
        raise SyntheticStressV2Error(f"{name} exceeds the report JSON nesting limit")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_JSON_INTEGER:
            raise SyntheticStressV2Error(f"{name} contains an integer outside the supported JSON range")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SyntheticStressV2Error(f"{name} contains a non-finite JSON number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, name=f"{name}[{index}]", depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SyntheticStressV2Error(f"{name} contains a non-string JSON key")
            _validate_json_value(item, name=f"{name}.{key}", depth=depth + 1)
        return
    raise SyntheticStressV2Error(f"{name} contains a value that is not JSON-compatible")


def _parse_report_json(raw: bytes) -> dict[str, Any]:
    """Parse one bounded JSON report while rejecting duplicate and unsafe numbers."""

    if not isinstance(raw, bytes):
        raise SyntheticStressV2Error("synthetic-stimulus response report JSON must be bytes")
    if len(raw) > MAX_REPORT_JSON_BYTES:
        raise SyntheticStressV2Error("synthetic-stimulus response report JSON exceeds the size limit")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SyntheticStressV2Error(f"synthetic-stimulus response report contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise SyntheticStressV2Error(f"synthetic-stimulus response report contains a non-finite JSON value: {value}")

    def bounded_integer(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise SyntheticStressV2Error("synthetic-stimulus response report contains an invalid JSON integer") from error
        if abs(parsed) > MAX_SAFE_JSON_INTEGER:
            raise SyntheticStressV2Error("synthetic-stimulus response report contains an oversized JSON integer")
        return parsed

    def finite_float(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise SyntheticStressV2Error("synthetic-stimulus response report contains an invalid JSON number") from error
        if not math.isfinite(parsed):
            raise SyntheticStressV2Error("synthetic-stimulus response report contains a non-finite JSON number")
        return parsed

    try:
        document = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
            parse_int=bounded_integer,
            parse_float=finite_float,
        )
    except SyntheticStressV2Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError, RecursionError, ValueError) as error:
        raise SyntheticStressV2Error("unable to parse synthetic-stimulus response report JSON") from error
    if not isinstance(document, dict):
        raise SyntheticStressV2Error("synthetic-stimulus response report JSON must be an object")
    _validate_json_value(document, name="synthetic-stimulus response report")
    return document


def _require_bounded_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise SyntheticStressV2Error(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _validate_response_condition(
    value: object,
    *,
    name: str,
    expected_raw_count: int | None = None,
    expected_stimulus_count: int | None = None,
    expected_pair_count: int | None = None,
) -> dict[str, Any]:
    condition = _require_exact_fields(value, name=name, fields=CONDITION_REPORT_FIELDS)
    counts = _require_exact_fields(condition["responseCounts"], name=f"{name} responseCounts", fields=RESPONSE_COUNT_FIELDS)
    raw_count = _require_bounded_int(counts["rawQueryCount"], name=f"{name} rawQueryCount", minimum=1, maximum=108)
    raw_above = _require_bounded_int(
        counts["rawQueryAboveThresholdCount"], name=f"{name} rawQueryAboveThresholdCount", minimum=0, maximum=raw_count
    )
    stimulus_count = _require_bounded_int(
        counts["syntheticStimulusCount"], name=f"{name} syntheticStimulusCount", minimum=1, maximum=972
    )
    stimulus_above = _require_bounded_int(
        counts["syntheticStimulusAboveThresholdCount"],
        name=f"{name} syntheticStimulusAboveThresholdCount",
        minimum=0,
        maximum=stimulus_count,
    )
    if expected_raw_count is not None and raw_count != expected_raw_count:
        raise SyntheticStressV2Error(f"{name} raw query count is inconsistent")
    if expected_stimulus_count is not None and stimulus_count != expected_stimulus_count:
        raise SyntheticStressV2Error(f"{name} synthetic-stimulus count is inconsistent")

    rates = _require_exact_fields(condition["responseRates"], name=f"{name} responseRates", fields=RESPONSE_RATE_FIELDS)
    raw_rate = _require_finite(rates["rawQueryAboveThresholdRate"], name=f"{name} raw-query response rate")
    stimulus_rate = _require_finite(
        rates["syntheticStimulusAboveThresholdRate"], name=f"{name} synthetic-stimulus response rate"
    )
    if not math.isclose(raw_rate, raw_above / raw_count, rel_tol=0.0, abs_tol=1e-12):
        raise SyntheticStressV2Error(f"{name} raw-query response rate is inconsistent")
    if not math.isclose(stimulus_rate, stimulus_above / stimulus_count, rel_tol=0.0, abs_tol=1e-12):
        raise SyntheticStressV2Error(f"{name} synthetic-stimulus response rate is inconsistent")

    paired = _require_exact_fields(
        condition["pairedScoreDeltaSummary"], name=f"{name} pairedScoreDeltaSummary", fields=PAIRED_SCORE_DELTA_FIELDS
    )
    pair_count = _require_bounded_int(paired["pairCount"], name=f"{name} pairCount", minimum=1, maximum=972)
    if expected_pair_count is not None and pair_count != expected_pair_count:
        raise SyntheticStressV2Error(f"{name} pair count is inconsistent")
    for field in (
        "meanChildMinusParentScore",
        "medianChildMinusParentScore",
        "minimumChildMinusParentScore",
        "maximumChildMinusParentScore",
        "positiveChildMinusParentRate",
    ):
        _require_finite(paired[field], name=f"{name} {field}")
    positive = _require_bounded_int(
        paired["positiveChildMinusParentCount"], name=f"{name} positiveChildMinusParentCount", minimum=0, maximum=pair_count
    )
    zero = _require_bounded_int(
        paired["zeroChildMinusParentCount"], name=f"{name} zeroChildMinusParentCount", minimum=0, maximum=pair_count
    )
    negative = _require_bounded_int(
        paired["negativeChildMinusParentCount"], name=f"{name} negativeChildMinusParentCount", minimum=0, maximum=pair_count
    )
    if positive + zero + negative != pair_count:
        raise SyntheticStressV2Error(f"{name} paired score-delta counts are inconsistent")
    if not math.isclose(
        _require_finite(paired["positiveChildMinusParentRate"], name=f"{name} positiveChildMinusParentRate"),
        positive / pair_count,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SyntheticStressV2Error(f"{name} positive child-minus-parent rate is inconsistent")
    return condition


def _reject_classification_fields(value: object, *, name: str) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_classification_fields(item, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = "".join(
                character
                for character in unicodedata.normalize("NFKC", key).casefold()
                if character.isalnum()
            )
            if (
                key in FORBIDDEN_CLASSIFICATION_FIELD_NAMES
                or normalized in FORBIDDEN_CLASSIFICATION_FIELD_NAME_NORMALIZATIONS
            ):
                raise SyntheticStressV2Error(f"{name} contains a forbidden classification field: {key}")
            _reject_classification_fields(item, name=f"{name}.{key}")


def _validate_report_configuration(value: object) -> dict[str, Any]:
    configuration = _require_exact_fields(value, name="synthetic-stimulus report testConfiguration", fields=TEST_CONFIGURATION_FIELDS)
    expected_scalars = {
        "algorithmId": SYNTHETIC_STRESS_V2_ALGORITHM,
        "splitAlgorithm": SYNTHETIC_STRESS_V2_PARENT_SPLIT_ALGORITHM,
        "maxPrototypePatches": SYNTHETIC_STRESS_V2_MAX_PROTOTYPE_PATCHES,
        "topKMostAnomalousPatches": SYNTHETIC_STRESS_V2_TOP_K,
        "prototypeBlockSize": SYNTHETIC_STRESS_V2_BLOCK_SIZE,
        "batchSize": SYNTHETIC_STRESS_V2_BATCH_SIZE,
        "decisionRule": SYNTHETIC_STRESS_V2_DECISION_RULE,
        "queryPairing": "RAW_QUERY_PARENT_AND_SYNTHETIC_CHILDREN_ARE_NOT_INDEPENDENT",
    }
    for field, expected in expected_scalars.items():
        if configuration[field] != expected:
            raise SyntheticStressV2Error(f"synthetic-stimulus report testConfiguration {field} is unsupported")
    if configuration["parentCountsPerCategory"] != dict(SYNTHETIC_STRESS_V2_PARENT_COUNTS):
        raise SyntheticStressV2Error("synthetic-stimulus report parent counts are unsupported")
    if configuration["families"] != list(SYNTHETIC_STRESS_V2_FAMILIES):
        raise SyntheticStressV2Error("synthetic-stimulus report families are unsupported")
    if configuration["renderIntensityLevels"] != list(SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS):
        raise SyntheticStressV2Error("synthetic-stimulus report render levels are unsupported")
    if configuration["rawCalibrationThresholdEstablishedBeforePackageLoad"] is not True:
        raise SyntheticStressV2Error("synthetic-stimulus report must establish the raw threshold before package load")
    if configuration["rawCalibrationThresholdEstablishedBeforePackageScoring"] is not True:
        raise SyntheticStressV2Error("synthetic-stimulus report must establish the raw threshold before package scoring")
    return configuration


def _validate_score_records(value: object, *, name: str, fields: set[str], expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise SyntheticStressV2Error(f"{name} must contain exactly {expected_count} records")
    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for index, item in enumerate(value):
        record = _require_exact_fields(item, name=f"{name}[{index}]", fields=fields)
        case_id = _require_string(record["caseId"], name=f"{name}[{index}] caseId")
        if case_id in seen_case_ids:
            raise SyntheticStressV2Error(f"{name} contains a duplicate caseId")
        seen_case_ids.add(case_id)
        if _require_string(record["category"], name=f"{name}[{index}] category") not in SYNTHETIC_STRESS_V2_CATEGORIES:
            raise SyntheticStressV2Error(f"{name} contains an unsupported category")
        _require_sha256(record["sourceSha256"], name=f"{name}[{index}] sourceSha256")
        components = {
            field: _require_finite(record[field], name=f"{name}[{index}] {field}")
            for field in ("score", "maxPatchDistance", "meanNearestPatchDistance")
        }
        if any(component < 0.0 or component > 2.0 for component in components.values()):
            raise SyntheticStressV2Error(f"{name}[{index}] patch-distance score components must be in [0, 2]")
        if not (
            components["meanNearestPatchDistance"]
            <= components["score"]
            <= components["maxPatchDistance"]
        ):
            raise SyntheticStressV2Error(
                f"{name}[{index}] must satisfy meanNearestPatchDistance <= score <= maxPatchDistance"
            )
        if "parentCaseId" in fields:
            _require_string(record["parentCaseId"], name=f"{name}[{index}] parentCaseId")
        if "aboveRawCalibrationThreshold" in fields and not isinstance(record["aboveRawCalibrationThreshold"], bool):
            raise SyntheticStressV2Error(f"{name}[{index}] aboveRawCalibrationThreshold must be a boolean")
        if "syntheticDefectFamily" in fields and record["syntheticDefectFamily"] not in SYNTHETIC_STRESS_V2_FAMILIES:
            raise SyntheticStressV2Error(f"{name}[{index}] syntheticDefectFamily is unsupported")
        if "renderIntensityLevel" in fields and record["renderIntensityLevel"] not in SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS:
            raise SyntheticStressV2Error(f"{name}[{index}] renderIntensityLevel is unsupported")
        result.append(record)
    return result


def _validate_and_index_parent_split(value: object) -> dict[str, dict[str, dict[str, Any]]]:
    """Validate the immutable parent split and index it by role/case.

    Count-only validation is insufficient here: a self-digested report could
    replace one parent record with another record in the same role/category.
    The score details below are meaningful only when every parent identity is
    unique and can be reconciled to exactly one split record.
    """

    if not isinstance(value, list) or len(value) != 36:
        raise SyntheticStressV2Error("synthetic-stimulus response report parent split is incomplete")
    split_counts = {
        role: {category: 0 for category in SYNTHETIC_STRESS_V2_CATEGORIES}
        for role in SYNTHETIC_STRESS_V2_PARENT_COUNTS
    }
    by_role: dict[str, dict[str, dict[str, Any]]] = {
        role: {} for role in SYNTHETIC_STRESS_V2_PARENT_COUNTS
    }
    seen_case_ids: set[str] = set()
    seen_source_sha256s: set[str] = set()
    seen_source_group_ids: set[str] = set()
    observed_case_ids: list[str] = []
    for index, item in enumerate(value):
        record = _require_exact_fields(
            item,
            name=f"synthetic-stimulus response report parentSplit[{index}]",
            fields=PARENT_SPLIT_RECORD_FIELDS,
        )
        role = record["role"]
        if role not in SYNTHETIC_STRESS_V2_PARENT_COUNTS:
            raise SyntheticStressV2Error("synthetic-stimulus response report parent split record has an unsupported role")
        case_id = _require_string(record["caseId"], name=f"synthetic-stimulus response report parentSplit[{index}] caseId")
        source_group_id = _require_string(
            record["sourceGroupId"], name=f"synthetic-stimulus response report parentSplit[{index}] sourceGroupId"
        )
        source_sha256 = _require_sha256(
            record["sourceSha256"], name=f"synthetic-stimulus response report parentSplit[{index}] sourceSha256"
        )
        category = record["category"]
        if category not in SYNTHETIC_STRESS_V2_CATEGORIES:
            raise SyntheticStressV2Error("synthetic-stimulus response report parent split record has an unsupported category")
        if case_id in seen_case_ids:
            raise SyntheticStressV2Error("synthetic-stimulus response report parent split has a duplicate caseId")
        if source_sha256 in seen_source_sha256s:
            raise SyntheticStressV2Error("synthetic-stimulus response report parent split has a duplicate sourceSha256")
        if source_group_id in seen_source_group_ids:
            raise SyntheticStressV2Error("synthetic-stimulus response report parent split has a duplicate sourceGroupId")
        seen_case_ids.add(case_id)
        seen_source_sha256s.add(source_sha256)
        seen_source_group_ids.add(source_group_id)
        observed_case_ids.append(case_id)
        by_role[role][case_id] = record
        split_counts[role][category] += 1
    if observed_case_ids != sorted(observed_case_ids):
        raise SyntheticStressV2Error("synthetic-stimulus response report parent split must be sorted by caseId")
    for role, expected_per_category in SYNTHETIC_STRESS_V2_PARENT_COUNTS.items():
        if any(count != expected_per_category for count in split_counts[role].values()):
            raise SyntheticStressV2Error(f"synthetic-stimulus response report {role} split is inconsistent")
    return by_role


def _numbers_match(actual: object, expected: float, *, name: str) -> None:
    value = _require_finite(actual, name=name)
    # Every compared value originates in the same JSON score detail and is
    # serialized with Python's exact float representation.  A tolerance would
    # let a self-digested near-threshold mutation claim a false flag/summary.
    if value != expected:
        raise SyntheticStressV2Error(f"{name} is inconsistent with score details")


def _validate_feature_extractor(value: object) -> dict[str, Any]:
    """Close the report identity surface to the fixed V2 DINO contract."""

    if not isinstance(value, dict):
        raise SyntheticStressV2Error("synthetic-stimulus response report feature extractor must be an object")
    required_fields = FEATURE_EXTRACTOR_FIELDS | {sealed_snapshot.SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_FIELD}
    allowed_fields = {frozenset(required_fields)}
    if frozenset(value) not in allowed_fields:
        missing = required_fields.difference(value)
        unknown = set(value).difference(required_fields)
        detail: list[str] = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unsupported {', '.join(sorted(unknown))}")
        if not detail:
            detail.append("unsupported feature extractor field combination")
        raise SyntheticStressV2Error(
            f"synthetic-stimulus response report feature extractor has {'; '.join(detail)} fields"
        )
    identity = value
    expected_scalars = {
        "schemaVersion": "phone-dino.mvtec-ad-synthetic-stimulus-feature-extractor/2.0",
        "modelEntrypoint": "dinov2_vits14",
        "device": "cpu",
        "preprocessingId": knn.SUCCESSOR_V2_PREPROCESSING_ID,
        "algorithmId": SYNTHETIC_STRESS_V2_ALGORITHM,
        "prototypeSelection": SYNTHETIC_STRESS_V2_PROTOTYPE_SELECTION,
    }
    for field, expected in expected_scalars.items():
        if identity[field] != expected:
            raise SyntheticStressV2Error(f"synthetic-stimulus response report feature extractor {field} is unsupported")
    for field in (
        "modelWeightsSha256",
        "modelRepositorySha256",
        "syntheticStressEvaluatorModuleSha256",
        "syntheticStressAugmentationModuleSha256",
        "successorModuleSha256",
        "patchKnnModuleSha256",
        "productionModuleSha256",
        "enginesModuleSha256",
    ):
        _require_sha256(identity[field], name=f"synthetic-stimulus response report feature extractor {field}")
    for field in ("pythonVersion", "numpyVersion", "torchVersion", "torchvisionVersion"):
        _require_string(identity[field], name=f"synthetic-stimulus response report feature extractor {field}")
    if sealed_snapshot.SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_FIELD in identity:
        sealed_provenance = identity[sealed_snapshot.SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_FIELD]
        provenance = _require_exact_fields(
            sealed_provenance,
            name="synthetic-stimulus response report sealed DINO snapshot provenance",
            fields=SEALED_DINO_SNAPSHOT_PROVENANCE_FIELDS,
        )
        if provenance["schemaVersion"] != sealed_snapshot.SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_SCHEMA:
            raise SyntheticStressV2Error("synthetic-stimulus response report sealed DINO provenance schema is unsupported")
        if provenance["snapshotSchemaVersion"] != sealed_snapshot.SEALED_DINO_SNAPSHOT_SCHEMA:
            raise SyntheticStressV2Error("synthetic-stimulus response report sealed DINO snapshot schema is unsupported")
        if provenance["repositoryDigestAlgorithm"] != sealed_snapshot.SEALED_DINO_REPOSITORY_DIGEST_ALGORITHM:
            raise SyntheticStressV2Error("synthetic-stimulus response report sealed DINO repository digest algorithm is unsupported")
        if provenance["weightsDigestAlgorithm"] != sealed_snapshot.SEALED_DINO_WEIGHTS_DIGEST_ALGORITHM:
            raise SyntheticStressV2Error("synthetic-stimulus response report sealed DINO weights digest algorithm is unsupported")
        if (
            provenance["snapshotGuardModuleDigestAlgorithm"]
            != sealed_snapshot.SEALED_DINO_SNAPSHOT_GUARD_MODULE_DIGEST_ALGORITHM
        ):
            raise SyntheticStressV2Error(
                "synthetic-stimulus response report sealed DINO snapshot guard module digest algorithm is unsupported"
            )
        for field in (
            "snapshotManifestSha256",
            "snapshotRepositorySha256",
            "snapshotWeightsSha256",
            "snapshotGuardModuleSha256",
        ):
            _require_sha256(provenance[field], name=f"synthetic-stimulus response report sealed DINO provenance {field}")
    return identity


def _validate_condition_matches_score_details(
    value: object,
    *,
    name: str,
    raw_scores: list[dict[str, Any]],
    stimulus_scores: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    """Require every published condition summary to be derived from details."""

    condition = _require_exact_fields(value, name=name, fields=CONDITION_REPORT_FIELDS)
    actual_counts = _require_exact_fields(
        condition["responseCounts"], name=f"{name} responseCounts", fields=RESPONSE_COUNT_FIELDS
    )
    expected_counts = _response_counts(raw_scores, stimulus_scores)
    if actual_counts != expected_counts:
        raise SyntheticStressV2Error(f"{name} response counts are inconsistent with score details")
    actual_rates = _require_exact_fields(
        condition["responseRates"], name=f"{name} responseRates", fields=RESPONSE_RATE_FIELDS
    )
    expected_rates = _response_rates(expected_counts)
    for field, expected in expected_rates.items():
        _numbers_match(actual_rates[field], expected, name=f"{name} {field}")
    actual_paired = _require_exact_fields(
        condition["pairedScoreDeltaSummary"],
        name=f"{name} pairedScoreDeltaSummary",
        fields=PAIRED_SCORE_DELTA_FIELDS,
    )
    expected_paired = _paired_score_delta_summary(pairs)
    for field in (
        "pairCount",
        "positiveChildMinusParentCount",
        "zeroChildMinusParentCount",
        "negativeChildMinusParentCount",
    ):
        if actual_paired[field] != expected_paired[field]:
            raise SyntheticStressV2Error(f"{name} {field} is inconsistent with score details")
    for field in (
        "meanChildMinusParentScore",
        "medianChildMinusParentScore",
        "minimumChildMinusParentScore",
        "maximumChildMinusParentScore",
        "positiveChildMinusParentRate",
    ):
        _numbers_match(actual_paired[field], expected_paired[field], name=f"{name} {field}")


def _reconcile_detail_scores(
    *,
    calibration_scores: list[dict[str, Any]],
    raw_query_scores: list[dict[str, Any]],
    stimulus_scores: list[dict[str, Any]],
    thresholds: dict[str, float],
    parent_split: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Bind every detailed score to the split, threshold, and one parent."""

    calibration_parents = parent_split["SYNTHETIC_CALIBRATION"]
    query_parents = parent_split["SYNTHETIC_QUERY"]
    if {record["caseId"] for record in calibration_scores} != set(calibration_parents):
        raise SyntheticStressV2Error("synthetic-stimulus calibration scores do not match the parent split")
    for record in calibration_scores:
        parent = calibration_parents[record["caseId"]]
        if record["category"] != parent["category"] or record["sourceSha256"] != parent["sourceSha256"]:
            raise SyntheticStressV2Error("synthetic-stimulus calibration score does not bind its parent split record")
    if {record["caseId"] for record in raw_query_scores} != set(query_parents):
        raise SyntheticStressV2Error("synthetic-stimulus raw query scores do not match the parent split")
    raw_by_case: dict[str, dict[str, Any]] = {}
    for record in raw_query_scores:
        parent = query_parents[record["caseId"]]
        if (
            record["parentCaseId"] != record["caseId"]
            or record["category"] != parent["category"]
            or record["sourceSha256"] != parent["sourceSha256"]
        ):
            raise SyntheticStressV2Error("synthetic-stimulus raw query score does not bind its parent split record")
        if record["aboveRawCalibrationThreshold"] != (record["score"] > thresholds[record["category"]]):
            raise SyntheticStressV2Error("synthetic-stimulus raw query threshold flag is inconsistent with score details")
        raw_by_case[record["caseId"]] = record
    for category in SYNTHETIC_STRESS_V2_CATEGORIES:
        calibration_values = [record["score"] for record in calibration_scores if record["category"] == category]
        if len(calibration_values) != SYNTHETIC_STRESS_V2_PARENT_COUNTS["SYNTHETIC_CALIBRATION"]:
            raise SyntheticStressV2Error("synthetic-stimulus calibration category coverage is inconsistent")
        _numbers_match(
            thresholds[category], max(calibration_values), name=f"synthetic-stimulus report threshold {category}"
        )

    expected_combinations = {
        (parent_case_id, family, level)
        for parent_case_id in raw_by_case
        for family in SYNTHETIC_STRESS_V2_FAMILIES
        for level in SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS
    }
    observed_combinations: set[tuple[str, str, str]] = set()
    pairs: list[dict[str, Any]] = []
    for record in stimulus_scores:
        parent = raw_by_case.get(record["parentCaseId"])
        if parent is None or record["category"] != parent["category"]:
            raise SyntheticStressV2Error("synthetic-stimulus score does not pair to one raw query parent")
        expected_case_id = (
            f"{record['parentCaseId']}/synthetic-only-stress-v2/"
            f"{record['renderIntensityLevel'].lower()}/{record['syntheticDefectFamily'].lower()}"
        )
        if record["caseId"] != expected_case_id:
            raise SyntheticStressV2Error("synthetic-stimulus score caseId is not the deterministic V2 child identity")
        if record["aboveRawCalibrationThreshold"] != (record["score"] > thresholds[record["category"]]):
            raise SyntheticStressV2Error("synthetic-stimulus threshold flag is inconsistent with score details")
        combination = (
            record["parentCaseId"],
            record["syntheticDefectFamily"],
            record["renderIntensityLevel"],
        )
        if combination in observed_combinations:
            raise SyntheticStressV2Error("synthetic-stimulus score parent/family/level is duplicated")
        observed_combinations.add(combination)
        pairs.append(
            {
                "caseId": record["caseId"],
                "parentCaseId": record["parentCaseId"],
                "category": record["category"],
                "syntheticDefectFamily": record["syntheticDefectFamily"],
                "renderIntensityLevel": record["renderIntensityLevel"],
                "parentScore": parent["score"],
                "childScore": record["score"],
                "childMinusParentScore": record["score"] - parent["score"],
            }
        )
    if observed_combinations != expected_combinations:
        raise SyntheticStressV2Error("synthetic-stimulus scores do not cover every parent, family, and render level")
    return pairs


def _reconcile_report_summaries(
    *,
    categories: dict[str, Any],
    aggregate: object,
    calibration_scores: list[dict[str, Any]],
    raw_query_scores: list[dict[str, Any]],
    stimulus_scores: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    thresholds: dict[str, float],
    parent_split: dict[str, dict[str, dict[str, Any]]],
) -> None:
    _validate_condition_matches_score_details(
        aggregate,
        name="synthetic-stimulus report aggregate",
        raw_scores=raw_query_scores,
        stimulus_scores=stimulus_scores,
        pairs=pairs,
    )
    for category in SYNTHETIC_STRESS_V2_CATEGORIES:
        report = categories[category]
        category_calibration = [record for record in calibration_scores if record["category"] == category]
        category_raw = [record for record in raw_query_scores if record["category"] == category]
        category_stimuli = [record for record in stimulus_scores if record["category"] == category]
        category_pairs = [record for record in pairs if record["category"] == category]
        if report["prototypeParentCount"] != sum(
            record["category"] == category for record in parent_split["SYNTHETIC_PROTOTYPE"].values()
        ):
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} prototype count is inconsistent with parent split")
        if report["calibrationParentCount"] != len(category_calibration):
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} calibration count is inconsistent with score details")
        if report["rawQueryParentCount"] != len(category_raw):
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} raw query count is inconsistent with score details")
        if report["syntheticStimulusCount"] != len(category_stimuli):
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} stimulus count is inconsistent with score details")
        _numbers_match(
            report["thresholdFromRawCalibration"], thresholds[category],
            name=f"synthetic-stimulus report category {category} threshold",
        )
        _validate_condition_matches_score_details(
            {field: report[field] for field in CONDITION_REPORT_FIELDS},
            name=f"synthetic-stimulus report category {category}",
            raw_scores=category_raw,
            stimulus_scores=category_stimuli,
            pairs=category_pairs,
        )
        for family in SYNTHETIC_STRESS_V2_FAMILIES:
            family_stimuli = [record for record in category_stimuli if record["syntheticDefectFamily"] == family]
            family_pairs = [record for record in category_pairs if record["syntheticDefectFamily"] == family]
            _validate_condition_matches_score_details(
                report["bySyntheticDefectFamily"][family],
                name=f"synthetic-stimulus report category {category} family {family}",
                raw_scores=category_raw,
                stimulus_scores=family_stimuli,
                pairs=family_pairs,
            )
            for level in SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS:
                condition_stimuli = [
                    record
                    for record in family_stimuli
                    if record["renderIntensityLevel"] == level
                ]
                condition_pairs = [
                    record
                    for record in family_pairs
                    if record["renderIntensityLevel"] == level
                ]
                _validate_condition_matches_score_details(
                    report["bySyntheticDefectFamilyAndRenderIntensityLevel"][family][level],
                    name=f"synthetic-stimulus report category {category} family {family} level {level}",
                    raw_scores=category_raw,
                    stimulus_scores=condition_stimuli,
                    pairs=condition_pairs,
                )
        for level in SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS:
            level_stimuli = [record for record in category_stimuli if record["renderIntensityLevel"] == level]
            level_pairs = [record for record in category_pairs if record["renderIntensityLevel"] == level]
            _validate_condition_matches_score_details(
                report["byRenderIntensityLevel"][level],
                name=f"synthetic-stimulus report category {category} render level {level}",
                raw_scores=category_raw,
                stimulus_scores=level_stimuli,
                pairs=level_pairs,
            )


def _validate_category_reports(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(SYNTHETIC_STRESS_V2_CATEGORIES):
        raise SyntheticStressV2Error("synthetic-stimulus report categories must cover the fixed categories")
    for category, item in value.items():
        report = _require_exact_fields(item, name=f"synthetic-stimulus report category {category}", fields=CATEGORY_REPORT_FIELDS)
        if report["prototypeParentCount"] != SYNTHETIC_STRESS_V2_PARENT_COUNTS["SYNTHETIC_PROTOTYPE"]:
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} prototype count is inconsistent")
        if report["calibrationParentCount"] != SYNTHETIC_STRESS_V2_PARENT_COUNTS["SYNTHETIC_CALIBRATION"]:
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} calibration count is inconsistent")
        if report["rawQueryParentCount"] != SYNTHETIC_STRESS_V2_PARENT_COUNTS["SYNTHETIC_QUERY"]:
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} raw query count is inconsistent")
        if report["syntheticStimulusCount"] != 36:
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} stimulus count is inconsistent")
        _require_bounded_int(report["prototypePatchCount"], name=f"synthetic-stimulus report category {category} prototypePatchCount", minimum=1, maximum=SYNTHETIC_STRESS_V2_MAX_PROTOTYPE_PATCHES)
        _require_finite(report["thresholdFromRawCalibration"], name=f"synthetic-stimulus report category {category} threshold")
        _validate_response_condition(
            {field: report[field] for field in CONDITION_REPORT_FIELDS},
            name=f"synthetic-stimulus report category {category}",
            expected_raw_count=4,
            expected_stimulus_count=36,
            expected_pair_count=36,
        )
        by_family = report["bySyntheticDefectFamily"]
        if not isinstance(by_family, dict) or set(by_family) != set(SYNTHETIC_STRESS_V2_FAMILIES):
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} family coverage is inconsistent")
        for family, condition in by_family.items():
            _validate_response_condition(condition, name=f"synthetic-stimulus report category {category} family {family}", expected_raw_count=4, expected_stimulus_count=12, expected_pair_count=12)
        by_level = report["byRenderIntensityLevel"]
        if not isinstance(by_level, dict) or set(by_level) != set(SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS):
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} render-level coverage is inconsistent")
        for level, condition in by_level.items():
            _validate_response_condition(condition, name=f"synthetic-stimulus report category {category} render level {level}", expected_raw_count=4, expected_stimulus_count=12, expected_pair_count=12)
        by_family_level = report["bySyntheticDefectFamilyAndRenderIntensityLevel"]
        if not isinstance(by_family_level, dict) or set(by_family_level) != set(SYNTHETIC_STRESS_V2_FAMILIES):
            raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} family-level coverage is inconsistent")
        for family, levels in by_family_level.items():
            if not isinstance(levels, dict) or set(levels) != set(SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS):
                raise SyntheticStressV2Error(f"synthetic-stimulus report category {category} family-level matrix is inconsistent")
            for level, condition in levels.items():
                _validate_response_condition(
                    condition,
                    name=f"synthetic-stimulus report category {category} family {family} level {level}",
                    expected_raw_count=4,
                    expected_stimulus_count=4,
                    expected_pair_count=4,
                )
    return value


def validate_response_only_report_v2_r2(document: object) -> dict[str, Any]:
    """Validate the r2 JSON-only response contract without opening any images."""

    _validate_json_value(document, name="synthetic-stimulus response report")
    report = _require_exact_fields(document, name="synthetic-stimulus response report", fields=REPORT_FIELDS)
    if report["schemaVersion"] == SYNTHETIC_STRESS_V2_R1_REPORT_SCHEMA:
        raise SyntheticStressV2Error(
            "historical r1 synthetic-stimulus response reports do not satisfy the r2 JSON contract"
        )
    if report["schemaVersion"] != SYNTHETIC_STRESS_V2_R2_REPORT_SCHEMA:
        raise SyntheticStressV2Error("synthetic-stimulus response report schema is unsupported")
    expected_scope = {
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "postV1Exploratory": True,
        "comparisonOrPromotionAllowed": False,
        "purpose": SYNTHETIC_STRESS_V2_PURPOSE,
        "phase": SYNTHETIC_STRESS_V2_PHASE,
        "metricScope": SYNTHETIC_STRESS_V2_METRIC_SCOPE,
        "realAnomalyPerformance": SYNTHETIC_STRESS_V2_REAL_PERFORMANCE,
        "realPrecisionRecall": SYNTHETIC_STRESS_V2_REAL_PRECISION_RECALL,
        "evidenceClass": SYNTHETIC_STRESS_V2_EVIDENCE_CLASS,
        "inputPolicy": SYNTHETIC_STRESS_V2_INPUT_POLICY,
        "blindPolicy": SYNTHETIC_STRESS_V2_BLIND_POLICY,
        "resultLabel": SYNTHETIC_STRESS_V2_RESULT_LABEL,
    }
    for field, expected in expected_scope.items():
        if report[field] != expected:
            raise SyntheticStressV2Error(f"synthetic-stimulus response report {field} is unsupported")
    if report["forbiddenUses"] != list(SYNTHETIC_STRESS_V2_FORBIDDEN_USES):
        raise SyntheticStressV2Error("synthetic-stimulus response report forbiddenUses is unsupported")
    _reject_classification_fields(report, name="synthetic-stimulus response report")
    _validate_report_configuration(report["testConfiguration"])
    for field in (
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
        "syntheticStressReportSha256",
    ):
        _require_sha256(report[field], name=f"synthetic-stimulus response report {field}")
    if report["augmentationSchemaVersion"] != augmentation.SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA:
        raise SyntheticStressV2Error("synthetic-stimulus response report augmentation schema is unsupported")
    calibration_scores = _validate_score_records(
        report["calibrationScores"],
        name="synthetic-stimulus calibrationScores",
        fields=CALIBRATION_SCORE_FIELDS,
        expected_count=6,
    )
    raw_query_scores = _validate_score_records(
        report["rawQueryScores"],
        name="synthetic-stimulus rawQueryScores",
        fields=RAW_QUERY_SCORE_FIELDS,
        expected_count=12,
    )
    stimulus_scores = _validate_score_records(
        report["stimulusScores"],
        name="synthetic-stimulus stimulusScores",
        fields=STIMULUS_SCORE_FIELDS,
        expected_count=108,
    )
    if not isinstance(report["thresholds"], dict) or set(report["thresholds"]) != set(SYNTHETIC_STRESS_V2_CATEGORIES):
        raise SyntheticStressV2Error("synthetic-stimulus response report thresholds are incomplete")
    for category, value in report["thresholds"].items():
        _require_finite(value, name=f"synthetic-stimulus response report threshold {category}")
    categories = _validate_category_reports(report["categories"])
    _validate_response_condition(
        report["aggregate"],
        name="synthetic-stimulus report aggregate",
        expected_raw_count=12,
        expected_stimulus_count=108,
        expected_pair_count=108,
    )
    parent_split = _validate_and_index_parent_split(report["parentSplit"])
    if successor.canonical_json_sha256(report["parentSplit"]) != report["parentSplitIdentitySha256"]:
        raise SyntheticStressV2Error("synthetic-stimulus response report parent split digest is inconsistent")
    pairs = _reconcile_detail_scores(
        calibration_scores=calibration_scores,
        raw_query_scores=raw_query_scores,
        stimulus_scores=stimulus_scores,
        thresholds=report["thresholds"],
        parent_split=parent_split,
    )
    _reconcile_report_summaries(
        categories=categories,
        aggregate=report["aggregate"],
        calibration_scores=calibration_scores,
        raw_query_scores=raw_query_scores,
        stimulus_scores=stimulus_scores,
        pairs=pairs,
        thresholds=report["thresholds"],
        parent_split=parent_split,
    )
    feature_extractor = _validate_feature_extractor(report["featureExtractor"])
    if successor.canonical_json_sha256(feature_extractor) != report["featureExtractorIdentitySha256"]:
        raise SyntheticStressV2Error("synthetic-stimulus response report feature extractor digest is inconsistent")
    execution = _require_exact_fields(report["execution"], name="synthetic-stimulus response report execution", fields=EXECUTION_FIELDS)
    _require_sha256(execution["syntheticStressEvaluatorModuleSha256"], name="synthetic-stimulus response report evaluator digest")
    _require_sha256(execution["entrypointSha256"], name="synthetic-stimulus response report entrypoint digest")
    if execution["syntheticStressEvaluatorModuleSha256"] != feature_extractor["syntheticStressEvaluatorModuleSha256"]:
        raise SyntheticStressV2Error(
            "synthetic-stimulus response report evaluator digest does not match the feature extractor identity"
        )
    if execution["timingBasis"] != "PROCESS_CPU_TIME_EXCLUDES_SUSPEND":
        raise SyntheticStressV2Error("synthetic-stimulus response report timing basis is unsupported")
    if not isinstance(execution["phaseTimingsSeconds"], dict) or set(execution["phaseTimingsSeconds"]) != {
        "inputAssemblySeconds", "provenanceSeconds", "featureProcessingSeconds", "scoringSeconds", "totalElapsedSeconds"
    }:
        raise SyntheticStressV2Error("synthetic-stimulus response report phase timings are unsupported")
    for field, value in execution["phaseTimingsSeconds"].items():
        if _require_finite(value, name=f"synthetic-stimulus response report timing {field}") < 0.0:
            raise SyntheticStressV2Error("synthetic-stimulus response report timings must be non-negative")
    _require_string(execution["python"], name="synthetic-stimulus response report python")
    _require_string(execution["platform"], name="synthetic-stimulus response report platform")
    if execution["gitRevision"] is not None:
        _require_string(execution["gitRevision"], name="synthetic-stimulus response report gitRevision")
    if _document_digest(report) != report["syntheticStressReportSha256"]:
        raise SyntheticStressV2Error("synthetic-stimulus response report digest is inconsistent")
    return report


def parse_response_only_report_v2_r2_json(raw: bytes) -> dict[str, Any]:
    """Parse and strictly validate an r2 response report without opening artifacts."""

    return validate_response_only_report_v2_r2(_parse_report_json(raw))


def _validate_v2_package_metadata_against_report_parent_split(
    manifest: dict[str, Any],
    *,
    recipe: dict[str, Any],
    recipe_file_sha256: str,
    report: dict[str, Any],
    repository_root: Path,
) -> list[dict[str, Any]]:
    """Validate V2 manifest declarations against a report split, never pixels.

    This deliberately mirrors the package loader's deterministic record
    checks while omitting only child file access, output decoding, and
    re-rendering.  It is suitable for a report/package metadata binding, not
    a replacement for the byte-level package loader.
    """

    try:
        augmentation._require_exact_fields(
            manifest,
            name="V2 synthetic-stimulus package metadata manifest",
            required=augmentation.MANIFEST_FIELDS,
        )
        _require_manifest_scope(manifest)
        if manifest.get("augmentationManifestSha256") != augmentation._document_digest(
            manifest, "augmentationManifestSha256"
        ):
            raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata declared digest is inconsistent")
        if manifest.get("recipe") != recipe or manifest.get("recipeFileSha256") != recipe_file_sha256:
            raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata recipe binding is inconsistent")
        augmentation._validate_generation_provenance(manifest.get("generation"), repository_root=repository_root)

        role_map = {
            "SYNTHETIC_PROTOTYPE": "PROTOTYPE",
            "SYNTHETIC_CALIBRATION": "CALIBRATION",
            "SYNTHETIC_QUERY": "QUERY",
        }
        split_parents = [
            {
                "caseId": record["caseId"],
                "category": record["category"],
                "sourceSha256": record["sourceSha256"],
                "sourceGroupId": record["sourceGroupId"],
                "syntheticTestRole": role_map[record["role"]],
            }
            for record in report["parentSplit"]
        ]
        query_parents = [parent for parent in split_parents if parent["syntheticTestRole"] == "QUERY"]
        if manifest.get("syntheticQueryParentIdentitySha256") != augmentation._query_parent_identity(split_parents):
            raise SyntheticStressV2Error(
                "V2 synthetic-stimulus package query-parent identity does not match the response report parent split"
            )
        raw_records = manifest.get("records")
        if not isinstance(raw_records, list) or len(raw_records) != len(query_parents) * len(
            augmentation.SYNTHETIC_ANOMALY_STRESS_VARIANTS
        ):
            raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata record coverage is inconsistent")
        parent_by_case = {str(parent["caseId"]): parent for parent in query_parents}
        expected_case_ids = {
            augmentation._expected_child_identity(parent, recipe_sha256=recipe_file_sha256, variant_id=int(variant["variantId"]))[0]
            for parent in query_parents
            for variant in augmentation.SYNTHETIC_ANOMALY_STRESS_VARIANTS
        }
        expected_combinations = {
            (str(parent["caseId"]), str(variant["syntheticDefectFamily"]), str(variant["renderIntensityLevel"]))
            for parent in query_parents
            for variant in augmentation.SYNTHETIC_ANOMALY_STRESS_VARIANTS
        }
        seen_case_ids: set[str] = set()
        seen_relative_paths: set[str] = set()
        found: set[tuple[str, str, str]] = set()
        validated: list[dict[str, Any]] = []
        for index, value in enumerate(raw_records):
            record = augmentation._require_exact_fields(
                value,
                name=f"V2 synthetic-stimulus package metadata record[{index}]",
                required=augmentation.RECORD_FIELDS,
            )
            case_id = augmentation._require_string(record.get("caseId"), name="V2 package metadata caseId")
            if case_id in seen_case_ids:
                raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata has a duplicate child caseId")
            seen_case_ids.add(case_id)
            parent_case_id = augmentation._require_string(
                record.get("parentCaseId"), name="V2 package metadata parentCaseId"
            )
            parent = parent_by_case.get(parent_case_id)
            if parent is None:
                raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata child has an unknown query parent")
            if (
                augmentation._require_sha256(
                    record.get("parentSourceSha256"), name="V2 package metadata parentSourceSha256"
                )
                != parent["sourceSha256"]
                or record.get("sourceGroupId") != parent["sourceGroupId"]
                or record.get("category") != parent["category"]
                or record.get("parentPartition") != "FIT"
                or record.get("syntheticTestRole") != "QUERY"
                or record.get("syntheticLabel") != "SYNTHETIC_STIMULUS"
            ):
                raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata child parent binding is inconsistent")
            variant_id = augmentation._require_positive_int(
                record.get("variantId"), name="V2 package metadata variantId"
            )
            variant = augmentation._variant_for_id(variant_id)
            family = str(variant["syntheticDefectFamily"])
            level = str(variant["renderIntensityLevel"])
            if record.get("syntheticDefectFamily") != family or record.get("renderIntensityLevel") != level:
                raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata family/level does not match variantId")
            combination = (parent_case_id, family, level)
            if combination in found:
                raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata parent/family/level is duplicated")
            found.add(combination)
            expected_case_id, expected_relative_path = augmentation._expected_child_identity(
                parent,
                recipe_sha256=recipe_file_sha256,
                variant_id=variant_id,
            )
            if case_id != expected_case_id:
                raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata caseId is not deterministic")
            relative_path = augmentation.v1._safe_relative_path(
                record.get("relativePath"), name="V2 package metadata relativePath"
            )
            if relative_path != expected_relative_path or relative_path.as_posix() in seen_relative_paths:
                raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata relativePath is inconsistent")
            seen_relative_paths.add(relative_path.as_posix())
            parameters = augmentation._validate_parameters(
                record.get("parameters"),
                render_intensity_level=level,
                synthetic_defect_family=family,
            )
            expected_parameters = augmentation.sample_synthetic_anomaly_stress_parameters_v2(
                recipe,
                recipe_sha256=recipe_file_sha256,
                parent_case_id=parent_case_id,
                parent_source_sha256=str(parent["sourceSha256"]),
                variant_id=variant_id,
            )
            if parameters != expected_parameters:
                raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata parameters are inconsistent")
            if record.get("outputEncoding") != augmentation.SYNTHETIC_ANOMALY_STRESS_OUTPUT_ENCODING:
                raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata output encoding is inconsistent")
            augmentation._require_sha256(record.get("sourceSha256"), name="V2 package metadata sourceSha256")
            validated.append(record)
        if found != expected_combinations or seen_case_ids != expected_case_ids:
            raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata does not cover the fixed query matrix")
        if [record["caseId"] for record in raw_records] != sorted(record["caseId"] for record in raw_records):
            raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata records must be sorted by caseId")
        return validated
    except (
        augmentation.SyntheticAnomalyStressV2Error,
        augmentation.v1.SyntheticAnomalyAugmentationError,
    ) as error:
        raise SyntheticStressV2Error(str(error)) from error


def validate_response_only_report_v2_r2_package_metadata(
    document: object,
    *,
    augmentation_manifest_path: Path,
    recipe_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Bind a valid r2 report to V2 package JSON without opening image bytes.

    ``validate_response_only_report_v2_r2`` intentionally remains a pure JSON
    structural verifier.  This companion verifier is opt-in for consumers who
    also have the immutable package metadata available: it reads only the
    manifest and recipe JSON, never a rendered child image or raw parent.
    """

    report = validate_response_only_report_v2_r2(document)
    if not isinstance(augmentation_manifest_path, Path) or not isinstance(recipe_path, Path):
        raise SyntheticStressV2Error("V2 report package metadata paths must be Path instances")
    try:
        manifest, manifest_file_sha256 = augmentation.v1._read_external_json(
            augmentation_manifest_path,
            description="V2 synthetic-stimulus package metadata manifest",
            repository_root=repository_root,
        )
        recipe, recipe_file_sha256 = augmentation.load_synthetic_anomaly_stress_recipe_v2(recipe_path)
    except (augmentation.v1.SyntheticAnomalyAugmentationError, augmentation.SyntheticAnomalyStressV2Error) as error:
        raise SyntheticStressV2Error(str(error)) from error
    try:
        augmentation._require_exact_fields(
            manifest,
            name="V2 synthetic-stimulus package metadata manifest",
            required=augmentation.MANIFEST_FIELDS,
        )
        if manifest.get("schemaVersion") != augmentation.SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA:
            raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata schema is unsupported")
        if manifest.get("augmentationManifestSha256") != augmentation._document_digest(
            manifest, "augmentationManifestSha256"
        ):
            raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata declared digest is inconsistent")
        if manifest.get("recipe") != recipe or manifest.get("recipeFileSha256") != recipe_file_sha256:
            raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata recipe binding is inconsistent")
        validated_records = _validate_v2_package_metadata_against_report_parent_split(
            manifest,
            recipe=recipe,
            recipe_file_sha256=recipe_file_sha256,
            report=report,
            repository_root=repository_root,
        )
        report_to_manifest = {
            "augmentationManifestFileSha256": manifest_file_sha256,
            "augmentationManifestDeclaredSha256": manifest.get("augmentationManifestSha256"),
            "augmentationSchemaVersion": manifest.get("schemaVersion"),
            "recipeFileSha256": recipe_file_sha256,
            "parentHoldoutManifestFileSha256": manifest.get("parentHoldoutManifestFileSha256"),
            "parentHoldoutManifestDeclaredSha256": manifest.get("parentHoldoutManifestDeclaredSha256"),
            "parentSelectionContractFileSha256": manifest.get("parentSelectionContractFileSha256"),
            "parentSelectionContractDeclaredSha256": manifest.get("parentSelectionContractDeclaredSha256"),
            "successorSealFileSha256": manifest.get("successorSealFileSha256"),
            "successorSealDeclaredSha256": manifest.get("successorSealDeclaredSha256"),
            "successorPlanFileSha256": manifest.get("successorPlanFileSha256"),
            "successorPlanDeclaredSha256": manifest.get("successorPlanDeclaredSha256"),
            "successorEnvelopeFileSha256": manifest.get("successorEnvelopeFileSha256"),
            "successorEnvelopeDeclaredSha256": manifest.get("successorEnvelopeDeclaredSha256"),
            "successorFitIdentitySha256": manifest.get("successorFitIdentitySha256"),
            "parentNormalConfirmationIdentitySha256": manifest.get("parentNormalConfirmationIdentitySha256"),
        }
        for field, expected in report_to_manifest.items():
            if report[field] != expected:
                raise SyntheticStressV2Error(
                    f"synthetic-stimulus response report {field} does not match V2 package metadata"
                )
        role_map = {
            "SYNTHETIC_PROTOTYPE": "PROTOTYPE",
            "SYNTHETIC_CALIBRATION": "CALIBRATION",
            "SYNTHETIC_QUERY": "QUERY",
        }
        query_identity_parents = [
            {
                "caseId": record["caseId"],
                "category": record["category"],
                "sourceSha256": record["sourceSha256"],
                "sourceGroupId": record["sourceGroupId"],
                "syntheticTestRole": role_map[record["role"]],
            }
            for record in report["parentSplit"]
        ]
        if manifest.get("syntheticQueryParentIdentitySha256") != augmentation._query_parent_identity(
            query_identity_parents
        ):
            raise SyntheticStressV2Error(
                "V2 synthetic-stimulus package query-parent identity does not match the response report parent split"
            )
        raw_records = validated_records
        if len(raw_records) != len(report["stimulusScores"]):
            raise SyntheticStressV2Error("V2 synthetic-stimulus package/report stimulus coverage is inconsistent")
        manifest_projection: set[tuple[str, str, str, str, str, str]] = set()
        for index, item in enumerate(raw_records):
            record = augmentation._require_exact_fields(
                item,
                name=f"V2 synthetic-stimulus package metadata record[{index}]",
                required=augmentation.RECORD_FIELDS,
            )
            projection = (
                augmentation._require_string(record.get("caseId"), name="V2 package metadata caseId"),
                augmentation._require_string(record.get("parentCaseId"), name="V2 package metadata parentCaseId"),
                augmentation._require_string(record.get("category"), name="V2 package metadata category"),
                augmentation._require_sha256(record.get("sourceSha256"), name="V2 package metadata sourceSha256"),
                augmentation._require_string(
                    record.get("syntheticDefectFamily"), name="V2 package metadata syntheticDefectFamily"
                ),
                augmentation._require_string(
                    record.get("renderIntensityLevel"), name="V2 package metadata renderIntensityLevel"
                ),
            )
            if projection in manifest_projection:
                raise SyntheticStressV2Error("V2 synthetic-stimulus package metadata projection is duplicated")
            manifest_projection.add(projection)
        report_projection = {
            (
                record["caseId"],
                record["parentCaseId"],
                record["category"],
                record["sourceSha256"],
                record["syntheticDefectFamily"],
                record["renderIntensityLevel"],
            )
            for record in report["stimulusScores"]
        }
        if manifest_projection != report_projection:
            raise SyntheticStressV2Error(
                "V2 synthetic-stimulus package metadata projection does not match response score details"
            )
    except augmentation.SyntheticAnomalyStressV2Error as error:
        raise SyntheticStressV2Error(str(error)) from error
    return report


def _recheck_response_report_parent_chain(path: Path, *, repository_root: Path) -> None:
    try:
        knn._reject_links_on_existing_path(path.parent, description="synthetic-stimulus response report output")
        if knn._is_under(repository_root, path) or knn._is_under(path, repository_root):
            raise SyntheticStressV2Error("synthetic-stimulus response report output must stay outside the Git working tree")
        if not path.parent.is_dir() or knn._is_link_or_reparse_point(path.parent):
            raise SyntheticStressV2Error("synthetic-stimulus response report output parent is unsafe")
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticStressV2Error(str(error)) from error
    except SyntheticStressV2Error:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise SyntheticStressV2Error("unable to verify synthetic-stimulus response report output parent") from error


def _capture_response_report_parent_signatures(path: Path) -> tuple[tuple[Path, tuple[int, int, int]], ...]:
    """Bind every existing parent so regular-directory swaps are visible."""

    signatures: list[tuple[Path, tuple[int, int, int]]] = []
    current = path.parent
    try:
        while True:
            if knn._is_link_or_reparse_point(current):
                raise SyntheticStressV2Error("synthetic-stimulus response report output parent became a link or reparse point")
            signatures.append((current, _directory_identity(current)))
            parent = current.parent
            if parent == current:
                return tuple(signatures)
            current = parent
    except SyntheticStressV2Error:
        raise
    except OSError as error:
        raise SyntheticStressV2Error("unable to bind synthetic-stimulus response report output parent chain") from error


def _verify_response_report_parent_signatures(
    signatures: tuple[tuple[Path, tuple[int, int, int]], ...],
    *,
    path: Path,
    repository_root: Path,
) -> None:
    _recheck_response_report_parent_chain(path, repository_root=repository_root)
    for parent, expected_signature in signatures:
        if _directory_identity(parent) != expected_signature:
            raise SyntheticStressV2Error("synthetic-stimulus response report parent chain changed while it was written")


def write_response_only_report(path: Path, document: dict[str, Any], *, repository_root: Path) -> Path:
    """Write a new immutable r2 response report with no-follow race checks."""

    try:
        prepared = knn._require_external_output(path, repository_root=repository_root)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticStressV2Error(str(error)) from error
    validate_response_only_report_v2_r2(document)
    data = _serialize_response_report(document)
    _recheck_response_report_parent_chain(prepared, repository_root=repository_root)
    parent_signatures = _capture_response_report_parent_signatures(prepared)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(prepared, flags, 0o600)
    except FileExistsError as error:
        raise SyntheticStressV2Error("synthetic-stimulus response report already exists") from error
    except OSError as error:
        raise SyntheticStressV2Error("unable to write synthetic-stimulus response report") from error
    try:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("unable to write synthetic-stimulus response report")
            offset += written
        os.fsync(fd)
        fd_signature = _fd_signature(fd)
        if fd_signature[3] != len(data):
            raise SyntheticStressV2Error("synthetic-stimulus response report size is inconsistent after the full write")
        _verify_response_report_parent_signatures(
            parent_signatures,
            path=prepared,
            repository_root=repository_root,
        )
        if knn._is_link_or_reparse_point(prepared) or _stat_signature(prepared) != fd_signature:
            raise SyntheticStressV2Error("synthetic-stimulus response report changed while it was written")
    except SyntheticStressV2Error:
        raise
    except OSError as error:
        raise SyntheticStressV2Error("unable to verify synthetic-stimulus response report output") from error
    finally:
        os.close(fd)
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
    expected_manifest_file_sha256: str | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Load a V2 package only after raw-calibration thresholds are fixed.

    The generator performs byte-level re-render validation.  This evaluator
    additionally binds every returned child to the independently recreated
    fixed query-parent split and requires all family/level combinations.
    """

    if expected_manifest_file_sha256 is not None:
        # This is a metadata-only re-read performed before the generator
        # loader can open any rendered child byte.  V3 uses it to bind its
        # irreversible registry receipt to the exact preflight manifest.
        try:
            _manifest_snapshot, manifest_snapshot_sha256 = augmentation.v1._read_external_json(
                augmentation_manifest_path,
                description="V2 synthetic-stimulus package preflight manifest",
                repository_root=repository_root,
            )
        except augmentation.v1.SyntheticAnomalyAugmentationError as error:
            raise SyntheticStressV2Error(str(error)) from error
        if manifest_snapshot_sha256 != _require_sha256(
            expected_manifest_file_sha256,
            name="V2 expected preflight manifest digest",
        ):
            raise SyntheticStressV2Error(
                "V2 synthetic-stimulus manifest changed after preflight; refusing before child-byte access"
            )
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

    # A hardened r2 report requires a complete sealed identity.  Validate it
    # before requesting even FIT parents so the public API cannot bypass the
    # sealed CLI path and consume image inputs first.
    provenance_started = time.process_time()
    extractor_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    _validate_feature_extractor(extractor_identity)
    extractor_identity_sha256 = successor.canonical_json_sha256(extractor_identity)
    timings["provenanceSeconds"] += time.process_time() - provenance_started
    embedder = embedder_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    provenance_started = time.process_time()
    loaded_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    _validate_feature_extractor(loaded_identity)
    timings["provenanceSeconds"] += time.process_time() - provenance_started
    if loaded_identity != extractor_identity:
        raise SyntheticStressV2Error("feature extractor changed while DINO loaded")

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
    _validate_feature_extractor(completed_identity)
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
        "realPrecisionRecall": SYNTHETIC_STRESS_V2_REAL_PRECISION_RECALL,
        "evidenceClass": SYNTHETIC_STRESS_V2_EVIDENCE_CLASS,
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
    validate_response_only_report_v2_r2(document)
    # ``prepared_output`` was preflighted before any source image was opened;
    # exclusive write inside ``write_response_only_report`` preserves new-only output.
    write_response_only_report(prepared_output, document, repository_root=repository_root)
    return document
