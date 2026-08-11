"""Paired generic-capture nuisance-control audit for offline MVTec research.

This V3 audit is intentionally a new, non-promotional experiment.  It pairs
the already materialised V2 synthetic-stimulus children with three R3 generic
capture controls for the *same* twelve raw FIT query parents.  It never opens
blind, anomaly, mask, tuning, selection, confirmation, or reserve image
bytes.  It also never consumes a V1 or V2 response report.

The result is a response-only engineering trace.  In particular, a difference
between synthetic children and camera controls is not a real-defect result,
and it is not a basis for model, threshold, or production decisions.
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
from phone_dino import mvtec_successor_fit_augmentation_v2 as camera
from phone_dino import mvtec_synthetic_anomaly_stress_v2 as stimulus
from phone_dino import mvtec_synthetic_stress_v2 as stress


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_NUISANCE_CONTROL_V3_REPORT_SCHEMA = (
    "phone-dino.mvtec-ad-synthetic-nuisance-control-response-report/3.0"
)
SYNTHETIC_NUISANCE_CONTROL_V3_RECEIPT_SCHEMA = (
    "phone-dino.mvtec-ad-synthetic-nuisance-control-receipt/3.0"
)
SYNTHETIC_NUISANCE_CONTROL_V3_PURPOSE = (
    "OFFLINE_MVTEC_PAIRED_GENERIC_CAPTURE_NUISANCE_CONTROL_AUDIT"
)
SYNTHETIC_NUISANCE_CONTROL_V3_PHASE = "POST_V2_PAIRED_NUISANCE_CONTROL_RESPONSE_AUDIT"
SYNTHETIC_NUISANCE_CONTROL_V3_METRIC_SCOPE = "SYNTHETIC_STIMULUS_AND_GENERIC_CAPTURE_RESPONSE_ONLY"
SYNTHETIC_NUISANCE_CONTROL_V3_EVIDENCE_CLASS = "SYNTHETIC_ENGINEERING_ONLY"
SYNTHETIC_NUISANCE_CONTROL_V3_RESULT_LABEL = "SYNTHETIC_ONLY_NOT_REAL_ANOMALY_PERFORMANCE"
SYNTHETIC_NUISANCE_CONTROL_V3_REAL_PERFORMANCE = "NOT_ESTIMATED"
SYNTHETIC_NUISANCE_CONTROL_V3_INPUT_POLICY = "SUCCESSOR_V2_FIT_RAW_QUERY_PARENTS_ONLY"
SYNTHETIC_NUISANCE_CONTROL_V3_BLIND_POLICY = "NO_BLIND_OR_TRUE_ANOMALY_DATA"
SYNTHETIC_NUISANCE_CONTROL_V3_FORBIDDEN_USES = [
    "MODEL_SELECTION",
    "ALGORITHM_SELECTION",
    "HYPERPARAMETER_SELECTION",
    "THRESHOLD_SELECTION",
    "PACKAGE_COMPARISON_OR_PROMOTION",
    "PRODUCTION_VALIDATION",
    "PHYSICAL_QUALIFICATION",
    "REAL_ANOMALY_PERFORMANCE_CLAIM",
]
SYNTHETIC_NUISANCE_CONTROL_V3_CATEGORIES = stress.SYNTHETIC_STRESS_V2_CATEGORIES
SYNTHETIC_NUISANCE_CONTROL_V3_FAMILIES = stress.SYNTHETIC_STRESS_V2_FAMILIES
SYNTHETIC_NUISANCE_CONTROL_V3_LEVELS = stress.SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS
SYNTHETIC_NUISANCE_CONTROL_V3_CONTROL_COMPONENTS = tuple(
    str(item["component"]) for item in camera.SUCCESSOR_FIT_VARIANTS
)
SYNTHETIC_NUISANCE_CONTROL_V3_PARENT_COUNTS = dict(stress.SYNTHETIC_STRESS_V2_PARENT_COUNTS)
SYNTHETIC_NUISANCE_CONTROL_V3_DECISION_RULE = stress.SYNTHETIC_STRESS_V2_DECISION_RULE

REPORT_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "syntheticOnly",
    "comparisonOrPromotionAllowed",
    "purpose",
    "phase",
    "metricScope",
    "evidenceClass",
    "realAnomalyPerformance",
    "realPrecisionRecall",
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
    "stimulusPackageManifestFileSha256",
    "stimulusPackageManifestDeclaredSha256",
    "stimulusPackageSchemaVersion",
    "stimulusPackageRecipeFileSha256",
    "captureControlPackageManifestFileSha256",
    "captureControlPackageManifestDeclaredSha256",
    "captureControlPackageSchemaVersion",
    "captureControlPackageRecipeFileSha256",
    "parentSplit",
    "parentSplitIdentitySha256",
    "featureExtractor",
    "featureExtractorIdentitySha256",
    "registryReceiptKeySha256",
    "registryReceiptFileSha256",
    "registryReceiptDeclaredSha256",
    "thresholds",
    "calibrationScores",
    "rawQueryScores",
    "genericCaptureControlScores",
    "syntheticStimulusScores",
    "parentLevelContrasts",
    "categories",
    "aggregate",
    "execution",
    "syntheticNuisanceControlReportSha256",
}
RECEIPT_FIELDS = {
    "schemaVersion",
    "receiptKeySha256",
    "receiptIdentity",
    "syntheticNuisanceControlReceiptSha256",
}
RECEIPT_IDENTITY_FIELDS = {
    "schemaVersion",
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
    "parentSplitIdentitySha256",
    "featureExtractorIdentitySha256",
    "stimulusPackageManifestFileSha256",
    "stimulusPackageManifestDeclaredSha256",
    "stimulusPackageRecipeFileSha256",
    "captureControlPackageManifestFileSha256",
    "captureControlPackageManifestDeclaredSha256",
    "captureControlPackageRecipeFileSha256",
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
    "category",
    "sourceSha256",
    "score",
    "maxPatchDistance",
    "meanNearestPatchDistance",
    "aboveRawCalibrationThreshold",
}
CONTROL_SCORE_FIELDS = {
    "caseId",
    "parentCaseId",
    "category",
    "sourceSha256",
    "captureComponent",
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
PARENT_CONTRAST_FIELDS = {
    "parentCaseId",
    "category",
    "rawParentScore",
    "rawParentAboveThreshold",
    "genericCaptureControlCount",
    "genericCaptureControlAboveThresholdCount",
    "genericCaptureMeanChildMinusParentScore",
    "syntheticStimulusCount",
    "syntheticStimulusAboveThresholdCount",
    "syntheticStimulusMeanChildMinusParentScore",
    "syntheticStimulusMinusGenericCaptureMeanDelta",
}


class SyntheticNuisanceControlV3Error(ValueError):
    """Raised when the V3 nuisance-control boundary is not closed."""


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without retaining an entire file in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SyntheticNuisanceControlV3Error(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise SyntheticNuisanceControlV3Error(f"{name} must be a SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise SyntheticNuisanceControlV3Error(f"{name} must be a SHA-256 digest") from error
    return digest


def _require_finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SyntheticNuisanceControlV3Error(f"{name} must be finite")
    return float(value)


def _require_exact_fields(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SyntheticNuisanceControlV3Error(f"{name} must be an object")
    missing = fields.difference(value)
    unknown = set(value).difference(fields)
    if missing or unknown:
        detail: list[str] = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unsupported {', '.join(sorted(unknown))}")
        raise SyntheticNuisanceControlV3Error(f"{name} has {'; '.join(detail)} fields")
    return value


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return successor.canonical_json_sha256(unsigned)


def _parse_json_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SyntheticNuisanceControlV3Error(f"{description} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise SyntheticNuisanceControlV3Error(f"{description} contains a non-finite JSON value: {value}")

    try:
        parsed = json.loads(
            raw.decode("utf-8-sig"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SyntheticNuisanceControlV3Error(f"unable to read {description}") from error
    if not isinstance(parsed, dict):
        raise SyntheticNuisanceControlV3Error(f"{description} must be a JSON object")
    return parsed


def _read_external_metadata(path: Path, *, description: str, repository_root: Path) -> tuple[dict[str, Any], str]:
    """Read package metadata only; this deliberately never touches a child image."""

    if not isinstance(path, Path):
        raise SyntheticNuisanceControlV3Error(f"{description} path must be a Path")
    try:
        knn._reject_links_on_existing_path(path, description=description)
        external_root = path.parent
        if knn._is_under(repository_root, path) or knn._is_under(external_root, repository_root):
            raise SyntheticNuisanceControlV3Error(f"{description} must stay outside the Git working tree")
        if not path.is_file() or knn._is_link_or_reparse_point(path):
            raise SyntheticNuisanceControlV3Error(f"{description} must be a regular non-link file")
        raw = path.read_bytes()
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    except OSError as error:
        raise SyntheticNuisanceControlV3Error(f"unable to read {description}") from error
    return _parse_json_bytes(raw, description=description), f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _assert_chain_binding(
    manifest: dict[str, Any], *, envelope: dict[str, Any], envelope_file_sha256: str, label: str
) -> None:
    parent_evidence = envelope.get("parentEvidence")
    partition_identities = envelope.get("successorPartitionIdentities")
    if not isinstance(parent_evidence, dict) or not isinstance(partition_identities, dict):
        raise SyntheticNuisanceControlV3Error("successor envelope is missing closed parent evidence")
    expected = {
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
    }
    for name, expected_value in expected.items():
        if _require_sha256(manifest.get(name), name=f"{label} {name}") != _require_sha256(
            expected_value, name=f"successor envelope {name}"
        ):
            raise SyntheticNuisanceControlV3Error(f"{label} is not bound to the closed successor contract")


def _validate_v2_stimulus_metadata_records(
    document: dict[str, Any],
    *,
    recipe: dict[str, Any],
    recipe_sha256: str,
    fit_parents: list[dict[str, Any]],
    package_root: Path,
    repository_root: Path,
) -> None:
    """Validate every V2 child declaration without opening any child bytes."""

    try:
        split_parents = stimulus._closed_stress_split(fit_parents)
        query_parents = [parent for parent in split_parents if parent["syntheticTestRole"] == "QUERY"]
        if _require_sha256(
            document.get("syntheticQueryParentIdentitySha256"),
            name="V2-r2 stimulus package syntheticQueryParentIdentitySha256",
        ) != stimulus._query_parent_identity(split_parents):
            raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package query-parent identity is unsafe")
        stimulus._validate_generation_provenance(document.get("generation"), repository_root=repository_root)
        camera._require_external_package_root(package_root, repository_root=repository_root)
        raw_records = document.get("records")
        if not isinstance(raw_records, list) or len(raw_records) != len(query_parents) * len(stimulus.SYNTHETIC_ANOMALY_STRESS_VARIANTS):
            raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package record coverage is inconsistent")
        parent_by_case = {str(parent["caseId"]): parent for parent in query_parents}
        expected_case_ids = {
            stimulus._expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=int(variant["variantId"]))[0]
            for parent in query_parents
            for variant in stimulus.SYNTHETIC_ANOMALY_STRESS_VARIANTS
        }
        seen_case_ids: set[str] = set()
        seen_relative_paths: set[str] = set()
        found: set[tuple[str, str, str]] = set()
        for raw_record in raw_records:
            record = stimulus._require_exact_fields(
                raw_record,
                name="V2-r2 stimulus metadata record",
                required=stimulus.RECORD_FIELDS,
            )
            case_id = stimulus._require_string(record.get("caseId"), name="V2-r2 stimulus metadata caseId")
            if case_id in seen_case_ids:
                raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package has a duplicate child caseId")
            seen_case_ids.add(case_id)
            parent_case_id = stimulus._require_string(
                record.get("parentCaseId"), name="V2-r2 stimulus metadata parentCaseId"
            )
            parent = parent_by_case.get(parent_case_id)
            if parent is None:
                raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package child is not bound to a fixed query parent")
            if (
                stimulus._require_sha256(
                    record.get("parentSourceSha256"), name="V2-r2 stimulus metadata parentSourceSha256"
                )
                != parent["sourceSha256"]
                or record.get("sourceGroupId") != parent["sourceGroupId"]
                or record.get("category") != parent["category"]
                or record.get("parentPartition") != "FIT"
                or record.get("syntheticTestRole") != "QUERY"
                or record.get("syntheticLabel") != "SYNTHETIC_STIMULUS"
            ):
                raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package child parent binding is unsafe")
            variant_id = stimulus._require_positive_int(
                record.get("variantId"), name="V2-r2 stimulus metadata variantId"
            )
            variant = stimulus._variant_for_id(variant_id)
            family = str(variant["syntheticDefectFamily"])
            level = str(variant["renderIntensityLevel"])
            if record.get("syntheticDefectFamily") != family or record.get("renderIntensityLevel") != level:
                raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package child family/level does not match variant")
            combination = (parent_case_id, family, level)
            if combination in found:
                raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package parent/family/level is duplicated")
            found.add(combination)
            expected_case_id, expected_relative_path = stimulus._expected_child_identity(
                parent,
                recipe_sha256=recipe_sha256,
                variant_id=variant_id,
            )
            if case_id != expected_case_id:
                raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package child caseId is not deterministic")
            relative_path = stimulus.v1._safe_relative_path(
                record.get("relativePath"), name="V2-r2 stimulus metadata relativePath"
            )
            if relative_path != expected_relative_path or relative_path.as_posix() in seen_relative_paths:
                raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package child relativePath is unsafe")
            seen_relative_paths.add(relative_path.as_posix())
            stimulus.v1._safe_file_under(
                package_root,
                relative_path,
                description="V2-r2 stimulus metadata child path",
                repository_root=repository_root,
            )
            parameters = stimulus._validate_parameters(
                record.get("parameters"),
                render_intensity_level=level,
                synthetic_defect_family=family,
            )
            expected_parameters = stimulus.sample_synthetic_anomaly_stress_parameters_v2(
                recipe,
                recipe_sha256=recipe_sha256,
                parent_case_id=parent_case_id,
                parent_source_sha256=str(parent["sourceSha256"]),
                variant_id=variant_id,
            )
            if parameters != expected_parameters:
                raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package child parameters are unsafe")
            if record.get("outputEncoding") != stimulus.SYNTHETIC_ANOMALY_STRESS_OUTPUT_ENCODING:
                raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package child output encoding is unsafe")
            stimulus._require_sha256(record.get("sourceSha256"), name="V2-r2 stimulus metadata sourceSha256")
        expected_combinations = {
            (str(parent["caseId"]), str(variant["syntheticDefectFamily"]), str(variant["renderIntensityLevel"]))
            for parent in query_parents
            for variant in stimulus.SYNTHETIC_ANOMALY_STRESS_VARIANTS
        }
        if found != expected_combinations or seen_case_ids != expected_case_ids:
            raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package metadata does not cover the fixed query matrix")
        if [record.get("caseId") for record in raw_records] != sorted(record.get("caseId") for record in raw_records):
            raise SyntheticNuisanceControlV3Error("V2-r2 stimulus package metadata records must be sorted by caseId")
    except (
        stimulus.SyntheticAnomalyStressV2Error,
        stimulus.v1.SyntheticAnomalyAugmentationError,
        camera.SuccessorFitAugmentationV2Error,
    ) as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error


def _validate_r3_capture_metadata_records(
    document: dict[str, Any],
    *,
    recipe: dict[str, Any],
    recipe_sha256: str,
    fit_parents: list[dict[str, Any]],
    package_root: Path,
    repository_root: Path,
) -> None:
    """Validate every R3 declaration and coverage before the receipt is spent."""

    try:
        parents = camera._validate_fit_parents(fit_parents)
        raw_records = document.get("records")
        expected_count = len(parents) * camera.SUCCESSOR_FIT_VARIANTS_PER_PARENT
        if not isinstance(raw_records, list) or len(raw_records) != expected_count:
            raise SyntheticNuisanceControlV3Error("R3 generic-capture package record coverage is inconsistent")
        camera._require_external_package_root(package_root, repository_root=repository_root)
        parent_by_case = {str(parent["caseId"]): parent for parent in parents}
        expected_case_ids = {
            camera._expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=int(variant["variantId"]))[0]
            for parent in parents
            for variant in camera.SUCCESSOR_FIT_VARIANTS
        }
        seen_case_ids: set[str] = set()
        seen_relative_paths: set[str] = set()
        found: set[tuple[str, str]] = set()
        for raw_record in raw_records:
            record = camera._require_exact_fields(
                raw_record,
                name="R3 generic-capture metadata record",
                required=camera.RECORD_FIELDS,
            )
            case_id = camera._require_string(record.get("caseId"), name="R3 generic-capture metadata caseId")
            if case_id in seen_case_ids:
                raise SyntheticNuisanceControlV3Error("R3 generic-capture package has a duplicate child caseId")
            seen_case_ids.add(case_id)
            parent_case_id = camera._require_string(
                record.get("parentCaseId"), name="R3 generic-capture metadata parentCaseId"
            )
            parent = parent_by_case.get(parent_case_id)
            if parent is None:
                raise SyntheticNuisanceControlV3Error("R3 generic-capture package child has an unknown parent")
            if (
                camera._require_sha256(
                    record.get("parentSourceSha256"), name="R3 generic-capture metadata parentSourceSha256"
                )
                != parent["sourceSha256"]
                or record.get("sourceGroupId") != parent["sourceGroupId"]
                or record.get("category") != parent["category"]
                or record.get("parentPartition") != "FIT"
                or record.get("kind") != "NOMINAL"
                or record.get("defect") != "good"
            ):
                raise SyntheticNuisanceControlV3Error("R3 generic-capture package child parent binding is unsafe")
            variant_id = camera._require_positive_int(
                record.get("variantId"), name="R3 generic-capture metadata variantId"
            )
            component = camera._component_for_variant(variant_id)
            if record.get("component") != component:
                raise SyntheticNuisanceControlV3Error("R3 generic-capture package child component does not match variant")
            combination = (parent_case_id, component)
            if combination in found:
                raise SyntheticNuisanceControlV3Error("R3 generic-capture package parent/component is duplicated")
            found.add(combination)
            expected_case_id, expected_relative_path = camera._expected_child_identity(
                parent,
                recipe_sha256=recipe_sha256,
                variant_id=variant_id,
            )
            if case_id != expected_case_id:
                raise SyntheticNuisanceControlV3Error("R3 generic-capture package child caseId is not deterministic")
            relative_path = camera._safe_relative_path(
                record.get("relativePath"), name="R3 generic-capture metadata relativePath"
            )
            if relative_path != expected_relative_path or relative_path.as_posix() in seen_relative_paths:
                raise SyntheticNuisanceControlV3Error("R3 generic-capture package child relativePath is unsafe")
            seen_relative_paths.add(relative_path.as_posix())
            camera._safe_file_under(
                package_root,
                relative_path,
                description="R3 generic-capture metadata child path",
                repository_root=repository_root,
            )
            parameters = camera._validate_parameters(record.get("parameters"), variant_id=variant_id)
            expected_parameters = camera.sample_successor_fit_parameters_v2(
                recipe,
                recipe_sha256=recipe_sha256,
                parent_case_id=parent_case_id,
                parent_source_sha256=str(parent["sourceSha256"]),
                variant_id=variant_id,
            )
            if parameters != expected_parameters:
                raise SyntheticNuisanceControlV3Error("R3 generic-capture package child parameters are unsafe")
            camera._validate_output_encoding(record.get("outputEncoding"))
            camera._require_sha256(record.get("sourceSha256"), name="R3 generic-capture metadata sourceSha256")
        expected_combinations = {
            (str(parent["caseId"]), str(variant["component"]))
            for parent in parents
            for variant in camera.SUCCESSOR_FIT_VARIANTS
        }
        if found != expected_combinations or seen_case_ids != expected_case_ids:
            raise SyntheticNuisanceControlV3Error("R3 generic-capture metadata does not cover every FIT parent/component")
        if [record.get("caseId") for record in raw_records] != sorted(record.get("caseId") for record in raw_records):
            raise SyntheticNuisanceControlV3Error("R3 generic-capture metadata records must be sorted by caseId")
    except camera.SuccessorFitAugmentationV2Error as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error


def _preflight_stimulus_package(
    augmentation_manifest_path: Path,
    *,
    recipe_path: Path,
    envelope: dict[str, Any],
    envelope_file_sha256: str,
    repository_root: Path,
    fit_parents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate immutable V2 package metadata without loading a stimulus byte."""

    document, file_sha256 = _read_external_metadata(
        augmentation_manifest_path,
        description="V2-r2 synthetic-stimulus augmentation manifest",
        repository_root=repository_root,
    )
    _require_exact_fields(document, name="V2-r2 synthetic-stimulus augmentation manifest", fields=stimulus.MANIFEST_FIELDS)
    if document.get("schemaVersion") != stimulus.SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA:
        raise SyntheticNuisanceControlV3Error("V2-r2 synthetic-stimulus package schema is unsupported")
    expected_scope = {
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "postV1Exploratory": True,
        "comparisonOrPromotionAllowed": False,
        "parentPartition": "FIT",
        "inputPolicy": stimulus.SYNTHETIC_ANOMALY_STRESS_INPUT_POLICY,
        "blindPolicy": stimulus.SYNTHETIC_ANOMALY_STRESS_BLIND_POLICY,
        "resultLabel": stimulus.SYNTHETIC_ANOMALY_STRESS_RESULT_LABEL,
        "parentSplitAlgorithm": stimulus.SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_ALGORITHM,
        "parentSplitCountsPerCategory": stimulus.SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_COUNTS_PER_CATEGORY,
        "variantsPerParent": stimulus.SYNTHETIC_ANOMALY_STRESS_VARIANTS_PER_PARENT,
    }
    if any(document.get(name) != value for name, value in expected_scope.items()):
        raise SyntheticNuisanceControlV3Error("V2-r2 synthetic-stimulus package scope is unsafe")
    if document.get("augmentationManifestSha256") != _document_digest(document, "augmentationManifestSha256"):
        raise SyntheticNuisanceControlV3Error("V2-r2 synthetic-stimulus package declared digest does not match")
    _assert_chain_binding(document, envelope=envelope, envelope_file_sha256=envelope_file_sha256, label="V2-r2 stimulus package")
    try:
        recipe, recipe_sha256 = stimulus.load_synthetic_anomaly_stress_recipe_v2(recipe_path)
    except ValueError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    if document.get("recipe") != recipe or _require_sha256(
        document.get("recipeFileSha256"), name="V2-r2 stimulus package recipeFileSha256"
    ) != recipe_sha256:
        raise SyntheticNuisanceControlV3Error("V2-r2 synthetic-stimulus package recipe binding is unsafe")
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != 108:
        raise SyntheticNuisanceControlV3Error("V2-r2 synthetic-stimulus package does not contain 108 records")
    if fit_parents is not None:
        _validate_v2_stimulus_metadata_records(
            document,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
            fit_parents=fit_parents,
            package_root=augmentation_manifest_path.parent,
            repository_root=repository_root,
        )
    return {
        "document": document,
        "fileSha256": file_sha256,
        "recipeSha256": recipe_sha256,
    }


def _preflight_camera_package(
    augmentation_manifest_path: Path,
    *,
    recipe_path: Path,
    envelope: dict[str, Any],
    envelope_file_sha256: str,
    fit_parents: list[dict[str, Any]],
    repository_root: Path,
) -> dict[str, Any]:
    """Validate immutable R3 metadata without loading a camera-control byte."""

    document, file_sha256 = _read_external_metadata(
        augmentation_manifest_path,
        description="R3 generic-capture augmentation manifest",
        repository_root=repository_root,
    )
    try:
        recipe, recipe_sha256 = camera.load_successor_fit_camera_recipe_v2(recipe_path)
        raw_records = camera._validate_manifest_document(
            document,
            manifest_file_sha256=file_sha256,
            envelope=envelope,
            envelope_file_sha256=envelope_file_sha256,
            parents=fit_parents,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
            repository_root=repository_root,
        )
    except ValueError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    _assert_chain_binding(document, envelope=envelope, envelope_file_sha256=envelope_file_sha256, label="R3 capture package")
    if document.get("schemaVersion") != camera.SUCCESSOR_FIT_AUGMENTATION_V2_SCHEMA:
        raise SyntheticNuisanceControlV3Error("R3 generic-capture package schema is unsupported")
    if len(raw_records) != len(fit_parents) * camera.SUCCESSOR_FIT_VARIANTS_PER_PARENT:
        raise SyntheticNuisanceControlV3Error("R3 generic-capture package coverage is inconsistent")
    _validate_r3_capture_metadata_records(
        document,
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        fit_parents=fit_parents,
        package_root=augmentation_manifest_path.parent,
        repository_root=repository_root,
    )
    return {
        "document": document,
        "fileSha256": file_sha256,
        "recipeSha256": recipe_sha256,
    }


def _prepare_registry_root(registry_root: Path, *, repository_root: Path) -> Path:
    """Prepare an external receipt directory without making a receipt yet."""

    if not isinstance(registry_root, Path):
        raise SyntheticNuisanceControlV3Error("registry_root must be a Path")
    try:
        knn._reject_links_on_existing_path(registry_root, description="V3 registry root")
        if knn._is_under(repository_root, registry_root) or knn._is_under(registry_root, repository_root):
            raise SyntheticNuisanceControlV3Error("registry_root must stay outside the Git working tree")
        if registry_root.exists():
            if not registry_root.is_dir() or knn._is_link_or_reparse_point(registry_root):
                raise SyntheticNuisanceControlV3Error("registry_root must be a non-link directory")
        else:
            registry_root.mkdir(parents=True, exist_ok=False)
        knn._reject_links_on_existing_path(registry_root, description="V3 registry root")
        if not registry_root.is_dir() or knn._is_link_or_reparse_point(registry_root):
            raise SyntheticNuisanceControlV3Error("registry_root became unsafe while it was prepared")
        return registry_root.resolve(strict=True)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    except (OSError, RuntimeError) as error:
        raise SyntheticNuisanceControlV3Error("unable to prepare external registry_root") from error


def _recheck_registry_root_chain(root: Path, *, repository_root: Path) -> None:
    """Require the registry root and every existing ancestor to stay stable-safe."""

    try:
        knn._reject_links_on_existing_path(root, description="V3 registry root")
        if knn._is_under(repository_root, root) or knn._is_under(root, repository_root):
            raise SyntheticNuisanceControlV3Error("registry_root must stay outside the Git working tree")
        if not root.is_dir() or knn._is_link_or_reparse_point(root):
            raise SyntheticNuisanceControlV3Error("registry_root became unsafe")
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    except SyntheticNuisanceControlV3Error:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise SyntheticNuisanceControlV3Error("unable to verify registry_root") from error


def _capture_registry_root_signatures(
    root: Path, *, repository_root: Path
) -> tuple[tuple[Path, tuple[int, int, int]], ...]:
    """Bind the root plus ancestor directory identities across receipt creation."""

    _recheck_registry_root_chain(root, repository_root=repository_root)
    signatures: list[tuple[Path, tuple[int, int, int]]] = []
    current = root
    try:
        while True:
            if knn._is_link_or_reparse_point(current):
                raise SyntheticNuisanceControlV3Error("registry_root ancestor became a link or reparse point")
            signatures.append((current, _directory_identity(current)))
            parent = current.parent
            if parent == current:
                return tuple(signatures)
            current = parent
    except SyntheticNuisanceControlV3Error:
        raise
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to bind registry_root ancestor identities") from error


def _verify_registry_root_signatures(
    signatures: tuple[tuple[Path, tuple[int, int, int]], ...],
    *,
    root: Path,
    repository_root: Path,
) -> None:
    _recheck_registry_root_chain(root, repository_root=repository_root)
    try:
        for ancestor, expected_signature in signatures:
            if _directory_identity(ancestor) != expected_signature:
                raise SyntheticNuisanceControlV3Error(
                    "registry_root ancestor chain changed while the one-time receipt was created"
                )
    except SyntheticNuisanceControlV3Error:
        raise
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to recheck registry_root ancestor identities") from error


def _verify_registry_receipt_binding(
    root: Path,
    signatures: tuple[tuple[Path, tuple[int, int, int]], ...],
    *,
    receipt: dict[str, Any],
    receipt_file_sha256: str,
    repository_root: Path,
) -> None:
    """Prove the irreversible receipt remains in its bound registry before children open."""

    _verify_registry_root_signatures(signatures, root=root, repository_root=repository_root)
    key = _require_sha256(receipt.get("receiptKeySha256"), name="V3 registry receipt key")
    expected_file_sha256 = _require_sha256(receipt_file_sha256, name="V3 registry receipt file digest")
    target = root / f"{key[7:]}.json"
    try:
        if not target.is_file() or knn._is_link_or_reparse_point(target):
            raise SyntheticNuisanceControlV3Error("V3 registry receipt became unsafe before child package loading")
        if sha256_file(target) != expected_file_sha256:
            raise SyntheticNuisanceControlV3Error("V3 registry receipt changed before child package loading")
        persisted = _parse_json_bytes(target.read_bytes(), description="V3 registry receipt")
        if persisted != receipt:
            raise SyntheticNuisanceControlV3Error("V3 registry receipt content changed before child package loading")
    except SyntheticNuisanceControlV3Error:
        raise
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to verify V3 registry receipt before child package loading") from error


def _receipt_identity(
    *,
    envelope: dict[str, Any],
    envelope_file_sha256: str,
    parent_split_records: list[dict[str, Any]],
    feature_extractor_identity_sha256: str,
    stimulus_metadata: dict[str, Any],
    camera_metadata: dict[str, Any],
) -> dict[str, Any]:
    parent_evidence = envelope.get("parentEvidence")
    partition_identities = envelope.get("successorPartitionIdentities")
    if not isinstance(parent_evidence, dict) or not isinstance(partition_identities, dict):
        raise SyntheticNuisanceControlV3Error("successor envelope is missing receipt evidence")
    stimulus_document = stimulus_metadata["document"]
    camera_document = camera_metadata["document"]
    identity = {
        "schemaVersion": SYNTHETIC_NUISANCE_CONTROL_V3_RECEIPT_SCHEMA,
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
        "parentSplitIdentitySha256": successor.canonical_json_sha256(parent_split_records),
        "featureExtractorIdentitySha256": feature_extractor_identity_sha256,
        "stimulusPackageManifestFileSha256": stimulus_metadata["fileSha256"],
        "stimulusPackageManifestDeclaredSha256": stimulus_document.get("augmentationManifestSha256"),
        "stimulusPackageRecipeFileSha256": stimulus_metadata["recipeSha256"],
        "captureControlPackageManifestFileSha256": camera_metadata["fileSha256"],
        "captureControlPackageManifestDeclaredSha256": camera_document.get("augmentationManifestSha256"),
        "captureControlPackageRecipeFileSha256": camera_metadata["recipeSha256"],
    }
    for name, value in identity.items():
        if name != "schemaVersion":
            identity[name] = _require_sha256(value, name=f"receipt identity {name}")
    return identity


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat(follow_symlinks=False)
    return status.st_dev, status.st_ino, status.st_mode, status.st_size


def _directory_identity(path: Path) -> tuple[int, int, int]:
    status = path.stat(follow_symlinks=False)
    return status.st_dev, status.st_ino, status.st_mode


def _fd_signature(fd: int) -> tuple[int, int, int, int]:
    status = os.fstat(fd)
    return status.st_dev, status.st_ino, status.st_mode, status.st_size


def _recheck_v3_report_parent_chain(path: Path, *, repository_root: Path) -> None:
    """Reject parent-path reparse points before opening a V3 report slot."""

    try:
        knn._reject_links_on_existing_path(path.parent, description="V3 nuisance-control report output")
        if knn._is_under(repository_root, path) or knn._is_under(path, repository_root):
            raise SyntheticNuisanceControlV3Error("V3 nuisance-control report output must stay outside the Git working tree")
        if not path.parent.is_dir() or knn._is_link_or_reparse_point(path.parent):
            raise SyntheticNuisanceControlV3Error("V3 nuisance-control report output parent is unsafe")
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    except SyntheticNuisanceControlV3Error:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise SyntheticNuisanceControlV3Error("unable to verify V3 nuisance-control report output parent") from error


def _capture_v3_report_parent_signatures(path: Path) -> tuple[tuple[Path, tuple[int, int, int]], ...]:
    """Bind every existing report ancestor to make directory swaps visible."""

    signatures: list[tuple[Path, tuple[int, int, int]]] = []
    current = path.parent
    try:
        while True:
            if knn._is_link_or_reparse_point(current):
                raise SyntheticNuisanceControlV3Error(
                    "V3 nuisance-control report output parent became a link or reparse point"
                )
            signatures.append((current, _directory_identity(current)))
            parent = current.parent
            if parent == current:
                return tuple(signatures)
            current = parent
    except SyntheticNuisanceControlV3Error:
        raise
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to bind V3 nuisance-control report output parent chain") from error


def _verify_v3_report_parent_signatures(
    signatures: tuple[tuple[Path, tuple[int, int, int]], ...], *, path: Path, repository_root: Path
) -> None:
    _recheck_v3_report_parent_chain(path, repository_root=repository_root)
    try:
        for parent, expected_signature in signatures:
            if _directory_identity(parent) != expected_signature:
                raise SyntheticNuisanceControlV3Error(
                    "V3 nuisance-control report parent chain changed while it was written"
                )
    except SyntheticNuisanceControlV3Error:
        raise
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to recheck V3 nuisance-control report output parent chain") from error


def create_one_time_registry_receipt(
    registry_root: Path,
    *,
    identity: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str]:
    """Atomically consume the immutable audit identity before any child loader runs.

    A repeated identity is intentionally rejected.  The receipt content and
    filename are deterministic; exclusive creation is the one-time gate.
    """

    _require_exact_fields(identity, name="V3 registry receipt identity", fields=RECEIPT_IDENTITY_FIELDS)
    if identity.get("schemaVersion") != SYNTHETIC_NUISANCE_CONTROL_V3_RECEIPT_SCHEMA:
        raise SyntheticNuisanceControlV3Error("V3 registry receipt identity schema is unsupported")
    for name in RECEIPT_IDENTITY_FIELDS - {"schemaVersion"}:
        _require_sha256(identity.get(name), name=f"V3 registry receipt identity {name}")
    root = _prepare_registry_root(registry_root, repository_root=repository_root)
    root_signatures = _capture_registry_root_signatures(root, repository_root=repository_root)
    key = successor.canonical_json_sha256(identity)
    receipt = {
        "schemaVersion": SYNTHETIC_NUISANCE_CONTROL_V3_RECEIPT_SCHEMA,
        "receiptKeySha256": key,
        "receiptIdentity": identity,
    }
    receipt["syntheticNuisanceControlReceiptSha256"] = _document_digest(
        receipt, "syntheticNuisanceControlReceiptSha256"
    )
    _require_exact_fields(receipt, name="V3 one-time registry receipt", fields=RECEIPT_FIELDS)
    target = root / f"{key[7:]}.json"
    data = (json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(target), flags, 0o600)
    except FileExistsError as error:
        raise SyntheticNuisanceControlV3Error(
            "V3 registry identity was already consumed; choose immutable new packages/model/contract"
        ) from error
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to create V3 one-time registry receipt") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            signature = os.fstat(stream.fileno())
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to write V3 one-time registry receipt") from error
    try:
        knn._reject_links_on_existing_path(root, description="V3 registry receipt")
        _verify_registry_root_signatures(
            root_signatures,
            root=root,
            repository_root=repository_root,
        )
        if knn._is_link_or_reparse_point(target) or _stat_signature(target) != (
            signature.st_dev,
            signature.st_ino,
            signature.st_mode,
            signature.st_size,
        ):
            raise SyntheticNuisanceControlV3Error("V3 registry receipt changed while it was written")
        return receipt, sha256_file(target)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to verify V3 one-time registry receipt") from error


def _validate_camera_control_record(
    record: dict[str, Any],
    *,
    parent: dict[str, Any],
    package_root: Path,
    recipe: dict[str, Any],
    recipe_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    try:
        parsed = camera._require_exact_fields(record, name="R3 capture-control record", required=camera.RECORD_FIELDS)
        case_id = camera._require_string(parsed.get("caseId"), name="R3 capture-control caseId")
        parent_case_id = camera._require_string(parsed.get("parentCaseId"), name="R3 capture-control parentCaseId")
        if parent_case_id != parent["caseId"]:
            raise SyntheticNuisanceControlV3Error("R3 capture-control parent does not match the fixed query split")
        if (
            camera._require_sha256(parsed.get("parentSourceSha256"), name="R3 capture-control parentSourceSha256")
            != parent["sourceSha256"]
            or parsed.get("sourceGroupId") != parent["sourceGroupId"]
            or parsed.get("category") != parent["category"]
            or parsed.get("parentPartition") != "FIT"
            or parsed.get("kind") != "NOMINAL"
            or parsed.get("defect") != "good"
        ):
            raise SyntheticNuisanceControlV3Error("R3 capture-control does not bind its FIT query parent")
        variant_id = camera._require_positive_int(parsed.get("variantId"), name="R3 capture-control variantId")
        component = camera._component_for_variant(variant_id)
        if parsed.get("component") != component:
            raise SyntheticNuisanceControlV3Error("R3 capture-control component does not match variantId")
        expected_case_id, expected_relative = camera._expected_child_identity(
            parent, recipe_sha256=recipe_sha256, variant_id=variant_id
        )
        if case_id != expected_case_id:
            raise SyntheticNuisanceControlV3Error("R3 capture-control caseId is not deterministic")
        relative = camera._safe_relative_path(parsed.get("relativePath"), name="R3 capture-control relativePath")
        if relative != expected_relative:
            raise SyntheticNuisanceControlV3Error("R3 capture-control relativePath is not deterministic")
        expected_parameters = camera.sample_successor_fit_parameters_v2(
            recipe,
            recipe_sha256=recipe_sha256,
            parent_case_id=parent_case_id,
            parent_source_sha256=str(parent["sourceSha256"]),
            variant_id=variant_id,
        )
        if camera._validate_parameters(parsed.get("parameters"), variant_id=variant_id) != expected_parameters:
            raise SyntheticNuisanceControlV3Error("R3 capture-control parameters do not match the locked recipe")
        camera._validate_output_encoding(parsed.get("outputEncoding"))
        image_path = camera._safe_file_under(
            package_root,
            relative,
            description="R3 capture-control output",
            repository_root=repository_root,
        )
        actual = image_path.read_bytes()
        if camera._sha256_bytes(actual) != camera._require_sha256(
            parsed.get("sourceSha256"), name="R3 capture-control sourceSha256"
        ):
            raise SyntheticNuisanceControlV3Error("R3 capture-control output digest does not match")
        if camera._inspect_jpeg_output_encoding(actual) != camera._expected_output_encoding():
            raise SyntheticNuisanceControlV3Error("R3 capture-control JPEG profile does not match")
        expected = camera._render_augmented_jpeg(
            camera._load_parent_rgb(parent), expected_parameters, variant_id=variant_id
        )
        if actual != expected:
            raise SyntheticNuisanceControlV3Error("R3 capture-control pixels do not match the deterministic renderer")
        return {**parsed, "imagePath": image_path}
    except camera.SuccessorFitAugmentationV2Error as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to read R3 capture-control output") from error


def load_and_validate_r3_camera_controls(
    augmentation_manifest_path: Path,
    *,
    recipe_path: Path,
    envelope: dict[str, Any],
    envelope_file_sha256: str,
    fit_parents: list[dict[str, Any]],
    query_parents: list[dict[str, Any]],
    expected_manifest_file_sha256: str | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Open exactly 36 R3 camera children from the fixed 12 query parents.

    The complete manifest remains cryptographically and contract validated,
    while child-image access is deliberately restricted to query parents.
    """

    document, manifest_file_sha256 = _read_external_metadata(
        augmentation_manifest_path,
        description="R3 generic-capture augmentation manifest",
        repository_root=repository_root,
    )
    if expected_manifest_file_sha256 is not None and manifest_file_sha256 != _require_sha256(
        expected_manifest_file_sha256,
        name="R3 expected preflight manifest digest",
    ):
        raise SyntheticNuisanceControlV3Error(
            "R3 generic-capture manifest changed after preflight; refusing before child-byte access"
        )
    try:
        recipe, recipe_sha256 = camera.load_successor_fit_camera_recipe_v2(recipe_path)
        raw_records = camera._validate_manifest_document(
            document,
            manifest_file_sha256=manifest_file_sha256,
            envelope=envelope,
            envelope_file_sha256=envelope_file_sha256,
            parents=fit_parents,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
            repository_root=repository_root,
        )
        camera._require_external_package_root(augmentation_manifest_path.parent, repository_root=repository_root)
    except ValueError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    _assert_chain_binding(document, envelope=envelope, envelope_file_sha256=envelope_file_sha256, label="R3 capture package")
    parent_by_case = {str(record["caseId"]): record for record in query_parents}
    if len(parent_by_case) != 12:
        raise SyntheticNuisanceControlV3Error("fixed query-parent split must contain exactly 12 parents")
    expected = {
        (parent_case_id, component)
        for parent_case_id in parent_by_case
        for component in SYNTHETIC_NUISANCE_CONTROL_V3_CONTROL_COMPONENTS
    }
    seen_case_ids: set[str] = set()
    found: set[tuple[str, str]] = set()
    validated: list[dict[str, Any]] = []
    for raw_record in raw_records:
        parent_case_id = raw_record.get("parentCaseId") if isinstance(raw_record, dict) else None
        if parent_case_id not in parent_by_case:
            continue
        parent = parent_by_case[parent_case_id]
        record = _validate_camera_control_record(
            raw_record,
            parent=parent,
            package_root=augmentation_manifest_path.parent,
            recipe=recipe,
            recipe_sha256=recipe_sha256,
            repository_root=repository_root,
        )
        case_id = str(record["caseId"])
        if case_id in seen_case_ids:
            raise SyntheticNuisanceControlV3Error("R3 capture-control caseId is duplicated")
        seen_case_ids.add(case_id)
        component = str(record["component"])
        key = (str(record["parentCaseId"]), component)
        if key in found:
            raise SyntheticNuisanceControlV3Error("R3 capture-control parent/component is duplicated")
        found.add(key)
        validated.append(record)
    if found != expected or len(validated) != 36:
        raise SyntheticNuisanceControlV3Error("R3 generic-capture package does not cover exactly 3 controls per query parent")
    return document, manifest_file_sha256, sorted(validated, key=lambda item: str(item["caseId"]))


def _feature_extractor_identity(*, model_repo: Path, model_weights: Path, device: str) -> dict[str, Any]:
    """Bind this auditor in addition to the frozen DINO feature implementation."""

    try:
        identity = dict(stress._feature_extractor_identity(model_repo=model_repo, model_weights=model_weights, device=device))
    except ValueError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    identity["syntheticNuisanceControlModuleSha256"] = sha256_file(Path(__file__))
    return identity


def _validate_v3_feature_extractor_identity(value: object) -> dict[str, Any]:
    """Require the sealed V2 identity plus the V3 auditor module binding."""

    if not isinstance(value, dict):
        raise SyntheticNuisanceControlV3Error("V3 feature extractor identity must be an object")
    identity = dict(value)
    try:
        nuisance_digest = stress._require_sha256(
            identity.pop("syntheticNuisanceControlModuleSha256", None),
            name="V3 feature extractor nuisance-control module digest",
        )
        stress._validate_feature_extractor(identity)
    except stress.SyntheticStressV2Error as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    if nuisance_digest != sha256_file(Path(__file__)):
        raise SyntheticNuisanceControlV3Error("V3 feature extractor nuisance-control module digest is inconsistent")
    return value


def _extract_features(records: list[dict[str, Any]], *, embedder: Any, timings: dict[str, float]) -> dict[str, object]:
    try:
        return stress._extract_features(records, embedder=embedder, timings=timings)
    except ValueError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error


def _score_records(
    records: list[dict[str, Any]], features: dict[str, object], *, prototype_banks: dict[str, object]
) -> list[dict[str, Any]]:
    try:
        return stress._score_records(records, features, prototype_banks=prototype_banks)
    except ValueError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error


def _adapt_camera_control(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "partition": "SYNTHETIC_QUERY",
        "kind": "NOMINAL",
        "defect": "good",
        "sourceSha256": record["sourceSha256"],
        "sourceGroupId": record["sourceGroupId"],
        "imagePath": record["imagePath"],
        "isAugmentation": False,
        "parentCaseId": record["parentCaseId"],
        "variantId": record["variantId"],
        "component": record["component"],
    }


def _calibration_output_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {
        "caseId": record["caseId"],
        "category": record["category"],
        "sourceSha256": record["sourceSha256"],
        "score": _require_finite(record.get("score"), name="raw calibration score"),
        "maxPatchDistance": _require_finite(record.get("maxPatchDistance"), name="raw calibration max patch distance"),
        "meanNearestPatchDistance": _require_finite(
            record.get("meanNearestPatchDistance"), name="raw calibration mean nearest patch distance"
        ),
    }
    _require_exact_fields(output, name="raw calibration score", fields=CALIBRATION_SCORE_FIELDS)
    return output


def _score_raw_queries(
    components: list[dict[str, Any]], *, thresholds: dict[str, float]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    expected = len(SYNTHETIC_NUISANCE_CONTROL_V3_CATEGORIES) * SYNTHETIC_NUISANCE_CONTROL_V3_PARENT_COUNTS[
        "SYNTHETIC_QUERY"
    ]
    if len(components) != expected:
        raise SyntheticNuisanceControlV3Error("raw query score count is inconsistent")
    records: list[dict[str, Any]] = []
    by_case: dict[str, dict[str, Any]] = {}
    for component in components:
        case_id = _require_string(component.get("caseId"), name="raw query caseId")
        category = _require_string(component.get("category"), name="raw query category")
        if category not in thresholds or case_id in by_case:
            raise SyntheticNuisanceControlV3Error("raw query score is unsafe")
        output = {
            "caseId": case_id,
            "category": category,
            "sourceSha256": _require_sha256(component.get("sourceSha256"), name="raw query sourceSha256"),
            "score": _require_finite(component.get("score"), name="raw query score"),
            "maxPatchDistance": _require_finite(component.get("maxPatchDistance"), name="raw query max patch distance"),
            "meanNearestPatchDistance": _require_finite(
                component.get("meanNearestPatchDistance"), name="raw query mean nearest patch distance"
            ),
            "aboveRawCalibrationThreshold": _require_finite(component.get("score"), name="raw query score")
            > thresholds[category],
        }
        _require_exact_fields(output, name="raw query score", fields=RAW_QUERY_SCORE_FIELDS)
        records.append(output)
        by_case[case_id] = output
    return sorted(records, key=lambda item: str(item["caseId"])), by_case


def _score_camera_controls(
    components: list[dict[str, Any]], *, raw_by_case: dict[str, dict[str, Any]], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    expected = len(raw_by_case) * len(SYNTHETIC_NUISANCE_CONTROL_V3_CONTROL_COMPONENTS)
    if len(components) != expected:
        raise SyntheticNuisanceControlV3Error("generic-capture control score count is inconsistent")
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for component in components:
        parent_case_id = _require_string(component.get("parentCaseId"), name="capture-control parentCaseId")
        parent = raw_by_case.get(parent_case_id)
        category = _require_string(component.get("category"), name="capture-control category")
        capture_component = component.get("component")
        if parent is None or parent["category"] != category or capture_component not in SYNTHETIC_NUISANCE_CONTROL_V3_CONTROL_COMPONENTS:
            raise SyntheticNuisanceControlV3Error("capture-control score does not pair to one raw query parent")
        key = (parent_case_id, str(capture_component))
        if key in seen:
            raise SyntheticNuisanceControlV3Error("capture-control score parent/component is duplicated")
        seen.add(key)
        score = _require_finite(component.get("score"), name="capture-control score")
        record = {
            "caseId": _require_string(component.get("caseId"), name="capture-control caseId"),
            "parentCaseId": parent_case_id,
            "category": category,
            "sourceSha256": _require_sha256(component.get("sourceSha256"), name="capture-control sourceSha256"),
            "captureComponent": capture_component,
            "score": score,
            "maxPatchDistance": _require_finite(component.get("maxPatchDistance"), name="capture-control max patch distance"),
            "meanNearestPatchDistance": _require_finite(
                component.get("meanNearestPatchDistance"), name="capture-control mean nearest patch distance"
            ),
            "aboveRawCalibrationThreshold": score > thresholds[category],
        }
        _require_exact_fields(record, name="capture-control score", fields=CONTROL_SCORE_FIELDS)
        output.append(record)
    expected_keys = {
        (parent_case_id, component)
        for parent_case_id in raw_by_case
        for component in SYNTHETIC_NUISANCE_CONTROL_V3_CONTROL_COMPONENTS
    }
    if seen != expected_keys:
        raise SyntheticNuisanceControlV3Error("generic-capture controls do not cover every query parent and component")
    return sorted(output, key=lambda item: str(item["caseId"]))


def _score_stimuli(
    components: list[dict[str, Any]], *, raw_by_case: dict[str, dict[str, Any]], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    expected = len(raw_by_case) * len(SYNTHETIC_NUISANCE_CONTROL_V3_FAMILIES) * len(SYNTHETIC_NUISANCE_CONTROL_V3_LEVELS)
    if len(components) != expected:
        raise SyntheticNuisanceControlV3Error("synthetic-stimulus score count is inconsistent")
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for component in components:
        parent_case_id = _require_string(component.get("parentCaseId"), name="stimulus parentCaseId")
        parent = raw_by_case.get(parent_case_id)
        category = _require_string(component.get("category"), name="stimulus category")
        family = component.get("syntheticDefectFamily")
        level = component.get("renderIntensityLevel")
        if (
            parent is None
            or parent["category"] != category
            or family not in SYNTHETIC_NUISANCE_CONTROL_V3_FAMILIES
            or level not in SYNTHETIC_NUISANCE_CONTROL_V3_LEVELS
        ):
            raise SyntheticNuisanceControlV3Error("synthetic-stimulus score does not pair to one raw query parent")
        key = (parent_case_id, str(family), str(level))
        if key in seen:
            raise SyntheticNuisanceControlV3Error("synthetic-stimulus score parent/family/level is duplicated")
        seen.add(key)
        score = _require_finite(component.get("score"), name="synthetic-stimulus score")
        record = {
            "caseId": _require_string(component.get("caseId"), name="synthetic-stimulus caseId"),
            "parentCaseId": parent_case_id,
            "category": category,
            "sourceSha256": _require_sha256(component.get("sourceSha256"), name="synthetic-stimulus sourceSha256"),
            "syntheticDefectFamily": family,
            "renderIntensityLevel": level,
            "score": score,
            "maxPatchDistance": _require_finite(component.get("maxPatchDistance"), name="synthetic-stimulus max patch distance"),
            "meanNearestPatchDistance": _require_finite(
                component.get("meanNearestPatchDistance"), name="synthetic-stimulus mean nearest patch distance"
            ),
            "aboveRawCalibrationThreshold": score > thresholds[category],
        }
        _require_exact_fields(record, name="synthetic-stimulus score", fields=STIMULUS_SCORE_FIELDS)
        output.append(record)
    expected_keys = {
        (parent_case_id, family, level)
        for parent_case_id in raw_by_case
        for family in SYNTHETIC_NUISANCE_CONTROL_V3_FAMILIES
        for level in SYNTHETIC_NUISANCE_CONTROL_V3_LEVELS
    }
    if seen != expected_keys:
        raise SyntheticNuisanceControlV3Error("synthetic stimuli do not cover every query parent, family, and level")
    return sorted(output, key=lambda item: str(item["caseId"]))


def build_parent_level_contrasts(
    raw_scores: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    stimuli: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Make the control-vs-stimulus comparison at parent level (n=12)."""

    raw_by_case = {str(record["caseId"]): record for record in raw_scores}
    if len(raw_by_case) != 12:
        raise SyntheticNuisanceControlV3Error("parent-level contrast requires exactly 12 raw query parents")
    controls_by_parent: dict[str, list[dict[str, Any]]] = {key: [] for key in raw_by_case}
    stimuli_by_parent: dict[str, list[dict[str, Any]]] = {key: [] for key in raw_by_case}
    for record in controls:
        controls_by_parent.setdefault(str(record["parentCaseId"]), []).append(record)
    for record in stimuli:
        stimuli_by_parent.setdefault(str(record["parentCaseId"]), []).append(record)
    result: list[dict[str, Any]] = []
    for parent_case_id, parent in raw_by_case.items():
        parent_controls = controls_by_parent.get(parent_case_id, [])
        parent_stimuli = stimuli_by_parent.get(parent_case_id, [])
        if len(parent_controls) != 3 or len(parent_stimuli) != 9:
            raise SyntheticNuisanceControlV3Error("parent-level contrast requires 3 controls and 9 stimuli per parent")
        control_deltas = [float(record["score"]) - float(parent["score"]) for record in parent_controls]
        stimulus_deltas = [float(record["score"]) - float(parent["score"]) for record in parent_stimuli]
        record = {
            "parentCaseId": parent_case_id,
            "category": parent["category"],
            "rawParentScore": float(parent["score"]),
            "rawParentAboveThreshold": bool(parent["aboveRawCalibrationThreshold"]),
            "genericCaptureControlCount": len(parent_controls),
            "genericCaptureControlAboveThresholdCount": sum(
                bool(item["aboveRawCalibrationThreshold"]) for item in parent_controls
            ),
            "genericCaptureMeanChildMinusParentScore": statistics.fmean(control_deltas),
            "syntheticStimulusCount": len(parent_stimuli),
            "syntheticStimulusAboveThresholdCount": sum(
                bool(item["aboveRawCalibrationThreshold"]) for item in parent_stimuli
            ),
            "syntheticStimulusMeanChildMinusParentScore": statistics.fmean(stimulus_deltas),
            "syntheticStimulusMinusGenericCaptureMeanDelta": statistics.fmean(stimulus_deltas)
            - statistics.fmean(control_deltas),
        }
        _require_exact_fields(record, name="parent-level contrast", fields=PARENT_CONTRAST_FIELDS)
        result.append(record)
    return sorted(result, key=lambda item: str(item["parentCaseId"]))


def _response_summary(
    raw_scores: list[dict[str, Any]], controls: list[dict[str, Any]], stimuli: list[dict[str, Any]]
) -> dict[str, Any]:
    if not raw_scores or not controls or not stimuli:
        raise SyntheticNuisanceControlV3Error("response summary requires raw parents, controls, and stimuli")
    counts = {
        "rawQueryCount": len(raw_scores),
        "rawQueryAboveThresholdCount": sum(bool(item["aboveRawCalibrationThreshold"]) for item in raw_scores),
        "genericCaptureControlCount": len(controls),
        "genericCaptureControlAboveThresholdCount": sum(bool(item["aboveRawCalibrationThreshold"]) for item in controls),
        "syntheticStimulusCount": len(stimuli),
        "syntheticStimulusAboveThresholdCount": sum(bool(item["aboveRawCalibrationThreshold"]) for item in stimuli),
    }
    return {
        "responseCounts": counts,
        "responseRates": {
            "rawQueryAboveThresholdRate": counts["rawQueryAboveThresholdCount"] / counts["rawQueryCount"],
            "genericCaptureControlAboveThresholdRate": counts["genericCaptureControlAboveThresholdCount"]
            / counts["genericCaptureControlCount"],
            "syntheticStimulusAboveThresholdRate": counts["syntheticStimulusAboveThresholdCount"]
            / counts["syntheticStimulusCount"],
        },
    }


def _contrast_summary(parent_contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parent_contrasts:
        raise SyntheticNuisanceControlV3Error("parent-level contrast summary requires parent contrasts")
    values = [
        _require_finite(item.get("syntheticStimulusMinusGenericCaptureMeanDelta"), name="parent-level contrast delta")
        for item in parent_contrasts
    ]
    control_values = [
        _require_finite(item.get("genericCaptureMeanChildMinusParentScore"), name="control parent delta")
        for item in parent_contrasts
    ]
    stimulus_values = [
        _require_finite(item.get("syntheticStimulusMeanChildMinusParentScore"), name="stimulus parent delta")
        for item in parent_contrasts
    ]
    return {
        "parentCount": len(parent_contrasts),
        "genericCaptureControlsPerParent": 3,
        "syntheticStimuliPerParent": 9,
        "meanGenericCaptureChildMinusParentScore": statistics.fmean(control_values),
        "meanSyntheticStimulusChildMinusParentScore": statistics.fmean(stimulus_values),
        "meanSyntheticStimulusMinusGenericCaptureDelta": statistics.fmean(values),
        "medianSyntheticStimulusMinusGenericCaptureDelta": statistics.median(values),
        "minimumSyntheticStimulusMinusGenericCaptureDelta": min(values),
        "maximumSyntheticStimulusMinusGenericCaptureDelta": max(values),
        "positiveSyntheticStimulusMinusGenericCaptureParentCount": sum(value > 0.0 for value in values),
        "zeroSyntheticStimulusMinusGenericCaptureParentCount": sum(value == 0.0 for value in values),
    }


def _category_report(
    category: str,
    *,
    threshold: float,
    prototype_patch_count: int,
    calibration_scores: list[dict[str, Any]],
    raw_scores: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    stimuli: list[dict[str, Any]],
    parent_contrasts: list[dict[str, Any]],
) -> dict[str, Any]:
    category_raw = [item for item in raw_scores if item["category"] == category]
    category_controls = [item for item in controls if item["category"] == category]
    category_stimuli = [item for item in stimuli if item["category"] == category]
    category_contrasts = [item for item in parent_contrasts if item["category"] == category]
    return {
        "prototypeParentCount": SYNTHETIC_NUISANCE_CONTROL_V3_PARENT_COUNTS["SYNTHETIC_PROTOTYPE"],
        "calibrationParentCount": sum(item["category"] == category for item in calibration_scores),
        "rawQueryParentCount": len(category_raw),
        "genericCaptureControlCount": len(category_controls),
        "syntheticStimulusCount": len(category_stimuli),
        "prototypePatchCount": prototype_patch_count,
        "thresholdFromRawCalibration": threshold,
        **_response_summary(category_raw, category_controls, category_stimuli),
        "parentLevelContrast": _contrast_summary(category_contrasts),
        "byGenericCaptureComponent": {
            component: _response_summary(
                category_raw,
                [item for item in category_controls if item["captureComponent"] == component],
                category_stimuli,
            )
            for component in SYNTHETIC_NUISANCE_CONTROL_V3_CONTROL_COMPONENTS
        },
        "bySyntheticDefectFamily": {
            family: _response_summary(
                category_raw,
                category_controls,
                [item for item in category_stimuli if item["syntheticDefectFamily"] == family],
            )
            for family in SYNTHETIC_NUISANCE_CONTROL_V3_FAMILIES
        },
        "byRenderIntensityLevel": {
            level: _response_summary(
                category_raw,
                category_controls,
                [item for item in category_stimuli if item["renderIntensityLevel"] == level],
            )
            for level in SYNTHETIC_NUISANCE_CONTROL_V3_LEVELS
        },
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
        "syntheticNuisanceControlModuleSha256": sha256_file(Path(__file__)),
        "entrypointSha256": sha256_file(repository_root / "tools" / "run_mvtec_ad_synthetic_nuisance_control_v3.py"),
        "timingBasis": "PROCESS_CPU_TIME_EXCLUDES_SUSPEND",
        "phaseTimingsSeconds": {name: round(value, 6) for name, value in timings.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "gitRevision": revision,
    }


def _write_new_report(path: Path, document: dict[str, Any], *, repository_root: Path) -> None:
    """Write a V3-specific immutable report without invoking a V2 validator."""

    try:
        prepared = knn._require_external_output(path, repository_root=repository_root)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    try:
        data = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise SyntheticNuisanceControlV3Error("unable to serialize V3 nuisance-control report as finite JSON") from error
    _recheck_v3_report_parent_chain(prepared, repository_root=repository_root)
    parent_signatures = _capture_v3_report_parent_signatures(prepared)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(prepared), flags, 0o600)
    except FileExistsError as error:
        raise SyntheticNuisanceControlV3Error("V3 nuisance-control report already exists") from error
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to write V3 nuisance-control report") from error
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("unable to write V3 nuisance-control report")
            offset += written
        os.fsync(descriptor)
        signature = _fd_signature(descriptor)
        _verify_v3_report_parent_signatures(
            parent_signatures, path=prepared, repository_root=repository_root
        )
        if knn._is_link_or_reparse_point(prepared) or _stat_signature(prepared) != signature:
            raise SyntheticNuisanceControlV3Error("V3 nuisance-control report changed while it was written")
    except SyntheticNuisanceControlV3Error:
        raise
    except OSError as error:
        raise SyntheticNuisanceControlV3Error("unable to verify V3 nuisance-control report output") from error
    finally:
        os.close(descriptor)


def run_synthetic_nuisance_control_v3(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    stimulus_augmentation_manifest_path: Path,
    capture_control_augmentation_manifest_path: Path,
    output_path: Path,
    *,
    source_root: Path,
    stimulus_recipe_path: Path,
    capture_control_recipe_path: Path,
    registry_root: Path,
    model_repo: Path,
    model_weights: Path,
    device: str = "cpu",
    repository_root: Path = REPOSITORY_ROOT,
    embedder_factory: Callable[..., Any] = knn.SuccessorV2BatchEmbedder,
    identity_factory: Callable[..., dict[str, Any]] = _feature_extractor_identity,
) -> dict[str, Any]:
    """Run one immutable V3 paired nuisance-control response-only audit.

    The raw FIT threshold is fixed before either package child loader can run.
    The one-time registry receipt is then atomically written before the 36
    camera controls or 108 synthetic stimuli are opened.
    """

    if device != "cpu":
        raise SyntheticNuisanceControlV3Error("V3 nuisance-control audit supports CPU only")
    # This deliberately includes output-slot and registry-root preparation.
    # Report serialization/write is excluded and named explicitly below.
    started = time.process_time()
    try:
        prepared_output = knn._require_external_output(output_path, repository_root=repository_root)
    except knn.SuccessorV2EvaluatorError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    # It is safe to create the registry directory early, but the immutable
    # receipt itself is intentionally delayed until raw and model preflight.
    prepared_registry_root = _prepare_registry_root(registry_root, repository_root=repository_root)
    registry_root_signatures = _capture_registry_root_signatures(
        prepared_registry_root,
        repository_root=repository_root,
    )
    timings = {
        "rawFitAndThresholdSeconds": 0.0,
        "modelPreflightSeconds": 0.0,
        "packageMetadataAndReceiptSeconds": 0.0,
        "childValidationSeconds": 0.0,
        "featureProcessingSeconds": 0.0,
        "scoringSeconds": 0.0,
        "totalBeforeReportWriteSeconds": 0.0,
    }

    # V3 hardened evidence requires the sealed V2 identity plus its own
    # module binding before any FIT parent can be requested through the
    # public API.
    model_started = time.process_time()
    extractor_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    _validate_v3_feature_extractor_identity(extractor_identity)
    extractor_identity_sha256 = successor.canonical_json_sha256(extractor_identity)
    embedder = embedder_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    loaded_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    _validate_v3_feature_extractor_identity(loaded_identity)
    if loaded_identity != extractor_identity:
        raise SyntheticNuisanceControlV3Error("feature extractor changed while DINO loaded")
    timings["modelPreflightSeconds"] += time.process_time() - model_started

    raw_started = time.process_time()
    try:
        envelope, envelope_file_sha256, parents = stress.load_safe_v2_fit_inputs(
            parent_holdout_path,
            parent_selection_contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            repository_root=repository_root,
        )
        split = stress.build_fixed_parent_split(parents)
    except ValueError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    timings["rawFitAndThresholdSeconds"] += time.process_time() - raw_started

    # Start a new interval after model preflight so rawFitAndThresholdSeconds
    # neither contains model load/identity checks nor double-counts FIT setup.
    raw_threshold_started = time.process_time()
    raw_feature_records = [stress._adapt_raw(record) for record in parents]
    raw_features = _extract_features(raw_feature_records, embedder=embedder, timings=timings)
    prototype_records = [stress._adapt_raw(record) for record in split["SYNTHETIC_PROTOTYPE"]]
    calibration_records = [stress._adapt_raw(record) for record in split["SYNTHETIC_CALIBRATION"]]
    raw_query_records = [stress._adapt_raw(record) for record in split["SYNTHETIC_QUERY"]]
    try:
        prototype_banks = {
            category: stress._prototype_bank(prototype_records, raw_features, category=category)
            for category in SYNTHETIC_NUISANCE_CONTROL_V3_CATEGORIES
        }
        scoring_started = time.process_time()
        raw_calibration_components = _score_records(calibration_records, raw_features, prototype_banks=prototype_banks)
        thresholds = stress.calibrate_raw_thresholds(raw_calibration_components)
        raw_query_components = _score_records(raw_query_records, raw_features, prototype_banks=prototype_banks)
        timings["scoringSeconds"] += time.process_time() - scoring_started
    except ValueError as error:
        raise SyntheticNuisanceControlV3Error(str(error)) from error
    timings["rawFitAndThresholdSeconds"] += time.process_time() - raw_threshold_started

    # Only manifest/recipe metadata is read in this block.  No camera or
    # stimulus child byte can be accessed before the immutable receipt exists.
    receipt_started = time.process_time()
    stimulus_metadata = _preflight_stimulus_package(
        stimulus_augmentation_manifest_path,
        recipe_path=stimulus_recipe_path,
        envelope=envelope,
        envelope_file_sha256=envelope_file_sha256,
        repository_root=repository_root,
        fit_parents=parents,
    )
    camera_metadata = _preflight_camera_package(
        capture_control_augmentation_manifest_path,
        recipe_path=capture_control_recipe_path,
        envelope=envelope,
        envelope_file_sha256=envelope_file_sha256,
        fit_parents=parents,
        repository_root=repository_root,
    )
    parent_split_records = stress._parent_split_records(split)
    if any(set(item) != PARENT_SPLIT_RECORD_FIELDS for item in parent_split_records):  # pragma: no cover - invariant guard
        raise SyntheticNuisanceControlV3Error("parent split record shape is unsafe")
    receipt_identity = _receipt_identity(
        envelope=envelope,
        envelope_file_sha256=envelope_file_sha256,
        parent_split_records=parent_split_records,
        feature_extractor_identity_sha256=extractor_identity_sha256,
        stimulus_metadata=stimulus_metadata,
        camera_metadata=camera_metadata,
    )
    receipt, receipt_file_sha256 = create_one_time_registry_receipt(
        prepared_registry_root, identity=receipt_identity, repository_root=repository_root
    )
    timings["packageMetadataAndReceiptSeconds"] += time.process_time() - receipt_started

    # Recheck the originally prepared root, not merely the path resolved by
    # receipt creation.  A same-privilege root swap/reparse must fail before
    # either child-package loader can open a camera or stimulus image.
    _verify_registry_receipt_binding(
        prepared_registry_root,
        registry_root_signatures,
        receipt=receipt,
        receipt_file_sha256=receipt_file_sha256,
        repository_root=repository_root,
    )

    child_started = time.process_time()
    capture_manifest, capture_file_sha256, control_records = load_and_validate_r3_camera_controls(
        capture_control_augmentation_manifest_path,
        recipe_path=capture_control_recipe_path,
        envelope=envelope,
        envelope_file_sha256=envelope_file_sha256,
        fit_parents=parents,
        query_parents=split["SYNTHETIC_QUERY"],
        expected_manifest_file_sha256=camera_metadata["fileSha256"],
        repository_root=repository_root,
    )
    stimulus_manifest, stimulus_file_sha256, stimulus_records = stress.load_and_validate_v2_package(
        stimulus_augmentation_manifest_path,
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=stimulus_recipe_path,
        query_parents=split["SYNTHETIC_QUERY"],
        expected_manifest_file_sha256=stimulus_metadata["fileSha256"],
        repository_root=repository_root,
    )
    if (
        capture_file_sha256 != camera_metadata["fileSha256"]
        or capture_manifest.get("augmentationManifestSha256")
        != camera_metadata["document"].get("augmentationManifestSha256")
        or stimulus_file_sha256 != stimulus_metadata["fileSha256"]
        or stimulus_manifest.get("augmentationManifestSha256")
        != stimulus_metadata["document"].get("augmentationManifestSha256")
    ):
        raise SyntheticNuisanceControlV3Error("package metadata changed after one-time receipt creation")
    timings["childValidationSeconds"] += time.process_time() - child_started

    control_features = _extract_features(
        [_adapt_camera_control(record) for record in control_records], embedder=embedder, timings=timings
    )
    stimulus_features = _extract_features(
        [stress._adapt_stimulus_record(record) for record in stimulus_records], embedder=embedder, timings=timings
    )
    scoring_started = time.process_time()
    control_components = _score_records(
        [_adapt_camera_control(record) for record in control_records], control_features, prototype_banks=prototype_banks
    )
    stimulus_components = _score_records(
        [stress._adapt_stimulus_record(record) for record in stimulus_records],
        stimulus_features,
        prototype_banks=prototype_banks,
    )
    timings["scoringSeconds"] += time.process_time() - scoring_started

    calibration_scores = [_calibration_output_record(item) for item in raw_calibration_components]
    raw_query_scores, raw_by_case = _score_raw_queries(raw_query_components, thresholds=thresholds)
    control_scores = _score_camera_controls(control_components, raw_by_case=raw_by_case, thresholds=thresholds)
    stimulus_scores = _score_stimuli(stimulus_components, raw_by_case=raw_by_case, thresholds=thresholds)
    parent_contrasts = build_parent_level_contrasts(raw_query_scores, control_scores, stimulus_scores)

    completed_identity = identity_factory(model_repo=model_repo, model_weights=model_weights, device=device)
    _validate_v3_feature_extractor_identity(completed_identity)
    if completed_identity != extractor_identity:
        raise SyntheticNuisanceControlV3Error("feature extractor changed while V3 nuisance-control audit ran")
    timings["totalBeforeReportWriteSeconds"] = time.process_time() - started

    parent_evidence = envelope.get("parentEvidence")
    partition_identities = envelope.get("successorPartitionIdentities")
    if not isinstance(parent_evidence, dict) or not isinstance(partition_identities, dict):
        raise SyntheticNuisanceControlV3Error("successor envelope is missing closed parent evidence")
    aggregate = {
        **_response_summary(raw_query_scores, control_scores, stimulus_scores),
        "parentLevelContrast": _contrast_summary(parent_contrasts),
    }
    categories = {
        category: _category_report(
            category,
            threshold=thresholds[category],
            prototype_patch_count=int(prototype_banks[category].shape[0]),
            calibration_scores=calibration_scores,
            raw_scores=raw_query_scores,
            controls=control_scores,
            stimuli=stimulus_scores,
            parent_contrasts=parent_contrasts,
        )
        for category in SYNTHETIC_NUISANCE_CONTROL_V3_CATEGORIES
    }
    document: dict[str, Any] = {
        "schemaVersion": SYNTHETIC_NUISANCE_CONTROL_V3_REPORT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "comparisonOrPromotionAllowed": False,
        "purpose": SYNTHETIC_NUISANCE_CONTROL_V3_PURPOSE,
        "phase": SYNTHETIC_NUISANCE_CONTROL_V3_PHASE,
        "metricScope": SYNTHETIC_NUISANCE_CONTROL_V3_METRIC_SCOPE,
        "evidenceClass": SYNTHETIC_NUISANCE_CONTROL_V3_EVIDENCE_CLASS,
        "realAnomalyPerformance": SYNTHETIC_NUISANCE_CONTROL_V3_REAL_PERFORMANCE,
        "realPrecisionRecall": SYNTHETIC_NUISANCE_CONTROL_V3_REAL_PERFORMANCE,
        "forbiddenUses": list(SYNTHETIC_NUISANCE_CONTROL_V3_FORBIDDEN_USES),
        "inputPolicy": SYNTHETIC_NUISANCE_CONTROL_V3_INPUT_POLICY,
        "blindPolicy": SYNTHETIC_NUISANCE_CONTROL_V3_BLIND_POLICY,
        "resultLabel": SYNTHETIC_NUISANCE_CONTROL_V3_RESULT_LABEL,
        "testConfiguration": {
            "splitAlgorithm": stress.SYNTHETIC_STRESS_V2_PARENT_SPLIT_ALGORITHM,
            "parentCountsPerCategory": dict(SYNTHETIC_NUISANCE_CONTROL_V3_PARENT_COUNTS),
            "genericCaptureControlsPerQueryParent": 3,
            "syntheticStimuliPerQueryParent": 9,
            "decisionRule": SYNTHETIC_NUISANCE_CONTROL_V3_DECISION_RULE,
            "rawCalibrationThresholdEstablishedBeforeChildPackageLoad": True,
            "rawCalibrationThresholdEstablishedBeforeChildPackageScoring": True,
            "registryReceiptCreatedBeforeChildPackageLoad": True,
            "parentLevelContrastUnit": "RAW_QUERY_PARENT_N_EQUALS_12",
            "genericCaptureComponents": list(SYNTHETIC_NUISANCE_CONTROL_V3_CONTROL_COMPONENTS),
            "syntheticFamilies": list(SYNTHETIC_NUISANCE_CONTROL_V3_FAMILIES),
            "syntheticRenderIntensityLevels": list(SYNTHETIC_NUISANCE_CONTROL_V3_LEVELS),
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
        "stimulusPackageManifestFileSha256": stimulus_file_sha256,
        "stimulusPackageManifestDeclaredSha256": stimulus_manifest.get("augmentationManifestSha256"),
        "stimulusPackageSchemaVersion": stimulus_manifest.get("schemaVersion"),
        "stimulusPackageRecipeFileSha256": stimulus_manifest.get("recipeFileSha256"),
        "captureControlPackageManifestFileSha256": capture_file_sha256,
        "captureControlPackageManifestDeclaredSha256": capture_manifest.get("augmentationManifestSha256"),
        "captureControlPackageSchemaVersion": capture_manifest.get("schemaVersion"),
        "captureControlPackageRecipeFileSha256": capture_manifest.get("recipeFileSha256"),
        "parentSplit": parent_split_records,
        "parentSplitIdentitySha256": successor.canonical_json_sha256(parent_split_records),
        "featureExtractor": extractor_identity,
        "featureExtractorIdentitySha256": extractor_identity_sha256,
        "registryReceiptKeySha256": receipt["receiptKeySha256"],
        "registryReceiptFileSha256": receipt_file_sha256,
        "registryReceiptDeclaredSha256": receipt["syntheticNuisanceControlReceiptSha256"],
        "thresholds": thresholds,
        "calibrationScores": sorted(calibration_scores, key=lambda item: str(item["caseId"])),
        "rawQueryScores": raw_query_scores,
        "genericCaptureControlScores": control_scores,
        "syntheticStimulusScores": stimulus_scores,
        "parentLevelContrasts": parent_contrasts,
        "categories": categories,
        "aggregate": aggregate,
        "execution": _execution_metadata(timings, repository_root=repository_root),
    }
    sha_fields = {
        name
        for name in REPORT_FIELDS
        if name.endswith("Sha256") and name != "syntheticNuisanceControlReportSha256"
    }
    for name in sha_fields:
        document[name] = _require_sha256(document.get(name), name=f"V3 report {name}")
    if (
        document["stimulusPackageSchemaVersion"] != stimulus.SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA
        or document["captureControlPackageSchemaVersion"] != camera.SUCCESSOR_FIT_AUGMENTATION_V2_SCHEMA
    ):
        raise SyntheticNuisanceControlV3Error("child package schema changed while V3 audit ran")
    _require_exact_fields(document, name="V3 nuisance-control response report", fields=REPORT_FIELDS - {"syntheticNuisanceControlReportSha256"})
    document["syntheticNuisanceControlReportSha256"] = _document_digest(document, "syntheticNuisanceControlReportSha256")
    _require_exact_fields(document, name="V3 nuisance-control response report", fields=REPORT_FIELDS)
    _write_new_report(prepared_output, document, repository_root=repository_root)
    return document
