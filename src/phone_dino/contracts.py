from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
PrefixedSha256 = Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
Identifier = Annotated[str, Field(min_length=1, max_length=160)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class ExecutionBundle(StrictModel):
    recipe_version: PrefixedSha256 = Field(alias="recipeVersion")
    golden_set_version: PrefixedSha256 = Field(alias="goldenSetVersion")
    capture_policy_version: PrefixedSha256 = Field(alias="capturePolicyVersion")
    decision_policy_version: PrefixedSha256 = Field(alias="decisionPolicyVersion")
    normalization_pipeline_version: PrefixedSha256 = Field(alias="normalizationPipelineVersion")
    analyzer_model_version: PrefixedSha256 = Field(alias="analyzerModelVersion")
    client_asset_version: PrefixedSha256 = Field(alias="clientAssetVersion")
    board_installation_version: PrefixedSha256 = Field(alias="boardInstallationVersion")


class AnalyzeRequest(StrictModel):
    schema_version: Literal["1.0"] = Field(alias="schemaVersion")
    request_id: Identifier = Field(alias="requestId")
    session_id: Identifier = Field(alias="sessionId")
    capture_ordinal: Annotated[int, Field(ge=1, le=1000)] = Field(alias="captureOrdinal")
    correlation_id: Identifier = Field(alias="correlationId")
    deadline: datetime
    raw_sha256: Sha256 = Field(alias="rawSha256")
    content_type: Literal["image/jpeg", "image/png"] = Field(alias="contentType")
    recipe_id: Identifier = Field(alias="recipeId")
    machine_id: Identifier = Field(alias="machineId")
    board_id: Identifier = Field(alias="boardId")
    inspection_intent: Literal["PM_SIMILARITY"] = Field(alias="inspectionIntent")
    execution_bundle_digest: PrefixedSha256 = Field(alias="executionBundleDigest")
    execution_bundle: ExecutionBundle = Field(alias="executionBundle")
    artifact_package_digest: PrefixedSha256 = Field(alias="artifactPackageDigest")
    simulation: bool

    @field_validator("deadline")
    @classmethod
    def deadline_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("deadline must include a timezone")
        return value


class CaptureState(str, Enum):
    ACCEPTED = "ACCEPTED"
    RECAPTURE_REQUIRED = "RECAPTURE_REQUIRED"


class AnalysisState(str, Enum):
    RUN = "RUN"
    NOT_RUN = "NOT_RUN"


class CaptureAssessment(StrictModel):
    state: CaptureState
    reason_codes: Annotated[list[str], Field(alias="reasonCodes", max_length=32)]


class AlignmentObservation(StrictModel):
    state: Literal["ALIGNED", "NOT_ALIGNED"]
    method: Literal["TARGET_HOMOGRAPHY", "TARGET_AFFINE", "LIGHTGLUE_RESIDUAL", "SIMULATION_FIXTURE"]
    target_relative: Literal[True] = Field(alias="targetRelative")
    inlier_count: Annotated[int, Field(alias="inlierCount", ge=0, le=10000)]
    inlier_ratio: Annotated[float, Field(alias="inlierRatio", ge=0, le=1)]
    reprojection_error_px: Annotated[float, Field(alias="reprojectionErrorPx", ge=0, le=1000)]
    coverage_ratio: Annotated[float, Field(alias="coverageRatio", ge=0, le=1)]
    transform_within_bounds: bool = Field(alias="transformWithinBounds")
    inspection_mask_applied: bool = Field(alias="inspectionMaskApplied")


class NormalizationObservation(StrictModel):
    canonical_sha256: Sha256 | None = Field(default=None, alias="canonicalSha256")
    alignment: AlignmentObservation | None = None


class AnalysisObservation(StrictModel):
    state: AnalysisState
    metric: Literal["cosine_distance"] | None = None
    global_distance: Annotated[float, Field(ge=0, le=2)] | None = Field(default=None, alias="globalDistance")
    nearest_golden_id: str | None = Field(default=None, alias="nearestGoldenId", max_length=160)
    nearest_golden_distance: Annotated[float, Field(ge=0, le=2)] | None = Field(default=None, alias="nearestGoldenDistance")
    uncertainty: Annotated[float, Field(ge=0, le=1)] | None = None


class ResolvedVersions(StrictModel):
    execution_bundle_digest: PrefixedSha256 = Field(alias="executionBundleDigest")
    artifact_package_digest: PrefixedSha256 = Field(alias="artifactPackageDigest")
    analyzer_model_version: PrefixedSha256 = Field(alias="analyzerModelVersion")
    analyzer_runtime_version: PrefixedSha256 = Field(alias="analyzerRuntimeVersion")
    normalization_runtime_version: PrefixedSha256 = Field(alias="normalizationRuntimeVersion")


class AnalyzeObservation(StrictModel):
    schema_version: Literal["1.0"] = Field(alias="schemaVersion")
    request_id: Identifier = Field(alias="requestId")
    analysis_id: Sha256 = Field(alias="analysisId")
    raw_sha256: Sha256 = Field(alias="rawSha256")
    simulation: bool
    resolved_versions: ResolvedVersions = Field(alias="resolvedVersions")
    capture_assessment: CaptureAssessment = Field(alias="captureAssessment")
    normalization: NormalizationObservation | None = None
    analysis: AnalysisObservation
