from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
PrefixedSha256 = Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
Identifier = Annotated[str, Field(min_length=1, max_length=160)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class GoldenDimensionBoardCandidate(StrictModel):
    """Server-qualified printable geometry; QR is intentionally not used for measurement."""

    board_id: Identifier = Field(alias="boardId")
    revision: Annotated[int, Field(ge=1, le=1_000_000)]
    profile: Identifier
    manifest_sha256: PrefixedSha256 = Field(alias="manifestSha256")
    dictionary: Annotated[str, Field(pattern=r"^DICT_[A-Z0-9_]+$")]
    squares_x: Annotated[int, Field(alias="squaresX", ge=3, le=100)]
    squares_y: Annotated[int, Field(alias="squaresY", ge=3, le=100)]
    square_length_mm: Annotated[float, Field(alias="squareLengthMm", gt=0, le=1000)]
    marker_length_mm: Annotated[float, Field(alias="markerLengthMm", gt=0, le=1000)]
    marker_ids: Annotated[list[int], Field(alias="markerIds", min_length=1, max_length=5000)]

    @model_validator(mode="after")
    def validate_charuco_geometry(self) -> "GoldenDimensionBoardCandidate":
        expected = self.squares_x * self.squares_y // 2
        if self.marker_length_mm >= self.square_length_mm:
            raise ValueError("markerLengthMm must be smaller than squareLengthMm")
        if len(self.marker_ids) != expected or len(set(self.marker_ids)) != expected:
            raise ValueError("markerIds must contain one unique ID per ChArUco marker cell")
        return self


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
    schema_version: Literal["1.0", "1.1", "1.2", "1.3", "1.4"] = Field(alias="schemaVersion")
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
    recipe_analysis_profile_digest: PrefixedSha256 | None = Field(default=None, alias="recipeAnalysisProfileDigest")
    board_candidates: Annotated[
        list[GoldenDimensionBoardCandidate], Field(alias="boardCandidates", max_length=8),
    ] = Field(default_factory=list)
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
    method: Literal[
        "TARGET_HOMOGRAPHY", "TARGET_AFFINE", "CONTOUR_ANCHOR_AFFINE",
        "SUBJECT_CONTOUR_ECC_AFFINE", "LIGHTGLUE_RESIDUAL", "SIMULATION_FIXTURE",
    ]
    target_relative: Literal[True] = Field(alias="targetRelative")
    inlier_count: Annotated[int, Field(alias="inlierCount", ge=0, le=10000)]
    inlier_ratio: Annotated[float, Field(alias="inlierRatio", ge=0, le=1)]
    reprojection_error_px: Annotated[float, Field(alias="reprojectionErrorPx", ge=0, le=1000)]
    coverage_ratio: Annotated[float, Field(alias="coverageRatio", ge=0, le=1)]
    transform_within_bounds: bool = Field(alias="transformWithinBounds")
    inspection_mask_applied: bool = Field(alias="inspectionMaskApplied")


class ScorerInputTileDigest(StrictModel):
    id: Identifier
    bbox_normalized: "BboxNormalized" = Field(alias="bboxNormalized")
    current_sha256: Sha256 = Field(alias="currentSha256")
    reference_sha256: Sha256 = Field(alias="referenceSha256")


class NormalizationObservation(StrictModel):
    canonical_sha256: Sha256 | None = Field(default=None, alias="canonicalSha256")
    target_canonical_sha256: Sha256 | None = Field(default=None, alias="targetCanonicalSha256")
    canonical_image_png_base64: str | None = Field(default=None, alias="canonicalImagePngBase64", max_length=16_000_000)
    golden_canonical_sha256: Sha256 | None = Field(default=None, alias="goldenCanonicalSha256")
    golden_canonical_image_png_base64: str | None = Field(default=None, alias="goldenCanonicalImagePngBase64", max_length=16_000_000)
    canonical_width: Annotated[int, Field(ge=1, le=8192)] | None = Field(default=None, alias="canonicalWidth")
    canonical_height: Annotated[int, Field(ge=1, le=8192)] | None = Field(default=None, alias="canonicalHeight")
    inspection_roi_contract_digest: PrefixedSha256 | None = Field(default=None, alias="inspectionRoiContractDigest")
    scorer_input_sha256: Sha256 | None = Field(default=None, alias="scorerInputSha256")
    golden_scorer_input_sha256: Sha256 | None = Field(default=None, alias="goldenScorerInputSha256")
    scorer_input_tile_digests: list[ScorerInputTileDigest] | None = Field(
        default=None, alias="scorerInputTileDigests", min_length=1, max_length=256,
    )
    analyzed_region_normalized: BboxNormalized | None = Field(default=None, alias="analyzedRegionNormalized")
    evidence_coordinate_space: Literal["TARGET_CANONICAL_IMAGE"] | None = Field(
        default=None, alias="evidenceCoordinateSpace",
    )
    alignment: AlignmentObservation | None = None


class BboxNormalized(StrictModel):
    x: Annotated[float, Field(ge=0, le=1)]
    y: Annotated[float, Field(ge=0, le=1)]
    width: Annotated[float, Field(gt=0, le=1)]
    height: Annotated[float, Field(gt=0, le=1)]

    def model_post_init(self, __context: object) -> None:
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("bboxNormalized exceeds canonical bounds")


class CandidateVerification(StrictModel):
    state: Literal["EVALUATED"]
    method: Literal["DINO_CROP_COSINE_V1"]
    mode: Literal["SHADOW", "GATE"]
    priority: Literal["HIGH", "REVIEW", "LOW"]
    crop_distance: Annotated[float, Field(alias="cropDistance", ge=0, le=2)]
    disclaimer_code: Literal["CANDIDATE_VERIFICATION_NOT_DEFECT_PROOF"] = Field(alias="disclaimerCode")


class CandidateVerificationV2(StrictModel):
    state: Literal["EVALUATED"]
    method: Literal["DINO_CROP_COSINE_LOCAL_STRUCTURE_V2"]
    mode: Literal["SHADOW", "GATE"]
    priority: Literal["HIGH", "REVIEW", "LOW"]
    crop_distance: Annotated[float, Field(alias="cropDistance", ge=0, le=2)]
    local_alignment_state: Literal["ALIGNED", "UNQUALIFIED"] = Field(alias="localAlignmentState")
    local_alignment_method: Literal["GRADIENT_ECC_TRANSLATION_V1"] = Field(alias="localAlignmentMethod")
    local_alignment_correlation: Annotated[float, Field(alias="localAlignmentCorrelation", ge=-1, le=1)]
    local_translation_x_px: Annotated[float, Field(alias="localTranslationXPx", ge=-1024, le=1024)]
    local_translation_y_px: Annotated[float, Field(alias="localTranslationYPx", ge=-1024, le=1024)]
    photometric_normalization: Literal["OPENCV_LAB_CONTEXT_MEDIAN_MAD_V1"] = Field(
        alias="photometricNormalization",
    )
    appearance_changed_area_ratio: Annotated[
        float | None, Field(alias="appearanceChangedAreaRatio", ge=0, le=1),
    ] = None
    edge_changed_area_ratio: Annotated[
        float | None, Field(alias="edgeChangedAreaRatio", ge=0, le=1),
    ] = None
    structure_confirmation: Literal[
        "CONFIRMED", "UNCONFIRMED", "LOCAL_ALIGNMENT_UNQUALIFIED",
    ] = Field(alias="structureConfirmation")
    disclaimer_code: Literal["CANDIDATE_VERIFICATION_NOT_DEFECT_PROOF"] = Field(alias="disclaimerCode")

    def model_post_init(self, __context: object) -> None:
        ratios = (self.appearance_changed_area_ratio, self.edge_changed_area_ratio)
        if self.local_alignment_state == "ALIGNED":
            if any(value is None for value in ratios):
                raise ValueError("aligned candidate verification requires changed-area ratios")
            if self.structure_confirmation == "LOCAL_ALIGNMENT_UNQUALIFIED":
                raise ValueError("aligned candidate cannot be local-alignment-unqualified")
        else:
            if any(value is not None for value in ratios):
                raise ValueError("unqualified local alignment cannot report changed-area ratios")
            if self.structure_confirmation != "LOCAL_ALIGNMENT_UNQUALIFIED":
                raise ValueError("unqualified local alignment must not confirm candidate structure")


class DifferenceRegion(StrictModel):
    id: Identifier
    bbox_normalized: BboxNormalized = Field(alias="bboxNormalized")
    peak_score: Annotated[float, Field(ge=0, le=1)] = Field(alias="peakScore")
    mean_score: Annotated[float, Field(ge=0, le=1)] = Field(alias="meanScore")
    kind: Literal["SUBJECT_INTERIOR", "SUBJECT_BOUNDARY"] | None = None
    verification: CandidateVerification | CandidateVerificationV2 | None = None


class CandidateFilter(StrictModel):
    raw_component_count: Annotated[int, Field(alias="rawComponentCount", ge=0)]
    retained_component_count: Annotated[int, Field(alias="retainedComponentCount", ge=0)]
    suppressed_by_background_count: Annotated[int, Field(alias="suppressedByBackgroundCount", ge=0)] = 0
    suppressed_small_region_count: Annotated[int, Field(alias="suppressedSmallRegionCount", ge=0)]
    suppressed_by_limit_count: Annotated[int, Field(alias="suppressedByLimitCount", ge=0)]
    suppressed_by_verifier_count: Annotated[int, Field(alias="suppressedByVerifierCount", ge=0)] = 0
    mask_semantics: Literal["RETAINED_CANDIDATES"] = Field(alias="maskSemantics")

    def model_post_init(self, __context: object) -> None:
        classified = (
            self.retained_component_count
            + self.suppressed_by_background_count
            + self.suppressed_small_region_count
            + self.suppressed_by_limit_count
            + self.suppressed_by_verifier_count
        )
        if classified != self.raw_component_count:
            raise ValueError("candidate filter counts must account for every raw component")


class SubjectSegmentationEvidence(StrictModel):
    """Golden and optional Current subject scope; never defect proof."""

    state: Literal["AVAILABLE", "UNAVAILABLE"]
    disclaimer_code: Literal["SUBJECT_MASK_NOT_DEFECT_PROOF"] = Field(alias="disclaimerCode")
    reason_code: str | None = Field(default=None, alias="reasonCode", max_length=160)
    method: Literal["MOBILE_SAM_VIT_T_BOX_PROMPT"] | None = None
    usage_mode: Literal["SPATIAL_GATE"] | None = Field(default=None, alias="usageMode")
    golden_id: Identifier | None = Field(default=None, alias="goldenId")
    model_repository_version: PrefixedSha256 | None = Field(default=None, alias="modelRepositoryVersion")
    model_weights_sha256: PrefixedSha256 | None = Field(default=None, alias="modelWeightsSha256")
    prompt_region_normalized: BboxNormalized | None = Field(default=None, alias="promptRegionNormalized")
    foreground_ratio: Annotated[float, Field(ge=0, le=1)] | None = Field(default=None, alias="foregroundRatio")
    support_ratio: Annotated[float, Field(ge=0, le=1)] | None = Field(default=None, alias="supportRatio")
    background_suppressed_ratio: Annotated[float, Field(ge=0, le=1)] | None = Field(
        default=None, alias="backgroundSuppressedRatio",
    )
    subject_mask_png_base64: str | None = Field(default=None, alias="subjectMaskPngBase64", max_length=4_000_000)
    subject_mask_sha256: Sha256 | None = Field(default=None, alias="subjectMaskSha256")
    support_mask_png_base64: str | None = Field(default=None, alias="supportMaskPngBase64", max_length=4_000_000)
    support_mask_sha256: Sha256 | None = Field(default=None, alias="supportMaskSha256")
    boundary_mask_png_base64: str | None = Field(default=None, alias="boundaryMaskPngBase64", max_length=4_000_000)
    boundary_mask_sha256: Sha256 | None = Field(default=None, alias="boundaryMaskSha256")
    current_foreground_ratio: Annotated[float, Field(ge=0, le=1)] | None = Field(
        default=None, alias="currentForegroundRatio",
    )
    mask_intersection_over_union: Annotated[float, Field(ge=0, le=1)] | None = Field(
        default=None, alias="maskIntersectionOverUnion",
    )
    interior_ratio: Annotated[float, Field(ge=0, le=1)] | None = Field(default=None, alias="interiorRatio")
    current_subject_mask_png_base64: str | None = Field(
        default=None, alias="currentSubjectMaskPngBase64", max_length=4_000_000,
    )
    current_subject_mask_sha256: Sha256 | None = Field(default=None, alias="currentSubjectMaskSha256")
    interior_mask_png_base64: str | None = Field(default=None, alias="interiorMaskPngBase64", max_length=4_000_000)
    interior_mask_sha256: Sha256 | None = Field(default=None, alias="interiorMaskSha256")

    def model_post_init(self, __context: object) -> None:
        generation_fields = (
            self.method, self.usage_mode, self.golden_id, self.model_repository_version,
            self.model_weights_sha256, self.prompt_region_normalized, self.foreground_ratio,
            self.support_ratio, self.background_suppressed_ratio, self.subject_mask_png_base64,
            self.subject_mask_sha256, self.support_mask_png_base64, self.support_mask_sha256,
            self.boundary_mask_png_base64, self.boundary_mask_sha256,
        )
        paired_fields = (
            self.current_foreground_ratio, self.mask_intersection_over_union, self.interior_ratio,
            self.current_subject_mask_png_base64, self.current_subject_mask_sha256,
            self.interior_mask_png_base64, self.interior_mask_sha256,
        )
        if self.state == "UNAVAILABLE":
            if any(value is not None for value in generation_fields + paired_fields):
                raise ValueError("UNAVAILABLE subject segmentation evidence must not include generation data")
            if not self.reason_code:
                raise ValueError("UNAVAILABLE subject segmentation evidence requires reasonCode")
        else:
            if any(value is None for value in generation_fields):
                raise ValueError("AVAILABLE subject segmentation evidence requires complete generation data")
            if self.reason_code is not None:
                raise ValueError("AVAILABLE subject segmentation evidence must not include reasonCode")
            if any(value is not None for value in paired_fields) and any(value is None for value in paired_fields):
                raise ValueError("paired subject segmentation evidence requires complete Current and interior data")


class BoundaryDifferenceRegion(StrictModel):
    id: Identifier
    bbox_normalized: BboxNormalized = Field(alias="bboxNormalized")
    change_type: Literal["MISSING_FROM_CURRENT", "PROTRUDING_FROM_CURRENT"] = Field(alias="changeType")
    area_ratio: Annotated[float, Field(alias="areaRatio", gt=0, le=1)]


class BoundaryDifferenceEvidence(StrictModel):
    """Mask geometry change kept separate from DINO interior evidence."""

    state: Literal["AVAILABLE", "UNAVAILABLE"]
    disclaimer_code: Literal["BOUNDARY_GEOMETRY_NOT_DEFECT_PROOF"] = Field(alias="disclaimerCode")
    reason_code: str | None = Field(default=None, alias="reasonCode", max_length=160)
    method: Literal["ALIGNED_SUBJECT_MASK_GEOMETRY_V1"] | None = None
    mask_intersection_over_union: Annotated[float, Field(ge=0, le=1)] | None = Field(
        default=None, alias="maskIntersectionOverUnion",
    )
    area_delta_ratio: Annotated[float, Field(ge=0, le=1)] | None = Field(default=None, alias="areaDeltaRatio")
    missing_area_ratio: Annotated[float, Field(ge=0, le=1)] | None = Field(default=None, alias="missingAreaRatio")
    protruding_area_ratio: Annotated[float, Field(ge=0, le=1)] | None = Field(
        default=None, alias="protrudingAreaRatio",
    )
    mean_contour_distance_px: Annotated[float, Field(ge=0)] | None = Field(
        default=None, alias="meanContourDistancePx",
    )
    p95_contour_distance_px: Annotated[float, Field(ge=0)] | None = Field(
        default=None, alias="p95ContourDistancePx",
    )
    max_contour_distance_px: Annotated[float, Field(ge=0)] | None = Field(
        default=None, alias="maxContourDistancePx",
    )
    regions: list[BoundaryDifferenceRegion] | None = Field(default=None, max_length=64)
    mask_png_base64: str | None = Field(default=None, alias="maskPngBase64", max_length=4_000_000)
    mask_sha256: Sha256 | None = Field(default=None, alias="maskSha256")

    def model_post_init(self, __context: object) -> None:
        available = (
            self.method, self.mask_intersection_over_union, self.area_delta_ratio,
            self.missing_area_ratio, self.protruding_area_ratio, self.mean_contour_distance_px,
            self.p95_contour_distance_px, self.max_contour_distance_px, self.regions,
            self.mask_png_base64, self.mask_sha256,
        )
        if self.state == "UNAVAILABLE":
            if any(value is not None for value in available):
                raise ValueError("UNAVAILABLE boundary evidence must not include geometry data")
            if not self.reason_code:
                raise ValueError("UNAVAILABLE boundary evidence requires reasonCode")
        else:
            if any(value is None for value in available):
                raise ValueError("AVAILABLE boundary evidence requires complete geometry data")
            if self.reason_code is not None:
                raise ValueError("AVAILABLE boundary evidence must not include reasonCode")


class MetricCalibrationEvidence(StrictModel):
    source: Literal["CHARUCO_BOARD_PLANE_V1"]
    detected_corner_count: Annotated[int, Field(alias="detectedCornerCount", ge=4, le=10000)]
    inlier_corner_count: Annotated[int, Field(alias="inlierCornerCount", ge=4, le=10000)]
    plane_reprojection_error_px: Annotated[
        float, Field(alias="planeReprojectionErrorPx", ge=0, le=1000),
    ]
    pixels_per_mm_x: Annotated[float, Field(alias="pixelsPerMmX", gt=0, le=10000)]
    pixels_per_mm_y: Annotated[float, Field(alias="pixelsPerMmY", gt=0, le=10000)]

    def model_post_init(self, __context: object) -> None:
        if self.inlier_corner_count > self.detected_corner_count:
            raise ValueError("calibration inlier corners cannot exceed detected corners")


class DimensionUncertaintyEvidence(StrictModel):
    method: Literal["CONSERVATIVE_CALIBRATION_PLUS_SEGMENTATION_V1"]
    linear_mm: Annotated[float, Field(alias="linearMm", gt=0, le=10000)]
    area_mm2: Annotated[float, Field(alias="areaMm2", gt=0, le=100_000_000)]
    relative_linear: Annotated[float, Field(alias="relativeLinear", gt=0, le=1)]


class PhysicalDimensionEvidence(StrictModel):
    """Planar dimensions from this capture's ChArUco scale and Current mask."""

    state: Literal["AVAILABLE", "UNAVAILABLE"]
    disclaimer_code: Literal["ENGINEERING_DIMENSION_NOT_METROLOGY_PROOF"] = Field(alias="disclaimerCode")
    reason_code: str | None = Field(default=None, alias="reasonCode", max_length=160)
    method: Literal[
        "CHARUCO_PLANE_CURRENT_MASK_MIN_AREA_RECT_V1",
        "CHARUCO_PLANE_GOLDEN_MASK_MIN_AREA_RECT_V1",
    ] | None = None
    approval_state: Literal["ENGINEERING_AUTO", "APPROVED"] | None = Field(
        default=None, alias="approvalState",
    )
    coordinate_space: Literal["CHARUCO_BOARD_PLANE_MM"] | None = Field(
        default=None, alias="coordinateSpace",
    )
    current_subject_mask_sha256: Sha256 | None = Field(default=None, alias="currentSubjectMaskSha256")
    length_mm: Annotated[float | None, Field(alias="lengthMm", gt=0, le=100000)] = None
    width_mm: Annotated[float | None, Field(alias="widthMm", gt=0, le=100000)] = None
    area_mm2: Annotated[float | None, Field(alias="areaMm2", gt=0, le=100_000_000)] = None
    rotated_rect_angle_degrees: Annotated[
        float | None, Field(alias="rotatedRectAngleDegrees", ge=-180, le=180),
    ] = None
    calibration: MetricCalibrationEvidence | None = None
    uncertainty: DimensionUncertaintyEvidence | None = None

    def model_post_init(self, __context: object) -> None:
        values = (
            self.method, self.approval_state, self.coordinate_space, self.current_subject_mask_sha256,
            self.length_mm, self.width_mm, self.area_mm2, self.rotated_rect_angle_degrees,
            self.calibration, self.uncertainty,
        )
        if self.state == "UNAVAILABLE":
            if any(value is not None for value in values):
                raise ValueError("UNAVAILABLE physical dimensions must not include metric values")
            if not self.reason_code:
                raise ValueError("UNAVAILABLE physical dimensions require reasonCode")
        else:
            if any(value is None for value in values):
                raise ValueError("AVAILABLE physical dimensions require complete metric evidence")
            if self.reason_code is not None:
                raise ValueError("AVAILABLE physical dimensions must not include reasonCode")
            if self.length_mm is not None and self.width_mm is not None and self.length_mm < self.width_mm:
                raise ValueError("physical dimension length must be the long side")


class GoldenDimensionRequest(StrictModel):
    """Server-owned request to establish one Golden planar dimension baseline."""

    schema_version: Literal["1.0"] = Field(alias="schemaVersion")
    request_id: Identifier = Field(alias="requestId")
    recipe_id: Identifier = Field(alias="recipeId")
    raw_sha256: Sha256 = Field(alias="rawSha256")
    content_type: Literal["image/jpeg", "image/png"] = Field(alias="contentType")
    prompt_region_normalized: BboxNormalized = Field(alias="promptRegionNormalized")
    measurement_plane: Literal["TOP", "FRONT", "SIDE"] = Field(alias="measurementPlane")
    board_candidates: Annotated[
        list[GoldenDimensionBoardCandidate], Field(alias="boardCandidates", max_length=8),
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_board_candidates(self) -> "GoldenDimensionRequest":
        profiles = [candidate.profile for candidate in self.board_candidates]
        if len(profiles) != len(set(profiles)):
            raise ValueError("boardCandidates must contain at most one revision per profile")
        return self


class GoldenDimensionObservation(StrictModel):
    """Engineering measurement evidence only; never a product disposition."""

    schema_version: Literal["1.0"] = Field(alias="schemaVersion")
    request_id: Identifier = Field(alias="requestId")
    recipe_id: Identifier = Field(alias="recipeId")
    raw_sha256: Sha256 = Field(alias="rawSha256")
    artifact_package_digest: PrefixedSha256 = Field(alias="artifactPackageDigest")
    analyzer_runtime_version: PrefixedSha256 = Field(alias="analyzerRuntimeVersion")
    calibration_board_profile: Identifier = Field(alias="calibrationBoardProfile")
    calibration_board_id: Identifier | None = Field(default=None, alias="calibrationBoardId")
    calibration_board_revision: int | None = Field(default=None, alias="calibrationBoardRevision", ge=1)
    calibration_board_manifest_sha256: PrefixedSha256 | None = Field(
        default=None, alias="calibrationBoardManifestSha256",
    )
    measurement_plane: Literal["TOP", "FRONT", "SIDE"] = Field(alias="measurementPlane")
    physical_dimensions: PhysicalDimensionEvidence = Field(alias="physicalDimensions")
    subject_mask_png_base64: str | None = Field(
        default=None, alias="subjectMaskPngBase64", max_length=4_000_000,
    )


class SpatialDifferenceEvidence(StrictModel):
    """Patch-level DINO difference evidence. UNAVAILABLE is honest-by-default:

    it is returned whenever the nearest Golden lacks patch features or no
    spatial policy is pinned, rather than fabricating a map/mask.
    """

    state: Literal["AVAILABLE", "UNAVAILABLE"]
    disclaimer_code: Literal["DIFFERENCE_NOT_DEFECT_PROOF"] = Field(alias="disclaimerCode")
    reason_code: str | None = Field(default=None, alias="reasonCode", max_length=160)
    generation_method: Literal[
        "PATCH_DISTANCE", "ROI_TILED_PATCH_DISTANCE", "SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE",
        "PAIRED_INTERIOR_ROI_TILED_PATCH_DISTANCE",
    ] | None = Field(
        default=None, alias="generationMethod",
    )
    # The map/mask are native-resolution patch-grid renders of exactly what DINO
    # saw (post Resize+CenterCrop), not a resample of the full canonical ROI.
    # This region locates that analyzed rectangle within canonical-ROI
    # normalized coordinates so a consumer never overlays evidence onto pixels
    # the model never examined.
    evidence_region_normalized: BboxNormalized | None = Field(default=None, alias="evidenceRegionNormalized")
    evidence_coordinate_space: Literal["TARGET_CANONICAL_IMAGE"] | None = Field(
        default=None, alias="evidenceCoordinateSpace",
    )
    scorer_input_tile_digests: list[ScorerInputTileDigest] | None = Field(
        default=None, alias="scorerInputTileDigests", min_length=1, max_length=256,
    )
    regions: list[DifferenceRegion] | None = Field(default=None, max_length=64)
    map_png_base64: str | None = Field(default=None, alias="mapPngBase64", max_length=4_000_000)
    mask_png_base64: str | None = Field(default=None, alias="maskPngBase64", max_length=4_000_000)
    map_sha256: Sha256 | None = Field(default=None, alias="mapSha256")
    mask_sha256: Sha256 | None = Field(default=None, alias="maskSha256")
    raw_threshold_mask_png_base64: str | None = Field(
        default=None, alias="rawThresholdMaskPngBase64", max_length=4_000_000,
    )
    raw_threshold_mask_sha256: Sha256 | None = Field(default=None, alias="rawThresholdMaskSha256")
    candidate_filter: CandidateFilter | None = Field(default=None, alias="candidateFilter")

    def model_post_init(self, __context: object) -> None:
        if self.state == "UNAVAILABLE":
            available_fields = (
                self.generation_method, self.regions, self.map_png_base64, self.mask_png_base64,
                self.evidence_region_normalized, self.map_sha256, self.mask_sha256,
                self.raw_threshold_mask_png_base64, self.raw_threshold_mask_sha256, self.candidate_filter,
                self.evidence_coordinate_space, self.scorer_input_tile_digests,
            )
            if any(field is not None for field in available_fields):
                raise ValueError("UNAVAILABLE spatial difference evidence must not include generation data")
            if not self.reason_code:
                raise ValueError("UNAVAILABLE spatial difference evidence requires reasonCode")
        else:
            required_fields = (
                self.generation_method, self.regions, self.map_png_base64, self.mask_png_base64,
                self.evidence_region_normalized, self.map_sha256, self.mask_sha256,
            )
            if any(field is None for field in required_fields):
                raise ValueError("AVAILABLE spatial difference evidence requires generation method, region, map, and mask")
            if self.reason_code is not None:
                raise ValueError("AVAILABLE spatial difference evidence must not include reasonCode")
            extended = (
                self.raw_threshold_mask_png_base64,
                self.raw_threshold_mask_sha256,
                self.candidate_filter,
            )
            if self.generation_method in {
                "SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE", "PAIRED_INTERIOR_ROI_TILED_PATCH_DISTANCE",
            }:
                if any(value is None for value in extended):
                    raise ValueError("subject-gated evidence requires raw mask and candidate filter")
            elif any(value is not None for value in extended):
                raise ValueError("legacy spatial evidence cannot include subject-gated fields")


class AnalysisObservation(StrictModel):
    state: AnalysisState
    metric: Literal["cosine_distance"] | None = None
    global_distance: Annotated[float, Field(ge=0, le=2)] | None = Field(default=None, alias="globalDistance")
    nearest_golden_id: str | None = Field(default=None, alias="nearestGoldenId", max_length=160)
    nearest_golden_distance: Annotated[float, Field(ge=0, le=2)] | None = Field(default=None, alias="nearestGoldenDistance")
    uncertainty: Annotated[float, Field(ge=0, le=1)] | None = None
    scoring_scope: Literal["INSPECTION_ROI_ONLY"] | None = Field(default=None, alias="scoringScope")
    spatial_difference_evidence: SpatialDifferenceEvidence | None = Field(default=None, alias="spatialDifferenceEvidence")
    subject_segmentation_evidence: SubjectSegmentationEvidence | None = Field(
        default=None, alias="subjectSegmentationEvidence",
    )
    boundary_difference_evidence: BoundaryDifferenceEvidence | None = Field(
        default=None, alias="boundaryDifferenceEvidence",
    )
    physical_dimension_evidence: PhysicalDimensionEvidence | None = Field(
        default=None, alias="physicalDimensionEvidence",
    )


class ResolvedVersions(StrictModel):
    execution_bundle_digest: PrefixedSha256 = Field(alias="executionBundleDigest")
    artifact_package_digest: PrefixedSha256 = Field(alias="artifactPackageDigest")
    analyzer_model_version: PrefixedSha256 = Field(alias="analyzerModelVersion")
    analyzer_runtime_version: PrefixedSha256 = Field(alias="analyzerRuntimeVersion")
    normalization_runtime_version: PrefixedSha256 = Field(alias="normalizationRuntimeVersion")
    recipe_analysis_profile_digest: PrefixedSha256 | None = Field(default=None, alias="recipeAnalysisProfileDigest")
    scorer_input_contract_digest: PrefixedSha256 | None = Field(default=None, alias="scorerInputContractDigest")


class AnalyzeObservation(StrictModel):
    schema_version: Literal["1.0", "1.1", "1.2", "1.3", "1.4"] = Field(alias="schemaVersion")
    request_id: Identifier = Field(alias="requestId")
    analysis_id: Sha256 = Field(alias="analysisId")
    raw_sha256: Sha256 = Field(alias="rawSha256")
    simulation: bool
    resolved_versions: ResolvedVersions = Field(alias="resolvedVersions")
    capture_assessment: CaptureAssessment = Field(alias="captureAssessment")
    normalization: NormalizationObservation | None = None
    analysis: AnalysisObservation
