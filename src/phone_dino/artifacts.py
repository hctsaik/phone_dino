from __future__ import annotations

import hashlib
import hmac
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageChops, ImageDraw, UnidentifiedImageError

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contracts import AnalyzeRequest, Identifier, PrefixedSha256
from .security import digest_file


class ArtifactError(RuntimeError):
    pass


class StrictArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class CharucoBoard(StrictArtifactModel):
    profile_id: str | None = Field(default=None, alias="profileId", pattern=r"^[A-Z0-9_]+$")
    squares_x: int = Field(alias="squaresX", ge=3, le=100)
    squares_y: int = Field(alias="squaresY", ge=3, le=100)
    square_length_mm: float = Field(alias="squareLengthMm", gt=0, le=1000)
    marker_length_mm: float = Field(alias="markerLengthMm", gt=0, le=1000)
    marker_ids: list[int] | None = Field(default=None, alias="markerIds", min_length=1, max_length=5000)
    dictionary: str = Field(pattern=r"^DICT_[A-Z0-9_]+$")
    canonical_width: int = Field(alias="canonicalWidth", ge=224, le=8192)
    canonical_height: int = Field(alias="canonicalHeight", ge=224, le=8192)

    def model_post_init(self, __context: object) -> None:
        if self.marker_length_mm >= self.square_length_mm:
            raise ValueError("ChArUco markerLengthMm must be smaller than squareLengthMm")
        if self.marker_ids is None:
            return
        expected = self.squares_x * self.squares_y // 2
        if len(self.marker_ids) != expected or len(set(self.marker_ids)) != expected:
            raise ValueError("ChArUco markerIds must contain one unique ID per marker cell")
        if any(marker_id < 0 or marker_id > 9999 for marker_id in self.marker_ids):
            raise ValueError("ChArUco markerIds are outside the supported range")


class StillGate(StrictArtifactModel):
    min_charuco_corners: int = Field(alias="minCharucoCorners", ge=4, le=10000)
    min_laplacian_variance: float = Field(alias="minLaplacianVariance", ge=0)
    max_over_exposure_ratio: float = Field(alias="maxOverExposureRatio", ge=0, le=1)


class ImageRegion(StrictArtifactModel):
    x: int = Field(ge=0, le=8191)
    y: int = Field(ge=0, le=8191)
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)


class ScorerInputContract(StrictArtifactModel):
    """Immutable definition of the pixels allowed to affect DINO scores."""

    schema_version: Literal["1.0", "1.1"] = Field(alias="schemaVersion")
    policy: Literal[
        "INSPECTION_ROI_SUBJECT_SUPPORT_TILES_NEUTRAL_OUTSIDE",
        "INSPECTION_ROI_PAIRED_INTERIOR_TILES_NEUTRAL_OUTSIDE",
    ]
    coordinate_space: Literal["TARGET_CANONICAL_IMAGE"] = Field(alias="coordinateSpace")
    inspection_roi_contract_digest: PrefixedSha256 = Field(alias="inspectionRoiContractDigest")
    tile_order: Literal["TOP_TO_BOTTOM_LEFT_TO_RIGHT"] = Field(alias="tileOrder")
    neutral_rgb: list[int] = Field(alias="neutralRgb", min_length=3, max_length=3)
    digest: PrefixedSha256

    def model_post_init(self, __context: object) -> None:
        if any(channel < 0 or channel > 255 for channel in self.neutral_rgb):
            raise ValueError("scorer input neutral RGB channels must be bytes")
        expected_policy = (
            "INSPECTION_ROI_SUBJECT_SUPPORT_TILES_NEUTRAL_OUTSIDE"
            if self.schema_version == "1.0"
            else "INSPECTION_ROI_PAIRED_INTERIOR_TILES_NEUTRAL_OUTSIDE"
        )
        if self.policy != expected_policy:
            raise ValueError("scorer input policy does not match schema version")
        payload = self.model_dump(by_alias=True, mode="json", exclude={"digest"})
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if not hmac.compare_digest(expected, self.digest):
            raise ValueError("scorer input contract digest does not match policy")


class ReferenceRoleBinding(StrictArtifactModel):
    role: Literal["ALIGNMENT_TEMPLATE", "TARGET_REFERENCE", "NORMAL_REFERENCE_SET", "DISPLAY_REFERENCE"]
    id: Identifier
    version: Identifier
    source_digest: PrefixedSha256 = Field(alias="sourceDigest")
    digest: PrefixedSha256

    def model_post_init(self, __context: object) -> None:
        payload = self.model_dump(by_alias=True, mode="json", exclude={"digest"})
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if not hmac.compare_digest(expected, self.digest):
            raise ValueError("reference role digest does not match its immutable binding")


class RecipeAnalysisProfile(StrictArtifactModel):
    schema_version: Literal["1.0"] = Field(alias="schemaVersion")
    id: Identifier
    version: Identifier
    alignment_template: ReferenceRoleBinding = Field(alias="alignmentTemplate")
    target_reference: ReferenceRoleBinding = Field(alias="targetReference")
    normal_reference_set: ReferenceRoleBinding = Field(alias="normalReferenceSet")
    display_reference: ReferenceRoleBinding = Field(alias="displayReference")
    inspection_roi_contract_digest: PrefixedSha256 = Field(alias="inspectionRoiContractDigest")
    scorer_input_contract_digest: PrefixedSha256 = Field(alias="scorerInputContractDigest")
    digest: PrefixedSha256

    def model_post_init(self, __context: object) -> None:
        bindings = (
            (self.alignment_template, "ALIGNMENT_TEMPLATE"),
            (self.target_reference, "TARGET_REFERENCE"),
            (self.normal_reference_set, "NORMAL_REFERENCE_SET"),
            (self.display_reference, "DISPLAY_REFERENCE"),
        )
        if any(binding.role != expected for binding, expected in bindings):
            raise ValueError("recipe analysis profile reference roles are not distinct and explicit")
        if len({binding.digest for binding, _ in bindings}) != len(bindings):
            raise ValueError("recipe analysis profile role digests must be distinct")
        payload = self.model_dump(by_alias=True, mode="json", exclude={"digest"})
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if not hmac.compare_digest(expected, self.digest):
            raise ValueError("recipe analysis profile digest does not match its bindings")


class CanonicalPolygon(StrictArtifactModel):
    """A closed inspection boundary in canonical pixel coordinates."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)


class InspectionRoiContract(StrictArtifactModel):
    """Immutable ROI contract; kept separate so schema 1.1 remains readable."""

    version: str = Field(pattern=r"^roi-[0-9]+\.[0-9]+$")
    canonical_width: int = Field(alias="canonicalWidth", ge=224, le=8192)
    canonical_height: int = Field(alias="canonicalHeight", ge=224, le=8192)
    polygon: list[CanonicalPolygon] = Field(min_length=3, max_length=128)
    inspection_regions: list[ImageRegion] = Field(alias="inspectionRegions", min_length=1, max_length=64)
    digest: PrefixedSha256

    def model_post_init(self, __context: object) -> None:
        points = [(point.x, point.y) for point in self.polygon]
        if len(set(points)) != len(points):
            raise ValueError("inspection ROI polygon vertices must be unique")
        for x, y in points:
            if x > self.canonical_width or y > self.canonical_height:
                raise ValueError("inspection ROI polygon exceeds canonical image")
        signed_area = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True)
        ) / 2.0
        if abs(signed_area) <= 1e-6:
            raise ValueError("inspection ROI polygon must have non-zero area")

        def orientation(a, b, c) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def on_segment(a, b, point) -> bool:
            return (
                min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= point[1] <= max(a[1], b[1])
            )

        edge_count = len(points)
        for left_index in range(edge_count):
            a, b = points[left_index], points[(left_index + 1) % edge_count]
            for right_index in range(left_index + 1, edge_count):
                if right_index in {left_index, (left_index + 1) % edge_count}:
                    continue
                if left_index == 0 and right_index == edge_count - 1:
                    continue
                c, d = points[right_index], points[(right_index + 1) % edge_count]
                orientations = (
                    orientation(a, b, c), orientation(a, b, d),
                    orientation(c, d, a), orientation(c, d, b),
                )
                intersects = orientations[0] * orientations[1] < 0 and orientations[2] * orientations[3] < 0
                intersects = intersects or any((
                    abs(orientations[0]) <= 1e-9 and on_segment(a, b, c),
                    abs(orientations[1]) <= 1e-9 and on_segment(a, b, d),
                    abs(orientations[2]) <= 1e-9 and on_segment(c, d, a),
                    abs(orientations[3]) <= 1e-9 and on_segment(c, d, b),
                ))
                if intersects:
                    raise ValueError("inspection ROI polygon must not self-intersect")
        regions = self.inspection_regions
        for region in regions:
            if region.x + region.width > self.canonical_width or region.y + region.height > self.canonical_height:
                raise ValueError("inspection region exceeds canonical image")
        for index, left in enumerate(regions):
            for right in regions[index + 1:]:
                if (left.x < right.x + right.width and right.x < left.x + left.width
                        and left.y < right.y + right.height and right.y < left.y + left.height):
                    raise ValueError("inspection regions must be pairwise disjoint")
        # Bind the exact geometry and bounds to the contract digest.
        payload = {
            "version": self.version,
            "canonicalWidth": self.canonical_width,
            "canonicalHeight": self.canonical_height,
            "polygon": [{"x": point.x, "y": point.y} for point in self.polygon],
            "inspectionRegions": [region.model_dump(by_alias=True) for region in regions],
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if not hmac.compare_digest(expected, self.digest):
            raise ValueError("inspection ROI digest does not match geometry")


def require_inspection_roi(artifact: "ProductionArtifact") -> InspectionRoiContract:
    """Production gate: legacy artifacts without immutable ROI fail closed."""
    roi = getattr(artifact, "inspection_roi", None)
    if roi is None:
        raise ArtifactError("INSPECTION_ROI_CONTRACT_REQUIRED")
    if roi.canonical_width != artifact.target_alignment.canonical_width or roi.canonical_height != artifact.target_alignment.canonical_height:
        raise ArtifactError("INSPECTION_ROI_CANONICAL_BOUNDS_MISMATCH")
    # The immutable ROI must remain disjoint from alignment and held-out
    # evidence.  The target policy is the authority for those regions; do not
    # trust a phone_cv candidate to redefine them at runtime.
    for inspection in roi.inspection_regions:
        for protected in (*artifact.target_alignment.alignment_regions, *artifact.target_alignment.held_out_regions):
            overlaps = (
                inspection.x < protected.x + protected.width and protected.x < inspection.x + inspection.width
                and inspection.y < protected.y + protected.height and protected.y < inspection.y + inspection.height
            )
            if overlaps:
                raise ArtifactError("INSPECTION_ROI_OVERLAPS_ALIGNMENT_OR_HELD_OUT")
    return roi


def inspection_roi_image(roi: InspectionRoiContract) -> Image.Image:
    """Rasterize the authoritative polygon intersected with tile regions."""
    polygon = Image.new("L", (roi.canonical_width, roi.canonical_height), color=0)
    ImageDraw.Draw(polygon).polygon([(point.x, point.y) for point in roi.polygon], fill=255)
    regions = Image.new("L", polygon.size, color=0)
    draw = ImageDraw.Draw(regions)
    for region in roi.inspection_regions:
        draw.rectangle(
            (region.x, region.y, region.x + region.width - 1, region.y + region.height - 1),
            fill=255,
        )
    return ImageChops.multiply(polygon, regions)


class TargetAlignmentPolicy(StrictArtifactModel):
    """Hash-bound target reference, masks, and fail-closed geometry bounds."""

    method: Literal["TARGET_AFFINE"]
    reference_image_base64: str = Field(alias="referenceImageBase64", min_length=16, max_length=16_000_000)
    reference_image_sha256: PrefixedSha256 = Field(alias="referenceImageSha256")
    canonical_width: int = Field(alias="canonicalWidth", ge=224, le=8192)
    canonical_height: int = Field(alias="canonicalHeight", ge=224, le=8192)
    alignment_regions: list[ImageRegion] = Field(alias="alignmentRegions", min_length=1, max_length=64)
    held_out_regions: list[ImageRegion] = Field(alias="heldOutRegions", min_length=1, max_length=64)
    inspection_regions: list[ImageRegion] = Field(alias="inspectionRegions", min_length=1, max_length=64)
    contour_anchor_region: ImageRegion | None = Field(default=None, alias="contourAnchorRegion")
    min_matches: int = Field(alias="minMatches", ge=6, le=10000)
    min_inliers: int = Field(alias="minInliers", ge=4, le=10000)
    min_inlier_ratio: float = Field(alias="minInlierRatio", gt=0, le=1)
    min_coverage_ratio: float = Field(alias="minCoverageRatio", gt=0, le=1)
    max_reprojection_error_px: float = Field(alias="maxReprojectionErrorPx", gt=0, le=1000)
    min_scale: float = Field(alias="minScale", gt=0, le=10)
    max_scale: float = Field(alias="maxScale", gt=0, le=10)
    max_rotation_degrees: float = Field(alias="maxRotationDegrees", ge=0, le=180)
    max_shear: float = Field(alias="maxShear", ge=0, le=1)
    max_translation_px: float = Field(alias="maxTranslationPx", ge=0, le=12000)
    max_secondary_inlier_ratio: float = Field(alias="maxSecondaryInlierRatio", ge=0, lt=1)
    min_held_out_matches: int = Field(alias="minHeldOutMatches", ge=4, le=10000)
    max_held_out_reprojection_error_px: float = Field(alias="maxHeldOutReprojectionErrorPx", gt=0, le=1000)

    def model_post_init(self, __context: object) -> None:
        if self.min_inliers > self.min_matches:
            raise ValueError("minInliers cannot exceed minMatches")
        if self.min_scale > self.max_scale:
            raise ValueError("minScale cannot exceed maxScale")
        try:
            reference = base64.b64decode(self.reference_image_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("referenceImageBase64 is invalid") from exc
        if not hmac.compare_digest("sha256:" + hashlib.sha256(reference).hexdigest(), self.reference_image_sha256):
            raise ValueError("referenceImageSha256 does not match referenceImageBase64")
        regions = (*self.alignment_regions, *self.held_out_regions, *self.inspection_regions)
        for region in regions:
            if region.x + region.width > self.canonical_width or region.y + region.height > self.canonical_height:
                raise ValueError("target alignment region exceeds canonical image")
        for index, left in enumerate(regions):
            for right in regions[index + 1:]:
                overlaps = (
                    left.x < right.x + right.width and right.x < left.x + left.width
                    and left.y < right.y + right.height and right.y < left.y + left.height
                )
                if overlaps:
                    raise ValueError("alignment, held-out, and inspection regions must be pairwise disjoint")
        if self.contour_anchor_region is not None and (
            self.contour_anchor_region.x + self.contour_anchor_region.width > self.canonical_width
            or self.contour_anchor_region.y + self.contour_anchor_region.height > self.canonical_height
        ):
            raise ValueError("contour anchor region exceeds canonical image")


class GoldenEmbedding(StrictArtifactModel):
    id: Identifier
    source_sha256: PrefixedSha256 = Field(alias="sourceSha256")
    values: list[float] = Field(min_length=2, max_length=4096)
    canonical_sha256: PrefixedSha256 | None = Field(default=None, alias="canonicalSha256")
    canonical_image_png_base64: str | None = Field(
        default=None, alias="canonicalImagePngBase64", min_length=16, max_length=16_000_000,
    )
    patch_values: list[list[float]] | None = Field(default=None, alias="patchValues", min_length=1, max_length=4096)
    patch_grid_height: int | None = Field(default=None, alias="patchGridHeight", ge=1, le=256)
    patch_grid_width: int | None = Field(default=None, alias="patchGridWidth", ge=1, le=256)
    scorer_input_sha256: PrefixedSha256 | None = Field(default=None, alias="scorerInputSha256")
    scorer_input_tiles: list["ScorerInputTileEmbedding"] | None = Field(
        default=None, alias="scorerInputTiles", min_length=1, max_length=256,
    )

    def model_post_init(self, __context: object) -> None:
        if (self.canonical_sha256 is None) != (self.canonical_image_png_base64 is None):
            raise ValueError("canonicalSha256 and canonicalImagePngBase64 must be set together")
        if self.canonical_image_png_base64 is not None:
            try:
                canonical = base64.b64decode(self.canonical_image_png_base64, validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("canonicalImagePngBase64 is invalid") from exc
            if not canonical.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("canonicalImagePngBase64 must contain PNG bytes")
            if not hmac.compare_digest("sha256:" + hashlib.sha256(canonical).hexdigest(), self.canonical_sha256):
                raise ValueError("canonicalSha256 does not match canonicalImagePngBase64")
        has_patch_fields = (self.patch_values is not None, self.patch_grid_height is not None, self.patch_grid_width is not None)
        if any(has_patch_fields) and not all(has_patch_fields):
            raise ValueError("patchValues, patchGridHeight, and patchGridWidth must be set together")
        if self.patch_values is None:
            pass
        else:
            if len(self.patch_values) != self.patch_grid_height * self.patch_grid_width:
                raise ValueError("patchValues length must equal patchGridHeight * patchGridWidth")
            for patch in self.patch_values:
                if len(patch) != len(self.values):
                    raise ValueError("every patch vector must have the same dimension as the Golden embedding")
                if not all(value == value and abs(value) != float("inf") for value in patch):
                    raise ValueError("patch vectors must be finite")
        if (self.scorer_input_sha256 is None) != (self.scorer_input_tiles is None):
            raise ValueError("scorerInputSha256 and scorerInputTiles must be set together")
        if self.scorer_input_tiles is not None:
            if len({item.id for item in self.scorer_input_tiles}) != len(self.scorer_input_tiles):
                raise ValueError("scorer input tile ids must be unique")
            dimensions = {len(item.values) for item in self.scorer_input_tiles}
            if dimensions != {len(self.values)}:
                raise ValueError("scorer input tile embedding dimensions must match the aggregate embedding")


class ScorerInputTileEmbedding(StrictArtifactModel):
    id: Identifier
    x: int = Field(ge=0, le=8191)
    y: int = Field(ge=0, le=8191)
    side: int = Field(ge=1, le=8192)
    tile_sha256: PrefixedSha256 = Field(alias="tileSha256")
    values: list[float] = Field(min_length=2, max_length=4096)

    def model_post_init(self, __context: object) -> None:
        if not all(value == value and abs(value) != float("inf") for value in self.values):
            raise ValueError("scorer input tile embeddings must be finite")
        if sum(value * value for value in self.values) ** 0.5 <= 1e-12:
            raise ValueError("scorer input tile embeddings must have non-zero norm")


class SpatialDifferencePolicy(StrictArtifactModel):
    """Per-Recipe, per-Model anomaly map/mask thresholds; never hardcoded in service code."""

    anomaly_distance_threshold: float = Field(alias="anomalyDistanceThreshold", ge=0, le=2)
    min_region_area_ratio: float = Field(alias="minRegionAreaRatio", ge=0, lt=1)
    max_regions: int = Field(alias="maxRegions", ge=1, le=64)


class SubjectAlignmentContract(StrictArtifactModel):
    """Immutable Golden-subject fallback used only after target ORB fails."""

    version: str = Field(pattern=r"^subject-align-[0-9]+\.[0-9]+$")
    method: Literal["SUBJECT_CONTOUR_ECC_AFFINE"]
    approval_state: Literal["ENGINEERING_AUTO", "APPROVED"] = Field(alias="approvalState")
    mask_source: Literal["GOLDEN_SUBJECT_MASK"] = Field(alias="maskSource")
    alignment_band_px: int = Field(alias="alignmentBandPx", ge=2, le=256)
    held_out_block_px: int = Field(alias="heldOutBlockPx", ge=8, le=512)
    max_iterations: int = Field(alias="maxIterations", ge=10, le=1000)
    convergence_epsilon: float = Field(alias="convergenceEpsilon", gt=0, le=0.1)
    min_ecc_correlation: float = Field(alias="minEccCorrelation", ge=0, le=1)
    max_held_out_residual_px: float = Field(alias="maxHeldOutResidualPx", gt=0, le=100)
    min_held_out_coverage_ratio: float = Field(alias="minHeldOutCoverageRatio", gt=0, le=1)
    max_residual_translation_px: float = Field(alias="maxResidualTranslationPx", ge=0, le=1000)
    max_residual_rotation_degrees: float = Field(alias="maxResidualRotationDegrees", ge=0, le=45)
    max_residual_scale_delta: float = Field(alias="maxResidualScaleDelta", ge=0, le=1)
    max_residual_shear: float = Field(alias="maxResidualShear", ge=0, le=1)


class CandidateVerificationPolicy(StrictArtifactModel):
    """Artifact-bound DINO crop verifier; priority is never a defect decision."""

    version: str = Field(pattern=r"^candidate-verify-[0-9]+\.[0-9]+$")
    method: Literal["DINO_CROP_COSINE_V1"]
    mode: Literal["SHADOW", "GATE"]
    approval_state: Literal["ENGINEERING_AUTO", "APPROVED"] = Field(alias="approvalState")
    context_padding_ratio: float = Field(alias="contextPaddingRatio", ge=0, le=2)
    minimum_crop_side_px: int = Field(alias="minimumCropSidePx", ge=64, le=1024)
    max_candidates: int = Field(alias="maxCandidates", ge=1, le=64)
    review_priority_distance: float = Field(alias="reviewPriorityDistance", ge=0, le=2)
    high_priority_distance: float = Field(alias="highPriorityDistance", ge=0, le=2)

    def model_post_init(self, __context: object) -> None:
        if self.review_priority_distance >= self.high_priority_distance:
            raise ValueError("reviewPriorityDistance must be less than highPriorityDistance")
        if self.mode == "GATE" and self.approval_state != "APPROVED":
            raise ValueError("candidate verification GATE mode requires APPROVED policy")


class CandidateVerificationPolicyV2(CandidateVerificationPolicy):
    """DINO priority plus bounded local appearance/edge confirmation."""

    method: Literal["DINO_CROP_COSINE_LOCAL_STRUCTURE_V2"]
    local_alignment_method: Literal["GRADIENT_ECC_TRANSLATION_V1"] = Field(alias="localAlignmentMethod")
    photometric_normalization: Literal["OPENCV_LAB_CONTEXT_MEDIAN_MAD_V1"] = Field(
        alias="photometricNormalization",
    )
    structure_method: Literal["LAB_DELTA_OR_CANNY_EDGE_V1"] = Field(alias="structureMethod")
    max_local_translation_px: float = Field(alias="maxLocalTranslationPx", gt=0, le=64)
    min_local_alignment_correlation: float = Field(alias="minLocalAlignmentCorrelation", ge=0, le=1)
    candidate_exclusion_padding_px: int = Field(alias="candidateExclusionPaddingPx", ge=0, le=64)
    minimum_context_pixels: int = Field(alias="minimumContextPixels", ge=64, le=1_000_000)
    appearance_delta_threshold: float = Field(alias="appearanceDeltaThreshold", gt=0, le=2)
    min_appearance_changed_area_ratio: float = Field(alias="minAppearanceChangedAreaRatio", gt=0, le=1)
    min_edge_changed_area_ratio: float = Field(alias="minEdgeChangedAreaRatio", gt=0, le=1)


class GoldenSubjectMask(StrictArtifactModel):
    golden_id: Identifier = Field(alias="goldenId")
    canonical_sha256: PrefixedSha256 = Field(alias="canonicalSha256")
    mask_png_base64: str = Field(alias="maskPngBase64", min_length=16, max_length=4_000_000)
    mask_sha256: PrefixedSha256 = Field(alias="maskSha256")
    prompt_region_normalized: dict[str, float] = Field(alias="promptRegionNormalized")
    model_quality_score: float = Field(alias="modelQualityScore", ge=0, le=1)
    foreground_ratio: float = Field(alias="foregroundRatio", gt=0, lt=1)

    def model_post_init(self, __context: object) -> None:
        expected_keys = {"x", "y", "width", "height"}
        if set(self.prompt_region_normalized) != expected_keys:
            raise ValueError("promptRegionNormalized must contain x, y, width, and height")
        x, y = self.prompt_region_normalized["x"], self.prompt_region_normalized["y"]
        width, height = self.prompt_region_normalized["width"], self.prompt_region_normalized["height"]
        if not all(isinstance(value, float) and value == value for value in (x, y, width, height)):
            raise ValueError("promptRegionNormalized values must be finite floats")
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("promptRegionNormalized exceeds canonical bounds")
        try:
            mask = base64.b64decode(self.mask_png_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("maskPngBase64 is invalid") from exc
        if not hmac.compare_digest("sha256:" + hashlib.sha256(mask).hexdigest(), self.mask_sha256):
            raise ValueError("maskSha256 does not match maskPngBase64")


class SubjectSegmentationContract(StrictArtifactModel):
    version: str = Field(pattern=r"^subject-[0-9]+\.[0-9]+$")
    method: Literal["MOBILE_SAM_VIT_T_BOX_PROMPT"]
    usage_mode: Literal["SPATIAL_GATE"] = Field(alias="usageMode")
    approval_state: Literal["ENGINEERING_AUTO", "APPROVED"] = Field(alias="approvalState")
    prompt_policy: Literal["INSPECTION_ROI_BOUNDING_BOX_V1"] = Field(alias="promptPolicy")
    canonical_width: int = Field(alias="canonicalWidth", ge=224, le=8192)
    canonical_height: int = Field(alias="canonicalHeight", ge=224, le=8192)
    model_repository_version: PrefixedSha256 = Field(alias="modelRepositoryVersion")
    model_weights_sha256: PrefixedSha256 = Field(alias="modelWeightsSha256")
    min_model_quality_score: float = Field(alias="minModelQualityScore", ge=0, le=1)
    min_foreground_ratio: float = Field(alias="minForegroundRatio", gt=0, lt=1)
    max_foreground_ratio: float = Field(alias="maxForegroundRatio", gt=0, le=1)
    support_padding_px: int = Field(alias="supportPaddingPx", ge=0, le=512)
    boundary_band_px: int = Field(alias="boundaryBandPx", ge=0, le=256)
    golden_masks: list[GoldenSubjectMask] = Field(alias="goldenMasks", min_length=1, max_length=256)

    def model_post_init(self, __context: object) -> None:
        if self.min_foreground_ratio >= self.max_foreground_ratio:
            raise ValueError("minForegroundRatio must be less than maxForegroundRatio")
        if len({item.golden_id for item in self.golden_masks}) != len(self.golden_masks):
            raise ValueError("Golden subject mask ids must be unique")
        for item in self.golden_masks:
            if item.model_quality_score < self.min_model_quality_score:
                raise ValueError("Golden subject mask quality is below policy")
            if not self.min_foreground_ratio <= item.foreground_ratio <= self.max_foreground_ratio:
                raise ValueError("Golden subject mask foreground ratio is outside policy")
            try:
                mask_bytes = base64.b64decode(item.mask_png_base64, validate=True)
                with Image.open(BytesIO(mask_bytes)) as image:
                    if image.format != "PNG" or image.size != (self.canonical_width, self.canonical_height):
                        raise ValueError("Golden subject mask dimensions must match canonical dimensions")
                    if image.mode not in {"1", "L"}:
                        raise ValueError("Golden subject mask must be a single-channel binary PNG")
                    colors = image.convert("L").getcolors(maxcolors=3)
                    if not colors or any(value not in {0, 255} for _, value in colors):
                        raise ValueError("Golden subject mask must contain only binary pixels")
                    if not any(value == 255 and count > 0 for count, value in colors):
                        raise ValueError("Golden subject mask must not be empty")
                    prompt = item.prompt_region_normalized
                    left = round(prompt["x"] * self.canonical_width)
                    top = round(prompt["y"] * self.canonical_height)
                    right = round((prompt["x"] + prompt["width"]) * self.canonical_width)
                    bottom = round((prompt["y"] + prompt["height"]) * self.canonical_height)
                    prompt_area = max(1, (right - left) * (bottom - top))
                    foreground = sum(
                        count for count, value in colors if value == 255
                    )
                    actual_ratio = foreground / prompt_area
                    tolerance = max(1e-9, 1.0 / prompt_area)
                    if abs(actual_ratio - item.foreground_ratio) > tolerance:
                        raise ValueError("Golden subject mask foreground ratio does not match PNG")
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError("Golden subject mask PNG is invalid") from exc


class CurrentSubjectSegmentationPolicy(StrictArtifactModel):
    """Runtime Current-mask gates and paired-interior construction policy."""

    version: str = Field(pattern=r"^current-subject-[0-9]+\.[0-9]+$")
    method: Literal["MOBILE_SAM_VIT_T_BOX_PROMPT"]
    prompt_policy: Literal["INSPECTION_ROI_BOUNDING_BOX_V1"] = Field(alias="promptPolicy")
    interior_erosion_px: int = Field(alias="interiorErosionPx", ge=1, le=256)
    min_mask_iou: float = Field(alias="minMaskIou", gt=0, le=1)
    max_area_delta_ratio: float = Field(alias="maxAreaDeltaRatio", ge=0, lt=1)
    min_interior_ratio: float = Field(alias="minInteriorRatio", gt=0, le=1)
    boundary_min_region_area_ratio: float = Field(alias="boundaryMinRegionAreaRatio", ge=0, lt=1)
    max_boundary_regions: int = Field(alias="maxBoundaryRegions", ge=1, le=64)


class DimensionMeasurementPolicy(StrictArtifactModel):
    """Pinned engineering policy for fail-closed planar metric measurement."""

    version: Literal["dimension-1.0"]
    method: Literal["CHARUCO_PLANE_CURRENT_MASK_MIN_AREA_RECT_V1"]
    approval_state: Literal["ENGINEERING_AUTO", "APPROVED"] = Field(alias="approvalState")
    calibration_source: Literal["CHARUCO_BOARD_PLANE_V1"] = Field(alias="calibrationSource")
    max_plane_reprojection_error_px: float = Field(alias="maxPlaneReprojectionErrorPx", gt=0, le=100)
    segmentation_boundary_uncertainty_px: float = Field(
        alias="segmentationBoundaryUncertaintyPx", gt=0, le=100,
    )
    max_relative_linear_uncertainty: float = Field(
        alias="maxRelativeLinearUncertainty", gt=0, le=1,
    )
    min_contour_area_px: int = Field(alias="minContourAreaPx", ge=16, le=100_000_000)
    min_contour_points: int = Field(alias="minContourPoints", ge=4, le=1_000_000)


class ProductionArtifact(StrictArtifactModel):
    schema_version: Literal["1.1"] = Field(alias="schemaVersion")
    recipe_id: Identifier = Field(alias="recipeId")
    machine_id: Identifier = Field(alias="machineId")
    board_id: Identifier = Field(alias="boardId")
    golden_set_version: PrefixedSha256 = Field(alias="goldenSetVersion")
    normalization_pipeline_version: PrefixedSha256 = Field(alias="normalizationPipelineVersion")
    analyzer_model_version: PrefixedSha256 = Field(alias="analyzerModelVersion")
    decision_policy_version: PrefixedSha256 = Field(alias="decisionPolicyVersion")
    analyzer_runtime_version: PrefixedSha256 = Field(alias="analyzerRuntimeVersion")
    model_repository_version: PrefixedSha256 = Field(alias="modelRepositoryVersion")
    board_installation_version: PrefixedSha256 = Field(alias="boardInstallationVersion")
    model_weights_sha256: PrefixedSha256 = Field(alias="modelWeightsSha256")
    board: CharucoBoard
    still_gate: StillGate = Field(alias="stillGate")
    target_alignment: TargetAlignmentPolicy = Field(alias="targetAlignment")
    golden_embeddings: list[GoldenEmbedding] = Field(alias="goldenEmbeddings", min_length=1, max_length=256)
    spatial_difference_policy: SpatialDifferencePolicy | None = Field(default=None, alias="spatialDifferencePolicy")

    def model_post_init(self, __context: object) -> None:
        if self.board.marker_length_mm >= self.board.square_length_mm:
            raise ValueError("markerLengthMm must be less than squareLengthMm")
        lengths = {len(item.values) for item in self.golden_embeddings}
        if len(lengths) != 1:
            raise ValueError("all Golden embeddings must have one dimension")
        if len({item.id for item in self.golden_embeddings}) != len(self.golden_embeddings):
            raise ValueError("Golden embedding ids must be unique")
        for item in self.golden_embeddings:
            if not all(value == value and abs(value) != float("inf") for value in item.values):
                raise ValueError("Golden embeddings must be finite")
            norm = sum(value * value for value in item.values) ** 0.5
            if norm <= 1e-12:
                raise ValueError("Golden embeddings must have non-zero norm")
            if item.canonical_image_png_base64 is not None:
                try:
                    payload = base64.b64decode(item.canonical_image_png_base64, validate=True)
                    with Image.open(BytesIO(payload)) as image:
                        if image.format != "PNG" or image.size != (
                            self.target_alignment.canonical_width,
                            self.target_alignment.canonical_height,
                        ):
                            raise ValueError("Golden canonical PNG dimensions are invalid")
                        image.load()
                except (OSError, UnidentifiedImageError) as exc:
                    raise ValueError("Golden canonical PNG is invalid") from exc


class ProductionArtifactV12(ProductionArtifact):
    """Schema 1.2: immutable canonical inspection ROI is mandatory."""

    schema_version: Literal["1.2"] = Field(alias="schemaVersion")
    inspection_roi: InspectionRoiContract = Field(alias="inspectionRoi")

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        contract_regions = [item.model_dump() for item in self.inspection_roi.inspection_regions]
        alignment_regions = [item.model_dump() for item in self.target_alignment.inspection_regions]
        if contract_regions != alignment_regions:
            raise ValueError("inspection ROI regions must match target alignment inspection regions")
        if inspection_roi_image(self.inspection_roi).getbbox() is None:
            raise ValueError("inspection ROI polygon and regions must overlap")


class ProductionArtifactV13(ProductionArtifactV12):
    """Schema 1.3: immutable Golden subject masks constrain spatial evidence."""

    schema_version: Literal["1.3"] = Field(alias="schemaVersion")
    subject_segmentation: SubjectSegmentationContract = Field(alias="subjectSegmentation")

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        contract = self.subject_segmentation
        if (
            contract.canonical_width != self.target_alignment.canonical_width
            or contract.canonical_height != self.target_alignment.canonical_height
        ):
            raise ValueError("subject segmentation canonical bounds mismatch")
        golden_by_id = {item.id: item for item in self.golden_embeddings}
        if set(golden_by_id) != {item.golden_id for item in contract.golden_masks}:
            raise ValueError("Golden subject masks must match Golden embeddings one-to-one")
        for subject in contract.golden_masks:
            golden = golden_by_id[subject.golden_id]
            if golden.canonical_sha256 is None or not hmac.compare_digest(golden.canonical_sha256, subject.canonical_sha256):
                raise ValueError("Golden subject mask canonical digest mismatch")
        bounds = inspection_roi_image(self.inspection_roi).getbbox()
        if bounds is None:
            raise ValueError("inspection ROI is empty")
        left_px, top_px, right_px, bottom_px = bounds
        left = left_px / contract.canonical_width
        top = top_px / contract.canonical_height
        right = right_px / contract.canonical_width
        bottom = bottom_px / contract.canonical_height
        expected_prompt = {"x": left, "y": top, "width": right - left, "height": bottom - top}
        for subject in contract.golden_masks:
            if any(abs(subject.prompt_region_normalized[key] - value) > 1e-9 for key, value in expected_prompt.items()):
                raise ValueError("Golden subject prompt does not match inspection ROI bounding box")


class ProductionArtifactV14(ProductionArtifactV13):
    """Schema 1.4: bounded subject alignment and candidate verification."""

    schema_version: Literal["1.4"] = Field(alias="schemaVersion")
    subject_alignment: SubjectAlignmentContract = Field(alias="subjectAlignment")
    candidate_verification_policy: CandidateVerificationPolicy = Field(alias="candidateVerificationPolicy")

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        if self.spatial_difference_policy is None:
            raise ValueError("schema 1.4 requires spatialDifferencePolicy")
        if self.candidate_verification_policy.max_candidates < self.spatial_difference_policy.max_regions:
            raise ValueError("candidate verifier must cover every retained spatial candidate")


class ProductionArtifactV15(ProductionArtifactV14):
    """Schema 1.5: ROI-only scoring plus explicit four-role reference identity."""

    schema_version: Literal["1.5"] = Field(alias="schemaVersion")
    scorer_input_contract: ScorerInputContract = Field(alias="scorerInputContract")
    recipe_analysis_profile: RecipeAnalysisProfile = Field(alias="recipeAnalysisProfile")

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        scorer = self.scorer_input_contract
        profile = self.recipe_analysis_profile
        if not hmac.compare_digest(scorer.inspection_roi_contract_digest, self.inspection_roi.digest):
            raise ValueError("scorer input contract is not bound to the inspection ROI")
        if not hmac.compare_digest(profile.inspection_roi_contract_digest, self.inspection_roi.digest):
            raise ValueError("recipe analysis profile is not bound to the inspection ROI")
        if not hmac.compare_digest(profile.scorer_input_contract_digest, scorer.digest):
            raise ValueError("recipe analysis profile is not bound to the scorer input contract")
        if not hmac.compare_digest(profile.target_reference.source_digest, self.target_alignment.reference_image_sha256):
            raise ValueError("TargetReference does not identify the target alignment reference")
        if not hmac.compare_digest(profile.normal_reference_set.source_digest, self.golden_set_version):
            raise ValueError("NormalReferenceSet does not identify the compiled Golden set")
        if profile.display_reference.source_digest not in {item.source_sha256 for item in self.golden_embeddings}:
            raise ValueError("DisplayReference does not identify a member of the normal reference set")
        for golden in self.golden_embeddings:
            if golden.scorer_input_sha256 is None or golden.scorer_input_tiles is None:
                raise ValueError("schema 1.5 requires ROI-only scorer inputs for every normal reference")


class ProductionArtifactV16(ProductionArtifactV15):
    """Schema 1.6: runtime Current mask, paired interior and boundary geometry."""

    schema_version: Literal["1.6"] = Field(alias="schemaVersion")
    current_subject_segmentation: CurrentSubjectSegmentationPolicy = Field(alias="currentSubjectSegmentation")

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        current = self.current_subject_segmentation
        subject = self.subject_segmentation
        if current.method != subject.method or current.prompt_policy != subject.prompt_policy:
            raise ValueError("Current subject segmentation must reuse the Golden model and prompt policy")
        if self.scorer_input_contract.schema_version != "1.1":
            raise ValueError("schema 1.6 requires paired-interior scorer input contract")


class ProductionArtifactV17(ProductionArtifactV16):
    """Schema 1.7: local calibration and structural candidate confirmation."""

    schema_version: Literal["1.7"] = Field(alias="schemaVersion")
    candidate_verification_policy: CandidateVerificationPolicyV2 = Field(alias="candidateVerificationPolicy")


class ProductionArtifactV18(ProductionArtifactV17):
    """Schema 1.8: ChArUco-calibrated Current-mask physical dimensions."""

    schema_version: Literal["1.8"] = Field(alias="schemaVersion")
    dimension_measurement_policy: DimensionMeasurementPolicy = Field(alias="dimensionMeasurementPolicy")


def require_subject_segmentation(artifact: "ProductionArtifactV13") -> SubjectSegmentationContract:
    contract = artifact.subject_segmentation
    roi = require_inspection_roi(artifact)
    inspection = inspection_roi_image(roi)
    for golden in contract.golden_masks:
        mask_bytes = base64.b64decode(golden.mask_png_base64, validate=True)
        with Image.open(BytesIO(mask_bytes)) as image:
            mask = image.convert("L")
            outside = Image.new("L", mask.size, color=0)
            outside.paste(mask, mask=Image.eval(inspection, lambda value: 255 - value))
            if outside.getbbox() is not None:
                raise ArtifactError("SUBJECT_MASK_OUTSIDE_INSPECTION_ROI")
    return contract


def load_artifact(path: Path, expected_digest: str, weights: Path) -> ProductionArtifact | ProductionArtifactV12 | ProductionArtifactV13 | ProductionArtifactV14 | ProductionArtifactV15 | ProductionArtifactV16 | ProductionArtifactV17 | ProductionArtifactV18:
    try:
        actual = digest_file(path)
        if not hmac.compare_digest(actual, expected_digest):
            raise ArtifactError("ARTIFACT_PACKAGE_DIGEST_MISMATCH")
        raw = path.read_bytes()
        document = json.loads(raw)
        schema_version = document.get("schemaVersion")
        model = (
            ProductionArtifactV18 if schema_version == "1.8"
            else ProductionArtifactV17 if schema_version == "1.7"
            else ProductionArtifactV16 if schema_version == "1.6"
            else ProductionArtifactV15 if schema_version == "1.5"
            else ProductionArtifactV14 if schema_version == "1.4"
            else ProductionArtifactV13 if schema_version == "1.3"
            else ProductionArtifactV12 if schema_version == "1.2"
            else ProductionArtifact
        )
        artifact = model.model_validate(document)
        if isinstance(artifact, ProductionArtifactV13):
            require_subject_segmentation(artifact)
    except ArtifactError:
        raise
    except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError("ARTIFACT_MANIFEST_INVALID") from exc
    actual_weights = "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest()
    if not hmac.compare_digest(actual_weights, artifact.model_weights_sha256):
        raise ArtifactError("MODEL_WEIGHTS_DIGEST_MISMATCH")
    return artifact


def verify_artifact_binding(artifact: ProductionArtifact, request: AnalyzeRequest) -> None:
    paired_schema_matches = (
        request.schema_version == "1.4"
        if isinstance(artifact, ProductionArtifactV18)
        else request.schema_version == "1.3"
        if isinstance(artifact, ProductionArtifactV17)
        else request.schema_version == "1.2"
        if isinstance(artifact, ProductionArtifactV16)
        else request.schema_version not in {"1.2", "1.3"}
    )
    if not paired_schema_matches:
        raise ArtifactError("PAIRED_INTERIOR_SCHEMA_ARTIFACT_MISMATCH")
    expected = {
        "recipe": (artifact.recipe_id, request.recipe_id),
        "machine": (artifact.machine_id, request.machine_id),
        "boardIdentity": (artifact.board_id, request.board_id),
        "golden": (artifact.golden_set_version, request.execution_bundle.golden_set_version),
        "normalization": (artifact.normalization_pipeline_version, request.execution_bundle.normalization_pipeline_version),
        "model": (artifact.analyzer_model_version, request.execution_bundle.analyzer_model_version),
        "decision": (artifact.decision_policy_version, request.execution_bundle.decision_policy_version),
        "board": (artifact.board_installation_version, request.execution_bundle.board_installation_version),
    }
    if any(not hmac.compare_digest(left, right) for left, right in expected.values()):
        raise ArtifactError("ARTIFACT_EXECUTION_BUNDLE_MISMATCH")
    if isinstance(artifact, ProductionArtifactV15):
        if request.recipe_analysis_profile_digest is None:
            raise ArtifactError("RECIPE_ANALYSIS_PROFILE_DIGEST_REQUIRED")
        if not hmac.compare_digest(request.recipe_analysis_profile_digest, artifact.recipe_analysis_profile.digest):
            raise ArtifactError("RECIPE_ANALYSIS_PROFILE_DIGEST_MISMATCH")
