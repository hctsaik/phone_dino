from __future__ import annotations

import hashlib
import hmac
import json
import base64
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contracts import AnalyzeRequest, Identifier, PrefixedSha256
from .security import digest_file


class ArtifactError(RuntimeError):
    pass


class StrictArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class CharucoBoard(StrictArtifactModel):
    squares_x: int = Field(alias="squaresX", ge=3, le=100)
    squares_y: int = Field(alias="squaresY", ge=3, le=100)
    square_length_mm: float = Field(alias="squareLengthMm", gt=0, le=1000)
    marker_length_mm: float = Field(alias="markerLengthMm", gt=0, le=1000)
    dictionary: str = Field(pattern=r"^DICT_[A-Z0-9_]+$")
    canonical_width: int = Field(alias="canonicalWidth", ge=224, le=8192)
    canonical_height: int = Field(alias="canonicalHeight", ge=224, le=8192)


class StillGate(StrictArtifactModel):
    min_charuco_corners: int = Field(alias="minCharucoCorners", ge=4, le=10000)
    min_laplacian_variance: float = Field(alias="minLaplacianVariance", ge=0)
    max_over_exposure_ratio: float = Field(alias="maxOverExposureRatio", ge=0, le=1)


class ImageRegion(StrictArtifactModel):
    x: int = Field(ge=0, le=8191)
    y: int = Field(ge=0, le=8191)
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)


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


class GoldenEmbedding(StrictArtifactModel):
    id: Identifier
    source_sha256: PrefixedSha256 = Field(alias="sourceSha256")
    values: list[float] = Field(min_length=2, max_length=4096)


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


class ProductionArtifactV12(ProductionArtifact):
    """Schema 1.2: immutable canonical inspection ROI is mandatory."""

    schema_version: Literal["1.2"] = Field(alias="schemaVersion")
    inspection_roi: InspectionRoiContract = Field(alias="inspectionRoi")


def load_artifact(path: Path, expected_digest: str, weights: Path) -> ProductionArtifact | ProductionArtifactV12:
    try:
        actual = digest_file(path)
        if not hmac.compare_digest(actual, expected_digest):
            raise ArtifactError("ARTIFACT_PACKAGE_DIGEST_MISMATCH")
        raw = path.read_bytes()
        document = json.loads(raw)
        model = ProductionArtifactV12 if document.get("schemaVersion") == "1.2" else ProductionArtifact
        artifact = model.model_validate(document)
    except ArtifactError:
        raise
    except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError("ARTIFACT_MANIFEST_INVALID") from exc
    actual_weights = "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest()
    if not hmac.compare_digest(actual_weights, artifact.model_weights_sha256):
        raise ArtifactError("MODEL_WEIGHTS_DIGEST_MISMATCH")
    return artifact


def verify_artifact_binding(artifact: ProductionArtifact, request: AnalyzeRequest) -> None:
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
