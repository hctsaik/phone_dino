from __future__ import annotations

import hashlib
import math
import base64
import json
import logging
from threading import Lock
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Protocol

from PIL import Image, ImageOps

from .analyzer import RUNTIME_DIGEST
from .artifacts import (
    ArtifactError, CandidateVerificationPolicy, CandidateVerificationPolicyV2, GoldenEmbedding, ProductionArtifact,
    ProductionArtifactV12, ProductionArtifactV13, ProductionArtifactV14, ProductionArtifactV15,
    ProductionArtifactV16, ProductionArtifactV17, ProductionArtifactV18, ProductionArtifactV19, SpatialDifferencePolicy,
    inspection_roi_image, load_artifact, require_inspection_roi, require_subject_segmentation,
    verify_artifact_binding,
)
from .config import Settings
from .contracts import (
    AnalysisObservation, AnalysisState, AnalyzeObservation, AnalyzeRequest,
    AlignmentObservation, BboxNormalized, BoundaryDifferenceEvidence, BoundaryDifferenceRegion,
    CandidateDimensionScaleEvidence, CandidateDimensionUncertaintyEvidence,
    CandidateFilter, CandidatePhysicalDimensionEvidence, CandidateVerification, CandidateVerificationV2,
    CaptureAssessment, CaptureState, DifferenceRegion, NormalizationObservation, ResolvedVersions,
    DimensionUncertaintyEvidence, GoldenDimensionObservation, GoldenDimensionRequest,
    GoldenDimensionBoardCandidate,
    DepthOffsetEstimateEvidence, MetricCalibrationEvidence, PhysicalDimensionEvidence,
    ScorerInputTileDigest, SpatialDifferenceEvidence, SubjectSegmentationEvidence,
)
from .decoder import DecodedImage
from .reference_board import ReferenceBoardVerifier
from .engines import LocalDinoV2Adapter
from .offset_plane import (
    BoardPose, CameraIntrinsics, OffsetPlaneGeometryError, estimate_board_pose,
    estimate_square_focal_length_from_planar_board, intersect_pixels_with_front_offset_plane,
)
from .depth_offset import (
    RelativeDepthOffsetPolicy,
    estimate_front_offset_from_relative_inverse_depth,
    relative_depth_posterior_improves_prior,
)
from .depth_anything import DepthAnythingV2RelativeDepthEstimator
from .security import digest_directory, digest_file
from .segmenters import MobileSamSegmenter, SubjectMaskPrediction, SubjectSegmenter

DINO_INPUT_SIZE = 224
DINO_RESIZE_SHORT_EDGE = 256
# Target alignment is intentionally conservative for similarity scoring.  It
# must not crop a real foreground boundary when reused as a raw-image prompt
# for physical dimensions, especially after a hand-held perspective change.
SOURCE_METRIC_PROMPT_PADDING_RATIO = 0.10


@dataclass(frozen=True, slots=True)
class NormalizedCapture:
    rgb: object
    encoded: bytes
    reason_codes: tuple[str, ...] = ()
    alignment: AlignmentObservation | None = None
    target_from_input: object | None = None
    metric_calibration: "TargetMetricCalibration | None" = None
    # Metric dimensions are deliberately measured in the decoded source image,
    # not in the 896px DINO comparison image.  Keep the transforms separately
    # so the comparison pipeline may evolve without changing a millimetre
    # measurement.
    source_rgb: object | None = None
    source_to_plane: object | None = None
    target_from_source: object | None = None
    calibration_support_plane: object | None = None
    # The immutable board geometry selected from this capture.  It is retained
    # separately from a board-plane homography because offset-plane PnP uses
    # the original source marker corners.
    calibration_board: GoldenDimensionBoardCandidate | None = None
    calibration_fiducial: str | None = None


@dataclass(frozen=True, slots=True)
class PlaneMetricCalibration:
    pixels_per_mm_x: float
    pixels_per_mm_y: float
    detected_corner_count: int
    inlier_corner_count: int
    reprojection_error_px: float
    calibration_fiducial: str = "CHARUCO_CORNERS"


@dataclass(frozen=True, slots=True)
class TargetMetricCalibration:
    target_to_plane: object
    pixels_per_mm_x: float
    pixels_per_mm_y: float
    detected_corner_count: int
    inlier_corner_count: int
    reprojection_error_px: float
    calibration_fiducial: str = "CHARUCO_CORNERS"


@dataclass(frozen=True, slots=True)
class PlaneNormalizedCapture:
    rgb: object
    reason_codes: tuple[str, ...] = ()
    metric_calibration: PlaneMetricCalibration | None = None
    source_rgb: object | None = None
    input_to_plane: object | None = None
    calibration_board: GoldenDimensionBoardCandidate | None = None
    calibration_fiducial: str | None = None
    # Polygon in canonical board-plane pixels covered by the physical board.
    # It is evidence of where a 2-D planar measurement may be made, not proof
    # that an arbitrary foreground object is coplanar.
    calibration_support_plane: object | None = None
    calibration_diagnostics: tuple[dict[str, object], ...] = ()


class Normalizer(Protocol):
    def normalize(self, image: DecodedImage, artifact: ProductionArtifact) -> NormalizedCapture: ...


class Embedder(Protocol):
    def embed(self, rgb: object) -> list[float]: ...


class RelativeDepthEstimator(Protocol):
    """Optional per-image relative inverse-depth provider (for example Depth Anything)."""

    def estimate_inverse_depth(self, rgb: object) -> object: ...


@dataclass(frozen=True, slots=True)
class PatchEmbedding:
    global_vector: list[float]
    patch_grid: list[list[float]]
    grid_height: int
    grid_width: int


@dataclass(frozen=True, slots=True)
class ScorerInputTile:
    id: str
    x: int
    y: int
    side: int
    rgb: object
    sha256: str


@dataclass(frozen=True, slots=True)
class SubjectScope:
    core_mask: object
    support_mask: object
    boundary_mask: object
    evidence: SubjectSegmentationEvidence
    paired_interior: bool = False


@dataclass(frozen=True, slots=True)
class PairedSubjectResult:
    scope: SubjectScope | None
    boundary_evidence: BoundaryDifferenceEvidence | None
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class PairedScoringCandidate:
    id: str
    distance: float
    golden: GoldenEmbedding
    scope: SubjectScope
    current_inputs: list[ScorerInputTile]
    golden_inputs: list[ScorerInputTile]
    current_cache: dict[tuple[int, int, int], PatchEmbedding]
    golden_cache: dict[tuple[str, int, int, int], PatchEmbedding]
    golden_rgb: object
    boundary_evidence: BoundaryDifferenceEvidence


class PatchEmbedder(Protocol):
    """Optional capability: an Embedder that can also return patch tokens.

    Kept separate from Embedder so every existing stub/fixture embedder used
    by contract and synthetic tests keeps working unchanged; only a real
    PatchEmbedder unlocks spatialDifferenceEvidence.
    """

    def embed_with_patches(self, rgb: object) -> PatchEmbedding: ...


class TargetAligner(Protocol):
    def align(self, plane_rgb: object, artifact: ProductionArtifact) -> NormalizedCapture: ...


def _alignment_observation(
    *, state: str, inliers: int = 0, ratio: float = 0.0, error: float = 0.0,
    coverage: float = 0.0, within_bounds: bool = False, method: str = "TARGET_AFFINE",
) -> AlignmentObservation:
    return AlignmentObservation(
        state=state, method=method, targetRelative=True, inlierCount=inliers,
        inlierRatio=ratio, reprojectionErrorPx=error, coverageRatio=coverage,
        transformWithinBounds=within_bounds, inspectionMaskApplied=True,
    )


@dataclass(frozen=True, slots=True)
class DarkBodyContour:
    x: float
    y: float
    width: float
    height: float
    center_x: float
    center_y: float
    angle_degrees: float
    rectangularity: float
    contour_points: int


def _detect_dark_body_contour(image_bgr: object, expected_aspect: float) -> DarkBodyContour | None:
    """Locate the guided, near-centre rectangular equipment body.

    This is an explicitly engineering-only bootstrap for low-texture black
    equipment where ORB cannot obtain enough target-only correspondences. It
    intentionally uses the large rigid body contour, not background texture.
    """
    import cv2
    import numpy as np

    image = np.asarray(image_bgr, dtype=np.uint8)
    height, width = image.shape[:2]
    resize_scale = min(1.0, 1008.0 / max(height, width))
    resized_width = max(1, round(width * resize_scale))
    resized_height = max(1, round(height * resize_scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 90)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = resized_width * resized_height * 0.01
    candidates: list[tuple[float, object, tuple[float, float, float, float, float], float]] = []
    for contour in contours:
        (center_x_px, center_y_px), (rect_width, rect_height), angle = cv2.minAreaRect(contour)
        if rect_width <= 0 or rect_height <= 0:
            continue
        # Normalize every rotated rectangle to width=short side and
        # height=long side.  OpenCV otherwise changes its angle convention at
        # 90 degrees, which makes the same equipment discontinuous as the
        # phone rotates.
        if rect_width > rect_height:
            rect_width, rect_height = rect_height, rect_width
            angle += 90.0
        angle = (float(angle) + 90.0) % 180.0 - 90.0
        area = float(cv2.contourArea(contour))
        rotated_area = float(rect_width * rect_height)
        aspect = rect_width / rect_height
        center_x = center_x_px / resized_width
        center_y = center_y_px / resized_height
        height_ratio = rect_height / resized_height
        rectangularity = area / max(1.0, rotated_area)
        if not (
            rotated_area >= minimum_area and 0.4 <= aspect <= 0.85
            and 0.25 <= center_x <= 0.75 and 0.3 <= center_y <= 0.8
            and 0.18 <= height_ratio <= 0.55 and rectangularity >= 0.15
            and abs(angle) <= 35.0
        ):
            continue
        aspect_similarity = max(0.0, 1.0 - abs(aspect - expected_aspect) / max(expected_aspect, 1e-6))
        # A protruding anomaly can merge with the outer subject silhouette.
        # The inner equipment edge remains geometrically stable but may have a
        # small contourArea due to Canny topology.  Strongly prefer the Golden
        # aspect while retaining an area/rectangularity prior; this keeps the
        # bootstrap tied to the rigid subject rather than the new object.
        score = rotated_area * (aspect_similarity ** 8) * (0.5 + min(0.5, rectangularity))
        candidates.append((
            score, contour,
            (center_x_px, center_y_px, rect_width, rect_height, angle),
            rectangularity,
        ))
    if not candidates:
        return None
    _, contour, (center_x_px, center_y_px, rect_width, rect_height, angle), rectangularity = max(
        candidates, key=lambda item: item[0],
    )
    inverse = 1.0 / resize_scale
    center_x = center_x_px * inverse
    center_y = center_y_px * inverse
    candidate_width = rect_width * inverse
    candidate_height = rect_height * inverse
    return DarkBodyContour(
        x=center_x - candidate_width / 2.0,
        y=center_y - candidate_height / 2.0,
        width=candidate_width, height=candidate_height,
        center_x=center_x, center_y=center_y, angle_degrees=angle,
        rectangularity=min(1.0, rectangularity), contour_points=len(contour),
    )


def _contour_bootstrap_transform(detected: DarkBodyContour, anchor, policy):
    """Map a rotated Current subject rectangle into the canonical Golden box."""
    import cv2
    import numpy as np

    scale_x = anchor.width / detected.width
    scale_y = anchor.height / detected.height
    anchor_center = np.asarray(
        (anchor.x + anchor.width / 2.0, anchor.y + anchor.height / 2.0), dtype=np.float64,
    )
    detected_center = np.asarray((detected.center_x, detected.center_y), dtype=np.float64)
    # minAreaRect reports the same signed angle needed by OpenCV's
    # getRotationMatrix2D to undo the observed Current orientation.
    rotation = cv2.getRotationMatrix2D((0.0, 0.0), detected.angle_degrees, 1.0)[:, :2]
    linear = np.diag((scale_x, scale_y)) @ rotation
    translation = anchor_center - linear @ detected_center
    transform = np.column_stack((linear, translation))
    bounded = (
        policy.min_scale <= scale_x <= policy.max_scale
        and policy.min_scale <= scale_y <= policy.max_scale
        and abs(detected.angle_degrees) <= getattr(policy, "max_rotation_degrees", 45.0)
        and float(np.linalg.norm(translation)) <= policy.max_translation_px
    )
    return transform, scale_x, scale_y, translation, bounded


def _align_with_dark_body_contour(current_bgr: object, policy) -> NormalizedCapture:
    import cv2
    import numpy as np

    anchor = policy.contour_anchor_region
    expected_aspect = anchor.width / anchor.height
    detected = _detect_dark_body_contour(current_bgr, expected_aspect)
    if detected is None:
        return NormalizedCapture(
            None, b"", ("TARGET_CONTOUR_NOT_FOUND",),
            _alignment_observation(state="NOT_ALIGNED", method="CONTOUR_ANCHOR_AFFINE"),
        )
    detected_aspect = detected.width / detected.height
    aspect_delta = abs(detected_aspect - expected_aspect) / expected_aspect
    transform, scale_x, scale_y, translation, bounded = _contour_bootstrap_transform(
        detected, anchor, policy,
    )
    coverage = min(1.0, anchor.width * anchor.height / (policy.canonical_width * policy.canonical_height))
    accepted = bounded and aspect_delta <= 0.25 and detected.rectangularity >= 0.15
    observation = _alignment_observation(
        state="ALIGNED" if accepted else "NOT_ALIGNED",
        inliers=max(4, min(detected.contour_points, 10_000)), ratio=detected.rectangularity,
        error=min(1000.0, aspect_delta * policy.max_reprojection_error_px),
        coverage=coverage, within_bounds=bounded, method="CONTOUR_ANCHOR_AFFINE",
    )
    if not accepted:
        return NormalizedCapture(None, b"", ("TARGET_CONTOUR_ANCHOR_INVALID",), observation)
    target = cv2.warpAffine(
        np.asarray(current_bgr, dtype=np.uint8), transform,
        (policy.canonical_width, policy.canonical_height), flags=cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".png", target)
    if not ok:
        raise RuntimeError("TARGET_CANONICAL_ENCODING_FAILED")
    return NormalizedCapture(
        rgb, encoded.tobytes(), (), observation,
        target_from_input=np.asarray(transform, dtype=np.float64),
    )


def _subject_alignment_feature(gray: object) -> object:
    """Lighting-normalized intensity image used by subject-boundary ECC."""
    import cv2
    import numpy as np

    image = np.asarray(gray, dtype=np.uint8)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    return (equalized.astype(np.float32) / 255.0)


def _held_out_subject_edge_metrics(
    reference_gray: object, current_gray: object, boundary_band: object,
    held_out_mask: object, reference_edge_support: object,
    maximum_residual_px: float,
) -> tuple[float, int, float]:
    """Measure immutable Golden strong edges against a Current hypothesis.

    Golden Canny edges must also touch the immutable subject-mask boundary.
    This prevents monitor, table, cable, or other background texture that
    happens to cross the wider search band from masquerading as subject pose
    error. Only Golden samples are held out; Current edges may lie anywhere in
    the subject boundary search band so a small residual pose can be measured.
    """
    import cv2
    import numpy as np

    reference_edges = cv2.bitwise_and(
        cv2.Canny(np.asarray(reference_gray, dtype=np.uint8), 100, 200),
        np.asarray(held_out_mask, dtype=np.uint8),
    )
    reference_edges = cv2.bitwise_and(
        reference_edges,
        np.asarray(reference_edge_support, dtype=np.uint8),
    )
    current_edges = cv2.bitwise_and(
        cv2.Canny(np.asarray(current_gray, dtype=np.uint8), 100, 200),
        np.asarray(boundary_band, dtype=np.uint8),
    )
    template_edge_count = int(np.count_nonzero(reference_edges))
    if template_edge_count == 0 or cv2.countNonZero(current_edges) == 0:
        return 1000.0, 0, 0.0
    distance_to_current = cv2.distanceTransform(
        (current_edges == 0).astype(np.uint8), cv2.DIST_L2, 3,
    )
    residuals = distance_to_current[reference_edges > 0]
    # Q75 tolerates a localized added/missing component, while the independent
    # coverage gate still rejects a capture where most of the subject is gone.
    residual = float(np.percentile(residuals, 75))
    matched = int(np.count_nonzero(residuals <= maximum_residual_px))
    return residual, matched, min(1.0, matched / template_edge_count)


def _align_with_subject_ecc(
    current_bgr: object, reference_bgr: object, policy, subject_mask: object, contract,
) -> NormalizedCapture:
    """Contour bootstrap plus bounded ECC on an immutable Golden boundary band."""
    import cv2
    import numpy as np

    anchor = policy.contour_anchor_region
    if anchor is None:
        return NormalizedCapture(
            None, b"", ("SUBJECT_ALIGNMENT_CONTOUR_ANCHOR_REQUIRED",),
            _alignment_observation(state="NOT_ALIGNED", method="SUBJECT_CONTOUR_ECC_AFFINE"),
        )
    detected = _detect_dark_body_contour(current_bgr, anchor.width / anchor.height)
    if detected is None:
        return NormalizedCapture(
            None, b"", ("SUBJECT_ALIGNMENT_CONTOUR_NOT_FOUND",),
            _alignment_observation(state="NOT_ALIGNED", method="SUBJECT_CONTOUR_ECC_AFFINE"),
        )
    coarse_transform, scale_x, scale_y, coarse_translation, coarse_bounded = _contour_bootstrap_transform(
        detected, anchor, policy,
    )
    if not coarse_bounded:
        return NormalizedCapture(
            None, b"", ("SUBJECT_ALIGNMENT_BOOTSTRAP_OUT_OF_BOUNDS",),
            _alignment_observation(state="NOT_ALIGNED", method="SUBJECT_CONTOUR_ECC_AFFINE"),
        )
    canonical_size = (policy.canonical_width, policy.canonical_height)
    coarse = cv2.warpAffine(
        np.asarray(current_bgr, dtype=np.uint8), coarse_transform, canonical_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    reference = np.asarray(reference_bgr, dtype=np.uint8)
    mask = (np.asarray(subject_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if mask.shape != (policy.canonical_height, policy.canonical_width) or cv2.countNonZero(mask) == 0:
        raise RuntimeError("SUBJECT_ALIGNMENT_MASK_INVALID")
    diameter = contract.alignment_band_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    boundary_band = cv2.subtract(cv2.dilate(mask, kernel), cv2.erode(mask, kernel))
    # The broad band is the Current-edge search area. Golden evidence is much
    # narrower: only immutable strong edges within three pixels of the approved
    # subject-mask boundary may contribute to the held-out residual.
    subject_boundary = cv2.Canny(mask, 50, 150)
    reference_edge_support = cv2.dilate(
        subject_boundary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    rows, columns = np.indices(boundary_band.shape)
    checker = ((rows // contract.held_out_block_px + columns // contract.held_out_block_px) % 2) == 0
    fit_mask = np.where((boundary_band > 0) & checker, 255, 0).astype(np.uint8)
    held_out_mask = np.where((boundary_band > 0) & ~checker, 255, 0).astype(np.uint8)
    if cv2.countNonZero(fit_mask) < 256 or cv2.countNonZero(held_out_mask) < 256:
        return NormalizedCapture(
            None, b"", ("SUBJECT_ALIGNMENT_BAND_INSUFFICIENT",),
            _alignment_observation(state="NOT_ALIGNED", method="SUBJECT_CONTOUR_ECC_AFFINE"),
        )

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    coarse_gray = cv2.cvtColor(coarse, cv2.COLOR_BGR2GRAY)
    reference_feature = _subject_alignment_feature(reference_gray)
    coarse_feature = _subject_alignment_feature(coarse_gray)
    residual_transform = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        contract.max_iterations,
        contract.convergence_epsilon,
    )
    fit_pixels = fit_mask > 0
    reference_fit = reference_feature[fit_pixels]
    coarse_fit = coarse_feature[fit_pixels]
    if float(np.std(reference_fit)) <= 1e-9 or float(np.std(coarse_fit)) <= 1e-9:
        baseline_correlation = 0.0
    else:
        baseline_correlation = float(np.corrcoef(reference_fit, coarse_fit)[0, 1])
        if not math.isfinite(baseline_correlation):
            baseline_correlation = 0.0
    ecc_converged = True
    try:
        correlation, residual_transform = cv2.findTransformECC(
            reference_feature, coarse_feature, residual_transform, cv2.MOTION_AFFINE,
            criteria, fit_mask, 5,
        )
    except cv2.error:
        # The rotated-contour hypothesis is independently held out below.  A
        # failed optional refinement must not invalidate an already good pose.
        ecc_converged = False
        correlation = baseline_correlation
        residual_transform = np.eye(2, 3, dtype=np.float32)
    correlation = float(correlation)
    a, b = float(residual_transform[0, 0]), float(residual_transform[0, 1])
    c, d = float(residual_transform[1, 0]), float(residual_transform[1, 1])
    residual_scale_x, residual_scale_y = math.hypot(a, c), math.hypot(b, d)
    residual_rotation = abs(math.degrees(math.atan2(c, a)))
    residual_shear = abs((a * b + c * d) / max(1e-12, residual_scale_x * residual_scale_y))
    residual_translation = math.hypot(float(residual_transform[0, 2]), float(residual_transform[1, 2]))
    residual_bounded = (
        residual_translation <= contract.max_residual_translation_px
        and residual_rotation <= contract.max_residual_rotation_degrees
        and abs(residual_scale_x - 1.0) <= contract.max_residual_scale_delta
        and abs(residual_scale_y - 1.0) <= contract.max_residual_scale_delta
        and residual_shear <= contract.max_residual_shear
    )
    refined = cv2.warpAffine(
        coarse, residual_transform, canonical_size,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
    )
    refined_gray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)
    coarse_metrics = _held_out_subject_edge_metrics(
        reference_gray, coarse_gray, boundary_band, held_out_mask,
        reference_edge_support,
        contract.max_held_out_residual_px,
    )
    refined_metrics = _held_out_subject_edge_metrics(
        reference_gray, refined_gray, boundary_band, held_out_mask,
        reference_edge_support,
        contract.max_held_out_residual_px,
    )
    coarse_passes = (
        coarse_metrics[0] <= contract.max_held_out_residual_px
        and coarse_metrics[2] >= contract.min_held_out_coverage_ratio
    )
    refined_passes = (
        refined_metrics[0] <= contract.max_held_out_residual_px
        and refined_metrics[2] >= contract.min_held_out_coverage_ratio
    )
    # Prefer ECC only when its independently held-out geometry is at least as
    # good as the stable contour hypothesis. This stops a localized anomaly in
    # the ECC fit band from pulling a correct pose toward the anomaly.
    use_refined = (
        ecc_converged and residual_bounded
        and (refined_passes and not coarse_passes
             or refined_passes == coarse_passes
             and (refined_metrics[0], -refined_metrics[2])
             < (coarse_metrics[0], -coarse_metrics[2]))
    )
    selected = refined if use_refined else coarse
    held_out_residual, matched_count, held_out_coverage = (
        refined_metrics if use_refined else coarse_metrics
    )
    selected_correlation = correlation if use_refined else baseline_correlation
    selected_residual_bounded = residual_bounded if use_refined else True
    accepted = (
        math.isfinite(selected_correlation) and selected_correlation >= contract.min_ecc_correlation
        and held_out_residual <= contract.max_held_out_residual_px
        and held_out_coverage >= contract.min_held_out_coverage_ratio
        and selected_residual_bounded
    )
    observation = _alignment_observation(
        state="ALIGNED" if accepted else "NOT_ALIGNED",
        inliers=matched_count,
        ratio=max(0.0, min(1.0, selected_correlation if math.isfinite(selected_correlation) else 0.0)),
        error=min(1000.0, held_out_residual),
        coverage=max(0.0, min(1.0, held_out_coverage)),
        within_bounds=coarse_bounded and selected_residual_bounded,
        method="SUBJECT_CONTOUR_ECC_AFFINE",
    )
    if not accepted:
        reasons: list[str] = []
        if not math.isfinite(selected_correlation) or selected_correlation < contract.min_ecc_correlation:
            reasons.append("SUBJECT_ALIGNMENT_CORRELATION_LOW")
        if held_out_residual > contract.max_held_out_residual_px:
            reasons.append("SUBJECT_ALIGNMENT_HELD_OUT_RESIDUAL_HIGH")
        if held_out_coverage < contract.min_held_out_coverage_ratio:
            reasons.append("SUBJECT_ALIGNMENT_HELD_OUT_COVERAGE_LOW")
        if not selected_residual_bounded:
            reasons.append("SUBJECT_ALIGNMENT_RESIDUAL_OUT_OF_BOUNDS")
        return NormalizedCapture(None, b"", tuple(reasons), observation)
    selected_transform = np.asarray(coarse_transform, dtype=np.float64)
    if use_refined:
        residual_inverse = cv2.invertAffineTransform(np.asarray(residual_transform, dtype=np.float64))
        selected_transform = (
            np.vstack((residual_inverse, (0.0, 0.0, 1.0)))
            @ np.vstack((selected_transform, (0.0, 0.0, 1.0)))
        )[:2]
    ok, encoded = cv2.imencode(".png", selected)
    if not ok:
        raise RuntimeError("TARGET_CANONICAL_ENCODING_FAILED")
    return NormalizedCapture(
        cv2.cvtColor(selected, cv2.COLOR_BGR2RGB), encoded.tobytes(), (), observation,
        target_from_input=selected_transform,
    )


class OpenCvTargetAligner:
    """Offline ORB/RANSAC baseline; LightGlue can implement the same protocol later.

    Reference descriptors are computed only inside alignment regions after every
    inspection region is removed.  Consequently a changed/removed inspection
    feature cannot drive the transform.
    """

    def __init__(self, *, allow_contour_anchor_alignment: bool = False) -> None:
        self._lock = Lock()
        self._allow_contour_anchor_alignment = allow_contour_anchor_alignment
        import cv2
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)

    def align(self, plane_rgb: object, artifact: ProductionArtifact) -> NormalizedCapture:
        # OpenCV RANSAC uses process-global RNG state. Serialize and seed from
        # immutable pixels so repeated requests and process histories agree.
        with self._lock:
            return self._align(plane_rgb, artifact)

    def _align(self, plane_rgb: object, artifact: ProductionArtifact) -> NormalizedCapture:
        feature_result = self._align_feature(plane_rgb, artifact)
        if not feature_result.reason_codes:
            return feature_result
        if not self._allow_contour_anchor_alignment or artifact.target_alignment.contour_anchor_region is None:
            return feature_result
        import cv2
        import numpy as np

        current = cv2.cvtColor(np.asarray(plane_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if isinstance(artifact, ProductionArtifactV14):
            reference_bytes = base64.b64decode(artifact.target_alignment.reference_image_base64, validate=True)
            reference = cv2.imdecode(np.frombuffer(reference_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if reference is None:
                raise RuntimeError("TARGET_REFERENCE_INVALID")
            subject = artifact.subject_segmentation.golden_masks[0]
            subject_bytes = base64.b64decode(subject.mask_png_base64, validate=True)
            subject_mask = cv2.imdecode(np.frombuffer(subject_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            fallback = _align_with_subject_ecc(
                current, reference, artifact.target_alignment, subject_mask, artifact.subject_alignment,
            )
        else:
            fallback = _align_with_dark_body_contour(current, artifact.target_alignment)
        if fallback.reason_codes:
            return NormalizedCapture(
                fallback.rgb, fallback.encoded,
                tuple(dict.fromkeys((*feature_result.reason_codes, *fallback.reason_codes))),
                fallback.alignment,
            )
        return fallback

    def _align_feature(self, plane_rgb: object, artifact: ProductionArtifact) -> NormalizedCapture:
        import cv2
        import numpy as np

        policy = artifact.target_alignment
        try:
            reference_bytes = base64.b64decode(policy.reference_image_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("TARGET_REFERENCE_INVALID") from exc
        if "sha256:" + hashlib.sha256(reference_bytes).hexdigest() != policy.reference_image_sha256:
            raise RuntimeError("TARGET_REFERENCE_DIGEST_MISMATCH")
        reference = cv2.imdecode(np.frombuffer(reference_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if reference is None or reference.shape[1] != policy.canonical_width or reference.shape[0] != policy.canonical_height:
            raise RuntimeError("TARGET_REFERENCE_INVALID")
        current = cv2.cvtColor(np.asarray(plane_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        deterministic_seed = int.from_bytes(hashlib.sha256(current.tobytes() + reference_bytes).digest()[:4], "big")
        cv2.setRNGSeed(deterministic_seed & 0x7FFFFFFF)
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        reference_mask = np.zeros(ref_gray.shape, dtype=np.uint8)
        for region in policy.alignment_regions:
            reference_mask[region.y:region.y + region.height, region.x:region.x + region.width] = 255
        held_out_mask = np.zeros(ref_gray.shape, dtype=np.uint8)
        for region in policy.held_out_regions:
            held_out_mask[region.y:region.y + region.height, region.x:region.x + region.width] = 255

        detector = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
        ref_keys, ref_desc = detector.detectAndCompute(ref_gray, reference_mask)
        held_keys, held_desc = detector.detectAndCompute(ref_gray, held_out_mask)
        cur_keys, cur_desc = detector.detectAndCompute(current_gray, None)
        if ref_desc is None or held_desc is None or cur_desc is None:
            return NormalizedCapture(None, b"", ("TARGET_NOT_FOUND",), _alignment_observation(state="NOT_ALIGNED"))
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(cur_desc, ref_desc, k=2)
        raw_matches = [
            first for pair in pairs if len(pair) == 2
            for first, second in [pair] if first.distance < 0.75 * second.distance
        ]
        reverse_pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ref_desc, cur_desc, k=2)
        reverse = {
            first.queryIdx: first.trainIdx for pair in reverse_pairs if len(pair) == 2
            for first, second in [pair] if first.distance < 0.75 * second.distance
        }
        # Primary fitting uses mutual, one-to-one correspondences.  Unrestricted
        # current-frame board texture may remain in raw_matches only for the
        # explicit secondary-hypothesis ambiguity check below.
        matches = [item for item in raw_matches if reverse.get(item.trainIdx) == item.queryIdx]
        if len(matches) < policy.min_matches:
            return NormalizedCapture(None, b"", ("TARGET_MATCHES_INSUFFICIENT",), _alignment_observation(state="NOT_ALIGNED"))
        source = np.float32([cur_keys[item.queryIdx].pt for item in matches])
        destination = np.float32([ref_keys[item.trainIdx].pt for item in matches])
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            source, destination, method=cv2.RANSAC, ransacReprojThreshold=policy.max_reprojection_error_px,
            maxIters=3000, confidence=0.995, refineIters=10,
        )
        if transform is None or inlier_mask is None:
            return NormalizedCapture(None, b"", ("TARGET_TRANSFORM_INVALID",), _alignment_observation(state="NOT_ALIGNED"))
        selected = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(selected.sum())
        ratio = inlier_count / len(matches)
        predicted = cv2.transform(source[selected].reshape(1, -1, 2), transform).reshape(-1, 2)
        errors = np.linalg.norm(predicted - destination[selected], axis=1)
        reprojection_p95 = float(np.percentile(errors, 95)) if len(errors) else 1000.0
        hull_area = float(cv2.contourArea(cv2.convexHull(destination[selected]))) if inlier_count >= 3 else 0.0
        alignment_area = float(sum(region.width * region.height for region in policy.alignment_regions))
        coverage = min(1.0, hull_area / max(1.0, alignment_area))
        a, b = float(transform[0, 0]), float(transform[0, 1])
        c, d = float(transform[1, 0]), float(transform[1, 1])
        scale_x, scale_y = math.hypot(a, c), math.hypot(b, d)
        rotation = abs(math.degrees(math.atan2(c, a)))
        shear = abs((a * b + c * d) / max(1e-12, scale_x * scale_y))
        translation = math.hypot(float(transform[0, 2]), float(transform[1, 2]))
        bounded = (
            policy.min_scale <= scale_x <= policy.max_scale
            and policy.min_scale <= scale_y <= policy.max_scale
            and rotation <= policy.max_rotation_degrees and shear <= policy.max_shear
            and translation <= policy.max_translation_px
        )

        # Query each immutable held-out landmark once; this avoids unrelated
        # current-frame keypoints dominating the residual distribution.
        held_pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(held_desc, cur_desc, k=2)
        held_matches = [
            first for pair in held_pairs if len(pair) == 2
            for first, second in [pair] if first.distance < 0.75 * second.distance
        ]
        held_p95 = 1000.0
        held_valid = False
        if len(held_matches) >= policy.min_held_out_matches:
            held_source = np.float32([cur_keys[item.trainIdx].pt for item in held_matches])
            held_destination = np.float32([held_keys[item.queryIdx].pt for item in held_matches])
            held_predicted = cv2.transform(held_source.reshape(1, -1, 2), transform).reshape(-1, 2)
            held_errors = np.linalg.norm(held_predicted - held_destination, axis=1)
            held_p95 = float(np.percentile(held_errors, 95))
            held_valid = held_p95 <= policy.max_held_out_reprojection_error_px
        # Deterministic duplicate gate.  For every raw alignment match, invert
        # the primary transform to predict its source position. A second copy of
        # the same target creates a large, coherent offset cluster. This avoids
        # a second randomized RANSAC and remains stable across process history.
        ambiguous = False
        raw_source = np.float32([cur_keys[item.queryIdx].pt for item in raw_matches])
        raw_destination = np.float32([ref_keys[item.trainIdx].pt for item in raw_matches])
        primary_3x3 = np.vstack((transform, (0.0, 0.0, 1.0)))
        try:
            inverse_primary = np.linalg.inv(primary_3x3)
        except np.linalg.LinAlgError:
            inverse_primary = None
        if inverse_primary is not None and len(raw_source) >= policy.min_matches:
            expected_source = cv2.perspectiveTransform(
                raw_destination.reshape(1, -1, 2), inverse_primary
            ).reshape(-1, 2)
            offsets = raw_source - expected_source
            bin_size = max(12.0, min(policy.canonical_width, policy.canonical_height) * 0.08)
            groups: dict[tuple[int, int], list[int]] = {}
            minimum_separation = min(policy.canonical_width, policy.canonical_height) * 0.35
            for index, offset in enumerate(offsets):
                if float(np.linalg.norm(offset)) < minimum_separation:
                    continue
                key = (int(round(float(offset[0]) / bin_size)), int(round(float(offset[1]) / bin_size)))
                groups.setdefault(key, []).append(index)

            secondary_pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(cur_desc, held_desc, k=2)
            secondary_matches = [
                first for pair in secondary_pairs if len(pair) == 2
                for first, second in [pair] if first.distance < 0.75 * second.distance
            ]
            held_offsets = np.empty((0, 2), dtype=np.float32)
            if secondary_matches:
                secondary_source = np.float32([cur_keys[item.queryIdx].pt for item in secondary_matches])
                secondary_destination = np.float32([held_keys[item.trainIdx].pt for item in secondary_matches])
                expected_held_source = cv2.perspectiveTransform(
                    secondary_destination.reshape(1, -1, 2), inverse_primary
                ).reshape(-1, 2)
                held_offsets = secondary_source - expected_held_source

            for indices in groups.values():
                unique_destination = {raw_matches[index].trainIdx for index in indices}
                if len(unique_destination) < policy.min_inliers:
                    continue
                cluster_center = np.median(offsets[indices], axis=0)
                cluster_destination = raw_destination[indices]
                cluster_hull = (
                    float(cv2.contourArea(cv2.convexHull(cluster_destination))) if len(indices) >= 3 else 0.0
                )
                held_support = int((np.linalg.norm(held_offsets - cluster_center, axis=1) <= bin_size * 1.5).sum())
                if (
                    len(unique_destination) / max(1, inlier_count) > policy.max_secondary_inlier_ratio
                    and cluster_hull / max(1.0, alignment_area) >= max(policy.min_coverage_ratio, 0.15)
                    and held_support >= max(4, policy.min_held_out_matches // 2)
                ):
                    ambiguous = True
                    break
        accepted = (
            inlier_count >= policy.min_inliers and ratio >= policy.min_inlier_ratio
            and coverage >= policy.min_coverage_ratio and reprojection_p95 <= policy.max_reprojection_error_px
            and bounded and not ambiguous and held_valid
        )
        reported_p95 = max(reprojection_p95, held_p95)
        observation = _alignment_observation(
            state="ALIGNED" if accepted else "NOT_ALIGNED", inliers=inlier_count, ratio=ratio,
            error=min(1000.0, reported_p95), coverage=coverage, within_bounds=bounded,
        )
        if not accepted:
            reasons = []
            if inlier_count < policy.min_inliers or ratio < policy.min_inlier_ratio:
                reasons.append("TARGET_INLIERS_INSUFFICIENT")
            if coverage < policy.min_coverage_ratio:
                reasons.append("TARGET_INLIERS_CLUSTERED")
            if reprojection_p95 > policy.max_reprojection_error_px:
                reasons.append("TARGET_REPROJECTION_ERROR")
            if not held_valid:
                reasons.append("TARGET_PARALLAX_OR_MISMATCH")
            if not bounded:
                reasons.append("TARGET_TRANSFORM_OUT_OF_BOUNDS")
            if ambiguous:
                reasons.append("TARGET_AMBIGUOUS")
            return NormalizedCapture(None, b"", tuple(reasons), observation)
        target = cv2.warpAffine(current, transform, (policy.canonical_width, policy.canonical_height))
        rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".png", target)
        if not ok:
            raise RuntimeError("TARGET_CANONICAL_ENCODING_FAILED")
        return NormalizedCapture(
            rgb, encoded.tobytes(), (), observation,
            target_from_input=np.asarray(transform, dtype=np.float64),
        )


class OpenCvCharucoPlaneNormalizer:
    """Still Gate and camera/plane normalization only; it never locates the target."""

    def normalize_plane(self, image: DecodedImage, artifact: ProductionArtifact) -> PlaneNormalizedCapture:
        import cv2
        import numpy as np

        try:
            with Image.open(BytesIO(image.data)) as source:
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                decoded = cv2.cvtColor(np.asarray(oriented), cv2.COLOR_RGB2BGR)
        except OSError as exc:
            raise RuntimeError("IMAGE_DECODE_FAILED") from exc
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        overexposure = float(np.count_nonzero(gray >= 250) / gray.size)
        reasons: list[str] = []
        if sharpness < artifact.still_gate.min_laplacian_variance:
            reasons.append("BLUR")
        if overexposure > artifact.still_gate.max_over_exposure_ratio:
            reasons.append("OVER_EXPOSURE")

        board = artifact.board
        dictionary_id = getattr(cv2.aruco, board.dictionary, None)
        if dictionary_id is None:
            raise RuntimeError("CHARUCO_DICTIONARY_UNSUPPORTED")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        board_arguments = (
            (board.squares_x, board.squares_y), board.square_length_mm,
            board.marker_length_mm, dictionary,
        )
        charuco = (
            cv2.aruco.CharucoBoard(*board_arguments, np.asarray(board.marker_ids, dtype=np.int32))
            if board.marker_ids is not None
            else cv2.aruco.CharucoBoard(*board_arguments)
        )
        detector = cv2.aruco.CharucoDetector(charuco)
        corners, ids, _, _ = detector.detectBoard(gray)
        count = 0 if ids is None else len(ids)
        if count < artifact.still_gate.min_charuco_corners:
            reasons.append("CHARUCO_CORNERS_INSUFFICIENT")
        if reasons:
            return PlaneNormalizedCapture(rgb=None, reason_codes=tuple(reasons))

        object_points = charuco.getChessboardCorners()[ids.flatten()][:, :2].astype(np.float32)
        pixels_per_mm_x = board.canonical_width / ((board.squares_x - 1) * board.square_length_mm)
        pixels_per_mm_y = board.canonical_height / ((board.squares_y - 1) * board.square_length_mm)
        destination = object_points.copy()
        destination[:, 0] *= pixels_per_mm_x
        destination[:, 1] *= pixels_per_mm_y
        transform, mask = cv2.findHomography(corners.reshape(-1, 2), destination, cv2.RANSAC, 3.0)
        if transform is None or mask is None or int(mask.sum()) < artifact.still_gate.min_charuco_corners:
            return PlaneNormalizedCapture(rgb=None, reason_codes=("CHARUCO_HOMOGRAPHY_INVALID",))
        selected = mask.reshape(-1).astype(bool)
        predicted = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2).astype(np.float32), transform,
        ).reshape(-1, 2)
        residuals = np.linalg.norm(predicted[selected] - destination[selected], axis=1)
        reprojection_p95 = float(np.percentile(residuals, 95)) if residuals.size else 1000.0
        canonical = cv2.warpPerspective(decoded, transform, (board.canonical_width, board.canonical_height))
        rgb = cv2.cvtColor(canonical, cv2.COLOR_BGR2RGB)
        return PlaneNormalizedCapture(
            rgb=rgb,
            metric_calibration=PlaneMetricCalibration(
                pixels_per_mm_x=float(pixels_per_mm_x),
                pixels_per_mm_y=float(pixels_per_mm_y),
                detected_corner_count=count,
                inlier_corner_count=int(mask.sum()),
                reprojection_error_px=reprojection_p95,
                calibration_fiducial="CHARUCO_CORNERS",
            ),
            source_rgb=cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB),
            input_to_plane=transform,
            calibration_support_plane=np.asarray((
                (0.0, 0.0),
                (float(board.canonical_width), 0.0),
                (float(board.canonical_width), float(board.canonical_height)),
                (0.0, float(board.canonical_height)),
            ), dtype=np.float32),
        )

    def normalize_golden_plane(
        self,
        image: DecodedImage,
        artifact: ProductionArtifact,
        candidates: list[GoldenDimensionBoardCandidate],
    ) -> PlaneNormalizedCapture:
        """Select one Server-qualified ChArUco profile by marker layout, never by QR."""
        if not candidates:
            return self.normalize_plane(image, artifact)
        import cv2
        import numpy as np

        try:
            with Image.open(BytesIO(image.data)) as source:
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                decoded = cv2.cvtColor(np.asarray(oriented), cv2.COLOR_RGB2BGR)
        except OSError as exc:
            raise RuntimeError("IMAGE_DECODE_FAILED") from exc
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        overexposure = float(np.count_nonzero(gray >= 250) / gray.size)
        still_reasons: list[str] = []
        if sharpness < artifact.still_gate.min_laplacian_variance:
            still_reasons.append("BLUR")
        if overexposure > artifact.still_gate.max_over_exposure_ratio:
            still_reasons.append("OVER_EXPOSURE")
        if still_reasons:
            return PlaneNormalizedCapture(rgb=None, reason_codes=tuple(still_reasons))

        qualified: list[tuple[GoldenDimensionBoardCandidate, object, float, int, int, float, float, str]] = []
        diagnostics: list[dict[str, object]] = []
        marker_detection_cache: dict[str, tuple[list[object], object | None]] = {}
        for candidate in candidates:
            dictionary_id = getattr(cv2.aruco, candidate.dictionary, None)
            if dictionary_id is None:
                diagnostics.append({
                    "profile": candidate.profile,
                    "boardId": candidate.board_id,
                    "revision": candidate.revision,
                    "reason": "DICTIONARY_UNSUPPORTED",
                })
                continue
            dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            diagnostic: dict[str, object] = {
                "profile": candidate.profile,
                "boardId": candidate.board_id,
                "revision": candidate.revision,
                "charucoGeometryQualified": candidate.charuco_geometry_qualified,
                "outerArucoGeometryQualified": candidate.outer_aruco_geometry_qualified,
            }
            candidate_fit: tuple[GoldenDimensionBoardCandidate, object, float, int, int, float, float, str] | None = None

            if candidate.charuco_geometry_qualified:
                charuco = cv2.aruco.CharucoBoard(
                    (candidate.squares_x, candidate.squares_y),
                    candidate.square_length_mm,
                    candidate.marker_length_mm,
                    dictionary,
                    np.asarray(candidate.marker_ids, dtype=np.int32),
                )
                corners, ids, _, _ = cv2.aruco.CharucoDetector(charuco).detectBoard(gray)
                count = 0 if ids is None else len(ids)
                diagnostic["charucoDetectedCornerCount"] = count
                if count >= artifact.still_gate.min_charuco_corners:
                    object_points = charuco.getChessboardCorners()[ids.flatten()][:, :2].astype(np.float32)
                    pixels_per_mm_x = artifact.board.canonical_width / (
                        (candidate.squares_x - 1) * candidate.square_length_mm
                    )
                    pixels_per_mm_y = artifact.board.canonical_height / (
                        (candidate.squares_y - 1) * candidate.square_length_mm
                    )
                    destination = object_points.copy()
                    destination[:, 0] *= pixels_per_mm_x
                    destination[:, 1] *= pixels_per_mm_y
                    transform, mask = cv2.findHomography(
                        corners.reshape(-1, 2), destination, cv2.RANSAC, 3.0,
                    )
                    if transform is not None and mask is not None:
                        inliers = int(mask.sum())
                        if inliers >= artifact.still_gate.min_charuco_corners:
                            selected = mask.reshape(-1).astype(bool)
                            predicted = cv2.perspectiveTransform(
                                corners.reshape(-1, 1, 2).astype(np.float32), transform,
                            ).reshape(-1, 2)
                            residuals = np.linalg.norm(predicted[selected] - destination[selected], axis=1)
                            reprojection_p95 = float(np.percentile(residuals, 95)) if residuals.size else 1000.0
                            candidate_fit = (
                                candidate, transform, reprojection_p95, count, inliers,
                                float(pixels_per_mm_x), float(pixels_per_mm_y), "CHARUCO_CORNERS",
                            )
                            diagnostic["charucoInlierCornerCount"] = inliers
                            diagnostic["charucoReprojectionP95Px"] = round(reprojection_p95, 4)

            if candidate.outer_aruco_geometry_qualified:
                if candidate.dictionary not in marker_detection_cache:
                    marker_corners, marker_ids, _ = cv2.aruco.ArucoDetector(
                        dictionary, cv2.aruco.DetectorParameters(),
                    ).detectMarkers(gray)
                    marker_detection_cache[candidate.dictionary] = (marker_corners, marker_ids)
                marker_corners, marker_ids = marker_detection_cache[candidate.dictionary]
                detected_markers = {
                    int(marker_id): np.asarray(corners, dtype=np.float32).reshape(4, 2)
                    for corners, marker_id in zip(
                        marker_corners,
                        [] if marker_ids is None else marker_ids.flatten(),
                    )
                }
                source_points: list[list[float]] = []
                physical_points: list[list[float]] = []
                matched_marker_ids: list[int] = []
                for marker in candidate.outer_markers:
                    detected = detected_markers.get(marker.id)
                    if detected is None:
                        continue
                    matched_marker_ids.append(marker.id)
                    source_points.extend(detected.tolist())
                    physical_points.extend([corner[:2] for corner in marker.corners_mm])
                diagnostic["outerDetectedMarkerIds"] = matched_marker_ids
                if (
                    len(matched_marker_ids) >= 3
                    and candidate.finished_width_mm is not None
                    and candidate.finished_height_mm is not None
                ):
                    scale = min(
                        artifact.board.canonical_width / candidate.finished_width_mm,
                        artifact.board.canonical_height / candidate.finished_height_mm,
                    )
                    offset_x = (artifact.board.canonical_width - candidate.finished_width_mm * scale) / 2.0
                    offset_y = (artifact.board.canonical_height - candidate.finished_height_mm * scale) / 2.0
                    source = np.asarray(source_points, dtype=np.float32)
                    physical = np.asarray(physical_points, dtype=np.float32)
                    destination = physical * float(scale)
                    destination[:, 0] += float(offset_x)
                    destination[:, 1] += float(offset_y)
                    outer_transform, outer_mask = cv2.findHomography(
                        source, destination, cv2.RANSAC, 3.0,
                    )
                    if outer_transform is not None and outer_mask is not None:
                        outer_inliers = int(outer_mask.sum())
                        minimum_outer_inliers = max(8, artifact.still_gate.min_charuco_corners)
                        if outer_inliers >= minimum_outer_inliers:
                            selected = outer_mask.reshape(-1).astype(bool)
                            predicted = cv2.perspectiveTransform(
                                source.reshape(-1, 1, 2), outer_transform,
                            ).reshape(-1, 2)
                            residuals = np.linalg.norm(predicted[selected] - destination[selected], axis=1)
                            outer_reprojection_p95 = (
                                float(np.percentile(residuals, 95)) if residuals.size else 1000.0
                            )
                            diagnostic["outerInlierCornerCount"] = outer_inliers
                            diagnostic["outerReprojectionP95Px"] = round(outer_reprojection_p95, 4)
                            if candidate_fit is None:
                                candidate_fit = (
                                    candidate, outer_transform, outer_reprojection_p95,
                                    len(source_points), outer_inliers, float(scale), float(scale),
                                    "OUTER_ARUCO_CORNERS",
                                )

            if candidate_fit is not None:
                diagnostic["selectedFiducial"] = candidate_fit[7]
                qualified.append(candidate_fit)
            diagnostics.append(diagnostic)

        if not qualified:
            return PlaneNormalizedCapture(
                rgb=None,
                reason_codes=("CHARUCO_CORNERS_INSUFFICIENT",),
                calibration_diagnostics=tuple(diagnostics),
            )
        qualified.sort(key=lambda item: (-item[4], item[2], -item[3], item[0].profile))
        best = qualified[0]
        if len(qualified) > 1:
            second = qualified[1]
            # A nearly tied fit means a partial view can satisfy more than one
            # legal layout. Requiring a clearly stronger inlier set prevents
            # silently guessing small versus large board geometry.
            if best[4] <= second[4] + 1 and best[2] >= second[2] * 0.5:
                return PlaneNormalizedCapture(
                    rgb=None,
                    reason_codes=("CHARUCO_BOARD_PROFILE_AMBIGUOUS",),
                    calibration_diagnostics=tuple(diagnostics),
                )

        candidate, transform, reprojection_p95, count, inliers, pixels_per_mm_x, pixels_per_mm_y, fiducial = best
        if fiducial == "OUTER_ARUCO_CORNERS":
            # The outer-marker normalization deliberately letterboxes the
            # physical board.  Preserve that exact support instead of treating
            # all 896x896 pixels as calibrated extrapolation space.
            assert candidate.finished_width_mm is not None
            assert candidate.finished_height_mm is not None
            support_scale = min(
                artifact.board.canonical_width / candidate.finished_width_mm,
                artifact.board.canonical_height / candidate.finished_height_mm,
            )
            support_left = (artifact.board.canonical_width - candidate.finished_width_mm * support_scale) / 2.0
            support_top = (artifact.board.canonical_height - candidate.finished_height_mm * support_scale) / 2.0
            support_right = support_left + candidate.finished_width_mm * support_scale
            support_bottom = support_top + candidate.finished_height_mm * support_scale
        else:
            support_left, support_top = 0.0, 0.0
            support_right = float(artifact.board.canonical_width)
            support_bottom = float(artifact.board.canonical_height)
        canonical = cv2.warpPerspective(
            decoded, transform, (artifact.board.canonical_width, artifact.board.canonical_height),
        )
        return PlaneNormalizedCapture(
            rgb=cv2.cvtColor(canonical, cv2.COLOR_BGR2RGB),
            metric_calibration=PlaneMetricCalibration(
                pixels_per_mm_x=pixels_per_mm_x,
                pixels_per_mm_y=pixels_per_mm_y,
                detected_corner_count=count,
                inlier_corner_count=inliers,
                reprojection_error_px=reprojection_p95,
                calibration_fiducial=fiducial,
            ),
            source_rgb=cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB),
            input_to_plane=transform,
            calibration_board=candidate,
            calibration_fiducial=fiducial,
            calibration_support_plane=np.asarray((
                (support_left, support_top),
                (support_right, support_top),
                (support_right, support_bottom),
                (support_left, support_bottom),
            ), dtype=np.float32),
            calibration_diagnostics=tuple(diagnostics),
        )


class OpenCvCharucoNormalizer:
    """Composes board-plane normalization with independent target alignment."""

    def __init__(
        self, target_aligner: TargetAligner | None = None, *,
        allow_target_only_alignment: bool = False,
        allow_contour_anchor_alignment: bool = False,
    ):
        self._plane = OpenCvCharucoPlaneNormalizer()
        self._target = target_aligner or OpenCvTargetAligner(
            allow_contour_anchor_alignment=allow_contour_anchor_alignment,
        )
        self._allow_target_only_alignment = allow_target_only_alignment

    def normalize(
        self,
        image: DecodedImage,
        artifact: ProductionArtifact,
        board_candidates: list[GoldenDimensionBoardCandidate] | None = None,
    ) -> NormalizedCapture:
        plane = (
            self._plane.normalize_golden_plane(image, artifact, board_candidates)
            if board_candidates
            else self._plane.normalize_plane(image, artifact)
        )
        if plane.reason_codes:
            # ChArUco is preferred for plane normalization, but it is not a
            # mandatory capture element. If the board is outside the frame,
            # target-relative alignment may still be safe when the immutable
            # target reference/held-out gates pass. Blur and exposure failures
            # remain fail-closed.
            board_only_failures = {"CHARUCO_CORNERS_INSUFFICIENT", "CHARUCO_HOMOGRAPHY_INVALID"}
            if self._allow_target_only_alignment and set(plane.reason_codes).issubset(board_only_failures):
                import cv2
                import numpy as np
                with Image.open(BytesIO(image.data)) as source:
                    oriented = ImageOps.exif_transpose(source).convert("RGB")
                    raw_rgb = np.asarray(oriented, dtype=np.uint8)
                return self._target.align(raw_rgb, artifact)
            return NormalizedCapture(None, b"", plane.reason_codes)
        import numpy as np

        aligned = self._target.align(plane.rgb, artifact)
        input_to_plane = np.eye(3, dtype=np.float64)
        target_input_is_source = False
        if aligned.reason_codes and self._allow_target_only_alignment and plane.source_rgb is not None:
            raw_aligned = self._target.align(plane.source_rgb, artifact)
            if raw_aligned.reason_codes:
                return raw_aligned
            aligned = raw_aligned
            if plane.input_to_plane is None:
                return aligned
            input_to_plane = np.asarray(plane.input_to_plane, dtype=np.float64)
            if input_to_plane.shape != (3, 3):
                return aligned
            target_input_is_source = True
        calibration = plane.metric_calibration
        if aligned.reason_codes or calibration is None or aligned.target_from_input is None:
            return aligned

        target_from_input = np.asarray(aligned.target_from_input, dtype=np.float64)
        if target_from_input.shape == (2, 3):
            target_from_input = np.vstack((target_from_input, (0.0, 0.0, 1.0)))
        if target_from_input.shape != (3, 3):
            return aligned
        try:
            target_to_plane = input_to_plane @ np.linalg.inv(target_from_input)
            target_from_source = (
                target_from_input
                if target_input_is_source
                else target_from_input @ input_to_plane
            )
        except np.linalg.LinAlgError:
            return aligned
        return replace(
            aligned,
            metric_calibration=TargetMetricCalibration(
                target_to_plane=target_to_plane,
                pixels_per_mm_x=calibration.pixels_per_mm_x,
                pixels_per_mm_y=calibration.pixels_per_mm_y,
                detected_corner_count=calibration.detected_corner_count,
                inlier_corner_count=calibration.inlier_corner_count,
                reprojection_error_px=calibration.reprojection_error_px,
                calibration_fiducial=calibration.calibration_fiducial,
            ),
            source_rgb=plane.source_rgb,
            source_to_plane=input_to_plane,
            target_from_source=target_from_source,
            calibration_support_plane=plane.calibration_support_plane,
            calibration_board=plane.calibration_board,
            calibration_fiducial=plane.calibration_fiducial,
        )


class DinoV2Embedder:
    def __init__(self, adapter: LocalDinoV2Adapter, device: str = "cpu"):
        self._adapter = adapter
        self._device = device
        self._model: object | None = None
        self._lock = Lock()

    def warm_up(self) -> None:
        """Load weights into memory ahead of the first request."""
        with self._lock:
            if self._model is None:
                self._model = self._adapter.smoke_load().to(self._device)

    def _transform(self, rgb: object) -> object:
        import numpy as np
        import torch
        from torchvision.transforms import v2

        pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        return v2.Compose([
            v2.Resize(DINO_RESIZE_SHORT_EDGE, antialias=True), v2.CenterCrop(DINO_INPUT_SIZE), v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])(pil).unsqueeze(0).to(self._device)

    def embed(self, rgb: object) -> list[float]:
        import torch

        with self._lock:
            if self._model is None:
                self._model = self._adapter.smoke_load().to(self._device)
            tensor = self._transform(rgb)
            with torch.inference_mode():
                value = self._model(tensor).detach().cpu().reshape(-1)
        return [float(item) for item in value]

    def embed_with_patches(self, rgb: object) -> PatchEmbedding:
        import torch

        with self._lock:
            if self._model is None:
                self._model = self._adapter.smoke_load().to(self._device)
            tensor = self._transform(rgb)
            with torch.inference_mode():
                features = self._model.forward_features(tensor)
                cls = features["x_norm_clstoken"].detach().cpu().reshape(-1)
                patches = features["x_norm_patchtokens"].detach().cpu()[0]
        patch_size = getattr(self._model, "patch_size", 14)
        grid = DINO_INPUT_SIZE // patch_size
        if patches.shape[0] != grid * grid:
            raise RuntimeError("DINO_PATCH_GRID_UNEXPECTED")
        return PatchEmbedding(
            global_vector=[float(item) for item in cls],
            patch_grid=[[float(item) for item in row] for row in patches],
            grid_height=grid, grid_width=grid,
        )


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError("EMBEDDING_DIMENSION_MISMATCH")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise RuntimeError("EMBEDDING_NORM_INVALID")
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(0.0, min(2.0, 1.0 - similarity))


def _dino_input_crop_box(canonical_width: int, canonical_height: int) -> tuple[float, float, float, float]:
    """Canonical-pixel (left, top, width, height) that Resize+CenterCrop feeds the model.

    Mirrors torchvision v2.Resize(256) [shorter edge] + v2.CenterCrop(224) exactly, so
    spatial evidence never claims coverage of canonical-ROI pixels the model never saw.
    """
    scale = DINO_RESIZE_SHORT_EDGE / min(canonical_width, canonical_height)
    resized_width, resized_height = canonical_width * scale, canonical_height * scale
    left = max(0.0, (resized_width - DINO_INPUT_SIZE) / 2.0) / scale
    top = max(0.0, (resized_height - DINO_INPUT_SIZE) / 2.0) / scale
    width = min(canonical_width - left, DINO_INPUT_SIZE / scale)
    height = min(canonical_height - top, DINO_INPUT_SIZE / scale)
    return left, top, width, height


def _unavailable_spatial_evidence(reason_code: str) -> SpatialDifferenceEvidence:
    return SpatialDifferenceEvidence(state="UNAVAILABLE", disclaimerCode="DIFFERENCE_NOT_DEFECT_PROOF", reasonCode=reason_code)


def _overlaps(left: float, top: float, width: float, height: float, region) -> bool:
    return (
        left < region.x + region.width and region.x < left + width
        and top < region.y + region.height and region.y < top + height
    )


def _patch_distance_grid(current: PatchEmbedding, golden: PatchEmbedding):
    """Return the raw cosine-distance patch grid shared by both evidence paths."""
    import numpy as np

    if current.grid_height != golden.grid_height or current.grid_width != golden.grid_width:
        raise RuntimeError("GOLDEN_PATCH_FEATURES_UNAVAILABLE")
    current_values = np.asarray(current.patch_grid, dtype=np.float64)
    golden_values = np.asarray(golden.patch_grid, dtype=np.float64)
    if current_values.shape != golden_values.shape:
        raise RuntimeError("GOLDEN_PATCH_FEATURES_UNAVAILABLE")
    current_norm = np.linalg.norm(current_values, axis=1)
    golden_norm = np.linalg.norm(golden_values, axis=1)
    valid = (current_norm > 1e-12) & (golden_norm > 1e-12)
    similarity = np.zeros(current_values.shape[0], dtype=np.float64)
    similarity[valid] = (
        np.sum(current_values[valid] * golden_values[valid], axis=1)
        / (current_norm[valid] * golden_norm[valid])
    )
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    distance[~valid] = 0.0
    return distance.reshape(current.grid_height, current.grid_width).astype(np.float32)


def _axis_tile_starts(region_start: float, region_length: float, analyzed_length: float) -> list[float]:
    """Cover one ROI axis with overlapping analyzed windows, including both ends."""
    if region_length <= analyzed_length:
        return [region_start + (region_length - analyzed_length) / 2.0]
    final = region_start + region_length - analyzed_length
    stride = analyzed_length * 0.70
    starts = [region_start]
    while starts[-1] + stride < final:
        starts.append(starts[-1] + stride)
    if final - starts[-1] > 1e-6:
        starts.append(final)
    return starts


def _roi_tile_boxes(inspection_regions, canonical_width: int, canonical_height: int) -> list[tuple[int, int, int]]:
    """Build deterministic square tiles whose DINO center-crops cover every ROI."""
    boxes: set[tuple[int, int, int]] = set()
    maximum_side = min(384, canonical_width, canonical_height)
    for region in inspection_regions:
        side = int(math.ceil(min(maximum_side, max(224.0, min(region.width, region.height) * 1.25))))
        _, _, analyzed_width, analyzed_height = _dino_input_crop_box(side, side)
        margin_x = (side - analyzed_width) / 2.0
        margin_y = (side - analyzed_height) / 2.0
        for analyzed_y in _axis_tile_starts(float(region.y), float(region.height), analyzed_height):
            for analyzed_x in _axis_tile_starts(float(region.x), float(region.width), analyzed_width):
                tile_x = int(round(analyzed_x - margin_x))
                tile_y = int(round(analyzed_y - margin_y))
                tile_x = max(0, min(canonical_width - side, tile_x))
                tile_y = max(0, min(canonical_height - side, tile_y))
                boxes.add((tile_x, tile_y, side))
    return sorted(boxes, key=lambda item: (item[1], item[0], item[2]))


def _scorer_input_tiles(
    rgb: object,
    inspection_regions,
    canonical_width: int,
    canonical_height: int,
    analysis_mask: object,
    neutral_rgb: tuple[int, int, int] = (127, 127, 127),
) -> list[ScorerInputTile]:
    """Create deterministic inputs whose only non-neutral pixels are in scope."""
    import numpy as np

    image = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(analysis_mask, dtype=np.uint8) > 0
    if image.shape[:2] != (canonical_height, canonical_width) or mask.shape != (canonical_height, canonical_width):
        raise RuntimeError("SCORER_INPUT_CANONICAL_DIMENSIONS_MISMATCH")
    tiles: list[ScorerInputTile] = []
    for index, (x, y, side) in enumerate(_roi_tile_boxes(inspection_regions, canonical_width, canonical_height), start=1):
        tile = image[y:y + side, x:x + side].copy()
        tile_mask = mask[y:y + side, x:x + side]
        tile[~tile_mask] = np.asarray(neutral_rgb, dtype=np.uint8)
        identity = hashlib.sha256()
        identity.update(f"{x}:{y}:{side}:RGB8:".encode("ascii"))
        identity.update(tile.tobytes(order="C"))
        tiles.append(ScorerInputTile(
            id=f"ROI-TILE-{index:03d}", x=x, y=y, side=side, rgb=tile,
            sha256="sha256:" + identity.hexdigest(),
        ))
    if not tiles:
        raise RuntimeError("SCORER_INPUT_TILES_EMPTY")
    return tiles


def _scorer_input_digest(tiles: list[ScorerInputTile], contract_digest: str) -> str:
    payload = {
        "contractDigest": contract_digest,
        "tiles": [
            {"id": tile.id, "x": tile.x, "y": tile.y, "side": tile.side, "tileSha256": tile.sha256}
            for tile in tiles
        ],
    }
    encoded = __import__("json").dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _mean_embedding(vectors: list[list[float]]) -> list[float]:
    if not vectors or len({len(vector) for vector in vectors}) != 1:
        raise RuntimeError("SCORER_INPUT_EMBEDDING_DIMENSION_MISMATCH")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(vectors[0]))]


def _encode_binary_png(mask: object) -> tuple[bytes, str, str]:
    import cv2
    import numpy as np

    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    ok, encoded = cv2.imencode(".png", binary)
    if not ok:
        raise RuntimeError("BINARY_MASK_ENCODING_FAILED")
    payload = encoded.tobytes()
    return payload, base64.b64encode(payload).decode("ascii"), hashlib.sha256(payload).hexdigest()


def _subject_scope(
    artifact: ProductionArtifactV13,
    nearest_golden: GoldenEmbedding,
) -> SubjectScope:
    """Materialize the immutable Golden mask and its padded runtime support."""
    import cv2
    import numpy as np

    contract = require_subject_segmentation(artifact)
    bound_mask = next((item for item in contract.golden_masks if item.golden_id == nearest_golden.id), None)
    if bound_mask is None:
        raise RuntimeError("GOLDEN_SUBJECT_MASK_NOT_FOUND")
    encoded = base64.b64decode(bound_mask.mask_png_base64, validate=True)
    core = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    expected_shape = (contract.canonical_height, contract.canonical_width)
    if core is None or core.shape != expected_shape:
        raise RuntimeError("GOLDEN_SUBJECT_MASK_INVALID")
    core = (core > 0).astype(np.uint8) * 255
    inspection = np.asarray(inspection_roi_image(artifact.inspection_roi), dtype=np.uint8)

    if contract.support_padding_px > 0:
        diameter = contract.support_padding_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        support = cv2.dilate(core, kernel)
    else:
        support = core.copy()
    support[inspection == 0] = 0

    if contract.boundary_band_px > 0:
        diameter = contract.boundary_band_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        outer = cv2.dilate(core, kernel)
        inner = cv2.erode(core, kernel)
        boundary = cv2.subtract(outer, inner)
    else:
        boundary = np.zeros_like(core)
    boundary[inspection == 0] = 0

    # Preserve the artifact bytes exactly. Re-encoding an identical pixel mask
    # can change PNG bytes and would break the immutable Golden-mask pin.
    subject_b64 = bound_mask.mask_png_base64
    subject_sha = bound_mask.mask_sha256.removeprefix("sha256:")
    _, support_b64, support_sha = _encode_binary_png(support)
    _, boundary_b64, boundary_sha = _encode_binary_png(boundary)
    inspection_area = max(1, int(np.count_nonzero(inspection)))
    support_ratio = float(np.count_nonzero(support)) / inspection_area
    background_suppressed_ratio = 1.0 - support_ratio
    prompt = bound_mask.prompt_region_normalized
    evidence = SubjectSegmentationEvidence(
        state="AVAILABLE",
        disclaimerCode="SUBJECT_MASK_NOT_DEFECT_PROOF",
        method=contract.method,
        usageMode=contract.usage_mode,
        goldenId=bound_mask.golden_id,
        modelRepositoryVersion=contract.model_repository_version,
        modelWeightsSha256=contract.model_weights_sha256,
        promptRegionNormalized=BboxNormalized(**prompt),
        foregroundRatio=bound_mask.foreground_ratio,
        supportRatio=max(0.0, min(1.0, support_ratio)),
        backgroundSuppressedRatio=max(0.0, min(1.0, background_suppressed_ratio)),
        subjectMaskPngBase64=subject_b64,
        subjectMaskSha256=subject_sha,
        supportMaskPngBase64=support_b64,
        supportMaskSha256=support_sha,
        boundaryMaskPngBase64=boundary_b64,
        boundaryMaskSha256=boundary_sha,
    )
    return SubjectScope(core_mask=core, support_mask=support, boundary_mask=boundary, evidence=evidence)


def _boundary_difference_evidence(
    golden_mask: object,
    current_mask: object,
    *,
    canonical_width: int,
    canonical_height: int,
    min_region_area_ratio: float,
    max_regions: int,
) -> BoundaryDifferenceEvidence:
    """Measure aligned mask geometry without mixing it into DINO evidence."""
    import cv2
    import numpy as np

    golden = np.asarray(golden_mask, dtype=np.uint8) > 0
    current = np.asarray(current_mask, dtype=np.uint8) > 0
    golden_area = max(1, int(np.count_nonzero(golden)))
    current_area = int(np.count_nonzero(current))
    intersection = int(np.count_nonzero(golden & current))
    union = max(1, int(np.count_nonzero(golden | current)))
    missing = golden & ~current
    protruding = current & ~golden
    difference = missing | protruding

    kernel = np.ones((3, 3), dtype=np.uint8)
    golden_edge = cv2.morphologyEx(golden.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    current_edge = cv2.morphologyEx(current.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    distances: list[float] = []
    if np.any(golden_edge) and np.any(current_edge):
        distance_to_current = cv2.distanceTransform((~current_edge).astype(np.uint8), cv2.DIST_L2, 3)
        distance_to_golden = cv2.distanceTransform((~golden_edge).astype(np.uint8), cv2.DIST_L2, 3)
        distances = np.concatenate((
            distance_to_current[golden_edge], distance_to_golden[current_edge],
        )).astype(float).tolist()
    elif np.any(golden_edge) or np.any(current_edge):
        distances = [float(max(canonical_width, canonical_height))]

    regions: list[tuple[int, BoundaryDifferenceRegion]] = []
    minimum_area = max(1, int(math.ceil(min_region_area_ratio * golden_area)))
    for change_type, changed in (
        ("MISSING_FROM_CURRENT", missing),
        ("PROTRUDING_FROM_CURRENT", protruding),
    ):
        count, _, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            regions.append((area, BoundaryDifferenceRegion(
                id="B-PLACEHOLDER",
                bboxNormalized=BboxNormalized(
                    x=x / canonical_width,
                    y=y / canonical_height,
                    width=min(width / canonical_width, 1.0 - x / canonical_width),
                    height=min(height / canonical_height, 1.0 - y / canonical_height),
                ),
                changeType=change_type,
                areaRatio=min(1.0, area / golden_area),
            )))
    regions.sort(key=lambda item: item[0], reverse=True)
    selected_regions = [
        region.model_copy(update={"id": f"B-{index + 1:03d}"})
        for index, (_, region) in enumerate(regions[:max_regions])
    ]
    _, mask_b64, mask_sha = _encode_binary_png(difference)
    distance_values = np.asarray(distances or [0.0], dtype=np.float64)
    return BoundaryDifferenceEvidence(
        state="AVAILABLE",
        disclaimerCode="BOUNDARY_GEOMETRY_NOT_DEFECT_PROOF",
        method="ALIGNED_SUBJECT_MASK_GEOMETRY_V1",
        maskIntersectionOverUnion=intersection / union,
        areaDeltaRatio=min(1.0, abs(current_area - golden_area) / golden_area),
        missingAreaRatio=min(1.0, float(np.count_nonzero(missing)) / golden_area),
        protrudingAreaRatio=min(1.0, float(np.count_nonzero(protruding)) / golden_area),
        meanContourDistancePx=float(distance_values.mean()),
        p95ContourDistancePx=float(np.percentile(distance_values, 95)),
        maxContourDistancePx=float(distance_values.max()),
        regions=selected_regions,
        maskPngBase64=mask_b64,
        maskSha256=mask_sha,
    )


def _paired_subject_scope(
    artifact: ProductionArtifactV16,
    nearest_golden: GoldenEmbedding,
    current_prediction: SubjectMaskPrediction,
) -> PairedSubjectResult:
    """Build erode(Golden ∩ Current) and fail closed on unreliable masks."""
    import cv2
    import numpy as np

    golden_scope = _subject_scope(artifact, nearest_golden)
    golden = np.asarray(golden_scope.core_mask, dtype=np.uint8) > 0
    current = np.asarray(current_prediction.mask, dtype=bool)
    expected_shape = (artifact.target_alignment.canonical_height, artifact.target_alignment.canonical_width)
    if current.shape != expected_shape:
        return PairedSubjectResult(None, None, "CURRENT_SUBJECT_MASK_DIMENSIONS_INVALID")
    inspection = np.asarray(inspection_roi_image(artifact.inspection_roi), dtype=np.uint8) > 0
    current &= inspection
    golden_area = max(1, int(np.count_nonzero(golden)))
    current_area = int(np.count_nonzero(current))
    intersection = golden & current
    union_area = max(1, int(np.count_nonzero(golden | current)))
    mask_iou = float(np.count_nonzero(intersection)) / union_area
    area_delta = abs(current_area - golden_area) / golden_area

    erosion = artifact.current_subject_segmentation.interior_erosion_px
    diameter = erosion * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    interior = cv2.erode(intersection.astype(np.uint8) * 255, kernel)
    interior[~inspection] = 0
    interior_ratio = float(np.count_nonzero(interior)) / golden_area
    boundary = _boundary_difference_evidence(
        golden, current,
        canonical_width=artifact.target_alignment.canonical_width,
        canonical_height=artifact.target_alignment.canonical_height,
        min_region_area_ratio=artifact.current_subject_segmentation.boundary_min_region_area_ratio,
        max_regions=artifact.current_subject_segmentation.max_boundary_regions,
    )
    policy = artifact.current_subject_segmentation
    if mask_iou < policy.min_mask_iou:
        return PairedSubjectResult(None, boundary, "CURRENT_SUBJECT_MASK_IOU_BELOW_POLICY")
    if area_delta > policy.max_area_delta_ratio:
        return PairedSubjectResult(None, boundary, "CURRENT_SUBJECT_MASK_AREA_DELTA_ABOVE_POLICY")
    if interior_ratio < policy.min_interior_ratio or not np.any(interior):
        return PairedSubjectResult(None, boundary, "PAIRED_SUBJECT_INTERIOR_BELOW_POLICY")

    _, current_b64, current_sha = _encode_binary_png(current)
    _, interior_b64, interior_sha = _encode_binary_png(interior)
    _, empty_boundary_b64, empty_boundary_sha = _encode_binary_png(np.zeros_like(interior))
    inspection_area = max(1, int(np.count_nonzero(inspection)))
    prompt = golden_scope.evidence.prompt_region_normalized
    if prompt is None:
        raise RuntimeError("GOLDEN_SUBJECT_PROMPT_NOT_FOUND")
    evidence = SubjectSegmentationEvidence(
        state="AVAILABLE",
        disclaimerCode="SUBJECT_MASK_NOT_DEFECT_PROOF",
        method=golden_scope.evidence.method,
        usageMode=golden_scope.evidence.usage_mode,
        goldenId=nearest_golden.id,
        modelRepositoryVersion=golden_scope.evidence.model_repository_version,
        modelWeightsSha256=golden_scope.evidence.model_weights_sha256,
        promptRegionNormalized=prompt,
        foregroundRatio=golden_scope.evidence.foreground_ratio,
        supportRatio=float(np.count_nonzero(interior)) / inspection_area,
        backgroundSuppressedRatio=1.0 - float(np.count_nonzero(interior)) / inspection_area,
        subjectMaskPngBase64=golden_scope.evidence.subject_mask_png_base64,
        subjectMaskSha256=golden_scope.evidence.subject_mask_sha256,
        supportMaskPngBase64=interior_b64,
        supportMaskSha256=interior_sha,
        boundaryMaskPngBase64=empty_boundary_b64,
        boundaryMaskSha256=empty_boundary_sha,
        currentForegroundRatio=current_prediction.foreground_ratio,
        maskIntersectionOverUnion=mask_iou,
        interiorRatio=interior_ratio,
        currentSubjectMaskPngBase64=current_b64,
        currentSubjectMaskSha256=current_sha,
        interiorMaskPngBase64=interior_b64,
        interiorMaskSha256=interior_sha,
    )
    return PairedSubjectResult(
        SubjectScope(
            core_mask=interior,
            support_mask=interior,
            boundary_mask=np.zeros_like(interior),
            evidence=evidence,
            paired_interior=True,
        ),
        boundary,
    )


def _unavailable_physical_dimensions(reason_code: str) -> PhysicalDimensionEvidence:
    return PhysicalDimensionEvidence(
        state="UNAVAILABLE",
        disclaimerCode="ENGINEERING_DIMENSION_NOT_METROLOGY_PROOF",
        reasonCode=reason_code,
    )


def _unavailable_candidate_dimensions(reason_code: str) -> CandidatePhysicalDimensionEvidence:
    return CandidatePhysicalDimensionEvidence(
        state="UNAVAILABLE",
        disclaimerCode="CANDIDATE_DIMENSION_ESTIMATE_NOT_DEFECT_PROOF",
        reasonCode=reason_code,
    )


def _candidate_component_contours(
    mask: object,
    regions: list[DifferenceRegion],
    canonical_width: int,
    canonical_height: int,
) -> dict[str, object]:
    """Bind retained regions back to their exact connected candidate components."""
    import cv2
    import numpy as np

    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    unused = set(range(1, count))
    result: dict[str, object] = {}
    for region in regions:
        expected = (
            int(round(region.bbox_normalized.x * canonical_width)),
            int(round(region.bbox_normalized.y * canonical_height)),
            int(round(region.bbox_normalized.width * canonical_width)),
            int(round(region.bbox_normalized.height * canonical_height)),
        )
        candidates = sorted(
            unused,
            key=lambda label: sum(
                abs(int(stats[label, field]) - expected[field]) for field in range(4)
            ),
        )
        if not candidates:
            continue
        label = candidates[0]
        observed = tuple(int(stats[label, field]) for field in range(4))
        if any(abs(observed[field] - expected[field]) > 2 for field in range(4)):
            continue
        unused.remove(label)
        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [item for item in contours if len(item) >= 4 and cv2.contourArea(item) > 0]
        if contours:
            result[region.id] = max(contours, key=cv2.contourArea)
    return result


def _candidate_physical_dimensions(
    artifact: ProductionArtifactV18,
    _request: AnalyzeRequest,
    normalized: NormalizedCapture,
    current_prediction: SubjectMaskPrediction | None,
    spatial: SpatialDifferenceEvidence,
) -> None:
    """Attach per-candidate dimensions only on the qualified ChArUco plane.

    Difference regions are still parts of the photographed subject.  A Golden
    ratio therefore cannot repair a missing plane calibration: it would repeat
    the same foreground/background parallax mistake that invalidated the
    legacy whole-subject readings.  Candidate millimetres use the same true
    ChArUco/support boundary as whole-subject millimetres.
    """
    import cv2
    import numpy as np

    regions = spatial.regions or []
    if not regions:
        return
    unavailable_reason: str | None = None
    if spatial.mask_png_base64 is None or spatial.mask_sha256 is None:
        unavailable_reason = "CANDIDATE_MASK_REQUIRED"
    elif current_prediction is None:
        unavailable_reason = "CURRENT_SUBJECT_SEGMENTATION_REQUIRED"
    alignment = normalized.alignment
    if unavailable_reason is None and (
        alignment is None or alignment.state != "ALIGNED"
        or not alignment.transform_within_bounds or not alignment.inspection_mask_applied
    ):
        unavailable_reason = "TARGET_ALIGNMENT_UNQUALIFIED"
    if unavailable_reason is not None:
        for region in regions:
            region.physical_dimensions = _unavailable_candidate_dimensions(unavailable_reason)
        return

    encoded_mask = base64.b64decode(spatial.mask_png_base64, validate=True)
    candidate_mask = cv2.imdecode(np.frombuffer(encoded_mask, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    expected_shape = (artifact.target_alignment.canonical_height, artifact.target_alignment.canonical_width)
    if candidate_mask is None or candidate_mask.shape != expected_shape:
        for region in regions:
            region.physical_dimensions = _unavailable_candidate_dimensions("CANDIDATE_MASK_DIMENSIONS_INVALID")
        return
    contours = _candidate_component_contours(
        candidate_mask, regions, artifact.target_alignment.canonical_width,
        artifact.target_alignment.canonical_height,
    )
    policy = artifact.dimension_measurement_policy
    calibration = normalized.metric_calibration

    metric_from_target = None
    direct_edge_uncertainty_mm = None
    target_to_plane = None
    support = None
    if calibration is None:
        unavailable_reason = "CANDIDATE_CHARUCO_CALIBRATION_REQUIRED"
    elif calibration.calibration_fiducial != "CHARUCO_CORNERS":
        unavailable_reason = "CHARUCO_CORNERS_REQUIRED_FOR_CANDIDATE_METRICS"
    elif calibration.reprojection_error_px > policy.max_plane_reprojection_error_px:
        unavailable_reason = "CALIBRATION_REPROJECTION_ERROR_ABOVE_POLICY"
    else:
        target_to_plane = np.asarray(calibration.target_to_plane, dtype=np.float64)
        support = np.asarray(normalized.calibration_support_plane, dtype=np.float32)
        if target_to_plane.shape != (3, 3) or not np.all(np.isfinite(target_to_plane)) \
                or abs(float(np.linalg.det(target_to_plane))) <= 1e-12:
            unavailable_reason = "METRIC_TRANSFORM_INVALID"
        elif support.shape != (4, 2) or not np.all(np.isfinite(support)) \
                or abs(float(cv2.contourArea(support))) <= 1.0:
            unavailable_reason = "CALIBRATION_SUPPORT_REQUIRED"
        else:
            plane_to_mm = np.diag((
                1.0 / calibration.pixels_per_mm_x,
                1.0 / calibration.pixels_per_mm_y,
                1.0,
            ))
            metric_from_target = plane_to_mm @ target_to_plane
            if metric_from_target.shape != (3, 3) or not np.all(np.isfinite(metric_from_target)) \
                    or abs(float(np.linalg.det(metric_from_target))) <= 1e-12:
                unavailable_reason = "METRIC_TRANSFORM_INVALID"
            else:
                center = np.asarray([[[
                    artifact.target_alignment.canonical_width / 2.0,
                    artifact.target_alignment.canonical_height / 2.0,
                ], [
                    artifact.target_alignment.canonical_width / 2.0 + 1.0,
                    artifact.target_alignment.canonical_height / 2.0,
                ], [
                    artifact.target_alignment.canonical_width / 2.0,
                    artifact.target_alignment.canonical_height / 2.0 + 1.0,
                ]]], dtype=np.float32)
                probe = cv2.perspectiveTransform(center, metric_from_target).reshape(-1, 2)
                mm_per_target_px = max(
                    float(np.linalg.norm(probe[1] - probe[0])),
                    float(np.linalg.norm(probe[2] - probe[0])),
                )
                direct_edge_uncertainty_mm = (
                    calibration.reprojection_error_px
                    / min(calibration.pixels_per_mm_x, calibration.pixels_per_mm_y)
                    + policy.segmentation_boundary_uncertainty_px * mm_per_target_px
                )

    for region in regions:
        if unavailable_reason is not None:
            region.physical_dimensions = _unavailable_candidate_dimensions(unavailable_reason)
            continue
        if region.kind != "SUBJECT_INTERIOR":
            region.physical_dimensions = _unavailable_candidate_dimensions("CANDIDATE_NOT_SUBJECT_INTERIOR")
            continue
        contour = contours.get(region.id)
        if contour is None:
            region.physical_dimensions = _unavailable_candidate_dimensions("CANDIDATE_CONTOUR_INVALID")
            continue
        points = np.asarray(contour, dtype=np.float32)
        assert metric_from_target is not None and target_to_plane is not None and support is not None
        plane_points = cv2.perspectiveTransform(points, target_to_plane).reshape(-1, 2)
        if any(cv2.pointPolygonTest(support, (float(point[0]), float(point[1])), False) < 0 for point in plane_points):
            region.physical_dimensions = _unavailable_candidate_dimensions(
                "CANDIDATE_OUTSIDE_CALIBRATION_PLANE_SUPPORT",
            )
            continue
        metric_points = cv2.perspectiveTransform(points, metric_from_target).astype(np.float32)
        method = "CHARUCO_PLANE_CANDIDATE_MASK_MIN_AREA_RECT_V1"
        approval_state = "ENGINEERING_AUTO"
        coordinate_space = "CHARUCO_BOARD_PLANE_MM"
        scale = CandidateDimensionScaleEvidence(source="CURRENT_CHARUCO_BOARD")
        uncertainty_method = "CONSERVATIVE_CALIBRATION_PLUS_CANDIDATE_BOUNDARY_V1"
        edge_uncertainty_mm = direct_edge_uncertainty_mm
        baseline_relative_uncertainty = 0.0
        (_, _), (side_a, side_b), angle = cv2.minAreaRect(metric_points)
        length_mm, width_mm = float(max(side_a, side_b)), float(min(side_a, side_b))
        area_mm2 = float(abs(cv2.contourArea(metric_points)))
        if not all(math.isfinite(value) and value > 0 for value in (length_mm, width_mm, area_mm2)):
            region.physical_dimensions = _unavailable_candidate_dimensions("CANDIDATE_DIMENSION_GEOMETRY_INVALID")
            continue
        linear_uncertainty = 2.0 * float(edge_uncertainty_mm) + baseline_relative_uncertainty * width_mm
        relative_linear = linear_uncertainty / width_mm
        perimeter_mm = float(cv2.arcLength(metric_points, True))
        area_uncertainty = (
            perimeter_mm * float(edge_uncertainty_mm)
            + math.pi * float(edge_uncertainty_mm) ** 2
            + baseline_relative_uncertainty * area_mm2
        )
        if not all(math.isfinite(value) and value > 0 for value in (
            linear_uncertainty, relative_linear, area_uncertainty,
        )) or relative_linear > 1:
            region.physical_dimensions = _unavailable_candidate_dimensions(
                "CANDIDATE_MEASUREMENT_UNCERTAINTY_ABOVE_POLICY",
            )
            continue
        region.physical_dimensions = CandidatePhysicalDimensionEvidence(
            state="AVAILABLE",
            disclaimerCode="CANDIDATE_DIMENSION_ESTIMATE_NOT_DEFECT_PROOF",
            method=method,
            approvalState=approval_state,
            coordinateSpace=coordinate_space,
            candidateMaskSha256=spatial.mask_sha256,
            lengthMm=length_mm,
            widthMm=width_mm,
            areaMm2=area_mm2,
            rotatedRectAngleDegrees=float(angle),
            scale=scale,
            uncertainty=CandidateDimensionUncertaintyEvidence(
                method=uncertainty_method,
                linearMm=linear_uncertainty,
                areaMm2=area_uncertainty,
                relativeLinear=relative_linear,
            ),
        )


def _source_plane_physical_dimension_evidence(
    artifact: ProductionArtifactV18,
    *,
    source_shape: tuple[int, int],
    source_to_plane: object | None,
    calibration: PlaneMetricCalibration | TargetMetricCalibration | None,
    calibration_support_plane: object | None,
    prediction: SubjectMaskPrediction | None,
    method: str,
    missing_prediction_reason: str,
    invalid_mask_reason: str,
    small_contour_reason: str,
    invalid_contour_reason: str,
) -> PhysicalDimensionEvidence:
    """Measure one full-resolution source mask in the calibrated board plane.

    Golden and Current calls intentionally share this implementation.  The
    canonical DINO image remains useful for similarity scoring, but is not a
    ruler and must never determine the pixel resolution used for millimetres.
    """
    import cv2
    import numpy as np

    if calibration is None or source_to_plane is None:
        return _unavailable_physical_dimensions("CHARUCO_CALIBRATION_REQUIRED")
    policy = artifact.dimension_measurement_policy
    if calibration.reprojection_error_px > policy.max_plane_reprojection_error_px:
        return _unavailable_physical_dimensions("CALIBRATION_REPROJECTION_ERROR_ABOVE_POLICY")
    if calibration.calibration_fiducial != "CHARUCO_CORNERS":
        # Outer markers may identify/normalize a board, but cannot establish
        # whole-subject metric evidence.  In particular a side strip or a
        # background card gives no support for a foreground 3-D contour.
        return _unavailable_physical_dimensions("CHARUCO_CORNERS_REQUIRED_FOR_WHOLE_SUBJECT_METRICS")
    if prediction is None:
        return _unavailable_physical_dimensions(missing_prediction_reason)

    mask = (np.asarray(prediction.mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if mask.shape != source_shape:
        return _unavailable_physical_dimensions(invalid_mask_reason)
    if int(cv2.countNonZero(mask)) < policy.min_contour_area_px:
        return _unavailable_physical_dimensions(small_contour_reason)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [item for item in contours if cv2.contourArea(item) > 0]
    if not contours or sum(len(item) for item in contours) < policy.min_contour_points:
        return _unavailable_physical_dimensions(invalid_contour_reason)

    source_to_plane_matrix = np.asarray(source_to_plane, dtype=np.float64)
    if source_to_plane_matrix.shape != (3, 3) or not np.all(np.isfinite(source_to_plane_matrix)):
        return _unavailable_physical_dimensions("METRIC_TRANSFORM_INVALID")
    if abs(float(np.linalg.det(source_to_plane_matrix))) <= 1e-12:
        return _unavailable_physical_dimensions("METRIC_TRANSFORM_INVALID")
    support = np.asarray(calibration_support_plane, dtype=np.float32)
    if support.shape != (4, 2) or not np.all(np.isfinite(support)) or abs(float(cv2.contourArea(support))) <= 1.0:
        return _unavailable_physical_dimensions("CALIBRATION_SUPPORT_REQUIRED")

    plane_contours = [
        cv2.perspectiveTransform(item.astype(np.float32), source_to_plane_matrix).astype(np.float32)
        for item in contours
    ]
    # A board transform is only supported on the physical board.  Any
    # extrapolation (the exact failure in the supplied photographs) is a
    # recapture condition, not a low-confidence millimetre result.
    for contour in plane_contours:
        points = contour.reshape(-1, 2)
        if any(cv2.pointPolygonTest(support, (float(point[0]), float(point[1])), False) < 0 for point in points):
            return _unavailable_physical_dimensions("SUBJECT_OUTSIDE_CALIBRATION_PLANE_SUPPORT")

    metric_from_source = np.diag((
        1.0 / calibration.pixels_per_mm_x,
        1.0 / calibration.pixels_per_mm_y,
        1.0,
    )) @ source_to_plane_matrix
    if abs(float(np.linalg.det(metric_from_source))) <= 1e-12:
        return _unavailable_physical_dimensions("METRIC_TRANSFORM_INVALID")
    metric_contours = [
        cv2.perspectiveTransform(item.astype(np.float32), metric_from_source).astype(np.float32)
        for item in contours
    ]
    all_points = np.concatenate(metric_contours, axis=0)
    (_, _), (side_a, side_b), angle = cv2.minAreaRect(all_points)
    length_mm, width_mm = float(max(side_a, side_b)), float(min(side_a, side_b))
    area_mm2 = float(sum(abs(cv2.contourArea(item)) for item in metric_contours))
    if not all(math.isfinite(value) and value > 0 for value in (length_mm, width_mm, area_mm2)):
        return _unavailable_physical_dimensions("PHYSICAL_DIMENSION_GEOMETRY_INVALID")

    source_points = np.concatenate(contours, axis=0).reshape(-1, 2)
    center = np.mean(source_points, axis=0)
    probe = np.asarray([[[center[0], center[1]], [center[0] + 1.0, center[1]], [center[0], center[1] + 1.0]]], dtype=np.float32)
    metric_probe = cv2.perspectiveTransform(probe, metric_from_source).reshape(-1, 2)
    mm_per_source_px = max(
        float(np.linalg.norm(metric_probe[1] - metric_probe[0])),
        float(np.linalg.norm(metric_probe[2] - metric_probe[0])),
    )
    if not math.isfinite(mm_per_source_px) or mm_per_source_px <= 0:
        return _unavailable_physical_dimensions("MEASUREMENT_UNCERTAINTY_INVALID")
    calibration_uncertainty_mm = calibration.reprojection_error_px / min(
        calibration.pixels_per_mm_x, calibration.pixels_per_mm_y,
    )
    edge_uncertainty_mm = calibration_uncertainty_mm + policy.segmentation_boundary_uncertainty_px * mm_per_source_px
    linear_uncertainty_mm = 2.0 * edge_uncertainty_mm
    relative_linear = linear_uncertainty_mm / width_mm
    perimeter_mm = float(sum(cv2.arcLength(item, True) for item in metric_contours))
    area_uncertainty_mm2 = perimeter_mm * edge_uncertainty_mm + math.pi * edge_uncertainty_mm ** 2
    if not all(math.isfinite(value) and value > 0 for value in (
        linear_uncertainty_mm, relative_linear, area_uncertainty_mm2,
    )):
        return _unavailable_physical_dimensions("MEASUREMENT_UNCERTAINTY_INVALID")
    if relative_linear > policy.max_relative_linear_uncertainty:
        return _unavailable_physical_dimensions("MEASUREMENT_UNCERTAINTY_ABOVE_POLICY")

    _, _, mask_sha = _encode_binary_png(mask)
    return PhysicalDimensionEvidence(
        state="AVAILABLE",
        disclaimerCode="ENGINEERING_DIMENSION_NOT_METROLOGY_PROOF",
        method=method,
        approvalState=policy.approval_state,
        coordinateSpace="CHARUCO_BOARD_PLANE_MM",
        currentSubjectMaskSha256=mask_sha,
        lengthMm=length_mm,
        widthMm=width_mm,
        areaMm2=area_mm2,
        rotatedRectAngleDegrees=float(angle),
        calibration=MetricCalibrationEvidence(
            source=policy.calibration_source,
            fiducial=calibration.calibration_fiducial,
            detectedCornerCount=calibration.detected_corner_count,
            inlierCornerCount=calibration.inlier_corner_count,
            planeReprojectionErrorPx=calibration.reprojection_error_px,
            pixelsPerMmX=calibration.pixels_per_mm_x,
            pixelsPerMmY=calibration.pixels_per_mm_y,
        ),
        uncertainty=DimensionUncertaintyEvidence(
            method="CONSERVATIVE_CALIBRATION_PLUS_SEGMENTATION_V1",
            linearMm=linear_uncertainty_mm,
            areaMm2=area_uncertainty_mm2,
            relativeLinear=relative_linear,
        ),
    )


def _background_board_marker_correspondences(
    source_rgb: object,
    board: GoldenDimensionBoardCandidate,
) -> tuple[object, object]:
    """Return matched declared outer-marker points in source-image coordinates."""
    import cv2
    import numpy as np

    if not board.outer_aruco_geometry_qualified or len(board.outer_markers) < 3:
        raise OffsetPlaneGeometryError("BACKGROUND_BOARD_OUTER_GEOMETRY_REQUIRED")
    dictionary_id = getattr(cv2.aruco, board.dictionary, None)
    if dictionary_id is None:
        raise OffsetPlaneGeometryError("BACKGROUND_BOARD_DICTIONARY_UNSUPPORTED")
    image = np.asarray(source_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise OffsetPlaneGeometryError("BACKGROUND_BOARD_SOURCE_IMAGE_INVALID")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detected_corners, detected_ids, _ = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary_id), cv2.aruco.DetectorParameters(),
    ).detectMarkers(gray)
    image_by_id = {
        int(marker_id): np.asarray(corners, dtype=np.float64).reshape(4, 2)
        for corners, marker_id in zip(detected_corners, [] if detected_ids is None else detected_ids.flatten())
    }
    object_points: list[list[float]] = []
    image_points: list[list[float]] = []
    matched_markers = 0
    for marker in board.outer_markers:
        observed = image_by_id.get(marker.id)
        if observed is None:
            continue
        matched_markers += 1
        object_points.extend(marker.corners_mm)
        image_points.extend(observed.tolist())
    if matched_markers < 3:
        raise OffsetPlaneGeometryError("BACKGROUND_BOARD_OUTER_MARKERS_INSUFFICIENT")
    return object_points, image_points


def _background_board_pnp_pose(
    source_rgb: object,
    board: GoldenDimensionBoardCandidate,
    intrinsics: CameraIntrinsics,
    *,
    minimum_inliers: int,
    maximum_reprojection_error_px: float,
    correspondences: tuple[object, object] | None = None,
) -> BoardPose:
    """Estimate background-board pose from distributed immutable outer markers.

    Unlike a homography, this pose is usable for a separate, known parallel
    front plane.  At least three outer markers are required so a cropped card
    cannot silently become a 3-D datum.
    """
    object_points, image_points = correspondences or _background_board_marker_correspondences(source_rgb, board)
    pose = estimate_board_pose(
        object_points, image_points, intrinsics,
        minimum_inliers=minimum_inliers,
        ransac_reprojection_error_px=maximum_reprojection_error_px,
    )
    if pose.reprojection_error_px > maximum_reprojection_error_px:
        raise OffsetPlaneGeometryError("BACKGROUND_BOARD_PNP_REPROJECTION_ABOVE_POLICY")
    return pose


def _robust_body_width_from_projected_mask_points(
    contour_points: object,
    interior_points: object,
    *,
    fallback_width_mm: float,
) -> float:
    """Return the sustained central cross-section width of one foreground.

    A full-silhouette min-area rectangle is appropriate for the total length,
    but its short side is unstable for a hand-held product: a pipe, hook, or
    one angled corner can widen one end of the rectangle even though it is not
    the enclosure width that an operator measures.  This recipe-agnostic
    geometry keeps the full long side while taking the 90th percentile of
    cross-section widths from the middle 20--80 percent of that long axis.

    The percentile means the reported width must persist through a central
    cross-section rather than occur only at an excluded end attachment. It is still
    deterministic image geometry, never a learned size correction.
    """
    import cv2
    import numpy as np

    boundary = np.asarray(contour_points, dtype=np.float32).reshape(-1, 2)
    interior = np.asarray(interior_points, dtype=np.float64).reshape(-1, 2)
    if len(boundary) < 4 or len(interior) < 128:
        return float(fallback_width_mm)
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(boundary.reshape(-1, 1, 2))
    if side_a <= 0 or side_b <= 0:
        return float(fallback_width_mm)
    box = cv2.boxPoints(cv2.minAreaRect(boundary.reshape(-1, 1, 2)))
    edges = np.roll(box, -1, axis=0) - box
    edge_lengths = np.linalg.norm(edges, axis=1)
    long_edge_index = int(np.argmax(edge_lengths))
    long_axis = edges[long_edge_index] / max(float(edge_lengths[long_edge_index]), 1e-12)
    short_axis = np.asarray((-long_axis[1], long_axis[0]), dtype=np.float64)
    boundary_along = boundary.astype(np.float64) @ long_axis
    interior_along = interior @ long_axis
    interior_across = interior @ short_axis
    low = float(np.min(boundary_along))
    high = float(np.max(boundary_along))
    span = high - low
    if not math.isfinite(span) or span <= 1e-9:
        return float(fallback_width_mm)

    # Twelve samples in the middle 60% deliberately exclude the attachment
    # zones at either end, while remaining broad enough for a rotated body.
    slice_widths: list[float] = []
    start, end, slice_count = 0.20, 0.80, 12
    for index in range(slice_count):
        fraction_low = start + (end - start) * index / slice_count
        fraction_high = start + (end - start) * (index + 1) / slice_count
        cross_section = interior_across[
            (interior_along >= low + span * fraction_low)
            & (interior_along < low + span * fraction_high)
        ]
        if len(cross_section) < 32:
            continue
        # Trim only isolated raster/segmentation specks.  A 2.5% trim would
        # shrink a genuinely rectangular 59 mm body by almost 3 mm.
        cross_width = float(np.quantile(cross_section, 0.995) - np.quantile(cross_section, 0.005))
        if math.isfinite(cross_width) and cross_width > 0:
            slice_widths.append(cross_width)
    if len(slice_widths) < 8:
        return float(fallback_width_mm)
    return float(np.quantile(np.asarray(slice_widths, dtype=np.float64), 0.90))


def _background_board_offset_plane_physical_dimension_evidence(
    artifact: ProductionArtifactV18,
    request: AnalyzeRequest | GoldenDimensionRequest,
    normalized: NormalizedCapture,
    prediction: SubjectMaskPrediction | None,
    relative_depth_estimator: RelativeDepthEstimator | None = None,
    current_subject_mask_sha256: str | None = None,
) -> PhysicalDimensionEvidence:
    """Measure a foreground mask from a board pose without assuming coplanarity.

    A fixed rig datum is the most accurate path.  When the configured capture
    allows it, a relative-depth provider may refine the datum per photo after
    being calibrated against visible board pixels.  If that provider is absent
    or inconclusive, we still emit the configured best estimate and its full
    prior interval; this is intentionally not turned into a silent refusal.
    """
    import cv2
    import numpy as np

    calibration = request.offset_plane_calibration
    if calibration is None:
        return _unavailable_physical_dimensions("BACKGROUND_BOARD_OFFSET_PLANE_CALIBRATION_REQUIRED")
    if prediction is None:
        return _unavailable_physical_dimensions("CURRENT_SUBJECT_SEGMENTATION_REQUIRED")
    if normalized.source_rgb is None or normalized.calibration_board is None:
        return _unavailable_physical_dimensions("BACKGROUND_BOARD_CALIBRATION_REQUIRED")
    source = np.asarray(normalized.source_rgb, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        return _unavailable_physical_dimensions("BACKGROUND_BOARD_SOURCE_IMAGE_INVALID")
    source_height, source_width = source.shape[:2]
    camera = calibration.camera
    self_calibrated_intrinsics = getattr(camera, "source", "NATIVE_CAPTURE_V2") == "BOARD_SELF_CALIBRATED_V1"
    dimensions_match = (camera.image_width, camera.image_height) == (source_width, source_height)
    # JPEG EXIF orientation is removed by the decoder before marker detection.
    # PhoneCV's upload dimensions describe the encoded SOF, which may be the
    # transposed pair.  A board-only focal fallback derives K from the already
    # oriented pixels, so accepting that pair is safe and keeps ordinary phone
    # stills on the best-effort estimate path. Native K remains strict because
    # its principal point/distortion need an explicit coordinate rebinding.
    oriented_jpeg_pair = (
        getattr(request, "content_type", None) == "image/jpeg"
        and self_calibrated_intrinsics
        and (camera.image_width, camera.image_height) == (source_height, source_width)
    )
    if not dimensions_match and not oriented_jpeg_pair:
        return _unavailable_physical_dimensions("BACKGROUND_BOARD_CAMERA_IMAGE_DIMENSIONS_MISMATCH")
    try:
        correspondences = None
        if self_calibrated_intrinsics:
            correspondences = _background_board_marker_correspondences(source, normalized.calibration_board)
            focal = estimate_square_focal_length_from_planar_board(
                correspondences[0], correspondences[1],
                image_width=source_width, image_height=source_height,
            )
            intrinsics = focal.intrinsics
        else:
            intrinsics = CameraIntrinsics(
                camera.fx_px, camera.fy_px, camera.cx_px, camera.cy_px,
                tuple(camera.distortion_coefficients),
            )
        # Native intrinsics retain the rig's tight residual gate.  A planar
        # self-calibration deliberately has unmodelled principal-point and
        # distortion error, so retain an estimate with a wider residual term
        # rather than treating a 2–4 px fit as no result at all.
        pnp_reprojection_limit = calibration.max_board_pnp_reprojection_error_px * (
            3.0 if self_calibrated_intrinsics else 1.0
        )
        pose = _background_board_pnp_pose(
            source, normalized.calibration_board, intrinsics,
            minimum_inliers=calibration.min_board_pnp_inliers,
            maximum_reprojection_error_px=pnp_reprojection_limit,
            correspondences=correspondences,
        )
    except OffsetPlaneGeometryError as exc:
        return _unavailable_physical_dimensions(str(exc))

    policy = artifact.dimension_measurement_policy
    mask = (np.asarray(prediction.mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if mask.shape != (source_height, source_width):
        return _unavailable_physical_dimensions("CURRENT_SOURCE_SUBJECT_MASK_DIMENSIONS_INVALID")
    if int(cv2.countNonZero(mask)) < policy.min_contour_area_px:
        return _unavailable_physical_dimensions("CURRENT_SUBJECT_CONTOUR_AREA_BELOW_POLICY")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [item for item in contours if cv2.contourArea(item) > 0]
    if not contours or sum(len(item) for item in contours) < policy.min_contour_points:
        return _unavailable_physical_dimensions("CURRENT_SUBJECT_CONTOUR_INVALID")
    foreground_y, foreground_x = np.nonzero(mask > 0)
    # A bounded evenly-spaced sample has more than enough pixels per cross
    # section, keeps latency stable, and is deterministic for the audit trail.
    sample_stride = max(1, int(math.ceil(len(foreground_x) / 30_000)))
    source_interior_samples = np.column_stack((
        foreground_x[::sample_stride], foreground_y[::sample_stride],
    )).astype(np.float64)

    # Some deterministic migration tests use a lightweight namespace instead
    # of the request model.  Treat its missing policy exactly as the backwards
    # compatible fixed-rig default.
    depth_policy = getattr(calibration, "depth_estimate_policy", None)
    offset_mm = calibration.front_plane_offset_mm
    lower_offset = (
        getattr(depth_policy, "lower95_mm", None)
        if getattr(depth_policy, "lower95_mm", None) is not None else calibration.front_plane_offset_mm
    )
    upper_offset = (
        getattr(depth_policy, "upper95_mm", None)
        if getattr(depth_policy, "upper95_mm", None) is not None else calibration.front_plane_offset_mm
    )
    # A generic, geometry-aware prior is expressed as a fraction of this
    # capture's solved board distance, not as a magic fixed millimetre value.
    # It is useful when the foreground is free in depth and no rig datum has
    # yet been surveyed; the deliberately broad bounds are shown to the user.
    ratio_center = getattr(depth_policy, "board_distance_ratio_center", None)
    ratio_lower = getattr(depth_policy, "board_distance_ratio_lower95", None)
    ratio_upper = getattr(depth_policy, "board_distance_ratio_upper95", None)
    if all(value is not None for value in (ratio_center, ratio_lower, ratio_upper)):
        board_distance_mm = float(pose.rotation()[:, 2] @ pose.translation())
        if math.isfinite(board_distance_mm) and board_distance_mm > 0:
            offset_mm = board_distance_mm * float(ratio_center)
            lower_offset = board_distance_mm * float(ratio_lower)
            upper_offset = board_distance_mm * float(ratio_upper)
    depth_estimate = DepthOffsetEstimateEvidence(
        source=(
            "FIXED_RIG_DATUM_V1"
            if lower_offset == upper_offset else "BOARD_POSE_UNCALIBRATED_PRIOR_V1"
        ),
        offsetMm=offset_mm,
        lower95Mm=lower_offset,
        upper95Mm=upper_offset,
        intervalKind=(
            "FIXED_RIG_TOLERANCE"
            if lower_offset == upper_offset else "UNCALIBRATED_SCENARIO_ENVELOPE"
        ),
    )
    board = normalized.calibration_board
    if (
        bool(getattr(depth_policy, "relative_depth_enabled", False)) and relative_depth_estimator is not None
        and board.finished_width_mm is not None and board.finished_height_mm is not None
    ):
        # Use the entire visible board (minus the subject) for the per-capture
        # relative-depth-to-millimetres fit.  Sampling only black marker cells
        # would let printing contrast dominate the learned depth prediction.
        try:
            board_corners, _ = cv2.projectPoints(
                np.asarray(((0.0, 0.0, 0.0), (board.finished_width_mm, 0.0, 0.0),
                            (board.finished_width_mm, board.finished_height_mm, 0.0),
                            (0.0, board.finished_height_mm, 0.0)), dtype=np.float64),
                np.asarray(pose.rvec, dtype=np.float64), np.asarray(pose.tvec_mm, dtype=np.float64),
                intrinsics.matrix(), intrinsics.distortion(),
            )
            board_mask = np.zeros((source_height, source_width), dtype=np.uint8)
            cv2.fillConvexPoly(board_mask, np.round(board_corners.reshape(-1, 2)).astype(np.int32), 255)
            board_mask[mask > 0] = 0
            relative_depth = relative_depth_estimator.estimate_inverse_depth(source)
            posterior = estimate_front_offset_from_relative_inverse_depth(
                relative_depth, board_mask > 0, mask > 0, intrinsics, pose,
                RelativeDepthOffsetPolicy(
                    maximum_front_offset_mm=max(upper_offset, calibration.front_plane_offset_mm, 1.0),
                    model_systematic_error_mm=float(getattr(depth_policy, "model_systematic_error_mm", 8.0)),
                    dominant_plane_enabled=bool(getattr(depth_policy, "dominant_plane_enabled", False)),
                    dominant_plane_half_width_mm=float(getattr(depth_policy, "dominant_plane_half_width_mm", 8.0)),
                    minimum_dominant_plane_support_ratio=float(
                        getattr(depth_policy, "minimum_dominant_plane_support_ratio", 0.35),
                    ),
                ),
            )
            if relative_depth_posterior_improves_prior(posterior, lower_offset, upper_offset):
                assert posterior.offset_mm is not None
                assert posterior.lower95_mm is not None and posterior.upper95_mm is not None
                assert posterior.board_fit_p95_mm is not None
                assert posterior.subject_depth_spread_p95_mm is not None
                offset_mm = posterior.offset_mm
                lower_offset = posterior.lower95_mm
                upper_offset = posterior.upper95_mm
                depth_estimate = DepthOffsetEstimateEvidence(
                    source="RELATIVE_DEPTH_BOARD_CALIBRATED_V1",
                    offsetMm=offset_mm,
                    lower95Mm=lower_offset,
                    upper95Mm=upper_offset,
                    # Fitting the map to the board gives metric scale, but it
                    # does not by itself validate the foreground extrapolation
                    # against independent physical measurements.
                    intervalKind="MODEL_UNVALIDATED_INTERVAL",
                    boardFitP95Mm=posterior.board_fit_p95_mm,
                    subjectSpreadP95Mm=posterior.subject_depth_spread_p95_mm,
                )
            # An unqualified learned map must not widen or contradict the
            # configured board/rig scenario envelope.  The explicit physical
            # prior above remains visible whenever the posterior is broad.
        except (OffsetPlaneGeometryError, RuntimeError, ValueError, TypeError):
            # The static prior remains explicit in the returned interval.  A
            # learned model is a correction source, never a reason to hide a
            # useful board-pose estimate from the operator.
            pass

    def projected_geometry(front_offset_mm: float) -> tuple[list[object], float, float, float, float]:
        metric_contours = [
            intersect_pixels_with_front_offset_plane(
                contour.reshape(-1, 2), intrinsics, pose,
                front_plane_offset_mm=front_offset_mm,
            ).astype(np.float32).reshape(-1, 1, 2)
            for contour in contours
        ]
        all_points = np.concatenate(metric_contours, axis=0)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(all_points)
        length_mm, min_area_rect_width_mm = float(max(side_a, side_b)), float(min(side_a, side_b))
        interior_points = intersect_pixels_with_front_offset_plane(
            source_interior_samples, intrinsics, pose, front_plane_offset_mm=front_offset_mm,
        )
        width_mm = _robust_body_width_from_projected_mask_points(
            all_points, interior_points, fallback_width_mm=min_area_rect_width_mm,
        )
        area_mm2 = float(sum(abs(cv2.contourArea(item)) for item in metric_contours))
        perimeter_mm = float(sum(cv2.arcLength(item, True) for item in metric_contours))
        if not all(math.isfinite(value) and value > 0 for value in (length_mm, width_mm, area_mm2, perimeter_mm)):
            raise OffsetPlaneGeometryError("PHYSICAL_DIMENSION_GEOMETRY_INVALID")
        return metric_contours, length_mm, width_mm, area_mm2, perimeter_mm

    try:
        metric_contours, length_mm, width_mm, area_mm2, perimeter_mm = projected_geometry(offset_mm)
        _, lower_length_mm, lower_width_mm, lower_area_mm2, _ = projected_geometry(lower_offset)
        _, upper_length_mm, upper_width_mm, upper_area_mm2, _ = projected_geometry(upper_offset)
    except OffsetPlaneGeometryError as exc:
        return _unavailable_physical_dimensions(str(exc))
    all_points = np.concatenate(metric_contours, axis=0)
    (_, _), _, angle = cv2.minAreaRect(all_points)
    source_points = np.concatenate(contours, axis=0).reshape(-1, 2)
    center = np.mean(source_points, axis=0)
    try:
        probe = intersect_pixels_with_front_offset_plane(
            ((center[0], center[1]), (center[0] + 1.0, center[1]), (center[0], center[1] + 1.0)),
            intrinsics, pose, front_plane_offset_mm=offset_mm,
        )
    except OffsetPlaneGeometryError as exc:
        return _unavailable_physical_dimensions(str(exc))
    mm_per_source_px = max(float(np.linalg.norm(probe[1] - probe[0])), float(np.linalg.norm(probe[2] - probe[0])))
    if not math.isfinite(mm_per_source_px) or mm_per_source_px <= 0:
        return _unavailable_physical_dimensions("MEASUREMENT_UNCERTAINTY_INVALID")
    edge_uncertainty_mm = (
        pose.reprojection_error_px + policy.segmentation_boundary_uncertainty_px
    ) * mm_per_source_px
    length_lower95_mm = max(1e-9, min(lower_length_mm, upper_length_mm) - 2.0 * edge_uncertainty_mm)
    length_upper95_mm = max(lower_length_mm, upper_length_mm) + 2.0 * edge_uncertainty_mm
    width_lower95_mm = max(1e-9, min(lower_width_mm, upper_width_mm) - 2.0 * edge_uncertainty_mm)
    width_upper95_mm = max(lower_width_mm, upper_width_mm) + 2.0 * edge_uncertainty_mm
    linear_uncertainty_mm = max(
        abs(length_mm - length_lower95_mm), abs(length_upper95_mm - length_mm),
        abs(width_mm - width_lower95_mm), abs(width_upper95_mm - width_mm),
    )
    relative_linear = linear_uncertainty_mm / width_mm
    area_uncertainty_mm2 = max(
        abs(area_mm2 - lower_area_mm2), abs(upper_area_mm2 - area_mm2),
        perimeter_mm * edge_uncertainty_mm + math.pi * edge_uncertainty_mm ** 2,
    )
    if not all(math.isfinite(value) and value > 0 for value in (
        linear_uncertainty_mm, relative_linear, area_uncertainty_mm2,
    )):
        return _unavailable_physical_dimensions("MEASUREMENT_UNCERTAINTY_INVALID")
    # A broad interval is an explicit result, not a reason to replace a real
    # photograph with a false "no estimate".  Callers can display the range
    # and improve it with a rig datum or a validated depth model.
    _, _, mask_sha = _encode_binary_png(mask)
    return PhysicalDimensionEvidence(
        state="AVAILABLE",
        disclaimerCode="ENGINEERING_DIMENSION_NOT_METROLOGY_PROOF",
        method="BACKGROUND_BOARD_PNP_FRONT_OFFSET_BODY_CROSS_SECTION_V1",
        approvalState=policy.approval_state,
        coordinateSpace="BACKGROUND_BOARD_FRONT_OFFSET_PLANE_MM",
        # The projected contour is source-resolution, while PhoneCV's
        # subject gate is bound to the canonical Current mask.  Supply both
        # digests at construction time so the AVAILABLE contract is complete
        # before Pydantic validation runs.
        currentSubjectMaskSha256=current_subject_mask_sha256 or mask_sha,
        metricSubjectMaskSha256=mask_sha,
        lengthMm=length_mm,
        widthMm=width_mm,
        areaMm2=area_mm2,
        rotatedRectAngleDegrees=float(angle),
        calibration=MetricCalibrationEvidence(
            source=(
                "BACKGROUND_BOARD_PNP_SELF_CALIBRATED_INTRINSICS_V1"
                if self_calibrated_intrinsics else "BACKGROUND_BOARD_PNP_FRONT_OFFSET_V1"
            ),
            fiducial="OUTER_ARUCO_CORNERS",
            detectedCornerCount=len(normalized.calibration_board.outer_markers) * 4,
            inlierCornerCount=pose.inlier_count,
            planeReprojectionErrorPx=pose.reprojection_error_px,
            pixelsPerMmX=1.0 / mm_per_source_px,
            pixelsPerMmY=1.0 / mm_per_source_px,
        ),
        uncertainty=DimensionUncertaintyEvidence(
            method="BOARD_POSE_DEPTH_INTERVAL_PLUS_SEGMENTATION_V1",
            linearMm=linear_uncertainty_mm,
            areaMm2=area_uncertainty_mm2,
            relativeLinear=relative_linear,
            lengthLower95Mm=length_lower95_mm,
            lengthUpper95Mm=length_upper95_mm,
            widthLower95Mm=width_lower95_mm,
            widthUpper95Mm=width_upper95_mm,
            intervalKind=depth_estimate.interval_kind,
        ),
        depthOffsetEstimate=depth_estimate,
    )


def _source_metric_prediction(
    artifact: ProductionArtifactV18,
    normalized: NormalizedCapture,
    segmenter: SubjectSegmenter,
) -> SubjectMaskPrediction:
    """Segment the metric subject on the decoded source using the target ROI.

    Similarity analysis keeps its canonical MobileSAM mask. This separate
    prediction maps the pinned canonical ROI back to the raw source, then
    expands that prompt before segmentation. Alignment ROI is deliberately
    tight for comparison, but it is not a physical object boundary; hard
    clipping it previously cut off the handle in a legitimate hand-held shot.
    """
    import cv2
    import numpy as np

    if normalized.source_rgb is None or normalized.target_from_source is None:
        raise RuntimeError("SOURCE_METRIC_PIPELINE_UNAVAILABLE")
    source = np.asarray(normalized.source_rgb, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise RuntimeError("SOURCE_METRIC_IMAGE_INVALID")
    target_from_source = np.asarray(normalized.target_from_source, dtype=np.float64)
    if target_from_source.shape != (3, 3) or not np.all(np.isfinite(target_from_source)):
        raise RuntimeError("SOURCE_METRIC_TRANSFORM_INVALID")
    try:
        source_from_target = np.linalg.inv(target_from_source)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("SOURCE_METRIC_TRANSFORM_INVALID") from exc
    roi_box = inspection_roi_image(artifact.inspection_roi).getbbox()
    if roi_box is None:
        raise RuntimeError("INSPECTION_ROI_EMPTY")
    left, top, right, bottom = roi_box
    target_roi = np.asarray([[[left, top], [right, top], [right, bottom], [left, bottom]]], dtype=np.float32)
    source_roi = cv2.perspectiveTransform(target_roi, source_from_target).reshape(-1, 2)
    height, width = source.shape[:2]
    if not np.all(np.isfinite(source_roi)) or np.any(source_roi[:, 0] < 0) or np.any(source_roi[:, 1] < 0) \
            or np.any(source_roi[:, 0] >= width) or np.any(source_roi[:, 1] >= height):
        raise RuntimeError("SOURCE_METRIC_ROI_OUT_OF_BOUNDS")
    roi_left = int(math.floor(float(np.min(source_roi[:, 0]))))
    roi_top = int(math.floor(float(np.min(source_roi[:, 1]))))
    roi_right = int(math.ceil(float(np.max(source_roi[:, 0]))))
    roi_bottom = int(math.ceil(float(np.max(source_roi[:, 1]))))
    padding = int(math.ceil(max(roi_right - roi_left, roi_bottom - roi_top) * SOURCE_METRIC_PROMPT_PADDING_RATIO))
    prompt_left = max(0, roi_left - padding)
    prompt_top = max(0, roi_top - padding)
    prompt_right = min(width, roi_right + padding)
    prompt_bottom = min(height, roi_bottom + padding)
    if prompt_right <= prompt_left or prompt_bottom <= prompt_top:
        raise RuntimeError("SOURCE_METRIC_ROI_INVALID")
    prediction = segmenter.segment(
        source,
        (prompt_left, prompt_top, prompt_right, prompt_bottom),
        min_foreground_ratio=artifact.subject_segmentation.min_foreground_ratio,
        max_foreground_ratio=artifact.subject_segmentation.max_foreground_ratio,
        min_quality_score=artifact.subject_segmentation.min_model_quality_score,
    )
    raw_mask = np.asarray(prediction.mask, dtype=np.uint8) > 0
    if raw_mask.shape != (height, width):
        raise RuntimeError("SOURCE_METRIC_MASK_DIMENSIONS_INVALID")
    # Retain the prompt's expanded support instead of the similarity ROI.
    # This is still bounded to the source image and lets the segmenter include
    # a valid boundary that shifted just outside the canonical comparison ROI.
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(roi_mask, (prompt_left, prompt_top), (prompt_right - 1, prompt_bottom - 1), 255, thickness=-1)
    clipped = raw_mask & (roi_mask > 0)
    clipped = _clean_source_metric_mask(artifact, normalized, clipped)
    roi_area = int(cv2.countNonZero(roi_mask))
    foreground_ratio = float(np.count_nonzero(clipped)) / max(1, roi_area)
    if not artifact.subject_segmentation.min_foreground_ratio <= foreground_ratio <= artifact.subject_segmentation.max_foreground_ratio:
        raise RuntimeError("SOURCE_METRIC_MASK_AREA_OUT_OF_POLICY")
    return SubjectMaskPrediction(
        mask=(clipped > 0).astype(np.uint8) * 255,
        quality_score=prediction.quality_score,
        prompt_box_xyxy=(prompt_left, prompt_top, prompt_right, prompt_bottom),
        foreground_ratio=foreground_ratio,
    )


def _clean_source_metric_mask(
    artifact: ProductionArtifactV18,
    normalized: NormalizedCapture,
    mask: object,
) -> object:
    """Remove disconnected MobileSAM specks before projecting a metric mask.

    A box-prompted segmenter can return small, detached foreground islands in a
    patterned board.  Those islands are especially dangerous here: a single
    island can increase the long-side rectangle and make an uncalibrated depth
    correction look accidentally accurate.  The immutable Golden mask is used
    only as a bounded *support* prior in source coordinates; it never supplies
    a boundary or a size.  The Current prediction remains the sole measured
    contour.

    The fallback keeps the original proposal if the prior would remove most of
    the foreground.  This preserves the estimate path for genuinely different
    poses while making the common detached-speck failure explicit and
    deterministic.
    """
    import cv2
    import numpy as np

    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if binary.ndim != 2 or not np.any(binary):
        return binary.astype(np.uint8) * 255

    bounded = binary
    target_from_source = normalized.target_from_source
    golden_contract = getattr(getattr(artifact, "subject_segmentation", None), "golden_masks", ())
    if target_from_source is not None and golden_contract:
        try:
            transform = np.asarray(target_from_source, dtype=np.float64)
            source_from_target = np.linalg.inv(transform)
            encoded = base64.b64decode(golden_contract[0].mask_png_base64, validate=True)
            golden = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if golden is not None and golden.ndim == 2:
                support = cv2.warpPerspective(
                    (golden > 0).astype(np.uint8) * 255,
                    source_from_target,
                    (binary.shape[1], binary.shape[0]),
                    flags=cv2.INTER_NEAREST,
                )
                # Convert the canonical support padding to source pixels.  A
                # small floor is important for masks whose target transform is
                # close to identity (the unit tests and low-resolution phones).
                target_padding = max(8.0, float(getattr(
                    getattr(artifact, "subject_segmentation", None), "support_padding_px", 0,
                )))
                center = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
                source_probe = cv2.perspectiveTransform(center, source_from_target).reshape(-1, 2)
                scale = max(
                    float(np.linalg.norm(source_probe[1] - source_probe[0])),
                    float(np.linalg.norm(source_probe[2] - source_probe[0])),
                    1.0,
                )
                radius = int(min(96, max(8, math.ceil(target_padding * scale))))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1),
                )
                support = cv2.dilate(support, kernel)
                candidate = binary & (support > 0)
                # A support prior is conservative only when it retains most of
                # the model proposal.  Do not turn a changed pose into a hard
                # crop solely because the Golden was captured from one view.
                if int(np.count_nonzero(candidate)) >= max(256, int(np.count_nonzero(binary) * 0.50)):
                    bounded = candidate.astype(np.uint8)
        except (ValueError, TypeError, cv2.error, np.linalg.LinAlgError):
            bounded = binary

    count, labels, stats, _ = cv2.connectedComponentsWithStats(bounded, connectivity=8)
    if count <= 2:
        return bounded.astype(np.uint8) * 255
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest = int(areas.max()) if len(areas) else 0
    if largest <= 0:
        return bounded.astype(np.uint8) * 255
    # Keep substantial attached pieces (e.g. an occluded handle), but reject
    # isolated texture islands below 0.5% of the main subject area.
    minimum = max(256, int(math.ceil(largest * 0.005)))
    keep_labels = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= minimum)
    keep_labels = keep_labels[keep_labels != 0]
    if len(keep_labels) == 0:
        keep_labels = np.asarray([1 + int(np.argmax(areas))])
    cleaned = np.isin(labels, keep_labels)
    return cleaned.astype(np.uint8) * 255


def _physical_dimension_evidence(
    artifact: ProductionArtifactV18,
    normalized: NormalizedCapture,
    current_prediction: SubjectMaskPrediction | None,
) -> PhysicalDimensionEvidence:
    """Measure the Current mask in the ChArUco board plane, never target pixels."""
    import cv2
    import numpy as np

    calibration = normalized.metric_calibration
    if calibration is None:
        return _unavailable_physical_dimensions("CHARUCO_CALIBRATION_REQUIRED")
    policy = artifact.dimension_measurement_policy
    if calibration.reprojection_error_px > policy.max_plane_reprojection_error_px:
        return _unavailable_physical_dimensions("CALIBRATION_REPROJECTION_ERROR_ABOVE_POLICY")
    if calibration.calibration_fiducial != "CHARUCO_CORNERS":
        return _unavailable_physical_dimensions("CHARUCO_CORNERS_REQUIRED_FOR_WHOLE_SUBJECT_METRICS")
    if normalized.calibration_support_plane is None:
        return _unavailable_physical_dimensions("CALIBRATION_SUPPORT_REQUIRED")
    alignment = normalized.alignment
    if (
        alignment is None or alignment.state != "ALIGNED"
        or not alignment.transform_within_bounds or not alignment.inspection_mask_applied
    ):
        return _unavailable_physical_dimensions("TARGET_ALIGNMENT_UNQUALIFIED")
    if current_prediction is None:
        return _unavailable_physical_dimensions("CURRENT_SUBJECT_SEGMENTATION_REQUIRED")

    # Production supplies a separate source-resolution prediction for metrics.
    # Keep the canonical prediction below only as a migration path for older
    # callers/tests; it is never selected by ProductionAnalyzer.
    if normalized.source_rgb is not None and normalized.source_to_plane is not None:
        source_shape = np.asarray(normalized.source_rgb).shape[:2]
        if np.asarray(current_prediction.mask).shape == source_shape:
            return _source_plane_physical_dimension_evidence(
                artifact,
                source_shape=source_shape,
                source_to_plane=normalized.source_to_plane,
                calibration=calibration,
                calibration_support_plane=normalized.calibration_support_plane,
                prediction=current_prediction,
                method=policy.method,
                missing_prediction_reason="CURRENT_SUBJECT_SEGMENTATION_REQUIRED",
                invalid_mask_reason="CURRENT_SOURCE_SUBJECT_MASK_DIMENSIONS_INVALID",
                small_contour_reason="CURRENT_SUBJECT_CONTOUR_AREA_BELOW_POLICY",
                invalid_contour_reason="CURRENT_SUBJECT_CONTOUR_INVALID",
            )

    mask = (np.asarray(current_prediction.mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    expected_shape = (artifact.target_alignment.canonical_height, artifact.target_alignment.canonical_width)
    if mask.shape != expected_shape:
        return _unavailable_physical_dimensions("CURRENT_SUBJECT_MASK_DIMENSIONS_INVALID")
    foreground_area = int(cv2.countNonZero(mask))
    if foreground_area < policy.min_contour_area_px:
        return _unavailable_physical_dimensions("CURRENT_SUBJECT_CONTOUR_AREA_BELOW_POLICY")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [item for item in contours if cv2.contourArea(item) > 0]
    contour_points = sum(len(item) for item in contours)
    if not contours or contour_points < policy.min_contour_points:
        return _unavailable_physical_dimensions("CURRENT_SUBJECT_CONTOUR_INVALID")

    target_to_plane = np.asarray(calibration.target_to_plane, dtype=np.float64)
    if target_to_plane.shape != (3, 3) or not np.all(np.isfinite(target_to_plane)):
        return _unavailable_physical_dimensions("METRIC_TRANSFORM_INVALID")
    plane_to_mm = np.diag((
        1.0 / calibration.pixels_per_mm_x,
        1.0 / calibration.pixels_per_mm_y,
        1.0,
    ))
    metric_from_target = plane_to_mm @ target_to_plane
    if abs(float(np.linalg.det(metric_from_target))) <= 1e-12:
        return _unavailable_physical_dimensions("METRIC_TRANSFORM_INVALID")

    mapped_contours = [
        cv2.perspectiveTransform(item.astype(np.float32), metric_from_target).astype(np.float32)
        for item in contours
    ]
    support = np.asarray(normalized.calibration_support_plane, dtype=np.float32)
    if support.shape != (4, 2) or not np.all(np.isfinite(support)) or abs(float(cv2.contourArea(support))) <= 1.0:
        return _unavailable_physical_dimensions("CALIBRATION_SUPPORT_REQUIRED")
    plane_contours = [
        cv2.perspectiveTransform(item.astype(np.float32), target_to_plane).astype(np.float32)
        for item in contours
    ]
    if any(
        cv2.pointPolygonTest(support, (float(point[0]), float(point[1])), False) < 0
        for contour in plane_contours
        for point in contour.reshape(-1, 2)
    ):
        return _unavailable_physical_dimensions("SUBJECT_OUTSIDE_CALIBRATION_PLANE_SUPPORT")
    all_points = np.concatenate(mapped_contours, axis=0)
    (_, _), (side_a, side_b), angle = cv2.minAreaRect(all_points)
    length_mm = float(max(side_a, side_b))
    width_mm = float(min(side_a, side_b))
    area_mm2 = float(sum(abs(cv2.contourArea(item)) for item in mapped_contours))
    if not all(math.isfinite(value) and value > 0 for value in (length_mm, width_mm, area_mm2)):
        return _unavailable_physical_dimensions("PHYSICAL_DIMENSION_GEOMETRY_INVALID")

    target_center = np.asarray([[[
        artifact.target_alignment.canonical_width / 2.0,
        artifact.target_alignment.canonical_height / 2.0,
    ], [
        artifact.target_alignment.canonical_width / 2.0 + 1.0,
        artifact.target_alignment.canonical_height / 2.0,
    ], [
        artifact.target_alignment.canonical_width / 2.0,
        artifact.target_alignment.canonical_height / 2.0 + 1.0,
    ]]], dtype=np.float32)
    metric_probe = cv2.perspectiveTransform(target_center, metric_from_target).reshape(-1, 2)
    mm_per_target_px = max(
        float(np.linalg.norm(metric_probe[1] - metric_probe[0])),
        float(np.linalg.norm(metric_probe[2] - metric_probe[0])),
    )
    calibration_uncertainty_mm = calibration.reprojection_error_px / min(
        calibration.pixels_per_mm_x, calibration.pixels_per_mm_y,
    )
    boundary_uncertainty_mm = policy.segmentation_boundary_uncertainty_px * mm_per_target_px
    edge_uncertainty_mm = calibration_uncertainty_mm + boundary_uncertainty_mm
    linear_uncertainty_mm = 2.0 * edge_uncertainty_mm
    relative_linear = linear_uncertainty_mm / width_mm
    perimeter_mm = float(sum(cv2.arcLength(item, True) for item in mapped_contours))
    area_uncertainty_mm2 = perimeter_mm * edge_uncertainty_mm + math.pi * edge_uncertainty_mm ** 2
    uncertainty_values = (linear_uncertainty_mm, relative_linear, area_uncertainty_mm2)
    if not all(math.isfinite(value) and value > 0 for value in uncertainty_values):
        return _unavailable_physical_dimensions("MEASUREMENT_UNCERTAINTY_INVALID")
    if relative_linear > policy.max_relative_linear_uncertainty:
        return _unavailable_physical_dimensions("MEASUREMENT_UNCERTAINTY_ABOVE_POLICY")

    _, _, current_mask_sha = _encode_binary_png(mask)
    return PhysicalDimensionEvidence(
        state="AVAILABLE",
        disclaimerCode="ENGINEERING_DIMENSION_NOT_METROLOGY_PROOF",
        method=policy.method,
        approvalState=policy.approval_state,
        coordinateSpace="CHARUCO_BOARD_PLANE_MM",
        currentSubjectMaskSha256=current_mask_sha,
        lengthMm=length_mm,
        widthMm=width_mm,
        areaMm2=area_mm2,
        rotatedRectAngleDegrees=float(angle),
        calibration=MetricCalibrationEvidence(
            source=policy.calibration_source,
            fiducial=calibration.calibration_fiducial,
            detectedCornerCount=calibration.detected_corner_count,
            inlierCornerCount=calibration.inlier_corner_count,
            planeReprojectionErrorPx=calibration.reprojection_error_px,
            pixelsPerMmX=calibration.pixels_per_mm_x,
            pixelsPerMmY=calibration.pixels_per_mm_y,
        ),
        uncertainty=DimensionUncertaintyEvidence(
            method="CONSERVATIVE_CALIBRATION_PLUS_SEGMENTATION_V1",
            linearMm=linear_uncertainty_mm,
            areaMm2=area_uncertainty_mm2,
            relativeLinear=relative_linear,
        ),
    )


def _golden_physical_dimension_evidence(
    artifact: ProductionArtifactV18,
    plane: PlaneNormalizedCapture,
    prediction: SubjectMaskPrediction | None,
) -> PhysicalDimensionEvidence:
    """Measure a Golden source mask through the shared full-resolution path."""
    import cv2
    import numpy as np

    calibration = plane.metric_calibration
    if calibration is None or plane.input_to_plane is None or plane.source_rgb is None:
        return _unavailable_physical_dimensions("CHARUCO_CALIBRATION_REQUIRED")
    policy = artifact.dimension_measurement_policy
    if calibration.reprojection_error_px > policy.max_plane_reprojection_error_px:
        return _unavailable_physical_dimensions("CALIBRATION_REPROJECTION_ERROR_ABOVE_POLICY")
    if prediction is None:
        return _unavailable_physical_dimensions("GOLDEN_SUBJECT_SEGMENTATION_REQUIRED")

    return _source_plane_physical_dimension_evidence(
        artifact,
        source_shape=np.asarray(plane.source_rgb).shape[:2],
        source_to_plane=plane.input_to_plane,
        calibration=calibration,
        calibration_support_plane=plane.calibration_support_plane,
        prediction=prediction,
        method="CHARUCO_PLANE_GOLDEN_MASK_MIN_AREA_RECT_V1",
        missing_prediction_reason="GOLDEN_SUBJECT_SEGMENTATION_REQUIRED",
        invalid_mask_reason="GOLDEN_SUBJECT_MASK_DIMENSIONS_INVALID",
        small_contour_reason="GOLDEN_SUBJECT_CONTOUR_AREA_BELOW_POLICY",
        invalid_contour_reason="GOLDEN_SUBJECT_CONTOUR_INVALID",
    )

    mask = (np.asarray(prediction.mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    expected_shape = np.asarray(plane.source_rgb).shape[:2]
    if mask.shape != expected_shape:
        return _unavailable_physical_dimensions("GOLDEN_SUBJECT_MASK_DIMENSIONS_INVALID")
    if int(cv2.countNonZero(mask)) < policy.min_contour_area_px:
        return _unavailable_physical_dimensions("GOLDEN_SUBJECT_CONTOUR_AREA_BELOW_POLICY")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [item for item in contours if cv2.contourArea(item) > 0]
    if not contours or sum(len(item) for item in contours) < policy.min_contour_points:
        return _unavailable_physical_dimensions("GOLDEN_SUBJECT_CONTOUR_INVALID")

    input_to_plane = np.asarray(plane.input_to_plane, dtype=np.float64)
    if input_to_plane.shape != (3, 3) or not np.all(np.isfinite(input_to_plane)):
        return _unavailable_physical_dimensions("METRIC_TRANSFORM_INVALID")
    metric_from_input = np.diag((
        1.0 / calibration.pixels_per_mm_x,
        1.0 / calibration.pixels_per_mm_y,
        1.0,
    )) @ input_to_plane
    if abs(float(np.linalg.det(metric_from_input))) <= 1e-12:
        return _unavailable_physical_dimensions("METRIC_TRANSFORM_INVALID")

    mapped_contours = [
        cv2.perspectiveTransform(item.astype(np.float32), metric_from_input).astype(np.float32)
        for item in contours
    ]
    all_points = np.concatenate(mapped_contours, axis=0)
    (_, _), (side_a, side_b), angle = cv2.minAreaRect(all_points)
    length_mm = float(max(side_a, side_b))
    width_mm = float(min(side_a, side_b))
    area_mm2 = float(sum(abs(cv2.contourArea(item)) for item in mapped_contours))
    if not all(math.isfinite(value) and value > 0 for value in (length_mm, width_mm, area_mm2)):
        return _unavailable_physical_dimensions("PHYSICAL_DIMENSION_GEOMETRY_INVALID")

    calibration_uncertainty_mm = calibration.reprojection_error_px / min(
        calibration.pixels_per_mm_x, calibration.pixels_per_mm_y,
    )
    boundary_uncertainty_mm = policy.segmentation_boundary_uncertainty_px / min(
        calibration.pixels_per_mm_x, calibration.pixels_per_mm_y,
    )
    edge_uncertainty_mm = calibration_uncertainty_mm + boundary_uncertainty_mm
    linear_uncertainty_mm = 2.0 * edge_uncertainty_mm
    relative_linear = linear_uncertainty_mm / width_mm
    perimeter_mm = float(sum(cv2.arcLength(item, True) for item in mapped_contours))
    area_uncertainty_mm2 = perimeter_mm * edge_uncertainty_mm + math.pi * edge_uncertainty_mm ** 2
    if not all(math.isfinite(value) and value > 0 for value in (
        linear_uncertainty_mm, relative_linear, area_uncertainty_mm2,
    )):
        return _unavailable_physical_dimensions("MEASUREMENT_UNCERTAINTY_INVALID")
    if relative_linear > policy.max_relative_linear_uncertainty:
        return _unavailable_physical_dimensions("MEASUREMENT_UNCERTAINTY_ABOVE_POLICY")

    _, _, mask_sha = _encode_binary_png(mask)
    return PhysicalDimensionEvidence(
        state="AVAILABLE",
        disclaimerCode="ENGINEERING_DIMENSION_NOT_METROLOGY_PROOF",
        method="CHARUCO_PLANE_GOLDEN_MASK_MIN_AREA_RECT_V1",
        approvalState=policy.approval_state,
        coordinateSpace="CHARUCO_BOARD_PLANE_MM",
        currentSubjectMaskSha256=mask_sha,
        lengthMm=length_mm,
        widthMm=width_mm,
        areaMm2=area_mm2,
        rotatedRectAngleDegrees=float(angle),
        calibration=MetricCalibrationEvidence(
            source=policy.calibration_source,
            fiducial=calibration.calibration_fiducial,
            detectedCornerCount=calibration.detected_corner_count,
            inlierCornerCount=calibration.inlier_corner_count,
            planeReprojectionErrorPx=calibration.reprojection_error_px,
            pixelsPerMmX=calibration.pixels_per_mm_x,
            pixelsPerMmY=calibration.pixels_per_mm_y,
        ),
        uncertainty=DimensionUncertaintyEvidence(
            method="CONSERVATIVE_CALIBRATION_PLUS_SEGMENTATION_V1",
            linearMm=linear_uncertainty_mm,
            areaMm2=area_uncertainty_mm2,
            relativeLinear=relative_linear,
        ),
    )


def _candidate_crop_box(
    x: int, y: int, width: int, height: int,
    canonical_width: int, canonical_height: int,
    policy: CandidateVerificationPolicy,
) -> tuple[int, int, int, int]:
    side = max(
        policy.minimum_crop_side_px,
        int(math.ceil(max(width, height) * (1.0 + 2.0 * policy.context_padding_ratio))),
    )
    side = min(side, canonical_width, canonical_height)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    left = max(0, min(canonical_width - side, int(round(center_x - side / 2.0))))
    top = max(0, min(canonical_height - side, int(round(center_y - side / 2.0))))
    return left, top, side, side


def _local_candidate_structure_verification(
    current_rgb: object,
    golden_rgb: object,
    subject_scope: SubjectScope,
    policy: CandidateVerificationPolicyV2,
    crop_bbox: tuple[int, int, int, int],
    candidate_bbox: tuple[int, int, int, int],
) -> tuple[str, float, float, float, float | None, float | None, str]:
    """Return bounded local fit plus photometrically normalized change ratios.

    The candidate itself is excluded from the ECC context so a real local
    change cannot pull the fit toward the anomaly. Unqualified local alignment
    intentionally returns no appearance/edge ratios and can never confirm a
    candidate.
    """
    import cv2
    import numpy as np

    current = np.asarray(current_rgb, dtype=np.uint8)
    golden = np.asarray(golden_rgb, dtype=np.uint8)
    support = np.asarray(subject_scope.support_mask, dtype=np.uint8) > 0
    left, top, width, height = crop_bbox
    x, y, candidate_width, candidate_height = candidate_bbox
    current_crop = current[top:top + height, left:left + width]
    golden_crop = golden[top:top + height, left:left + width]
    support_crop = support[top:top + height, left:left + width]

    candidate = np.zeros((height, width), dtype=np.uint8)
    candidate_left = max(0, x - left)
    candidate_top = max(0, y - top)
    candidate_right = min(width, x + candidate_width - left)
    candidate_bottom = min(height, y + candidate_height - top)
    if candidate_right <= candidate_left or candidate_bottom <= candidate_top:
        return "UNQUALIFIED", -1.0, 0.0, 0.0, None, None, "LOCAL_ALIGNMENT_UNQUALIFIED"
    candidate[candidate_top:candidate_bottom, candidate_left:candidate_right] = 1
    padding = policy.candidate_exclusion_padding_px
    kernel_side = 2 * padding + 1
    excluded = (
        cv2.dilate(candidate, np.ones((kernel_side, kernel_side), dtype=np.uint8)) > 0
        if padding > 0 else candidate > 0
    )
    context = support_crop & ~excluded
    if int(np.count_nonzero(context)) < policy.minimum_context_pixels:
        return "UNQUALIFIED", -1.0, 0.0, 0.0, None, None, "LOCAL_ALIGNMENT_UNQUALIFIED"

    def gradient_magnitude(rgb: object) -> object:
        gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(dx, dy)

    current_gradient = gradient_magnitude(current_crop)
    golden_gradient = gradient_magnitude(golden_crop)
    warp = np.eye(2, 3, dtype=np.float32)
    correlation = -1.0
    try:
        correlation, warp = cv2.findTransformECC(
            golden_gradient,
            current_gradient,
            warp,
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6),
            context.astype(np.uint8) * 255,
            5,
        )
    except cv2.error:
        pass
    correlation = float(np.clip(correlation, -1.0, 1.0)) if np.isfinite(correlation) else -1.0
    dx_px = float(np.clip(warp[0, 2], -1024.0, 1024.0)) if np.isfinite(warp[0, 2]) else 0.0
    dy_px = float(np.clip(warp[1, 2], -1024.0, 1024.0)) if np.isfinite(warp[1, 2]) else 0.0
    if (
        correlation < policy.min_local_alignment_correlation
        or abs(dx_px) > policy.max_local_translation_px
        or abs(dy_px) > policy.max_local_translation_px
    ):
        return (
            "UNQUALIFIED", correlation, dx_px, dy_px,
            None, None, "LOCAL_ALIGNMENT_UNQUALIFIED",
        )

    aligned_current = cv2.warpAffine(
        current_crop,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT101,
    )
    aligned_support = cv2.warpAffine(
        support_crop.astype(np.uint8) * 255,
        warp,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    normalization_context = context & aligned_support
    target = (candidate > 0) & support_crop & aligned_support
    if (
        int(np.count_nonzero(normalization_context)) < policy.minimum_context_pixels
        or not np.any(target)
    ):
        return (
            "UNQUALIFIED", correlation, dx_px, dy_px,
            None, None, "LOCAL_ALIGNMENT_UNQUALIFIED",
        )

    current_lab = cv2.cvtColor(aligned_current, cv2.COLOR_RGB2LAB).astype(np.float32)
    golden_lab = cv2.cvtColor(golden_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    normalized_lab = current_lab.copy()
    # Scale limits are part of OPENCV_LAB_CONTEXT_MEDIAN_MAD_V1 and the
    # analyzer runtime digest. They prevent low-texture context from producing
    # an unbounded photometric transform.
    for channel in range(3):
        current_values = current_lab[..., channel][normalization_context]
        golden_values = golden_lab[..., channel][normalization_context]
        current_median = float(np.median(current_values))
        golden_median = float(np.median(golden_values))
        current_scale = max(1.0, 1.4826 * float(np.median(np.abs(current_values - current_median))))
        golden_scale = max(1.0, 1.4826 * float(np.median(np.abs(golden_values - golden_median))))
        scale = float(np.clip(golden_scale / current_scale, 0.7, 1.4))
        normalized_lab[..., channel] = (current_lab[..., channel] - current_median) * scale + golden_median
    normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)
    delta = np.linalg.norm(
        (normalized_lab.astype(np.float32) - golden_lab) / 255.0,
        axis=2,
    )
    appearance_ratio = float(np.mean(delta[target] >= policy.appearance_delta_threshold))

    normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
    normalized_edges = cv2.Canny(cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2GRAY), 60, 140) > 0
    golden_edges = cv2.Canny(cv2.cvtColor(golden_crop, cv2.COLOR_RGB2GRAY), 60, 140) > 0
    edge_ratio = float(np.mean((normalized_edges ^ golden_edges)[target]))
    confirmation = (
        "CONFIRMED"
        if appearance_ratio >= policy.min_appearance_changed_area_ratio
        or edge_ratio >= policy.min_edge_changed_area_ratio
        else "UNCONFIRMED"
    )
    return "ALIGNED", correlation, dx_px, dy_px, appearance_ratio, edge_ratio, confirmation


def _verify_candidate_crop(
    embedder: PatchEmbedder,
    current_rgb: object,
    golden_rgb: object,
    subject_scope: SubjectScope,
    policy: CandidateVerificationPolicy | CandidateVerificationPolicyV2,
    bbox: tuple[int, int, int, int],
    golden_cache: dict[tuple[str, int, int, int, int], list[float]] | None,
    golden_cache_key: str,
    neutral_rgb: tuple[int, int, int] | None = None,
    candidate_bbox: tuple[int, int, int, int] | None = None,
) -> CandidateVerification | CandidateVerificationV2:
    import numpy as np

    raw_current = np.asarray(current_rgb, dtype=np.uint8)
    raw_golden = np.asarray(golden_rgb, dtype=np.uint8)
    current = raw_current.copy()
    golden = raw_golden.copy()
    support = np.asarray(subject_scope.support_mask, dtype=np.uint8) > 0
    if neutral_rgb is None:
        current[~support] = golden[~support]
    else:
        neutral = np.asarray(neutral_rgb, dtype=np.uint8)
        current[~support] = neutral
        golden[~support] = neutral
    left, top, width, height = bbox
    current_crop = current[top:top + height, left:left + width]
    golden_crop = golden[top:top + height, left:left + width]
    current_vector = embedder.embed_with_patches(current_crop).global_vector
    cache_key = (golden_cache_key, left, top, width, height)
    golden_vector = None if golden_cache is None else golden_cache.get(cache_key)
    if golden_vector is None:
        golden_vector = embedder.embed_with_patches(golden_crop).global_vector
        if golden_cache is not None:
            golden_cache[cache_key] = golden_vector
    distance = _cosine_distance(current_vector, golden_vector)
    priority = (
        "HIGH" if distance >= policy.high_priority_distance
        else "REVIEW" if distance >= policy.review_priority_distance
        else "LOW"
    )
    common = {
        "state": "EVALUATED",
        "method": policy.method,
        "mode": policy.mode,
        "priority": priority,
        "cropDistance": distance,
        "disclaimerCode": "CANDIDATE_VERIFICATION_NOT_DEFECT_PROOF",
    }
    if isinstance(policy, CandidateVerificationPolicyV2):
        if candidate_bbox is None:
            raise RuntimeError("LOCAL_STRUCTURE_VERIFICATION_REQUIRES_CANDIDATE_BBOX")
        (
            alignment_state, correlation, dx_px, dy_px,
            appearance_ratio, edge_ratio, confirmation,
        ) = _local_candidate_structure_verification(
            raw_current, raw_golden, subject_scope, policy, bbox, candidate_bbox,
        )
        return CandidateVerificationV2(
            **common,
            localAlignmentState=alignment_state,
            localAlignmentMethod=policy.local_alignment_method,
            localAlignmentCorrelation=correlation,
            localTranslationXPx=dx_px,
            localTranslationYPx=dy_px,
            photometricNormalization=policy.photometric_normalization,
            appearanceChangedAreaRatio=appearance_ratio,
            edgeChangedAreaRatio=edge_ratio,
            structureConfirmation=confirmation,
        )
    return CandidateVerification(**common)


def _roi_tiled_spatial_difference_evidence(
    embedder: PatchEmbedder, current_rgb: object, golden_rgb: object,
    policy: SpatialDifferencePolicy | None, canonical_width: int, canonical_height: int,
    inspection_regions, *, golden_cache: dict[tuple[str, int, int, int], PatchEmbedding] | None = None,
    golden_cache_key: str = "golden",
    current_cache: dict[tuple[int, int, int], PatchEmbedding] | None = None,
    subject_scope: SubjectScope | None = None,
    inspection_mask_override: object | None = None,
    candidate_verification_policy: CandidateVerificationPolicy | CandidateVerificationPolicyV2 | None = None,
    golden_candidate_cache: dict[tuple[str, int, int, int, int], list[float]] | None = None,
    neutral_rgb: tuple[int, int, int] | None = None,
) -> SpatialDifferenceEvidence:
    """Run real DINO patch comparison on overlapping ROI tiles.

    The output is a full-canonical evidence raster. Each local patch grid is
    projected only onto the pixels seen by DINO's Resize+CenterCrop transform;
    overlapping windows are averaged and everything outside the immutable ROI
    is zeroed before encoding.
    """
    if policy is None:
        return _unavailable_spatial_evidence("SPATIAL_DIFFERENCE_POLICY_NOT_CONFIGURED")
    if not inspection_regions:
        return _unavailable_spatial_evidence("INSPECTION_ROI_NOT_CONFIGURED")

    import cv2
    import numpy as np

    current_image = np.asarray(current_rgb, dtype=np.uint8)
    golden_image = np.asarray(golden_rgb, dtype=np.uint8)
    expected_shape = (canonical_height, canonical_width)
    if current_image.shape[:2] != expected_shape or golden_image.shape[:2] != expected_shape:
        return _unavailable_spatial_evidence("CANONICAL_IMAGE_DIMENSIONS_MISMATCH")

    accumulated = np.zeros(expected_shape, dtype=np.float32)
    weights = np.zeros(expected_shape, dtype=np.float32)
    support_mask = (
        np.asarray(subject_scope.support_mask, dtype=np.uint8)
        if subject_scope is not None else None
    )
    core_mask = (
        np.asarray(subject_scope.core_mask, dtype=np.uint8)
        if subject_scope is not None else None
    )
    boundary_mask = (
        np.asarray(subject_scope.boundary_mask, dtype=np.uint8)
        if subject_scope is not None else None
    )
    inspection_mask = np.zeros(expected_shape, dtype=np.uint8)
    for region in inspection_regions:
        inspection_mask[region.y:region.y + region.height, region.x:region.x + region.width] = 255
    if inspection_mask_override is not None:
        authoritative_mask = np.asarray(inspection_mask_override, dtype=np.uint8)
        if authoritative_mask.shape != expected_shape:
            return _unavailable_spatial_evidence("INSPECTION_ROI_MASK_DIMENSIONS_MISMATCH")
        inspection_mask = cv2.bitwise_and(inspection_mask, authoritative_mask)
    analysis_mask = inspection_mask if support_mask is None else cv2.bitwise_and(inspection_mask, support_mask)
    scorer_tile_digests: list[ScorerInputTileDigest] = []
    tile_boxes = _roi_tile_boxes(inspection_regions, canonical_width, canonical_height)
    for tile_index, (tile_x, tile_y, side) in enumerate(tile_boxes, start=1):
        current_tile = current_image[tile_y:tile_y + side, tile_x:tile_x + side].copy()
        golden_tile = golden_image[tile_y:tile_y + side, tile_x:tile_x + side].copy()
        if neutral_rgb is not None:
            tile_scope = analysis_mask[tile_y:tile_y + side, tile_x:tile_x + side] > 0
            neutral = np.asarray(neutral_rgb, dtype=np.uint8)
            current_tile[~tile_scope] = neutral
            golden_tile[~tile_scope] = neutral
            current_identity = hashlib.sha256()
            current_identity.update(f"{tile_x}:{tile_y}:{side}:RGB8:".encode("ascii"))
            current_identity.update(current_tile.tobytes(order="C"))
            golden_identity = hashlib.sha256()
            golden_identity.update(f"{tile_x}:{tile_y}:{side}:RGB8:".encode("ascii"))
            golden_identity.update(golden_tile.tobytes(order="C"))
            scorer_tile_digests.append(ScorerInputTileDigest(
                id=f"ROI-TILE-{tile_index:03d}",
                bboxNormalized=BboxNormalized(
                    x=tile_x / canonical_width, y=tile_y / canonical_height,
                    width=side / canonical_width, height=side / canonical_height,
                ),
                currentSha256=current_identity.hexdigest(),
                referenceSha256=golden_identity.hexdigest(),
            ))
        elif support_mask is not None:
            tile_support = support_mask[tile_y:tile_y + side, tile_x:tile_x + side] > 0
            # Suppress background before DINO inference so background features
            # cannot change attention and leak back into subject patch scores.
            current_tile[~tile_support] = golden_tile[~tile_support]
        current_cache_key = (tile_x, tile_y, side)
        current_patches = None if current_cache is None else current_cache.get(current_cache_key)
        if current_patches is None:
            current_patches = embedder.embed_with_patches(current_tile)
            if current_cache is not None:
                current_cache[current_cache_key] = current_patches
        cache_key = (golden_cache_key, tile_x, tile_y, side)
        golden_patches = None if golden_cache is None else golden_cache.get(cache_key)
        if golden_patches is None:
            golden_patches = embedder.embed_with_patches(golden_tile)
            if golden_cache is not None:
                golden_cache[cache_key] = golden_patches
        grid = _patch_distance_grid(current_patches, golden_patches)

        crop_left, crop_top, crop_width, crop_height = _dino_input_crop_box(side, side)
        x0 = max(0, int(round(tile_x + crop_left)))
        y0 = max(0, int(round(tile_y + crop_top)))
        x1 = min(canonical_width, int(round(tile_x + crop_left + crop_width)))
        y1 = min(canonical_height, int(round(tile_y + crop_top + crop_height)))
        if x1 <= x0 or y1 <= y0:
            continue
        projected = cv2.resize(grid, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        accumulated[y0:y1, x0:x1] += projected
        weights[y0:y1, x0:x1] += 1.0

    covered = weights > 0
    if not np.any(covered):
        return _unavailable_spatial_evidence("ROI_TILES_DID_NOT_COVER_CANONICAL_IMAGE")
    distance = np.zeros(expected_shape, dtype=np.float32)
    distance[covered] = accumulated[covered] / weights[covered]

    distance[analysis_mask == 0] = 0.0

    display = np.clip(distance / 2.0, 0.0, 1.0)
    heatmap_color = cv2.applyColorMap((display * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap_color[analysis_mask == 0] = 0
    ok, map_encoded = cv2.imencode(".png", heatmap_color)
    if not ok:
        raise RuntimeError("SPATIAL_MAP_ENCODING_FAILED")

    raw_binary = (distance >= policy.anomaly_distance_threshold).astype(np.uint8) * 255
    raw_binary[analysis_mask == 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((raw_binary > 0).astype(np.uint8), connectivity=8)
    inspection_area = float(np.count_nonzero(analysis_mask))
    min_area = max(1.0, policy.min_region_area_ratio * inspection_area)
    candidates: list[tuple[float, int, int, int, int, float, float, int, str | None]] = []
    background_suppressed = 0
    small_region_suppressed = 0
    for label in range(1, num_labels):
        component = labels == label
        if core_mask is not None and not np.any(core_mask[component] > 0):
            # The padded support deliberately gives DINO some edge context,
            # but a component that never reaches the immutable Golden subject
            # is background leakage and must not become a returned candidate.
            background_suppressed += 1
            continue
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            small_region_suppressed += 1
            continue
        x, y = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        width, height = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
        values = distance[component]
        kind = None
        if core_mask is not None and boundary_mask is not None:
            inward_subject = (core_mask > 0) & (boundary_mask == 0)
            kind = "SUBJECT_INTERIOR" if np.all(inward_subject[component]) else "SUBJECT_BOUNDARY"
        candidates.append((
            area, x, y, width, height,
            float(np.clip(values.max() / 2.0, 0.0, 1.0)),
            float(np.clip(values.mean() / 2.0, 0.0, 1.0)),
            label, kind,
        ))
    candidates.sort(
        key=lambda item: (
            1 if item[8] == "SUBJECT_INTERIOR" else 0,
            item[5], item[0],
        ),
        reverse=True,
    )
    selected = candidates[:policy.max_regions]
    verified: list[tuple[
        tuple[float, int, int, int, int, float, float, int, str | None],
        CandidateVerification | CandidateVerificationV2 | None,
    ]] = []
    for candidate in selected:
        verification = None
        if candidate_verification_policy is not None:
            if subject_scope is None:
                raise RuntimeError("CANDIDATE_VERIFICATION_REQUIRES_SUBJECT_SCOPE")
            _, x, y, width, height, _, _, _, _ = candidate
            crop_box = _candidate_crop_box(
                x, y, width, height, canonical_width, canonical_height,
                candidate_verification_policy,
            )
            verification = _verify_candidate_crop(
                embedder, current_image, golden_image, subject_scope,
                candidate_verification_policy, crop_box, golden_candidate_cache,
                golden_cache_key, neutral_rgb, (x, y, width, height),
            )
        verified.append((candidate, verification))
    priority_rank = {"HIGH": 2, "REVIEW": 1, "LOW": 0}
    verified.sort(
        key=lambda item: (
            1 if item[0][8] == "SUBJECT_INTERIOR" else 0,
            -1 if item[1] is None else priority_rank[item[1].priority],
            -1.0 if item[1] is None else item[1].crop_distance,
            item[0][5], item[0][0],
        ),
        reverse=True,
    )
    verifier_suppressed = 0
    if candidate_verification_policy is not None and candidate_verification_policy.mode == "GATE":
        def passes_verifier(verification: CandidateVerification | CandidateVerificationV2 | None) -> bool:
            if verification is None or verification.priority == "LOW":
                return verification is None
            return (
                not isinstance(verification, CandidateVerificationV2)
                or verification.structure_confirmation == "CONFIRMED"
            )

        verifier_suppressed = sum(1 for _, verification in verified if not passes_verifier(verification))
        verified = [
            item for item in verified
            if passes_verifier(item[1])
        ]
    retained_binary = np.zeros_like(raw_binary)
    for candidate, _ in verified:
        retained_binary[labels == candidate[7]] = 255
    ok, mask_encoded = cv2.imencode(".png", retained_binary)
    if not ok:
        raise RuntimeError("SPATIAL_MASK_ENCODING_FAILED")
    regions = [DifferenceRegion(
        id=f"D-{index + 1:03d}",
        bboxNormalized=BboxNormalized(
            x=x / canonical_width, y=y / canonical_height,
            width=min(width / canonical_width, 1.0 - x / canonical_width),
            height=min(height / canonical_height, 1.0 - y / canonical_height),
        ),
        peakScore=peak, meanScore=mean, kind=kind, verification=verification,
    ) for index, ((_, x, y, width, height, peak, mean, _, kind), verification) in enumerate(verified)]

    map_bytes = map_encoded.tobytes()
    mask_bytes = mask_encoded.tobytes()
    fields: dict[str, object] = {}
    generation_method = "ROI_TILED_PATCH_DISTANCE"
    if subject_scope is not None:
        raw_mask_bytes, raw_mask_b64, raw_mask_sha = _encode_binary_png(raw_binary)
        del raw_mask_bytes
        fields = {
            "rawThresholdMaskPngBase64": raw_mask_b64,
            "rawThresholdMaskSha256": raw_mask_sha,
            "candidateFilter": CandidateFilter(
                rawComponentCount=max(0, num_labels - 1),
                retainedComponentCount=len(verified),
                suppressedByBackgroundCount=background_suppressed,
                suppressedSmallRegionCount=small_region_suppressed,
                suppressedByLimitCount=max(0, len(candidates) - len(selected)),
                suppressedByVerifierCount=verifier_suppressed,
                maskSemantics="RETAINED_CANDIDATES",
            ),
        }
        generation_method = (
            "PAIRED_INTERIOR_ROI_TILED_PATCH_DISTANCE"
            if subject_scope.paired_interior
            else "SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE"
        )
    analyzed_y, analyzed_x = np.where(analysis_mask > 0)
    if analyzed_x.size == 0 or analyzed_y.size == 0:
        return _unavailable_spatial_evidence("ANALYZED_REGION_EMPTY")
    analyzed_left, analyzed_top = int(analyzed_x.min()), int(analyzed_y.min())
    analyzed_right, analyzed_bottom = int(analyzed_x.max()) + 1, int(analyzed_y.max()) + 1
    return SpatialDifferenceEvidence(
        state="AVAILABLE", disclaimerCode="DIFFERENCE_NOT_DEFECT_PROOF",
        generationMethod=generation_method,
        evidenceRegionNormalized=(
            BboxNormalized(
                x=analyzed_left / canonical_width,
                y=analyzed_top / canonical_height,
                width=(analyzed_right - analyzed_left) / canonical_width,
                height=(analyzed_bottom - analyzed_top) / canonical_height,
            ) if neutral_rgb is not None else BboxNormalized(x=0.0, y=0.0, width=1.0, height=1.0)
        ),
        evidenceCoordinateSpace="TARGET_CANONICAL_IMAGE" if neutral_rgb is not None else None,
        scorerInputTileDigests=scorer_tile_digests or None,
        regions=regions,
        mapPngBase64=base64.b64encode(map_bytes).decode("ascii"),
        maskPngBase64=base64.b64encode(mask_bytes).decode("ascii"),
        mapSha256=hashlib.sha256(map_bytes).hexdigest(),
        maskSha256=hashlib.sha256(mask_bytes).hexdigest(),
        **fields,
    )


def _spatial_difference_evidence(
    patches: PatchEmbedding, golden: GoldenEmbedding, policy: SpatialDifferencePolicy | None,
    canonical_width: int, canonical_height: int, inspection_regions=(), inspection_mask_override: object | None = None,
) -> SpatialDifferenceEvidence:
    if policy is None:
        return _unavailable_spatial_evidence("SPATIAL_DIFFERENCE_POLICY_NOT_CONFIGURED")
    if golden.patch_values is None or golden.patch_grid_height != patches.grid_height or golden.patch_grid_width != patches.grid_width:
        return _unavailable_spatial_evidence("GOLDEN_PATCH_FEATURES_UNAVAILABLE")

    import cv2
    import numpy as np

    current = np.asarray(patches.patch_grid, dtype=np.float64)
    golden_patches = np.asarray(golden.patch_values, dtype=np.float64)
    current_norm = np.linalg.norm(current, axis=1)
    golden_norm = np.linalg.norm(golden_patches, axis=1)
    valid = (current_norm > 1e-12) & (golden_norm > 1e-12)
    similarity = np.zeros(current.shape[0], dtype=np.float64)
    similarity[valid] = np.sum(current[valid] * golden_patches[valid], axis=1) / (current_norm[valid] * golden_norm[valid])
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    distance[~valid] = 0.0
    grid = distance.reshape(patches.grid_height, patches.grid_width).astype(np.float32)

    # Upscale the coarse patch grid to the exact pixel footprint DINO analyzed
    # (DINO_INPUT_SIZE x DINO_INPUT_SIZE), never to the full canonical ROI.
    upscaled = cv2.resize(grid, (DINO_INPUT_SIZE, DINO_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    # Rasterize the immutable inspection scope into the exact 224x224 DINO
    # crop before any evidence bytes are encoded. Filtering only the reported
    # component list would leave misleading heatmap/mask pixels outside ROI.
    left, top, crop_width, crop_height = _dino_input_crop_box(canonical_width, canonical_height)
    inspection_mask = np.full((DINO_INPUT_SIZE, DINO_INPUT_SIZE), 255, dtype=np.uint8)
    if inspection_regions:
        inspection_mask.fill(0)
        for region in inspection_regions:
            intersection_left = max(left, float(region.x))
            intersection_top = max(top, float(region.y))
            intersection_right = min(left + crop_width, float(region.x + region.width))
            intersection_bottom = min(top + crop_height, float(region.y + region.height))
            if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
                continue
            x0 = max(0, min(DINO_INPUT_SIZE, int(math.floor((intersection_left - left) / crop_width * DINO_INPUT_SIZE))))
            y0 = max(0, min(DINO_INPUT_SIZE, int(math.floor((intersection_top - top) / crop_height * DINO_INPUT_SIZE))))
            x1 = max(0, min(DINO_INPUT_SIZE, int(math.ceil((intersection_right - left) / crop_width * DINO_INPUT_SIZE))))
            y1 = max(0, min(DINO_INPUT_SIZE, int(math.ceil((intersection_bottom - top) / crop_height * DINO_INPUT_SIZE))))
            inspection_mask[y0:y1, x0:x1] = 255
    if inspection_mask_override is not None:
        canonical_mask = np.asarray(inspection_mask_override, dtype=np.uint8)
        if canonical_mask.shape != (canonical_height, canonical_width):
            return _unavailable_spatial_evidence("INSPECTION_ROI_MASK_DIMENSIONS_MISMATCH")
        x0 = max(0, int(math.floor(left)))
        y0 = max(0, int(math.floor(top)))
        x1 = min(canonical_width, int(math.ceil(left + crop_width)))
        y1 = min(canonical_height, int(math.ceil(top + crop_height)))
        crop_mask = canonical_mask[y0:y1, x0:x1]
        authoritative_crop = cv2.resize(crop_mask, (DINO_INPUT_SIZE, DINO_INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
        inspection_mask = cv2.bitwise_and(inspection_mask, authoritative_crop)

    display = np.clip(upscaled / 2.0, 0.0, 1.0)
    display[inspection_mask == 0] = 0.0
    heatmap_color = cv2.applyColorMap((display * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap_color[inspection_mask == 0] = 0
    ok, map_encoded = cv2.imencode(".png", heatmap_color)
    if not ok:
        raise RuntimeError("SPATIAL_MAP_ENCODING_FAILED")

    binary = (upscaled >= policy.anomaly_distance_threshold).astype(np.uint8) * 255
    binary[inspection_mask == 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    min_area = policy.min_region_area_ratio * DINO_INPUT_SIZE * DINO_INPUT_SIZE
    candidates: list[tuple[float, float, float, float, float, float, float, int]] = []
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        w, h = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
        component_values = upscaled[labels == label]
        peak = float(np.clip(component_values.max() / 2.0, 0.0, 1.0))
        mean = float(np.clip(component_values.mean() / 2.0, 0.0, 1.0))
        # Component bbox is in DINO_INPUT_SIZE pixel space; project through the
        # crop box into canonical-ROI pixel coordinates immediately so the
        # inspection-region filter and the reported bbox use the same values.
        canon_x = left + (x / DINO_INPUT_SIZE) * crop_width
        canon_y = top + (y / DINO_INPUT_SIZE) * crop_height
        canon_w = (w / DINO_INPUT_SIZE) * crop_width
        canon_h = (h / DINO_INPUT_SIZE) * crop_height
        candidates.append((area, canon_x, canon_y, canon_w, canon_h, peak, mean, label))
    # A difference driven by a stable alignment/held-out landmark (not the
    # inspected content) is not evidence about the equipment; only report
    # components that actually overlap a declared inspection region.
    if inspection_regions:
        candidates = [
            candidate for candidate in candidates
            if any(_overlaps(candidate[1], candidate[2], candidate[3], candidate[4], region) for region in inspection_regions)
        ]
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[: policy.max_regions]
    retained_binary = np.zeros_like(binary)
    for candidate in selected:
        retained_binary[labels == candidate[7]] = 255
    ok, mask_encoded = cv2.imencode(".png", retained_binary)
    if not ok:
        raise RuntimeError("SPATIAL_MASK_ENCODING_FAILED")
    regions: list[DifferenceRegion] = []
    for index, (_, canon_x, canon_y, canon_w, canon_h, peak, mean, _) in enumerate(selected):
        x_norm = canon_x / canonical_width
        y_norm = canon_y / canonical_height
        width_norm = min(canon_w / canonical_width, 1.0 - x_norm)
        height_norm = min(canon_h / canonical_height, 1.0 - y_norm)
        regions.append(DifferenceRegion(
            id=f"D-{index + 1:03d}",
            bboxNormalized=BboxNormalized(x=max(0.0, x_norm), y=max(0.0, y_norm), width=max(1e-6, width_norm), height=max(1e-6, height_norm)),
            peakScore=peak, meanScore=mean,
        ))

    map_bytes = map_encoded.tobytes()
    mask_bytes = mask_encoded.tobytes()
    evidence_x_norm = left / canonical_width
    evidence_y_norm = top / canonical_height
    return SpatialDifferenceEvidence(
        state="AVAILABLE", disclaimerCode="DIFFERENCE_NOT_DEFECT_PROOF", generationMethod="PATCH_DISTANCE",
        evidenceRegionNormalized=BboxNormalized(
            x=max(0.0, evidence_x_norm), y=max(0.0, evidence_y_norm),
            width=min(crop_width / canonical_width, 1.0 - evidence_x_norm),
            height=min(crop_height / canonical_height, 1.0 - evidence_y_norm),
        ),
        regions=regions,
        mapPngBase64=base64.b64encode(map_bytes).decode("ascii"),
        maskPngBase64=base64.b64encode(mask_bytes).decode("ascii"),
        mapSha256=hashlib.sha256(map_bytes).hexdigest(),
        maskSha256=hashlib.sha256(mask_bytes).hexdigest(),
    )


class ProductionAnalyzer:
    def __init__(
        self,
        settings: Settings,
        normalizer: Normalizer | None = None,
        embedder: Embedder | None = None,
        subject_segmenter: SubjectSegmenter | None = None,
        relative_depth_estimator: RelativeDepthEstimator | None = None,
        reference_board_verifier: ReferenceBoardVerifier | None = None,
    ):
        self.settings = settings
        self.adapter = LocalDinoV2Adapter(settings.model_repo, settings.model_weights)
        self.normalizer = normalizer or OpenCvCharucoNormalizer(
            allow_target_only_alignment=settings.allow_target_only_alignment,
            allow_contour_anchor_alignment=settings.engineering_contour_alignment_enabled,
        )
        self.embedder = embedder or DinoV2Embedder(self.adapter, device=settings.device)
        self.subject_segmenter = subject_segmenter
        self.reference_board_verifier = reference_board_verifier or ReferenceBoardVerifier()
        # A missing learned-depth model is non-fatal: the configured rig datum
        # or conservative offset prior still produces a bounded estimate.
        self.relative_depth_estimator = relative_depth_estimator
        self._relative_depth_status = "INJECTED" if relative_depth_estimator is not None else "UNCONFIGURED"
        if self.relative_depth_estimator is None and (
            settings.relative_depth_repo is not None or settings.relative_depth_weights is not None
        ):
            try:
                if settings.relative_depth_repo is None or settings.relative_depth_weights is None:
                    raise RuntimeError("DEPTH_ANYTHING_LOCAL_ARTIFACT_UNAVAILABLE")
                if (
                    settings.relative_depth_repository_version is None
                    or settings.relative_depth_weights_sha256 is None
                ):
                    raise RuntimeError("DEPTH_ANYTHING_DIGEST_NOT_PINNED")
                if digest_directory(settings.relative_depth_repo) != settings.relative_depth_repository_version:
                    raise RuntimeError("DEPTH_ANYTHING_REPOSITORY_DIGEST_MISMATCH")
                if digest_file(settings.relative_depth_weights) != settings.relative_depth_weights_sha256:
                    raise RuntimeError("DEPTH_ANYTHING_WEIGHTS_DIGEST_MISMATCH")
                self.relative_depth_estimator = DepthAnythingV2RelativeDepthEstimator(
                    settings.relative_depth_repo, settings.relative_depth_weights,
                    encoder=settings.relative_depth_encoder, device=settings.relative_depth_device,
                )
                self._relative_depth_status = "READY"
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                # Keep the board-pose prior available and make the absence
                # observable in readiness metadata rather than altering a
                # request's metric result unpredictably.
                self._relative_depth_status = str(exc)[:160]
        self._artifact: (
            ProductionArtifact | ProductionArtifactV12 | ProductionArtifactV13
            | ProductionArtifactV14 | ProductionArtifactV15 | ProductionArtifactV16
            | ProductionArtifactV17 | ProductionArtifactV18 | ProductionArtifactV19 | None
        ) = None
        # Golden ROI tiles are immutable and hash-bound to the artifact. Keep
        # their real DINO patch features for the process lifetime so subsequent
        # engineering comparisons only infer the current capture tiles.
        self._golden_tile_cache: dict[tuple[str, int, int, int], PatchEmbedding] = {}
        self._golden_candidate_cache: dict[tuple[str, int, int, int, int], list[float]] = {}
        # Readiness re-verifies digests over an 88MB weights file and the full
        # vendored model repository tree; both are immutable, read-only mounts
        # in production (see docs/phone_dino_service_development_plan.md §12),
        # so a *positive* result is cached for the process lifetime instead of
        # re-hashing hundreds of MB on every request. A negative result is
        # never cached, so a not-yet-mounted artifact keeps being re-checked.
        self._readiness_cache: tuple[bool, str | None] | None = None

    def _warm_golden_spatial_cache(self) -> None:
        """Precompute immutable Golden patch tokens before accepting traffic."""
        artifact = self._artifact
        if (
            not isinstance(artifact, ProductionArtifactV15)
            or isinstance(artifact, ProductionArtifactV16)
            or artifact.spatial_difference_policy is None
            or not hasattr(self.embedder, "embed_with_patches")
        ):
            return

        import cv2
        import numpy as np

        inspection_mask = np.asarray(inspection_roi_image(artifact.inspection_roi), dtype=np.uint8)
        for golden in artifact.golden_embeddings:
            golden_bytes = base64.b64decode(golden.canonical_image_png_base64, validate=True)
            golden_bgr = cv2.imdecode(np.frombuffer(golden_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if golden_bgr is None:
                raise RuntimeError("GOLDEN_CANONICAL_IMAGE_INVALID")
            golden_rgb = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2RGB)
            scope = _subject_scope(artifact, golden)
            analysis_mask = cv2.bitwise_and(inspection_mask, np.asarray(scope.support_mask, dtype=np.uint8))
            generated_tiles = _scorer_input_tiles(
                golden_rgb,
                artifact.inspection_roi.inspection_regions,
                artifact.target_alignment.canonical_width,
                artifact.target_alignment.canonical_height,
                analysis_mask,
                artifact.scorer_input_contract.neutral_rgb,
            )
            stored_tiles = golden.scorer_input_tiles or []
            if len(generated_tiles) != len(stored_tiles):
                raise RuntimeError("GOLDEN_SCORER_INPUT_TILE_COUNT_MISMATCH")
            cache_key = golden.canonical_sha256 or golden.id
            cache_key += ":" + scope.evidence.subject_mask_sha256
            for generated, stored in zip(generated_tiles, stored_tiles, strict=True):
                if (
                    generated.id != stored.id
                    or (generated.x, generated.y, generated.side) != (stored.x, stored.y, stored.side)
                    or generated.sha256 != stored.tile_sha256
                ):
                    raise RuntimeError("GOLDEN_SCORER_INPUT_TILE_IDENTITY_MISMATCH")
                patch_key = (cache_key, generated.x, generated.y, generated.side)
                if patch_key not in self._golden_tile_cache:
                    self._golden_tile_cache[patch_key] = self.embedder.embed_with_patches(generated.rgb)  # type: ignore[union-attr]

    def readiness(self) -> tuple[bool, str | None]:
        if self._readiness_cache is not None:
            return self._readiness_cache
        result = self._check_readiness()
        if result[0]:
            self._readiness_cache = result
        return result

    def _check_readiness(self) -> tuple[bool, str | None]:
        if self.settings.artifact_manifest is None or not self.settings.artifact_manifest.is_file():
            return False, "ARTIFACT_MANIFEST_NOT_AVAILABLE"
        if self.settings.artifact_package_digest is None:
            return False, "ARTIFACT_PACKAGE_DIGEST_NOT_PINNED"
        ready, reason = self.adapter.readiness()
        if not ready:
            return False, reason
        try:
            self._artifact = load_artifact(
                self.settings.artifact_manifest, self.settings.artifact_package_digest, self.settings.model_weights
            )
            if isinstance(self._artifact, ProductionArtifactV12):
                require_inspection_roi(self._artifact)
            if isinstance(self._artifact, ProductionArtifactV13):
                require_subject_segmentation(self._artifact)
                if not isinstance(self._artifact, ProductionArtifactV15) and len(self._artifact.golden_embeddings) != 1:
                    return False, "SUBJECT_GATED_MULTI_GOLDEN_SELECTION_NOT_QUALIFIED"
                if (
                    not self.settings.engineering_real_model_enabled
                    and self._artifact.subject_segmentation.approval_state != "APPROVED"
                ):
                    return False, "SUBJECT_MASK_NOT_APPROVED"
            if isinstance(self._artifact, ProductionArtifactV14) and not self.settings.engineering_real_model_enabled:
                if self._artifact.subject_alignment.approval_state != "APPROVED":
                    return False, "SUBJECT_ALIGNMENT_NOT_APPROVED"
                if self._artifact.candidate_verification_policy.approval_state != "APPROVED":
                    return False, "CANDIDATE_VERIFICATION_NOT_APPROVED"
            if isinstance(self._artifact, ProductionArtifactV18) and not self.settings.engineering_real_model_enabled:
                if self._artifact.dimension_measurement_policy.approval_state != "APPROVED":
                    return False, "DIMENSION_MEASUREMENT_NOT_APPROVED"
            if isinstance(self._artifact, ProductionArtifactV16) and self.subject_segmenter is None:
                repository = self.settings.subject_segmenter_repo
                weights = self.settings.subject_segmenter_weights
                if repository is None or not repository.is_dir():
                    return False, "SUBJECT_SEGMENTER_REPOSITORY_NOT_AVAILABLE"
                if weights is None or not weights.is_file():
                    return False, "SUBJECT_SEGMENTER_WEIGHTS_NOT_AVAILABLE"
                subject = self._artifact.subject_segmentation
                if digest_directory(repository) != subject.model_repository_version:
                    return False, "SUBJECT_SEGMENTER_REPOSITORY_DIGEST_MISMATCH"
                if digest_file(weights) != subject.model_weights_sha256:
                    return False, "SUBJECT_SEGMENTER_WEIGHTS_DIGEST_MISMATCH"
                self.subject_segmenter = MobileSamSegmenter(
                    repository, weights, self.settings.subject_segmenter_device,
                )
            if self._artifact.analyzer_runtime_version != RUNTIME_DIGEST:
                return False, "ANALYZER_RUNTIME_VERSION_MISMATCH"
            if self.settings.model_repository_version is None:
                return False, "MODEL_REPOSITORY_VERSION_NOT_PINNED"
            if self._artifact.model_repository_version != self.settings.model_repository_version:
                return False, "MODEL_REPOSITORY_VERSION_MISMATCH"
            if digest_directory(self.settings.model_repo) != self._artifact.model_repository_version:
                return False, "MODEL_REPOSITORY_DIGEST_MISMATCH"
            if isinstance(self.normalizer, OpenCvCharucoNormalizer):
                import cv2
                import numpy
                # Decode and verify the hash-bound target reference during readiness.
                reference = base64.b64decode(self._artifact.target_alignment.reference_image_base64, validate=True)
                if "sha256:" + hashlib.sha256(reference).hexdigest() != self._artifact.target_alignment.reference_image_sha256:
                    return False, "TARGET_REFERENCE_DIGEST_MISMATCH"
                decoded_reference = cv2.imdecode(numpy.frombuffer(reference, dtype=numpy.uint8), cv2.IMREAD_COLOR)
                policy = self._artifact.target_alignment
                if decoded_reference is None or decoded_reference.shape[:2] != (policy.canonical_height, policy.canonical_width):
                    return False, "TARGET_REFERENCE_INVALID"
            if isinstance(self.embedder, DinoV2Embedder):
                import torch
                import torchvision
        except ArtifactError as exc:
            return False, str(exc)
        except (ImportError, OSError):
            return False, "PRODUCTION_DEPENDENCY_NOT_AVAILABLE"
        return True, None

    def warm_up(self) -> tuple[bool, str | None]:
        """Pay artifact, model and immutable Golden inference costs before traffic arrives."""
        ready, reason = self.readiness()
        if ready:
            try:
                if isinstance(self.embedder, DinoV2Embedder):
                    self.embedder.warm_up()
                if isinstance(self._artifact, ProductionArtifactV16):
                    if self.subject_segmenter is None:
                        raise RuntimeError("SUBJECT_SEGMENTER_NOT_CONFIGURED")
                    warm_segmenter = getattr(self.subject_segmenter, "warm_up", None)
                    if callable(warm_segmenter):
                        warm_segmenter()
                self._warm_golden_spatial_cache()
            except (ArtifactError, RuntimeError, OSError, ValueError) as exc:
                self._readiness_cache = None
                return False, f"MODEL_WARM_UP_FAILED:{exc}"
        return ready, reason

    def readiness_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "supportedSchemas": ["1.0", "1.1", "1.2", "1.3"],
            "capabilities": [],
        }
        if isinstance(self._artifact, ProductionArtifactV13):
            subject = self._artifact.subject_segmentation
            metadata.update({
                "capabilities": ["SUBJECT_SEGMENTATION_V1"],
                "subjectSegmentation": {
                    "method": subject.method,
                    "usageMode": subject.usage_mode,
                    "approvalState": subject.approval_state,
                    "subjectMaskSha256": subject.golden_masks[0].mask_sha256,
                    "modelRepositoryVersion": subject.model_repository_version,
                    "modelWeightsSha256": subject.model_weights_sha256,
                    "goldenMasks": [
                        {"goldenId": item.golden_id, "maskSha256": item.mask_sha256}
                        for item in subject.golden_masks
                    ],
                },
            })
        if isinstance(self._artifact, ProductionArtifactV14):
            alignment = self._artifact.subject_alignment
            verification = self._artifact.candidate_verification_policy
            metadata["capabilities"] = [
                "SUBJECT_SEGMENTATION_V1", "SUBJECT_ALIGNMENT_ECC_V1", "CANDIDATE_VERIFICATION_V1",
            ]
            metadata["subjectAlignment"] = {
                "method": alignment.method,
                "approvalState": alignment.approval_state,
            }
            metadata["candidateVerification"] = {
                "method": verification.method,
                "mode": verification.mode,
                "approvalState": verification.approval_state,
            }
        if isinstance(self._artifact, ProductionArtifactV15):
            profile = self._artifact.recipe_analysis_profile
            scorer = self._artifact.scorer_input_contract
            metadata["capabilities"] = [
                "SUBJECT_SEGMENTATION_V1", "SUBJECT_ALIGNMENT_ECC_V1", "CANDIDATE_VERIFICATION_V1",
                "INSPECTION_ROI_ONLY_SCORING_V1", "RECIPE_ANALYSIS_PROFILE_V1",
            ]
            metadata["recipeAnalysisProfile"] = profile.model_dump(by_alias=True, mode="json")
            metadata["scorerInputContract"] = scorer.model_dump(by_alias=True, mode="json")
        if isinstance(self._artifact, ProductionArtifactV16):
            metadata["capabilities"] = [
                "SUBJECT_SEGMENTATION_V1", "SUBJECT_ALIGNMENT_ECC_V1", "CANDIDATE_VERIFICATION_V1",
                "INSPECTION_ROI_ONLY_SCORING_V1", "RECIPE_ANALYSIS_PROFILE_V1",
                "CURRENT_SUBJECT_SEGMENTATION_V1", "PAIRED_SUBJECT_INTERIOR_V1",
                "SUBJECT_BOUNDARY_GEOMETRY_V1", "ALIGNMENT_FAIL_CLOSED_DINO_V1",
            ]
            metadata["currentSubjectSegmentation"] = (
                self._artifact.current_subject_segmentation.model_dump(by_alias=True, mode="json")
            )
        if isinstance(self._artifact, ProductionArtifactV17):
            metadata["capabilities"] = [
                "SUBJECT_SEGMENTATION_V1", "SUBJECT_ALIGNMENT_ECC_V1", "CANDIDATE_VERIFICATION_V1",
                "INSPECTION_ROI_ONLY_SCORING_V1", "RECIPE_ANALYSIS_PROFILE_V1",
                "CURRENT_SUBJECT_SEGMENTATION_V1", "PAIRED_SUBJECT_INTERIOR_V1",
                "SUBJECT_BOUNDARY_GEOMETRY_V1", "ALIGNMENT_FAIL_CLOSED_DINO_V1",
                "CANDIDATE_LOCAL_ALIGNMENT_V1", "CANDIDATE_PHOTOMETRIC_NORMALIZATION_V1",
                "CANDIDATE_STRUCTURE_CONFIRMATION_V1",
            ]
        if isinstance(self._artifact, ProductionArtifactV18):
            metadata["supportedSchemas"] = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]
            metadata["capabilities"] = [
                *metadata["capabilities"],
                "CHARUCO_CAPTURE_SCALE_V1", "CURRENT_SUBJECT_PHYSICAL_DIMENSIONS_V1",
                "SOURCE_RESOLUTION_METRICS_V1", "CALIBRATION_SUPPORT_GATE_V1",
                "DIMENSION_UNCERTAINTY_V1", "DIMENSION_FAIL_CLOSED_V1",
                "GOLDEN_DIMENSION_BASELINE_V1", "OUTER_ARUCO_IDENTIFICATION_ONLY_V1",
                "CANDIDATE_DIMENSION_CHARUCO_V1", "CANDIDATE_DIMENSION_SUPPORT_GATE_V1",
                "BACKGROUND_BOARD_PNP_FRONT_OFFSET_DIMENSIONS_V1",
                "BACKGROUND_BOARD_DEPTH_INTERVAL_ESTIMATE_V1",
            ]
            metadata["dimensionMeasurement"] = {
                **self._artifact.dimension_measurement_policy.model_dump(by_alias=True, mode="json"),
                "calibrationBoard": self._artifact.board.model_dump(by_alias=True, mode="json"),
            }
            metadata["relativeDepth"] = {
                "available": self.relative_depth_estimator is not None,
                "status": self._relative_depth_status,
                "provider": "DEPTH_ANYTHING_V2" if self.relative_depth_estimator is not None else None,
            }
        if isinstance(self._artifact, ProductionArtifactV19):
            metadata["supportedSchemas"] = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]
            metadata["capabilities"] = [
                *metadata["capabilities"],
                "HYBRID_REFERENCE_BOARD_V1", "QR_CHARUCO_COLOCATION_GATE_V1",
                "QR_IDENTITY_CHARUCO_SCALE_ONLY_V1",
            ]
            metadata["referenceBoard"] = {
                "metricScaleSource": self._artifact.reference_board.metric_scale_source,
                "templateManifestSha256": self._artifact.reference_board.template_manifest_sha256,
                "installationDigest": self._artifact.reference_board.installation_digest,
                "minQrSidePx": self._artifact.reference_board.min_qr_side_px,
                "minCharucoCorners": self._artifact.reference_board.min_charuco_corners,
                "maxQrToCharucoResidualMm": self._artifact.reference_board.max_qr_to_charuco_residual_mm,
            }
        return metadata

    def measure_golden_dimensions(
        self,
        request: GoldenDimensionRequest,
        image: DecodedImage,
    ) -> GoldenDimensionObservation:
        """Establish a Golden baseline without comparing it to the active Golden."""
        ready, reason = self.readiness()
        if not ready or not isinstance(self._artifact, ProductionArtifactV18):
            raise RuntimeError(reason or "GOLDEN_DIMENSION_BASELINE_NOT_CONFIGURED")
        if self.subject_segmenter is None:
            raise RuntimeError("SUBJECT_SEGMENTER_NOT_CONFIGURED")
        if self._artifact.board.profile_id is None:
            raise RuntimeError("CALIBRATION_BOARD_PROFILE_NOT_PINNED")

        plane_normalizer = OpenCvCharucoPlaneNormalizer()
        plane = plane_normalizer.normalize_golden_plane(
            image, self._artifact, request.board_candidates,
        )
        logging.getLogger("phone_dino").info(json.dumps({
            "event": "golden_board_detection_completed",
            "requestId": request.request_id,
            "recipeId": request.recipe_id,
            "selectedProfile": plane.calibration_board.profile if plane.calibration_board else None,
            "selectedFiducial": plane.calibration_fiducial,
            "reasonCodes": list(plane.reason_codes),
            "candidates": list(plane.calibration_diagnostics),
        }, separators=(",", ":"), sort_keys=True))
        evidence: PhysicalDimensionEvidence
        background_offset_plane_dimensions: PhysicalDimensionEvidence | None = None
        subject_mask_png_base64: str | None = None
        if plane.reason_codes:
            reason_code = (
                "CHARUCO_CALIBRATION_REQUIRED"
                if "CHARUCO_CORNERS_INSUFFICIENT" in plane.reason_codes
                else f"GOLDEN_CAPTURE_{plane.reason_codes[0]}"
            )
            evidence = _unavailable_physical_dimensions(reason_code)
        elif plane.source_rgb is None:
            evidence = _unavailable_physical_dimensions("GOLDEN_SOURCE_IMAGE_UNAVAILABLE")
        else:
            import numpy as np

            source = np.asarray(plane.source_rgb)
            height, width = source.shape[:2]
            region = request.prompt_region_normalized
            prompt_box = (
                max(0, min(width - 1, int(math.floor(region.x * width)))),
                max(0, min(height - 1, int(math.floor(region.y * height)))),
                max(1, min(width, int(math.ceil((region.x + region.width) * width)))),
                max(1, min(height, int(math.ceil((region.y + region.height) * height)))),
            )
            try:
                prediction = self.subject_segmenter.segment(
                    source,
                    prompt_box,
                    min_foreground_ratio=self._artifact.subject_segmentation.min_foreground_ratio,
                    max_foreground_ratio=self._artifact.subject_segmentation.max_foreground_ratio,
                    min_quality_score=self._artifact.subject_segmentation.min_model_quality_score,
                )
            except RuntimeError as exc:
                evidence = _unavailable_physical_dimensions(
                    f"GOLDEN_SUBJECT_SEGMENTATION_FAILED:{str(exc)[:100]}"
                )
            else:
                evidence = _golden_physical_dimension_evidence(self._artifact, plane, prediction)
                if (
                    evidence.state == "UNAVAILABLE"
                    and evidence.reason_code == "SUBJECT_OUTSIDE_CALIBRATION_PLANE_SUPPORT"
                    and request.offset_plane_calibration is not None
                ):
                    # The formal baseline remains unavailable: the object is
                    # not on the ChArUco plane.  POC may nevertheless retain
                    # a separately labelled board-pose/front-offset estimate.
                    _, _, source_mask_sha = _encode_binary_png(prediction.mask)
                    background_offset_plane_dimensions = (
                        _background_board_offset_plane_physical_dimension_evidence(
                            self._artifact,
                            request,
                            plane,
                            prediction,
                            current_subject_mask_sha256=source_mask_sha,
                        )
                    )
                if evidence.state == "AVAILABLE":
                    _, subject_mask_png_base64, subject_mask_sha = _encode_binary_png(prediction.mask)
                    if subject_mask_sha != evidence.current_subject_mask_sha256:
                        raise RuntimeError("GOLDEN_SUBJECT_MASK_DIGEST_MISMATCH")

        selected_board = plane.calibration_board
        return GoldenDimensionObservation(
            schemaVersion="1.0",
            requestId=request.request_id,
            recipeId=request.recipe_id,
            rawSha256=request.raw_sha256,
            artifactPackageDigest=self.settings.artifact_package_digest,
            analyzerRuntimeVersion=RUNTIME_DIGEST,
            calibrationBoardProfile=(
                selected_board.profile if selected_board is not None else self._artifact.board.profile_id
            ),
            calibrationBoardId=selected_board.board_id if selected_board is not None else None,
            calibrationBoardRevision=selected_board.revision if selected_board is not None else None,
            calibrationBoardManifestSha256=(
                selected_board.manifest_sha256 if selected_board is not None else None
            ),
            measurementPlane=request.measurement_plane,
            physicalDimensions=evidence,
            backgroundOffsetPlaneDimensions=background_offset_plane_dimensions,
            subjectMaskPngBase64=subject_mask_png_base64,
        )

    def analyze(self, request: AnalyzeRequest, image: DecodedImage) -> AnalyzeObservation:
        ready, reason = self.readiness()
        if not ready or self._artifact is None:
            raise RuntimeError(reason or "ANALYZER_NOT_READY")
        verify_artifact_binding(self._artifact, request)
        reference_board_evidence = None
        if isinstance(self._artifact, ProductionArtifactV19):
            # This must happen before either normalisation or embedding. In
            # particular it prevents the target-only fallback from becoming a
            # bypass around an artifact that requires a reference board.
            reference_board_evidence = self.reference_board_verifier.verify(image, self._artifact)
        identity = "|".join((request.session_id, str(request.capture_ordinal), request.raw_sha256, request.execution_bundle_digest, RUNTIME_DIGEST))
        analysis_id = hashlib.sha256(identity.encode()).hexdigest()
        resolved = ResolvedVersions(
            executionBundleDigest=request.execution_bundle_digest,
            artifactPackageDigest=request.artifact_package_digest,
            analyzerModelVersion=self._artifact.analyzer_model_version,
            analyzerRuntimeVersion=RUNTIME_DIGEST,
            normalizationRuntimeVersion=self._artifact.normalization_pipeline_version,
            recipeAnalysisProfileDigest=(
                self._artifact.recipe_analysis_profile.digest
                if isinstance(self._artifact, ProductionArtifactV15) else None
            ),
            scorerInputContractDigest=(
                self._artifact.scorer_input_contract.digest
                if isinstance(self._artifact, ProductionArtifactV15) else None
            ),
        )
        if reference_board_evidence is not None and reference_board_evidence.state != "VERIFIED":
            return AnalyzeObservation(
                schemaVersion=request.schema_version, requestId=request.request_id, analysisId=analysis_id,
                rawSha256=request.raw_sha256, simulation=request.simulation, resolvedVersions=resolved,
                captureAssessment=CaptureAssessment(
                    state=CaptureState.RECAPTURE_REQUIRED,
                    reasonCodes=list(reference_board_evidence.reason_codes),
                ),
                referenceBoardEvidence=reference_board_evidence,
                normalization=None,
                analysis=AnalysisObservation(state=AnalysisState.NOT_RUN),
            )
        normalized = (
            self.normalizer.normalize(image, self._artifact, request.board_candidates)
            if request.board_candidates and isinstance(self.normalizer, OpenCvCharucoNormalizer)
            else self.normalizer.normalize(image, self._artifact)
        )
        if normalized.reason_codes:
            normalization = None
            if normalized.alignment is not None:
                normalization = NormalizationObservation(alignment=normalized.alignment)
            return AnalyzeObservation(
                schemaVersion=request.schema_version, requestId=request.request_id, analysisId=analysis_id,
                rawSha256=request.raw_sha256, simulation=request.simulation, resolvedVersions=resolved,
                captureAssessment=CaptureAssessment(state=CaptureState.RECAPTURE_REQUIRED, reasonCodes=list(normalized.reason_codes)),
                referenceBoardEvidence=reference_board_evidence,
                normalization=normalization,
                analysis=AnalysisObservation(state=AnalysisState.NOT_RUN),
            )
        alignment = normalized.alignment
        if (
            alignment is None or alignment.state != "ALIGNED" or not alignment.target_relative
            or not alignment.transform_within_bounds or not alignment.inspection_mask_applied
        ):
            return AnalyzeObservation(
                schemaVersion=request.schema_version, requestId=request.request_id, analysisId=analysis_id,
                rawSha256=request.raw_sha256, simulation=request.simulation, resolvedVersions=resolved,
                captureAssessment=CaptureAssessment(
                    state=CaptureState.RECAPTURE_REQUIRED, reasonCodes=["TARGET_ALIGNMENT_REQUIRED"],
                ),
                referenceBoardEvidence=reference_board_evidence,
                normalization=None if alignment is None else NormalizationObservation(alignment=alignment),
                analysis=AnalysisObservation(state=AnalysisState.NOT_RUN),
            )
        paired_interior_scoring = isinstance(self._artifact, ProductionArtifactV16)
        use_subject_gate = request.schema_version == "1.1" and isinstance(self._artifact, ProductionArtifactV13)
        roi_only_scoring = isinstance(self._artifact, ProductionArtifactV15)
        subject_evidence: SubjectSegmentationEvidence | None = None
        boundary_evidence: BoundaryDifferenceEvidence | None = None
        scope: SubjectScope | None = None
        golden_rgb: object | None = None
        scorer_input_sha256: str | None = None
        golden_scorer_input_sha256: str | None = None
        scorer_tile_digests: list[ScorerInputTileDigest] | None = None
        analyzed_region: BboxNormalized | None = None
        patches: PatchEmbedding | None = None
        selected_current_patch_cache: dict[tuple[int, int, int], PatchEmbedding] | None = None
        selected_golden_patch_cache: dict[tuple[str, int, int, int], PatchEmbedding] | None = None
        selected_golden_candidate_cache: dict[tuple[str, int, int, int, int], list[float]] | None = None
        current_prediction: SubjectMaskPrediction | None = None
        metric_prediction: SubjectMaskPrediction | None = None
        metric_prediction_reason: str | None = None
        if paired_interior_scoring:
            if self.subject_segmenter is None:
                raise RuntimeError("SUBJECT_SEGMENTER_NOT_CONFIGURED")
            prompt_box = inspection_roi_image(self._artifact.inspection_roi).getbbox()
            if prompt_box is None:
                raise RuntimeError("INSPECTION_ROI_EMPTY")
            try:
                current_prediction = self.subject_segmenter.segment(
                    normalized.rgb,
                    prompt_box,
                    min_foreground_ratio=self._artifact.subject_segmentation.min_foreground_ratio,
                    max_foreground_ratio=self._artifact.subject_segmentation.max_foreground_ratio,
                    min_quality_score=self._artifact.subject_segmentation.min_model_quality_score,
                )
            except RuntimeError as exc:
                return AnalyzeObservation(
                    schemaVersion=request.schema_version,
                    requestId=request.request_id,
                    analysisId=analysis_id,
                    rawSha256=request.raw_sha256,
                    simulation=request.simulation,
                    resolvedVersions=resolved,
                    captureAssessment=CaptureAssessment(
                        state=CaptureState.RECAPTURE_REQUIRED,
                        reasonCodes=[f"CURRENT_SUBJECT_SEGMENTATION_FAILED:{str(exc)[:100]}"],
                    ),
                    referenceBoardEvidence=reference_board_evidence,
                    normalization=NormalizationObservation(alignment=alignment),
                    analysis=AnalysisObservation(state=AnalysisState.NOT_RUN),
                )

        if isinstance(self._artifact, ProductionArtifactV18):
            if self.subject_segmenter is None:
                metric_prediction_reason = "CURRENT_SUBJECT_SEGMENTATION_REQUIRED"
            else:
                try:
                    metric_prediction = _source_metric_prediction(
                        self._artifact, normalized, self.subject_segmenter,
                    )
                except RuntimeError as exc:
                    metric_prediction_reason = f"SOURCE_METRIC_SEGMENTATION_FAILED:{str(exc)[:100]}"

        if paired_interior_scoring:
            if not hasattr(self.embedder, "embed_with_patches") or current_prediction is None:
                raise RuntimeError("PAIRED_INTERIOR_PATCH_EMBEDDER_REQUIRED")
            import cv2
            import numpy as np

            inspection_mask = np.asarray(inspection_roi_image(self._artifact.inspection_roi), dtype=np.uint8)
            paired_candidates: list[PairedScoringCandidate] = []
            rejected_reasons: list[str] = []
            for golden in self._artifact.golden_embeddings:
                paired = _paired_subject_scope(self._artifact, golden, current_prediction)
                if paired.scope is None or paired.boundary_evidence is None:
                    rejected_reasons.append(paired.reason_code or "PAIRED_SUBJECT_SCOPE_UNAVAILABLE")
                    continue
                golden_bytes = base64.b64decode(golden.canonical_image_png_base64, validate=True)
                golden_bgr = cv2.imdecode(np.frombuffer(golden_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if golden_bgr is None:
                    raise RuntimeError("GOLDEN_CANONICAL_IMAGE_INVALID")
                candidate_golden_rgb = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2RGB)
                analysis_mask = cv2.bitwise_and(
                    inspection_mask, np.asarray(paired.scope.support_mask, dtype=np.uint8),
                )
                current_inputs = _scorer_input_tiles(
                    normalized.rgb,
                    self._artifact.inspection_roi.inspection_regions,
                    self._artifact.target_alignment.canonical_width,
                    self._artifact.target_alignment.canonical_height,
                    analysis_mask,
                    self._artifact.scorer_input_contract.neutral_rgb,
                )
                golden_inputs = _scorer_input_tiles(
                    candidate_golden_rgb,
                    self._artifact.inspection_roi.inspection_regions,
                    self._artifact.target_alignment.canonical_width,
                    self._artifact.target_alignment.canonical_height,
                    analysis_mask,
                    self._artifact.scorer_input_contract.neutral_rgb,
                )
                if len(current_inputs) != len(golden_inputs):
                    raise RuntimeError("PAIRED_SCORER_INPUT_TILE_COUNT_MISMATCH")
                current_cache: dict[tuple[int, int, int], PatchEmbedding] = {}
                golden_cache: dict[tuple[str, int, int, int], PatchEmbedding] = {}
                tile_distances: list[float] = []
                current_vectors: list[list[float]] = []
                golden_vectors: list[list[float]] = []
                pair_key = (golden.canonical_sha256 or golden.id) + ":" + paired.scope.evidence.interior_mask_sha256
                for current_tile, golden_tile in zip(current_inputs, golden_inputs, strict=True):
                    if (
                        current_tile.id != golden_tile.id
                        or (current_tile.x, current_tile.y, current_tile.side)
                        != (golden_tile.x, golden_tile.y, golden_tile.side)
                    ):
                        raise RuntimeError("PAIRED_SCORER_INPUT_TILE_IDENTITY_MISMATCH")
                    current_patches = self.embedder.embed_with_patches(current_tile.rgb)  # type: ignore[union-attr]
                    golden_patches = self.embedder.embed_with_patches(golden_tile.rgb)  # type: ignore[union-attr]
                    current_cache[(current_tile.x, current_tile.y, current_tile.side)] = current_patches
                    golden_cache[(pair_key, golden_tile.x, golden_tile.y, golden_tile.side)] = golden_patches
                    current_vectors.append(current_patches.global_vector)
                    golden_vectors.append(golden_patches.global_vector)
                    tile_distances.append(_cosine_distance(
                        current_patches.global_vector, golden_patches.global_vector,
                    ))
                _mean_embedding(current_vectors)
                _mean_embedding(golden_vectors)
                paired_candidates.append(PairedScoringCandidate(
                    id=golden.id,
                    distance=sum(tile_distances) / len(tile_distances),
                    golden=golden,
                    scope=paired.scope,
                    current_inputs=current_inputs,
                    golden_inputs=golden_inputs,
                    current_cache=current_cache,
                    golden_cache=golden_cache,
                    golden_rgb=candidate_golden_rgb,
                    boundary_evidence=paired.boundary_evidence,
                ))
            if not paired_candidates:
                return AnalyzeObservation(
                    schemaVersion=request.schema_version,
                    requestId=request.request_id,
                    analysisId=analysis_id,
                    rawSha256=request.raw_sha256,
                    simulation=request.simulation,
                    resolvedVersions=resolved,
                    captureAssessment=CaptureAssessment(
                        state=CaptureState.RECAPTURE_REQUIRED,
                        reasonCodes=[rejected_reasons[0] if rejected_reasons else "PAIRED_SUBJECT_SCOPE_UNAVAILABLE"],
                    ),
                    referenceBoardEvidence=reference_board_evidence,
                    normalization=NormalizationObservation(alignment=alignment),
                    analysis=AnalysisObservation(state=AnalysisState.NOT_RUN),
                )
            selected = min(paired_candidates, key=lambda item: item.distance)
            nearest_id = selected.id
            nearest = selected.distance
            nearest_golden = selected.golden
            scope = selected.scope
            subject_evidence = scope.evidence
            boundary_evidence = selected.boundary_evidence
            golden_rgb = selected.golden_rgb
            selected_current_patch_cache = selected.current_cache
            selected_golden_patch_cache = selected.golden_cache
            selected_golden_candidate_cache = {}
            distances = [(candidate.id, candidate.distance) for candidate in paired_candidates]
            scorer_input_sha256 = _scorer_input_digest(
                selected.current_inputs, self._artifact.scorer_input_contract.digest,
            )
            golden_scorer_input_sha256 = _scorer_input_digest(
                selected.golden_inputs, self._artifact.scorer_input_contract.digest,
            )
            scorer_tile_digests = [
                ScorerInputTileDigest(
                    id=current.id,
                    bboxNormalized=BboxNormalized(
                        x=current.x / self._artifact.target_alignment.canonical_width,
                        y=current.y / self._artifact.target_alignment.canonical_height,
                        width=current.side / self._artifact.target_alignment.canonical_width,
                        height=current.side / self._artifact.target_alignment.canonical_height,
                    ),
                    currentSha256=current.sha256.removeprefix("sha256:"),
                    referenceSha256=reference.sha256.removeprefix("sha256:"),
                )
                for current, reference in zip(selected.current_inputs, selected.golden_inputs, strict=True)
            ]
            selected_mask = np.asarray(scope.support_mask, dtype=np.uint8)
            ys, xs = np.where(selected_mask > 0)
            if xs.size == 0 or ys.size == 0:
                raise RuntimeError("ANALYZED_REGION_EMPTY")
            left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            analyzed_region = BboxNormalized(
                x=left / self._artifact.target_alignment.canonical_width,
                y=top / self._artifact.target_alignment.canonical_height,
                width=(right - left) / self._artifact.target_alignment.canonical_width,
                height=(bottom - top) / self._artifact.target_alignment.canonical_height,
            )
        elif roi_only_scoring:
            if not hasattr(self.embedder, "embed_with_patches"):
                raise RuntimeError("ROI_ONLY_PATCH_EMBEDDER_REQUIRED")
            import cv2
            import numpy as np

            inspection_mask = np.asarray(inspection_roi_image(self._artifact.inspection_roi), dtype=np.uint8)
            candidates: list[
                tuple[
                    str, float, GoldenEmbedding, SubjectScope, list[ScorerInputTile],
                    dict[tuple[int, int, int], PatchEmbedding],
                ]
            ] = []
            for golden in self._artifact.golden_embeddings:
                golden_scope = _subject_scope(self._artifact, golden)
                analysis_mask = cv2.bitwise_and(inspection_mask, np.asarray(golden_scope.support_mask, dtype=np.uint8))
                current_inputs = _scorer_input_tiles(
                    normalized.rgb,
                    self._artifact.inspection_roi.inspection_regions,
                    self._artifact.target_alignment.canonical_width,
                    self._artifact.target_alignment.canonical_height,
                    analysis_mask,
                    self._artifact.scorer_input_contract.neutral_rgb,
                )
                stored_tiles = golden.scorer_input_tiles or []
                if len(current_inputs) != len(stored_tiles):
                    raise RuntimeError("SCORER_INPUT_TILE_COUNT_MISMATCH")
                current_vectors: list[list[float]] = []
                tile_distances: list[float] = []
                current_patch_cache: dict[tuple[int, int, int], PatchEmbedding] = {}
                for current_tile, reference_tile in zip(current_inputs, stored_tiles, strict=True):
                    if (
                        current_tile.id != reference_tile.id
                        or (current_tile.x, current_tile.y, current_tile.side) != (reference_tile.x, reference_tile.y, reference_tile.side)
                    ):
                        raise RuntimeError("SCORER_INPUT_TILE_IDENTITY_MISMATCH")
                    current_patches = self.embedder.embed_with_patches(current_tile.rgb)  # type: ignore[union-attr]
                    current_patch_cache[(current_tile.x, current_tile.y, current_tile.side)] = current_patches
                    current_vectors.append(current_patches.global_vector)
                    tile_distances.append(_cosine_distance(current_patches.global_vector, reference_tile.values))
                # Build and validate the aggregate as evidence that every tile
                # uses the same embedding dimension; the score itself is the
                # equal-weight mean of per-tile cosine distances.
                _mean_embedding(current_vectors)
                candidates.append((
                    golden.id, sum(tile_distances) / len(tile_distances), golden, golden_scope,
                    current_inputs, current_patch_cache,
                ))
            nearest_id, nearest, nearest_golden, scope, selected_inputs, selected_current_patch_cache = min(
                candidates, key=lambda item: item[1],
            )
            distances = [(candidate_id, distance) for candidate_id, distance, *_ in candidates]
            subject_evidence = scope.evidence
            scorer_input_sha256 = _scorer_input_digest(selected_inputs, self._artifact.scorer_input_contract.digest)
            golden_scorer_input_sha256 = nearest_golden.scorer_input_sha256
            stored_tiles = nearest_golden.scorer_input_tiles or []
            scorer_tile_digests = [
                ScorerInputTileDigest(
                    id=current.id,
                    bboxNormalized=BboxNormalized(
                        x=current.x / self._artifact.target_alignment.canonical_width,
                        y=current.y / self._artifact.target_alignment.canonical_height,
                        width=current.side / self._artifact.target_alignment.canonical_width,
                        height=current.side / self._artifact.target_alignment.canonical_height,
                    ),
                    currentSha256=current.sha256.removeprefix("sha256:"),
                    referenceSha256=reference.tile_sha256.removeprefix("sha256:"),
                )
                for current, reference in zip(selected_inputs, stored_tiles, strict=True)
            ]
            selected_mask = cv2.bitwise_and(inspection_mask, np.asarray(scope.support_mask, dtype=np.uint8))
            ys, xs = np.where(selected_mask > 0)
            if xs.size == 0 or ys.size == 0:
                raise RuntimeError("ANALYZED_REGION_EMPTY")
            left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            analyzed_region = BboxNormalized(
                x=left / self._artifact.target_alignment.canonical_width,
                y=top / self._artifact.target_alignment.canonical_height,
                width=(right - left) / self._artifact.target_alignment.canonical_width,
                height=(bottom - top) / self._artifact.target_alignment.canonical_height,
            )
        elif use_subject_gate:
            # V1.3 readiness deliberately permits one Active Golden only. This
            # prevents an ungated full-frame embedding from selecting a mask
            # based on changing background rather than the inspected subject.
            nearest_golden = self._artifact.golden_embeddings[0]
            golden_bytes = base64.b64decode(nearest_golden.canonical_image_png_base64, validate=True)
            import cv2
            import numpy as np

            golden_bgr = cv2.imdecode(np.frombuffer(golden_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if golden_bgr is None:
                raise RuntimeError("GOLDEN_CANONICAL_IMAGE_INVALID")
            golden_rgb = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2RGB)
            scope = _subject_scope(self._artifact, nearest_golden)
            subject_evidence = scope.evidence
            effective_current = np.asarray(normalized.rgb, dtype=np.uint8).copy()
            effective_golden = np.asarray(golden_rgb, dtype=np.uint8)
            effective_current[np.asarray(scope.support_mask) == 0] = effective_golden[np.asarray(scope.support_mask) == 0]
        else:
            effective_current = normalized.rgb

        if not roi_only_scoring:
            if hasattr(self.embedder, "embed_with_patches"):
                patches = self.embedder.embed_with_patches(effective_current)  # type: ignore[union-attr]
                embedding = patches.global_vector
            else:
                embedding = self.embedder.embed(effective_current)
            distances = [(item.id, _cosine_distance(embedding, item.values)) for item in self._artifact.golden_embeddings]
            nearest_id, nearest = min(distances, key=lambda item: item[1])
            nearest_golden = next(item for item in self._artifact.golden_embeddings if item.id == nearest_id)
        mean = sum(value for _, value in distances) / len(distances)
        global_distance = nearest if roi_only_scoring else mean
        uncertainty = min(1.0, max(0.0, max(value for _, value in distances) - min(value for _, value in distances)))
        if request.schema_version == "1.1" and not isinstance(self._artifact, ProductionArtifactV13):
            subject_evidence = SubjectSegmentationEvidence(
                state="UNAVAILABLE",
                disclaimerCode="SUBJECT_MASK_NOT_DEFECT_PROOF",
                reasonCode="SUBJECT_SEGMENTATION_ARTIFACT_NOT_CONFIGURED",
            )
        if patches is not None or roi_only_scoring:
            inspection_regions = (
                self._artifact.inspection_roi.inspection_regions
                if isinstance(self._artifact, ProductionArtifactV12)
                else self._artifact.target_alignment.inspection_regions
            )
            inspection_mask_override = None
            if isinstance(self._artifact, ProductionArtifactV12):
                import numpy as np
                inspection_mask_override = np.asarray(inspection_roi_image(self._artifact.inspection_roi), dtype=np.uint8)
            if (
                self.settings.engineering_real_model_enabled or use_subject_gate or paired_interior_scoring
            ) and nearest_golden.canonical_image_png_base64 is not None:
                import cv2
                import numpy as np

                if golden_rgb is None:
                    golden_bytes = base64.b64decode(nearest_golden.canonical_image_png_base64, validate=True)
                    golden_bgr = cv2.imdecode(np.frombuffer(golden_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    golden_rgb = None if golden_bgr is None else cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2RGB)
                if golden_rgb is None:
                    spatial_evidence = _unavailable_spatial_evidence("GOLDEN_CANONICAL_IMAGE_INVALID")
                else:
                    cache_key = nearest_golden.canonical_sha256 or nearest_golden.id
                    if scope is not None:
                        scope_digest = (
                            scope.evidence.interior_mask_sha256
                            if scope.paired_interior
                            else scope.evidence.subject_mask_sha256
                        )
                        cache_key += ":" + scope_digest
                    spatial_evidence = _roi_tiled_spatial_difference_evidence(
                        self.embedder, normalized.rgb, golden_rgb, self._artifact.spatial_difference_policy,
                        self._artifact.target_alignment.canonical_width,
                        self._artifact.target_alignment.canonical_height,
                        inspection_regions, golden_cache=(
                            selected_golden_patch_cache
                            if paired_interior_scoring else self._golden_tile_cache
                        ),
                        golden_cache_key=cache_key,
                        current_cache=selected_current_patch_cache,
                        subject_scope=scope,
                        inspection_mask_override=inspection_mask_override,
                        candidate_verification_policy=(
                            self._artifact.candidate_verification_policy
                            if isinstance(self._artifact, ProductionArtifactV14) else None
                        ),
                        golden_candidate_cache=(
                            selected_golden_candidate_cache
                            if paired_interior_scoring else self._golden_candidate_cache
                        ),
                        neutral_rgb=(
                            self._artifact.scorer_input_contract.neutral_rgb
                            if isinstance(self._artifact, ProductionArtifactV15) else None
                        ),
                    )  # type: ignore[arg-type]
            else:
                spatial_evidence = _spatial_difference_evidence(
                    patches, nearest_golden, self._artifact.spatial_difference_policy,
                    self._artifact.target_alignment.canonical_width, self._artifact.target_alignment.canonical_height,
                    inspection_regions,
                    inspection_mask_override,
                )
        else:
            spatial_evidence = _unavailable_spatial_evidence("PATCH_EMBEDDER_NOT_AVAILABLE")
        golden_canonical_sha256 = (
            nearest_golden.canonical_sha256
            or self._artifact.target_alignment.reference_image_sha256
        ).removeprefix("sha256:")
        golden_canonical_base64 = (
            nearest_golden.canonical_image_png_base64
            or self._artifact.target_alignment.reference_image_base64
        )
        dimension_evidence = (
            _background_board_offset_plane_physical_dimension_evidence(
                self._artifact, request, normalized, metric_prediction, self.relative_depth_estimator,
                current_subject_mask_sha256=(
                    _encode_binary_png(current_prediction.mask)[2]
                    if current_prediction is not None else None
                ),
            )
            if isinstance(self._artifact, ProductionArtifactV18)
            and metric_prediction is not None
            and request.offset_plane_calibration is not None
            else _physical_dimension_evidence(self._artifact, normalized, metric_prediction)
            if isinstance(self._artifact, ProductionArtifactV18) and metric_prediction is not None
            else _unavailable_physical_dimensions(metric_prediction_reason or "SOURCE_METRIC_PIPELINE_REQUIRED")
            if isinstance(self._artifact, ProductionArtifactV18)
            else None
        )
        if (
            dimension_evidence is not None
            and dimension_evidence.state == "AVAILABLE"
            and current_prediction is not None
            and metric_prediction is not None
        ):
            # PhoneCV binds physical dimensions to the canonical Current mask
            # used by the paired subject gate. Preserve the separate
            # source-resolution digest that actually supplied the projected
            # contour for an auditable metric-mask binding.
            _, _, canonical_subject_sha = _encode_binary_png(current_prediction.mask)
            source_metric_sha = dimension_evidence.metric_subject_mask_sha256
            dimension_evidence = dimension_evidence.model_copy(update={
                "current_subject_mask_sha256": canonical_subject_sha,
                "metric_subject_mask_sha256": source_metric_sha,
            })
        if (
            request.schema_version in {"1.5", "1.6"}
            and isinstance(self._artifact, ProductionArtifactV18)
            and spatial_evidence.state == "AVAILABLE"
        ):
            _candidate_physical_dimensions(
                self._artifact, request, normalized, current_prediction, spatial_evidence,
            )
        return AnalyzeObservation(
            schemaVersion=request.schema_version, requestId=request.request_id, analysisId=analysis_id,
            rawSha256=request.raw_sha256, simulation=request.simulation, resolvedVersions=resolved,
            captureAssessment=CaptureAssessment(state=CaptureState.ACCEPTED, reasonCodes=[]),
            referenceBoardEvidence=reference_board_evidence,
            normalization=NormalizationObservation(
                canonicalSha256=hashlib.sha256(normalized.encoded).hexdigest(),
                targetCanonicalSha256=hashlib.sha256(normalized.encoded).hexdigest(),
                canonicalImagePngBase64=base64.b64encode(normalized.encoded).decode("ascii"),
                goldenCanonicalSha256=golden_canonical_sha256,
                goldenCanonicalImagePngBase64=golden_canonical_base64,
                canonicalWidth=self._artifact.target_alignment.canonical_width,
                canonicalHeight=self._artifact.target_alignment.canonical_height,
                inspectionRoiContractDigest=(
                    self._artifact.inspection_roi.digest
                    if isinstance(self._artifact, ProductionArtifactV15) else None
                ),
                scorerInputSha256=None if scorer_input_sha256 is None else scorer_input_sha256.removeprefix("sha256:"),
                goldenScorerInputSha256=(
                    None if golden_scorer_input_sha256 is None
                    else golden_scorer_input_sha256.removeprefix("sha256:")
                ),
                scorerInputTileDigests=scorer_tile_digests,
                analyzedRegionNormalized=analyzed_region,
                evidenceCoordinateSpace="TARGET_CANONICAL_IMAGE" if roi_only_scoring else None,
                alignment=normalized.alignment,
            ),
            analysis=AnalysisObservation(
                state=AnalysisState.RUN, metric="cosine_distance", globalDistance=global_distance,
                nearestGoldenId=nearest_id, nearestGoldenDistance=nearest, uncertainty=uncertainty,
                scoringScope="INSPECTION_ROI_ONLY" if roi_only_scoring else None,
                spatialDifferenceEvidence=spatial_evidence,
                subjectSegmentationEvidence=subject_evidence,
                boundaryDifferenceEvidence=boundary_evidence,
                physicalDimensionEvidence=dimension_evidence,
            ),
        )
