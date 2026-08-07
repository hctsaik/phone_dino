"""Convert a relative-depth map into a conservative board-relative offset.

Monocular models such as Depth Anything output relative depth, not a ruler.
Here the printed background board supplies the missing per-capture scale: PnP
gives the true camera-Z depth for visible board pixels, and we robustly fit an
affine inverse-depth mapping there.  The fitted mapping is then used only for
the segmented subject.  Its residuals and the subject's own depth spread are
returned as a 95% interval rather than hidden behind a single guessed number.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .offset_plane import (
    BoardPose,
    CameraIntrinsics,
    OffsetPlaneGeometryError,
    board_plane_camera_depth_at_pixels,
    front_offset_from_camera_depth,
)


@dataclass(frozen=True, slots=True)
class RelativeDepthOffsetPolicy:
    min_board_pixels: int = 256
    min_subject_pixels: int = 128
    minimum_depth_variation: float = 1e-6
    maximum_front_offset_mm: float = 1_000.0
    model_systematic_error_mm: float = 8.0
    # The legacy path treats all segmented foreground pixels as one depth
    # population.  Enable this only for a rig/profile that declares a named
    # board-parallel front measurement plane.
    dominant_plane_enabled: bool = False
    dominant_plane_half_width_mm: float = 8.0
    minimum_dominant_plane_pixels: int = 256
    minimum_dominant_plane_support_ratio: float = 0.35


@dataclass(frozen=True, slots=True)
class RelativeDepthOffsetEstimate:
    state: str
    reason_code: str | None = None
    offset_mm: float | None = None
    lower95_mm: float | None = None
    upper95_mm: float | None = None
    board_fit_p95_mm: float | None = None
    subject_depth_spread_p95_mm: float | None = None
    board_pixel_count: int = 0
    subject_pixel_count: int = 0


def _unavailable(reason: str, board_count: int = 0, subject_count: int = 0) -> RelativeDepthOffsetEstimate:
    return RelativeDepthOffsetEstimate(
        state="UNAVAILABLE", reason_code=reason,
        board_pixel_count=board_count, subject_pixel_count=subject_count,
    )


def relative_depth_posterior_improves_prior(
    posterior: RelativeDepthOffsetEstimate,
    prior_lower_mm: float,
    prior_upper_mm: float,
) -> bool:
    """Select an unqualified learned posterior only when it improves the prior.

    The board/rig interval is the existing physical scenario envelope.  A
    relative-depth posterior may refine it only when it is wholly contained by
    that envelope and is strictly narrower.  This prevents a high-spread 3-D
    subject from appearing less certain merely because a learned map ran.
    """
    required = (posterior.offset_mm, posterior.lower95_mm, posterior.upper95_mm)
    if (
        posterior.state != "AVAILABLE"
        or any(value is None or not math.isfinite(value) for value in required)
        or not math.isfinite(prior_lower_mm)
        or not math.isfinite(prior_upper_mm)
        or prior_lower_mm > prior_upper_mm
    ):
        return False
    assert posterior.lower95_mm is not None and posterior.upper95_mm is not None
    posterior_width = posterior.upper95_mm - posterior.lower95_mm
    prior_width = prior_upper_mm - prior_lower_mm
    tolerance = 1e-9
    return (
        prior_width > tolerance
        and posterior_width + tolerance < prior_width
        and posterior.lower95_mm + tolerance >= prior_lower_mm
        and posterior.upper95_mm <= prior_upper_mm + tolerance
    )


def _robust_affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Fit y = a*x+b after a few MAD outlier rejection passes."""
    keep = np.ones(len(x), dtype=bool)
    for _ in range(4):
        if int(np.count_nonzero(keep)) < 16:
            return None
        design = np.column_stack((x[keep], np.ones(int(np.count_nonzero(keep)), dtype=np.float64)))
        if int(np.linalg.matrix_rank(design)) < 2:
            return None
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        if not math.isfinite(mad):
            return None
        if mad <= 1e-12:
            break
        next_keep = np.abs(residual - center) <= max(3.5 * 1.4826 * mad, 1e-10)
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    if not math.isfinite(float(slope)) or not math.isfinite(float(intercept)) or slope <= 0:
        return None
    return float(slope), float(intercept)


def _dominant_board_parallel_plane_offsets(
    offsets: np.ndarray,
    policy: RelativeDepthOffsetPolicy,
) -> np.ndarray | None:
    """Return a supported board-parallel foreground layer, if one exists.

    A single relative-depth map cannot safely recover arbitrary tilted surface
    geometry.  The background-board metric projection already measures a
    board-parallel front plane, so select only a depth-consensus layer that is
    compatible with that contract.  Selection is deterministic: the densest
    sliding depth window wins; ties prefer the nearer layer.  Its support must
    be substantial enough that a small edge, handle or shadow cannot become
    the reported measurement plane.
    """
    half_width = float(policy.dominant_plane_half_width_mm)
    support_ratio = float(policy.minimum_dominant_plane_support_ratio)
    if (
        not math.isfinite(half_width) or half_width <= 0.0
        or not math.isfinite(support_ratio) or not 0.0 < support_ratio <= 1.0
        or policy.minimum_dominant_plane_pixels < 1
    ):
        return None
    values = np.sort(np.asarray(offsets, dtype=np.float64))
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        return None
    window_end = np.searchsorted(values, values + 2.0 * half_width, side="right")
    counts = window_end - np.arange(len(values))
    greatest = int(np.max(counts))
    if greatest <= 0:
        return None
    minimum_support = max(
        int(policy.min_subject_pixels),
        int(policy.minimum_dominant_plane_pixels),
        int(math.ceil(len(values) * support_ratio)),
    )
    best: np.ndarray | None = None
    best_center = -math.inf
    # Different starts can describe the same depth layer.  Re-centre every
    # winning window on its robust median before applying the final inlier
    # band, so the resulting uncertainty uses only its local plane pixels.
    for start in np.flatnonzero(counts == greatest):
        candidate = values[int(start):int(window_end[int(start)])]
        center = float(np.median(candidate))
        inliers = offsets[np.abs(offsets - center) <= half_width]
        if len(inliers) < minimum_support:
            continue
        local_center = float(np.median(inliers))
        # Higher board-normal offset is nearer the camera and is the intended
        # front layer when equally supported surfaces are present.
        if best is None or len(inliers) > len(best) or (
            len(inliers) == len(best) and local_center > best_center
        ):
            best = inliers
            best_center = local_center
    return best


def estimate_front_offset_from_relative_inverse_depth(
    inverse_depth: object,
    board_mask: object,
    subject_mask: object,
    intrinsics: CameraIntrinsics,
    pose: BoardPose,
    policy: RelativeDepthOffsetPolicy = RelativeDepthOffsetPolicy(),
) -> RelativeDepthOffsetEstimate:
    """Estimate the subject's board-normal front offset and conservative 95% band.

    ``inverse_depth`` must have the same native resolution as the photo, with
    larger values representing nearer points.  Its arbitrary scale and shift
    are calibrated from the visible board in this particular capture.
    """
    depth = np.asarray(inverse_depth, dtype=np.float64)
    board = np.asarray(board_mask, dtype=bool)
    subject = np.asarray(subject_mask, dtype=bool)
    if depth.ndim != 2 or board.shape != depth.shape or subject.shape != depth.shape:
        return _unavailable("RELATIVE_DEPTH_MASK_DIMENSIONS_INVALID")
    valid = np.isfinite(depth)
    board &= valid
    subject &= valid
    board_count = int(np.count_nonzero(board))
    subject_count = int(np.count_nonzero(subject))
    if board_count < policy.min_board_pixels:
        return _unavailable("RELATIVE_DEPTH_BOARD_SUPPORT_INSUFFICIENT", board_count, subject_count)
    if subject_count < policy.min_subject_pixels:
        return _unavailable("RELATIVE_DEPTH_SUBJECT_SUPPORT_INSUFFICIENT", board_count, subject_count)
    board_values = depth[board]
    if float(np.percentile(board_values, 95) - np.percentile(board_values, 5)) < policy.minimum_depth_variation:
        return _unavailable("RELATIVE_DEPTH_BOARD_DYNAMIC_RANGE_INSUFFICIENT", board_count, subject_count)

    rows, columns = np.indices(depth.shape, dtype=np.float64)
    board_pixels = np.column_stack((columns[board], rows[board]))
    subject_pixels = np.column_stack((columns[subject], rows[subject]))
    try:
        board_z = board_plane_camera_depth_at_pixels(board_pixels, intrinsics, pose)
    except OffsetPlaneGeometryError as exc:
        return _unavailable(str(exc), board_count, subject_count)
    fit = _robust_affine_fit(board_values, 1.0 / board_z)
    if fit is None:
        return _unavailable("RELATIVE_DEPTH_BOARD_AFFINE_FIT_INVALID", board_count, subject_count)
    slope, intercept = fit
    fitted_inverse_z = slope * board_values + intercept
    if np.any(fitted_inverse_z <= 1e-12):
        return _unavailable("RELATIVE_DEPTH_BOARD_AFFINE_FIT_INVALID", board_count, subject_count)
    board_fit_error = np.abs(1.0 / fitted_inverse_z - board_z)
    board_fit_p95 = float(np.percentile(board_fit_error, 95))

    subject_inverse_z = slope * depth[subject] + intercept
    if np.any(~np.isfinite(subject_inverse_z)) or np.any(subject_inverse_z <= 1e-12):
        return _unavailable("RELATIVE_DEPTH_SUBJECT_AFFINE_EXTRAPOLATION", board_count, subject_count)
    try:
        offsets = front_offset_from_camera_depth(
            subject_pixels, 1.0 / subject_inverse_z, intrinsics, pose,
        )
    except OffsetPlaneGeometryError as exc:
        return _unavailable(str(exc), board_count, subject_count)
    offsets = offsets[np.isfinite(offsets)]
    if len(offsets) < policy.min_subject_pixels:
        return _unavailable("RELATIVE_DEPTH_SUBJECT_OFFSET_INVALID", board_count, subject_count)
    if policy.dominant_plane_enabled:
        dominant_offsets = _dominant_board_parallel_plane_offsets(offsets, policy)
        if dominant_offsets is None:
            return _unavailable("RELATIVE_DEPTH_DOMINANT_PLANE_SUPPORT_INSUFFICIENT", board_count, subject_count)
        offsets = dominant_offsets
    offset = float(np.median(offsets))
    spread = float(np.percentile(np.abs(offsets - offset), 95))
    systematic = max(0.0, float(policy.model_systematic_error_mm))
    half_width = max(board_fit_p95 + spread + systematic, systematic)
    lower = max(0.0, offset - half_width)
    upper = min(float(policy.maximum_front_offset_mm), offset + half_width)
    if not all(math.isfinite(value) for value in (offset, lower, upper, board_fit_p95, spread)) or upper < lower:
        return _unavailable("RELATIVE_DEPTH_POSTERIOR_INVALID", board_count, subject_count)
    return RelativeDepthOffsetEstimate(
        state="AVAILABLE", offset_mm=max(0.0, min(offset, float(policy.maximum_front_offset_mm))),
        lower95_mm=lower, upper95_mm=upper,
        board_fit_p95_mm=board_fit_p95, subject_depth_spread_p95_mm=spread,
        board_pixel_count=board_count, subject_pixel_count=subject_count,
    )
