from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class DepthShadowPolicy:
    min_board_pixels: int = 64
    min_subject_pixels: int = 64
    max_board_residual_ratio: float = 0.05
    max_subject_median_residual_ratio: float = 0.08
    max_subject_p95_residual_ratio: float = 0.15


@dataclass(frozen=True, slots=True)
class RelativeDepthCoplanarityEvidence:
    state: str
    reason_code: str | None = None
    observation: str | None = None
    board_residual_p95_ratio: float | None = None
    subject_median_residual_ratio: float | None = None
    subject_p95_residual_ratio: float | None = None
    board_pixel_count: int = 0
    subject_pixel_count: int = 0
    disclaimer_code: str = "DEPTH_SHADOW_NOT_METRIC_PROOF"


def assess_relative_depth_coplanarity(
    inverse_depth: object,
    board_mask: object,
    subject_mask: object,
    policy: DepthShadowPolicy = DepthShadowPolicy(),
) -> RelativeDepthCoplanarityEvidence:
    """Assess plane residuals without treating monocular depth as millimetres.

    A least-squares inverse-depth plane is fitted only on the detected board.
    Residuals are normalized by the frame's robust depth span, making the
    observation invariant to the scale and shift ambiguity of relative depth.
    The result is shadow evidence and must never alter a dimensional result.
    """
    import numpy as np

    depth = np.asarray(inverse_depth, dtype=np.float64)
    board = np.asarray(board_mask, dtype=bool)
    subject = np.asarray(subject_mask, dtype=bool)
    if depth.ndim != 2 or board.shape != depth.shape or subject.shape != depth.shape:
        return RelativeDepthCoplanarityEvidence(state="UNAVAILABLE", reason_code="DEPTH_MASK_DIMENSIONS_INVALID")
    valid = np.isfinite(depth)
    board &= valid
    subject &= valid
    board_count = int(np.count_nonzero(board))
    subject_count = int(np.count_nonzero(subject))
    if board_count < policy.min_board_pixels:
        return RelativeDepthCoplanarityEvidence(
            state="UNAVAILABLE", reason_code="DEPTH_BOARD_SUPPORT_INSUFFICIENT",
            board_pixel_count=board_count, subject_pixel_count=subject_count,
        )
    if subject_count < policy.min_subject_pixels:
        return RelativeDepthCoplanarityEvidence(
            state="UNAVAILABLE", reason_code="DEPTH_SUBJECT_SUPPORT_INSUFFICIENT",
            board_pixel_count=board_count, subject_pixel_count=subject_count,
        )
    values = depth[valid]
    robust_span = float(np.percentile(values, 95) - np.percentile(values, 5))
    if not math.isfinite(robust_span) or robust_span <= 1e-9:
        return RelativeDepthCoplanarityEvidence(
            state="UNAVAILABLE", reason_code="DEPTH_DYNAMIC_RANGE_INSUFFICIENT",
            board_pixel_count=board_count, subject_pixel_count=subject_count,
        )

    rows, columns = np.indices(depth.shape, dtype=np.float64)
    design = np.column_stack((columns[board], rows[board], np.ones(board_count, dtype=np.float64)))
    if int(np.linalg.matrix_rank(design)) < 3:
        return RelativeDepthCoplanarityEvidence(
            state="UNAVAILABLE", reason_code="DEPTH_BOARD_GEOMETRY_DEGENERATE",
            board_pixel_count=board_count, subject_pixel_count=subject_count,
        )
    coefficients, *_ = np.linalg.lstsq(design, depth[board], rcond=None)
    predicted = coefficients[0] * columns + coefficients[1] * rows + coefficients[2]
    board_residual = np.abs(depth[board] - predicted[board]) / robust_span
    subject_residual = np.abs(depth[subject] - predicted[subject]) / robust_span
    board_p95 = float(np.percentile(board_residual, 95))
    subject_median = float(np.median(subject_residual))
    subject_p95 = float(np.percentile(subject_residual, 95))
    metrics = (board_p95, subject_median, subject_p95)
    if not all(math.isfinite(value) and value >= 0 for value in metrics):
        return RelativeDepthCoplanarityEvidence(
            state="UNAVAILABLE", reason_code="DEPTH_RESIDUAL_INVALID",
            board_pixel_count=board_count, subject_pixel_count=subject_count,
        )
    observation = (
        "COPLANAR_CANDIDATE"
        if board_p95 <= policy.max_board_residual_ratio
        and subject_median <= policy.max_subject_median_residual_ratio
        and subject_p95 <= policy.max_subject_p95_residual_ratio
        else "NON_COPLANAR_RISK"
    )
    return RelativeDepthCoplanarityEvidence(
        state="AVAILABLE",
        observation=observation,
        board_residual_p95_ratio=board_p95,
        subject_median_residual_ratio=subject_median,
        subject_p95_residual_ratio=subject_p95,
        board_pixel_count=board_count,
        subject_pixel_count=subject_count,
    )
