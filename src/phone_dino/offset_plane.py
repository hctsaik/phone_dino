"""Metric projection from a background board onto a fixed front measurement plane.

The board may be behind the subject.  A solved board pose and calibrated camera
intrinsics make every image pixel a 3-D ray; intersecting that ray with a known
parallel front plane produces board-frame millimetres without pretending the
subject is on the board itself.

This module deliberately has no learned-depth dependency.  A monocular depth
model can flag an object that does not remain near the declared plane, but it
cannot establish the absolute millimetre scale of this projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


class OffsetPlaneGeometryError(ValueError):
    """Raised when pose or ray geometry cannot support a metric projection."""


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Undistorted camera model in native-image pixel coordinates."""

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    distortion_coefficients: tuple[float, ...] = ()

    def matrix(self) -> np.ndarray:
        values = (self.fx_px, self.fy_px, self.cx_px, self.cy_px)
        if not all(np.isfinite(values)) or self.fx_px <= 0 or self.fy_px <= 0:
            raise OffsetPlaneGeometryError("CAMERA_INTRINSICS_INVALID")
        return np.asarray(
            ((self.fx_px, 0.0, self.cx_px), (0.0, self.fy_px, self.cy_px), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

    def distortion(self) -> np.ndarray:
        values = np.asarray(self.distortion_coefficients, dtype=np.float64)
        if values.size and not np.all(np.isfinite(values)):
            raise OffsetPlaneGeometryError("CAMERA_DISTORTION_INVALID")
        return values.reshape(-1, 1)


@dataclass(frozen=True, slots=True)
class BoardPose:
    """Board-frame to camera-frame transform from PnP.

    Board X is right and Y is down.  Board +Z is away from a front-facing
    camera, so a positive ``front_plane_offset_mm`` maps to board Z = -offset.
    """

    rvec: tuple[float, float, float]
    tvec_mm: tuple[float, float, float]
    reprojection_error_px: float
    inlier_count: int

    def rotation(self) -> np.ndarray:
        import cv2

        vector = np.asarray(self.rvec, dtype=np.float64).reshape(3, 1)
        if not np.all(np.isfinite(vector)):
            raise OffsetPlaneGeometryError("BOARD_POSE_INVALID")
        rotation, _ = cv2.Rodrigues(vector)
        return rotation

    def translation(self) -> np.ndarray:
        vector = np.asarray(self.tvec_mm, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(vector)):
            raise OffsetPlaneGeometryError("BOARD_POSE_INVALID")
        return vector


@dataclass(frozen=True, slots=True)
class OffsetPlaneMeasurement:
    length_mm: float
    width_mm: float
    area_mm2: float
    contour_points_mm: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class PlanarBoardFocalEstimate:
    """Provisional square-pixel intrinsics inferred from one known planar board.

    This is deliberately a fallback, not a replacement for native camera
    calibration.  It assumes zero skew, square pixels, principal point at the
    encoded image centre, and negligible lens distortion.  Callers must expose
    its use as an uncalibrated estimate to the operator.
    """

    intrinsics: CameraIntrinsics
    homography_residual: float


def estimate_square_focal_length_from_planar_board(
    object_points_mm: Iterable[Iterable[float]],
    image_points_px: Iterable[Iterable[float]],
    *,
    image_width: int,
    image_height: int,
) -> PlanarBoardFocalEstimate:
    """Infer focal length from a known planar board under explicit assumptions.

    A plane homography has columns ``K r1``, ``K r2``, and ``K t``.  With
    square pixels and a centred principal point, equality of the first two
    rotation-column norms yields ``f``.  The orthogonality equation is retained
    as a residual check but is not forced: real mobile JPEGs can have a small
    principal-point shift and unmodelled distortion.
    """
    import cv2

    object_points = np.asarray(tuple(tuple(point) for point in object_points_mm), dtype=np.float64)
    image_points = np.asarray(tuple(tuple(point) for point in image_points_px), dtype=np.float64)
    if (object_points.ndim != 2 or object_points.shape[1] != 3 or len(object_points) < 4
            or image_points.shape != (len(object_points), 2)
            or not np.all(np.isfinite(object_points)) or not np.all(np.isfinite(image_points))
            or image_width < 64 or image_height < 64):
        raise OffsetPlaneGeometryError("BOARD_SELF_CALIBRATION_INPUT_INVALID")
    homography, inlier_mask = cv2.findHomography(
        object_points[:, :2], image_points, cv2.RANSAC, 2.5,
    )
    if homography is None or inlier_mask is None or int(np.count_nonzero(inlier_mask)) < 4:
        raise OffsetPlaneGeometryError("BOARD_SELF_CALIBRATION_HOMOGRAPHY_INVALID")
    h1, h2 = homography[:, 0], homography[:, 1]
    cx, cy = float(image_width) / 2.0, float(image_height) / 2.0
    a1 = np.asarray((h1[0] - cx * h1[2], h1[1] - cy * h1[2]), dtype=np.float64)
    a2 = np.asarray((h2[0] - cx * h2[2], h2[1] - cy * h2[2]), dtype=np.float64)
    norm_difference = float(np.dot(a1, a1) - np.dot(a2, a2))
    homogeneous_difference = float(h1[2] ** 2 - h2[2] ** 2)
    if abs(norm_difference) <= 1e-14:
        raise OffsetPlaneGeometryError("BOARD_SELF_CALIBRATION_FOCAL_UNIDENTIFIABLE")
    inverse_focal_squared = -homogeneous_difference / norm_difference
    if not math.isfinite(inverse_focal_squared) or inverse_focal_squared <= 0:
        raise OffsetPlaneGeometryError("BOARD_SELF_CALIBRATION_FOCAL_INVALID")
    focal = 1.0 / math.sqrt(inverse_focal_squared)
    if not math.isfinite(focal) or focal < 100.0 or focal > 100_000.0:
        raise OffsetPlaneGeometryError("BOARD_SELF_CALIBRATION_FOCAL_INVALID")
    q1 = np.asarray((a1[0] / focal, a1[1] / focal, h1[2]), dtype=np.float64)
    q2 = np.asarray((a2[0] / focal, a2[1] / focal, h2[2]), dtype=np.float64)
    scale = max(float(np.linalg.norm(q1) * np.linalg.norm(q2)), 1e-12)
    orthogonality_residual = abs(float(np.dot(q1, q2))) / scale
    if not math.isfinite(orthogonality_residual) or orthogonality_residual > 0.25:
        raise OffsetPlaneGeometryError("BOARD_SELF_CALIBRATION_ASSUMPTION_VIOLATED")
    return PlanarBoardFocalEstimate(
        intrinsics=CameraIntrinsics(focal, focal, cx, cy),
        homography_residual=orthogonality_residual,
    )


def estimate_board_pose(
    object_points_mm: Iterable[Iterable[float]],
    image_points_px: Iterable[Iterable[float]],
    intrinsics: CameraIntrinsics,
    *,
    minimum_inliers: int = 6,
    ransac_reprojection_error_px: float = 2.0,
) -> BoardPose:
    """Solve an immutable board's pose from known 3-D/2-D correspondences."""
    import cv2

    object_points = np.asarray(tuple(tuple(point) for point in object_points_mm), dtype=np.float64)
    image_points = np.asarray(tuple(tuple(point) for point in image_points_px), dtype=np.float64)
    if object_points.ndim != 2 or object_points.shape[1] != 3 or len(object_points) < 4:
        raise OffsetPlaneGeometryError("BOARD_OBJECT_POINTS_INVALID")
    if image_points.shape != (len(object_points), 2) or not np.all(np.isfinite(image_points)):
        raise OffsetPlaneGeometryError("BOARD_IMAGE_POINTS_INVALID")
    if not np.all(np.isfinite(object_points)):
        raise OffsetPlaneGeometryError("BOARD_OBJECT_POINTS_INVALID")
    if minimum_inliers < 4 or ransac_reprojection_error_px <= 0:
        raise OffsetPlaneGeometryError("PNP_POLICY_INVALID")

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        intrinsics.matrix(),
        intrinsics.distortion(),
        iterationsCount=200,
        reprojectionError=float(ransac_reprojection_error_px),
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    inlier_indices = np.asarray(inliers if inliers is not None else (), dtype=np.int32).reshape(-1)
    if not ok or rvec is None or tvec is None or len(inlier_indices) < minimum_inliers:
        raise OffsetPlaneGeometryError("BOARD_PNP_INSUFFICIENT_INLIERS")
    projected, _ = cv2.projectPoints(
        object_points[inlier_indices], rvec, tvec, intrinsics.matrix(), intrinsics.distortion(),
    )
    residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points[inlier_indices], axis=1)
    return BoardPose(
        rvec=tuple(float(value) for value in rvec.reshape(3)),
        tvec_mm=tuple(float(value) for value in tvec.reshape(3)),
        reprojection_error_px=float(np.percentile(residuals, 95)),
        inlier_count=int(len(inlier_indices)),
    )


def intersect_pixels_with_front_offset_plane(
    image_points_px: Iterable[Iterable[float]],
    intrinsics: CameraIntrinsics,
    pose: BoardPose,
    *,
    front_plane_offset_mm: float,
) -> np.ndarray:
    """Return board-frame X/Y intersections on the declared front plane.

    ``front_plane_offset_mm`` is the signed physical distance from the printed
    board toward the camera to the measurement plane.  It is a fixture datum,
    not an inferred per-photo parameter.  Zero means the board plane itself.
    """
    import cv2

    pixels = np.asarray(tuple(tuple(point) for point in image_points_px), dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 1 or not np.all(np.isfinite(pixels)):
        raise OffsetPlaneGeometryError("MEASUREMENT_IMAGE_POINTS_INVALID")
    if not np.isfinite(front_plane_offset_mm) or front_plane_offset_mm < 0:
        raise OffsetPlaneGeometryError("FRONT_PLANE_OFFSET_INVALID")
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), intrinsics.matrix(), intrinsics.distortion(),
    ).reshape(-1, 2)
    rays_camera = np.column_stack((normalized, np.ones(len(normalized), dtype=np.float64)))
    rotation = pose.rotation()
    translation = pose.translation()
    camera_center_board = -rotation.T @ translation
    rays_board = (rotation.T @ rays_camera.T).T
    plane_z = -float(front_plane_offset_mm)
    denominators = rays_board[:, 2]
    if np.any(np.abs(denominators) <= 1e-12):
        raise OffsetPlaneGeometryError("MEASUREMENT_RAY_PARALLEL_TO_PLANE")
    distances = (plane_z - camera_center_board[2]) / denominators
    if np.any(~np.isfinite(distances)) or np.any(distances <= 0):
        raise OffsetPlaneGeometryError("MEASUREMENT_PLANE_BEHIND_CAMERA")
    points_board = camera_center_board + distances[:, None] * rays_board
    if not np.all(np.isfinite(points_board)):
        raise OffsetPlaneGeometryError("MEASUREMENT_PLANE_INTERSECTION_INVALID")
    return points_board[:, :2]


def board_plane_camera_depth_at_pixels(
    image_points_px: Iterable[Iterable[float]],
    intrinsics: CameraIntrinsics,
    pose: BoardPose,
) -> np.ndarray:
    """Return the camera-Z depth where each pixel ray meets the board plane.

    This is deliberately exposed separately from the X/Y metric projection.
    A relative-depth network can be calibrated *per capture* against this
    metric depth field; it never needs to invent a global millimetre scale.
    """
    import cv2

    pixels = np.asarray(tuple(tuple(point) for point in image_points_px), dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 1 or not np.all(np.isfinite(pixels)):
        raise OffsetPlaneGeometryError("MEASUREMENT_IMAGE_POINTS_INVALID")
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), intrinsics.matrix(), intrinsics.distortion(),
    ).reshape(-1, 2)
    rays_camera = np.column_stack((normalized, np.ones(len(normalized), dtype=np.float64)))
    normal_camera = pose.rotation()[:, 2]
    plane_distance = float(normal_camera @ pose.translation())
    denominators = rays_camera @ normal_camera
    if np.any(np.abs(denominators) <= 1e-12):
        raise OffsetPlaneGeometryError("MEASUREMENT_RAY_PARALLEL_TO_PLANE")
    distances = plane_distance / denominators
    depths = distances * rays_camera[:, 2]
    if np.any(~np.isfinite(depths)) or np.any(depths <= 0):
        raise OffsetPlaneGeometryError("MEASUREMENT_PLANE_BEHIND_CAMERA")
    return depths


def front_offset_from_camera_depth(
    image_points_px: Iterable[Iterable[float]],
    camera_depth_mm: Iterable[float],
    intrinsics: CameraIntrinsics,
    pose: BoardPose,
) -> np.ndarray:
    """Convert per-pixel camera-Z depth to positive board-front offset mm.

    A positive return value is closer to the camera than the board.  The
    function supports a tilted board and does not assume a fronto-parallel
    phone pose.
    """
    import cv2

    pixels = np.asarray(tuple(tuple(point) for point in image_points_px), dtype=np.float64)
    depths = np.asarray(tuple(camera_depth_mm), dtype=np.float64).reshape(-1)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 1
            or len(depths) != len(pixels) or not np.all(np.isfinite(pixels))
            or not np.all(np.isfinite(depths)) or np.any(depths <= 0)):
        raise OffsetPlaneGeometryError("MEASUREMENT_CAMERA_DEPTH_INVALID")
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), intrinsics.matrix(), intrinsics.distortion(),
    ).reshape(-1, 2)
    rays_camera = np.column_stack((normalized, np.ones(len(normalized), dtype=np.float64)))
    normal_camera = pose.rotation()[:, 2]
    plane_distance = float(normal_camera @ pose.translation())
    ray_distance = depths / rays_camera[:, 2]
    offsets = plane_distance - ray_distance * (rays_camera @ normal_camera)
    if not np.all(np.isfinite(offsets)):
        raise OffsetPlaneGeometryError("MEASUREMENT_CAMERA_DEPTH_INVALID")
    return offsets


def measure_contour_on_front_offset_plane(
    contour_px: Iterable[Iterable[float]],
    intrinsics: CameraIntrinsics,
    pose: BoardPose,
    *,
    front_plane_offset_mm: float,
) -> OffsetPlaneMeasurement:
    """Project a source-resolution contour and report its minimum-area box."""
    import cv2

    contour_mm = intersect_pixels_with_front_offset_plane(
        contour_px, intrinsics, pose, front_plane_offset_mm=front_plane_offset_mm,
    )
    if len(contour_mm) < 3:
        raise OffsetPlaneGeometryError("MEASUREMENT_CONTOUR_INVALID")
    rectangle = cv2.minAreaRect(contour_mm.astype(np.float32).reshape(-1, 1, 2))
    first, second = (float(value) for value in rectangle[1])
    if first <= 0 or second <= 0:
        raise OffsetPlaneGeometryError("MEASUREMENT_CONTOUR_DEGENERATE")
    return OffsetPlaneMeasurement(
        length_mm=max(first, second),
        width_mm=min(first, second),
        area_mm2=abs(float(cv2.contourArea(contour_mm.astype(np.float32).reshape(-1, 1, 2)))),
        contour_points_mm=tuple((float(x), float(y)) for x, y in contour_mm),
    )
