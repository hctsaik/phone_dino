from __future__ import annotations

import numpy as np
import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from phone_dino.offset_plane import (
    BoardPose,
    CameraIntrinsics,
    OffsetPlaneGeometryError,
    estimate_board_pose,
    estimate_square_focal_length_from_planar_board,
    measure_contour_on_front_offset_plane,
)


def _project(points, rvec, tvec, intrinsics):
    projected, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64),
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        intrinsics.matrix(),
        intrinsics.distortion(),
    )
    return projected.reshape(-1, 2)


def test_pnp_offset_plane_recovers_a_background_board_subject_footprint():
    intrinsics = CameraIntrinsics(1180.0, 1160.0, 640.0, 480.0, (-0.08, 0.015, 0.0003, -0.0002, 0.0))
    rvec = (0.10, -0.16, 0.025)
    tvec = (-92.0, -76.0, 760.0)
    board_points = np.asarray(((0, 0, 0), (200, 0, 0), (200, 230, 0), (0, 230, 0), (30, 25, 0), (170, 205, 0)), dtype=np.float64)
    pose = estimate_board_pose(board_points, _project(board_points, rvec, tvec, intrinsics), intrinsics)
    assert pose.inlier_count == len(board_points)
    assert pose.reprojection_error_px < 0.001

    front_offset = 34.0
    subject = np.asarray(((9, 21, -front_offset), (192, 21, -front_offset), (192, 80, -front_offset), (9, 80, -front_offset)), dtype=np.float64)
    measurement = measure_contour_on_front_offset_plane(
        _project(subject, rvec, tvec, intrinsics), intrinsics, pose, front_plane_offset_mm=front_offset,
    )
    assert measurement.length_mm == pytest.approx(183.0, abs=0.02)
    assert measurement.width_mm == pytest.approx(59.0, abs=0.02)
    assert measurement.area_mm2 == pytest.approx(10797.0, abs=1.0)


def test_using_the_board_plane_for_a_front_subject_remains_visibly_wrong():
    intrinsics = CameraIntrinsics(1200.0, 1200.0, 640.0, 480.0)
    pose = BoardPose(rvec=(0.0, 0.0, 0.0), tvec_mm=(0.0, 0.0, 700.0), reprojection_error_px=0.1, inlier_count=12)
    subject = np.asarray(((0, 0, -35), (183, 0, -35), (183, 59, -35), (0, 59, -35)), dtype=np.float64)
    pixels = _project(subject, pose.rvec, pose.tvec_mm, intrinsics)
    corrected = measure_contour_on_front_offset_plane(pixels, intrinsics, pose, front_plane_offset_mm=35)
    board_plane = measure_contour_on_front_offset_plane(pixels, intrinsics, pose, front_plane_offset_mm=0)
    assert corrected.length_mm == pytest.approx(183.0, abs=0.01)
    assert board_plane.length_mm > 192.0


def test_planar_board_self_calibration_recovers_square_focal_length_under_its_assumptions():
    intrinsics = CameraIntrinsics(1500.0, 1500.0, 640.0, 480.0)
    board_points = np.asarray(
        ((0, 0, 0), (200, 0, 0), (200, 230, 0), (0, 230, 0), (30, 25, 0), (170, 205, 0)),
        dtype=np.float64,
    )
    image_points = _project(board_points, (0.12, -0.18, 0.04), (-92.0, -76.0, 760.0), intrinsics)
    estimate = estimate_square_focal_length_from_planar_board(
        board_points, image_points, image_width=1280, image_height=960,
    )
    assert estimate.intrinsics.fx_px == pytest.approx(1500.0, abs=0.1)
    assert estimate.intrinsics.fy_px == pytest.approx(1500.0, abs=0.1)
    assert estimate.homography_residual < 1e-6


def test_offset_plane_rejects_an_unknown_negative_board_to_subject_distance():
    intrinsics = CameraIntrinsics(1200.0, 1200.0, 640.0, 480.0)
    pose = BoardPose(rvec=(0.0, 0.0, 0.0), tvec_mm=(0.0, 0.0, 700.0), reprojection_error_px=0.1, inlier_count=12)
    with pytest.raises(OffsetPlaneGeometryError, match="FRONT_PLANE_OFFSET_INVALID"):
        measure_contour_on_front_offset_plane(((640.0, 480.0), (650.0, 480.0), (650.0, 490.0)), intrinsics, pose, front_plane_offset_mm=-1.0)
