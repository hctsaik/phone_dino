import numpy as np
import pytest

from phone_dino.depth_offset import (
    RelativeDepthOffsetEstimate,
    RelativeDepthOffsetPolicy,
    estimate_front_offset_from_relative_inverse_depth,
    relative_depth_posterior_improves_prior,
)
from phone_dino.offset_plane import (
    BoardPose,
    CameraIntrinsics,
    board_plane_camera_depth_at_pixels,
    front_offset_from_camera_depth,
)


def _scene(offset_mm: float = 34.0):
    height, width = 420, 640
    intrinsics = CameraIntrinsics(900.0, 896.0, width / 2, height / 2)
    pose = BoardPose((0.11, -0.08, 0.02), (8.0, -5.0, 620.0), 0.2, 24)
    rows, columns = np.indices((height, width), dtype=np.float64)
    pixels = np.column_stack((columns.reshape(-1), rows.reshape(-1)))
    board_z = board_plane_camera_depth_at_pixels(pixels, intrinsics, pose).reshape(height, width)
    board = np.zeros((height, width), dtype=bool)
    board[30:380, 35:605] = True
    subject = np.zeros((height, width), dtype=bool)
    subject[160:275, 240:465] = True
    board &= ~subject
    subject_pixels = np.column_stack((columns[subject], rows[subject]))
    subject_z = np.full(int(np.count_nonzero(subject)), 1.0, dtype=np.float64)
    # Work backwards from a parallel plane with the chosen positive offset.
    board_normal = pose.rotation()[:, 2]
    plane_distance = float(board_normal @ pose.translation()) - offset_mm
    normalized_x = (subject_pixels[:, 0] - intrinsics.cx_px) / intrinsics.fx_px
    normalized_y = (subject_pixels[:, 1] - intrinsics.cy_px) / intrinsics.fy_px
    rays = np.column_stack((normalized_x, normalized_y, np.ones(len(subject_pixels))))
    subject_z[:] = plane_distance / (rays @ board_normal)
    depth_z = board_z.copy()
    depth_z[subject] = subject_z
    # Relative network output has arbitrary affine inverse-depth coordinates.
    relative_inverse_depth = (1.0 / depth_z - 0.0002) / 0.002
    return relative_inverse_depth, board, subject, intrinsics, pose


def _set_parallel_offset(
    relative_inverse_depth: np.ndarray,
    selector: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: BoardPose,
    offset_mm: float,
) -> np.ndarray:
    """Render one board-parallel depth layer into the synthetic map."""
    result = relative_inverse_depth.copy()
    rows, columns = np.nonzero(selector)
    pixels = np.column_stack((columns, rows)).astype(np.float64)
    normal = pose.rotation()[:, 2]
    plane_distance = float(normal @ pose.translation()) - offset_mm
    rays = np.column_stack((
        (pixels[:, 0] - intrinsics.cx_px) / intrinsics.fx_px,
        (pixels[:, 1] - intrinsics.cy_px) / intrinsics.fy_px,
        np.ones(len(pixels)),
    ))
    z = plane_distance / (rays @ normal)
    result[selector] = (1.0 / z - 0.0002) / 0.002
    return result


def test_relative_depth_is_calibrated_to_board_pose_and_recovers_front_offset():
    depth, board, subject, intrinsics, pose = _scene()

    evidence = estimate_front_offset_from_relative_inverse_depth(
        depth, board, subject, intrinsics, pose,
        RelativeDepthOffsetPolicy(model_systematic_error_mm=2.0),
    )

    assert evidence.state == "AVAILABLE"
    assert evidence.offset_mm == pytest.approx(34.0, abs=0.05)
    assert evidence.lower95_mm is not None and evidence.lower95_mm <= 34.0
    assert evidence.upper95_mm is not None and evidence.upper95_mm >= 34.0


def test_relative_depth_returns_an_interval_wider_than_its_board_fit_error():
    depth, board, subject, intrinsics, pose = _scene()
    depth = depth.copy()
    depth[subject] += np.linspace(-0.02, 0.02, int(np.count_nonzero(subject)))

    evidence = estimate_front_offset_from_relative_inverse_depth(
        depth, board, subject, intrinsics, pose,
        RelativeDepthOffsetPolicy(model_systematic_error_mm=5.0),
    )

    assert evidence.state == "AVAILABLE"
    assert evidence.lower95_mm is not None and evidence.upper95_mm is not None
    assert evidence.upper95_mm - evidence.lower95_mm >= 10.0
    assert evidence.subject_depth_spread_p95_mm is not None
    assert evidence.subject_depth_spread_p95_mm > 0.0


def test_dominant_plane_uses_only_the_supported_front_layer_for_its_interval():
    depth, board, subject, intrinsics, pose = _scene()
    rows, columns = np.indices(depth.shape)
    front = subject & (columns < 375)
    rear = subject & ~front
    depth = _set_parallel_offset(depth, front, intrinsics, pose, 34.0)
    depth = _set_parallel_offset(depth, rear, intrinsics, pose, 74.0)

    full_subject = estimate_front_offset_from_relative_inverse_depth(
        depth, board, subject, intrinsics, pose,
        RelativeDepthOffsetPolicy(model_systematic_error_mm=2.0),
    )
    selected_plane = estimate_front_offset_from_relative_inverse_depth(
        depth, board, subject, intrinsics, pose,
        RelativeDepthOffsetPolicy(
            model_systematic_error_mm=2.0,
            dominant_plane_enabled=True,
            dominant_plane_half_width_mm=4.0,
            minimum_dominant_plane_support_ratio=0.35,
        ),
    )

    assert full_subject.state == "AVAILABLE"
    assert selected_plane.state == "AVAILABLE"
    assert selected_plane.offset_mm == pytest.approx(34.0, abs=0.05)
    assert selected_plane.subject_depth_spread_p95_mm is not None
    assert selected_plane.subject_depth_spread_p95_mm < 0.1
    assert selected_plane.lower95_mm is not None and selected_plane.upper95_mm is not None
    assert full_subject.lower95_mm is not None and full_subject.upper95_mm is not None
    assert selected_plane.upper95_mm - selected_plane.lower95_mm < (
        full_subject.upper95_mm - full_subject.lower95_mm
    )


def test_dominant_plane_refuses_a_subject_without_one_supported_depth_layer():
    depth, board, subject, intrinsics, pose = _scene()
    rows, _ = np.indices(depth.shape)
    subject_rows = rows[subject]
    for index, offset_mm in enumerate((8.0, 24.0, 40.0, 56.0, 72.0, 88.0)):
        layer = subject & ((rows - int(np.min(subject_rows))) * 6 // len(np.unique(subject_rows)) == index)
        depth = _set_parallel_offset(depth, layer, intrinsics, pose, offset_mm)

    evidence = estimate_front_offset_from_relative_inverse_depth(
        depth, board, subject, intrinsics, pose,
        RelativeDepthOffsetPolicy(
            dominant_plane_enabled=True,
            dominant_plane_half_width_mm=4.0,
            minimum_dominant_plane_support_ratio=0.35,
        ),
    )

    assert evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "RELATIVE_DEPTH_DOMINANT_PLANE_SUPPORT_INSUFFICIENT"


def test_relative_depth_posterior_must_be_contained_and_narrower_than_the_prior():
    narrow = RelativeDepthOffsetEstimate(
        state="AVAILABLE", offset_mm=34.0, lower95_mm=30.0, upper95_mm=38.0,
    )
    wider = RelativeDepthOffsetEstimate(
        state="AVAILABLE", offset_mm=34.0, lower95_mm=0.0, upper95_mm=70.0,
    )
    crossing = RelativeDepthOffsetEstimate(
        state="AVAILABLE", offset_mm=30.0, lower95_mm=18.0, upper95_mm=38.0,
    )

    assert relative_depth_posterior_improves_prior(narrow, 20.0, 50.0)
    assert not relative_depth_posterior_improves_prior(wider, 0.0, 70.0)
    assert not relative_depth_posterior_improves_prior(crossing, 20.0, 50.0)


def test_relative_depth_refuses_only_when_the_depth_map_cannot_be_bound_to_the_board():
    depth, board, subject, intrinsics, pose = _scene()
    board[:] = False

    evidence = estimate_front_offset_from_relative_inverse_depth(depth, board, subject, intrinsics, pose)

    assert evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "RELATIVE_DEPTH_BOARD_SUPPORT_INSUFFICIENT"
