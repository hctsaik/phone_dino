import numpy as np
import pytest

from phone_dino.depth_shadow import DepthShadowPolicy, assess_relative_depth_coplanarity


def scene(offset: float = 0.0):
    rows, columns = np.indices((80, 100), dtype=np.float64)
    depth = 1.0 + columns * 0.002 + rows * 0.001
    board = np.zeros_like(depth, dtype=bool)
    board[8:72, 5:30] = True
    subject = np.zeros_like(depth, dtype=bool)
    subject[20:65, 45:90] = True
    depth[subject] += offset
    return depth, board, subject


def test_relative_depth_shadow_accepts_a_subject_on_the_fitted_board_plane():
    evidence = assess_relative_depth_coplanarity(*scene())

    assert evidence.state == "AVAILABLE"
    assert evidence.observation == "COPLANAR_CANDIDATE"
    assert evidence.subject_p95_residual_ratio == pytest.approx(0.0, abs=1e-10)
    assert evidence.disclaimer_code == "DEPTH_SHADOW_NOT_METRIC_PROOF"


def test_relative_depth_shadow_flags_depth_separation_without_emitting_metric_depth():
    evidence = assess_relative_depth_coplanarity(*scene(offset=0.2))

    assert evidence.state == "AVAILABLE"
    assert evidence.observation == "NON_COPLANAR_RISK"
    assert evidence.subject_median_residual_ratio is not None
    assert evidence.subject_median_residual_ratio > 0.08


def test_relative_depth_shadow_is_invariant_to_affine_depth_scale_and_shift():
    depth, board, subject = scene(offset=0.2)
    original = assess_relative_depth_coplanarity(depth, board, subject)
    transformed = assess_relative_depth_coplanarity(depth * 7.0 - 3.0, board, subject)

    assert transformed.observation == original.observation
    assert transformed.subject_p95_residual_ratio == pytest.approx(original.subject_p95_residual_ratio)


def test_relative_depth_shadow_fails_closed_without_enough_board_support():
    depth, board, subject = scene()
    board[:] = False
    board[0:2, 0:2] = True

    evidence = assess_relative_depth_coplanarity(
        depth, board, subject, DepthShadowPolicy(min_board_pixels=10),
    )

    assert evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "DEPTH_BOARD_SUPPORT_INSUFFICIENT"
