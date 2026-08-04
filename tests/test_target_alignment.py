from __future__ import annotations

import base64
import hashlib
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

np = pytest.importorskip("numpy", reason="production vision extra is not installed")
cv2 = pytest.importorskip("cv2", reason="production OpenCV extra is not installed")

from phone_dino.artifacts import SubjectAlignmentContract, TargetAlignmentPolicy
from phone_dino.production import (
    OpenCvTargetAligner, _align_with_dark_body_contour, _align_with_subject_ecc,
    _detect_dark_body_contour,
)


def _reference() -> np.ndarray:
    rng = np.random.default_rng(20260801)
    image = np.full((300, 400, 3), 235, dtype=np.uint8)
    # Stable structure is deliberately spread around the future inspection area.
    for _ in range(350):
        x, y = int(rng.integers(10, 390)), int(rng.integers(10, 290))
        color = tuple(int(value) for value in rng.integers(0, 180, 3))
        cv2.circle(image, (x, y), int(rng.integers(2, 5)), color, -1)
    cv2.rectangle(image, (145, 105), (255, 195), (20, 20, 20), -1)
    cv2.putText(image, "INSPECT", (150, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    return image


def _policy(reference: np.ndarray, **updates) -> TargetAlignmentPolicy:
    ok, encoded = cv2.imencode(".png", reference)
    assert ok
    body = {
        "method": "TARGET_AFFINE",
        "referenceImageBase64": base64.b64encode(encoded.tobytes()).decode(),
        "referenceImageSha256": "sha256:" + hashlib.sha256(encoded.tobytes()).hexdigest(),
        "canonicalWidth": 400, "canonicalHeight": 300,
        "alignmentRegions": [
            {"x": 0, "y": 0, "width": 400, "height": 90},
            {"x": 0, "y": 210, "width": 400, "height": 90},
        ],
        "heldOutRegions": [
            {"x": 0, "y": 100, "width": 130, "height": 100},
            {"x": 270, "y": 100, "width": 130, "height": 100},
        ],
        "inspectionRegions": [{"x": 140, "y": 100, "width": 120, "height": 100}],
        "minMatches": 20, "minInliers": 15, "minInlierRatio": 0.5, "minCoverageRatio": 0.12,
        "maxReprojectionErrorPx": 2.5, "minScale": 0.8, "maxScale": 1.2,
        "maxRotationDegrees": 15.0, "maxShear": 0.03, "maxTranslationPx": 250.0,
        "maxSecondaryInlierRatio": 0.35, "minHeldOutMatches": 12, "maxHeldOutReprojectionErrorPx": 3.0,
    }
    body.update(updates)
    return TargetAlignmentPolicy.model_validate(body)


def _move_target(reference: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(reference, matrix, (620, 480), borderValue=(250, 250, 250))


def test_target_alignment_is_independent_of_board_relative_offset_and_masks_inspection():
    reference = _reference()
    current = _move_target(reference, np.float32([[1, 0, 91], [0, 1, 67]]))
    # Simulate a real defect: it must not contribute descriptors used by alignment.
    current[172:262, 236:346] = (0, 0, 255)

    result = OpenCvTargetAligner().align(
        cv2.cvtColor(current, cv2.COLOR_BGR2RGB), SimpleNamespace(target_alignment=_policy(reference))
    )

    assert result.reason_codes == ()
    assert result.alignment is not None
    assert result.alignment.state == "ALIGNED"
    assert result.alignment.target_relative is True
    assert result.alignment.inspection_mask_applied is True
    assert result.alignment.transform_within_bounds is True
    assert result.alignment.inlier_count >= 15
    assert result.alignment.coverage_ratio >= 0.12
    assert result.rgb.shape == reference.shape


def test_target_missing_fails_closed_without_canonical_pixels():
    reference = _reference()
    blank = np.full((480, 620, 3), 255, dtype=np.uint8)

    result = OpenCvTargetAligner().align(
        cv2.cvtColor(blank, cv2.COLOR_BGR2RGB), SimpleNamespace(target_alignment=_policy(reference))
    )

    assert result.rgb is None and result.encoded == b""
    assert result.reason_codes == ("TARGET_NOT_FOUND",)
    assert result.alignment is not None and result.alignment.state == "NOT_ALIGNED"


def test_clustered_evidence_and_out_of_bounds_rotation_fail_closed():
    reference = _reference()
    shifted = _move_target(reference, np.float32([[1, 0, 70], [0, 1, 50]]))
    clustered = OpenCvTargetAligner().align(
        cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB),
        SimpleNamespace(target_alignment=_policy(reference, minCoverageRatio=0.99)),
    )
    assert clustered.rgb is None
    assert "TARGET_INLIERS_CLUSTERED" in clustered.reason_codes

    center = (reference.shape[1] / 2, reference.shape[0] / 2)
    rotated_matrix = cv2.getRotationMatrix2D(center, 28.0, 1.0)
    rotated_matrix[:, 2] += (100, 80)
    rotated = _move_target(reference, rotated_matrix.astype(np.float32))
    bounded = OpenCvTargetAligner().align(
        cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), SimpleNamespace(target_alignment=_policy(reference))
    )
    assert bounded.rgb is None
    assert "TARGET_TRANSFORM_OUT_OF_BOUNDS" in bounded.reason_codes
    assert bounded.alignment is not None and bounded.alignment.transform_within_bounds is False


def test_excessive_translation_duplicate_target_and_held_out_mismatch_fail_closed():
    reference = _reference()
    translated = cv2.warpAffine(
        reference, np.float32([[1, 0, 280], [0, 1, 100]]), (850, 520), borderValue=(250, 250, 250)
    )
    translation_result = OpenCvTargetAligner().align(
        cv2.cvtColor(translated, cv2.COLOR_BGR2RGB), SimpleNamespace(target_alignment=_policy(reference))
    )
    assert "TARGET_TRANSFORM_OUT_OF_BOUNDS" in translation_result.reason_codes

    duplicate = np.full((480, 930, 3), 250, dtype=np.uint8)
    duplicate[50:350, 40:440] = reference
    duplicate[50:350, 490:890] = reference
    duplicate_result = OpenCvTargetAligner().align(
        cv2.cvtColor(duplicate, cv2.COLOR_BGR2RGB),
        SimpleNamespace(target_alignment=_policy(reference, maxTranslationPx=600.0)),
    )
    assert duplicate_result.rgb is None
    assert "TARGET_AMBIGUOUS" in duplicate_result.reason_codes

    mismatched = _move_target(reference, np.float32([[1, 0, 70], [0, 1, 50]]))
    # Remove only the independent held-out regions after applying target offset.
    mismatched[150:250, 70:200] = 255
    mismatched[150:250, 340:470] = 255
    held_out_result = OpenCvTargetAligner().align(
        cv2.cvtColor(mismatched, cv2.COLOR_BGR2RGB), SimpleNamespace(target_alignment=_policy(reference))
    )
    assert held_out_result.rgb is None
    assert "TARGET_PARALLAX_OR_MISMATCH" in held_out_result.reason_codes


def test_alignment_is_repeatable_across_process_history_and_concurrent_calls():
    reference = _reference()
    current = _move_target(reference, np.float32([[1, 0, 91], [0, 1, 67]]))
    rgb = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
    artifact = SimpleNamespace(target_alignment=_policy(reference))
    aligner = OpenCvTargetAligner()

    def observe(_index):
        result = aligner.align(rgb, artifact)
        return result.reason_codes, result.encoded, result.alignment.model_dump_json() if result.alignment else None

    sequential = [observe(index) for index in range(10)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        concurrent = list(pool.map(observe, range(12)))
    assert all(value == sequential[0] for value in (*sequential, *concurrent))


def test_engineering_dark_body_contour_finds_only_a_central_equipment_body():
    image = np.full((900, 700, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (230, 315), (470, 700), (18, 18, 18), -1)

    detected = _detect_dark_body_contour(image, expected_aspect=240 / 385)

    assert detected is not None
    assert detected.x == pytest.approx(230, abs=4)
    assert detected.y == pytest.approx(315, abs=4)
    assert detected.width == pytest.approx(241, abs=5)
    assert detected.height == pytest.approx(386, abs=5)
    assert detected.rectangularity >= 0.95

    blank = np.full_like(image, 235)
    assert _detect_dark_body_contour(blank, expected_aspect=240 / 385) is None

    off_center = blank.copy()
    cv2.rectangle(off_center, (0, 315), (180, 700), (18, 18, 18), -1)
    assert _detect_dark_body_contour(off_center, expected_aspect=240 / 385) is None


def test_engineering_contour_alignment_reports_its_limited_method_truthfully():
    image = np.full((900, 700, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (230, 315), (470, 700), (18, 18, 18), -1)
    policy = SimpleNamespace(
        contour_anchor_region=SimpleNamespace(x=230, y=315, width=241, height=386),
        canonical_width=700, canonical_height=900,
        min_scale=0.8, max_scale=1.2, max_translation_px=30.0,
        max_reprojection_error_px=8.0,
    )

    result = _align_with_dark_body_contour(image, policy)

    assert result.reason_codes == ()
    assert result.alignment is not None
    assert result.alignment.method == "CONTOUR_ANCHOR_AFFINE"


def test_subject_boundary_ecc_refines_rotation_and_validates_held_out_edges():
    height, width = 900, 700
    reference = np.full((height, width, 3), 220, dtype=np.uint8)
    cv2.rectangle(reference, (230, 315), (470, 700), (18, 18, 18), -1)
    cv2.rectangle(reference, (248, 340), (452, 675), (35, 35, 35), 3)
    cv2.circle(reference, (350, 560), 80, (80, 80, 80), 4)
    cv2.line(reference, (250, 400), (450, 400), (120, 120, 120), 5)
    subject = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(subject, (230, 315), (470, 700), 255, -1)
    transform = cv2.getRotationMatrix2D((350, 507), 7.0, 1.05)
    transform[:, 2] += (18, -12)
    current = cv2.warpAffine(reference, transform, (width, height), borderValue=(220, 220, 220))
    # An added object crossing the current silhouette must remain comparison
    # evidence; its extra edges cannot veto an otherwise valid pose.
    cv2.rectangle(current, (285, 270), (430, 335), (255, 80, 0), -1)
    policy = SimpleNamespace(
        contour_anchor_region=SimpleNamespace(x=230, y=315, width=241, height=386),
        canonical_width=width, canonical_height=height,
        min_scale=0.6, max_scale=1.5, max_translation_px=500.0,
        max_reprojection_error_px=8.0,
    )
    contract = SubjectAlignmentContract.model_validate({
        "version": "subject-align-1.0", "method": "SUBJECT_CONTOUR_ECC_AFFINE",
        "approvalState": "ENGINEERING_AUTO", "maskSource": "GOLDEN_SUBJECT_MASK",
        "alignmentBandPx": 24, "heldOutBlockPx": 32, "maxIterations": 200,
        "convergenceEpsilon": 0.00001, "minEccCorrelation": 0.2,
        "maxHeldOutResidualPx": 8.0, "minHeldOutCoverageRatio": 0.35,
        "maxResidualTranslationPx": 120.0, "maxResidualRotationDegrees": 15.0,
        "maxResidualScaleDelta": 0.25, "maxResidualShear": 0.15,
    })

    result = _align_with_subject_ecc(current, reference, policy, subject, contract)

    assert result.reason_codes == ()
    assert result.alignment is not None
    assert result.alignment.method == "SUBJECT_CONTOUR_ECC_AFFINE"
    assert result.alignment.inlier_ratio >= 0.9
    assert result.alignment.reprojection_error_px <= 2.0
    assert result.alignment.coverage_ratio >= 0.2
    assert result.rgb.shape == reference.shape


def test_subject_boundary_ecc_ignores_background_edges_inside_the_search_band():
    height, width = 900, 700
    reference = np.full((height, width, 3), 220, dtype=np.uint8)
    cv2.rectangle(reference, (230, 315), (470, 700), (18, 18, 18), -1)
    # Immutable background texture sits inside the broad +/-24 px search band,
    # but does not touch the approved subject-mask boundary.
    for y in range(330, 690, 30):
        cv2.line(reference, (210, y), (225, y), (0, 0, 0), 3)
        cv2.line(reference, (475, y), (490, y), (0, 0, 0), 3)
    subject = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(subject, (230, 315), (470, 700), 255, -1)
    current = reference.copy()
    current[:, :230] = 245
    current[:, 471:] = 245
    policy = SimpleNamespace(
        contour_anchor_region=SimpleNamespace(x=230, y=315, width=241, height=386),
        canonical_width=width, canonical_height=height,
        min_scale=0.6, max_scale=1.5, max_rotation_degrees=35.0,
        max_translation_px=500.0, max_reprojection_error_px=8.0,
    )
    contract = SubjectAlignmentContract.model_validate({
        "version": "subject-align-1.0", "method": "SUBJECT_CONTOUR_ECC_AFFINE",
        "approvalState": "ENGINEERING_AUTO", "maskSource": "GOLDEN_SUBJECT_MASK",
        "alignmentBandPx": 24, "heldOutBlockPx": 32, "maxIterations": 200,
        "convergenceEpsilon": 0.00001, "minEccCorrelation": 0.2,
        "maxHeldOutResidualPx": 8.0, "minHeldOutCoverageRatio": 0.35,
        "maxResidualTranslationPx": 120.0, "maxResidualRotationDegrees": 15.0,
        "maxResidualScaleDelta": 0.25, "maxResidualShear": 0.15,
    })

    result = _align_with_subject_ecc(current, reference, policy, subject, contract)

    assert result.reason_codes == ()
    assert result.alignment is not None
    assert result.alignment.state == "ALIGNED"
    assert result.alignment.reprojection_error_px <= 1.0


def test_subject_boundary_ecc_still_rejects_non_affine_subject_boundary_mismatch():
    height, width = 900, 700
    reference = np.full((height, width, 3), 220, dtype=np.uint8)
    cv2.rectangle(reference, (230, 315), (470, 700), (18, 18, 18), -1)
    subject = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(subject, (230, 315), (470, 700), 255, -1)
    current = np.full_like(reference, 220)
    # A trapezoid has a similar global Canny silhouette but no bounded affine
    # hypothesis can make its top and bottom boundaries agree simultaneously.
    cv2.fillConvexPoly(
        current,
        np.asarray([(285, 315), (415, 315), (470, 700), (230, 700)], dtype=np.int32),
        (18, 18, 18),
    )
    policy = SimpleNamespace(
        contour_anchor_region=SimpleNamespace(x=230, y=315, width=241, height=386),
        canonical_width=width, canonical_height=height,
        min_scale=0.6, max_scale=1.5, max_rotation_degrees=35.0,
        max_translation_px=500.0, max_reprojection_error_px=8.0,
    )
    contract = SubjectAlignmentContract.model_validate({
        "version": "subject-align-1.0", "method": "SUBJECT_CONTOUR_ECC_AFFINE",
        "approvalState": "ENGINEERING_AUTO", "maskSource": "GOLDEN_SUBJECT_MASK",
        "alignmentBandPx": 24, "heldOutBlockPx": 32, "maxIterations": 200,
        "convergenceEpsilon": 0.00001, "minEccCorrelation": 0.2,
        "maxHeldOutResidualPx": 8.0, "minHeldOutCoverageRatio": 0.35,
        "maxResidualTranslationPx": 120.0, "maxResidualRotationDegrees": 15.0,
        "maxResidualScaleDelta": 0.25, "maxResidualShear": 0.15,
    })

    result = _align_with_subject_ecc(current, reference, policy, subject, contract)

    assert result.rgb is None
    assert result.alignment is not None
    assert result.alignment.state == "NOT_ALIGNED"
    assert "SUBJECT_ALIGNMENT_HELD_OUT_RESIDUAL_HIGH" in result.reason_codes


def test_feature_alignment_remains_first_when_contour_fallback_is_enabled():
    reference = _reference()
    current = _move_target(reference, np.float32([[1, 0, 91], [0, 1, 67]]))

    result = OpenCvTargetAligner(allow_contour_anchor_alignment=True).align(
        cv2.cvtColor(current, cv2.COLOR_BGR2RGB),
        SimpleNamespace(target_alignment=_policy(
            reference,
            contourAnchorRegion={"x": 145, "y": 105, "width": 110, "height": 90},
        )),
    )

    assert result.reason_codes == ()
    assert result.alignment is not None
    assert result.alignment.method == "TARGET_AFFINE"
