from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy", reason="production vision extra is not installed")
cv2 = pytest.importorskip("cv2", reason="production OpenCV extra is not installed")

from phone_dino.artifacts import CharucoBoard, DimensionMeasurementPolicy, StillGate
from phone_dino.contracts import AlignmentObservation, GoldenDimensionBoardCandidate, PhysicalDimensionEvidence
from phone_dino.decoder import DecodedImage
from phone_dino.production import (
    NormalizedCapture,
    OpenCvCharucoPlaneNormalizer,
    PlaneMetricCalibration,
    PlaneNormalizedCapture,
    TargetMetricCalibration,
    _golden_physical_dimension_evidence,
    _physical_dimension_evidence,
)
from phone_dino.segmenters import SubjectMaskPrediction


def _policy(**updates) -> DimensionMeasurementPolicy:
    body = {
        "version": "dimension-1.0",
        "method": "CHARUCO_PLANE_CURRENT_MASK_MIN_AREA_RECT_V1",
        "approvalState": "ENGINEERING_AUTO",
        "calibrationSource": "CHARUCO_BOARD_PLANE_V1",
        "maxPlaneReprojectionErrorPx": 3.0,
        "segmentationBoundaryUncertaintyPx": 2.0,
        "maxRelativeLinearUncertainty": 0.1,
        "minContourAreaPx": 1000,
        "minContourPoints": 100,
    }
    body.update(updates)
    return DimensionMeasurementPolicy.model_validate(body)


def _artifact(**policy_updates):
    return SimpleNamespace(
        dimension_measurement_policy=_policy(**policy_updates),
        target_alignment=SimpleNamespace(canonical_width=400, canonical_height=300),
    )


def _alignment() -> AlignmentObservation:
    return AlignmentObservation(
        state="ALIGNED", method="TARGET_AFFINE", targetRelative=True,
        inlierCount=24, inlierRatio=0.9, reprojectionErrorPx=0.5,
        coverageRatio=0.5, transformWithinBounds=True, inspectionMaskApplied=True,
    )


def _prediction() -> SubjectMaskPrediction:
    mask = np.zeros((300, 400), dtype=np.uint8)
    rectangle = ((200.0, 150.0), (200.0, 100.0), 17.0)
    cv2.fillConvexPoly(mask, np.int32(np.round(cv2.boxPoints(rectangle))), 255)
    return SubjectMaskPrediction(
        mask=mask, quality_score=0.99, prompt_box_xyxy=(0, 0, 400, 300),
        foreground_ratio=float(np.count_nonzero(mask)) / mask.size,
    )


def _normalized(*, calibration=True, reprojection_error_px=0.5) -> NormalizedCapture:
    metric = None
    if calibration:
        metric = TargetMetricCalibration(
            target_to_plane=np.eye(3, dtype=np.float64),
            pixels_per_mm_x=10.0,
            pixels_per_mm_y=10.0,
            detected_corner_count=18,
            inlier_corner_count=17,
            reprojection_error_px=reprojection_error_px,
        )
    return NormalizedCapture(
        rgb=np.zeros((300, 400, 3), dtype=np.uint8), encoded=b"png",
        alignment=_alignment(), metric_calibration=metric,
    )


def test_current_mask_is_measured_in_charuco_plane_mm_with_uncertainty():
    evidence = _physical_dimension_evidence(_artifact(), _normalized(), _prediction())

    assert evidence.state == "AVAILABLE"
    assert evidence.coordinate_space == "CHARUCO_BOARD_PLANE_MM"
    assert evidence.length_mm == pytest.approx(20.0, abs=0.2)
    assert evidence.width_mm == pytest.approx(10.0, abs=0.2)
    assert evidence.area_mm2 == pytest.approx(200.0, abs=3.0)
    assert evidence.calibration is not None
    assert evidence.calibration.inlier_corner_count == 17
    assert evidence.uncertainty is not None
    assert evidence.uncertainty.linear_mm == pytest.approx(0.5, abs=1e-4)
    assert evidence.current_subject_mask_sha256 is not None


def test_golden_source_mask_is_measured_in_its_own_charuco_plane():
    plane = PlaneNormalizedCapture(
        rgb=np.zeros((300, 400, 3), dtype=np.uint8),
        reason_codes=(),
        metric_calibration=PlaneMetricCalibration(
            pixels_per_mm_x=10.0,
            pixels_per_mm_y=10.0,
            detected_corner_count=18,
            inlier_corner_count=17,
            reprojection_error_px=0.5,
        ),
        source_rgb=np.zeros((300, 400, 3), dtype=np.uint8),
        input_to_plane=np.eye(3, dtype=np.float64),
    )

    evidence = _golden_physical_dimension_evidence(_artifact(), plane, _prediction())

    assert evidence.state == "AVAILABLE"
    assert evidence.method == "CHARUCO_PLANE_GOLDEN_MASK_MIN_AREA_RECT_V1"
    assert evidence.length_mm == pytest.approx(20.0, abs=0.2)
    assert evidence.width_mm == pytest.approx(10.0, abs=0.2)
    assert evidence.current_subject_mask_sha256 is not None


def test_golden_dimensions_fail_closed_without_source_to_plane_transform():
    plane = PlaneNormalizedCapture(
        rgb=np.zeros((300, 400, 3), dtype=np.uint8),
        reason_codes=(),
        metric_calibration=PlaneMetricCalibration(
            pixels_per_mm_x=10.0,
            pixels_per_mm_y=10.0,
            detected_corner_count=18,
            inlier_corner_count=17,
            reprojection_error_px=0.5,
        ),
        source_rgb=np.zeros((300, 400, 3), dtype=np.uint8),
        input_to_plane=None,
    )

    evidence = _golden_physical_dimension_evidence(_artifact(), plane, _prediction())

    assert evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CHARUCO_CALIBRATION_REQUIRED"


@pytest.mark.parametrize(
    ("normalized", "artifact", "reason"),
    [
        (_normalized(calibration=False), _artifact(), "CHARUCO_CALIBRATION_REQUIRED"),
        (
            _normalized(reprojection_error_px=3.1), _artifact(),
            "CALIBRATION_REPROJECTION_ERROR_ABOVE_POLICY",
        ),
        (
            _normalized(), _artifact(maxRelativeLinearUncertainty=0.01),
            "MEASUREMENT_UNCERTAINTY_ABOVE_POLICY",
        ),
    ],
)
def test_physical_dimensions_fail_closed_without_qualified_evidence(normalized, artifact, reason):
    evidence = _physical_dimension_evidence(artifact, normalized, _prediction())

    assert evidence.model_dump(by_alias=True, exclude_none=True) == {
        "state": "UNAVAILABLE",
        "disclaimerCode": "ENGINEERING_DIMENSION_NOT_METROLOGY_PROOF",
        "reasonCode": reason,
    }


def test_unavailable_dimension_contract_rejects_metric_values():
    with pytest.raises(ValueError, match="must not include metric values"):
        PhysicalDimensionEvidence(
            state="UNAVAILABLE",
            disclaimerCode="ENGINEERING_DIMENSION_NOT_METROLOGY_PROOF",
            reasonCode="CHARUCO_CALIBRATION_REQUIRED",
            lengthMm=10.0,
        )


def test_phonecv_large_profile_marker_ids_produce_metric_charuco_calibration():
    marker_ids = np.arange(100, 117, dtype=np.int32)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    source_board = cv2.aruco.CharucoBoard((7, 5), 10.0, 7.0, dictionary, marker_ids)
    image = source_board.generateImage((700, 500), marginSize=24, borderBits=1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    artifact = SimpleNamespace(
        board=CharucoBoard.model_validate({
            "profileId": "COMPACT_130X90_V1",
            "squaresX": 7, "squaresY": 5,
            "squareLengthMm": 10.0, "markerLengthMm": 7.0,
            "markerIds": marker_ids.tolist(),
            "dictionary": "DICT_5X5_1000",
            "canonicalWidth": 896, "canonicalHeight": 896,
        }),
        still_gate=StillGate.model_validate({
            "minCharucoCorners": 6,
            "minLaplacianVariance": 1.0,
            "maxOverExposureRatio": 0.95,
        }),
    )

    normalized = OpenCvCharucoPlaneNormalizer().normalize_plane(
        DecodedImage(data=encoded.tobytes(), width=700, height=500, format="PNG", elapsed_ms=0),
        artifact,
    )

    assert normalized.reason_codes == ()
    assert normalized.metric_calibration is not None
    assert normalized.metric_calibration.detected_corner_count == 24
    assert normalized.metric_calibration.inlier_corner_count == 24
    assert normalized.metric_calibration.reprojection_error_px < 1.0


def _board_candidate(profile: str) -> GoldenDimensionBoardCandidate:
    small = profile == "CREDIT_CARD_85P6X54_V1"
    squares_x, squares_y = (5, 3) if small else (7, 5)
    return GoldenDimensionBoardCandidate(
        boardId="BC-042",
        revision=5 if small else 6,
        profile=profile,
        manifestSha256="sha256:" + ("5" if small else "6") * 64,
        dictionary="DICT_5X5_1000",
        squaresX=squares_x,
        squaresY=squares_y,
        squareLengthMm=8.0 if small else 10.0,
        markerLengthMm=5.6 if small else 7.0,
        markerIds=list(range(100, 100 + squares_x * squares_y // 2)),
    )


@pytest.mark.parametrize(
    ("profile", "image_size"),
    [
        ("CREDIT_CARD_85P6X54_V1", (600, 360)),
        ("COMPACT_130X90_V1", (700, 500)),
    ],
)
def test_golden_plane_selects_charuco_profile_from_marker_count_and_layout(profile, image_size):
    candidate = _board_candidate(profile)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard(
        (candidate.squares_x, candidate.squares_y),
        candidate.square_length_mm,
        candidate.marker_length_mm,
        dictionary,
        np.asarray(candidate.marker_ids, dtype=np.int32),
    )
    image = board.generateImage(image_size, marginSize=24, borderBits=1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    artifact = SimpleNamespace(
        board=SimpleNamespace(canonical_width=896, canonical_height=896),
        still_gate=StillGate.model_validate({
            "minCharucoCorners": 6,
            "minLaplacianVariance": 1.0,
            "maxOverExposureRatio": 0.95,
        }),
    )

    normalized = OpenCvCharucoPlaneNormalizer().normalize_golden_plane(
        DecodedImage(
            data=encoded.tobytes(), width=image_size[0], height=image_size[1],
            format="PNG", elapsed_ms=0,
        ),
        artifact,
        [_board_candidate("CREDIT_CARD_85P6X54_V1"), _board_candidate("COMPACT_130X90_V1")],
    )

    assert normalized.reason_codes == ()
    assert normalized.calibration_board is not None
    assert normalized.calibration_board.profile == profile
    assert normalized.metric_calibration is not None
    assert normalized.metric_calibration.inlier_corner_count >= 6


def test_golden_plane_refuses_equally_valid_profile_geometry_instead_of_guessing():
    candidate = _board_candidate("CREDIT_CARD_85P6X54_V1")
    duplicate = GoldenDimensionBoardCandidate.model_validate({
        **candidate.model_dump(by_alias=True, mode="json"),
        "boardId": "BC-OTHER",
        "revision": 1,
        "profile": "CREDIT_CARD_DUPLICATE_V1",
        "manifestSha256": "sha256:" + "d" * 64,
    })
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard(
        (candidate.squares_x, candidate.squares_y), candidate.square_length_mm,
        candidate.marker_length_mm, dictionary, np.asarray(candidate.marker_ids, dtype=np.int32),
    )
    image = board.generateImage((600, 360), marginSize=24, borderBits=1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    artifact = SimpleNamespace(
        board=SimpleNamespace(canonical_width=896, canonical_height=896),
        still_gate=StillGate.model_validate({
            "minCharucoCorners": 6, "minLaplacianVariance": 1.0, "maxOverExposureRatio": 0.95,
        }),
    )

    normalized = OpenCvCharucoPlaneNormalizer().normalize_golden_plane(
        DecodedImage(data=encoded.tobytes(), width=600, height=360, format="PNG", elapsed_ms=0),
        artifact,
        [candidate, duplicate],
    )

    assert normalized.reason_codes == ("CHARUCO_BOARD_PROFILE_AMBIGUOUS",)
    assert normalized.metric_calibration is None
