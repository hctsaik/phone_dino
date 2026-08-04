from __future__ import annotations

from types import SimpleNamespace
import base64
import hashlib

import pytest

np = pytest.importorskip("numpy", reason="production vision extra is not installed")
cv2 = pytest.importorskip("cv2", reason="production OpenCV extra is not installed")

from phone_dino.artifacts import CharucoBoard, DimensionMeasurementPolicy, StillGate
from phone_dino.contracts import (
    AlignmentObservation, BboxNormalized, CandidateFilter, DifferenceRegion,
    GoldenDimensionBoardCandidate, GoldenRatioScaleReference, PhysicalDimensionEvidence,
    SpatialDifferenceEvidence,
)
from phone_dino.decoder import DecodedImage
from phone_dino.production import (
    NormalizedCapture,
    OpenCvCharucoPlaneNormalizer,
    PlaneMetricCalibration,
    PlaneNormalizedCapture,
    TargetMetricCalibration,
    _candidate_physical_dimensions,
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


def _candidate_spatial() -> SpatialDifferenceEvidence:
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[140:160, 175:225] = 255
    ok, encoded = cv2.imencode(".png", mask)
    assert ok
    payload = encoded.tobytes()
    return SpatialDifferenceEvidence(
        state="AVAILABLE",
        disclaimerCode="DIFFERENCE_NOT_DEFECT_PROOF",
        generationMethod="PAIRED_INTERIOR_ROI_TILED_PATCH_DISTANCE",
        evidenceRegionNormalized=BboxNormalized(x=0.1, y=0.1, width=0.8, height=0.8),
        evidenceCoordinateSpace="TARGET_CANONICAL_IMAGE",
        regions=[DifferenceRegion(
            id="D-001",
            bboxNormalized=BboxNormalized(x=175 / 400, y=140 / 300, width=50 / 400, height=20 / 300),
            peakScore=0.8,
            meanScore=0.6,
            kind="SUBJECT_INTERIOR",
        )],
        mapPngBase64=base64.b64encode(payload).decode("ascii"),
        maskPngBase64=base64.b64encode(payload).decode("ascii"),
        mapSha256=hashlib.sha256(payload).hexdigest(),
        maskSha256=hashlib.sha256(payload).hexdigest(),
        rawThresholdMaskPngBase64=base64.b64encode(payload).decode("ascii"),
        rawThresholdMaskSha256=hashlib.sha256(payload).hexdigest(),
        candidateFilter=CandidateFilter(
            rawComponentCount=1,
            retainedComponentCount=1,
            suppressedSmallRegionCount=0,
            suppressedByLimitCount=0,
            maskSemantics="RETAINED_CANDIDATES",
        ),
    )


def _axis_aligned_prediction() -> SubjectMaskPrediction:
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[100:200, 100:300] = 255
    return SubjectMaskPrediction(
        mask=mask, quality_score=0.99, prompt_box_xyxy=(0, 0, 400, 300),
        foreground_ratio=float(np.count_nonzero(mask)) / mask.size,
    )


def test_candidate_dimensions_prefer_direct_charuco_projection():
    spatial = _candidate_spatial()
    request = SimpleNamespace(golden_ratio_scale_reference=GoldenRatioScaleReference(
        source="CONFIRMED_GOLDEN_DIMENSION_BASELINE",
        templateId="AT-001",
        sourcePhotoSha256="a" * 64,
        measurementPlane="FRONT",
        lengthMm=200.0,
        widthMm=100.0,
        relativeLinearUncertainty=0.02,
        confirmationSource="AUTO_MEASURED_ACCEPTED",
    ))

    _candidate_physical_dimensions(
        _artifact(), request, _normalized(), _axis_aligned_prediction(), spatial,
    )

    evidence = spatial.regions[0].physical_dimensions
    assert evidence is not None and evidence.state == "AVAILABLE"
    assert evidence.method == "CHARUCO_PLANE_CANDIDATE_MASK_MIN_AREA_RECT_V1"
    assert evidence.length_mm == pytest.approx(4.9, abs=0.2)
    assert evidence.width_mm == pytest.approx(1.9, abs=0.2)
    assert evidence.scale is not None and evidence.scale.source == "CURRENT_CHARUCO_BOARD"


def test_candidate_dimensions_fall_back_to_confirmed_golden_subject_ratio():
    spatial = _candidate_spatial()
    reference = GoldenRatioScaleReference(
        source="CONFIRMED_GOLDEN_DIMENSION_BASELINE",
        templateId="AT-001",
        sourcePhotoSha256="b" * 64,
        measurementPlane="FRONT",
        lengthMm=200.0,
        widthMm=100.0,
        relativeLinearUncertainty=0.02,
        confirmationSource="USER_EDITED",
    )

    _candidate_physical_dimensions(
        _artifact(), SimpleNamespace(golden_ratio_scale_reference=reference),
        _normalized(calibration=False), _axis_aligned_prediction(), spatial,
    )

    evidence = spatial.regions[0].physical_dimensions
    assert evidence is not None and evidence.state == "AVAILABLE"
    assert evidence.method == "GOLDEN_BASELINE_RATIO_CANDIDATE_MASK_MIN_AREA_RECT_V1"
    assert evidence.length_mm == pytest.approx(49.0, abs=1.0)
    assert evidence.width_mm == pytest.approx(19.0, abs=1.0)
    assert evidence.scale is not None
    assert evidence.scale.source == "CONFIRMED_GOLDEN_DIMENSION_BASELINE"
    assert evidence.scale.template_id == "AT-001"


def test_candidate_dimensions_refuse_mm_without_charuco_or_confirmed_golden():
    spatial = _candidate_spatial()

    _candidate_physical_dimensions(
        _artifact(), SimpleNamespace(golden_ratio_scale_reference=None),
        _normalized(calibration=False), _axis_aligned_prediction(), spatial,
    )

    evidence = spatial.regions[0].physical_dimensions
    assert evidence is not None and evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CANDIDATE_SCALE_REQUIRED"


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


def _board_candidate(
    profile: str, *, charuco_geometry_qualified: bool = True,
) -> GoldenDimensionBoardCandidate:
    small = profile == "CREDIT_CARD_85P6X54_V1"
    squares_x, squares_y = (5, 3) if small else (7, 5)
    finished_width, finished_height = (85.6, 53.98) if small else (130.0, 90.0)
    inset, marker_size = (3.0, 9.0) if small else (4.0, 12.0)
    right = finished_width - inset - marker_size
    middle_y = (finished_height - marker_size) / 2.0
    bottom = finished_height - inset - marker_size
    def marker(marker_id: int, x: float, y: float) -> dict[str, object]:
        return {
            "id": marker_id,
            "cornersMm": [
                [x, y, 0.0], [x + marker_size, y, 0.0],
                [x + marker_size, y + marker_size, 0.0], [x, y + marker_size, 0.0],
            ],
        }
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
        finishedWidthMm=finished_width,
        finishedHeightMm=finished_height,
        charucoOriginMm=[22.8, 3.0, 0.0] if small else [30.0, 5.0, 0.0],
        outerMarkers=[
            marker(0, inset, inset), marker(1, right, inset),
            marker(2, inset, middle_y), marker(3, right, middle_y),
            marker(4, inset, bottom), marker(5, right, bottom),
        ],
        charucoGeometryQualified=charuco_geometry_qualified,
        outerArucoGeometryQualified=True,
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


def test_legacy_small_board_uses_outer_aruco_geometry_when_charuco_artwork_is_unqualified():
    candidate = _board_candidate("CREDIT_CARD_85P6X54_V1", charuco_geometry_qualified=False)
    scale = 10
    image = np.full(
        (round(candidate.finished_height_mm * scale), round(candidate.finished_width_mm * scale)),
        255,
        dtype=np.uint8,
    )
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    for marker in candidate.outer_markers:
        top_left = marker.corners_mm[0]
        marker_size = round((marker.corners_mm[1][0] - top_left[0]) * scale)
        rendered = cv2.aruco.generateImageMarker(dictionary, marker.id, marker_size, borderBits=1)
        x, y = round(top_left[0] * scale), round(top_left[1] * scale)
        image[y:y + marker_size, x:x + marker_size] = rendered
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    artifact = SimpleNamespace(
        board=SimpleNamespace(canonical_width=896, canonical_height=896),
        still_gate=StillGate.model_validate({
            "minCharucoCorners": 6,
            "minLaplacianVariance": 1.0,
            "maxOverExposureRatio": 0.99,
        }),
    )

    normalized = OpenCvCharucoPlaneNormalizer().normalize_golden_plane(
        DecodedImage(
            data=encoded.tobytes(), width=image.shape[1], height=image.shape[0],
            format="PNG", elapsed_ms=0,
        ),
        artifact,
        [candidate, _board_candidate("COMPACT_130X90_V1")],
    )

    assert normalized.reason_codes == ()
    assert normalized.calibration_board is not None
    assert normalized.calibration_board.profile == "CREDIT_CARD_85P6X54_V1"
    assert normalized.calibration_fiducial == "OUTER_ARUCO_CORNERS"
    assert normalized.metric_calibration is not None
    assert normalized.metric_calibration.inlier_corner_count == 24
    assert normalized.metric_calibration.reprojection_error_px < 1.0


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
