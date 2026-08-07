from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import base64
import hashlib

import pytest

np = pytest.importorskip("numpy", reason="production vision extra is not installed")
cv2 = pytest.importorskip("cv2", reason="production OpenCV extra is not installed")

from phone_dino.artifacts import CharucoBoard, DimensionMeasurementPolicy, StillGate
from phone_dino.contracts import (
    AlignmentObservation, BboxNormalized, CandidateFilter, DifferenceRegion,
    GoldenDimensionBoardCandidate, PhysicalDimensionEvidence,
    SpatialDifferenceEvidence,
)
from phone_dino.decoder import DecodedImage
from phone_dino.production import (
    NormalizedCapture,
    OpenCvCharucoPlaneNormalizer,
    PlaneMetricCalibration,
    PlaneNormalizedCapture,
    TargetMetricCalibration,
    _background_board_offset_plane_physical_dimension_evidence,
    _candidate_physical_dimensions,
    _golden_physical_dimension_evidence,
    _physical_dimension_evidence,
    _robust_body_width_from_projected_mask_points,
    _clean_source_metric_mask,
    _source_metric_prediction,
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


def _normalized(*, calibration=True, reprojection_error_px=0.5, fiducial="CHARUCO_CORNERS", support=True) -> NormalizedCapture:
    metric = None
    if calibration:
        metric = TargetMetricCalibration(
            target_to_plane=np.eye(3, dtype=np.float64),
            pixels_per_mm_x=10.0,
            pixels_per_mm_y=10.0,
            detected_corner_count=18,
            inlier_corner_count=17,
            reprojection_error_px=reprojection_error_px,
            calibration_fiducial=fiducial,
        )
    return NormalizedCapture(
        rgb=np.zeros((300, 400, 3), dtype=np.uint8), encoded=b"png",
        alignment=_alignment(), metric_calibration=metric,
        calibration_support_plane=(
            np.asarray(((0, 0), (400, 0), (400, 300), (0, 300)), dtype=np.float32)
            if support else None
        ),
    )


def test_current_mask_is_measured_in_charuco_plane_mm_with_uncertainty():
    evidence = _physical_dimension_evidence(_artifact(), _normalized(), _prediction())

    assert evidence.state == "AVAILABLE"
    assert evidence.coordinate_space == "CHARUCO_BOARD_PLANE_MM"
    assert evidence.length_mm == pytest.approx(20.0, abs=0.2)
    assert evidence.width_mm == pytest.approx(10.0, abs=0.2)
    assert evidence.area_mm2 == pytest.approx(200.0, abs=3.0)
    assert evidence.calibration is not None
    assert evidence.calibration.fiducial == "CHARUCO_CORNERS"
    assert evidence.calibration.inlier_corner_count == 17
    assert evidence.uncertainty is not None
    assert evidence.uncertainty.linear_mm == pytest.approx(0.5, abs=1e-4)
    assert evidence.current_subject_mask_sha256 is not None


def test_current_source_mask_uses_the_same_full_resolution_metric_path_as_golden():
    normalized = replace(
        _normalized(),
        source_rgb=np.zeros((300, 400, 3), dtype=np.uint8),
        source_to_plane=np.eye(3, dtype=np.float64),
    )

    evidence = _physical_dimension_evidence(_artifact(), normalized, _prediction())

    assert evidence.state == "AVAILABLE"
    assert evidence.length_mm == pytest.approx(20.0, abs=0.2)
    assert evidence.width_mm == pytest.approx(10.0, abs=0.2)


def test_source_metric_prompt_expands_beyond_similarity_roi_without_clipping_the_subject():
    class Segmenter:
        def __init__(self):
            self.prompt = None

        def segment(self, source, prompt_box_xyxy, **_kwargs):
            self.prompt = prompt_box_xyxy
            mask = np.zeros(source.shape[:2], dtype=np.uint8)
            # Its top edge deliberately lies outside the tight comparison ROI
            # but inside the physical-metric prompt padding.
            cv2.rectangle(mask, (110, 90), (290, 190), 255, thickness=-1)
            return SubjectMaskPrediction(
                mask=mask, quality_score=0.99, prompt_box_xyxy=prompt_box_xyxy,
                foreground_ratio=float(np.count_nonzero(mask)) / mask.size,
            )

    roi = SimpleNamespace(
        canonical_width=400,
        canonical_height=300,
        polygon=[
            SimpleNamespace(x=100, y=100), SimpleNamespace(x=300, y=100),
            SimpleNamespace(x=300, y=200), SimpleNamespace(x=100, y=200),
        ],
        inspection_regions=[SimpleNamespace(x=100, y=100, width=200, height=100)],
    )
    artifact = SimpleNamespace(
        inspection_roi=roi,
        subject_segmentation=SimpleNamespace(
            min_foreground_ratio=0.01, max_foreground_ratio=0.9, min_model_quality_score=0.5,
        ),
    )
    normalized = replace(
        _normalized(),
        source_rgb=np.zeros((300, 400, 3), dtype=np.uint8),
        target_from_source=np.eye(3, dtype=np.float64),
    )
    segmenter = Segmenter()

    prediction = _source_metric_prediction(artifact, normalized, segmenter)

    assert segmenter.prompt == (80, 80, 320, 220)
    assert prediction.mask[90, 200] == 255


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
        calibration_support_plane=np.asarray(((0, 0), (400, 0), (400, 300), (0, 300)), dtype=np.float32),
    )

    evidence = _golden_physical_dimension_evidence(_artifact(), plane, _prediction())

    assert evidence.state == "AVAILABLE"
    assert evidence.method == "CHARUCO_PLANE_GOLDEN_MASK_MIN_AREA_RECT_V1"
    assert evidence.length_mm == pytest.approx(20.0, abs=0.2)
    assert evidence.width_mm == pytest.approx(10.0, abs=0.2)
    assert evidence.calibration is not None
    assert evidence.calibration.fiducial == "CHARUCO_CORNERS"
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
        calibration_support_plane=np.asarray(((0, 0), (400, 0), (400, 300), (0, 300)), dtype=np.float32),
    )

    evidence = _golden_physical_dimension_evidence(_artifact(), plane, _prediction())

    assert evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CHARUCO_CALIBRATION_REQUIRED"


def test_golden_outer_aruco_fallback_is_identification_not_whole_subject_metrology():
    plane = PlaneNormalizedCapture(
        rgb=np.zeros((300, 400, 3), dtype=np.uint8),
        metric_calibration=PlaneMetricCalibration(
            pixels_per_mm_x=10.0,
            pixels_per_mm_y=10.0,
            detected_corner_count=24,
            inlier_corner_count=21,
            reprojection_error_px=0.5,
            calibration_fiducial="OUTER_ARUCO_CORNERS",
        ),
        source_rgb=np.zeros((300, 400, 3), dtype=np.uint8),
        input_to_plane=np.eye(3, dtype=np.float64),
        calibration_support_plane=np.asarray(((0, 0), (400, 0), (400, 300), (0, 300)), dtype=np.float32),
    )

    evidence = _golden_physical_dimension_evidence(_artifact(), plane, _prediction())

    assert evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CHARUCO_CORNERS_REQUIRED_FOR_WHOLE_SUBJECT_METRICS"


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


def test_outer_aruco_fallback_cannot_authorize_whole_subject_dimensions():
    evidence = _physical_dimension_evidence(
        _artifact(), _normalized(fiducial="OUTER_ARUCO_CORNERS"), _prediction(),
    )

    assert evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CHARUCO_CORNERS_REQUIRED_FOR_WHOLE_SUBJECT_METRICS"


def test_subject_outside_board_support_is_rejected_not_extrapolated():
    evidence = _physical_dimension_evidence(
        _artifact(), _normalized(support=False), _prediction(),
    )

    assert evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CALIBRATION_SUPPORT_REQUIRED"

    normalized = _normalized()
    normalized = replace(
        normalized,
        calibration_support_plane=np.asarray(((0, 0), (180, 0), (180, 300), (0, 300)), dtype=np.float32),
    )
    evidence = _physical_dimension_evidence(_artifact(), normalized, _prediction())
    assert evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "SUBJECT_OUTSIDE_CALIBRATION_PLANE_SUPPORT"


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
    request = SimpleNamespace()

    _candidate_physical_dimensions(
        _artifact(), request, _normalized(), _axis_aligned_prediction(), spatial,
    )

    evidence = spatial.regions[0].physical_dimensions
    assert evidence is not None and evidence.state == "AVAILABLE"
    assert evidence.method == "CHARUCO_PLANE_CANDIDATE_MASK_MIN_AREA_RECT_V1"
    assert evidence.length_mm == pytest.approx(4.9, abs=0.2)
    assert evidence.width_mm == pytest.approx(1.9, abs=0.2)
    assert evidence.scale is not None and evidence.scale.source == "CURRENT_CHARUCO_BOARD"


def test_candidate_dimensions_do_not_fall_back_to_confirmed_golden_subject_ratio():
    spatial = _candidate_spatial()
    _candidate_physical_dimensions(
        _artifact(), SimpleNamespace(),
        _normalized(calibration=False), _axis_aligned_prediction(), spatial,
    )

    evidence = spatial.regions[0].physical_dimensions
    assert evidence is not None and evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CANDIDATE_CHARUCO_CALIBRATION_REQUIRED"


def test_candidate_dimensions_refuse_mm_without_true_charuco():
    spatial = _candidate_spatial()

    _candidate_physical_dimensions(
        _artifact(), SimpleNamespace(),
        _normalized(calibration=False), _axis_aligned_prediction(), spatial,
    )

    evidence = spatial.regions[0].physical_dimensions
    assert evidence is not None and evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CANDIDATE_CHARUCO_CALIBRATION_REQUIRED"


def test_candidate_dimensions_reject_outer_aruco_or_missing_board_support():
    spatial = _candidate_spatial()
    _candidate_physical_dimensions(
        _artifact(), SimpleNamespace(),
        _normalized(fiducial="OUTER_ARUCO_CORNERS"), _axis_aligned_prediction(), spatial,
    )
    evidence = spatial.regions[0].physical_dimensions
    assert evidence is not None and evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CHARUCO_CORNERS_REQUIRED_FOR_CANDIDATE_METRICS"

    spatial = _candidate_spatial()
    _candidate_physical_dimensions(
        _artifact(), SimpleNamespace(),
        _normalized(support=False), _axis_aligned_prediction(), spatial,
    )
    evidence = spatial.regions[0].physical_dimensions
    assert evidence is not None and evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CALIBRATION_SUPPORT_REQUIRED"


def test_candidate_dimensions_reject_extrapolation_outside_board_support():
    spatial = _candidate_spatial()
    normalized = replace(
        _normalized(),
        calibration_support_plane=np.asarray(((0, 0), (160, 0), (160, 300), (0, 300)), dtype=np.float32),
    )
    _candidate_physical_dimensions(
        _artifact(), SimpleNamespace(),
        normalized, _axis_aligned_prediction(), spatial,
    )

    evidence = spatial.regions[0].physical_dimensions
    assert evidence is not None and evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "CANDIDATE_OUTSIDE_CALIBRATION_PLANE_SUPPORT"


def test_background_outer_aruco_pnp_projects_a_front_offset_subject_in_mm():
    image_width, image_height = 1280, 960
    camera_matrix = np.asarray(((1180.0, 0.0, 640.0), (0.0, 1180.0, 480.0), (0.0, 0.0, 1.0)))
    rvec = np.asarray((0.10, -0.16, 0.025), dtype=np.float64)
    tvec = np.asarray((-92.0, -76.0, 760.0), dtype=np.float64)
    outer_positions = ((6, 6), (178, 6), (6, 107), (178, 107), (6, 208), (178, 208))
    outer_markers = [
        {"id": marker_id, "cornersMm": [[x, y, 0], [x + 16, y, 0], [x + 16, y + 16, 0], [x, y + 16, 0]]}
        for marker_id, (x, y) in enumerate(outer_positions)
    ]
    board = GoldenDimensionBoardCandidate.model_validate({
        "boardId": "A4-METRIC", "revision": 1, "profile": "A4_METRIC_200X230_V1",
        "manifestSha256": "sha256:" + "a" * 64, "dictionary": "DICT_5X5_1000",
        "squaresX": 7, "squaresY": 9, "squareLengthMm": 20.0, "markerLengthMm": 14.0,
        "markerIds": list(range(100, 131)), "finishedWidthMm": 200.0, "finishedHeightMm": 230.0,
        "charucoOriginMm": [30.0, 25.0, 0.0], "outerMarkers": outer_markers,
        "charucoGeometryQualified": True, "outerArucoGeometryQualified": True,
    })
    source = np.full((image_height, image_width), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    marker_source = np.asarray(((0, 0), (160, 0), (160, 160), (0, 160)), dtype=np.float32)
    for marker in board.outer_markers:
        image_marker = cv2.aruco.generateImageMarker(dictionary, marker.id, 160, borderBits=1)
        points = np.asarray(marker.corners_mm, dtype=np.float64)
        projected, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, None)
        target = projected.reshape(4, 2).astype(np.float32)
        transform = cv2.getPerspectiveTransform(marker_source, target)
        rendered = cv2.warpPerspective(image_marker, transform, (image_width, image_height), borderValue=255)
        mask = cv2.warpPerspective(np.full_like(image_marker, 255), transform, (image_width, image_height))
        source[mask > 0] = rendered[mask > 0]

    front_offset = 34.0
    subject_mm = np.asarray(((9, 21, -front_offset), (192, 21, -front_offset), (192, 80, -front_offset), (9, 80, -front_offset)), dtype=np.float64)
    subject_pixels, _ = cv2.projectPoints(subject_mm, rvec, tvec, camera_matrix, None)
    mask = np.zeros_like(source)
    cv2.fillConvexPoly(mask, np.int32(np.round(subject_pixels.reshape(-1, 2))), 255)
    source[mask > 0] = 96
    prediction = SubjectMaskPrediction(
        mask=mask, quality_score=0.99, prompt_box_xyxy=(0, 0, image_width, image_height),
        foreground_ratio=float(np.count_nonzero(mask)) / mask.size,
    )
    request = SimpleNamespace(offset_plane_calibration=SimpleNamespace(
        front_plane_offset_mm=front_offset,
        camera=SimpleNamespace(
            image_width=image_width, image_height=image_height,
                fx_px=1180.0, fy_px=1180.0, cx_px=640.0, cy_px=480.0,
            distortion_coefficients=[],
        ),
        min_board_pnp_inliers=8,
        max_board_pnp_reprojection_error_px=1.5,
    ))
    normalized = NormalizedCapture(
        rgb=np.dstack((source, source, source)), encoded=b"png", alignment=_alignment(),
        source_rgb=np.dstack((source, source, source)), calibration_board=board,
    )
    evidence = _background_board_offset_plane_physical_dimension_evidence(
        _artifact(segmentationBoundaryUncertaintyPx=1.0), request, normalized, prediction,
    )
    assert evidence.state == "AVAILABLE"
    assert evidence.method == "BACKGROUND_BOARD_PNP_FRONT_OFFSET_BODY_CROSS_SECTION_V1"
    assert evidence.coordinate_space == "BACKGROUND_BOARD_FRONT_OFFSET_PLANE_MM"
    assert evidence.length_mm == pytest.approx(183.0, abs=1.6)
    assert evidence.width_mm == pytest.approx(59.0, abs=1.6)
    assert evidence.calibration is not None
    assert evidence.calibration.source == "BACKGROUND_BOARD_PNP_FRONT_OFFSET_V1"
    assert evidence.depth_offset_estimate is not None
    assert evidence.depth_offset_estimate.interval_kind == "FIXED_RIG_TOLERANCE"
    assert evidence.uncertainty is not None
    assert evidence.uncertainty.interval_kind == "FIXED_RIG_TOLERANCE"

    board_only_request = SimpleNamespace(offset_plane_calibration=SimpleNamespace(
        front_plane_offset_mm=front_offset,
        camera=SimpleNamespace(
            source="BOARD_SELF_CALIBRATED_V1",
            image_width=image_width, image_height=image_height,
            # The board-only solver deliberately ignores these placeholders.
            fx_px=1.0, fy_px=1.0, cx_px=0.0, cy_px=0.0, distortion_coefficients=[],
        ),
        min_board_pnp_inliers=8,
        max_board_pnp_reprojection_error_px=1.5,
    ))
    board_only = _background_board_offset_plane_physical_dimension_evidence(
        _artifact(segmentationBoundaryUncertaintyPx=1.0), board_only_request, normalized, prediction,
    )
    assert board_only.state == "AVAILABLE"
    assert board_only.length_mm == pytest.approx(183.0, abs=3.0)
    assert board_only.width_mm == pytest.approx(59.0, abs=2.0)
    assert board_only.calibration is not None
    assert board_only.calibration.source == "BACKGROUND_BOARD_PNP_SELF_CALIBRATED_INTRINSICS_V1"


def test_robust_body_cross_section_ignores_an_end_attachment_but_keeps_the_body_width():
    # A 183 x 59 mm enclosure with a 12 mm-wide attachment at its last 8 mm
    # has a much wider global min-area rectangle.  The attachment is not the
    # repeated enclosure cross section an operator calls the product width.
    body_x, body_y = np.meshgrid(np.linspace(0, 183, 92), np.linspace(-29.5, 29.5, 30))
    body = np.column_stack((body_x.ravel(), body_y.ravel()))
    attachment_x, attachment_y = np.meshgrid(np.linspace(175, 183, 8), np.linspace(-41.5, 41.5, 42))
    attachment = np.column_stack((attachment_x.ravel(), attachment_y.ravel()))
    all_points = np.concatenate((body, attachment), axis=0)
    width = _robust_body_width_from_projected_mask_points(
        all_points, all_points, fallback_width_mm=83.0,
    )
    assert width == pytest.approx(59.0, abs=0.5)


def test_source_metric_mask_cleanup_rejects_detached_board_specks():
    # The MobileSAM box prompt can produce isolated white islands on a marker
    # board.  They must not become the long side of the projected object.
    golden = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(golden, (35, 20), (125, 100), 255, thickness=-1)
    ok, encoded = cv2.imencode(".png", golden)
    assert ok
    mask = np.zeros_like(golden)
    cv2.rectangle(mask, (38, 23), (122, 97), 255, thickness=-1)
    cv2.rectangle(mask, (145, 4), (157, 28), 255, thickness=-1)
    artifact = SimpleNamespace(subject_segmentation=SimpleNamespace(
        golden_masks=(SimpleNamespace(mask_png_base64=base64.b64encode(encoded).decode("ascii")),),
        support_padding_px=0,
    ))
    normalized = SimpleNamespace(target_from_source=np.eye(3, dtype=np.float64))
    cleaned = _clean_source_metric_mask(artifact, normalized, mask)
    assert int(np.count_nonzero(cleaned)) == int(np.count_nonzero(mask[20:101, 35:126]))
    assert np.count_nonzero(cleaned[0:30, 140:160]) == 0


def test_background_board_relative_depth_refines_the_non_coplanar_offset_and_reports_a_range():
    image_width, image_height = 1280, 960
    camera_matrix = np.asarray(((1180.0, 0.0, 640.0), (0.0, 1160.0, 480.0), (0.0, 0.0, 1.0)))
    rvec = np.asarray((0.10, -0.16, 0.025), dtype=np.float64)
    tvec = np.asarray((-92.0, -76.0, 760.0), dtype=np.float64)
    outer_positions = ((6, 6), (178, 6), (6, 107), (178, 107), (6, 208), (178, 208))
    outer_markers = [
        {"id": marker_id, "cornersMm": [[x, y, 0], [x + 16, y, 0], [x + 16, y + 16, 0], [x, y + 16, 0]]}
        for marker_id, (x, y) in enumerate(outer_positions)
    ]
    board = GoldenDimensionBoardCandidate.model_validate({
        "boardId": "A4-METRIC", "revision": 1, "profile": "A4_METRIC_200X230_V1",
        "manifestSha256": "sha256:" + "a" * 64, "dictionary": "DICT_5X5_1000",
        "squaresX": 7, "squaresY": 9, "squareLengthMm": 20.0, "markerLengthMm": 14.0,
        "markerIds": list(range(100, 131)), "finishedWidthMm": 200.0, "finishedHeightMm": 230.0,
        "charucoOriginMm": [30.0, 25.0, 0.0], "outerMarkers": outer_markers,
        "charucoGeometryQualified": True, "outerArucoGeometryQualified": True,
    })
    source = np.full((image_height, image_width), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    marker_source = np.asarray(((0, 0), (160, 0), (160, 160), (0, 160)), dtype=np.float32)
    for marker in board.outer_markers:
        image_marker = cv2.aruco.generateImageMarker(dictionary, marker.id, 160, borderBits=1)
        projected, _ = cv2.projectPoints(np.asarray(marker.corners_mm, dtype=np.float64), rvec, tvec, camera_matrix, None)
        transform = cv2.getPerspectiveTransform(marker_source, projected.reshape(4, 2).astype(np.float32))
        rendered = cv2.warpPerspective(image_marker, transform, (image_width, image_height), borderValue=255)
        rendered_mask = cv2.warpPerspective(np.full_like(image_marker, 255), transform, (image_width, image_height))
        source[rendered_mask > 0] = rendered[rendered_mask > 0]

    front_offset = 34.0
    subject_mm = np.asarray(((9, 21, -front_offset), (192, 21, -front_offset), (192, 80, -front_offset), (9, 80, -front_offset)), dtype=np.float64)
    subject_pixels, _ = cv2.projectPoints(subject_mm, rvec, tvec, camera_matrix, None)
    mask = np.zeros_like(source)
    cv2.fillConvexPoly(mask, np.int32(np.round(subject_pixels.reshape(-1, 2))), 255)
    source[mask > 0] = 96
    prediction = SubjectMaskPrediction(
        mask=mask, quality_score=0.99, prompt_box_xyxy=(0, 0, image_width, image_height),
        foreground_ratio=float(np.count_nonzero(mask)) / mask.size,
    )

    class PerfectRelativeDepth:
        def estimate_inverse_depth(self, _rgb):
            rows, columns = np.indices((image_height, image_width), dtype=np.float64)
            rays = np.stack(((columns - 640.0) / 1180.0, (rows - 480.0) / 1160.0, np.ones_like(rows)), axis=-1)
            rotation, _ = cv2.Rodrigues(rvec)
            normal = rotation[:, 2]
            board_distance = float(normal @ tvec)
            z = board_distance / np.einsum("...i,i->...", rays, normal)
            z[mask > 0] = (board_distance - front_offset) / np.einsum("...i,i->...", rays[mask > 0], normal)
            return (1.0 / z - 0.0002) / 0.002

    request = SimpleNamespace(offset_plane_calibration=SimpleNamespace(
        front_plane_offset_mm=20.0,
        camera=SimpleNamespace(
            image_width=image_width, image_height=image_height,
            fx_px=1180.0, fy_px=1160.0, cx_px=640.0, cy_px=480.0,
            distortion_coefficients=[],
        ),
        min_board_pnp_inliers=8,
        max_board_pnp_reprojection_error_px=1.5,
        depth_estimate_policy=SimpleNamespace(
            relative_depth_enabled=True, lower95_mm=0.0, upper95_mm=70.0, model_systematic_error_mm=2.0,
        ),
    ))
    normalized = NormalizedCapture(
        rgb=np.dstack((source, source, source)), encoded=b"png", alignment=_alignment(),
        source_rgb=np.dstack((source, source, source)), calibration_board=board,
    )

    evidence = _background_board_offset_plane_physical_dimension_evidence(
        _artifact(segmentationBoundaryUncertaintyPx=1.0), request, normalized, prediction, PerfectRelativeDepth(),
    )

    assert evidence.state == "AVAILABLE"
    assert evidence.length_mm == pytest.approx(183.0, abs=1.6)
    assert evidence.width_mm == pytest.approx(59.0, abs=1.6)
    assert evidence.depth_offset_estimate is not None
    assert evidence.depth_offset_estimate.source == "RELATIVE_DEPTH_BOARD_CALIBRATED_V1"
    assert evidence.depth_offset_estimate.interval_kind == "MODEL_UNVALIDATED_INTERVAL"
    assert evidence.depth_offset_estimate.offset_mm == pytest.approx(front_offset, abs=3.2)
    assert evidence.uncertainty is not None
    assert evidence.uncertainty.interval_kind == "MODEL_UNVALIDATED_INTERVAL"
    assert evidence.uncertainty.length_lower95_mm is not None
    assert evidence.uncertainty.length_upper95_mm is not None
    assert evidence.uncertainty.length_lower95_mm <= 183.0 <= evidence.uncertainty.length_upper95_mm

    class LayeredRelativeDepth(PerfectRelativeDepth):
        def estimate_inverse_depth(self, _rgb):
            rows, columns = np.indices((image_height, image_width), dtype=np.float64)
            rays = np.stack(((columns - 640.0) / 1180.0, (rows - 480.0) / 1160.0, np.ones_like(rows)), axis=-1)
            rotation, _ = cv2.Rodrigues(rvec)
            normal = rotation[:, 2]
            board_distance = float(normal @ tvec)
            z = board_distance / np.einsum("...i,i->...", rays, normal)
            foreground = mask > 0
            front = foreground & (columns < np.percentile(columns[foreground], 60))
            rear = foreground & ~front
            z[front] = (board_distance - front_offset) / np.einsum("...i,i->...", rays[front], normal)
            z[rear] = (board_distance - 64.0) / np.einsum("...i,i->...", rays[rear], normal)
            return (1.0 / z - 0.0002) / 0.002

    request.offset_plane_calibration.depth_estimate_policy.dominant_plane_enabled = True
    request.offset_plane_calibration.depth_estimate_policy.dominant_plane_half_width_mm = 4.0
    request.offset_plane_calibration.depth_estimate_policy.minimum_dominant_plane_support_ratio = 0.35
    layered = _background_board_offset_plane_physical_dimension_evidence(
        _artifact(segmentationBoundaryUncertaintyPx=1.0), request, normalized, prediction, LayeredRelativeDepth(),
    )
    assert layered.depth_offset_estimate is not None
    assert layered.depth_offset_estimate.source == "RELATIVE_DEPTH_BOARD_CALIBRATED_V1"
    assert layered.depth_offset_estimate.offset_mm == pytest.approx(front_offset, abs=3.2)
    assert layered.depth_offset_estimate.subject_spread_p95_mm is not None
    assert layered.depth_offset_estimate.subject_spread_p95_mm < 1.0

    # A learned posterior that spills beyond the surveyed scenario envelope is
    # retained as diagnostic evidence only; it cannot widen the displayed
    # board/rig interval before physical model qualification.
    request.offset_plane_calibration.depth_estimate_policy.model_systematic_error_mm = 100.0
    fallback = _background_board_offset_plane_physical_dimension_evidence(
        _artifact(segmentationBoundaryUncertaintyPx=1.0), request, normalized, prediction, PerfectRelativeDepth(),
    )
    assert fallback.depth_offset_estimate is not None
    assert fallback.depth_offset_estimate.source == "BOARD_POSE_UNCALIBRATED_PRIOR_V1"
    assert fallback.depth_offset_estimate.offset_mm == pytest.approx(20.0)
    assert fallback.depth_offset_estimate.lower95_mm == pytest.approx(0.0)
    assert fallback.depth_offset_estimate.upper95_mm == pytest.approx(70.0)
    assert fallback.uncertainty is not None
    assert fallback.uncertainty.interval_kind == "UNCALIBRATED_SCENARIO_ENVELOPE"


def test_background_offset_plane_refuses_missing_background_board_calibration():
    request = SimpleNamespace(offset_plane_calibration=SimpleNamespace(
        front_plane_offset_mm=10.0,
        camera=SimpleNamespace(
            image_width=401, image_height=300, fx_px=500.0, fy_px=500.0, cx_px=200.0, cy_px=150.0,
            distortion_coefficients=[],
        ),
        min_board_pnp_inliers=8,
        max_board_pnp_reprojection_error_px=1.5,
    ))
    normalized = replace(_normalized(), source_rgb=np.zeros((300, 400, 3), dtype=np.uint8))
    evidence = _background_board_offset_plane_physical_dimension_evidence(_artifact(), request, normalized, _prediction())
    assert evidence.state == "UNAVAILABLE"
    assert evidence.reason_code == "BACKGROUND_BOARD_CALIBRATION_REQUIRED"


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
    assert normalized.metric_calibration.calibration_fiducial == "OUTER_ARUCO_CORNERS"
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
