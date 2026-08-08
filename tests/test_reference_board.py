from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from phone_dino.analyzer import RUNTIME_DIGEST
from phone_dino.artifacts import CharucoBoard, ProductionArtifactV19, ReferenceBoardPolicy, StillGate
from phone_dino.contracts import AnalyzeRequest, ReferenceBoardEvidence
from phone_dino.decoder import DecodedImage
from phone_dino.production import ProductionAnalyzer
from phone_dino.reference_board import ReferenceBoardVerifier, qr_to_charuco_residual_mm


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def policy(payload: str = "reference-tag-v1") -> ReferenceBoardPolicy:
    document = {
        "schemaVersion": "1.0",
        "metricScaleSource": "CHARUCO_ONLY",
        "tagUidSha256": "1" * 64,
        "boardSerialSha256": "2" * 64,
        "templateManifestSha256": "sha256:" + "3" * 64,
        "installationDigest": "sha256:" + "4" * 64,
        "qrPayloadSha256": hashlib.sha256(payload.encode()).hexdigest(),
        "qrSymbolBoundsMm": {"x": 53.0, "y": 61.0, "width": 24.0, "height": 24.0},
        "charucoOriginMm": [30.0, 5.0, 0.0],
        "minQrSidePx": 24.0,
        "minCharucoCorners": 6,
        "maxQrToCharucoResidualMm": 0.75,
    }
    return ReferenceBoardPolicy.model_validate({
        **document,
        "digest": digest(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()),
    })


def test_qr_corner_residual_ignores_detector_corner_rotation_but_not_translation():
    expected = __import__("numpy").asarray([[53, 61], [77, 61], [77, 85], [53, 85]], dtype="float32")
    assert qr_to_charuco_residual_mm(expected[[2, 3, 0, 1]], expected) == pytest.approx(0.0)
    assert qr_to_charuco_residual_mm(expected + [3.0, 0.0], expected) == pytest.approx(3.0)


def test_reference_gate_detects_charuco_and_rejects_qr_moved_outside_pinned_bounds():
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    if not hasattr(cv2, "aruco"):
        pytest.skip("OpenCV ArUco support is not installed")
    reference_policy = policy()
    marker_ids = numpy.arange(100, 117, dtype=numpy.int32)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    source_board = cv2.aruco.CharucoBoard((7, 5), 10.0, 7.0, dictionary, marker_ids)
    image = source_board.generateImage((700, 500), marginSize=24, borderBits=1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    artifact = SimpleNamespace(
        board=CharucoBoard.model_validate({
            "profileId": "COMPACT_130X90_V1", "squaresX": 7, "squaresY": 5,
            "squareLengthMm": 10.0, "markerLengthMm": 7.0, "markerIds": marker_ids.tolist(),
            "dictionary": "DICT_5X5_1000", "canonicalWidth": 896, "canonicalHeight": 896,
        }),
        still_gate=StillGate.model_validate({
            "minCharucoCorners": 6, "minLaplacianVariance": 1.0, "maxOverExposureRatio": 0.95,
        }),
        reference_board=reference_policy,
    )
    decoded = DecodedImage(data=encoded.tobytes(), width=700, height=500, format="PNG", elapsed_ms=0)
    plane = ReferenceBoardVerifier._charuco_plane(cv2.imdecode(encoded, cv2.IMREAD_COLOR), artifact)
    assert plane is not None
    bounds = reference_policy.qr_symbol_bounds_mm
    expected_mm = numpy.asarray([
        [bounds.x, bounds.y], [bounds.x + bounds.width, bounds.y],
        [bounds.x + bounds.width, bounds.y + bounds.height], [bounds.x, bounds.y + bounds.height],
    ], dtype=numpy.float32)
    image_points = cv2.perspectiveTransform(
        expected_mm.reshape(1, 4, 2), numpy.linalg.inv(plane.homography_image_to_mm),
    ).reshape(4, 2)
    verifier = ReferenceBoardVerifier(qr_decoder=lambda _source: ("reference-tag-v1", image_points))
    verified = verifier.verify(decoded, artifact)
    assert verified.state == "VERIFIED"
    assert verified.metric_scale_source == "CHARUCO_ONLY"
    assert verified.qr_to_charuco_residual_mm == pytest.approx(0.0, abs=0.02)

    moved = ReferenceBoardVerifier(qr_decoder=lambda _source: ("reference-tag-v1", image_points + [80.0, 0.0]))
    rejected = moved.verify(decoded, artifact)
    assert rejected.state == "REJECTED"
    assert rejected.reason_codes == ["REFERENCE_QR_CHARUCO_COLOCATION_FAILED"]


def _v19_request(reference_policy: ReferenceBoardPolicy) -> AnalyzeRequest:
    version = "sha256:" + "a" * 64
    return AnalyzeRequest.model_validate({
        "schemaVersion": "1.6", "requestId": "reference-gate-1", "sessionId": "reference-session-1",
        "captureOrdinal": 1, "correlationId": "reference-correlation-1",
        "deadline": datetime(2099, 1, 1, tzinfo=timezone.utc), "rawSha256": "b" * 64,
        "contentType": "image/png", "recipeId": "recipe-1", "machineId": "machine-1", "boardId": "board-1",
        "inspectionIntent": "PM_SIMILARITY", "executionBundleDigest": version,
        "executionBundle": {
            "recipeVersion": version, "goldenSetVersion": version, "capturePolicyVersion": version,
            "decisionPolicyVersion": version, "normalizationPipelineVersion": version,
            "analyzerModelVersion": version, "clientAssetVersion": version, "boardInstallationVersion": version,
        },
        "artifactPackageDigest": version, "recipeAnalysisProfileDigest": version,
        "referenceBoard": {
            "schemaVersion": "1.0", "tagUidSha256": reference_policy.tag_uid_sha256,
            "boardSerialSha256": reference_policy.board_serial_sha256,
            "templateManifestSha256": reference_policy.template_manifest_sha256,
            "installationDigest": reference_policy.installation_digest,
            "qrPayloadSha256": reference_policy.qr_payload_sha256,
        }, "simulation": True,
    })


def test_production_analyzer_returns_before_normalization_or_embedding_when_reference_gate_rejects():
    reference_policy = policy()
    version = "sha256:" + "a" * 64
    artifact = object.__new__(ProductionArtifactV19)
    object.__setattr__(artifact, "__dict__", {
        "recipe_id": "recipe-1", "machine_id": "machine-1", "board_id": "board-1",
        "golden_set_version": version, "normalization_pipeline_version": version,
        "analyzer_model_version": version, "decision_policy_version": version,
        "board_installation_version": version, "recipe_analysis_profile": SimpleNamespace(digest=version),
        "scorer_input_contract": SimpleNamespace(digest=version), "reference_board": reference_policy,
    })
    rejected = ReferenceBoardEvidence(
        state="REJECTED", metricScaleSource="CHARUCO_ONLY", qrPayloadSha256=reference_policy.qr_payload_sha256,
        reasonCodes=["REFERENCE_QR_NOT_DETECTED"],
    )

    class RejectingVerifier:
        def verify(self, image, artifact):
            return rejected

    class ExplodingNormalizer:
        def normalize(self, image, artifact):
            raise AssertionError("normalizer must not run after reference-board rejection")

    analyzer = object.__new__(ProductionAnalyzer)
    object.__setattr__(analyzer, "_artifact", artifact)
    object.__setattr__(analyzer, "reference_board_verifier", RejectingVerifier())
    object.__setattr__(analyzer, "normalizer", ExplodingNormalizer())
    object.__setattr__(analyzer, "readiness", lambda: (True, None))
    observation = analyzer.analyze(
        _v19_request(reference_policy),
        DecodedImage(data=b"not-decoded-after-rejection", width=1, height=1, format="PNG", elapsed_ms=0),
    )
    assert observation.capture_assessment.state.value == "RECAPTURE_REQUIRED"
    assert observation.analysis.state.value == "NOT_RUN"
    assert observation.reference_board_evidence == rejected
    assert observation.reference_board_evidence.metric_scale_source == "CHARUCO_ONLY"
    assert RUNTIME_DIGEST in observation.analysis_id or len(observation.analysis_id) == 64
