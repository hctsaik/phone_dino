from __future__ import annotations

import hashlib
import base64
import json
from dataclasses import replace

from fastapi.testclient import TestClient

from phone_dino.app import create_app
from phone_dino.analyzer import RUNTIME_DIGEST
from phone_dino.config import Settings
from phone_dino.production import NormalizedCapture, ProductionAnalyzer
from phone_dino.contracts import (
    AlignmentObservation,
    DimensionUncertaintyEvidence,
    GoldenDimensionObservation,
    MetricCalibrationEvidence,
    PhysicalDimensionEvidence,
)
from phone_dino.security import digest_directory


_GOLDEN_MASK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def settings(tmp_path, *, enabled: bool = True, token: str | None = "secret", max_bytes: int = 1024 * 1024) -> Settings:
    return Settings(
        service_token=token,
        fixture_enabled=enabled,
        fixture_dir=tmp_path,
        artifact_manifest=None,
        artifact_package_digest=None,
        model_repo=None,
        model_weights=None,
        max_image_bytes=max_bytes,
        max_image_pixels=10000,
        max_image_width=100,
        max_image_height=100,
    )


def post(client, manifest, image, *, token="secret"):
    return client.post(
        "/internal/v1/analyze",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "manifest": ("manifest.json", json.dumps(manifest).encode(), "application/json"),
            "image": ("capture.png", image, "image/png"),
        },
    )


def post_golden(client, manifest, image, *, token="secret"):
    return client.post(
        "/internal/v1/golden-dimension-assessments",
        headers={"Authorization": f"Bearer {token}"},
        files={
            "manifest": ("manifest.json", json.dumps(manifest).encode(), "application/json"),
            "image": ("golden.png", image, "image/png"),
        },
    )


class GoldenDimensionAnalyzer:
    def measure_golden_dimensions(self, request, _image):
        return GoldenDimensionObservation(
            schemaVersion="1.0",
            requestId=request.request_id,
            recipeId=request.recipe_id,
            rawSha256=request.raw_sha256,
            artifactPackageDigest="sha256:" + "a" * 64,
            analyzerRuntimeVersion=RUNTIME_DIGEST,
            calibrationBoardProfile="COMPACT_130X90_V1",
            measurementPlane=request.measurement_plane,
            physicalDimensions=PhysicalDimensionEvidence(
                state="AVAILABLE",
                disclaimerCode="ENGINEERING_DIMENSION_NOT_METROLOGY_PROOF",
                method="CHARUCO_PLANE_GOLDEN_MASK_MIN_AREA_RECT_V1",
                approvalState="ENGINEERING_AUTO",
                coordinateSpace="CHARUCO_BOARD_PLANE_MM",
                currentSubjectMaskSha256=hashlib.sha256(_GOLDEN_MASK_PNG).hexdigest(),
                lengthMm=20.0,
                widthMm=10.0,
                areaMm2=200.0,
                rotatedRectAngleDegrees=0.0,
                calibration=MetricCalibrationEvidence(
                    source="CHARUCO_BOARD_PLANE_V1",
                    detectedCornerCount=18,
                    inlierCornerCount=17,
                    planeReprojectionErrorPx=0.4,
                    pixelsPerMmX=10.0,
                    pixelsPerMmY=10.0,
                ),
                uncertainty=DimensionUncertaintyEvidence(
                    method="CONSERVATIVE_CALIBRATION_PLUS_SEGMENTATION_V1",
                    linearMm=0.5,
                    areaMm2=15.0,
                    relativeLinear=0.025,
                ),
            ),
            subjectMaskPngBase64=base64.b64encode(_GOLDEN_MASK_PNG).decode("ascii"),
        )


def test_health_and_fixture_readiness(tmp_path):
    client = TestClient(create_app(settings(tmp_path)))
    assert client.get("/healthz").status_code == 200
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready", "simulation": True, "analysisMode": "FIXTURE",
        "supportedSchemas": ["1.0"], "capabilities": [],
    }


def test_golden_dimension_endpoint_returns_observation_without_decision(tmp_path, png_bytes):
    raw_sha = hashlib.sha256(png_bytes).hexdigest()
    manifest = {
        "schemaVersion": "1.0",
        "requestId": "golden-request-1",
        "recipeId": "PM-ABC-001",
        "rawSha256": raw_sha,
        "contentType": "image/png",
        "promptRegionNormalized": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
        "measurementPlane": "TOP",
    }
    client = TestClient(create_app(settings(tmp_path), production_analyzer=GoldenDimensionAnalyzer()))

    response = post_golden(client, manifest, png_bytes)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rawSha256"] == raw_sha
    assert body["measurementPlane"] == "TOP"
    assert body["physicalDimensions"]["lengthMm"] == 20.0
    assert body["physicalDimensions"]["method"] == "CHARUCO_PLANE_GOLDEN_MASK_MIN_AREA_RECT_V1"
    assert base64.b64decode(body["subjectMaskPngBase64"]) == _GOLDEN_MASK_PNG
    assert "decision" not in body and "manufacturingAction" not in body


def test_golden_dimension_endpoint_rejects_image_digest_mismatch(tmp_path, png_bytes):
    manifest = {
        "schemaVersion": "1.0",
        "requestId": "golden-request-2",
        "recipeId": "PM-ABC-001",
        "rawSha256": "0" * 64,
        "contentType": "image/png",
        "promptRegionNormalized": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
        "measurementPlane": "FRONT",
    }
    client = TestClient(create_app(settings(tmp_path), production_analyzer=GoldenDimensionAnalyzer()))

    response = post_golden(client, manifest, png_bytes)

    assert response.status_code == 409
    assert response.json()["detail"] == "RAW_SHA256_MISMATCH"


def test_analyze_fixture_returns_observation_not_decision(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    assert manifest["executionBundleDigest"] != manifest["artifactPackageDigest"]
    raw_sha = manifest["rawSha256"]
    (tmp_path / f"{raw_sha}.json").write_text(
        json.dumps({"outcome": "pass", "globalDistance": 0.12, "uncertainty": 0.03}), encoding="utf-8"
    )
    response = post(TestClient(create_app(settings(tmp_path))), manifest, png_bytes)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["simulation"] is True
    assert body["analysis"]["state"] == "RUN"
    assert body["analysis"]["globalDistance"] == 0.12
    assert body["resolvedVersions"]["executionBundleDigest"] == manifest["executionBundleDigest"]
    assert body["resolvedVersions"]["artifactPackageDigest"] == manifest["artifactPackageDigest"]
    assert "manufacturingAction" not in body
    assert "decision" not in body


def test_recapture_does_not_run_analysis(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    (tmp_path / f"{manifest['rawSha256']}.json").write_text(
        json.dumps({"outcome": "recapture", "reasonCodes": ["BLUR"]}), encoding="utf-8"
    )
    response = post(TestClient(create_app(settings(tmp_path))), manifest, png_bytes)
    assert response.status_code == 200
    body = response.json()
    assert body["captureAssessment"]["state"] == "RECAPTURE_REQUIRED"
    assert body["analysis"]["state"] == "NOT_RUN"
    assert "normalization" not in body
    assert "globalDistance" not in body["analysis"]


def test_system_error_fixture_returns_503(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    (tmp_path / f"{manifest['rawSha256']}.json").write_text(json.dumps({"outcome": "system-error"}), encoding="utf-8")
    assert post(TestClient(create_app(settings(tmp_path))), manifest, png_bytes).status_code == 503


def test_auth_is_required(tmp_path, png_bytes, manifest_factory):
    response = post(TestClient(create_app(settings(tmp_path))), manifest_factory(png_bytes), png_bytes, token="wrong")
    assert response.status_code == 401


def test_raw_digest_mismatch_is_conflict(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    manifest["rawSha256"] = "0" * 64
    response = post(TestClient(create_app(settings(tmp_path))), manifest, png_bytes)
    assert response.status_code == 409
    assert response.json()["detail"] == "RAW_SHA256_MISMATCH"


def test_bundle_digest_mismatch_is_conflict(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    manifest["executionBundleDigest"] = "sha256:" + "0" * 64
    response = post(TestClient(create_app(settings(tmp_path))), manifest, png_bytes)
    assert response.status_code == 409


def test_unknown_manifest_field_is_rejected(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    manifest["clientThreshold"] = 0.5
    response = post(TestClient(create_app(settings(tmp_path))), manifest, png_bytes)
    assert response.status_code == 422


def test_oversized_upload_is_rejected(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    response = post(TestClient(create_app(settings(tmp_path, max_bytes=8))), manifest, png_bytes)
    assert response.status_code == 413


def test_non_fixture_mode_fails_closed_without_artifact(tmp_path, png_bytes, manifest_factory):
    client = TestClient(create_app(settings(tmp_path, enabled=False)))
    assert client.get("/readyz").status_code == 503
    manifest = manifest_factory(png_bytes)
    manifest["simulation"] = False
    response = post(client, manifest, png_bytes)
    assert response.status_code == 503
    assert response.json()["detail"] == "ARTIFACT_MANIFEST_NOT_AVAILABLE"


def test_production_request_must_match_independent_artifact_pin(tmp_path, png_bytes, manifest_factory):
    artifact = tmp_path / "artifact-manifest.json"
    artifact.write_bytes(b"immutable artifact package manifest")
    configured_digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    configured = replace(
        settings(tmp_path, enabled=False),
        artifact_manifest=artifact,
        artifact_package_digest=configured_digest,
    )
    manifest = manifest_factory(png_bytes)
    manifest["simulation"] = False
    assert manifest["artifactPackageDigest"] != configured_digest

    response = post(TestClient(create_app(configured)), manifest, png_bytes)

    assert response.status_code == 409
    assert response.json()["detail"] == "ARTIFACT_PACKAGE_DIGEST_MISMATCH"


def test_fixture_lookup_uses_actual_image_hash(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    wrong = hashlib.sha256(b"different").hexdigest()
    (tmp_path / f"{wrong}.json").write_text(json.dumps({"outcome": "pass", "globalDistance": 0.1}), encoding="utf-8")
    response = post(TestClient(create_app(settings(tmp_path))), manifest, png_bytes)
    assert response.status_code == 503
    assert response.json()["detail"] == "ENGINEERING_FIXTURE_NOT_AVAILABLE"


def test_simulation_mode_must_match_engine(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    manifest["simulation"] = False
    response = post(TestClient(create_app(settings(tmp_path))), manifest, png_bytes)
    assert response.status_code == 409
    assert response.json()["detail"] == "SIMULATION_MODE_MISMATCH"


def test_corrupt_image_is_rejected(tmp_path, manifest_factory):
    corrupt = b"not a png"
    manifest = manifest_factory(corrupt)
    response = post(TestClient(create_app(settings(tmp_path))), manifest, corrupt)
    assert response.status_code == 400
    assert response.json()["detail"] == "IMAGE_DECODE_FAILED"


def test_non_finite_fixture_value_fails_closed(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    (tmp_path / f"{manifest['rawSha256']}.json").write_text(
        '{"outcome":"pass","globalDistance":NaN}', encoding="utf-8"
    )
    response = post(TestClient(create_app(settings(tmp_path))), manifest, png_bytes)
    assert response.status_code == 503
    assert response.json()["detail"] == "ENGINEERING_FIXTURE_NOT_AVAILABLE"


def test_expired_deadline_is_rejected(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    manifest["deadline"] = "2020-01-01T00:00:00Z"
    response = post(TestClient(create_app(settings(tmp_path))), manifest, png_bytes)
    assert response.status_code == 408
    assert response.json()["detail"] == "ANALYSIS_DEADLINE_EXPIRED"


def test_manifest_requires_json_part(tmp_path, png_bytes, manifest_factory):
    client = TestClient(create_app(settings(tmp_path)))
    response = client.post(
        "/internal/v1/analyze",
        headers={"Authorization": "Bearer secret"},
        files={
            "manifest": ("manifest.txt", json.dumps(manifest_factory(png_bytes)).encode(), "text/plain"),
            "image": ("capture.png", png_bytes, "image/png"),
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "MANIFEST_CONTENT_TYPE_INVALID"


class AcceptedNormalizer:
    def normalize(self, image, artifact):
        return NormalizedCapture(
            rgb="canonical", encoded=b"canonical-png",
            alignment=AlignmentObservation(
                state="ALIGNED", method="TARGET_AFFINE", targetRelative=True, inlierCount=20,
                inlierRatio=0.8, reprojectionErrorPx=0.5, coverageRatio=0.4,
                transformWithinBounds=True, inspectionMaskApplied=True,
            ),
        )


class RecaptureNormalizer:
    def normalize(self, image, artifact):
        return NormalizedCapture(rgb=None, encoded=b"", reason_codes=("BLUR",))


class MissingTargetAlignmentNormalizer:
    def normalize(self, image, artifact):
        return NormalizedCapture(rgb="unsafe-board-canvas", encoded=b"unsafe-board-canvas")


class DeterministicEmbedder:
    def embed(self, rgb):
        assert rgb == "canonical"
        return [1.0, 0.0, 0.0]


def production_setup(tmp_path, manifest, *, normalizer=None, engineering: bool = False):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"reviewed-model-weights")
    repo = tmp_path / "dinov2"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# reviewed local repository\n", encoding="utf-8")
    component = manifest["executionBundle"]["goldenSetVersion"]
    repository_version = digest_directory(repo)
    artifact_body = {
        "schemaVersion": "1.1",
        "recipeId": manifest["recipeId"],
        "machineId": manifest["machineId"],
        "boardId": manifest["boardId"],
        "goldenSetVersion": component,
        "normalizationPipelineVersion": manifest["executionBundle"]["normalizationPipelineVersion"],
        "analyzerModelVersion": manifest["executionBundle"]["analyzerModelVersion"],
        "decisionPolicyVersion": manifest["executionBundle"]["decisionPolicyVersion"],
        "analyzerRuntimeVersion": RUNTIME_DIGEST,
        "modelRepositoryVersion": repository_version,
        "boardInstallationVersion": manifest["executionBundle"]["boardInstallationVersion"],
        "modelWeightsSha256": "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest(),
        "board": {
            "squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
            "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896,
        },
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
        "targetAlignment": target_alignment(png_bytes_for_reference()),
        "goldenEmbeddings": [
            {"id": "golden-near", "sourceSha256": component, "values": [0.9, 0.1, 0.0]},
            {"id": "golden-far", "sourceSha256": component, "values": [0.0, 1.0, 0.0]},
        ],
    }
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(artifact_body, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    artifact_digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest["artifactPackageDigest"] = artifact_digest
    manifest["simulation"] = engineering
    configured = replace(
        settings(tmp_path, enabled=False), artifact_manifest=artifact,
        artifact_package_digest=artifact_digest, model_repo=repo, model_weights=weights,
        model_repository_version=repository_version, engineering_real_model_enabled=engineering,
    )
    analyzer = ProductionAnalyzer(
        configured, normalizer=normalizer or AcceptedNormalizer(), embedder=DeterministicEmbedder()
    )
    return TestClient(create_app(configured, production_analyzer=analyzer)), artifact_body


def png_bytes_for_reference():
    # Readiness only validates the hash for injected test normalizers.
    return b"test-target-reference"


def target_alignment(reference):
    return {
        "method": "TARGET_AFFINE", "referenceImageBase64": base64.b64encode(reference).decode(),
        "referenceImageSha256": "sha256:" + hashlib.sha256(reference).hexdigest(),
        "canonicalWidth": 320, "canonicalHeight": 240,
        "alignmentRegions": [{"x": 0, "y": 0, "width": 320, "height": 80}],
        "heldOutRegions": [{"x": 0, "y": 200, "width": 320, "height": 40}],
        "inspectionRegions": [{"x": 100, "y": 100, "width": 100, "height": 80}],
        "minMatches": 8, "minInliers": 6, "minInlierRatio": 0.5, "minCoverageRatio": 0.02,
        "maxReprojectionErrorPx": 3.0, "minScale": 0.8, "maxScale": 1.2,
        "maxRotationDegrees": 15.0, "maxShear": 0.05, "maxTranslationPx": 300.0,
        "maxSecondaryInlierRatio": 0.35, "minHeldOutMatches": 4, "maxHeldOutReprojectionErrorPx": 3.0,
    }


def test_production_analyzer_returns_version_bound_observation(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    client, _ = production_setup(tmp_path, manifest)
    assert client.get("/readyz").json() == {
        "status": "ready", "simulation": False, "analysisMode": "PRODUCTION",
        "supportedSchemas": ["1.0", "1.1", "1.2", "1.3"], "capabilities": [],
    }

    response = post(client, manifest, png_bytes)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["simulation"] is False
    assert body["captureAssessment"] == {"state": "ACCEPTED", "reasonCodes": []}
    assert body["analysis"]["state"] == "RUN"
    assert body["analysis"]["metric"] == "cosine_distance"
    assert body["analysis"]["nearestGoldenId"] == "golden-near"
    assert 0 <= body["analysis"]["nearestGoldenDistance"] < body["analysis"]["globalDistance"] <= 2
    assert "decision" not in body and "manufacturingAction" not in body


def test_engineering_real_mode_runs_production_analyzer_with_simulation_identity(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    client, _ = production_setup(tmp_path, manifest, engineering=True)
    assert client.get("/readyz").json() == {
        "status": "ready", "simulation": True, "analysisMode": "ENGINEERING_REAL_DINO",
        "supportedSchemas": ["1.0", "1.1", "1.2", "1.3"], "capabilities": [],
    }

    response = post(client, manifest, png_bytes)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["simulation"] is True
    assert body["analysis"]["state"] == "RUN"
    assert body["analysis"]["nearestGoldenId"] == "golden-near"


def test_engineering_real_mode_still_enforces_pinned_artifact_digest(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    client, _ = production_setup(tmp_path, manifest, engineering=True)
    manifest["artifactPackageDigest"] = "sha256:" + "f" * 64

    response = post(client, manifest, png_bytes)

    assert response.status_code == 409
    assert response.json()["detail"] == "ARTIFACT_PACKAGE_DIGEST_MISMATCH"


def test_production_still_gate_recapture_never_embeds(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    client, _ = production_setup(tmp_path, manifest, normalizer=RecaptureNormalizer())

    response = post(client, manifest, png_bytes)

    assert response.status_code == 200
    body = response.json()
    assert body["captureAssessment"] == {"state": "RECAPTURE_REQUIRED", "reasonCodes": ["BLUR"]}
    assert body["analysis"] == {"state": "NOT_RUN"}
    assert "normalization" not in body


def test_production_never_embeds_board_canvas_without_target_alignment(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    client, _ = production_setup(tmp_path, manifest, normalizer=MissingTargetAlignmentNormalizer())

    response = post(client, manifest, png_bytes)

    assert response.status_code == 200
    body = response.json()
    assert body["captureAssessment"] == {
        "state": "RECAPTURE_REQUIRED", "reasonCodes": ["TARGET_ALIGNMENT_REQUIRED"],
    }
    assert body["analysis"] == {"state": "NOT_RUN"}
    assert "normalization" not in body


def test_production_artifact_bundle_mismatch_is_conflict(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    client, _ = production_setup(tmp_path, manifest)
    manifest["recipeId"] = "different-recipe"

    response = post(client, manifest, png_bytes)

    assert response.status_code == 409
    assert response.json()["detail"] == "ARTIFACT_EXECUTION_BUNDLE_MISMATCH"


def test_production_weights_digest_mismatch_is_not_ready(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    client, _ = production_setup(tmp_path, manifest)
    configured = client.app.state.settings
    configured.model_weights.write_bytes(b"tampered")

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["detail"]["reason"] == "MODEL_WEIGHTS_DIGEST_MISMATCH"
