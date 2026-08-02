from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from phone_dino.app import create_app
from phone_dino.artifacts import GoldenEmbedding, SpatialDifferencePolicy
from phone_dino.config import Settings
from phone_dino.contracts import AlignmentObservation
from phone_dino.production import (
    NormalizedCapture, PatchEmbedding, ProductionAnalyzer, _dino_input_crop_box, _spatial_difference_evidence,
)
from phone_dino.security import digest_directory

from .test_api import AcceptedNormalizer, png_bytes_for_reference, settings, target_alignment


class GridPatchEmbedder:
    """Deterministic small-grid embedder: fast, exact pure-function coverage."""

    def __init__(self, grid: list[list[float]], grid_size: int = 2, dim: int = 2, global_vector: list[float] | None = None):
        self._grid = grid
        self._grid_size = grid_size
        self._dim = dim
        self._global_vector = global_vector or [1.0, 0.0]

    def embed(self, rgb):
        return self._global_vector

    def embed_with_patches(self, rgb):
        return PatchEmbedding(
            global_vector=self._global_vector, patch_grid=self._grid,
            grid_height=self._grid_size, grid_width=self._grid_size,
        )


IDENTICAL_GRID = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
DIFFERENT_GRID = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
POLICY = SpatialDifferencePolicy(anomalyDistanceThreshold=0.5, minRegionAreaRatio=0.01, maxRegions=8)


def _golden(patch_values=None, grid_h=None, grid_w=None) -> GoldenEmbedding:
    kwargs = {}
    if patch_values is not None:
        kwargs = {"patchValues": patch_values, "patchGridHeight": grid_h, "patchGridWidth": grid_w}
    return GoldenEmbedding(id="G-1", sourceSha256="sha256:" + "0" * 64, values=[1.0, 0.0], **kwargs)


class TestGoldenEmbeddingPatchValidation:
    def test_patch_fields_must_be_set_together(self):
        with pytest.raises(Exception):
            GoldenEmbedding(id="G-1", sourceSha256="sha256:" + "0" * 64, values=[1.0, 0.0], patchGridHeight=2, patchGridWidth=2)

    def test_patch_grid_dimensions_must_match_declared_size(self):
        with pytest.raises(Exception):
            _golden(patch_values=IDENTICAL_GRID, grid_h=3, grid_w=3)

    def test_patch_vector_dimension_must_match_global_embedding(self):
        with pytest.raises(Exception):
            _golden(patch_values=[[1.0, 0.0, 0.0]] * 4, grid_h=2, grid_w=2)

    def test_non_finite_patch_values_rejected(self):
        with pytest.raises(Exception):
            _golden(patch_values=[[1.0, math.inf], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], grid_h=2, grid_w=2)

    def test_valid_patch_golden_accepted(self):
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        assert golden.patch_grid_height == 2 and golden.patch_grid_width == 2


class TestSpatialDifferencePolicyValidation:
    def test_threshold_out_of_range_rejected(self):
        with pytest.raises(Exception):
            SpatialDifferencePolicy(anomalyDistanceThreshold=3.0, minRegionAreaRatio=0.01, maxRegions=8)

    def test_valid_policy_accepted(self):
        policy = SpatialDifferencePolicy(anomalyDistanceThreshold=0.4, minRegionAreaRatio=0.02, maxRegions=16)
        assert policy.max_regions == 16


class TestSpatialDifferenceEvidenceFunction:
    def test_unavailable_when_no_policy_configured(self):
        embedder = GridPatchEmbedder(IDENTICAL_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        evidence = _spatial_difference_evidence(patches, golden, None, 320, 240)
        assert evidence.state == "UNAVAILABLE"
        assert evidence.reason_code == "SPATIAL_DIFFERENCE_POLICY_NOT_CONFIGURED"
        assert evidence.regions is None
        assert evidence.map_png_base64 is None

    def test_unavailable_when_golden_lacks_patch_features(self):
        embedder = GridPatchEmbedder(IDENTICAL_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden()
        evidence = _spatial_difference_evidence(patches, golden, POLICY, 320, 240)
        assert evidence.state == "UNAVAILABLE"
        assert evidence.reason_code == "GOLDEN_PATCH_FEATURES_UNAVAILABLE"

    def test_unavailable_when_grid_size_mismatches_golden(self):
        embedder = GridPatchEmbedder(IDENTICAL_GRID, grid_size=2)
        patches = embedder.embed_with_patches(None)
        golden = GoldenEmbedding(
            id="G-1", sourceSha256="sha256:" + "0" * 64, values=[1.0, 0.0],
            patchValues=[[1.0, 0.0]] * 9, patchGridHeight=3, patchGridWidth=3,
        )
        evidence = _spatial_difference_evidence(patches, golden, POLICY, 320, 240)
        assert evidence.state == "UNAVAILABLE"
        assert evidence.reason_code == "GOLDEN_PATCH_FEATURES_UNAVAILABLE"

    def test_identical_patches_produce_no_regions(self):
        embedder = GridPatchEmbedder(IDENTICAL_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        evidence = _spatial_difference_evidence(patches, golden, POLICY, 320, 240)
        assert evidence.state == "AVAILABLE"
        assert evidence.generation_method == "PATCH_DISTANCE"
        assert evidence.regions == []
        assert evidence.map_png_base64 is not None and evidence.mask_png_base64 is not None
        map_bytes = base64.b64decode(evidence.map_png_base64)
        mask_bytes = base64.b64decode(evidence.mask_png_base64)
        assert hashlib.sha256(map_bytes).hexdigest() == evidence.map_sha256
        assert hashlib.sha256(mask_bytes).hexdigest() == evidence.mask_sha256

    def test_different_patches_produce_bounded_region(self):
        embedder = GridPatchEmbedder(DIFFERENT_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        evidence = _spatial_difference_evidence(patches, golden, POLICY, 320, 240)
        assert evidence.state == "AVAILABLE"
        assert len(evidence.regions) >= 1
        for region in evidence.regions:
            box = region.bbox_normalized
            assert 0 <= box.x and 0 <= box.y
            assert box.x + box.width <= 1.0 + 1e-9
            assert box.y + box.height <= 1.0 + 1e-9
            assert 0 <= region.peak_score <= 1 and 0 <= region.mean_score <= 1

    def test_evidence_region_never_exceeds_dino_input_crop(self):
        embedder = GridPatchEmbedder(DIFFERENT_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        canonical_width, canonical_height = 320, 240
        evidence = _spatial_difference_evidence(patches, golden, POLICY, canonical_width, canonical_height)
        left, top, width, height = _dino_input_crop_box(canonical_width, canonical_height)
        region_box = evidence.evidence_region_normalized
        assert region_box.x == pytest.approx(left / canonical_width, abs=1e-6)
        assert region_box.y == pytest.approx(top / canonical_height, abs=1e-6)
        for region in evidence.regions:
            assert region.bbox_normalized.x + 1e-6 >= region_box.x
            assert region.bbox_normalized.y + 1e-6 >= region_box.y
            assert region.bbox_normalized.x + region.bbox_normalized.width <= region_box.x + region_box.width + 1e-6
            assert region.bbox_normalized.y + region.bbox_normalized.height <= region_box.y + region_box.height + 1e-6

    def test_max_regions_is_enforced(self):
        # A checkerboard-like grid maximizes distinct connected components.
        big = 8
        grid = [[1.0, 0.0] if (r + c) % 2 == 0 else [0.0, 1.0] for r in range(big) for c in range(big)]
        embedder = GridPatchEmbedder(grid, grid_size=big)
        patches = embedder.embed_with_patches(None)
        golden = GoldenEmbedding(
            id="G-1", sourceSha256="sha256:" + "0" * 64, values=[1.0, 0.0],
            patchValues=[[1.0, 0.0]] * (big * big), patchGridHeight=big, patchGridWidth=big,
        )
        tight_policy = SpatialDifferencePolicy(anomalyDistanceThreshold=0.5, minRegionAreaRatio=0.0001, maxRegions=3)
        evidence = _spatial_difference_evidence(patches, golden, tight_policy, 320, 240)
        assert evidence.state == "AVAILABLE"
        assert len(evidence.regions) <= 3


def _spatial_production_setup(tmp_path, manifest, *, spatial_policy=None, embedder=None, golden_patch=None, golden_values=None):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"reviewed-model-weights")
    repo = tmp_path / "dinov2"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# reviewed local repository\n", encoding="utf-8")
    component = manifest["executionBundle"]["goldenSetVersion"]
    repository_version = digest_directory(repo)
    resolved_values = golden_values if golden_values is not None else [0.9, 0.1, 0.0]
    golden_embedding = {"id": "golden-near", "sourceSha256": component, "values": resolved_values}
    if golden_patch is not None:
        golden_embedding.update(golden_patch)
    artifact_body = {
        "schemaVersion": "1.1",
        "recipeId": manifest["recipeId"], "machineId": manifest["machineId"], "boardId": manifest["boardId"],
        "goldenSetVersion": component,
        "normalizationPipelineVersion": manifest["executionBundle"]["normalizationPipelineVersion"],
        "analyzerModelVersion": manifest["executionBundle"]["analyzerModelVersion"],
        "decisionPolicyVersion": manifest["executionBundle"]["decisionPolicyVersion"],
        "analyzerRuntimeVersion": "sha256:" + hashlib.sha256(b"phone_dino:0.3.0").hexdigest(),
        "modelRepositoryVersion": repository_version,
        "boardInstallationVersion": manifest["executionBundle"]["boardInstallationVersion"],
        "modelWeightsSha256": "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest(),
        "board": {
            "squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
            "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896,
        },
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
        "targetAlignment": target_alignment(png_bytes_for_reference()),
        "goldenEmbeddings": [golden_embedding, {"id": "golden-far", "sourceSha256": component, "values": [0.0] * (len(resolved_values) - 1) + [1.0]}],
    }
    if spatial_policy is not None:
        artifact_body["spatialDifferencePolicy"] = spatial_policy
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(artifact_body, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    artifact_digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest["artifactPackageDigest"] = artifact_digest
    manifest["simulation"] = False
    configured = replace(
        settings(tmp_path, enabled=False), artifact_manifest=artifact,
        artifact_package_digest=artifact_digest, model_repo=repo, model_weights=weights,
        model_repository_version=repository_version,
    )
    analyzer = ProductionAnalyzer(configured, normalizer=AcceptedNormalizer(), embedder=embedder)
    return TestClient(create_app(configured, production_analyzer=analyzer)), artifact_body


def _post(client, manifest, image):
    return client.post(
        "/internal/v1/analyze", headers={"Authorization": "Bearer secret"},
        files={
            "manifest": ("manifest.json", json.dumps(manifest).encode(), "application/json"),
            "image": ("capture.png", image, "image/png"),
        },
    )


def test_end_to_end_response_omits_spatial_evidence_generation_fields_when_unavailable(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    from .test_api import DeterministicEmbedder
    client, _ = _spatial_production_setup(tmp_path, manifest, embedder=DeterministicEmbedder())
    response = _post(client, manifest, png_bytes)
    assert response.status_code == 200, response.text
    body = response.json()
    evidence = body["analysis"]["spatialDifferenceEvidence"]
    assert evidence["state"] == "UNAVAILABLE"
    assert evidence["reasonCode"] == "PATCH_EMBEDDER_NOT_AVAILABLE"
    assert "generationMethod" not in evidence
    assert "regions" not in evidence
    assert "mapPngBase64" not in evidence


def test_end_to_end_response_includes_patch_distance_evidence_when_available(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    embedder = GridPatchEmbedder(DIFFERENT_GRID, global_vector=[0.9, 0.1])
    client, _ = _spatial_production_setup(
        tmp_path, manifest,
        spatial_policy={"anomalyDistanceThreshold": 0.5, "minRegionAreaRatio": 0.01, "maxRegions": 8},
        embedder=embedder,
        golden_values=[0.9, 0.1],
        golden_patch={"patchValues": IDENTICAL_GRID, "patchGridHeight": 2, "patchGridWidth": 2},
    )
    response = _post(client, manifest, png_bytes)
    assert response.status_code == 200, response.text
    evidence = response.json()["analysis"]["spatialDifferenceEvidence"]
    assert evidence["state"] == "AVAILABLE"
    assert evidence["generationMethod"] == "PATCH_DISTANCE"
    assert evidence["disclaimerCode"] == "DIFFERENCE_NOT_DEFECT_PROOF"
    assert "reasonCode" not in evidence
    map_bytes = base64.b64decode(evidence["mapPngBase64"])
    assert hashlib.sha256(map_bytes).hexdigest() == evidence["mapSha256"]


class TestReadinessCaching:
    def test_positive_readiness_is_cached_and_avoids_rehashing(self, tmp_path, manifest_factory, png_bytes, monkeypatch):
        manifest = manifest_factory(png_bytes)
        # Build the analyzer directly for white-box access to the cache.
        weights = tmp_path / "weights2.pth"
        weights.write_bytes(b"reviewed-model-weights")
        repo = tmp_path / "dinov2-2"
        repo.mkdir()
        (repo / "hubconf.py").write_text("# reviewed\n", encoding="utf-8")
        repository_version = digest_directory(repo)
        component = manifest["executionBundle"]["goldenSetVersion"]
        artifact_body = {
            "schemaVersion": "1.1", "recipeId": manifest["recipeId"], "machineId": manifest["machineId"],
            "boardId": manifest["boardId"], "goldenSetVersion": component,
            "normalizationPipelineVersion": manifest["executionBundle"]["normalizationPipelineVersion"],
            "analyzerModelVersion": manifest["executionBundle"]["analyzerModelVersion"],
            "decisionPolicyVersion": manifest["executionBundle"]["decisionPolicyVersion"],
            "analyzerRuntimeVersion": "sha256:" + hashlib.sha256(b"phone_dino:0.3.0").hexdigest(),
            "modelRepositoryVersion": repository_version,
            "boardInstallationVersion": manifest["executionBundle"]["boardInstallationVersion"],
            "modelWeightsSha256": "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest(),
            "board": {
                "squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
                "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896,
            },
            "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
            "targetAlignment": target_alignment(png_bytes_for_reference()),
            "goldenEmbeddings": [{"id": "g", "sourceSha256": component, "values": [1.0, 0.0]}],
        }
        artifact = tmp_path / "artifact2.json"
        artifact.write_text(json.dumps(artifact_body, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        artifact_digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
        configured = replace(
            settings(tmp_path, enabled=False), artifact_manifest=artifact,
            artifact_package_digest=artifact_digest, model_repo=repo, model_weights=weights,
            model_repository_version=repository_version,
        )
        from .test_api import DeterministicEmbedder
        analyzer = ProductionAnalyzer(configured, normalizer=AcceptedNormalizer(), embedder=DeterministicEmbedder())

        calls = {"count": 0}
        real_digest_directory = digest_directory

        def counting_digest_directory(path):
            calls["count"] += 1
            return real_digest_directory(path)

        monkeypatch.setattr("phone_dino.production.digest_directory", counting_digest_directory)
        first = analyzer.readiness()
        second = analyzer.readiness()
        third = analyzer.readiness()
        assert first == (True, None)
        assert second == (True, None) and third == (True, None)
        assert calls["count"] == 1, "positive readiness must be cached, not re-hashed every call"

    def test_negative_readiness_is_not_cached(self, tmp_path):
        configured = Settings(
            service_token="secret", fixture_enabled=False, fixture_dir=None,
            artifact_manifest=None, artifact_package_digest=None, model_repo=None, model_weights=None,
            max_image_bytes=1024, max_image_pixels=1000, max_image_width=100, max_image_height=100,
        )
        analyzer = ProductionAnalyzer(configured)
        assert analyzer.readiness() == (False, "ARTIFACT_MANIFEST_NOT_AVAILABLE")
        assert analyzer.readiness() == (False, "ARTIFACT_MANIFEST_NOT_AVAILABLE")
        assert analyzer._readiness_cache is None


def test_analysis_timeout_returns_504(tmp_path, manifest_factory, png_bytes):
    class SlowNormalizer:
        def normalize(self, image, artifact):
            time.sleep(0.3)
            return NormalizedCapture(
                rgb="canonical", encoded=b"canonical-png",
                alignment=AlignmentObservation(
                    state="ALIGNED", method="TARGET_AFFINE", targetRelative=True, inlierCount=20,
                    inlierRatio=0.8, reprojectionErrorPx=0.5, coverageRatio=0.4,
                    transformWithinBounds=True, inspectionMaskApplied=True,
                ),
            )

    manifest = manifest_factory(png_bytes)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"reviewed-model-weights")
    repo = tmp_path / "dinov2"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# reviewed\n", encoding="utf-8")
    repository_version = digest_directory(repo)
    component = manifest["executionBundle"]["goldenSetVersion"]
    artifact_body = {
        "schemaVersion": "1.1", "recipeId": manifest["recipeId"], "machineId": manifest["machineId"],
        "boardId": manifest["boardId"], "goldenSetVersion": component,
        "normalizationPipelineVersion": manifest["executionBundle"]["normalizationPipelineVersion"],
        "analyzerModelVersion": manifest["executionBundle"]["analyzerModelVersion"],
        "decisionPolicyVersion": manifest["executionBundle"]["decisionPolicyVersion"],
        "analyzerRuntimeVersion": "sha256:" + hashlib.sha256(b"phone_dino:0.3.0").hexdigest(),
        "modelRepositoryVersion": repository_version,
        "boardInstallationVersion": manifest["executionBundle"]["boardInstallationVersion"],
        "modelWeightsSha256": "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest(),
        "board": {
            "squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
            "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896,
        },
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
        "targetAlignment": target_alignment(png_bytes_for_reference()),
        "goldenEmbeddings": [{"id": "g", "sourceSha256": component, "values": [1.0, 0.0]}],
    }
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(artifact_body, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    artifact_digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest["artifactPackageDigest"] = artifact_digest
    manifest["simulation"] = False
    configured = replace(
        settings(tmp_path, enabled=False), artifact_manifest=artifact,
        artifact_package_digest=artifact_digest, model_repo=repo, model_weights=weights,
        model_repository_version=repository_version, analysis_timeout_seconds=0.05,
    )
    from .test_api import DeterministicEmbedder
    analyzer = ProductionAnalyzer(configured, normalizer=SlowNormalizer(), embedder=DeterministicEmbedder())
    client = TestClient(create_app(configured, production_analyzer=analyzer))
    response = _post(client, manifest, png_bytes)
    assert response.status_code == 504
    assert response.json()["detail"] == "ANALYSIS_TIMEOUT"


@pytest.mark.skipif(
    not __import__("pathlib").Path(r"C:\code\claude\phone_dino\runtime\models\dinov2_vits14_pretrain.pth").is_file(),
    reason="Real DINOv2 weights are not present on disk; this is a real-model smoke test, not fixture-based.",
)
def test_real_model_patch_embedding_shape_and_self_consistency():
    """Runs the actual DINOv2 ViT-S/14 forward pass; proves embed_with_patches works
    against the real model, not just protocol-shaped stubs."""
    from pathlib import Path
    import numpy as np
    from PIL import Image as PILImage
    from phone_dino.engines import LocalDinoV2Adapter
    from phone_dino.production import DinoV2Embedder

    # Earlier tests in this process set PIL's global MAX_IMAGE_PIXELS via
    # decoder.py's per-request bomb guard; this raw in-memory array is safe.
    PILImage.MAX_IMAGE_PIXELS = None
    repo = Path(r"C:\code\claude\phone_dino\runtime\models\dinov2")
    weights = Path(r"C:\code\claude\phone_dino\runtime\models\dinov2_vits14_pretrain.pth")
    adapter = LocalDinoV2Adapter(repo, weights)
    ready, reason = adapter.readiness()
    assert ready, reason
    embedder = DinoV2Embedder(adapter)

    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    result = embedder.embed_with_patches(rgb)
    assert len(result.global_vector) == 384
    assert result.grid_height == 16 and result.grid_width == 16
    assert len(result.patch_grid) == 256
    assert all(len(patch) == 384 for patch in result.patch_grid)
    assert all(math.isfinite(v) for v in result.global_vector)

    golden = GoldenEmbedding(
        id="G-1", sourceSha256="sha256:" + "0" * 64, values=result.global_vector,
        patchValues=result.patch_grid, patchGridHeight=16, patchGridWidth=16,
    )
    evidence = _spatial_difference_evidence(result, golden, POLICY, 320, 240)
    assert evidence.state == "AVAILABLE"
    assert evidence.regions == [], "comparing an image against its own Golden embedding must not fabricate anomalies"
