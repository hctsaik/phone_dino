from __future__ import annotations

import hashlib
import base64
import io
import json

from PIL import Image
import pytest

from phone_dino.analyzer import RUNTIME_DIGEST
from phone_dino.artifacts import ProductionArtifact
from phone_dino.compiler import CompilerError, compile_artifact
from phone_dino.contracts import AlignmentObservation
from phone_dino.production import NormalizedCapture
from phone_dino.security import digest_directory


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (320, 240), color).save(output, "PNG")
    return output.getvalue()


class AcceptedNormalizer:
    def normalize(self, image, artifact):
        return NormalizedCapture(
            rgb=image.data, encoded=b"canonical:" + image.data,
            alignment=AlignmentObservation(
                state="ALIGNED", method="TARGET_AFFINE", targetRelative=True, inlierCount=20,
                inlierRatio=0.8, reprojectionErrorPx=0.5, coverageRatio=0.4,
                transformWithinBounds=True, inspectionMaskApplied=True,
            ),
        )


class RecaptureNormalizer:
    def normalize(self, image, artifact):
        return NormalizedCapture(rgb=None, encoded=b"", reason_codes=("BLUR",))


class StableEmbedder:
    def embed(self, rgb):
        value = int(hashlib.sha256(rgb).hexdigest()[:8], 16) / 0xFFFFFFFF
        return [1.0, value, 0.25]


def setup_build(tmp_path):
    repo = tmp_path / "dinov2"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# approved local source\n", encoding="utf-8")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"approved weights")
    first = png((10, 20, 30))
    second = png((30, 20, 10))
    (tmp_path / "first.png").write_bytes(first)
    (tmp_path / "second.png").write_bytes(second)
    version = digest(b"version")
    body = {
        "schemaVersion": "1.1", "recipeId": "PM-ABC-001", "machineId": "MC-07", "boardId": "CB-001",
        "goldenSetVersion": version,
        "normalizationPipelineVersion": version, "analyzerModelVersion": version, "decisionPolicyVersion": version,
        "modelRepositoryVersion": digest_directory(repo),
        "boardInstallationVersion": version, "modelWeightsSha256": digest(weights.read_bytes()),
        "board": {"squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
                  "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896},
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
        "targetAlignment": target_alignment(),
        "goldenSources": [
            {"id": "GOLDEN-B", "path": "second.png", "sourceSha256": digest(second)},
            {"id": "GOLDEN-A", "path": "first.png", "sourceSha256": digest(first)},
        ],
    }
    spec = tmp_path / "build-spec.json"
    spec.write_text(json.dumps(body), encoding="utf-8")
    return spec, repo, weights, body


def target_alignment():
    reference = png((20, 20, 20))
    return {
        "method": "TARGET_AFFINE", "referenceImageBase64": base64.b64encode(reference).decode(),
        "referenceImageSha256": digest(reference), "canonicalWidth": 320, "canonicalHeight": 240,
        "alignmentRegions": [{"x": 0, "y": 0, "width": 320, "height": 80}],
        "heldOutRegions": [{"x": 0, "y": 200, "width": 320, "height": 40}],
        "inspectionRegions": [{"x": 100, "y": 100, "width": 100, "height": 80}],
        "minMatches": 8, "minInliers": 6, "minInlierRatio": 0.5, "minCoverageRatio": 0.02,
        "maxReprojectionErrorPx": 3.0, "minScale": 0.8, "maxScale": 1.2,
        "maxRotationDegrees": 15.0, "maxShear": 0.05, "maxTranslationPx": 300.0,
        "maxSecondaryInlierRatio": 0.35, "minHeldOutMatches": 4, "maxHeldOutReprojectionErrorPx": 3.0,
    }


def test_compiler_emits_deterministic_strict_artifact_and_evidence(tmp_path):
    spec, repo, weights, body = setup_build(tmp_path)
    output = tmp_path / "artifact.json"

    result = compile_artifact(
        spec, output, repo, weights, normalizer=AcceptedNormalizer(), embedder=StableEmbedder()
    )

    assert result.artifact_package_digest == digest(output.read_bytes())
    artifact = ProductionArtifact.model_validate_json(output.read_bytes())
    assert artifact.analyzer_runtime_version == RUNTIME_DIGEST
    assert artifact.model_repository_version == body["modelRepositoryVersion"]
    assert [item.id for item in artifact.golden_embeddings] == ["GOLDEN-A", "GOLDEN-B"]
    assert all(len(item.values) == 3 for item in artifact.golden_embeddings)
    evidence = json.loads(result.evidence_path.read_text(encoding="utf-8"))
    assert evidence["artifactPackageDigest"] == result.artifact_package_digest
    assert [item["id"] for item in evidence["goldenSources"]] == ["GOLDEN-A", "GOLDEN-B"]
    with pytest.raises(CompilerError, match="ARTIFACT_OUTPUT_ALREADY_EXISTS"):
        compile_artifact(spec, output, repo, weights, normalizer=AcceptedNormalizer(), embedder=StableEmbedder())


def test_compiler_fails_closed_before_output_for_digest_or_still_gate_failure(tmp_path):
    spec, repo, weights, body = setup_build(tmp_path)
    body["goldenSources"][0]["sourceSha256"] = digest(b"wrong")
    spec.write_text(json.dumps(body), encoding="utf-8")
    output = tmp_path / "artifact.json"
    with pytest.raises(CompilerError, match="GOLDEN_SOURCE_DIGEST_MISMATCH"):
        compile_artifact(spec, output, repo, weights, normalizer=AcceptedNormalizer(), embedder=StableEmbedder())
    assert not output.exists()

    body["goldenSources"][0]["sourceSha256"] = digest((tmp_path / "second.png").read_bytes())
    spec.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(CompilerError, match="GOLDEN_SOURCE_RECAPTURE_REQUIRED:GOLDEN-A:BLUR"):
        compile_artifact(spec, output, repo, weights, normalizer=RecaptureNormalizer(), embedder=StableEmbedder())
    assert not output.exists()
