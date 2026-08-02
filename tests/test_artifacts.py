from __future__ import annotations

import hashlib
import base64
import json

import pytest

from phone_dino.artifacts import ArtifactError, load_artifact


def prefixed(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def valid_artifact(weights: bytes) -> dict[str, object]:
    version = prefixed(b"version")
    return {
        "schemaVersion": "1.1", "recipeId": "recipe-1", "machineId": "machine-1", "boardId": "board-1",
        "goldenSetVersion": version, "normalizationPipelineVersion": version,
        "analyzerModelVersion": version, "decisionPolicyVersion": version, "boardInstallationVersion": version,
        "analyzerRuntimeVersion": prefixed(b"phone_dino:0.3.0"), "modelRepositoryVersion": version,
        "modelWeightsSha256": prefixed(weights),
        "board": {"squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
                  "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896},
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
        "targetAlignment": target_alignment(),
        "goldenEmbeddings": [{"id": "golden-1", "sourceSha256": version, "values": [1.0, 0.0]}],
    }


def target_alignment() -> dict[str, object]:
    reference = b"not-decoded-in-artifact-schema-test"
    return {
        "method": "TARGET_AFFINE", "referenceImageBase64": base64.b64encode(reference).decode(),
        "referenceImageSha256": prefixed(reference), "canonicalWidth": 320, "canonicalHeight": 240,
        "alignmentRegions": [{"x": 0, "y": 0, "width": 320, "height": 80}],
        "heldOutRegions": [{"x": 0, "y": 200, "width": 320, "height": 40}],
        "inspectionRegions": [{"x": 100, "y": 100, "width": 100, "height": 80}],
        "minMatches": 8, "minInliers": 6, "minInlierRatio": 0.5, "minCoverageRatio": 0.02,
        "maxReprojectionErrorPx": 3.0, "minScale": 0.8, "maxScale": 1.2,
        "maxRotationDegrees": 15.0, "maxShear": 0.05, "maxTranslationPx": 300.0,
        "maxSecondaryInlierRatio": 0.35, "minHeldOutMatches": 4, "maxHeldOutReprojectionErrorPx": 3.0,
    }


def test_artifact_rejects_unknown_fields_and_zero_embeddings(tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    body = valid_artifact(weights.read_bytes())
    body["unexpected"] = True
    body["goldenEmbeddings"][0]["values"] = [0.0, 0.0]
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ArtifactError, match="ARTIFACT_MANIFEST_INVALID"):
        load_artifact(artifact, prefixed(artifact.read_bytes()), weights)


def test_artifact_digest_is_checked_before_parsing(tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(valid_artifact(weights.read_bytes())), encoding="utf-8")

    with pytest.raises(ArtifactError, match="ARTIFACT_PACKAGE_DIGEST_MISMATCH"):
        load_artifact(artifact, prefixed(b"different"), weights)


def test_artifact_rejects_overlapping_alignment_held_out_and_inspection_regions(tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    body = valid_artifact(weights.read_bytes())
    body["targetAlignment"]["heldOutRegions"] = [{"x": 0, "y": 20, "width": 100, "height": 40}]
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ArtifactError, match="ARTIFACT_MANIFEST_INVALID"):
        load_artifact(artifact, prefixed(artifact.read_bytes()), weights)
