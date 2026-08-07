from __future__ import annotations

import hashlib
import base64
import json

import pytest

from phone_dino.analyzer import RUNTIME_DIGEST
from phone_dino.artifacts import (
    ArtifactError, CandidateVerificationPolicy, CharucoBoard, ProductionArtifactV18,
    ScorerInputContract, _paired_wire_schema_matches, load_artifact,
)


def prefixed(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def valid_artifact(weights: bytes) -> dict[str, object]:
    version = prefixed(b"version")
    return {
        "schemaVersion": "1.1", "recipeId": "recipe-1", "machineId": "machine-1", "boardId": "board-1",
        "goldenSetVersion": version, "normalizationPipelineVersion": version,
        "analyzerModelVersion": version, "decisionPolicyVersion": version, "boardInstallationVersion": version,
        "analyzerRuntimeVersion": RUNTIME_DIGEST, "modelRepositoryVersion": version,
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


def test_schema_18_accepts_existing_wire_14_and_candidate_dimension_wire_15():
    artifact = object.__new__(ProductionArtifactV18)

    assert _paired_wire_schema_matches(artifact, "1.4")
    assert _paired_wire_schema_matches(artifact, "1.5")
    assert not _paired_wire_schema_matches(artifact, "1.3")


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


def test_candidate_gate_requires_approved_artifact_policy():
    body = {
        "version": "candidate-verify-1.0",
        "method": "DINO_CROP_COSINE_V1",
        "mode": "GATE",
        "approvalState": "ENGINEERING_AUTO",
        "contextPaddingRatio": 0.35,
        "minimumCropSidePx": 112,
        "maxCandidates": 12,
        "reviewPriorityDistance": 0.1,
        "highPriorityDistance": 0.25,
    }
    with pytest.raises(ValueError, match="GATE mode requires APPROVED"):
        CandidateVerificationPolicy.model_validate(body)

    body["approvalState"] = "APPROVED"
    assert CandidateVerificationPolicy.model_validate(body).mode == "GATE"


def test_scorer_input_contract_accepts_exact_three_channel_json_array():
    payload = {
        "schemaVersion": "1.0",
        "policy": "INSPECTION_ROI_SUBJECT_SUPPORT_TILES_NEUTRAL_OUTSIDE",
        "coordinateSpace": "TARGET_CANONICAL_IMAGE",
        "inspectionRoiContractDigest": prefixed(b"roi"),
        "tileOrder": "TOP_TO_BOTTOM_LEFT_TO_RIGHT",
        "neutralRgb": [127, 127, 127],
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")

    contract = ScorerInputContract.model_validate({**payload, "digest": prefixed(encoded)})

    assert contract.neutral_rgb == [127, 127, 127]
    with pytest.raises(ValueError):
        ScorerInputContract.model_validate({**payload, "neutralRgb": [127, 127], "digest": prefixed(encoded)})


def test_phonecv_charuco_profile_pins_exact_marker_ids():
    board = CharucoBoard.model_validate({
        "profileId": "COMPACT_130X90_V1",
        "squaresX": 7, "squaresY": 5,
        "squareLengthMm": 10.0, "markerLengthMm": 7.0,
        "markerIds": list(range(100, 117)),
        "dictionary": "DICT_5X5_1000",
        "canonicalWidth": 896, "canonicalHeight": 896,
    })

    assert board.profile_id == "COMPACT_130X90_V1"
    assert board.marker_ids == list(range(100, 117))
    with pytest.raises(ValueError, match="one unique ID per marker cell"):
        CharucoBoard.model_validate({
            **board.model_dump(by_alias=True),
            "markerIds": list(range(100, 116)),
        })


def test_phonecv_a4_metric_profile_has_the_complete_7_by_9_charuco_layout():
    board = CharucoBoard.model_validate({
        "profileId": "A4_METRIC_200X230_V1",
        "squaresX": 7, "squaresY": 9,
        "squareLengthMm": 20.0, "markerLengthMm": 14.0,
        "markerIds": list(range(100, 131)),
        "dictionary": "DICT_5X5_1000",
        "canonicalWidth": 896, "canonicalHeight": 896,
    })

    assert board.profile_id == "A4_METRIC_200X230_V1"
    assert board.marker_ids == list(range(100, 131))
    assert len(board.marker_ids) == 31
