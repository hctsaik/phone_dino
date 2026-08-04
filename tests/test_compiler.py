from __future__ import annotations

import hashlib
import base64
import io
import json
from datetime import datetime, timezone
from copy import deepcopy

from PIL import Image
import pytest

from phone_dino.analyzer import RUNTIME_DIGEST
from phone_dino.artifacts import (
    ArtifactError, ProductionArtifact, ProductionArtifactV12, ProductionArtifactV13,
    ProductionArtifactV15, ProductionArtifactV16, verify_artifact_binding,
)
from phone_dino.compiler import CompilerError, compile_artifact
from phone_dino.config import Settings
from phone_dino.contracts import AlignmentObservation, AnalyzeRequest
from phone_dino.decoder import DecodedImage
from phone_dino.production import NormalizedCapture, PatchEmbedding, ProductionAnalyzer, _subject_scope
from phone_dino.security import digest_directory
from phone_dino.segmenters import SubjectMaskPrediction


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (320, 240), color).save(output, "PNG")
    return output.getvalue()


class AcceptedNormalizer:
    def normalize(self, image, artifact):
        return NormalizedCapture(
            rgb=image.data, encoded=image.data,
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


class StableSubjectSegmenter:
    def segment(self, canonical_rgb, prompt_box_xyxy, **kwargs):
        import numpy as np

        mask = np.zeros((240, 320), dtype=bool)
        mask[110:170, 110:190] = True
        return SubjectMaskPrediction(
            mask=mask, quality_score=0.95, prompt_box_xyxy=prompt_box_xyxy,
            foreground_ratio=0.6,
        )


class RecordingSubjectSegmenter(StableSubjectSegmenter):
    def __init__(self):
        self.calls = 0

    def segment(self, canonical_rgb, prompt_box_xyxy, **kwargs):
        self.calls += 1
        return super().segment(canonical_rgb, prompt_box_xyxy, **kwargs)


class CanonicalImageNormalizer:
    def normalize(self, image, artifact):
        import numpy as np

        with Image.open(io.BytesIO(image.data)) as decoded:
            rgb = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        return NormalizedCapture(
            rgb=rgb, encoded=image.data,
            alignment=AlignmentObservation(
                state="ALIGNED", method="TARGET_AFFINE", targetRelative=True, inlierCount=20,
                inlierRatio=0.8, reprojectionErrorPx=0.5, coverageRatio=0.4,
                transformWithinBounds=True, inspectionMaskApplied=True,
            ),
        )


class StablePatchEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_with_patches(self, rgb):
        import numpy as np

        self.calls += 1
        payload = np.asarray(rgb, dtype=np.uint8).tobytes(order="C")
        value = int(hashlib.sha256(payload).hexdigest()[:8], 16) / 0xFFFFFFFF
        vector = [1.0, value, 0.25]
        return PatchEmbedding(global_vector=vector, patch_grid=[vector] * 4, grid_height=2, grid_width=2)


def with_digest(payload):
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return {**payload, "digest": digest(encoded)}


def upgrade_to_v15(body):
    body["schemaVersion"] = "1.5"
    roi_payload = {
        "version": "roi-1.0", "canonicalWidth": 320, "canonicalHeight": 240,
        "polygon": [
            {"x": 100.0, "y": 100.0}, {"x": 200.0, "y": 100.0},
            {"x": 200.0, "y": 180.0}, {"x": 100.0, "y": 180.0},
        ],
        "inspectionRegions": [{"x": 100, "y": 100, "width": 100, "height": 80}],
    }
    body["inspectionRoi"] = with_digest(roi_payload)
    body["spatialDifferencePolicy"] = {
        "anomalyDistanceThreshold": 0.35, "minRegionAreaRatio": 0.003, "maxRegions": 2,
    }
    body["subjectSegmentation"] = {
        "version": "subject-1.0", "method": "MOBILE_SAM_VIT_T_BOX_PROMPT",
        "usageMode": "SPATIAL_GATE", "approvalState": "ENGINEERING_AUTO",
        "promptPolicy": "INSPECTION_ROI_BOUNDING_BOX_V1",
        "modelRepositoryVersion": digest(b"mobile-sam-repo"),
        "modelWeightsSha256": digest(b"mobile-sam-weights"),
        "minModelQualityScore": 0.8, "minForegroundRatio": 0.1,
        "maxForegroundRatio": 0.9, "supportPaddingPx": 8, "boundaryBandPx": 4,
    }
    body["subjectAlignment"] = {
        "version": "subject-align-1.0", "method": "SUBJECT_CONTOUR_ECC_AFFINE",
        "approvalState": "ENGINEERING_AUTO", "maskSource": "GOLDEN_SUBJECT_MASK",
        "alignmentBandPx": 12, "heldOutBlockPx": 16, "maxIterations": 100,
        "convergenceEpsilon": 0.00001, "minEccCorrelation": 0.2,
        "maxHeldOutResidualPx": 8.0, "minHeldOutCoverageRatio": 0.35,
        "maxResidualTranslationPx": 30.0, "maxResidualRotationDegrees": 10.0,
        "maxResidualScaleDelta": 0.2, "maxResidualShear": 0.1,
    }
    body["candidateVerificationPolicy"] = {
        "version": "candidate-verify-1.0", "method": "DINO_CROP_COSINE_V1",
        "mode": "SHADOW", "approvalState": "ENGINEERING_AUTO",
        "contextPaddingRatio": 0.35, "minimumCropSidePx": 112, "maxCandidates": 2,
        "reviewPriorityDistance": 0.1, "highPriorityDistance": 0.25,
    }
    scorer = with_digest({
        "schemaVersion": "1.0", "policy": "INSPECTION_ROI_SUBJECT_SUPPORT_TILES_NEUTRAL_OUTSIDE",
        "coordinateSpace": "TARGET_CANONICAL_IMAGE",
        "inspectionRoiContractDigest": body["inspectionRoi"]["digest"],
        "tileOrder": "TOP_TO_BOTTOM_LEFT_TO_RIGHT", "neutralRgb": [127, 127, 127],
    })
    bindings = {
        "alignmentTemplate": with_digest({
            "role": "ALIGNMENT_TEMPLATE", "id": "ALIGNMENT-1", "version": "1",
            "sourceDigest": body["goldenSources"][0]["sourceSha256"],
        }),
        "targetReference": with_digest({
            "role": "TARGET_REFERENCE", "id": "TARGET-1", "version": "1",
            "sourceDigest": body["targetAlignment"]["referenceImageSha256"],
        }),
        "normalReferenceSet": with_digest({
            "role": "NORMAL_REFERENCE_SET", "id": "NORMALS-1", "version": "1",
            "sourceDigest": body["goldenSetVersion"],
        }),
        "displayReference": with_digest({
            "role": "DISPLAY_REFERENCE", "id": "DISPLAY-1", "version": "1",
            "sourceDigest": body["goldenSources"][0]["sourceSha256"],
        }),
    }
    profile = with_digest({
        "schemaVersion": "1.0", "id": "PROFILE-1", "version": "1", **bindings,
        "inspectionRoiContractDigest": body["inspectionRoi"]["digest"],
        "scorerInputContractDigest": scorer["digest"],
    })
    body["scorerInputContract"] = scorer
    body["recipeAnalysisProfile"] = profile


def upgrade_to_v16(body):
    upgrade_to_v15(body)
    body["schemaVersion"] = "1.6"
    scorer = with_digest({
        "schemaVersion": "1.1", "policy": "INSPECTION_ROI_PAIRED_INTERIOR_TILES_NEUTRAL_OUTSIDE",
        "coordinateSpace": "TARGET_CANONICAL_IMAGE",
        "inspectionRoiContractDigest": body["inspectionRoi"]["digest"],
        "tileOrder": "TOP_TO_BOTTOM_LEFT_TO_RIGHT", "neutralRgb": [127, 127, 127],
    })
    profile_payload = {
        key: value for key, value in body["recipeAnalysisProfile"].items() if key != "digest"
    }
    profile_payload["scorerInputContractDigest"] = scorer["digest"]
    body["scorerInputContract"] = scorer
    body["recipeAnalysisProfile"] = with_digest(profile_payload)
    body["currentSubjectSegmentation"] = {
        "version": "current-subject-1.0", "method": "MOBILE_SAM_VIT_T_BOX_PROMPT",
        "promptPolicy": "INSPECTION_ROI_BOUNDING_BOX_V1", "interiorErosionPx": 4,
        "minMaskIou": 0.8, "maxAreaDeltaRatio": 0.2, "minInteriorRatio": 0.6,
        "boundaryMinRegionAreaRatio": 0.001, "maxBoundaryRegions": 8,
    }


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


def test_compiler_emits_schema_1_2_with_digest_bound_inspection_roi(tmp_path):
    spec, repo, weights, body = setup_build(tmp_path)
    body["schemaVersion"] = "1.2"
    roi_payload = {
        "version": "roi-1.0",
        "canonicalWidth": 320,
        "canonicalHeight": 240,
        "polygon": [{"x": 100.0, "y": 100.0}, {"x": 200.0, "y": 100.0}, {"x": 200.0, "y": 180.0}, {"x": 100.0, "y": 180.0}],
        "inspectionRegions": [{"x": 100, "y": 100, "width": 100, "height": 80}],
    }
    roi_encoded = json.dumps(roi_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    body["inspectionRoi"] = {**roi_payload, "digest": digest(roi_encoded)}
    spec.write_text(json.dumps(body), encoding="utf-8")

    output = tmp_path / "artifact-v12.json"
    result = compile_artifact(spec, output, repo, weights, normalizer=AcceptedNormalizer(), embedder=StableEmbedder())

    artifact = ProductionArtifactV12.model_validate_json(output.read_bytes())
    assert artifact.schema_version == "1.2"
    assert artifact.inspection_roi.digest == body["inspectionRoi"]["digest"]
    assert result.artifact_package_digest == digest(output.read_bytes())


def test_compiler_emits_schema_1_3_with_hash_bound_subject_masks(tmp_path):
    spec, repo, weights, body = setup_build(tmp_path)
    body["schemaVersion"] = "1.3"
    roi_payload = {
        "version": "roi-1.0", "canonicalWidth": 320, "canonicalHeight": 240,
        "polygon": [
            {"x": 100.0, "y": 100.0}, {"x": 200.0, "y": 100.0},
            {"x": 200.0, "y": 180.0}, {"x": 100.0, "y": 180.0},
        ],
        "inspectionRegions": [{"x": 100, "y": 100, "width": 100, "height": 80}],
    }
    encoded = json.dumps(roi_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    body["inspectionRoi"] = {**roi_payload, "digest": digest(encoded)}
    body["subjectSegmentation"] = {
        "version": "subject-1.0", "method": "MOBILE_SAM_VIT_T_BOX_PROMPT",
        "usageMode": "SPATIAL_GATE", "approvalState": "ENGINEERING_AUTO",
        "promptPolicy": "INSPECTION_ROI_BOUNDING_BOX_V1",
        "modelRepositoryVersion": digest(b"mobile-sam-repo"),
        "modelWeightsSha256": digest(b"mobile-sam-weights"),
        "minModelQualityScore": 0.8, "minForegroundRatio": 0.1,
        "maxForegroundRatio": 0.9, "supportPaddingPx": 8, "boundaryBandPx": 4,
    }
    spec.write_text(json.dumps(body), encoding="utf-8")

    output = tmp_path / "artifact-v13.json"
    result = compile_artifact(
        spec, output, repo, weights, normalizer=AcceptedNormalizer(), embedder=StableEmbedder(),
        subject_segmenter=StableSubjectSegmenter(),
    )

    artifact = ProductionArtifactV13.model_validate_json(output.read_bytes())
    assert artifact.schema_version == "1.3"
    assert [item.golden_id for item in artifact.subject_segmentation.golden_masks] == ["GOLDEN-A", "GOLDEN-B"]
    assert all(item.mask_sha256.startswith("sha256:") for item in artifact.subject_segmentation.golden_masks)
    first_mask = artifact.subject_segmentation.golden_masks[0]
    first_golden = next(item for item in artifact.golden_embeddings if item.id == first_mask.golden_id)
    scope = _subject_scope(artifact, first_golden)
    assert scope.evidence.subject_mask_png_base64 == first_mask.mask_png_base64
    assert scope.evidence.subject_mask_sha256 == first_mask.mask_sha256.removeprefix("sha256:")
    evidence = json.loads(result.evidence_path.read_text(encoding="utf-8"))
    assert evidence["subjectSegmentation"]["method"] == "MOBILE_SAM_VIT_T_BOX_PROMPT"


def test_compiler_emits_schema_1_5_roi_only_profile_and_enforces_request_pin(tmp_path):
    spec, repo, weights, body = setup_build(tmp_path)
    upgrade_to_v15(body)
    spec.write_text(json.dumps(body), encoding="utf-8")
    output = tmp_path / "artifact-v15.json"

    result = compile_artifact(
        spec, output, repo, weights, normalizer=CanonicalImageNormalizer(),
        embedder=StablePatchEmbedder(), subject_segmenter=StableSubjectSegmenter(),
    )

    artifact = ProductionArtifactV15.model_validate_json(output.read_bytes())
    assert artifact.schema_version == "1.5"
    assert result.artifact_package_digest == digest(output.read_bytes())
    assert artifact.recipe_analysis_profile.scorer_input_contract_digest == artifact.scorer_input_contract.digest
    assert len({
        artifact.recipe_analysis_profile.alignment_template.digest,
        artifact.recipe_analysis_profile.target_reference.digest,
        artifact.recipe_analysis_profile.normal_reference_set.digest,
        artifact.recipe_analysis_profile.display_reference.digest,
    }) == 4
    assert len(artifact.golden_embeddings) == 2
    assert all(golden.scorer_input_sha256 and golden.scorer_input_tiles for golden in artifact.golden_embeddings)
    assert all(
        [tile.id for tile in golden.scorer_input_tiles] == ["ROI-TILE-001"]
        for golden in artifact.golden_embeddings
    )
    evidence = json.loads(result.evidence_path.read_text(encoding="utf-8"))
    assert evidence["recipeAnalysisProfileDigest"] == artifact.recipe_analysis_profile.digest
    assert evidence["scorerInputContractDigest"] == artifact.scorer_input_contract.digest

    document = json.loads(output.read_text(encoding="utf-8"))
    for role, message in (
        ("targetReference", "TargetReference"),
        ("normalReferenceSet", "NormalReferenceSet"),
        ("displayReference", "DisplayReference"),
    ):
        invalid = deepcopy(document)
        binding = {key: value for key, value in invalid["recipeAnalysisProfile"][role].items() if key != "digest"}
        binding["sourceDigest"] = digest(f"wrong-{role}".encode())
        invalid["recipeAnalysisProfile"][role] = with_digest(binding)
        profile_payload = {key: value for key, value in invalid["recipeAnalysisProfile"].items() if key != "digest"}
        invalid["recipeAnalysisProfile"] = with_digest(profile_payload)
        with pytest.raises(ValueError, match=message):
            ProductionArtifactV15.model_validate(invalid)

    duplicate_role = deepcopy(document)
    duplicate_role["recipeAnalysisProfile"]["displayReference"]["digest"] = duplicate_role["recipeAnalysisProfile"]["targetReference"]["digest"]
    with pytest.raises(ValueError, match="reference role digest|profile role digests"):
        ProductionArtifactV15.model_validate(duplicate_role)

    request_body = {
        "schemaVersion": "1.1", "requestId": "request-1", "sessionId": "session-1",
        "captureOrdinal": 1, "correlationId": "correlation-1", "deadline": datetime(2099, 1, 1, tzinfo=timezone.utc),
        "rawSha256": "a" * 64, "contentType": "image/png", "recipeId": body["recipeId"],
        "machineId": body["machineId"], "boardId": body["boardId"], "inspectionIntent": "PM_SIMILARITY",
        "executionBundleDigest": digest(b"bundle"),
        "executionBundle": {
            "recipeVersion": digest(b"recipe"), "goldenSetVersion": body["goldenSetVersion"],
            "capturePolicyVersion": digest(b"capture"), "decisionPolicyVersion": body["decisionPolicyVersion"],
            "normalizationPipelineVersion": body["normalizationPipelineVersion"],
            "analyzerModelVersion": body["analyzerModelVersion"], "clientAssetVersion": digest(b"client"),
            "boardInstallationVersion": body["boardInstallationVersion"],
        },
        "artifactPackageDigest": result.artifact_package_digest, "simulation": True,
    }
    with pytest.raises(ArtifactError, match="RECIPE_ANALYSIS_PROFILE_DIGEST_REQUIRED"):
        verify_artifact_binding(artifact, AnalyzeRequest.model_validate(request_body))
    request_body["recipeAnalysisProfileDigest"] = digest(b"wrong-profile")
    with pytest.raises(ArtifactError, match="RECIPE_ANALYSIS_PROFILE_DIGEST_MISMATCH"):
        verify_artifact_binding(artifact, AnalyzeRequest.model_validate(request_body))
    request_body["recipeAnalysisProfileDigest"] = artifact.recipe_analysis_profile.digest
    request = AnalyzeRequest.model_validate(request_body)
    verify_artifact_binding(artifact, request)

    configured = Settings(
        service_token="secret", fixture_enabled=False, fixture_dir=None,
        artifact_manifest=output, artifact_package_digest=result.artifact_package_digest,
        model_repo=repo, model_weights=weights, max_image_bytes=1_000_000,
        max_image_pixels=1_000_000, max_image_width=1000, max_image_height=1000,
        model_repository_version=digest_directory(repo), engineering_real_model_enabled=True,
        allow_target_only_alignment=True,
    )
    embedder = StablePatchEmbedder()
    analyzer = ProductionAnalyzer(
        configured, normalizer=CanonicalImageNormalizer(), embedder=embedder,
    )
    assert analyzer.warm_up() == (True, None)
    warmed_calls = embedder.calls
    assert warmed_calls == sum(len(item.scorer_input_tiles or []) for item in artifact.golden_embeddings)
    readiness = analyzer.readiness_metadata()
    assert {"INSPECTION_ROI_ONLY_SCORING_V1", "RECIPE_ANALYSIS_PROFILE_V1"}.issubset(readiness["capabilities"])
    assert readiness["recipeAnalysisProfile"]["digest"] == artifact.recipe_analysis_profile.digest
    assert readiness["scorerInputContract"]["digest"] == artifact.scorer_input_contract.digest
    current_bytes = (tmp_path / "first.png").read_bytes()
    request_body["rawSha256"] = hashlib.sha256(current_bytes).hexdigest()
    observation = analyzer.analyze(
        AnalyzeRequest.model_validate(request_body),
        DecodedImage(data=current_bytes, width=320, height=240, format="PNG", elapsed_ms=0),
    )
    assert observation.analysis.scoring_scope == "INSPECTION_ROI_ONLY"
    assert observation.analysis.nearest_golden_id == "GOLDEN-A"
    assert observation.analysis.global_distance == pytest.approx(observation.analysis.nearest_golden_distance)
    assert observation.analysis.global_distance == pytest.approx(0.0)
    assert observation.resolved_versions.recipe_analysis_profile_digest == artifact.recipe_analysis_profile.digest
    assert observation.resolved_versions.scorer_input_contract_digest == artifact.scorer_input_contract.digest
    assert observation.normalization.evidence_coordinate_space == "TARGET_CANONICAL_IMAGE"
    assert observation.normalization.scorer_input_tile_digests == observation.analysis.spatial_difference_evidence.scorer_input_tile_digests
    assert embedder.calls - warmed_calls == sum(len(item.scorer_input_tiles or []) for item in artifact.golden_embeddings)


def test_schema_1_6_segments_current_and_scores_only_eroded_paired_interior(tmp_path):
    import numpy as np

    spec, repo, weights, body = setup_build(tmp_path)
    upgrade_to_v16(body)
    spec.write_text(json.dumps(body), encoding="utf-8")
    output = tmp_path / "artifact-v16.json"
    result = compile_artifact(
        spec, output, repo, weights,
        normalizer=CanonicalImageNormalizer(), embedder=StablePatchEmbedder(),
        subject_segmenter=StableSubjectSegmenter(),
    )
    artifact = ProductionArtifactV16.model_validate_json(output.read_bytes())
    assert artifact.scorer_input_contract.policy == "INSPECTION_ROI_PAIRED_INTERIOR_TILES_NEUTRAL_OUTSIDE"

    current = np.full((240, 320, 3), (10, 20, 30), dtype=np.uint8)
    current[:100] = (240, 240, 240)
    current[:, :100] = (240, 240, 240)
    current[:, 200:] = (240, 240, 240)
    current[180:] = (240, 240, 240)
    current_buffer = io.BytesIO()
    Image.fromarray(current).save(current_buffer, "PNG")
    current_bytes = current_buffer.getvalue()
    request_body = {
        "schemaVersion": "1.2", "requestId": "request-paired", "sessionId": "session-paired",
        "captureOrdinal": 1, "correlationId": "correlation-paired",
        "deadline": datetime(2099, 1, 1, tzinfo=timezone.utc),
        "rawSha256": hashlib.sha256(current_bytes).hexdigest(), "contentType": "image/png",
        "recipeId": body["recipeId"], "machineId": body["machineId"], "boardId": body["boardId"],
        "inspectionIntent": "PM_SIMILARITY", "executionBundleDigest": digest(b"bundle"),
        "executionBundle": {
            "recipeVersion": digest(b"recipe"), "goldenSetVersion": body["goldenSetVersion"],
            "capturePolicyVersion": digest(b"capture"), "decisionPolicyVersion": body["decisionPolicyVersion"],
            "normalizationPipelineVersion": body["normalizationPipelineVersion"],
            "analyzerModelVersion": body["analyzerModelVersion"], "clientAssetVersion": digest(b"client"),
            "boardInstallationVersion": body["boardInstallationVersion"],
        },
        "artifactPackageDigest": result.artifact_package_digest,
        "recipeAnalysisProfileDigest": artifact.recipe_analysis_profile.digest,
        "simulation": True,
    }
    configured = Settings(
        service_token="secret", fixture_enabled=False, fixture_dir=None,
        artifact_manifest=output, artifact_package_digest=result.artifact_package_digest,
        model_repo=repo, model_weights=weights, max_image_bytes=1_000_000,
        max_image_pixels=1_000_000, max_image_width=1000, max_image_height=1000,
        model_repository_version=digest_directory(repo), engineering_real_model_enabled=True,
        allow_target_only_alignment=True,
    )
    runtime_segmenter = RecordingSubjectSegmenter()
    runtime_embedder = StablePatchEmbedder()
    analyzer = ProductionAnalyzer(
        configured, normalizer=CanonicalImageNormalizer(), embedder=runtime_embedder,
        subject_segmenter=runtime_segmenter,
    )
    assert analyzer.warm_up() == (True, None)
    observation = analyzer.analyze(
        AnalyzeRequest.model_validate(request_body),
        DecodedImage(data=current_bytes, width=320, height=240, format="PNG", elapsed_ms=0),
    )

    assert runtime_segmenter.calls == 1
    assert observation.analysis.state.value == "RUN"
    assert observation.analysis.global_distance == pytest.approx(0.0)
    subject = observation.analysis.subject_segmentation_evidence
    assert subject.current_subject_mask_sha256
    assert subject.interior_mask_sha256 == subject.support_mask_sha256
    assert subject.interior_ratio < 1.0
    spatial = observation.analysis.spatial_difference_evidence
    assert spatial.generation_method == "PAIRED_INTERIOR_ROI_TILED_PATCH_DISTANCE"
    assert all(region.kind == "SUBJECT_INTERIOR" for region in spatial.regions)
    boundary = observation.analysis.boundary_difference_evidence
    assert boundary.state == "AVAILABLE"
    assert boundary.mask_intersection_over_union == pytest.approx(1.0)
    assert boundary.regions == []
    assert {
        "CURRENT_SUBJECT_SEGMENTATION_V1", "PAIRED_SUBJECT_INTERIOR_V1",
        "SUBJECT_BOUNDARY_GEOMETRY_V1", "ALIGNMENT_FAIL_CLOSED_DINO_V1",
    }.issubset(analyzer.readiness_metadata()["capabilities"])


def test_schema_1_6_stops_before_dino_when_current_mask_is_unqualified(tmp_path):
    import numpy as np

    class UnqualifiedSegmenter(RecordingSubjectSegmenter):
        def segment(self, canonical_rgb, prompt_box_xyxy, **kwargs):
            self.calls += 1
            mask = np.zeros((240, 320), dtype=bool)
            mask[100:120, 100:120] = True
            return SubjectMaskPrediction(
                mask=mask, quality_score=0.95, prompt_box_xyxy=prompt_box_xyxy,
                foreground_ratio=0.2,
            )

    spec, repo, weights, body = setup_build(tmp_path)
    upgrade_to_v16(body)
    spec.write_text(json.dumps(body), encoding="utf-8")
    output = tmp_path / "artifact-v16-unqualified.json"
    result = compile_artifact(
        spec, output, repo, weights,
        normalizer=CanonicalImageNormalizer(), embedder=StablePatchEmbedder(),
        subject_segmenter=StableSubjectSegmenter(),
    )
    artifact = ProductionArtifactV16.model_validate_json(output.read_bytes())
    current_bytes = (tmp_path / "first.png").read_bytes()
    request = AnalyzeRequest.model_validate({
        "schemaVersion": "1.2", "requestId": "request-mask-gate", "sessionId": "session-mask-gate",
        "captureOrdinal": 1, "correlationId": "correlation-mask-gate",
        "deadline": datetime(2099, 1, 1, tzinfo=timezone.utc),
        "rawSha256": hashlib.sha256(current_bytes).hexdigest(), "contentType": "image/png",
        "recipeId": body["recipeId"], "machineId": body["machineId"], "boardId": body["boardId"],
        "inspectionIntent": "PM_SIMILARITY", "executionBundleDigest": digest(b"bundle"),
        "executionBundle": {
            "recipeVersion": digest(b"recipe"), "goldenSetVersion": body["goldenSetVersion"],
            "capturePolicyVersion": digest(b"capture"), "decisionPolicyVersion": body["decisionPolicyVersion"],
            "normalizationPipelineVersion": body["normalizationPipelineVersion"],
            "analyzerModelVersion": body["analyzerModelVersion"], "clientAssetVersion": digest(b"client"),
            "boardInstallationVersion": body["boardInstallationVersion"],
        },
        "artifactPackageDigest": result.artifact_package_digest,
        "recipeAnalysisProfileDigest": artifact.recipe_analysis_profile.digest, "simulation": True,
    })
    configured = Settings(
        service_token="secret", fixture_enabled=False, fixture_dir=None,
        artifact_manifest=output, artifact_package_digest=result.artifact_package_digest,
        model_repo=repo, model_weights=weights, max_image_bytes=1_000_000,
        max_image_pixels=1_000_000, max_image_width=1000, max_image_height=1000,
        model_repository_version=digest_directory(repo), engineering_real_model_enabled=True,
        allow_target_only_alignment=True,
    )
    embedder = StablePatchEmbedder()
    segmenter = UnqualifiedSegmenter()
    analyzer = ProductionAnalyzer(
        configured, normalizer=CanonicalImageNormalizer(), embedder=embedder,
        subject_segmenter=segmenter,
    )
    observation = analyzer.analyze(
        request, DecodedImage(data=current_bytes, width=320, height=240, format="PNG", elapsed_ms=0),
    )
    assert segmenter.calls == 1
    assert embedder.calls == 0
    assert observation.capture_assessment.state.value == "RECAPTURE_REQUIRED"
    assert observation.analysis.state.value == "NOT_RUN"
    assert observation.capture_assessment.reason_codes[0] in {
        "CURRENT_SUBJECT_MASK_IOU_BELOW_POLICY",
        "CURRENT_SUBJECT_MASK_AREA_DELTA_ABOVE_POLICY",
    }
