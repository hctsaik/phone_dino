from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

np = pytest.importorskip("numpy", reason="production vision extra is not installed")
cv2 = pytest.importorskip("cv2", reason="production OpenCV extra is not installed")

from phone_dino.artifacts import ProductionArtifact
from phone_dino.config import Settings
from phone_dino.contracts import AnalyzeRequest
from phone_dino.decoder import DecodedImage
from phone_dino.production import OpenCvCharucoNormalizer, ProductionAnalyzer
from phone_dino.synthetic_scene import SyntheticSceneError, generate_suite, verify_suite_manifest


def _artifact(document: dict[str, object]) -> ProductionArtifact:
    digest = "sha256:" + "1" * 64
    return ProductionArtifact.model_validate({
        "schemaVersion": "1.1", "recipeId": "synthetic-recipe", "machineId": "synthetic-machine",
        "boardId": "synthetic-board", "goldenSetVersion": digest,
        "normalizationPipelineVersion": digest, "analyzerModelVersion": digest,
        "decisionPolicyVersion": digest, "analyzerRuntimeVersion": digest,
        "modelRepositoryVersion": digest, "boardInstallationVersion": digest,
        "modelWeightsSha256": digest, **document["artifactInputs"],
        "goldenEmbeddings": [{"id": "synthetic-golden", "sourceSha256": digest, "values": [1.0, 0.0, 0.0]}],
    })


def _normalized(root: Path, scene: dict[str, object], artifact: ProductionArtifact):
    record = scene["image"]
    content = (root / record["path"]).read_bytes()
    decoded = DecodedImage(
        data=content, width=record["width"], height=record["height"],
        format="JPEG" if record["contentType"] == "image/jpeg" else "PNG", elapsed_ms=0,
    )
    return OpenCvCharucoNormalizer().normalize(decoded, artifact)


def _request(scene: dict[str, object], artifact: ProductionArtifact) -> tuple[AnalyzeRequest, DecodedImage]:
    record = scene["image"]
    content = (Path(scene["_root"]) / record["path"]).read_bytes()
    component = artifact.golden_set_version
    bundle = {
        "recipeVersion": component, "goldenSetVersion": component, "capturePolicyVersion": component,
        "decisionPolicyVersion": component, "normalizationPipelineVersion": component,
        "analyzerModelVersion": component, "clientAssetVersion": component,
        "boardInstallationVersion": component,
    }
    bundle_bytes = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    request = AnalyzeRequest.model_validate({
        "schemaVersion": "1.0", "requestId": "synthetic-" + scene["sceneId"],
        "sessionId": "synthetic-session", "captureOrdinal": 1, "correlationId": "synthetic-correlation",
        "deadline": datetime.now(timezone.utc) + timedelta(minutes=1),
        "rawSha256": hashlib.sha256(content).hexdigest(), "contentType": record["contentType"],
        "recipeId": artifact.recipe_id, "machineId": artifact.machine_id, "boardId": artifact.board_id,
        "inspectionIntent": "PM_SIMILARITY",
        "executionBundleDigest": "sha256:" + hashlib.sha256(bundle_bytes).hexdigest(),
        "executionBundle": bundle, "artifactPackageDigest": "sha256:" + "2" * 64, "simulation": False,
    })
    decoded = DecodedImage(
        data=content, width=record["width"], height=record["height"],
        format="JPEG" if record["contentType"] == "image/jpeg" else "PNG", elapsed_ms=0,
    )
    return request, decoded


def _all_files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.iterdir())}


@pytest.fixture(scope="module")
def synthetic_suite(tmp_path_factory):
    manifest = generate_suite(tmp_path_factory.mktemp("synthetic-suite") / "suite")
    return manifest, verify_suite_manifest(manifest)


def test_suite_is_byte_reproducible_in_fresh_process_and_seed_changes_bytes(tmp_path):
    first, second, different = tmp_path / "first", tmp_path / "second", tmp_path / "different"
    source_root = str(Path(__file__).parents[1] / "src")
    environment = {**os.environ, "PYTHONPATH": source_root}
    for output, seed in ((first, 20260802), (second, 20260802), (different, 20260803)):
        subprocess.run(
            [sys.executable, "-m", "phone_dino.synthetic_scene", str(output), "--seed", str(seed)],
            check=True, env=environment, capture_output=True, text=True,
        )

    assert _all_files(first) == _all_files(second)
    assert (first / "target-reference.png").read_bytes() != (different / "target-reference.png").read_bytes()


def test_duplicate_is_never_accepted_across_five_fresh_processes(synthetic_suite):
    manifest, _ = synthetic_suite
    source_root = str(Path(__file__).parents[1] / "src")
    environment = {**os.environ, "PYTHONPATH": source_root}
    program = (
        "import json,sys; from pathlib import Path; "
        "from phone_dino.synthetic_scene import evaluate_suite_alignment; "
        "print(json.dumps(evaluate_suite_alignment(Path(sys.argv[1]))['rejected-target-duplicate'],sort_keys=True))"
    )
    for _ in range(5):
        completed = subprocess.run(
            [sys.executable, "-c", program, str(manifest)], check=True, env=environment,
            capture_output=True, text=True,
        )
        observation = json.loads(completed.stdout)
        assert observation["captureState"] == "RECAPTURE_REQUIRED"
        assert any(reason.startswith("TARGET_") for reason in observation["reasonCodes"])


def test_manifest_truth_is_verified_and_mutations_are_rejected(tmp_path):
    manifest = generate_suite(tmp_path / "suite")
    document = verify_suite_manifest(manifest)
    assert document["evidenceClass"] == "SYNTHETIC_ENGINEERING_ONLY"
    assert len(document["scenes"]) == 16
    assert document["scenes"][0]["conditions"]["measured"]["detectedCharucoCorners"] >= 6

    mutated = json.loads(manifest.read_text(encoding="utf-8"))
    mutated["scenes"][0]["geometry"]["targetProjectedCorners"][0][0] += 3
    manifest.write_text(json.dumps(mutated, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(SyntheticSceneError, match="MANIFEST_TARGET_CORNERS_MISMATCH"):
        verify_suite_manifest(manifest)


def test_board_and_target_poses_are_independent_metamorphic_pairs(synthetic_suite):
    _, document = synthetic_suite
    scenes = {scene["sceneId"]: scene["geometry"] for scene in document["scenes"]}
    reference = scenes["accepted-reference"]
    board_motion = scenes["accepted-board-motion"]
    target_motion = scenes["accepted-target-offset"]
    combined = scenes["accepted-combined-motion"]
    defect = scenes["accepted-defect"]

    assert board_motion["boardObjectToImageTransform"] != reference["boardObjectToImageTransform"]
    assert board_motion["targetReferenceToImageTransform"] == reference["targetReferenceToImageTransform"]
    assert target_motion["boardObjectToImageTransform"] == reference["boardObjectToImageTransform"]
    assert target_motion["targetReferenceToImageTransform"] != reference["targetReferenceToImageTransform"]
    assert combined["boardObjectToImageTransform"] != reference["boardObjectToImageTransform"]
    assert combined["targetReferenceToImageTransform"] != reference["targetReferenceToImageTransform"]
    assert defect["boardObjectToImageTransform"] == reference["boardObjectToImageTransform"]
    assert defect["targetReferenceToImageTransform"] == reference["targetReferenceToImageTransform"]


def test_all_scenes_follow_real_charuco_and_target_alignment_safe_outcomes(synthetic_suite):
    manifest, document = synthetic_suite
    artifact = _artifact(document)

    for scene in document["scenes"]:
        result = _normalized(manifest.parent, scene, artifact)
        accepted = not result.reason_codes
        assert accepted is (scene["expected"]["captureState"] == "ACCEPTED"), (
            scene["sceneId"], result.reason_codes
        )
        if accepted:
            assert result.alignment is not None and result.alignment.state == "ALIGNED"
            assert result.rgb is not None and result.encoded
        else:
            assert result.rgb is None and result.encoded == b""
            assert scene["expected"]["analysisState"] == "NOT_RUN"
            expected_reason = scene["expected"]["reasonClass"]
            if expected_reason == "TARGET_ALIGNMENT":
                assert any(reason.startswith("TARGET_") for reason in result.reason_codes)
            else:
                assert expected_reason in result.reason_codes


def test_production_analyzer_never_embeds_any_expected_failure_scene(synthetic_suite):
    manifest, document = synthetic_suite
    artifact = _artifact(document)

    class CountingEmbedder:
        def __init__(self):
            self.calls = 0

        def embed(self, rgb):
            self.calls += 1
            return [1.0, 0.5, 0.25]

    embedder = CountingEmbedder()
    settings = Settings(
        service_token="synthetic", fixture_enabled=False, fixture_dir=None, artifact_manifest=None,
        artifact_package_digest=None, model_repo=None, model_weights=None, max_image_bytes=5_000_000,
        max_image_pixels=2_000_000, max_image_width=2000, max_image_height=2000,
    )
    analyzer = ProductionAnalyzer(settings, normalizer=OpenCvCharucoNormalizer(), embedder=embedder)
    analyzer._artifact = artifact
    analyzer.readiness = lambda: (True, None)
    for original in document["scenes"]:
        scene = {**original, "_root": str(manifest.parent)}
        request, decoded = _request(scene, artifact)
        before = embedder.calls
        observation = analyzer.analyze(request, decoded)
        if scene["expected"]["captureState"] == "RECAPTURE_REQUIRED":
            assert observation.analysis.state == "NOT_RUN"
            assert embedder.calls == before, scene["sceneId"]
        else:
            assert observation.analysis.state == "RUN"
            assert embedder.calls == before + 1


def test_defect_is_nonempty_inspection_only_and_preserved_in_canonical_target(synthetic_suite):
    manifest, document = synthetic_suite
    artifact = _artifact(document)
    scenes = {scene["sceneId"]: scene for scene in document["scenes"]}
    clean = _normalized(manifest.parent, scenes["accepted-reference"], artifact)
    defect = _normalized(manifest.parent, scenes["accepted-defect"], artifact)

    measured = scenes["accepted-defect"]["conditions"]["measured"]
    assert measured["defectChangedPixels"] > 0
    assert measured["defectChangedPixelsOutsideInspection"] == 0
    assert clean.alignment is not None and clean.alignment.state == "ALIGNED"
    assert defect.alignment is not None and defect.alignment.state == "ALIGNED"
    clean_gray = cv2.cvtColor(np.asarray(clean.rgb), cv2.COLOR_RGB2GRAY)
    defect_gray = cv2.cvtColor(np.asarray(defect.rgb), cv2.COLOR_RGB2GRAY)
    delta = cv2.absdiff(clean_gray, defect_gray)
    inspection_delta = int(cv2.countNonZero(delta[82:159, 105:216]))
    assert inspection_delta > 500


def test_output_bounds_and_measured_quality_are_from_encoded_files(synthetic_suite):
    manifest, document = synthetic_suite
    by_id = {scene["sceneId"]: scene for scene in document["scenes"]}
    assert by_id["rejected-blur"]["conditions"]["measured"]["laplacianVariance"] < 35.0
    assert by_id["rejected-overexposure"]["conditions"]["measured"]["overExposureRatio"] > 0.38
    assert by_id["rejected-board-occluded"]["conditions"]["measured"]["detectedCharucoCorners"] < 6
    for scene in document["scenes"]:
        assert scene["image"]["width"] == 960 and scene["image"]["height"] == 1040
        assert (manifest.parent / scene["image"]["path"]).stat().st_size < 5_000_000
