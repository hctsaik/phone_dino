from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from phone_dino.mvtec_research import generate_normal_augmentations


REPOSITORY_ROOT = Path(__file__).parents[1]
RECIPE_V3_PATH = REPOSITORY_ROOT / "tools" / "mvtec_ad_camera_lighting_recipe_v3.json"
RECIPE_V5_PATH = REPOSITORY_ROOT / "tools" / "mvtec_ad_camera_lighting_recipe_v5.json"


def _tool_module():
    path = Path(__file__).parents[1] / "tools" / "run_mvtec_ad_iteration.py"
    spec = importlib.util.spec_from_file_location("mvtec_ad_iteration", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature_extractor_identity(tool: object, *, marker: str) -> tuple[dict[str, object], str]:
    identity: dict[str, object] = {
        "schemaVersion": "phone-dino.mvtec-ad-feature-extractor/1.0",
        "modelWeightsSha256": f"sha256:{marker * 64}",
        "modelRepositorySha256": f"sha256:{'2' * 64}",
        "preprocessingId": "DINO_TEST_PREPROCESSING",
        "preprocessing": {
            "colorSpace": "RGB",
            "resizeShortEdge": 256,
            "centerCropWidth": 224,
            "centerCropHeight": 224,
            "resizeAntialias": True,
            "normalizeMean": [0.485, 0.456, 0.406],
            "normalizeStd": [0.229, 0.224, 0.225],
        },
        "modelEntrypoint": "dinov2_vits14",
        "device": "cpu",
        "iterationToolSha256": f"sha256:{'3' * 64}",
        "productionModuleSha256": f"sha256:{'4' * 64}",
        "enginesModuleSha256": f"sha256:{'5' * 64}",
        "mvtecResearchModuleSha256": f"sha256:{'6' * 64}",
        "pythonVersion": "3.11.0",
        "numpyVersion": "2.0.0",
        "torchVersion": "2.0.0",
        "torchvisionVersion": "0.0.0",
        "pillowVersion": "10.0.0",
        "torchThreadCount": 1,
        "torchBackend": {
            "deterministicAlgorithmsEnabled": False,
            "mkldnnAvailable": True,
            "mkldnnEnabled": True,
        },
    }
    return identity, tool.canonical_json_sha256(identity)


def test_blocked_patch_scoring_matches_full_matrix_reference() -> None:
    tool = _tool_module()
    rng = np.random.default_rng(7)
    query = rng.normal(size=(3, 5, 4)).astype(np.float32)
    prototypes = rng.normal(size=(9, 4)).astype(np.float32)
    scored = tool.patch_knn_scores_blocked(query, prototypes, top_k=2, prototype_block_size=3)

    normalized_query = query / np.linalg.norm(query, axis=2, keepdims=True)
    normalized_prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
    expected = np.min(np.clip(1.0 - normalized_query.reshape(-1, 4) @ normalized_prototypes.T, 0.0, 2.0), axis=1)
    expected = expected.reshape(3, 5)
    for index, result in enumerate(scored):
        assert np.allclose(result["patchDistanceGrid"], expected[index], atol=1e-6)
        assert result["score"] == pytest.approx(np.mean(np.partition(expected[index], -2)[-2:]), abs=1e-6)


def test_blocked_patch_scoring_rejects_invalid_features() -> None:
    tool = _tool_module()
    with pytest.raises(tool.IterationError, match="zero-norm"):
        tool.patch_knn_scores_blocked(np.zeros((1, 2, 2)), np.ones((1, 2)), top_k=1, prototype_block_size=1)
    with pytest.raises(tool.IterationError, match="positive"):
        tool.patch_knn_scores_blocked(np.ones((1, 2, 2)), np.ones((1, 2)), top_k=1, prototype_block_size=0)


def test_input_sort_key_is_stable_across_input_enumeration() -> None:
    tool = _tool_module()
    records = [
        {"caseId": "b/camera-augmentation/01", "parentCaseId": "b", "variantId": 1},
        {"caseId": "a/camera-augmentation/02", "parentCaseId": "a", "variantId": 2},
        {"caseId": "a", "isAugmentation": False},
        {"caseId": "a/camera-augmentation/01", "parentCaseId": "a", "variantId": 1},
    ]
    assert [record["caseId"] for record in sorted(reversed(records), key=tool.input_sort_key)] == [
        "a", "a/camera-augmentation/01", "a/camera-augmentation/02", "b/camera-augmentation/01"
    ]


def test_feature_cache_key_binds_source_and_full_extractor_identity(tmp_path: Path) -> None:
    tool = _tool_module()
    record = {"sourceSha256": f"sha256:{'1' * 64}", "augmentationRecipeSha256": None}
    first_identity, first_digest = _feature_extractor_identity(tool, marker="1")
    changed_identity, changed_digest = _feature_extractor_identity(tool, marker="7")
    first = tool.FeatureCache(
        tmp_path / "outside",
        feature_extractor=first_identity,
        feature_extractor_identity_sha256=first_digest,
    )
    changed_model = tool.FeatureCache(
        tmp_path / "outside-model",
        feature_extractor=changed_identity,
        feature_extractor_identity_sha256=changed_digest,
    )
    changed_input = dict(record, sourceSha256=f"sha256:{'5' * 64}")
    assert first.key_for(record, "patch") != changed_model.key_for(record, "patch")
    assert first.key_for(record, "patch") != first.key_for(changed_input, "patch")
    assert first.key_for(record, "patch") != first.key_for(record, "global")


def test_feature_cache_metadata_detects_finite_array_tampering(tmp_path: Path) -> None:
    tool = _tool_module()
    identity, digest = _feature_extractor_identity(tool, marker="1")
    cache = tool.FeatureCache(
        tmp_path / "outside",
        feature_extractor=identity,
        feature_extractor_identity_sha256=digest,
    )
    record = {"sourceSha256": f"sha256:{'1' * 64}", "augmentationRecipeSha256": None}
    key = cache.key_for(record, "patch")
    values = np.ones((256, 384), dtype=np.float32)
    cache.put(key, values, feature_kind="patch", record=record)
    assert np.array_equal(cache.get(key, feature_kind="patch", record=record), values)
    metadata = json.loads((tmp_path / "outside" / f"{key}.json").read_text(encoding="utf-8"))
    data_path = tmp_path / "outside" / metadata["dataFileName"]
    tampered = values.copy()
    tampered[0, 0] = 2.0
    np.save(data_path, tampered, allow_pickle=False)
    with pytest.raises(tool.IterationError, match="data digest"):
        cache.get(key, feature_kind="patch", record=record)
    (tmp_path / "outside" / f"{key}.json").unlink()
    assert cache.get(key, feature_kind="patch", record=record) is None


def test_model_repository_tree_digest_binds_file_and_directory_structure(tmp_path: Path) -> None:
    tool = _tool_module()
    repository = tmp_path / "model-repository"
    repository.mkdir()
    (repository / "hubconf.py").write_text("model = 1\n", encoding="utf-8")
    initial = tool.sha256_directory(repository)
    (repository / "hubconf.py").write_text("model = 2\n", encoding="utf-8")
    assert tool.sha256_directory(repository) != initial
    after_content_change = tool.sha256_directory(repository)
    (repository / "empty-package").mkdir()
    assert tool.sha256_directory(repository) != after_content_change


def test_model_repository_tree_digest_rejects_directory_links_when_supported(tmp_path: Path) -> None:
    tool = _tool_module()
    repository = tmp_path / "model-repository"
    repository.mkdir()
    (repository / "hubconf.py").write_text("model = 1\n", encoding="utf-8")
    target = tmp_path / "link-target"
    target.mkdir()
    try:
        (repository / "linked-package").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("this Windows test environment cannot create directory links")
    with pytest.raises(tool.IterationError, match="link or reparse point"):
        tool.sha256_directory(repository)


def test_pixel_metrics_use_dino_crop_and_perfect_patch_localization(tmp_path: Path) -> None:
    tool = _tool_module()
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[16:30, 16:30] = 255  # This becomes the top-left 14x14 DINO patch after the 16 px crop.
    Image.fromarray(mask).save(mask_path)
    mask_sha256 = f"sha256:{hashlib.sha256(mask_path.read_bytes()).hexdigest()}"
    empty_grid = np.zeros((16, 16), dtype=np.float32)
    anomaly_grid = np.zeros((16, 16), dtype=np.float32)
    anomaly_grid[0, 0] = 1.0
    records = [
        {"kind": "NOMINAL", "localization": {"patchDistanceGrid": empty_grid}},
        {
            "kind": "ANOMALY",
            "maskPath": mask_path,
            "maskSha256": mask_sha256,
            "localization": {"patchDistanceGrid": anomaly_grid},
        },
    ]
    result = tool.calculate_pixel_localization_metrics(records)
    assert result["state"] == "AVAILABLE"
    assert result["pixelAuRoc"] == pytest.approx(1.0)
    assert result["auproAt30Fpr"] == pytest.approx(1.0)


def test_pixel_metrics_returns_unavailable_without_anomaly_masks() -> None:
    tool = _tool_module()
    grid = np.zeros((16, 16), dtype=np.float32)
    result = tool.calculate_pixel_localization_metrics([{"kind": "NOMINAL", "localization": {"patchDistanceGrid": grid}}])
    assert result["state"] == "UNAVAILABLE"
    assert result["pixelAuRoc"] is None


def test_normal_calibration_summary_uses_only_scored_normal_records() -> None:
    tool = _tool_module()
    summary = tool.normal_calibration_summary([{"score": 0.1}, {"score": 0.2}, {"score": 0.9}])
    assert summary == {
        "normalCalibrationCases": 3,
        "normalScoreMedian": 0.2,
        "normalScoreP95": 0.9,
        "normalScoreMax": 0.9,
        "originalTuningNormalScoreCases": 3,
        "originalTuningNormalScoreMedian": 0.2,
        "originalTuningNormalScoreP95": 0.9,
        "originalTuningNormalScoreMax": 0.9,
    }


def test_iteration_identity_and_score_records_preserve_augmentation_variant_ids() -> None:
    tool = _tool_module()
    original = {
        "caseId": "capsule/tuning/001", "category": "capsule", "role": "THRESHOLD_TUNING",
        "kind": "NOMINAL", "defect": "good", "sourceSha256": f"sha256:{'1' * 64}",
        "isAugmentation": False, "variantId": None, "score": 0.1,
    }
    derived = {
        "caseId": "capsule/tuning/001/camera-augmentation/04", "category": "capsule", "role": "THRESHOLD_TUNING",
        "kind": "NOMINAL", "defect": "good", "sourceSha256": f"sha256:{'2' * 64}",
        "isAugmentation": True, "variantId": 4, "parentCaseId": original["caseId"],
        "parentSourceSha256": original["sourceSha256"], "augmentationRecipeSha256": f"sha256:{'3' * 64}", "score": 0.2,
    }
    assert tool.input_identity_record(original)["variantId"] is None
    assert tool._score_record(original)["variantId"] is None
    assert tool.input_identity_record(derived)["variantId"] == 4
    assert tool._score_record(derived)["variantId"] == 4


def _write_normal_only_manifest(
    tmp_path: Path, *, fit_kind: str = "NOMINAL", fit_defect: str = "good"
) -> tuple[Path, Path]:
    image_path = tmp_path / "normal.png"
    Image.new("RGB", (32, 32), (90, 130, 180)).save(image_path)
    source_sha256 = f"sha256:{hashlib.sha256(image_path.read_bytes()).hexdigest()}"
    records = [
        {"caseId": "capsule/fit/001", "category": "capsule", "role": "FIT", "kind": fit_kind,
         "defect": fit_defect, "relativePath": image_path.name, "sourceSha256": source_sha256},
        {"caseId": "capsule/tuning/001", "category": "capsule", "role": "THRESHOLD_TUNING", "kind": "NOMINAL",
         "defect": "good", "relativePath": image_path.name, "sourceSha256": source_sha256},
        {"caseId": "capsule/blind/001", "category": "capsule", "role": "BLIND", "kind": "NOMINAL",
         "defect": "good", "relativePath": image_path.name, "sourceSha256": source_sha256},
    ]
    manifest_path = tmp_path / "subset_manifest.json"
    manifest_path.write_text(json.dumps({
        "schemaVersion": "phone-dino.mvtec-ad-smoke/1.0",
        "authoritative": False,
        "manifestSha256": f"sha256:{'a' * 64}",
        "records": records,
    }), encoding="utf-8")
    return manifest_path, image_path


def test_iteration_rejects_anomalous_fit_or_tuning_records_before_feature_loading(tmp_path: Path) -> None:
    tool = _tool_module()
    manifest_path, _ = _write_normal_only_manifest(tmp_path, fit_kind="ANOMALY")
    with pytest.raises(tool.IterationError, match="FIT and THRESHOLD_TUNING records must be good nominal"):
        tool.load_frozen_records(manifest_path)


def test_iteration_rejects_non_good_nominal_fit_before_feature_loading(tmp_path: Path) -> None:
    tool = _tool_module()
    manifest_path, _ = _write_normal_only_manifest(tmp_path, fit_defect="crack")
    with pytest.raises(tool.IterationError, match="FIT and THRESHOLD_TUNING records must be good nominal"):
        tool.load_frozen_records(manifest_path)


def test_normal_only_loader_skips_blind_payload_before_label_or_mask_validation(tmp_path: Path) -> None:
    tool = _tool_module()
    manifest_path, _ = _write_normal_only_manifest(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    blind = document["records"][2]
    blind["kind"] = "ANOMALY"
    blind["defect"] = "forbidden-to-normal-only"
    blind["maskRelativePath"] = "..\\blind-mask.png"
    blind["maskSha256"] = "not-a-digest"
    manifest_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    _, normal_records = tool.load_frozen_records(manifest_path, normal_only=True)
    assert [record["caseId"] for record in normal_records] == ["capsule/fit/001", "capsule/tuning/001"]
    with pytest.raises(tool.IterationError):
        tool.load_frozen_records(manifest_path)


def test_normal_only_iteration_never_embeds_blind_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _tool_module()
    manifest_path, image_path = _write_normal_only_manifest(tmp_path)
    weights_path = tmp_path / "weights.pth"
    weights_path.write_bytes(b"offline-test-weights")

    class SpyEmbedder:
        embedded_images = 0

        def __init__(self, **_: object) -> None:
            pass

        def extract(self, images: list[Image.Image], *, feature_kind: str) -> list[object]:
            assert feature_kind == "global"
            type(self).embedded_images += len(images)
            return [np.asarray([1.0, 0.0], dtype=np.float32) for _ in images]

    monkeypatch.setattr(tool, "ResearchBatchEmbedder", SpyEmbedder)
    report = tool.run(
        manifest_path,
        tmp_path / "normal-only-report.json",
        model_repo=tmp_path,
        model_weights=weights_path,
        device="cpu",
        algorithm="global-knn",
        max_prototypes=1,
        top_k_patches=1,
        batch_size=4,
        prototype_block_size=1,
        include_pixel_metrics=False,
        normal_only=True,
    )
    assert SpyEmbedder.embedded_images == 2
    assert report["blindReporting"]["state"] == "NOT_RUN"
    assert [record["caseId"] for record in report["scores"]] == ["capsule/tuning/001"]
    evidence = report["normalOnlyEvidence"]
    assert evidence["featureInputCount"] == 2
    assert evidence["featureInputRoles"] == ["FIT", "THRESHOLD_TUNING"]
    assert evidence["featureInputKinds"] == ["NOMINAL"]
    assert evidence["blindFeatureInputCount"] == 0
    assert evidence["anomalyFeatureInputCount"] == 0
    assert evidence["reportedScoreRoles"] == ["THRESHOLD_TUNING"]
    assert evidence["reportedScoreKinds"] == ["NOMINAL"]
    assert evidence["calibrationScoreRoles"] == ["THRESHOLD_TUNING"]
    assert evidence["calibrationScoreKinds"] == ["NOMINAL"]
    assert evidence["normalInputIdentitySha256"].startswith("sha256:")
    assert [record["caseId"] for record in evidence["featureInputs"]] == ["capsule/fit/001", "capsule/tuning/001"]
    assert [record["caseId"] for record in evidence["calibrationInputs"]] == ["capsule/tuning/001"]
    assert evidence["calibrationInputIdentitySha256"].startswith("sha256:")
    assert evidence["originalTuningInputCount"] == 1
    assert evidence["originalTuningInputIdentitySha256"] == evidence["calibrationInputIdentitySha256"]
    assert report["schemaVersion"] == "phone-dino.mvtec-ad-iteration-report/1.4"
    assert report["algorithm"]["modelRepositorySha256"] == report["featureExtractor"]["modelRepositorySha256"]
    assert report["featureExtractorIdentitySha256"] == tool.canonical_json_sha256(report["featureExtractor"])
    assert report["execution"]["featureCacheSchemaVersion"] == "phone-dino.mvtec-ad-feature-cache/1.1"
    assert report["execution"]["phaseTimingsSeconds"]["cacheValidationSeconds"] >= 0.0
    assert report["execution"]["phaseTimingsSeconds"]["cacheWriteSeconds"] >= 0.0
    assert report["candidateConfiguration"] == {
        "algorithmId": "DINOV2_GLOBAL_NEAREST_NORMAL_COSINE_V1",
        "batchSize": 4,
    }
    assert report["candidateConfigurationSha256"] == tool.canonical_json_sha256(report["candidateConfiguration"])


@pytest.mark.parametrize("recipe_path", [RECIPE_V3_PATH, RECIPE_V5_PATH])
def test_normal_only_iteration_reports_all_four_generated_variant_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recipe_path: Path,
) -> None:
    tool = _tool_module()
    manifest_path, _ = _write_normal_only_manifest(tmp_path)
    augmentation_directory = tmp_path / "augmented"
    generate_normal_augmentations(
        manifest_path,
        recipe_path,
        augmentation_directory,
        variants_per_parent=4,
        repository_root=tmp_path / "repo",
    )
    model_repo = tmp_path / "model-repository"
    model_repo.mkdir()
    (model_repo / "hubconf.py").write_text("# fake model repository\n", encoding="utf-8")
    weights_path = tmp_path / "weights.pth"
    weights_path.write_bytes(b"offline-test-weights")

    class SpyEmbedder:
        def __init__(self, **_: object) -> None:
            pass

        def extract(self, images: list[Image.Image], *, feature_kind: str) -> list[object]:
            assert feature_kind == "global"
            return [np.asarray([1.0] * 384, dtype=np.float32) for _ in images]

    monkeypatch.setattr(tool, "ResearchBatchEmbedder", SpyEmbedder)
    report = tool.run(
        manifest_path,
        tmp_path / "normal-only-r4-report.json",
        model_repo=model_repo,
        model_weights=weights_path,
        device="cpu",
        algorithm="global-knn",
        max_prototypes=1,
        top_k_patches=1,
        batch_size=4,
        prototype_block_size=1,
        augmentation_manifest_path=augmentation_directory / "augmentation_manifest.json",
        include_pixel_metrics=False,
        normal_only=True,
    )
    assert report["augmentation"]["variantsPerParent"] == 4
    feature_inputs = report["normalOnlyEvidence"]["featureInputs"]
    assert {record["variantId"] for record in feature_inputs if not record["isAugmentation"]} == {None}
    assert {record["variantId"] for record in feature_inputs if record["isAugmentation"]} == {1, 2, 3, 4}
    calibration_scores = report["calibrationScores"]
    assert {record["variantId"] for record in calibration_scores if not record["isAugmentation"]} == {None}
    assert {record["variantId"] for record in calibration_scores if record["isAugmentation"]} == {1, 2, 3, 4}


def test_iteration_rejects_extractor_input_changes_during_model_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _tool_module()
    manifest_path, _ = _write_normal_only_manifest(tmp_path)
    weights_path = tmp_path / "weights.pth"
    weights_path.write_bytes(b"offline-test-weights")

    class SpyEmbedder:
        def __init__(self, **_: object) -> None:
            pass

        def extract(self, images: list[Image.Image], *, feature_kind: str) -> list[object]:
            return [np.asarray([1.0] * 384, dtype=np.float32) for _ in images]

    identities = [
        {"modelRepositorySha256": f"sha256:{'1' * 64}"},
        {"modelRepositorySha256": f"sha256:{'2' * 64}"},
    ]

    def changing_identity(**_: object) -> dict[str, object]:
        return identities.pop(0)

    monkeypatch.setattr(tool, "ResearchBatchEmbedder", SpyEmbedder)
    monkeypatch.setattr(tool, "feature_extractor_identity", changing_identity)
    output_path = tmp_path / "normal-only-report.json"
    with pytest.raises(tool.IterationError, match="changed while the model was loading"):
        tool.run(
            manifest_path,
            output_path,
            model_repo=tmp_path,
            model_weights=weights_path,
            device="cpu",
            algorithm="global-knn",
            max_prototypes=1,
            top_k_patches=1,
            batch_size=4,
            prototype_block_size=1,
            include_pixel_metrics=False,
            normal_only=True,
        )
    assert not output_path.exists()


def test_iteration_does_not_publish_cache_when_extractor_changes_after_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool_module()
    manifest_path, _ = _write_normal_only_manifest(tmp_path)
    weights_path = tmp_path / "weights.pth"
    weights_path.write_bytes(b"offline-test-weights")
    first_identity, _ = _feature_extractor_identity(tool, marker="1")
    changed_identity, _ = _feature_extractor_identity(tool, marker="7")

    class SpyEmbedder:
        def __init__(self, **_: object) -> None:
            pass

        def extract(self, images: list[Image.Image], *, feature_kind: str) -> list[object]:
            assert feature_kind == "global"
            return [np.asarray([1.0] * 384, dtype=np.float32) for _ in images]

    identities = [first_identity, dict(first_identity), changed_identity]

    def changing_identity(**_: object) -> dict[str, object]:
        return identities.pop(0)

    cache_path = tmp_path / "outside-cache"
    monkeypatch.setattr(tool, "ResearchBatchEmbedder", SpyEmbedder)
    monkeypatch.setattr(tool, "feature_extractor_identity", changing_identity)
    with pytest.raises(tool.IterationError, match="changed while features were extracted"):
        tool.run(
            manifest_path,
            tmp_path / "normal-only-report.json",
            model_repo=tmp_path,
            model_weights=weights_path,
            device="cpu",
            algorithm="global-knn",
            max_prototypes=1,
            top_k_patches=1,
            batch_size=4,
            prototype_block_size=1,
            feature_cache_path=cache_path,
            include_pixel_metrics=False,
            normal_only=True,
        )
    assert cache_path.is_dir()
    assert not list(cache_path.iterdir())


def test_normal_input_identity_is_stable_and_excludes_blind_or_anomalous_inputs() -> None:
    tool = _tool_module()
    records = [
        {"caseId": "tuning", "category": "tile", "role": "THRESHOLD_TUNING", "kind": "NOMINAL", "sourceSha256": f"sha256:{'2' * 64}"},
        {"caseId": "blind", "category": "tile", "role": "BLIND", "kind": "ANOMALY", "sourceSha256": f"sha256:{'3' * 64}"},
        {"caseId": "fit", "category": "tile", "role": "FIT", "kind": "NOMINAL", "sourceSha256": f"sha256:{'1' * 64}"},
    ]
    first = tool.normal_input_identity(records)
    second = tool.normal_input_identity(list(reversed(records)))
    assert first == second
    changed = [dict(record) for record in records]
    changed[2]["sourceSha256"] = f"sha256:{'4' * 64}"
    assert tool.normal_input_identity(changed) != first
