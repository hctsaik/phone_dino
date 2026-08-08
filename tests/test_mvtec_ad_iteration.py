from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _tool_module():
    path = Path(__file__).parents[1] / "tools" / "run_mvtec_ad_iteration.py"
    spec = importlib.util.spec_from_file_location("mvtec_ad_iteration", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_feature_cache_key_binds_source_model_and_preprocessing(tmp_path: Path) -> None:
    tool = _tool_module()
    record = {"sourceSha256": f"sha256:{'1' * 64}", "augmentationRecipeSha256": None}
    first = tool.FeatureCache(tmp_path / "outside", model_weights_sha256=f"sha256:{'2' * 64}", source_sha256=f"sha256:{'3' * 64}")
    changed_model = tool.FeatureCache(tmp_path / "outside-model", model_weights_sha256=f"sha256:{'4' * 64}", source_sha256=f"sha256:{'3' * 64}")
    changed_input = dict(record, sourceSha256=f"sha256:{'5' * 64}")
    assert first.key_for(record, "patch") != changed_model.key_for(record, "patch")
    assert first.key_for(record, "patch") != first.key_for(changed_input, "patch")
    assert first.key_for(record, "patch") != first.key_for(record, "global")


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


def _write_normal_only_manifest(tmp_path: Path, *, fit_kind: str = "NOMINAL") -> tuple[Path, Path]:
    image_path = tmp_path / "normal.png"
    Image.new("RGB", (32, 32), (90, 130, 180)).save(image_path)
    source_sha256 = f"sha256:{hashlib.sha256(image_path.read_bytes()).hexdigest()}"
    records = [
        {"caseId": "capsule/fit/001", "category": "capsule", "role": "FIT", "kind": fit_kind,
         "defect": "good", "relativePath": image_path.name, "sourceSha256": source_sha256},
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
    with pytest.raises(tool.IterationError, match="FIT and THRESHOLD_TUNING records must be nominal"):
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
