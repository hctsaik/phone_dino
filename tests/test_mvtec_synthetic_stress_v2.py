from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from phone_dino import mvtec_synthetic_anomaly_stress_v2 as generator
from phone_dino import mvtec_synthetic_stress_v2 as stress


def _digest(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _write_image(path: Path, colour: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), colour)
    draw = ImageDraw.Draw(image)
    draw.line((2, 2, 42, 37), fill=((colour[0] + 31) % 255, (colour[1] + 47) % 255, (colour[2] + 73) % 255), width=2)
    image.save(path, format="PNG")
    return stress.sha256_file(path)


def _parents(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for category_index, category in enumerate(stress.SYNTHETIC_STRESS_V2_CATEGORIES):
        for ordinal in range(12):
            path = root / category / f"parent-{ordinal:02d}.png"
            source = _write_image(
                path,
                (
                    (category_index * 70 + ordinal * 13 + 10) % 255,
                    (ordinal * 29 + 40) % 255,
                    (category_index * 41 + ordinal * 17 + 80) % 255,
                ),
            )
            records.append({
                "caseId": f"successor/{category}/fit/{ordinal:02d}",
                "category": category,
                "partition": "FIT",
                "kind": "NOMINAL",
                "defect": "good",
                "sourceSha256": source,
                "sourceGroupId": f"CONTENT_SHA256:{source[7:]}",
                "imagePath": path,
            })
    return records


def _envelope() -> dict[str, Any]:
    return {
        "parentEvidence": {
            "holdoutManifestFileSha256": _digest("holdout-file"),
            "holdoutManifestDeclaredSha256": _digest("holdout-declared"),
            "selectionContractFileSha256": _digest("contract-file"),
            "selectionContractDeclaredSha256": _digest("contract-declared"),
            "parentNormalConfirmationIdentitySha256": _digest("confirmation"),
        },
        "sealFileSha256": _digest("seal-file"),
        "sealDeclaredSha256": _digest("seal-declared"),
        "planFileSha256": _digest("plan-file"),
        "planDeclaredSha256": _digest("plan-declared"),
        "successorEnvelopeSha256": _digest("envelope-declared"),
        "successorPartitionIdentities": {"FIT": _digest("fit-identity")},
    }


def _manifest() -> dict[str, Any]:
    return {
        "schemaVersion": "phone-dino.mvtec-ad-synthetic-only-stress-augmentation/2.0",
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "postV1Exploratory": True,
        "comparisonOrPromotionAllowed": False,
        "parentPartition": "FIT",
        "inputPolicy": stress.SYNTHETIC_STRESS_V2_INPUT_POLICY,
        "blindPolicy": stress.SYNTHETIC_STRESS_V2_BLIND_POLICY,
        "resultLabel": stress.SYNTHETIC_STRESS_V2_RESULT_LABEL,
        "parentSplitAlgorithm": stress.SYNTHETIC_STRESS_V2_PARENT_SPLIT_ALGORITHM,
        "parentSplitCountsPerCategory": {"PROTOTYPE": 6, "CALIBRATION": 2, "QUERY": 4},
        "variantsPerParent": 9,
        "augmentationManifestSha256": _digest("augmentation-declared"),
        "recipeFileSha256": _digest("recipe-file"),
    }


def _children(root: Path, parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split = stress.build_fixed_parent_split(parents)
    children: list[dict[str, Any]] = []
    variants = [
        (level, family)
        for level in stress.SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS
        for family in stress.SYNTHETIC_STRESS_V2_FAMILIES
    ]
    variant_id = {pair: index for index, pair in enumerate(variants, start=1)}
    for parent in split["SYNTHETIC_QUERY"]:
        with Image.open(parent["imagePath"]) as opened:
            source = opened.convert("RGB")
        for level_index, level in enumerate(stress.SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS, start=1):
            for family_index, family in enumerate(stress.SYNTHETIC_STRESS_V2_FAMILIES, start=1):
                identifier = variant_id[level, family]
                image = source.copy()
                draw = ImageDraw.Draw(image)
                position = 3 + family_index * 5
                size = 2 + level_index * 4
                draw.rectangle((position, position, position + size, position + size), fill=(0, 0, 0))
                path = root / parent["category"] / f"{parent['caseId'].split('/')[-1]}-{identifier}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, format="PNG")
                children.append({
                    "caseId": f"synthetic-v2/{parent['caseId']}/{identifier}",
                    "parentCaseId": parent["caseId"],
                    "parentSourceSha256": parent["sourceSha256"],
                    "sourceGroupId": parent["sourceGroupId"],
                    "category": parent["category"],
                    "parentPartition": "FIT",
                    "syntheticTestRole": "QUERY",
                    "syntheticLabel": "SYNTHETIC_STIMULUS",
                    "syntheticDefectFamily": family,
                    "renderIntensityLevel": level,
                    "variantId": identifier,
                    "relativePath": path.name,
                    "sourceSha256": stress.sha256_file(path),
                    "parameters": {},
                    "outputEncoding": {},
                    "imagePath": path,
                })
    return sorted(children, key=lambda record: str(record["caseId"]))


class _FakeEmbedder:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def extract_patches(self, images: list[Image.Image]) -> list[object]:
        result: list[object] = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
            mean = array.mean(axis=(0, 1)) / 255.0
            spatial_signal = float(array.mean(axis=2).std()) / 255.0
            base = np.array([1.0, spatial_signal, *mean, 0.2, 0.3, 0.4], dtype=np.float32)
            result.append(np.stack([base + index * 0.0001 for index in range(16)], axis=0))
        return result


def _identity_factory(**_kwargs: object) -> dict[str, Any]:
    return {"synthetic": "fixture-identity"}


def _install_safe_loader(
    monkeypatch: pytest.MonkeyPatch,
    parents: list[dict[str, Any]],
    calls: list[set[str]],
) -> None:
    def safe_loader(*_args: object, **kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        requested = set(kwargs["partitions"])
        calls.append(requested)
        assert requested == {"FIT"}
        return _envelope(), _digest("envelope-file"), [dict(record) for record in parents]

    monkeypatch.setattr(stress.successor, "load_successor_safe_normal_inputs", safe_loader)


def test_fixed_parent_split_is_stable_and_disjoint(tmp_path: Path) -> None:
    parents = _parents(tmp_path / "source")
    first = stress.build_fixed_parent_split(parents)
    second = stress.build_fixed_parent_split(list(reversed(parents)))
    assert {
        role: [record["caseId"] for record in values]
        for role, values in first.items()
    } == {
        role: [record["caseId"] for record in values]
        for role, values in second.items()
    }
    assert {role: len(values) for role, values in first.items()} == {
        "SYNTHETIC_PROTOTYPE": 18,
        "SYNTHETIC_CALIBRATION": 6,
        "SYNTHETIC_QUERY": 12,
    }
    groups = {role: {record["sourceGroupId"] for record in values} for role, values in first.items()}
    assert not groups["SYNTHETIC_PROTOTYPE"] & groups["SYNTHETIC_CALIBRATION"]
    assert not groups["SYNTHETIC_PROTOTYPE"] & groups["SYNTHETIC_QUERY"]
    assert not groups["SYNTHETIC_CALIBRATION"] & groups["SYNTHETIC_QUERY"]


def test_raw_calibration_uses_maximum_per_category() -> None:
    rows = [
        {"category": category, "score": value}
        for category, values in {
            "capsule": (0.2, 0.5),
            "metal_nut": (0.1, 0.4),
            "tile": (0.3, 0.7),
        }.items()
        for value in values
    ]
    assert stress.calibrate_raw_thresholds(rows) == {"capsule": 0.5, "metal_nut": 0.4, "tile": 0.7}


def test_v2_response_run_is_fit_only_threshold_first_and_response_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = _parents(tmp_path / "source")
    children = _children(tmp_path / "package", parents)
    calls: list[set[str]] = []
    events: list[str] = []
    _install_safe_loader(monkeypatch, parents, calls)
    original_calibrate = stress.calibrate_raw_thresholds

    def calibrate(scores: list[dict[str, Any]]) -> dict[str, float]:
        events.append("threshold")
        return original_calibrate(scores)

    def package_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        assert events == ["threshold"]
        events.append("package")
        return _manifest(), _digest("augmentation-file"), [dict(record) for record in children]

    monkeypatch.setattr(stress, "calibrate_raw_thresholds", calibrate)
    monkeypatch.setattr(stress.augmentation, "load_validated_synthetic_stress_v2", package_loader)
    output = tmp_path / "report" / "synthetic-stress-v2.json"
    report = stress.run_synthetic_stress_v2(
        tmp_path / "chain" / "holdout.json",
        tmp_path / "chain" / "contract.json",
        tmp_path / "chain" / "plan.json",
        tmp_path / "chain" / "envelope.json",
        tmp_path / "package" / "augmentation_manifest.json",
        output,
        source_root=tmp_path / "source",
        recipe_path=tmp_path / "recipe.json",
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_FakeEmbedder,
        identity_factory=_identity_factory,
    )
    assert calls == [{"FIT"}]
    assert events == ["threshold", "package"]
    assert output.is_file()
    assert report["syntheticOnly"] is True
    assert report["postV1Exploratory"] is True
    assert report["comparisonOrPromotionAllowed"] is False
    assert report["metricScope"] == "SYNTHETIC_STIMULUS_RESPONSE_ONLY"
    assert report["realAnomalyPerformance"] == "NOT_ESTIMATED"
    assert report["testConfiguration"]["rawCalibrationThresholdEstablishedBeforePackageLoad"] is True
    assert report["testConfiguration"]["rawCalibrationThresholdEstablishedBeforePackageScoring"] is True
    assert report["aggregate"]["responseCounts"] == {
        "rawQueryCount": 12,
        "rawQueryAboveThresholdCount": report["aggregate"]["responseCounts"]["rawQueryAboveThresholdCount"],
        "syntheticStimulusCount": 108,
        "syntheticStimulusAboveThresholdCount": report["aggregate"]["responseCounts"]["syntheticStimulusAboveThresholdCount"],
    }
    assert report["aggregate"]["pairedScoreDeltaSummary"]["pairCount"] == 108
    assert len(report["rawQueryScores"]) == 12
    assert len(report["stimulusScores"]) == 108
    assert {record["renderIntensityLevel"] for record in report["stimulusScores"]} == set(stress.SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS)
    assert {record["syntheticDefectFamily"] for record in report["stimulusScores"]} == set(stress.SYNTHETIC_STRESS_V2_FAMILIES)
    assert all(values["responseCounts"]["syntheticStimulusCount"] == 36 for values in report["categories"].values())
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in ("syntheticTP", "syntheticFP", "syntheticFN", "syntheticTN", "syntheticPrecision", "syntheticRecall", "syntheticF1", "AUROC"):
        assert forbidden not in serialized


def test_v2_evaluator_consumes_a_real_validated_generator_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = _parents(tmp_path / "source")
    calls: list[set[str]] = []
    _install_safe_loader(monkeypatch, parents, calls)
    chain_paths = [
        tmp_path / "chain" / "holdout.json",
        tmp_path / "chain" / "contract.json",
        tmp_path / "chain" / "plan.json",
        tmp_path / "chain" / "envelope.json",
    ]
    for path in chain_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    package_dir = tmp_path / "real-package"
    generator.generate_synthetic_anomaly_stress_v2(
        chain_paths[0],
        chain_paths[1],
        chain_paths[2],
        chain_paths[3],
        source_root=tmp_path / "source",
        recipe_path=generator.REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
        output_dir=package_dir,
    )
    report = stress.run_synthetic_stress_v2(
        chain_paths[0],
        chain_paths[1],
        chain_paths[2],
        chain_paths[3],
        package_dir / "augmentation_manifest.json",
        tmp_path / "report" / "response.json",
        source_root=tmp_path / "source",
        recipe_path=generator.REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_FakeEmbedder,
        identity_factory=_identity_factory,
    )
    assert calls == [{"FIT"}, {"FIT"}, {"FIT"}]
    assert report["augmentationSchemaVersion"] == generator.SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA
    assert report["aggregate"]["responseCounts"]["syntheticStimulusCount"] == 108


def test_v2_package_requires_every_query_family_level_combination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = _parents(tmp_path / "source")
    children = _children(tmp_path / "package", parents)

    def package_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        return _manifest(), _digest("augmentation-file"), children[:-1]

    monkeypatch.setattr(stress.augmentation, "load_validated_synthetic_stress_v2", package_loader)
    query_parents = stress.build_fixed_parent_split(parents)["SYNTHETIC_QUERY"]
    with pytest.raises(stress.SyntheticStressV2Error, match="cover every query parent, family, and render level"):
        stress.load_and_validate_v2_package(
            tmp_path / "package" / "augmentation_manifest.json",
            tmp_path / "chain" / "holdout.json",
            tmp_path / "chain" / "contract.json",
            tmp_path / "chain" / "plan.json",
            tmp_path / "chain" / "envelope.json",
            source_root=tmp_path / "source",
            recipe_path=tmp_path / "recipe.json",
            query_parents=query_parents,
        )


def test_existing_output_rejects_before_any_fit_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "report.json"
    output.write_text("already exists", encoding="utf-8")
    calls: list[str] = []

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        calls.append("FIT")
        raise AssertionError("existing output must fail before the FIT loader")

    monkeypatch.setattr(stress.successor, "load_successor_safe_normal_inputs", forbidden_loader)
    with pytest.raises(stress.SyntheticStressV2Error, match="already exists"):
        stress.run_synthetic_stress_v2(
            tmp_path / "holdout.json",
            tmp_path / "contract.json",
            tmp_path / "plan.json",
            tmp_path / "envelope.json",
            tmp_path / "augmentation_manifest.json",
            output,
            source_root=tmp_path,
            recipe_path=tmp_path / "recipe.json",
            model_repo=tmp_path / "model",
            model_weights=tmp_path / "weights.pth",
            identity_factory=_identity_factory,
        )
    assert calls == []


def test_no_v1_report_api_or_classification_summary_fields() -> None:
    signature = inspect.signature(stress.run_synthetic_stress_v2)
    assert "v1" not in " ".join(signature.parameters).lower()
    source = inspect.getsource(stress)
    assert "mvtec_synthetic_anomaly_test" not in source
    assert "synthetic_confusion_metrics" not in source
    assert "syntheticPrecision" not in source
