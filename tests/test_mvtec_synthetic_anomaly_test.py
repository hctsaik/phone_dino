from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from phone_dino import mvtec_synthetic_anomaly_test as synthetic


def _digest(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _write_solid(path: Path, colour: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), colour).save(path, format="PNG")
    return synthetic.sha256_file(path)


def _parents(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for category_ordinal, category in enumerate(synthetic.SYNTHETIC_ANOMALY_TEST_V1_CATEGORIES):
        for ordinal in range(12):
            path = root / category / f"parent-{ordinal:02d}.png"
            source = _write_solid(path, ((category_ordinal * 60 + ordinal * 7) % 255, 40, 90))
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


def _synthetic_children(root: Path, parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split = synthetic.split_synthetic_fit_parents(parents)
    children: list[dict[str, Any]] = []
    families = ("LOCAL_SCRATCH", "LOCAL_SPOT", "LOCAL_OCCLUSION")
    for parent in split["SYNTHETIC_QUERY"]:
        with Image.open(parent["imagePath"]) as opened:
            source_image = opened.convert("RGB")
        for variant_id, family in enumerate(families, start=1):
            image = source_image.copy()
            draw = ImageDraw.Draw(image)
            draw.rectangle((10 + variant_id, 10, 20, 20), fill=(0, 0, 0))
            path = root / parent["category"] / f"{parent['caseId'].split('/')[-1]}-{variant_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG")
            children.append({
                "caseId": f"synthetic/{parent['caseId']}/{variant_id}",
                "parentCaseId": parent["caseId"],
                "parentSourceSha256": parent["sourceSha256"],
                "sourceGroupId": parent["sourceGroupId"],
                "category": parent["category"],
                "parentPartition": "FIT",
                "syntheticTestRole": "QUERY",
                "syntheticLabel": "SYNTHETIC_ANOMALY",
                "syntheticDefectFamily": family,
                "variantId": variant_id,
                "relativePath": path.name,
                "sourceSha256": synthetic.sha256_file(path),
                "parameters": {},
                "outputEncoding": {},
                "imagePath": path,
            })
    return children


class _FakeEmbedder:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def extract_patches(self, images: list[Image.Image]) -> list[object]:
        result: list[object] = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
            spatial_signal = float(array.mean(axis=2).std()) / 255.0
            base = np.array([1.0, spatial_signal, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
            result.append(np.stack([base + index * 0.0001 for index in range(16)], axis=0))
        return result


def _identity_factory(**_kwargs: object) -> dict[str, Any]:
    return {"synthetic": "fixture-identity"}


def test_parent_split_is_stable_and_disjoint(tmp_path: Path) -> None:
    parents = _parents(tmp_path / "source")
    first = synthetic.split_synthetic_fit_parents(parents)
    second = synthetic.split_synthetic_fit_parents(list(reversed(parents)))
    assert {
        role: [record["caseId"] for record in values]
        for role, values in first.items()
    } == {
        role: [record["caseId"] for record in values]
        for role, values in second.items()
    }
    groups = {role: {record["sourceGroupId"] for record in values} for role, values in first.items()}
    assert not groups["SYNTHETIC_PROTOTYPE"] & groups["SYNTHETIC_CALIBRATION"]
    assert not groups["SYNTHETIC_PROTOTYPE"] & groups["SYNTHETIC_QUERY"]
    assert not groups["SYNTHETIC_CALIBRATION"] & groups["SYNTHETIC_QUERY"]
    assert {role: len(records) for role, records in first.items()} == {
        "SYNTHETIC_PROTOTYPE": 18,
        "SYNTHETIC_CALIBRATION": 6,
        "SYNTHETIC_QUERY": 12,
    }


def test_synthetic_metrics_use_a_strict_threshold() -> None:
    metrics = synthetic.synthetic_confusion_metrics([
        {"syntheticLabel": "SYNTHETIC_ANOMALY", "score": 0.51},
        {"syntheticLabel": "SYNTHETIC_ANOMALY", "score": 0.50},
        {"syntheticLabel": "SYNTHETIC_NOMINAL", "score": 0.50},
        {"syntheticLabel": "SYNTHETIC_NOMINAL", "score": 0.51},
    ], threshold=0.50)
    assert metrics == {
        "syntheticTP": 1,
        "syntheticFP": 1,
        "syntheticFN": 1,
        "syntheticTN": 1,
        "syntheticPrecision": 0.5,
        "syntheticRecall": 0.5,
        "syntheticF1": 0.5,
    }


def test_synthetic_test_requests_fit_only_and_reports_scoped_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    parents = _parents(source_root)
    children = _synthetic_children(tmp_path / "synthetic", parents)
    calls: list[set[str]] = []

    def safe_loader(*_args: object, **kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        requested = set(kwargs["partitions"])
        calls.append(requested)
        assert requested == {"FIT"}
        return _envelope(), _digest("envelope-file"), [dict(record) for record in parents]

    def synthetic_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        return {
            "augmentationManifestSha256": _digest("augmentation-declared"),
            "recipeFileSha256": _digest("recipe-file"),
        }, _digest("augmentation-file"), [dict(record) for record in children]

    monkeypatch.setattr(synthetic.successor, "load_successor_safe_normal_inputs", safe_loader)
    monkeypatch.setattr(synthetic.augmentation, "load_validated_synthetic_anomaly_augmentations", synthetic_loader)
    output = tmp_path / "report" / "synthetic.json"
    report = synthetic.run_synthetic_anomaly_test(
        tmp_path / "chain" / "holdout.json",
        tmp_path / "chain" / "contract.json",
        tmp_path / "chain" / "plan.json",
        tmp_path / "chain" / "envelope.json",
        tmp_path / "synthetic" / "augmentation_manifest.json",
        output,
        source_root=source_root,
        recipe_path=tmp_path / "recipe.json",
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_FakeEmbedder,
        identity_factory=_identity_factory,
    )
    assert calls == [{"FIT"}]
    assert output.is_file()
    assert report["syntheticOnly"] is True
    assert report["metricScope"] == "SYNTHETIC_RENDERING_DISCRIMINATION_ONLY"
    assert report["realAnomalyPerformance"] == "NOT_ESTIMATED"
    assert report["forbiddenUses"] == synthetic.SYNTHETIC_ANOMALY_TEST_V1_FORBIDDEN_USES
    assert report["aggregate"] == {
        "syntheticTP": 36,
        "syntheticFP": 0,
        "syntheticFN": 0,
        "syntheticTN": 12,
        "syntheticPrecision": 1.0,
        "syntheticRecall": 1.0,
        "syntheticF1": 1.0,
    }
    assert all(value["querySyntheticAnomalyCount"] == 12 for value in report["categories"].values())
    assert all(value["queryNominalParentCount"] == 4 for value in report["categories"].values())


def test_existing_output_rejects_before_fit_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "report.json"
    output.write_text("already exists", encoding="utf-8")
    calls: list[str] = []

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        calls.append("FIT")
        raise AssertionError("an existing output must fail before a FIT image loader is called")

    monkeypatch.setattr(synthetic.successor, "load_successor_safe_normal_inputs", forbidden_loader)
    with pytest.raises(synthetic.SyntheticAnomalyTestError, match="already exists"):
        synthetic.run_synthetic_anomaly_test(
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
