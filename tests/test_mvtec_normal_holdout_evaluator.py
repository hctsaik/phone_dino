from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from phone_dino import mvtec_fresh_fit_augmentation as augmentation
from phone_dino import mvtec_normal_holdout as holdout
from phone_dino import mvtec_normal_holdout_evaluator as evaluator


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def _write_image(path: Path, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (31, 25), (index * 41 % 255, index * 73 % 255, index * 107 % 255)).save(path, format="PNG")


def _fresh_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Path]]:
    source_root = tmp_path / "source_bytes"
    assignments = (
        ("FIT", 0),
        ("FIT", 1),
        ("THRESHOLD_TUNING", 2),
        ("NORMAL_SELECTION", 3),
        ("NORMAL_CONFIRMATION", 4),
        ("RESERVE_UNTOUCHED", 5),
    )
    records: list[dict] = []
    paths: dict[str, Path] = {}
    for partition, index in assignments:
        relative = Path("images") / f"{index}.png"
        path = source_root / relative
        _write_image(path, index)
        source_sha256 = holdout.sha256_file(path)
        records.append({
            "caseId": f"mvtec-ad/capsule/train-good/{source_sha256[7:]}",
            "category": "capsule",
            "relativePath": relative.as_posix(),
            "sourceSha256": source_sha256,
            "sourceGroupId": f"CONTENT_SHA256:{source_sha256[7:]}",
            "acquisitionStratum": "OFFICIAL_MVTEC_TRAIN_GOOD",
            "sourceRemotePath": f"data/data_6/{index}.png",
            "expectedRemoteSha256": source_sha256,
            "expectedRemoteBytes": path.stat().st_size,
            "kind": "NOMINAL",
            "defect": "good",
            "partition": partition,
        })
        paths[partition] = path
    records.sort(key=lambda record: record["caseId"])
    document = {
        "schemaVersion": holdout.NORMAL_HOLDOUT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": holdout.HOLDOUT_PURPOSE,
        "blindPolicy": holdout.HOLDOUT_BLIND_POLICY,
        "sourcePoolFileSha256": "sha256:" + "1" * 64,
        "sourcePoolDeclaredSha256": "sha256:" + "2" * 64,
        "historicalLedgerFileSha256": "sha256:" + "3" * 64,
        "historicalLedgerDeclaredSha256": "sha256:" + "4" * 64,
        "planFileSha256": "sha256:" + "5" * 64,
        "planDeclaredSha256": "sha256:" + "6" * 64,
        "historyExclusion": {
            "algorithm": holdout.HISTORY_EXCLUSION_ALGORITHM,
            "matchedHistoricalSourceCount": 0,
            "excludedSourceGroupCount": 0,
            "eligibleSourceCount": len(records),
            "eligibleSourceIdentitySha256": holdout.canonical_json_sha256([]),
        },
        "records": records,
        "developmentIdentitySha256": holdout._holdout_partition_identity(records, {"FIT", "THRESHOLD_TUNING"}),
        "normalSelectionIdentitySha256": holdout._holdout_partition_identity(records, {"NORMAL_SELECTION"}),
        "normalConfirmationIdentitySha256": holdout._holdout_partition_identity(records, {"NORMAL_CONFIRMATION"}),
        "reserveUntouchedIdentitySha256": holdout._holdout_partition_identity(records, {"RESERVE_UNTOUCHED"}),
    }
    document["normalHoldoutManifestSha256"] = holdout._document_digest(document, "normalHoldoutManifestSha256")
    holdout_path = tmp_path / "holdout" / "normal_holdout.json"
    _write_json(holdout_path, document)
    recipe_path = augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json"
    augmentation_root = tmp_path / "augmentation"
    augmentation.generate_fresh_fit_augmentations(
        holdout_path,
        source_root,
        recipe_path,
        augmentation_root,
        variants_per_parent=2,
    )
    return holdout_path, source_root, augmentation_root / "augmentation_manifest.json", paths


class _FakeEmbedder:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def extract_patches(self, images: list[Image.Image]) -> list[object]:
        result: list[object] = []
        base = np.linspace(1.0, 2.0, 384, dtype=np.float32)
        for image in images:
            signal = float(np.asarray(image, dtype=np.uint8).mean()) / 255.0
            result.append(np.tile(base + signal, (256, 1)).astype(np.float32))
        return result


def _identity_factory(**_kwargs: object) -> dict:
    return {"fixture": "feature-extractor"}


def _configuration(identifier: str = "patch-1024") -> dict:
    return {
        "id": identifier,
        "algorithmId": evaluator.ALGORITHM_ID,
        "memoryBankSelection": evaluator.PROTOTYPE_SELECTION,
        "maxPrototypePatches": 1024,
        "topKMostAnomalousPatches": 5,
        "prototypeBlockSize": 64,
        "batchSize": 2,
    }


def test_development_evaluator_opens_only_fit_and_raw_tuning(tmp_path: Path) -> None:
    holdout_path, source_root, augmentation_manifest, paths = _fresh_fixture(tmp_path)
    # Development must never decode these partitions. Their corruption would
    # fail a broad loader but must not affect this phase-safe evaluator.
    for partition in ("NORMAL_SELECTION", "NORMAL_CONFIRMATION", "RESERVE_UNTOUCHED"):
        paths[partition].write_bytes(b"must-not-be-opened")
    output_path = tmp_path / "reports" / "patch-1024.json"
    report = evaluator.run_development_evaluation(
        holdout_path,
        augmentation_manifest,
        augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json",
        output_path,
        source_root=source_root,
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        device="cpu",
        candidate_configuration=_configuration(),
        embedder_factory=_FakeEmbedder,
        identity_factory=_identity_factory,
    )
    assert output_path.is_file()
    assert report["phase"] == evaluator.DEVELOPMENT_PHASE
    assert report["normalOnlyEvidence"]["featureInputPartitions"] == ["FIT", "THRESHOLD_TUNING"]
    assert report["normalOnlyEvidence"]["blindFeatureInputCount"] == 0
    assert report["normalOnlyEvidence"]["anomalyFeatureInputCount"] == 0
    assert {record["partition"] for record in report["calibrationInputs"]} == {"THRESHOLD_TUNING"}
    assert all(not record["isAugmentation"] for record in report["calibrationInputs"])
    assert len(report["calibrationScores"]) == 1
    assert report["categories"]["capsule"]["fitOriginalCount"] == 2
    assert report["categories"]["capsule"]["fitAugmentedCount"] == 4


def test_candidate_configuration_is_closed() -> None:
    invalid = _configuration()
    invalid["unknown"] = True
    with pytest.raises(evaluator.NormalHoldoutEvaluatorError, match="unsupported fields"):
        evaluator.validate_candidate_configuration(invalid)
    invalid = _configuration()
    invalid["id"] = "PATCH-1024"
    with pytest.raises(evaluator.NormalHoldoutEvaluatorError, match="lowercase"):
        evaluator.validate_candidate_configuration(invalid)
