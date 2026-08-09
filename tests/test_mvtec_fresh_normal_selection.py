from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from phone_dino import mvtec_fresh_fit_augmentation as augmentation
from phone_dino import mvtec_fresh_normal_selection as selection
from phone_dino import mvtec_normal_holdout as holdout
from phone_dino import mvtec_normal_holdout_evaluator as evaluator


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _write_image(path: Path, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (31, 25), (index * 41 % 255, index * 73 % 255, index * 107 % 255)).save(path, format="PNG")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Path]]:
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
        image_path = source_root / relative
        _write_image(image_path, index)
        source_sha256 = holdout.sha256_file(image_path)
        records.append({
            "caseId": f"mvtec-ad/capsule/train-good/{source_sha256[7:]}",
            "category": "capsule",
            "relativePath": relative.as_posix(),
            "sourceSha256": source_sha256,
            "sourceGroupId": f"CONTENT_SHA256:{source_sha256[7:]}",
            "acquisitionStratum": "OFFICIAL_MVTEC_TRAIN_GOOD",
            "sourceRemotePath": f"data/data_6/{index}.png",
            "expectedRemoteSha256": source_sha256,
            "expectedRemoteBytes": image_path.stat().st_size,
            "kind": "NOMINAL",
            "defect": "good",
            "partition": partition,
        })
        paths[partition] = image_path
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
    augmentation_root = tmp_path / "augmentation"
    recipe_path = augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json"
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
    return {"fixture": "same-feature-extractor"}


def _configuration(identifier: str, patches: int) -> dict:
    return {
        "id": identifier,
        "algorithmId": evaluator.ALGORITHM_ID,
        "memoryBankSelection": evaluator.PROTOTYPE_SELECTION,
        "maxPrototypePatches": patches,
        "topKMostAnomalousPatches": 5,
        "prototypeBlockSize": 64,
        "batchSize": 2,
    }


def _development_reports(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, Path]]:
    holdout_path, source_root, augmentation_manifest, paths = _fixture(tmp_path)
    recipe_path = augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json"
    reports: list[Path] = []
    for identifier, patches in (("fresh-patch-1024", 1024), ("fresh-patch-2048", 2048)):
        output = tmp_path / "development_reports" / f"{identifier}.json"
        evaluator.run_development_evaluation(
            holdout_path,
            augmentation_manifest,
            recipe_path,
            output,
            source_root=source_root,
            model_repo=tmp_path / "model",
            model_weights=tmp_path / "weights.pth",
            device="cpu",
            candidate_configuration=_configuration(identifier, patches),
            embedder_factory=_FakeEmbedder,
            identity_factory=_identity_factory,
        )
        reports.append(output)
    return holdout_path, augmentation_manifest, reports[0], reports[1], paths


def _gates() -> dict[str, float]:
    return {
        "maxAboveThresholdRate": 0.25,
        "maxP95ScoreMinusThreshold": 0.05,
        "maxMaximumScoreMinusThreshold": 0.10,
    }


def _objective() -> dict:
    return json.loads(json.dumps(selection.FRESH_NORMAL_SELECTION_OBJECTIVE))


def test_contract_and_fixed_claim_are_json_only_and_bind_the_frozen_inputs(tmp_path: Path) -> None:
    holdout_path, augmentation_manifest, first_report, second_report, paths = _development_reports(tmp_path)
    # If contract creation ever opens a held-out source image, this corruption
    # would fail it.  Contract/claim creation only reads closed JSON documents.
    paths["NORMAL_SELECTION"].write_bytes(b"selection-must-not-be-opened")
    paths["NORMAL_CONFIRMATION"].write_bytes(b"confirmation-must-not-be-opened")
    paths["RESERVE_UNTOUCHED"].write_bytes(b"reserve-must-not-be-opened")
    contract_path = tmp_path / "selection" / "fresh_normal_selection_contract.json"
    contract = selection.create_fresh_normal_selection_contract(
        holdout_path,
        augmentation_manifest,
        [second_report, first_report],
        contract_path,
        selection_gates=_gates(),
        selection_objective=_objective(),
    )
    assert [entry["candidateId"] for entry in contract["candidateReports"]] == [
        "fresh-patch-1024",
        "fresh-patch-2048",
    ]
    assert set(contract["candidateReports"][0]["thresholds"]) == {"capsule"}
    assert contract["candidateReports"][0]["thresholdsIdentitySha256"] == selection.canonical_json_sha256(
        contract["candidateReports"][0]["thresholds"]
    )
    assert contract["normalSelectionInputs"][0]["partition"] == "NORMAL_SELECTION"
    assert contract["normalConfirmationInputs"][0]["partition"] == "NORMAL_CONFIRMATION"
    assert "relativePath" not in json.dumps(contract)
    loaded_contract, contract_file_sha256 = selection.load_validated_fresh_selection_contract(contract_path)
    assert loaded_contract == contract
    assert contract_file_sha256 == selection.sha256_file(contract_path)
    loaded_report, report_file_sha256 = selection.load_validated_fresh_development_report(first_report)
    assert loaded_report["developmentReportSha256"]
    assert report_file_sha256 == selection.sha256_file(first_report)

    claim = selection.create_fresh_normal_selection_claim(contract_path)
    claim_path = selection.fresh_selection_claim_path(contract_path)
    assert claim_path.is_file()
    loaded_claim, claim_file_sha256 = selection.load_validated_fresh_selection_claim_for_contract(contract_path)
    assert loaded_claim == claim
    assert claim_file_sha256 == selection.sha256_file(claim_path)
    selection.validate_fresh_selection_claim_binding(
        loaded_contract,
        contract_file_sha256,
        loaded_claim,
        contract_path=contract_path,
        claim_path=claim_path,
    )
    with pytest.raises(selection.FreshNormalSelectionError, match="already exists"):
        selection.create_fresh_normal_selection_claim(contract_path)


def test_contract_rejects_missing_gates_and_duplicate_json_keys(tmp_path: Path) -> None:
    holdout_path, augmentation_manifest, first_report, second_report, _ = _development_reports(tmp_path)
    with pytest.raises(selection.FreshNormalSelectionError, match="missing required fields"):
        selection.create_fresh_normal_selection_contract(
            holdout_path,
            augmentation_manifest,
            [first_report, second_report],
            tmp_path / "selection" / "bad-gates.json",
            selection_gates={},
            selection_objective=_objective(),
        )
    duplicate = tmp_path / "selection" / "duplicate-contract.json"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_text('{"schemaVersion":"one","schemaVersion":"two"}', encoding="utf-8")
    with pytest.raises(selection.FreshNormalSelectionError, match="duplicate JSON key"):
        selection.load_validated_fresh_selection_contract(duplicate)
