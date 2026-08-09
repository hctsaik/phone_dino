from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from phone_dino import mvtec_fresh_fit_augmentation as augmentation
from phone_dino import mvtec_fresh_normal_observation as observation
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


class _ConstantEmbedder:
    calls: list[list[int]] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def extract_patches(self, images: list[Image.Image]) -> list[object]:
        self.calls.append([int(np.asarray(image, dtype=np.uint8)[0, 0, 0]) for image in images])
        base = np.linspace(1.0, 2.0, 384, dtype=np.float32)
        return [np.tile(base, (256, 1)).astype(np.float32) for _ in images]


def _identity_factory(**_kwargs: object) -> dict:
    return {"fixture": "same-feature-extractor"}


def _configuration(identifier: str, patches: int, *, batch_size: int = 2) -> dict:
    return {
        "id": identifier,
        "algorithmId": evaluator.ALGORITHM_ID,
        "memoryBankSelection": evaluator.PROTOTYPE_SELECTION,
        "maxPrototypePatches": patches,
        "topKMostAnomalousPatches": 5,
        "prototypeBlockSize": 64,
        "batchSize": batch_size,
    }


def _boundary(tmp_path: Path, *, second_batch_size: int = 2) -> tuple[Path, Path, Path, Path, dict[str, Path], dict]:
    holdout_path, source_root, augmentation_manifest, paths = _fixture(tmp_path)
    recipe_path = augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json"
    reports: list[Path] = []
    for identifier, patches, batch_size in (
        ("fresh-patch-1024", 1024, 2),
        ("fresh-patch-2048", 2048, second_batch_size),
    ):
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
            candidate_configuration=_configuration(identifier, patches, batch_size=batch_size),
            embedder_factory=_ConstantEmbedder,
            identity_factory=_identity_factory,
        )
        reports.append(output)
    contract_path = tmp_path / "contracts" / "fresh_normal_selection_contract.json"
    contract = selection.create_fresh_normal_selection_contract(
        holdout_path,
        augmentation_manifest,
        reports,
        contract_path,
        selection_gates={
            "maxAboveThresholdRate": 0.125,
            "maxP95ScoreMinusThreshold": 0.05,
            "maxMaximumScoreMinusThreshold": 0.10,
        },
        selection_objective=dict(selection.FRESH_NORMAL_SELECTION_OBJECTIVE),
    )
    selection.create_fresh_normal_selection_claim(contract_path)
    return contract_path, holdout_path, source_root, augmentation_manifest, paths, contract


def test_selection_receipt_precedes_one_query_decode_and_excludes_other_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, holdout_path, source_root, augmentation_manifest, paths, contract = _boundary(tmp_path)
    paths["THRESHOLD_TUNING"].write_bytes(b"tuning-must-not-be-reopened")
    paths["NORMAL_CONFIRMATION"].write_bytes(b"confirmation-must-not-be-opened")
    paths["RESERVE_UNTOUCHED"].write_bytes(b"reserve-must-not-be-opened")
    receipt_path = observation.fresh_selection_receipt_path(contract)
    real_loader = holdout.load_evaluation_safe_normal_holdout_inputs
    calls: list[set[str]] = []

    def guarded_loader(*args: object, **kwargs: object) -> tuple[dict, str, list[dict]]:
        partitions = set(kwargs["partitions"])
        calls.append(partitions)
        assert partitions.issubset({"FIT", "NORMAL_SELECTION"})
        if partitions == {"NORMAL_SELECTION"}:
            assert receipt_path.is_file(), "receipt must be committed before NORMAL_SELECTION opens"
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(observation.holdout, "load_evaluation_safe_normal_holdout_inputs", guarded_loader)
    decoded: list[str] = []
    original_decode = observation._load_rgb_and_verify

    def tracked_decode(record: dict) -> Image.Image:
        if record["partition"] == "NORMAL_SELECTION":
            decoded.append(record["caseId"])
            assert receipt_path.is_file()
        return original_decode(record)

    monkeypatch.setattr(observation, "_load_rgb_and_verify", tracked_decode)
    _ConstantEmbedder.calls.clear()
    result = observation.run_fresh_normal_selection_observation(
        contract_path,
        holdout_path,
        augmentation_manifest,
        augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json",
        source_root=source_root,
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_ConstantEmbedder,
        identity_factory=_identity_factory,
    )
    assert calls[-1] == {"NORMAL_SELECTION"}
    assert len(decoded) == len(contract["normalSelectionInputs"])
    assert len(result["candidateObservations"]) == 2
    assert observation.fresh_selection_observation_path(contract).is_file()
    loaded, _ = observation.load_validated_fresh_selection_observation_for_contract(contract_path)
    assert loaded["selectionObservationSha256"] == result["selectionObservationSha256"]


def test_selection_rejects_mixed_batch_sizes_before_receipt_or_selection_open(tmp_path: Path) -> None:
    # The contract itself closes this degree of freedom before any claim or
    # observer exists, which guarantees one decoded/inferred query pass.
    with pytest.raises(selection.FreshNormalSelectionError, match="share one batchSize"):
        _boundary(tmp_path, second_batch_size=1)


def test_receipt_identity_recheck_blocks_query_after_durable_slot_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, holdout_path, source_root, augmentation_manifest, _paths, _contract = _boundary(tmp_path)
    real_loader = holdout.load_evaluation_safe_normal_holdout_inputs
    requested: list[set[str]] = []

    def guarded_loader(*args: object, **kwargs: object) -> tuple[dict, str, list[dict]]:
        partitions = set(kwargs["partitions"])
        requested.append(partitions)
        assert partitions == {"FIT"}
        return real_loader(*args, **kwargs)

    def fail_identity(*_args: object, **_kwargs: object) -> None:
        raise observation.FreshNormalObservationError("simulated receipt identity substitution")

    monkeypatch.setattr(observation.holdout, "load_evaluation_safe_normal_holdout_inputs", guarded_loader)
    monkeypatch.setattr(observation, "_assert_written_slot_identity", fail_identity)
    with pytest.raises(observation.FreshNormalObservationError, match="simulated receipt identity substitution"):
        observation.run_fresh_normal_selection_observation(
            contract_path,
            holdout_path,
            augmentation_manifest,
            augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json",
            source_root=source_root,
            model_repo=tmp_path / "model",
            model_weights=tmp_path / "weights.pth",
            embedder_factory=_ConstantEmbedder,
            identity_factory=_identity_factory,
        )
    assert requested and all(partitions == {"FIT"} for partitions in requested)


def test_contract_copy_uses_the_same_cohort_wide_selection_receipt_slot(tmp_path: Path) -> None:
    contract_path, holdout_path, source_root, augmentation_manifest, _paths, contract = _boundary(tmp_path)
    copied_contract = tmp_path / "copied_contract" / "same-contract.json"
    copied_contract.parent.mkdir(parents=True)
    copied_contract.write_bytes(contract_path.read_bytes())
    copy_document, _ = selection.load_validated_fresh_selection_contract(copied_contract)
    assert observation.fresh_selection_receipt_path(copy_document) == observation.fresh_selection_receipt_path(contract)
    observation.run_fresh_normal_selection_observation(
        copied_contract,
        holdout_path,
        augmentation_manifest,
        augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json",
        source_root=source_root,
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_ConstantEmbedder,
        identity_factory=_identity_factory,
    )
    with pytest.raises(observation.FreshNormalObservationError, match="already exists"):
        observation.run_fresh_normal_selection_observation(
            contract_path,
            holdout_path,
            augmentation_manifest,
            augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json",
            source_root=source_root,
            model_repo=tmp_path / "model",
            model_weights=tmp_path / "weights.pth",
            embedder_factory=_ConstantEmbedder,
            identity_factory=_identity_factory,
        )


def test_json_only_lock_then_explicit_confirmation_with_receipt_before_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, holdout_path, source_root, augmentation_manifest, paths, contract = _boundary(tmp_path)
    recipe = augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json"
    observation.run_fresh_normal_selection_observation(
        contract_path,
        holdout_path,
        augmentation_manifest,
        recipe,
        source_root=source_root,
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_ConstantEmbedder,
        identity_factory=_identity_factory,
    )
    # The lock must be able to run after all images have become unreadable: it
    # reads only the external JSON evidence and must not manufacture a claim.
    for image_path in paths.values():
        image_path.write_bytes(b"json-only-lock-must-not-open-images")
    with pytest.raises(observation.FreshNormalObservationError):
        observation.create_fresh_normal_confirmation_claim(contract_path)
    lock = observation.create_fresh_normal_selection_lock(contract_path)
    assert lock["decision"]["state"] == "RESEARCH_CONFIGURATION_LOCKED"
    assert lock["decision"]["automaticConfirmation"] is False
    assert not observation.fresh_confirmation_claim_path(contract).exists()


def test_confirmation_receipt_precedes_only_confirmation_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract_path, holdout_path, source_root, augmentation_manifest, paths, contract = _boundary(tmp_path)
    recipe = augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json"
    observation.run_fresh_normal_selection_observation(
        contract_path,
        holdout_path,
        augmentation_manifest,
        recipe,
        source_root=source_root,
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_ConstantEmbedder,
        identity_factory=_identity_factory,
    )
    observation.create_fresh_normal_selection_lock(contract_path)
    claim = observation.create_fresh_normal_confirmation_claim(contract_path)
    assert claim["selectedCandidateId"] == "fresh-patch-1024"
    paths["THRESHOLD_TUNING"].write_bytes(b"tuning-must-not-be-reopened")
    paths["NORMAL_SELECTION"].write_bytes(b"selection-must-not-be-reopened")
    paths["RESERVE_UNTOUCHED"].write_bytes(b"reserve-must-not-be-opened")
    receipt_path = observation.fresh_confirmation_receipt_path(contract)
    real_loader = holdout.load_evaluation_safe_normal_holdout_inputs
    calls: list[set[str]] = []

    def guarded_loader(*args: object, **kwargs: object) -> tuple[dict, str, list[dict]]:
        partitions = set(kwargs["partitions"])
        calls.append(partitions)
        assert partitions.issubset({"FIT", "NORMAL_CONFIRMATION"})
        if partitions == {"NORMAL_CONFIRMATION"}:
            assert receipt_path.is_file(), "receipt must be committed before NORMAL_CONFIRMATION opens"
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(observation.holdout, "load_evaluation_safe_normal_holdout_inputs", guarded_loader)
    decoded: list[str] = []
    original_decode = observation._load_rgb_and_verify

    def tracked_decode(record: dict) -> Image.Image:
        if record["partition"] == "NORMAL_CONFIRMATION":
            decoded.append(record["caseId"])
            assert receipt_path.is_file()
        return original_decode(record)

    monkeypatch.setattr(observation, "_load_rgb_and_verify", tracked_decode)
    result = observation.run_fresh_normal_confirmation_observation(
        contract_path,
        holdout_path,
        augmentation_manifest,
        recipe,
        source_root=source_root,
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_ConstantEmbedder,
        identity_factory=_identity_factory,
    )
    assert calls[-1] == {"NORMAL_CONFIRMATION"}
    assert len(decoded) == len(contract["normalConfirmationInputs"])
    assert result["resultScope"] == observation.FRESH_NORMAL_CONFIRMATION_RESULT_SCOPE
    loaded, _ = observation.load_validated_fresh_confirmation_observation_for_contract(contract_path)
    assert loaded["confirmationObservationSha256"] == result["confirmationObservationSha256"]


def test_lock_rejects_duplicate_or_nonfinite_selection_observation_json(tmp_path: Path) -> None:
    contract_path, holdout_path, source_root, augmentation_manifest, _paths, contract = _boundary(tmp_path)
    observation.run_fresh_normal_selection_observation(
        contract_path,
        holdout_path,
        augmentation_manifest,
        augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json",
        source_root=source_root,
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_ConstantEmbedder,
        identity_factory=_identity_factory,
    )
    path = observation.fresh_selection_observation_path(contract)
    path.write_text('{"selectionObservationSha256": NaN, "selectionObservationSha256": "again"}', encoding="utf-8")
    with pytest.raises(observation.FreshNormalObservationError, match="duplicate JSON key|non-finite JSON"):
        observation.create_fresh_normal_selection_lock(contract_path)
