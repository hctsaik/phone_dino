from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from phone_dino import mvtec_successor_evaluator_v2 as evaluator


def _digest(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _write_image(path: Path, ordinal: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(
        "RGB",
        (19, 17),
        ((ordinal * 29) % 255, (ordinal * 47) % 255, (ordinal * 71) % 255),
    ).save(path, format="PNG")
    return evaluator.sha256_file(path)


def _envelope(plan_path: Path) -> dict[str, Any]:
    return {
        "parentEvidence": {
            "holdoutManifestFileSha256": _digest("parent-holdout-file"),
            "holdoutManifestDeclaredSha256": _digest("parent-holdout-declared"),
            "selectionContractFileSha256": _digest("parent-contract-file"),
            "selectionContractDeclaredSha256": _digest("parent-contract-declared"),
            "parentNormalConfirmationIdentitySha256": _digest("parent-confirmation-identity"),
        },
        "sealFileSha256": _digest("seal-file"),
        "sealDeclaredSha256": _digest("seal-declared"),
        "planFileSha256": evaluator.sha256_file(plan_path),
        "planDeclaredSha256": _digest("plan-declared"),
        "successorEnvelopeSha256": _digest("envelope-declared"),
        "successorPartitionIdentities": {
            "FIT": _digest("fit-identity"),
            "THRESHOLD_TUNING": _digest("tuning-identity"),
        },
    }


def _original_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = {}
    ordinal = 1
    for category in ("capsule", "metal_nut", "tile"):
        category_records: list[dict[str, Any]] = []
        for partition, count in (("FIT", 12), ("THRESHOLD_TUNING", 4)):
            for index in range(count):
                path = root / category / partition / f"{index}.png"
                source_sha256 = _write_image(path, ordinal)
                ordinal += 1
                record = {
                    "caseId": f"successor/{category}/{partition.lower()}/{index:02d}",
                    "category": category,
                    "partition": partition,
                    "kind": "NOMINAL",
                    "defect": "good",
                    "sourceSha256": source_sha256,
                    "sourceGroupId": f"CONTENT_SHA256:{source_sha256[7:]}",
                    "imagePath": path,
                }
                records.append(record)
                category_records.append(record)
        by_category[category] = category_records
    return records, by_category


class _FakeEmbedder:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def extract_patches(self, images: list[Image.Image]) -> list[object]:
        result: list[object] = []
        base = np.linspace(1.0, 2.0, 8, dtype=np.float32)
        for image in images:
            signal = float(np.asarray(image, dtype=np.uint8).mean()) / 255.0
            patches = np.stack([base + signal + patch * 0.001 for patch in range(16)], axis=0)
            result.append(patches.astype(np.float32))
        return result


class _CountingFakeEmbedder(_FakeEmbedder):
    image_count = 0

    def extract_patches(self, images: list[Image.Image]) -> list[object]:
        type(self).image_count += len(images)
        return super().extract_patches(images)


def _identity_factory(**_kwargs: object) -> dict[str, object]:
    return {"fixture": "successor-v2-feature-extractor"}


def _safe_loader(
    expected_envelope: dict[str, Any],
    records: list[dict[str, Any]],
    calls: list[set[str]],
):
    def load(
        _parent_holdout: Path,
        _parent_contract: Path,
        _plan: Path,
        _envelope: Path,
        *,
        source_root: Path,
        partitions: object,
        repository_root: Path,
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        assert source_root.is_dir()
        assert repository_root == evaluator.REPOSITORY_ROOT
        requested = set(partitions)  # type: ignore[arg-type]
        calls.append(requested)
        assert requested == {"FIT", "THRESHOLD_TUNING"}
        return expected_envelope, _digest("envelope-file"), [
            dict(record) for record in records if record["partition"] in requested
        ]

    return load


def _run_raw(
    tmp_path: Path,
    monkeypatch: object,
) -> tuple[dict[str, Any], list[set[str]], list[dict[str, Any]], Path, Path]:
    source_root = tmp_path / "source"
    originals, _by_category = _original_records(source_root)
    # These bytes emulate protected partitions. The loader spy must ensure the
    # evaluator never asks for a partition that could make them observable.
    (source_root / "protected").mkdir(parents=True, exist_ok=True)
    (source_root / "protected" / "selection.bin").write_bytes(b"must-not-open-selection")
    (source_root / "protected" / "confirmation.bin").write_bytes(b"must-not-open-confirmation")
    plan_path = tmp_path / "chain" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("{}", encoding="utf-8")
    envelope = _envelope(plan_path)
    calls: list[set[str]] = []
    monkeypatch.setattr(evaluator, "load_successor_safe_normal_inputs", _safe_loader(envelope, originals, calls))
    output_path = tmp_path / "reports" / "raw.json"
    report = evaluator.run_successor_v2_development_evaluation(
        tmp_path / "chain" / "parent_holdout.json",
        tmp_path / "chain" / "contract.json",
        plan_path,
        tmp_path / "chain" / "envelope.json",
        output_path,
        source_root=source_root,
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        device="cpu",
        candidate_configuration=evaluator.pre_registered_candidate_configuration("reserve-v2-raw-p2048-k5"),
        embedder_factory=_FakeEmbedder,
        identity_factory=_identity_factory,
    )
    return report, calls, originals, plan_path, source_root


def test_successor_v2_development_is_phase_isolated_and_raw_only(tmp_path: Path, monkeypatch: object) -> None:
    report, calls, originals, _plan_path, _source_root = _run_raw(tmp_path, monkeypatch)
    assert calls == [{"FIT", "THRESHOLD_TUNING"}]
    assert report["candidateConfiguration"]["prototypeInputPolicy"] == evaluator.RAW_FIT_ONLY
    assert report["augmentationManifestFileSha256"] is None
    assert report["normalOnlyEvidence"]["featureInputPartitions"] == ["FIT", "THRESHOLD_TUNING"]
    assert report["normalOnlyEvidence"]["blindFeatureInputCount"] == 0
    assert report["normalOnlyEvidence"]["anomalyFeatureInputCount"] == 0
    assert report["normalOnlyEvidence"]["maskFeatureInputCount"] == 0
    assert all(record["partition"] != "NORMAL_SELECTION" for record in report["featureInputs"])
    assert all(record["partition"] != "NORMAL_CONFIRMATION" for record in report["featureInputs"])
    assert len(report["featureInputs"]) == len(originals)
    assert all(not record["isAugmentation"] for record in report["featureInputs"])


def test_successor_v2_nested_prototype_prefix_is_true_for_1024_and_2048() -> None:
    records = [
        {
            "caseId": "fit/parent-a/raw",
            "partition": "FIT",
            "sourceSha256": _digest("raw-a"),
            "isAugmentation": False,
            "parentCaseId": None,
            "variantId": None,
        },
        {
            "caseId": "fit/parent-a/registration",
            "partition": "FIT",
            "sourceSha256": _digest("aug-a"),
            "isAugmentation": True,
            "parentCaseId": "fit/parent-a/raw",
            "variantId": 1,
        },
        {
            "caseId": "fit/parent-b/raw",
            "partition": "FIT",
            "sourceSha256": _digest("raw-b"),
            "isAugmentation": False,
            "parentCaseId": None,
            "variantId": None,
        },
        {
            "caseId": "fit/parent-b/illumination",
            "partition": "FIT",
            "sourceSha256": _digest("aug-b"),
            "isAugmentation": True,
            "parentCaseId": "fit/parent-b/raw",
            "variantId": 2,
        },
    ]
    prefix_1024 = evaluator.deterministic_stratified_hash_ranked_patch_indices(records, [800, 800, 800, 800], 1024)
    prefix_2048 = evaluator.deterministic_stratified_hash_ranked_patch_indices(records, [800, 800, 800, 800], 2048)
    assert len(prefix_1024) == 1024
    assert len(prefix_2048) == 2048
    assert prefix_1024 == prefix_2048[:1024]
    assert len(set(prefix_2048)) == len(prefix_2048)


def test_multicandidate_preflight_rejects_alias_output_slots_before_any_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    def fail_if_inputs_are_loaded(*_args: object, **_kwargs: object) -> object:
        calls.append("input-loader")
        raise AssertionError("duplicate output aliases must fail before any input is loaded")

    monkeypatch.setattr(evaluator, "load_successor_safe_normal_inputs", fail_if_inputs_are_loaded)
    raw = evaluator.pre_registered_candidate_configuration("reserve-v2-raw-p2048-k5")
    r3 = evaluator.pre_registered_candidate_configuration("reserve-v2-r3-p1024-k3")
    report_path = tmp_path / "reports" / "shared.json"
    alias_path = tmp_path / "reports" / "alias" / ".." / "shared.json"
    with pytest.raises(evaluator.SuccessorV2EvaluatorError, match="same canonical report slot"):
        evaluator.run_successor_v2_development_evaluations(
            tmp_path / "parent.json",
            tmp_path / "contract.json",
            tmp_path / "plan.json",
            tmp_path / "envelope.json",
            {raw["id"]: report_path, r3["id"]: alias_path},
            source_root=tmp_path,
            model_repo=tmp_path / "model",
            model_weights=tmp_path / "weights.pth",
            device="cpu",
            candidate_configurations=[raw, r3],
        )
    assert calls == []
    assert not report_path.exists()


def test_external_output_reparse_check_precedes_containment_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "reparse-parent" / "report.json"
    output_path.parent.mkdir()

    def fail_if_containment_is_checked_first(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("containment resolution must follow the reparse check")

    monkeypatch.setattr(Path, "resolve", fail_if_containment_is_checked_first)
    monkeypatch.setattr(evaluator, "_is_link_or_reparse_point", lambda path: path == output_path.parent)
    with pytest.raises(evaluator.SuccessorV2EvaluatorError, match="contains a symbolic link"):
        evaluator._canonical_external_output_slot(output_path, repository_root=evaluator.REPOSITORY_ROOT)


def test_r3_rejects_manifest_swap_after_validated_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    augmentation_manifest = tmp_path / "augmentation" / "augmentation_manifest.json"
    augmentation_manifest.parent.mkdir()
    augmentation_manifest.write_text('{"state":"validated"}', encoding="utf-8")
    validated_file_sha256 = evaluator.sha256_file(augmentation_manifest)
    original_records = [
        {
            "caseId": "successor/capsule/fit/00",
            "category": "capsule",
            "partition": "FIT",
            "kind": "NOMINAL",
            "defect": "good",
            "sourceSha256": _digest("fit"),
            "sourceGroupId": "CONTENT_SHA256:fit",
            "imagePath": source_root / "fit.png",
        },
        {
            "caseId": "successor/capsule/tuning/00",
            "category": "capsule",
            "partition": "THRESHOLD_TUNING",
            "kind": "NOMINAL",
            "defect": "good",
            "sourceSha256": _digest("tuning"),
            "sourceGroupId": "CONTENT_SHA256:tuning",
            "imagePath": source_root / "tuning.png",
        },
    ]

    def safe_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        return {}, _digest("envelope"), original_records

    def validated_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        augmentation_manifest.write_text('{"state":"swapped"}', encoding="utf-8")
        return {}, validated_file_sha256, []

    monkeypatch.setattr(evaluator, "load_successor_safe_normal_inputs", safe_loader)
    monkeypatch.setattr(evaluator, "_load_validated_successor_fit_augmentations", validated_loader)
    with pytest.raises(evaluator.SuccessorV2EvaluatorError, match="manifest changed after validation"):
        evaluator._development_inputs(
            tmp_path / "parent.json",
            tmp_path / "contract.json",
            tmp_path / "plan.json",
            tmp_path / "envelope.json",
            source_root=source_root,
            configuration=evaluator.pre_registered_candidate_configuration("reserve-v2-r3-p1024-k3"),
            augmentation_manifest_path=augmentation_manifest,
            recipe_path=tmp_path / "recipe.json",
            repository_root=evaluator.REPOSITORY_ROOT,
        )


def test_successor_v2_augmented_calibration_uses_only_four_raw_tuning_inputs(
    tmp_path: Path, monkeypatch: object
) -> None:
    source_root = tmp_path / "source"
    originals, by_category = _original_records(source_root)
    plan_path = tmp_path / "chain" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("{}", encoding="utf-8")
    envelope = _envelope(plan_path)
    calls: list[set[str]] = []
    monkeypatch.setattr(evaluator, "load_successor_safe_normal_inputs", _safe_loader(envelope, originals, calls))

    augmentation_manifest = tmp_path / "augmentation" / "augmentation_manifest.json"
    augmentation_manifest.parent.mkdir(parents=True, exist_ok=True)
    augmentation_manifest.write_text("{}", encoding="utf-8")
    recipe_path = tmp_path / "augmentation" / "recipe.json"
    recipe_path.write_text("{}", encoding="utf-8")
    augmented_records: list[dict[str, Any]] = []
    ordinal = 1000
    component_by_variant = {1: "registration", 2: "illumination", 3: "sensor_transport"}
    for category, category_records in by_category.items():
        for parent in (record for record in category_records if record["partition"] == "FIT"):
            for variant_id, component in component_by_variant.items():
                relative = Path("images") / category / f"{parent['caseId'].replace('/', '-')}-{variant_id}.png"
                child_path = augmentation_manifest.parent / relative
                source_sha256 = _write_image(child_path, ordinal)
                ordinal += 1
                augmented_records.append({
                    "caseId": f"{parent['caseId']}/r3/{variant_id}",
                    "parentCaseId": parent["caseId"],
                    "parentSourceSha256": parent["sourceSha256"],
                    "sourceGroupId": parent["sourceGroupId"],
                    "category": category,
                    "parentPartition": "FIT",
                    "kind": "NOMINAL",
                    "defect": "good",
                    "variantId": variant_id,
                    "component": component,
                    "relativePath": relative.as_posix(),
                    "sourceSha256": source_sha256,
                })
    augmentation_document = {
        "augmentationManifestSha256": _digest("augmentation-declared"),
        "successorFitIdentitySha256": envelope["successorPartitionIdentities"]["FIT"],
        "successorEnvelopeFileSha256": _digest("envelope-file"),
        "successorEnvelopeDeclaredSha256": envelope["successorEnvelopeSha256"],
        "successorPlanFileSha256": evaluator.sha256_file(plan_path),
        "successorPlanDeclaredSha256": envelope["planDeclaredSha256"],
        "recipeFileSha256": evaluator.sha256_file(recipe_path),
        "variantsPerParent": 3,
    }

    def load_augmentation(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        return augmentation_document, evaluator.sha256_file(augmentation_manifest), augmented_records

    monkeypatch.setattr(evaluator, "_load_validated_successor_fit_augmentations", load_augmentation)
    _CountingFakeEmbedder.image_count = 0
    reports = evaluator.run_successor_v2_development_evaluations(
        tmp_path / "chain" / "parent_holdout.json",
        tmp_path / "chain" / "contract.json",
        plan_path,
        tmp_path / "chain" / "envelope.json",
        {
            configuration["id"]: tmp_path / "reports" / f"{configuration['id']}.json"
            for configuration in evaluator.PRE_REGISTERED_CANDIDATES
        },
        source_root=source_root,
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        device="cpu",
        candidate_configurations=[dict(configuration) for configuration in evaluator.PRE_REGISTERED_CANDIDATES],
        augmentation_manifest_path=augmentation_manifest,
        recipe_path=recipe_path,
        embedder_factory=_CountingFakeEmbedder,
        identity_factory=_identity_factory,
    )
    report = reports["reserve-v2-r3-p2048-k5"]
    assert calls == [{"FIT", "THRESHOLD_TUNING"}] * 4
    # Every unique source is embedded at most once. The fixture deliberately
    # permits byte-identical tiny colour images, so calculate the expected
    # in-memory cache cardinality from source digests rather than image count.
    # The two later R3 reports reuse only those identity-keyed features after
    # re-verifying their source bytes.
    expected_unique_features = {
        record["sourceSha256"] for record in originals + augmented_records
    }
    assert _CountingFakeEmbedder.image_count == len(expected_unique_features)
    assert _CountingFakeEmbedder.image_count < 48 + (3 * 156)
    assert set(reports) == {configuration["id"] for configuration in evaluator.PRE_REGISTERED_CANDIDATES}
    assert report["candidateConfiguration"]["prototypeInputPolicy"] == evaluator.RAW_FIT_PLUS_AUGMENTATION_R3
    assert report["normalOnlyEvidence"]["fitAugmentedFeatureInputCount"] == 108
    assert report["normalOnlyEvidence"]["calibrationInputCount"] == 12
    assert {record["partition"] for record in report["calibrationInputs"]} == {"THRESHOLD_TUNING"}
    assert all(not record["isAugmentation"] for record in report["calibrationInputs"])
    assert len(report["calibrationScores"]) == 12
    for category in ("capsule", "metal_nut", "tile"):
        scores = [item["score"] for item in report["calibrationScores"] if item["category"] == category]
        assert len(scores) == 4
        assert report["thresholds"][category] == max(scores)
        assert report["categories"][category]["fitOriginalCount"] == 12
        assert report["categories"][category]["fitAugmentedCount"] == 36
        assert report["categories"][category]["tuningOriginalCount"] == 4
