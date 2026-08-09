from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from phone_dino import mvtec_fresh_fit_augmentation as augmentation
from phone_dino import mvtec_normal_holdout as holdout


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def _write_image(path: Path, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (29, 23), (index * 37 % 255, index * 71 % 255, index * 103 % 255)).save(path, format="PNG")


def _holdout_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    source_root = tmp_path / "fresh_source_bytes"
    assignments = (
        ("FIT", 0),
        ("FIT", 1),
        ("THRESHOLD_TUNING", 2),
        ("NORMAL_SELECTION", 3),
        ("NORMAL_CONFIRMATION", 4),
        ("RESERVE_UNTOUCHED", 5),
    )
    records: list[dict] = []
    image_paths: dict[str, Path] = {}
    for partition, index in assignments:
        relative_path = Path("images") / f"{index}.png"
        image_path = source_root / relative_path
        _write_image(image_path, index)
        source_sha256 = holdout.sha256_file(image_path)
        records.append({
            "caseId": f"mvtec-ad/capsule/train-good/{source_sha256[7:]}",
            "category": "capsule",
            "relativePath": relative_path.as_posix(),
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
        image_paths[f"{partition}-{index}"] = image_path
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
    manifest_path = tmp_path / "holdout" / "normal_holdout.json"
    _write_json(manifest_path, document)
    return manifest_path, source_root, image_paths


def _recipe_path() -> Path:
    return augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_fresh_fit_camera_recipe_v1.json"


def test_generates_and_validates_fit_only_without_opening_other_partitions(tmp_path: Path) -> None:
    manifest_path, source_root, image_paths = _holdout_fixture(tmp_path)
    # The generator asks the phase-safe loader for FIT only. Broken non-FIT
    # bytes prove tuning/selection/confirmation/reserve are not decoded.
    for name, image_path in image_paths.items():
        if not name.startswith("FIT-"):
            image_path.write_bytes(b"not-a-fit-parent")
    output_path = tmp_path / "augmentation_package"
    document = augmentation.generate_fresh_fit_augmentations(
        manifest_path,
        source_root,
        _recipe_path(),
        output_path,
        variants_per_parent=2,
    )
    assert document["parentPartition"] == "FIT"
    assert document["inputPolicy"] == augmentation.FRESH_FIT_INPUT_POLICY
    assert len(document["records"]) == 4
    assert {record["parentPartition"] for record in document["records"]} == {"FIT"}
    assert {record["parentCaseId"] for record in document["records"]}
    assert all(record["outputEncoding"]["subsampling"] == "4:2:0" for record in document["records"])
    validated, records = augmentation.load_validated_fresh_fit_augmentations(
        output_path / "augmentation_manifest.json",
        manifest_path,
        source_root=source_root,
        recipe_path=_recipe_path(),
    )
    assert validated == document
    assert records == document["records"]


def test_fresh_fit_generation_is_deterministic_and_rejects_reencoded_child(tmp_path: Path) -> None:
    manifest_path, source_root, _ = _holdout_fixture(tmp_path)
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first = augmentation.generate_fresh_fit_augmentations(
        manifest_path, source_root, _recipe_path(), first_path, variants_per_parent=2
    )
    second = augmentation.generate_fresh_fit_augmentations(
        manifest_path, source_root, _recipe_path(), second_path, variants_per_parent=2
    )
    assert first == second
    for record in first["records"]:
        assert (first_path / record["relativePath"]).read_bytes() == (second_path / record["relativePath"]).read_bytes()

    manifest_file = first_path / "augmentation_manifest.json"
    tampered = json.loads(manifest_file.read_text(encoding="utf-8"))
    record = tampered["records"][0]
    child = first_path / record["relativePath"]
    with Image.open(child) as opened:
        opened.convert("RGB").save(child, format="JPEG", quality=96, subsampling=2, optimize=False, progressive=False)
    record["sourceSha256"] = augmentation.sha256_file(child)
    tampered["augmentationManifestSha256"] = augmentation._document_digest(tampered, "augmentationManifestSha256")
    _write_json(manifest_file, tampered)
    with pytest.raises(augmentation.FreshFitAugmentationError, match="JPEG headers|pixels"):
        augmentation.load_validated_fresh_fit_augmentations(
            manifest_file,
            manifest_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
        )


def test_fresh_fit_recipe_is_closed_and_output_is_new_only(tmp_path: Path) -> None:
    recipe = json.loads(_recipe_path().read_text(encoding="utf-8"))
    recipe["productionAuthorized"] = True
    altered_recipe = tmp_path / "altered_recipe.json"
    _write_json(altered_recipe, recipe)
    with pytest.raises(augmentation.FreshFitAugmentationError, match="non-authoritative"):
        augmentation.load_fresh_fit_camera_recipe(altered_recipe)

    manifest_path, source_root, _ = _holdout_fixture(tmp_path / "cohort")
    output_path = tmp_path / "already_exists"
    output_path.mkdir()
    with pytest.raises(augmentation.FreshFitAugmentationError, match="already exists"):
        augmentation.generate_fresh_fit_augmentations(
            manifest_path,
            source_root,
            _recipe_path(),
            output_path,
            variants_per_parent=1,
        )
    with pytest.raises(augmentation.FreshFitAugmentationError, match="outside the Git"):
        augmentation._require_external_package_root(
            augmentation.REPOSITORY_ROOT.parent,
            repository_root=augmentation.REPOSITORY_ROOT,
        )
