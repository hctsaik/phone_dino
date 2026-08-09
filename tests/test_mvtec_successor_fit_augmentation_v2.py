from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from phone_dino import mvtec_normal_holdout as holdout
from phone_dino import mvtec_normal_successor as successor
from phone_dino import mvtec_successor_fit_augmentation_v2 as augmentation


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _record(
    source_root: Path,
    *,
    category: str,
    partition: str,
    ordinal: int,
    decodable: bool,
) -> tuple[dict, Path]:
    extension = "png" if decodable else "bin"
    relative = Path("images") / partition.lower() / category / f"{ordinal}.{extension}"
    path = source_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if decodable:
        Image.new("RGB", (29, 23), (ordinal % 255, (ordinal * 37) % 255, (ordinal * 71) % 255)).save(path, format="PNG")
    else:
        path.write_bytes(f"{partition}:{category}:{ordinal}:must-not-be-opened".encode("utf-8"))
    source_sha256 = holdout.sha256_file(path)
    return {
        "caseId": f"mvtec-ad/{category}/train-good/{source_sha256[7:]}",
        "category": category,
        "relativePath": relative.as_posix(),
        "sourceSha256": source_sha256,
        "sourceGroupId": f"CONTENT_SHA256:{source_sha256[7:]}",
        "acquisitionStratum": "OFFICIAL_MVTEC_TRAIN_GOOD",
        "partition": partition,
        "sourceRemotePath": f"data/{partition.lower()}/{category}/{ordinal}.{extension}",
        "expectedRemoteSha256": source_sha256,
        "expectedRemoteBytes": path.stat().st_size,
        "kind": "NOMINAL",
        "defect": "good",
    }, path


def _successor_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Path]]:
    """Build the sealed chain while making non-FIT bytes deliberately toxic."""

    source_root = tmp_path / "source_bytes"
    records: list[dict] = []
    image_paths: dict[str, Path] = {}
    for index, partition in enumerate(successor.PARENT_HISTORICAL_PARTITIONS):
        record, path = _record(
            source_root,
            category="capsule",
            partition=partition,
            ordinal=1_000 + index,
            decodable=False,
        )
        records.append(record)
        image_paths[record["sourceSha256"]] = path

    quota_totals = {
        quota["category"]: sum(
            int(quota[name])
            for name in ("fitCount", "thresholdTuningCount", "normalSelectionCount", "reserveUntouchedCount")
        )
        for quota in successor.SUCCESSOR_CATEGORY_QUOTAS
    }
    ordinal = 0
    for category, count in quota_totals.items():
        for _ in range(count):
            record, path = _record(
                source_root,
                category=category,
                partition=successor.PARENT_RESERVE_PARTITION,
                ordinal=ordinal,
                decodable=True,
            )
            records.append(record)
            image_paths[record["sourceSha256"]] = path
            ordinal += 1

    for category in ("capsule", "metal_nut", "tile"):
        for confirmation_ordinal in range(32):
            record, path = _record(
                source_root,
                category=category,
                partition=successor.PARENT_CONFIRMATION_PARTITION,
                ordinal=confirmation_ordinal,
                decodable=False,
            )
            records.append(record)
            image_paths[record["sourceSha256"]] = path

    records.sort(key=lambda item: item["caseId"])
    holdout_document: dict = {
        "schemaVersion": holdout.NORMAL_HOLDOUT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": holdout.HOLDOUT_PURPOSE,
        "blindPolicy": holdout.HOLDOUT_BLIND_POLICY,
        "sourcePoolFileSha256": _digest("1"),
        "sourcePoolDeclaredSha256": _digest("2"),
        "historicalLedgerFileSha256": _digest("3"),
        "historicalLedgerDeclaredSha256": _digest("4"),
        "planFileSha256": _digest("5"),
        "planDeclaredSha256": _digest("6"),
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
    holdout_document["normalHoldoutManifestSha256"] = holdout._document_digest(
        holdout_document,
        "normalHoldoutManifestSha256",
    )
    holdout_path = tmp_path / "parent" / "normal_holdout.json"
    _write_json(holdout_path, holdout_document)
    parsed_records = holdout._validate_closed_normal_holdout_document(holdout_document)
    historical, reserve, confirmation = successor._classify_parent_records(
        parsed_records,
        holdout_document=holdout_document,
    )
    context = successor._ParentContext(
        holdout_path=holdout_path,
        holdout_document=holdout_document,
        holdout_file_sha256=successor.sha256_file(holdout_path),
        contract={"contractSha256": _digest("7")},
        contract_file_sha256=_digest("8"),
        claim={"claimSha256": _digest("9")},
        claim_file_sha256=_digest("a"),
        selection_observation={
            "selectionReceiptFileSha256": _digest("b"),
            "selectionReceiptDeclaredSha256": _digest("c"),
            "selectionObservationSha256": _digest("d"),
        },
        selection_observation_file_sha256=_digest("e"),
        selection_lock={
            "selectionLockSha256": _digest("f"),
            "decision": {"state": "NO_ELIGIBLE_CONFIGURATION", "selectedCandidateId": None},
        },
        selection_lock_file_sha256=_digest("0"),
        historical_sources=historical,
        reserve_inputs=reserve,
        preserved_confirmation=confirmation,
    )
    contract_path = tmp_path / "parent" / "selection_contract.json"
    _write_json(contract_path, {})
    monkeypatch.setattr(successor, "_load_parent_context", lambda *_args, **_kwargs: context)
    successor.create_fresh_normal_successor_seal(holdout_path, contract_path)
    plan_path = tmp_path / "successor" / "successor_plan.json"
    successor.create_fresh_normal_successor_plan(holdout_path, contract_path, plan_path)
    envelope_path = tmp_path / "successor" / "successor_envelope.json"
    envelope = successor.create_fresh_normal_successor_envelope(holdout_path, contract_path, plan_path, envelope_path)

    # A FIT-only loader must never attempt these source bytes.  They were
    # initially decodable to permit sealed allocation, then poisoned after
    # the envelope has frozen their identities.
    fit_hashes = {record["sourceSha256"] for record in envelope["records"] if record["partition"] == "FIT"}
    for source_sha256, path in image_paths.items():
        if source_sha256 not in fit_hashes:
            path.write_bytes(b"must-not-be-opened-by-successor-fit-augmentation")
    return holdout_path, contract_path, plan_path, envelope_path, source_root, image_paths


def _recipe_path() -> Path:
    return augmentation.REPOSITORY_ROOT / "tools" / "mvtec_ad_successor_fit_camera_recipe_v2.json"


def test_generates_r3_from_only_successor_fit_and_revalidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    holdout_path, contract_path, plan_path, envelope_path, source_root, _image_paths = _successor_fixture(tmp_path, monkeypatch)
    first_path = tmp_path / "first"
    document = augmentation.generate_successor_fit_augmentations(
        holdout_path,
        contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=_recipe_path(),
        output_dir=first_path,
    )
    assert document["parentPartition"] == "FIT"
    assert document["variantsPerParent"] == 3
    assert len(document["records"]) == 36 * 3
    assert document["successorFitIdentitySha256"]
    by_parent: dict[str, set[str]] = {}
    for record in document["records"]:
        by_parent.setdefault(record["parentCaseId"], set()).add(record["component"])
        assert record["outputEncoding"]["subsampling"] == "4:2:0"
        assert set(record["parameters"]) == {
            "component",
            "namedSubstream",
            "seed",
            record["component"],
        }
    assert all(components == {"registration", "illumination", "sensor_transport"} for components in by_parent.values())
    loaded, records = augmentation.load_validated_successor_fit_augmentations(
        first_path / "augmentation_manifest.json",
        holdout_path,
        contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=_recipe_path(),
    )
    assert loaded == document
    assert records == document["records"]
    loaded_with_digest, manifest_file_sha256, records_with_digest = (
        augmentation.load_validated_successor_fit_augmentations_with_file_sha256(
            first_path / "augmentation_manifest.json",
            holdout_path,
            contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
        )
    )
    assert loaded_with_digest == document
    assert manifest_file_sha256 == augmentation.sha256_file(first_path / "augmentation_manifest.json")
    assert records_with_digest == document["records"]

    second_path = tmp_path / "second"
    second = augmentation.generate_successor_fit_augmentations(
        holdout_path,
        contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=_recipe_path(),
        output_dir=second_path,
    )
    assert second == document
    for record in document["records"]:
        assert (first_path / record["relativePath"]).read_bytes() == (second_path / record["relativePath"]).read_bytes()


def test_reencoded_v2_child_is_rejected_by_header_and_rerender_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holdout_path, contract_path, plan_path, envelope_path, source_root, _image_paths = _successor_fixture(tmp_path, monkeypatch)
    output_path = tmp_path / "package"
    document = augmentation.generate_successor_fit_augmentations(
        holdout_path,
        contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=_recipe_path(),
        output_dir=output_path,
    )
    manifest_path = output_path / "augmentation_manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = tampered["records"][0]
    child = output_path / record["relativePath"]
    with Image.open(child) as opened:
        opened.convert("RGB").save(child, format="JPEG", quality=96, subsampling=2, optimize=False, progressive=False)
    record["sourceSha256"] = augmentation.sha256_file(child)
    tampered["augmentationManifestSha256"] = augmentation._document_digest(tampered, "augmentationManifestSha256")
    _write_json(manifest_path, tampered)
    with pytest.raises(augmentation.SuccessorFitAugmentationV2Error, match="JPEG headers|pixels"):
        augmentation.load_validated_successor_fit_augmentations(
            manifest_path,
            holdout_path,
            contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
        )


def test_recipe_is_closed_and_output_must_be_new_external_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = json.loads(_recipe_path().read_text(encoding="utf-8"))
    recipe["productionAuthorized"] = True
    changed_recipe = tmp_path / "changed_recipe.json"
    _write_json(changed_recipe, recipe)
    with pytest.raises(augmentation.SuccessorFitAugmentationV2Error, match="non-authoritative"):
        augmentation.load_successor_fit_camera_recipe_v2(changed_recipe)

    holdout_path, contract_path, plan_path, envelope_path, source_root, _image_paths = _successor_fixture(tmp_path / "chain", monkeypatch)
    output_path = tmp_path / "exists"
    output_path.mkdir()
    with pytest.raises(augmentation.SuccessorFitAugmentationV2Error, match="already exists"):
        augmentation.generate_successor_fit_augmentations(
            holdout_path,
            contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
            output_dir=output_path,
        )
    with pytest.raises(augmentation.SuccessorFitAugmentationV2Error, match="outside the Git"):
        augmentation._require_external_package_root(
            augmentation.REPOSITORY_ROOT.parent,
            repository_root=augmentation.REPOSITORY_ROOT,
        )
    with pytest.raises(augmentation.SuccessorFitAugmentationV2Error, match="outside the Git"):
        augmentation._require_external_input_file(
            augmentation.REPOSITORY_ROOT.parent / "would-enclose-repository.json",
            description="test input manifest",
            repository_root=augmentation.REPOSITORY_ROOT,
        )


def test_reparse_check_precedes_output_containment_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "reparse-parent" / "output"
    output_path.parent.mkdir()
    original_resolve = Path.resolve

    def fail_if_containment_is_checked_first(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("containment resolution must follow the reparse check")

    monkeypatch.setattr(Path, "resolve", fail_if_containment_is_checked_first)
    monkeypatch.setattr(augmentation, "_is_link_or_reparse_point", lambda path: path == output_path.parent)
    with pytest.raises(augmentation.SuccessorFitAugmentationV2Error, match="contains a symbolic link"):
        augmentation._prepare_external_output_directory(output_path, repository_root=augmentation.REPOSITORY_ROOT)
    monkeypatch.setattr(Path, "resolve", original_resolve)
