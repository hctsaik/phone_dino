from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from phone_dino import mvtec_synthetic_anomaly_stress_v2 as stress


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _recipe_path() -> Path:
    return stress.REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json"


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, Path, list[dict], dict[str, object]]:
    """Provide exactly twelve FIT parents/category via a fake safe loader."""

    source_root = tmp_path / "source_bytes"
    parents: list[dict] = []
    for category_index, category in enumerate(("capsule", "metal_nut", "tile"), start=1):
        for ordinal in range(1, 13):
            image_path = source_root / "images" / category / f"{ordinal}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (61, 59),
                (
                    (ordinal * 17 + category_index * 5) % 255,
                    (ordinal * 31 + category_index * 7) % 255,
                    (ordinal * 47 + category_index * 11) % 255,
                ),
            ).save(image_path, format="PNG")
            source_sha256 = stress.sha256_file(image_path)
            parents.append({
                "caseId": f"mvtec-ad/{category}/successor-fit/{source_sha256[7:]}",
                "category": category,
                "sourceSha256": source_sha256,
                "sourceGroupId": f"CONTENT_SHA256:{source_sha256[7:]}",
                "partition": "FIT",
                "kind": "NOMINAL",
                "defect": "good",
                "imagePath": image_path,
            })

    evidence = {
        "holdoutManifestFileSha256": _digest("1"),
        "holdoutManifestDeclaredSha256": _digest("2"),
        "selectionContractFileSha256": _digest("3"),
        "selectionContractDeclaredSha256": _digest("4"),
        "parentNormalConfirmationIdentitySha256": _digest("5"),
    }
    envelope = {
        "parentEvidence": evidence,
        "successorPartitionIdentities": {"FIT": _digest("6")},
        "sealFileSha256": _digest("7"),
        "sealDeclaredSha256": _digest("8"),
        "planFileSha256": _digest("9"),
        "planDeclaredSha256": _digest("a"),
        "successorEnvelopeSha256": _digest("b"),
    }
    call: dict[str, object] = {"count": 0}

    def fake_safe_loader(
        parent_holdout_path: Path,
        parent_selection_contract_path: Path,
        plan_path: Path,
        envelope_path: Path,
        *,
        source_root: Path,
        partitions: object,
        repository_root: Path,
    ) -> tuple[dict, str, list[dict]]:
        call["count"] = int(call["count"]) + 1
        call["partitions"] = partitions
        call["sourceRoot"] = source_root
        assert parent_holdout_path.is_file()
        assert parent_selection_contract_path.is_file()
        assert plan_path.is_file()
        assert envelope_path.is_file()
        return envelope, _digest("c"), [dict(parent) for parent in parents]

    monkeypatch.setattr(stress.successor, "load_successor_safe_normal_inputs", fake_safe_loader)
    parent_holdout_path = tmp_path / "parent" / "normal_holdout.json"
    parent_contract_path = tmp_path / "parent" / "selection_contract.json"
    plan_path = tmp_path / "successor" / "plan.json"
    envelope_path = tmp_path / "successor" / "envelope.json"
    for path in (parent_holdout_path, parent_contract_path, plan_path, envelope_path):
        _write_json(path, {})
    return parent_holdout_path, parent_contract_path, plan_path, envelope_path, source_root, parents, call


def _generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_name: str = "package",
) -> tuple[dict, Path, tuple[Path, Path, Path, Path, Path, list[dict], dict[str, object]]]:
    fixture = _fixture(tmp_path, monkeypatch)
    parent_holdout_path, contract_path, plan_path, envelope_path, source_root, _parents, _call = fixture
    output = tmp_path / output_name
    document = stress.generate_synthetic_anomaly_stress_v2(
        parent_holdout_path,
        contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=_recipe_path(),
        output_dir=output,
    )
    return document, output, fixture


def test_generates_108_fit_only_stress_children_and_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, output, fixture = _generate(tmp_path, monkeypatch)
    parent_holdout_path, contract_path, plan_path, envelope_path, source_root, parents, call = fixture

    assert call["partitions"] == {"FIT"}
    assert call["sourceRoot"] == source_root
    assert document["syntheticOnly"] is True
    assert document["postV1Exploratory"] is True
    assert document["comparisonOrPromotionAllowed"] is False
    assert document["parentPartition"] == "FIT"
    assert document["variantsPerParent"] == 9
    split = stress.split_synthetic_anomaly_stress_fit_parents(parents)
    query_parents = [parent for parent in split if parent["syntheticTestRole"] == "QUERY"]
    assert len(query_parents) == 12
    assert len(document["records"]) == 108
    assert document["syntheticQueryParentIdentitySha256"] == stress._query_parent_identity(split)
    assert document["parentSplitCountsPerCategory"] == {"PROTOTYPE": 6, "CALIBRATION": 2, "QUERY": 4}
    assert {record["renderIntensityLevel"] for record in document["records"]} == {
        "SUBTLE",
        "MODERATE",
        "PRONOUNCED",
    }
    assert {record["syntheticDefectFamily"] for record in document["records"]} == {
        "LOCAL_SCRATCH",
        "LOCAL_SPOT",
        "LOCAL_OCCLUSION",
    }
    assert {record["syntheticLabel"] for record in document["records"]} == {"SYNTHETIC_STIMULUS"}
    assert {record["syntheticTestRole"] for record in document["records"]} == {"QUERY"}
    assert all(record["outputEncoding"] == stress.SYNTHETIC_ANOMALY_STRESS_OUTPUT_ENCODING for record in document["records"])
    coverage = {
        (record["parentCaseId"], record["renderIntensityLevel"], record["syntheticDefectFamily"])
        for record in document["records"]
    }
    assert len(coverage) == 108

    loaded, manifest_file_sha256, records = stress.load_validated_synthetic_stress_v2(
        output / "augmentation_manifest.json",
        parent_holdout_path,
        contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=_recipe_path(),
    )
    assert loaded == document
    assert manifest_file_sha256 == stress.sha256_file(output / "augmentation_manifest.json")
    assert [{key: value for key, value in record.items() if key != "imagePath"} for record in records] == document["records"]
    assert all(record["imagePath"] == output / record["relativePath"] for record in records)
    assert call["count"] == 2


def test_generation_is_deterministic_and_stimuli_change_parent_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, first_output, fixture = _generate(tmp_path, monkeypatch, output_name="first")
    parent_holdout_path, contract_path, plan_path, envelope_path, source_root, parents, _call = fixture
    second_output = tmp_path / "second"
    second = stress.generate_synthetic_anomaly_stress_v2(
        parent_holdout_path,
        contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=_recipe_path(),
        output_dir=second_output,
    )
    assert second == document
    by_parent = {parent["caseId"]: parent for parent in parents}
    for record in document["records"]:
        first = first_output / record["relativePath"]
        second_child = second_output / record["relativePath"]
        assert first.read_bytes() == second_child.read_bytes()
        with Image.open(first) as child, Image.open(by_parent[record["parentCaseId"]]["imagePath"]) as parent:
            assert child.convert("RGB").tobytes() != parent.convert("RGB").tobytes()


def test_reencoded_or_modified_child_is_rejected_by_deterministic_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _document, output, fixture = _generate(tmp_path, monkeypatch)
    parent_holdout_path, contract_path, plan_path, envelope_path, source_root, _parents, _call = fixture
    manifest_path = output / "augmentation_manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = tampered["records"][0]
    child_path = output / record["relativePath"]
    with Image.open(child_path) as opened:
        changed = opened.convert("RGB")
    changed.putpixel((0, 0), (255, 0, 255))
    changed.save(child_path, format="PNG", optimize=False, compress_level=9)
    record["sourceSha256"] = stress.sha256_file(child_path)
    tampered["augmentationManifestSha256"] = stress._document_digest(tampered, "augmentationManifestSha256")
    _write_json(manifest_path, tampered)

    with pytest.raises(stress.SyntheticAnomalyStressV2Error, match="pixels do not match"):
        stress.load_validated_synthetic_stress_v2(
            manifest_path,
            parent_holdout_path,
            contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
        )


def test_manifest_path_substitution_is_rejected_before_generation_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    parent_holdout_path, contract_path, plan_path, envelope_path, source_root, _parents, _call = fixture
    output = tmp_path / "substitution"
    manifest_path = output / "augmentation_manifest.json"
    original_path_signature = stress._path_signature
    observed = {"count": 0}

    def substituted_path_signature(path: Path) -> tuple[int, int, int, int]:
        signature = original_path_signature(path)
        if path == manifest_path:
            observed["count"] += 1
            # Simulate a competing replacement observed after our descriptor
            # was synced: its directory entry has a different file identity.
            return signature[0], signature[1] + 1, signature[2], signature[3]
        return signature

    monkeypatch.setattr(stress, "_path_signature", substituted_path_signature)
    with pytest.raises(stress.SyntheticAnomalyStressV2Error, match="manifest changed while it was written"):
        stress.generate_synthetic_anomaly_stress_v2(
            parent_holdout_path,
            contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
            output_dir=output,
        )
    assert observed["count"] == 1
    assert manifest_path.is_file()


def test_output_preflight_rejects_existing_or_repo_local_slot_before_fit_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    parent_holdout_path, contract_path, plan_path, envelope_path, source_root, _parents, call = fixture
    occupied_output = tmp_path / "occupied"
    occupied_output.mkdir()
    with pytest.raises(stress.SyntheticAnomalyStressV2Error, match="already exists"):
        stress.generate_synthetic_anomaly_stress_v2(
            parent_holdout_path,
            contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
            output_dir=occupied_output,
        )
    assert call["count"] == 0

    repository_root = tmp_path / "simulated_repository"
    repository_root.mkdir()
    with pytest.raises(stress.SyntheticAnomalyStressV2Error, match="outside the Git working tree"):
        stress.generate_synthetic_anomaly_stress_v2(
            parent_holdout_path,
            contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
            output_dir=repository_root / "unsafe_output",
            repository_root=repository_root,
        )
    assert call["count"] == 0


def test_recipe_scope_and_existing_output_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = json.loads(_recipe_path().read_text(encoding="utf-8"))
    recipe["comparisonOrPromotionAllowed"] = True
    changed_recipe = tmp_path / "changed_recipe.json"
    _write_json(changed_recipe, recipe)
    with pytest.raises(stress.SyntheticAnomalyStressV2Error, match="comparison or promotion"):
        stress.load_synthetic_anomaly_stress_recipe_v2(changed_recipe)

    fixture = _fixture(tmp_path / "chain", monkeypatch)
    parent_holdout_path, contract_path, plan_path, envelope_path, source_root, _parents, _call = fixture
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(stress.SyntheticAnomalyStressV2Error, match="already exists"):
        stress.generate_synthetic_anomaly_stress_v2(
            parent_holdout_path,
            contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
            output_dir=output,
        )


def test_non_fit_parent_from_safe_loader_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    parent_holdout_path, contract_path, plan_path, envelope_path, source_root, parents, _call = fixture
    parents[0]["partition"] = "THRESHOLD_TUNING"
    with pytest.raises(stress.SyntheticAnomalyStressV2Error, match="FIT nominal-good"):
        stress.generate_synthetic_anomaly_stress_v2(
            parent_holdout_path,
            contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
            output_dir=tmp_path / "output",
        )


def test_generator_requires_the_fixed_three_category_108_child_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    parent_holdout_path, contract_path, plan_path, envelope_path, source_root, parents, _call = fixture
    parents[:] = [parent for parent in parents if parent["category"] != "tile"]
    with pytest.raises(stress.SyntheticAnomalyStressV2Error, match="fixed capsule, metal_nut, and tile"):
        stress.generate_synthetic_anomaly_stress_v2(
            parent_holdout_path,
            contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            recipe_path=_recipe_path(),
            output_dir=tmp_path / "output",
        )


def test_public_parent_split_is_stable_and_matches_v1_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _parent_holdout_path, _contract_path, _plan_path, _envelope_path, _source_root, parents, _call = fixture
    first = stress.split_synthetic_anomaly_stress_fit_parents(parents)
    second = stress.split_synthetic_anomaly_stress_fit_parents(list(reversed(parents)))
    assert [(parent["caseId"], parent["syntheticTestRole"]) for parent in first] == [
        (parent["caseId"], parent["syntheticTestRole"]) for parent in second
    ]
    for category in ("capsule", "metal_nut", "tile"):
        roles = [parent["syntheticTestRole"] for parent in first if parent["category"] == category]
        assert roles.count("PROTOTYPE") == 6
        assert roles.count("CALIBRATION") == 2
        assert roles.count("QUERY") == 4
    with pytest.raises(stress.SyntheticAnomalyStressV2Error, match="exactly 12"):
        stress.split_synthetic_anomaly_stress_fit_parents(parents[:-1])
