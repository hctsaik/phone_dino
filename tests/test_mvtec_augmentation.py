from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from phone_dino.mvtec_research import (
    MvtecResearchError,
    canonical_json_sha256,
    derive_augmentation_seed,
    generate_normal_augmentations,
    load_camera_recipe,
    load_validated_normal_augmentations,
    sample_camera_parameters,
    sha256_file,
    validate_normal_augmentation_parent,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
RECIPE_PATH = REPOSITORY_ROOT / "tools" / "mvtec_ad_camera_lighting_recipe_v1.json"
RECIPE_V2_PATH = REPOSITORY_ROOT / "tools" / "mvtec_ad_camera_lighting_recipe_v2.json"
RECIPE_V3_PATH = REPOSITORY_ROOT / "tools" / "mvtec_ad_camera_lighting_recipe_v3.json"


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _record(case_id: str, role: str, source_sha256: str, relative_path: str, **extra: object) -> dict[str, object]:
    return {
        "caseId": case_id,
        "category": "capsule",
        "defect": "good",
        "kind": "NOMINAL",
        "role": role,
        "relativePath": relative_path,
        "sourceSha256": source_sha256,
        "maskSourcePath": None,
        **extra,
    }


def _frozen_manifest(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    image_path = source_dir / "normal.png"
    Image.new("RGB", (32, 28), (90, 130, 180)).save(image_path)
    source_hash = _sha256(image_path)
    document = {
        "schemaVersion": "phone-dino.mvtec-ad-smoke/1.0",
        "authoritative": False,
        "manifestSha256": f"sha256:{'a' * 64}",
        "records": [
            _record("case/fit/001", "FIT", source_hash, "source/normal.png"),
            _record("case/tuning/001", "THRESHOLD_TUNING", source_hash, "source/normal.png"),
            _record("case/blind/001", "BLIND", source_hash, "source/normal.png"),
        ],
    }
    manifest_path = tmp_path / "subset_manifest.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    return manifest_path


@pytest.mark.parametrize("recipe_path", [RECIPE_PATH, RECIPE_V2_PATH, RECIPE_V3_PATH])
def test_parameters_and_seed_are_stable_without_record_order(recipe_path: Path) -> None:
    recipe, recipe_sha256 = load_camera_recipe(recipe_path)
    first = sample_camera_parameters(
        recipe,
        recipe_sha256=recipe_sha256,
        parent_case_id="case/fit/001",
        parent_source_sha256=f"sha256:{'1' * 64}",
        variant_id=1,
    )
    second = sample_camera_parameters(
        recipe,
        recipe_sha256=recipe_sha256,
        parent_case_id="case/fit/001",
        parent_source_sha256=f"sha256:{'1' * 64}",
        variant_id=1,
    )
    assert first == second
    assert int(first["seed"]) == derive_augmentation_seed(
        recipe.get("samplingSeedAnchor", recipe_sha256), "case/fit/001", f"sha256:{'1' * 64}", 1
    )


@pytest.mark.parametrize("record", [
    _record("case/blind/001", "BLIND", f"sha256:{'1' * 64}", "normal.png"),
    _record("case/fit/001", "FIT", f"sha256:{'1' * 64}", "normal.png", kind="ANOMALY"),
    _record("case/fit/001", "FIT", f"sha256:{'1' * 64}", "normal.png", maskRelativePath="mask.png"),
])
def test_non_normal_or_blind_parent_is_rejected(record: dict[str, object]) -> None:
    with pytest.raises(MvtecResearchError):
        validate_normal_augmentation_parent(record)


def test_generator_is_deterministic_and_keeps_blind_original(tmp_path: Path) -> None:
    manifest_path = _frozen_manifest(tmp_path)
    first_dir = tmp_path / "outside-one"
    second_dir = tmp_path / "outside-two"
    first = generate_normal_augmentations(manifest_path, RECIPE_PATH, first_dir, variants_per_parent=1, repository_root=tmp_path / "repo")
    second = generate_normal_augmentations(manifest_path, RECIPE_PATH, second_dir, variants_per_parent=1, repository_root=tmp_path / "repo")
    assert first["blindPolicy"] == "BLIND_ORIGINAL_ONLY"
    assert len(first["records"]) == 2
    assert {record["role"] for record in first["records"]} == {"FIT", "THRESHOLD_TUNING"}
    assert [record["sourceSha256"] for record in first["records"]] == [record["sourceSha256"] for record in second["records"]]
    document, records = load_validated_normal_augmentations(first_dir / "augmentation_manifest.json", manifest_path)
    assert document["recipeSha256"] == sha256_file(RECIPE_PATH)
    assert len(records) == 2


def test_generator_refuses_git_output_and_tampered_derived_data(tmp_path: Path) -> None:
    manifest_path = _frozen_manifest(tmp_path)
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    with pytest.raises(MvtecResearchError, match="outside the Git"):
        generate_normal_augmentations(
            manifest_path, RECIPE_PATH, repository_root / "outputs", variants_per_parent=1, repository_root=repository_root
        )

    output_dir = tmp_path / "outside"
    document = generate_normal_augmentations(
        manifest_path, RECIPE_PATH, output_dir, variants_per_parent=1, repository_root=repository_root
    )
    image_path = output_dir / str(document["records"][0]["relativePath"])
    image_path.write_bytes(b"tampered")
    with pytest.raises(MvtecResearchError, match="output digest"):
        load_validated_normal_augmentations(output_dir / "augmentation_manifest.json", manifest_path)


def _refresh_augmentation_manifest(path: Path, document: dict[str, object]) -> None:
    unsigned = dict(document)
    unsigned.pop("augmentationManifestSha256", None)
    document["augmentationManifestSha256"] = canonical_json_sha256(unsigned)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_v3_preserves_v2_draws_and_adds_only_off_axis_lens_fields() -> None:
    recipe_v2, recipe_v2_sha256 = load_camera_recipe(RECIPE_V2_PATH)
    recipe_v3, recipe_v3_sha256 = load_camera_recipe(RECIPE_V3_PATH)
    arguments = {
        "parent_case_id": "case/fit/001",
        "parent_source_sha256": f"sha256:{'1' * 64}",
        "variant_id": 1,
    }
    v2 = sample_camera_parameters(recipe_v2, recipe_sha256=recipe_v2_sha256, **arguments)
    v3 = sample_camera_parameters(recipe_v3, recipe_sha256=recipe_v3_sha256, **arguments)
    assert recipe_v3["samplingSeedAnchor"] == recipe_v2_sha256
    assert {name: v3[name] for name in v2} == v2
    assert set(v3) - set(v2) == {
        "offAxisLensShadingStrength",
        "lensShadingCenterOffsetXFraction",
        "lensShadingCenterOffsetYFraction",
    }
    assert 0.0 <= v3["offAxisLensShadingStrength"] <= 0.025
    assert abs(v3["lensShadingCenterOffsetXFraction"]) <= 0.12
    assert abs(v3["lensShadingCenterOffsetYFraction"]) <= 0.12


def test_v3_manifest_requires_full_coverage_and_rederived_parameters(tmp_path: Path) -> None:
    manifest_path = _frozen_manifest(tmp_path)
    output_dir = tmp_path / "outside-v3"
    generated = generate_normal_augmentations(
        manifest_path, RECIPE_V3_PATH, output_dir, variants_per_parent=1, repository_root=tmp_path / "repo"
    )
    manifest_output = output_dir / "augmentation_manifest.json"
    assert generated["schemaVersion"] == "phone-dino.mvtec-ad-normal-augmentation/1.1"
    assert generated["generation"]["generatorModuleSha256"] == sha256_file(REPOSITORY_ROOT / "src" / "phone_dino" / "mvtec_research.py")
    load_validated_normal_augmentations(manifest_output, manifest_path)

    incomplete = json.loads(manifest_output.read_text(encoding="utf-8"))
    incomplete["records"].pop()
    _refresh_augmentation_manifest(manifest_output, incomplete)
    with pytest.raises(MvtecResearchError, match="cover every eligible normal parent"):
        load_validated_normal_augmentations(manifest_output, manifest_path)

    regenerated_dir = tmp_path / "outside-v3-regenerated"
    generate_normal_augmentations(
        manifest_path, RECIPE_V3_PATH, regenerated_dir, variants_per_parent=1, repository_root=tmp_path / "repo"
    )
    regenerated_output = regenerated_dir / "augmentation_manifest.json"
    changed_parameters = json.loads(regenerated_output.read_text(encoding="utf-8"))
    changed_parameters["records"][0]["parameters"]["rotationDegrees"] = 0.0
    _refresh_augmentation_manifest(regenerated_output, changed_parameters)
    with pytest.raises(MvtecResearchError, match="parameters do not match"):
        load_validated_normal_augmentations(regenerated_output, manifest_path)


def test_v3_manifest_detects_altered_seed_anchor_and_out_of_bounds_lens_recipe(tmp_path: Path) -> None:
    manifest_path = _frozen_manifest(tmp_path)
    output_dir = tmp_path / "outside-v3"
    generate_normal_augmentations(
        manifest_path, RECIPE_V3_PATH, output_dir, variants_per_parent=1, repository_root=tmp_path / "repo"
    )
    manifest_output = output_dir / "augmentation_manifest.json"
    altered_document = json.loads(manifest_output.read_text(encoding="utf-8"))
    altered_recipe = json.loads(RECIPE_V3_PATH.read_text(encoding="utf-8"))
    altered_recipe["samplingSeedAnchor"] = f"sha256:{'b' * 64}"
    altered_recipe_path = tmp_path / "altered-v3-recipe.json"
    altered_recipe_path.write_text(json.dumps(altered_recipe, indent=2) + "\n", encoding="utf-8")
    altered_document["recipePath"] = str(altered_recipe_path)
    altered_document["recipeSha256"] = sha256_file(altered_recipe_path)
    altered_document["recipe"] = altered_recipe
    _refresh_augmentation_manifest(manifest_output, altered_document)
    with pytest.raises(MvtecResearchError, match="parameters do not match"):
        load_validated_normal_augmentations(manifest_output, manifest_path)

    unsafe_recipe = json.loads(RECIPE_V3_PATH.read_text(encoding="utf-8"))
    unsafe_recipe["offAxisLensShading"]["maxCenterOffsetFraction"] = 0.251
    unsafe_recipe_path = tmp_path / "unsafe-v3-recipe.json"
    unsafe_recipe_path.write_text(json.dumps(unsafe_recipe, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MvtecResearchError, match="maxCenterOffsetFraction"):
        load_camera_recipe(unsafe_recipe_path)
