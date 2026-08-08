from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from phone_dino.mvtec_research import (
    MvtecResearchError,
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


@pytest.mark.parametrize("recipe_path", [RECIPE_PATH, RECIPE_V2_PATH])
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
        recipe_sha256, "case/fit/001", f"sha256:{'1' * 64}", 1
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
