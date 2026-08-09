from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, JpegImagePlugin

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
RECIPE_V4_PATH = REPOSITORY_ROOT / "tools" / "mvtec_ad_camera_lighting_recipe_v4.json"


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


@pytest.mark.parametrize("recipe_path", [RECIPE_PATH, RECIPE_V2_PATH, RECIPE_V3_PATH, RECIPE_V4_PATH])
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
    _record("case/fit/001", "FIT", f"sha256:{'1' * 64}", "normal.png", defect="crack"),
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


@pytest.mark.parametrize("recipe_path", [RECIPE_PATH, RECIPE_V2_PATH, RECIPE_V3_PATH])
def test_legacy_recipes_emit_attested_jpeg_444(tmp_path: Path, recipe_path: Path) -> None:
    manifest_path = _frozen_manifest(tmp_path)
    output_dir = tmp_path / recipe_path.stem
    generated = generate_normal_augmentations(
        manifest_path, recipe_path, output_dir, variants_per_parent=1, repository_root=tmp_path / "repo"
    )
    expected = {
        "format": "JPEG",
        "mode": "RGB",
        "subsampling": "4:4:4",
        "componentIds": [1, 2, 3],
        "samplingFactors": [[1, 1], [1, 1], [1, 1]],
        "quantizationTableSelectors": [0, 1, 1],
        "progressive": False,
    }
    for record in generated["records"]:
        actual = _jpeg_encoding(output_dir / str(record["relativePath"]))
        assert record["outputEncoding"] == actual
        assert {name: actual[name] for name in expected} == expected
        assert str(actual["quantizationTablesSha256"]).startswith("sha256:")
    load_validated_normal_augmentations(output_dir / "augmentation_manifest.json", manifest_path)


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


def _jpeg_encoding(path: Path) -> dict[str, object]:
    with Image.open(path) as opened:
        assert opened.format == "JPEG"
        assert opened.mode == "RGB"
        opened.load()
        assert isinstance(opened.layer, list)
        assert isinstance(opened.quantization, dict)
        return {
            "format": "JPEG",
            "mode": "RGB",
            "subsampling": {0: "4:4:4", 2: "4:2:0"}[JpegImagePlugin.get_sampling(opened)],
            "componentIds": [component[0] for component in opened.layer],
            "samplingFactors": [[component[1], component[2]] for component in opened.layer],
            "quantizationTableSelectors": [component[3] for component in opened.layer],
            "progressive": bool(opened.info.get("progressive") or opened.info.get("progression")),
            "quantizationTablesSha256": canonical_json_sha256({
                str(table_id): list(opened.quantization[table_id]) for table_id in sorted(opened.quantization)
            }),
        }


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


def test_v4_preserves_every_v3_draw_and_changes_only_jpeg_subsampling() -> None:
    recipe_v3, recipe_v3_sha256 = load_camera_recipe(RECIPE_V3_PATH)
    recipe_v4, recipe_v4_sha256 = load_camera_recipe(RECIPE_V4_PATH)
    assert recipe_v4["samplingSeedAnchor"] == recipe_v3["samplingSeedAnchor"]
    assert recipe_v4["samplingSeedAnchor"] == sha256_file(RECIPE_V2_PATH)
    for parent_case_id, parent_source_sha256, variant_id in (
        ("case/fit/001", f"sha256:{'1' * 64}", 1),
        ("case/fit/002", f"sha256:{'2' * 64}", 4),
        ("case/tuning/001", f"sha256:{'3' * 64}", 2),
    ):
        arguments = {
            "parent_case_id": parent_case_id,
            "parent_source_sha256": parent_source_sha256,
            "variant_id": variant_id,
        }
        assert sample_camera_parameters(recipe_v4, recipe_sha256=recipe_v4_sha256, **arguments) == sample_camera_parameters(
            recipe_v3, recipe_sha256=recipe_v3_sha256, **arguments
        )


def test_v4_jpeg_420_manifest_is_attested_and_header_checked(tmp_path: Path) -> None:
    manifest_path = _frozen_manifest(tmp_path)
    output_dir = tmp_path / "outside-v4-r4"
    generated = generate_normal_augmentations(
        manifest_path, RECIPE_V4_PATH, output_dir, variants_per_parent=4, repository_root=tmp_path / "repo"
    )
    expected = {
        "format": "JPEG",
        "mode": "RGB",
        "subsampling": "4:2:0",
        "componentIds": [1, 2, 3],
        "samplingFactors": [[2, 2], [1, 1], [1, 1]],
        "quantizationTableSelectors": [0, 1, 1],
        "progressive": False,
    }
    assert generated["schemaVersion"] == "phone-dino.mvtec-ad-normal-augmentation/1.2"
    assert len(generated["records"]) == 8
    assert {record["role"] for record in generated["records"]} == {"FIT", "THRESHOLD_TUNING"}
    for record in generated["records"]:
        actual = _jpeg_encoding(output_dir / str(record["relativePath"]))
        assert record["outputEncoding"] == actual
        assert {name: actual[name] for name in expected} == expected
        assert str(actual["quantizationTablesSha256"]).startswith("sha256:")
    _, validated = load_validated_normal_augmentations(output_dir / "augmentation_manifest.json", manifest_path)
    assert len(validated) == 8

    wrong_metadata = json.loads((output_dir / "augmentation_manifest.json").read_text(encoding="utf-8"))
    wrong_metadata["records"][0]["outputEncoding"]["subsampling"] = "4:4:4"
    wrong_metadata["records"][0]["outputEncoding"]["samplingFactors"] = [[1, 1], [1, 1], [1, 1]]
    _refresh_augmentation_manifest(output_dir / "augmentation_manifest.json", wrong_metadata)
    with pytest.raises(MvtecResearchError, match="outputEncoding does not match the frozen recipe"):
        load_validated_normal_augmentations(output_dir / "augmentation_manifest.json", manifest_path)

    tampered_dir = tmp_path / "outside-v4-tampered"
    tampered = generate_normal_augmentations(
        manifest_path, RECIPE_V4_PATH, tampered_dir, variants_per_parent=1, repository_root=tmp_path / "repo"
    )
    tampered_path = tampered_dir / str(tampered["records"][0]["relativePath"])
    with Image.open(tampered_path) as opened:
        opened.convert("RGB").save(tampered_path, format="JPEG", quality=97, subsampling=0, optimize=False, progressive=False)
    tampered["records"][0]["sourceSha256"] = _sha256(tampered_path)
    _refresh_augmentation_manifest(tampered_dir / "augmentation_manifest.json", tampered)
    with pytest.raises(MvtecResearchError, match="outputEncoding does not match decoded JPEG headers"):
        load_validated_normal_augmentations(tampered_dir / "augmentation_manifest.json", manifest_path)

    quality_tampered_dir = tmp_path / "outside-v4-quality-tampered"
    quality_tampered = generate_normal_augmentations(
        manifest_path, RECIPE_V4_PATH, quality_tampered_dir, variants_per_parent=1, repository_root=tmp_path / "repo"
    )
    quality_tampered_path = quality_tampered_dir / str(quality_tampered["records"][0]["relativePath"])
    with Image.open(quality_tampered_path) as opened:
        opened.convert("RGB").save(
            quality_tampered_path, format="JPEG", quality=90, subsampling=2, optimize=False, progressive=False
        )
    quality_tampered["records"][0]["sourceSha256"] = _sha256(quality_tampered_path)
    _refresh_augmentation_manifest(quality_tampered_dir / "augmentation_manifest.json", quality_tampered)
    with pytest.raises(MvtecResearchError, match="outputEncoding does not match decoded JPEG headers"):
        load_validated_normal_augmentations(quality_tampered_dir / "augmentation_manifest.json", manifest_path)

    progressive_tampered_dir = tmp_path / "outside-v4-progressive-tampered"
    progressive_tampered = generate_normal_augmentations(
        manifest_path, RECIPE_V4_PATH, progressive_tampered_dir, variants_per_parent=1, repository_root=tmp_path / "repo"
    )
    progressive_tampered_path = progressive_tampered_dir / str(progressive_tampered["records"][0]["relativePath"])
    with Image.open(progressive_tampered_path) as opened:
        opened.convert("RGB").save(
            progressive_tampered_path,
            format="JPEG",
            quality=int(progressive_tampered["records"][0]["parameters"]["jpegQuality"]),
            subsampling=2,
            optimize=False,
            progressive=True,
        )
    progressive_tampered["records"][0]["sourceSha256"] = _sha256(progressive_tampered_path)
    _refresh_augmentation_manifest(progressive_tampered_dir / "augmentation_manifest.json", progressive_tampered)
    with pytest.raises(MvtecResearchError, match="outputEncoding does not match decoded JPEG headers"):
        load_validated_normal_augmentations(progressive_tampered_dir / "augmentation_manifest.json", manifest_path)


def test_v4_recipe_rejects_hidden_encoding_knobs_and_invalid_subsampling(tmp_path: Path) -> None:
    def write_recipe(name: str, recipe: dict[str, object]) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        return path

    unknown_root = json.loads(RECIPE_V4_PATH.read_text(encoding="utf-8"))
    unknown_root["progressive"] = False
    with pytest.raises(MvtecResearchError, match="unsupported fields"):
        load_camera_recipe(write_recipe("unknown-root.json", unknown_root))

    unknown_geometry = json.loads(RECIPE_V4_PATH.read_text(encoding="utf-8"))
    unknown_geometry["geometry"]["extraWarp"] = 0.001
    with pytest.raises(MvtecResearchError, match="unsupported fields"):
        load_camera_recipe(write_recipe("unknown-geometry.json", unknown_geometry))

    unknown_encoding = json.loads(RECIPE_V4_PATH.read_text(encoding="utf-8"))
    unknown_encoding["encoding"]["qtables"] = "custom"
    with pytest.raises(MvtecResearchError, match="unsupported fields"):
        load_camera_recipe(write_recipe("unknown-encoding.json", unknown_encoding))

    missing_subsampling = json.loads(RECIPE_V4_PATH.read_text(encoding="utf-8"))
    del missing_subsampling["encoding"]["jpegSubsampling"]
    with pytest.raises(MvtecResearchError, match="missing required fields"):
        load_camera_recipe(write_recipe("missing-subsampling.json", missing_subsampling))

    for label in ("4:4:4", "4:2:2"):
        invalid_subsampling = json.loads(RECIPE_V4_PATH.read_text(encoding="utf-8"))
        invalid_subsampling["encoding"]["jpegSubsampling"] = label
        with pytest.raises(MvtecResearchError, match="jpegSubsampling"):
            load_camera_recipe(write_recipe(f"invalid-{label.replace(':', '-')}.json", invalid_subsampling))

    changed_geometry = json.loads(RECIPE_V4_PATH.read_text(encoding="utf-8"))
    changed_geometry["geometry"]["maxRotationDegrees"] = 0.34
    with pytest.raises(MvtecResearchError, match="locked V3 geometry"):
        load_camera_recipe(write_recipe("changed-geometry.json", changed_geometry))

    changed_photometry = json.loads(RECIPE_V4_PATH.read_text(encoding="utf-8"))
    changed_photometry["photometry"]["maxExposureEv"] = 0.09
    with pytest.raises(MvtecResearchError, match="locked V3 photometry"):
        load_camera_recipe(write_recipe("changed-photometry.json", changed_photometry))

    changed_quality = json.loads(RECIPE_V4_PATH.read_text(encoding="utf-8"))
    changed_quality["encoding"]["jpegQualityMin"] = 96
    with pytest.raises(MvtecResearchError, match="locked V3 JPEG quality"):
        load_camera_recipe(write_recipe("changed-quality.json", changed_quality))

    changed_lens = json.loads(RECIPE_V4_PATH.read_text(encoding="utf-8"))
    changed_lens["offAxisLensShading"]["maxStrength"] = 0.02
    with pytest.raises(MvtecResearchError, match="locked V3 offAxisLensShading"):
        load_camera_recipe(write_recipe("changed-lens.json", changed_lens))

    changed_anchor = json.loads(RECIPE_V4_PATH.read_text(encoding="utf-8"))
    changed_anchor["samplingSeedAnchor"] = f"sha256:{'b' * 64}"
    with pytest.raises(MvtecResearchError, match="V2 samplingSeedAnchor"):
        load_camera_recipe(write_recipe("changed-anchor.json", changed_anchor))


def test_v3_manifest_requires_full_coverage_and_rederived_parameters(tmp_path: Path) -> None:
    manifest_path = _frozen_manifest(tmp_path)
    output_dir = tmp_path / "outside-v3"
    generated = generate_normal_augmentations(
        manifest_path, RECIPE_V3_PATH, output_dir, variants_per_parent=1, repository_root=tmp_path / "repo"
    )
    manifest_output = output_dir / "augmentation_manifest.json"
    assert generated["schemaVersion"] == "phone-dino.mvtec-ad-normal-augmentation/1.2"
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


def test_v3_multi_seed_manifest_covers_every_normal_parent_and_variant(tmp_path: Path) -> None:
    manifest_path = _frozen_manifest(tmp_path)
    output_dir = tmp_path / "outside-v3-r4"
    generated = generate_normal_augmentations(
        manifest_path, RECIPE_V3_PATH, output_dir, variants_per_parent=4, repository_root=tmp_path / "repo"
    )
    assert generated["variantsPerParent"] == 4
    by_parent: dict[str, set[int]] = {}
    for record in generated["records"]:
        by_parent.setdefault(str(record["parentCaseId"]), set()).add(int(record["variantId"]))
    assert by_parent == {"case/fit/001": {1, 2, 3, 4}, "case/tuning/001": {1, 2, 3, 4}}
    _, validated = load_validated_normal_augmentations(output_dir / "augmentation_manifest.json", manifest_path)
    assert len(validated) == 8


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
