"""Reproducible, offline-only helpers for the MVTec AD research tools.

This module is intentionally isolated from PhoneDINO's runtime/artifact
contracts.  It creates and validates *normal-only* camera-condition
augmentations for non-commercial MVTec research.  It must never be used to
choose a production threshold or an equipment decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, JpegImagePlugin, __version__ as PILLOW_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CAMERA_AUGMENTATION_SCHEMA = "phone-dino.mvtec-ad-normal-augmentation/1.3"
CAMERA_RECIPE_SCHEMA = "phone-dino.mvtec-ad-camera-recipe/1.1"
FIXED_JPEG_QUALITY_CAMERA_RECIPE_SCHEMA = "phone-dino.mvtec-ad-camera-recipe/1.2"
LEGACY_CAMERA_RECIPE_SCHEMA = "phone-dino.mvtec-ad-camera-recipe/1.0"
NORMAL_AUGMENTATION_ROLES = frozenset({"FIT", "THRESHOLD_TUNING"})
V4_RECIPE_ID = "GENERIC_PHONE_CAPTURE_NORMAL_V4_NARROW_OFF_AXIS_LENS_SHADING_JPEG_420"
V5_RECIPE_ID = "GENERIC_PHONE_CAPTURE_NORMAL_V5_NARROW_OFF_AXIS_LENS_SHADING_JPEG_420_Q95"
V4_SAMPLING_SEED_ANCHOR = "sha256:1f0b49bb26066936a63122cc5c53588f5ad051156243464a31d20dbf03bf653f"
V4_BASELINE_GEOMETRY = {
    "maxCornerJitterFraction": 0.002,
    "maxRotationDegrees": 0.35,
    "maxScaleDelta": 0.005,
    "maxTranslationFraction": 0.002,
}
V4_BASELINE_PHOTOMETRY = {
    "maxExposureEv": 0.1,
    "maxGammaDelta": 0.025,
    "maxSensorNoiseStdDn": 0.5,
    "maxShadingStrength": 0.04,
    "maxVignetteStrength": 0.025,
    "maxWhiteBalanceDelta": 0.025,
}
V4_BASELINE_ENCODING_QUALITY = {"jpegQualityMin": 95, "jpegQualityMax": 98}
V4_BASELINE_LENS_SHADING = {"maxCenterOffsetFraction": 0.12, "maxStrength": 0.025}
V5_OUTPUT_JPEG_QUALITY = 95
V5_OUTPUT_QUANTIZATION_TABLES_SHA256 = "sha256:f67e35fd0dcd2fd9f999077e2aae8560e6327a8477c45427f6ea2e0a224cd187"


class MvtecResearchError(ValueError):
    """Raised when an offline MVTec research input breaks its protocol."""


def canonical_json_sha256(document: Any) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require_string(document: dict[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value.strip():
        raise MvtecResearchError(f"{name} must be a non-empty string")
    return value


def _require_sha256(document: dict[str, Any], name: str) -> str:
    value = _require_string(document, name)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise MvtecResearchError(f"{name} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise MvtecResearchError(f"{name} must be a sha256 digest") from error
    return value


def _require_mapping(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise MvtecResearchError(f"{name} must be an object")
    return value


def _require_exact_fields(
    document: dict[str, Any], *, name: str, required: frozenset[str], optional: frozenset[str] = frozenset()
) -> None:
    missing = required.difference(document)
    if missing:
        raise MvtecResearchError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    unknown = set(document).difference(required | optional)
    if unknown:
        raise MvtecResearchError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")


def _require_finite_number(document: dict[str, Any], name: str) -> float:
    value = document.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise MvtecResearchError(f"{name} must be a finite number")
    return float(value)


def _under(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _safe_relative_path(value: str, *, name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise MvtecResearchError(f"{name} must be a safe relative path")
    return path


def _document_digest(document: dict[str, Any], field: str) -> str:
    without_digest = dict(document)
    without_digest.pop(field, None)
    return canonical_json_sha256(without_digest)


def _camera_recipe_seed_anchor(recipe: dict[str, Any], recipe_sha256: str) -> str:
    """Return the immutable seed source for a controlled recipe experiment."""

    anchor = recipe.get("samplingSeedAnchor", recipe_sha256)
    if not isinstance(anchor, str) or len(anchor) != 71 or not anchor.startswith("sha256:"):
        raise MvtecResearchError("samplingSeedAnchor must be a sha256 digest")
    try:
        int(anchor[7:], 16)
    except ValueError as error:
        raise MvtecResearchError("samplingSeedAnchor must be a sha256 digest") from error
    return anchor


def _resolve_recipe_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _generator_provenance() -> dict[str, Any]:
    """Capture the code and runtime that materialized an external package."""

    try:
        revision = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        worktree_clean = not subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - Git availability is environment-specific
        revision = None
        worktree_clean = None
    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - generation itself requires these dependencies
        raise RuntimeError("MVTec camera augmentation requires the optional vision dependencies") from error
    entrypoint = REPOSITORY_ROOT / "tools" / "generate_mvtec_ad_normal_augmentations.py"
    return {
        "generatorModuleSha256": sha256_file(Path(__file__)),
        "generatorEntrypointSha256": sha256_file(entrypoint),
        "gitRevision": revision,
        "gitWorktreeClean": worktree_clean,
        "python": sys.version,
        "platform": platform.platform(),
        "pillowVersion": PILLOW_VERSION,
        "opencvVersion": cv2.__version__,
        "numpyVersion": np.__version__,
    }


def _output_encoding_profile(recipe: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Return the only JPEG encoding profile approved for a validated recipe."""

    schema_version = recipe.get("schemaVersion")
    if schema_version == LEGACY_CAMERA_RECIPE_SCHEMA:
        subsampling = "4:4:4"
    elif schema_version in {CAMERA_RECIPE_SCHEMA, FIXED_JPEG_QUALITY_CAMERA_RECIPE_SCHEMA}:
        encoding = _require_mapping(recipe, "encoding")
        subsampling = _require_string(encoding, "jpegSubsampling")
    else:
        raise MvtecResearchError("unsupported camera augmentation recipe schema")
    profiles = {
        "4:4:4": (
            0,
            {
                "format": "JPEG",
                "mode": "RGB",
                "subsampling": "4:4:4",
                "componentIds": [1, 2, 3],
                "samplingFactors": [[1, 1], [1, 1], [1, 1]],
                "quantizationTableSelectors": [0, 1, 1],
                "progressive": False,
            },
        ),
        "4:2:0": (
            2,
            {
                "format": "JPEG",
                "mode": "RGB",
                "subsampling": "4:2:0",
                "componentIds": [1, 2, 3],
                "samplingFactors": [[2, 2], [1, 1], [1, 1]],
                "quantizationTableSelectors": [0, 1, 1],
                "progressive": False,
            },
        ),
    }
    try:
        pillow_subsampling, profile = profiles[subsampling]
    except KeyError as error:
        raise MvtecResearchError("jpegSubsampling is outside the approved output-encoding profiles") from error
    return pillow_subsampling, profile


def _quantization_tables_sha256(opened: Image.Image) -> str:
    quantization = getattr(opened, "quantization", None)
    if not isinstance(quantization, dict) or set(quantization) != {0, 1}:
        raise MvtecResearchError("augmentation JPEG quantization tables are invalid")
    normalized: dict[str, list[int]] = {}
    for table_id in (0, 1):
        values = quantization.get(table_id)
        if not isinstance(values, (list, tuple)) or len(values) != 64:
            raise MvtecResearchError("augmentation JPEG quantization table length is invalid")
        table = list(values)
        if any(not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 255 for value in table):
            raise MvtecResearchError("augmentation JPEG quantization table values are invalid")
        normalized[str(table_id)] = table
    return canonical_json_sha256(normalized)


def _jpeg_output_encoding_from_opened_image(opened: Image.Image) -> dict[str, Any]:
    """Read only the JPEG coding profile needed for this research envelope."""

    if opened.format != "JPEG" or opened.mode != "RGB":
        raise MvtecResearchError("augmentation output must be an RGB JPEG")
    layer = getattr(opened, "layer", None)
    sampling = JpegImagePlugin.get_sampling(opened)
    if not isinstance(layer, list) or len(layer) != 3:
        raise MvtecResearchError("augmentation JPEG must have exactly three components")
    component_ids: list[int] = []
    factors: list[list[int]] = []
    table_selectors: list[int] = []
    for component in layer:
        if not isinstance(component, tuple) or len(component) != 4:
            raise MvtecResearchError("augmentation JPEG component metadata is invalid")
        component_id, horizontal, vertical, table_selector = component
        if (
            not isinstance(component_id, int)
            or isinstance(component_id, bool)
            or component_id <= 0
            or not isinstance(horizontal, int)
            or isinstance(horizontal, bool)
            or not isinstance(vertical, int)
            or isinstance(vertical, bool)
            or horizontal <= 0
            or vertical <= 0
            or not isinstance(table_selector, int)
            or isinstance(table_selector, bool)
            or table_selector < 0
        ):
            raise MvtecResearchError("augmentation JPEG component metadata is invalid")
        component_ids.append(component_id)
        factors.append([horizontal, vertical])
        table_selectors.append(table_selector)
    sampling_names = {0: "4:4:4", 2: "4:2:0"}
    if sampling not in sampling_names:
        raise MvtecResearchError("augmentation JPEG subsampling is outside the approved output-encoding profiles")
    return {
        "format": "JPEG",
        "mode": "RGB",
        "subsampling": sampling_names[sampling],
        "componentIds": component_ids,
        "samplingFactors": factors,
        "quantizationTableSelectors": table_selectors,
        "progressive": bool(opened.info.get("progressive") or opened.info.get("progression")),
        "quantizationTablesSha256": _quantization_tables_sha256(opened),
    }


def _inspect_jpeg_output_encoding(path: Path) -> dict[str, Any]:
    """Decode a generated JPEG header and return the auditable encoding profile."""

    try:
        with Image.open(path) as opened:
            opened.load()
            return _jpeg_output_encoding_from_opened_image(opened)
    except MvtecResearchError:
        raise
    except (OSError, SyntaxError, ValueError) as error:
        raise MvtecResearchError("unable to decode augmentation JPEG output") from error


@lru_cache(maxsize=8)
def _expected_jpeg_output_encoding(pillow_subsampling: int, jpeg_quality: int) -> dict[str, Any]:
    """Derive the approved JPEG tables from the exact Pillow encoder arguments."""

    if not 0 <= jpeg_quality <= 100:
        raise MvtecResearchError("JPEG quality is outside the supported encoder range")
    buffer = BytesIO()
    Image.new("RGB", (16, 16), (90, 130, 180)).save(
        buffer,
        format="JPEG",
        quality=jpeg_quality,
        subsampling=pillow_subsampling,
        optimize=False,
        progressive=False,
    )
    buffer.seek(0)
    try:
        with Image.open(buffer) as opened:
            opened.load()
            return _jpeg_output_encoding_from_opened_image(opened)
    except MvtecResearchError:
        raise
    except (OSError, SyntaxError, ValueError) as error:
        raise MvtecResearchError("unable to derive the approved JPEG output encoding") from error


def _expected_output_encoding(recipe: dict[str, Any], jpeg_quality: int) -> dict[str, Any]:
    pillow_subsampling, static_profile = _output_encoding_profile(recipe)
    expected = _expected_jpeg_output_encoding(pillow_subsampling, jpeg_quality)
    if {name: expected[name] for name in static_profile} != static_profile:
        raise MvtecResearchError("installed Pillow encoder does not emit the approved JPEG output profile")
    if recipe.get("schemaVersion") == FIXED_JPEG_QUALITY_CAMERA_RECIPE_SCHEMA:
        expected_tables = _require_sha256(_require_mapping(recipe, "encoding"), "jpegQuantizationTablesSha256")
        if expected["quantizationTablesSha256"] != expected_tables:
            raise MvtecResearchError("installed Pillow encoder does not emit the locked V5 Q95 quantization tables")
    return expected


def _validate_output_encoding(value: Any, expected: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise MvtecResearchError("augmentation outputEncoding must be an object")
    _require_exact_fields(
        value,
        name="augmentation outputEncoding",
        required=frozenset({
            "format",
            "mode",
            "subsampling",
            "componentIds",
            "samplingFactors",
            "quantizationTableSelectors",
            "progressive",
            "quantizationTablesSha256",
        }),
    )
    if value != expected:
        raise MvtecResearchError("augmentation outputEncoding does not match the frozen recipe")


def _validate_v4_baseline(recipe: dict[str, Any]) -> None:
    """Lock schema 1.1 to V3 parameters so V4 changes only chroma sampling."""

    if recipe.get("id") != V4_RECIPE_ID:
        raise MvtecResearchError("camera recipe 1.1 is reserved for the locked V4 JPEG 4:2:0 experiment")
    if recipe.get("samplingSeedAnchor") != V4_SAMPLING_SEED_ANCHOR:
        raise MvtecResearchError("camera recipe 1.1 must retain the V2 samplingSeedAnchor")
    expected_sections = {
        "geometry": V4_BASELINE_GEOMETRY,
        "photometry": V4_BASELINE_PHOTOMETRY,
        "offAxisLensShading": V4_BASELINE_LENS_SHADING,
    }
    for name, expected in expected_sections.items():
        if recipe.get(name) != expected:
            raise MvtecResearchError(f"camera recipe 1.1 must retain the locked V3 {name} values")
    encoding = _require_mapping(recipe, "encoding")
    if {name: encoding.get(name) for name in V4_BASELINE_ENCODING_QUALITY} != V4_BASELINE_ENCODING_QUALITY:
        raise MvtecResearchError("camera recipe 1.1 must retain the locked V3 JPEG quality range")


def _validate_v5_baseline(recipe: dict[str, Any]) -> None:
    """Lock schema 1.2 to V4 samples with a fixed Q95 output profile only."""

    if recipe.get("id") != V5_RECIPE_ID:
        raise MvtecResearchError("camera recipe 1.2 is reserved for the locked V5 JPEG Q95 experiment")
    if recipe.get("samplingSeedAnchor") != V4_SAMPLING_SEED_ANCHOR:
        raise MvtecResearchError("camera recipe 1.2 must retain the V2 samplingSeedAnchor")
    expected_sections = {
        "geometry": V4_BASELINE_GEOMETRY,
        "photometry": V4_BASELINE_PHOTOMETRY,
        "offAxisLensShading": V4_BASELINE_LENS_SHADING,
    }
    for name, expected in expected_sections.items():
        if recipe.get(name) != expected:
            raise MvtecResearchError(f"camera recipe 1.2 must retain the locked V4 {name} values")
    encoding = _require_mapping(recipe, "encoding")
    expected_encoding = {
        **V4_BASELINE_ENCODING_QUALITY,
        "jpegSubsampling": "4:2:0",
        "jpegQualityOutputOverride": V5_OUTPUT_JPEG_QUALITY,
        "jpegQuantizationTablesSha256": V5_OUTPUT_QUANTIZATION_TABLES_SHA256,
    }
    if encoding != expected_encoding:
        raise MvtecResearchError("camera recipe 1.2 must retain the locked V4 sampling range and fixed Q95 output")


def output_jpeg_quality(recipe: dict[str, Any], sampled_jpeg_quality: int) -> int:
    """Return the encoder quality without changing the established sampling stream.

    V5 retains the V4 random ``jpegQuality`` draw in ``parameters`` so geometry,
    photometry, noise, and all later RNG draws remain comparable.  It alone
    overrides the quality passed to Pillow, yielding a fixed Q95 coding profile.
    """

    if not isinstance(sampled_jpeg_quality, int) or isinstance(sampled_jpeg_quality, bool):
        raise MvtecResearchError("sampled JPEG quality must be an integer")
    if recipe.get("schemaVersion") != FIXED_JPEG_QUALITY_CAMERA_RECIPE_SCHEMA:
        return sampled_jpeg_quality
    override = _require_finite_number(_require_mapping(recipe, "encoding"), "jpegQualityOutputOverride")
    if not override.is_integer():
        raise MvtecResearchError("jpegQualityOutputOverride must be an integer")
    return int(override)


def load_camera_recipe(recipe_path: Path) -> tuple[dict[str, Any], str]:
    """Load the bounded generic camera/lighting recipe used by this protocol."""

    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MvtecResearchError(f"Unable to read augmentation recipe: {recipe_path}") from error
    if not isinstance(recipe, dict) or recipe.get("schemaVersion") not in {
        LEGACY_CAMERA_RECIPE_SCHEMA,
        CAMERA_RECIPE_SCHEMA,
        FIXED_JPEG_QUALITY_CAMERA_RECIPE_SCHEMA,
    }:
        raise MvtecResearchError("unsupported camera augmentation recipe schema")
    _require_exact_fields(
        recipe,
        name="camera augmentation recipe",
        required=frozenset({
            "schemaVersion",
            "id",
            "description",
            "authoritative",
            "productionAuthorized",
            "blindPolicy",
            "geometry",
            "photometry",
            "encoding",
        }),
        optional=frozenset({"samplingSeedAnchor", "offAxisLensShading"}),
    )
    _require_string(recipe, "id")
    _require_string(recipe, "description")
    if recipe.get("authoritative") is not False or recipe.get("productionAuthorized") is not False:
        raise MvtecResearchError("camera augmentation recipe must be explicitly non-authoritative and non-production")
    if recipe.get("blindPolicy") != "BLIND_ORIGINAL_ONLY":
        raise MvtecResearchError("camera augmentation recipe must keep blind inputs original")

    geometry = _require_mapping(recipe, "geometry")
    photometry = _require_mapping(recipe, "photometry")
    encoding = _require_mapping(recipe, "encoding")
    _require_exact_fields(
        geometry,
        name="camera augmentation geometry",
        required=frozenset({
            "maxRotationDegrees",
            "maxScaleDelta",
            "maxTranslationFraction",
            "maxCornerJitterFraction",
        }),
    )
    _require_exact_fields(
        photometry,
        name="camera augmentation photometry",
        required=frozenset({
            "maxExposureEv",
            "maxGammaDelta",
            "maxWhiteBalanceDelta",
            "maxShadingStrength",
            "maxVignetteStrength",
            "maxSensorNoiseStdDn",
        }),
    )
    if recipe["schemaVersion"] == LEGACY_CAMERA_RECIPE_SCHEMA:
        _require_exact_fields(
            encoding,
            name="legacy camera augmentation encoding",
            required=frozenset({"jpegQualityMin", "jpegQualityMax"}),
        )
    elif recipe["schemaVersion"] == CAMERA_RECIPE_SCHEMA:
        _require_exact_fields(
            encoding,
            name="camera augmentation encoding",
            required=frozenset({"jpegQualityMin", "jpegQualityMax", "jpegSubsampling"}),
        )
        if _require_string(encoding, "jpegSubsampling") != "4:2:0":
            raise MvtecResearchError("jpegSubsampling is outside the approved output-encoding profiles")
        if "samplingSeedAnchor" not in recipe or "offAxisLensShading" not in recipe:
            raise MvtecResearchError("camera recipe 1.1 must retain the controlled V3 sampling anchor and lens shading")
    else:
        _require_exact_fields(
            encoding,
            name="fixed-quality camera augmentation encoding",
            required=frozenset({
                "jpegQualityMin", "jpegQualityMax", "jpegSubsampling", "jpegQualityOutputOverride",
                "jpegQuantizationTablesSha256",
            }),
        )
        if _require_string(encoding, "jpegSubsampling") != "4:2:0":
            raise MvtecResearchError("jpegSubsampling is outside the approved output-encoding profiles")
        if "samplingSeedAnchor" not in recipe or "offAxisLensShading" not in recipe:
            raise MvtecResearchError("camera recipe 1.2 must retain the controlled V3 sampling anchor and lens shading")
    bounds = (
        (geometry, "maxRotationDegrees", 0.0, 1.0),
        (geometry, "maxScaleDelta", 0.0, 0.015),
        (geometry, "maxTranslationFraction", 0.0, 0.005),
        (geometry, "maxCornerJitterFraction", 0.0, 0.005),
        (photometry, "maxExposureEv", 0.0, 0.35),
        (photometry, "maxGammaDelta", 0.0, 0.08),
        (photometry, "maxWhiteBalanceDelta", 0.0, 0.08),
        (photometry, "maxShadingStrength", 0.0, 0.15),
        (photometry, "maxVignetteStrength", 0.0, 0.10),
        (photometry, "maxSensorNoiseStdDn", 0.0, 1.5),
    )
    for section, name, minimum, maximum in bounds:
        value = _require_finite_number(section, name)
        if value < minimum or value > maximum:
            raise MvtecResearchError(f"{name} is outside the approved generic-simulation range")
    quality_min = _require_finite_number(encoding, "jpegQualityMin")
    quality_max = _require_finite_number(encoding, "jpegQualityMax")
    if not quality_min.is_integer() or not quality_max.is_integer() or not (90 <= quality_min <= quality_max <= 98):
        raise MvtecResearchError("JPEG quality must be an integer range within 90..98")
    if recipe["schemaVersion"] == FIXED_JPEG_QUALITY_CAMERA_RECIPE_SCHEMA:
        output_quality = _require_finite_number(encoding, "jpegQualityOutputOverride")
        if not output_quality.is_integer() or not 90 <= output_quality <= 98:
            raise MvtecResearchError("jpegQualityOutputOverride must be an integer within 90..98")
        _require_sha256(encoding, "jpegQuantizationTablesSha256")
    recipe_sha256 = sha256_file(recipe_path)
    _camera_recipe_seed_anchor(recipe, recipe_sha256)
    off_axis_lens_shading = recipe.get("offAxisLensShading")
    if off_axis_lens_shading is not None:
        lens = _require_mapping(recipe, "offAxisLensShading")
        _require_exact_fields(
            lens,
            name="camera augmentation offAxisLensShading",
            required=frozenset({"maxStrength", "maxCenterOffsetFraction"}),
        )
        maximum_strength = _require_finite_number(lens, "maxStrength")
        maximum_offset = _require_finite_number(lens, "maxCenterOffsetFraction")
        if not 0.0 <= maximum_strength <= 0.04:
            raise MvtecResearchError("offAxisLensShading maxStrength is outside the approved generic-simulation range")
        if not 0.0 <= maximum_offset <= 0.25:
            raise MvtecResearchError("offAxisLensShading maxCenterOffsetFraction is outside the approved generic-simulation range")
    if recipe["schemaVersion"] == CAMERA_RECIPE_SCHEMA:
        _validate_v4_baseline(recipe)
    elif recipe["schemaVersion"] == FIXED_JPEG_QUALITY_CAMERA_RECIPE_SCHEMA:
        _validate_v5_baseline(recipe)
    _output_encoding_profile(recipe)
    return recipe, recipe_sha256


def derive_augmentation_seed(recipe_sha256: str, parent_case_id: str, parent_source_sha256: str, variant_id: int) -> int:
    if variant_id <= 0:
        raise MvtecResearchError("variant_id must be positive")
    payload = "\0".join((recipe_sha256, parent_case_id, parent_source_sha256, str(variant_id))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def sample_camera_parameters(recipe: dict[str, Any], *, recipe_sha256: str, parent_case_id: str,
                             parent_source_sha256: str, variant_id: int) -> dict[str, Any]:
    """Return stable generic phone-capture perturbations for one normal input."""

    sampling_seed_anchor = _camera_recipe_seed_anchor(recipe, recipe_sha256)
    seed = derive_augmentation_seed(sampling_seed_anchor, parent_case_id, parent_source_sha256, variant_id)
    rng = random.Random(seed)
    geometry = _require_mapping(recipe, "geometry")
    photometry = _require_mapping(recipe, "photometry")
    encoding = _require_mapping(recipe, "encoding")

    def symmetric(section: dict[str, Any], name: str) -> float:
        return rng.uniform(-_require_finite_number(section, name), _require_finite_number(section, name))

    def rounded(value: float) -> float:
        return round(value, 8)

    parameters = {
        "seed": str(seed),
        "rotationDegrees": rounded(symmetric(geometry, "maxRotationDegrees")),
        "scale": rounded(1.0 + symmetric(geometry, "maxScaleDelta")),
        "translationFractionX": rounded(symmetric(geometry, "maxTranslationFraction")),
        "translationFractionY": rounded(symmetric(geometry, "maxTranslationFraction")),
        "cornerJitterFractions": [rounded(symmetric(geometry, "maxCornerJitterFraction")) for _ in range(8)],
        "exposureEv": rounded(symmetric(photometry, "maxExposureEv")),
        "gamma": rounded(1.0 + symmetric(photometry, "maxGammaDelta")),
        "redGain": rounded(1.0 + symmetric(photometry, "maxWhiteBalanceDelta")),
        "blueGain": rounded(1.0 + symmetric(photometry, "maxWhiteBalanceDelta")),
        "shadingStrength": rounded(rng.uniform(0.0, _require_finite_number(photometry, "maxShadingStrength"))),
        "shadingAngleDegrees": rounded(rng.uniform(0.0, 360.0)),
        "vignetteStrength": rounded(rng.uniform(0.0, _require_finite_number(photometry, "maxVignetteStrength"))),
        "sensorNoiseStdDn": rounded(rng.uniform(0.0, _require_finite_number(photometry, "maxSensorNoiseStdDn"))),
        "jpegQuality": int(rng.randint(int(encoding["jpegQualityMin"]), int(encoding["jpegQualityMax"]))),
        "noiseSeed": str(rng.getrandbits(64)),
    }
    off_axis_lens_shading = recipe.get("offAxisLensShading")
    if off_axis_lens_shading is not None:
        lens = _require_mapping(recipe, "offAxisLensShading")
        parameters |= {
            "offAxisLensShadingStrength": rounded(rng.uniform(0.0, _require_finite_number(lens, "maxStrength"))),
            "lensShadingCenterOffsetXFraction": rounded(symmetric(lens, "maxCenterOffsetFraction")),
            "lensShadingCenterOffsetYFraction": rounded(symmetric(lens, "maxCenterOffsetFraction")),
        }
    return parameters


def validate_normal_augmentation_parent(record: dict[str, Any]) -> None:
    """Reject any input that could turn blind/anomalous data into development data."""

    role = _require_string(record, "role")
    kind = _require_string(record, "kind")
    if role not in NORMAL_AUGMENTATION_ROLES:
        raise MvtecResearchError("only FIT and THRESHOLD_TUNING records may be augmented")
    if kind != "NOMINAL":
        raise MvtecResearchError("only nominal records may be augmented")
    if record.get("defect") != "good":
        raise MvtecResearchError("only good nominal records may be augmented")
    if record.get("maskRelativePath") is not None or record.get("maskSourcePath") is not None:
        raise MvtecResearchError("mask-bearing records may not be augmented")
    _require_string(record, "caseId")
    _require_string(record, "category")
    _require_sha256(record, "sourceSha256")
    _safe_relative_path(_require_string(record, "relativePath"), name="relativePath")


def apply_camera_augmentation(image: Image.Image, parameters: dict[str, Any]) -> Image.Image:
    """Apply bounded geometry and camera/lighting simulation to one RGB image.

    The recipe intentionally excludes crop, flip, cutout, synthetic defects,
    strong glare and blur: those are not safe normal-label-preserving inputs.
    """

    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment-specific dependency guard
        raise RuntimeError("MVTec camera augmentation requires the optional vision dependencies") from error

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    if height < 2 or width < 2:
        raise MvtecResearchError("image is too small for camera augmentation")
    source = np.asarray([
        [0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)],
    ], dtype=np.float32)
    angle = math.radians(float(parameters["rotationDegrees"]))
    scale = float(parameters["scale"])
    center = np.asarray([[(width - 1) / 2.0], [(height - 1) / 2.0]], dtype=np.float32)
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    destination = ((source - center.reshape(1, 2)) @ rotation.T * scale) + center.reshape(1, 2)
    destination[:, 0] += float(parameters["translationFractionX"]) * width
    destination[:, 1] += float(parameters["translationFractionY"]) * height
    jitters = parameters["cornerJitterFractions"]
    if not isinstance(jitters, list) or len(jitters) != 8:
        raise MvtecResearchError("cornerJitterFractions must contain eight values")
    destination += np.asarray(jitters, dtype=np.float32).reshape(4, 2) * min(width, height)
    homography = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
    warped = cv2.warpPerspective(rgb, homography, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    values = warped.astype(np.float32) / 255.0
    values *= 2.0 ** float(parameters["exposureEv"])
    values[..., 0] *= float(parameters["redGain"])
    values[..., 2] *= float(parameters["blueGain"])
    values = np.power(np.clip(values, 0.0, 1.0), float(parameters["gamma"]))
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (x / max(width - 1, 1)) * 2.0 - 1.0
    y = (y / max(height - 1, 1)) * 2.0 - 1.0
    shading_angle = math.radians(float(parameters["shadingAngleDegrees"]))
    directional = x * math.cos(shading_angle) + y * math.sin(shading_angle)
    shading = 1.0 - float(parameters["shadingStrength"]) * (directional + 1.0) / 2.0
    radius_squared = np.minimum(1.0, x * x + y * y)
    vignette = 1.0 - float(parameters["vignetteStrength"]) * radius_squared
    lens_shading = np.ones_like(vignette)
    lens_keys = {
        "offAxisLensShadingStrength",
        "lensShadingCenterOffsetXFraction",
        "lensShadingCenterOffsetYFraction",
    }
    present_lens_keys = lens_keys.intersection(parameters)
    if present_lens_keys and present_lens_keys != lens_keys:
        raise MvtecResearchError("off-axis lens shading parameters must be complete")
    if present_lens_keys:
        lens_strength = _require_finite_number(parameters, "offAxisLensShadingStrength")
        lens_offset_x = _require_finite_number(parameters, "lensShadingCenterOffsetXFraction")
        lens_offset_y = _require_finite_number(parameters, "lensShadingCenterOffsetYFraction")
        if not 0.0 <= lens_strength <= 0.04 or abs(lens_offset_x) > 0.25 or abs(lens_offset_y) > 0.25:
            raise MvtecResearchError("off-axis lens shading parameters are outside the approved generic-simulation range")
        lens_radius_squared = np.minimum(1.0, (x - 2.0 * lens_offset_x) ** 2 + (y - 2.0 * lens_offset_y) ** 2)
        lens_shading = 1.0 - lens_strength * lens_radius_squared
    values *= (shading * vignette * lens_shading)[..., None]
    noise_rng = np.random.default_rng(int(parameters["noiseSeed"]))
    values += noise_rng.normal(0.0, float(parameters["sensorNoiseStdDn"]) / 255.0, values.shape).astype(np.float32)
    return Image.fromarray(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8))


def _prepare_output_directory(output_dir: Path, *, repository_root: Path) -> None:
    resolved_output = output_dir.resolve()
    if _under(repository_root, resolved_output):
        raise MvtecResearchError("augmentation output must stay outside the Git working tree")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MvtecResearchError("augmentation output directory must be empty; existing data is never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_source_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MvtecResearchError(f"Unable to read frozen MVTec manifest: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != "phone-dino.mvtec-ad-smoke/1.0":
        raise MvtecResearchError("unsupported frozen MVTec manifest schema")
    if manifest.get("authoritative") is not False:
        raise MvtecResearchError("frozen MVTec manifest must be non-authoritative")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise MvtecResearchError("frozen MVTec manifest has no records")
    return manifest


def generate_normal_augmentations(
    manifest_path: Path,
    recipe_path: Path,
    output_dir: Path,
    *,
    variants_per_parent: int,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Materialize a deterministic normal-only augmentation package outside Git."""

    if variants_per_parent <= 0 or variants_per_parent > 8:
        raise MvtecResearchError("variants_per_parent must be between 1 and 8")
    manifest = _load_source_manifest(manifest_path)
    recipe, recipe_sha256 = load_camera_recipe(recipe_path)
    pillow_subsampling, _ = _output_encoding_profile(recipe)
    _prepare_output_directory(output_dir, repository_root=repository_root)
    source_root = manifest_path.parent
    output_records: list[dict[str, Any]] = []
    parent_records = [record for record in manifest["records"] if isinstance(record, dict) and record.get("role") in NORMAL_AUGMENTATION_ROLES]
    for parent in sorted(parent_records, key=lambda item: str(item.get("caseId", ""))):
        validate_normal_augmentation_parent(parent)
        source_path = source_root / _safe_relative_path(str(parent["relativePath"]), name="relativePath")
        if sha256_file(source_path) != parent["sourceSha256"]:
            raise MvtecResearchError(f"parent input digest mismatch: {source_path}")
        with Image.open(source_path) as opened:
            source_image = opened.convert("RGB")
        for variant_id in range(1, variants_per_parent + 1):
            parameters = sample_camera_parameters(
                recipe,
                recipe_sha256=recipe_sha256,
                parent_case_id=str(parent["caseId"]),
                parent_source_sha256=str(parent["sourceSha256"]),
                variant_id=variant_id,
            )
            encoded_jpeg_quality = output_jpeg_quality(recipe, int(parameters["jpegQuality"]))
            child_name = hashlib.sha256(
                f"{parent['caseId']}\0{parent['sourceSha256']}\0{variant_id}".encode("utf-8")
            ).hexdigest()
            relative_path = Path("images") / f"{child_name}-v{variant_id}.jpg"
            target_path = output_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            augmented = apply_camera_augmentation(source_image, parameters)
            augmented.save(
                target_path,
                format="JPEG",
                quality=encoded_jpeg_quality,
                subsampling=pillow_subsampling,
                optimize=False,
                progressive=False,
            )
            expected_output_encoding = _expected_output_encoding(recipe, encoded_jpeg_quality)
            output_encoding = _inspect_jpeg_output_encoding(target_path)
            if output_encoding != expected_output_encoding:
                raise MvtecResearchError("generated JPEG headers do not match the frozen recipe")
            output_records.append({
                "caseId": f"{parent['caseId']}/camera-augmentation/{variant_id:02d}",
                "parentCaseId": parent["caseId"],
                "parentSourceSha256": parent["sourceSha256"],
                "category": parent["category"],
                "defect": "good",
                "kind": "NOMINAL",
                "role": parent["role"],
                "relativePath": relative_path.as_posix(),
                "sourceSha256": sha256_file(target_path),
                "variantId": variant_id,
                "parameters": parameters,
                "outputJpegQuality": encoded_jpeg_quality,
                "outputEncoding": output_encoding,
            })
    document: dict[str, Any] = {
        "schemaVersion": CAMERA_AUGMENTATION_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": "OFFLINE_MVTEC_RESEARCH_ONLY",
        "blindPolicy": "BLIND_ORIGINAL_ONLY",
        "sourceManifestPath": str(manifest_path),
        "sourceManifestFileSha256": sha256_file(manifest_path),
        "sourceManifestDeclaredSha256": manifest.get("manifestSha256"),
        "recipePath": str(recipe_path.resolve()),
        "recipeSha256": recipe_sha256,
        "recipe": recipe,
        "variantsPerParent": variants_per_parent,
        "generation": _generator_provenance(),
        "records": output_records,
    }
    document["augmentationManifestSha256"] = _document_digest(document, "augmentationManifestSha256")
    (output_dir / "augmentation_manifest.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def _validate_generation_provenance(document: dict[str, Any]) -> None:
    generation = _require_mapping(document, "generation")
    expected_hashes = {
        "generatorModuleSha256": sha256_file(Path(__file__)),
        "generatorEntrypointSha256": sha256_file(REPOSITORY_ROOT / "tools" / "generate_mvtec_ad_normal_augmentations.py"),
    }
    for name, expected in expected_hashes.items():
        if _require_sha256(generation, name) != expected:
            raise MvtecResearchError(f"augmentation {name} does not match this generator")
    for name in ("python", "platform", "pillowVersion", "opencvVersion", "numpyVersion"):
        _require_string(generation, name)
    if generation.get("gitRevision") is not None:
        _require_string(generation, "gitRevision")
    if generation.get("gitWorktreeClean") is not None and not isinstance(generation.get("gitWorktreeClean"), bool):
        raise MvtecResearchError("augmentation gitWorktreeClean must be a boolean or null")


def load_validated_normal_augmentations(augmentation_manifest_path: Path, source_manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate an augmentation package before an offline scorer can consume it."""

    source_manifest = _load_source_manifest(source_manifest_path)
    try:
        document = json.loads(augmentation_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MvtecResearchError(f"Unable to read augmentation manifest: {augmentation_manifest_path}") from error
    if not isinstance(document, dict) or document.get("schemaVersion") != CAMERA_AUGMENTATION_SCHEMA:
        raise MvtecResearchError("unsupported augmentation manifest schema")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise MvtecResearchError("augmentation manifest must be non-authoritative and non-production")
    if document.get("purpose") != "OFFLINE_MVTEC_RESEARCH_ONLY" or document.get("blindPolicy") != "BLIND_ORIGINAL_ONLY":
        raise MvtecResearchError("augmentation manifest has an unsafe research scope")
    if document.get("sourceManifestFileSha256") != sha256_file(source_manifest_path):
        raise MvtecResearchError("augmentation source manifest file digest does not match")
    if document.get("sourceManifestDeclaredSha256") != source_manifest.get("manifestSha256"):
        raise MvtecResearchError("augmentation source manifest declared digest does not match")
    if document.get("augmentationManifestSha256") != _document_digest(document, "augmentationManifestSha256"):
        raise MvtecResearchError("augmentation manifest digest does not match")
    recipe_sha256 = _require_sha256(document, "recipeSha256")
    recipe_path = _resolve_recipe_path(_require_string(document, "recipePath"))
    if not recipe_path.is_file():
        raise MvtecResearchError("augmentation recipe file is missing")
    recipe, actual_recipe_sha256 = load_camera_recipe(recipe_path)
    if actual_recipe_sha256 != recipe_sha256:
        raise MvtecResearchError("augmentation recipe digest does not match")
    if document.get("recipe") != recipe:
        raise MvtecResearchError("augmentation embedded recipe does not match the recipe file")
    _validate_generation_provenance(document)
    variants_per_parent = document.get("variantsPerParent")
    if not isinstance(variants_per_parent, int) or isinstance(variants_per_parent, bool) or not 1 <= variants_per_parent <= 8:
        raise MvtecResearchError("augmentation variantsPerParent must be between 1 and 8")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise MvtecResearchError("augmentation manifest has no records")
    parents = {str(record.get("caseId")): record for record in source_manifest["records"] if isinstance(record, dict)}
    eligible_parents = sorted(
        (record for record in parents.values() if isinstance(record, dict) and record.get("role") in NORMAL_AUGMENTATION_ROLES),
        key=lambda record: str(record.get("caseId", "")),
    )
    for parent in eligible_parents:
        validate_normal_augmentation_parent(parent)
    expected_case_ids = {
        f"{parent['caseId']}/camera-augmentation/{variant_id:02d}"
        for parent in eligible_parents
        for variant_id in range(1, variants_per_parent + 1)
    }
    output_root = augmentation_manifest_path.parent
    seen_case_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise MvtecResearchError("augmentation record is not an object")
        case_id = _require_string(record, "caseId")
        if case_id in seen_case_ids:
            raise MvtecResearchError("augmentation caseId is duplicated")
        seen_case_ids.add(case_id)
        parent_case_id = _require_string(record, "parentCaseId")
        parent = parents.get(parent_case_id)
        if not isinstance(parent, dict):
            raise MvtecResearchError("augmentation parent case is unknown")
        validate_normal_augmentation_parent(parent)
        if record.get("parentSourceSha256") != parent.get("sourceSha256"):
            raise MvtecResearchError("augmentation parent digest does not match")
        for name in ("category", "kind", "role"):
            if record.get(name) != parent.get(name):
                raise MvtecResearchError(f"augmentation {name} does not match its parent")
        if record.get("defect") != "good":
            raise MvtecResearchError("augmentation defect must remain good")
        relative_path = _safe_relative_path(_require_string(record, "relativePath"), name="relativePath")
        output_path = output_root / relative_path
        if not _under(output_root, output_path) or not output_path.is_file():
            raise MvtecResearchError("augmentation output file is missing or escapes its package")
        if sha256_file(output_path) != _require_sha256(record, "sourceSha256"):
            raise MvtecResearchError("augmentation output digest does not match")
        variant_id = record.get("variantId")
        if not isinstance(variant_id, int) or isinstance(variant_id, bool) or not 1 <= variant_id <= variants_per_parent:
            raise MvtecResearchError("augmentation variantId is outside the declared coverage")
        parameters = record.get("parameters")
        if not isinstance(parameters, dict):
            raise MvtecResearchError("augmentation parameters must be an object")
        expected_parameters = sample_camera_parameters(
            recipe,
            recipe_sha256=recipe_sha256,
            parent_case_id=parent_case_id,
            parent_source_sha256=str(parent["sourceSha256"]),
            variant_id=variant_id,
        )
        if parameters != expected_parameters:
            raise MvtecResearchError("augmentation parameters do not match the frozen recipe and parent")
        expected_output_jpeg_quality = output_jpeg_quality(recipe, int(expected_parameters["jpegQuality"]))
        record_output_jpeg_quality = record.get("outputJpegQuality")
        if (
            not isinstance(record_output_jpeg_quality, int)
            or isinstance(record_output_jpeg_quality, bool)
            or record_output_jpeg_quality != expected_output_jpeg_quality
        ):
            raise MvtecResearchError("augmentation outputJpegQuality does not match the frozen recipe and parent")
        expected_output_encoding = _expected_output_encoding(recipe, expected_output_jpeg_quality)
        output_encoding = record.get("outputEncoding")
        _validate_output_encoding(output_encoding, expected_output_encoding)
        if _inspect_jpeg_output_encoding(output_path) != expected_output_encoding:
            raise MvtecResearchError("augmentation outputEncoding does not match decoded JPEG headers")
        expected_case_id = f"{parent_case_id}/camera-augmentation/{variant_id:02d}"
        if case_id != expected_case_id:
            raise MvtecResearchError("augmentation caseId does not match its parent and variant")
        child_name = hashlib.sha256(
            f"{parent_case_id}\0{parent['sourceSha256']}\0{variant_id}".encode("utf-8")
        ).hexdigest()
        expected_relative_path = Path("images") / f"{child_name}-v{variant_id}.jpg"
        if relative_path != expected_relative_path:
            raise MvtecResearchError("augmentation relativePath does not match its parent and variant")
        validated.append(dict(record))
    if seen_case_ids != expected_case_ids:
        raise MvtecResearchError("augmentation records do not cover every eligible normal parent and declared variant")
    return document, validated
