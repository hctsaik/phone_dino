"""Fresh-cohort FIT-only camera augmentation for offline MVTec research.

This module deliberately does not reuse the historical V3--V5 augmentation
schemas or generators.  It consumes only the phase-safe normal-holdout view,
opens FIT parent images only, and materializes a fully deterministic external
package.  It is not device calibration, production authorization, or evidence
of defect-detection performance.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import stat
import subprocess
import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import Any

from PIL import Image, ImageOps, JpegImagePlugin, __version__ as PILLOW_VERSION

from phone_dino.mvtec_normal_holdout import (
    NORMAL_HOLDOUT_SCHEMA,
    NormalHoldoutError,
    canonical_json_sha256,
    load_evaluation_safe_normal_holdout_inputs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FRESH_FIT_AUGMENTATION_SCHEMA = "phone-dino.mvtec-ad-fresh-fit-augmentation/1.0"
FRESH_FIT_RECIPE_SCHEMA = "phone-dino.mvtec-ad-fresh-fit-camera-recipe/1.0"
FRESH_FIT_AUGMENTATION_PURPOSE = "OFFLINE_MVTEC_FRESH_FIT_AUGMENTATION_ONLY"
FRESH_FIT_RECIPE_ID = "FRESH_GENERIC_FIT_CAPTURE_V1_NARROW_CAMERA_420_Q95"
FRESH_FIT_RECIPE_DESCRIPTION = (
    "Closed generic fresh-cohort FIT-only camera prior: narrow geometry, exposure, white balance, "
    "low-frequency shading, off-axis lens shading, conservative Poisson-Gaussian-like sensor noise, "
    "and fixed JPEG 4:2:0 Q95. It is not device calibration or production authorization."
)
FRESH_FIT_INPUT_POLICY = "FIT_RAW_NORMAL_PARENTS_ONLY"
FRESH_FIT_BLIND_POLICY = "NO_BLIND_OR_ANOMALY_DATA"
FRESH_FIT_SAMPLING_ALGORITHM = "NAMED_SHA256_SUBSTREAMS_V1"
FRESH_FIT_CALIBRATION_EVIDENCE = {"state": "GENERIC_PRIOR_NOT_DEVICE_CALIBRATED"}
FRESH_FIT_GEOMETRY = {
    "maxCornerJitterFraction": 0.002,
    "maxRotationDegrees": 0.35,
    "maxScaleDelta": 0.005,
    "maxTranslationFraction": 0.002,
}
FRESH_FIT_PHOTOMETRY = {
    "maxDirectionalShadingStrength": 0.04,
    "maxExposureEv": 0.1,
    "maxGammaDelta": 0.025,
    "maxVignetteStrength": 0.025,
    "maxWhiteBalanceDelta": 0.025,
}
FRESH_FIT_LENS_SHADING = {"maxCenterOffsetFraction": 0.12, "maxStrength": 0.025}
FRESH_FIT_SENSOR_NOISE = {"maxReadNoiseStdDn": 0.35, "maxShotNoiseStdDnAtFullScale": 0.35}
FRESH_FIT_JPEG_Q95_TABLES_SHA256 = "sha256:f67e35fd0dcd2fd9f999077e2aae8560e6327a8477c45427f6ea2e0a224cd187"
FRESH_FIT_OUTPUT_ENCODING = {
    "format": "JPEG",
    "jpegQuality": 95,
    "jpegQuantizationTablesSha256": FRESH_FIT_JPEG_Q95_TABLES_SHA256,
    "jpegSubsampling": "4:2:0",
    "progressive": False,
}
RECIPE_FIELDS = {
    "schemaVersion",
    "id",
    "description",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "calibrationEvidence",
    "samplingAlgorithm",
    "geometry",
    "photometry",
    "lensShading",
    "sensorNoise",
    "outputEncoding",
}
AUGMENTATION_MANIFEST_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "inputPolicy",
    "blindPolicy",
    "parentPartition",
    "holdoutManifestFileSha256",
    "holdoutManifestDeclaredSha256",
    "developmentIdentitySha256",
    "fitParentIdentitySha256",
    "recipeFileSha256",
    "recipe",
    "variantsPerParent",
    "generation",
    "records",
    "augmentationManifestSha256",
}
GENERATION_FIELDS = {
    "generatorModuleSha256",
    "generatorEntrypointSha256",
    "normalHoldoutModuleSha256",
    "gitRevision",
    "gitWorktreeClean",
    "python",
    "platform",
    "pillowVersion",
    "opencvVersion",
    "numpyVersion",
}
AUGMENTATION_RECORD_FIELDS = {
    "caseId",
    "parentCaseId",
    "parentSourceSha256",
    "sourceGroupId",
    "category",
    "parentPartition",
    "kind",
    "defect",
    "variantId",
    "relativePath",
    "sourceSha256",
    "parameters",
    "outputEncoding",
}
PARAMETER_FIELDS = {"geometry", "photometry", "lensShading", "sensorNoise", "seeds"}
GEOMETRY_PARAMETER_FIELDS = {
    "rotationDegrees",
    "scale",
    "translationFractionX",
    "translationFractionY",
    "cornerJitterFractions",
}
PHOTOMETRY_PARAMETER_FIELDS = {
    "exposureEv",
    "gamma",
    "redGain",
    "blueGain",
    "directionalShadingStrength",
    "directionalShadingAngleDegrees",
    "vignetteStrength",
}
LENS_PARAMETER_FIELDS = {"strength", "centerOffsetXFraction", "centerOffsetYFraction"}
SENSOR_NOISE_PARAMETER_FIELDS = {"readNoiseStdDn", "shotNoiseStdDnAtFullScale"}
SEED_FIELDS = {"geometry", "photometry", "lensShading", "sensorNoise"}
OUTPUT_ENCODING_FIELDS = {
    "format",
    "mode",
    "subsampling",
    "componentIds",
    "samplingFactors",
    "quantizationTableSelectors",
    "progressive",
    "quantizationTablesSha256",
}


class FreshFitAugmentationError(ValueError):
    """Raised when a fresh FIT-only augmentation artifact breaks its protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return canonical_json_sha256(unsigned)


def _require_exact_fields(document: dict[str, Any], *, name: str, required: set[str]) -> None:
    missing = required.difference(document)
    if missing:
        raise FreshFitAugmentationError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    unknown = set(document).difference(required)
    if unknown:
        raise FreshFitAugmentationError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")


def _require_string(document: dict[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value.strip():
        raise FreshFitAugmentationError(f"{name} must be a non-empty string")
    return value


def _require_sha256(document: dict[str, Any], name: str) -> str:
    value = _require_string(document, name)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise FreshFitAugmentationError(f"{name} must be a SHA-256 digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise FreshFitAugmentationError(f"{name} must be a SHA-256 digest") from error
    return value


def _require_finite_number(document: dict[str, Any], name: str) -> float:
    value = document.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise FreshFitAugmentationError(f"{name} must be a finite number")
    return float(value)


def _require_positive_int(value: object, *, name: str, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FreshFitAugmentationError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise FreshFitAugmentationError(f"{name} must be at most {maximum}")
    return value


def _safe_relative_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FreshFitAugmentationError(f"{name} must be a non-empty safe relative path")
    windows = PureWindowsPath(value)
    path = Path(value)
    if windows.is_absolute() or windows.drive or windows.root or path.is_absolute() or path.root or not path.parts:
        raise FreshFitAugmentationError(f"{name} must be a safe relative path")
    if any(
        part in {"", ".", ".."}
        or ":" in part
        or part != part.rstrip(" .")
        or PureWindowsPath(part).is_reserved()
        for part in path.parts
    ):
        raise FreshFitAugmentationError(f"{name} must be a safe relative path")
    return path


def _is_under(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError:
        return False
    if stat.S_ISLNK(status.st_mode):
        return True
    attributes = getattr(status, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    return bool(attributes & reparse)


def _reject_links_on_existing_path(path: Path, *, description: str) -> None:
    current = path
    while True:
        if current.exists() or current.is_symlink():
            if _is_link_or_reparse_point(current):
                raise FreshFitAugmentationError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _require_external_input_file(path: Path, *, description: str, repository_root: Path) -> None:
    if _is_under(repository_root, path):
        raise FreshFitAugmentationError(f"{description} must stay outside the Git working tree")
    _reject_links_on_existing_path(path, description=description)
    if not path.is_file():
        raise FreshFitAugmentationError(f"{description} is missing")


def _safe_file_under(root: Path, relative: Path, *, description: str, repository_root: Path) -> Path:
    _reject_links_on_existing_path(root, description=description)
    candidate = root.joinpath(*relative.parts)
    _reject_links_on_existing_path(candidate, description=description)
    if not candidate.is_file() or not _is_under(root, candidate) or _is_under(repository_root, candidate):
        raise FreshFitAugmentationError(f"{description} is missing or escapes its external root")
    return candidate


def _parse_json_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreshFitAugmentationError(f"{description} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise FreshFitAugmentationError(f"{description} contains a non-finite JSON value: {value}")

    try:
        document = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshFitAugmentationError(f"unable to read {description}") from error
    if not isinstance(document, dict):
        raise FreshFitAugmentationError(f"{description} must be a JSON object")
    return document


def _read_external_json(path: Path, *, description: str, repository_root: Path) -> tuple[dict[str, Any], str]:
    _require_external_input_file(path, description=description, repository_root=repository_root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FreshFitAugmentationError(f"unable to read {description}") from error
    return _parse_json_bytes(raw, description=description), _sha256_bytes(raw)


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FreshFitAugmentationError(f"{name} must be an object")
    return value


def _expect_exact_mapping(value: object, *, name: str, expected: dict[str, Any]) -> None:
    mapping = _require_mapping(value, name=name)
    _require_exact_fields(mapping, name=name, required=set(expected))
    if mapping != expected:
        raise FreshFitAugmentationError(f"{name} does not match the fixed fresh FIT V1 recipe")


def load_fresh_fit_camera_recipe(recipe_path: Path) -> tuple[dict[str, Any], str]:
    """Load the one closed generic prior permitted for fresh FIT V1."""

    try:
        raw = recipe_path.read_bytes()
    except OSError as error:
        raise FreshFitAugmentationError("unable to read fresh FIT camera recipe") from error
    recipe = _parse_json_bytes(raw, description="fresh FIT camera recipe")
    _require_exact_fields(recipe, name="fresh FIT camera recipe", required=RECIPE_FIELDS)
    if recipe.get("schemaVersion") != FRESH_FIT_RECIPE_SCHEMA or recipe.get("id") != FRESH_FIT_RECIPE_ID:
        raise FreshFitAugmentationError("fresh FIT camera recipe schema or id is unsupported")
    if recipe.get("description") != FRESH_FIT_RECIPE_DESCRIPTION:
        raise FreshFitAugmentationError("fresh FIT camera recipe description is not the locked V1 declaration")
    if recipe.get("authoritative") is not False or recipe.get("productionAuthorized") is not False:
        raise FreshFitAugmentationError("fresh FIT camera recipe must be non-authoritative and non-production")
    if recipe.get("purpose") != FRESH_FIT_AUGMENTATION_PURPOSE:
        raise FreshFitAugmentationError("fresh FIT camera recipe has an unsafe purpose")
    _expect_exact_mapping(
        recipe.get("calibrationEvidence"),
        name="fresh FIT camera recipe calibrationEvidence",
        expected=FRESH_FIT_CALIBRATION_EVIDENCE,
    )
    if recipe.get("samplingAlgorithm") != FRESH_FIT_SAMPLING_ALGORITHM:
        raise FreshFitAugmentationError("fresh FIT camera recipe sampling algorithm is unsupported")
    _expect_exact_mapping(recipe.get("geometry"), name="fresh FIT camera recipe geometry", expected=FRESH_FIT_GEOMETRY)
    _expect_exact_mapping(recipe.get("photometry"), name="fresh FIT camera recipe photometry", expected=FRESH_FIT_PHOTOMETRY)
    _expect_exact_mapping(recipe.get("lensShading"), name="fresh FIT camera recipe lensShading", expected=FRESH_FIT_LENS_SHADING)
    _expect_exact_mapping(recipe.get("sensorNoise"), name="fresh FIT camera recipe sensorNoise", expected=FRESH_FIT_SENSOR_NOISE)
    _expect_exact_mapping(recipe.get("outputEncoding"), name="fresh FIT camera recipe outputEncoding", expected=FRESH_FIT_OUTPUT_ENCODING)
    return recipe, _sha256_bytes(raw)


def _derive_stream_seed(
    recipe_sha256: str,
    parent_case_id: str,
    parent_source_sha256: str,
    variant_id: int,
    stream_name: str,
) -> int:
    if variant_id <= 0:
        raise FreshFitAugmentationError("variantId must be positive")
    payload = "\0".join((
        FRESH_FIT_SAMPLING_ALGORITHM,
        recipe_sha256,
        parent_case_id,
        parent_source_sha256,
        str(variant_id),
        stream_name,
    )).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _rounded(value: float) -> float:
    return round(value, 8)


def _symmetric(rng: random.Random, maximum: float) -> float:
    return rng.uniform(-maximum, maximum)


def sample_fresh_fit_parameters(
    recipe: dict[str, Any],
    *,
    recipe_sha256: str,
    parent_case_id: str,
    parent_source_sha256: str,
    variant_id: int,
) -> dict[str, Any]:
    """Sample independent named substreams for one FIT parent and variant."""

    if recipe.get("samplingAlgorithm") != FRESH_FIT_SAMPLING_ALGORITHM:
        raise FreshFitAugmentationError("fresh FIT camera recipe sampling algorithm is unsupported")
    geometry_seed = _derive_stream_seed(recipe_sha256, parent_case_id, parent_source_sha256, variant_id, "geometry")
    photometry_seed = _derive_stream_seed(recipe_sha256, parent_case_id, parent_source_sha256, variant_id, "photometry")
    lens_seed = _derive_stream_seed(recipe_sha256, parent_case_id, parent_source_sha256, variant_id, "lensShading")
    noise_seed = _derive_stream_seed(recipe_sha256, parent_case_id, parent_source_sha256, variant_id, "sensorNoise")
    geometry_rng = random.Random(geometry_seed)
    photometry_rng = random.Random(photometry_seed)
    lens_rng = random.Random(lens_seed)
    noise_rng = random.Random(noise_seed)
    parameters = {
        "seeds": {
            "geometry": str(geometry_seed),
            "photometry": str(photometry_seed),
            "lensShading": str(lens_seed),
            "sensorNoise": str(noise_seed),
        },
        "geometry": {
            "rotationDegrees": _rounded(_symmetric(geometry_rng, FRESH_FIT_GEOMETRY["maxRotationDegrees"])),
            "scale": _rounded(1.0 + _symmetric(geometry_rng, FRESH_FIT_GEOMETRY["maxScaleDelta"])),
            "translationFractionX": _rounded(
                _symmetric(geometry_rng, FRESH_FIT_GEOMETRY["maxTranslationFraction"])
            ),
            "translationFractionY": _rounded(
                _symmetric(geometry_rng, FRESH_FIT_GEOMETRY["maxTranslationFraction"])
            ),
            "cornerJitterFractions": [
                _rounded(_symmetric(geometry_rng, FRESH_FIT_GEOMETRY["maxCornerJitterFraction"]))
                for _ in range(8)
            ],
        },
        "photometry": {
            "exposureEv": _rounded(_symmetric(photometry_rng, FRESH_FIT_PHOTOMETRY["maxExposureEv"])),
            "gamma": _rounded(1.0 + _symmetric(photometry_rng, FRESH_FIT_PHOTOMETRY["maxGammaDelta"])),
            "redGain": _rounded(
                1.0 + _symmetric(photometry_rng, FRESH_FIT_PHOTOMETRY["maxWhiteBalanceDelta"])
            ),
            "blueGain": _rounded(
                1.0 + _symmetric(photometry_rng, FRESH_FIT_PHOTOMETRY["maxWhiteBalanceDelta"])
            ),
            "directionalShadingStrength": _rounded(
                photometry_rng.uniform(0.0, FRESH_FIT_PHOTOMETRY["maxDirectionalShadingStrength"])
            ),
            "directionalShadingAngleDegrees": _rounded(photometry_rng.uniform(0.0, 360.0)),
            "vignetteStrength": _rounded(photometry_rng.uniform(0.0, FRESH_FIT_PHOTOMETRY["maxVignetteStrength"])),
        },
        "lensShading": {
            "strength": _rounded(lens_rng.uniform(0.0, FRESH_FIT_LENS_SHADING["maxStrength"])),
            "centerOffsetXFraction": _rounded(
                _symmetric(lens_rng, FRESH_FIT_LENS_SHADING["maxCenterOffsetFraction"])
            ),
            "centerOffsetYFraction": _rounded(
                _symmetric(lens_rng, FRESH_FIT_LENS_SHADING["maxCenterOffsetFraction"])
            ),
        },
        "sensorNoise": {
            "readNoiseStdDn": _rounded(noise_rng.uniform(0.0, FRESH_FIT_SENSOR_NOISE["maxReadNoiseStdDn"])),
            "shotNoiseStdDnAtFullScale": _rounded(
                noise_rng.uniform(0.0, FRESH_FIT_SENSOR_NOISE["maxShotNoiseStdDnAtFullScale"])
            ),
        },
    }
    _validate_parameters(parameters)
    return parameters


def _validate_parameter_section(
    value: object,
    *,
    name: str,
    fields: set[str],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    section = _require_mapping(value, name=name)
    _require_exact_fields(section, name=name, required=fields)
    for field, (minimum, maximum) in bounds.items():
        number = _require_finite_number(section, field)
        if number < minimum or number > maximum:
            raise FreshFitAugmentationError(f"{name}.{field} is outside the approved fresh FIT V1 range")
    return section


def _validate_parameters(value: object) -> dict[str, Any]:
    parameters = _require_mapping(value, name="fresh FIT augmentation parameters")
    _require_exact_fields(parameters, name="fresh FIT augmentation parameters", required=PARAMETER_FIELDS)
    seeds = _require_mapping(parameters.get("seeds"), name="fresh FIT augmentation parameters.seeds")
    _require_exact_fields(seeds, name="fresh FIT augmentation parameters.seeds", required=SEED_FIELDS)
    for name in sorted(SEED_FIELDS):
        seed = _require_string(seeds, name)
        if not seed.isdecimal() or int(seed) < 0 or int(seed) > 2**64 - 1:
            raise FreshFitAugmentationError(f"fresh FIT augmentation seed {name} is invalid")
    geometry = _validate_parameter_section(
        parameters.get("geometry"),
        name="fresh FIT augmentation geometry parameters",
        fields=GEOMETRY_PARAMETER_FIELDS,
        bounds={
            "rotationDegrees": (-FRESH_FIT_GEOMETRY["maxRotationDegrees"], FRESH_FIT_GEOMETRY["maxRotationDegrees"]),
            "scale": (1.0 - FRESH_FIT_GEOMETRY["maxScaleDelta"], 1.0 + FRESH_FIT_GEOMETRY["maxScaleDelta"]),
            "translationFractionX": (
                -FRESH_FIT_GEOMETRY["maxTranslationFraction"], FRESH_FIT_GEOMETRY["maxTranslationFraction"]
            ),
            "translationFractionY": (
                -FRESH_FIT_GEOMETRY["maxTranslationFraction"], FRESH_FIT_GEOMETRY["maxTranslationFraction"]
            ),
        },
    )
    corner_jitter = parameters["geometry"].get("cornerJitterFractions")
    if not isinstance(corner_jitter, list) or len(corner_jitter) != 8:
        raise FreshFitAugmentationError("fresh FIT augmentation cornerJitterFractions must contain eight values")
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        or abs(float(item)) > FRESH_FIT_GEOMETRY["maxCornerJitterFraction"]
        for item in corner_jitter
    ):
        raise FreshFitAugmentationError("fresh FIT augmentation cornerJitterFractions are outside the approved range")
    _validate_parameter_section(
        parameters.get("photometry"),
        name="fresh FIT augmentation photometry parameters",
        fields=PHOTOMETRY_PARAMETER_FIELDS,
        bounds={
            "exposureEv": (-FRESH_FIT_PHOTOMETRY["maxExposureEv"], FRESH_FIT_PHOTOMETRY["maxExposureEv"]),
            "gamma": (1.0 - FRESH_FIT_PHOTOMETRY["maxGammaDelta"], 1.0 + FRESH_FIT_PHOTOMETRY["maxGammaDelta"]),
            "redGain": (1.0 - FRESH_FIT_PHOTOMETRY["maxWhiteBalanceDelta"], 1.0 + FRESH_FIT_PHOTOMETRY["maxWhiteBalanceDelta"]),
            "blueGain": (1.0 - FRESH_FIT_PHOTOMETRY["maxWhiteBalanceDelta"], 1.0 + FRESH_FIT_PHOTOMETRY["maxWhiteBalanceDelta"]),
            "directionalShadingStrength": (0.0, FRESH_FIT_PHOTOMETRY["maxDirectionalShadingStrength"]),
            "directionalShadingAngleDegrees": (0.0, 360.0),
            "vignetteStrength": (0.0, FRESH_FIT_PHOTOMETRY["maxVignetteStrength"]),
        },
    )
    _validate_parameter_section(
        parameters.get("lensShading"),
        name="fresh FIT augmentation lens-shading parameters",
        fields=LENS_PARAMETER_FIELDS,
        bounds={
            "strength": (0.0, FRESH_FIT_LENS_SHADING["maxStrength"]),
            "centerOffsetXFraction": (
                -FRESH_FIT_LENS_SHADING["maxCenterOffsetFraction"], FRESH_FIT_LENS_SHADING["maxCenterOffsetFraction"]
            ),
            "centerOffsetYFraction": (
                -FRESH_FIT_LENS_SHADING["maxCenterOffsetFraction"], FRESH_FIT_LENS_SHADING["maxCenterOffsetFraction"]
            ),
        },
    )
    _validate_parameter_section(
        parameters.get("sensorNoise"),
        name="fresh FIT augmentation sensor-noise parameters",
        fields=SENSOR_NOISE_PARAMETER_FIELDS,
        bounds={
            "readNoiseStdDn": (0.0, FRESH_FIT_SENSOR_NOISE["maxReadNoiseStdDn"]),
            "shotNoiseStdDnAtFullScale": (0.0, FRESH_FIT_SENSOR_NOISE["maxShotNoiseStdDnAtFullScale"]),
        },
    )
    return parameters


def apply_fresh_fit_camera_augmentation(image: Image.Image, parameters: dict[str, Any]) -> Image.Image:
    """Apply a narrow generic phone-capture perturbation to one FIT image.

    V1 intentionally excludes crop, flip, blur, glare, occlusion, hot pixels,
    synthetic defects, and capture-reject effects.  It is a generic normal
    prior, not a device-calibrated reconstruction.
    """

    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment-specific dependency guard
        raise RuntimeError("fresh FIT augmentation requires the optional vision dependencies") from error
    _validate_parameters(parameters)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    if height < 2 or width < 2:
        raise FreshFitAugmentationError("image is too small for fresh FIT camera augmentation")
    geometry = parameters["geometry"]
    source = np.asarray(
        [[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]],
        dtype=np.float32,
    )
    angle = math.radians(float(geometry["rotationDegrees"]))
    scale = float(geometry["scale"])
    center = np.asarray([[(width - 1) / 2.0], [(height - 1) / 2.0]], dtype=np.float32)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32
    )
    destination = ((source - center.reshape(1, 2)) @ rotation.T * scale) + center.reshape(1, 2)
    destination[:, 0] += float(geometry["translationFractionX"]) * width
    destination[:, 1] += float(geometry["translationFractionY"]) * height
    destination += np.asarray(geometry["cornerJitterFractions"], dtype=np.float32).reshape(4, 2) * min(width, height)
    homography = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
    warped = cv2.warpPerspective(
        rgb,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    photometry = parameters["photometry"]
    values = warped.astype(np.float32) / 255.0
    values *= 2.0 ** float(photometry["exposureEv"])
    values[..., 0] *= float(photometry["redGain"])
    values[..., 2] *= float(photometry["blueGain"])
    values = np.power(np.clip(values, 0.0, 1.0), float(photometry["gamma"]))
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (x / max(width - 1, 1)) * 2.0 - 1.0
    y = (y / max(height - 1, 1)) * 2.0 - 1.0
    shading_angle = math.radians(float(photometry["directionalShadingAngleDegrees"]))
    directional = x * math.cos(shading_angle) + y * math.sin(shading_angle)
    directional_shading = 1.0 - float(photometry["directionalShadingStrength"]) * (directional + 1.0) / 2.0
    vignette = 1.0 - float(photometry["vignetteStrength"]) * np.minimum(1.0, x * x + y * y)
    lens = parameters["lensShading"]
    lens_radius_squared = np.minimum(
        1.0,
        (x - 2.0 * float(lens["centerOffsetXFraction"])) ** 2
        + (y - 2.0 * float(lens["centerOffsetYFraction"])) ** 2,
    )
    lens_shading = 1.0 - float(lens["strength"]) * lens_radius_squared
    values *= (directional_shading * vignette * lens_shading)[..., None]

    sensor_noise = parameters["sensorNoise"]
    noise_rng = np.random.default_rng(int(parameters["seeds"]["sensorNoise"]))
    signal = np.clip(values, 0.0, 1.0)
    standard_deviation_dn = np.sqrt(
        float(sensor_noise["readNoiseStdDn"]) ** 2
        + signal * float(sensor_noise["shotNoiseStdDnAtFullScale"]) ** 2
    )
    values += noise_rng.normal(0.0, 1.0, values.shape).astype(np.float32) * (standard_deviation_dn / 255.0)
    return Image.fromarray(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8))


def _quantization_tables_sha256(opened: Image.Image) -> str:
    quantization = getattr(opened, "quantization", None)
    if not isinstance(quantization, dict) or set(quantization) != {0, 1}:
        raise FreshFitAugmentationError("fresh FIT JPEG quantization tables are invalid")
    normalized: dict[str, list[int]] = {}
    for table_id in (0, 1):
        values = quantization.get(table_id)
        if not isinstance(values, (list, tuple)) or len(values) != 64:
            raise FreshFitAugmentationError("fresh FIT JPEG quantization table length is invalid")
        table = list(values)
        if any(not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 255 for value in table):
            raise FreshFitAugmentationError("fresh FIT JPEG quantization table values are invalid")
        normalized[str(table_id)] = table
    return canonical_json_sha256(normalized)


def _jpeg_output_encoding_from_opened_image(opened: Image.Image) -> dict[str, Any]:
    if opened.format != "JPEG" or opened.mode != "RGB":
        raise FreshFitAugmentationError("fresh FIT output must be an RGB JPEG")
    layer = getattr(opened, "layer", None)
    sampling = JpegImagePlugin.get_sampling(opened)
    if not isinstance(layer, list) or len(layer) != 3:
        raise FreshFitAugmentationError("fresh FIT JPEG must have exactly three components")
    component_ids: list[int] = []
    sampling_factors: list[list[int]] = []
    selectors: list[int] = []
    for component in layer:
        if not isinstance(component, tuple) or len(component) != 4:
            raise FreshFitAugmentationError("fresh FIT JPEG component metadata is invalid")
        component_id, horizontal, vertical, selector = component
        if (
            not isinstance(component_id, int)
            or isinstance(component_id, bool)
            or component_id <= 0
            or not isinstance(horizontal, int)
            or isinstance(horizontal, bool)
            or horizontal <= 0
            or not isinstance(vertical, int)
            or isinstance(vertical, bool)
            or vertical <= 0
            or not isinstance(selector, int)
            or isinstance(selector, bool)
            or selector < 0
        ):
            raise FreshFitAugmentationError("fresh FIT JPEG component metadata is invalid")
        component_ids.append(component_id)
        sampling_factors.append([horizontal, vertical])
        selectors.append(selector)
    sampling_names = {0: "4:4:4", 2: "4:2:0"}
    if sampling not in sampling_names:
        raise FreshFitAugmentationError("fresh FIT JPEG subsampling is unsupported")
    return {
        "format": "JPEG",
        "mode": "RGB",
        "subsampling": sampling_names[sampling],
        "componentIds": component_ids,
        "samplingFactors": sampling_factors,
        "quantizationTableSelectors": selectors,
        "progressive": bool(opened.info.get("progressive") or opened.info.get("progression")),
        "quantizationTablesSha256": _quantization_tables_sha256(opened),
    }


def _inspect_jpeg_output_encoding(data: bytes) -> dict[str, Any]:
    try:
        with Image.open(BytesIO(data)) as opened:
            opened.load()
            return _jpeg_output_encoding_from_opened_image(opened)
    except FreshFitAugmentationError:
        raise
    except (OSError, SyntaxError, ValueError) as error:
        raise FreshFitAugmentationError("unable to decode fresh FIT JPEG output") from error


@lru_cache(maxsize=1)
def _expected_output_encoding() -> dict[str, Any]:
    """Derive the exact V1 coding profile from the installed Pillow encoder."""

    buffer = BytesIO()
    Image.new("RGB", (16, 16), (90, 130, 180)).save(
        buffer,
        format="JPEG",
        quality=95,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    expected = _inspect_jpeg_output_encoding(buffer.getvalue())
    static = {
        "format": "JPEG",
        "mode": "RGB",
        "subsampling": "4:2:0",
        "componentIds": [1, 2, 3],
        "samplingFactors": [[2, 2], [1, 1], [1, 1]],
        "quantizationTableSelectors": [0, 1, 1],
        "progressive": False,
    }
    if {name: expected[name] for name in static} != static:
        raise FreshFitAugmentationError("installed Pillow does not emit the required fresh FIT V1 JPEG profile")
    if expected["quantizationTablesSha256"] != FRESH_FIT_JPEG_Q95_TABLES_SHA256:
        raise FreshFitAugmentationError("installed Pillow does not emit the locked fresh FIT V1 Q95 quantization tables")
    return expected


def _validate_output_encoding(value: object) -> dict[str, Any]:
    encoding = _require_mapping(value, name="fresh FIT augmentation outputEncoding")
    _require_exact_fields(encoding, name="fresh FIT augmentation outputEncoding", required=OUTPUT_ENCODING_FIELDS)
    expected = _expected_output_encoding()
    if encoding != expected:
        raise FreshFitAugmentationError("fresh FIT augmentation outputEncoding does not match the fixed recipe")
    return encoding


def _render_augmented_jpeg(image: Image.Image, parameters: dict[str, Any]) -> bytes:
    augmented = apply_fresh_fit_camera_augmentation(image, parameters)
    buffer = BytesIO()
    augmented.save(
        buffer,
        format="JPEG",
        quality=95,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    data = buffer.getvalue()
    if _inspect_jpeg_output_encoding(data) != _expected_output_encoding():
        raise FreshFitAugmentationError("fresh FIT JPEG headers do not match the fixed recipe")
    return data


def _load_parent_rgb(parent: dict[str, Any]) -> Image.Image:
    source_path = parent.get("imagePath")
    if not isinstance(source_path, Path):
        raise FreshFitAugmentationError("fresh FIT parent has no validated image path")
    if sha256_file(source_path) != parent.get("sourceSha256"):
        raise FreshFitAugmentationError("fresh FIT parent input changed before augmentation")
    try:
        with Image.open(source_path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, SyntaxError, ValueError) as error:
        raise FreshFitAugmentationError("unable to decode fresh FIT parent image") from error
    if sha256_file(source_path) != parent.get("sourceSha256"):
        raise FreshFitAugmentationError("fresh FIT parent input changed while it was decoded")
    return image


def _fit_parent_identity(parents: list[dict[str, Any]]) -> str:
    return canonical_json_sha256([
        {
            "caseId": parent["caseId"],
            "category": parent["category"],
            "sourceSha256": parent["sourceSha256"],
            "sourceGroupId": parent["sourceGroupId"],
            "partition": parent["partition"],
        }
        for parent in sorted(parents, key=lambda item: str(item["caseId"]))
    ])


def _expected_child_identity(parent: dict[str, Any], *, recipe_sha256: str, variant_id: int) -> tuple[str, Path]:
    case_id = f"{parent['caseId']}/fresh-fit-camera-v1/{variant_id:02d}"
    name = hashlib.sha256(
        "\0".join((
            FRESH_FIT_AUGMENTATION_SCHEMA,
            recipe_sha256,
            str(parent["caseId"]),
            str(parent["sourceSha256"]),
            str(variant_id),
        )).encode("utf-8")
    ).hexdigest()
    return case_id, Path("images") / f"{name}-v{variant_id}.jpg"


def _prepare_external_output_directory(output_dir: Path, *, repository_root: Path) -> None:
    if _is_under(repository_root, output_dir) or _is_under(output_dir, repository_root):
        raise FreshFitAugmentationError("fresh FIT augmentation output must stay outside the Git working tree")
    if output_dir.exists():
        raise FreshFitAugmentationError("fresh FIT augmentation output already exists; choose a new immutable path")
    _reject_links_on_existing_path(output_dir.parent, description="fresh FIT augmentation output")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise FreshFitAugmentationError("unable to create fresh FIT augmentation output directory") from error
    _reject_links_on_existing_path(output_dir, description="fresh FIT augmentation output")


def _require_external_package_root(root: Path, *, repository_root: Path) -> None:
    if _is_under(repository_root, root) or _is_under(root, repository_root):
        raise FreshFitAugmentationError("fresh FIT augmentation package must stay outside the Git working tree")
    _reject_links_on_existing_path(root, description="fresh FIT augmentation package")
    if not root.is_dir():
        raise FreshFitAugmentationError("fresh FIT augmentation package root is missing")


def _generation_provenance(*, repository_root: Path) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - generation itself requires these dependencies
        raise RuntimeError("fresh FIT augmentation requires the optional vision dependencies") from error
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        worktree_clean = not subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - environment-specific Git availability
        revision = None
        worktree_clean = None
    return {
        "generatorModuleSha256": sha256_file(Path(__file__)),
        "generatorEntrypointSha256": sha256_file(
            repository_root / "tools" / "generate_mvtec_ad_fresh_fit_augmentations.py"
        ),
        "normalHoldoutModuleSha256": sha256_file(repository_root / "src" / "phone_dino" / "mvtec_normal_holdout.py"),
        "gitRevision": revision,
        "gitWorktreeClean": worktree_clean,
        "python": sys.version,
        "platform": platform.platform(),
        "pillowVersion": PILLOW_VERSION,
        "opencvVersion": cv2.__version__,
        "numpyVersion": np.__version__,
    }


def _validate_generation_provenance(value: object, *, repository_root: Path) -> None:
    generation = _require_mapping(value, name="fresh FIT augmentation generation")
    _require_exact_fields(generation, name="fresh FIT augmentation generation", required=GENERATION_FIELDS)
    expected_hashes = {
        "generatorModuleSha256": sha256_file(Path(__file__)),
        "generatorEntrypointSha256": sha256_file(
            repository_root / "tools" / "generate_mvtec_ad_fresh_fit_augmentations.py"
        ),
        "normalHoldoutModuleSha256": sha256_file(repository_root / "src" / "phone_dino" / "mvtec_normal_holdout.py"),
    }
    for name, expected in expected_hashes.items():
        if _require_sha256(generation, name) != expected:
            raise FreshFitAugmentationError(f"fresh FIT augmentation {name} does not match this implementation")
    for name in ("python", "platform", "pillowVersion", "opencvVersion", "numpyVersion"):
        _require_string(generation, name)
    if generation.get("gitRevision") is not None:
        _require_string(generation, "gitRevision")
    if generation.get("gitWorktreeClean") is not None and not isinstance(generation.get("gitWorktreeClean"), bool):
        raise FreshFitAugmentationError("fresh FIT augmentation gitWorktreeClean must be a boolean or null")


def _validate_fit_parents(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not parents:
        raise FreshFitAugmentationError("fresh FIT augmentation has no FIT parent records")
    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for parent in parents:
        if parent.get("partition") != "FIT" or parent.get("kind") != "NOMINAL" or parent.get("defect") != "good":
            raise FreshFitAugmentationError("fresh FIT augmentation may use FIT nominal-good parents only")
        case_id = parent.get("caseId")
        if not isinstance(case_id, str) or case_id in seen_case_ids:
            raise FreshFitAugmentationError("fresh FIT augmentation parent caseId is invalid or duplicated")
        seen_case_ids.add(case_id)
        if not isinstance(parent.get("imagePath"), Path):
            raise FreshFitAugmentationError("fresh FIT augmentation parent was not phase-safely loaded")
        result.append(parent)
    return sorted(result, key=lambda parent: str(parent["caseId"]))


def generate_fresh_fit_augmentations(
    holdout_manifest_path: Path,
    source_root: Path,
    recipe_path: Path,
    output_dir: Path,
    *,
    variants_per_parent: int,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Materialize deterministic FIT-only camera variants in a fresh external directory."""

    variants = _require_positive_int(variants_per_parent, name="variantsPerParent", maximum=8)
    try:
        holdout, holdout_file_sha256, parents = load_evaluation_safe_normal_holdout_inputs(
            holdout_manifest_path,
            source_root=source_root,
            partitions={"FIT"},
        )
    except NormalHoldoutError as error:
        raise FreshFitAugmentationError(str(error)) from error
    if holdout.get("schemaVersion") != NORMAL_HOLDOUT_SCHEMA:
        raise FreshFitAugmentationError("fresh FIT augmentation requires a normal-holdout manifest")
    parents = _validate_fit_parents(parents)
    recipe, recipe_sha256 = load_fresh_fit_camera_recipe(recipe_path)
    # Fail before producing a package if the installed encoder cannot satisfy
    # the fixed coding profile claimed by the recipe.
    _expected_output_encoding()
    _prepare_external_output_directory(output_dir, repository_root=repository_root)
    records: list[dict[str, Any]] = []
    for parent in parents:
        source_image = _load_parent_rgb(parent)
        for variant_id in range(1, variants + 1):
            parameters = sample_fresh_fit_parameters(
                recipe,
                recipe_sha256=recipe_sha256,
                parent_case_id=str(parent["caseId"]),
                parent_source_sha256=str(parent["sourceSha256"]),
                variant_id=variant_id,
            )
            case_id, relative_path = _expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=variant_id)
            data = _render_augmented_jpeg(source_image, parameters)
            target_path = output_dir.joinpath(*relative_path.parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target_path.open("xb") as stream:
                    stream.write(data)
            except OSError as error:
                raise FreshFitAugmentationError("unable to write fresh FIT augmentation image") from error
            records.append({
                "caseId": case_id,
                "parentCaseId": parent["caseId"],
                "parentSourceSha256": parent["sourceSha256"],
                "sourceGroupId": parent["sourceGroupId"],
                "category": parent["category"],
                "parentPartition": "FIT",
                "kind": "NOMINAL",
                "defect": "good",
                "variantId": variant_id,
                "relativePath": relative_path.as_posix(),
                "sourceSha256": _sha256_bytes(data),
                "parameters": parameters,
                "outputEncoding": _expected_output_encoding(),
            })
    records.sort(key=lambda record: record["caseId"])
    document: dict[str, Any] = {
        "schemaVersion": FRESH_FIT_AUGMENTATION_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": FRESH_FIT_AUGMENTATION_PURPOSE,
        "inputPolicy": FRESH_FIT_INPUT_POLICY,
        "blindPolicy": FRESH_FIT_BLIND_POLICY,
        "parentPartition": "FIT",
        "holdoutManifestFileSha256": holdout_file_sha256,
        "holdoutManifestDeclaredSha256": holdout["normalHoldoutManifestSha256"],
        "developmentIdentitySha256": holdout["developmentIdentitySha256"],
        "fitParentIdentitySha256": _fit_parent_identity(parents),
        "recipeFileSha256": recipe_sha256,
        "recipe": recipe,
        "variantsPerParent": variants,
        "generation": _generation_provenance(repository_root=repository_root),
        "records": records,
    }
    document["augmentationManifestSha256"] = _document_digest(document, "augmentationManifestSha256")
    manifest_path = output_dir / "augmentation_manifest.json"
    try:
        with manifest_path.open("xb") as stream:
            stream.write((json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    except OSError as error:
        raise FreshFitAugmentationError("unable to write fresh FIT augmentation manifest") from error
    return document


def _validate_augmentation_document(
    document: dict[str, Any],
    *,
    manifest_file_sha256: str,
    holdout: dict[str, Any],
    holdout_file_sha256: str,
    parents: list[dict[str, Any]],
    recipe: dict[str, Any],
    recipe_sha256: str,
    repository_root: Path,
) -> tuple[int, list[dict[str, Any]]]:
    _require_exact_fields(document, name="fresh FIT augmentation manifest", required=AUGMENTATION_MANIFEST_FIELDS)
    if document.get("schemaVersion") != FRESH_FIT_AUGMENTATION_SCHEMA:
        raise FreshFitAugmentationError("fresh FIT augmentation manifest schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise FreshFitAugmentationError("fresh FIT augmentation manifest must be non-authoritative and non-production")
    if (
        document.get("purpose") != FRESH_FIT_AUGMENTATION_PURPOSE
        or document.get("inputPolicy") != FRESH_FIT_INPUT_POLICY
        or document.get("blindPolicy") != FRESH_FIT_BLIND_POLICY
        or document.get("parentPartition") != "FIT"
    ):
        raise FreshFitAugmentationError("fresh FIT augmentation manifest scope is unsafe")
    if document.get("augmentationManifestSha256") != _document_digest(document, "augmentationManifestSha256"):
        raise FreshFitAugmentationError("fresh FIT augmentation manifest digest does not match")
    bindings = {
        "holdoutManifestFileSha256": holdout_file_sha256,
        "holdoutManifestDeclaredSha256": holdout["normalHoldoutManifestSha256"],
        "developmentIdentitySha256": holdout["developmentIdentitySha256"],
        "fitParentIdentitySha256": _fit_parent_identity(parents),
        "recipeFileSha256": recipe_sha256,
    }
    for name, expected in bindings.items():
        if _require_sha256(document, name) != expected:
            raise FreshFitAugmentationError(f"fresh FIT augmentation manifest {name} does not match")
    if document.get("recipe") != recipe:
        raise FreshFitAugmentationError("fresh FIT augmentation embedded recipe does not match the supplied recipe")
    _validate_generation_provenance(document.get("generation"), repository_root=repository_root)
    variants = _require_positive_int(document.get("variantsPerParent"), name="variantsPerParent", maximum=8)
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise FreshFitAugmentationError("fresh FIT augmentation manifest has no records")
    if manifest_file_sha256 == "":  # Defensive invariant: callers must hash the same bytes they parsed.
        raise FreshFitAugmentationError("fresh FIT augmentation manifest file digest is missing")
    return variants, raw_records


def load_validated_fresh_fit_augmentations(
    augmentation_manifest_path: Path,
    holdout_manifest_path: Path,
    *,
    source_root: Path,
    recipe_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a fresh FIT-only package and re-render every declared child.

    The validator reopens FIT parent images only.  It never accepts a source
    pool, historical ledger, allocation plan, public inventory, tuning image,
    selection image, confirmation image, blind image, anomaly label, or mask.
    """

    try:
        holdout, holdout_file_sha256, parents = load_evaluation_safe_normal_holdout_inputs(
            holdout_manifest_path,
            source_root=source_root,
            partitions={"FIT"},
        )
    except NormalHoldoutError as error:
        raise FreshFitAugmentationError(str(error)) from error
    parents = _validate_fit_parents(parents)
    recipe, recipe_sha256 = load_fresh_fit_camera_recipe(recipe_path)
    _expected_output_encoding()
    document, manifest_file_sha256 = _read_external_json(
        augmentation_manifest_path,
        description="fresh FIT augmentation manifest",
        repository_root=repository_root,
    )
    variants, raw_records = _validate_augmentation_document(
        document,
        manifest_file_sha256=manifest_file_sha256,
        holdout=holdout,
        holdout_file_sha256=holdout_file_sha256,
        parents=parents,
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        repository_root=repository_root,
    )
    output_root = augmentation_manifest_path.parent
    _require_external_package_root(output_root, repository_root=repository_root)
    parent_by_case_id = {str(parent["caseId"]): parent for parent in parents}
    parent_images: dict[str, Image.Image] = {}
    expected_case_ids = {
        _expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=variant_id)[0]
        for parent in parents
        for variant_id in range(1, variants + 1)
    }
    seen_case_ids: set[str] = set()
    seen_relative_paths: set[str] = set()
    validated: list[dict[str, Any]] = []
    for value in raw_records:
        if not isinstance(value, dict):
            raise FreshFitAugmentationError("fresh FIT augmentation record must be an object")
        _require_exact_fields(value, name="fresh FIT augmentation record", required=AUGMENTATION_RECORD_FIELDS)
        case_id = _require_string(value, "caseId")
        if case_id in seen_case_ids:
            raise FreshFitAugmentationError("fresh FIT augmentation caseId is duplicated")
        seen_case_ids.add(case_id)
        parent_case_id = _require_string(value, "parentCaseId")
        parent = parent_by_case_id.get(parent_case_id)
        if parent is None:
            raise FreshFitAugmentationError("fresh FIT augmentation parent case is unknown")
        if value.get("parentSourceSha256") != parent["sourceSha256"]:
            raise FreshFitAugmentationError("fresh FIT augmentation parent source digest does not match")
        if value.get("sourceGroupId") != parent["sourceGroupId"] or value.get("category") != parent["category"]:
            raise FreshFitAugmentationError("fresh FIT augmentation parent group or category does not match")
        if value.get("parentPartition") != "FIT" or value.get("kind") != "NOMINAL" or value.get("defect") != "good":
            raise FreshFitAugmentationError("fresh FIT augmentation record scope is unsafe")
        variant_id = _require_positive_int(value.get("variantId"), name="variantId", maximum=variants)
        expected_case_id, expected_relative_path = _expected_child_identity(
            parent,
            recipe_sha256=recipe_sha256,
            variant_id=variant_id,
        )
        if case_id != expected_case_id:
            raise FreshFitAugmentationError("fresh FIT augmentation caseId does not match parent and variant")
        relative_path = _safe_relative_path(value.get("relativePath"), name="fresh FIT augmentation relativePath")
        if relative_path != expected_relative_path:
            raise FreshFitAugmentationError("fresh FIT augmentation relativePath does not match parent and variant")
        if relative_path.as_posix() in seen_relative_paths:
            raise FreshFitAugmentationError("fresh FIT augmentation relativePath is duplicated")
        seen_relative_paths.add(relative_path.as_posix())
        expected_parameters = sample_fresh_fit_parameters(
            recipe,
            recipe_sha256=recipe_sha256,
            parent_case_id=parent_case_id,
            parent_source_sha256=str(parent["sourceSha256"]),
            variant_id=variant_id,
        )
        parameters = _validate_parameters(value.get("parameters"))
        if parameters != expected_parameters:
            raise FreshFitAugmentationError("fresh FIT augmentation parameters do not match recipe and parent")
        _validate_output_encoding(value.get("outputEncoding"))
        output_path = _safe_file_under(
            output_root,
            relative_path,
            description="fresh FIT augmentation output image",
            repository_root=repository_root,
        )
        try:
            actual_bytes = output_path.read_bytes()
        except OSError as error:
            raise FreshFitAugmentationError("unable to read fresh FIT augmentation output image") from error
        if _sha256_bytes(actual_bytes) != _require_sha256(value, "sourceSha256"):
            raise FreshFitAugmentationError("fresh FIT augmentation output digest does not match")
        if _inspect_jpeg_output_encoding(actual_bytes) != _expected_output_encoding():
            raise FreshFitAugmentationError("fresh FIT augmentation JPEG headers do not match the fixed recipe")
        source_image = parent_images.get(parent_case_id)
        if source_image is None:
            source_image = _load_parent_rgb(parent)
            parent_images[parent_case_id] = source_image
        expected_bytes = _render_augmented_jpeg(source_image, expected_parameters)
        if actual_bytes != expected_bytes:
            raise FreshFitAugmentationError("fresh FIT augmentation pixels do not match the deterministic renderer")
        validated.append(dict(value))
    if [record["caseId"] for record in raw_records] != sorted(record["caseId"] for record in raw_records):
        raise FreshFitAugmentationError("fresh FIT augmentation records must be sorted by caseId")
    if seen_case_ids != expected_case_ids:
        raise FreshFitAugmentationError("fresh FIT augmentation records do not cover every FIT parent and variant")
    return document, validated
