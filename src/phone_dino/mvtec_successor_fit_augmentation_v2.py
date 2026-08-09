"""V2 successor FIT-only generic camera augmentation for offline MVTec research.

The parent V1 cohort is closed.  This module deliberately has its own schema,
recipe, renderer, and package validator so it cannot silently alter or reuse
the frozen V1 augmentation contract.  It receives parent bytes exclusively
through :func:`mvtec_normal_successor.load_successor_safe_normal_inputs` with
the ``FIT`` partition, and it is an exploratory normal-robustness aid only.
It is not device calibration, a defect result, or production authorization.
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
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image, ImageOps, JpegImagePlugin, __version__ as PILLOW_VERSION

from phone_dino import mvtec_normal_successor as successor


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SUCCESSOR_FIT_AUGMENTATION_V2_SCHEMA = "phone-dino.mvtec-ad-reserve-successor-fit-augmentation/2.0"
SUCCESSOR_FIT_CAMERA_RECIPE_V2_SCHEMA = "phone-dino.mvtec-ad-reserve-successor-fit-camera-recipe/2.0"
SUCCESSOR_FIT_AUGMENTATION_V2_PURPOSE = "OFFLINE_MVTEC_RESERVE_SUCCESSOR_V2_FIT_AUGMENTATION_ONLY"
SUCCESSOR_FIT_RECIPE_V2_ID = "RESERVE_SUCCESSOR_V2_GENERIC_FIT_CAMERA_R3_420_Q95"
SUCCESSOR_FIT_RECIPE_V2_DESCRIPTION = (
    "Closed V2 successor FIT-only component-separated generic camera prior: R3 registration, illumination, "
    "and sensor_transport variants. It is generic and non-device-calibrated; it excludes crops, flips, glare, "
    "synthetic defects, masks, and capture rejection, and uses fixed RGB JPEG 4:2:0 Q95. Q95 is retained only "
    "from the available engineering still coding profile; it does not establish deployed-camera calibration or "
    "production authorization."
)
SUCCESSOR_FIT_INPUT_POLICY = "SUCCESSOR_FIT_RAW_NORMAL_PARENTS_ONLY"
SUCCESSOR_FIT_BLIND_POLICY = "NO_BLIND_OR_ANOMALY_DATA"
SUCCESSOR_FIT_DELEGATION_POLICY = "TOOL_MEDIATED_UNCONSUMED_ONLY"
SUCCESSOR_FIT_RESULT_LABEL = "EXPLORATORY_NOT_INDEPENDENT"
SUCCESSOR_FIT_INDEPENDENCE_LABEL = "NOT_INDEPENDENT_PARENT_RESERVE_DERIVATION"
SUCCESSOR_FIT_SAMPLING_ALGORITHM = "NAMED_SHA256_COMPONENT_SUBSTREAMS_V2"
SUCCESSOR_FIT_CALIBRATION_EVIDENCE = {"state": "GENERIC_PRIOR_NOT_DEVICE_CALIBRATED"}
SUCCESSOR_FIT_VARIANTS = (
    {"variantId": 1, "component": "registration"},
    {"variantId": 2, "component": "illumination"},
    {"variantId": 3, "component": "sensor_transport"},
)
SUCCESSOR_FIT_VARIANTS_PER_PARENT = len(SUCCESSOR_FIT_VARIANTS)

# The V2 profile is intentionally component-separated.  It never combines
# registration, illumination, and transport effects into one derivative.
SUCCESSOR_FIT_REGISTRATION_PROFILE = {
    "maxCornerJitterFraction": 0.002,
    "maxGaussianBlurSigmaPixels": 0.20,
    "maxRotationDegrees": 0.35,
    "maxScaleDelta": 0.005,
    "maxTranslationFraction": 0.002,
}
SUCCESSOR_FIT_ILLUMINATION_PROFILE = {
    "maxDirectionalShadingStrength": 0.04,
    "maxExposureEv": 0.10,
    "maxGammaDelta": 0.025,
    "maxLensCenterOffsetFraction": 0.12,
    "maxLensShadingStrength": 0.025,
    "maxVignetteStrength": 0.025,
    "maxWhiteBalanceDelta": 0.025,
}
SUCCESSOR_FIT_SENSOR_TRANSPORT_PROFILE = {
    "maxGaussianBlurSigmaPixels": 0.20,
    "maxReadNoiseStdDn": 0.35,
    "maxShotNoiseStdDnAtFullScale": 0.35,
    "minDownUpSamplingScale": 0.99,
}
SUCCESSOR_FIT_JPEG_Q95_TABLES_SHA256 = "sha256:f67e35fd0dcd2fd9f999077e2aae8560e6327a8477c45427f6ea2e0a224cd187"
SUCCESSOR_FIT_OUTPUT_ENCODING = {
    "format": "JPEG",
    "jpegQuality": 95,
    "jpegQuantizationTablesSha256": SUCCESSOR_FIT_JPEG_Q95_TABLES_SHA256,
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
    "componentProfiles",
    "outputEncoding",
}
COMPONENT_PROFILE_FIELDS = {"registration", "illumination", "sensor_transport"}
MANIFEST_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "purpose",
    "inputPolicy",
    "blindPolicy",
    "delegationPolicy",
    "resultLabel",
    "independenceLabel",
    "parentPartition",
    "parentHoldoutManifestFileSha256",
    "parentHoldoutManifestDeclaredSha256",
    "parentSelectionContractFileSha256",
    "parentSelectionContractDeclaredSha256",
    "successorSealFileSha256",
    "successorSealDeclaredSha256",
    "successorPlanFileSha256",
    "successorPlanDeclaredSha256",
    "successorEnvelopeFileSha256",
    "successorEnvelopeDeclaredSha256",
    "successorFitIdentitySha256",
    "parentNormalConfirmationIdentitySha256",
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
    "normalSuccessorModuleSha256",
    "gitRevision",
    "gitWorktreeClean",
    "python",
    "platform",
    "pillowVersion",
    "opencvVersion",
    "numpyVersion",
}
RECORD_FIELDS = {
    "caseId",
    "parentCaseId",
    "parentSourceSha256",
    "sourceGroupId",
    "category",
    "parentPartition",
    "kind",
    "defect",
    "variantId",
    "component",
    "relativePath",
    "sourceSha256",
    "parameters",
    "outputEncoding",
}
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


class SuccessorFitAugmentationV2Error(ValueError):
    """Raised when a V2 successor FIT augmentation artifact is unsafe."""


def sha256_file(path: Path) -> str:
    """Return a ``sha256:`` digest while avoiding unbounded reads."""

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
    return successor.canonical_json_sha256(unsigned)


def _require_exact_fields(document: object, *, name: str, required: set[str]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise SuccessorFitAugmentationV2Error(f"{name} must be an object")
    missing = required.difference(document)
    unknown = set(document).difference(required)
    if missing:
        raise SuccessorFitAugmentationV2Error(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise SuccessorFitAugmentationV2Error(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")
    return document


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SuccessorFitAugmentationV2Error(f"{name} must be an object")
    return value


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SuccessorFitAugmentationV2Error(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    result = _require_string(value, name=name)
    if len(result) != 71 or not result.startswith("sha256:"):
        raise SuccessorFitAugmentationV2Error(f"{name} must be a SHA-256 digest")
    try:
        int(result[7:], 16)
    except ValueError as error:
        raise SuccessorFitAugmentationV2Error(f"{name} must be a SHA-256 digest") from error
    return result


def _require_finite_number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SuccessorFitAugmentationV2Error(f"{name} must be a finite number")
    return float(value)


def _require_positive_int(value: object, *, name: str, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SuccessorFitAugmentationV2Error(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise SuccessorFitAugmentationV2Error(f"{name} must be at most {maximum}")
    return value


def _expect_exact_mapping(value: object, *, name: str, expected: dict[str, Any]) -> dict[str, Any]:
    mapping = _require_exact_fields(value, name=name, required=set(expected))
    if mapping != expected:
        raise SuccessorFitAugmentationV2Error(f"{name} does not match the closed V2 recipe")
    return mapping


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
                raise SuccessorFitAugmentationV2Error(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _require_external_input_file(path: Path, *, description: str, repository_root: Path) -> None:
    # Inspect raw path components before any containment test calls resolve().
    # A manifest beside an ancestor of the repository is also unsafe: that
    # artifact root would contain the working tree rather than be external to
    # it.
    _reject_links_on_existing_path(path, description=description)
    external_root = path if path.is_dir() else path.parent
    if _is_under(repository_root, path) or _is_under(external_root, repository_root):
        raise SuccessorFitAugmentationV2Error(f"{description} must stay outside the Git working tree")
    if not path.is_file():
        raise SuccessorFitAugmentationV2Error(f"{description} is missing")


def _safe_relative_path(value: object, *, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or "\\" in value:
        raise SuccessorFitAugmentationV2Error(f"{name} must be a non-empty safe relative path")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute() or windows.drive or windows.root or posix.is_absolute() or not posix.parts:
        raise SuccessorFitAugmentationV2Error(f"{name} must be a safe relative path")
    if any(
        part in {"", ".", ".."}
        or ":" in part
        or part != part.rstrip(" .")
        or PureWindowsPath(part).is_reserved()
        for part in posix.parts
    ):
        raise SuccessorFitAugmentationV2Error(f"{name} must be a safe relative path")
    return posix


def _safe_file_under(root: Path, relative: PurePosixPath, *, description: str, repository_root: Path) -> Path:
    _reject_links_on_existing_path(root, description=description)
    candidate = root.joinpath(*relative.parts)
    _reject_links_on_existing_path(candidate, description=description)
    if not candidate.is_file() or not _is_under(root, candidate) or _is_under(repository_root, candidate):
        raise SuccessorFitAugmentationV2Error(f"{description} is missing or escapes its external root")
    return candidate


def _parse_json_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SuccessorFitAugmentationV2Error(f"{description} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise SuccessorFitAugmentationV2Error(f"{description} contains a non-finite JSON value: {value}")

    try:
        parsed = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SuccessorFitAugmentationV2Error(f"unable to read {description}") from error
    if not isinstance(parsed, dict):
        raise SuccessorFitAugmentationV2Error(f"{description} must be a JSON object")
    return parsed


def _read_external_json(
    path: Path,
    *,
    description: str,
    repository_root: Path,
) -> tuple[dict[str, Any], str]:
    _require_external_input_file(path, description=description, repository_root=repository_root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SuccessorFitAugmentationV2Error(f"unable to read {description}") from error
    return _parse_json_bytes(raw, description=description), _sha256_bytes(raw)


def _read_recipe_json(path: Path) -> tuple[dict[str, Any], str]:
    _reject_links_on_existing_path(path, description="successor FIT V2 camera recipe")
    if not path.is_file():
        raise SuccessorFitAugmentationV2Error("successor FIT V2 camera recipe is missing")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SuccessorFitAugmentationV2Error("unable to read successor FIT V2 camera recipe") from error
    return _parse_json_bytes(raw, description="successor FIT V2 camera recipe"), _sha256_bytes(raw)


def load_successor_fit_camera_recipe_v2(recipe_path: Path) -> tuple[dict[str, Any], str]:
    """Load the one closed generic camera prior allowed by successor V2."""

    recipe, file_sha256 = _read_recipe_json(recipe_path)
    _require_exact_fields(recipe, name="successor FIT V2 camera recipe", required=RECIPE_FIELDS)
    if recipe.get("schemaVersion") != SUCCESSOR_FIT_CAMERA_RECIPE_V2_SCHEMA or recipe.get("id") != SUCCESSOR_FIT_RECIPE_V2_ID:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 camera recipe schema or id is unsupported")
    if recipe.get("description") != SUCCESSOR_FIT_RECIPE_V2_DESCRIPTION:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 camera recipe description is not locked")
    if recipe.get("authoritative") is not False or recipe.get("productionAuthorized") is not False:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 camera recipe must be non-authoritative and non-production")
    if recipe.get("purpose") != SUCCESSOR_FIT_AUGMENTATION_V2_PURPOSE:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 camera recipe purpose is unsafe")
    _expect_exact_mapping(
        recipe.get("calibrationEvidence"),
        name="successor FIT V2 camera recipe calibrationEvidence",
        expected=SUCCESSOR_FIT_CALIBRATION_EVIDENCE,
    )
    if recipe.get("samplingAlgorithm") != SUCCESSOR_FIT_SAMPLING_ALGORITHM:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 camera recipe sampling algorithm is unsupported")
    profiles = _require_exact_fields(
        recipe.get("componentProfiles"),
        name="successor FIT V2 camera recipe componentProfiles",
        required=COMPONENT_PROFILE_FIELDS,
    )
    _expect_exact_mapping(
        profiles.get("registration"),
        name="successor FIT V2 registration profile",
        expected=SUCCESSOR_FIT_REGISTRATION_PROFILE,
    )
    _expect_exact_mapping(
        profiles.get("illumination"),
        name="successor FIT V2 illumination profile",
        expected=SUCCESSOR_FIT_ILLUMINATION_PROFILE,
    )
    _expect_exact_mapping(
        profiles.get("sensor_transport"),
        name="successor FIT V2 sensor transport profile",
        expected=SUCCESSOR_FIT_SENSOR_TRANSPORT_PROFILE,
    )
    _expect_exact_mapping(
        recipe.get("outputEncoding"),
        name="successor FIT V2 camera recipe outputEncoding",
        expected=SUCCESSOR_FIT_OUTPUT_ENCODING,
    )
    return recipe, file_sha256


def _component_for_variant(variant_id: int) -> str:
    for item in SUCCESSOR_FIT_VARIANTS:
        if item["variantId"] == variant_id:
            return str(item["component"])
    raise SuccessorFitAugmentationV2Error("successor FIT V2 variantId is unsupported")


def _derive_component_seed(
    recipe_sha256: str,
    parent_case_id: str,
    parent_source_sha256: str,
    variant_id: int,
    component: str,
) -> int:
    if component != _component_for_variant(variant_id):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 component does not match variantId")
    payload = "\0".join((
        SUCCESSOR_FIT_SAMPLING_ALGORITHM,
        recipe_sha256,
        parent_case_id,
        parent_source_sha256,
        str(variant_id),
        component,
    )).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _rounded(value: float) -> float:
    return round(value, 8)


def _symmetric(rng: random.Random, maximum: float) -> float:
    return rng.uniform(-maximum, maximum)


def sample_successor_fit_parameters_v2(
    recipe: dict[str, Any],
    *,
    recipe_sha256: str,
    parent_case_id: str,
    parent_source_sha256: str,
    variant_id: int,
) -> dict[str, Any]:
    """Derive one deterministic, component-isolated V2 transform."""

    if recipe.get("samplingAlgorithm") != SUCCESSOR_FIT_SAMPLING_ALGORITHM:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 recipe sampling algorithm is unsupported")
    component = _component_for_variant(variant_id)
    seed = _derive_component_seed(
        recipe_sha256,
        parent_case_id,
        parent_source_sha256,
        variant_id,
        component,
    )
    rng = random.Random(seed)
    common = {"component": component, "namedSubstream": component, "seed": str(seed)}
    if component == "registration":
        parameters = {
            **common,
            "registration": {
                "rotationDegrees": _rounded(_symmetric(rng, SUCCESSOR_FIT_REGISTRATION_PROFILE["maxRotationDegrees"])),
                "scale": _rounded(1.0 + _symmetric(rng, SUCCESSOR_FIT_REGISTRATION_PROFILE["maxScaleDelta"])),
                "translationFractionX": _rounded(
                    _symmetric(rng, SUCCESSOR_FIT_REGISTRATION_PROFILE["maxTranslationFraction"])
                ),
                "translationFractionY": _rounded(
                    _symmetric(rng, SUCCESSOR_FIT_REGISTRATION_PROFILE["maxTranslationFraction"])
                ),
                "cornerJitterFractions": [
                    _rounded(_symmetric(rng, SUCCESSOR_FIT_REGISTRATION_PROFILE["maxCornerJitterFraction"]))
                    for _ in range(8)
                ],
                "gaussianBlurSigmaPixels": _rounded(
                    rng.uniform(0.0, SUCCESSOR_FIT_REGISTRATION_PROFILE["maxGaussianBlurSigmaPixels"])
                ),
            },
        }
    elif component == "illumination":
        parameters = {
            **common,
            "illumination": {
                "exposureEv": _rounded(_symmetric(rng, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxExposureEv"])),
                "gamma": _rounded(1.0 + _symmetric(rng, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxGammaDelta"])),
                "redGain": _rounded(
                    1.0 + _symmetric(rng, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxWhiteBalanceDelta"])
                ),
                "blueGain": _rounded(
                    1.0 + _symmetric(rng, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxWhiteBalanceDelta"])
                ),
                "directionalShadingStrength": _rounded(
                    rng.uniform(0.0, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxDirectionalShadingStrength"])
                ),
                "directionalShadingAngleDegrees": _rounded(rng.uniform(0.0, 360.0)),
                "vignetteStrength": _rounded(rng.uniform(0.0, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxVignetteStrength"])),
                "lensShadingStrength": _rounded(
                    rng.uniform(0.0, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxLensShadingStrength"])
                ),
                "lensCenterOffsetXFraction": _rounded(
                    _symmetric(rng, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxLensCenterOffsetFraction"])
                ),
                "lensCenterOffsetYFraction": _rounded(
                    _symmetric(rng, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxLensCenterOffsetFraction"])
                ),
            },
        }
    else:
        parameters = {
            **common,
            "sensor_transport": {
                "readNoiseStdDn": _rounded(rng.uniform(0.0, SUCCESSOR_FIT_SENSOR_TRANSPORT_PROFILE["maxReadNoiseStdDn"])),
                "shotNoiseStdDnAtFullScale": _rounded(
                    rng.uniform(0.0, SUCCESSOR_FIT_SENSOR_TRANSPORT_PROFILE["maxShotNoiseStdDnAtFullScale"])
                ),
                "downUpSamplingScale": _rounded(
                    rng.uniform(SUCCESSOR_FIT_SENSOR_TRANSPORT_PROFILE["minDownUpSamplingScale"], 1.0)
                ),
                "gaussianBlurSigmaPixels": _rounded(
                    rng.uniform(0.0, SUCCESSOR_FIT_SENSOR_TRANSPORT_PROFILE["maxGaussianBlurSigmaPixels"])
                ),
            },
        }
    _validate_parameters(parameters, variant_id=variant_id)
    return parameters


def _validate_bounds(
    mapping: dict[str, Any],
    *,
    name: str,
    fields: set[str],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    _require_exact_fields(mapping, name=name, required=fields)
    for field, (minimum, maximum) in bounds.items():
        value = _require_finite_number(mapping.get(field), name=f"{name}.{field}")
        if value < minimum or value > maximum:
            raise SuccessorFitAugmentationV2Error(f"{name}.{field} is outside the approved V2 range")
    return mapping


def _validate_parameters(value: object, *, variant_id: int) -> dict[str, Any]:
    parameters = _require_mapping(value, name="successor FIT V2 augmentation parameters")
    component = _component_for_variant(variant_id)
    expected_fields = {"component", "namedSubstream", "seed", component}
    _require_exact_fields(parameters, name="successor FIT V2 augmentation parameters", required=expected_fields)
    if parameters.get("component") != component or parameters.get("namedSubstream") != component:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 parameters do not use the expected component substream")
    seed = _require_string(parameters.get("seed"), name="successor FIT V2 augmentation seed")
    if not seed.isdecimal() or int(seed) < 0 or int(seed) > 2**64 - 1:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation seed is invalid")
    if component == "registration":
        section = _validate_bounds(
            _require_mapping(parameters.get(component), name="successor FIT V2 registration parameters"),
            name="successor FIT V2 registration parameters",
            fields={
                "rotationDegrees",
                "scale",
                "translationFractionX",
                "translationFractionY",
                "cornerJitterFractions",
                "gaussianBlurSigmaPixels",
            },
            bounds={
                "rotationDegrees": (-SUCCESSOR_FIT_REGISTRATION_PROFILE["maxRotationDegrees"], SUCCESSOR_FIT_REGISTRATION_PROFILE["maxRotationDegrees"]),
                "scale": (1.0 - SUCCESSOR_FIT_REGISTRATION_PROFILE["maxScaleDelta"], 1.0 + SUCCESSOR_FIT_REGISTRATION_PROFILE["maxScaleDelta"]),
                "translationFractionX": (-SUCCESSOR_FIT_REGISTRATION_PROFILE["maxTranslationFraction"], SUCCESSOR_FIT_REGISTRATION_PROFILE["maxTranslationFraction"]),
                "translationFractionY": (-SUCCESSOR_FIT_REGISTRATION_PROFILE["maxTranslationFraction"], SUCCESSOR_FIT_REGISTRATION_PROFILE["maxTranslationFraction"]),
                "gaussianBlurSigmaPixels": (0.0, SUCCESSOR_FIT_REGISTRATION_PROFILE["maxGaussianBlurSigmaPixels"]),
            },
        )
        jitter = section.get("cornerJitterFractions")
        if not isinstance(jitter, list) or len(jitter) != 8:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 cornerJitterFractions must contain eight values")
        if any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or abs(float(item)) > SUCCESSOR_FIT_REGISTRATION_PROFILE["maxCornerJitterFraction"]
            for item in jitter
        ):
            raise SuccessorFitAugmentationV2Error("successor FIT V2 cornerJitterFractions are outside the approved range")
    elif component == "illumination":
        _validate_bounds(
            _require_mapping(parameters.get(component), name="successor FIT V2 illumination parameters"),
            name="successor FIT V2 illumination parameters",
            fields={
                "exposureEv",
                "gamma",
                "redGain",
                "blueGain",
                "directionalShadingStrength",
                "directionalShadingAngleDegrees",
                "vignetteStrength",
                "lensShadingStrength",
                "lensCenterOffsetXFraction",
                "lensCenterOffsetYFraction",
            },
            bounds={
                "exposureEv": (-SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxExposureEv"], SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxExposureEv"]),
                "gamma": (1.0 - SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxGammaDelta"], 1.0 + SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxGammaDelta"]),
                "redGain": (1.0 - SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxWhiteBalanceDelta"], 1.0 + SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxWhiteBalanceDelta"]),
                "blueGain": (1.0 - SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxWhiteBalanceDelta"], 1.0 + SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxWhiteBalanceDelta"]),
                "directionalShadingStrength": (0.0, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxDirectionalShadingStrength"]),
                "directionalShadingAngleDegrees": (0.0, 360.0),
                "vignetteStrength": (0.0, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxVignetteStrength"]),
                "lensShadingStrength": (0.0, SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxLensShadingStrength"]),
                "lensCenterOffsetXFraction": (-SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxLensCenterOffsetFraction"], SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxLensCenterOffsetFraction"]),
                "lensCenterOffsetYFraction": (-SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxLensCenterOffsetFraction"], SUCCESSOR_FIT_ILLUMINATION_PROFILE["maxLensCenterOffsetFraction"]),
            },
        )
    else:
        _validate_bounds(
            _require_mapping(parameters.get(component), name="successor FIT V2 sensor transport parameters"),
            name="successor FIT V2 sensor transport parameters",
            fields={"readNoiseStdDn", "shotNoiseStdDnAtFullScale", "downUpSamplingScale", "gaussianBlurSigmaPixels"},
            bounds={
                "readNoiseStdDn": (0.0, SUCCESSOR_FIT_SENSOR_TRANSPORT_PROFILE["maxReadNoiseStdDn"]),
                "shotNoiseStdDnAtFullScale": (0.0, SUCCESSOR_FIT_SENSOR_TRANSPORT_PROFILE["maxShotNoiseStdDnAtFullScale"]),
                "downUpSamplingScale": (SUCCESSOR_FIT_SENSOR_TRANSPORT_PROFILE["minDownUpSamplingScale"], 1.0),
                "gaussianBlurSigmaPixels": (0.0, SUCCESSOR_FIT_SENSOR_TRANSPORT_PROFILE["maxGaussianBlurSigmaPixels"]),
            },
        )
    return parameters


def _gaussian_blur(array: Any, sigma: float, cv2: Any) -> Any:
    if sigma <= 0.0:
        return array
    # A zero kernel tells OpenCV to select the Gaussian-support size from the
    # bounded sigma.  This does not crop or alter the image geometry.
    return cv2.GaussianBlur(array, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def apply_successor_fit_camera_augmentation_v2(image: Image.Image, parameters: dict[str, Any], *, variant_id: int) -> Image.Image:
    """Render one isolated generic V2 component without a crop, flip, or defect."""

    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("successor FIT V2 augmentation requires the optional vision dependencies") from error
    parsed = _validate_parameters(parameters, variant_id=variant_id)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    if height < 2 or width < 2:
        raise SuccessorFitAugmentationV2Error("image is too small for successor FIT V2 augmentation")
    component = str(parsed["component"])
    if component == "registration":
        values = parsed["registration"]
        source = np.asarray(
            [[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]],
            dtype=np.float32,
        )
        angle = math.radians(float(values["rotationDegrees"]))
        scale = float(values["scale"])
        center = np.asarray([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
        rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
        destination = ((source - center) @ rotation.T * scale) + center
        destination[:, 0] += float(values["translationFractionX"]) * width
        destination[:, 1] += float(values["translationFractionY"]) * height
        destination += np.asarray(values["cornerJitterFractions"], dtype=np.float32).reshape(4, 2) * min(width, height)
        homography = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
        result = cv2.warpPerspective(
            rgb,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        result = _gaussian_blur(result, float(values["gaussianBlurSigmaPixels"]), cv2)
    elif component == "illumination":
        values = parsed["illumination"]
        result_values = rgb.astype(np.float32) / 255.0
        result_values *= 2.0 ** float(values["exposureEv"])
        result_values[..., 0] *= float(values["redGain"])
        result_values[..., 2] *= float(values["blueGain"])
        result_values = np.power(np.clip(result_values, 0.0, 1.0), float(values["gamma"]))
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        x = (x / max(width - 1, 1)) * 2.0 - 1.0
        y = (y / max(height - 1, 1)) * 2.0 - 1.0
        angle = math.radians(float(values["directionalShadingAngleDegrees"]))
        directional = x * math.cos(angle) + y * math.sin(angle)
        directional_shading = 1.0 - float(values["directionalShadingStrength"]) * (directional + 1.0) / 2.0
        vignette = 1.0 - float(values["vignetteStrength"]) * np.minimum(1.0, x * x + y * y)
        lens_radius_squared = np.minimum(
            1.0,
            (x - 2.0 * float(values["lensCenterOffsetXFraction"])) ** 2
            + (y - 2.0 * float(values["lensCenterOffsetYFraction"])) ** 2,
        )
        lens = 1.0 - float(values["lensShadingStrength"]) * lens_radius_squared
        result = np.rint(np.clip(result_values * (directional_shading * vignette * lens)[..., None], 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        values = parsed["sensor_transport"]
        scale = float(values["downUpSamplingScale"])
        down_width = max(1, int(round(width * scale)))
        down_height = max(1, int(round(height * scale)))
        if down_width != width or down_height != height:
            reduced = cv2.resize(rgb, (down_width, down_height), interpolation=cv2.INTER_AREA)
            transported = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            transported = rgb.copy()
        image_values = transported.astype(np.float32) / 255.0
        noise_rng = np.random.default_rng(int(parsed["seed"]))
        standard_deviation_dn = np.sqrt(
            float(values["readNoiseStdDn"]) ** 2
            + np.clip(image_values, 0.0, 1.0) * float(values["shotNoiseStdDnAtFullScale"]) ** 2
        )
        image_values += noise_rng.normal(0.0, 1.0, image_values.shape).astype(np.float32) * (standard_deviation_dn / 255.0)
        result = np.rint(np.clip(image_values, 0.0, 1.0) * 255.0).astype(np.uint8)
        result = _gaussian_blur(result, float(values["gaussianBlurSigmaPixels"]), cv2)
    return Image.fromarray(result.astype(np.uint8))


def _quantization_tables_sha256(opened: Image.Image) -> str:
    quantization = getattr(opened, "quantization", None)
    if not isinstance(quantization, dict) or set(quantization) != {0, 1}:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 JPEG quantization tables are invalid")
    normalized: dict[str, list[int]] = {}
    for table_id in (0, 1):
        values = quantization.get(table_id)
        if not isinstance(values, (list, tuple)) or len(values) != 64:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 JPEG quantization table length is invalid")
        table = list(values)
        if any(not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 255 for value in table):
            raise SuccessorFitAugmentationV2Error("successor FIT V2 JPEG quantization table values are invalid")
        normalized[str(table_id)] = table
    return successor.canonical_json_sha256(normalized)


def _jpeg_output_encoding_from_opened_image(opened: Image.Image) -> dict[str, Any]:
    if opened.format != "JPEG" or opened.mode != "RGB":
        raise SuccessorFitAugmentationV2Error("successor FIT V2 output must be an RGB JPEG")
    layer = getattr(opened, "layer", None)
    sampling = JpegImagePlugin.get_sampling(opened)
    if not isinstance(layer, list) or len(layer) != 3:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 JPEG must have exactly three components")
    component_ids: list[int] = []
    sampling_factors: list[list[int]] = []
    selectors: list[int] = []
    for component in layer:
        if not isinstance(component, tuple) or len(component) != 4:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 JPEG component metadata is invalid")
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
            raise SuccessorFitAugmentationV2Error("successor FIT V2 JPEG component metadata is invalid")
        component_ids.append(component_id)
        sampling_factors.append([horizontal, vertical])
        selectors.append(selector)
    sampling_names = {0: "4:4:4", 2: "4:2:0"}
    if sampling not in sampling_names:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 JPEG subsampling is unsupported")
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
    except SuccessorFitAugmentationV2Error:
        raise
    except (OSError, SyntaxError, ValueError) as error:
        raise SuccessorFitAugmentationV2Error("unable to decode successor FIT V2 JPEG output") from error


@lru_cache(maxsize=1)
def _expected_output_encoding() -> dict[str, Any]:
    """Attest the installed encoder's exact JPEG Q95 4:2:0 profile."""

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
        raise SuccessorFitAugmentationV2Error("installed Pillow does not emit the required successor FIT V2 JPEG profile")
    if expected["quantizationTablesSha256"] != SUCCESSOR_FIT_JPEG_Q95_TABLES_SHA256:
        raise SuccessorFitAugmentationV2Error("installed Pillow does not emit the locked successor FIT V2 Q95 tables")
    return expected


def _validate_output_encoding(value: object) -> dict[str, Any]:
    encoding = _require_exact_fields(
        value,
        name="successor FIT V2 augmentation outputEncoding",
        required=OUTPUT_ENCODING_FIELDS,
    )
    expected = _expected_output_encoding()
    if encoding != expected:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 outputEncoding does not match the fixed recipe")
    return encoding


def _render_augmented_jpeg(image: Image.Image, parameters: dict[str, Any], *, variant_id: int) -> bytes:
    augmented = apply_successor_fit_camera_augmentation_v2(image, parameters, variant_id=variant_id)
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
        raise SuccessorFitAugmentationV2Error("successor FIT V2 JPEG headers do not match the fixed recipe")
    return data


def _load_parent_rgb(parent: dict[str, Any]) -> Image.Image:
    source_path = parent.get("imagePath")
    if not isinstance(source_path, Path):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 parent has no validated image path")
    if sha256_file(source_path) != parent.get("sourceSha256"):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 parent changed before augmentation")
    try:
        with Image.open(source_path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, SyntaxError, ValueError) as error:
        raise SuccessorFitAugmentationV2Error("unable to decode successor FIT V2 parent image") from error
    if sha256_file(source_path) != parent.get("sourceSha256"):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 parent changed while it was decoded")
    return image


def _fit_parent_identity(parents: list[dict[str, Any]]) -> str:
    return successor.canonical_json_sha256([
        {
            "caseId": parent["caseId"],
            "category": parent["category"],
            "sourceSha256": parent["sourceSha256"],
            "sourceGroupId": parent["sourceGroupId"],
            "partition": parent["partition"],
        }
        for parent in sorted(parents, key=lambda item: str(item["caseId"]))
    ])


def _expected_child_identity(
    parent: dict[str, Any],
    *,
    recipe_sha256: str,
    variant_id: int,
) -> tuple[str, PurePosixPath]:
    component = _component_for_variant(variant_id)
    case_id = f"{parent['caseId']}/reserve-successor-fit-camera-v2/{component}"
    name = hashlib.sha256(
        "\0".join((
            SUCCESSOR_FIT_AUGMENTATION_V2_SCHEMA,
            recipe_sha256,
            str(parent["caseId"]),
            str(parent["sourceSha256"]),
            str(variant_id),
            component,
        )).encode("utf-8")
    ).hexdigest()
    return case_id, PurePosixPath("images") / f"{name}-{component}.jpg"


def _prepare_external_output_directory(output_dir: Path, *, repository_root: Path) -> None:
    # Reparse checks intentionally precede .resolve()-based containment.
    _reject_links_on_existing_path(output_dir, description="successor FIT V2 augmentation output")
    if _is_under(repository_root, output_dir) or _is_under(output_dir, repository_root):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation output must stay outside the Git working tree")
    if output_dir.exists() or output_dir.is_symlink():
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation output already exists; choose a new immutable path")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise SuccessorFitAugmentationV2Error("unable to create successor FIT V2 augmentation output directory") from error
    _reject_links_on_existing_path(output_dir, description="successor FIT V2 augmentation output")


def _require_external_package_root(root: Path, *, repository_root: Path) -> None:
    # Reparse checks intentionally precede .resolve()-based containment.
    _reject_links_on_existing_path(root, description="successor FIT V2 augmentation package")
    if _is_under(repository_root, root) or _is_under(root, repository_root):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation package must stay outside the Git working tree")
    if not root.is_dir():
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation package root is missing")


def _require_external_source_root(root: Path, *, repository_root: Path) -> None:
    """Reject a source directory that is in, or encloses, the Git tree."""

    _reject_links_on_existing_path(root, description="successor FIT V2 source root")
    if _is_under(repository_root, root) or _is_under(root, repository_root):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 source root must stay outside the Git working tree")
    if not root.is_dir():
        raise SuccessorFitAugmentationV2Error("successor FIT V2 source root is missing")


def _generation_provenance(*, repository_root: Path) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - generation itself needs optional vision dependencies
        raise RuntimeError("successor FIT V2 augmentation requires the optional vision dependencies") from error
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
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - Git availability is host-specific
        revision = None
        worktree_clean = None
    return {
        "generatorModuleSha256": sha256_file(Path(__file__)),
        "generatorEntrypointSha256": sha256_file(
            repository_root / "tools" / "generate_mvtec_ad_successor_fit_augmentations.py"
        ),
        "normalSuccessorModuleSha256": sha256_file(repository_root / "src" / "phone_dino" / "mvtec_normal_successor.py"),
        "gitRevision": revision,
        "gitWorktreeClean": worktree_clean,
        "python": sys.version,
        "platform": platform.platform(),
        "pillowVersion": PILLOW_VERSION,
        "opencvVersion": cv2.__version__,
        "numpyVersion": np.__version__,
    }


def _validate_generation_provenance(value: object, *, repository_root: Path) -> dict[str, Any]:
    generation = _require_exact_fields(value, name="successor FIT V2 augmentation generation", required=GENERATION_FIELDS)
    expected_hashes = {
        "generatorModuleSha256": sha256_file(Path(__file__)),
        "generatorEntrypointSha256": sha256_file(
            repository_root / "tools" / "generate_mvtec_ad_successor_fit_augmentations.py"
        ),
        "normalSuccessorModuleSha256": sha256_file(repository_root / "src" / "phone_dino" / "mvtec_normal_successor.py"),
    }
    for name, expected in expected_hashes.items():
        if _require_sha256(generation.get(name), name=f"successor FIT V2 augmentation {name}") != expected:
            raise SuccessorFitAugmentationV2Error(f"successor FIT V2 {name} does not match this implementation")
    for name in ("python", "platform", "pillowVersion", "opencvVersion", "numpyVersion"):
        _require_string(generation.get(name), name=f"successor FIT V2 augmentation {name}")
    if generation.get("gitRevision") is not None:
        _require_string(generation.get("gitRevision"), name="successor FIT V2 augmentation gitRevision")
    if generation.get("gitWorktreeClean") is not None and not isinstance(generation.get("gitWorktreeClean"), bool):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 gitWorktreeClean must be a boolean or null")
    return generation


def _validate_fit_parents(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not parents:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation has no FIT parent records")
    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_sources: set[str] = set()
    for parent in parents:
        if parent.get("partition") != "FIT" or parent.get("kind") != "NOMINAL" or parent.get("defect") != "good":
            raise SuccessorFitAugmentationV2Error("successor FIT V2 may use FIT nominal-good parents only")
        case_id = parent.get("caseId")
        source_sha256 = parent.get("sourceSha256")
        if not isinstance(case_id, str) or case_id in seen_case_ids or not isinstance(source_sha256, str) or source_sha256 in seen_sources:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 parent caseId or source digest is invalid or duplicated")
        if not isinstance(parent.get("imagePath"), Path):
            raise SuccessorFitAugmentationV2Error("successor FIT V2 parent was not phase-safely loaded")
        seen_case_ids.add(case_id)
        seen_sources.add(source_sha256)
        result.append(parent)
    return sorted(result, key=lambda parent: str(parent["caseId"]))


def _load_successor_fit_parents(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    # Add the V2 two-way external-boundary guard before delegating to the
    # parent-chain loader.  These checks only inspect paths/metadata; no image
    # bytes are opened until that loader receives the exact FIT request below.
    for path, description in (
        (parent_holdout_path, "successor FIT V2 parent holdout manifest"),
        (parent_selection_contract_path, "successor FIT V2 parent selection contract"),
        (plan_path, "successor FIT V2 allocation plan"),
        (envelope_path, "successor FIT V2 phase envelope"),
    ):
        _require_external_input_file(path, description=description, repository_root=repository_root)
    _require_external_source_root(source_root, repository_root=repository_root)
    try:
        envelope, envelope_file_sha256, parents = successor.load_successor_safe_normal_inputs(
            parent_holdout_path,
            parent_selection_contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            partitions={"FIT"},
            repository_root=repository_root,
        )
    except successor.FreshNormalSuccessorError as error:
        raise SuccessorFitAugmentationV2Error(str(error)) from error
    return envelope, envelope_file_sha256, _validate_fit_parents(parents)


def _chain_bindings(envelope: dict[str, Any], *, envelope_file_sha256: str) -> dict[str, str]:
    parent_evidence = envelope.get("parentEvidence")
    if not isinstance(parent_evidence, dict):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 envelope parent evidence is missing")
    partition_identities = envelope.get("successorPartitionIdentities")
    if not isinstance(partition_identities, dict):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 envelope partition identities are missing")
    bindings = {
        "parentHoldoutManifestFileSha256": parent_evidence.get("holdoutManifestFileSha256"),
        "parentHoldoutManifestDeclaredSha256": parent_evidence.get("holdoutManifestDeclaredSha256"),
        "parentSelectionContractFileSha256": parent_evidence.get("selectionContractFileSha256"),
        "parentSelectionContractDeclaredSha256": parent_evidence.get("selectionContractDeclaredSha256"),
        "successorSealFileSha256": envelope.get("sealFileSha256"),
        "successorSealDeclaredSha256": envelope.get("sealDeclaredSha256"),
        "successorPlanFileSha256": envelope.get("planFileSha256"),
        "successorPlanDeclaredSha256": envelope.get("planDeclaredSha256"),
        "successorEnvelopeFileSha256": envelope_file_sha256,
        "successorEnvelopeDeclaredSha256": envelope.get("successorEnvelopeSha256"),
        "successorFitIdentitySha256": partition_identities.get("FIT"),
        "parentNormalConfirmationIdentitySha256": parent_evidence.get("parentNormalConfirmationIdentitySha256"),
    }
    return {
        name: _require_sha256(value, name=f"successor FIT V2 {name}")
        for name, value in bindings.items()
    }


def generate_successor_fit_augmentations(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    recipe_path: Path,
    output_dir: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Materialize the immutable R3 V2 package from successor ``FIT`` only.

    The parent data loader is deliberately invoked with exactly ``{"FIT"}``.
    It validates the complete JSON chain first, then opens and decodes only
    successor FIT parents.  No tuning, selection, remaining reserve, or parent
    confirmation image is accepted by this entry point.
    """

    envelope, envelope_file_sha256, parents = _load_successor_fit_parents(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        repository_root=repository_root,
    )
    recipe, recipe_sha256 = load_successor_fit_camera_recipe_v2(recipe_path)
    _expected_output_encoding()
    _prepare_external_output_directory(output_dir, repository_root=repository_root)
    records: list[dict[str, Any]] = []
    for parent in parents:
        source_image = _load_parent_rgb(parent)
        for variant in SUCCESSOR_FIT_VARIANTS:
            variant_id = int(variant["variantId"])
            component = str(variant["component"])
            parameters = sample_successor_fit_parameters_v2(
                recipe,
                recipe_sha256=recipe_sha256,
                parent_case_id=str(parent["caseId"]),
                parent_source_sha256=str(parent["sourceSha256"]),
                variant_id=variant_id,
            )
            case_id, relative_path = _expected_child_identity(
                parent,
                recipe_sha256=recipe_sha256,
                variant_id=variant_id,
            )
            if parameters["component"] != component:
                raise SuccessorFitAugmentationV2Error("successor FIT V2 variant component is inconsistent")
            data = _render_augmented_jpeg(source_image, parameters, variant_id=variant_id)
            target_path = output_dir.joinpath(*relative_path.parts)
            _reject_links_on_existing_path(target_path.parent, description="successor FIT V2 augmentation output")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            _reject_links_on_existing_path(target_path.parent, description="successor FIT V2 augmentation output")
            try:
                with target_path.open("xb") as stream:
                    stream.write(data)
            except OSError as error:
                raise SuccessorFitAugmentationV2Error("unable to write successor FIT V2 augmentation image") from error
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
                "component": component,
                "relativePath": relative_path.as_posix(),
                "sourceSha256": _sha256_bytes(data),
                "parameters": parameters,
                "outputEncoding": _expected_output_encoding(),
            })
    records.sort(key=lambda record: str(record["caseId"]))
    document: dict[str, Any] = {
        "schemaVersion": SUCCESSOR_FIT_AUGMENTATION_V2_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": SUCCESSOR_FIT_AUGMENTATION_V2_PURPOSE,
        "inputPolicy": SUCCESSOR_FIT_INPUT_POLICY,
        "blindPolicy": SUCCESSOR_FIT_BLIND_POLICY,
        "delegationPolicy": SUCCESSOR_FIT_DELEGATION_POLICY,
        "resultLabel": SUCCESSOR_FIT_RESULT_LABEL,
        "independenceLabel": SUCCESSOR_FIT_INDEPENDENCE_LABEL,
        "parentPartition": "FIT",
        **_chain_bindings(envelope, envelope_file_sha256=envelope_file_sha256),
        "recipeFileSha256": recipe_sha256,
        "recipe": recipe,
        "variantsPerParent": SUCCESSOR_FIT_VARIANTS_PER_PARENT,
        "generation": _generation_provenance(repository_root=repository_root),
        "records": records,
    }
    document["augmentationManifestSha256"] = _document_digest(document, "augmentationManifestSha256")
    manifest_path = output_dir / "augmentation_manifest.json"
    try:
        with manifest_path.open("xb") as stream:
            stream.write((json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    except OSError as error:
        raise SuccessorFitAugmentationV2Error("unable to write successor FIT V2 augmentation manifest") from error
    return document


def _validate_manifest_document(
    document: dict[str, Any],
    *,
    manifest_file_sha256: str,
    envelope: dict[str, Any],
    envelope_file_sha256: str,
    parents: list[dict[str, Any]],
    recipe: dict[str, Any],
    recipe_sha256: str,
    repository_root: Path,
) -> list[dict[str, Any]]:
    _require_exact_fields(document, name="successor FIT V2 augmentation manifest", required=MANIFEST_FIELDS)
    if document.get("schemaVersion") != SUCCESSOR_FIT_AUGMENTATION_V2_SCHEMA:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation manifest schema is unsupported")
    if document.get("authoritative") is not False or document.get("productionAuthorized") is not False:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation manifest must be non-authoritative and non-production")
    scope = {
        "purpose": SUCCESSOR_FIT_AUGMENTATION_V2_PURPOSE,
        "inputPolicy": SUCCESSOR_FIT_INPUT_POLICY,
        "blindPolicy": SUCCESSOR_FIT_BLIND_POLICY,
        "delegationPolicy": SUCCESSOR_FIT_DELEGATION_POLICY,
        "resultLabel": SUCCESSOR_FIT_RESULT_LABEL,
        "independenceLabel": SUCCESSOR_FIT_INDEPENDENCE_LABEL,
        "parentPartition": "FIT",
    }
    if any(document.get(name) != expected for name, expected in scope.items()):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation manifest scope is unsafe")
    if document.get("augmentationManifestSha256") != _document_digest(document, "augmentationManifestSha256"):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation manifest digest does not match")
    if not manifest_file_sha256:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation manifest file digest is missing")
    expected_bindings = _chain_bindings(envelope, envelope_file_sha256=envelope_file_sha256)
    expected_bindings["recipeFileSha256"] = recipe_sha256
    for name, expected in expected_bindings.items():
        if _require_sha256(document.get(name), name=f"successor FIT V2 manifest {name}") != expected:
            raise SuccessorFitAugmentationV2Error(f"successor FIT V2 manifest {name} does not match the closed chain")
    if document.get("recipe") != recipe:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 embedded recipe does not match the supplied recipe")
    if _require_positive_int(document.get("variantsPerParent"), name="variantsPerParent") != SUCCESSOR_FIT_VARIANTS_PER_PARENT:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 requires exactly three component-separated variants")
    _validate_generation_provenance(document.get("generation"), repository_root=repository_root)
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation manifest has no records")
    expected_count = len(parents) * SUCCESSOR_FIT_VARIANTS_PER_PARENT
    if len(raw_records) != expected_count:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation records do not have R3 parent coverage")
    return raw_records


def load_validated_successor_fit_augmentations_with_file_sha256(
    augmentation_manifest_path: Path,
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    recipe_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Validate the V2 package, source chain, JPEG profile, and every re-render.

    This invokes the successor loader with ``FIT`` only.  The JSON chain can
    be fully revalidated, but no image bytes from tuning, selection, remaining
    reserve, parent confirmation, blind, anomaly, or mask partitions are
    requested or opened.
    """

    envelope, envelope_file_sha256, parents = _load_successor_fit_parents(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        repository_root=repository_root,
    )
    recipe, recipe_sha256 = load_successor_fit_camera_recipe_v2(recipe_path)
    _expected_output_encoding()
    document, manifest_file_sha256 = _read_external_json(
        augmentation_manifest_path,
        description="successor FIT V2 augmentation manifest",
        repository_root=repository_root,
    )
    raw_records = _validate_manifest_document(
        document,
        manifest_file_sha256=manifest_file_sha256,
        envelope=envelope,
        envelope_file_sha256=envelope_file_sha256,
        parents=parents,
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        repository_root=repository_root,
    )
    output_root = augmentation_manifest_path.parent
    _require_external_package_root(output_root, repository_root=repository_root)
    parent_by_case_id = {str(parent["caseId"]): parent for parent in parents}
    expected_case_ids = {
        _expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=int(variant["variantId"]))[0]
        for parent in parents
        for variant in SUCCESSOR_FIT_VARIANTS
    }
    parent_images: dict[str, Image.Image] = {}
    seen_case_ids: set[str] = set()
    seen_relative_paths: set[str] = set()
    validated: list[dict[str, Any]] = []
    for value in raw_records:
        record = _require_exact_fields(value, name="successor FIT V2 augmentation record", required=RECORD_FIELDS)
        case_id = _require_string(record.get("caseId"), name="successor FIT V2 record caseId")
        if case_id in seen_case_ids:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation caseId is duplicated")
        seen_case_ids.add(case_id)
        parent_case_id = _require_string(record.get("parentCaseId"), name="successor FIT V2 record parentCaseId")
        parent = parent_by_case_id.get(parent_case_id)
        if parent is None:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation parent case is unknown")
        if _require_sha256(record.get("parentSourceSha256"), name="successor FIT V2 record parentSourceSha256") != parent["sourceSha256"]:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 parent source digest does not match")
        if record.get("sourceGroupId") != parent["sourceGroupId"] or record.get("category") != parent["category"]:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 parent group or category does not match")
        if record.get("parentPartition") != "FIT" or record.get("kind") != "NOMINAL" or record.get("defect") != "good":
            raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation record scope is unsafe")
        variant_id = _require_positive_int(record.get("variantId"), name="successor FIT V2 record variantId")
        component = _component_for_variant(variant_id)
        if record.get("component") != component:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 record component does not match variantId")
        expected_case_id, expected_relative_path = _expected_child_identity(
            parent,
            recipe_sha256=recipe_sha256,
            variant_id=variant_id,
        )
        if case_id != expected_case_id:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 caseId does not match parent and component")
        relative_path = _safe_relative_path(record.get("relativePath"), name="successor FIT V2 record relativePath")
        if relative_path != expected_relative_path:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 relativePath does not match parent and component")
        if relative_path.as_posix() in seen_relative_paths:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation relativePath is duplicated")
        seen_relative_paths.add(relative_path.as_posix())
        expected_parameters = sample_successor_fit_parameters_v2(
            recipe,
            recipe_sha256=recipe_sha256,
            parent_case_id=parent_case_id,
            parent_source_sha256=str(parent["sourceSha256"]),
            variant_id=variant_id,
        )
        parameters = _validate_parameters(record.get("parameters"), variant_id=variant_id)
        if parameters != expected_parameters:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 parameters do not match recipe and parent")
        _validate_output_encoding(record.get("outputEncoding"))
        output_path = _safe_file_under(
            output_root,
            relative_path,
            description="successor FIT V2 augmentation output image",
            repository_root=repository_root,
        )
        try:
            actual_bytes = output_path.read_bytes()
        except OSError as error:
            raise SuccessorFitAugmentationV2Error("unable to read successor FIT V2 augmentation output image") from error
        if _sha256_bytes(actual_bytes) != _require_sha256(record.get("sourceSha256"), name="successor FIT V2 record sourceSha256"):
            raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation output digest does not match")
        if _inspect_jpeg_output_encoding(actual_bytes) != _expected_output_encoding():
            raise SuccessorFitAugmentationV2Error("successor FIT V2 JPEG headers do not match the fixed recipe")
        source_image = parent_images.get(parent_case_id)
        if source_image is None:
            source_image = _load_parent_rgb(parent)
            parent_images[parent_case_id] = source_image
        expected_bytes = _render_augmented_jpeg(source_image, expected_parameters, variant_id=variant_id)
        if actual_bytes != expected_bytes:
            raise SuccessorFitAugmentationV2Error("successor FIT V2 pixels do not match the deterministic renderer")
        validated.append(dict(record))
    if [record["caseId"] for record in raw_records] != sorted(record["caseId"] for record in raw_records):
        raise SuccessorFitAugmentationV2Error("successor FIT V2 augmentation records must be sorted by caseId")
    if seen_case_ids != expected_case_ids:
        raise SuccessorFitAugmentationV2Error("successor FIT V2 records do not cover every FIT parent and component")
    return document, manifest_file_sha256, validated


def load_validated_successor_fit_augmentations(
    augmentation_manifest_path: Path,
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    recipe_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compatibility wrapper that omits the validated manifest file digest."""

    document, _manifest_file_sha256, records = load_validated_successor_fit_augmentations_with_file_sha256(
        augmentation_manifest_path,
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        recipe_path=recipe_path,
        repository_root=repository_root,
    )
    return document, records
