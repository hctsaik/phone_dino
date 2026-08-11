"""Synthetic-only local-defect augmentation for offline MVTec harness tests.

This module is intentionally separate from the frozen V1/V2 normal-only
augmentation packages.  It accepts parent bytes only through the successor
FIT-only safe loader, and renders declared *synthetic* patterns solely for
offline engineering tests.  Its outputs are not real anomalies, physical
defect models, qualification evidence, or production inputs.
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
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps, __version__ as PILLOW_VERSION

from phone_dino import mvtec_normal_successor as successor


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_ANOMALY_AUGMENTATION_V1_SCHEMA = "phone-dino.mvtec-ad-synthetic-only-augmentation/1.0"
SYNTHETIC_ANOMALY_RECIPE_V1_SCHEMA = "phone-dino.mvtec-ad-synthetic-only-recipe/1.0"
SYNTHETIC_ANOMALY_AUGMENTATION_V1_PURPOSE = "OFFLINE_MVTEC_SYNTHETIC_ONLY_AUGMENTATION_AND_TESTING"
SYNTHETIC_ANOMALY_RECIPE_V1_ID = "SYNTHETIC_ONLY_LOCAL_DEFECTS_V1_PNG"
SYNTHETIC_ANOMALY_RECIPE_V1_DESCRIPTION = (
    "Closed synthetic-only offline MVTec augmentation recipe. It renders deterministic local scratch, spot, and "
    "occlusion patterns on FIT nominal parents solely to exercise an engineering test harness. The patterns are not "
    "observed defects, do not model physical failure mechanisms, have no real-anomaly ground truth, and must not be "
    "used for qualification, threshold selection, release decisions, or production authorization."
)
SYNTHETIC_ANOMALY_INPUT_POLICY = "SUCCESSOR_FIT_RAW_NORMAL_PARENTS_ONLY"
SYNTHETIC_ANOMALY_BLIND_POLICY = "NO_BLIND_OR_TRUE_ANOMALY_DATA"
SYNTHETIC_ANOMALY_RESULT_LABEL = "SYNTHETIC_ONLY_NOT_REAL_ANOMALY_PERFORMANCE"
SYNTHETIC_ANOMALY_SAMPLING_ALGORITHM = "NAMED_SHA256_SYNTHETIC_DEFECT_SUBSTREAMS_V1"
SYNTHETIC_ANOMALY_PARENT_SPLIT_ALGORITHM = "SOURCE_SHA256_THEN_CASE_ID_PER_CATEGORY_V1"
SYNTHETIC_ANOMALY_PARENT_SPLIT_COUNTS_PER_CATEGORY = {
    "PROTOTYPE": 6,
    "CALIBRATION": 2,
    "QUERY": 4,
}
SYNTHETIC_ANOMALY_FIT_PARENTS_PER_CATEGORY = sum(SYNTHETIC_ANOMALY_PARENT_SPLIT_COUNTS_PER_CATEGORY.values())
SYNTHETIC_ANOMALY_OUTPUT_ENCODING = {
    "format": "PNG",
    "mode": "RGB",
    "compressLevel": 9,
    "optimize": False,
}
SYNTHETIC_ANOMALY_DEFECT_PROFILES = {
    "LOCAL_SCRATCH": {
        "centerMarginFraction": 0.22,
        "minLengthFraction": 0.16,
        "maxLengthFraction": 0.34,
        "minWidthFraction": 0.009,
        "maxWidthFraction": 0.024,
        "minOpacity": 0.70,
        "maxOpacity": 0.94,
    },
    "LOCAL_SPOT": {
        "centerMarginFraction": 0.22,
        "minRadiusFraction": 0.065,
        "maxRadiusFraction": 0.15,
        "minSoftnessFraction": 0.008,
        "maxSoftnessFraction": 0.035,
        "minOpacity": 0.52,
        "maxOpacity": 0.80,
    },
    "LOCAL_OCCLUSION": {
        "centerMarginFraction": 0.22,
        "minWidthFraction": 0.12,
        "maxWidthFraction": 0.24,
        "minHeightFraction": 0.10,
        "maxHeightFraction": 0.20,
        "minCornerRadiusFraction": 0.02,
        "maxCornerRadiusFraction": 0.06,
        "minOpacity": 0.74,
        "maxOpacity": 0.92,
    },
}
SYNTHETIC_ANOMALY_VARIANTS = (
    {"variantId": 1, "syntheticDefectFamily": "LOCAL_SCRATCH"},
    {"variantId": 2, "syntheticDefectFamily": "LOCAL_SPOT"},
    {"variantId": 3, "syntheticDefectFamily": "LOCAL_OCCLUSION"},
)
SYNTHETIC_ANOMALY_VARIANTS_PER_PARENT = len(SYNTHETIC_ANOMALY_VARIANTS)

RECIPE_FIELDS = {
    "schemaVersion",
    "id",
    "description",
    "authoritative",
    "productionAuthorized",
    "syntheticOnly",
    "purpose",
    "samplingAlgorithm",
    "defectProfiles",
    "outputEncoding",
}
MANIFEST_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "syntheticOnly",
    "purpose",
    "inputPolicy",
    "blindPolicy",
    "resultLabel",
    "parentPartition",
    "parentSplitAlgorithm",
    "parentSplitCountsPerCategory",
    "syntheticQueryParentIdentitySha256",
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
}
RECORD_FIELDS = {
    "caseId",
    "parentCaseId",
    "parentSourceSha256",
    "sourceGroupId",
    "category",
    "parentPartition",
    "syntheticTestRole",
    "syntheticLabel",
    "syntheticDefectFamily",
    "variantId",
    "relativePath",
    "sourceSha256",
    "parameters",
    "outputEncoding",
}


class SyntheticAnomalyAugmentationError(ValueError):
    """Raised when a synthetic-only augmentation artifact is unsafe."""


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without reading an entire file into memory."""

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
        raise SyntheticAnomalyAugmentationError(f"{name} must be an object")
    missing = required.difference(document)
    unknown = set(document).difference(required)
    if missing:
        raise SyntheticAnomalyAugmentationError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise SyntheticAnomalyAugmentationError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")
    return document


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SyntheticAnomalyAugmentationError(f"{name} must be an object")
    return value


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SyntheticAnomalyAugmentationError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    result = _require_string(value, name=name)
    if len(result) != 71 or not result.startswith("sha256:"):
        raise SyntheticAnomalyAugmentationError(f"{name} must be a SHA-256 digest")
    try:
        int(result[7:], 16)
    except ValueError as error:
        raise SyntheticAnomalyAugmentationError(f"{name} must be a SHA-256 digest") from error
    return result


def _require_positive_int(value: object, *, name: str, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SyntheticAnomalyAugmentationError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise SyntheticAnomalyAugmentationError(f"{name} must be at most {maximum}")
    return value


def _require_finite_number(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SyntheticAnomalyAugmentationError(f"{name} must be a finite number")
    return float(value)


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
                raise SyntheticAnomalyAugmentationError(f"{description} contains a symbolic link or reparse point")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _require_external_input_file(path: Path, *, description: str, repository_root: Path) -> None:
    _reject_links_on_existing_path(path, description=description)
    external_root = path if path.is_dir() else path.parent
    if _is_under(repository_root, path) or _is_under(external_root, repository_root):
        raise SyntheticAnomalyAugmentationError(f"{description} must stay outside the Git working tree")
    if not path.is_file():
        raise SyntheticAnomalyAugmentationError(f"{description} is missing")


def _require_external_directory(path: Path, *, description: str, repository_root: Path) -> None:
    _reject_links_on_existing_path(path, description=description)
    if _is_under(repository_root, path) or _is_under(path, repository_root):
        raise SyntheticAnomalyAugmentationError(f"{description} must stay outside the Git working tree")
    if not path.is_dir():
        raise SyntheticAnomalyAugmentationError(f"{description} is missing")


def _prepare_external_output_directory(output_dir: Path, *, repository_root: Path) -> None:
    _reject_links_on_existing_path(output_dir, description="synthetic-only augmentation output")
    if _is_under(repository_root, output_dir) or _is_under(output_dir, repository_root):
        raise SyntheticAnomalyAugmentationError("synthetic-only augmentation output must stay outside the Git working tree")
    if output_dir.exists() or output_dir.is_symlink():
        raise SyntheticAnomalyAugmentationError("synthetic-only augmentation output already exists; choose a new immutable path")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise SyntheticAnomalyAugmentationError("unable to create synthetic-only augmentation output directory") from error
    _reject_links_on_existing_path(output_dir, description="synthetic-only augmentation output")


def _safe_relative_path(value: object, *, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or "\\" in value:
        raise SyntheticAnomalyAugmentationError(f"{name} must be a non-empty safe relative path")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value)
    if windows.is_absolute() or windows.drive or windows.root or posix.is_absolute() or not posix.parts:
        raise SyntheticAnomalyAugmentationError(f"{name} must be a safe relative path")
    if any(
        part in {"", ".", ".."}
        or ":" in part
        or part != part.rstrip(" .")
        or PureWindowsPath(part).is_reserved()
        for part in posix.parts
    ):
        raise SyntheticAnomalyAugmentationError(f"{name} must be a safe relative path")
    return posix


def _safe_file_under(root: Path, relative: PurePosixPath, *, description: str, repository_root: Path) -> Path:
    _reject_links_on_existing_path(root, description=description)
    candidate = root.joinpath(*relative.parts)
    _reject_links_on_existing_path(candidate, description=description)
    if not candidate.is_file() or not _is_under(root, candidate) or _is_under(repository_root, candidate):
        raise SyntheticAnomalyAugmentationError(f"{description} is missing or escapes its external root")
    return candidate


def _parse_json_bytes(raw: bytes, *, description: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SyntheticAnomalyAugmentationError(f"{description} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise SyntheticAnomalyAugmentationError(f"{description} contains a non-finite JSON value: {value}")

    try:
        parsed = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SyntheticAnomalyAugmentationError(f"unable to read {description}") from error
    if not isinstance(parsed, dict):
        raise SyntheticAnomalyAugmentationError(f"{description} must be a JSON object")
    return parsed


def _read_external_json(path: Path, *, description: str, repository_root: Path) -> tuple[dict[str, Any], str]:
    _require_external_input_file(path, description=description, repository_root=repository_root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SyntheticAnomalyAugmentationError(f"unable to read {description}") from error
    return _parse_json_bytes(raw, description=description), _sha256_bytes(raw)


def _read_recipe_json(path: Path) -> tuple[dict[str, Any], str]:
    _reject_links_on_existing_path(path, description="synthetic-only recipe")
    if not path.is_file():
        raise SyntheticAnomalyAugmentationError("synthetic-only recipe is missing")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SyntheticAnomalyAugmentationError("unable to read synthetic-only recipe") from error
    return _parse_json_bytes(raw, description="synthetic-only recipe"), _sha256_bytes(raw)


def _expect_exact_mapping(value: object, *, name: str, expected: dict[str, Any]) -> dict[str, Any]:
    mapping = _require_exact_fields(value, name=name, required=set(expected))
    if mapping != expected:
        raise SyntheticAnomalyAugmentationError(f"{name} does not match the closed synthetic-only recipe")
    return mapping


def load_synthetic_anomaly_recipe_v1(recipe_path: Path) -> tuple[dict[str, Any], str]:
    """Load the one closed synthetic-only local-pattern recipe."""

    recipe, file_sha256 = _read_recipe_json(recipe_path)
    _require_exact_fields(recipe, name="synthetic-only recipe", required=RECIPE_FIELDS)
    if recipe.get("schemaVersion") != SYNTHETIC_ANOMALY_RECIPE_V1_SCHEMA or recipe.get("id") != SYNTHETIC_ANOMALY_RECIPE_V1_ID:
        raise SyntheticAnomalyAugmentationError("synthetic-only recipe schema or id is unsupported")
    if recipe.get("description") != SYNTHETIC_ANOMALY_RECIPE_V1_DESCRIPTION:
        raise SyntheticAnomalyAugmentationError("synthetic-only recipe description is not locked")
    if recipe.get("authoritative") is not False or recipe.get("productionAuthorized") is not False:
        raise SyntheticAnomalyAugmentationError("synthetic-only recipe must be non-authoritative and non-production")
    if recipe.get("syntheticOnly") is not True or recipe.get("purpose") != SYNTHETIC_ANOMALY_AUGMENTATION_V1_PURPOSE:
        raise SyntheticAnomalyAugmentationError("synthetic-only recipe scope is unsafe")
    if recipe.get("samplingAlgorithm") != SYNTHETIC_ANOMALY_SAMPLING_ALGORITHM:
        raise SyntheticAnomalyAugmentationError("synthetic-only recipe sampling algorithm is unsupported")
    _expect_exact_mapping(
        recipe.get("defectProfiles"),
        name="synthetic-only recipe defectProfiles",
        expected=SYNTHETIC_ANOMALY_DEFECT_PROFILES,
    )
    _expect_exact_mapping(
        recipe.get("outputEncoding"),
        name="synthetic-only recipe outputEncoding",
        expected=SYNTHETIC_ANOMALY_OUTPUT_ENCODING,
    )
    return recipe, file_sha256


def _variant_for_id(variant_id: int) -> dict[str, Any]:
    for variant in SYNTHETIC_ANOMALY_VARIANTS:
        if variant["variantId"] == variant_id:
            return variant
    raise SyntheticAnomalyAugmentationError("synthetic-only variantId is unsupported")


def _rounded(value: float) -> float:
    return round(value, 8)


def _derive_seed(
    recipe_sha256: str,
    parent_case_id: str,
    parent_source_sha256: str,
    variant_id: int,
    synthetic_defect_family: str,
) -> int:
    expected = _variant_for_id(variant_id)["syntheticDefectFamily"]
    if synthetic_defect_family != expected:
        raise SyntheticAnomalyAugmentationError("synthetic-only defect family does not match variantId")
    payload = "\0".join((
        SYNTHETIC_ANOMALY_SAMPLING_ALGORITHM,
        recipe_sha256,
        parent_case_id,
        parent_source_sha256,
        str(variant_id),
        synthetic_defect_family,
    )).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _sample_center(rng: random.Random, *, profile: dict[str, Any]) -> tuple[float, float]:
    margin = float(profile["centerMarginFraction"])
    return _rounded(rng.uniform(margin, 1.0 - margin)), _rounded(rng.uniform(margin, 1.0 - margin))


def _sample_gray(rng: random.Random) -> list[int]:
    # Use a high-contrast neutral tone, not a claim about defect appearance.
    tone = rng.randint(12, 70) if rng.randrange(2) == 0 else rng.randint(185, 243)
    return [tone, tone, tone]


def sample_synthetic_anomaly_parameters_v1(
    recipe: dict[str, Any],
    *,
    recipe_sha256: str,
    parent_case_id: str,
    parent_source_sha256: str,
    variant_id: int,
) -> dict[str, Any]:
    """Derive one deterministic synthetic-only local pattern specification."""

    if recipe.get("samplingAlgorithm") != SYNTHETIC_ANOMALY_SAMPLING_ALGORITHM:
        raise SyntheticAnomalyAugmentationError("synthetic-only recipe sampling algorithm is unsupported")
    variant = _variant_for_id(variant_id)
    family = str(variant["syntheticDefectFamily"])
    profile = SYNTHETIC_ANOMALY_DEFECT_PROFILES[family]
    seed = _derive_seed(recipe_sha256, parent_case_id, parent_source_sha256, variant_id, family)
    rng = random.Random(seed)
    center_x, center_y = _sample_center(rng, profile=profile)
    common = {"syntheticDefectFamily": family, "seed": str(seed)}
    if family == "LOCAL_SCRATCH":
        return {
            **common,
            "scratch": {
                "centerXFraction": center_x,
                "centerYFraction": center_y,
                "angleDegrees": _rounded(rng.uniform(0.0, 180.0)),
                "lengthFraction": _rounded(rng.uniform(profile["minLengthFraction"], profile["maxLengthFraction"])),
                "widthFraction": _rounded(rng.uniform(profile["minWidthFraction"], profile["maxWidthFraction"])),
                "colorRgb": _sample_gray(rng),
                "opacity": _rounded(rng.uniform(profile["minOpacity"], profile["maxOpacity"])),
            },
        }
    if family == "LOCAL_SPOT":
        return {
            **common,
            "spot": {
                "centerXFraction": center_x,
                "centerYFraction": center_y,
                "radiusFraction": _rounded(rng.uniform(profile["minRadiusFraction"], profile["maxRadiusFraction"])),
                "softnessFraction": _rounded(rng.uniform(profile["minSoftnessFraction"], profile["maxSoftnessFraction"])),
                "colorRgb": _sample_gray(rng),
                "opacity": _rounded(rng.uniform(profile["minOpacity"], profile["maxOpacity"])),
            },
        }
    if family == "LOCAL_OCCLUSION":
        return {
            **common,
            "occlusion": {
                "centerXFraction": center_x,
                "centerYFraction": center_y,
                "widthFraction": _rounded(rng.uniform(profile["minWidthFraction"], profile["maxWidthFraction"])),
                "heightFraction": _rounded(rng.uniform(profile["minHeightFraction"], profile["maxHeightFraction"])),
                "cornerRadiusFraction": _rounded(
                    rng.uniform(profile["minCornerRadiusFraction"], profile["maxCornerRadiusFraction"])
                ),
                "colorRgb": _sample_gray(rng),
                "opacity": _rounded(rng.uniform(profile["minOpacity"], profile["maxOpacity"])),
            },
        }
    raise SyntheticAnomalyAugmentationError("synthetic-only defect family is unsupported")


def _validate_color(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 3 or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise SyntheticAnomalyAugmentationError(f"{name} must be three integer RGB components")
    if any(item < 0 or item > 255 for item in value):
        raise SyntheticAnomalyAugmentationError(f"{name} RGB components are out of range")
    return list(value)


def _validate_parameter_mapping(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    mapping = _require_exact_fields(value, name=name, required=fields)
    for key, item in mapping.items():
        if key == "colorRgb":
            _validate_color(item, name=f"{name} colorRgb")
        else:
            _require_finite_number(item, name=f"{name} {key}")
    return mapping


def _validate_parameters(value: object, *, synthetic_defect_family: str) -> dict[str, Any]:
    fields_by_family = {
        "LOCAL_SCRATCH": ("scratch", {"centerXFraction", "centerYFraction", "angleDegrees", "lengthFraction", "widthFraction", "colorRgb", "opacity"}),
        "LOCAL_SPOT": ("spot", {"centerXFraction", "centerYFraction", "radiusFraction", "softnessFraction", "colorRgb", "opacity"}),
        "LOCAL_OCCLUSION": (
            "occlusion",
            {"centerXFraction", "centerYFraction", "widthFraction", "heightFraction", "cornerRadiusFraction", "colorRgb", "opacity"},
        ),
    }
    expected = fields_by_family.get(synthetic_defect_family)
    if expected is None:
        raise SyntheticAnomalyAugmentationError("synthetic-only defect family is unsupported")
    child_name, child_fields = expected
    mapping = _require_exact_fields(
        value,
        name="synthetic-only parameters",
        required={"syntheticDefectFamily", "seed", child_name},
    )
    if mapping.get("syntheticDefectFamily") != synthetic_defect_family:
        raise SyntheticAnomalyAugmentationError("synthetic-only parameter family is inconsistent")
    seed = _require_string(mapping.get("seed"), name="synthetic-only parameter seed")
    if not seed.isdigit():
        raise SyntheticAnomalyAugmentationError("synthetic-only parameter seed must be a decimal integer")
    _validate_parameter_mapping(mapping.get(child_name), name=f"synthetic-only {child_name} parameters", fields=child_fields)
    return mapping


def _pixel_coordinate(fraction: float, length: int) -> float:
    return max(0.0, min(float(length - 1), fraction * float(length - 1)))


def _alpha(value: float) -> int:
    return max(0, min(255, int(round(value * 255.0))))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def apply_synthetic_anomaly_augmentation(
    image: Image.Image,
    parameters: dict[str, Any],
    *,
    synthetic_defect_family: str,
) -> Image.Image:
    """Render a declared local synthetic pattern and return an RGB image.

    This renderer has no physical-defect interpretation.  It is public so the
    test harness can reproduce the exact synthetic fixture bytes.
    """

    _validate_parameters(parameters, synthetic_defect_family=synthetic_defect_family)
    if image.width < 8 or image.height < 8:
        raise SyntheticAnomalyAugmentationError("synthetic-only parent image dimensions must be at least 8 by 8")
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    minimum_dimension = float(min(base.width, base.height))
    if synthetic_defect_family == "LOCAL_SCRATCH":
        item = parameters["scratch"]
        center_x = _pixel_coordinate(float(item["centerXFraction"]), base.width)
        center_y = _pixel_coordinate(float(item["centerYFraction"]), base.height)
        half_length = float(item["lengthFraction"]) * minimum_dimension / 2.0
        radians = math.radians(float(item["angleDegrees"]))
        delta_x = math.cos(radians) * half_length
        delta_y = math.sin(radians) * half_length
        line_width = max(1, int(round(float(item["widthFraction"]) * minimum_dimension)))
        ImageDraw.Draw(layer).line(
            (
                _clamp(center_x - delta_x, 0.0, base.width - 1.0),
                _clamp(center_y - delta_y, 0.0, base.height - 1.0),
                _clamp(center_x + delta_x, 0.0, base.width - 1.0),
                _clamp(center_y + delta_y, 0.0, base.height - 1.0),
            ),
            fill=(*_validate_color(item["colorRgb"], name="synthetic-only scratch color"), _alpha(float(item["opacity"]))),
            width=line_width,
        )
    elif synthetic_defect_family == "LOCAL_SPOT":
        item = parameters["spot"]
        center_x = _pixel_coordinate(float(item["centerXFraction"]), base.width)
        center_y = _pixel_coordinate(float(item["centerYFraction"]), base.height)
        radius = max(1.0, float(item["radiusFraction"]) * minimum_dimension)
        ImageDraw.Draw(layer).ellipse(
            (
                _clamp(center_x - radius, 0.0, base.width - 1.0),
                _clamp(center_y - radius, 0.0, base.height - 1.0),
                _clamp(center_x + radius, 0.0, base.width - 1.0),
                _clamp(center_y + radius, 0.0, base.height - 1.0),
            ),
            fill=(*_validate_color(item["colorRgb"], name="synthetic-only spot color"), _alpha(float(item["opacity"]))),
        )
        layer = layer.filter(ImageFilter.GaussianBlur(radius=float(item["softnessFraction"]) * minimum_dimension))
    elif synthetic_defect_family == "LOCAL_OCCLUSION":
        item = parameters["occlusion"]
        center_x = _pixel_coordinate(float(item["centerXFraction"]), base.width)
        center_y = _pixel_coordinate(float(item["centerYFraction"]), base.height)
        half_width = max(0.5, float(item["widthFraction"]) * minimum_dimension / 2.0)
        half_height = max(0.5, float(item["heightFraction"]) * minimum_dimension / 2.0)
        ImageDraw.Draw(layer).rounded_rectangle(
            (
                _clamp(center_x - half_width, 0.0, base.width - 1.0),
                _clamp(center_y - half_height, 0.0, base.height - 1.0),
                _clamp(center_x + half_width, 0.0, base.width - 1.0),
                _clamp(center_y + half_height, 0.0, base.height - 1.0),
            ),
            radius=max(0.0, float(item["cornerRadiusFraction"]) * minimum_dimension),
            fill=(*_validate_color(item["colorRgb"], name="synthetic-only occlusion color"), _alpha(float(item["opacity"]))),
        )
    else:  # _validate_parameters already rejects this; keep the renderer fail-closed.
        raise SyntheticAnomalyAugmentationError("synthetic-only defect family is unsupported")
    return Image.alpha_composite(base, layer).convert("RGB")


def _inspect_png_output(data: bytes) -> dict[str, Any]:
    try:
        with Image.open(BytesIO(data)) as opened:
            opened.load()
            result = {"format": opened.format, "mode": opened.mode}
    except (OSError, SyntaxError, ValueError) as error:
        raise SyntheticAnomalyAugmentationError("synthetic-only output is not a decodable PNG") from error
    if result != {"format": "PNG", "mode": "RGB"}:
        raise SyntheticAnomalyAugmentationError("synthetic-only output encoding is unsupported")
    return result


def _render_augmented_png(image: Image.Image, parameters: dict[str, Any], *, synthetic_defect_family: str) -> bytes:
    augmented = apply_synthetic_anomaly_augmentation(
        image,
        parameters,
        synthetic_defect_family=synthetic_defect_family,
    )
    buffer = BytesIO()
    augmented.save(buffer, format="PNG", optimize=False, compress_level=9)
    data = buffer.getvalue()
    _inspect_png_output(data)
    return data


def _load_parent_rgb(parent: dict[str, Any]) -> Image.Image:
    source_path = parent.get("imagePath")
    if not isinstance(source_path, Path):
        raise SyntheticAnomalyAugmentationError("synthetic-only parent has no validated image path")
    if sha256_file(source_path) != parent.get("sourceSha256"):
        raise SyntheticAnomalyAugmentationError("synthetic-only parent changed before augmentation")
    try:
        with Image.open(source_path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, SyntaxError, ValueError) as error:
        raise SyntheticAnomalyAugmentationError("unable to decode synthetic-only FIT parent image") from error
    if sha256_file(source_path) != parent.get("sourceSha256"):
        raise SyntheticAnomalyAugmentationError("synthetic-only parent changed while it was decoded")
    return image


def _expected_child_identity(
    parent: dict[str, Any],
    *,
    recipe_sha256: str,
    variant_id: int,
) -> tuple[str, PurePosixPath]:
    variant = _variant_for_id(variant_id)
    family = str(variant["syntheticDefectFamily"])
    case_id = f"{parent['caseId']}/synthetic-only-v1/{family.lower()}"
    name = hashlib.sha256(
        "\0".join((
            SYNTHETIC_ANOMALY_AUGMENTATION_V1_SCHEMA,
            recipe_sha256,
            str(parent["caseId"]),
            str(parent["sourceSha256"]),
            str(variant_id),
            family,
        )).encode("utf-8")
    ).hexdigest()
    return case_id, PurePosixPath("images") / f"{name}-{family.lower()}.png"


def _generation_provenance(*, repository_root: Path) -> dict[str, Any]:
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
            repository_root / "tools" / "generate_mvtec_ad_synthetic_anomaly_augmentations.py"
        ),
        "normalSuccessorModuleSha256": sha256_file(repository_root / "src" / "phone_dino" / "mvtec_normal_successor.py"),
        "gitRevision": revision,
        "gitWorktreeClean": worktree_clean,
        "python": sys.version,
        "platform": platform.platform(),
        "pillowVersion": PILLOW_VERSION,
    }


def _validate_generation_provenance(value: object, *, repository_root: Path) -> dict[str, Any]:
    generation = _require_exact_fields(value, name="synthetic-only augmentation generation", required=GENERATION_FIELDS)
    expected_hashes = {
        "generatorModuleSha256": sha256_file(Path(__file__)),
        "generatorEntrypointSha256": sha256_file(
            repository_root / "tools" / "generate_mvtec_ad_synthetic_anomaly_augmentations.py"
        ),
        "normalSuccessorModuleSha256": sha256_file(repository_root / "src" / "phone_dino" / "mvtec_normal_successor.py"),
    }
    for name, expected in expected_hashes.items():
        if _require_sha256(generation.get(name), name=f"synthetic-only augmentation {name}") != expected:
            raise SyntheticAnomalyAugmentationError(f"synthetic-only augmentation {name} does not match this implementation")
    for name in ("python", "platform", "pillowVersion"):
        _require_string(generation.get(name), name=f"synthetic-only augmentation {name}")
    if generation.get("gitRevision") is not None:
        _require_string(generation.get("gitRevision"), name="synthetic-only augmentation gitRevision")
    if generation.get("gitWorktreeClean") is not None and not isinstance(generation.get("gitWorktreeClean"), bool):
        raise SyntheticAnomalyAugmentationError("synthetic-only augmentation gitWorktreeClean must be a boolean or null")
    return generation


def _validate_fit_parents(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not parents:
        raise SyntheticAnomalyAugmentationError("synthetic-only augmentation has no FIT parent records")
    result: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_source_groups: set[str] = set()
    for parent in parents:
        if parent.get("partition") != "FIT" or parent.get("kind") != "NOMINAL" or parent.get("defect") != "good":
            raise SyntheticAnomalyAugmentationError("synthetic-only augmentation may use FIT nominal-good parents only")
        case_id = parent.get("caseId")
        source_sha256 = parent.get("sourceSha256")
        if not isinstance(case_id, str) or case_id in seen_case_ids or not isinstance(source_sha256, str) or source_sha256 in seen_sources:
            raise SyntheticAnomalyAugmentationError("synthetic-only parent caseId or source digest is invalid or duplicated")
        source_group_id = parent.get("sourceGroupId")
        if not isinstance(source_group_id, str) or not source_group_id or not isinstance(parent.get("category"), str):
            raise SyntheticAnomalyAugmentationError("synthetic-only parent group or category is invalid")
        if not isinstance(parent.get("imagePath"), Path):
            raise SyntheticAnomalyAugmentationError("synthetic-only parent was not phase-safely loaded")
        seen_case_ids.add(case_id)
        seen_sources.add(source_sha256)
        if source_group_id in seen_source_groups:
            raise SyntheticAnomalyAugmentationError("synthetic-only parent source group is duplicated")
        seen_source_groups.add(source_group_id)
        result.append(parent)
    return sorted(result, key=lambda parent: str(parent["caseId"]))


def split_synthetic_anomaly_fit_parents(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign the closed per-category prototype/calibration/query roles.

    The split uses only immutable FIT identities, before any synthetic label or
    pixel rendering is considered.  Each category must supply exactly 12 FIT
    parents, ranked by ``(sourceSha256, caseId)``: six ``PROTOTYPE``, two
    ``CALIBRATION``, and four ``QUERY``.  Returned records are copies with a
    non-serialized ``syntheticTestRole`` helper field.
    """

    validated = _validate_fit_parents(parents)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for parent in validated:
        by_category.setdefault(str(parent["category"]), []).append(parent)
    result: list[dict[str, Any]] = []
    prototype_limit = SYNTHETIC_ANOMALY_PARENT_SPLIT_COUNTS_PER_CATEGORY["PROTOTYPE"]
    calibration_limit = prototype_limit + SYNTHETIC_ANOMALY_PARENT_SPLIT_COUNTS_PER_CATEGORY["CALIBRATION"]
    for category in sorted(by_category):
        ranked = sorted(
            by_category[category],
            key=lambda parent: (str(parent["sourceSha256"]), str(parent["caseId"])),
        )
        if len(ranked) != SYNTHETIC_ANOMALY_FIT_PARENTS_PER_CATEGORY:
            raise SyntheticAnomalyAugmentationError(
                "synthetic-only FIT parent split requires exactly "
                f"{SYNTHETIC_ANOMALY_FIT_PARENTS_PER_CATEGORY} parents per category"
            )
        for index, parent in enumerate(ranked):
            if index < prototype_limit:
                role = "PROTOTYPE"
            elif index < calibration_limit:
                role = "CALIBRATION"
            else:
                role = "QUERY"
            result.append({**parent, "syntheticTestRole": role})
    return sorted(result, key=lambda parent: str(parent["caseId"]))


def _query_parent_identity(parents: list[dict[str, Any]]) -> str:
    query_parents = [parent for parent in parents if parent.get("syntheticTestRole") == "QUERY"]
    if not query_parents:
        raise SyntheticAnomalyAugmentationError("synthetic-only parent split has no QUERY parents")
    return successor.canonical_json_sha256([
        {
            "caseId": parent["caseId"],
            "category": parent["category"],
            "sourceSha256": parent["sourceSha256"],
            "sourceGroupId": parent["sourceGroupId"],
            "syntheticTestRole": parent["syntheticTestRole"],
        }
        for parent in sorted(query_parents, key=lambda item: str(item["caseId"]))
    ])


def _load_synthetic_fit_parents(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    # The prechecks inspect only external paths/metadata.  The exact FIT-only
    # request below is the first path permitted to open parent image bytes.
    for path, description in (
        (parent_holdout_path, "synthetic-only parent holdout manifest"),
        (parent_selection_contract_path, "synthetic-only parent selection contract"),
        (plan_path, "synthetic-only allocation plan"),
        (envelope_path, "synthetic-only phase envelope"),
    ):
        _require_external_input_file(path, description=description, repository_root=repository_root)
    _require_external_directory(source_root, description="synthetic-only source root", repository_root=repository_root)
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
        raise SyntheticAnomalyAugmentationError(str(error)) from error
    return envelope, envelope_file_sha256, _validate_fit_parents(parents)


def _chain_bindings(envelope: dict[str, Any], *, envelope_file_sha256: str) -> dict[str, str]:
    parent_evidence = _require_mapping(envelope.get("parentEvidence"), name="synthetic-only envelope parent evidence")
    partition_identities = _require_mapping(
        envelope.get("successorPartitionIdentities"),
        name="synthetic-only envelope partition identities",
    )
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
        name: _require_sha256(value, name=f"synthetic-only {name}")
        for name, value in bindings.items()
    }


def generate_synthetic_anomaly_augmentations(
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
    """Materialize an immutable synthetic-only package from successor ``FIT`` only.

    The safe parent loader is always called with exactly ``{"FIT"}``.  This
    API cannot request tuning, selection, reserve, confirmation, blind, or
    true-anomaly image inputs.
    """

    envelope, envelope_file_sha256, fit_parents = _load_synthetic_fit_parents(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        repository_root=repository_root,
    )
    recipe, recipe_sha256 = load_synthetic_anomaly_recipe_v1(recipe_path)
    parents = split_synthetic_anomaly_fit_parents(fit_parents)
    query_parents = [parent for parent in parents if parent["syntheticTestRole"] == "QUERY"]
    _prepare_external_output_directory(output_dir, repository_root=repository_root)
    records: list[dict[str, Any]] = []
    for parent in query_parents:
        source_image = _load_parent_rgb(parent)
        for variant in SYNTHETIC_ANOMALY_VARIANTS:
            variant_id = int(variant["variantId"])
            family = str(variant["syntheticDefectFamily"])
            parameters = sample_synthetic_anomaly_parameters_v1(
                recipe,
                recipe_sha256=recipe_sha256,
                parent_case_id=str(parent["caseId"]),
                parent_source_sha256=str(parent["sourceSha256"]),
                variant_id=variant_id,
            )
            case_id, relative_path = _expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=variant_id)
            if parameters["syntheticDefectFamily"] != family:
                raise SyntheticAnomalyAugmentationError("synthetic-only variant family is inconsistent")
            data = _render_augmented_png(source_image, parameters, synthetic_defect_family=family)
            target_path = output_dir.joinpath(*relative_path.parts)
            _reject_links_on_existing_path(target_path.parent, description="synthetic-only augmentation output")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            _reject_links_on_existing_path(target_path.parent, description="synthetic-only augmentation output")
            try:
                with target_path.open("xb") as stream:
                    stream.write(data)
            except OSError as error:
                raise SyntheticAnomalyAugmentationError("unable to write synthetic-only augmentation image") from error
            records.append({
                "caseId": case_id,
                "parentCaseId": parent["caseId"],
                "parentSourceSha256": parent["sourceSha256"],
                "sourceGroupId": parent["sourceGroupId"],
                "category": parent["category"],
                "parentPartition": "FIT",
                "syntheticTestRole": "QUERY",
                "syntheticLabel": "SYNTHETIC_ANOMALY",
                "syntheticDefectFamily": family,
                "variantId": variant_id,
                "relativePath": relative_path.as_posix(),
                "sourceSha256": _sha256_bytes(data),
                "parameters": parameters,
                "outputEncoding": dict(SYNTHETIC_ANOMALY_OUTPUT_ENCODING),
            })
    records.sort(key=lambda record: str(record["caseId"]))
    document: dict[str, Any] = {
        "schemaVersion": SYNTHETIC_ANOMALY_AUGMENTATION_V1_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "purpose": SYNTHETIC_ANOMALY_AUGMENTATION_V1_PURPOSE,
        "inputPolicy": SYNTHETIC_ANOMALY_INPUT_POLICY,
        "blindPolicy": SYNTHETIC_ANOMALY_BLIND_POLICY,
        "resultLabel": SYNTHETIC_ANOMALY_RESULT_LABEL,
        "parentPartition": "FIT",
        "parentSplitAlgorithm": SYNTHETIC_ANOMALY_PARENT_SPLIT_ALGORITHM,
        "parentSplitCountsPerCategory": dict(SYNTHETIC_ANOMALY_PARENT_SPLIT_COUNTS_PER_CATEGORY),
        "syntheticQueryParentIdentitySha256": _query_parent_identity(parents),
        **_chain_bindings(envelope, envelope_file_sha256=envelope_file_sha256),
        "recipeFileSha256": recipe_sha256,
        "recipe": recipe,
        "variantsPerParent": SYNTHETIC_ANOMALY_VARIANTS_PER_PARENT,
        "generation": _generation_provenance(repository_root=repository_root),
        "records": records,
    }
    document["augmentationManifestSha256"] = _document_digest(document, "augmentationManifestSha256")
    manifest_path = output_dir / "augmentation_manifest.json"
    try:
        with manifest_path.open("xb") as stream:
            stream.write((json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    except OSError as error:
        raise SyntheticAnomalyAugmentationError("unable to write synthetic-only augmentation manifest") from error
    return document


def _validate_manifest_document(
    document: dict[str, Any],
    *,
    manifest_file_sha256: str,
    output_root: Path,
    envelope: dict[str, Any],
    envelope_file_sha256: str,
    parents: list[dict[str, Any]],
    recipe: dict[str, Any],
    recipe_sha256: str,
    repository_root: Path,
) -> list[dict[str, Any]]:
    _require_exact_fields(document, name="synthetic-only augmentation manifest", required=MANIFEST_FIELDS)
    if document.get("schemaVersion") != SYNTHETIC_ANOMALY_AUGMENTATION_V1_SCHEMA:
        raise SyntheticAnomalyAugmentationError("synthetic-only augmentation manifest schema is unsupported")
    scope = {
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "purpose": SYNTHETIC_ANOMALY_AUGMENTATION_V1_PURPOSE,
        "inputPolicy": SYNTHETIC_ANOMALY_INPUT_POLICY,
        "blindPolicy": SYNTHETIC_ANOMALY_BLIND_POLICY,
        "resultLabel": SYNTHETIC_ANOMALY_RESULT_LABEL,
        "parentPartition": "FIT",
    }
    if any(document.get(name) != expected for name, expected in scope.items()):
        raise SyntheticAnomalyAugmentationError("synthetic-only augmentation manifest scope is unsafe")
    if document.get("parentSplitAlgorithm") != SYNTHETIC_ANOMALY_PARENT_SPLIT_ALGORITHM:
        raise SyntheticAnomalyAugmentationError("synthetic-only parent split algorithm is unsupported")
    if document.get("parentSplitCountsPerCategory") != SYNTHETIC_ANOMALY_PARENT_SPLIT_COUNTS_PER_CATEGORY:
        raise SyntheticAnomalyAugmentationError("synthetic-only parent split counts are unsupported")
    split_parents = split_synthetic_anomaly_fit_parents(parents)
    query_parents = [parent for parent in split_parents if parent["syntheticTestRole"] == "QUERY"]
    if _require_sha256(
        document.get("syntheticQueryParentIdentitySha256"),
        name="synthetic-only manifest syntheticQueryParentIdentitySha256",
    ) != _query_parent_identity(split_parents):
        raise SyntheticAnomalyAugmentationError("synthetic-only query parent identity does not match the closed split")
    if document.get("augmentationManifestSha256") != _document_digest(document, "augmentationManifestSha256"):
        raise SyntheticAnomalyAugmentationError("synthetic-only augmentation manifest digest does not match")
    _require_sha256(manifest_file_sha256, name="synthetic-only augmentation manifest file digest")
    expected_bindings = _chain_bindings(envelope, envelope_file_sha256=envelope_file_sha256)
    expected_bindings["recipeFileSha256"] = recipe_sha256
    for name, expected in expected_bindings.items():
        if _require_sha256(document.get(name), name=f"synthetic-only manifest {name}") != expected:
            raise SyntheticAnomalyAugmentationError(f"synthetic-only manifest {name} does not match the closed chain")
    if document.get("recipe") != recipe:
        raise SyntheticAnomalyAugmentationError("synthetic-only embedded recipe does not match the supplied recipe")
    if _require_positive_int(document.get("variantsPerParent"), name="synthetic-only variantsPerParent") != SYNTHETIC_ANOMALY_VARIANTS_PER_PARENT:
        raise SyntheticAnomalyAugmentationError("synthetic-only variantsPerParent is unsupported")
    _validate_generation_provenance(document.get("generation"), repository_root=repository_root)
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(query_parents) * SYNTHETIC_ANOMALY_VARIANTS_PER_PARENT:
        raise SyntheticAnomalyAugmentationError("synthetic-only augmentation record count is inconsistent")
    parent_by_case = {str(parent["caseId"]): parent for parent in query_parents}
    expected_case_ids = {
        _expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=int(variant["variantId"]))[0]
        for parent in query_parents
        for variant in SYNTHETIC_ANOMALY_VARIANTS
    }
    seen_case_ids: set[str] = set()
    seen_relative_paths: set[str] = set()
    parent_images: dict[str, Image.Image] = {}
    validated: list[dict[str, Any]] = []
    for raw_record in raw_records:
        record = _require_exact_fields(raw_record, name="synthetic-only augmentation record", required=RECORD_FIELDS)
        case_id = _require_string(record.get("caseId"), name="synthetic-only record caseId")
        if case_id in seen_case_ids:
            raise SyntheticAnomalyAugmentationError("synthetic-only augmentation caseId is duplicated")
        seen_case_ids.add(case_id)
        parent_case_id = _require_string(record.get("parentCaseId"), name="synthetic-only record parentCaseId")
        parent = parent_by_case.get(parent_case_id)
        if parent is None:
            raise SyntheticAnomalyAugmentationError("synthetic-only augmentation parent case is unknown")
        if _require_sha256(record.get("parentSourceSha256"), name="synthetic-only record parentSourceSha256") != parent["sourceSha256"]:
            raise SyntheticAnomalyAugmentationError("synthetic-only parent source digest does not match")
        if record.get("sourceGroupId") != parent["sourceGroupId"] or record.get("category") != parent["category"]:
            raise SyntheticAnomalyAugmentationError("synthetic-only parent group or category does not match")
        if (
            record.get("parentPartition") != "FIT"
            or record.get("syntheticTestRole") != "QUERY"
            or record.get("syntheticLabel") != "SYNTHETIC_ANOMALY"
        ):
            raise SyntheticAnomalyAugmentationError("synthetic-only augmentation record scope is unsafe")
        variant_id = _require_positive_int(record.get("variantId"), name="synthetic-only record variantId")
        family = str(_variant_for_id(variant_id)["syntheticDefectFamily"])
        if record.get("syntheticDefectFamily") != family:
            raise SyntheticAnomalyAugmentationError("synthetic-only record family does not match variantId")
        expected_case_id, expected_relative_path = _expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=variant_id)
        if case_id != expected_case_id:
            raise SyntheticAnomalyAugmentationError("synthetic-only caseId does not match parent and family")
        relative_path = _safe_relative_path(record.get("relativePath"), name="synthetic-only record relativePath")
        if relative_path != expected_relative_path:
            raise SyntheticAnomalyAugmentationError("synthetic-only relativePath does not match parent and family")
        if relative_path.as_posix() in seen_relative_paths:
            raise SyntheticAnomalyAugmentationError("synthetic-only augmentation relativePath is duplicated")
        seen_relative_paths.add(relative_path.as_posix())
        parameters = _validate_parameters(record.get("parameters"), synthetic_defect_family=family)
        expected_parameters = sample_synthetic_anomaly_parameters_v1(
            recipe,
            recipe_sha256=recipe_sha256,
            parent_case_id=parent_case_id,
            parent_source_sha256=str(parent["sourceSha256"]),
            variant_id=variant_id,
        )
        if parameters != expected_parameters:
            raise SyntheticAnomalyAugmentationError("synthetic-only parameters do not match recipe and parent")
        if record.get("outputEncoding") != SYNTHETIC_ANOMALY_OUTPUT_ENCODING:
            raise SyntheticAnomalyAugmentationError("synthetic-only output encoding does not match the closed recipe")
        output_path = _safe_file_under(
            output_root,
            relative_path,
            description="synthetic-only augmentation output image",
            repository_root=repository_root,
        )
        try:
            actual_bytes = output_path.read_bytes()
        except OSError as error:
            raise SyntheticAnomalyAugmentationError("unable to read synthetic-only augmentation output image") from error
        if _sha256_bytes(actual_bytes) != _require_sha256(record.get("sourceSha256"), name="synthetic-only record sourceSha256"):
            raise SyntheticAnomalyAugmentationError("synthetic-only augmentation output digest does not match")
        _inspect_png_output(actual_bytes)
        source_image = parent_images.get(parent_case_id)
        if source_image is None:
            source_image = _load_parent_rgb(parent)
            parent_images[parent_case_id] = source_image
        expected_bytes = _render_augmented_png(source_image, expected_parameters, synthetic_defect_family=family)
        if actual_bytes != expected_bytes:
            raise SyntheticAnomalyAugmentationError("synthetic-only output pixels do not match the deterministic renderer")
        # ``imagePath`` is a post-validation convenience for offline test
        # runners.  It is deliberately not serialized in the immutable
        # manifest record schema.
        validated.append({**record, "imagePath": output_path})
    if [record["caseId"] for record in raw_records] != sorted(record["caseId"] for record in raw_records):
        raise SyntheticAnomalyAugmentationError("synthetic-only augmentation records must be sorted by caseId")
    if seen_case_ids != expected_case_ids:
        raise SyntheticAnomalyAugmentationError("synthetic-only records do not cover every FIT parent and family")
    return validated


def load_validated_synthetic_anomaly_augmentations(
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
    """Revalidate an external synthetic-only package and return its records.

    The returned tuple is ``(manifest, manifest_file_sha256, records)``.
    Validation reopens only the permitted FIT parents and their declared
    synthetic child files; it never accepts true anomaly or blind data.
    """

    _require_external_directory(
        augmentation_manifest_path.parent,
        description="synthetic-only augmentation package root",
        repository_root=repository_root,
    )
    document, manifest_file_sha256 = _read_external_json(
        augmentation_manifest_path,
        description="synthetic-only augmentation manifest",
        repository_root=repository_root,
    )
    envelope, envelope_file_sha256, parents = _load_synthetic_fit_parents(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        repository_root=repository_root,
    )
    recipe, recipe_sha256 = load_synthetic_anomaly_recipe_v1(recipe_path)
    records = _validate_manifest_document(
        document,
        manifest_file_sha256=manifest_file_sha256,
        output_root=augmentation_manifest_path.parent,
        envelope=envelope,
        envelope_file_sha256=envelope_file_sha256,
        parents=parents,
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        repository_root=repository_root,
    )
    return document, manifest_file_sha256, records
