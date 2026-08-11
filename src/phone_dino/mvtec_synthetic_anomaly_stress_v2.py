"""Post-V1 synthetic-only stress augmentation for offline MVTec tests.

The V1 synthetic package deliberately exercised one pattern per family.  This
separate V2 package expands that *engineering-only* fixture space while
preserving the same closed successor-V2 FIT-parent boundary and the exact
identity-only 6/2/4 parent split.  It does not contain observed defects and
must not be interpreted as a comparison, promotion, real-anomaly result, or
production evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps, __version__ as PILLOW_VERSION

from phone_dino import mvtec_normal_successor as successor
from phone_dino import mvtec_synthetic_anomaly_augmentation as v1


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA = "phone-dino.mvtec-ad-synthetic-only-stress-augmentation/2.0"
SYNTHETIC_ANOMALY_STRESS_RECIPE_V2_SCHEMA = "phone-dino.mvtec-ad-synthetic-only-stress-recipe/2.0"
SYNTHETIC_ANOMALY_STRESS_V2_PURPOSE = "OFFLINE_MVTEC_SYNTHETIC_ONLY_POST_V1_EXPLORATORY_STRESS_TESTING"
SYNTHETIC_ANOMALY_STRESS_RECIPE_V2_ID = "SYNTHETIC_ONLY_STRESS_LEVELS_V2_PNG"
SYNTHETIC_ANOMALY_STRESS_RECIPE_V2_DESCRIPTION = (
    "Closed post-V1 synthetic-only offline MVTec stress recipe. It renders deterministic subtle, moderate, and pronounced "
    "local scratch, spot, and occlusion stimuli on successor FIT nominal parents solely to exercise an engineering "
    "test harness. The stimuli are not observed defects, do not model physical failure mechanisms, have no "
    "real-anomaly ground truth, and must not be used for comparison, promotion, qualification, threshold selection, "
    "release decisions, or production authorization."
)
SYNTHETIC_ANOMALY_STRESS_INPUT_POLICY = "SUCCESSOR_V2_FIT_RAW_NORMAL_PARENTS_ONLY"
SYNTHETIC_ANOMALY_STRESS_BLIND_POLICY = "NO_BLIND_OR_TRUE_ANOMALY_DATA"
SYNTHETIC_ANOMALY_STRESS_RESULT_LABEL = "SYNTHETIC_ONLY_NOT_REAL_ANOMALY_PERFORMANCE"
SYNTHETIC_ANOMALY_STRESS_SAMPLING_ALGORITHM = "NAMED_SHA256_SYNTHETIC_STRESS_SUBSTREAMS_V2"
SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_ALGORITHM = v1.SYNTHETIC_ANOMALY_PARENT_SPLIT_ALGORITHM
SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_COUNTS_PER_CATEGORY = dict(v1.SYNTHETIC_ANOMALY_PARENT_SPLIT_COUNTS_PER_CATEGORY)
SYNTHETIC_ANOMALY_STRESS_FIT_PARENTS_PER_CATEGORY = sum(SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_COUNTS_PER_CATEGORY.values())
SYNTHETIC_ANOMALY_STRESS_OUTPUT_ENCODING = dict(v1.SYNTHETIC_ANOMALY_OUTPUT_ENCODING)
SYNTHETIC_ANOMALY_STRESS_CATEGORIES = ("capsule", "metal_nut", "tile")
SYNTHETIC_ANOMALY_STRESS_LEVELS = ("SUBTLE", "MODERATE", "PRONOUNCED")
SYNTHETIC_ANOMALY_STRESS_FAMILIES = ("LOCAL_SCRATCH", "LOCAL_SPOT", "LOCAL_OCCLUSION")

# These are procedural stimulus ranges, not estimates of any physical defect
# distribution.  The three locked levels deliberately make a test harder than
# V1 without allowing a later caller to cherry-pick severity after scoring.
SYNTHETIC_ANOMALY_STRESS_PROFILES = {
    "SUBTLE": {
        "LOCAL_SCRATCH": {
            "centerMarginFraction": 0.22,
            "minLengthFraction": 0.080,
            "maxLengthFraction": 0.160,
            "minWidthFraction": 0.004,
            "maxWidthFraction": 0.011,
            "minOpacity": 0.20,
            "maxOpacity": 0.46,
        },
        "LOCAL_SPOT": {
            "centerMarginFraction": 0.22,
            "minRadiusFraction": 0.030,
            "maxRadiusFraction": 0.080,
            "minSoftnessFraction": 0.010,
            "maxSoftnessFraction": 0.045,
            "minOpacity": 0.16,
            "maxOpacity": 0.44,
        },
        "LOCAL_OCCLUSION": {
            "centerMarginFraction": 0.22,
            "minWidthFraction": 0.060,
            "maxWidthFraction": 0.135,
            "minHeightFraction": 0.050,
            "maxHeightFraction": 0.115,
            "minCornerRadiusFraction": 0.020,
            "maxCornerRadiusFraction": 0.070,
            "minOpacity": 0.24,
            "maxOpacity": 0.50,
        },
    },
    "MODERATE": {
        "LOCAL_SCRATCH": {
            "centerMarginFraction": 0.22,
            "minLengthFraction": 0.120,
            "maxLengthFraction": 0.270,
            "minWidthFraction": 0.007,
            "maxWidthFraction": 0.022,
            "minOpacity": 0.45,
            "maxOpacity": 0.75,
        },
        "LOCAL_SPOT": {
            "centerMarginFraction": 0.22,
            "minRadiusFraction": 0.055,
            "maxRadiusFraction": 0.140,
            "minSoftnessFraction": 0.010,
            "maxSoftnessFraction": 0.050,
            "minOpacity": 0.38,
            "maxOpacity": 0.72,
        },
        "LOCAL_OCCLUSION": {
            "centerMarginFraction": 0.22,
            "minWidthFraction": 0.100,
            "maxWidthFraction": 0.220,
            "minHeightFraction": 0.080,
            "maxHeightFraction": 0.180,
            "minCornerRadiusFraction": 0.025,
            "maxCornerRadiusFraction": 0.080,
            "minOpacity": 0.46,
            "maxOpacity": 0.76,
        },
    },
    "PRONOUNCED": {
        "LOCAL_SCRATCH": {
            "centerMarginFraction": 0.22,
            "minLengthFraction": 0.180,
            "maxLengthFraction": 0.420,
            "minWidthFraction": 0.012,
            "maxWidthFraction": 0.035,
            "minOpacity": 0.75,
            "maxOpacity": 0.98,
        },
        "LOCAL_SPOT": {
            "centerMarginFraction": 0.22,
            "minRadiusFraction": 0.090,
            "maxRadiusFraction": 0.200,
            "minSoftnessFraction": 0.010,
            "maxSoftnessFraction": 0.055,
            "minOpacity": 0.55,
            "maxOpacity": 0.92,
        },
        "LOCAL_OCCLUSION": {
            "centerMarginFraction": 0.22,
            "minWidthFraction": 0.140,
            "maxWidthFraction": 0.300,
            "minHeightFraction": 0.120,
            "maxHeightFraction": 0.240,
            "minCornerRadiusFraction": 0.030,
            "maxCornerRadiusFraction": 0.090,
            "minOpacity": 0.76,
            "maxOpacity": 0.96,
        },
    },
}
SYNTHETIC_ANOMALY_STRESS_VARIANTS = (
    {"variantId": 1, "renderIntensityLevel": "SUBTLE", "syntheticDefectFamily": "LOCAL_SCRATCH"},
    {"variantId": 2, "renderIntensityLevel": "SUBTLE", "syntheticDefectFamily": "LOCAL_SPOT"},
    {"variantId": 3, "renderIntensityLevel": "SUBTLE", "syntheticDefectFamily": "LOCAL_OCCLUSION"},
    {"variantId": 4, "renderIntensityLevel": "MODERATE", "syntheticDefectFamily": "LOCAL_SCRATCH"},
    {"variantId": 5, "renderIntensityLevel": "MODERATE", "syntheticDefectFamily": "LOCAL_SPOT"},
    {"variantId": 6, "renderIntensityLevel": "MODERATE", "syntheticDefectFamily": "LOCAL_OCCLUSION"},
    {"variantId": 7, "renderIntensityLevel": "PRONOUNCED", "syntheticDefectFamily": "LOCAL_SCRATCH"},
    {"variantId": 8, "renderIntensityLevel": "PRONOUNCED", "syntheticDefectFamily": "LOCAL_SPOT"},
    {"variantId": 9, "renderIntensityLevel": "PRONOUNCED", "syntheticDefectFamily": "LOCAL_OCCLUSION"},
)
SYNTHETIC_ANOMALY_STRESS_VARIANTS_PER_PARENT = len(SYNTHETIC_ANOMALY_STRESS_VARIANTS)

RECIPE_FIELDS = {
    "schemaVersion",
    "id",
    "description",
    "authoritative",
    "productionAuthorized",
    "syntheticOnly",
    "purpose",
    "postV1Exploratory",
    "comparisonOrPromotionAllowed",
    "samplingAlgorithm",
    "stimulusProfiles",
    "outputEncoding",
}
MANIFEST_FIELDS = {
    "schemaVersion",
    "authoritative",
    "productionAuthorized",
    "syntheticOnly",
    "purpose",
    "postV1Exploratory",
    "comparisonOrPromotionAllowed",
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
    "v1SyntheticAugmentationModuleSha256",
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
    "renderIntensityLevel",
    "syntheticDefectFamily",
    "variantId",
    "relativePath",
    "sourceSha256",
    "parameters",
    "outputEncoding",
}


class SyntheticAnomalyStressV2Error(ValueError):
    """Raised when a post-V1 synthetic stress artifact is unsafe."""


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without retaining a complete file in memory."""

    return v1.sha256_file(path)


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _document_digest(document: dict[str, Any], field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return successor.canonical_json_sha256(unsigned)


def _fd_signature(fd: int) -> tuple[int, int, int, int]:
    """Return the immutable identity fields for an already-open manifest."""

    status = os.fstat(fd)
    return status.st_dev, status.st_ino, status.st_mode, status.st_size


def _path_signature(path: Path) -> tuple[int, int, int, int]:
    """Return a no-follow identity signature for the manifested path."""

    status = path.stat(follow_symlinks=False)
    return status.st_dev, status.st_ino, status.st_mode, status.st_size


def _write_new_external_manifest(
    path: Path,
    document: dict[str, Any],
    *,
    repository_root: Path,
) -> None:
    """Create a manifest once and prove its final path still names our file.

    The output directory is already new-only, but the final manifest is the
    package's root of trust.  Re-check its parent after image creation, create
    it with ``O_EXCL``, fsync its bytes, then compare the open handle against
    the no-follow path after a second reparse-point inspection.  A concurrent
    rename, link substitution, or replacement cannot be reported as success.
    """

    try:
        v1._require_external_directory(
            path.parent,
            description="synthetic-only stress augmentation output",
            repository_root=repository_root,
        )
        v1._reject_links_on_existing_path(path.parent, description="synthetic-only stress manifest output")
        if path.exists() or path.is_symlink():
            raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation manifest already exists")
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error

    data = (json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation manifest already exists") from error
    except OSError as error:
        raise SyntheticAnomalyStressV2Error("unable to exclusively create synthetic-only stress augmentation manifest") from error
    try:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("unable to write synthetic-only stress augmentation manifest")
            offset += written
        os.fsync(fd)
        fd_identity = _fd_signature(fd)
        try:
            v1._require_external_directory(
                path.parent,
                description="synthetic-only stress augmentation output",
                repository_root=repository_root,
            )
            v1._reject_links_on_existing_path(path.parent, description="synthetic-only stress manifest output")
        except v1.SyntheticAnomalyAugmentationError as error:
            raise _raise_v1(error) from error
        if v1._is_link_or_reparse_point(path):
            raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation manifest became a link or reparse point")
        if _path_signature(path) != fd_identity:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation manifest changed while it was written")
    except SyntheticAnomalyStressV2Error:
        raise
    except OSError as error:
        raise SyntheticAnomalyStressV2Error("unable to verify synthetic-only stress augmentation manifest") from error
    finally:
        os.close(fd)


def _preflight_external_output_directory(output_dir: Path, *, repository_root: Path) -> None:
    """Reject an unsafe or occupied output slot before opening any FIT bytes.

    This deliberately performs no write.  After the FIT-only safe loader has
    returned and the fixed identity split is established, generation calls the
    V1 hardened new-directory creator, which repeats this check immediately
    before creating the directory.
    """

    try:
        v1._reject_links_on_existing_path(output_dir, description="synthetic-only stress augmentation output")
        if v1._is_under(repository_root, output_dir) or v1._is_under(output_dir, repository_root):
            raise SyntheticAnomalyStressV2Error(
                "synthetic-only stress augmentation output must stay outside the Git working tree"
            )
        if output_dir.exists() or output_dir.is_symlink():
            raise SyntheticAnomalyStressV2Error(
                "synthetic-only stress augmentation output already exists; choose a new immutable path"
            )
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def _raise_v1(error: Exception) -> SyntheticAnomalyStressV2Error:
    return SyntheticAnomalyStressV2Error(str(error))


def _require_exact_fields(document: object, *, name: str, required: set[str]) -> dict[str, Any]:
    try:
        return v1._require_exact_fields(document, name=name, required=required)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    try:
        return v1._require_mapping(value, name=name)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def _require_string(value: object, *, name: str) -> str:
    try:
        return v1._require_string(value, name=name)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def _require_sha256(value: object, *, name: str) -> str:
    try:
        return v1._require_sha256(value, name=name)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def _require_positive_int(value: object, *, name: str, maximum: int | None = None) -> int:
    try:
        return v1._require_positive_int(value, name=name, maximum=maximum)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def _require_finite_number(value: object, *, name: str) -> float:
    try:
        return v1._require_finite_number(value, name=name)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def _read_recipe_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        return v1._read_recipe_json(path)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def _expect_exact_mapping(value: object, *, name: str, expected: dict[str, Any]) -> dict[str, Any]:
    mapping = _require_exact_fields(value, name=name, required=set(expected))
    if mapping != expected:
        raise SyntheticAnomalyStressV2Error(f"{name} does not match the closed synthetic-only stress recipe")
    return mapping


def load_synthetic_anomaly_stress_recipe_v2(recipe_path: Path) -> tuple[dict[str, Any], str]:
    """Load the one closed post-V1 exploratory stress recipe."""

    recipe, file_sha256 = _read_recipe_json(recipe_path)
    _require_exact_fields(recipe, name="synthetic-only stress recipe", required=RECIPE_FIELDS)
    if recipe.get("schemaVersion") != SYNTHETIC_ANOMALY_STRESS_RECIPE_V2_SCHEMA:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress recipe schema is unsupported")
    if recipe.get("id") != SYNTHETIC_ANOMALY_STRESS_RECIPE_V2_ID:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress recipe id is unsupported")
    if recipe.get("description") != SYNTHETIC_ANOMALY_STRESS_RECIPE_V2_DESCRIPTION:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress recipe description is not locked")
    if recipe.get("authoritative") is not False or recipe.get("productionAuthorized") is not False:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress recipe must be non-authoritative and non-production")
    if recipe.get("syntheticOnly") is not True or recipe.get("purpose") != SYNTHETIC_ANOMALY_STRESS_V2_PURPOSE:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress recipe scope is unsafe")
    if recipe.get("postV1Exploratory") is not True or recipe.get("comparisonOrPromotionAllowed") is not False:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress recipe cannot support comparison or promotion")
    if recipe.get("samplingAlgorithm") != SYNTHETIC_ANOMALY_STRESS_SAMPLING_ALGORITHM:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress recipe sampling algorithm is unsupported")
    _expect_exact_mapping(
        recipe.get("stimulusProfiles"),
        name="synthetic-only stress recipe stimulusProfiles",
        expected=SYNTHETIC_ANOMALY_STRESS_PROFILES,
    )
    _expect_exact_mapping(
        recipe.get("outputEncoding"),
        name="synthetic-only stress recipe outputEncoding",
        expected=SYNTHETIC_ANOMALY_STRESS_OUTPUT_ENCODING,
    )
    return recipe, file_sha256


def _variant_for_id(variant_id: int) -> dict[str, Any]:
    for variant in SYNTHETIC_ANOMALY_STRESS_VARIANTS:
        if variant["variantId"] == variant_id:
            return variant
    raise SyntheticAnomalyStressV2Error("synthetic-only stress variantId is unsupported")


def _rounded(value: float) -> float:
    return round(value, 8)


def _derive_seed(
    recipe_sha256: str,
    parent_case_id: str,
    parent_source_sha256: str,
    variant_id: int,
    render_intensity_level: str,
    synthetic_defect_family: str,
) -> int:
    expected = _variant_for_id(variant_id)
    if expected["renderIntensityLevel"] != render_intensity_level or expected["syntheticDefectFamily"] != synthetic_defect_family:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress level or family does not match variantId")
    payload = "\0".join((
        SYNTHETIC_ANOMALY_STRESS_SAMPLING_ALGORITHM,
        recipe_sha256,
        parent_case_id,
        parent_source_sha256,
        str(variant_id),
        render_intensity_level,
        synthetic_defect_family,
    )).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _sample_center(rng: random.Random, *, profile: dict[str, Any]) -> tuple[float, float]:
    margin = float(profile["centerMarginFraction"])
    return _rounded(rng.uniform(margin, 1.0 - margin)), _rounded(rng.uniform(margin, 1.0 - margin))


def _sample_gray(rng: random.Random) -> list[int]:
    # A neutral overlay avoids an implicit claim about the colour of a defect.
    tone = rng.randint(12, 70) if rng.randrange(2) == 0 else rng.randint(185, 243)
    return [tone, tone, tone]


def sample_synthetic_anomaly_stress_parameters_v2(
    recipe: dict[str, Any],
    *,
    recipe_sha256: str,
    parent_case_id: str,
    parent_source_sha256: str,
    variant_id: int,
) -> dict[str, Any]:
    """Derive one deterministic severity-and-family stress specification."""

    if recipe.get("samplingAlgorithm") != SYNTHETIC_ANOMALY_STRESS_SAMPLING_ALGORITHM:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress recipe sampling algorithm is unsupported")
    variant = _variant_for_id(variant_id)
    level = str(variant["renderIntensityLevel"])
    family = str(variant["syntheticDefectFamily"])
    profile = SYNTHETIC_ANOMALY_STRESS_PROFILES[level][family]
    seed = _derive_seed(recipe_sha256, parent_case_id, parent_source_sha256, variant_id, level, family)
    rng = random.Random(seed)
    center_x, center_y = _sample_center(rng, profile=profile)
    common = {"syntheticDefectFamily": family, "renderIntensityLevel": level, "seed": str(seed)}
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
    raise SyntheticAnomalyStressV2Error("synthetic-only stress family is unsupported")


def _validate_color(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 3 or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise SyntheticAnomalyStressV2Error(f"{name} must be three integer RGB components")
    if any(item < 0 or item > 255 for item in value):
        raise SyntheticAnomalyStressV2Error(f"{name} RGB components are out of range")
    return list(value)


def _validate_parameter_mapping(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    mapping = _require_exact_fields(value, name=name, required=fields)
    for key, item in mapping.items():
        if key == "colorRgb":
            _validate_color(item, name=f"{name} colorRgb")
        else:
            _require_finite_number(item, name=f"{name} {key}")
    return mapping


def _validate_parameters(
    value: object,
    *,
    render_intensity_level: str,
    synthetic_defect_family: str,
) -> dict[str, Any]:
    if render_intensity_level not in SYNTHETIC_ANOMALY_STRESS_LEVELS:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress level is unsupported")
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
        raise SyntheticAnomalyStressV2Error("synthetic-only stress family is unsupported")
    child_name, child_fields = expected
    mapping = _require_exact_fields(
        value,
        name="synthetic-only stress parameters",
        required={"syntheticDefectFamily", "renderIntensityLevel", "seed", child_name},
    )
    if mapping.get("syntheticDefectFamily") != synthetic_defect_family or mapping.get("renderIntensityLevel") != render_intensity_level:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress parameter family or level is inconsistent")
    seed = _require_string(mapping.get("seed"), name="synthetic-only stress parameter seed")
    if not seed.isdigit():
        raise SyntheticAnomalyStressV2Error("synthetic-only stress parameter seed must be a decimal integer")
    _validate_parameter_mapping(mapping.get(child_name), name=f"synthetic-only stress {child_name} parameters", fields=child_fields)
    return mapping


def _pixel_coordinate(fraction: float, length: int) -> float:
    return max(0.0, min(float(length - 1), fraction * float(length - 1)))


def _alpha(value: float) -> int:
    return max(0, min(255, int(round(value * 255.0))))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def apply_synthetic_anomaly_stress_v2(
    image: Image.Image,
    parameters: dict[str, Any],
    *,
    render_intensity_level: str,
    synthetic_defect_family: str,
) -> Image.Image:
    """Render one declared stress stimulus as RGB without physical interpretation."""

    _validate_parameters(
        parameters,
        render_intensity_level=render_intensity_level,
        synthetic_defect_family=synthetic_defect_family,
    )
    if image.width < 8 or image.height < 8:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress parent image dimensions must be at least 8 by 8")
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
            fill=(*_validate_color(item["colorRgb"], name="synthetic-only stress scratch color"), _alpha(float(item["opacity"]))),
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
            fill=(*_validate_color(item["colorRgb"], name="synthetic-only stress spot color"), _alpha(float(item["opacity"]))),
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
            fill=(*_validate_color(item["colorRgb"], name="synthetic-only stress occlusion color"), _alpha(float(item["opacity"]))),
        )
    else:  # The parameter validator already rejects this; retain fail-closed behavior.
        raise SyntheticAnomalyStressV2Error("synthetic-only stress family is unsupported")
    rendered = Image.alpha_composite(base, layer).convert("RGB")
    if rendered.tobytes() == image.convert("RGB").tobytes():
        raise SyntheticAnomalyStressV2Error("synthetic-only stress renderer did not alter the parent pixels")
    return rendered


def _inspect_png_output(data: bytes) -> dict[str, Any]:
    try:
        with Image.open(BytesIO(data)) as opened:
            opened.load()
            result = {"format": opened.format, "mode": opened.mode}
    except (OSError, SyntaxError, ValueError) as error:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress output is not a decodable PNG") from error
    if result != {"format": "PNG", "mode": "RGB"}:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress output encoding is unsupported")
    return result


def _render_augmented_png(
    image: Image.Image,
    parameters: dict[str, Any],
    *,
    render_intensity_level: str,
    synthetic_defect_family: str,
) -> bytes:
    augmented = apply_synthetic_anomaly_stress_v2(
        image,
        parameters,
        render_intensity_level=render_intensity_level,
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
        raise SyntheticAnomalyStressV2Error("synthetic-only stress parent has no validated image path")
    if sha256_file(source_path) != parent.get("sourceSha256"):
        raise SyntheticAnomalyStressV2Error("synthetic-only stress parent changed before augmentation")
    try:
        with Image.open(source_path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, SyntaxError, ValueError) as error:
        raise SyntheticAnomalyStressV2Error("unable to decode synthetic-only stress FIT parent image") from error
    if sha256_file(source_path) != parent.get("sourceSha256"):
        raise SyntheticAnomalyStressV2Error("synthetic-only stress parent changed while it was decoded")
    return image


def _expected_child_identity(
    parent: dict[str, Any],
    *,
    recipe_sha256: str,
    variant_id: int,
) -> tuple[str, PurePosixPath]:
    variant = _variant_for_id(variant_id)
    level = str(variant["renderIntensityLevel"])
    family = str(variant["syntheticDefectFamily"])
    case_id = f"{parent['caseId']}/synthetic-only-stress-v2/{level.lower()}/{family.lower()}"
    name = hashlib.sha256(
        "\0".join((
            SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA,
            recipe_sha256,
            str(parent["caseId"]),
            str(parent["sourceSha256"]),
            str(variant_id),
            level,
            family,
        )).encode("utf-8")
    ).hexdigest()
    return case_id, PurePosixPath("images") / f"{name}-{level.lower()}-{family.lower()}.png"


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
            repository_root / "tools" / "generate_mvtec_ad_synthetic_anomaly_stress_v2.py"
        ),
        "normalSuccessorModuleSha256": sha256_file(repository_root / "src" / "phone_dino" / "mvtec_normal_successor.py"),
        "v1SyntheticAugmentationModuleSha256": sha256_file(
            repository_root / "src" / "phone_dino" / "mvtec_synthetic_anomaly_augmentation.py"
        ),
        "gitRevision": revision,
        "gitWorktreeClean": worktree_clean,
        "python": sys.version,
        "platform": platform.platform(),
        "pillowVersion": PILLOW_VERSION,
    }


def _validate_generation_provenance(value: object, *, repository_root: Path) -> dict[str, Any]:
    generation = _require_exact_fields(value, name="synthetic-only stress augmentation generation", required=GENERATION_FIELDS)
    expected_hashes = {
        "generatorModuleSha256": sha256_file(Path(__file__)),
        "generatorEntrypointSha256": sha256_file(
            repository_root / "tools" / "generate_mvtec_ad_synthetic_anomaly_stress_v2.py"
        ),
        "normalSuccessorModuleSha256": sha256_file(repository_root / "src" / "phone_dino" / "mvtec_normal_successor.py"),
        "v1SyntheticAugmentationModuleSha256": sha256_file(
            repository_root / "src" / "phone_dino" / "mvtec_synthetic_anomaly_augmentation.py"
        ),
    }
    for name, expected in expected_hashes.items():
        if _require_sha256(generation.get(name), name=f"synthetic-only stress augmentation {name}") != expected:
            raise SyntheticAnomalyStressV2Error(f"synthetic-only stress augmentation {name} does not match this implementation")
    for name in ("python", "platform", "pillowVersion"):
        _require_string(generation.get(name), name=f"synthetic-only stress augmentation {name}")
    if generation.get("gitRevision") is not None:
        _require_string(generation.get("gitRevision"), name="synthetic-only stress augmentation gitRevision")
    if generation.get("gitWorktreeClean") is not None and not isinstance(generation.get("gitWorktreeClean"), bool):
        raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation gitWorktreeClean must be a boolean or null")
    return generation


def split_synthetic_anomaly_stress_fit_parents(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose V1's closed identity-only 6/2/4 split for this V2 package.

    The wrapper intentionally delegates to the already exercised split rather
    than copying or tuning allocation logic.  It cannot depend on rendered
    pixels, a synthetic label, a score, or the requested stress level.
    """

    try:
        return v1.split_synthetic_anomaly_fit_parents(parents)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def _closed_stress_split(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require the fixed three-category allocation that makes 108 children."""

    split = split_synthetic_anomaly_stress_fit_parents(parents)
    actual_categories = tuple(sorted({str(parent["category"]) for parent in split}))
    expected_categories = tuple(sorted(SYNTHETIC_ANOMALY_STRESS_CATEGORIES))
    if actual_categories != expected_categories:
        raise SyntheticAnomalyStressV2Error(
            "synthetic-only stress requires exactly the fixed capsule, metal_nut, and tile FIT categories"
        )
    query_parents = [parent for parent in split if parent["syntheticTestRole"] == "QUERY"]
    if len(query_parents) != len(SYNTHETIC_ANOMALY_STRESS_CATEGORIES) * SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_COUNTS_PER_CATEGORY["QUERY"]:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress requires exactly twelve FIT query parents")
    return split


def _query_parent_identity(parents: list[dict[str, Any]]) -> str:
    try:
        return v1._query_parent_identity(parents)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def _load_synthetic_stress_fit_parents(
    parent_holdout_path: Path,
    parent_selection_contract_path: Path,
    plan_path: Path,
    envelope_path: Path,
    *,
    source_root: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Request only the successor-V2 FIT image partition through its safe loader."""

    for path, description in (
        (parent_holdout_path, "synthetic-only stress parent holdout manifest"),
        (parent_selection_contract_path, "synthetic-only stress parent selection contract"),
        (plan_path, "synthetic-only stress allocation plan"),
        (envelope_path, "synthetic-only stress phase envelope"),
    ):
        try:
            v1._require_external_input_file(path, description=description, repository_root=repository_root)
        except v1.SyntheticAnomalyAugmentationError as error:
            raise _raise_v1(error) from error
    try:
        v1._require_external_directory(source_root, description="synthetic-only stress source root", repository_root=repository_root)
        envelope, envelope_file_sha256, parents = successor.load_successor_safe_normal_inputs(
            parent_holdout_path,
            parent_selection_contract_path,
            plan_path,
            envelope_path,
            source_root=source_root,
            partitions={"FIT"},
            repository_root=repository_root,
        )
        return envelope, envelope_file_sha256, v1._validate_fit_parents(parents)
    except (v1.SyntheticAnomalyAugmentationError, successor.FreshNormalSuccessorError) as error:
        raise _raise_v1(error) from error


def _chain_bindings(envelope: dict[str, Any], *, envelope_file_sha256: str) -> dict[str, str]:
    try:
        return v1._chain_bindings(envelope, envelope_file_sha256=envelope_file_sha256)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error


def generate_synthetic_anomaly_stress_v2(
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
    """Materialize 108 closed, query-parent-only post-V1 stress stimuli.

    With three expected categories this means four query parents per category,
    three fixed levels, and three fixed families: ``12 * 3 * 3 == 108``.  The
    function has no argument capable of expanding the permitted input
    partition or selecting a level after a score is known.
    """

    _preflight_external_output_directory(output_dir, repository_root=repository_root)
    envelope, envelope_file_sha256, fit_parents = _load_synthetic_stress_fit_parents(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        repository_root=repository_root,
    )
    recipe, recipe_sha256 = load_synthetic_anomaly_stress_recipe_v2(recipe_path)
    parents = _closed_stress_split(fit_parents)
    query_parents = [parent for parent in parents if parent["syntheticTestRole"] == "QUERY"]
    try:
        v1._prepare_external_output_directory(output_dir, repository_root=repository_root)
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error
    records: list[dict[str, Any]] = []
    for parent in query_parents:
        source_image = _load_parent_rgb(parent)
        for variant in SYNTHETIC_ANOMALY_STRESS_VARIANTS:
            variant_id = int(variant["variantId"])
            level = str(variant["renderIntensityLevel"])
            family = str(variant["syntheticDefectFamily"])
            parameters = sample_synthetic_anomaly_stress_parameters_v2(
                recipe,
                recipe_sha256=recipe_sha256,
                parent_case_id=str(parent["caseId"]),
                parent_source_sha256=str(parent["sourceSha256"]),
                variant_id=variant_id,
            )
            case_id, relative_path = _expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=variant_id)
            if parameters["renderIntensityLevel"] != level or parameters["syntheticDefectFamily"] != family:
                raise SyntheticAnomalyStressV2Error("synthetic-only stress variant level or family is inconsistent")
            data = _render_augmented_png(
                source_image,
                parameters,
                render_intensity_level=level,
                synthetic_defect_family=family,
            )
            target_path = output_dir.joinpath(*relative_path.parts)
            try:
                v1._reject_links_on_existing_path(target_path.parent, description="synthetic-only stress output")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                v1._reject_links_on_existing_path(target_path.parent, description="synthetic-only stress output")
                with target_path.open("xb") as stream:
                    stream.write(data)
            except v1.SyntheticAnomalyAugmentationError as error:
                raise _raise_v1(error) from error
            except OSError as error:
                raise SyntheticAnomalyStressV2Error("unable to write synthetic-only stress augmentation image") from error
            records.append({
                "caseId": case_id,
                "parentCaseId": parent["caseId"],
                "parentSourceSha256": parent["sourceSha256"],
                "sourceGroupId": parent["sourceGroupId"],
                "category": parent["category"],
                "parentPartition": "FIT",
                "syntheticTestRole": "QUERY",
                "syntheticLabel": "SYNTHETIC_STIMULUS",
                "renderIntensityLevel": level,
                "syntheticDefectFamily": family,
                "variantId": variant_id,
                "relativePath": relative_path.as_posix(),
                "sourceSha256": _sha256_bytes(data),
                "parameters": parameters,
                "outputEncoding": dict(SYNTHETIC_ANOMALY_STRESS_OUTPUT_ENCODING),
            })
    records.sort(key=lambda record: str(record["caseId"]))
    document: dict[str, Any] = {
        "schemaVersion": SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "purpose": SYNTHETIC_ANOMALY_STRESS_V2_PURPOSE,
        "postV1Exploratory": True,
        "comparisonOrPromotionAllowed": False,
        "inputPolicy": SYNTHETIC_ANOMALY_STRESS_INPUT_POLICY,
        "blindPolicy": SYNTHETIC_ANOMALY_STRESS_BLIND_POLICY,
        "resultLabel": SYNTHETIC_ANOMALY_STRESS_RESULT_LABEL,
        "parentPartition": "FIT",
        "parentSplitAlgorithm": SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_ALGORITHM,
        "parentSplitCountsPerCategory": dict(SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_COUNTS_PER_CATEGORY),
        "syntheticQueryParentIdentitySha256": _query_parent_identity(parents),
        **_chain_bindings(envelope, envelope_file_sha256=envelope_file_sha256),
        "recipeFileSha256": recipe_sha256,
        "recipe": recipe,
        "variantsPerParent": SYNTHETIC_ANOMALY_STRESS_VARIANTS_PER_PARENT,
        "generation": _generation_provenance(repository_root=repository_root),
        "records": records,
    }
    document["augmentationManifestSha256"] = _document_digest(document, "augmentationManifestSha256")
    manifest_path = output_dir / "augmentation_manifest.json"
    _write_new_external_manifest(manifest_path, document, repository_root=repository_root)
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
    _require_exact_fields(document, name="synthetic-only stress augmentation manifest", required=MANIFEST_FIELDS)
    if document.get("schemaVersion") != SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation manifest schema is unsupported")
    scope = {
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "purpose": SYNTHETIC_ANOMALY_STRESS_V2_PURPOSE,
        "postV1Exploratory": True,
        "comparisonOrPromotionAllowed": False,
        "inputPolicy": SYNTHETIC_ANOMALY_STRESS_INPUT_POLICY,
        "blindPolicy": SYNTHETIC_ANOMALY_STRESS_BLIND_POLICY,
        "resultLabel": SYNTHETIC_ANOMALY_STRESS_RESULT_LABEL,
        "parentPartition": "FIT",
    }
    if any(document.get(name) != expected for name, expected in scope.items()):
        raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation manifest scope is unsafe")
    if document.get("parentSplitAlgorithm") != SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_ALGORITHM:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress parent split algorithm is unsupported")
    if document.get("parentSplitCountsPerCategory") != SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_COUNTS_PER_CATEGORY:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress parent split counts are unsupported")
    split_parents = _closed_stress_split(parents)
    query_parents = [parent for parent in split_parents if parent["syntheticTestRole"] == "QUERY"]
    if _require_sha256(
        document.get("syntheticQueryParentIdentitySha256"),
        name="synthetic-only stress manifest syntheticQueryParentIdentitySha256",
    ) != _query_parent_identity(split_parents):
        raise SyntheticAnomalyStressV2Error("synthetic-only stress query parent identity does not match the closed split")
    if document.get("augmentationManifestSha256") != _document_digest(document, "augmentationManifestSha256"):
        raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation manifest digest does not match")
    _require_sha256(manifest_file_sha256, name="synthetic-only stress augmentation manifest file digest")
    expected_bindings = _chain_bindings(envelope, envelope_file_sha256=envelope_file_sha256)
    expected_bindings["recipeFileSha256"] = recipe_sha256
    for name, expected in expected_bindings.items():
        if _require_sha256(document.get(name), name=f"synthetic-only stress manifest {name}") != expected:
            raise SyntheticAnomalyStressV2Error(f"synthetic-only stress manifest {name} does not match the closed chain")
    if document.get("recipe") != recipe:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress embedded recipe does not match the supplied recipe")
    if _require_positive_int(document.get("variantsPerParent"), name="synthetic-only stress variantsPerParent") != SYNTHETIC_ANOMALY_STRESS_VARIANTS_PER_PARENT:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress variantsPerParent is unsupported")
    _validate_generation_provenance(document.get("generation"), repository_root=repository_root)
    raw_records = document.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(query_parents) * SYNTHETIC_ANOMALY_STRESS_VARIANTS_PER_PARENT:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation record count is inconsistent")
    parent_by_case = {str(parent["caseId"]): parent for parent in query_parents}
    expected_case_ids = {
        _expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=int(variant["variantId"]))[0]
        for parent in query_parents
        for variant in SYNTHETIC_ANOMALY_STRESS_VARIANTS
    }
    seen_case_ids: set[str] = set()
    seen_relative_paths: set[str] = set()
    parent_images: dict[str, Image.Image] = {}
    validated: list[dict[str, Any]] = []
    for raw_record in raw_records:
        record = _require_exact_fields(raw_record, name="synthetic-only stress augmentation record", required=RECORD_FIELDS)
        case_id = _require_string(record.get("caseId"), name="synthetic-only stress record caseId")
        if case_id in seen_case_ids:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation caseId is duplicated")
        seen_case_ids.add(case_id)
        parent_case_id = _require_string(record.get("parentCaseId"), name="synthetic-only stress record parentCaseId")
        parent = parent_by_case.get(parent_case_id)
        if parent is None:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation parent case is unknown")
        if _require_sha256(record.get("parentSourceSha256"), name="synthetic-only stress record parentSourceSha256") != parent["sourceSha256"]:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress parent source digest does not match")
        if record.get("sourceGroupId") != parent["sourceGroupId"] or record.get("category") != parent["category"]:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress parent group or category does not match")
        if (
            record.get("parentPartition") != "FIT"
            or record.get("syntheticTestRole") != "QUERY"
            or record.get("syntheticLabel") != "SYNTHETIC_STIMULUS"
        ):
            raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation record scope is unsafe")
        variant_id = _require_positive_int(record.get("variantId"), name="synthetic-only stress record variantId")
        variant = _variant_for_id(variant_id)
        level = str(variant["renderIntensityLevel"])
        family = str(variant["syntheticDefectFamily"])
        if record.get("renderIntensityLevel") != level or record.get("syntheticDefectFamily") != family:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress record level or family does not match variantId")
        expected_case_id, expected_relative_path = _expected_child_identity(parent, recipe_sha256=recipe_sha256, variant_id=variant_id)
        if case_id != expected_case_id:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress caseId does not match parent, level, and family")
        try:
            relative_path = v1._safe_relative_path(record.get("relativePath"), name="synthetic-only stress record relativePath")
        except v1.SyntheticAnomalyAugmentationError as error:
            raise _raise_v1(error) from error
        if relative_path != expected_relative_path:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress relativePath does not match parent, level, and family")
        if relative_path.as_posix() in seen_relative_paths:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation relativePath is duplicated")
        seen_relative_paths.add(relative_path.as_posix())
        parameters = _validate_parameters(
            record.get("parameters"),
            render_intensity_level=level,
            synthetic_defect_family=family,
        )
        expected_parameters = sample_synthetic_anomaly_stress_parameters_v2(
            recipe,
            recipe_sha256=recipe_sha256,
            parent_case_id=parent_case_id,
            parent_source_sha256=str(parent["sourceSha256"]),
            variant_id=variant_id,
        )
        if parameters != expected_parameters:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress parameters do not match recipe and parent")
        if record.get("outputEncoding") != SYNTHETIC_ANOMALY_STRESS_OUTPUT_ENCODING:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress output encoding does not match the closed recipe")
        try:
            output_path = v1._safe_file_under(
                output_root,
                relative_path,
                description="synthetic-only stress augmentation output image",
                repository_root=repository_root,
            )
        except v1.SyntheticAnomalyAugmentationError as error:
            raise _raise_v1(error) from error
        try:
            actual_bytes = output_path.read_bytes()
        except OSError as error:
            raise SyntheticAnomalyStressV2Error("unable to read synthetic-only stress augmentation output image") from error
        if _sha256_bytes(actual_bytes) != _require_sha256(record.get("sourceSha256"), name="synthetic-only stress record sourceSha256"):
            raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation output digest does not match")
        _inspect_png_output(actual_bytes)
        source_image = parent_images.get(parent_case_id)
        if source_image is None:
            source_image = _load_parent_rgb(parent)
            parent_images[parent_case_id] = source_image
        expected_bytes = _render_augmented_png(
            source_image,
            expected_parameters,
            render_intensity_level=level,
            synthetic_defect_family=family,
        )
        if actual_bytes != expected_bytes:
            raise SyntheticAnomalyStressV2Error("synthetic-only stress output pixels do not match the deterministic renderer")
        # ``imagePath`` is a post-validation convenience and is never serialized.
        validated.append({**record, "imagePath": output_path})
    if [record["caseId"] for record in raw_records] != sorted(record["caseId"] for record in raw_records):
        raise SyntheticAnomalyStressV2Error("synthetic-only stress augmentation records must be sorted by caseId")
    if seen_case_ids != expected_case_ids:
        raise SyntheticAnomalyStressV2Error("synthetic-only stress records do not cover every query parent, level, and family")
    return validated


def load_validated_synthetic_stress_v2(
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
    """Revalidate an external V2 stress package and return ``imagePath`` records."""

    try:
        v1._require_external_directory(
            augmentation_manifest_path.parent,
            description="synthetic-only stress augmentation package root",
            repository_root=repository_root,
        )
        document, manifest_file_sha256 = v1._read_external_json(
            augmentation_manifest_path,
            description="synthetic-only stress augmentation manifest",
            repository_root=repository_root,
        )
    except v1.SyntheticAnomalyAugmentationError as error:
        raise _raise_v1(error) from error
    envelope, envelope_file_sha256, parents = _load_synthetic_stress_fit_parents(
        parent_holdout_path,
        parent_selection_contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        repository_root=repository_root,
    )
    recipe, recipe_sha256 = load_synthetic_anomaly_stress_recipe_v2(recipe_path)
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
