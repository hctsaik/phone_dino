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
import random
from pathlib import Path
from typing import Any

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CAMERA_AUGMENTATION_SCHEMA = "phone-dino.mvtec-ad-normal-augmentation/1.0"
CAMERA_RECIPE_SCHEMA = "phone-dino.mvtec-ad-camera-recipe/1.0"
NORMAL_AUGMENTATION_ROLES = frozenset({"FIT", "THRESHOLD_TUNING"})


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


def load_camera_recipe(recipe_path: Path) -> tuple[dict[str, Any], str]:
    """Load the bounded generic camera/lighting recipe used by this protocol."""

    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MvtecResearchError(f"Unable to read augmentation recipe: {recipe_path}") from error
    if not isinstance(recipe, dict) or recipe.get("schemaVersion") != CAMERA_RECIPE_SCHEMA:
        raise MvtecResearchError("unsupported camera augmentation recipe schema")
    if recipe.get("authoritative") is not False or recipe.get("productionAuthorized") is not False:
        raise MvtecResearchError("camera augmentation recipe must be explicitly non-authoritative and non-production")
    if recipe.get("blindPolicy") != "BLIND_ORIGINAL_ONLY":
        raise MvtecResearchError("camera augmentation recipe must keep blind inputs original")

    geometry = _require_mapping(recipe, "geometry")
    photometry = _require_mapping(recipe, "photometry")
    encoding = _require_mapping(recipe, "encoding")
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
    return recipe, sha256_file(recipe_path)


def derive_augmentation_seed(recipe_sha256: str, parent_case_id: str, parent_source_sha256: str, variant_id: int) -> int:
    if variant_id <= 0:
        raise MvtecResearchError("variant_id must be positive")
    payload = "\0".join((recipe_sha256, parent_case_id, parent_source_sha256, str(variant_id))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def sample_camera_parameters(recipe: dict[str, Any], *, recipe_sha256: str, parent_case_id: str,
                             parent_source_sha256: str, variant_id: int) -> dict[str, Any]:
    """Return stable generic phone-capture perturbations for one normal input."""

    seed = derive_augmentation_seed(recipe_sha256, parent_case_id, parent_source_sha256, variant_id)
    rng = random.Random(seed)
    geometry = _require_mapping(recipe, "geometry")
    photometry = _require_mapping(recipe, "photometry")
    encoding = _require_mapping(recipe, "encoding")

    def symmetric(section: dict[str, Any], name: str) -> float:
        return rng.uniform(-_require_finite_number(section, name), _require_finite_number(section, name))

    def rounded(value: float) -> float:
        return round(value, 8)

    return {
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


def validate_normal_augmentation_parent(record: dict[str, Any]) -> None:
    """Reject any input that could turn blind/anomalous data into development data."""

    role = _require_string(record, "role")
    kind = _require_string(record, "kind")
    if role not in NORMAL_AUGMENTATION_ROLES:
        raise MvtecResearchError("only FIT and THRESHOLD_TUNING records may be augmented")
    if kind != "NOMINAL":
        raise MvtecResearchError("only nominal records may be augmented")
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
    values *= (shading * vignette)[..., None]
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
            child_name = hashlib.sha256(
                f"{parent['caseId']}\0{parent['sourceSha256']}\0{variant_id}".encode("utf-8")
            ).hexdigest()
            relative_path = Path("images") / f"{child_name}-v{variant_id}.jpg"
            target_path = output_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            augmented = apply_camera_augmentation(source_image, parameters)
            augmented.save(target_path, format="JPEG", quality=int(parameters["jpegQuality"]), subsampling=0, optimize=False, progressive=False)
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
        "recipePath": str(recipe_path),
        "recipeSha256": recipe_sha256,
        "variantsPerParent": variants_per_parent,
        "records": output_records,
    }
    document["augmentationManifestSha256"] = _document_digest(document, "augmentationManifestSha256")
    (output_dir / "augmentation_manifest.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


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
    _require_sha256(document, "recipeSha256")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise MvtecResearchError("augmentation manifest has no records")
    parents = {str(record.get("caseId")): record for record in source_manifest["records"] if isinstance(record, dict)}
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
        if not isinstance(variant_id, int) or isinstance(variant_id, bool) or variant_id <= 0:
            raise MvtecResearchError("augmentation variantId must be positive")
        parameters = record.get("parameters")
        if not isinstance(parameters, dict):
            raise MvtecResearchError("augmentation parameters must be an object")
        validated.append(dict(record))
    return document, validated
