"""Replay a frozen external physical-alignment cohort without changing runtime state.

The cohort format deliberately binds the raw still, saved request/response,
artifact bytes, and the saved readiness response.  It is an engineering
evidence tool only: it never scores an image, changes a threshold, or makes a
qualification/disposition decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import hmac
import json
from pathlib import Path
import stat
from typing import Annotated, Any, Literal

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .artifacts import (
    ArtifactError,
    ProductionArtifactV14,
    ProductionArtifactV15,
    ProductionArtifactV16,
    ProductionArtifactV17,
    ProductionArtifactV18,
    ProductionArtifactV19,
    require_inspection_roi,
    require_subject_segmentation,
    verify_artifact_binding,
)
from .analyzer import RUNTIME_DIGEST
from .contracts import AnalyzeObservation, AnalyzeRequest, Identifier, PrefixedSha256
from .decoder import DecodedImage
from .production import OpenCvCharucoNormalizer
from .reference_board import ReferenceBoardVerifier
from .security import canonical_bundle_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COHORT_SCHEMA_VERSION = "phone-dino.physical-alignment-cohort/1.0"
REPORT_SCHEMA_VERSION = "phone-dino.physical-alignment-replay-report/1.0"
ARTIFACT_MODELS = {
    "1.4": ProductionArtifactV14,
    "1.5": ProductionArtifactV15,
    "1.6": ProductionArtifactV16,
    "1.7": ProductionArtifactV17,
    "1.8": ProductionArtifactV18,
    "1.9": ProductionArtifactV19,
}
MAX_COHORT_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_READYZ_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
OpaqueIdentifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")]


class PhysicalAlignmentReplayError(ValueError):
    """Raised when a physical-alignment replay input is not immutable evidence."""


class StrictPhysicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class EvidenceFile(StrictPhysicalModel):
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=1024)
    sha256: PrefixedSha256


class PinnedArtifact(StrictPhysicalModel):
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=1024)
    artifact_package_digest: PrefixedSha256 = Field(alias="artifactPackageDigest")
    schema_version: Literal["1.4", "1.5", "1.6", "1.7", "1.8", "1.9"] = Field(alias="schemaVersion")
    recipe_id: Identifier = Field(alias="recipeId")
    machine_id: Identifier = Field(alias="machineId")
    board_id: Identifier = Field(alias="boardId")


class CaptureStrata(StrictPhysicalModel):
    device: OpaqueIdentifier
    camera: OpaqueIdentifier
    lens: OpaqueIdentifier
    lighting: OpaqueIdentifier
    distance: OpaqueIdentifier
    view: OpaqueIdentifier


class ReplayPolicy(StrictPhysicalModel):
    allow_target_only_alignment: bool = Field(alias="allowTargetOnlyAlignment")
    allow_contour_anchor_alignment: bool = Field(alias="allowContourAnchorAlignment")


class CohortCase(StrictPhysicalModel):
    case_id: OpaqueIdentifier = Field(alias="caseId")
    partition: Literal["DEVELOPMENT", "HELD_OUT"]
    acquisition_group_id: OpaqueIdentifier = Field(alias="acquisitionGroupId")
    intended_alignment_state: Literal["ALIGNED", "NOT_ALIGNED"] = Field(alias="intendedAlignmentState")
    capture_strata: CaptureStrata = Field(alias="captureStrata")
    raw: EvidenceFile
    request: EvidenceFile
    response: EvidenceFile
    opaque_native_attestation_digest: PrefixedSha256 | None = Field(
        default=None, alias="opaqueNativeAttestationDigest",
    )


class ReadyzReplayProvenance(StrictPhysicalModel):
    artifact_package_digest: PrefixedSha256 = Field(alias="artifactPackageDigest")
    analyzer_runtime_version: PrefixedSha256 = Field(alias="analyzerRuntimeVersion")
    allow_target_only_alignment: bool = Field(alias="allowTargetOnlyAlignment")
    allow_contour_anchor_alignment: bool = Field(alias="allowContourAnchorAlignment")
    max_image_bytes: int = Field(alias="maxImageBytes", ge=1, le=1024 * 1024 * 1024)
    max_image_pixels: int = Field(alias="maxImagePixels", ge=1, le=100_000_000)
    max_image_width: int = Field(alias="maxImageWidth", ge=1, le=100_000)
    max_image_height: int = Field(alias="maxImageHeight", ge=1, le=100_000)


class PhysicalAlignmentCohort(StrictPhysicalModel):
    schema_version: Literal[COHORT_SCHEMA_VERSION] = Field(alias="schemaVersion")
    cohort_sha256: PrefixedSha256 = Field(alias="cohortSha256")
    authoritative: Literal[False]
    production_authorized: Literal[False] = Field(alias="productionAuthorized")
    purpose: Literal["PHYSICAL_ALIGNMENT_REPLAY_ONLY"]
    artifact: PinnedArtifact
    readyz_evidence: EvidenceFile = Field(alias="readyzEvidence")
    replay_policy: ReplayPolicy = Field(alias="replayPolicy")
    cases: list[CohortCase] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_membership(self) -> "PhysicalAlignmentCohort":
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("cohort caseId values must be unique")
        raw_digests = [case.raw.sha256 for case in self.cases]
        if len(set(raw_digests)) != len(raw_digests):
            raise ValueError("cohort raw stills must not be reused")
        partitions_by_group: dict[str, set[str]] = {}
        for case in self.cases:
            partitions_by_group.setdefault(case.acquisition_group_id, set()).add(case.partition)
        if any(len(partitions) != 1 for partitions in partitions_by_group.values()):
            raise ValueError("acquisition groups must not cross DEVELOPMENT and HELD_OUT")
        return self


@dataclass(frozen=True, slots=True)
class LoadedCohort:
    root: Path
    path: Path
    file_sha256: str
    model: PhysicalAlignmentCohort


@dataclass(frozen=True, slots=True)
class RuntimeDecodedCapture:
    rgb: Any
    output_encoding: dict[str, object]
    encoded_width: int
    encoded_height: int


def canonical_json_sha256(document: Any) -> str:
    try:
        encoded = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PhysicalAlignmentReplayError("document cannot be canonicalized") from error
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _is_under(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise PhysicalAlignmentReplayError(f"cannot stat immutable evidence path: {path}") from error
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(getattr(metadata, "st_file_attributes", 0) & reparse_attribute)


def _require_regular_file(path: Path, *, description: str) -> None:
    if not path.is_file() or _is_link_or_reparse_point(path):
        raise PhysicalAlignmentReplayError(f"{description} must be a regular non-link file: {path}")


def _require_non_link_directory(path: Path, *, description: str) -> None:
    if not path.is_dir() or _is_link_or_reparse_point(path):
        raise PhysicalAlignmentReplayError(f"{description} must be a non-link directory: {path}")


def _safe_relative_file(root: Path, relative_path: str, *, description: str) -> Path:
    relative = Path(relative_path)
    if (
        not relative_path
        or relative.is_absolute()
        or relative.drive
        or any(part in {"", ".", ".."} or ":" in part for part in relative.parts)
    ):
        raise PhysicalAlignmentReplayError(f"{description} must be a safe relative path")
    _require_non_link_directory(root, description="cohort root")
    candidate = root
    for index, part in enumerate(relative.parts):
        candidate = candidate / part
        if index < len(relative.parts) - 1:
            _require_non_link_directory(candidate, description=f"{description} parent")
    _require_regular_file(candidate, description=description)
    if not _is_under(root, candidate):
        raise PhysicalAlignmentReplayError(f"{description} escapes cohort root")
    return candidate


def _read_bytes(path: Path, *, description: str, maximum_bytes: int | None = None) -> bytes:
    _require_regular_file(path, description=description)
    try:
        if maximum_bytes is not None and path.stat(follow_symlinks=False).st_size > maximum_bytes:
            raise PhysicalAlignmentReplayError(f"{description} exceeds the configured byte limit")
        with path.open("rb") as handle:
            data = handle.read() if maximum_bytes is None else handle.read(maximum_bytes + 1)
    except OSError as error:
        raise PhysicalAlignmentReplayError(f"cannot read {description}: {path}") from error
    if maximum_bytes is not None and len(data) > maximum_bytes:
        raise PhysicalAlignmentReplayError(f"{description} exceeds the configured byte limit")
    _require_regular_file(path, description=description)
    return data


def _read_json_bytes(data: bytes, *, description: str) -> dict[str, Any]:
    try:
        document = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhysicalAlignmentReplayError(f"{description} must be UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise PhysicalAlignmentReplayError(f"{description} must be a JSON object")
    return document


def _read_bound_file(
    root: Path,
    binding: EvidenceFile,
    *,
    description: str,
    maximum_bytes: int | None = None,
) -> tuple[Path, bytes]:
    path = _safe_relative_file(root, binding.relative_path, description=description)
    data = _read_bytes(path, description=description, maximum_bytes=maximum_bytes)
    actual = f"sha256:{hashlib.sha256(data).hexdigest()}"
    if not hmac.compare_digest(actual, binding.sha256):
        raise PhysicalAlignmentReplayError(f"{description} digest mismatch")
    return path, data


def load_cohort(path: Path) -> LoadedCohort:
    """Load a self-digested cohort manifest from an external non-link directory."""

    if _is_under(REPOSITORY_ROOT, path):
        raise PhysicalAlignmentReplayError("cohort manifest must be outside the repository")
    _require_regular_file(path, description="cohort manifest")
    root = path.parent
    _require_non_link_directory(root, description="cohort root")
    data = _read_bytes(path, description="cohort manifest", maximum_bytes=MAX_COHORT_BYTES)
    document = _read_json_bytes(data, description="cohort manifest")
    try:
        cohort = PhysicalAlignmentCohort.model_validate(document)
    except ValidationError as error:
        raise PhysicalAlignmentReplayError(f"cohort manifest schema is invalid: {error}") from error
    unsigned = {key: value for key, value in document.items() if key != "cohortSha256"}
    expected = canonical_json_sha256(unsigned)
    if not hmac.compare_digest(expected, cohort.cohort_sha256):
        raise PhysicalAlignmentReplayError("cohort manifest self digest mismatch")
    return LoadedCohort(
        root=root,
        path=path,
        file_sha256=f"sha256:{hashlib.sha256(data).hexdigest()}",
        model=cohort,
    )


def _load_pinned_artifact(loaded: LoadedCohort):
    binding = EvidenceFile(
        relativePath=loaded.model.artifact.relative_path,
        sha256=loaded.model.artifact.artifact_package_digest,
    )
    _path, data = _read_bound_file(
        loaded.root, binding, description="pinned artifact", maximum_bytes=MAX_ARTIFACT_BYTES,
    )
    document = _read_json_bytes(data, description="pinned artifact")
    schema_version = document.get("schemaVersion")
    model = ARTIFACT_MODELS.get(schema_version)
    if model is None:
        raise PhysicalAlignmentReplayError("pinned artifact schema is unsupported for alignment replay")
    if schema_version != loaded.model.artifact.schema_version:
        raise PhysicalAlignmentReplayError("pinned artifact schema does not match cohort")
    try:
        artifact = model.model_validate(document)
        # Match the production artifact loader's extra semantic checks instead
        # of treating a merely Pydantic-valid subject mask as a replay target.
        require_inspection_roi(artifact)
        require_subject_segmentation(artifact)
    except (ArtifactError, ValidationError, ValueError) as error:
        raise PhysicalAlignmentReplayError("pinned artifact is invalid") from error
    expected = loaded.model.artifact
    identities = (
        (artifact.recipe_id, expected.recipe_id),
        (artifact.machine_id, expected.machine_id),
        (artifact.board_id, expected.board_id),
    )
    if any(not hmac.compare_digest(actual, declared) for actual, declared in identities):
        raise PhysicalAlignmentReplayError("pinned artifact identity does not match cohort")
    return artifact


def _load_readyz(loaded: LoadedCohort, artifact: Any) -> tuple[dict[str, Any], ReadyzReplayProvenance, str]:
    _path, data = _read_bound_file(
        loaded.root, loaded.model.readyz_evidence, description="readyz evidence", maximum_bytes=MAX_READYZ_BYTES,
    )
    readyz = _read_json_bytes(data, description="readyz evidence")
    if (
        readyz.get("status") != "ready"
        or readyz.get("analysisMode") != "ENGINEERING_REAL_DINO"
        or not isinstance(readyz.get("simulation"), bool)
    ):
        raise PhysicalAlignmentReplayError("readyz evidence is not an engineering real-DINO readiness snapshot")
    supported = readyz.get("supportedSchemas")
    if not isinstance(supported, list) or not supported or any(not isinstance(item, str) for item in supported):
        raise PhysicalAlignmentReplayError("readyz evidence must declare supported schemas")
    if isinstance(artifact, ProductionArtifactV19) and "1.6" not in supported:
        raise PhysicalAlignmentReplayError("V19 replay requires readyz support for wire schema 1.6")
    try:
        provenance = ReadyzReplayProvenance.model_validate(readyz.get("replayProvenance"))
    except ValidationError as error:
        raise PhysicalAlignmentReplayError("readyz evidence lacks replay provenance") from error
    if (
        provenance.artifact_package_digest != loaded.model.artifact.artifact_package_digest
        or provenance.analyzer_runtime_version != artifact.analyzer_runtime_version
        or provenance.allow_target_only_alignment != loaded.model.replay_policy.allow_target_only_alignment
        or provenance.allow_contour_anchor_alignment != loaded.model.replay_policy.allow_contour_anchor_alignment
    ):
        raise PhysicalAlignmentReplayError("readyz replay provenance does not match cohort artifact or policy")
    return readyz, provenance, f"sha256:{hashlib.sha256(data).hexdigest()}"


def _jpeg_encoding_from_opened_image(image: Image.Image) -> dict[str, object]:
    layer = getattr(image, "layer", None)
    components = [] if layer is None else [
        [int(component[0]), int(component[1]), int(component[2]), int(component[3])]
        for component in layer
    ]
    quantization = getattr(image, "quantization", {}) or {}
    normalized_tables = [
        [int(table_id), [int(value) for value in values]]
        for table_id, values in sorted(quantization.items(), key=lambda item: int(item[0]))
    ]
    return {
        "format": "JPEG",
        "mode": image.mode,
        "samplingFactors": [[component[1], component[2]] for component in components],
        "componentIds": [component[0] for component in components],
        "quantizationTableSelectors": [component[3] for component in components],
        "progressive": bool(image.info.get("progressive") or image.info.get("progression")),
        "quantizationTablesSha256": canonical_json_sha256(normalized_tables),
    }


def decode_runtime_equivalent(
    raw: bytes,
    *,
    content_type: str,
    max_image_width: int = 8000,
    max_image_height: int = 8000,
    max_image_pixels: int = 24_000_000,
) -> RuntimeDecodedCapture:
    """Decode bytes with the exact PIL/EXIF/RGB semantics used by production."""

    expected_format = "JPEG" if content_type == "image/jpeg" else "PNG" if content_type == "image/png" else None
    if expected_format is None:
        raise PhysicalAlignmentReplayError("request content type is unsupported for replay")
    try:
        with Image.open(BytesIO(raw)) as source:
            if source.format != expected_format:
                raise PhysicalAlignmentReplayError("raw still format does not match saved request")
            source.load()
            source_width, source_height = source.size
            if (
                source_width <= 0
                or source_height <= 0
                or source_width > max_image_width
                or source_height > max_image_height
                or source_width * source_height > max_image_pixels
            ):
                raise PhysicalAlignmentReplayError("raw still dimensions exceed replay limits")
            output_encoding: dict[str, object]
            if source.format == "JPEG":
                output_encoding = _jpeg_encoding_from_opened_image(source)
            else:
                output_encoding = {"format": "PNG", "mode": source.mode}
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            import numpy as np

            rgb = np.asarray(oriented, dtype=np.uint8).copy()
    except PhysicalAlignmentReplayError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise PhysicalAlignmentReplayError("raw still cannot be decoded") from error
    output_encoding["orientedWidth"] = int(rgb.shape[1])
    output_encoding["orientedHeight"] = int(rgb.shape[0])
    return RuntimeDecodedCapture(
        rgb=rgb,
        output_encoding=output_encoding,
        encoded_width=source_width,
        encoded_height=source_height,
    )


def _load_case_documents(
    loaded: LoadedCohort,
    case: CohortCase,
    artifact: Any,
    *,
    readyz_simulation: bool,
    supported_schemas: frozenset[str],
    replay_provenance: ReadyzReplayProvenance,
) -> tuple[bytes, AnalyzeRequest, AnalyzeObservation, RuntimeDecodedCapture]:
    _raw_path, raw = _read_bound_file(
        loaded.root,
        case.raw,
        description=f"raw still for {case.case_id}",
        maximum_bytes=replay_provenance.max_image_bytes,
    )
    _request_path, request_bytes = _read_bound_file(
        loaded.root, case.request, description=f"request for {case.case_id}", maximum_bytes=MAX_REQUEST_BYTES,
    )
    _response_path, response_bytes = _read_bound_file(
        loaded.root, case.response, description=f"response for {case.case_id}", maximum_bytes=MAX_RESPONSE_BYTES,
    )
    try:
        request = AnalyzeRequest.model_validate_json(request_bytes)
        response = AnalyzeObservation.model_validate_json(response_bytes)
    except (ValidationError, ValueError) as error:
        raise PhysicalAlignmentReplayError(f"saved request or response schema is invalid for {case.case_id}") from error
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if request.raw_sha256 != raw_sha256 or response.raw_sha256 != raw_sha256:
        raise PhysicalAlignmentReplayError(f"raw still digest does not match request/response for {case.case_id}")
    if request.artifact_package_digest != loaded.model.artifact.artifact_package_digest:
        raise PhysicalAlignmentReplayError(f"request artifact digest does not match cohort for {case.case_id}")
    if request.content_type not in {"image/jpeg", "image/png"}:
        raise PhysicalAlignmentReplayError(f"request content type is unsupported for {case.case_id}")
    if request.schema_version not in supported_schemas:
        raise PhysicalAlignmentReplayError(f"request wire schema is not supported by saved readyz evidence for {case.case_id}")
    expected_bundle = canonical_bundle_digest(request)
    if not hmac.compare_digest(expected_bundle, request.execution_bundle_digest):
        raise PhysicalAlignmentReplayError(f"request execution bundle digest is invalid for {case.case_id}")
    try:
        verify_artifact_binding(artifact, request)
    except ArtifactError as error:
        raise PhysicalAlignmentReplayError(f"request artifact identity is invalid for {case.case_id}") from error
    if (
        response.schema_version != request.schema_version
        or response.request_id != request.request_id
        or response.raw_sha256 != request.raw_sha256
        or response.simulation != request.simulation
    ):
        raise PhysicalAlignmentReplayError(f"response does not bind the saved request for {case.case_id}")
    resolved = response.resolved_versions
    if (
        resolved.execution_bundle_digest != request.execution_bundle_digest
        or resolved.artifact_package_digest != loaded.model.artifact.artifact_package_digest
        or resolved.analyzer_model_version != artifact.analyzer_model_version
        or resolved.analyzer_runtime_version != artifact.analyzer_runtime_version
        or resolved.normalization_runtime_version != artifact.normalization_pipeline_version
    ):
        raise PhysicalAlignmentReplayError(f"response resolved identity is invalid for {case.case_id}")
    if isinstance(artifact, ProductionArtifactV15):
        if (
            resolved.recipe_analysis_profile_digest != artifact.recipe_analysis_profile.digest
            or resolved.scorer_input_contract_digest != artifact.scorer_input_contract.digest
        ):
            raise PhysicalAlignmentReplayError(f"response scorer/profile identity is invalid for {case.case_id}")
    elif resolved.recipe_analysis_profile_digest is not None or resolved.scorer_input_contract_digest is not None:
        raise PhysicalAlignmentReplayError(f"legacy response has unexpected scorer/profile identity for {case.case_id}")
    expected_analysis_id = hashlib.sha256(
        "|".join((
            request.session_id,
            str(request.capture_ordinal),
            request.raw_sha256,
            request.execution_bundle_digest,
            artifact.analyzer_runtime_version,
        )).encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(expected_analysis_id, response.analysis_id):
        raise PhysicalAlignmentReplayError(f"response analysis identity is invalid for {case.case_id}")
    if request.simulation != readyz_simulation:
        raise PhysicalAlignmentReplayError(f"request simulation does not match readyz evidence for {case.case_id}")
    if isinstance(artifact, ProductionArtifactV19):
        evidence = response.reference_board_evidence
        if (
            evidence is None
            or evidence.state != "VERIFIED"
            or evidence.metric_scale_source != "CHARUCO_ONLY"
            or evidence.qr_payload_sha256 != artifact.reference_board.qr_payload_sha256
        ):
            raise PhysicalAlignmentReplayError(f"V19 response lacks verified reference-board evidence for {case.case_id}")
    decoded = decode_runtime_equivalent(
        raw,
        content_type=request.content_type,
        max_image_width=replay_provenance.max_image_width,
        max_image_height=replay_provenance.max_image_height,
        max_image_pixels=replay_provenance.max_image_pixels,
    )
    return raw, request, response, decoded


def _validate_output_path(path: Path) -> None:
    if _is_under(REPOSITORY_ROOT, path):
        raise PhysicalAlignmentReplayError("replay report must be outside the repository")
    if path.exists():
        raise PhysicalAlignmentReplayError("replay report output already exists")
    _require_non_link_directory(path.parent, description="replay report parent")
    if _is_link_or_reparse_point(path.parent):
        raise PhysicalAlignmentReplayError("replay report parent must not be a link or reparse point")


def _bind_capture_identity(
    bindings: dict[str, tuple[str, str]],
    *,
    value: str,
    acquisition_group_id: str,
    partition: str,
    description: str,
) -> None:
    claimed = (acquisition_group_id, partition)
    prior = bindings.setdefault(value, claimed)
    if prior != claimed:
        raise PhysicalAlignmentReplayError(
            f"saved {description} crosses acquisition-group or partition boundaries"
        )


def run_cohort(
    cohort_path: Path,
    output_path: Path,
    *,
    reference_board_verifier: ReferenceBoardVerifier | None = None,
) -> dict[str, object]:
    """Validate and replay one frozen cohort, then write one external report."""

    _validate_output_path(output_path)
    loaded = load_cohort(cohort_path)
    artifact = _load_pinned_artifact(loaded)
    readyz, replay_provenance, readyz_sha256 = _load_readyz(loaded, artifact)
    readyz_simulation = readyz["simulation"]
    supported_schemas = frozenset(readyz["supportedSchemas"])
    loaded_cases: list[tuple[CohortCase, bytes, AnalyzeRequest, AnalyzeObservation, RuntimeDecodedCapture]] = []
    session_bindings: dict[str, tuple[str, str]] = {}
    correlation_bindings: dict[str, tuple[str, str]] = {}
    for case in sorted(loaded.model.cases, key=lambda item: item.case_id):
        raw, request, response, decoded = _load_case_documents(
            loaded,
            case,
            artifact,
            readyz_simulation=readyz_simulation,
            supported_schemas=supported_schemas,
            replay_provenance=replay_provenance,
        )
        _bind_capture_identity(
            session_bindings,
            value=request.session_id,
            acquisition_group_id=case.acquisition_group_id,
            partition=case.partition,
            description="sessionId",
        )
        _bind_capture_identity(
            correlation_bindings,
            value=request.correlation_id,
            acquisition_group_id=case.acquisition_group_id,
            partition=case.partition,
            description="correlationId",
        )
        loaded_cases.append((case, raw, request, response, decoded))
    normalizer = OpenCvCharucoNormalizer(
        allow_target_only_alignment=loaded.model.replay_policy.allow_target_only_alignment,
        allow_contour_anchor_alignment=loaded.model.replay_policy.allow_contour_anchor_alignment,
    )
    verifier = reference_board_verifier or ReferenceBoardVerifier()
    reports: list[dict[str, object]] = []
    for case, raw, request, response, decoded in loaded_cases:
        image = DecodedImage(
            data=raw,
            width=decoded.encoded_width,
            height=decoded.encoded_height,
            format="JPEG" if request.content_type == "image/jpeg" else "PNG",
            elapsed_ms=0,
        )
        recorded_reference_state = None
        if response.reference_board_evidence is not None:
            recorded_reference_state = response.reference_board_evidence.state
        replayed_reference_state = None
        replayed_reference_reasons: list[str] = []
        if isinstance(artifact, ProductionArtifactV19):
            try:
                replayed_reference = verifier.verify(image, artifact)
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                raise PhysicalAlignmentReplayError(
                    f"V19 reference-board replay failed for {case.case_id}"
                ) from error
            replayed_reference_state = replayed_reference.state
            replayed_reference_reasons = list(replayed_reference.reason_codes)
            if replayed_reference.state != "VERIFIED":
                alignment = None
                reason_codes = tuple(replayed_reference.reason_codes)
            else:
                normalized = (
                    normalizer.normalize(image, artifact, request.board_candidates)
                    if request.board_candidates
                    else normalizer.normalize(image, artifact)
                )
                alignment = normalized.alignment
                reason_codes = normalized.reason_codes
        else:
            normalized = (
                normalizer.normalize(image, artifact, request.board_candidates)
                if request.board_candidates
                else normalizer.normalize(image, artifact)
            )
            alignment = normalized.alignment
            reason_codes = normalized.reason_codes
        replayed_state = alignment.state if alignment is not None else "NOT_ALIGNED"
        recorded_alignment = None
        if response.normalization is not None and response.normalization.alignment is not None:
            recorded_alignment = response.normalization.alignment.state
        reports.append({
            "caseId": case.case_id,
            "partition": case.partition,
            "acquisitionGroupId": case.acquisition_group_id,
            "rawSha256": case.raw.sha256,
            "requestSha256": case.request.sha256,
            "responseSha256": case.response.sha256,
            "intendedAlignmentState": case.intended_alignment_state,
            "recordedAlignmentState": recorded_alignment,
            "replayedAlignmentState": replayed_state,
            "expectationMet": replayed_state == case.intended_alignment_state,
            "recordedReferenceBoardState": recorded_reference_state,
            "replayedReferenceBoardState": replayed_reference_state,
            "replayedReferenceBoardReasonCodes": replayed_reference_reasons,
            "method": None if alignment is None else alignment.method,
            "correlationOrInlierRatio": 0.0 if alignment is None else alignment.inlier_ratio,
            "heldOutOrReprojectionResidualPx": 1000.0 if alignment is None else alignment.reprojection_error_px,
            "coverageRatio": 0.0 if alignment is None else alignment.coverage_ratio,
            "reasonCodes": list(reason_codes),
            "sourceEncoding": decoded.output_encoding,
        })
    engineering_only = bool(readyz_simulation)
    report: dict[str, object] = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": "PHYSICAL_ALIGNMENT_REPLAY_ONLY",
        "cohortSha256": loaded.model.cohort_sha256,
        "cohortFileSha256": loaded.file_sha256,
        "artifact": {
            "artifactPackageDigest": loaded.model.artifact.artifact_package_digest,
            "schemaVersion": loaded.model.artifact.schema_version,
            "recipeId": loaded.model.artifact.recipe_id,
            "machineId": loaded.model.artifact.machine_id,
            "boardId": loaded.model.artifact.board_id,
        },
        "readyzEvidenceSha256": readyz_sha256,
        "readyzReplayProvenance": replay_provenance.model_dump(by_alias=True, mode="json"),
        "replayPolicy": loaded.model.replay_policy.model_dump(by_alias=True, mode="json"),
        "replayRuntimeDigest": RUNTIME_DIGEST,
        "runtimeCompatibility": (
            "SAME_PINNED_RUNTIME" if RUNTIME_DIGEST == artifact.analyzer_runtime_version else "CROSS_RUNTIME_REPLAY_ONLY"
        ),
        "evidenceClassification": (
            "ENGINEERING_REPLAY_ONLY" if engineering_only else "UNAUTHORIZED_CAPTURE_REPLAY_ONLY"
        ),
        "qualificationStatement": "NOT_PHYSICAL_QUALIFICATION_OR_PRODUCTION_AUTHORIZATION",
        "cases": reports,
    }
    report["reportSha256"] = canonical_json_sha256(report)
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as error:
        raise PhysicalAlignmentReplayError("unable to write external replay report") from error
    return report
