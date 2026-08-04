from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import ConfigDict, BaseModel, Field, ValidationError

from .analyzer import RUNTIME_DIGEST
from .artifacts import (
    CandidateVerificationPolicy, CandidateVerificationPolicyV2, CharucoBoard, GoldenEmbedding, GoldenSubjectMask,
    CurrentSubjectSegmentationPolicy, DimensionMeasurementPolicy,
    InspectionRoiContract, ProductionArtifact, ProductionArtifactV12, ProductionArtifactV13,
    ProductionArtifactV14, ProductionArtifactV15, ProductionArtifactV16, ProductionArtifactV17,
    ProductionArtifactV18,
    RecipeAnalysisProfile, ScorerInputContract,
    ScorerInputTileEmbedding, SpatialDifferencePolicy, StillGate, SubjectAlignmentContract,
    SubjectSegmentationContract, TargetAlignmentPolicy, inspection_roi_image,
)
from .contracts import Identifier, PrefixedSha256
from .decoder import DecodedImage
from .engines import LocalDinoV2Adapter
from .production import (
    DinoV2Embedder, Embedder, Normalizer, OpenCvCharucoNormalizer, PatchEmbedding,
    _mean_embedding, _scorer_input_digest, _scorer_input_tiles,
)
from .security import digest_directory, digest_file
from .segmenters import MobileSamSegmenter, SubjectSegmenter


class CompilerError(RuntimeError):
    pass


class StrictCompilerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class GoldenSource(StrictCompilerModel):
    id: Identifier
    path: str = Field(min_length=1, max_length=1024)
    source_sha256: PrefixedSha256 = Field(alias="sourceSha256")


class SubjectSegmentationBuildSpec(StrictCompilerModel):
    version: str = Field(pattern=r"^subject-[0-9]+\.[0-9]+$")
    method: Literal["MOBILE_SAM_VIT_T_BOX_PROMPT"]
    usage_mode: Literal["SPATIAL_GATE"] = Field(alias="usageMode")
    approval_state: Literal["ENGINEERING_AUTO", "APPROVED"] = Field(alias="approvalState")
    prompt_policy: Literal["INSPECTION_ROI_BOUNDING_BOX_V1"] = Field(alias="promptPolicy")
    model_repository_version: PrefixedSha256 = Field(alias="modelRepositoryVersion")
    model_weights_sha256: PrefixedSha256 = Field(alias="modelWeightsSha256")
    min_model_quality_score: float = Field(alias="minModelQualityScore", ge=0, le=1)
    min_foreground_ratio: float = Field(alias="minForegroundRatio", gt=0, lt=1)
    max_foreground_ratio: float = Field(alias="maxForegroundRatio", gt=0, le=1)
    support_padding_px: int = Field(alias="supportPaddingPx", ge=0, le=512)
    boundary_band_px: int = Field(alias="boundaryBandPx", ge=0, le=256)

    def model_post_init(self, __context: object) -> None:
        if self.min_foreground_ratio >= self.max_foreground_ratio:
            raise ValueError("minForegroundRatio must be less than maxForegroundRatio")


class ArtifactBuildSpec(StrictCompilerModel):
    schema_version: str = Field(alias="schemaVersion", pattern=r"^1\.[12345678]$")
    recipe_id: Identifier = Field(alias="recipeId")
    machine_id: Identifier = Field(alias="machineId")
    board_id: Identifier = Field(alias="boardId")
    golden_set_version: PrefixedSha256 = Field(alias="goldenSetVersion")
    normalization_pipeline_version: PrefixedSha256 = Field(alias="normalizationPipelineVersion")
    analyzer_model_version: PrefixedSha256 = Field(alias="analyzerModelVersion")
    decision_policy_version: PrefixedSha256 = Field(alias="decisionPolicyVersion")
    model_repository_version: PrefixedSha256 = Field(alias="modelRepositoryVersion")
    board_installation_version: PrefixedSha256 = Field(alias="boardInstallationVersion")
    model_weights_sha256: PrefixedSha256 = Field(alias="modelWeightsSha256")
    board: CharucoBoard
    still_gate: StillGate = Field(alias="stillGate")
    target_alignment: TargetAlignmentPolicy = Field(alias="targetAlignment")
    golden_sources: list[GoldenSource] = Field(alias="goldenSources", min_length=1, max_length=256)
    spatial_difference_policy: SpatialDifferencePolicy | None = Field(default=None, alias="spatialDifferencePolicy")
    inspection_roi: InspectionRoiContract | None = Field(default=None, alias="inspectionRoi")
    subject_segmentation: SubjectSegmentationBuildSpec | None = Field(default=None, alias="subjectSegmentation")
    subject_alignment: SubjectAlignmentContract | None = Field(default=None, alias="subjectAlignment")
    candidate_verification_policy: CandidateVerificationPolicy | CandidateVerificationPolicyV2 | None = Field(
        default=None, alias="candidateVerificationPolicy",
    )
    scorer_input_contract: ScorerInputContract | None = Field(default=None, alias="scorerInputContract")
    recipe_analysis_profile: RecipeAnalysisProfile | None = Field(default=None, alias="recipeAnalysisProfile")
    current_subject_segmentation: CurrentSubjectSegmentationPolicy | None = Field(
        default=None, alias="currentSubjectSegmentation",
    )
    dimension_measurement_policy: DimensionMeasurementPolicy | None = Field(
        default=None, alias="dimensionMeasurementPolicy",
    )

    def model_post_init(self, __context: object) -> None:
        if len({source.id for source in self.golden_sources}) != len(self.golden_sources):
            raise ValueError("Golden source ids must be unique")
        if self.schema_version in {"1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8"} and self.inspection_roi is None:
            raise ValueError("schema 1.2+ requires inspectionRoi")
        if self.schema_version == "1.1" and self.inspection_roi is not None:
            raise ValueError("schema 1.1 cannot include inspectionRoi")
        if self.schema_version in {"1.3", "1.4", "1.5", "1.6", "1.7", "1.8"} and self.subject_segmentation is None:
            raise ValueError("schema 1.3+ requires subjectSegmentation")
        if self.schema_version not in {"1.3", "1.4", "1.5", "1.6", "1.7", "1.8"} and self.subject_segmentation is not None:
            raise ValueError("subjectSegmentation requires schema 1.3+")
        if self.schema_version in {"1.4", "1.5", "1.6", "1.7", "1.8"}:
            if self.subject_alignment is None or self.candidate_verification_policy is None:
                raise ValueError("schema 1.4 requires subjectAlignment and candidateVerificationPolicy")
        elif self.subject_alignment is not None or self.candidate_verification_policy is not None:
            raise ValueError("subject alignment and candidate verification require schema 1.4")
        if self.schema_version in {"1.5", "1.6", "1.7", "1.8"}:
            if self.scorer_input_contract is None or self.recipe_analysis_profile is None:
                raise ValueError("schema 1.5 requires scorerInputContract and recipeAnalysisProfile")
        elif self.scorer_input_contract is not None or self.recipe_analysis_profile is not None:
            raise ValueError("ROI-only scorer and recipe analysis profile require schema 1.5")
        if self.schema_version in {"1.6", "1.7", "1.8"}:
            if self.current_subject_segmentation is None:
                raise ValueError("schema 1.6 requires currentSubjectSegmentation")
        elif self.current_subject_segmentation is not None:
            raise ValueError("currentSubjectSegmentation requires schema 1.6")
        if self.schema_version in {"1.7", "1.8"} and not isinstance(
            self.candidate_verification_policy, CandidateVerificationPolicyV2,
        ):
            raise ValueError("schema 1.7 requires local-structure candidate verification")
        if self.schema_version == "1.8" and self.dimension_measurement_policy is None:
            raise ValueError("schema 1.8 requires dimensionMeasurementPolicy")
        if self.schema_version != "1.8" and self.dimension_measurement_policy is not None:
            raise ValueError("dimensionMeasurementPolicy requires schema 1.8")


@dataclass(frozen=True, slots=True)
class CompileResult:
    artifact_path: Path
    artifact_package_digest: str
    evidence_path: Path


def _decoded_image(path: Path, expected_digest: str) -> DecodedImage:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CompilerError(f"GOLDEN_SOURCE_NOT_READABLE:{path.name}") from exc
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if actual != expected_digest:
        raise CompilerError(f"GOLDEN_SOURCE_DIGEST_MISMATCH:{path.name}")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise CompilerError(f"GOLDEN_SOURCE_FORMAT_INVALID:{path.name}")
            width, height = image.size
            if width <= 0 or height <= 0 or width > 8000 or height > 8000 or width * height > 24_000_000:
                raise CompilerError(f"GOLDEN_SOURCE_DIMENSIONS_INVALID:{path.name}")
            image.load()
            image_format = image.format
    except CompilerError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise CompilerError(f"GOLDEN_SOURCE_DECODE_FAILED:{path.name}") from exc
    return DecodedImage(data=data, width=width, height=height, format=image_format, elapsed_ms=0)


def compile_artifact(
    spec_path: Path,
    output_path: Path,
    model_repository: Path,
    model_weights: Path,
    *,
    evidence_path: Path | None = None,
    normalizer: Normalizer | None = None,
    embedder: Embedder | None = None,
    subject_segmenter: SubjectSegmenter | None = None,
    segmenter_repository: Path | None = None,
    segmenter_weights: Path | None = None,
    segmenter_device: str = "cpu",
    allow_target_only_alignment: bool = False,
) -> CompileResult:
    try:
        spec = ArtifactBuildSpec.model_validate_json(spec_path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise CompilerError("ARTIFACT_BUILD_SPEC_INVALID") from exc
    adapter = LocalDinoV2Adapter(model_repository, model_weights)
    ready, reason = adapter.readiness()
    if not ready:
        raise CompilerError(reason or "MODEL_NOT_READY")
    if digest_file(model_weights) != spec.model_weights_sha256:
        raise CompilerError("MODEL_WEIGHTS_DIGEST_MISMATCH")
    try:
        repository_digest = digest_directory(model_repository)
    except OSError as exc:
        raise CompilerError("MODEL_REPOSITORY_NOT_READABLE") from exc
    if repository_digest != spec.model_repository_version:
        raise CompilerError("MODEL_REPOSITORY_DIGEST_MISMATCH")
    resolved_normalizer = normalizer or OpenCvCharucoNormalizer(allow_target_only_alignment=allow_target_only_alignment)
    resolved_embedder = embedder or DinoV2Embedder(adapter)
    resolved_segmenter = subject_segmenter
    if spec.subject_segmentation is not None and resolved_segmenter is None:
        if segmenter_repository is None or segmenter_weights is None:
            raise CompilerError("SUBJECT_SEGMENTER_PATHS_REQUIRED")
        if not segmenter_weights.is_file():
            raise CompilerError("SUBJECT_SEGMENTER_WEIGHTS_NOT_AVAILABLE")
        try:
            segmenter_repository_digest = digest_directory(segmenter_repository)
            segmenter_weights_digest = digest_file(segmenter_weights)
        except OSError as exc:
            raise CompilerError("SUBJECT_SEGMENTER_NOT_READABLE") from exc
        if segmenter_repository_digest != spec.subject_segmentation.model_repository_version:
            raise CompilerError("SUBJECT_SEGMENTER_REPOSITORY_DIGEST_MISMATCH")
        if segmenter_weights_digest != spec.subject_segmentation.model_weights_sha256:
            raise CompilerError("SUBJECT_SEGMENTER_WEIGHTS_DIGEST_MISMATCH")
        resolved_segmenter = MobileSamSegmenter(segmenter_repository, segmenter_weights, segmenter_device)
    placeholder = GoldenEmbedding(id="COMPILER-PLACEHOLDER", sourceSha256=spec.golden_set_version, values=[1.0, 0.0])
    # Schema 1.3 needs generated masks, but normalization only needs the
    # already-pinned 1.2 geometry. Build a temporary 1.2 view, then validate
    # the complete 1.3 artifact after every Golden has been segmented.
    artifact_type = ProductionArtifactV12 if spec.schema_version in {"1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8"} else ProductionArtifact
    normalization_schema = "1.2" if spec.schema_version in {"1.3", "1.4", "1.5", "1.6", "1.7", "1.8"} else spec.schema_version
    artifact_fields: dict[str, object] = {}
    if spec.inspection_roi is not None:
        artifact_fields["inspectionRoi"] = spec.inspection_roi
    normalization_artifact = artifact_type(
        schemaVersion=normalization_schema, recipeId=spec.recipe_id, machineId=spec.machine_id, boardId=spec.board_id,
        goldenSetVersion=spec.golden_set_version,
        normalizationPipelineVersion=spec.normalization_pipeline_version,
        analyzerModelVersion=spec.analyzer_model_version, decisionPolicyVersion=spec.decision_policy_version,
        analyzerRuntimeVersion=RUNTIME_DIGEST,
        modelRepositoryVersion=spec.model_repository_version,
        boardInstallationVersion=spec.board_installation_version,
        modelWeightsSha256=spec.model_weights_sha256, board=spec.board, stillGate=spec.still_gate,
        targetAlignment=spec.target_alignment,
        goldenEmbeddings=[placeholder], **artifact_fields,
    )
    embeddings: list[GoldenEmbedding] = []
    subject_masks: list[GoldenSubjectMask] = []
    evidence: list[dict[str, object]] = []
    supports_patches = hasattr(resolved_embedder, "embed_with_patches")
    for source in sorted(spec.golden_sources, key=lambda item: item.id):
        source_path = (spec_path.parent / source.path).resolve()
        decoded = _decoded_image(source_path, source.source_sha256)
        normalized = resolved_normalizer.normalize(decoded, normalization_artifact)
        if normalized.reason_codes:
            raise CompilerError(f"GOLDEN_SOURCE_RECAPTURE_REQUIRED:{source.id}:{','.join(normalized.reason_codes)}")
        if normalized.alignment is None or normalized.alignment.state != "ALIGNED":
            raise CompilerError(f"GOLDEN_SOURCE_TARGET_ALIGNMENT_REQUIRED:{source.id}")
        subject_mask_evidence: dict[str, object] | None = None
        subject_support_mask: object | None = None
        subject_core_mask: object | None = None
        if spec.subject_segmentation is not None:
            if resolved_segmenter is None or spec.inspection_roi is None:
                raise CompilerError("SUBJECT_SEGMENTER_NOT_AVAILABLE")
            regions = spec.inspection_roi.inspection_regions
            inspection_image = inspection_roi_image(spec.inspection_roi)
            bounds = inspection_image.getbbox()
            if bounds is None:
                raise CompilerError("INSPECTION_ROI_EMPTY")
            left, top, right, bottom = bounds
            try:
                prediction = resolved_segmenter.segment(
                    normalized.rgb, (left, top, right, bottom),
                    min_foreground_ratio=spec.subject_segmentation.min_foreground_ratio,
                    max_foreground_ratio=spec.subject_segmentation.max_foreground_ratio,
                    min_quality_score=spec.subject_segmentation.min_model_quality_score,
                )
            except RuntimeError as exc:
                raise CompilerError(f"GOLDEN_SUBJECT_SEGMENTATION_FAILED:{source.id}:{exc}") from exc

            import numpy as np

            mask_array = np.asarray(prediction.mask, dtype=bool)
            expected_shape = (
                spec.inspection_roi.canonical_height,
                spec.inspection_roi.canonical_width,
            )
            if mask_array.shape != expected_shape:
                raise CompilerError(f"GOLDEN_SUBJECT_MASK_DIMENSIONS_INVALID:{source.id}")
            inspection_mask = np.asarray(inspection_image, dtype=np.uint8) > 0
            mask_array &= inspection_mask
            prompt_area = float((right - left) * (bottom - top))
            foreground_ratio = float(np.count_nonzero(mask_array)) / prompt_area
            if not (
                spec.subject_segmentation.min_foreground_ratio
                <= foreground_ratio
                <= spec.subject_segmentation.max_foreground_ratio
            ):
                raise CompilerError(f"GOLDEN_SUBJECT_MASK_AREA_OUT_OF_POLICY:{source.id}")
            mask_image = Image.fromarray(mask_array.astype("uint8") * 255)
            mask_buffer = BytesIO()
            mask_image.save(mask_buffer, format="PNG", optimize=False)
            mask_bytes = mask_buffer.getvalue()
            canonical_sha = "sha256:" + hashlib.sha256(normalized.encoded).hexdigest()
            prompt_normalized = {
                "x": float(left / spec.inspection_roi.canonical_width),
                "y": float(top / spec.inspection_roi.canonical_height),
                "width": float((right - left) / spec.inspection_roi.canonical_width),
                "height": float((bottom - top) / spec.inspection_roi.canonical_height),
            }
            subject_masks.append(GoldenSubjectMask(
                goldenId=source.id,
                canonicalSha256=canonical_sha,
                maskPngBase64=base64.b64encode(mask_bytes).decode("ascii"),
                maskSha256="sha256:" + hashlib.sha256(mask_bytes).hexdigest(),
                promptRegionNormalized=prompt_normalized,
                modelQualityScore=prediction.quality_score,
                foregroundRatio=foreground_ratio,
            ))
            subject_mask_evidence = {
                "method": spec.subject_segmentation.method,
                "maskSha256": "sha256:" + hashlib.sha256(mask_bytes).hexdigest(),
                "modelQualityScore": prediction.quality_score,
                "foregroundRatio": foreground_ratio,
                "promptRegionNormalized": prompt_normalized,
            }
            import cv2

            core = mask_array.astype("uint8") * 255
            subject_core_mask = core
            if spec.subject_segmentation.support_padding_px > 0:
                diameter = spec.subject_segmentation.support_padding_px * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
                subject_support_mask = cv2.dilate(core, kernel)
            else:
                subject_support_mask = core.copy()
            subject_support_mask[~inspection_mask] = 0
        patches: PatchEmbedding | None = None
        scorer_tiles = None
        scorer_input_sha = None
        if spec.schema_version in {"1.5", "1.6", "1.7", "1.8"}:
            if not supports_patches or spec.inspection_roi is None or spec.scorer_input_contract is None:
                raise CompilerError("ROI_ONLY_PATCH_EMBEDDER_REQUIRED")
            import numpy as np

            analysis_mask = np.asarray(inspection_roi_image(spec.inspection_roi), dtype=np.uint8)
            if spec.schema_version in {"1.6", "1.7", "1.8"}:
                if subject_core_mask is None or spec.current_subject_segmentation is None:
                    raise CompilerError("PAIRED_INTERIOR_MASK_REQUIRED")
                import cv2
                diameter = spec.current_subject_segmentation.interior_erosion_px * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
                golden_interior = cv2.erode(np.asarray(subject_core_mask, dtype=np.uint8), kernel)
                analysis_mask = np.where(golden_interior > 0, analysis_mask, 0).astype(np.uint8)
            elif subject_support_mask is not None:
                analysis_mask = np.where(np.asarray(subject_support_mask, dtype=np.uint8) > 0, analysis_mask, 0).astype(np.uint8)
            inputs = _scorer_input_tiles(
                normalized.rgb,
                spec.inspection_roi.inspection_regions,
                spec.inspection_roi.canonical_width,
                spec.inspection_roi.canonical_height,
                analysis_mask,
                spec.scorer_input_contract.neutral_rgb,
            )
            tile_embeddings = [resolved_embedder.embed_with_patches(tile.rgb).global_vector for tile in inputs]  # type: ignore[union-attr]
            values = _mean_embedding(tile_embeddings)
            scorer_tiles = [
                ScorerInputTileEmbedding(
                    id=tile.id, x=tile.x, y=tile.y, side=tile.side,
                    tileSha256=tile.sha256, values=vector,
                )
                for tile, vector in zip(inputs, tile_embeddings, strict=True)
            ]
            scorer_input_sha = _scorer_input_digest(inputs, spec.scorer_input_contract.digest)
        elif supports_patches:
            patches = resolved_embedder.embed_with_patches(normalized.rgb)  # type: ignore[union-attr]
            values = patches.global_vector
        else:
            values = resolved_embedder.embed(normalized.rgb)
        if not values or not all(math.isfinite(value) for value in values) or not any(value != 0 for value in values):
            raise CompilerError(f"GOLDEN_EMBEDDING_INVALID:{source.id}")
        if scorer_tiles is not None:
            embeddings.append(GoldenEmbedding(
                id=source.id, sourceSha256=source.source_sha256, values=values,
                canonicalSha256="sha256:" + hashlib.sha256(normalized.encoded).hexdigest(),
                canonicalImagePngBase64=base64.b64encode(normalized.encoded).decode("ascii"),
                scorerInputSha256=scorer_input_sha, scorerInputTiles=scorer_tiles,
            ))
        elif patches is not None:
            if not all(math.isfinite(value) for row in patches.patch_grid for value in row):
                raise CompilerError(f"GOLDEN_PATCH_EMBEDDING_INVALID:{source.id}")
            embeddings.append(GoldenEmbedding(
                id=source.id, sourceSha256=source.source_sha256, values=values,
                canonicalSha256="sha256:" + hashlib.sha256(normalized.encoded).hexdigest(),
                canonicalImagePngBase64=base64.b64encode(normalized.encoded).decode("ascii"),
                patchValues=patches.patch_grid, patchGridHeight=patches.grid_height, patchGridWidth=patches.grid_width,
            ))
        else:
            embeddings.append(GoldenEmbedding(
                id=source.id, sourceSha256=source.source_sha256, values=values,
                canonicalSha256="sha256:" + hashlib.sha256(normalized.encoded).hexdigest(),
                canonicalImagePngBase64=base64.b64encode(normalized.encoded).decode("ascii"),
            ))
        evidence.append({
            "id": source.id,
            "sourceSha256": source.source_sha256,
            "canonicalSha256": "sha256:" + hashlib.sha256(normalized.encoded).hexdigest(),
            "alignment": normalized.alignment.model_dump(by_alias=True, mode="json"),
            "embeddingDimension": len(values),
            "patchFeaturesIncluded": patches is not None,
            **({
                "scorerInputSha256": scorer_input_sha,
                "scorerInputTiles": [
                    {"id": item.id, "x": item.x, "y": item.y, "side": item.side, "tileSha256": item.tile_sha256}
                    for item in (scorer_tiles or [])
                ],
            } if scorer_input_sha is not None else {}),
            **({"subjectSegmentation": subject_mask_evidence} if subject_mask_evidence is not None else {}),
        })
    if len({len(item.values) for item in embeddings}) != 1:
        raise CompilerError("GOLDEN_EMBEDDING_DIMENSION_MISMATCH")
    artifact_document = normalization_artifact.model_dump(by_alias=True, mode="json")
    artifact_document.update({
        "schemaVersion": spec.schema_version,
        "goldenEmbeddings": [item.model_dump(by_alias=True, mode="json") for item in embeddings],
        "spatialDifferencePolicy": (
            None if spec.spatial_difference_policy is None
            else spec.spatial_difference_policy.model_dump(by_alias=True, mode="json")
        ),
    })
    if spec.subject_segmentation is not None:
        subject_contract = SubjectSegmentationContract(
            **spec.subject_segmentation.model_dump(by_alias=True, mode="json"),
            canonicalWidth=spec.target_alignment.canonical_width,
            canonicalHeight=spec.target_alignment.canonical_height,
            goldenMasks=[item.model_dump(by_alias=True, mode="json") for item in subject_masks],
        )
        artifact_document["subjectSegmentation"] = subject_contract.model_dump(by_alias=True, mode="json")
    if spec.subject_alignment is not None:
        artifact_document["subjectAlignment"] = spec.subject_alignment.model_dump(by_alias=True, mode="json")
    if spec.candidate_verification_policy is not None:
        artifact_document["candidateVerificationPolicy"] = spec.candidate_verification_policy.model_dump(
            by_alias=True, mode="json",
        )
    if spec.scorer_input_contract is not None:
        artifact_document["scorerInputContract"] = spec.scorer_input_contract.model_dump(by_alias=True, mode="json")
    if spec.recipe_analysis_profile is not None:
        artifact_document["recipeAnalysisProfile"] = spec.recipe_analysis_profile.model_dump(by_alias=True, mode="json")
    if spec.current_subject_segmentation is not None:
        artifact_document["currentSubjectSegmentation"] = spec.current_subject_segmentation.model_dump(
            by_alias=True, mode="json",
        )
    if spec.dimension_measurement_policy is not None:
        artifact_document["dimensionMeasurementPolicy"] = spec.dimension_measurement_policy.model_dump(
            by_alias=True, mode="json",
        )
    artifact_model = (
        ProductionArtifactV18 if spec.schema_version == "1.8"
        else ProductionArtifactV17 if spec.schema_version == "1.7"
        else ProductionArtifactV16 if spec.schema_version == "1.6"
        else ProductionArtifactV15 if spec.schema_version == "1.5"
        else ProductionArtifactV14 if spec.schema_version == "1.4"
        else ProductionArtifactV13 if spec.schema_version == "1.3"
        else ProductionArtifactV12 if spec.schema_version == "1.2"
        else ProductionArtifact
    )
    artifact = artifact_model.model_validate(artifact_document)
    encoded = json.dumps(
        artifact.model_dump(by_alias=True, mode="json"), ensure_ascii=False,
        separators=(",", ":"), sort_keys=True, allow_nan=False,
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise CompilerError("ARTIFACT_OUTPUT_ALREADY_EXISTS") from exc
    artifact_digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    resolved_evidence = evidence_path or output_path.with_suffix(output_path.suffix + ".evidence.json")
    evidence_document = {
        "schemaVersion": "1.0", "artifactPackageDigest": artifact_digest,
        "analyzerRuntimeVersion": RUNTIME_DIGEST, "modelRepositoryVersion": spec.model_repository_version,
        "modelWeightsSha256": spec.model_weights_sha256, "goldenSources": evidence,
    }
    if spec.subject_segmentation is not None:
        evidence_document["subjectSegmentation"] = {
            "method": spec.subject_segmentation.method,
            "modelRepositoryVersion": spec.subject_segmentation.model_repository_version,
            "modelWeightsSha256": spec.subject_segmentation.model_weights_sha256,
            "promptPolicy": spec.subject_segmentation.prompt_policy,
            "approvalState": spec.subject_segmentation.approval_state,
        }
    if spec.subject_alignment is not None:
        evidence_document["subjectAlignment"] = spec.subject_alignment.model_dump(by_alias=True, mode="json")
    if spec.candidate_verification_policy is not None:
        evidence_document["candidateVerificationPolicy"] = spec.candidate_verification_policy.model_dump(
            by_alias=True, mode="json",
        )
    if spec.scorer_input_contract is not None:
        evidence_document["scorerInputContractDigest"] = spec.scorer_input_contract.digest
    if spec.recipe_analysis_profile is not None:
        evidence_document["recipeAnalysisProfileDigest"] = spec.recipe_analysis_profile.digest
        evidence_document["referenceRoles"] = {
            "alignmentTemplate": spec.recipe_analysis_profile.alignment_template.model_dump(by_alias=True, mode="json"),
            "targetReference": spec.recipe_analysis_profile.target_reference.model_dump(by_alias=True, mode="json"),
            "normalReferenceSet": spec.recipe_analysis_profile.normal_reference_set.model_dump(by_alias=True, mode="json"),
            "displayReference": spec.recipe_analysis_profile.display_reference.model_dump(by_alias=True, mode="json"),
        }
    if spec.current_subject_segmentation is not None:
        evidence_document["currentSubjectSegmentation"] = spec.current_subject_segmentation.model_dump(
            by_alias=True, mode="json",
        )
    if spec.dimension_measurement_policy is not None:
        evidence_document["dimensionMeasurementPolicy"] = spec.dimension_measurement_policy.model_dump(
            by_alias=True, mode="json",
        )
    try:
        with resolved_evidence.open("x", encoding="utf-8") as handle:
            json.dump(evidence_document, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except OSError as exc:
        output_path.unlink(missing_ok=True)
        raise CompilerError("EVIDENCE_OUTPUT_NOT_WRITABLE") from exc
    return CompileResult(output_path, artifact_digest, resolved_evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile an immutable phone_dino production artifact offline")
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-repository", required=True, type=Path)
    parser.add_argument("--model-weights", required=True, type=Path)
    parser.add_argument("--segmenter-repository", type=Path)
    parser.add_argument("--segmenter-weights", type=Path)
    parser.add_argument("--segmenter-device", default="cpu")
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--allow-target-only-alignment", action="store_true")
    arguments = parser.parse_args()
    try:
        result = compile_artifact(
            arguments.spec.resolve(), arguments.output.resolve(), arguments.model_repository.resolve(),
            arguments.model_weights.resolve(), evidence_path=arguments.evidence_output.resolve() if arguments.evidence_output else None,
            segmenter_repository=arguments.segmenter_repository.resolve() if arguments.segmenter_repository else None,
            segmenter_weights=arguments.segmenter_weights.resolve() if arguments.segmenter_weights else None,
            segmenter_device=arguments.segmenter_device,
            allow_target_only_alignment=arguments.allow_target_only_alignment,
        )
    except CompilerError as exc:
        raise SystemExit(f"Production artifact compilation failed: {exc}") from exc
    print(json.dumps({
        "artifactPath": str(result.artifact_path), "artifactPackageDigest": result.artifact_package_digest,
        "evidencePath": str(result.evidence_path),
    }, separators=(",", ":"), sort_keys=True))
