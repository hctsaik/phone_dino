from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import ConfigDict, BaseModel, Field, ValidationError

from .analyzer import RUNTIME_DIGEST
from .artifacts import CharucoBoard, GoldenEmbedding, ProductionArtifact, StillGate, TargetAlignmentPolicy
from .contracts import Identifier, PrefixedSha256
from .decoder import DecodedImage
from .engines import LocalDinoV2Adapter
from .production import DinoV2Embedder, Embedder, Normalizer, OpenCvCharucoNormalizer
from .security import digest_directory, digest_file


class CompilerError(RuntimeError):
    pass


class StrictCompilerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class GoldenSource(StrictCompilerModel):
    id: Identifier
    path: str = Field(min_length=1, max_length=1024)
    source_sha256: PrefixedSha256 = Field(alias="sourceSha256")


class ArtifactBuildSpec(StrictCompilerModel):
    schema_version: str = Field(alias="schemaVersion", pattern=r"^1\.1$")
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

    def model_post_init(self, __context: object) -> None:
        if len({source.id for source in self.golden_sources}) != len(self.golden_sources):
            raise ValueError("Golden source ids must be unique")


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
    Image.MAX_IMAGE_PIXELS = 24_000_000
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
    placeholder = GoldenEmbedding(id="COMPILER-PLACEHOLDER", sourceSha256=spec.golden_set_version, values=[1.0, 0.0])
    normalization_artifact = ProductionArtifact(
        schemaVersion="1.1", recipeId=spec.recipe_id, machineId=spec.machine_id, boardId=spec.board_id,
        goldenSetVersion=spec.golden_set_version,
        normalizationPipelineVersion=spec.normalization_pipeline_version,
        analyzerModelVersion=spec.analyzer_model_version, decisionPolicyVersion=spec.decision_policy_version,
        analyzerRuntimeVersion=RUNTIME_DIGEST,
        modelRepositoryVersion=spec.model_repository_version,
        boardInstallationVersion=spec.board_installation_version,
        modelWeightsSha256=spec.model_weights_sha256, board=spec.board, stillGate=spec.still_gate,
        targetAlignment=spec.target_alignment,
        goldenEmbeddings=[placeholder],
    )
    embeddings: list[GoldenEmbedding] = []
    evidence: list[dict[str, object]] = []
    for source in sorted(spec.golden_sources, key=lambda item: item.id):
        source_path = (spec_path.parent / source.path).resolve()
        decoded = _decoded_image(source_path, source.source_sha256)
        normalized = resolved_normalizer.normalize(decoded, normalization_artifact)
        if normalized.reason_codes:
            raise CompilerError(f"GOLDEN_SOURCE_RECAPTURE_REQUIRED:{source.id}:{','.join(normalized.reason_codes)}")
        if normalized.alignment is None or normalized.alignment.state != "ALIGNED":
            raise CompilerError(f"GOLDEN_SOURCE_TARGET_ALIGNMENT_REQUIRED:{source.id}")
        values = resolved_embedder.embed(normalized.rgb)
        if not values or not all(math.isfinite(value) for value in values) or not any(value != 0 for value in values):
            raise CompilerError(f"GOLDEN_EMBEDDING_INVALID:{source.id}")
        embeddings.append(GoldenEmbedding(id=source.id, sourceSha256=source.source_sha256, values=values))
        evidence.append({
            "id": source.id,
            "sourceSha256": source.source_sha256,
            "canonicalSha256": "sha256:" + hashlib.sha256(normalized.encoded).hexdigest(),
            "alignment": normalized.alignment.model_dump(by_alias=True, mode="json"),
            "embeddingDimension": len(values),
        })
    if len({len(item.values) for item in embeddings}) != 1:
        raise CompilerError("GOLDEN_EMBEDDING_DIMENSION_MISMATCH")
    artifact = normalization_artifact.model_copy(update={"golden_embeddings": embeddings})
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
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--allow-target-only-alignment", action="store_true")
    arguments = parser.parse_args()
    try:
        result = compile_artifact(
            arguments.spec.resolve(), arguments.output.resolve(), arguments.model_repository.resolve(),
            arguments.model_weights.resolve(), evidence_path=arguments.evidence_output.resolve() if arguments.evidence_output else None,
            allow_target_only_alignment=arguments.allow_target_only_alignment,
        )
    except CompilerError as exc:
        raise SystemExit(f"Production artifact compilation failed: {exc}") from exc
    print(json.dumps({
        "artifactPath": str(result.artifact_path), "artifactPackageDigest": result.artifact_package_digest,
        "evidencePath": str(result.evidence_path),
    }, separators=(",", ":"), sort_keys=True))
