from __future__ import annotations

import hashlib
import math
import base64
from threading import Lock
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

from PIL import Image, ImageOps

from .analyzer import RUNTIME_DIGEST
from .artifacts import ArtifactError, ProductionArtifact, ProductionArtifactV12, load_artifact, require_inspection_roi, verify_artifact_binding
from .config import Settings
from .contracts import (
    AnalysisObservation, AnalysisState, AnalyzeObservation, AnalyzeRequest,
    AlignmentObservation, CaptureAssessment, CaptureState, NormalizationObservation, ResolvedVersions,
)
from .decoder import DecodedImage
from .engines import LocalDinoV2Adapter
from .security import digest_directory


@dataclass(frozen=True, slots=True)
class NormalizedCapture:
    rgb: object
    encoded: bytes
    reason_codes: tuple[str, ...] = ()
    alignment: AlignmentObservation | None = None


@dataclass(frozen=True, slots=True)
class PlaneNormalizedCapture:
    rgb: object
    reason_codes: tuple[str, ...] = ()


class Normalizer(Protocol):
    def normalize(self, image: DecodedImage, artifact: ProductionArtifact) -> NormalizedCapture: ...


class Embedder(Protocol):
    def embed(self, rgb: object) -> list[float]: ...


class TargetAligner(Protocol):
    def align(self, plane_rgb: object, artifact: ProductionArtifact) -> NormalizedCapture: ...


def _alignment_observation(
    *, state: str, inliers: int = 0, ratio: float = 0.0, error: float = 0.0,
    coverage: float = 0.0, within_bounds: bool = False,
) -> AlignmentObservation:
    return AlignmentObservation(
        state=state, method="TARGET_AFFINE", targetRelative=True, inlierCount=inliers,
        inlierRatio=ratio, reprojectionErrorPx=error, coverageRatio=coverage,
        transformWithinBounds=within_bounds, inspectionMaskApplied=True,
    )


class OpenCvTargetAligner:
    """Offline ORB/RANSAC baseline; LightGlue can implement the same protocol later.

    Reference descriptors are computed only inside alignment regions after every
    inspection region is removed.  Consequently a changed/removed inspection
    feature cannot drive the transform.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        import cv2
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)

    def align(self, plane_rgb: object, artifact: ProductionArtifact) -> NormalizedCapture:
        # OpenCV RANSAC uses process-global RNG state. Serialize and seed from
        # immutable pixels so repeated requests and process histories agree.
        with self._lock:
            return self._align(plane_rgb, artifact)

    def _align(self, plane_rgb: object, artifact: ProductionArtifact) -> NormalizedCapture:
        import cv2
        import numpy as np

        policy = artifact.target_alignment
        try:
            reference_bytes = base64.b64decode(policy.reference_image_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("TARGET_REFERENCE_INVALID") from exc
        if "sha256:" + hashlib.sha256(reference_bytes).hexdigest() != policy.reference_image_sha256:
            raise RuntimeError("TARGET_REFERENCE_DIGEST_MISMATCH")
        reference = cv2.imdecode(np.frombuffer(reference_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if reference is None or reference.shape[1] != policy.canonical_width or reference.shape[0] != policy.canonical_height:
            raise RuntimeError("TARGET_REFERENCE_INVALID")
        current = cv2.cvtColor(np.asarray(plane_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        deterministic_seed = int.from_bytes(hashlib.sha256(current.tobytes() + reference_bytes).digest()[:4], "big")
        cv2.setRNGSeed(deterministic_seed & 0x7FFFFFFF)
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        reference_mask = np.zeros(ref_gray.shape, dtype=np.uint8)
        for region in policy.alignment_regions:
            reference_mask[region.y:region.y + region.height, region.x:region.x + region.width] = 255
        held_out_mask = np.zeros(ref_gray.shape, dtype=np.uint8)
        for region in policy.held_out_regions:
            held_out_mask[region.y:region.y + region.height, region.x:region.x + region.width] = 255

        detector = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
        ref_keys, ref_desc = detector.detectAndCompute(ref_gray, reference_mask)
        held_keys, held_desc = detector.detectAndCompute(ref_gray, held_out_mask)
        cur_keys, cur_desc = detector.detectAndCompute(current_gray, None)
        if ref_desc is None or held_desc is None or cur_desc is None:
            return NormalizedCapture(None, b"", ("TARGET_NOT_FOUND",), _alignment_observation(state="NOT_ALIGNED"))
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(cur_desc, ref_desc, k=2)
        raw_matches = [
            first for pair in pairs if len(pair) == 2
            for first, second in [pair] if first.distance < 0.75 * second.distance
        ]
        reverse_pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ref_desc, cur_desc, k=2)
        reverse = {
            first.queryIdx: first.trainIdx for pair in reverse_pairs if len(pair) == 2
            for first, second in [pair] if first.distance < 0.75 * second.distance
        }
        # Primary fitting uses mutual, one-to-one correspondences.  Unrestricted
        # current-frame board texture may remain in raw_matches only for the
        # explicit secondary-hypothesis ambiguity check below.
        matches = [item for item in raw_matches if reverse.get(item.trainIdx) == item.queryIdx]
        if len(matches) < policy.min_matches:
            return NormalizedCapture(None, b"", ("TARGET_MATCHES_INSUFFICIENT",), _alignment_observation(state="NOT_ALIGNED"))
        source = np.float32([cur_keys[item.queryIdx].pt for item in matches])
        destination = np.float32([ref_keys[item.trainIdx].pt for item in matches])
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            source, destination, method=cv2.RANSAC, ransacReprojThreshold=policy.max_reprojection_error_px,
            maxIters=3000, confidence=0.995, refineIters=10,
        )
        if transform is None or inlier_mask is None:
            return NormalizedCapture(None, b"", ("TARGET_TRANSFORM_INVALID",), _alignment_observation(state="NOT_ALIGNED"))
        selected = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(selected.sum())
        ratio = inlier_count / len(matches)
        predicted = cv2.transform(source[selected].reshape(1, -1, 2), transform).reshape(-1, 2)
        errors = np.linalg.norm(predicted - destination[selected], axis=1)
        reprojection_p95 = float(np.percentile(errors, 95)) if len(errors) else 1000.0
        hull_area = float(cv2.contourArea(cv2.convexHull(destination[selected]))) if inlier_count >= 3 else 0.0
        alignment_area = float(sum(region.width * region.height for region in policy.alignment_regions))
        coverage = min(1.0, hull_area / max(1.0, alignment_area))
        a, b = float(transform[0, 0]), float(transform[0, 1])
        c, d = float(transform[1, 0]), float(transform[1, 1])
        scale_x, scale_y = math.hypot(a, c), math.hypot(b, d)
        rotation = abs(math.degrees(math.atan2(c, a)))
        shear = abs((a * b + c * d) / max(1e-12, scale_x * scale_y))
        translation = math.hypot(float(transform[0, 2]), float(transform[1, 2]))
        bounded = (
            policy.min_scale <= scale_x <= policy.max_scale
            and policy.min_scale <= scale_y <= policy.max_scale
            and rotation <= policy.max_rotation_degrees and shear <= policy.max_shear
            and translation <= policy.max_translation_px
        )

        # Query each immutable held-out landmark once; this avoids unrelated
        # current-frame keypoints dominating the residual distribution.
        held_pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(held_desc, cur_desc, k=2)
        held_matches = [
            first for pair in held_pairs if len(pair) == 2
            for first, second in [pair] if first.distance < 0.75 * second.distance
        ]
        held_p95 = 1000.0
        held_valid = False
        if len(held_matches) >= policy.min_held_out_matches:
            held_source = np.float32([cur_keys[item.trainIdx].pt for item in held_matches])
            held_destination = np.float32([held_keys[item.queryIdx].pt for item in held_matches])
            held_predicted = cv2.transform(held_source.reshape(1, -1, 2), transform).reshape(-1, 2)
            held_errors = np.linalg.norm(held_predicted - held_destination, axis=1)
            held_p95 = float(np.percentile(held_errors, 95))
            held_valid = held_p95 <= policy.max_held_out_reprojection_error_px
        # Deterministic duplicate gate.  For every raw alignment match, invert
        # the primary transform to predict its source position. A second copy of
        # the same target creates a large, coherent offset cluster. This avoids
        # a second randomized RANSAC and remains stable across process history.
        ambiguous = False
        raw_source = np.float32([cur_keys[item.queryIdx].pt for item in raw_matches])
        raw_destination = np.float32([ref_keys[item.trainIdx].pt for item in raw_matches])
        primary_3x3 = np.vstack((transform, (0.0, 0.0, 1.0)))
        try:
            inverse_primary = np.linalg.inv(primary_3x3)
        except np.linalg.LinAlgError:
            inverse_primary = None
        if inverse_primary is not None and len(raw_source) >= policy.min_matches:
            expected_source = cv2.perspectiveTransform(
                raw_destination.reshape(1, -1, 2), inverse_primary
            ).reshape(-1, 2)
            offsets = raw_source - expected_source
            bin_size = max(12.0, min(policy.canonical_width, policy.canonical_height) * 0.08)
            groups: dict[tuple[int, int], list[int]] = {}
            minimum_separation = min(policy.canonical_width, policy.canonical_height) * 0.35
            for index, offset in enumerate(offsets):
                if float(np.linalg.norm(offset)) < minimum_separation:
                    continue
                key = (int(round(float(offset[0]) / bin_size)), int(round(float(offset[1]) / bin_size)))
                groups.setdefault(key, []).append(index)

            secondary_pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(cur_desc, held_desc, k=2)
            secondary_matches = [
                first for pair in secondary_pairs if len(pair) == 2
                for first, second in [pair] if first.distance < 0.75 * second.distance
            ]
            held_offsets = np.empty((0, 2), dtype=np.float32)
            if secondary_matches:
                secondary_source = np.float32([cur_keys[item.queryIdx].pt for item in secondary_matches])
                secondary_destination = np.float32([held_keys[item.trainIdx].pt for item in secondary_matches])
                expected_held_source = cv2.perspectiveTransform(
                    secondary_destination.reshape(1, -1, 2), inverse_primary
                ).reshape(-1, 2)
                held_offsets = secondary_source - expected_held_source

            for indices in groups.values():
                unique_destination = {raw_matches[index].trainIdx for index in indices}
                if len(unique_destination) < policy.min_inliers:
                    continue
                cluster_center = np.median(offsets[indices], axis=0)
                cluster_destination = raw_destination[indices]
                cluster_hull = (
                    float(cv2.contourArea(cv2.convexHull(cluster_destination))) if len(indices) >= 3 else 0.0
                )
                held_support = int((np.linalg.norm(held_offsets - cluster_center, axis=1) <= bin_size * 1.5).sum())
                if (
                    len(unique_destination) / max(1, inlier_count) > policy.max_secondary_inlier_ratio
                    and cluster_hull / max(1.0, alignment_area) >= max(policy.min_coverage_ratio, 0.15)
                    and held_support >= max(4, policy.min_held_out_matches // 2)
                ):
                    ambiguous = True
                    break
        accepted = (
            inlier_count >= policy.min_inliers and ratio >= policy.min_inlier_ratio
            and coverage >= policy.min_coverage_ratio and reprojection_p95 <= policy.max_reprojection_error_px
            and bounded and not ambiguous and held_valid
        )
        reported_p95 = max(reprojection_p95, held_p95)
        observation = _alignment_observation(
            state="ALIGNED" if accepted else "NOT_ALIGNED", inliers=inlier_count, ratio=ratio,
            error=min(1000.0, reported_p95), coverage=coverage, within_bounds=bounded,
        )
        if not accepted:
            reasons = []
            if inlier_count < policy.min_inliers or ratio < policy.min_inlier_ratio:
                reasons.append("TARGET_INLIERS_INSUFFICIENT")
            if coverage < policy.min_coverage_ratio:
                reasons.append("TARGET_INLIERS_CLUSTERED")
            if reprojection_p95 > policy.max_reprojection_error_px:
                reasons.append("TARGET_REPROJECTION_ERROR")
            if not held_valid:
                reasons.append("TARGET_PARALLAX_OR_MISMATCH")
            if not bounded:
                reasons.append("TARGET_TRANSFORM_OUT_OF_BOUNDS")
            if ambiguous:
                reasons.append("TARGET_AMBIGUOUS")
            return NormalizedCapture(None, b"", tuple(reasons), observation)
        target = cv2.warpAffine(current, transform, (policy.canonical_width, policy.canonical_height))
        rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".png", target)
        if not ok:
            raise RuntimeError("TARGET_CANONICAL_ENCODING_FAILED")
        return NormalizedCapture(rgb, encoded.tobytes(), (), observation)


class OpenCvCharucoPlaneNormalizer:
    """Still Gate and camera/plane normalization only; it never locates the target."""

    def normalize_plane(self, image: DecodedImage, artifact: ProductionArtifact) -> PlaneNormalizedCapture:
        import cv2
        import numpy as np

        try:
            with Image.open(BytesIO(image.data)) as source:
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                decoded = cv2.cvtColor(np.asarray(oriented), cv2.COLOR_RGB2BGR)
        except OSError as exc:
            raise RuntimeError("IMAGE_DECODE_FAILED") from exc
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        overexposure = float(np.count_nonzero(gray >= 250) / gray.size)
        reasons: list[str] = []
        if sharpness < artifact.still_gate.min_laplacian_variance:
            reasons.append("BLUR")
        if overexposure > artifact.still_gate.max_over_exposure_ratio:
            reasons.append("OVER_EXPOSURE")

        board = artifact.board
        dictionary_id = getattr(cv2.aruco, board.dictionary, None)
        if dictionary_id is None:
            raise RuntimeError("CHARUCO_DICTIONARY_UNSUPPORTED")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        charuco = cv2.aruco.CharucoBoard(
            (board.squares_x, board.squares_y), board.square_length_mm, board.marker_length_mm, dictionary
        )
        detector = cv2.aruco.CharucoDetector(charuco)
        corners, ids, _, _ = detector.detectBoard(gray)
        count = 0 if ids is None else len(ids)
        if count < artifact.still_gate.min_charuco_corners:
            reasons.append("CHARUCO_CORNERS_INSUFFICIENT")
        if reasons:
            return PlaneNormalizedCapture(rgb=None, reason_codes=tuple(reasons))

        object_points = charuco.getChessboardCorners()[ids.flatten()][:, :2].astype(np.float32)
        pixels_per_mm_x = board.canonical_width / ((board.squares_x - 1) * board.square_length_mm)
        pixels_per_mm_y = board.canonical_height / ((board.squares_y - 1) * board.square_length_mm)
        destination = object_points.copy()
        destination[:, 0] *= pixels_per_mm_x
        destination[:, 1] *= pixels_per_mm_y
        transform, mask = cv2.findHomography(corners.reshape(-1, 2), destination, cv2.RANSAC, 3.0)
        if transform is None or mask is None or int(mask.sum()) < artifact.still_gate.min_charuco_corners:
            return PlaneNormalizedCapture(rgb=None, reason_codes=("CHARUCO_HOMOGRAPHY_INVALID",))
        canonical = cv2.warpPerspective(decoded, transform, (board.canonical_width, board.canonical_height))
        rgb = cv2.cvtColor(canonical, cv2.COLOR_BGR2RGB)
        return PlaneNormalizedCapture(rgb=rgb)


class OpenCvCharucoNormalizer:
    """Composes board-plane normalization with independent target alignment."""

    def __init__(self, target_aligner: TargetAligner | None = None, *, allow_target_only_alignment: bool = False):
        self._plane = OpenCvCharucoPlaneNormalizer()
        self._target = target_aligner or OpenCvTargetAligner()
        self._allow_target_only_alignment = allow_target_only_alignment

    def normalize(self, image: DecodedImage, artifact: ProductionArtifact) -> NormalizedCapture:
        plane = self._plane.normalize_plane(image, artifact)
        if plane.reason_codes:
            # ChArUco is preferred for plane normalization, but it is not a
            # mandatory capture element. If the board is outside the frame,
            # target-relative alignment may still be safe when the immutable
            # target reference/held-out gates pass. Blur and exposure failures
            # remain fail-closed.
            if self._allow_target_only_alignment and set(plane.reason_codes) == {"CHARUCO_CORNERS_INSUFFICIENT"}:
                import cv2
                import numpy as np
                with Image.open(BytesIO(image.data)) as source:
                    oriented = ImageOps.exif_transpose(source).convert("RGB")
                    raw_rgb = np.asarray(oriented, dtype=np.uint8)
                return self._target.align(raw_rgb, artifact)
            return NormalizedCapture(None, b"", plane.reason_codes)
        return self._target.align(plane.rgb, artifact)


class DinoV2Embedder:
    def __init__(self, adapter: LocalDinoV2Adapter):
        self._adapter = adapter
        self._model: object | None = None
        self._lock = Lock()

    def embed(self, rgb: object) -> list[float]:
        import numpy as np
        import torch
        from torchvision.transforms import v2

        with self._lock:
            if self._model is None:
                self._model = self._adapter.smoke_load()
            pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
            tensor = v2.Compose([
                v2.Resize(256, antialias=True), v2.CenterCrop(224), v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])(pil).unsqueeze(0)
            with torch.inference_mode():
                value = self._model(tensor).detach().cpu().reshape(-1)
        return [float(item) for item in value]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError("EMBEDDING_DIMENSION_MISMATCH")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise RuntimeError("EMBEDDING_NORM_INVALID")
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(0.0, min(2.0, 1.0 - similarity))


class ProductionAnalyzer:
    def __init__(self, settings: Settings, normalizer: Normalizer | None = None, embedder: Embedder | None = None):
        self.settings = settings
        self.adapter = LocalDinoV2Adapter(settings.model_repo, settings.model_weights)
        self.normalizer = normalizer or OpenCvCharucoNormalizer(allow_target_only_alignment=settings.allow_target_only_alignment)
        self.embedder = embedder or DinoV2Embedder(self.adapter)
        self._artifact: ProductionArtifact | None = None

    def readiness(self) -> tuple[bool, str | None]:
        if self.settings.artifact_manifest is None or not self.settings.artifact_manifest.is_file():
            return False, "ARTIFACT_MANIFEST_NOT_AVAILABLE"
        if self.settings.artifact_package_digest is None:
            return False, "ARTIFACT_PACKAGE_DIGEST_NOT_PINNED"
        ready, reason = self.adapter.readiness()
        if not ready:
            return False, reason
        try:
            self._artifact = load_artifact(
                self.settings.artifact_manifest, self.settings.artifact_package_digest, self.settings.model_weights
            )
            if isinstance(self._artifact, ProductionArtifactV12):
                require_inspection_roi(self._artifact)
            if self._artifact.analyzer_runtime_version != RUNTIME_DIGEST:
                return False, "ANALYZER_RUNTIME_VERSION_MISMATCH"
            if self.settings.model_repository_version is None:
                return False, "MODEL_REPOSITORY_VERSION_NOT_PINNED"
            if self._artifact.model_repository_version != self.settings.model_repository_version:
                return False, "MODEL_REPOSITORY_VERSION_MISMATCH"
            if digest_directory(self.settings.model_repo) != self._artifact.model_repository_version:
                return False, "MODEL_REPOSITORY_DIGEST_MISMATCH"
            if isinstance(self.normalizer, OpenCvCharucoNormalizer):
                import cv2
                import numpy
                # Decode and verify the hash-bound target reference during readiness.
                reference = base64.b64decode(self._artifact.target_alignment.reference_image_base64, validate=True)
                if "sha256:" + hashlib.sha256(reference).hexdigest() != self._artifact.target_alignment.reference_image_sha256:
                    return False, "TARGET_REFERENCE_DIGEST_MISMATCH"
                decoded_reference = cv2.imdecode(numpy.frombuffer(reference, dtype=numpy.uint8), cv2.IMREAD_COLOR)
                policy = self._artifact.target_alignment
                if decoded_reference is None or decoded_reference.shape[:2] != (policy.canonical_height, policy.canonical_width):
                    return False, "TARGET_REFERENCE_INVALID"
            if isinstance(self.embedder, DinoV2Embedder):
                import torch
                import torchvision
        except ArtifactError as exc:
            return False, str(exc)
        except (ImportError, OSError):
            return False, "PRODUCTION_DEPENDENCY_NOT_AVAILABLE"
        return True, None

    def analyze(self, request: AnalyzeRequest, image: DecodedImage) -> AnalyzeObservation:
        ready, reason = self.readiness()
        if not ready or self._artifact is None:
            raise RuntimeError(reason or "ANALYZER_NOT_READY")
        verify_artifact_binding(self._artifact, request)
        normalized = self.normalizer.normalize(image, self._artifact)
        identity = "|".join((request.session_id, str(request.capture_ordinal), request.raw_sha256, request.execution_bundle_digest, RUNTIME_DIGEST))
        analysis_id = hashlib.sha256(identity.encode()).hexdigest()
        resolved = ResolvedVersions(
            executionBundleDigest=request.execution_bundle_digest,
            artifactPackageDigest=request.artifact_package_digest,
            analyzerModelVersion=self._artifact.analyzer_model_version,
            analyzerRuntimeVersion=RUNTIME_DIGEST,
            normalizationRuntimeVersion=self._artifact.normalization_pipeline_version,
        )
        if normalized.reason_codes:
            normalization = None
            if normalized.alignment is not None:
                normalization = NormalizationObservation(alignment=normalized.alignment)
            return AnalyzeObservation(
                schemaVersion="1.0", requestId=request.request_id, analysisId=analysis_id,
                rawSha256=request.raw_sha256, simulation=False, resolvedVersions=resolved,
                captureAssessment=CaptureAssessment(state=CaptureState.RECAPTURE_REQUIRED, reasonCodes=list(normalized.reason_codes)),
                normalization=normalization,
                analysis=AnalysisObservation(state=AnalysisState.NOT_RUN),
            )
        alignment = normalized.alignment
        if (
            alignment is None or alignment.state != "ALIGNED" or not alignment.target_relative
            or not alignment.transform_within_bounds or not alignment.inspection_mask_applied
        ):
            return AnalyzeObservation(
                schemaVersion="1.0", requestId=request.request_id, analysisId=analysis_id,
                rawSha256=request.raw_sha256, simulation=False, resolvedVersions=resolved,
                captureAssessment=CaptureAssessment(
                    state=CaptureState.RECAPTURE_REQUIRED, reasonCodes=["TARGET_ALIGNMENT_REQUIRED"],
                ),
                normalization=None if alignment is None else NormalizationObservation(alignment=alignment),
                analysis=AnalysisObservation(state=AnalysisState.NOT_RUN),
            )
        embedding = self.embedder.embed(normalized.rgb)
        distances = [(item.id, _cosine_distance(embedding, item.values)) for item in self._artifact.golden_embeddings]
        nearest_id, nearest = min(distances, key=lambda item: item[1])
        mean = sum(value for _, value in distances) / len(distances)
        uncertainty = min(1.0, max(0.0, max(value for _, value in distances) - min(value for _, value in distances)))
        return AnalyzeObservation(
            schemaVersion="1.0", requestId=request.request_id, analysisId=analysis_id,
            rawSha256=request.raw_sha256, simulation=False, resolvedVersions=resolved,
            captureAssessment=CaptureAssessment(state=CaptureState.ACCEPTED, reasonCodes=[]),
            normalization=NormalizationObservation(
                canonicalSha256=hashlib.sha256(normalized.encoded).hexdigest(), alignment=normalized.alignment,
            ),
            analysis=AnalysisObservation(state=AnalysisState.RUN, metric="cosine_distance", globalDistance=mean,
                                         nearestGoldenId=nearest_id, nearestGoldenDistance=nearest, uncertainty=uncertainty),
        )
