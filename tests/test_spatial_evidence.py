from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from phone_dino.app import create_app
from phone_dino.analyzer import RUNTIME_DIGEST
from phone_dino.artifacts import (
    CandidateVerificationPolicy, CandidateVerificationPolicyV2, GoldenEmbedding, ImageRegion,
    SpatialDifferencePolicy,
)
from phone_dino.config import Settings
from phone_dino.contracts import AlignmentObservation
from phone_dino.production import (
    NormalizedCapture, PatchEmbedding, ProductionAnalyzer, SubjectScope, _dino_input_crop_box,
    _candidate_crop_box, _roi_tiled_spatial_difference_evidence, _scorer_input_digest,
    _scorer_input_tiles, _spatial_difference_evidence,
    _local_candidate_structure_verification, _verify_candidate_crop,
)
from phone_dino.security import digest_directory

from .test_api import AcceptedNormalizer, png_bytes_for_reference, settings, target_alignment


class GridPatchEmbedder:
    """Deterministic small-grid embedder: fast, exact pure-function coverage."""

    def __init__(self, grid: list[list[float]], grid_size: int = 2, dim: int = 2, global_vector: list[float] | None = None):
        self._grid = grid
        self._grid_size = grid_size
        self._dim = dim
        self._global_vector = global_vector or [1.0, 0.0]

    def embed(self, rgb):
        return self._global_vector

    def embed_with_patches(self, rgb):
        return PatchEmbedding(
            global_vector=self._global_vector, patch_grid=self._grid,
            grid_height=self._grid_size, grid_width=self._grid_size,
        )


class QuadrantPatchEmbedder:
    """Produces deterministic image-dependent patch tokens without a model."""

    def __init__(self):
        self.calls = 0

    def embed_with_patches(self, rgb):
        import numpy as np

        self.calls += 1
        image = np.asarray(rgb)
        rows = np.array_split(image, 2, axis=0)
        values = []
        for row in rows:
            for quadrant in np.array_split(row, 2, axis=1):
                values.append([1.0, 0.0] if float(quadrant.mean()) < 96 else [0.0, 1.0])
        return PatchEmbedding(global_vector=[1.0, 0.0], patch_grid=values, grid_height=2, grid_width=2)


class ImageGridPatchEmbedder:
    """Image-dependent grid used to create separated spatial components."""

    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size

    def embed_with_patches(self, rgb):
        import numpy as np

        image = np.asarray(rgb)
        values = []
        for row in np.array_split(image, self.grid_size, axis=0):
            for cell in np.array_split(row, self.grid_size, axis=1):
                values.append([1.0, 0.0] if float(cell.mean()) < 96 else [0.0, 1.0])
        return PatchEmbedding(
            global_vector=[1.0, 0.0], patch_grid=values,
            grid_height=self.grid_size, grid_width=self.grid_size,
        )


IDENTICAL_GRID = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
DIFFERENT_GRID = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
POLICY = SpatialDifferencePolicy(anomalyDistanceThreshold=0.5, minRegionAreaRatio=0.01, maxRegions=8)


def _golden(patch_values=None, grid_h=None, grid_w=None) -> GoldenEmbedding:
    kwargs = {}
    if patch_values is not None:
        kwargs = {"patchValues": patch_values, "patchGridHeight": grid_h, "patchGridWidth": grid_w}
    return GoldenEmbedding(id="G-1", sourceSha256="sha256:" + "0" * 64, values=[1.0, 0.0], **kwargs)


class TestGoldenEmbeddingPatchValidation:
    def test_patch_fields_must_be_set_together(self):
        with pytest.raises(Exception):
            GoldenEmbedding(id="G-1", sourceSha256="sha256:" + "0" * 64, values=[1.0, 0.0], patchGridHeight=2, patchGridWidth=2)

    def test_patch_grid_dimensions_must_match_declared_size(self):
        with pytest.raises(Exception):
            _golden(patch_values=IDENTICAL_GRID, grid_h=3, grid_w=3)

    def test_patch_vector_dimension_must_match_global_embedding(self):
        with pytest.raises(Exception):
            _golden(patch_values=[[1.0, 0.0, 0.0]] * 4, grid_h=2, grid_w=2)

    def test_non_finite_patch_values_rejected(self):
        with pytest.raises(Exception):
            _golden(patch_values=[[1.0, math.inf], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], grid_h=2, grid_w=2)

    def test_valid_patch_golden_accepted(self):
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        assert golden.patch_grid_height == 2 and golden.patch_grid_width == 2


class TestSpatialDifferencePolicyValidation:
    def test_threshold_out_of_range_rejected(self):
        with pytest.raises(Exception):
            SpatialDifferencePolicy(anomalyDistanceThreshold=3.0, minRegionAreaRatio=0.01, maxRegions=8)

    def test_valid_policy_accepted(self):
        policy = SpatialDifferencePolicy(anomalyDistanceThreshold=0.4, minRegionAreaRatio=0.02, maxRegions=16)
        assert policy.max_regions == 16


class ImageMeanPatchEmbedder:
    def embed_with_patches(self, rgb):
        import numpy as np

        vector = [1.0, 0.0] if float(np.asarray(rgb).mean()) < 64 else [0.0, 1.0]
        return PatchEmbedding(global_vector=vector, patch_grid=[vector], grid_height=1, grid_width=1)


def test_candidate_crop_verifier_reports_shadow_priority_without_defect_claim():
    import numpy as np

    golden = np.zeros((240, 320, 3), dtype=np.uint8)
    current = golden.copy()
    current[90:150, 120:200] = 255
    full_mask = np.full((240, 320), 255, dtype=np.uint8)
    scope = SubjectScope(core_mask=full_mask, support_mask=full_mask, boundary_mask=np.zeros_like(full_mask), evidence=None)
    policy = CandidateVerificationPolicy.model_validate({
        "version": "candidate-verify-1.0", "method": "DINO_CROP_COSINE_V1",
        "mode": "SHADOW", "approvalState": "ENGINEERING_AUTO",
        "contextPaddingRatio": 0.35, "minimumCropSidePx": 112, "maxCandidates": 8,
        "reviewPriorityDistance": 0.1, "highPriorityDistance": 0.25,
    })
    crop = _candidate_crop_box(120, 90, 80, 60, 320, 240, policy)

    verification = _verify_candidate_crop(
        ImageMeanPatchEmbedder(), current, golden, scope, policy, crop, {}, "golden",
    )

    assert verification.mode == "SHADOW"
    assert verification.priority == "HIGH"
    assert verification.crop_distance == pytest.approx(1.0)
    assert verification.disclaimer_code == "CANDIDATE_VERIFICATION_NOT_DEFECT_PROOF"


def _local_structure_policy() -> CandidateVerificationPolicyV2:
    return CandidateVerificationPolicyV2.model_validate({
        "version": "candidate-verify-2.0", "method": "DINO_CROP_COSINE_LOCAL_STRUCTURE_V2",
        "mode": "SHADOW", "approvalState": "ENGINEERING_AUTO",
        "contextPaddingRatio": 0.35, "minimumCropSidePx": 112, "maxCandidates": 8,
        "reviewPriorityDistance": 0.1, "highPriorityDistance": 0.25,
        "localAlignmentMethod": "GRADIENT_ECC_TRANSLATION_V1",
        "photometricNormalization": "OPENCV_LAB_CONTEXT_MEDIAN_MAD_V1",
        "structureMethod": "LAB_DELTA_OR_CANNY_EDGE_V1",
        "maxLocalTranslationPx": 6.0, "minLocalAlignmentCorrelation": 0.45,
        "candidateExclusionPaddingPx": 5, "minimumContextPixels": 512,
        "appearanceDeltaThreshold": 0.12, "minAppearanceChangedAreaRatio": 0.3,
        "minEdgeChangedAreaRatio": 0.15,
    })


def _textured_candidate_scene():
    import numpy as np

    rng = np.random.default_rng(23)
    golden = rng.integers(20, 220, size=(180, 180, 3), dtype=np.uint8)
    full_mask = np.full((180, 180), 255, dtype=np.uint8)
    scope = SubjectScope(
        core_mask=full_mask, support_mask=full_mask,
        boundary_mask=np.zeros_like(full_mask), evidence=None,
    )
    return golden, scope, (34, 34, 112, 112), (72, 72, 36, 36)


def test_local_structure_normalization_does_not_confirm_uniform_photometric_shift():
    import numpy as np

    golden, scope, crop, candidate = _textured_candidate_scene()
    current = np.clip(golden.astype(np.float32) * 1.08 + 10, 0, 255).astype(np.uint8)

    state, _, dx, dy, appearance, edge, confirmation = _local_candidate_structure_verification(
        current, golden, scope, _local_structure_policy(), crop, candidate,
    )

    assert state == "ALIGNED"
    assert abs(dx) <= 1 and abs(dy) <= 1
    assert appearance is not None and appearance < 0.3
    assert edge is not None and edge < 0.15
    assert confirmation == "UNCONFIRMED"


def test_local_structure_confirms_local_change_after_photometric_normalization():
    import numpy as np

    golden, scope, crop, candidate = _textured_candidate_scene()
    current = np.clip(golden.astype(np.float32) * 1.08 + 10, 0, 255).astype(np.uint8)
    x, y, width, height = candidate
    current[y:y + height, x:x + width] = (245, 220, 20)

    state, _, _, _, appearance, edge, confirmation = _local_candidate_structure_verification(
        current, golden, scope, _local_structure_policy(), crop, candidate,
    )

    assert state == "ALIGNED"
    assert appearance is not None and edge is not None
    assert appearance >= 0.3 or edge >= 0.15
    assert confirmation == "CONFIRMED"


def test_local_structure_rejects_out_of_bounds_translation_without_ratios():
    import cv2
    import numpy as np

    golden, scope, crop, candidate = _textured_candidate_scene()
    current = cv2.warpAffine(
        golden, np.float32([[1, 0, 10], [0, 1, 0]]), (180, 180),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101,
    )

    state, correlation, dx, _, appearance, edge, confirmation = _local_candidate_structure_verification(
        current, golden, scope, _local_structure_policy(), crop, candidate,
    )

    assert state == "UNQUALIFIED"
    assert correlation < 0.45 or abs(dx) > 6
    assert appearance is None and edge is None
    assert confirmation == "LOCAL_ALIGNMENT_UNQUALIFIED"


class TestSpatialDifferenceEvidenceFunction:
    def test_unavailable_when_no_policy_configured(self):
        embedder = GridPatchEmbedder(IDENTICAL_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        evidence = _spatial_difference_evidence(patches, golden, None, 320, 240)
        assert evidence.state == "UNAVAILABLE"
        assert evidence.reason_code == "SPATIAL_DIFFERENCE_POLICY_NOT_CONFIGURED"
        assert evidence.regions is None
        assert evidence.map_png_base64 is None

    def test_unavailable_when_golden_lacks_patch_features(self):
        embedder = GridPatchEmbedder(IDENTICAL_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden()
        evidence = _spatial_difference_evidence(patches, golden, POLICY, 320, 240)
        assert evidence.state == "UNAVAILABLE"
        assert evidence.reason_code == "GOLDEN_PATCH_FEATURES_UNAVAILABLE"

    def test_unavailable_when_grid_size_mismatches_golden(self):
        embedder = GridPatchEmbedder(IDENTICAL_GRID, grid_size=2)
        patches = embedder.embed_with_patches(None)
        golden = GoldenEmbedding(
            id="G-1", sourceSha256="sha256:" + "0" * 64, values=[1.0, 0.0],
            patchValues=[[1.0, 0.0]] * 9, patchGridHeight=3, patchGridWidth=3,
        )
        evidence = _spatial_difference_evidence(patches, golden, POLICY, 320, 240)
        assert evidence.state == "UNAVAILABLE"
        assert evidence.reason_code == "GOLDEN_PATCH_FEATURES_UNAVAILABLE"

    def test_identical_patches_produce_no_regions(self):
        embedder = GridPatchEmbedder(IDENTICAL_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        evidence = _spatial_difference_evidence(patches, golden, POLICY, 320, 240)
        assert evidence.state == "AVAILABLE"
        assert evidence.generation_method == "PATCH_DISTANCE"
        assert evidence.regions == []
        assert evidence.map_png_base64 is not None and evidence.mask_png_base64 is not None
        map_bytes = base64.b64decode(evidence.map_png_base64)
        mask_bytes = base64.b64decode(evidence.mask_png_base64)
        assert hashlib.sha256(map_bytes).hexdigest() == evidence.map_sha256
        assert hashlib.sha256(mask_bytes).hexdigest() == evidence.mask_sha256

    def test_different_patches_produce_bounded_region(self):
        embedder = GridPatchEmbedder(DIFFERENT_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        evidence = _spatial_difference_evidence(patches, golden, POLICY, 320, 240)
        assert evidence.state == "AVAILABLE"
        assert len(evidence.regions) >= 1
        for region in evidence.regions:
            box = region.bbox_normalized
            assert 0 <= box.x and 0 <= box.y
            assert box.x + box.width <= 1.0 + 1e-9
            assert box.y + box.height <= 1.0 + 1e-9
            assert 0 <= region.peak_score <= 1 and 0 <= region.mean_score <= 1

    def test_evidence_region_never_exceeds_dino_input_crop(self):
        embedder = GridPatchEmbedder(DIFFERENT_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        canonical_width, canonical_height = 320, 240
        evidence = _spatial_difference_evidence(patches, golden, POLICY, canonical_width, canonical_height)
        left, top, width, height = _dino_input_crop_box(canonical_width, canonical_height)
        region_box = evidence.evidence_region_normalized
        assert region_box.x == pytest.approx(left / canonical_width, abs=1e-6)
        assert region_box.y == pytest.approx(top / canonical_height, abs=1e-6)
        for region in evidence.regions:
            assert region.bbox_normalized.x + 1e-6 >= region_box.x
            assert region.bbox_normalized.y + 1e-6 >= region_box.y
            assert region.bbox_normalized.x + region.bbox_normalized.width <= region_box.x + region_box.width + 1e-6
            assert region.bbox_normalized.y + region.bbox_normalized.height <= region_box.y + region_box.height + 1e-6

    def test_regions_outside_every_inspection_region_are_dropped(self):
        from phone_dino.artifacts import ImageRegion
        import cv2
        import numpy as np

        # DIFFERENT_GRID's difference lands in the bottom half of the analyzed crop.
        embedder = GridPatchEmbedder(DIFFERENT_GRID)
        patches = embedder.embed_with_patches(None)
        golden = _golden(patch_values=IDENTICAL_GRID, grid_h=2, grid_w=2)
        canonical_width, canonical_height = 320, 240

        excluding = [ImageRegion(x=0, y=0, width=320, height=50)]
        dropped = _spatial_difference_evidence(patches, golden, POLICY, canonical_width, canonical_height, excluding)
        assert dropped.regions == []
        heatmap = cv2.imdecode(np.frombuffer(base64.b64decode(dropped.map_png_base64), dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.frombuffer(base64.b64decode(dropped.mask_png_base64), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # The declared ROI occupies only the top of the DINO crop. Evidence
        # bytes below it must be black/zero, not merely absent from regions[].
        assert not np.any(heatmap[60:, :, :])
        assert not np.any(mask[60:, :])

        covering = [ImageRegion(x=0, y=150, width=320, height=90)]
        kept = _spatial_difference_evidence(patches, golden, POLICY, canonical_width, canonical_height, covering)
        assert len(kept.regions) >= 1

        # No inspection regions supplied at all preserves the unfiltered behaviour.
        unfiltered = _spatial_difference_evidence(patches, golden, POLICY, canonical_width, canonical_height, [])
        assert len(unfiltered.regions) == len(kept.regions)

    def test_max_regions_is_enforced(self):
        # A checkerboard-like grid maximizes distinct connected components.
        big = 8
        grid = [[1.0, 0.0] if (r + c) % 2 == 0 else [0.0, 1.0] for r in range(big) for c in range(big)]
        embedder = GridPatchEmbedder(grid, grid_size=big)
        patches = embedder.embed_with_patches(None)
        golden = GoldenEmbedding(
            id="G-1", sourceSha256="sha256:" + "0" * 64, values=[1.0, 0.0],
            patchValues=[[1.0, 0.0]] * (big * big), patchGridHeight=big, patchGridWidth=big,
        )
        tight_policy = SpatialDifferencePolicy(anomalyDistanceThreshold=0.5, minRegionAreaRatio=0.0001, maxRegions=3)
        evidence = _spatial_difference_evidence(patches, golden, tight_policy, 320, 240)
        assert evidence.state == "AVAILABLE"
        assert len(evidence.regions) <= 3
        import cv2
        import numpy as np
        retained = cv2.imdecode(
            np.frombuffer(base64.b64decode(evidence.mask_png_base64), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        component_count, _ = cv2.connectedComponents((retained > 0).astype(np.uint8), connectivity=8)
        assert component_count - 1 == len(evidence.regions)

    def test_roi_tiled_evidence_is_full_canonical_roi_masked_and_caches_golden_tiles(self):
        import cv2
        import numpy as np

        width = height = 320
        golden = np.zeros((height, width, 3), dtype=np.uint8)
        current = golden.copy()
        current[60:140, 115:205] = 255
        roi = [ImageRegion(x=80, y=40, width=160, height=240)]
        embedder = QuadrantPatchEmbedder()
        cache = {}
        policy = SpatialDifferencePolicy(
            anomalyDistanceThreshold=0.5, minRegionAreaRatio=0.001, maxRegions=8,
        )

        evidence = _roi_tiled_spatial_difference_evidence(
            embedder, current, golden, policy, width, height, roi,
            golden_cache=cache, golden_cache_key="sha256:golden",
        )
        calls_after_first = embedder.calls
        repeated = _roi_tiled_spatial_difference_evidence(
            embedder, current, golden, policy, width, height, roi,
            golden_cache=cache, golden_cache_key="sha256:golden",
        )

        assert evidence.state == "AVAILABLE"
        assert evidence.generation_method == "ROI_TILED_PATCH_DISTANCE"
        assert evidence.evidence_region_normalized.model_dump() == {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
        assert repeated.map_sha256 == evidence.map_sha256
        assert embedder.calls - calls_after_first == len(cache)
        heatmap = cv2.imdecode(np.frombuffer(base64.b64decode(evidence.map_png_base64), dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.frombuffer(base64.b64decode(evidence.mask_png_base64), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        assert heatmap.shape[:2] == (height, width)
        assert mask.shape == (height, width)
        assert not np.any(heatmap[:, :80]) and not np.any(mask[:, :80])
        assert not np.any(heatmap[:, 240:]) and not np.any(mask[:, 240:])

    def test_roi_only_scorer_neutralizes_outside_pixels_and_shares_spatial_tile_identity(self):
        import numpy as np

        width = height = 320
        roi = [ImageRegion(x=80, y=40, width=160, height=240)]
        support = np.zeros((height, width), dtype=np.uint8)
        support[70:250, 105:215] = 255
        golden = np.full((height, width, 3), 24, dtype=np.uint8)
        outside_changed = golden.copy()
        outside_changed[support == 0] = 240
        neutral = (127, 127, 127)
        contract_digest = "sha256:" + "9" * 64

        golden_tiles = _scorer_input_tiles(golden, roi, width, height, support, neutral)
        outside_tiles = _scorer_input_tiles(outside_changed, roi, width, height, support, neutral)
        assert [tile.sha256 for tile in outside_tiles] == [tile.sha256 for tile in golden_tiles]
        assert _scorer_input_digest(outside_tiles, contract_digest) == _scorer_input_digest(golden_tiles, contract_digest)

        inside_changed = outside_changed.copy()
        inside_changed[120:180, 130:190] = 255
        inside_tiles = _scorer_input_tiles(inside_changed, roi, width, height, support, neutral)
        assert [tile.sha256 for tile in inside_tiles] != [tile.sha256 for tile in golden_tiles]
        assert _scorer_input_digest(inside_tiles, contract_digest) != _scorer_input_digest(golden_tiles, contract_digest)

        scope = SubjectScope(core_mask=support, support_mask=support, boundary_mask=np.zeros_like(support), evidence=None)  # type: ignore[arg-type]
        evidence = _roi_tiled_spatial_difference_evidence(
            QuadrantPatchEmbedder(), inside_changed, golden, POLICY, width, height, roi,
            subject_scope=scope, neutral_rgb=neutral,
        )
        assert evidence.state == "AVAILABLE"
        assert evidence.evidence_coordinate_space == "TARGET_CANONICAL_IMAGE"
        assert evidence.evidence_region_normalized.model_dump() == {
            "x": 105 / width, "y": 70 / height, "width": 110 / width, "height": 180 / height,
        }
        assert [item.current_sha256 for item in evidence.scorer_input_tile_digests] == [
            tile.sha256.removeprefix("sha256:") for tile in inside_tiles
        ]
        assert [item.reference_sha256 for item in evidence.scorer_input_tile_digests] == [
            tile.sha256.removeprefix("sha256:") for tile in golden_tiles
        ]

    def test_roi_only_spatial_evidence_reuses_current_scorer_patch_embeddings(self):
        import numpy as np

        width = height = 320
        roi = [ImageRegion(x=80, y=40, width=160, height=240)]
        support = np.zeros((height, width), dtype=np.uint8)
        support[70:250, 105:215] = 255
        current = np.zeros((height, width, 3), dtype=np.uint8)
        golden = current.copy()
        scope = SubjectScope(
            core_mask=support,
            support_mask=support,
            boundary_mask=np.zeros_like(support),
            evidence=None,  # type: ignore[arg-type]
        )
        embedder = QuadrantPatchEmbedder()
        current_tiles = _scorer_input_tiles(current, roi, width, height, support, (127, 127, 127))
        current_cache = {
            (tile.x, tile.y, tile.side): embedder.embed_with_patches(tile.rgb)
            for tile in current_tiles
        }
        calls_after_scorer = embedder.calls

        evidence = _roi_tiled_spatial_difference_evidence(
            embedder, current, golden, POLICY, width, height, roi,
            current_cache=current_cache, subject_scope=scope, neutral_rgb=(127, 127, 127),
        )

        assert evidence.state == "AVAILABLE"
        assert embedder.calls - calls_after_scorer == len(current_tiles)

    def test_subject_gate_suppresses_background_before_dino_and_retains_subject_change(self):
        import cv2
        import numpy as np

        width = height = 320
        golden = np.zeros((height, width, 3), dtype=np.uint8)
        support = np.zeros((height, width), dtype=np.uint8)
        support[:, :160] = 255
        boundary = np.zeros_like(support)
        boundary[:, 150:160] = 255
        scope = SubjectScope(core_mask=support, support_mask=support, boundary_mask=boundary, evidence=None)  # type: ignore[arg-type]
        roi = [ImageRegion(x=0, y=0, width=320, height=320)]
        policy = SpatialDifferencePolicy(
            anomalyDistanceThreshold=0.5, minRegionAreaRatio=0.001, maxRegions=8,
        )

        background_only = golden.copy()
        background_only[:, 160:] = 255
        suppressed = _roi_tiled_spatial_difference_evidence(
            QuadrantPatchEmbedder(), background_only, golden, policy, width, height, roi,
            subject_scope=scope,
        )
        suppressed_raw = cv2.imdecode(
            np.frombuffer(base64.b64decode(suppressed.raw_threshold_mask_png_base64), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        assert suppressed.generation_method == "SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE"
        assert not np.any(suppressed_raw)
        assert suppressed.regions == []

        subject_changed = golden.copy()
        subject_changed[:, :160] = 255
        retained = _roi_tiled_spatial_difference_evidence(
            QuadrantPatchEmbedder(), subject_changed, golden, policy, width, height, roi,
            subject_scope=scope,
        )
        assert retained.regions
        assert retained.candidate_filter.retained_component_count == len(retained.regions)
        final_mask = cv2.imdecode(
            np.frombuffer(base64.b64decode(retained.mask_png_base64), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        assert np.any(final_mask[:, :160])
        assert not np.any(final_mask[:, 160:])

    def test_padded_support_background_component_is_counted_and_not_returned(self):
        import cv2
        import numpy as np

        width = height = 320
        golden = np.zeros((height, width, 3), dtype=np.uint8)
        current = golden.copy()
        current[80:160, 240:320] = 255
        core = np.zeros((height, width), dtype=np.uint8)
        core[:, :80] = 255
        support = np.full_like(core, 255)
        boundary = np.zeros_like(core)
        boundary[:, 70:90] = 255
        scope = SubjectScope(core_mask=core, support_mask=support, boundary_mask=boundary, evidence=None)  # type: ignore[arg-type]
        policy = SpatialDifferencePolicy(
            anomalyDistanceThreshold=0.5, minRegionAreaRatio=0.001, maxRegions=8,
        )

        evidence = _roi_tiled_spatial_difference_evidence(
            ImageGridPatchEmbedder(), current, golden, policy, width, height,
            [ImageRegion(x=0, y=0, width=width, height=height)], subject_scope=scope,
        )
        raw = cv2.imdecode(
            np.frombuffer(base64.b64decode(evidence.raw_threshold_mask_png_base64), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )

        assert np.any(raw)
        assert evidence.regions == []
        assert evidence.candidate_filter.raw_component_count == 1
        assert evidence.candidate_filter.suppressed_by_background_count == 1
        assert evidence.candidate_filter.suppressed_small_region_count == 0

    def test_inward_subject_candidate_precedes_boundary_at_region_limit(self):
        import numpy as np

        width = height = 320
        golden = np.zeros((height, width, 3), dtype=np.uint8)
        current = golden.copy()
        current[80:160, 80:160] = 255
        current[80:160, 240:320] = 255
        core = np.full((height, width), 255, dtype=np.uint8)
        boundary = np.zeros_like(core)
        boundary[:, 240:] = 255
        scope = SubjectScope(core_mask=core, support_mask=core, boundary_mask=boundary, evidence=None)  # type: ignore[arg-type]
        policy = SpatialDifferencePolicy(
            anomalyDistanceThreshold=0.5, minRegionAreaRatio=0.001, maxRegions=1,
        )

        evidence = _roi_tiled_spatial_difference_evidence(
            ImageGridPatchEmbedder(), current, golden, policy, width, height,
            [ImageRegion(x=0, y=0, width=width, height=height)], subject_scope=scope,
        )

        assert evidence.candidate_filter.raw_component_count == 2
        assert evidence.candidate_filter.suppressed_by_limit_count == 1
        assert len(evidence.regions) == 1
        assert evidence.regions[0].kind == "SUBJECT_INTERIOR"


def _spatial_production_setup(tmp_path, manifest, *, spatial_policy=None, embedder=None, golden_patch=None, golden_values=None):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"reviewed-model-weights")
    repo = tmp_path / "dinov2"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# reviewed local repository\n", encoding="utf-8")
    component = manifest["executionBundle"]["goldenSetVersion"]
    repository_version = digest_directory(repo)
    resolved_values = golden_values if golden_values is not None else [0.9, 0.1, 0.0]
    golden_embedding = {"id": "golden-near", "sourceSha256": component, "values": resolved_values}
    if golden_patch is not None:
        golden_embedding.update(golden_patch)
    artifact_body = {
        "schemaVersion": "1.1",
        "recipeId": manifest["recipeId"], "machineId": manifest["machineId"], "boardId": manifest["boardId"],
        "goldenSetVersion": component,
        "normalizationPipelineVersion": manifest["executionBundle"]["normalizationPipelineVersion"],
        "analyzerModelVersion": manifest["executionBundle"]["analyzerModelVersion"],
        "decisionPolicyVersion": manifest["executionBundle"]["decisionPolicyVersion"],
        "analyzerRuntimeVersion": RUNTIME_DIGEST,
        "modelRepositoryVersion": repository_version,
        "boardInstallationVersion": manifest["executionBundle"]["boardInstallationVersion"],
        "modelWeightsSha256": "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest(),
        "board": {
            "squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
            "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896,
        },
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
        "targetAlignment": target_alignment(png_bytes_for_reference()),
        "goldenEmbeddings": [golden_embedding, {"id": "golden-far", "sourceSha256": component, "values": [0.0] * (len(resolved_values) - 1) + [1.0]}],
    }
    if spatial_policy is not None:
        artifact_body["spatialDifferencePolicy"] = spatial_policy
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(artifact_body, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    artifact_digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest["artifactPackageDigest"] = artifact_digest
    manifest["simulation"] = False
    configured = replace(
        settings(tmp_path, enabled=False), artifact_manifest=artifact,
        artifact_package_digest=artifact_digest, model_repo=repo, model_weights=weights,
        model_repository_version=repository_version,
    )
    analyzer = ProductionAnalyzer(configured, normalizer=AcceptedNormalizer(), embedder=embedder)
    return TestClient(create_app(configured, production_analyzer=analyzer)), artifact_body


def _post(client, manifest, image):
    return client.post(
        "/internal/v1/analyze", headers={"Authorization": "Bearer secret"},
        files={
            "manifest": ("manifest.json", json.dumps(manifest).encode(), "application/json"),
            "image": ("capture.png", image, "image/png"),
        },
    )


def _subject_gated_setup(tmp_path, manifest, *, engineering=True):
    import cv2
    import numpy as np

    weights = tmp_path / "weights-subject.pth"
    weights.write_bytes(b"reviewed-model-weights")
    repo = tmp_path / "dinov2-subject"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# reviewed local repository\n", encoding="utf-8")
    repository_version = digest_directory(repo)
    component = manifest["executionBundle"]["goldenSetVersion"]
    golden_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    golden_buffer = io.BytesIO()
    Image.fromarray(golden_rgb).save(golden_buffer, "PNG")
    golden_png = golden_buffer.getvalue()
    subject = np.zeros((240, 320), dtype=np.uint8)
    subject[110:170, 110:190] = 255
    subject_buffer = io.BytesIO()
    Image.fromarray(subject).save(subject_buffer, "PNG")
    subject_png = subject_buffer.getvalue()
    roi_payload = {
        "version": "roi-1.0", "canonicalWidth": 320, "canonicalHeight": 240,
        "polygon": [
            {"x": 100.0, "y": 100.0}, {"x": 200.0, "y": 100.0},
            {"x": 200.0, "y": 180.0}, {"x": 100.0, "y": 180.0},
        ],
        "inspectionRegions": [{"x": 100, "y": 100, "width": 100, "height": 80}],
    }
    roi_digest = "sha256:" + hashlib.sha256(
        json.dumps(roi_payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    artifact_body = {
        "schemaVersion": "1.3", "recipeId": manifest["recipeId"],
        "machineId": manifest["machineId"], "boardId": manifest["boardId"],
        "goldenSetVersion": component,
        "normalizationPipelineVersion": manifest["executionBundle"]["normalizationPipelineVersion"],
        "analyzerModelVersion": manifest["executionBundle"]["analyzerModelVersion"],
        "decisionPolicyVersion": manifest["executionBundle"]["decisionPolicyVersion"],
        "analyzerRuntimeVersion": RUNTIME_DIGEST, "modelRepositoryVersion": repository_version,
        "boardInstallationVersion": manifest["executionBundle"]["boardInstallationVersion"],
        "modelWeightsSha256": "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest(),
        "board": {
            "squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
            "dictionary": "DICT_4X4_50", "canonicalWidth": 320, "canonicalHeight": 240,
        },
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
        "targetAlignment": target_alignment(png_bytes_for_reference()),
        "inspectionRoi": {**roi_payload, "digest": roi_digest},
        "spatialDifferencePolicy": {"anomalyDistanceThreshold": 0.5, "minRegionAreaRatio": 0.001, "maxRegions": 8},
        "goldenEmbeddings": [{
            "id": "golden-subject", "sourceSha256": component, "values": [1.0, 0.0],
            "canonicalSha256": "sha256:" + hashlib.sha256(golden_png).hexdigest(),
            "canonicalImagePngBase64": base64.b64encode(golden_png).decode(),
        }],
        "subjectSegmentation": {
            "version": "subject-1.0", "method": "MOBILE_SAM_VIT_T_BOX_PROMPT",
            "usageMode": "SPATIAL_GATE", "approvalState": "ENGINEERING_AUTO",
            "promptPolicy": "INSPECTION_ROI_BOUNDING_BOX_V1", "canonicalWidth": 320, "canonicalHeight": 240,
            "modelRepositoryVersion": "sha256:" + "1" * 64,
            "modelWeightsSha256": "sha256:" + "2" * 64,
            "minModelQualityScore": 0.8, "minForegroundRatio": 0.1, "maxForegroundRatio": 0.9,
            "supportPaddingPx": 5, "boundaryBandPx": 3,
            "goldenMasks": [{
                "goldenId": "golden-subject",
                "canonicalSha256": "sha256:" + hashlib.sha256(golden_png).hexdigest(),
                "maskPngBase64": base64.b64encode(subject_png).decode(),
                "maskSha256": "sha256:" + hashlib.sha256(subject_png).hexdigest(),
                "promptRegionNormalized": {"x": 0.3125, "y": 100 / 240, "width": 0.3125, "height": 1 / 3},
                "modelQualityScore": 0.95, "foregroundRatio": 0.6,
            }],
        },
    }
    artifact = tmp_path / "artifact-subject.json"
    artifact.write_text(json.dumps(artifact_body, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    artifact_digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest.update({"schemaVersion": "1.1", "artifactPackageDigest": artifact_digest, "simulation": engineering})

    current_rgb = golden_rgb.copy()
    current_rgb[120:160, 120:180] = 255
    ok, encoded_current = cv2.imencode(".png", cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR))
    assert ok

    class ArrayNormalizer:
        def normalize(self, image, artifact):
            return NormalizedCapture(
                rgb=current_rgb, encoded=encoded_current.tobytes(),
                alignment=AlignmentObservation(
                    state="ALIGNED", method="TARGET_AFFINE", targetRelative=True,
                    inlierCount=20, inlierRatio=0.8, reprojectionErrorPx=0.5,
                    coverageRatio=0.4, transformWithinBounds=True, inspectionMaskApplied=True,
                ),
            )

    configured = replace(
        settings(tmp_path, enabled=False), artifact_manifest=artifact,
        artifact_package_digest=artifact_digest, model_repo=repo, model_weights=weights,
        model_repository_version=repository_version, engineering_real_model_enabled=engineering,
    )
    analyzer = ProductionAnalyzer(configured, normalizer=ArrayNormalizer(), embedder=QuadrantPatchEmbedder())
    return TestClient(create_app(configured, production_analyzer=analyzer)), artifact_body


def test_schema_1_1_subject_gated_production_analyzer_contract(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    client, artifact = _subject_gated_setup(tmp_path, manifest)
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["subjectSegmentation"]["subjectMaskSha256"] == artifact["subjectSegmentation"]["goldenMasks"][0]["maskSha256"]

    response = _post(client, manifest, png_bytes)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schemaVersion"] == "1.1"
    subject = body["analysis"]["subjectSegmentationEvidence"]
    assert subject["subjectMaskSha256"] == artifact["subjectSegmentation"]["goldenMasks"][0]["maskSha256"].removeprefix("sha256:")
    spatial = body["analysis"]["spatialDifferenceEvidence"]
    assert spatial["generationMethod"] == "SUBJECT_GATED_ROI_TILED_PATCH_DISTANCE"
    assert spatial["candidateFilter"]["retainedComponentCount"] == len(spatial["regions"])
    assert "suppressedByBackgroundCount" in spatial["candidateFilter"]
    assert hashlib.sha256(base64.b64decode(spatial["rawThresholdMaskPngBase64"])).hexdigest() == spatial["rawThresholdMaskSha256"]


def test_engineering_auto_subject_mask_is_rejected_in_production_mode(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    client, _ = _subject_gated_setup(tmp_path, manifest, engineering=False)
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["detail"]["reason"] == "SUBJECT_MASK_NOT_APPROVED"


def test_end_to_end_response_omits_spatial_evidence_generation_fields_when_unavailable(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    from .test_api import DeterministicEmbedder
    client, _ = _spatial_production_setup(tmp_path, manifest, embedder=DeterministicEmbedder())
    response = _post(client, manifest, png_bytes)
    assert response.status_code == 200, response.text
    body = response.json()
    evidence = body["analysis"]["spatialDifferenceEvidence"]
    assert evidence["state"] == "UNAVAILABLE"
    assert evidence["reasonCode"] == "PATCH_EMBEDDER_NOT_AVAILABLE"
    assert "generationMethod" not in evidence
    assert "regions" not in evidence
    assert "mapPngBase64" not in evidence


def test_end_to_end_response_includes_patch_distance_evidence_when_available(tmp_path, png_bytes, manifest_factory):
    manifest = manifest_factory(png_bytes)
    embedder = GridPatchEmbedder(DIFFERENT_GRID, global_vector=[0.9, 0.1])
    client, artifact_body = _spatial_production_setup(
        tmp_path, manifest,
        spatial_policy={"anomalyDistanceThreshold": 0.5, "minRegionAreaRatio": 0.01, "maxRegions": 8},
        embedder=embedder,
        golden_values=[0.9, 0.1],
        golden_patch={"patchValues": IDENTICAL_GRID, "patchGridHeight": 2, "patchGridWidth": 2},
    )
    response = _post(client, manifest, png_bytes)
    assert response.status_code == 200, response.text
    body = response.json()
    normalization = body["normalization"]
    assert normalization["canonicalWidth"] == artifact_body["targetAlignment"]["canonicalWidth"]
    assert normalization["canonicalHeight"] == artifact_body["targetAlignment"]["canonicalHeight"]
    assert base64.b64decode(normalization["canonicalImagePngBase64"]) == b"canonical-png"
    assert base64.b64decode(normalization["goldenCanonicalImagePngBase64"]) == png_bytes_for_reference()
    assert normalization["canonicalSha256"] == hashlib.sha256(b"canonical-png").hexdigest()
    assert normalization["goldenCanonicalSha256"] == hashlib.sha256(png_bytes_for_reference()).hexdigest()
    evidence = body["analysis"]["spatialDifferenceEvidence"]
    assert evidence["state"] == "AVAILABLE"
    assert evidence["generationMethod"] == "PATCH_DISTANCE"
    assert evidence["disclaimerCode"] == "DIFFERENCE_NOT_DEFECT_PROOF"
    assert "reasonCode" not in evidence
    map_bytes = base64.b64decode(evidence["mapPngBase64"])
    assert hashlib.sha256(map_bytes).hexdigest() == evidence["mapSha256"]


class TestReadinessCaching:
    def test_positive_readiness_is_cached_and_avoids_rehashing(self, tmp_path, manifest_factory, png_bytes, monkeypatch):
        manifest = manifest_factory(png_bytes)
        # Build the analyzer directly for white-box access to the cache.
        weights = tmp_path / "weights2.pth"
        weights.write_bytes(b"reviewed-model-weights")
        repo = tmp_path / "dinov2-2"
        repo.mkdir()
        (repo / "hubconf.py").write_text("# reviewed\n", encoding="utf-8")
        repository_version = digest_directory(repo)
        component = manifest["executionBundle"]["goldenSetVersion"]
        artifact_body = {
            "schemaVersion": "1.1", "recipeId": manifest["recipeId"], "machineId": manifest["machineId"],
            "boardId": manifest["boardId"], "goldenSetVersion": component,
            "normalizationPipelineVersion": manifest["executionBundle"]["normalizationPipelineVersion"],
            "analyzerModelVersion": manifest["executionBundle"]["analyzerModelVersion"],
            "decisionPolicyVersion": manifest["executionBundle"]["decisionPolicyVersion"],
            "analyzerRuntimeVersion": RUNTIME_DIGEST,
            "modelRepositoryVersion": repository_version,
            "boardInstallationVersion": manifest["executionBundle"]["boardInstallationVersion"],
            "modelWeightsSha256": "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest(),
            "board": {
                "squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
                "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896,
            },
            "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
            "targetAlignment": target_alignment(png_bytes_for_reference()),
            "goldenEmbeddings": [{"id": "g", "sourceSha256": component, "values": [1.0, 0.0]}],
        }
        artifact = tmp_path / "artifact2.json"
        artifact.write_text(json.dumps(artifact_body, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        artifact_digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
        configured = replace(
            settings(tmp_path, enabled=False), artifact_manifest=artifact,
            artifact_package_digest=artifact_digest, model_repo=repo, model_weights=weights,
            model_repository_version=repository_version,
        )
        from .test_api import DeterministicEmbedder
        analyzer = ProductionAnalyzer(configured, normalizer=AcceptedNormalizer(), embedder=DeterministicEmbedder())

        calls = {"count": 0}
        real_digest_directory = digest_directory

        def counting_digest_directory(path):
            calls["count"] += 1
            return real_digest_directory(path)

        monkeypatch.setattr("phone_dino.production.digest_directory", counting_digest_directory)
        first = analyzer.readiness()
        second = analyzer.readiness()
        third = analyzer.readiness()
        assert first == (True, None)
        assert second == (True, None) and third == (True, None)
        assert calls["count"] == 1, "positive readiness must be cached, not re-hashed every call"

    def test_negative_readiness_is_not_cached(self, tmp_path):
        configured = Settings(
            service_token="secret", fixture_enabled=False, fixture_dir=None,
            artifact_manifest=None, artifact_package_digest=None, model_repo=None, model_weights=None,
            max_image_bytes=1024, max_image_pixels=1000, max_image_width=100, max_image_height=100,
        )
        analyzer = ProductionAnalyzer(configured)
        assert analyzer.readiness() == (False, "ARTIFACT_MANIFEST_NOT_AVAILABLE")
        assert analyzer.readiness() == (False, "ARTIFACT_MANIFEST_NOT_AVAILABLE")
        assert analyzer._readiness_cache is None


def test_analysis_timeout_returns_504(tmp_path, manifest_factory, png_bytes):
    class SlowNormalizer:
        def normalize(self, image, artifact):
            time.sleep(0.3)
            return NormalizedCapture(
                rgb="canonical", encoded=b"canonical-png",
                alignment=AlignmentObservation(
                    state="ALIGNED", method="TARGET_AFFINE", targetRelative=True, inlierCount=20,
                    inlierRatio=0.8, reprojectionErrorPx=0.5, coverageRatio=0.4,
                    transformWithinBounds=True, inspectionMaskApplied=True,
                ),
            )

    manifest = manifest_factory(png_bytes)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"reviewed-model-weights")
    repo = tmp_path / "dinov2"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# reviewed\n", encoding="utf-8")
    repository_version = digest_directory(repo)
    component = manifest["executionBundle"]["goldenSetVersion"]
    artifact_body = {
        "schemaVersion": "1.1", "recipeId": manifest["recipeId"], "machineId": manifest["machineId"],
        "boardId": manifest["boardId"], "goldenSetVersion": component,
        "normalizationPipelineVersion": manifest["executionBundle"]["normalizationPipelineVersion"],
        "analyzerModelVersion": manifest["executionBundle"]["analyzerModelVersion"],
        "decisionPolicyVersion": manifest["executionBundle"]["decisionPolicyVersion"],
        "analyzerRuntimeVersion": RUNTIME_DIGEST,
        "modelRepositoryVersion": repository_version,
        "boardInstallationVersion": manifest["executionBundle"]["boardInstallationVersion"],
        "modelWeightsSha256": "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest(),
        "board": {
            "squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
            "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896,
        },
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
        "targetAlignment": target_alignment(png_bytes_for_reference()),
        "goldenEmbeddings": [{"id": "g", "sourceSha256": component, "values": [1.0, 0.0]}],
    }
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(artifact_body, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    artifact_digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest["artifactPackageDigest"] = artifact_digest
    manifest["simulation"] = False
    configured = replace(
        settings(tmp_path, enabled=False), artifact_manifest=artifact,
        artifact_package_digest=artifact_digest, model_repo=repo, model_weights=weights,
        model_repository_version=repository_version, analysis_timeout_seconds=0.05,
    )
    from .test_api import DeterministicEmbedder
    analyzer = ProductionAnalyzer(configured, normalizer=SlowNormalizer(), embedder=DeterministicEmbedder())
    client = TestClient(create_app(configured, production_analyzer=analyzer))
    response = _post(client, manifest, png_bytes)
    assert response.status_code == 504
    assert response.json()["detail"] == "ANALYSIS_TIMEOUT"


@pytest.mark.skipif(
    not __import__("pathlib").Path(r"C:\code\claude\phone_dino\runtime\models\dinov2_vits14_pretrain.pth").is_file(),
    reason="Real DINOv2 weights are not present on disk; this is a real-model smoke test, not fixture-based.",
)
def test_real_model_patch_embedding_shape_and_self_consistency():
    """Runs the actual DINOv2 ViT-S/14 forward pass; proves embed_with_patches works
    against the real model, not just protocol-shaped stubs."""
    from pathlib import Path
    import numpy as np
    from PIL import Image as PILImage
    from phone_dino.engines import LocalDinoV2Adapter
    from phone_dino.production import DinoV2Embedder

    # Earlier tests in this process set PIL's global MAX_IMAGE_PIXELS via
    # decoder.py's per-request bomb guard; this raw in-memory array is safe.
    PILImage.MAX_IMAGE_PIXELS = None
    repo = Path(r"C:\code\claude\phone_dino\runtime\models\dinov2")
    weights = Path(r"C:\code\claude\phone_dino\runtime\models\dinov2_vits14_pretrain.pth")
    adapter = LocalDinoV2Adapter(repo, weights)
    ready, reason = adapter.readiness()
    assert ready, reason
    embedder = DinoV2Embedder(adapter)

    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    result = embedder.embed_with_patches(rgb)
    assert len(result.global_vector) == 384
    assert result.grid_height == 16 and result.grid_width == 16
    assert len(result.patch_grid) == 256
    assert all(len(patch) == 384 for patch in result.patch_grid)
    assert all(math.isfinite(v) for v in result.global_vector)

    golden = GoldenEmbedding(
        id="G-1", sourceSha256="sha256:" + "0" * 64, values=result.global_vector,
        patchValues=result.patch_grid, patchGridHeight=16, patchGridWidth=16,
    )
    evidence = _spatial_difference_evidence(result, golden, POLICY, 320, 240)
    assert evidence.state == "AVAILABLE"
    assert evidence.regions == [], "comparing an image against its own Golden embedding must not fabricate anomalies"
