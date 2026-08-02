from __future__ import annotations

import base64
import hashlib
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

np = pytest.importorskip("numpy", reason="production vision extra is not installed")
cv2 = pytest.importorskip("cv2", reason="production OpenCV extra is not installed")

from phone_dino.artifacts import TargetAlignmentPolicy
from phone_dino.production import OpenCvTargetAligner


def _reference() -> np.ndarray:
    rng = np.random.default_rng(20260801)
    image = np.full((300, 400, 3), 235, dtype=np.uint8)
    # Stable structure is deliberately spread around the future inspection area.
    for _ in range(350):
        x, y = int(rng.integers(10, 390)), int(rng.integers(10, 290))
        color = tuple(int(value) for value in rng.integers(0, 180, 3))
        cv2.circle(image, (x, y), int(rng.integers(2, 5)), color, -1)
    cv2.rectangle(image, (145, 105), (255, 195), (20, 20, 20), -1)
    cv2.putText(image, "INSPECT", (150, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    return image


def _policy(reference: np.ndarray, **updates) -> TargetAlignmentPolicy:
    ok, encoded = cv2.imencode(".png", reference)
    assert ok
    body = {
        "method": "TARGET_AFFINE",
        "referenceImageBase64": base64.b64encode(encoded.tobytes()).decode(),
        "referenceImageSha256": "sha256:" + hashlib.sha256(encoded.tobytes()).hexdigest(),
        "canonicalWidth": 400, "canonicalHeight": 300,
        "alignmentRegions": [
            {"x": 0, "y": 0, "width": 400, "height": 90},
            {"x": 0, "y": 210, "width": 400, "height": 90},
        ],
        "heldOutRegions": [
            {"x": 0, "y": 100, "width": 130, "height": 100},
            {"x": 270, "y": 100, "width": 130, "height": 100},
        ],
        "inspectionRegions": [{"x": 140, "y": 100, "width": 120, "height": 100}],
        "minMatches": 20, "minInliers": 15, "minInlierRatio": 0.5, "minCoverageRatio": 0.12,
        "maxReprojectionErrorPx": 2.5, "minScale": 0.8, "maxScale": 1.2,
        "maxRotationDegrees": 15.0, "maxShear": 0.03, "maxTranslationPx": 250.0,
        "maxSecondaryInlierRatio": 0.35, "minHeldOutMatches": 12, "maxHeldOutReprojectionErrorPx": 3.0,
    }
    body.update(updates)
    return TargetAlignmentPolicy.model_validate(body)


def _move_target(reference: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(reference, matrix, (620, 480), borderValue=(250, 250, 250))


def test_target_alignment_is_independent_of_board_relative_offset_and_masks_inspection():
    reference = _reference()
    current = _move_target(reference, np.float32([[1, 0, 91], [0, 1, 67]]))
    # Simulate a real defect: it must not contribute descriptors used by alignment.
    current[172:262, 236:346] = (0, 0, 255)

    result = OpenCvTargetAligner().align(
        cv2.cvtColor(current, cv2.COLOR_BGR2RGB), SimpleNamespace(target_alignment=_policy(reference))
    )

    assert result.reason_codes == ()
    assert result.alignment is not None
    assert result.alignment.state == "ALIGNED"
    assert result.alignment.target_relative is True
    assert result.alignment.inspection_mask_applied is True
    assert result.alignment.transform_within_bounds is True
    assert result.alignment.inlier_count >= 15
    assert result.alignment.coverage_ratio >= 0.12
    assert result.rgb.shape == reference.shape


def test_target_missing_fails_closed_without_canonical_pixels():
    reference = _reference()
    blank = np.full((480, 620, 3), 255, dtype=np.uint8)

    result = OpenCvTargetAligner().align(
        cv2.cvtColor(blank, cv2.COLOR_BGR2RGB), SimpleNamespace(target_alignment=_policy(reference))
    )

    assert result.rgb is None and result.encoded == b""
    assert result.reason_codes == ("TARGET_NOT_FOUND",)
    assert result.alignment is not None and result.alignment.state == "NOT_ALIGNED"


def test_clustered_evidence_and_out_of_bounds_rotation_fail_closed():
    reference = _reference()
    shifted = _move_target(reference, np.float32([[1, 0, 70], [0, 1, 50]]))
    clustered = OpenCvTargetAligner().align(
        cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB),
        SimpleNamespace(target_alignment=_policy(reference, minCoverageRatio=0.99)),
    )
    assert clustered.rgb is None
    assert "TARGET_INLIERS_CLUSTERED" in clustered.reason_codes

    center = (reference.shape[1] / 2, reference.shape[0] / 2)
    rotated_matrix = cv2.getRotationMatrix2D(center, 28.0, 1.0)
    rotated_matrix[:, 2] += (100, 80)
    rotated = _move_target(reference, rotated_matrix.astype(np.float32))
    bounded = OpenCvTargetAligner().align(
        cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), SimpleNamespace(target_alignment=_policy(reference))
    )
    assert bounded.rgb is None
    assert "TARGET_TRANSFORM_OUT_OF_BOUNDS" in bounded.reason_codes
    assert bounded.alignment is not None and bounded.alignment.transform_within_bounds is False


def test_excessive_translation_duplicate_target_and_held_out_mismatch_fail_closed():
    reference = _reference()
    translated = cv2.warpAffine(
        reference, np.float32([[1, 0, 280], [0, 1, 100]]), (850, 520), borderValue=(250, 250, 250)
    )
    translation_result = OpenCvTargetAligner().align(
        cv2.cvtColor(translated, cv2.COLOR_BGR2RGB), SimpleNamespace(target_alignment=_policy(reference))
    )
    assert "TARGET_TRANSFORM_OUT_OF_BOUNDS" in translation_result.reason_codes

    duplicate = np.full((480, 930, 3), 250, dtype=np.uint8)
    duplicate[50:350, 40:440] = reference
    duplicate[50:350, 490:890] = reference
    duplicate_result = OpenCvTargetAligner().align(
        cv2.cvtColor(duplicate, cv2.COLOR_BGR2RGB),
        SimpleNamespace(target_alignment=_policy(reference, maxTranslationPx=600.0)),
    )
    assert duplicate_result.rgb is None
    assert "TARGET_AMBIGUOUS" in duplicate_result.reason_codes

    mismatched = _move_target(reference, np.float32([[1, 0, 70], [0, 1, 50]]))
    # Remove only the independent held-out regions after applying target offset.
    mismatched[150:250, 70:200] = 255
    mismatched[150:250, 340:470] = 255
    held_out_result = OpenCvTargetAligner().align(
        cv2.cvtColor(mismatched, cv2.COLOR_BGR2RGB), SimpleNamespace(target_alignment=_policy(reference))
    )
    assert held_out_result.rgb is None
    assert "TARGET_PARALLAX_OR_MISMATCH" in held_out_result.reason_codes


def test_alignment_is_repeatable_across_process_history_and_concurrent_calls():
    reference = _reference()
    current = _move_target(reference, np.float32([[1, 0, 91], [0, 1, 67]]))
    rgb = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
    artifact = SimpleNamespace(target_alignment=_policy(reference))
    aligner = OpenCvTargetAligner()

    def observe(_index):
        result = aligner.align(rgb, artifact)
        return result.reason_codes, result.encoded, result.alignment.model_dump_json() if result.alignment else None

    sequential = [observe(index) for index in range(10)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        concurrent = list(pool.map(observe, range(12)))
    assert all(value == sequential[0] for value in (*sequential, *concurrent))
