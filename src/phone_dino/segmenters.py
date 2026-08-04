from __future__ import annotations

import math
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SubjectMaskPrediction:
    """One canonical-space subject proposal produced during artifact compilation."""

    mask: object
    quality_score: float
    prompt_box_xyxy: tuple[int, int, int, int]
    foreground_ratio: float


class SubjectSegmenter(Protocol):
    def segment(
        self,
        canonical_rgb: object,
        prompt_box_xyxy: tuple[int, int, int, int],
        *,
        min_foreground_ratio: float,
        max_foreground_ratio: float,
        min_quality_score: float,
    ) -> SubjectMaskPrediction: ...


class MobileSamSegmenter:
    """Pinned local MobileSAM ViT-T adapter for compile-time and runtime masks.

    The adapter never downloads code or weights. Runtime use is enabled only
    by an artifact contract that pins the same reviewed repository and weights
    used to generate the Golden masks.
    """

    def __init__(self, repository: Path, weights: Path, device: str = "cpu"):
        self._repository = repository.resolve()
        self._weights = weights.resolve()
        self._device = device
        self._predictor: object | None = None
        self._lock = Lock()

    def _load(self) -> object:
        if self._predictor is not None:
            return self._predictor
        repository = str(self._repository)
        if repository not in sys.path:
            sys.path.insert(0, repository)
        try:
            module = importlib.import_module("mobile_sam")
            module_path = Path(module.__file__).resolve() if module.__file__ else None
            if module_path is None:
                raise RuntimeError("MOBILE_SAM_MODULE_ORIGIN_INVALID")
            try:
                module_path.relative_to(self._repository)
            except ValueError as exc:
                raise RuntimeError("MOBILE_SAM_MODULE_ORIGIN_MISMATCH") from exc
            from mobile_sam import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError("MOBILE_SAM_DEPENDENCY_NOT_AVAILABLE") from exc
        try:
            model = sam_model_registry["vit_t"](checkpoint=str(self._weights))
            model.to(device=self._device)
            model.eval()
            self._predictor = SamPredictor(model)
        except (KeyError, OSError, RuntimeError) as exc:
            raise RuntimeError("MOBILE_SAM_MODEL_LOAD_FAILED") from exc
        return self._predictor

    def warm_up(self) -> None:
        """Load the pinned model before the service starts accepting traffic."""
        with self._lock:
            self._load()

    def segment(
        self,
        canonical_rgb: object,
        prompt_box_xyxy: tuple[int, int, int, int],
        *,
        min_foreground_ratio: float,
        max_foreground_ratio: float,
        min_quality_score: float,
    ) -> SubjectMaskPrediction:
        import numpy as np

        image = np.asarray(canonical_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError("SUBJECT_SEGMENTATION_CANONICAL_IMAGE_INVALID")
        left, top, right, bottom = prompt_box_xyxy
        height, width = image.shape[:2]
        if left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
            raise RuntimeError("SUBJECT_SEGMENTATION_PROMPT_OUT_OF_BOUNDS")
        prompt_area = float((right - left) * (bottom - top))
        with self._lock:
            predictor = self._load()
            predictor.set_image(image)  # type: ignore[attr-defined]
            masks, scores, _ = predictor.predict(  # type: ignore[attr-defined]
                box=np.asarray(prompt_box_xyxy, dtype=np.float32),
                multimask_output=True,
            )

        candidates: list[tuple[float, float, object]] = []
        for raw_mask, raw_score in zip(masks, scores, strict=True):
            score = float(raw_score)
            if not math.isfinite(score):
                continue
            mask = np.asarray(raw_mask, dtype=bool)
            if mask.shape != (height, width):
                continue
            clipped = np.zeros_like(mask)
            clipped[top:bottom, left:right] = mask[top:bottom, left:right]
            ratio = float(np.count_nonzero(clipped)) / prompt_area
            if min_foreground_ratio <= ratio <= max_foreground_ratio:
                candidates.append((score, ratio, clipped))
        if not candidates:
            raise RuntimeError("SUBJECT_MASK_AREA_OUT_OF_POLICY")
        score, ratio, mask = max(candidates, key=lambda item: (item[0], item[1]))
        if score < min_quality_score:
            raise RuntimeError("SUBJECT_MASK_QUALITY_BELOW_POLICY")
        return SubjectMaskPrediction(
            mask=mask,
            quality_score=max(0.0, min(1.0, score)),
            prompt_box_xyxy=prompt_box_xyxy,
            foreground_ratio=ratio,
        )
