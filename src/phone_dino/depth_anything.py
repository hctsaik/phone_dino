"""Optional local Depth Anything V2 adapter.

Weights are deliberately never fetched at request time.  The engineering
service accepts a locally pinned checkout and weight file through Settings;
without both, board-pose measurements continue with their explicit offset
prior.  This keeps network availability from changing a physical estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys


@dataclass(slots=True)
class DepthAnythingV2RelativeDepthEstimator:
    repository: Path
    weights: Path
    encoder: str = "vits"
    device: str = "cpu"
    input_size: int = 518
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.repository.is_dir() or not self.weights.is_file():
            raise RuntimeError("DEPTH_ANYTHING_LOCAL_ARTIFACT_UNAVAILABLE")
        if self.encoder not in {"vits", "vitb", "vitl"}:
            raise RuntimeError("DEPTH_ANYTHING_ENCODER_UNSUPPORTED")
        import torch

        repository_text = str(self.repository)
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise RuntimeError("DEPTH_ANYTHING_REPOSITORY_INVALID") from exc
        configurations = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        model = DepthAnythingV2(**configurations[self.encoder])
        state = torch.load(self.weights, map_location=self.device, weights_only=True)
        model.load_state_dict(state, strict=True)
        self._model = model.to(self.device).eval()

    def estimate_inverse_depth(self, rgb: object) -> object:
        """Return native-resolution relative inverse depth (nearer is larger)."""
        import cv2
        import numpy as np

        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError("DEPTH_ANYTHING_IMAGE_INVALID")
        # The official V2 helper expects OpenCV's BGR image convention.
        depth = self._model.infer_image(image[:, :, ::-1], input_size=self.input_size)
        result = np.asarray(depth, dtype=np.float64)
        if result.ndim != 2 or not np.all(np.isfinite(result)):
            raise RuntimeError("DEPTH_ANYTHING_OUTPUT_INVALID")
        if result.shape != image.shape[:2]:
            result = cv2.resize(result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("DEPTH_ANYTHING_OUTPUT_INVALID")
        return result
