from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class AnalyzerEngine(Protocol):
    """Boundary for production analyzers; engines return observations, never decisions."""

    def readiness(self) -> tuple[bool, str | None]: ...


@dataclass(frozen=True, slots=True)
class LocalDinoV2Adapter:
    """Local-only DINOv2 loader.

    No network-backed torch.hub identifier is accepted. Loading is deliberately
    explicit so normal service import and engineering fixtures never execute model
    code or download weights.
    """

    repository: Path | None
    weights: Path | None

    def readiness(self) -> tuple[bool, str | None]:
        if self.repository is None or not self.repository.is_dir():
            return False, "MODEL_REPOSITORY_NOT_AVAILABLE"
        if not (self.repository / "hubconf.py").is_file():
            return False, "MODEL_REPOSITORY_INVALID"
        if self.weights is None or not self.weights.is_file():
            return False, "MODEL_WEIGHTS_NOT_AVAILABLE"
        return True, None

    def smoke_load(self) -> object:
        """Load ViT-S/14 from explicit local paths for an operator smoke test."""
        ready, reason = self.readiness()
        if not ready:
            raise RuntimeError(reason)
        import torch

        model = torch.hub.load(str(self.repository), "dinov2_vits14", source="local", pretrained=False)
        checkpoint = torch.load(str(self.weights), map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

