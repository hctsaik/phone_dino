from __future__ import annotations

from phone_dino.app import create_app
from phone_dino.config import Settings
from phone_dino.contracts import AlignmentObservation
from phone_dino.production import NormalizedCapture, ProductionAnalyzer


class ContractNormalizer:
    def normalize(self, image, artifact):
        return NormalizedCapture(
            rgb="contract-canonical", encoded=b"contract-canonical-png",
            alignment=AlignmentObservation(
                state="ALIGNED", method="TARGET_AFFINE", targetRelative=True, inlierCount=20,
                inlierRatio=0.8, reprojectionErrorPx=0.5, coverageRatio=0.4,
                transformWithinBounds=True, inspectionMaskApplied=True,
            ),
        )


class ContractEmbedder:
    def embed(self, rgb):
        if rgb != "contract-canonical":
            raise RuntimeError("CONTRACT_CANONICAL_INVALID")
        return [1.0, 0.0, 0.0]


settings = Settings.from_env()
if settings.fixture_enabled:
    raise RuntimeError("Production contract app refuses engineering fixtures")
app = create_app(
    settings,
    production_analyzer=ProductionAnalyzer(
        settings, normalizer=ContractNormalizer(), embedder=ContractEmbedder()
    ),
)
