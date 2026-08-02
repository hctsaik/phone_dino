"""Explicitly gated production-contract app for cross-repository synthetic E2E.

The decoder, Still Gate, ChArUco plane normalizer, target aligner, artifact and
request binding are production implementations.  Only the expensive DINO model
is replaced by a deterministic image embedder.
"""
from __future__ import annotations

import os

from phone_dino.app import create_app
from phone_dino.config import Settings
from phone_dino.production import ProductionAnalyzer


class DeterministicSyntheticEmbedder:
    def embed(self, rgb):
        import numpy as np

        pixels = np.asarray(rgb, dtype=np.float64) / 255.0
        return [1.0, float(pixels.mean()), float(pixels.std())]


if os.environ.get("PHONE_DINO_ENABLE_SYNTHETIC_CONTRACT_APP") != "true":
    raise RuntimeError("Synthetic contract app requires PHONE_DINO_ENABLE_SYNTHETIC_CONTRACT_APP=true")

settings = Settings.from_env()
if settings.fixture_enabled:
    raise RuntimeError("Synthetic contract app refuses engineering fixture mode")

app = create_app(
    settings,
    production_analyzer=ProductionAnalyzer(settings, embedder=DeterministicSyntheticEmbedder()),
)
