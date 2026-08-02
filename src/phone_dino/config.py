from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in TRUE_VALUES


@dataclass(frozen=True, slots=True)
class Settings:
    service_token: str | None
    fixture_enabled: bool
    fixture_dir: Path | None
    artifact_manifest: Path | None
    artifact_package_digest: str | None
    model_repo: Path | None
    model_weights: Path | None
    max_image_bytes: int
    max_image_pixels: int
    max_image_width: int
    max_image_height: int
    model_repository_version: str | None = None
    fixture_fallback_enabled: bool = False
    allow_target_only_alignment: bool = False
    device: str = "cpu"
    analysis_timeout_seconds: float = 8.0

    @classmethod
    def from_env(cls) -> "Settings":
        fixture_dir = os.getenv("PHONE_DINO_FIXTURE_DIR")
        artifact_manifest = os.getenv("PHONE_DINO_ARTIFACT_MANIFEST")
        model_repo = os.getenv("PHONE_DINO_MODEL_REPO")
        model_weights = os.getenv("PHONE_DINO_MODEL_WEIGHTS")
        return cls(
            service_token=os.getenv("PHONE_DINO_SERVICE_TOKEN"),
            fixture_enabled=_bool_env("PHONE_DINO_ENABLE_ENGINEERING_FIXTURES"),
            fixture_fallback_enabled=_bool_env("PHONE_DINO_ALLOW_UNMAPPED_FIXTURE"),
            fixture_dir=Path(fixture_dir).resolve() if fixture_dir else None,
            artifact_manifest=Path(artifact_manifest).resolve() if artifact_manifest else None,
            artifact_package_digest=os.getenv("PHONE_DINO_ARTIFACT_PACKAGE_DIGEST"),
            model_repo=Path(model_repo).resolve() if model_repo else None,
            model_weights=Path(model_weights).resolve() if model_weights else None,
            max_image_bytes=int(os.getenv("PHONE_DINO_MAX_IMAGE_BYTES", str(12 * 1024 * 1024))),
            max_image_pixels=int(os.getenv("PHONE_DINO_MAX_IMAGE_PIXELS", "24000000")),
            max_image_width=int(os.getenv("PHONE_DINO_MAX_IMAGE_WIDTH", "8000")),
            max_image_height=int(os.getenv("PHONE_DINO_MAX_IMAGE_HEIGHT", "8000")),
            model_repository_version=os.getenv("PHONE_DINO_MODEL_REPOSITORY_VERSION"),
            allow_target_only_alignment=_bool_env("PHONE_DINO_ALLOW_TARGET_ONLY_ALIGNMENT"),
            device=os.getenv("PHONE_DINO_DEVICE", "cpu"),
            analysis_timeout_seconds=float(os.getenv("PHONE_DINO_ANALYSIS_TIMEOUT_SECONDS", "8.0")),
        )
