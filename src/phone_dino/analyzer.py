from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import Settings
from .contracts import (
    AnalysisObservation,
    AnalysisState,
    AlignmentObservation,
    AnalyzeObservation,
    AnalyzeRequest,
    CaptureAssessment,
    CaptureState,
    NormalizationObservation,
    ResolvedVersions,
)
from .decoder import DecodedImage
from . import __version__


RUNTIME_DIGEST = "sha256:" + hashlib.sha256(f"phone_dino:{__version__}".encode()).hexdigest()
NORMALIZATION_DIGEST = "sha256:" + hashlib.sha256(b"decode_only:0.1.0").hexdigest()


class FixtureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    outcome: Literal["pass", "fail", "review", "recapture", "system-error"]
    global_distance: float | None = Field(default=None, alias="globalDistance", ge=0, le=2)
    nearest_golden_distance: float | None = Field(default=None, alias="nearestGoldenDistance", ge=0, le=2)
    uncertainty: float | None = Field(default=None, ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list, alias="reasonCodes", max_length=32)


def _load_fixture(settings: Settings, raw_sha256: str) -> FixtureRecord:
    if not settings.fixture_enabled or settings.fixture_dir is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ANALYZER_ARTIFACT_NOT_AVAILABLE")
    path = settings.fixture_dir / f"{raw_sha256}.json"
    try:
        path.relative_to(settings.fixture_dir)
        return FixtureRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValidationError, ValueError):
        if settings.fixture_fallback_enabled:
            # Development-only fallback: every uploaded image can exercise the
            # real HTTP contract without silently enabling this in production.
            # Keep the unmapped engineering sample in the REVIEW band. A
            # deterministic FAIL here would make every new phone photo look
            # defective before a real model is installed.
            distance = 0.42
            return FixtureRecord(
                outcome="review",
                globalDistance=round(distance, 4),
                nearestGoldenDistance=round(distance, 4),
                uncertainty=0.2,
                # ACCEPTED captures must carry no capture reason codes; the
                # simulation boundary is already visible in the envelope.
                reasonCodes=[],
            )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ENGINEERING_FIXTURE_NOT_AVAILABLE") from None


def analyze_fixture(
    request: AnalyzeRequest,
    image: DecodedImage,
    settings: Settings,
) -> AnalyzeObservation:
    fixture = _load_fixture(settings, request.raw_sha256)
    if fixture.outcome == "system-error":
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ENGINEERING_FIXTURE_SYSTEM_ERROR")

    recapture = fixture.outcome == "recapture"
    if not recapture and fixture.global_distance is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ENGINEERING_FIXTURE_DISTANCE_MISSING")

    identity = "|".join((request.session_id, str(request.capture_ordinal), request.raw_sha256, request.execution_bundle_digest, RUNTIME_DIGEST))
    analysis_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    canonical_sha = hashlib.sha256(image.data).hexdigest()
    return AnalyzeObservation(
        schemaVersion="1.0",
        requestId=request.request_id,
        analysisId=analysis_id,
        rawSha256=request.raw_sha256,
        simulation=True,
        resolvedVersions=ResolvedVersions(
            executionBundleDigest=request.execution_bundle_digest,
            artifactPackageDigest=request.artifact_package_digest,
            analyzerModelVersion=request.execution_bundle.analyzer_model_version,
            analyzerRuntimeVersion=RUNTIME_DIGEST,
            normalizationRuntimeVersion=NORMALIZATION_DIGEST,
        ),
        captureAssessment=CaptureAssessment(
            state=CaptureState.RECAPTURE_REQUIRED if recapture else CaptureState.ACCEPTED,
            reasonCodes=fixture.reason_codes or (["ENGINEERING_FIXTURE_RECAPTURE"] if recapture else []),
        ),
        normalization=None if recapture else NormalizationObservation(
            canonicalSha256=canonical_sha,
            alignment=AlignmentObservation(
                state="ALIGNED", method="SIMULATION_FIXTURE", targetRelative=True,
                inlierCount=0, inlierRatio=1.0, reprojectionErrorPx=0.0,
                coverageRatio=1.0, transformWithinBounds=True, inspectionMaskApplied=True,
            ),
        ),
        analysis=AnalysisObservation(
            state=AnalysisState.NOT_RUN if recapture else AnalysisState.RUN,
            metric=None if recapture else "cosine_distance",
            globalDistance=None if recapture else fixture.global_distance,
            nearestGoldenId=None if recapture else "ENGINEERING-GOLDEN",
            nearestGoldenDistance=None if recapture else (
                fixture.nearest_golden_distance
                if fixture.nearest_golden_distance is not None
                else fixture.global_distance
            ),
            uncertainty=None if recapture else (fixture.uncertainty if fixture.uncertainty is not None else 0.0),
        ),
    )
