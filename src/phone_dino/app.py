from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from . import __version__
from .analyzer import analyze_fixture
from .config import Settings
from .contracts import (
    AnalyzeObservation, AnalyzeRequest, GoldenDimensionObservation, GoldenDimensionRequest,
)
from .decoder import read_and_validate_image
from .artifacts import ArtifactError
from .production import ProductionAnalyzer
from .security import sha256_hex, verify_bundle_digest


logger = logging.getLogger("phone_dino")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _log(event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}, separators=(",", ":"), sort_keys=True))


def _artifact_state(settings: Settings, analyzer: ProductionAnalyzer | None = None) -> tuple[bool, str | None]:
    if settings.engineering_real_model_enabled:
        return (analyzer or ProductionAnalyzer(settings)).readiness()
    if settings.fixture_enabled:
        ready = settings.fixture_dir is not None and settings.fixture_dir.is_dir()
        return ready, None if ready else "FIXTURE_DIRECTORY_NOT_AVAILABLE"
    return (analyzer or ProductionAnalyzer(settings)).readiness()


def create_app(settings: Settings | None = None, production_analyzer: ProductionAnalyzer | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()
    production = production_analyzer or ProductionAnalyzer(resolved)
    expected_simulation = resolved.fixture_enabled or resolved.engineering_real_model_enabled

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not resolved.fixture_enabled or resolved.engineering_real_model_enabled:
            warm_up_started = time.perf_counter()
            ready, reason = await run_in_threadpool(production.warm_up)
            _log(
                "warm_up_completed", ready=ready, reason=reason,
                durationMs=round((time.perf_counter() - warm_up_started) * 1000),
            )
        yield

    app = FastAPI(title="phone_dino", version=__version__, lifespan=lifespan)
    app.state.settings = resolved

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = resolved.service_token
        if not expected:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="SERVICE_TOKEN_NOT_CONFIGURED")
        prefix = "Bearer "
        if authorization is None or not authorization.startswith(prefix):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="UNAUTHORIZED", headers={"WWW-Authenticate": "Bearer"})
        if not hmac.compare_digest(authorization[len(prefix):], expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="UNAUTHORIZED", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "alive", "version": __version__}

    @app.get("/readyz")
    async def readyz() -> dict[str, object]:
        token_ready = bool(resolved.service_token)
        artifact_ready, reason = _artifact_state(resolved, production)
        ready = token_ready and artifact_ready
        body = {
            "status": "ready" if ready else "not_ready",
            "simulation": expected_simulation,
            "analysisMode": "ENGINEERING_REAL_DINO" if resolved.engineering_real_model_enabled else ("FIXTURE" if resolved.fixture_enabled else "PRODUCTION"),
        }
        if ready and (not resolved.fixture_enabled or resolved.engineering_real_model_enabled):
            body.update(production.readiness_metadata())
        elif ready:
            body.update({"supportedSchemas": ["1.0"], "capabilities": []})
        if not token_ready:
            body["reason"] = "SERVICE_TOKEN_NOT_CONFIGURED"
        elif reason:
            body["reason"] = reason
        if not ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=body)
        return body

    @app.post(
        "/internal/v1/analyze",
        response_model=AnalyzeObservation,
        response_model_by_alias=True,
        response_model_exclude_none=True,
    )
    async def analyze(
        manifest: UploadFile = File(...),
        image: UploadFile = File(...),
        _: None = Depends(authorize),
    ) -> AnalyzeObservation:
        started = time.perf_counter()
        if manifest.content_type != "application/json":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MANIFEST_CONTENT_TYPE_INVALID")
        manifest_bytes = await manifest.read(64 * 1024 + 1)
        if len(manifest_bytes) > 64 * 1024:
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="MANIFEST_BYTES_LIMIT_EXCEEDED")
        try:
            request = AnalyzeRequest.model_validate_json(manifest_bytes)
        except ValidationError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=json.loads(exc.json())) from exc

        verify_bundle_digest(request)
        if request.deadline <= datetime.now(timezone.utc):
            raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="ANALYSIS_DEADLINE_EXPIRED")
        if request.simulation is not expected_simulation:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="SIMULATION_MODE_MISMATCH")
        if (not resolved.fixture_enabled or resolved.engineering_real_model_enabled) and resolved.artifact_package_digest is not None:
            if not hmac.compare_digest(request.artifact_package_digest, resolved.artifact_package_digest):
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="ARTIFACT_PACKAGE_DIGEST_MISMATCH")
        decoded = await read_and_validate_image(image, request.content_type, resolved)
        if not hmac.compare_digest(sha256_hex(decoded.data), request.raw_sha256):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="RAW_SHA256_MISMATCH")

        if resolved.engineering_real_model_enabled:
            ready, reason = _artifact_state(resolved, production)
            if not ready:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=reason)
            remaining = (request.deadline - datetime.now(timezone.utc)).total_seconds()
            bounded_timeout = max(0.1, min(remaining, resolved.analysis_timeout_seconds))
            try:
                result = await asyncio.wait_for(run_in_threadpool(production.analyze, request, decoded), timeout=bounded_timeout)
            except ArtifactError as exc:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
            except asyncio.TimeoutError as exc:
                raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="ANALYSIS_TIMEOUT") from exc
        elif resolved.fixture_enabled:
            result = analyze_fixture(request, decoded, resolved)
        else:
            ready, reason = _artifact_state(resolved, production)
            if not ready:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=reason)
            remaining = (request.deadline - datetime.now(timezone.utc)).total_seconds()
            bounded_timeout = max(0.1, min(remaining, resolved.analysis_timeout_seconds))
            try:
                result = await asyncio.wait_for(run_in_threadpool(production.analyze, request, decoded), timeout=bounded_timeout)
            except ArtifactError as exc:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
            except asyncio.TimeoutError as exc:
                raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="ANALYSIS_TIMEOUT") from exc

        _log(
            "analysis_completed",
            requestId=request.request_id,
            correlationId=request.correlation_id,
            analysisId=result.analysis_id,
            rawSha256=request.raw_sha256,
            simulation=result.simulation,
            durationMs=round((time.perf_counter() - started) * 1000),
        )
        return result

    @app.post(
        "/internal/v1/golden-dimension-assessments",
        response_model=GoldenDimensionObservation,
        response_model_by_alias=True,
        response_model_exclude_none=True,
    )
    async def assess_golden_dimensions(
        manifest: UploadFile = File(...),
        image: UploadFile = File(...),
        _: None = Depends(authorize),
    ) -> GoldenDimensionObservation:
        started = time.perf_counter()
        if manifest.content_type != "application/json":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MANIFEST_CONTENT_TYPE_INVALID")
        manifest_bytes = await manifest.read(16 * 1024 + 1)
        if len(manifest_bytes) > 16 * 1024:
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="MANIFEST_BYTES_LIMIT_EXCEEDED")
        try:
            request = GoldenDimensionRequest.model_validate_json(manifest_bytes)
        except ValidationError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=json.loads(exc.json())) from exc
        decoded = await read_and_validate_image(image, request.content_type, resolved)
        if not hmac.compare_digest(sha256_hex(decoded.data), request.raw_sha256):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="RAW_SHA256_MISMATCH")
        ready, reason = _artifact_state(resolved, production)
        if not ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=reason)
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(production.measure_golden_dimensions, request, decoded),
                timeout=resolved.analysis_timeout_seconds,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="GOLDEN_DIMENSION_TIMEOUT") from exc
        _log(
            "golden_dimension_assessment_completed",
            requestId=request.request_id,
            recipeId=request.recipe_id,
            rawSha256=request.raw_sha256,
            state=result.physical_dimensions.state,
            reasonCode=result.physical_dimensions.reason_code,
            durationMs=round((time.perf_counter() - started) * 1000),
        )
        return result

    return app


app = create_app()
