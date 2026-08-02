from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timedelta, timezone

from PIL import Image
import pytest


@pytest.fixture
def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 8), (20, 40, 60)).save(output, "PNG")
    return output.getvalue()


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


@pytest.fixture
def manifest_factory():
    def factory(image: bytes) -> dict[str, object]:
        component = digest(b"component")
        bundle = {
            "recipeVersion": component,
            "goldenSetVersion": component,
            "capturePolicyVersion": component,
            "decisionPolicyVersion": component,
            "normalizationPipelineVersion": component,
            "analyzerModelVersion": component,
            "clientAssetVersion": component,
            "boardInstallationVersion": component,
        }
        bundle_digest = digest(json.dumps(bundle, separators=(",", ":"), sort_keys=True).encode())
        return {
            "schemaVersion": "1.0",
            "requestId": "job-1",
            "sessionId": "session-1",
            "captureOrdinal": 1,
            "correlationId": "correlation-1",
            "deadline": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
            "rawSha256": hashlib.sha256(image).hexdigest(),
            "contentType": "image/png",
            "recipeId": "PM-ABC-001",
            "machineId": "MC-07",
            "boardId": "CB-001",
            "inspectionIntent": "PM_SIMILARITY",
            "executionBundleDigest": bundle_digest,
            "executionBundle": bundle,
            "artifactPackageDigest": digest(b"artifact"),
            "simulation": True,
        }
    return factory
