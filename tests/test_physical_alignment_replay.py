from __future__ import annotations

from io import BytesIO
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageOps
import pytest

from phone_dino.contracts import AnalyzeRequest
from phone_dino import physical_alignment_replay as replay


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _write_json(path: Path, document: dict[str, object]) -> bytes:
    data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _jpeg(*, orientation: int | None = None) -> bytes:
    image = Image.new("RGB", (5, 3), (25, 50, 75))
    image.putpixel((0, 0), (220, 10, 20))
    image.putpixel((4, 2), (10, 200, 30))
    output = BytesIO()
    if orientation is None:
        image.save(output, "JPEG", quality=95, subsampling=2, progressive=False)
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(output, "JPEG", quality=95, subsampling=2, progressive=False, exif=exif)
    return output.getvalue()


def _request(
    raw: bytes,
    artifact_digest: str,
    *,
    simulation: bool = True,
    request_id: str = "request-1",
    session_id: str = "session-1",
    correlation_id: str = "correlation-1",
    capture_ordinal: int = 1,
) -> dict[str, object]:
    version = "sha256:" + "1" * 64
    document: dict[str, object] = {
        "schemaVersion": "1.4",
        "requestId": request_id,
        "sessionId": session_id,
        "captureOrdinal": capture_ordinal,
        "correlationId": correlation_id,
        "deadline": "2099-01-01T00:00:00Z",
        "rawSha256": hashlib.sha256(raw).hexdigest(),
        "contentType": "image/jpeg",
        "recipeId": "recipe-1",
        "machineId": "machine-1",
        "boardId": "board-1",
        "inspectionIntent": "PM_SIMILARITY",
        "executionBundleDigest": version,
        "executionBundle": {
            "recipeVersion": version,
            "goldenSetVersion": version,
            "capturePolicyVersion": version,
            "decisionPolicyVersion": version,
            "normalizationPipelineVersion": version,
            "analyzerModelVersion": version,
            "clientAssetVersion": version,
            "boardInstallationVersion": version,
        },
        "artifactPackageDigest": artifact_digest,
        "simulation": simulation,
    }
    request = AnalyzeRequest.model_validate_json(json.dumps(document))
    document["executionBundleDigest"] = replay.canonical_bundle_digest(request)
    return document


def _response(request: dict[str, object], *, runtime: str, normalization: str) -> dict[str, object]:
    analysis_id = hashlib.sha256(
        "|".join((
            str(request["sessionId"]),
            str(request["captureOrdinal"]),
            str(request["rawSha256"]),
            str(request["executionBundleDigest"]),
            runtime,
        )).encode("utf-8")
    ).hexdigest()
    return {
        "schemaVersion": request["schemaVersion"],
        "requestId": request["requestId"],
        "analysisId": analysis_id,
        "rawSha256": request["rawSha256"],
        "simulation": request["simulation"],
        "resolvedVersions": {
            "executionBundleDigest": request["executionBundleDigest"],
            "artifactPackageDigest": request["artifactPackageDigest"],
            "analyzerModelVersion": request["executionBundle"]["analyzerModelVersion"],  # type: ignore[index]
            "analyzerRuntimeVersion": runtime,
            "normalizationRuntimeVersion": normalization,
            "recipeAnalysisProfileDigest": None,
            "scorerInputContractDigest": None,
        },
        "captureAssessment": {"state": "ACCEPTED", "reasonCodes": []},
        "normalization": {
            "alignment": {
                "state": "ALIGNED",
                "method": "TARGET_AFFINE",
                "targetRelative": True,
                "inlierCount": 20,
                "inlierRatio": 0.8,
                "reprojectionErrorPx": 0.5,
                "coverageRatio": 0.4,
                "transformWithinBounds": True,
                "inspectionMaskApplied": True,
            },
        },
        "analysis": {"state": "NOT_RUN"},
    }


def _cohort_document(root: Path, *, raw: bytes | None = None) -> dict[str, object]:
    raw = _jpeg() if raw is None else raw
    artifact = b"pinned-artifact-bytes"
    (root / "artifact").mkdir(parents=True)
    (root / "raw").mkdir()
    (root / "requests").mkdir()
    (root / "responses").mkdir()
    (root / "artifact" / "artifact.json").write_bytes(artifact)
    (root / "raw" / "capture.jpg").write_bytes(raw)
    artifact_digest = _digest(artifact)
    request = _request(raw, artifact_digest)
    request_bytes = _write_json(root / "requests" / "capture.json", request)
    response_bytes = _write_json(
        root / "responses" / "capture.json",
        _response(request, runtime="sha256:" + "2" * 64, normalization="sha256:" + "3" * 64),
    )
    readyz = {
        "status": "ready",
        "simulation": True,
        "analysisMode": "ENGINEERING_REAL_DINO",
        "supportedSchemas": ["1.4"],
        "replayProvenance": {
            "artifactPackageDigest": artifact_digest,
            "analyzerRuntimeVersion": "sha256:" + "2" * 64,
            "allowTargetOnlyAlignment": True,
            "allowContourAnchorAlignment": True,
            "maxImageBytes": 12 * 1024 * 1024,
            "maxImagePixels": 24_000_000,
            "maxImageWidth": 8000,
            "maxImageHeight": 8000,
        },
    }
    readyz_bytes = _write_json(root / "readyz.json", readyz)
    document: dict[str, object] = {
        "schemaVersion": replay.COHORT_SCHEMA_VERSION,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": "PHYSICAL_ALIGNMENT_REPLAY_ONLY",
        "artifact": {
            "relativePath": "artifact/artifact.json",
            "artifactPackageDigest": artifact_digest,
            "schemaVersion": "1.4",
            "recipeId": "recipe-1",
            "machineId": "machine-1",
            "boardId": "board-1",
        },
        "readyzEvidence": {"relativePath": "readyz.json", "sha256": _digest(readyz_bytes)},
        "replayPolicy": {"allowTargetOnlyAlignment": True, "allowContourAnchorAlignment": True},
        "cases": [{
            "caseId": "capture-001",
            "partition": "DEVELOPMENT",
            "acquisitionGroupId": "session-a",
            "intendedAlignmentState": "ALIGNED",
            "captureStrata": {
                "device": "device-a", "camera": "rear-wide", "lens": "native",
                "lighting": "bench-a", "distance": "fixed", "view": "top",
            },
            "raw": {"relativePath": "raw/capture.jpg", "sha256": _digest(raw)},
            "request": {"relativePath": "requests/capture.json", "sha256": _digest(request_bytes)},
            "response": {"relativePath": "responses/capture.json", "sha256": _digest(response_bytes)},
        }],
    }
    document["cohortSha256"] = replay.canonical_json_sha256(document)
    return document


def _write_cohort(root: Path, document: dict[str, object]) -> Path:
    path = root / "cohort.json"
    _write_json(path, document)
    return path


def _fake_artifact() -> SimpleNamespace:
    return SimpleNamespace(
        recipe_id="recipe-1",
        machine_id="machine-1",
        board_id="board-1",
        analyzer_model_version="sha256:" + "1" * 64,
        analyzer_runtime_version="sha256:" + "2" * 64,
        normalization_pipeline_version="sha256:" + "3" * 64,
    )


def _valid_v14_artifact_document() -> dict[str, object]:
    """A compact full artifact that exercises the real V14 parser and mask gate."""

    version = "sha256:" + "1" * 64
    canonical_image = Image.new("RGB", (320, 240), (20, 30, 40))
    canonical_buffer = BytesIO()
    canonical_image.save(canonical_buffer, "PNG")
    canonical = canonical_buffer.getvalue()
    mask = Image.new("L", (320, 240), 0)
    ImageDraw.Draw(mask).rectangle((110, 110, 189, 169), fill=255)
    mask_buffer = BytesIO()
    mask.save(mask_buffer, "PNG")
    mask_bytes = mask_buffer.getvalue()
    roi = {
        "version": "roi-1.0",
        "canonicalWidth": 320,
        "canonicalHeight": 240,
        "polygon": [
            {"x": 100.0, "y": 100.0}, {"x": 200.0, "y": 100.0},
            {"x": 200.0, "y": 180.0}, {"x": 100.0, "y": 180.0},
        ],
        "inspectionRegions": [{"x": 100, "y": 100, "width": 100, "height": 80}],
    }
    roi_digest = _digest(json.dumps(roi, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    target_alignment = {
        "method": "TARGET_AFFINE",
        "referenceImageBase64": base64.b64encode(canonical).decode("ascii"),
        "referenceImageSha256": _digest(canonical),
        "canonicalWidth": 320,
        "canonicalHeight": 240,
        "alignmentRegions": [{"x": 0, "y": 0, "width": 320, "height": 80}],
        "heldOutRegions": [{"x": 0, "y": 200, "width": 320, "height": 40}],
        "inspectionRegions": [{"x": 100, "y": 100, "width": 100, "height": 80}],
        "minMatches": 8,
        "minInliers": 6,
        "minInlierRatio": 0.5,
        "minCoverageRatio": 0.02,
        "maxReprojectionErrorPx": 3.0,
        "minScale": 0.8,
        "maxScale": 1.2,
        "maxRotationDegrees": 15.0,
        "maxShear": 0.05,
        "maxTranslationPx": 300.0,
        "maxSecondaryInlierRatio": 0.35,
        "minHeldOutMatches": 4,
        "maxHeldOutReprojectionErrorPx": 3.0,
    }
    return {
        "schemaVersion": "1.4",
        "recipeId": "recipe-1",
        "machineId": "machine-1",
        "boardId": "board-1",
        "goldenSetVersion": version,
        "normalizationPipelineVersion": version,
        "analyzerModelVersion": version,
        "decisionPolicyVersion": version,
        "analyzerRuntimeVersion": "sha256:" + "2" * 64,
        "modelRepositoryVersion": version,
        "boardInstallationVersion": version,
        "modelWeightsSha256": version,
        "board": {
            "squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
            "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896,
        },
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 50.0, "maxOverExposureRatio": 0.05},
        "targetAlignment": target_alignment,
        "goldenEmbeddings": [{
            "id": "golden-1", "sourceSha256": version, "values": [1.0, 0.0],
            "canonicalSha256": _digest(canonical), "canonicalImagePngBase64": base64.b64encode(canonical).decode("ascii"),
        }],
        "inspectionRoi": {**roi, "digest": roi_digest},
        "spatialDifferencePolicy": {"anomalyDistanceThreshold": 0.35, "minRegionAreaRatio": 0.003, "maxRegions": 2},
        "subjectSegmentation": {
            "version": "subject-1.0", "method": "MOBILE_SAM_VIT_T_BOX_PROMPT", "usageMode": "SPATIAL_GATE",
            "approvalState": "ENGINEERING_AUTO", "promptPolicy": "INSPECTION_ROI_BOUNDING_BOX_V1",
            "canonicalWidth": 320, "canonicalHeight": 240,
            "modelRepositoryVersion": version, "modelWeightsSha256": version,
            "minModelQualityScore": 0.8, "minForegroundRatio": 0.1, "maxForegroundRatio": 0.9,
            "supportPaddingPx": 8, "boundaryBandPx": 4,
            "goldenMasks": [{
                "goldenId": "golden-1", "canonicalSha256": _digest(canonical),
                "maskPngBase64": base64.b64encode(mask_bytes).decode("ascii"), "maskSha256": _digest(mask_bytes),
                "promptRegionNormalized": {"x": 0.3125, "y": 100 / 240, "width": 0.3125, "height": 80 / 240},
                "modelQualityScore": 0.95, "foregroundRatio": 0.6,
            }],
        },
        "subjectAlignment": {
            "version": "subject-align-1.0", "method": "SUBJECT_CONTOUR_ECC_AFFINE",
            "approvalState": "ENGINEERING_AUTO", "maskSource": "GOLDEN_SUBJECT_MASK", "alignmentBandPx": 12,
            "heldOutBlockPx": 16, "maxIterations": 100, "convergenceEpsilon": 0.00001,
            "minEccCorrelation": 0.2, "maxHeldOutResidualPx": 8.0, "minHeldOutCoverageRatio": 0.35,
            "maxResidualTranslationPx": 30.0, "maxResidualRotationDegrees": 10.0,
            "maxResidualScaleDelta": 0.2, "maxResidualShear": 0.1,
        },
        "candidateVerificationPolicy": {
            "version": "candidate-verify-1.0", "method": "DINO_CROP_COSINE_V1", "mode": "SHADOW",
            "approvalState": "ENGINEERING_AUTO", "contextPaddingRatio": 0.35, "minimumCropSidePx": 112,
            "maxCandidates": 2, "reviewPriorityDistance": 0.1, "highPriorityDistance": 0.25,
        },
    }


def test_cohort_replay_binds_evidence_and_uses_pinned_normalizer_policy(tmp_path, monkeypatch):
    external = tmp_path / "external"
    external.mkdir()
    cohort_path = _write_cohort(external, _cohort_document(external, raw=_jpeg(orientation=6)))
    output_path = external / "report.json"
    seen: dict[str, object] = {}

    class FakeNormalizer:
        def __init__(self, *, allow_target_only_alignment, allow_contour_anchor_alignment):
            seen["policy"] = (allow_target_only_alignment, allow_contour_anchor_alignment)

        def normalize(self, image, artifact):
            seen["image"] = image
            return SimpleNamespace(
                alignment=SimpleNamespace(
                    state="ALIGNED", method="TARGET_AFFINE", inlier_ratio=0.8,
                    reprojection_error_px=0.5, coverage_ratio=0.4,
                ),
                reason_codes=(),
            )

    monkeypatch.setattr(replay, "_load_pinned_artifact", lambda _loaded: _fake_artifact())
    monkeypatch.setattr(replay, "verify_artifact_binding", lambda _artifact, _request: None)
    monkeypatch.setattr(replay, "OpenCvCharucoNormalizer", FakeNormalizer)

    report = replay.run_cohort(cohort_path, output_path)

    assert output_path.is_file()
    assert report["evidenceClassification"] == "ENGINEERING_REPLAY_ONLY"
    assert report["reportSha256"] == replay.canonical_json_sha256({
        key: value for key, value in report.items() if key != "reportSha256"
    })
    assert seen["policy"] == (True, True)
    assert seen["image"].width == 5
    assert seen["image"].height == 3
    case = report["cases"][0]
    assert case["expectationMet"] is True
    assert case["sourceEncoding"]["samplingFactors"] == [[2, 2], [1, 1], [1, 1]]
    assert case["sourceEncoding"]["orientedWidth"] == 3
    assert case["sourceEncoding"]["orientedHeight"] == 5
    assert "raw/capture.jpg" not in json.dumps(report)
    assert "exif" not in json.dumps(report).lower()


def test_real_v14_artifact_parser_and_request_binding_accept_bom_bytes(tmp_path, monkeypatch):
    external = tmp_path / "external"
    external.mkdir()
    document = _cohort_document(external)
    artifact_bytes = b"\xef\xbb\xbf" + json.dumps(
        _valid_v14_artifact_document(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    artifact_path = external / "artifact" / "artifact.json"
    artifact_path.write_bytes(artifact_bytes)
    artifact_digest = _digest(artifact_bytes)
    document["artifact"]["artifactPackageDigest"] = artifact_digest
    raw = (external / "raw" / "capture.jpg").read_bytes()
    request = _request(raw, artifact_digest)
    request_bytes = _write_json(external / "requests" / "capture.json", request)
    response_bytes = _write_json(
        external / "responses" / "capture.json",
        _response(request, runtime="sha256:" + "2" * 64, normalization="sha256:" + "1" * 64),
    )
    document["cases"][0]["request"]["sha256"] = _digest(request_bytes)
    document["cases"][0]["response"]["sha256"] = _digest(response_bytes)
    readyz_path = external / "readyz.json"
    readyz = json.loads(readyz_path.read_text(encoding="utf-8"))
    readyz["replayProvenance"]["artifactPackageDigest"] = artifact_digest
    readyz_bytes = _write_json(readyz_path, readyz)
    document["readyzEvidence"]["sha256"] = _digest(readyz_bytes)
    document["cohortSha256"] = replay.canonical_json_sha256({
        key: value for key, value in document.items() if key != "cohortSha256"
    })
    cohort_path = _write_cohort(external, document)

    class FakeNormalizer:
        def __init__(self, **_kwargs):
            pass

        def normalize(self, *_args):
            return SimpleNamespace(
                alignment=SimpleNamespace(
                    state="ALIGNED", method="TARGET_AFFINE", inlier_ratio=0.8,
                    reprojection_error_px=0.5, coverage_ratio=0.4,
                ),
                reason_codes=(),
            )

    monkeypatch.setattr(replay, "OpenCvCharucoNormalizer", FakeNormalizer)
    report = replay.run_cohort(cohort_path, external / "report.json")

    assert report["cases"][0]["expectationMet"] is True
    assert report["runtimeCompatibility"] == "CROSS_RUNTIME_REPLAY_ONLY"


def test_tampered_raw_fails_before_report_is_written(tmp_path, monkeypatch):
    external = tmp_path / "external"
    external.mkdir()
    cohort_path = _write_cohort(external, _cohort_document(external))
    (external / "raw" / "capture.jpg").write_bytes(_jpeg(orientation=6))
    monkeypatch.setattr(replay, "_load_pinned_artifact", lambda _loaded: _fake_artifact())
    monkeypatch.setattr(replay, "verify_artifact_binding", lambda _artifact, _request: None)

    with pytest.raises(replay.PhysicalAlignmentReplayError, match="raw still for capture-001 digest mismatch"):
        replay.run_cohort(cohort_path, external / "report.json")

    assert not (external / "report.json").exists()


def test_readyz_policy_and_request_schema_must_match_frozen_service_snapshot(tmp_path, monkeypatch):
    external = tmp_path / "external"
    external.mkdir()
    document = _cohort_document(external)
    readyz_path = external / "readyz.json"
    readyz = json.loads(readyz_path.read_text(encoding="utf-8"))
    readyz["replayProvenance"]["allowContourAnchorAlignment"] = False
    readyz_bytes = _write_json(readyz_path, readyz)
    document["readyzEvidence"]["sha256"] = _digest(readyz_bytes)
    document["cohortSha256"] = replay.canonical_json_sha256({
        key: value for key, value in document.items() if key != "cohortSha256"
    })
    cohort_path = _write_cohort(external, document)
    monkeypatch.setattr(replay, "_load_pinned_artifact", lambda _loaded: _fake_artifact())

    with pytest.raises(replay.PhysicalAlignmentReplayError, match="replay provenance"):
        replay.run_cohort(cohort_path, external / "report.json")

    readyz["replayProvenance"]["allowContourAnchorAlignment"] = True
    readyz["supportedSchemas"] = ["1.3"]
    readyz_bytes = _write_json(readyz_path, readyz)
    document["readyzEvidence"]["sha256"] = _digest(readyz_bytes)
    document["cohortSha256"] = replay.canonical_json_sha256({
        key: value for key, value in document.items() if key != "cohortSha256"
    })
    cohort_path = _write_cohort(external, document)
    with pytest.raises(replay.PhysicalAlignmentReplayError, match="wire schema"):
        replay.run_cohort(cohort_path, external / "report-two.json")


def test_session_or_correlation_cannot_cross_development_and_holdout(tmp_path, monkeypatch):
    external = tmp_path / "external"
    external.mkdir()
    document = _cohort_document(external)
    second_raw = _jpeg(orientation=6)
    (external / "raw" / "capture-two.jpg").write_bytes(second_raw)
    artifact_digest = document["artifact"]["artifactPackageDigest"]
    second_request = _request(
        second_raw,
        artifact_digest,
        request_id="request-2",
        capture_ordinal=2,
        # Deliberately retain the first saved session/correlation identity.
        session_id="session-1",
        correlation_id="correlation-1",
    )
    request_bytes = _write_json(external / "requests" / "capture-two.json", second_request)
    response_bytes = _write_json(
        external / "responses" / "capture-two.json",
        _response(second_request, runtime="sha256:" + "2" * 64, normalization="sha256:" + "3" * 64),
    )
    second = json.loads(json.dumps(document["cases"][0]))
    second.update({
        "caseId": "capture-002",
        "partition": "HELD_OUT",
        "acquisitionGroupId": "session-b",
        "raw": {"relativePath": "raw/capture-two.jpg", "sha256": _digest(second_raw)},
        "request": {"relativePath": "requests/capture-two.json", "sha256": _digest(request_bytes)},
        "response": {"relativePath": "responses/capture-two.json", "sha256": _digest(response_bytes)},
    })
    document["cases"].append(second)
    document["cohortSha256"] = replay.canonical_json_sha256({
        key: value for key, value in document.items() if key != "cohortSha256"
    })
    cohort_path = _write_cohort(external, document)
    monkeypatch.setattr(replay, "_load_pinned_artifact", lambda _loaded: _fake_artifact())
    monkeypatch.setattr(replay, "verify_artifact_binding", lambda _artifact, _request: None)

    with pytest.raises(replay.PhysicalAlignmentReplayError, match="sessionId crosses"):
        replay.run_cohort(cohort_path, external / "report.json")

    assert not (external / "report.json").exists()


def test_cohort_self_digest_and_split_leakage_fail_closed(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    document = _cohort_document(external)
    document["purpose"] = "wrong"
    cohort_path = _write_cohort(external, document)
    with pytest.raises(replay.PhysicalAlignmentReplayError, match="cohort manifest schema is invalid"):
        replay.load_cohort(cohort_path)

    external_two = tmp_path / "external-two"
    external_two.mkdir()
    document = _cohort_document(external_two)
    second = json.loads(json.dumps(document["cases"][0]))
    second["caseId"] = "capture-002"
    second["partition"] = "HELD_OUT"
    document["cases"].append(second)
    document["cohortSha256"] = replay.canonical_json_sha256({
        key: value for key, value in document.items() if key != "cohortSha256"
    })
    cohort_path = _write_cohort(external_two, document)
    with pytest.raises(replay.PhysicalAlignmentReplayError, match="raw stills must not be reused"):
        replay.load_cohort(cohort_path)


def test_runtime_equivalent_decode_applies_exif_orientation_and_records_safe_headers():
    raw = _jpeg(orientation=6)
    decoded = replay.decode_runtime_equivalent(raw, content_type="image/jpeg")
    with Image.open(BytesIO(raw)) as image:
        expected = __import__("numpy").asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype="uint8")
    assert decoded.rgb.shape == expected.shape == (5, 3, 3)
    assert (decoded.rgb == expected).all()
    assert decoded.output_encoding["orientedWidth"] == 3
    assert decoded.output_encoding["orientedHeight"] == 5
    assert decoded.output_encoding["progressive"] is False
    assert str(decoded.output_encoding["quantizationTablesSha256"]).startswith("sha256:")


def test_utf8_bom_json_is_accepted_without_changing_its_byte_digest():
    document = replay._read_json_bytes(b"\xef\xbb\xbf{\"state\":\"ready\"}", description="test")
    assert document == {"state": "ready"}


def test_output_inside_repository_or_existing_output_is_rejected(tmp_path):
    with pytest.raises(replay.PhysicalAlignmentReplayError, match="outside the repository"):
        replay._validate_output_path(replay.REPOSITORY_ROOT / "physical-alignment-report.json")

    output = tmp_path / "existing.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(replay.PhysicalAlignmentReplayError, match="already exists"):
        replay._validate_output_path(output)


def test_relative_path_traversal_and_links_are_rejected(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_jpeg())
    with pytest.raises(replay.PhysicalAlignmentReplayError, match="safe relative path"):
        replay._safe_relative_file(root, "../outside.jpg", description="test evidence")
    with pytest.raises(replay.PhysicalAlignmentReplayError, match="safe relative path"):
        replay._safe_relative_file(root, "raw.jpg:alternate-stream", description="test evidence")

    linked = root / "linked.jpg"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(replay.PhysicalAlignmentReplayError, match="regular non-link file"):
        replay._safe_relative_file(root, "linked.jpg", description="test evidence")


def test_v19_mapping_and_readyz_wire_requirement_are_explicit(tmp_path):
    from phone_dino.artifacts import ProductionArtifactV19

    assert replay.ARTIFACT_MODELS["1.9"] is ProductionArtifactV19
    external = tmp_path / "external"
    external.mkdir()
    cohort_path = _write_cohort(external, _cohort_document(external))
    loaded = replay.load_cohort(cohort_path)
    artifact = object.__new__(ProductionArtifactV19)
    object.__setattr__(artifact, "__dict__", {"analyzer_runtime_version": "sha256:" + "2" * 64})
    with pytest.raises(replay.PhysicalAlignmentReplayError, match="wire schema 1.6"):
        replay._load_readyz(loaded, artifact)


def test_v19_reference_board_rejection_stops_before_normalizer(tmp_path, monkeypatch):
    from phone_dino.artifacts import ProductionArtifactV19

    external = tmp_path / "external"
    external.mkdir()
    document = _cohort_document(external)
    readyz_path = external / "readyz.json"
    readyz = json.loads(readyz_path.read_text(encoding="utf-8"))
    readyz["supportedSchemas"] = ["1.4", "1.6"]
    readyz_bytes = _write_json(readyz_path, readyz)
    document["readyzEvidence"]["sha256"] = _digest(readyz_bytes)
    document["cohortSha256"] = replay.canonical_json_sha256({
        key: value for key, value in document.items() if key != "cohortSha256"
    })
    cohort_path = _write_cohort(external, document)
    artifact = object.__new__(ProductionArtifactV19)
    object.__setattr__(artifact, "__dict__", {"analyzer_runtime_version": "sha256:" + "2" * 64})
    request = SimpleNamespace(
        session_id="session-1", correlation_id="correlation-1", content_type="image/jpeg", board_candidates=[],
    )
    response = SimpleNamespace(normalization=None, reference_board_evidence=SimpleNamespace(state="VERIFIED"))
    decoded = replay.RuntimeDecodedCapture(
        rgb=None,
        output_encoding={"format": "JPEG", "orientedWidth": 1, "orientedHeight": 1},
        encoded_width=1,
        encoded_height=1,
    )

    class ExplodingNormalizer:
        def __init__(self, **_kwargs):
            pass

        def normalize(self, *_args):
            raise AssertionError("normalizer must not run after V19 reference-board rejection")

    monkeypatch.setattr(replay, "_load_pinned_artifact", lambda _loaded: artifact)
    monkeypatch.setattr(replay, "_load_case_documents", lambda *_args, **_kwargs: (b"raw", request, response, decoded))
    monkeypatch.setattr(replay, "OpenCvCharucoNormalizer", ExplodingNormalizer)
    verifier = SimpleNamespace(verify=lambda _image, _artifact: SimpleNamespace(
        state="REJECTED", reason_codes=["REFERENCE_QR_NOT_DETECTED"],
    ))

    report = replay.run_cohort(cohort_path, external / "report.json", reference_board_verifier=verifier)

    case = report["cases"][0]
    assert case["replayedReferenceBoardState"] == "REJECTED"
    assert case["replayedReferenceBoardReasonCodes"] == ["REFERENCE_QR_NOT_DETECTED"]
    assert case["replayedAlignmentState"] == "NOT_ALIGNED"
