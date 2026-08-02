import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from phone_dino.artifacts import InspectionRoiContract, ProductionArtifact, ProductionArtifactV12, require_inspection_roi, ArtifactError


def _payload():
    return {
        "version": "roi-1.0",
        "canonicalWidth": 1000,
        "canonicalHeight": 800,
        "polygon": [{"x": 100.0, "y": 100.0}, {"x": 700.0, "y": 100.0}, {"x": 700.0, "y": 600.0}, {"x": 100.0, "y": 600.0}],
        "inspectionRegions": [{"x": 120, "y": 120, "width": 560, "height": 440}],
    }


def _contract():
    payload = _payload()
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return InspectionRoiContract.model_validate({**payload, "digest": "sha256:" + hashlib.sha256(body).hexdigest()})


def test_roi_digest_and_geometry_are_bound():
    roi = _contract()
    assert roi.version == "roi-1.0"
    with pytest.raises(ValidationError, match="digest"):
        InspectionRoiContract.model_validate({**_payload(), "digest": "sha256:" + "0" * 64})


def test_roi_rejects_out_of_bounds_and_overlapping_regions():
    payload = _payload()
    payload["polygon"][0] = {"x": 1001, "y": 100}
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(ValidationError, match="exceeds canonical"):
        InspectionRoiContract.model_validate({**payload, "digest": "sha256:" + hashlib.sha256(body).hexdigest()})

    payload = _payload()
    payload["inspectionRegions"] = [{"x": 100, "y": 100, "width": 300, "height": 300}, {"x": 200, "y": 200, "width": 300, "height": 300}]
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(ValidationError, match="pairwise disjoint"):
        InspectionRoiContract.model_validate({**payload, "digest": "sha256:" + hashlib.sha256(body).hexdigest()})


def test_legacy_artifact_without_roi_fails_closed():
    artifact = SimpleNamespace(inspection_roi=None)
    with pytest.raises(ArtifactError, match="INSPECTION_ROI_CONTRACT_REQUIRED"):
        require_inspection_roi(artifact)


def test_roi_must_match_target_canonical_dimensions():
    roi = _contract()
    artifact = SimpleNamespace(
        inspection_roi=roi,
        target_alignment=SimpleNamespace(canonical_width=999, canonical_height=800),
    )
    with pytest.raises(ArtifactError, match="CANONICAL_BOUNDS_MISMATCH"):
        require_inspection_roi(artifact)


def test_roi_rejects_alignment_or_held_out_overlap():
    roi = _contract()
    artifact = SimpleNamespace(
        inspection_roi=roi,
        target_alignment=SimpleNamespace(
            canonical_width=1000,
            canonical_height=800,
            alignment_regions=[SimpleNamespace(x=100, y=100, width=80, height=80)],
            held_out_regions=[SimpleNamespace(x=800, y=700, width=50, height=50)],
        ),
    )
    with pytest.raises(ArtifactError, match="OVERLAPS_ALIGNMENT_OR_HELD_OUT"):
        require_inspection_roi(artifact)


def test_schema_11_rejects_roi_and_schema_12_requires_it():
    # The explicit version split prevents silently changing the 1.1 contract.
    with pytest.raises(ValidationError):
        ProductionArtifact.model_validate({"schemaVersion": "1.1", "inspectionRoi": _contract().model_dump(by_alias=True)})
    fields = ProductionArtifactV12.model_fields
    assert fields["inspection_roi"].is_required()
    assert fields["schema_version"].annotation == __import__("typing").Literal["1.2"]
