from __future__ import annotations

import json
from pathlib import Path

import pytest

from phone_dino import mvtec_cohort_quarantine as quarantine


def _write_known_incident(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(quarantine.known_fresh_normal_holdout_v1_incident_document(), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _parent_manifest_path(root: Path, name: str = "parent-holdout.json") -> Path:
    return root / "cohort" / name


def test_known_incident_template_is_closed_and_matches_the_literal_pin() -> None:
    document = quarantine.known_fresh_normal_holdout_v1_incident_document()
    assert document["cohortQuarantineIncidentSha256"] == quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_DECLARED_SHA256
    assert document["cohortManifestFileSha256"] == quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_FILE_SHA256
    assert document["cohortManifestDeclaredSha256"] == quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_DECLARED_SHA256
    quarantine._validate_known_incident(document)


def test_missing_incident_fails_closed_before_parent_manifest_is_even_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_reads: list[object] = []
    monkeypatch.setattr(
        quarantine,
        "_load_normal_holdout_identity",
        lambda *_args, **_kwargs: parent_reads.append("parent")
        or (_ for _ in ()).throw(AssertionError("parent manifest must not be read when incident is missing")),
    )

    with pytest.raises(quarantine.CohortQuarantineError, match="V3 cohort quarantine incident must be a regular non-link file"):
        quarantine.assert_v3_parent_holdout_not_quarantined(
            _parent_manifest_path(tmp_path),
            quarantine_incident_path=tmp_path / "incident_ledger" / "missing-incident.json",
            expected_quarantine_incident_sha256=quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_DECLARED_SHA256,
        )
    assert parent_reads == []


def test_incident_ledger_cannot_live_under_the_quarantined_cohort_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort_root = tmp_path / "cohort"
    incident = _write_known_incident(cohort_root / "incident.json")
    monkeypatch.setattr(
        quarantine,
        "_load_normal_holdout_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parent manifest must not be read")),
    )

    with pytest.raises(quarantine.CohortQuarantineError, match="outside the parent cohort root"):
        quarantine.assert_v3_parent_holdout_not_quarantined(
            cohort_root / "normal_holdout.json",
            quarantine_incident_path=incident,
            expected_quarantine_incident_sha256=quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_DECLARED_SHA256,
        )


def test_self_consistent_forged_incident_cannot_change_the_known_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident = quarantine.known_fresh_normal_holdout_v1_incident_document()
    incident["reason"] = "PRETEND_RELEASE"
    incident["cohortQuarantineIncidentSha256"] = quarantine._document_digest(
        incident, "cohortQuarantineIncidentSha256"
    )
    path = tmp_path / "external" / "forged-incident.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(incident, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        quarantine,
        "_load_normal_holdout_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parent manifest must not be read after forged incident")),
    )

    with pytest.raises(quarantine.CohortQuarantineError, match="reason is unsafe"):
        quarantine.assert_v3_parent_holdout_not_quarantined(
            _parent_manifest_path(tmp_path),
            quarantine_incident_path=path,
            expected_quarantine_incident_sha256=quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_DECLARED_SHA256,
        )


def test_self_consistent_incident_with_a_different_target_cannot_bypass_the_known_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident = quarantine.known_fresh_normal_holdout_v1_incident_document()
    incident["cohortManifestDeclaredSha256"] = "sha256:" + "d" * 64
    incident["cohortQuarantineIncidentSha256"] = quarantine._document_digest(
        incident, "cohortQuarantineIncidentSha256"
    )
    path = tmp_path / "external" / "wrong-target-incident.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(incident, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        quarantine,
        "_load_normal_holdout_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parent manifest must not be read")),
    )

    with pytest.raises(quarantine.CohortQuarantineError, match="does not bind the known exposed cohort"):
        quarantine.assert_v3_parent_holdout_not_quarantined(
            _parent_manifest_path(tmp_path),
            quarantine_incident_path=path,
            expected_quarantine_incident_sha256=quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_DECLARED_SHA256,
        )


def test_wrong_independent_pin_fails_before_incident_or_parent_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident = _write_known_incident(tmp_path / "external" / "incident.json")
    reads: list[str] = []
    monkeypatch.setattr(
        quarantine,
        "_read_external_json",
        lambda *_args, **_kwargs: reads.append("incident")
        or (_ for _ in ()).throw(AssertionError("incident must not be read with the wrong independent pin")),
    )

    with pytest.raises(quarantine.CohortQuarantineError, match="approved immutable pin"):
        quarantine.assert_v3_parent_holdout_not_quarantined(
            _parent_manifest_path(tmp_path),
            quarantine_incident_path=incident,
            expected_quarantine_incident_sha256="sha256:" + "0" * 64,
        )
    assert reads == []


def test_known_exposed_cohort_is_rejected_with_correct_external_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident = _write_known_incident(tmp_path / "external" / "incident.json")
    monkeypatch.setattr(
        quarantine,
        "_load_normal_holdout_identity",
        lambda *_args, **_kwargs: (
            quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_FILE_SHA256,
            quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_MANIFEST_DECLARED_SHA256,
        ),
    )

    with pytest.raises(quarantine.CohortQuarantineError, match="permanently quarantined"):
        quarantine.assert_v3_parent_holdout_not_quarantined(
            _parent_manifest_path(tmp_path),
            quarantine_incident_path=incident,
            expected_quarantine_incident_sha256=quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_DECLARED_SHA256,
        )


def test_parent_manifest_must_be_closed_json_and_no_source_root_is_accepted(
    tmp_path: Path,
) -> None:
    incident = _write_known_incident(tmp_path / "external" / "incident.json")
    parent = _parent_manifest_path(tmp_path, "malformed-parent-holdout.json")
    parent.parent.mkdir(parents=True)
    parent.write_text("{}", encoding="utf-8")

    with pytest.raises(quarantine.CohortQuarantineError, match="parent normal holdout manifest is unsafe"):
        quarantine.assert_v3_parent_holdout_not_quarantined(
            parent,
            quarantine_incident_path=incident,
            expected_quarantine_incident_sha256=quarantine.KNOWN_FRESH_NORMAL_HOLDOUT_V1_INCIDENT_DECLARED_SHA256,
        )
