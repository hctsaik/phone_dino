from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

import pytest

from phone_dino import mvtec_recovery_v4_source_acquisition as recovery


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _write_json(path: Path, document: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _empty_validated_plan(quarantined_root: Path) -> recovery.RecoveryV4ValidatedSourcePlan:
    return recovery.RecoveryV4ValidatedSourcePlan(
        plan={"recoveryV4SourceAcquisitionPlanSha256": _digest("1")},
        plan_file_sha256=_digest("2"),
        records=(),
        excluded_remote_paths=frozenset(),
        excluded_source_hashes=frozenset(),
        allowlist_policy={"recoveryV4SourceAllowlistPolicySha256": _digest("3")},
        allowlist_policy_file_sha256=_digest("4"),
        non_overlap_ledger={"recoveryV4NonOverlapMetadataLedgerSha256": _digest("5")},
        non_overlap_ledger_file_sha256=_digest("6"),
        quarantined_cohort_root=quarantined_root,
    )


def _inventory_with_only_metadata_rows(
    policy_records: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Build an in-memory pinned-metadata-shaped inventory without image bytes."""

    inventory: dict[str, dict[str, object]] = {}
    for record in policy_records:
        inventory[str(record["sourceRemotePath"])] = {
            "category": record["category"],
            "split": "train",
            "defect": "good",
            "hasMask": False,
        }
    for category in recovery.RECOVERY_V4_CATEGORIES:
        selected = [record for record in policy_records if record["category"] == category]
        final_rank = str(selected[-1]["metadataRankSha256"])[7:]
        needed = recovery.RECOVERY_V4_EXPECTED_TRAIN_GOOD_COUNTS[category] - len(selected)
        added = 0
        candidate_index = 0
        while added < needed:
            remote_path = f"data/filler/{category}-{candidate_index}.png"
            candidate_index += 1
            if recovery._rank_remote_path(category, remote_path) <= final_rank:
                continue
            inventory[remote_path] = {
                "category": category,
                "split": "train",
                "defect": "good",
                "hasMask": False,
            }
            added += 1
        first_rank = str(selected[0]["metadataRankSha256"])[7:]
        for kind, split, defect, has_mask in (
            ("test-good", "test", "good", False),
            ("train-anomaly", "train", "broken", False),
            ("masked-good", "train", "good", True),
        ):
            poison_index = 0
            while True:
                remote_path = f"data/poison/{category}-{kind}-{poison_index}.png"
                poison_index += 1
                if recovery._rank_remote_path(category, remote_path) < first_rank:
                    inventory[remote_path] = {
                        "category": category,
                        "split": split,
                        "defect": defect,
                        "hasMask": has_mask,
                    }
                    break
    return inventory


def _forbid_recursive_enumeration(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("recursive/directory enumeration is forbidden in Recovery V4 source planning")

    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(os, "walk", forbidden)


def test_tracked_policy_has_the_independently_audited_query_allocation() -> None:
    policy, _file_sha256, records = recovery.load_recovery_v4_allowlist_policy()

    assert policy["rankingInputEncoding"] == "UTF8_PIPE_DELIMITED_V1"
    assert policy["rankingAlgorithm"] == "SHA256_SEED_REVISION_CATEGORY_REMOTE_PATH_V1"
    assert len(records) == 96
    assert [record["sourceRemotePath"] for record in records if record["role"] == "QUERY"] == [
        "data/data_47/114-13.png",
        "data/data_49/024-28.png",
        "data/data_47/117-13.png",
        "data/data_47/199-13.png",
        "data/data_34/163-10.png",
        "data/data_35/036-12.png",
        "data/data_35/142-10.png",
        "data/data_34/160-10.png",
        "data/data_27/078-8.png",
        "data/data_27/325.png",
        "data/data_26/106-8.png",
        "data/data_26/059-9.png",
    ]
    assert {record["role"] for record in records if record["rank"] in {7, 8}} == {"RAW_CALIBRATION"}


def test_metadata_only_validation_excludes_test_anomaly_and_mask_rows() -> None:
    _policy, _file_sha256, records = recovery.load_recovery_v4_allowlist_policy()
    inventory = _inventory_with_only_metadata_rows(records)

    recovery._validate_policy_against_pinned_metadata(records, inventory, excluded_remote_paths=set())

    selected_paths = {record["sourceRemotePath"] for record in records}
    poison_paths = {path for path in inventory if "/poison/" in path}
    assert poison_paths.isdisjoint(selected_paths)
    assert any(inventory[path]["split"] == "test" for path in poison_paths)
    assert any(inventory[path]["defect"] != "good" for path in poison_paths)
    assert any(bool(inventory[path]["hasMask"]) for path in poison_paths)


def test_source_plan_is_metadata_only_and_never_enumerates_or_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, policy_file_sha256, records = recovery.load_recovery_v4_allowlist_policy()
    inventory = _inventory_with_only_metadata_rows(records)
    ledger = {"recoveryV4NonOverlapMetadataLedgerSha256": "sha256:" + "a" * 64}
    quarantined_root = tmp_path / "quarantined-root"
    quarantined_root.mkdir()
    _forbid_recursive_enumeration(monkeypatch)
    monkeypatch.setattr(recovery, "_load_pinned_source_metadata", lambda _path: inventory)
    monkeypatch.setattr(
        recovery,
        "load_recovery_v4_non_overlap_metadata_ledger",
        lambda _path: (ledger, "sha256:" + "b" * 64, set(), set(), quarantined_root),
    )
    monkeypatch.setattr(
        recovery,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("planning must not download source bytes")),
    )
    output = tmp_path / "external" / "source-plan.json"

    document = recovery.create_recovery_v4_source_acquisition_plan(
        tmp_path / "ignored-samples.json",
        tmp_path / "ignored-ledger.json",
        output,
    )

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert document["allowlistPolicyFileSha256"] == policy_file_sha256
    assert document["allowlistPolicyDeclaredSha256"] == policy["recoveryV4SourceAllowlistPolicySha256"]
    assert document["records"] == records


def test_existing_acquisition_root_fails_before_any_remote_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "external" / "existing-source-root"
    source_root.mkdir(parents=True)
    plan = _empty_validated_plan(tmp_path / "external" / "quarantined-root")
    monkeypatch.setattr(
        recovery,
        "load_recovery_v4_source_acquisition_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        recovery,
        "_resolve_remote_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("existing source root must reject before remote resolve")),
    )

    with pytest.raises(recovery.RecoveryV4SourceError, match="already exists"):
        recovery.acquire_recovery_v4_sources(
            tmp_path / "plan.json",
            tmp_path / "ledger.json",
            source_root,
            tmp_path / "external" / "source-manifest.json",
        )


def test_policy_record_rejects_a_tampered_rank_digest() -> None:
    _policy, _file_sha256, records = recovery.load_recovery_v4_allowlist_policy()
    tampered = dict(records[0])
    tampered["metadataRankSha256"] = "sha256:" + "0" * 64

    with pytest.raises(recovery.RecoveryV4SourceError, match="metadata rank digest"):
        recovery._validate_policy_record(tampered)


def test_source_plan_output_under_quarantine_root_fails_before_metadata_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quarantined_root = tmp_path / "external" / "fresh_normal_holdout_v1"
    quarantined_root.mkdir(parents=True)
    ledger = {"recoveryV4NonOverlapMetadataLedgerSha256": _digest("a")}
    monkeypatch.setattr(
        recovery,
        "load_recovery_v4_non_overlap_metadata_ledger",
        lambda *_args, **_kwargs: (ledger, _digest("b"), set(), set(), quarantined_root),
    )
    monkeypatch.setattr(
        recovery,
        "_load_pinned_source_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("quarantined plan output must reject before metadata read")),
    )

    with pytest.raises(recovery.RecoveryV4SourceError, match="overlaps the known quarantined cohort root"):
        recovery.create_recovery_v4_source_acquisition_plan(
            tmp_path / "metadata.json",
            tmp_path / "ledger.json",
            quarantined_root / "source-plan.json",
        )


def test_non_overlap_ledger_output_under_quarantine_root_fails_before_source_metadata_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quarantined_root = tmp_path / "external" / "fresh_normal_holdout_v1"
    quarantined_root.mkdir(parents=True)
    monkeypatch.setattr(
        recovery,
        "_read_json_file",
        lambda *_args, **_kwargs: ({}, _digest("a"), b"{}"),
    )
    monkeypatch.setattr(recovery, "_validate_historical_usage_ledger", lambda *_args, **_kwargs: {_digest("b")})
    monkeypatch.setattr(recovery, "_validate_quarantine_incident", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        recovery,
        "_derive_quarantined_cohort_root",
        lambda *_args, **_kwargs: (quarantined_root, recovery._canonical_absolute_path_text(quarantined_root)),
    )
    monkeypatch.setattr(
        recovery,
        "_extract_known_quarantined_source_records",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must reject before extracting cohort metadata")),
    )

    with pytest.raises(recovery.RecoveryV4SourceError, match="overlaps the known quarantined cohort root"):
        recovery.freeze_recovery_v4_non_overlap_metadata_ledger(
            tmp_path / "historical.json",
            tmp_path / "incident.json",
            tmp_path / "normal_holdout.json",
            quarantined_root / "non-overlap-ledger.json",
        )


@pytest.mark.parametrize("target_kind", ("source-root", "manifest-output"))
def test_quarantined_root_overlap_rejects_before_remote_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    quarantined_root = tmp_path / "external" / "fresh_normal_holdout_v1"
    quarantined_root.mkdir(parents=True)
    source_root = tmp_path / "external" / "recovery-v4-source-root"
    manifest_output = tmp_path / "external" / "recovery-v4-manifest.json"
    if target_kind == "source-root":
        source_root = quarantined_root / "recovery-v4-source-root"
    else:
        manifest_output = quarantined_root / "recovery-v4-manifest.json"
    plan = _empty_validated_plan(quarantined_root)
    events: list[str] = []
    monkeypatch.setattr(
        recovery,
        "load_recovery_v4_source_acquisition_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        recovery,
        "_resolve_remote_identity",
        lambda *_args, **_kwargs: events.append("resolve") or (_ for _ in ()).throw(AssertionError("must not resolve")),
    )

    with pytest.raises(recovery.RecoveryV4SourceError, match="overlaps the known quarantined cohort root"):
        recovery.acquire_recovery_v4_sources(
            tmp_path / "plan.json",
            tmp_path / "ledger.json",
            source_root,
            manifest_output,
        )
    assert events == []


def test_quarantined_root_containment_also_rejects_an_ancestor_candidate(tmp_path: Path) -> None:
    quarantined_root = tmp_path / "external" / "fresh_normal_holdout_v1"
    quarantined_root.mkdir(parents=True)

    with pytest.raises(recovery.RecoveryV4SourceError, match="overlaps the known quarantined cohort root"):
        recovery._assert_outside_known_quarantined_cohort_root(
            quarantined_root.parent,
            quarantined_root=quarantined_root,
            description="Recovery V4 ancestor candidate",
        )


def test_self_digested_omitted_historical_hash_rejects_before_remote_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical_hashes = [_digest("1"), _digest("2")]
    quarantined_record = {"sourceRemotePath": "data/quarantined/old.png", "sourceSha256": _digest("3")}
    quarantined_records = [quarantined_record]
    quarantined_root = tmp_path / "external" / "fresh_normal_holdout_v1"
    quarantined_root.mkdir(parents=True)
    root_text = recovery._canonical_absolute_path_text(quarantined_root)
    monkeypatch.setattr(recovery, "KNOWN_HISTORICAL_USAGE_LEDGER_FILE_SHA256", _digest("4"))
    monkeypatch.setattr(recovery, "KNOWN_HISTORICAL_USAGE_LEDGER_DECLARED_SHA256", _digest("5"))
    monkeypatch.setattr(recovery, "KNOWN_HISTORICAL_USAGE_SOURCE_HASH_COUNT", len(historical_hashes))
    monkeypatch.setattr(
        recovery,
        "KNOWN_HISTORICAL_USAGE_SOURCE_HASH_IDENTITY_SHA256",
        recovery._historical_source_hash_identity(historical_hashes),
    )
    monkeypatch.setattr(recovery, "KNOWN_QUARANTINE_INCIDENT_DECLARED_SHA256", _digest("6"))
    monkeypatch.setattr(recovery, "KNOWN_QUARANTINED_COHORT_MANIFEST_FILE_SHA256", _digest("7"))
    monkeypatch.setattr(recovery, "KNOWN_QUARANTINED_COHORT_MANIFEST_DECLARED_SHA256", _digest("8"))
    monkeypatch.setattr(recovery, "KNOWN_QUARANTINED_COHORT_SOURCE_RECORD_COUNT", 1)
    monkeypatch.setattr(
        recovery,
        "KNOWN_QUARANTINED_COHORT_SOURCE_IDENTITY_SHA256",
        recovery._quarantined_source_record_identity(quarantined_records),
    )
    known_entry = {
        "quarantineIncidentFileSha256": _digest("9"),
        "quarantineIncidentDeclaredSha256": recovery.KNOWN_QUARANTINE_INCIDENT_DECLARED_SHA256,
        "cohortManifestFileSha256": recovery.KNOWN_QUARANTINED_COHORT_MANIFEST_FILE_SHA256,
        "cohortManifestDeclaredSha256": recovery.KNOWN_QUARANTINED_COHORT_MANIFEST_DECLARED_SHA256,
        "cohortSourceRecordCount": 1,
        "cohortSourceRecordIdentitySha256": recovery.KNOWN_QUARANTINED_COHORT_SOURCE_IDENTITY_SHA256,
        "cohortSourceRecords": quarantined_records,
        "quarantinedCohortRoot": root_text,
        "quarantinedCohortRootIdentitySha256": recovery._quarantined_cohort_root_identity(root_text),
    }
    # This is deliberately self-consistent, but omits historical_hashes[1].
    # Its own digest is recomputed, so only the compiled historical identity
    # can catch the narrowing attempt.
    forged: dict[str, object] = {
        "schemaVersion": recovery.RECOVERY_V4_NON_OVERLAP_LEDGER_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": recovery.NON_OVERLAP_LEDGER_PURPOSE,
        "historicalNormalUsageLedgerFileSha256": recovery.KNOWN_HISTORICAL_USAGE_LEDGER_FILE_SHA256,
        "historicalNormalUsageLedgerDeclaredSha256": recovery.KNOWN_HISTORICAL_USAGE_LEDGER_DECLARED_SHA256,
        "historicalSourceHashCount": 1,
        "historicalSourceHashIdentitySha256": recovery._historical_source_hash_identity([historical_hashes[0]]),
        "historicalSourceSha256": [historical_hashes[0]],
        "knownQuarantinedCohort": known_entry,
        "excludedRemotePaths": [quarantined_record["sourceRemotePath"]],
        "excludedSourceSha256": sorted([historical_hashes[0], quarantined_record["sourceSha256"]]),
    }
    forged["recoveryV4NonOverlapMetadataLedgerSha256"] = recovery._document_digest(
        forged, "recoveryV4NonOverlapMetadataLedgerSha256"
    )
    ledger_path = _write_json(tmp_path / "external" / "forged-ledger.json", forged)
    events: list[str] = []
    monkeypatch.setattr(
        recovery,
        "_resolve_remote_identity",
        lambda *_args, **_kwargs: events.append("resolve") or (_ for _ in ()).throw(AssertionError("must not resolve")),
    )

    with pytest.raises(recovery.RecoveryV4SourceError, match="historical source hashes do not match the trusted identity"):
        recovery.acquire_recovery_v4_sources(
            tmp_path / "missing-plan.json",
            ledger_path,
            tmp_path / "external" / "new-source-root",
            tmp_path / "external" / "new-source-manifest.json",
        )
    assert events == []


def test_partial_path_substitution_fails_before_hard_link_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010804000000b51c0c02"
        "0000000b4944415478da6364f80f00010501012718e3660000000049454e44ae426082"
    )
    expected_sha256 = "sha256:" + hashlib.sha256(image_bytes).hexdigest()

    class Response:
        def __init__(self) -> None:
            self._sent = False

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://huggingface.co/fake-content"

        def read(self, _size: int) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return image_bytes

    monkeypatch.setattr(recovery, "urlopen", lambda *_args, **_kwargs: Response())

    def substitute(path: Path) -> None:
        path.unlink()
        path.write_bytes(b"substituted-after-download")

    monkeypatch.setattr(recovery, "_verify_decodable_image", substitute)
    destination = tmp_path / "external" / "source-root" / "images" / "bottle" / "image.png"

    with pytest.raises(recovery.RecoveryV4SourceError, match="changed while it was verified"):
        recovery._stream_verified_image(
            "https://huggingface.co/fake-content",
            destination,
            expected_sha256=expected_sha256,
            expected_bytes=len(image_bytes),
            timeout_seconds=1.0,
        )
    assert not destination.exists()


def test_verified_image_uses_new_only_hard_link_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010804000000b51c0c02"
        "0000000b4944415478da6364f80f00010501012718e3660000000049454e44ae426082"
    )
    expected_sha256 = "sha256:" + hashlib.sha256(image_bytes).hexdigest()

    class Response:
        def __init__(self) -> None:
            self._sent = False

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://huggingface.co/fake-content"

        def read(self, _size: int) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return image_bytes

    monkeypatch.setattr(recovery, "urlopen", lambda *_args, **_kwargs: Response())
    destination = tmp_path / "external" / "source-root" / "images" / "bottle" / "image.png"

    recovery._stream_verified_image(
        "https://huggingface.co/fake-content",
        destination,
        expected_sha256=expected_sha256,
        expected_bytes=len(image_bytes),
        timeout_seconds=1.0,
    )

    assert destination.is_file()
    assert destination.read_bytes() == image_bytes
    assert not destination.with_name(destination.name + ".partial").exists()
