from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from phone_dino import mvtec_normal_holdout as holdout
from phone_dino import mvtec_normal_successor as successor


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _record(
    source_root: Path,
    *,
    category: str,
    partition: str,
    ordinal: int,
    decodable: bool,
) -> tuple[dict, Path]:
    extension = "png" if decodable else "bin"
    relative = Path("images") / partition.lower() / category / f"{ordinal}.{extension}"
    path = source_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if decodable:
        Image.new("RGB", (17, 13), (ordinal % 255, (ordinal * 37) % 255, (ordinal * 71) % 255)).save(path, format="PNG")
    else:
        path.write_bytes(f"{partition}:{category}:{ordinal}:must-not-be-opened".encode("utf-8"))
    source_sha256 = holdout.sha256_file(path)
    return {
        "caseId": f"mvtec-ad/{category}/train-good/{source_sha256[7:]}",
        "category": category,
        "relativePath": relative.as_posix(),
        "sourceSha256": source_sha256,
        "sourceGroupId": f"CONTENT_SHA256:{source_sha256[7:]}",
        "acquisitionStratum": "OFFICIAL_MVTEC_TRAIN_GOOD",
        "partition": partition,
        "sourceRemotePath": f"data/{partition.lower()}/{category}/{ordinal}.{extension}",
        "expectedRemoteSha256": source_sha256,
        "expectedRemoteBytes": path.stat().st_size,
        "kind": "NOMINAL",
        "defect": "good",
    }, path


def _parent_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, successor._ParentContext, dict[str, Path]]:
    source_root = tmp_path / "source_bytes"
    records: list[dict] = []
    reserve_paths: dict[str, Path] = {}

    # These deliberately non-image parent sources prove that seal/plan/envelope
    # construction does not open historical or confirmation image bytes.
    for index, partition in enumerate(successor.PARENT_HISTORICAL_PARTITIONS):
        record, _path = _record(
            source_root,
            category="capsule",
            partition=partition,
            ordinal=1_000 + index,
            decodable=False,
        )
        records.append(record)

    quota_totals = {
        quota["category"]: sum(
            int(quota[name])
            for name in ("fitCount", "thresholdTuningCount", "normalSelectionCount", "reserveUntouchedCount")
        )
        for quota in successor.SUCCESSOR_CATEGORY_QUOTAS
    }
    ordinal = 0
    for category, count in quota_totals.items():
        for _ in range(count):
            record, path = _record(
                source_root,
                category=category,
                partition=successor.PARENT_RESERVE_PARTITION,
                ordinal=ordinal,
                decodable=True,
            )
            records.append(record)
            reserve_paths[record["sourceSha256"]] = path
            ordinal += 1

    for category in ("capsule", "metal_nut", "tile"):
        for ordinal in range(32):
            record, _path = _record(
                source_root,
                category=category,
                partition=successor.PARENT_CONFIRMATION_PARTITION,
                ordinal=ordinal,
                decodable=False,
            )
            records.append(record)

    records.sort(key=lambda item: item["caseId"])
    holdout_document: dict = {
        "schemaVersion": holdout.NORMAL_HOLDOUT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": holdout.HOLDOUT_PURPOSE,
        "blindPolicy": holdout.HOLDOUT_BLIND_POLICY,
        "sourcePoolFileSha256": _digest("1"),
        "sourcePoolDeclaredSha256": _digest("2"),
        "historicalLedgerFileSha256": _digest("3"),
        "historicalLedgerDeclaredSha256": _digest("4"),
        "planFileSha256": _digest("5"),
        "planDeclaredSha256": _digest("6"),
        "historyExclusion": {
            "algorithm": holdout.HISTORY_EXCLUSION_ALGORITHM,
            "matchedHistoricalSourceCount": 0,
            "excludedSourceGroupCount": 0,
            "eligibleSourceCount": len(records),
            "eligibleSourceIdentitySha256": holdout.canonical_json_sha256([]),
        },
        "records": records,
        "developmentIdentitySha256": holdout._holdout_partition_identity(records, {"FIT", "THRESHOLD_TUNING"}),
        "normalSelectionIdentitySha256": holdout._holdout_partition_identity(records, {"NORMAL_SELECTION"}),
        "normalConfirmationIdentitySha256": holdout._holdout_partition_identity(records, {"NORMAL_CONFIRMATION"}),
        "reserveUntouchedIdentitySha256": holdout._holdout_partition_identity(records, {"RESERVE_UNTOUCHED"}),
    }
    holdout_document["normalHoldoutManifestSha256"] = holdout._document_digest(
        holdout_document,
        "normalHoldoutManifestSha256",
    )
    holdout_path = tmp_path / "parent" / "normal_holdout.json"
    _write_json(holdout_path, holdout_document)
    parsed_records = holdout._validate_closed_normal_holdout_document(holdout_document)
    historical, reserve, confirmation = successor._classify_parent_records(
        parsed_records,
        holdout_document=holdout_document,
    )
    context = successor._ParentContext(
        holdout_path=holdout_path,
        holdout_document=holdout_document,
        holdout_file_sha256=successor.sha256_file(holdout_path),
        contract={"contractSha256": _digest("7")},
        contract_file_sha256=_digest("8"),
        claim={"claimSha256": _digest("9")},
        claim_file_sha256=_digest("a"),
        selection_observation={
            "selectionReceiptFileSha256": _digest("b"),
            "selectionReceiptDeclaredSha256": _digest("c"),
            "selectionObservationSha256": _digest("d"),
        },
        selection_observation_file_sha256=_digest("e"),
        selection_lock={
            "selectionLockSha256": _digest("f"),
            "decision": {"state": "NO_ELIGIBLE_CONFIGURATION", "selectedCandidateId": None},
        },
        selection_lock_file_sha256=_digest("0"),
        historical_sources=historical,
        reserve_inputs=reserve,
        preserved_confirmation=confirmation,
    )
    contract_path = tmp_path / "parent" / "selection_contract.json"
    _write_json(contract_path, {})
    monkeypatch.setattr(successor, "_load_parent_context", lambda *_args, **_kwargs: context)
    return holdout_path, contract_path, source_root, context, reserve_paths


def test_successor_seal_is_one_time_reserve_only_and_preserves_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    holdout_path, contract_path, _source_root, _context, _reserve_paths = _parent_fixture(tmp_path, monkeypatch)

    seal = successor.create_fresh_normal_successor_seal(holdout_path, contract_path)
    seal_path = successor.fresh_normal_successor_seal_path(holdout_path)
    assert seal_path.is_file()
    assert seal["delegationPolicy"] == successor.FRESH_NORMAL_SUCCESSOR_DELEGATION_POLICY
    assert seal["preservedParentNormalConfirmation"]["recordCount"] == 96
    assert len(seal["delegatedReserveInputs"]) == 93
    assert set(item["parentPartition"] for item in seal["delegatedReserveInputs"]) == {"RESERVE_UNTOUCHED"}
    assert {item["partition"] for item in seal["historicalParentSources"]} == set(successor.PARENT_HISTORICAL_PARTITIONS)
    assert not {
        item["sourceSha256"] for item in seal["delegatedReserveInputs"]
    }.intersection({item["sourceSha256"] for item in seal["historicalParentSources"]})

    loaded, loaded_file_sha256 = successor.load_validated_fresh_normal_successor_seal(holdout_path, contract_path)
    assert loaded == seal
    assert loaded_file_sha256 == successor.sha256_file(seal_path)
    with pytest.raises(successor.FreshNormalSuccessorError, match="already exists"):
        successor.create_fresh_normal_successor_seal(holdout_path, contract_path)


def test_successor_seal_rejects_nonfailed_parent_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    holdout_path, contract_path, _source_root, context, _reserve_paths = _parent_fixture(tmp_path, monkeypatch)
    locked_context = replace(
        context,
        selection_lock={
            "selectionLockSha256": _digest("f"),
            "decision": {"state": "RESEARCH_CONFIGURATION_LOCKED", "selectedCandidateId": "candidate"},
        },
    )
    monkeypatch.setattr(successor, "_load_parent_context", lambda *_args, **_kwargs: locked_context)
    with pytest.raises(successor.FreshNormalSuccessorError, match="NO_ELIGIBLE_CONFIGURATION"):
        successor.create_fresh_normal_successor_seal(holdout_path, contract_path)


def test_successor_seal_tamper_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    holdout_path, contract_path, _source_root, _context, _reserve_paths = _parent_fixture(tmp_path, monkeypatch)
    successor.create_fresh_normal_successor_seal(holdout_path, contract_path)
    seal_path = successor.fresh_normal_successor_seal_path(holdout_path)
    sealed = json.loads(seal_path.read_text(encoding="utf-8"))
    sealed["delegatedReserveInputs"][0]["category"] = "tampered"
    _write_json(seal_path, sealed)
    with pytest.raises(successor.FreshNormalSuccessorError, match="fixed 93-image category allocation|digest does not match"):
        successor.load_validated_fresh_normal_successor_seal(holdout_path, contract_path)


def test_successor_envelope_phase_loader_opens_only_requested_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    holdout_path, contract_path, source_root, _context, reserve_paths = _parent_fixture(tmp_path, monkeypatch)
    successor.create_fresh_normal_successor_seal(holdout_path, contract_path)
    plan_path = tmp_path / "successor" / "successor_plan.json"
    plan = successor.create_fresh_normal_successor_plan(holdout_path, contract_path, plan_path)
    envelope_path = tmp_path / "successor" / "successor_envelope.json"
    envelope = successor.create_fresh_normal_successor_envelope(holdout_path, contract_path, plan_path, envelope_path)
    assert plan["resultLabel"] == successor.FRESH_NORMAL_SUCCESSOR_RESULT_LABEL
    assert envelope["independenceLabel"] == successor.FRESH_NORMAL_SUCCESSOR_INDEPENDENCE_LABEL

    fit_hashes = {record["sourceSha256"] for record in envelope["records"] if record["partition"] == "FIT"}
    for source_sha256, path in reserve_paths.items():
        if source_sha256 not in fit_hashes:
            path.write_bytes(b"non-fit-reserve-must-not-be-opened")

    loaded, _file_sha256, records = successor.load_successor_safe_normal_inputs(
        holdout_path,
        contract_path,
        plan_path,
        envelope_path,
        source_root=source_root,
        partitions={"FIT"},
    )
    assert loaded["successorEnvelopeSha256"] == envelope["successorEnvelopeSha256"]
    assert len(records) == 36
    assert {record["partition"] for record in records} == {"FIT"}


def test_phase_loader_rejects_a_self_consistent_envelope_not_bound_to_the_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holdout_path, contract_path, source_root, _context, _reserve_paths = _parent_fixture(tmp_path, monkeypatch)
    successor.create_fresh_normal_successor_seal(holdout_path, contract_path)
    plan_path = tmp_path / "successor" / "successor_plan.json"
    successor.create_fresh_normal_successor_plan(holdout_path, contract_path, plan_path)
    envelope_path = tmp_path / "successor" / "successor_envelope.json"
    successor.create_fresh_normal_successor_envelope(holdout_path, contract_path, plan_path, envelope_path)

    malicious = json.loads(envelope_path.read_text(encoding="utf-8"))
    capsule = next(item for item in malicious["records"] if item["partition"] == "FIT" and item["category"] == "capsule")
    metal_nut = next(item for item in malicious["records"] if item["partition"] == "FIT" and item["category"] == "metal_nut")
    capsule["category"], metal_nut["category"] = metal_nut["category"], capsule["category"]
    malicious["successorPartitionIdentities"] = {
        partition: successor._partition_identity(malicious["records"], partition)
        for partition in successor.SUCCESSOR_PARTITIONS
    }
    malicious["successorEnvelopeSha256"] = successor._document_digest(
        malicious,
        "successorEnvelopeSha256",
    )
    malicious_path = tmp_path / "successor" / "malicious_self_consistent_envelope.json"
    _write_json(malicious_path, malicious)

    def fail_if_an_image_is_opened(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("unbound envelope must be rejected before image access")

    monkeypatch.setattr(successor, "_safe_file_under", fail_if_an_image_is_opened)
    with pytest.raises(successor.FreshNormalSuccessorError, match="does not bind the sealed reserve"):
        successor.load_successor_safe_normal_inputs(
            holdout_path,
            contract_path,
            plan_path,
            malicious_path,
            source_root=source_root,
            partitions={"FIT"},
        )


def test_successor_seal_rejects_reparse_partition_access_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    holdout_path, contract_path, _source_root, _context, _reserve_paths = _parent_fixture(tmp_path, monkeypatch)
    access_root = holdout_path.parent / successor.FRESH_NORMAL_SUCCESSOR_PARTITION_ACCESS_DIRECTORY
    redirected_root = tmp_path / "redirected_partition_access"
    redirected_root.mkdir()

    # A real Windows junction/symlink resolves before inspection.  Simulate
    # that behavior without requiring the test host to grant symlink rights:
    # the raw root must reach the reparse check before any resolve call.
    original_resolve = Path.resolve

    def simulated_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if path == access_root:
            return redirected_root
        return original_resolve(path, *args, **kwargs)

    def simulated_reparse(path: Path) -> bool:
        return path == access_root

    monkeypatch.setattr(Path, "resolve", simulated_resolve)
    monkeypatch.setattr(successor, "_is_link_or_reparse_point", simulated_reparse)
    with pytest.raises(successor.FreshNormalSuccessorError, match="contains a symbolic link"):
        successor.create_fresh_normal_successor_seal(holdout_path, contract_path)
