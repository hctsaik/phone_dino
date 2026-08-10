from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from phone_dino import mvtec_successor_evaluator_v2 as evaluator
from phone_dino import mvtec_successor_selection_v2 as selection


_REAL_CONTRACT_LEDGER_VALIDATOR = selection._validate_contract_development_evidence_ledger


@pytest.fixture(autouse=True)
def _stub_remote_git_ledger_for_non_git_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most phase-isolation tests use synthetic contracts, not a pushed Git repo."""

    monkeypatch.setattr(selection, "_validate_contract_development_evidence_ledger", lambda *_args, **_kwargs: {})


def _digest(label: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _git(arguments: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(["git", *arguments], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")


def _contract_projection(contract: dict) -> dict:
    return {
        "augmentation": contract["augmentation"],
        "prototypeInputIdentities": contract["prototypeInputIdentities"],
        "prototypeInputCounts": contract["prototypeInputCounts"],
        "featureExtractor": contract["featureExtractor"],
        "featureExtractorIdentitySha256": contract["featureExtractorIdentitySha256"],
        "selectionProtocolModuleSha256": contract["selectionProtocolModuleSha256"],
        "candidateBindings": contract["candidateReports"],
        "candidateUniverseIdentitySha256": contract["candidateUniverseIdentitySha256"],
    }


def _pushed_ledger_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract: dict
) -> tuple[Path, Path, dict, dict]:
    """Create a local pushed-history analogue for Git-anchor tests."""

    repository = tmp_path / "ledger-repository"
    repository.mkdir()
    _git(["init", "-b", "master"], cwd=repository)
    _git(["config", "user.email", "test@example.invalid"], cwd=repository)
    _git(["config", "user.name", "PhoneDINO test"], cwd=repository)
    ledger = selection._development_evidence_ledger_document(_contract_projection(contract))
    ledger_path = repository / selection.DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH
    _write_json(ledger_path, ledger)
    _git(["add", selection.DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH], cwd=repository)
    _git(["commit", "-m", "freeze development evidence"], cwd=repository)
    bare_remote = tmp_path / "pushed-audit.git"
    _git(["init", "--bare", str(bare_remote)], cwd=tmp_path)
    remote_url = bare_remote.as_uri()
    _git(["remote", "add", "origin", remote_url], cwd=repository)
    _git(["push", "origin", "HEAD:refs/heads/master"], cwd=repository)
    monkeypatch.setattr(selection, "CANONICAL_GIT_AUDIT_REMOTE_URL", remote_url)
    monkeypatch.setattr(selection, "REQUIRED_GIT_AUDIT_REMOTE_REF", "refs/heads/master")
    monkeypatch.setattr(selection, "CANONICAL_GIT_AUDIT_OBJECT_FORMAT", "sha1")
    loaded_ledger, anchor = selection._load_head_development_evidence_ledger(ledger_path, repository_root=repository)
    return repository, ledger_path, loaded_ledger, anchor


def _record(category: str, partition: str, index: int, *, image_path: Path | None = None) -> dict:
    value = {
        "caseId": f"successor/{category}/{partition.lower()}/{index:02d}",
        "category": category,
        "partition": partition,
        "kind": "NOMINAL",
        "defect": "good",
        "sourceSha256": _digest(f"{category}/{partition}/{index}"),
        "sourceGroupId": f"CONTENT_SHA256:{index:02d}-{category}",
        "acquisitionStratum": "OFFICIAL_MVTEC_TRAIN_GOOD",
        "expectedRemoteSha256": _digest(f"remote/{category}/{partition}/{index}"),
        "expectedRemoteBytes": 17,
    }
    if image_path is not None:
        value["imagePath"] = image_path
    return value


def _selection_inputs() -> list[dict]:
    return [
        {field: record[field] for field in selection.NORMAL_INPUT_FIELDS}
        for category in evaluator.SUCCESSOR_V2_CATEGORIES
        for index in range(8)
        for record in [_record(category, "NORMAL_SELECTION", index)]
    ]


def _contract(registry_root: Path) -> dict:
    registry_root.mkdir(parents=True, exist_ok=True)
    extractor = {"fixture": "same-successor-v2-extractor"}
    extractor_digest = selection.canonical_json_sha256(extractor)
    bindings = []
    for configuration in evaluator.PRE_REGISTERED_CANDIDATES:
        thresholds = {category: 0.1 for category in evaluator.SUCCESSOR_V2_CATEGORIES}
        bindings.append({
            "candidateId": configuration["id"],
            "developmentReportFileSha256": _digest(f"report-file/{configuration['id']}"),
            "developmentReportDeclaredSha256": _digest(f"report-declared/{configuration['id']}"),
            "candidateConfiguration": dict(configuration),
            "candidateConfigurationSha256": selection.canonical_json_sha256(configuration),
            "prototypeInputPolicy": configuration["prototypeInputPolicy"],
            "featureInputIdentitySha256": _digest(f"features/{configuration['id']}"),
            "calibrationInputIdentitySha256": _digest(f"calibration/{configuration['id']}"),
            "featureExtractorIdentitySha256": extractor_digest,
            "thresholds": thresholds,
            "thresholdsIdentitySha256": selection.canonical_json_sha256(thresholds),
            "augmentationManifestFileSha256": None if configuration["prototypeInputPolicy"] == evaluator.RAW_FIT_ONLY else _digest("r3-file"),
            "augmentationManifestDeclaredSha256": None if configuration["prototypeInputPolicy"] == evaluator.RAW_FIT_ONLY else _digest("r3-declared"),
            "augmentationRecipeFileSha256": None if configuration["prototypeInputPolicy"] == evaluator.RAW_FIT_ONLY else _digest("r3-recipe"),
        })
    parent = {
        "holdoutManifestFileSha256": _digest("parent-holdout-file"),
        "holdoutManifestDeclaredSha256": _digest("parent-holdout-declared"),
        "selectionContractFileSha256": _digest("parent-contract-file"),
        "selectionContractDeclaredSha256": _digest("parent-contract-declared"),
        "selectionClaimFileSha256": _digest("parent-claim-file"),
        "selectionClaimDeclaredSha256": _digest("parent-claim-declared"),
        "selectionReceiptFileSha256": _digest("parent-receipt-file"),
        "selectionReceiptDeclaredSha256": _digest("parent-receipt-declared"),
        "selectionObservationFileSha256": _digest("parent-observation-file"),
        "selectionObservationDeclaredSha256": _digest("parent-observation-declared"),
        "selectionLockFileSha256": _digest("parent-lock-file"),
        "selectionLockDeclaredSha256": _digest("parent-lock-declared"),
        "selectionLockState": "NO_ELIGIBLE_CONFIGURATION",
        "parentReserveUntouchedIdentitySha256": _digest("parent-reserve"),
        "parentNormalConfirmationIdentitySha256": _digest("parent-confirmation"),
    }
    inputs = _selection_inputs()
    document = {
        "schemaVersion": selection.SUCCESSOR_V2_SELECTION_CONTRACT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": selection.CONTRACT_PURPOSE,
        "phase": selection.CONTRACT_PHASE,
        "blindPolicy": selection.BLIND_POLICY,
        "resultLabel": selection.RESULT_LABEL,
        "independenceLabel": selection.INDEPENDENCE_LABEL,
        "delegationPolicy": selection.DELEGATION_POLICY,
        "selectionInput": dict(selection.SELECTION_INPUT),
        "parentEvidence": parent,
        "successorPlanFileSha256": _digest("plan-file"),
        "successorPlanDeclaredSha256": _digest("plan-declared"),
        "successorEnvelopeFileSha256": _digest("envelope-file"),
        "successorEnvelopeDeclaredSha256": _digest("envelope-declared"),
        "successorEnvelopeSelectionIdentitySha256": _digest("successor-selection"),
        "successorSelectionInputs": inputs,
        "successorSelectionInputIdentitySha256": selection.canonical_json_sha256(inputs),
        "prototypeInputIdentities": {
            evaluator.RAW_FIT_ONLY: _digest("raw-fit-prototype-inputs"),
            evaluator.RAW_FIT_PLUS_AUGMENTATION_R3: _digest("r3-fit-prototype-inputs"),
        },
        "prototypeInputCounts": dict(selection.PROTOTYPE_INPUT_COUNTS),
        "developmentEvidenceLedger": {
            "mode": selection.PUSHED_GIT_AUDIT_ONLY,
            "canonicalRemoteUrl": selection.CANONICAL_GIT_AUDIT_REMOTE_URL,
            "requiredRemoteRef": selection.REQUIRED_GIT_AUDIT_REMOTE_REF,
            "gitObjectFormat": selection.CANONICAL_GIT_AUDIT_OBJECT_FORMAT,
            "gitCommitOid": "a" * 40,
            "repositoryPath": selection.DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH,
            "gitBlobOid": "b" * 40,
            "ledgerBlobSha256": _digest("ledger-blob"),
            "ledgerDeclaredSha256": _digest("ledger-declared"),
            "ledgerProjectionSha256": _digest("ledger-projection"),
        },
        "selectionProtocolModuleSha256": selection.sha256_file(Path(selection.__file__)),
        "augmentation": {
            "manifestFileSha256": _digest("r3-file"),
            "manifestDeclaredSha256": _digest("r3-declared"),
            "recipeFileSha256": _digest("r3-recipe"),
            "successorFitIdentitySha256": _digest("fit"),
            "variantsPerParent": 3,
        },
        "consumptionRegistry": {
            "schemaVersion": selection.CONSUMPTION_REGISTRY_SCHEMA,
            "root": str(registry_root.resolve()),
            "selectionSlotKey": "",
        },
        "featureExtractor": extractor,
        "featureExtractorIdentitySha256": extractor_digest,
        "candidateReports": bindings,
        "candidateUniverseIdentitySha256": selection.canonical_json_sha256(bindings),
        "selectionGates": dict(selection.SELECTION_GATES),
        "selectionObjective": dict(selection.SELECTION_OBJECTIVE),
    }
    document["consumptionRegistry"]["selectionSlotKey"] = selection.canonical_json_sha256({
        "schemaVersion": selection.CONSUMPTION_REGISTRY_SCHEMA,
        "parentPartitionAccessRoot": str(registry_root.resolve()),
        "parentHoldoutFileSha256": parent["holdoutManifestFileSha256"],
        "parentHoldoutDeclaredSha256": parent["holdoutManifestDeclaredSha256"],
        "successorEnvelopeSelectionIdentitySha256": document["successorEnvelopeSelectionIdentitySha256"],
    })
    document["contractSha256"] = selection._document_digest(document, "contractSha256")
    return document


def _stub_parent_registry(
    monkeypatch: pytest.MonkeyPatch, contract: dict, registry_root: Path
) -> Path:
    parent_contract_path = registry_root.parent / "parent-selection-contract.json"

    def loader(*_args: object, **_kwargs: object) -> tuple[dict, str]:
        return {
            "contractSha256": contract["parentEvidence"]["selectionContractDeclaredSha256"],
            "consumptionRegistry": {"root": str(registry_root.resolve())},
        }, contract["parentEvidence"]["selectionContractFileSha256"]

    monkeypatch.setattr(selection.parent_protocol, "load_validated_fresh_selection_contract", loader)
    return parent_contract_path


def _semantic_report_fixture(tmp_path: Path) -> tuple[dict, dict, dict, str, str, str]:
    """Return a fully self-consistent raw-FIT report and its JSON chain."""

    parent = _contract(tmp_path / "semantic-parent" / "partition_access")["parentEvidence"]
    originals = [
        _record(category, partition, index)
        for category in evaluator.SUCCESSOR_V2_CATEGORIES
        for partition, count in (("FIT", 12), ("THRESHOLD_TUNING", 4))
        for index in range(count)
    ]
    envelope = {
        "records": originals,
        "parentEvidence": parent,
        "sealFileSha256": _digest("seal-file"),
        "sealDeclaredSha256": _digest("seal-declared"),
        "planFileSha256": _digest("plan-file"),
        "planDeclaredSha256": _digest("plan-declared"),
        "successorEnvelopeSha256": _digest("envelope-declared"),
        "successorPartitionIdentities": {
            "FIT": _digest("fit"),
            "THRESHOLD_TUNING": _digest("tuning"),
            "NORMAL_SELECTION": _digest("selection"),
            "RESERVE_UNTOUCHED": _digest("reserve"),
        },
    }
    r3_records: list[dict] = []
    for parent_record in [record for record in originals if record["partition"] == "FIT"]:
        for variant_id, component in ((1, "registration"), (2, "illumination"), (3, "sensor_transport")):
            r3_records.append({
                "caseId": f"{parent_record['caseId']}/r3/{variant_id}",
                "parentCaseId": parent_record["caseId"],
                "parentSourceSha256": parent_record["sourceSha256"],
                "sourceGroupId": parent_record["sourceGroupId"],
                "category": parent_record["category"],
                "parentPartition": "FIT",
                "kind": "NOMINAL",
                "defect": "good",
                "variantId": variant_id,
                "component": component,
                "relativePath": f"images/{len(r3_records):03d}.jpg",
                "sourceSha256": _digest(f"r3/{parent_record['caseId']}/{variant_id}"),
                "parameters": {},
                "outputEncoding": {},
            })
    r3_records.sort(key=lambda item: item["caseId"])
    r3_manifest = {"augmentationManifestSha256": _digest("r3-declared"), "records": r3_records}
    raw_inputs, _r3_inputs = selection._expected_feature_inputs(envelope, r3_manifest=r3_manifest)
    calibration = [record for record in raw_inputs if record["partition"] == "THRESHOLD_TUNING"]
    scores = []
    for record in calibration:
        score = (int(record["caseId"].rsplit("/", 1)[1]) + 1) / 100.0
        scores.append({
            "caseId": record["caseId"], "category": record["category"], "partition": "THRESHOLD_TUNING",
            "kind": "NOMINAL", "defect": "good", "sourceSha256": record["sourceSha256"], "score": score,
            "maxPatchDistance": score + 0.01, "meanNearestPatchDistance": score - 0.005,
        })
    scores.sort(key=lambda item: item["caseId"])
    thresholds = {category: 0.04 for category in evaluator.SUCCESSOR_V2_CATEGORIES}
    categories = {
        category: {
            "fitOriginalCount": 12,
            "fitAugmentedCount": 0,
            "tuningOriginalCount": 4,
            "prototypePatchCount": 128,
            "fitPatchCount": 192,
            "patchGridHeight": 4,
            "patchGridWidth": 4,
            "thresholdFromRawTuning": 0.04,
            "tuningScoreMedian": 0.03,
            "tuningScoreP95": 0.04,
            "tuningScoreMax": 0.04,
        }
        for category in evaluator.SUCCESSOR_V2_CATEGORIES
    }
    configuration = evaluator.pre_registered_candidate_configuration("reserve-v2-raw-p2048-k5")
    extractor = {"fixture": "semantic-report-extractor", "evaluatorModuleSha256": _digest("evaluator-module")}
    report = {
        "schemaVersion": evaluator.SUCCESSOR_V2_DEVELOPMENT_REPORT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": evaluator.SUCCESSOR_V2_DEVELOPMENT_REPORT_PURPOSE,
        "phase": evaluator.SUCCESSOR_V2_DEVELOPMENT_PHASE,
        "blindPolicy": evaluator.SUCCESSOR_V2_BLIND_POLICY,
        "resultLabel": evaluator.SUCCESSOR_V2_RESULT_LABEL,
        "delegationPolicy": evaluator.SUCCESSOR_V2_DELEGATION_POLICY,
        "independenceLabel": "NOT_INDEPENDENT_PARENT_RESERVE_DERIVATION",
        "inputPolicy": "SUCCESSOR_RAW_FIT_PLUS_RAW_THRESHOLD_TUNING_ONLY",
        "parentHoldoutFileSha256": parent["holdoutManifestFileSha256"],
        "parentHoldoutDeclaredSha256": parent["holdoutManifestDeclaredSha256"],
        "parentSelectionContractFileSha256": parent["selectionContractFileSha256"],
        "parentSelectionContractDeclaredSha256": parent["selectionContractDeclaredSha256"],
        "parentNormalConfirmationIdentitySha256": parent["parentNormalConfirmationIdentitySha256"],
        "successorSealFileSha256": envelope["sealFileSha256"],
        "successorSealDeclaredSha256": envelope["sealDeclaredSha256"],
        "successorPlanFileSha256": envelope["planFileSha256"],
        "successorPlanDeclaredSha256": envelope["planDeclaredSha256"],
        "successorEnvelopeFileSha256": _digest("envelope-file"),
        "successorEnvelopeDeclaredSha256": envelope["successorEnvelopeSha256"],
        "successorFitIdentitySha256": envelope["successorPartitionIdentities"]["FIT"],
        "successorThresholdTuningIdentitySha256": envelope["successorPartitionIdentities"]["THRESHOLD_TUNING"],
        "augmentationManifestFileSha256": None,
        "augmentationManifestDeclaredSha256": None,
        "augmentationRecipeFileSha256": None,
        "augmentationParentFitIdentitySha256": None,
        "candidateConfiguration": configuration,
        "candidateConfigurationSha256": selection.canonical_json_sha256(configuration),
        "featureExtractor": extractor,
        "featureExtractorIdentitySha256": selection.canonical_json_sha256(extractor),
        "featureInputs": raw_inputs,
        "featureInputIdentitySha256": selection.canonical_json_sha256(raw_inputs),
        "calibrationInputs": calibration,
        "calibrationInputIdentitySha256": selection.canonical_json_sha256(calibration),
        "thresholds": thresholds,
        "categories": categories,
        "calibrationScores": scores,
        "normalOnlyEvidence": evaluator._build_normal_only_evidence(raw_inputs, calibration),
        "execution": {
            "evaluatorModuleSha256": extractor["evaluatorModuleSha256"],
            "evaluatorEntrypointSha256": _digest("evaluator-entrypoint"),
            "phaseTimingsSeconds": {
                "inputAssemblySeconds": 0.0,
                "provenanceSeconds": 0.0,
                "inputVerificationSeconds": 0.0,
                "featureInferenceSeconds": 0.0,
                "scoringSeconds": 0.0,
                "totalElapsedSeconds": 0.0,
            },
            "python": "fixture",
            "platform": "fixture",
            "numpyVersion": "fixture",
            "torchVersion": "fixture",
            "torchThreadCount": 1,
            "gitRevision": None,
            "gitWorktreeClean": None,
        },
    }
    report["developmentReportSha256"] = selection._document_digest(report, "developmentReportSha256")
    return report, envelope, r3_manifest, _digest("report-file"), _digest("r3-file"), _digest("r3-recipe")


def test_claim_slot_is_parent_registry_global_and_contract_copy_cannot_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "canonical-parent" / "partition_access"
    parent = tmp_path / "parent" / "normal_holdout.json"
    copied_parent = tmp_path / "copied-parent" / "normal_holdout.json"
    parent.parent.mkdir(parents=True)
    copied_parent.parent.mkdir(parents=True)
    contract = _contract(canonical_root)
    parent_contract = _stub_parent_registry(monkeypatch, contract, canonical_root)
    # This unit test is about registry-slot derivation.  The parent-chain
    # integration validation has its own adversarial regression below.
    monkeypatch.setattr(selection, "_validate_contract_chain", lambda *_args, **_kwargs: ({}, _digest("envelope-file")))
    contract_path = tmp_path / "contract" / "selection.json"
    _write_json(contract_path, contract)
    # The caller can hand the claim command a copied parent path, but it must
    # still consume the V1-contract-bound canonical registry slot.
    selection.create_successor_v2_selection_claim(
        copied_parent, parent_contract, tmp_path / "plan.json", tmp_path / "envelope.json", contract_path
    )
    assert selection.successor_v2_selection_path(contract, artifact="claim").parent == canonical_root.resolve()
    assert not (copied_parent.parent / "partition_access").exists()
    copied = tmp_path / "copied" / "selection.json"
    _write_json(copied, contract)
    with pytest.raises(selection.SuccessorV2SelectionError, match="already exists"):
        selection.create_successor_v2_selection_claim(
            parent, parent_contract, tmp_path / "plan.json", tmp_path / "envelope.json", copied
        )


def test_contract_rejects_report_policy_membership_mismatch(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "canonical-parent" / "partition_access")
    # A raw candidate cannot pretend to carry an R3 report identity once the
    # binding's declared policy is fixed in the contract.
    contract["candidateReports"][0]["prototypeInputPolicy"] = evaluator.RAW_FIT_PLUS_AUGMENTATION_R3
    contract["candidateUniverseIdentitySha256"] = selection.canonical_json_sha256(contract["candidateReports"])
    contract["contractSha256"] = selection._document_digest(contract, "contractSha256")
    path = tmp_path / "contract" / "selection.json"
    _write_json(path, contract)
    with pytest.raises(selection.SuccessorV2SelectionError, match="prototype policy"):
        selection.load_validated_successor_v2_selection_contract(path)


def test_self_digested_contract_cannot_relocate_canonical_parent_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "canonical-parent" / "partition_access"
    alternate_root = tmp_path / "copied-parent" / "partition_access"
    contract = _contract(canonical_root)
    parent_contract = _stub_parent_registry(monkeypatch, contract, canonical_root)
    altered = json.loads(json.dumps(contract))
    alternate_root.mkdir(parents=True)
    altered["consumptionRegistry"]["root"] = str(alternate_root.resolve())
    altered["consumptionRegistry"]["selectionSlotKey"] = selection.canonical_json_sha256({
        "schemaVersion": selection.CONSUMPTION_REGISTRY_SCHEMA,
        "parentPartitionAccessRoot": str(alternate_root.resolve()),
        "parentHoldoutFileSha256": altered["parentEvidence"]["holdoutManifestFileSha256"],
        "parentHoldoutDeclaredSha256": altered["parentEvidence"]["holdoutManifestDeclaredSha256"],
        "successorEnvelopeSelectionIdentitySha256": altered["successorEnvelopeSelectionIdentitySha256"],
    })
    altered["contractSha256"] = selection._document_digest(altered, "contractSha256")
    path = tmp_path / "altered" / "selection.json"
    _write_json(path, altered)
    with pytest.raises(selection.SuccessorV2SelectionError, match="canonical parent V1 registry"):
        selection.create_successor_v2_selection_claim(
            tmp_path / "parent" / "holdout.json", parent_contract, tmp_path / "plan.json", tmp_path / "envelope.json", path
        )
    assert not list(alternate_root.glob("*.claim.json"))


def test_self_digested_contract_cannot_rekey_canonical_registry_away_from_genuine_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The V1 registry root alone is insufficient: authenticate the envelope too."""

    canonical_root = tmp_path / "canonical-parent" / "partition_access"
    contract = _contract(canonical_root)
    parent_contract = _stub_parent_registry(monkeypatch, contract, canonical_root)
    altered = json.loads(json.dumps(contract))
    altered["successorEnvelopeSelectionIdentitySha256"] = _digest("forged-successor-selection")
    altered["consumptionRegistry"]["selectionSlotKey"] = selection.canonical_json_sha256({
        "schemaVersion": selection.CONSUMPTION_REGISTRY_SCHEMA,
        "parentPartitionAccessRoot": str(canonical_root.resolve()),
        "parentHoldoutFileSha256": altered["parentEvidence"]["holdoutManifestFileSha256"],
        "parentHoldoutDeclaredSha256": altered["parentEvidence"]["holdoutManifestDeclaredSha256"],
        "successorEnvelopeSelectionIdentitySha256": altered["successorEnvelopeSelectionIdentitySha256"],
    })
    altered["contractSha256"] = selection._document_digest(altered, "contractSha256")
    path = tmp_path / "altered" / "selection.json"
    _write_json(path, altered)
    genuine_envelope = {
        "parentEvidence": contract["parentEvidence"],
        "planFileSha256": contract["successorPlanFileSha256"],
        "planDeclaredSha256": contract["successorPlanDeclaredSha256"],
        "successorEnvelopeSha256": contract["successorEnvelopeDeclaredSha256"],
        "successorPartitionIdentities": {"NORMAL_SELECTION": contract["successorEnvelopeSelectionIdentitySha256"]},
        "records": contract["successorSelectionInputs"],
    }
    monkeypatch.setattr(
        selection, "_load_parent_chain", lambda *_args, **_kwargs: (genuine_envelope, contract["successorEnvelopeFileSha256"])
    )
    alternate_claim = selection.successor_v2_selection_path(altered, artifact="claim")
    with pytest.raises(selection.SuccessorV2SelectionError, match="parent/plan/envelope chain"):
        selection.create_successor_v2_selection_claim(
            tmp_path / "parent" / "holdout.json", parent_contract, tmp_path / "plan.json", tmp_path / "envelope.json", path
        )
    assert not alternate_claim.exists()


@pytest.mark.parametrize("raw_blob, message", [
    (b'{"schemaVersion":"first","schemaVersion":"second"}', "duplicate JSON key"),
    (b'{"score":NaN}', "non-finite JSON value"),
])
def test_recorded_git_ledger_blob_rejects_ambiguous_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_blob: bytes, message: str
) -> None:
    anchor = {
        "mode": selection.PUSHED_GIT_AUDIT_ONLY,
        "canonicalRemoteUrl": selection.CANONICAL_GIT_AUDIT_REMOTE_URL,
        "requiredRemoteRef": selection.REQUIRED_GIT_AUDIT_REMOTE_REF,
        "gitObjectFormat": selection.CANONICAL_GIT_AUDIT_OBJECT_FORMAT,
        "gitCommitOid": "a" * 40,
        "repositoryPath": selection.DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH,
        "gitBlobOid": "b" * 40,
        "ledgerBlobSha256": "sha256:" + hashlib.sha256(raw_blob).hexdigest(),
        "ledgerDeclaredSha256": _digest("declared"),
        "ledgerProjectionSha256": _digest("projection"),
    }
    monkeypatch.setattr(
        selection, "_resolve_pushed_git_ledger_blob", lambda **_kwargs: (anchor["gitBlobOid"], raw_blob)
    )
    with pytest.raises(selection.SuccessorV2SelectionError, match=message):
        selection._load_recorded_development_evidence_ledger(anchor, repository_root=tmp_path)


def test_pushed_git_ledger_uses_raw_blob_despite_worktree_or_index_ledger_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path / "partition_access")
    repository, ledger_path, expected_ledger, anchor = _pushed_ledger_repo(tmp_path, monkeypatch, contract)
    loaded, loaded_anchor = selection._load_recorded_development_evidence_ledger(anchor, repository_root=repository)
    assert loaded == expected_ledger
    assert loaded_anchor["ledgerBlobSha256"] == anchor["ledgerBlobSha256"]

    # Contract freeze also reads HEAD/blob bytes, not this mutable checkout
    # copy or index.  Line-ending filters and unrelated worktree state cannot
    # alter the audit authority.
    ledger_path.write_text('{"tampered": true}\n', encoding="utf-8")
    loaded_after_worktree_change, _ = selection._load_recorded_development_evidence_ledger(anchor, repository_root=repository)
    assert loaded_after_worktree_change == expected_ledger
    loaded_head, _ = selection._load_head_development_evidence_ledger(ledger_path, repository_root=repository)
    assert loaded_head == expected_ledger
    _git(["add", selection.DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH], cwd=repository)
    loaded_staged, _ = selection._load_head_development_evidence_ledger(ledger_path, repository_root=repository)
    assert loaded_staged == expected_ledger


def test_pushed_git_ledger_ignores_hostile_git_config_url_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller's GIT_CONFIG_* URL rewrite must not replace the fixed audit remote."""

    contract = _contract(tmp_path / "partition_access")
    repository, _ledger_path, expected_ledger, anchor = _pushed_ledger_repo(tmp_path, monkeypatch, contract)
    legitimate_remote = selection.CANONICAL_GIT_AUDIT_REMOTE_URL

    attacker_worktree = tmp_path / "attacker-worktree"
    attacker_worktree.mkdir()
    _git(["init", "-b", "master"], cwd=attacker_worktree)
    _git(["config", "user.email", "attacker@example.invalid"], cwd=attacker_worktree)
    _git(["config", "user.name", "Attacker"], cwd=attacker_worktree)
    _write_json(
        attacker_worktree / selection.DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH,
        {"attacker": "substituted ledger"},
    )
    _git(["add", selection.DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH], cwd=attacker_worktree)
    _git(["commit", "-m", "attacker ledger"], cwd=attacker_worktree)
    attacker_tip = subprocess.run(
        ["git", "-C", str(attacker_worktree), "rev-parse", "--verify", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True,
    ).stdout.strip()
    attacker_remote = tmp_path / "attacker.git"
    _git(["init", "--bare", str(attacker_remote)], cwd=tmp_path)
    _git(["remote", "add", "origin", attacker_remote.as_uri()], cwd=attacker_worktree)
    _git(["push", "origin", "HEAD:refs/heads/master"], cwd=attacker_worktree)

    # First prove the inherited environment is a real URL-rewrite attack,
    # rather than merely setting unused variables in this test.
    hostile_environment = os.environ.copy()
    hostile_environment.update({
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.{attacker_remote.as_uri()}.insteadOf",
        "GIT_CONFIG_VALUE_0": legitimate_remote,
    })
    unsafe_bare = tmp_path / "unsafe-inherited-config.git"
    completed = subprocess.run(
        ["git", "init", "--bare", str(unsafe_bare)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, env=hostile_environment,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    completed = subprocess.run(
        [
            "git", "-C", str(unsafe_bare), "fetch", "--no-tags", "--quiet", legitimate_remote,
            "refs/heads/master:refs/unsafe-audit",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, env=hostile_environment,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    redirected_tip = subprocess.run(
        ["git", "-C", str(unsafe_bare), "rev-parse", "--verify", "refs/unsafe-audit"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, env=hostile_environment,
    ).stdout.strip()
    assert redirected_tip == attacker_tip

    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{attacker_remote.as_uri()}.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", legitimate_remote)
    loaded, _ = selection._load_recorded_development_evidence_ledger(anchor, repository_root=repository)
    assert loaded == expected_ledger


def test_git_ledger_anchor_rejects_commit_outside_required_ref_and_non_100644_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path / "partition_access")
    repository, _ledger_path, _ledger, anchor = _pushed_ledger_repo(tmp_path, monkeypatch, contract)
    _git(["checkout", "-b", "unrequired-ledger-branch"], cwd=repository)
    _git(["commit", "--allow-empty", "-m", "commit outside required ref"], cwd=repository)
    _git(["push", "origin", "HEAD:refs/heads/unrequired-ledger-branch"], cwd=repository)
    unreachable = dict(anchor)
    unreachable["gitCommitOid"] = selection._git_text(
        ["rev-parse", "--verify", "HEAD"], repository_root=repository, description="test HEAD"
    )
    with pytest.raises(selection.SuccessorV2SelectionError, match="not reachable"):
        selection._load_recorded_development_evidence_ledger(unreachable, repository_root=repository)

    mode_root = tmp_path / "nonregular-mode"
    mode_root.mkdir()
    mode_repository, _mode_ledger_path, _mode_ledger, mode_anchor = _pushed_ledger_repo(mode_root, monkeypatch, contract)
    _git(["update-index", "--chmod=+x", selection.DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH], cwd=mode_repository)
    _git(["commit", "-m", "make ledger executable"], cwd=mode_repository)
    _git(["push", "origin", "HEAD:refs/heads/master"], cwd=mode_repository)
    executable_anchor = dict(mode_anchor)
    executable_anchor["gitCommitOid"] = selection._git_text(
        ["rev-parse", "--verify", "HEAD"], repository_root=mode_repository, description="executable ledger HEAD"
    )
    with pytest.raises(selection.SuccessorV2SelectionError, match="regular blob"):
        selection._load_recorded_development_evidence_ledger(executable_anchor, repository_root=mode_repository)


def test_pushed_git_ledger_resolver_rejects_present_nonancestor_before_blob_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the fail-closed merge-base return-1 branch explicitly."""

    commit_oid = "a" * 40
    fetched_tip = "b" * 40
    calls: list[list[str]] = []

    def fake_run(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    def fake_git_process(
        arguments: list[str], *, repository_root: Path, environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        del repository_root, environment
        calls.append(arguments)
        if arguments == ["rev-parse", "--show-object-format"]:
            return subprocess.CompletedProcess(arguments, 0, b"sha1\n", b"")
        if arguments[0] == "fetch":
            return subprocess.CompletedProcess(arguments, 0, b"", b"")
        if arguments[:2] == ["rev-parse", "--verify"]:
            return subprocess.CompletedProcess(arguments, 0, f"{fetched_tip}\n".encode("ascii"), b"")
        if arguments[:2] == ["cat-file", "-e"]:
            return subprocess.CompletedProcess(arguments, 0, b"", b"")
        if arguments[:2] == ["merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(arguments, 1, b"", b"")
        pytest.fail(f"unexpected Git audit command: {arguments!r}")

    monkeypatch.setattr(selection.subprocess, "run", fake_run)
    monkeypatch.setattr(selection, "_git_process", fake_git_process)
    with pytest.raises(selection.SuccessorV2SelectionError, match="not reachable"):
        selection._resolve_pushed_git_ledger_blob(
            commit_oid=commit_oid,
            repository_path=selection.DEVELOPMENT_EVIDENCE_LEDGER_REPOSITORY_PATH,
            canonical_remote_url=selection.CANONICAL_GIT_AUDIT_REMOTE_URL,
            required_remote_ref=selection.REQUIRED_GIT_AUDIT_REMOTE_REF,
            object_format="sha1",
        )
    assert any(arguments[:2] == ["merge-base", "--is-ancestor"] for arguments in calls)
    assert not any(arguments[:1] == ["ls-tree"] for arguments in calls)


def test_report_digest_must_match_ledger_before_semantic_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path / "partition_access")
    projection = _contract_projection(contract)
    monkeypatch.setattr(selection, "_load_parent_chain", lambda *_args, **_kwargs: ({}, _digest("envelope-file")))
    monkeypatch.setattr(selection, "_load_r3_manifest_json", lambda *_args, **_kwargs: ({}, _digest("r3-file"), _digest("recipe-file")))
    first = projection["candidateBindings"][0]
    monkeypatch.setattr(
        selection,
        "_read_json",
        lambda *_args, **_kwargs: (
            {"candidateConfiguration": {"id": first["candidateId"]}, "developmentReportSha256": first["developmentReportDeclaredSha256"]},
            _digest("substituted-report-file"),
        ),
    )
    monkeypatch.setattr(selection, "_validate_report", lambda *_args, **_kwargs: pytest.fail("semantic report validation was reached"))
    with pytest.raises(selection.SuccessorV2SelectionError, match="before semantic validation"):
        selection._load_closed_development_evidence(
            tmp_path / "holdout.json", tmp_path / "parent-contract.json", tmp_path / "plan.json", tmp_path / "envelope.json",
            [tmp_path / f"report-{index}.json" for index in range(4)], tmp_path / "manifest.json", tmp_path / "recipe.json",
            expected_projection=projection, repository_root=selection.REPOSITORY_ROOT,
        )


def test_self_digested_contract_cannot_bypass_immutable_ledger_at_any_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claim, observer, lock, and lock-loader all reject before any slot/query access."""

    canonical_root = tmp_path / "partition_access"
    contract = _contract(canonical_root)
    repository, _ledger_path, _ledger, anchor = _pushed_ledger_repo(tmp_path, monkeypatch, contract)
    contract["developmentEvidenceLedger"] = anchor
    contract["contractSha256"] = selection._document_digest(contract, "contractSha256")
    forged = json.loads(json.dumps(contract))
    forged["candidateReports"][0]["thresholds"]["capsule"] = 0.11
    forged["candidateReports"][0]["thresholdsIdentitySha256"] = selection.canonical_json_sha256(
        forged["candidateReports"][0]["thresholds"]
    )
    forged["candidateUniverseIdentitySha256"] = selection.canonical_json_sha256(forged["candidateReports"])
    forged["contractSha256"] = selection._document_digest(forged, "contractSha256")
    external = tmp_path / "external"
    forged_path = external / "forged-selection-contract.json"
    _write_json(forged_path, forged)
    monkeypatch.setattr(selection, "_validate_contract_development_evidence_ledger", _REAL_CONTRACT_LEDGER_VALIDATOR)
    monkeypatch.setattr(selection, "_load_parent_chain", lambda *_args, **_kwargs: pytest.fail("parent chain reached after forged ledger binding"))
    source_calls: list[object] = []

    def forbidden_source_loader(*_args: object, **_kwargs: object) -> object:
        source_calls.append(object())
        pytest.fail("selection source loader reached after forged ledger binding")

    monkeypatch.setattr(selection.successor, "load_successor_safe_normal_inputs", forbidden_source_loader)
    parent_holdout = external / "normal_holdout.json"
    parent_contract = external / "fresh_normal_selection_contract.json"
    plan = external / "fresh_normal_successor_plan.json"
    envelope = external / "fresh_normal_successor_envelope.json"
    manifest = external / "augmentation_manifest.json"
    recipe = external / "recipe.json"
    error = "immutable development evidence ledger projection"
    with pytest.raises(selection.SuccessorV2SelectionError, match=error):
        selection.create_successor_v2_selection_claim(
            parent_holdout, parent_contract, plan, envelope, forged_path, repository_root=repository
        )
    with pytest.raises(selection.SuccessorV2SelectionError, match=error):
        selection.run_successor_v2_selection_observation(
            parent_holdout, parent_contract, plan, envelope, forged_path, manifest, recipe,
            source_root=external, model_repo=external / "model", model_weights=external / "weights",
            repository_root=repository,
        )
    with pytest.raises(selection.SuccessorV2SelectionError, match=error):
        selection.create_successor_v2_selection_lock(
            parent_holdout, parent_contract, plan, envelope, forged_path, repository_root=repository
        )
    with pytest.raises(selection.SuccessorV2SelectionError, match=error):
        selection.load_validated_successor_v2_selection_lock(
            parent_holdout, parent_contract, plan, envelope, forged_path, repository_root=repository
        )
    assert not source_calls
    assert not list(canonical_root.glob("successor-v2-selection--*.json"))


def test_report_validator_recomputes_scores_thresholds_and_normal_only_evidence(tmp_path: Path) -> None:
    report, envelope, r3_manifest, report_file, r3_file, recipe_file = _semantic_report_fixture(tmp_path)
    selected = selection._validate_report(
        report, report_file_sha256=report_file, envelope=envelope, envelope_file_sha256=_digest("envelope-file"),
        r3_manifest=r3_manifest, r3_manifest_file_sha256=r3_file, r3_recipe_sha256=recipe_file,
    )
    assert selected["thresholds"] == {category: 0.04 for category in evaluator.SUCCESSOR_V2_CATEGORIES}

    threshold_tamper = json.loads(json.dumps(report))
    for category in evaluator.SUCCESSOR_V2_CATEGORIES:
        threshold_tamper["thresholds"][category] = 0.8
        threshold_tamper["categories"][category]["thresholdFromRawTuning"] = 0.8
        threshold_tamper["categories"][category]["tuningScoreMax"] = 0.8
    threshold_tamper["developmentReportSha256"] = selection._document_digest(threshold_tamper, "developmentReportSha256")
    with pytest.raises(selection.SuccessorV2SelectionError, match="raw-tuning maximum"):
        selection._validate_report(
            threshold_tamper, report_file_sha256=report_file, envelope=envelope, envelope_file_sha256=_digest("envelope-file"),
            r3_manifest=r3_manifest, r3_manifest_file_sha256=r3_file, r3_recipe_sha256=recipe_file,
        )

    score_tamper = json.loads(json.dumps(report))
    score_tamper["calibrationScores"][0]["score"] = 0.9
    score_tamper["calibrationScores"][0]["maxPatchDistance"] = 0.91
    score_tamper["developmentReportSha256"] = selection._document_digest(score_tamper, "developmentReportSha256")
    with pytest.raises(selection.SuccessorV2SelectionError, match="raw-tuning maximum"):
        selection._validate_report(
            score_tamper, report_file_sha256=report_file, envelope=envelope, envelope_file_sha256=_digest("envelope-file"),
            r3_manifest=r3_manifest, r3_manifest_file_sha256=r3_file, r3_recipe_sha256=recipe_file,
        )

    evidence_tamper = json.loads(json.dumps(report))
    evidence_tamper["normalOnlyEvidence"]["anomalyFeatureInputCount"] = 1
    evidence_tamper["developmentReportSha256"] = selection._document_digest(evidence_tamper, "developmentReportSha256")
    with pytest.raises(selection.SuccessorV2SelectionError, match="normal-only evidence"):
        selection._validate_report(
            evidence_tamper, report_file_sha256=report_file, envelope=envelope, envelope_file_sha256=_digest("envelope-file"),
            r3_manifest=r3_manifest, r3_manifest_file_sha256=r3_file, r3_recipe_sha256=recipe_file,
        )


def test_r3_manifest_json_membership_rejects_duplicate_or_wrong_parent(tmp_path: Path) -> None:
    _report, envelope, manifest, _report_file, _r3_file, _recipe_file = _semantic_report_fixture(tmp_path)
    duplicate = json.loads(json.dumps(manifest))
    duplicate["records"][1]["sourceSha256"] = duplicate["records"][0]["sourceSha256"]
    with pytest.raises(selection.SuccessorV2SelectionError, match="duplicated"):
        selection._expected_feature_inputs(envelope, r3_manifest=duplicate)
    wrong_parent = json.loads(json.dumps(manifest))
    wrong_parent["records"][0]["parentSourceSha256"] = _digest("wrong-parent")
    with pytest.raises(selection.SuccessorV2SelectionError, match="does not match its FIT parent"):
        selection._expected_feature_inputs(envelope, r3_manifest=wrong_parent)


def test_receipt_precedes_selection_and_no_nonselection_partition_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "parent" / "normal_holdout.json"
    parent.parent.mkdir(parents=True)
    canonical_root = tmp_path / "canonical-parent" / "partition_access"
    contract = _contract(canonical_root)
    parent_contract = _stub_parent_registry(monkeypatch, contract, canonical_root)
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    recipe_digest = selection.sha256_file(recipe)
    contract["augmentation"]["recipeFileSha256"] = recipe_digest
    for candidate in contract["candidateReports"]:
        if candidate["prototypeInputPolicy"] != evaluator.RAW_FIT_ONLY:
            candidate["augmentationRecipeFileSha256"] = recipe_digest
    contract["candidateUniverseIdentitySha256"] = selection.canonical_json_sha256(contract["candidateReports"])
    contract["contractSha256"] = selection._document_digest(contract, "contractSha256")
    contract_path = tmp_path / "contract" / "selection.json"

    fit = [
        _record(category, "FIT", index, image_path=tmp_path / "source" / f"{category}-{index}.jpg")
        for category in evaluator.SUCCESSOR_V2_CATEGORIES for index in range(12)
    ]
    query = [
        _record(category, "NORMAL_SELECTION", index, image_path=tmp_path / "source" / f"{category}-q-{index}.jpg")
        for category in evaluator.SUCCESSOR_V2_CATEGORIES for index in range(8)
    ]
    augmentation_root = tmp_path / "augmentation"
    augmentation_root.mkdir()
    r3_records = []
    for parent_record in fit:
        for variant_id, component in ((1, "registration"), (2, "illumination"), (3, "sensor_transport")):
            relative = Path("images") / f"{len(r3_records):03d}.jpg"
            path = augmentation_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-read-by-mocked-extractor")
            r3_records.append({
                "caseId": f"{parent_record['caseId']}/r3/{variant_id}",
                "parentCaseId": parent_record["caseId"],
                "parentSourceSha256": parent_record["sourceSha256"],
                "sourceGroupId": parent_record["sourceGroupId"],
                "category": parent_record["category"],
                "parentPartition": "FIT",
                "kind": "NOMINAL",
                "defect": "good",
                "variantId": variant_id,
                "component": component,
                "relativePath": relative.as_posix(),
                "sourceSha256": _digest(f"r3/{len(r3_records)}"),
            })
    r3_document = {"augmentationManifestSha256": contract["augmentation"]["manifestDeclaredSha256"]}
    envelope = {"records": fit + query, "successorEnvelopeSha256": contract["successorEnvelopeDeclaredSha256"]}
    raw_identity_records = sorted(
        [selection._identity_from_envelope_record(record) for record in fit], key=lambda item: item["caseId"]
    )
    r3_identity_records = sorted(
        raw_identity_records + [
            selection._identity_from_augmentation_record(
                record, manifest_declared=contract["augmentation"]["manifestDeclaredSha256"]
            )
            for record in r3_records
        ],
        key=lambda item: item["caseId"],
    )
    contract["prototypeInputIdentities"], contract["prototypeInputCounts"] = selection._prototype_input_binding_from_identities(
        raw_identity_records, r3_identity_records, name="fixture prototype commitment"
    )
    contract["contractSha256"] = selection._document_digest(contract, "contractSha256")
    _write_json(contract_path, contract)
    monkeypatch.setattr(selection, "_validate_contract_chain", lambda *_args, **_kwargs: (envelope, contract["successorEnvelopeFileSha256"]))
    selection.create_successor_v2_selection_claim(
        parent, parent_contract, tmp_path / "plan.json", tmp_path / "envelope.json", contract_path
    )
    calls: list[set[str]] = []
    receipt_path = selection.successor_v2_selection_path(parent, contract, artifact="receipt")

    def safe_loader(*_args: object, **kwargs: object) -> tuple[dict, str, list[dict]]:
        partitions = set(kwargs["partitions"])
        calls.append(partitions)
        assert partitions.issubset({"FIT", "NORMAL_SELECTION"})
        if partitions == {"NORMAL_SELECTION"}:
            assert receipt_path.is_file()
            return envelope, contract["successorEnvelopeFileSha256"], query
        return envelope, contract["successorEnvelopeFileSha256"], fit

    def fake_augment(*_args: object, **_kwargs: object) -> tuple[dict, str, list[dict]]:
        return r3_document, contract["augmentation"]["manifestFileSha256"], r3_records

    def fake_features(records: list[dict], **_kwargs: object) -> dict[str, object]:
        base = np.tile(np.linspace(1.0, 2.0, 8, dtype=np.float32), (16, 1))
        return {str(record["caseId"]): base for record in records}

    monkeypatch.setattr(selection.successor, "load_successor_safe_normal_inputs", safe_loader)
    monkeypatch.setattr(selection.augmentation, "load_validated_successor_fit_augmentations_with_file_sha256", fake_augment)
    monkeypatch.setattr(selection.evaluator, "_extract_patch_features", fake_features)
    monkeypatch.setattr(selection, "_assert_identity", lambda *_args, **_kwargs: contract["featureExtractor"])
    observed = selection.run_successor_v2_selection_observation(
        parent, tmp_path / "parent-contract.json", tmp_path / "plan.json", tmp_path / "envelope.json", contract_path,
        augmentation_root / "augmentation_manifest.json", recipe, source_root=tmp_path / "source",
        model_repo=tmp_path / "model", model_weights=tmp_path / "weights", embedder_factory=lambda **_kwargs: object(),
    )
    assert calls == [{"FIT"}, {"NORMAL_SELECTION"}]
    assert len(observed["candidateObservations"]) == 4
    assert observed["normalOnlyEvidence"]["parentConfirmationInputCount"] == 0
    assert observed["normalOnlyEvidence"]["remainingReserveInputCount"] == 0

    loaded_contract, contract_file = selection.load_validated_successor_v2_selection_contract(contract_path)
    claim, claim_file = selection._load_claim(parent_contract, loaded_contract, contract_file_sha256=contract_file, repository_root=selection.REPOSITORY_ROOT)
    receipt_path = selection.successor_v2_selection_path(loaded_contract, artifact="receipt")
    receipt, receipt_file = selection._read_json(receipt_path, description="fixture receipt", repository_root=selection.REPOSITORY_ROOT)
    tampered_receipt = json.loads(json.dumps(receipt))
    tampered_receipt["prototypeInputCounts"][evaluator.RAW_FIT_ONLY] = 35
    tampered_receipt["selectionReceiptSha256"] = selection._document_digest(tampered_receipt, "selectionReceiptSha256")
    with pytest.raises(selection.SuccessorV2SelectionError, match="raw=36"):
        selection._validate_receipt(
            tampered_receipt, contract=loaded_contract, contract_file_sha256=contract_file, claim=claim, claim_file_sha256=claim_file
        )
    tampered_identity = json.loads(json.dumps(receipt))
    tampered_identity["prototypeInputIdentities"][evaluator.RAW_FIT_ONLY] = _digest("forged-raw-prototype-identity")
    tampered_identity["selectionReceiptSha256"] = selection._document_digest(tampered_identity, "selectionReceiptSha256")
    with pytest.raises(selection.SuccessorV2SelectionError, match="prototype maps"):
        selection._validate_receipt(
            tampered_identity, contract=loaded_contract, contract_file_sha256=contract_file, claim=claim, claim_file_sha256=claim_file
        )
    tampered_observation = json.loads(json.dumps(observed))
    tampered_observation["normalOnlyEvidence"]["queryInputCount"] = 23
    tampered_observation["selectionObservationSha256"] = selection._document_digest(
        tampered_observation, "selectionObservationSha256"
    )
    with pytest.raises(selection.SuccessorV2SelectionError, match="normal-only evidence"):
        selection._validate_observation(
            tampered_observation, contract=loaded_contract, contract_file_sha256=contract_file, claim=claim,
            claim_file_sha256=claim_file, receipt=receipt, receipt_file_sha256=receipt_file,
        )


def test_no_eligible_decision_has_no_automatic_confirmation() -> None:
    metrics = {
        category: {
            "queryCount": 8, "aboveThresholdCount": 2, "aboveThresholdRate": 0.25,
            "p95Score": 0.2, "maximumScore": 0.2, "p95ScoreMinusThreshold": 0.1,
            "maximumScoreMinusThreshold": 0.1,
        }
        for category in evaluator.SUCCESSOR_V2_CATEGORIES
    }
    decision = selection._decision([{
        "candidateId": "reserve-v2-raw-p2048-k5", "categoryMetrics": metrics, "gatePassed": False,
        "gateRejectionReasons": selection._rejection_reasons(metrics), "objectiveValues": None,
    }])
    assert decision == {
        "state": "NO_ELIGIBLE_CONFIGURATION",
        "selectedCandidateId": None,
        "resultScope": "OFFLINE_RESEARCH_CONFIGURATION_LOCK_ONLY",
        "automaticProductionPromotion": False,
        "automaticConfirmation": False,
    }
