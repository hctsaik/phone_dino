from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from phone_dino import mvtec_normal_holdout as holdout
from phone_dino import mvtec_normal_source_acquisition as acquisition


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def _write_image(path: Path, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (9, 7), (index * 19 % 255, index * 41 % 255, index * 67 % 255)).save(path, format="PNG")


def _patch_source_constants(monkeypatch: pytest.MonkeyPatch, metadata_path: Path) -> None:
    raw = metadata_path.read_bytes()
    monkeypatch.setattr(holdout, "PINNED_MVTEC_DATASET_ID", "MVTec AD fixture")
    monkeypatch.setattr(holdout, "PINNED_MVTEC_OFFICIAL_SOURCE_URI", "https://official.example/mvtec")
    monkeypatch.setattr(holdout, "PINNED_MVTEC_MIRROR_SOURCE_URI", "https://mirror.example/mvtec")
    monkeypatch.setattr(holdout, "PINNED_MVTEC_MIRROR_REVISION", "a" * 40)
    monkeypatch.setattr(holdout, "PINNED_MVTEC_SAMPLES_SHA256", f"sha256:{hashlib.sha256(raw).hexdigest()}")
    monkeypatch.setattr(
        holdout,
        "PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1",
        hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest(),
    )
    monkeypatch.setattr(holdout, "PINNED_MVTEC_LICENSE_NOTICE", "fixture-license")


def _origin() -> dict:
    return {
        "datasetId": holdout.PINNED_MVTEC_DATASET_ID,
        "officialSourceUri": holdout.PINNED_MVTEC_OFFICIAL_SOURCE_URI,
        "mirrorSourceUri": holdout.PINNED_MVTEC_MIRROR_SOURCE_URI,
        "mirrorRevision": holdout.PINNED_MVTEC_MIRROR_REVISION,
        "sourceMetadataFileSha256": holdout.PINNED_MVTEC_SAMPLES_SHA256,
        "sourceMetadataGitBlobSha1": holdout.PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1,
        "sourceCriterion": "OFFICIAL_TRAIN_GOOD_ONLY",
        "licenseNotice": holdout.PINNED_MVTEC_LICENSE_NOTICE,
        "priorSubsetSourceIdentitySha256": holdout.canonical_json_sha256([]),
    }


def _self_digest(document: dict, field: str) -> str:
    unsigned = dict(document)
    unsigned.pop(field, None)
    return holdout.canonical_json_sha256(unsigned)


def _bind_candidate_to_ledger(candidate_path: Path, ledger: dict) -> None:
    document = json.loads(candidate_path.read_text(encoding="utf-8"))
    document["origin"]["priorSubsetSourceIdentitySha256"] = holdout.historical_normal_usage_identity(
        ledger, set(ledger["normalSourceSha256"])
    )
    document["candidateManifestSha256"] = _self_digest(document, "candidateManifestSha256")
    _write_json(candidate_path, document)


def _normal_feature(case_id: str, source_sha256: str, role: str) -> dict:
    return {
        "caseId": case_id,
        "category": "capsule",
        "role": role,
        "kind": "NOMINAL",
        "sourceSha256": source_sha256,
        "isAugmentation": False,
        "variantId": None,
        "parentCaseId": None,
        "parentSourceSha256": None,
        "augmentationRecipeSha256": None,
    }


def _normal_score(feature: dict) -> dict:
    return {
        "caseId": feature["caseId"],
        "category": feature["category"],
        "defect": "good",
        "isAugmentation": feature["isAugmentation"],
        "kind": "NOMINAL",
        "role": "THRESHOLD_TUNING",
        "score": 0.1,
        "sourceSha256": feature["sourceSha256"],
        "variantId": feature["variantId"],
    }


def _normal_only_report(path: Path, source_sha256: str) -> Path:
    feature_inputs = [
        _normal_feature("legacy/capsule/fit/000", source_sha256, "FIT"),
        _normal_feature("legacy/capsule/tuning/000", source_sha256, "THRESHOLD_TUNING"),
    ]
    identity = holdout.canonical_json_sha256(feature_inputs)
    evidence = {
        "featureInputCount": len(feature_inputs),
        "featureInputRoles": ["FIT", "THRESHOLD_TUNING"],
        "featureInputKinds": ["NOMINAL"],
        "blindFeatureInputCount": 0,
        "anomalyFeatureInputCount": 0,
        "normalInputRecordCount": len(feature_inputs),
        "featureInputs": feature_inputs,
        "featureInputIdentitySha256": identity,
        "normalInputIdentitySha256": identity,
        "reportedScoreCount": 1,
        "reportedScoreRoles": ["THRESHOLD_TUNING"],
        "reportedScoreKinds": ["NOMINAL"],
        "calibrationScoreCount": 1,
        "calibrationScoreRoles": ["THRESHOLD_TUNING"],
        "calibrationScoreKinds": ["NOMINAL"],
        "calibrationInputs": [feature_inputs[1]],
        "calibrationInputIdentitySha256": holdout.canonical_json_sha256([feature_inputs[1]]),
        "originalTuningInputCount": 1,
        "originalTuningInputs": [feature_inputs[1]],
        "originalTuningInputIdentitySha256": holdout.canonical_json_sha256([feature_inputs[1]]),
    }
    document = {
        "schemaVersion": holdout.ITERATION_REPORT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "selectionProtocol": holdout.ITERATION_REPORT_PURPOSE,
        "blindReporting": {
            "state": "NOT_RUN",
            "blindSourcePolicy": "ORIGINAL_ONLY",
            "reason": "NORMAL_ONLY_ITERATION",
        },
        "pixelLocalization": None,
        "normalOnlyEvidence": evidence,
        "algorithm": {},
        "augmentation": {},
        "calibrationScores": [_normal_score(feature_inputs[1])],
        "candidateConfiguration": {},
        "candidateConfigurationSha256": "sha256:" + "0" * 64,
        "categories": {},
        "disclaimer": "fixture",
        "execution": {},
        "featureExtractor": {},
        "featureExtractorIdentitySha256": "sha256:" + "1" * 64,
        "inputManifest": {},
        "inputManifestDeclaredSha256": "sha256:" + "2" * 64,
        "inputManifestFileSha256": "sha256:" + "3" * 64,
        "scores": [_normal_score(feature_inputs[1])],
    }
    assert set(document) == holdout.ITERATION_REPORT_FIELDS
    _write_json(path, document)
    return path


def _closed_normal_holdout_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    """Build a self-contained frozen manifest without public source metadata."""

    source_root = tmp_path / "normal_sources"
    partitions = (
        "FIT",
        "THRESHOLD_TUNING",
        "NORMAL_SELECTION",
        "NORMAL_CONFIRMATION",
        "RESERVE_UNTOUCHED",
    )
    records: list[dict] = []
    image_paths: dict[str, Path] = {}
    for index, partition in enumerate(partitions):
        relative_path = Path("images") / f"{index}.png"
        image_path = source_root / relative_path
        _write_image(image_path, index)
        source_sha256 = holdout.sha256_file(image_path)
        records.append({
            "caseId": f"mvtec-ad/capsule/train-good/{source_sha256[7:]}",
            "category": "capsule",
            "relativePath": relative_path.as_posix(),
            "sourceSha256": source_sha256,
            "sourceGroupId": f"CONTENT_SHA256:{source_sha256[7:]}",
            "acquisitionStratum": "OFFICIAL_MVTEC_TRAIN_GOOD",
            "sourceRemotePath": f"data/data_6/{index}.png",
            "expectedRemoteSha256": source_sha256,
            "expectedRemoteBytes": image_path.stat().st_size,
            "kind": "NOMINAL",
            "defect": "good",
            "partition": partition,
        })
        image_paths[partition] = image_path
    records.sort(key=lambda record: record["caseId"])
    document = {
        "schemaVersion": holdout.NORMAL_HOLDOUT_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": holdout.HOLDOUT_PURPOSE,
        "blindPolicy": holdout.HOLDOUT_BLIND_POLICY,
        "sourcePoolFileSha256": "sha256:" + "1" * 64,
        "sourcePoolDeclaredSha256": "sha256:" + "2" * 64,
        "historicalLedgerFileSha256": "sha256:" + "3" * 64,
        "historicalLedgerDeclaredSha256": "sha256:" + "4" * 64,
        "planFileSha256": "sha256:" + "5" * 64,
        "planDeclaredSha256": "sha256:" + "6" * 64,
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
    document["normalHoldoutManifestSha256"] = _self_digest(document, "normalHoldoutManifestSha256")
    manifest_path = tmp_path / "holdout" / "normal_holdout.json"
    _write_json(manifest_path, document)
    return manifest_path, source_root, image_paths


@pytest.fixture
def frozen_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path | list[dict]]:
    source_root = tmp_path / "fresh_sources"
    metadata_path = tmp_path / "source_metadata" / "samples.json"
    samples: list[dict] = []
    records: list[dict] = []
    for index in range(7):
        source_remote_path = f"data/data_6/{index:03d}.png"
        relative_path = Path("images") / "capsule" / f"{index:03d}.png"
        image_path = source_root / relative_path
        _write_image(image_path, index)
        source_sha256 = holdout.sha256_file(image_path)
        samples.append({
            "filepath": source_remote_path,
            "category": {"label": "capsule"},
            "defect": {"label": "good"},
            "split": "train",
        })
        records.append({
            "caseId": f"mvtec-ad/capsule/train-good/{source_sha256[7:]}",
            "category": "capsule",
            "relativePath": relative_path.as_posix(),
            "sourceGroupId": f"CONTENT_SHA256:{source_sha256[7:]}",
            "acquisitionStratum": "OFFICIAL_MVTEC_TRAIN_GOOD",
            "sourceRemotePath": source_remote_path,
            "expectedRemoteSha256": source_sha256,
            "expectedRemoteBytes": image_path.stat().st_size,
        })
    _write_json(metadata_path, {"samples": samples})
    _patch_source_constants(monkeypatch, metadata_path)
    records.sort(key=lambda record: record["caseId"])
    candidate = {
        "schemaVersion": holdout.NORMAL_SOURCE_CANDIDATES_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": holdout.SOURCE_CANDIDATES_PURPOSE,
        "origin": _origin(),
        "groupingStrength": "EXACT_CONTENT_ONLY",
        "records": records,
    }
    candidate["candidateManifestSha256"] = _self_digest(candidate, "candidateManifestSha256")
    candidate_path = tmp_path / "candidate" / "candidates.json"
    _write_json(candidate_path, candidate)
    return {
        "source_root": source_root,
        "metadata_path": metadata_path,
        "candidate_path": candidate_path,
        "records": records,
        "tmp_path": tmp_path,
    }


def test_freezes_external_source_pool_and_allocates_disjoint_holdout(frozen_inputs: dict[str, Path | list[dict]]) -> None:
    source_root = frozen_inputs["source_root"]
    metadata_path = frozen_inputs["metadata_path"]
    candidate_path = frozen_inputs["candidate_path"]
    records = frozen_inputs["records"]
    tmp_path = frozen_inputs["tmp_path"]
    assert isinstance(source_root, Path)
    assert isinstance(metadata_path, Path)
    assert isinstance(candidate_path, Path)
    assert isinstance(records, list)
    assert isinstance(tmp_path, Path)
    excluded_source = records[0]["expectedRemoteSha256"]
    report_path = _normal_only_report(tmp_path / "history" / "normal_only.json", excluded_source)
    ledger_path = tmp_path / "history" / "ledger.json"
    ledger = holdout.build_historical_normal_usage_ledger([report_path], ledger_path)
    assert excluded_source in ledger["normalSourceSha256"]
    _bind_candidate_to_ledger(candidate_path, ledger)
    pool_path = tmp_path / "frozen" / "pool.json"
    pool = holdout.freeze_normal_source_pool(
        candidate_path,
        pool_path,
        source_root=source_root,
        source_metadata_path=metadata_path,
    )
    assert pool["normalSourcePoolIdentitySha256"].startswith("sha256:")
    plan_path = tmp_path / "plan" / "plan.json"
    plan = holdout.create_normal_holdout_plan(
        pool_path,
        ledger_path,
        plan_path,
        source_root=source_root,
        source_metadata_path=metadata_path,
        category_group_quotas={
            "capsule": {
                "fitGroupCount": 1,
                "thresholdTuningGroupCount": 1,
                "normalSelectionGroupCount": 1,
                "normalConfirmationGroupCount": 1,
                "reserveUntouchedGroupCount": 2,
            }
        },
    )
    assert plan["partitionAlgorithm"] == holdout.PARTITION_ALGORITHM
    holdout_path = tmp_path / "holdout" / "holdout.json"
    document = holdout.build_normal_holdout_manifest(
        pool_path,
        ledger_path,
        plan_path,
        holdout_path,
        source_root=source_root,
        source_metadata_path=metadata_path,
    )
    validated, _ = holdout.load_validated_normal_holdout_manifest(
        holdout_path,
        pool_path,
        ledger_path,
        plan_path,
        source_root=source_root,
        source_metadata_path=metadata_path,
    )
    assert validated == document
    assert excluded_source not in {record["sourceSha256"] for record in document["records"]}
    assert {record["partition"] for record in document["records"]} == set(holdout.HOLDOUT_PARTITIONS)
    assert all(record["kind"] == "NOMINAL" and record["defect"] == "good" for record in document["records"])


def test_source_pool_rejects_metadata_non_normal_and_source_tamper(
    frozen_inputs: dict[str, Path | list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = frozen_inputs["source_root"]
    metadata_path = frozen_inputs["metadata_path"]
    candidate_path = frozen_inputs["candidate_path"]
    records = frozen_inputs["records"]
    tmp_path = frozen_inputs["tmp_path"]
    assert isinstance(source_root, Path)
    assert isinstance(metadata_path, Path)
    assert isinstance(candidate_path, Path)
    assert isinstance(records, list)
    assert isinstance(tmp_path, Path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    original_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["samples"][0]["split"] = "test"
    _write_json(metadata_path, metadata)
    _patch_source_constants(monkeypatch, metadata_path)
    candidate["origin"] = _origin()
    candidate["candidateManifestSha256"] = _self_digest(candidate, "candidateManifestSha256")
    _write_json(candidate_path, candidate)
    with pytest.raises(holdout.NormalHoldoutError, match="not a pinned train/good"):
        holdout.freeze_normal_source_pool(
            candidate_path,
            tmp_path / "bad" / "pool.json",
            source_root=source_root,
            source_metadata_path=metadata_path,
        )
    _write_json(metadata_path, original_metadata)
    # Restore the fixture's pinned metadata constants before proving byte tampering.
    _patch_source_constants(monkeypatch, metadata_path)
    candidate["origin"] = _origin()
    candidate["candidateManifestSha256"] = _self_digest(candidate, "candidateManifestSha256")
    _write_json(candidate_path, candidate)
    image_path = source_root / Path(records[0]["relativePath"])
    image_path.write_bytes(b"not-an-image")
    with pytest.raises(holdout.NormalHoldoutError, match="decodable image|pinned remote identity"):
        holdout.freeze_normal_source_pool(
            candidate_path,
            tmp_path / "bad2" / "pool.json",
            source_root=source_root,
            source_metadata_path=metadata_path,
        )


def test_historical_ledger_rejects_unlinked_augmented_parent(tmp_path: Path) -> None:
    source_sha256 = "sha256:" + "a" * 64
    report_path = _normal_only_report(tmp_path / "report.json", source_sha256)
    document = json.loads(report_path.read_text(encoding="utf-8"))
    augmented = {
        "caseId": "legacy/capsule/fit/000/augmentation/01",
        "category": "capsule",
        "role": "FIT",
        "kind": "NOMINAL",
        "sourceSha256": "sha256:" + "b" * 64,
        "isAugmentation": True,
        "variantId": 1,
        "parentCaseId": "legacy/capsule/fit/not-present",
        "parentSourceSha256": "sha256:" + "c" * 64,
        "augmentationRecipeSha256": "sha256:" + "d" * 64,
    }
    evidence = document["normalOnlyEvidence"]
    evidence["featureInputs"] = sorted(evidence["featureInputs"] + [augmented], key=lambda item: item["caseId"])
    evidence["featureInputCount"] = len(evidence["featureInputs"])
    evidence["normalInputRecordCount"] = len(evidence["featureInputs"])
    identity = holdout.canonical_json_sha256(evidence["featureInputs"])
    evidence["featureInputIdentitySha256"] = identity
    evidence["normalInputIdentitySha256"] = identity
    _write_json(report_path, document)
    with pytest.raises(holdout.NormalHoldoutError, match="augmented input parent"):
        holdout.build_historical_normal_usage_ledger([report_path], tmp_path / "ledger.json")


def test_evaluation_safe_loader_reads_only_requested_normal_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, source_root, image_paths = _closed_normal_holdout_fixture(tmp_path)
    # A broken selection file proves this reader neither opens nor decodes it
    # while preparing the FIT/tuning development envelope.
    image_paths["NORMAL_SELECTION"].write_bytes(b"not-a-normal-image")

    def unexpected_metadata_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("evaluation-safe loader must not read source metadata")

    monkeypatch.setattr(holdout, "_load_pinned_source_metadata", unexpected_metadata_read)
    document, _, records = holdout.load_evaluation_safe_normal_holdout_inputs(
        manifest_path,
        source_root=source_root,
        partitions={"FIT", "THRESHOLD_TUNING"},
    )
    assert document["blindPolicy"] == holdout.HOLDOUT_BLIND_POLICY
    assert {record["partition"] for record in records} == {"FIT", "THRESHOLD_TUNING"}
    assert all(record["imagePath"].is_file() for record in records)
    with pytest.raises(holdout.NormalHoldoutError, match="bytes do not match"):
        holdout.load_evaluation_safe_normal_holdout_inputs(
            manifest_path,
            source_root=source_root,
            partitions={"NORMAL_SELECTION"},
        )


def test_evaluation_safe_loader_rejects_non_normal_closed_record(tmp_path: Path) -> None:
    manifest_path, source_root, _ = _closed_normal_holdout_fixture(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["records"][0]["defect"] = "crack"
    document["normalHoldoutManifestSha256"] = _self_digest(document, "normalHoldoutManifestSha256")
    _write_json(manifest_path, document)
    with pytest.raises(holdout.NormalHoldoutError, match="nominal good"):
        holdout.load_evaluation_safe_normal_holdout_inputs(
            manifest_path,
            source_root=source_root,
            partitions={"FIT"},
        )


def test_rejects_duplicate_json_keys_repo_ancestor_and_manifest_tamper(
    frozen_inputs: dict[str, Path | list[dict]],
) -> None:
    source_root = frozen_inputs["source_root"]
    metadata_path = frozen_inputs["metadata_path"]
    candidate_path = frozen_inputs["candidate_path"]
    tmp_path = frozen_inputs["tmp_path"]
    assert isinstance(source_root, Path)
    assert isinstance(metadata_path, Path)
    assert isinstance(candidate_path, Path)
    assert isinstance(tmp_path, Path)
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"purpose":"bad","purpose":"good"}', encoding="utf-8")
    with pytest.raises(holdout.NormalHoldoutError, match="duplicate JSON key"):
        holdout._read_json(duplicate_path, description="duplicate fixture")
    with pytest.raises(holdout.NormalHoldoutError, match="outside the Git"):
        holdout._require_external_source_root(holdout.REPOSITORY_ROOT.parent)
    pool_path = tmp_path / "pool" / "pool.json"
    holdout.freeze_normal_source_pool(
        candidate_path,
        pool_path,
        source_root=source_root,
        source_metadata_path=metadata_path,
    )
    report_path = _normal_only_report(tmp_path / "report.json", "sha256:" + "f" * 64)
    ledger_path = tmp_path / "ledger" / "ledger.json"
    ledger = holdout.build_historical_normal_usage_ledger([report_path], ledger_path)
    with pytest.raises(holdout.NormalHoldoutError, match="not bound to the current historical"):
        holdout.create_normal_holdout_plan(
            pool_path,
            ledger_path,
            tmp_path / "plan" / "mismatched_plan.json",
            source_root=source_root,
            source_metadata_path=metadata_path,
            category_group_quotas={
                "capsule": {
                    "fitGroupCount": 1,
                    "thresholdTuningGroupCount": 1,
                    "normalSelectionGroupCount": 1,
                    "normalConfirmationGroupCount": 1,
                    "reserveUntouchedGroupCount": 3,
                }
            },
        )
    # Freeze a new pool after binding its candidate to the ledger that existed
    # at acquisition time; the earlier pool is intentionally not reused.
    bound_pool_path = tmp_path / "pool" / "bound_pool.json"
    _bind_candidate_to_ledger(candidate_path, ledger)
    holdout.freeze_normal_source_pool(
        candidate_path,
        bound_pool_path,
        source_root=source_root,
        source_metadata_path=metadata_path,
    )
    plan_path = tmp_path / "plan" / "plan.json"
    holdout.create_normal_holdout_plan(
        bound_pool_path,
        ledger_path,
        plan_path,
        source_root=source_root,
        source_metadata_path=metadata_path,
        category_group_quotas={
            "capsule": {
                "fitGroupCount": 1,
                "thresholdTuningGroupCount": 1,
                "normalSelectionGroupCount": 1,
                "normalConfirmationGroupCount": 1,
                "reserveUntouchedGroupCount": 3,
            }
        },
    )
    holdout_path = tmp_path / "holdout" / "holdout.json"
    holdout.build_normal_holdout_manifest(
        bound_pool_path,
        ledger_path,
        plan_path,
        holdout_path,
        source_root=source_root,
        source_metadata_path=metadata_path,
    )
    document = json.loads(holdout_path.read_text(encoding="utf-8"))
    next(record for record in document["records"] if record["partition"] != "FIT")["partition"] = "FIT"
    document["normalHoldoutManifestSha256"] = _self_digest(document, "normalHoldoutManifestSha256")
    _write_json(holdout_path, document)
    with pytest.raises(holdout.NormalHoldoutError, match="does not match"):
        holdout.load_validated_normal_holdout_manifest(
            holdout_path,
            bound_pool_path,
            ledger_path,
            plan_path,
            source_root=source_root,
            source_metadata_path=metadata_path,
        )


def test_acquisition_emits_only_fresh_pinned_train_good_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from io import BytesIO

    images: dict[str, bytes] = {}
    for remote_path, color in (("data/data_6/historical.png", (10, 20, 30)), ("data/data_6/fresh.png", (40, 50, 60))):
        buffer = BytesIO()
        Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
        images[remote_path] = buffer.getvalue()
    metadata_path = tmp_path / "metadata" / "samples.json"
    _write_json(metadata_path, {"samples": [
        {
            "filepath": remote_path,
            "category": {"label": "capsule"},
            "defect": {"label": "good"},
            "split": "train",
        }
        for remote_path in images
    ]})
    _patch_source_constants(monkeypatch, metadata_path)
    for name in (
        "PINNED_MVTEC_DATASET_ID",
        "PINNED_MVTEC_OFFICIAL_SOURCE_URI",
        "PINNED_MVTEC_MIRROR_SOURCE_URI",
        "PINNED_MVTEC_MIRROR_REVISION",
        "PINNED_MVTEC_SAMPLES_SHA256",
        "PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1",
        "PINNED_MVTEC_LICENSE_NOTICE",
    ):
        monkeypatch.setattr(acquisition, name, getattr(holdout, name))
    monkeypatch.setattr(acquisition, "TARGET_CATEGORIES", ("capsule",))
    monkeypatch.setattr(acquisition, "EXPECTED_TRAIN_GOOD_COUNTS", {"capsule": 2})
    monkeypatch.setattr(acquisition, "EXPECTED_FRESH_COUNTS", {"capsule": 1})
    historical_sha256 = f"sha256:{hashlib.sha256(images['data/data_6/historical.png']).hexdigest()}"
    ledger = {
        "schemaVersion": holdout.HISTORICAL_NORMAL_USAGE_LEDGER_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": holdout.HISTORICAL_LEDGER_PURPOSE,
        "exclusionScope": holdout.HISTORICAL_LEDGER_SCOPE,
        "evidence": [{
            "reportFileSha256": "sha256:" + "1" * 64,
            "featureInputIdentitySha256": "sha256:" + "2" * 64,
            "normalInputIdentitySha256": "sha256:" + "2" * 64,
        }],
        "normalSourceSha256": [historical_sha256],
    }
    ledger["historicalNormalUsageLedgerSha256"] = _self_digest(ledger, "historicalNormalUsageLedgerSha256")
    ledger_path = tmp_path / "history" / "ledger.json"
    _write_json(ledger_path, ledger)

    def fake_resolve(remote_path: str, *, timeout_seconds: float) -> tuple[str, int, str]:
        content = images[remote_path]
        return f"sha256:{hashlib.sha256(content).hexdigest()}", len(content), f"https://mock.hf.co/{remote_path}"

    def fake_stream(
        location: str,
        destination: Path,
        *,
        expected_sha256: str,
        expected_bytes: int,
        timeout_seconds: float,
    ) -> None:
        remote_path = location.removeprefix("https://mock.hf.co/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(images[remote_path])
        assert holdout.sha256_file(destination) == expected_sha256
        assert destination.stat().st_size == expected_bytes

    monkeypatch.setattr(acquisition, "_resolve_remote_identity", fake_resolve)
    monkeypatch.setattr(acquisition, "_stream_verified_file", fake_stream)
    source_root = tmp_path / "acquired_sources"
    candidate_output = tmp_path / "candidate" / "candidates.json"
    document = acquisition.acquire_fresh_normal_sources(
        metadata_path,
        ledger_path,
        source_root,
        candidate_output,
        workers=1,
    )
    assert len(document["records"]) == 1
    record = document["records"][0]
    assert record["sourceRemotePath"] == "data/data_6/fresh.png"
    assert record["expectedRemoteSha256"] != historical_sha256
    assert (source_root / record["relativePath"]).is_file()
