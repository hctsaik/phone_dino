from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from phone_dino import mvtec_synthetic_nuisance_control_v3 as audit
from phone_dino import mvtec_synthetic_stress_v2 as stress


def _digest(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _write_image(path: Path, colour: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), colour)
    draw = ImageDraw.Draw(image)
    draw.line(
        (2, 2, 42, 37),
        fill=((colour[0] + 31) % 255, (colour[1] + 47) % 255, (colour[2] + 73) % 255),
        width=2,
    )
    image.save(path, format="PNG")
    return audit.sha256_file(path)


def _parents(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for category_index, category in enumerate(audit.SYNTHETIC_NUISANCE_CONTROL_V3_CATEGORIES):
        for ordinal in range(12):
            path = root / category / f"parent-{ordinal:02d}.png"
            source = _write_image(
                path,
                (
                    (category_index * 70 + ordinal * 13 + 10) % 255,
                    (ordinal * 29 + 40) % 255,
                    (category_index * 41 + ordinal * 17 + 80) % 255,
                ),
            )
            records.append(
                {
                    "caseId": f"successor/{category}/fit/{ordinal:02d}",
                    "category": category,
                    "partition": "FIT",
                    "kind": "NOMINAL",
                    "defect": "good",
                    "sourceSha256": source,
                    "sourceGroupId": f"CONTENT_SHA256:{source[7:]}",
                    "imagePath": path,
                }
            )
    return records


def _envelope() -> dict[str, Any]:
    return {
        "parentEvidence": {
            "holdoutManifestFileSha256": _digest("holdout-file"),
            "holdoutManifestDeclaredSha256": _digest("holdout-declared"),
            "selectionContractFileSha256": _digest("contract-file"),
            "selectionContractDeclaredSha256": _digest("contract-declared"),
            "parentNormalConfirmationIdentitySha256": _digest("confirmation"),
        },
        "sealFileSha256": _digest("seal-file"),
        "sealDeclaredSha256": _digest("seal-declared"),
        "planFileSha256": _digest("plan-file"),
        "planDeclaredSha256": _digest("plan-declared"),
        "successorEnvelopeSha256": _digest("envelope-declared"),
        "successorPartitionIdentities": {"FIT": _digest("fit-identity")},
    }


def _package_manifest(label: str, schema: str) -> dict[str, Any]:
    return {
        "schemaVersion": schema,
        "augmentationManifestSha256": _digest(f"{label}-declared"),
        "recipeFileSha256": _digest(f"{label}-recipe"),
    }


def _package_metadata(label: str, schema: str) -> dict[str, Any]:
    document = _package_manifest(label, schema)
    return {
        "document": document,
        "fileSha256": _digest(f"{label}-file"),
        "recipeSha256": document["recipeFileSha256"],
    }


def _receipt_identity_fixture(label: str) -> dict[str, Any]:
    return {
        "schemaVersion": audit.SYNTHETIC_NUISANCE_CONTROL_V3_RECEIPT_SCHEMA,
        **{
            field: _digest(f"{label}-{field}")
            for field in audit.RECEIPT_IDENTITY_FIELDS
            if field != "schemaVersion"
        },
    }


def _controls(root: Path, parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split = stress.build_fixed_parent_split(parents)
    result: list[dict[str, Any]] = []
    for parent in split["SYNTHETIC_QUERY"]:
        with Image.open(parent["imagePath"]) as opened:
            source = opened.convert("RGB")
        for variant_id, component in enumerate(audit.SYNTHETIC_NUISANCE_CONTROL_V3_CONTROL_COMPONENTS, start=1):
            image = source.copy()
            draw = ImageDraw.Draw(image)
            draw.line((variant_id, 44 - variant_id, 30, 40 - variant_id), fill=(35, 35, 35), width=1)
            path = root / parent["category"] / f"{parent['caseId'].split('/')[-1]}-{component}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG")
            result.append(
                {
                    "caseId": f"camera-r3/{parent['caseId']}/{component}",
                    "parentCaseId": parent["caseId"],
                    "category": parent["category"],
                    "sourceSha256": audit.sha256_file(path),
                    "sourceGroupId": parent["sourceGroupId"],
                    "imagePath": path,
                    "variantId": variant_id,
                    "component": component,
                }
            )
    return sorted(result, key=lambda item: str(item["caseId"]))


def _stimuli(root: Path, parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split = stress.build_fixed_parent_split(parents)
    result: list[dict[str, Any]] = []
    variants = [
        (level, family)
        for level in audit.SYNTHETIC_NUISANCE_CONTROL_V3_LEVELS
        for family in audit.SYNTHETIC_NUISANCE_CONTROL_V3_FAMILIES
    ]
    for parent in split["SYNTHETIC_QUERY"]:
        with Image.open(parent["imagePath"]) as opened:
            source = opened.convert("RGB")
        for variant_id, (level, family) in enumerate(variants, start=1):
            image = source.copy()
            draw = ImageDraw.Draw(image)
            draw.rectangle((3 + variant_id, 3, 5 + variant_id, 5 + variant_id), fill=(0, 0, 0))
            path = root / parent["category"] / f"{parent['caseId'].split('/')[-1]}-{variant_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG")
            result.append(
                {
                    "caseId": f"synthetic-v2/{parent['caseId']}/{variant_id}",
                    "parentCaseId": parent["caseId"],
                    "parentSourceSha256": parent["sourceSha256"],
                    "sourceGroupId": parent["sourceGroupId"],
                    "category": parent["category"],
                    "parentPartition": "FIT",
                    "syntheticTestRole": "QUERY",
                    "syntheticLabel": "SYNTHETIC_STIMULUS",
                    "syntheticDefectFamily": family,
                    "renderIntensityLevel": level,
                    "variantId": variant_id,
                    "relativePath": path.name,
                    "sourceSha256": audit.sha256_file(path),
                    "parameters": {},
                    "outputEncoding": {},
                    "imagePath": path,
                }
            )
    return sorted(result, key=lambda item: str(item["caseId"]))


class _FakeEmbedder:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def extract_patches(self, images: list[Image.Image]) -> list[object]:
        result: list[object] = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
            mean = array.mean(axis=(0, 1)) / 255.0
            spatial_signal = float(array.mean(axis=2).std()) / 255.0
            base = np.array([1.0, spatial_signal, *mean, 0.2, 0.3, 0.4], dtype=np.float32)
            result.append(np.stack([base + index * 0.0001 for index in range(16)], axis=0))
        return result


def _identity_factory(**_kwargs: object) -> dict[str, Any]:
    identity = {
        "schemaVersion": "phone-dino.mvtec-ad-synthetic-stimulus-feature-extractor/2.0",
        "modelWeightsSha256": _digest("fixture-model-weights"),
        "modelRepositorySha256": _digest("fixture-model-repository"),
        "modelEntrypoint": "dinov2_vits14",
        "device": "cpu",
        "preprocessingId": stress.knn.SUCCESSOR_V2_PREPROCESSING_ID,
        "algorithmId": stress.SYNTHETIC_STRESS_V2_ALGORITHM,
        "prototypeSelection": stress.SYNTHETIC_STRESS_V2_PROTOTYPE_SELECTION,
        "syntheticStressEvaluatorModuleSha256": _digest("fixture-stress-evaluator"),
        "syntheticStressAugmentationModuleSha256": _digest("fixture-stress-augmentation"),
        "successorModuleSha256": _digest("fixture-successor"),
        "patchKnnModuleSha256": _digest("fixture-patch-knn"),
        "productionModuleSha256": _digest("fixture-production"),
        "enginesModuleSha256": _digest("fixture-engines"),
        "pythonVersion": "fixture-python",
        "numpyVersion": "fixture-numpy",
        "torchVersion": "fixture-torch",
        "torchvisionVersion": "fixture-torchvision",
        "syntheticNuisanceControlModuleSha256": audit.sha256_file(Path(audit.__file__)),
    }
    identity["sealedDinoSnapshot"] = {
        "schemaVersion": stress.sealed_snapshot.SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_SCHEMA,
        "snapshotSchemaVersion": stress.sealed_snapshot.SEALED_DINO_SNAPSHOT_SCHEMA,
        "repositoryDigestAlgorithm": stress.sealed_snapshot.SEALED_DINO_REPOSITORY_DIGEST_ALGORITHM,
        "weightsDigestAlgorithm": stress.sealed_snapshot.SEALED_DINO_WEIGHTS_DIGEST_ALGORITHM,
        "snapshotManifestSha256": _digest("fixture-sealed-manifest"),
        "snapshotRepositorySha256": identity["modelRepositorySha256"],
        "snapshotWeightsSha256": identity["modelWeightsSha256"],
    }
    return identity


def _install_run_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    *,
    parents: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    stimuli: list[dict[str, Any]],
    events: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    envelope = _envelope()
    stimulus_manifest = _package_manifest("stimulus", audit.stimulus.SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA)
    capture_manifest = _package_manifest("capture", audit.camera.SUCCESSOR_FIT_AUGMENTATION_V2_SCHEMA)
    stimulus_metadata = _package_metadata("stimulus", audit.stimulus.SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA)
    capture_metadata = _package_metadata("capture", audit.camera.SUCCESSOR_FIT_AUGMENTATION_V2_SCHEMA)

    def safe_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        events.append("fit")
        return envelope, _digest("envelope-file"), [dict(item) for item in parents]

    original_calibrate = stress.calibrate_raw_thresholds

    def calibrate(values: list[dict[str, Any]]) -> dict[str, float]:
        events.append("threshold")
        return original_calibrate(values)

    def stimulus_preflight(*_args: object, **_kwargs: object) -> dict[str, Any]:
        assert events[-1] == "threshold"
        events.append("stimulus-metadata")
        return stimulus_metadata

    def capture_preflight(*_args: object, **_kwargs: object) -> dict[str, Any]:
        events.append("capture-metadata")
        return capture_metadata

    original_receipt = audit.create_one_time_registry_receipt

    def receipt(*args: object, **kwargs: object) -> tuple[dict[str, Any], str]:
        events.append("receipt")
        return original_receipt(*args, **kwargs)

    def control_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        assert events[-1] == "receipt"
        events.append("controls")
        return capture_manifest, capture_metadata["fileSha256"], [dict(item) for item in controls]

    def stimulus_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        assert events[-1] == "controls"
        events.append("stimuli")
        return stimulus_manifest, stimulus_metadata["fileSha256"], [dict(item) for item in stimuli]

    monkeypatch.setattr(stress, "load_safe_v2_fit_inputs", safe_loader)
    monkeypatch.setattr(stress, "calibrate_raw_thresholds", calibrate)
    monkeypatch.setattr(audit, "_preflight_stimulus_package", stimulus_preflight)
    monkeypatch.setattr(audit, "_preflight_camera_package", capture_preflight)
    monkeypatch.setattr(audit, "create_one_time_registry_receipt", receipt)
    monkeypatch.setattr(audit, "load_and_validate_r3_camera_controls", control_loader)
    monkeypatch.setattr(stress, "load_and_validate_v2_package", stimulus_loader)
    return stimulus_manifest, capture_manifest


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_name: str = "report.json",
    registry_name: str = "registry",
    events: list[str] | None = None,
    identity_factory: Any = _identity_factory,
    after_install: Any = None,
) -> tuple[dict[str, Any], list[str], Path]:
    parents = _parents(tmp_path / "source")
    controls = _controls(tmp_path / "controls", parents)
    stimuli = _stimuli(tmp_path / "stimuli", parents)
    actual_events = events if events is not None else []
    _install_run_fixtures(
        monkeypatch, parents=parents, controls=controls, stimuli=stimuli, events=actual_events
    )
    registry = tmp_path / registry_name
    if after_install is not None:
        after_install(registry)
    report = audit.run_synthetic_nuisance_control_v3(
        tmp_path / "chain" / "holdout.json",
        tmp_path / "chain" / "contract.json",
        tmp_path / "chain" / "plan.json",
        tmp_path / "chain" / "envelope.json",
        tmp_path / "stimulus-package" / "augmentation_manifest.json",
        tmp_path / "camera-package" / "augmentation_manifest.json",
        tmp_path / output_name,
        source_root=tmp_path / "source",
        stimulus_recipe_path=tmp_path / "stimulus-recipe.json",
        capture_control_recipe_path=tmp_path / "camera-recipe.json",
        registry_root=registry,
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_FakeEmbedder,
        identity_factory=identity_factory,
    )
    return report, actual_events, registry


def test_v3_is_fit_only_threshold_receipt_first_and_response_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, events, registry = _run(tmp_path, monkeypatch)
    assert events == ["fit", "threshold", "stimulus-metadata", "capture-metadata", "receipt", "controls", "stimuli"]
    assert len(list(registry.glob("*.json"))) == 1
    assert report["schemaVersion"] == audit.SYNTHETIC_NUISANCE_CONTROL_V3_REPORT_SCHEMA
    assert report["syntheticOnly"] is True
    assert report["realAnomalyPerformance"] == "NOT_ESTIMATED"
    assert report["realPrecisionRecall"] == "NOT_ESTIMATED"
    assert report["evidenceClass"] == "SYNTHETIC_ENGINEERING_ONLY"
    assert report["testConfiguration"]["rawCalibrationThresholdEstablishedBeforeChildPackageLoad"] is True
    assert report["testConfiguration"]["registryReceiptCreatedBeforeChildPackageLoad"] is True
    assert len(report["rawQueryScores"]) == 12
    assert len(report["genericCaptureControlScores"]) == 36
    assert len(report["syntheticStimulusScores"]) == 108
    assert len(report["parentLevelContrasts"]) == 12
    assert report["aggregate"]["parentLevelContrast"]["parentCount"] == 12
    assert all(item["genericCaptureControlCount"] == 3 for item in report["parentLevelContrasts"])
    assert all(item["syntheticStimulusCount"] == 9 for item in report["parentLevelContrasts"])
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in (
        "syntheticTP",
        "syntheticFP",
        "syntheticFN",
        "syntheticTN",
        "syntheticPrecision",
        "syntheticRecall",
        "syntheticF1",
        "AUROC",
        "averagePrecision",
    ):
        assert forbidden not in serialized


def test_v3_raw_fit_timing_excludes_model_preflight_and_is_not_double_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"seconds": 0.0}
    monkeypatch.setattr(audit.time, "process_time", lambda: clock["seconds"])
    original_extract = audit._extract_features
    extract_calls = {"count": 0}

    def extract(records: list[dict[str, Any]], **kwargs: object) -> dict[str, object]:
        result = original_extract(records, **kwargs)
        if extract_calls["count"] == 0:
            clock["seconds"] += 2.0
        extract_calls["count"] += 1
        return result

    def identity(**kwargs: object) -> dict[str, Any]:
        clock["seconds"] += 100.0
        return _identity_factory(**kwargs)

    def instrument_after_install(_registry: Path) -> None:
        original_loader = stress.load_safe_v2_fit_inputs
        original_calibrate = stress.calibrate_raw_thresholds

        def loader(*args: object, **kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
            result = original_loader(*args, **kwargs)
            clock["seconds"] += 1.0
            return result

        def calibrate(values: list[dict[str, Any]]) -> dict[str, float]:
            result = original_calibrate(values)
            clock["seconds"] += 3.0
            return result

        monkeypatch.setattr(stress, "load_safe_v2_fit_inputs", loader)
        monkeypatch.setattr(stress, "calibrate_raw_thresholds", calibrate)

    monkeypatch.setattr(audit, "_extract_features", extract)
    report, _events, _registry = _run(
        tmp_path,
        monkeypatch,
        identity_factory=identity,
        after_install=instrument_after_install,
    )
    timings = report["execution"]["phaseTimingsSeconds"]
    assert timings["rawFitAndThresholdSeconds"] == 6.0
    assert timings["modelPreflightSeconds"] == 200.0
    assert "totalElapsedSeconds" not in timings
    assert "totalBeforeReportWriteSeconds" in timings


def test_v3_direct_api_rejects_unsealed_feature_identity_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def unsealed_identity(**kwargs: object) -> dict[str, Any]:
        identity = _identity_factory(**kwargs)
        del identity["sealedDinoSnapshot"]
        return identity

    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="missing sealedDinoSnapshot"):
        _run(tmp_path, monkeypatch, events=events, identity_factory=unsealed_identity)
    assert events == []
    assert not list((tmp_path / "registry").glob("*.json"))


def test_v3_parent_level_contrast_requires_exact_3_by_9_coverage() -> None:
    raw = [
        {
            "caseId": f"p-{index}",
            "category": "capsule",
            "score": 0.1,
            "aboveRawCalibrationThreshold": False,
        }
        for index in range(12)
    ]
    controls = [
        {
            "parentCaseId": f"p-{parent}",
            "score": 0.2,
            "aboveRawCalibrationThreshold": False,
        }
        for parent in range(12)
        for _ in range(3)
    ]
    stimuli = [
        {
            "parentCaseId": f"p-{parent}",
            "score": 0.3,
            "aboveRawCalibrationThreshold": True,
        }
        for parent in range(12)
        for _ in range(9)
    ]
    assert len(audit.build_parent_level_contrasts(raw, controls, stimuli)) == 12
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="3 controls and 9 stimuli"):
        audit.build_parent_level_contrasts(raw, controls[:-1], stimuli)
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="3 controls and 9 stimuli"):
        audit.build_parent_level_contrasts(raw, controls, stimuli[:-1])


def test_r3_loader_restricts_child_validation_to_exact_query_parent_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parents = _parents(tmp_path / "source")
    split = stress.build_fixed_parent_split(parents)
    query_parents = split["SYNTHETIC_QUERY"]
    selected = [
        {
            "caseId": f"control/{parent['caseId']}/{component}",
            "parentCaseId": parent["caseId"],
            "component": component,
        }
        for parent in query_parents
        for component in audit.SYNTHETIC_NUISANCE_CONTROL_V3_CONTROL_COMPONENTS
    ]
    unrelated = {"caseId": "control/unrelated", "parentCaseId": "not-a-query-parent", "component": "registration"}
    document = {"schemaVersion": audit.camera.SUCCESSOR_FIT_AUGMENTATION_V2_SCHEMA}
    seen_parent_ids: list[str] = []

    monkeypatch.setattr(audit, "_read_external_metadata", lambda *_args, **_kwargs: (document, _digest("camera-file")))
    monkeypatch.setattr(audit.camera, "load_successor_fit_camera_recipe_v2", lambda _path: ({}, _digest("camera-recipe")))
    monkeypatch.setattr(audit.camera, "_validate_manifest_document", lambda *_args, **_kwargs: [*selected, unrelated])
    monkeypatch.setattr(audit.camera, "_require_external_package_root", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(audit, "_assert_chain_binding", lambda *_args, **_kwargs: None)

    def validate(record: dict[str, Any], *, parent: dict[str, Any], **_kwargs: object) -> dict[str, Any]:
        seen_parent_ids.append(parent["caseId"])
        return {**record, "imagePath": tmp_path / "unused.png"}

    monkeypatch.setattr(audit, "_validate_camera_control_record", validate)
    _manifest, _file_sha, records = audit.load_and_validate_r3_camera_controls(
        tmp_path / "camera-package" / "augmentation_manifest.json",
        recipe_path=tmp_path / "camera-recipe.json",
        envelope=_envelope(),
        envelope_file_sha256=_digest("envelope-file"),
        fit_parents=parents,
        query_parents=query_parents,
    )
    assert len(records) == 36
    assert set(seen_parent_ids) == {parent["caseId"] for parent in query_parents}
    assert len(seen_parent_ids) == 36
    monkeypatch.setattr(audit.camera, "_validate_manifest_document", lambda *_args, **_kwargs: selected[:-1])
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="exactly 3 controls"):
        audit.load_and_validate_r3_camera_controls(
            tmp_path / "camera-package" / "augmentation_manifest.json",
            recipe_path=tmp_path / "camera-recipe.json",
            envelope=_envelope(),
            envelope_file_sha256=_digest("envelope-file"),
            fit_parents=parents,
            query_parents=query_parents,
        )


def test_registry_receipt_race_is_exclusive(tmp_path: Path) -> None:
    identity = _receipt_identity_fixture("race-fixture")
    root = tmp_path / "registry"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(audit.create_one_time_registry_receipt, root, identity=identity)
            for _ in range(2)
        ]
    outcomes: list[object] = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except audit.SyntheticNuisanceControlV3Error as error:
            outcomes.append(error)
    assert sum(isinstance(item, tuple) for item in outcomes) == 1
    assert sum(isinstance(item, audit.SyntheticNuisanceControlV3Error) for item in outcomes) == 1
    assert len(list(root.glob("*.json"))) == 1


def test_registry_receipt_rechecks_root_and_ancestor_identities_after_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "registry"
    original_identity = audit._directory_identity
    root_reads = {"count": 0}

    def substituted_identity(path: Path) -> tuple[int, int, int]:
        signature = original_identity(path)
        if path.name == "registry":
            root_reads["count"] += 1
            if root_reads["count"] == 2:
                return signature[0], signature[1] + 1, signature[2]
        return signature

    monkeypatch.setattr(audit, "_directory_identity", substituted_identity)
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="registry_root ancestor chain changed"):
        audit.create_one_time_registry_receipt(root, identity=_receipt_identity_fixture("identity-race"))
    # A raced slot is intentionally left immutable rather than silently reused.
    assert len(list(root.glob("*.json"))) == 1


def test_registry_root_replacement_rejects_before_any_child_package_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def replace_after_receipt(registry: Path) -> None:
        original_receipt = audit.create_one_time_registry_receipt

        def receipt(*args: object, **kwargs: object) -> tuple[dict[str, Any], str]:
            result = original_receipt(*args, **kwargs)
            replacement_source = registry.with_name("registry-original")
            registry.rename(replacement_source)
            registry.mkdir()
            return result

        monkeypatch.setattr(audit, "create_one_time_registry_receipt", receipt)

    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="registry_root ancestor chain changed"):
        _run(tmp_path, monkeypatch, events=events, after_install=replace_after_receipt)
    assert events == ["fit", "threshold", "stimulus-metadata", "capture-metadata", "receipt"]


def test_child_loaders_reject_changed_preflight_manifest_before_child_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_accesses: list[str] = []
    monkeypatch.setattr(
        audit,
        "_read_external_metadata",
        lambda *_args, **_kwargs: ({}, _digest("changed-camera-manifest")),
    )
    monkeypatch.setattr(
        audit,
        "_validate_camera_control_record",
        lambda *_args, **_kwargs: child_accesses.append("camera") or (_ for _ in ()).throw(AssertionError("child opened")),
    )
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="changed after preflight"):
        audit.load_and_validate_r3_camera_controls(
            tmp_path / "camera" / "augmentation_manifest.json",
            recipe_path=tmp_path / "camera-recipe.json",
            envelope=_envelope(),
            envelope_file_sha256=_digest("envelope-file"),
            fit_parents=[],
            query_parents=[],
            expected_manifest_file_sha256=_digest("preflight-camera-manifest"),
        )
    assert child_accesses == []

    loader_calls: list[str] = []
    monkeypatch.setattr(
        stress.augmentation.v1,
        "_read_external_json",
        lambda *_args, **_kwargs: ({}, _digest("changed-stimulus-manifest")),
    )
    monkeypatch.setattr(
        stress.augmentation,
        "load_validated_synthetic_stress_v2",
        lambda *_args, **_kwargs: loader_calls.append("stimulus")
        or (_ for _ in ()).throw(AssertionError("child opened")),
    )
    with pytest.raises(stress.SyntheticStressV2Error, match="changed after preflight"):
        stress.load_and_validate_v2_package(
            tmp_path / "stimulus" / "augmentation_manifest.json",
            tmp_path / "chain" / "holdout.json",
            tmp_path / "chain" / "contract.json",
            tmp_path / "chain" / "plan.json",
            tmp_path / "chain" / "envelope.json",
            source_root=tmp_path / "source",
            recipe_path=tmp_path / "stimulus-recipe.json",
            query_parents=[],
            expected_manifest_file_sha256=_digest("preflight-stimulus-manifest"),
        )
    assert loader_calls == []


def test_existing_receipt_blocks_child_loaders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # First run consumes the deterministic identity.  A second run uses a new
    # report slot but must fail before either child loader can be reached.
    _run(tmp_path, monkeypatch, output_name="first.json")
    monkeypatch.undo()
    parents = _parents(tmp_path / "source-second")
    controls = _controls(tmp_path / "controls-second", parents)
    stimuli = _stimuli(tmp_path / "stimuli-second", parents)
    events: list[str] = []
    _install_run_fixtures(monkeypatch, parents=parents, controls=controls, stimuli=stimuli, events=events)
    # The parent contents are intentionally the same logical fixture identity
    # only when they share the original source path and digest. Reinstall the
    # original raw fixture instead of making fresh parent bytes.
    monkeypatch.undo()
    parents = _parents(tmp_path / "source")
    controls = _controls(tmp_path / "controls-retry", parents)
    stimuli = _stimuli(tmp_path / "stimuli-retry", parents)
    _install_run_fixtures(monkeypatch, parents=parents, controls=controls, stimuli=stimuli, events=events)

    def forbidden_controls(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("existing receipt must reject before control child access")

    def forbidden_stimuli(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("existing receipt must reject before stimulus child access")

    monkeypatch.setattr(audit, "load_and_validate_r3_camera_controls", forbidden_controls)
    monkeypatch.setattr(stress, "load_and_validate_v2_package", forbidden_stimuli)
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="already consumed"):
        audit.run_synthetic_nuisance_control_v3(
            tmp_path / "chain" / "holdout.json",
            tmp_path / "chain" / "contract.json",
            tmp_path / "chain" / "plan.json",
            tmp_path / "chain" / "envelope.json",
            tmp_path / "stimulus-package" / "augmentation_manifest.json",
            tmp_path / "camera-package" / "augmentation_manifest.json",
            tmp_path / "second.json",
            source_root=tmp_path / "source",
            stimulus_recipe_path=tmp_path / "stimulus-recipe.json",
            capture_control_recipe_path=tmp_path / "camera-recipe.json",
            registry_root=tmp_path / "registry",
            model_repo=tmp_path / "model",
            model_weights=tmp_path / "weights.pth",
            embedder_factory=_FakeEmbedder,
            identity_factory=_identity_factory,
        )
    assert "controls" not in events and "stimuli" not in events


def test_package_tamper_after_receipt_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parents = _parents(tmp_path / "source")
    controls = _controls(tmp_path / "controls", parents)
    stimuli = _stimuli(tmp_path / "stimuli", parents)
    events: list[str] = []
    stimulus_manifest, _capture_manifest = _install_run_fixtures(
        monkeypatch, parents=parents, controls=controls, stimuli=stimuli, events=events
    )

    def tampered_stimulus_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        assert events[-1] == "controls"
        return {
            **stimulus_manifest,
            "augmentationManifestSha256": _digest("changed-after-receipt"),
        }, _digest("stimulus-file"), stimuli

    monkeypatch.setattr(stress, "load_and_validate_v2_package", tampered_stimulus_loader)
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="changed after one-time receipt"):
        audit.run_synthetic_nuisance_control_v3(
            tmp_path / "chain" / "holdout.json",
            tmp_path / "chain" / "contract.json",
            tmp_path / "chain" / "plan.json",
            tmp_path / "chain" / "envelope.json",
            tmp_path / "stimulus-package" / "augmentation_manifest.json",
            tmp_path / "camera-package" / "augmentation_manifest.json",
            tmp_path / "report.json",
            source_root=tmp_path / "source",
            stimulus_recipe_path=tmp_path / "stimulus-recipe.json",
            capture_control_recipe_path=tmp_path / "camera-recipe.json",
            registry_root=tmp_path / "registry",
            model_repo=tmp_path / "model",
            model_weights=tmp_path / "weights.pth",
            embedder_factory=_FakeEmbedder,
            identity_factory=_identity_factory,
        )
    assert events[-1] == "controls"
    assert len(list((tmp_path / "registry").glob("*.json"))) == 1


def test_malformed_v2_metadata_rejects_before_receipt_and_child_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata-only preflight must spend neither the receipt nor child bytes."""

    parents = _parents(tmp_path / "source")
    envelope = _envelope()
    chain_paths = [
        tmp_path / "chain" / "holdout.json",
        tmp_path / "chain" / "contract.json",
        tmp_path / "chain" / "plan.json",
        tmp_path / "chain" / "envelope.json",
    ]
    for path in chain_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    def safe_loader(*_args: object, **kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        assert set(kwargs["partitions"]) == {"FIT"}
        return envelope, _digest("envelope-file"), [dict(item) for item in parents]

    monkeypatch.setattr(stress.successor, "load_successor_safe_normal_inputs", safe_loader)
    package_dir = tmp_path / "stimulus-package"
    recipe_path = audit.REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json"
    audit.stimulus.generate_synthetic_anomaly_stress_v2(
        chain_paths[0],
        chain_paths[1],
        chain_paths[2],
        chain_paths[3],
        source_root=tmp_path / "source",
        recipe_path=recipe_path,
        output_dir=package_dir,
    )
    manifest_path = package_dir / "augmentation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["syntheticLabel"] = "NOT_SYNTHETIC_STIMULUS"
    manifest["augmentationManifestSha256"] = audit.stimulus._document_digest(
        manifest, "augmentationManifestSha256"
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    receipt_calls: list[str] = []
    child_calls: list[str] = []
    monkeypatch.setattr(
        audit,
        "create_one_time_registry_receipt",
        lambda *_args, **_kwargs: receipt_calls.append("receipt")
        or (_ for _ in ()).throw(AssertionError("receipt must not be created")),
    )
    monkeypatch.setattr(
        audit,
        "_preflight_camera_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("camera metadata must not be reached")),
    )
    monkeypatch.setattr(
        audit,
        "load_and_validate_r3_camera_controls",
        lambda *_args, **_kwargs: child_calls.append("camera")
        or (_ for _ in ()).throw(AssertionError("camera child must not be opened")),
    )
    monkeypatch.setattr(
        stress,
        "load_and_validate_v2_package",
        lambda *_args, **_kwargs: child_calls.append("stimulus")
        or (_ for _ in ()).throw(AssertionError("stimulus child must not be opened")),
    )
    registry = tmp_path / "registry"
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="child parent binding"):
        audit.run_synthetic_nuisance_control_v3(
            chain_paths[0],
            chain_paths[1],
            chain_paths[2],
            chain_paths[3],
            manifest_path,
            tmp_path / "camera-package" / "augmentation_manifest.json",
            tmp_path / "report.json",
            source_root=tmp_path / "source",
            stimulus_recipe_path=recipe_path,
            capture_control_recipe_path=tmp_path / "camera-recipe.json",
            registry_root=registry,
            model_repo=tmp_path / "model",
            model_weights=tmp_path / "weights.pth",
            embedder_factory=_FakeEmbedder,
            identity_factory=_identity_factory,
        )
    assert receipt_calls == []
    assert child_calls == []
    assert not list(registry.glob("*.json"))


def test_report_writer_uses_no_follow_exclusive_create_and_rechecks_parent_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"
    original_open = audit.os.open
    observed_flags: list[int] = []

    def recording_open(path: str, flags: int, mode: int = 0o777) -> int:
        observed_flags.append(flags)
        return original_open(path, flags, mode)

    def simulated_parent_swap(*_args: object, **_kwargs: object) -> None:
        raise audit.SyntheticNuisanceControlV3Error("simulated report parent swap")

    monkeypatch.setattr(audit.os, "open", recording_open)
    monkeypatch.setattr(audit, "_verify_v3_report_parent_signatures", simulated_parent_swap)
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="parent swap"):
        audit._write_new_report(output, {"finite": 1.0}, repository_root=audit.REPOSITORY_ROOT)
    assert observed_flags
    assert observed_flags[-1] & audit.os.O_EXCL
    if hasattr(audit.os, "O_NOFOLLOW"):
        assert observed_flags[-1] & audit.os.O_NOFOLLOW
    # The file is intentionally not removed after a detected race: it remains
    # an immutable failed slot and cannot be reused to mask the event.
    assert output.is_file()

    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="finite JSON"):
        audit._write_new_report(
            tmp_path / "nonfinite.json", {"nonfinite": float("nan")}, repository_root=audit.REPOSITORY_ROOT
        )


def _valid_stimulus_metadata_document(envelope: dict[str, Any]) -> dict[str, Any]:
    recipe = {"fixture": "recipe"}
    parent_evidence = envelope["parentEvidence"]
    document: dict[str, Any] = {
        "schemaVersion": audit.stimulus.SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "purpose": audit.stimulus.SYNTHETIC_ANOMALY_STRESS_V2_PURPOSE,
        "postV1Exploratory": True,
        "comparisonOrPromotionAllowed": False,
        "inputPolicy": audit.stimulus.SYNTHETIC_ANOMALY_STRESS_INPUT_POLICY,
        "blindPolicy": audit.stimulus.SYNTHETIC_ANOMALY_STRESS_BLIND_POLICY,
        "resultLabel": audit.stimulus.SYNTHETIC_ANOMALY_STRESS_RESULT_LABEL,
        "parentPartition": "FIT",
        "parentSplitAlgorithm": audit.stimulus.SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_ALGORITHM,
        "parentSplitCountsPerCategory": audit.stimulus.SYNTHETIC_ANOMALY_STRESS_PARENT_SPLIT_COUNTS_PER_CATEGORY,
        "syntheticQueryParentIdentitySha256": _digest("query-parents"),
        "parentHoldoutManifestFileSha256": parent_evidence["holdoutManifestFileSha256"],
        "parentHoldoutManifestDeclaredSha256": parent_evidence["holdoutManifestDeclaredSha256"],
        "parentSelectionContractFileSha256": parent_evidence["selectionContractFileSha256"],
        "parentSelectionContractDeclaredSha256": parent_evidence["selectionContractDeclaredSha256"],
        "successorSealFileSha256": envelope["sealFileSha256"],
        "successorSealDeclaredSha256": envelope["sealDeclaredSha256"],
        "successorPlanFileSha256": envelope["planFileSha256"],
        "successorPlanDeclaredSha256": envelope["planDeclaredSha256"],
        "successorEnvelopeFileSha256": _digest("envelope-file"),
        "successorEnvelopeDeclaredSha256": envelope["successorEnvelopeSha256"],
        "successorFitIdentitySha256": envelope["successorPartitionIdentities"]["FIT"],
        "parentNormalConfirmationIdentitySha256": parent_evidence["parentNormalConfirmationIdentitySha256"],
        "recipeFileSha256": _digest("recipe-file"),
        "recipe": recipe,
        "variantsPerParent": 9,
        "generation": {},
        "records": [{} for _ in range(108)],
    }
    document["augmentationManifestSha256"] = audit._document_digest(document, "augmentationManifestSha256")
    return document


def test_stimulus_metadata_rejects_bad_schema_digest_and_repo_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="Git working tree"):
        audit._read_external_metadata(
            audit.REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
            description="fixture",
            repository_root=audit.REPOSITORY_ROOT,
        )
    envelope = _envelope()
    document = _valid_stimulus_metadata_document(envelope)
    monkeypatch.setattr(audit, "_read_external_metadata", lambda *_args, **_kwargs: (document, _digest("manifest-file")))
    monkeypatch.setattr(audit.stimulus, "load_synthetic_anomaly_stress_recipe_v2", lambda _path: (document["recipe"], document["recipeFileSha256"]))
    result = audit._preflight_stimulus_package(
        tmp_path / "outside" / "manifest.json",
        recipe_path=tmp_path / "recipe.json",
        envelope=envelope,
        envelope_file_sha256=_digest("envelope-file"),
        repository_root=audit.REPOSITORY_ROOT,
    )
    assert result["document"] is document
    changed_schema = dict(document)
    changed_schema["schemaVersion"] = "unsupported"
    monkeypatch.setattr(audit, "_read_external_metadata", lambda *_args, **_kwargs: (changed_schema, _digest("manifest-file")))
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="schema"):
        audit._preflight_stimulus_package(
            tmp_path / "outside" / "manifest.json",
            recipe_path=tmp_path / "recipe.json",
            envelope=envelope,
            envelope_file_sha256=_digest("envelope-file"),
            repository_root=audit.REPOSITORY_ROOT,
        )
    changed_digest = dict(document)
    changed_digest["augmentationManifestSha256"] = _digest("tampered")
    monkeypatch.setattr(audit, "_read_external_metadata", lambda *_args, **_kwargs: (changed_digest, _digest("manifest-file")))
    with pytest.raises(audit.SyntheticNuisanceControlV3Error, match="declared digest"):
        audit._preflight_stimulus_package(
            tmp_path / "outside" / "manifest.json",
            recipe_path=tmp_path / "recipe.json",
            envelope=envelope,
            envelope_file_sha256=_digest("envelope-file"),
            repository_root=audit.REPOSITORY_ROOT,
        )


def test_no_v1_report_api_or_classification_metric_fields() -> None:
    signature = inspect.signature(audit.run_synthetic_nuisance_control_v3)
    assert "report" not in " ".join(signature.parameters).lower()
    source = inspect.getsource(audit)
    assert "mvtec_synthetic_anomaly_test" not in source
    assert "synthetic_confusion_metrics" not in source
    assert "syntheticPrecision" not in source
