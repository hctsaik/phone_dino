from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _tool_module():
    path = Path(__file__).parents[1] / "tools" / "select_mvtec_ad_normal_candidate.py"
    spec = importlib.util.spec_from_file_location("mvtec_normal_selection", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(character: str) -> str:
    if len(character) == 1:
        return f"sha256:{character * 64}"
    return f"sha256:{hashlib.sha256(character.encode('utf-8')).hexdigest()}"


def _file_sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _summary(values: list[float], prefix: str) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        f"{prefix}Cases": len(ordered),
        f"{prefix}Median": ordered[len(ordered) // 2],
        f"{prefix}P95": ordered[max(0, (len(ordered) * 95 + 99) // 100 - 1)],
        f"{prefix}Max": ordered[-1],
    }


def _identity_from_score(record: dict[str, object]) -> dict[str, object]:
    return {
        "caseId": record["caseId"],
        "category": record["category"],
        "role": record["role"],
        "kind": record["kind"],
        "sourceSha256": record["sourceSha256"],
        "isAugmentation": record["isAugmentation"],
        "variantId": record.get("variantId"),
        "parentCaseId": record.get("parentCaseId"),
        "parentSourceSha256": record.get("parentSourceSha256"),
        "augmentationRecipeSha256": record.get("augmentationRecipeSha256"),
    }


def _candidate_configuration(max_prototypes: int) -> dict[str, object]:
    return {
        "algorithmId": "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1",
        "batchSize": 4,
        "memoryBankSelection": "DETERMINISTIC_EVENLY_SPACED_PATCH_SUBSET_AFTER_STABLE_PARENT_SORT",
        "maxPrototypePatches": max_prototypes,
        "topKMostAnomalousPatches": 5,
        "prototypeBlockSize": 256,
    }


def _feature_extractor_identity() -> dict[str, object]:
    return {
        "schemaVersion": "phone-dino.mvtec-ad-feature-extractor/1.0",
        "modelWeightsSha256": _digest("e"),
        "modelRepositorySha256": _digest("9"),
        "preprocessingId": "DINO_TEST_PREPROCESSING",
        "preprocessing": {
            "colorSpace": "RGB",
            "resizeShortEdge": 256,
            "centerCropWidth": 224,
            "centerCropHeight": 224,
            "resizeAntialias": True,
            "normalizeMean": [0.485, 0.456, 0.406],
            "normalizeStd": [0.229, 0.224, 0.225],
        },
        "modelEntrypoint": "dinov2_vits14",
        "device": "cpu",
        "iterationToolSha256": _digest("f"),
        "productionModuleSha256": _digest("a"),
        "enginesModuleSha256": _digest("b"),
        "mvtecResearchModuleSha256": _digest("c"),
        "pythonVersion": "3.11.0",
        "numpyVersion": "2.0.0",
        "torchVersion": "2.0.0",
        "torchvisionVersion": "0.0.0",
        "pillowVersion": "10.0.0",
        "torchThreadCount": 1,
        "torchBackend": {
            "deterministicAlgorithmsEnabled": False,
            "mkldnnAvailable": True,
            "mkldnnEnabled": True,
        },
    }


def _report_document(
    *,
    original_score: float,
    augmented_scores: list[float],
    max_prototypes: int,
    blind_state: str = "NOT_RUN",
) -> dict[str, object]:
    tool = _tool_module()
    original: dict[str, object] = {
        "caseId": "capsule/tuning/001",
        "category": "capsule",
        "role": "THRESHOLD_TUNING",
        "kind": "NOMINAL",
        "defect": "good",
        "sourceSha256": _digest("1"),
        "isAugmentation": False,
        "variantId": None,
        "score": original_score,
    }
    augmented = [
        {
            "caseId": f"capsule/tuning/001/camera-augmentation/{index:02d}",
            "category": "capsule",
            "role": "THRESHOLD_TUNING",
            "kind": "NOMINAL",
            "defect": "good",
            "sourceSha256": _digest(str(index + 1)),
            "isAugmentation": True,
            "variantId": index,
            "parentCaseId": original["caseId"],
            "parentSourceSha256": original["sourceSha256"],
            "augmentationRecipeSha256": _digest("d"),
            "score": score,
        }
        for index, score in enumerate(augmented_scores, start=1)
    ]
    fit_original = {
        "caseId": "capsule/fit/001", "category": "capsule", "role": "FIT", "kind": "NOMINAL",
        "sourceSha256": _digest("7"), "isAugmentation": False,
        "variantId": None,
        "parentCaseId": None, "parentSourceSha256": None, "augmentationRecipeSha256": None,
    }
    fit_augmented = [
        {
            "caseId": f"capsule/fit/001/camera-augmentation/{index:02d}", "category": "capsule", "role": "FIT", "kind": "NOMINAL",
            "sourceSha256": _digest(str(index + 7)), "isAugmentation": True,
            "variantId": index,
            "parentCaseId": fit_original["caseId"], "parentSourceSha256": fit_original["sourceSha256"],
            "augmentationRecipeSha256": _digest("d"),
        }
        for index in range(1, len(augmented_scores) + 1)
    ]
    calibration_scores = [original, *augmented]
    feature_inputs = sorted(
        [fit_original, *fit_augmented, *(_identity_from_score(record) for record in calibration_scores)],
        key=lambda record: str(record["caseId"]),
    )
    document: dict[str, object] = {
        "schemaVersion": "phone-dino.mvtec-ad-iteration-report/1.4",
        "authoritative": False,
        "productionAuthorized": False,
        "disclaimer": "Offline MVTec research fixture only.",
        "selectionProtocol": "NORMAL_ONLY_ITERATION_THEN_BLIND_REPORTING_ONLY",
        "blindReporting": {"state": blind_state, "blindSourcePolicy": "ORIGINAL_ONLY", "reason": "NORMAL_ONLY_ITERATION"},
        "inputManifest": "C:/outside/subset_manifest.json",
        "inputManifestDeclaredSha256": _digest("a"),
        "inputManifestFileSha256": _digest("b"),
        "augmentation": {
            "state": "NORMAL_FIT_AND_TUNING_ONLY", "blindPolicy": "BLIND_ORIGINAL_ONLY", "blindAugmentedCount": 0,
            "augmentationManifestPath": "C:/outside/augmentation_manifest.json", "augmentationManifestSha256": _digest("c"),
            "recipeSha256": _digest("d"), "variantsPerParent": len(augmented_scores),
            "derivedRecordCount": len(augmented) + len(fit_augmented),
        },
        "algorithm": {
            "id": "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1", "modelRepository": "C:/outside/dinov2",
            "modelRepositorySha256": _digest("9"), "modelWeights": "C:/outside/weights.pth", "modelWeightsSha256": _digest("e"),
            "preprocessingId": "DINO_TEST_PREPROCESSING", "device": "cpu",
            "memoryBankSelection": "DETERMINISTIC_EVENLY_SPACED_PATCH_SUBSET_AFTER_STABLE_PARENT_SORT",
            "maxPrototypePatches": max_prototypes, "topKMostAnomalousPatches": 5, "prototypeBlockSize": 256,
        },
        "execution": {
            "batchSize": 4, "featureCache": "C:/outside/cache", "featureCacheHits": 0, "featureCacheMisses": 6,
            "iterationToolSha256": _digest("f"), "featureCacheSchemaVersion": "phone-dino.mvtec-ad-feature-cache/1.1",
            "phaseTimingsSeconds": {
                "provenanceSeconds": 0.1, "inputVerificationSeconds": 0.1, "cacheValidationSeconds": 0.1,
                "cacheWriteSeconds": 0.1, "featureInferenceSeconds": 0.2, "scoringSeconds": 0.3,
                "pixelMetricsSeconds": 0.0, "totalElapsedSeconds": 0.9,
            },
            "python": "3.11", "platform": "test", "numpyVersion": "2.0", "torchVersion": "2.0", "torchThreadCount": 1,
        },
        "calibrationScores": calibration_scores,
        "scores": [original],
        "pixelLocalization": None,
    }
    document["featureExtractor"] = _feature_extractor_identity()
    document["featureExtractorIdentitySha256"] = tool.canonical_json_sha256(document["featureExtractor"])
    document["candidateConfiguration"] = _candidate_configuration(max_prototypes)
    document["candidateConfigurationSha256"] = tool.canonical_json_sha256(document["candidateConfiguration"])
    document["normalOnlyEvidence"] = _normal_evidence(tool, feature_inputs, calibration_scores)
    document["categories"] = {"capsule": _category(calibration_scores, max_prototypes)}
    return document


def _normal_evidence(tool: object, feature_inputs: list[dict[str, object]], calibration_scores: list[dict[str, object]]) -> dict[str, object]:
    calibration_inputs = sorted((_identity_from_score(record) for record in calibration_scores), key=lambda record: str(record["caseId"]))
    originals = [record for record in calibration_inputs if not record["isAugmentation"]]
    return {
        "featureInputCount": len(feature_inputs), "featureInputRoles": ["FIT", "THRESHOLD_TUNING"],
        "featureInputKinds": ["NOMINAL"], "blindFeatureInputCount": 0, "anomalyFeatureInputCount": 0,
        "normalInputRecordCount": len(feature_inputs), "featureInputs": feature_inputs,
        "featureInputIdentitySha256": tool.canonical_json_sha256(feature_inputs),
        "normalInputIdentitySha256": tool.canonical_json_sha256(feature_inputs),
        "reportedScoreCount": 1, "reportedScoreRoles": ["THRESHOLD_TUNING"], "reportedScoreKinds": ["NOMINAL"],
        "calibrationScoreCount": len(calibration_scores), "calibrationScoreRoles": ["THRESHOLD_TUNING"],
        "calibrationScoreKinds": ["NOMINAL"], "calibrationInputs": calibration_inputs,
        "calibrationInputIdentitySha256": tool.canonical_json_sha256(calibration_inputs),
        "originalTuningInputCount": len(originals), "originalTuningInputs": originals,
        "originalTuningInputIdentitySha256": tool.canonical_json_sha256(originals),
    }


def _category(records: list[dict[str, object]], max_prototypes: int) -> dict[str, object]:
    original_scores = [float(record["score"]) for record in records if not record["isAugmentation"]]
    augmented_scores = [float(record["score"]) for record in records if record["isAugmentation"]]
    scores = original_scores + augmented_scores
    category: dict[str, object] = {
        "blindCases": 0, "blindNominalCases": 0, "blindAnomalyCases": 0, "imageAuRoc": None,
        "thresholdFromNominalTuning": max(scores), "nominalAboveThresholdRate": None, "anomalyAboveThresholdRate": None,
        "fitOriginalCount": 1, "fitAugmentedCount": len(augmented_scores),
        "tuningOriginalCount": len(original_scores), "tuningAugmentedCount": len(augmented_scores),
        "fitPatchCount": 256, "prototypePatchCount": max_prototypes, "patchGridHeight": 16, "patchGridWidth": 16,
    }
    category |= _summary(scores, "normalScore")
    category["normalCalibrationCases"] = category.pop("normalScoreCases")
    category |= _summary(original_scores, "originalTuningNormalScore")
    category |= _summary(augmented_scores, "augmentedTuningNormalScore")
    return category


def _write_report(path: Path, **kwargs: object) -> Path:
    path.write_text(json.dumps(_report_document(**kwargs), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_contract(
    path: Path,
    *,
    reference_path: Path,
    candidate_configurations: dict[str, dict[str, object]],
    paired_cap: float = 0.1,
    per_variant_paired_cap: float | None = None,
) -> Path:
    tool = _tool_module()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    evidence = reference["normalOnlyEvidence"]
    document: dict[str, object] = {
        "schemaVersion": "phone-dino.mvtec-ad-normal-selection-contract/1.2",
        "authoritative": False, "productionAuthorized": False, "purpose": "OFFLINE_MVTEC_NORMAL_ONLY_CONFIGURATION_LOCK",
        "inputManifestDeclaredSha256": _digest("a"), "inputManifestFileSha256": _digest("b"),
        "candidates": [
            {"id": candidate_id, "candidateConfiguration": configuration}
            for candidate_id, configuration in candidate_configurations.items()
        ],
        "candidateUniverse": {
            "algorithmId": "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1", "modelWeightsSha256": _digest("e"),
            "modelRepositorySha256": _digest("9"),
            "preprocessingId": "DINO_TEST_PREPROCESSING", "device": "cpu", "iterationToolSha256": _digest("f"),
            "featureExtractorIdentitySha256": reference["featureExtractorIdentitySha256"],
            "categories": ["capsule"], "normalInputIdentitySha256": evidence["normalInputIdentitySha256"],
            "calibrationInputIdentitySha256": evidence["calibrationInputIdentitySha256"],
            "originalTuningInputIdentitySha256": evidence["originalTuningInputIdentitySha256"],
            "augmentationManifestSha256": _digest("c"), "recipeSha256": _digest("d"),
            "augmentationVariantsPerParent": reference["augmentation"]["variantsPerParent"],
        },
        "referenceCandidate": {"id": next(iter(candidate_configurations)), "reportSha256": _file_sha256(reference_path)},
        "gate": {
            "maxThresholdIncreaseVsReference": 0.0, "maxOriginalP95IncreaseVsReference": 0.0,
            "maxAugmentedP95MinusOriginalP95": 0.1, "maxPairedAugmentedScoreDeltaP95": paired_cap,
            "maxPairedAugmentedScoreDeltaMax": paired_cap,
            "perVariantPairedDeltaGates": [
                {
                    "variantId": variant_id,
                    "maxPairedAugmentedScoreDeltaP95": paired_cap if per_variant_paired_cap is None else per_variant_paired_cap,
                    "maxPairedAugmentedScoreDeltaMax": paired_cap if per_variant_paired_cap is None else per_variant_paired_cap,
                }
                for variant_id in range(1, int(reference["augmentation"]["variantsPerParent"]) + 1)
            ],
        },
        "selection": {"objective": tool.NORMAL_OBJECTIVE},
    }
    document["contractSha256"] = tool.canonical_json_sha256(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_normal_selection_locks_best_eligible_candidate_without_blind_data(tmp_path: Path) -> None:
    tool = _tool_module()
    reference = _write_report(tmp_path / "reference.json", original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    candidate = _write_report(tmp_path / "candidate.json", original_score=0.10, augmented_scores=[0.12, 0.11], max_prototypes=2048)
    configurations = {"v3-1024": _candidate_configuration(1024), "v3-2048": _candidate_configuration(2048)}
    contract = _write_contract(tmp_path / "contract.json", reference_path=reference, candidate_configurations=configurations)
    output_path = tmp_path / "selection.json"
    result = tool.run_selection(contract, {"v3-1024": reference, "v3-2048": candidate}, output_path)
    assert result["decision"]["state"] == "RESEARCH_CONFIGURATION_LOCKED"
    assert result["decision"]["selectedCandidateId"] == "v3-2048"
    assert result["decision"]["selectionUses"] == "NORMAL_ONLY_NO_BLIND_OR_ANOMALY_INPUT"
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["selectionSha256"] == tool.canonical_json_sha256({key: value for key, value in persisted.items() if key != "selectionSha256"})


def test_normal_selection_rejects_blind_or_hyperparameter_mismatched_reports(tmp_path: Path) -> None:
    tool = _tool_module()
    reference = _write_report(tmp_path / "reference.json", original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    unsafe = _write_report(tmp_path / "unsafe.json", original_score=0.10, augmented_scores=[0.12, 0.11], max_prototypes=4096, blind_state="REPORTED_ONCE_AFTER_CONFIGURATION_LOCK")
    configurations = {"reference": _candidate_configuration(1024), "candidate": _candidate_configuration(2048)}
    contract = _write_contract(tmp_path / "contract.json", reference_path=reference, candidate_configurations=configurations)
    result = tool.run_selection(contract, {"reference": reference, "candidate": unsafe}, tmp_path / "selection.json")
    evaluation = next(item for item in result["candidateEvaluations"] if item["id"] == "candidate")
    assert evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "not a normal-only report" in evaluation["rejectionReasons"][0]

    mismatch = _write_report(tmp_path / "mismatch.json", original_score=0.10, augmented_scores=[0.12, 0.11], max_prototypes=4096)
    result = tool.run_selection(contract, {"reference": reference, "candidate": mismatch}, tmp_path / "selection-mismatch.json")
    evaluation = next(item for item in result["candidateEvaluations"] if item["id"] == "candidate")
    assert evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "predeclared candidate" in evaluation["rejectionReasons"][0]


def test_normal_selection_rejects_calibration_subset_even_when_report_is_self_consistent(tmp_path: Path) -> None:
    tool = _tool_module()
    reference = _write_report(tmp_path / "reference.json", original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    subset = _report_document(original_score=0.10, augmented_scores=[0.12, 0.11], max_prototypes=2048)
    subset["calibrationScores"] = subset["calibrationScores"][:-1]
    subset["categories"] = {"capsule": _category(subset["calibrationScores"], 2048)}
    subset["normalOnlyEvidence"] = _normal_evidence(
        tool,
        subset["normalOnlyEvidence"]["featureInputs"],
        subset["calibrationScores"],
    )
    subset_path = tmp_path / "subset.json"
    subset_path.write_text(json.dumps(subset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    configurations = {"reference": _candidate_configuration(1024), "subset": _candidate_configuration(2048)}
    contract = _write_contract(tmp_path / "contract.json", reference_path=reference, candidate_configurations=configurations)
    result = tool.run_selection(contract, {"reference": reference, "subset": subset_path}, tmp_path / "selection.json")
    evaluation = next(item for item in result["candidateEvaluations"] if item["id"] == "subset")
    assert evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "calibration membership does not match tuning feature membership" in evaluation["rejectionReasons"][0]


def test_normal_selection_rejects_calibration_recipe_mismatched_from_feature_membership(tmp_path: Path) -> None:
    tool = _tool_module()
    mismatched = _report_document(original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    for record in mismatched["calibrationScores"]:
        if record["isAugmentation"]:
            record["augmentationRecipeSha256"] = _digest("e")
    mismatched["normalOnlyEvidence"] = _normal_evidence(
        tool, mismatched["normalOnlyEvidence"]["featureInputs"], mismatched["calibrationScores"]
    )
    mismatched_path = tmp_path / "mismatched-calibration-recipe.json"
    mismatched_path.write_text(json.dumps(mismatched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract = _write_contract(
        tmp_path / "mismatched-calibration-recipe-contract.json",
        reference_path=mismatched_path,
        candidate_configurations={"reference": _candidate_configuration(1024)},
    )
    result = tool.run_selection(
        contract, {"reference": mismatched_path}, tmp_path / "mismatched-calibration-recipe-selection.json"
    )
    evaluation = result["candidateEvaluations"][0]
    assert evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "calibration membership does not match tuning feature membership" in evaluation["rejectionReasons"][0]


def test_normal_selection_rejects_out_of_range_score_and_duplicate_report_path(tmp_path: Path) -> None:
    tool = _tool_module()
    reference = _write_report(tmp_path / "reference.json", original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    candidate = _report_document(original_score=0.10, augmented_scores=[3.0, 0.11], max_prototypes=2048)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    configurations = {"reference": _candidate_configuration(1024), "candidate": _candidate_configuration(2048)}
    contract = _write_contract(tmp_path / "contract.json", reference_path=reference, candidate_configurations=configurations)
    result = tool.run_selection(contract, {"reference": reference, "candidate": candidate_path}, tmp_path / "selection.json")
    evaluation = next(item for item in result["candidateEvaluations"] if item["id"] == "candidate")
    assert evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "cosine-distance range" in evaluation["rejectionReasons"][0]
    with pytest.raises(tool.SelectionError, match="distinct report path"):
        tool.run_selection(contract, {"reference": reference, "candidate": reference}, tmp_path / "duplicate.json")


def test_normal_selection_rejects_changed_feature_extractor_identity(tmp_path: Path) -> None:
    tool = _tool_module()
    reference = _write_report(tmp_path / "reference.json", original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    changed = _report_document(original_score=0.10, augmented_scores=[0.12, 0.11], max_prototypes=2048)
    changed["featureExtractor"]["productionModuleSha256"] = _digest("8")
    changed["featureExtractorIdentitySha256"] = tool.canonical_json_sha256(changed["featureExtractor"])
    changed_path = tmp_path / "changed-extractor.json"
    changed_path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    configurations = {"reference": _candidate_configuration(1024), "candidate": _candidate_configuration(2048)}
    contract = _write_contract(tmp_path / "contract.json", reference_path=reference, candidate_configurations=configurations)
    result = tool.run_selection(contract, {"reference": reference, "candidate": changed_path}, tmp_path / "selection.json")
    evaluation = next(item for item in result["candidateEvaluations"] if item["id"] == "candidate")
    assert evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "feature extractor identity does not match the contract" in evaluation["rejectionReasons"][0]


def test_normal_selection_rejects_hidden_blind_payload_and_inconsistent_membership_counts(tmp_path: Path) -> None:
    tool = _tool_module()
    reference = _write_report(tmp_path / "reference.json", original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    configurations = {"reference": _candidate_configuration(1024), "candidate": _candidate_configuration(2048)}
    contract = _write_contract(tmp_path / "contract.json", reference_path=reference, candidate_configurations=configurations)

    hidden_blind = _report_document(original_score=0.10, augmented_scores=[0.12, 0.11], max_prototypes=2048)
    hidden_blind["blindReporting"]["unobservedPayload"] = {"caseId": "capsule/blind/001"}
    hidden_blind_path = tmp_path / "hidden-blind.json"
    hidden_blind_path.write_text(json.dumps(hidden_blind, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = tool.run_selection(contract, {"reference": reference, "candidate": hidden_blind_path}, tmp_path / "hidden-blind-selection.json")
    evaluation = next(item for item in result["candidateEvaluations"] if item["id"] == "candidate")
    assert evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "blindReporting has unknown fields" in evaluation["rejectionReasons"][0]

    count_mismatch = _report_document(original_score=0.10, augmented_scores=[0.12, 0.11], max_prototypes=2048)
    count_mismatch["augmentation"]["derivedRecordCount"] = 1
    count_mismatch_path = tmp_path / "count-mismatch.json"
    count_mismatch_path.write_text(json.dumps(count_mismatch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = tool.run_selection(contract, {"reference": reference, "candidate": count_mismatch_path}, tmp_path / "count-mismatch-selection.json")
    evaluation = next(item for item in result["candidateEvaluations"] if item["id"] == "candidate")
    assert evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "derived record count does not match feature membership" in evaluation["rejectionReasons"][0]

    category_mismatch = _report_document(original_score=0.10, augmented_scores=[0.12, 0.11], max_prototypes=2048)
    category_mismatch["categories"]["capsule"]["fitAugmentedCount"] = 0
    category_mismatch_path = tmp_path / "category-mismatch.json"
    category_mismatch_path.write_text(json.dumps(category_mismatch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = tool.run_selection(contract, {"reference": reference, "candidate": category_mismatch_path}, tmp_path / "category-mismatch-selection.json")
    evaluation = next(item for item in result["candidateEvaluations"] if item["id"] == "candidate")
    assert evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "fitAugmentedCount does not match feature membership" in evaluation["rejectionReasons"][0]


def test_normal_selection_rejects_missing_or_duplicate_parent_variant_coverage(tmp_path: Path) -> None:
    tool = _tool_module()
    configurations = {"reference": _candidate_configuration(1024)}

    missing = _report_document(original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    missing["augmentation"]["variantsPerParent"] = 3
    missing_path = tmp_path / "missing-variant.json"
    missing_path.write_text(json.dumps(missing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    missing_contract = _write_contract(
        tmp_path / "missing-variant-contract.json", reference_path=missing_path, candidate_configurations=configurations
    )
    missing_result = tool.run_selection(
        missing_contract, {"reference": missing_path}, tmp_path / "missing-variant-selection.json"
    )
    missing_evaluation = missing_result["candidateEvaluations"][0]
    assert missing_evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "coverage does not match the frozen variant set" in missing_evaluation["rejectionReasons"][0]

    duplicate = _report_document(original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    for record in duplicate["calibrationScores"]:
        if record["isAugmentation"] and record["variantId"] == 2:
            record["variantId"] = 1
    for record in duplicate["normalOnlyEvidence"]["featureInputs"]:
        if record["isAugmentation"] and record["variantId"] == 2:
            record["variantId"] = 1
    duplicate["normalOnlyEvidence"] = _normal_evidence(
        tool, duplicate["normalOnlyEvidence"]["featureInputs"], duplicate["calibrationScores"]
    )
    duplicate_path = tmp_path / "duplicate-variant.json"
    duplicate_path.write_text(json.dumps(duplicate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    duplicate_contract = _write_contract(
        tmp_path / "duplicate-variant-contract.json", reference_path=duplicate_path, candidate_configurations=configurations
    )
    duplicate_result = tool.run_selection(
        duplicate_contract, {"reference": duplicate_path}, tmp_path / "duplicate-variant-selection.json"
    )
    duplicate_evaluation = duplicate_result["candidateEvaluations"][0]
    assert duplicate_evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "duplicate variantId" in duplicate_evaluation["rejectionReasons"][0]

    cross_role = _report_document(original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    for records in (cross_role["calibrationScores"], cross_role["normalOnlyEvidence"]["featureInputs"]):
        for record in records:
            if record.get("caseId") == "capsule/tuning/001/camera-augmentation/01":
                record["parentCaseId"] = "capsule/fit/001"
                record["parentSourceSha256"] = _digest("7")
    cross_role["normalOnlyEvidence"] = _normal_evidence(
        tool, cross_role["normalOnlyEvidence"]["featureInputs"], cross_role["calibrationScores"]
    )
    cross_role_path = tmp_path / "cross-role-variant.json"
    cross_role_path.write_text(json.dumps(cross_role, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cross_role_contract = _write_contract(
        tmp_path / "cross-role-variant-contract.json", reference_path=cross_role_path, candidate_configurations=configurations
    )
    cross_role_result = tool.run_selection(
        cross_role_contract, {"reference": cross_role_path}, tmp_path / "cross-role-variant-selection.json"
    )
    cross_role_evaluation = cross_role_result["candidateEvaluations"][0]
    assert cross_role_evaluation["state"] == "REJECTED_INVALID_REPORT"
    assert "does not match its original parent membership" in cross_role_evaluation["rejectionReasons"][0]


def test_normal_selection_applies_per_variant_gate_even_when_aggregate_gates_pass(tmp_path: Path) -> None:
    tool = _tool_module()
    reference = _write_report(
        tmp_path / "reference.json", original_score=0.12, augmented_scores=[0.16, 0.14, 0.14, 0.14], max_prototypes=1024
    )
    candidate = _write_report(
        tmp_path / "candidate.json", original_score=0.10, augmented_scores=[0.16, 0.11, 0.11, 0.11], max_prototypes=2048
    )
    configurations = {"reference": _candidate_configuration(1024), "candidate": _candidate_configuration(2048)}
    contract = _write_contract(
        tmp_path / "contract.json",
        reference_path=reference,
        candidate_configurations=configurations,
        paired_cap=0.1,
        per_variant_paired_cap=0.05,
    )
    result = tool.run_selection(contract, {"reference": reference, "candidate": candidate}, tmp_path / "selection.json")
    evaluation = next(item for item in result["candidateEvaluations"] if item["id"] == "candidate")
    assert evaluation["state"] == "REJECTED_GATE"
    assert evaluation["rejectionReasons"] == [
        "capsule.variant1.maxPairedAugmentedScoreDeltaP95=0.06 exceeds 0.05",
        "capsule.variant1.maxPairedAugmentedScoreDeltaMax=0.06 exceeds 0.05",
    ]


def test_normal_selection_contract_requires_an_ordered_gate_for_every_variant(tmp_path: Path) -> None:
    tool = _tool_module()
    reference = _write_report(
        tmp_path / "reference.json", original_score=0.10, augmented_scores=[0.15, 0.14, 0.13, 0.12], max_prototypes=1024
    )
    contract = _write_contract(
        tmp_path / "contract.json", reference_path=reference, candidate_configurations={"reference": _candidate_configuration(1024)}
    )
    document = json.loads(contract.read_text(encoding="utf-8"))
    document["gate"]["perVariantPairedDeltaGates"].pop()
    document["contractSha256"] = tool.canonical_json_sha256({key: value for key, value in document.items() if key != "contractSha256"})
    contract.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(tool.SelectionError, match="exactly cover sorted augmentation variants"):
        tool.run_selection(contract, {"reference": reference}, tmp_path / "selection.json")


def test_normal_selection_reports_no_eligible_configuration_and_refuses_repo_paths(tmp_path: Path) -> None:
    tool = _tool_module()
    reference = _write_report(tmp_path / "reference.json", original_score=0.10, augmented_scores=[0.15, 0.14], max_prototypes=1024)
    candidate = _write_report(tmp_path / "candidate.json", original_score=0.10, augmented_scores=[0.12, 0.11], max_prototypes=2048)
    configurations = {"reference": _candidate_configuration(1024), "candidate": _candidate_configuration(2048)}
    contract = _write_contract(tmp_path / "contract.json", reference_path=reference, candidate_configurations=configurations, paired_cap=0.0)
    result = tool.run_selection(contract, {"reference": reference, "candidate": candidate}, tmp_path / "selection.json")
    assert result["decision"]["state"] == "NO_ELIGIBLE_CONFIGURATION"
    with pytest.raises(tool.SelectionError, match="selection output must stay outside"):
        tool.run_selection(contract, {"reference": reference, "candidate": candidate}, tool.REPOSITORY_ROOT / "selection.json")
    with pytest.raises(tool.SelectionError, match="selection contract must stay outside"):
        tool.run_selection(tool.REPOSITORY_ROOT / "contract.json", {"reference": reference, "candidate": candidate}, tmp_path / "repo-contract.json")
