from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from phone_dino import mvtec_synthetic_anomaly_stress_v2 as generator
from phone_dino import mvtec_synthetic_stress_v2 as stress


def _digest(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _write_image(path: Path, colour: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), colour)
    draw = ImageDraw.Draw(image)
    draw.line((2, 2, 42, 37), fill=((colour[0] + 31) % 255, (colour[1] + 47) % 255, (colour[2] + 73) % 255), width=2)
    image.save(path, format="PNG")
    return stress.sha256_file(path)


def _parents(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for category_index, category in enumerate(stress.SYNTHETIC_STRESS_V2_CATEGORIES):
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
            records.append({
                "caseId": f"successor/{category}/fit/{ordinal:02d}",
                "category": category,
                "partition": "FIT",
                "kind": "NOMINAL",
                "defect": "good",
                "sourceSha256": source,
                "sourceGroupId": f"CONTENT_SHA256:{source[7:]}",
                "imagePath": path,
            })
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


def _manifest() -> dict[str, Any]:
    return {
        "schemaVersion": "phone-dino.mvtec-ad-synthetic-only-stress-augmentation/2.0",
        "authoritative": False,
        "productionAuthorized": False,
        "syntheticOnly": True,
        "postV1Exploratory": True,
        "comparisonOrPromotionAllowed": False,
        "parentPartition": "FIT",
        "inputPolicy": stress.SYNTHETIC_STRESS_V2_INPUT_POLICY,
        "blindPolicy": stress.SYNTHETIC_STRESS_V2_BLIND_POLICY,
        "resultLabel": stress.SYNTHETIC_STRESS_V2_RESULT_LABEL,
        "parentSplitAlgorithm": stress.SYNTHETIC_STRESS_V2_PARENT_SPLIT_ALGORITHM,
        "parentSplitCountsPerCategory": {"PROTOTYPE": 6, "CALIBRATION": 2, "QUERY": 4},
        "variantsPerParent": 9,
        "augmentationManifestSha256": _digest("augmentation-declared"),
        "recipeFileSha256": _digest("recipe-file"),
    }


def _children(root: Path, parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split = stress.build_fixed_parent_split(parents)
    children: list[dict[str, Any]] = []
    variants = [
        (level, family)
        for level in stress.SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS
        for family in stress.SYNTHETIC_STRESS_V2_FAMILIES
    ]
    variant_id = {pair: index for index, pair in enumerate(variants, start=1)}
    for parent in split["SYNTHETIC_QUERY"]:
        with Image.open(parent["imagePath"]) as opened:
            source = opened.convert("RGB")
        for level_index, level in enumerate(stress.SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS, start=1):
            for family_index, family in enumerate(stress.SYNTHETIC_STRESS_V2_FAMILIES, start=1):
                identifier = variant_id[level, family]
                image = source.copy()
                draw = ImageDraw.Draw(image)
                position = 3 + family_index * 5
                size = 2 + level_index * 4
                draw.rectangle((position, position, position + size, position + size), fill=(0, 0, 0))
                path = root / parent["category"] / f"{parent['caseId'].split('/')[-1]}-{identifier}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, format="PNG")
                children.append({
                    "caseId": (
                        f"{parent['caseId']}/synthetic-only-stress-v2/"
                        f"{level.lower()}/{family.lower()}"
                    ),
                    "parentCaseId": parent["caseId"],
                    "parentSourceSha256": parent["sourceSha256"],
                    "sourceGroupId": parent["sourceGroupId"],
                    "category": parent["category"],
                    "parentPartition": "FIT",
                    "syntheticTestRole": "QUERY",
                    "syntheticLabel": "SYNTHETIC_STIMULUS",
                    "syntheticDefectFamily": family,
                    "renderIntensityLevel": level,
                    "variantId": identifier,
                    "relativePath": path.name,
                    "sourceSha256": stress.sha256_file(path),
                    "parameters": {},
                    "outputEncoding": {},
                    "imagePath": path,
                })
    return sorted(children, key=lambda record: str(record["caseId"]))


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
        "syntheticStressEvaluatorModuleSha256": stress.sha256_file(Path(stress.__file__)),
        "syntheticStressAugmentationModuleSha256": _digest("fixture-stress-augmentation"),
        "successorModuleSha256": _digest("fixture-successor"),
        "patchKnnModuleSha256": _digest("fixture-patch-knn"),
        "productionModuleSha256": _digest("fixture-production"),
        "enginesModuleSha256": _digest("fixture-engines"),
        "pythonVersion": "fixture-python",
        "numpyVersion": "fixture-numpy",
        "torchVersion": "fixture-torch",
        "torchvisionVersion": "fixture-torchvision",
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


def _install_safe_loader(
    monkeypatch: pytest.MonkeyPatch,
    parents: list[dict[str, Any]],
    calls: list[set[str]],
) -> None:
    def safe_loader(*_args: object, **kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        requested = set(kwargs["partitions"])
        calls.append(requested)
        assert requested == {"FIT"}
        return _envelope(), _digest("envelope-file"), [dict(record) for record in parents]

    monkeypatch.setattr(stress.successor, "load_successor_safe_normal_inputs", safe_loader)


def _response_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build one complete report through fixtures for JSON-contract tests."""

    parents = _parents(tmp_path / "source")
    children = _children(tmp_path / "package", parents)
    calls: list[set[str]] = []
    _install_safe_loader(monkeypatch, parents, calls)

    def package_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        return _manifest(), _digest("augmentation-file"), [dict(record) for record in children]

    monkeypatch.setattr(stress.augmentation, "load_validated_synthetic_stress_v2", package_loader)
    report = stress.run_synthetic_stress_v2(
        tmp_path / "chain" / "holdout.json",
        tmp_path / "chain" / "contract.json",
        tmp_path / "chain" / "plan.json",
        tmp_path / "chain" / "envelope.json",
        tmp_path / "package" / "augmentation_manifest.json",
        tmp_path / "generated" / "response.json",
        source_root=tmp_path / "source",
        recipe_path=tmp_path / "recipe.json",
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_FakeEmbedder,
        identity_factory=_identity_factory,
    )
    assert calls == [{"FIT"}]
    return report


def test_fixed_parent_split_is_stable_and_disjoint(tmp_path: Path) -> None:
    parents = _parents(tmp_path / "source")
    first = stress.build_fixed_parent_split(parents)
    second = stress.build_fixed_parent_split(list(reversed(parents)))
    assert {
        role: [record["caseId"] for record in values]
        for role, values in first.items()
    } == {
        role: [record["caseId"] for record in values]
        for role, values in second.items()
    }
    assert {role: len(values) for role, values in first.items()} == {
        "SYNTHETIC_PROTOTYPE": 18,
        "SYNTHETIC_CALIBRATION": 6,
        "SYNTHETIC_QUERY": 12,
    }
    groups = {role: {record["sourceGroupId"] for record in values} for role, values in first.items()}
    assert not groups["SYNTHETIC_PROTOTYPE"] & groups["SYNTHETIC_CALIBRATION"]
    assert not groups["SYNTHETIC_PROTOTYPE"] & groups["SYNTHETIC_QUERY"]
    assert not groups["SYNTHETIC_CALIBRATION"] & groups["SYNTHETIC_QUERY"]


def test_raw_calibration_uses_maximum_per_category() -> None:
    rows = [
        {"category": category, "score": value}
        for category, values in {
            "capsule": (0.2, 0.5),
            "metal_nut": (0.1, 0.4),
            "tile": (0.3, 0.7),
        }.items()
        for value in values
    ]
    assert stress.calibrate_raw_thresholds(rows) == {"capsule": 0.5, "metal_nut": 0.4, "tile": 0.7}


def test_v2_response_run_is_fit_only_threshold_first_and_response_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = _parents(tmp_path / "source")
    children = _children(tmp_path / "package", parents)
    calls: list[set[str]] = []
    events: list[str] = []
    _install_safe_loader(monkeypatch, parents, calls)
    original_calibrate = stress.calibrate_raw_thresholds

    def calibrate(scores: list[dict[str, Any]]) -> dict[str, float]:
        events.append("threshold")
        return original_calibrate(scores)

    def package_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        assert events == ["threshold"]
        events.append("package")
        return _manifest(), _digest("augmentation-file"), [dict(record) for record in children]

    monkeypatch.setattr(stress, "calibrate_raw_thresholds", calibrate)
    monkeypatch.setattr(stress.augmentation, "load_validated_synthetic_stress_v2", package_loader)
    output = tmp_path / "report" / "synthetic-stress-v2.json"
    report = stress.run_synthetic_stress_v2(
        tmp_path / "chain" / "holdout.json",
        tmp_path / "chain" / "contract.json",
        tmp_path / "chain" / "plan.json",
        tmp_path / "chain" / "envelope.json",
        tmp_path / "package" / "augmentation_manifest.json",
        output,
        source_root=tmp_path / "source",
        recipe_path=tmp_path / "recipe.json",
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_FakeEmbedder,
        identity_factory=_identity_factory,
    )
    assert calls == [{"FIT"}]
    assert events == ["threshold", "package"]
    assert output.is_file()
    assert report["syntheticOnly"] is True
    assert report["postV1Exploratory"] is True
    assert report["comparisonOrPromotionAllowed"] is False
    assert report["metricScope"] == "SYNTHETIC_STIMULUS_RESPONSE_ONLY"
    assert report["realAnomalyPerformance"] == "NOT_ESTIMATED"
    assert report["realPrecisionRecall"] == "NOT_ESTIMATED"
    assert report["evidenceClass"] == "SYNTHETIC_ENGINEERING_ONLY"
    assert report["schemaVersion"] == stress.SYNTHETIC_STRESS_V2_R2_REPORT_SCHEMA
    assert report["testConfiguration"]["rawCalibrationThresholdEstablishedBeforePackageLoad"] is True
    assert report["testConfiguration"]["rawCalibrationThresholdEstablishedBeforePackageScoring"] is True
    assert report["aggregate"]["responseCounts"] == {
        "rawQueryCount": 12,
        "rawQueryAboveThresholdCount": report["aggregate"]["responseCounts"]["rawQueryAboveThresholdCount"],
        "syntheticStimulusCount": 108,
        "syntheticStimulusAboveThresholdCount": report["aggregate"]["responseCounts"]["syntheticStimulusAboveThresholdCount"],
    }
    assert report["aggregate"]["pairedScoreDeltaSummary"]["pairCount"] == 108
    assert len(report["rawQueryScores"]) == 12
    assert len(report["stimulusScores"]) == 108
    assert {record["renderIntensityLevel"] for record in report["stimulusScores"]} == set(stress.SYNTHETIC_STRESS_V2_RENDER_INTENSITY_LEVELS)
    assert {record["syntheticDefectFamily"] for record in report["stimulusScores"]} == set(stress.SYNTHETIC_STRESS_V2_FAMILIES)
    assert all(values["responseCounts"]["syntheticStimulusCount"] == 36 for values in report["categories"].values())
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in ("syntheticTP", "syntheticFP", "syntheticFN", "syntheticTN", "syntheticPrecision", "syntheticRecall", "syntheticF1", "AUROC"):
        assert forbidden not in serialized


def test_r2_json_contract_rejects_historical_r1_fields_and_unsafe_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _response_report(tmp_path, monkeypatch)
    assert stress.validate_response_only_report_v2_r2(report) is report
    assert stress.SYNTHETIC_STRESS_V2_R1_REPORT_SCHEMA in stress.SYNTHETIC_STRESS_V2_HISTORICAL_REPORT_SCHEMAS
    assert "R1_2_0" in stress.SYNTHETIC_STRESS_V2_R2_MIGRATION
    encoded = json.dumps(report, ensure_ascii=True, sort_keys=True, allow_nan=False).encode("utf-8")
    assert stress.parse_response_only_report_v2_r2_json(encoded) == report

    r1_missing_limits = deepcopy(report)
    del r1_missing_limits["realPrecisionRecall"]
    del r1_missing_limits["evidenceClass"]
    r1_missing_limits["syntheticStressReportSha256"] = stress._document_digest(r1_missing_limits)
    with pytest.raises(stress.SyntheticStressV2Error, match="missing.*evidenceClass.*realPrecisionRecall"):
        stress.validate_response_only_report_v2_r2(r1_missing_limits)

    r1_schema = deepcopy(report)
    r1_schema["schemaVersion"] = stress.SYNTHETIC_STRESS_V2_R1_REPORT_SCHEMA
    r1_schema["syntheticStressReportSha256"] = stress._document_digest(r1_schema)
    with pytest.raises(stress.SyntheticStressV2Error, match="historical r1"):
        stress.validate_response_only_report_v2_r2(r1_schema)

    with pytest.raises(stress.SyntheticStressV2Error, match="unable to parse"):
        stress.parse_response_only_report_v2_r2_json(b'{"schemaVersion":')
    with pytest.raises(stress.SyntheticStressV2Error, match="duplicate JSON key"):
        stress.parse_response_only_report_v2_r2_json(b'{"key":1,"key":2}')
    with pytest.raises(stress.SyntheticStressV2Error, match="non-finite JSON value"):
        stress.parse_response_only_report_v2_r2_json(b'{"value":NaN}')
    with pytest.raises(stress.SyntheticStressV2Error, match="oversized JSON integer"):
        stress.parse_response_only_report_v2_r2_json(b'{"oversized":9007199254740992}')
    with pytest.raises(stress.SyntheticStressV2Error, match="representable finite"):
        stress._require_finite(10**10000, name="fixture huge finite integer")
    with pytest.raises(stress.SyntheticStressV2Error, match="finite JSON"):
        stress._serialize_response_report({"score": float("nan")})


def test_r2_json_contract_rejects_self_digested_semantic_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A renewed self-digest cannot turn contradictory details into evidence."""

    report = _response_report(tmp_path, monkeypatch)

    def resign(value: dict[str, Any]) -> dict[str, Any]:
        value["syntheticStressReportSha256"] = stress._document_digest(value)
        return value

    wrong_threshold = deepcopy(report)
    wrong_threshold["thresholds"]["capsule"] += 5e-13
    with pytest.raises(stress.SyntheticStressV2Error, match="threshold capsule.*score details"):
        stress.validate_response_only_report_v2_r2(resign(wrong_threshold))

    wrong_flag = deepcopy(report)
    wrong_flag["rawQueryScores"][0]["aboveRawCalibrationThreshold"] = not wrong_flag["rawQueryScores"][0][
        "aboveRawCalibrationThreshold"
    ]
    with pytest.raises(stress.SyntheticStressV2Error, match="raw query threshold flag"):
        stress.validate_response_only_report_v2_r2(resign(wrong_flag))

    wrong_summary = deepcopy(report)
    wrong_summary["categories"]["capsule"]["pairedScoreDeltaSummary"]["meanChildMinusParentScore"] += 0.125
    with pytest.raises(stress.SyntheticStressV2Error, match="meanChildMinusParentScore.*score details"):
        stress.validate_response_only_report_v2_r2(resign(wrong_summary))

    duplicate_parent = deepcopy(report)
    candidates = [
        index
        for index, record in enumerate(duplicate_parent["parentSplit"])
        if record["role"] == "SYNTHETIC_QUERY" and record["category"] == "capsule"
    ]
    duplicate_parent["parentSplit"][candidates[-1]] = deepcopy(duplicate_parent["parentSplit"][candidates[0]])
    duplicate_parent["parentSplitIdentitySha256"] = stress.successor.canonical_json_sha256(duplicate_parent["parentSplit"])
    with pytest.raises(stress.SyntheticStressV2Error, match="duplicate caseId"):
        stress.validate_response_only_report_v2_r2(resign(duplicate_parent))

    arbitrary_child = deepcopy(report)
    arbitrary_child["stimulusScores"][0]["caseId"] = "not-a-v2-child"
    with pytest.raises(stress.SyntheticStressV2Error, match="deterministic V2 child identity"):
        stress.validate_response_only_report_v2_r2(resign(arbitrary_child))

    out_of_range_component = deepcopy(report)
    out_of_range_component["stimulusScores"][0]["maxPatchDistance"] = 3.0
    with pytest.raises(stress.SyntheticStressV2Error, match=r"components must be in \[0, 2\]"):
        stress.validate_response_only_report_v2_r2(resign(out_of_range_component))

    impossible_component_order = deepcopy(report)
    impossible_component_order["stimulusScores"][0]["score"] = 0.5
    impossible_component_order["stimulusScores"][0]["meanNearestPatchDistance"] = 1.0
    impossible_component_order["stimulusScores"][0]["maxPatchDistance"] = 1.5
    with pytest.raises(stress.SyntheticStressV2Error, match="meanNearestPatchDistance <= score <= maxPatchDistance"):
        stress.validate_response_only_report_v2_r2(resign(impossible_component_order))

    mismatched_execution = deepcopy(report)
    mismatched_execution["execution"]["syntheticStressEvaluatorModuleSha256"] = _digest("other-evaluator")
    with pytest.raises(stress.SyntheticStressV2Error, match="does not match the feature extractor"):
        stress.validate_response_only_report_v2_r2(resign(mismatched_execution))

    injected_metric = deepcopy(report)
    injected_metric["featureExtractor"]["AUROC"] = 1.0
    injected_metric["featureExtractorIdentitySha256"] = stress.successor.canonical_json_sha256(
        injected_metric["featureExtractor"]
    )
    with pytest.raises(stress.SyntheticStressV2Error, match="forbidden classification field: AUROC"):
        stress.validate_response_only_report_v2_r2(resign(injected_metric))

    unsealed_report = deepcopy(report)
    del unsealed_report["featureExtractor"]["sealedDinoSnapshot"]
    unsealed_report["featureExtractorIdentitySha256"] = stress.successor.canonical_json_sha256(
        unsealed_report["featureExtractor"]
    )
    with pytest.raises(stress.SyntheticStressV2Error, match="missing sealedDinoSnapshot"):
        stress.validate_response_only_report_v2_r2(resign(unsealed_report))

    null_sealed_snapshot = deepcopy(report)
    null_sealed_snapshot["featureExtractor"]["sealedDinoSnapshot"] = None
    null_sealed_snapshot["featureExtractorIdentitySha256"] = stress.successor.canonical_json_sha256(
        null_sealed_snapshot["featureExtractor"]
    )
    with pytest.raises(stress.SyntheticStressV2Error, match="sealed DINO snapshot provenance must be an object"):
        stress.validate_response_only_report_v2_r2(resign(null_sealed_snapshot))

    sealed_snapshot_report = deepcopy(report)
    sealed_snapshot_report["featureExtractor"]["sealedDinoSnapshot"] = {
        "schemaVersion": stress.sealed_snapshot.SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_SCHEMA,
        "snapshotSchemaVersion": stress.sealed_snapshot.SEALED_DINO_SNAPSHOT_SCHEMA,
        "repositoryDigestAlgorithm": stress.sealed_snapshot.SEALED_DINO_REPOSITORY_DIGEST_ALGORITHM,
        "weightsDigestAlgorithm": stress.sealed_snapshot.SEALED_DINO_WEIGHTS_DIGEST_ALGORITHM,
        "snapshotManifestSha256": _digest("sealed-manifest"),
        "snapshotRepositorySha256": _digest("sealed-repository-different-algorithm"),
        "snapshotWeightsSha256": _digest("sealed-weights-independent-pin"),
    }
    sealed_snapshot_report["featureExtractorIdentitySha256"] = stress.successor.canonical_json_sha256(
        sealed_snapshot_report["featureExtractor"]
    )
    assert stress.validate_response_only_report_v2_r2(resign(sealed_snapshot_report)) is sealed_snapshot_report

    wrong_sealed_snapshot = deepcopy(sealed_snapshot_report)
    wrong_sealed_snapshot["featureExtractor"]["sealedDinoSnapshot"]["weightsDigestAlgorithm"] = "UNPINNED"
    wrong_sealed_snapshot["featureExtractorIdentitySha256"] = stress.successor.canonical_json_sha256(
        wrong_sealed_snapshot["featureExtractor"]
    )
    with pytest.raises(stress.SyntheticStressV2Error, match="weights digest algorithm is unsupported"):
        stress.validate_response_only_report_v2_r2(resign(wrong_sealed_snapshot))


def test_response_report_writer_rejects_path_substitution_and_completes_partial_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _response_report(tmp_path, monkeypatch)
    target = tmp_path / "writer-race" / "response.json"
    original_signature = stress._stat_signature

    def substituted_signature(path: Path) -> tuple[int, int, int, int]:
        signature = original_signature(path)
        if path == target:
            return signature[0], signature[1] + 1, signature[2], signature[3]
        return signature

    monkeypatch.setattr(stress, "_stat_signature", substituted_signature)
    with pytest.raises(stress.SyntheticStressV2Error, match="changed while it was written"):
        stress.write_response_only_report(target, report, repository_root=stress.REPOSITORY_ROOT)
    assert target.is_file()

    parent_race_target = tmp_path / "writer-parent-race" / "response.json"
    original_directory_identity = stress._directory_identity
    parent_identity_reads = {"count": 0}

    def substituted_directory_identity(path: Path) -> tuple[int, int, int]:
        signature = original_directory_identity(path)
        if path == parent_race_target.parent:
            parent_identity_reads["count"] += 1
            if parent_identity_reads["count"] == 2:
                return signature[0], signature[1] + 1, signature[2]
        return signature

    monkeypatch.setattr(stress, "_directory_identity", substituted_directory_identity)
    with pytest.raises(stress.SyntheticStressV2Error, match="parent chain changed while it was written"):
        stress.write_response_only_report(parent_race_target, report, repository_root=stress.REPOSITORY_ROOT)
    assert parent_race_target.is_file()

    partial_target = tmp_path / "writer-partial" / "response.json"
    original_write = stress.os.write
    writes: list[int] = []

    def partial_write(fd: int, data: bytes) -> int:
        writes.append(len(data))
        return original_write(fd, data[: max(1, len(data) // 2)])

    monkeypatch.setattr(stress.os, "write", partial_write)
    monkeypatch.setattr(stress, "_directory_identity", original_directory_identity)
    stress.write_response_only_report(partial_target, report, repository_root=stress.REPOSITORY_ROOT)
    assert len(writes) > 1
    assert stress.parse_response_only_report_v2_r2_json(partial_target.read_bytes()) == report


def test_v2_evaluator_consumes_a_real_validated_generator_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = _parents(tmp_path / "source")
    calls: list[set[str]] = []
    _install_safe_loader(monkeypatch, parents, calls)
    chain_paths = [
        tmp_path / "chain" / "holdout.json",
        tmp_path / "chain" / "contract.json",
        tmp_path / "chain" / "plan.json",
        tmp_path / "chain" / "envelope.json",
    ]
    for path in chain_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    package_dir = tmp_path / "real-package"
    generator.generate_synthetic_anomaly_stress_v2(
        chain_paths[0],
        chain_paths[1],
        chain_paths[2],
        chain_paths[3],
        source_root=tmp_path / "source",
        recipe_path=generator.REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
        output_dir=package_dir,
    )
    report = stress.run_synthetic_stress_v2(
        chain_paths[0],
        chain_paths[1],
        chain_paths[2],
        chain_paths[3],
        package_dir / "augmentation_manifest.json",
        tmp_path / "report" / "response.json",
        source_root=tmp_path / "source",
        recipe_path=generator.REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
        model_repo=tmp_path / "model",
        model_weights=tmp_path / "weights.pth",
        embedder_factory=_FakeEmbedder,
        identity_factory=_identity_factory,
    )
    assert calls == [{"FIT"}, {"FIT"}, {"FIT"}]
    assert report["augmentationSchemaVersion"] == generator.SYNTHETIC_ANOMALY_STRESS_V2_SCHEMA
    assert report["aggregate"]["responseCounts"]["syntheticStimulusCount"] == 108
    manifest_path = package_dir / "augmentation_manifest.json"
    assert stress.validate_response_only_report_v2_r2_package_metadata(
        report,
        augmentation_manifest_path=manifest_path,
        recipe_path=generator.REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
    ) is report

    wrong_child_sha = deepcopy(report)
    wrong_child_sha["stimulusScores"][0]["sourceSha256"] = _digest("wrong-child-projection")
    wrong_child_sha["syntheticStressReportSha256"] = stress._document_digest(wrong_child_sha)
    assert stress.validate_response_only_report_v2_r2(wrong_child_sha) is wrong_child_sha
    with pytest.raises(stress.SyntheticStressV2Error, match="projection does not match"):
        stress.validate_response_only_report_v2_r2_package_metadata(
            wrong_child_sha,
            augmentation_manifest_path=manifest_path,
            recipe_path=generator.REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
        )

    malformed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    malformed_manifest["records"][0]["syntheticLabel"] = "NOT_SYNTHETIC_STIMULUS"
    malformed_manifest["augmentationManifestSha256"] = generator._document_digest(
        malformed_manifest, "augmentationManifestSha256"
    )
    manifest_path.write_text(json.dumps(malformed_manifest, sort_keys=True), encoding="utf-8")
    malformed_report = deepcopy(report)
    malformed_report["augmentationManifestFileSha256"] = stress.sha256_file(manifest_path)
    malformed_report["augmentationManifestDeclaredSha256"] = malformed_manifest["augmentationManifestSha256"]
    malformed_report["syntheticStressReportSha256"] = stress._document_digest(malformed_report)
    assert stress.validate_response_only_report_v2_r2(malformed_report) is malformed_report
    with pytest.raises(stress.SyntheticStressV2Error, match="child parent binding"):
        stress.validate_response_only_report_v2_r2_package_metadata(
            malformed_report,
            augmentation_manifest_path=manifest_path,
            recipe_path=generator.REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
        )


def test_v2_package_requires_every_query_family_level_combination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = _parents(tmp_path / "source")
    children = _children(tmp_path / "package", parents)

    def package_loader(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        return _manifest(), _digest("augmentation-file"), children[:-1]

    monkeypatch.setattr(stress.augmentation, "load_validated_synthetic_stress_v2", package_loader)
    query_parents = stress.build_fixed_parent_split(parents)["SYNTHETIC_QUERY"]
    with pytest.raises(stress.SyntheticStressV2Error, match="cover every query parent, family, and render level"):
        stress.load_and_validate_v2_package(
            tmp_path / "package" / "augmentation_manifest.json",
            tmp_path / "chain" / "holdout.json",
            tmp_path / "chain" / "contract.json",
            tmp_path / "chain" / "plan.json",
            tmp_path / "chain" / "envelope.json",
            source_root=tmp_path / "source",
            recipe_path=tmp_path / "recipe.json",
            query_parents=query_parents,
        )


def test_existing_output_rejects_before_any_fit_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "report.json"
    output.write_text("already exists", encoding="utf-8")
    calls: list[str] = []

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        calls.append("FIT")
        raise AssertionError("existing output must fail before the FIT loader")

    monkeypatch.setattr(stress.successor, "load_successor_safe_normal_inputs", forbidden_loader)
    with pytest.raises(stress.SyntheticStressV2Error, match="already exists"):
        stress.run_synthetic_stress_v2(
            tmp_path / "holdout.json",
            tmp_path / "contract.json",
            tmp_path / "plan.json",
            tmp_path / "envelope.json",
            tmp_path / "augmentation_manifest.json",
            output,
            source_root=tmp_path,
            recipe_path=tmp_path / "recipe.json",
            model_repo=tmp_path / "model",
            model_weights=tmp_path / "weights.pth",
            identity_factory=_identity_factory,
        )
    assert calls == []


def test_v2_direct_api_rejects_unsealed_feature_identity_before_package_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parents = _parents(tmp_path / "source")
    calls: list[set[str]] = []
    child_loads: list[str] = []
    _install_safe_loader(monkeypatch, parents, calls)

    def unsealed_identity(**kwargs: object) -> dict[str, Any]:
        identity = _identity_factory(**kwargs)
        del identity["sealedDinoSnapshot"]
        return identity

    monkeypatch.setattr(
        stress.augmentation,
        "load_validated_synthetic_stress_v2",
        lambda *_args, **_kwargs: child_loads.append("package")
        or (_ for _ in ()).throw(AssertionError("package child loader must not run")),
    )
    with pytest.raises(stress.SyntheticStressV2Error, match="missing sealedDinoSnapshot"):
        stress.run_synthetic_stress_v2(
            tmp_path / "chain" / "holdout.json",
            tmp_path / "chain" / "contract.json",
            tmp_path / "chain" / "plan.json",
            tmp_path / "chain" / "envelope.json",
            tmp_path / "package" / "augmentation_manifest.json",
            tmp_path / "report.json",
            source_root=tmp_path / "source",
            recipe_path=tmp_path / "recipe.json",
            model_repo=tmp_path / "model",
            model_weights=tmp_path / "weights.pth",
            embedder_factory=_FakeEmbedder,
            identity_factory=unsealed_identity,
        )
    assert calls == []
    assert child_loads == []


def test_no_v1_report_api_or_classification_summary_fields() -> None:
    signature = inspect.signature(stress.run_synthetic_stress_v2)
    assert "v1" not in " ".join(signature.parameters).lower()
    source = inspect.getsource(stress)
    assert "mvtec_synthetic_anomaly_test" not in source
    assert "synthetic_confusion_metrics" not in source
    assert stress.SYNTHETIC_STRESS_V2_REAL_PRECISION_RECALL == "NOT_ESTIMATED"
    assert stress.SYNTHETIC_STRESS_V2_EVIDENCE_CLASS == "SYNTHETIC_ENGINEERING_ONLY"
