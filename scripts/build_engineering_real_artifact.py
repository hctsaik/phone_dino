from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps, ImageStat

from phone_dino.compiler import compile_artifact
from phone_dino.contracts import GoldenDimensionBoardCandidate
from phone_dino.production import _detect_dark_body_contour
from phone_dino.security import digest_directory, digest_file


CALIBRATION_BOARD_PROFILES: dict[str, dict[str, object]] = {
    "LEGACY_5X7_20MM_V1": {
        "profileId": "LEGACY_5X7_20MM_V1",
        "squaresX": 5, "squaresY": 7,
        "squareLengthMm": 20.0, "markerLengthMm": 15.0,
        "dictionary": "DICT_4X4_50",
    },
    "COMPACT_130X90_V1": {
        "profileId": "COMPACT_130X90_V1",
        "squaresX": 7, "squaresY": 5,
        "squareLengthMm": 10.0, "markerLengthMm": 7.0,
        "markerIds": list(range(100, 117)),
        "dictionary": "DICT_5X5_1000",
    },
    "CREDIT_CARD_85P6X54_V1": {
        "profileId": "CREDIT_CARD_85P6X54_V1",
        "squaresX": 5, "squaresY": 3,
        "squareLengthMm": 8.0, "markerLengthMm": 5.6,
        "markerIds": list(range(100, 107)),
        "dictionary": "DICT_5X5_1000",
    },
}


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _roi_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _digest_bytes(encoded)


def _with_digest(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "digest": _roi_digest(payload)}


def _canonical_reference(
    golden_path: Path,
    output_path: Path,
    roi: tuple[float, float, float, float],
    canonical_size: int,
    model_crop_margin: int,
) -> tuple[dict[str, int], tuple[int, int]]:
    with Image.open(golden_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        x, y, roi_width, roi_height = roi
        box = (
            round(x * width), round(y * height),
            round((x + roi_width) * width), round((y + roi_height) * height),
        )
        crop = image.crop(box)
        usable = canonical_size - 2 * model_crop_margin
        scale = min(usable / crop.width, usable / crop.height)
        resized = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
        # Use the Golden crop's border median instead of black padding. This
        # avoids an artificial high-contrast frame becoming an alignment cue.
        border = resized.crop((0, 0, resized.width, max(1, min(16, resized.height))))
        background = tuple(round(value) for value in ImageStat.Stat(border).median)
        canvas = Image.new("RGB", (canonical_size, canonical_size), background)
        left = (canonical_size - resized.width) // 2
        top = (canonical_size - resized.height) // 2
        canvas.paste(resized, (left, top))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, "PNG", optimize=True)
    return {
        "x": left,
        "y": top,
        "width": resized.width,
        "height": resized.height,
    }, (width, height)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clearly non-production artifact that runs the real local DINOv2 model",
    )
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--golden-id", required=True)
    parser.add_argument("--recipe-id", required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--board-id", required=True)
    parser.add_argument("--golden-set-version", required=True)
    parser.add_argument("--normalization-version", required=True)
    parser.add_argument("--analyzer-model-version", required=True)
    parser.add_argument("--decision-policy-version", required=True)
    parser.add_argument("--board-installation-version", required=True)
    parser.add_argument("--model-repository", required=True, type=Path)
    parser.add_argument("--model-weights", required=True, type=Path)
    parser.add_argument("--segmenter-repository", required=True, type=Path)
    parser.add_argument("--segmenter-weights", required=True, type=Path)
    parser.add_argument("--roi", required=True, nargs=4, type=float, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--canonical-size", type=int, default=896)
    parser.add_argument("--model-crop-margin", type=int, default=56)
    parser.add_argument(
        "--calibration-board-profile",
        choices=tuple(CALIBRATION_BOARD_PROFILES),
        default="LEGACY_5X7_20MM_V1",
        help="Immutable ChArUco geometry expected in Current captures",
    )
    parser.add_argument(
        "--calibration-board-manifest",
        type=Path,
        help="Immutable PhoneCV board manifest used for Golden normalization",
    )
    parser.add_argument("--artifact-name", default="engineering-real-dino-artifact.json")
    args = parser.parse_args()

    golden_path = args.golden.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "canonical-target-reference.png"
    content, source_dimensions = _canonical_reference(
        golden_path, reference_path, tuple(args.roi), args.canonical_size, args.model_crop_margin,
    )
    reference_bytes = reference_path.read_bytes()
    golden_bytes = golden_path.read_bytes()
    import cv2
    golden_bgr = cv2.imread(str(golden_path), cv2.IMREAD_COLOR)
    detected_body = None if golden_bgr is None else _detect_dark_body_contour(golden_bgr, 0.61)
    if detected_body is None:
        raise SystemExit("The Golden target body contour could not be established")
    source_width, source_height = source_dimensions
    crop_left = round(args.roi[0] * source_width)
    crop_top = round(args.roi[1] * source_height)
    crop_right = round((args.roi[0] + args.roi[2]) * source_width)
    crop_bottom = round((args.roi[1] + args.roi[3]) * source_height)
    fit_scale = min(content["width"] / (crop_right - crop_left), content["height"] / (crop_bottom - crop_top))
    contour_anchor = {
        "x": max(0, round(content["x"] + (detected_body.x - crop_left) * fit_scale)),
        "y": max(0, round(content["y"] + (detected_body.y - crop_top) * fit_scale)),
        "width": max(1, round(detected_body.width * fit_scale)),
        "height": max(1, round(detected_body.height * fit_scale)),
    }

    # Keep two narrow immutable target strips for fitting and held-out proof;
    # the wide center stays exclusively available for visual anomaly evidence.
    protected_width = max(50, round(content["width"] * 0.14))
    gap = max(8, round(content["width"] * 0.025))
    alignment = {"x": content["x"], "y": content["y"], "width": protected_width, "height": content["height"]}
    held_out = {
        "x": content["x"] + content["width"] - protected_width,
        "y": content["y"], "width": protected_width, "height": content["height"],
    }
    inspection = {
        "x": alignment["x"] + alignment["width"] + gap,
        "y": content["y"],
        "width": held_out["x"] - gap - (alignment["x"] + alignment["width"] + gap),
        "height": content["height"],
    }
    if inspection["width"] < 64:
        raise SystemExit("Confirmed ROI is too narrow after reserving target-alignment evidence")

    roi_payload: dict[str, object] = {
        "version": "roi-1.0",
        "canonicalWidth": args.canonical_size,
        "canonicalHeight": args.canonical_size,
        "polygon": [
            {"x": float(inspection["x"]), "y": float(inspection["y"])},
            {"x": float(inspection["x"] + inspection["width"]), "y": float(inspection["y"])},
            {"x": float(inspection["x"] + inspection["width"]), "y": float(inspection["y"] + inspection["height"])},
            {"x": float(inspection["x"]), "y": float(inspection["y"] + inspection["height"])},
        ],
        "inspectionRegions": [inspection],
    }
    roi_contract = {**roi_payload, "digest": _roi_digest(roi_payload)}
    scorer_input_contract = _with_digest({
        "schemaVersion": "1.1",
        "policy": "INSPECTION_ROI_PAIRED_INTERIOR_TILES_NEUTRAL_OUTSIDE",
        "coordinateSpace": "TARGET_CANONICAL_IMAGE",
        "inspectionRoiContractDigest": roi_contract["digest"],
        "tileOrder": "TOP_TO_BOTTOM_LEFT_TO_RIGHT",
        "neutralRgb": [127, 127, 127],
    })
    role_bindings = {
        "alignmentTemplate": _with_digest({
            "role": "ALIGNMENT_TEMPLATE", "id": f"{args.golden_id}-ALIGNMENT",
            "version": "1", "sourceDigest": _digest_bytes(golden_bytes),
        }),
        "targetReference": _with_digest({
            "role": "TARGET_REFERENCE", "id": f"{args.golden_id}-TARGET",
            "version": "1", "sourceDigest": _digest_bytes(reference_bytes),
        }),
        "normalReferenceSet": _with_digest({
            "role": "NORMAL_REFERENCE_SET", "id": f"{args.recipe_id}-NORMALS",
            "version": "1", "sourceDigest": args.golden_set_version,
        }),
        "displayReference": _with_digest({
            "role": "DISPLAY_REFERENCE", "id": f"{args.golden_id}-DISPLAY",
            "version": "1", "sourceDigest": _digest_bytes(golden_bytes),
        }),
    }
    recipe_analysis_profile = _with_digest({
        "schemaVersion": "1.0",
        "id": f"{args.recipe_id}-ANALYSIS-PROFILE",
        "version": "2",
        **role_bindings,
        "inspectionRoiContractDigest": roi_contract["digest"],
        "scorerInputContractDigest": scorer_input_contract["digest"],
    })
    spec = {
        "schemaVersion": "1.8",
        "recipeId": args.recipe_id,
        "machineId": args.machine_id,
        "boardId": args.board_id,
        "goldenSetVersion": args.golden_set_version,
        "normalizationPipelineVersion": args.normalization_version,
        "analyzerModelVersion": args.analyzer_model_version,
        "decisionPolicyVersion": args.decision_policy_version,
        "modelRepositoryVersion": digest_directory(args.model_repository.resolve()),
        "boardInstallationVersion": args.board_installation_version,
        "modelWeightsSha256": digest_file(args.model_weights.resolve()),
        "board": {
            **CALIBRATION_BOARD_PROFILES[args.calibration_board_profile],
            "canonicalWidth": args.canonical_size, "canonicalHeight": args.canonical_size,
        },
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 15.0, "maxOverExposureRatio": 0.35},
        "targetAlignment": {
            "method": "TARGET_AFFINE",
            "referenceImageBase64": base64.b64encode(reference_bytes).decode("ascii"),
            "referenceImageSha256": _digest_bytes(reference_bytes),
            "canonicalWidth": args.canonical_size, "canonicalHeight": args.canonical_size,
            "alignmentRegions": [alignment], "heldOutRegions": [held_out], "inspectionRegions": [inspection],
            "contourAnchorRegion": contour_anchor,
            "minMatches": 6, "minInliers": 4, "minInlierRatio": 0.25, "minCoverageRatio": 0.01,
            "maxReprojectionErrorPx": 8.0, "minScale": 0.1, "maxScale": 4.0,
            "maxRotationDegrees": 35.0, "maxShear": 0.12, "maxTranslationPx": 12000.0,
            "maxSecondaryInlierRatio": 0.6, "minHeldOutMatches": 4,
            "maxHeldOutReprojectionErrorPx": 12.0,
        },
        "goldenSources": [{
            "id": args.golden_id, "path": golden_path.name, "sourceSha256": _digest_bytes(golden_bytes),
        }],
        "spatialDifferencePolicy": {
            "anomalyDistanceThreshold": 0.35, "minRegionAreaRatio": 0.003, "maxRegions": 12,
        },
        "inspectionRoi": roi_contract,
        "scorerInputContract": scorer_input_contract,
        "recipeAnalysisProfile": recipe_analysis_profile,
        "subjectSegmentation": {
            "version": "subject-1.0",
            "method": "MOBILE_SAM_VIT_T_BOX_PROMPT",
            "usageMode": "SPATIAL_GATE",
            "approvalState": "ENGINEERING_AUTO",
            "promptPolicy": "INSPECTION_ROI_BOUNDING_BOX_V1",
            "modelRepositoryVersion": digest_directory(args.segmenter_repository.resolve()),
            "modelWeightsSha256": digest_file(args.segmenter_weights.resolve()),
            "minModelQualityScore": 0.8,
            "minForegroundRatio": 0.2,
            "maxForegroundRatio": 0.95,
            "supportPaddingPx": 0,
            "boundaryBandPx": 0,
        },
        "currentSubjectSegmentation": {
            "version": "current-subject-1.0",
            "method": "MOBILE_SAM_VIT_T_BOX_PROMPT",
            "promptPolicy": "INSPECTION_ROI_BOUNDING_BOX_V1",
            "interiorErosionPx": 8,
            "minMaskIou": 0.85,
            "maxAreaDeltaRatio": 0.15,
            "minInteriorRatio": 0.7,
            "boundaryMinRegionAreaRatio": 0.001,
            "maxBoundaryRegions": 16,
        },
        "dimensionMeasurementPolicy": {
            "version": "dimension-1.0",
            "method": "CHARUCO_PLANE_CURRENT_MASK_MIN_AREA_RECT_V1",
            "approvalState": "ENGINEERING_AUTO",
            "calibrationSource": "CHARUCO_BOARD_PLANE_V1",
            "maxPlaneReprojectionErrorPx": 3.0,
            "segmentationBoundaryUncertaintyPx": 2.0,
            "maxRelativeLinearUncertainty": 0.05,
            "minContourAreaPx": 2000,
            "minContourPoints": 100,
        },
        "subjectAlignment": {
            "version": "subject-align-1.0",
            "method": "SUBJECT_CONTOUR_ECC_AFFINE",
            "approvalState": "ENGINEERING_AUTO",
            "maskSource": "GOLDEN_SUBJECT_MASK",
            "alignmentBandPx": 24,
            "heldOutBlockPx": 32,
            "maxIterations": 200,
            "convergenceEpsilon": 0.00001,
            "minEccCorrelation": 0.2,
            "maxHeldOutResidualPx": 8.0,
            "minHeldOutCoverageRatio": 0.35,
            "maxResidualTranslationPx": 120.0,
            "maxResidualRotationDegrees": 15.0,
            "maxResidualScaleDelta": 0.25,
            "maxResidualShear": 0.15,
        },
        "candidateVerificationPolicy": {
            "version": "candidate-verify-2.0",
            "method": "DINO_CROP_COSINE_LOCAL_STRUCTURE_V2",
            "mode": "SHADOW",
            "approvalState": "ENGINEERING_AUTO",
            "contextPaddingRatio": 0.35,
            "minimumCropSidePx": 112,
            "maxCandidates": 12,
            "reviewPriorityDistance": 0.1,
            "highPriorityDistance": 0.25,
            "localAlignmentMethod": "GRADIENT_ECC_TRANSLATION_V1",
            "photometricNormalization": "OPENCV_LAB_CONTEXT_MEDIAN_MAD_V1",
            "structureMethod": "LAB_DELTA_OR_CANNY_EDGE_V1",
            "maxLocalTranslationPx": 6.0,
            "minLocalAlignmentCorrelation": 0.45,
            "candidateExclusionPaddingPx": 5,
            "minimumContextPixels": 512,
            "appearanceDeltaThreshold": 0.12,
            "minAppearanceChangedAreaRatio": 0.3,
            "minEdgeChangedAreaRatio": 0.15,
        },
    }
    # The compiler resolves Golden paths relative to the spec. Keep an exact
    # local copy beside it when the source lives elsewhere.
    local_golden = output_dir / golden_path.name
    if local_golden.resolve() != golden_path:
        local_golden.write_bytes(golden_bytes)
    spec_path = output_dir / "engineering-real-dino-build-spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_path = output_dir / args.artifact_name
    board_candidates = None
    if args.calibration_board_manifest is not None:
        manifest_path = args.calibration_board_manifest.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        charuco = manifest["charuco"]
        aruco = manifest["aruco"]
        finished = manifest["finishedSizeMm"]
        board_candidates = [GoldenDimensionBoardCandidate.model_validate({
            "boardId": manifest["boardId"],
            "revision": manifest["revision"],
            "profile": manifest["profile"],
            "manifestSha256": digest_file(manifest_path),
            "dictionary": aruco["dictionary"],
            "squaresX": charuco["squaresX"],
            "squaresY": charuco["squaresY"],
            "squareLengthMm": charuco["squareLengthMm"],
            "markerLengthMm": charuco["markerLengthMm"],
            "markerIds": charuco["markerIds"],
            "finishedWidthMm": finished["width"],
            "finishedHeightMm": finished["height"],
            "charucoOriginMm": charuco["originMm"],
            "outerMarkers": [
                {"id": marker["id"], "cornersMm": marker["cornersMm"]}
                for marker in aruco["markers"]
            ],
            "charucoGeometryQualified": True,
            "outerArucoGeometryQualified": len(aruco["markers"]) >= 3,
        })]
    result = compile_artifact(
        spec_path, artifact_path, args.model_repository.resolve(), args.model_weights.resolve(),
        segmenter_repository=args.segmenter_repository.resolve(),
        segmenter_weights=args.segmenter_weights.resolve(),
        allow_target_only_alignment=True,
        board_candidates=board_candidates,
    )
    summary = {
        "mode": "ENGINEERING_REAL_DINO",
        "productionAuthorized": False,
        "artifactPath": str(result.artifact_path),
        "artifactPackageDigest": result.artifact_package_digest,
        "evidencePath": str(result.evidence_path),
        "goldenSourceSha256": _digest_bytes(golden_bytes),
        "sourceDimensions": {"width": source_dimensions[0], "height": source_dimensions[1]},
        "confirmedGoldenRoiNormalized": {"x": args.roi[0], "y": args.roi[1], "width": args.roi[2], "height": args.roi[3]},
        "canonicalContentBounds": content,
        "canonicalContourAnchorRegion": contour_anchor,
        "canonicalInspectionRegion": inspection,
    }
    summary_path = output_dir / "engineering-real-dino-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
