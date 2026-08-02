from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import platform
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal


GENERATOR_VERSION = "phone_dino.synthetic-suite/1.0"
DEFAULT_SEED = 20260802


class SyntheticSceneError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SceneSpec:
    scene_id: str
    board_dx: float = 0.0
    board_dy: float = 0.0
    board_rotation: float = 0.0
    board_scale: float = 1.0
    target_dx: float = 0.0
    target_dy: float = 0.0
    target_rotation: float = 0.0
    target_scale: float = 1.0
    perspective_x: float = 0.0
    perspective_y: float = 0.0
    blur_sigma: float = 0.0
    exposure_gain: float = 1.0
    glare_ratio: float = 0.0
    shadow_ratio: float = 0.0
    jpeg_quality: int = 0
    target_state: Literal["PRESENT", "ABSENT", "DUPLICATE", "PARTIAL"] = "PRESENT"
    defect_type: Literal["NONE", "MISSING_COMPONENT", "SHIFTED_COMPONENT"] = "NONE"
    held_out_parallax_px: float = 0.0
    local_warp_px: float = 0.0
    board_occlusion: float = 0.0
    expected_capture_state: Literal["ACCEPTED", "RECAPTURE_REQUIRED"] = "ACCEPTED"
    reason_class: str = "NONE"


SUITE_SCENES = (
    SceneSpec("accepted-reference", target_rotation=2.0),
    SceneSpec("accepted-board-motion", board_dx=10, board_dy=-8, board_rotation=1.0, target_rotation=2.0),
    SceneSpec("accepted-target-offset", target_dx=38, target_dy=-25, target_rotation=3.0),
    SceneSpec("accepted-combined-motion", board_dx=-24, board_dy=18, board_rotation=-2.0,
              target_dx=30, target_dy=16, target_rotation=2.5),
    SceneSpec("accepted-camera-perspective", perspective_x=8, perspective_y=4),
    SceneSpec("accepted-defect", target_rotation=2.0, defect_type="MISSING_COMPONENT"),
    SceneSpec("accepted-jpeg-shadow", target_rotation=1.5, jpeg_quality=95, shadow_ratio=0.08),
    SceneSpec("rejected-target-absent", target_state="ABSENT", expected_capture_state="RECAPTURE_REQUIRED",
              reason_class="TARGET_MATCHES_INSUFFICIENT"),
    SceneSpec("rejected-target-duplicate", target_state="DUPLICATE", held_out_parallax_px=15,
              expected_capture_state="RECAPTURE_REQUIRED",
              reason_class="TARGET_PARALLAX_OR_MISMATCH"),
    SceneSpec("rejected-heldout-parallax", held_out_parallax_px=20, local_warp_px=10,
              expected_capture_state="RECAPTURE_REQUIRED", reason_class="TARGET_PARALLAX_OR_MISMATCH"),
    SceneSpec("rejected-blur", blur_sigma=10, expected_capture_state="RECAPTURE_REQUIRED", reason_class="BLUR"),
    SceneSpec("rejected-overexposure", exposure_gain=3.5, expected_capture_state="RECAPTURE_REQUIRED",
              reason_class="OVER_EXPOSURE"),
    SceneSpec("rejected-glare", glare_ratio=0.82, expected_capture_state="RECAPTURE_REQUIRED",
              reason_class="OVER_EXPOSURE"),
    SceneSpec("rejected-board-occluded", board_occlusion=0.72, expected_capture_state="RECAPTURE_REQUIRED",
              reason_class="CHARUCO_CORNERS_INSUFFICIENT"),
    SceneSpec("rejected-target-partial", target_state="PARTIAL", expected_capture_state="RECAPTURE_REQUIRED",
              reason_class="TARGET_ALIGNMENT"),
    SceneSpec("rejected-transform-out-of-bounds", target_rotation=20,
              expected_capture_state="RECAPTURE_REQUIRED", reason_class="TARGET_TRANSFORM_OUT_OF_BOUNDS"),
)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _matrix(values) -> list[float]:
    return [round(float(value), 9) for value in values.reshape(-1)]


def _points(values) -> list[list[float]]:
    return [[round(float(x), 6), round(float(y), 6)] for x, y in values]


def _pose(width: int, height: int, center_x: float, center_y: float, scale: float, rotation: float):
    import cv2
    import numpy as np

    radians = math.radians(rotation)
    cosine, sine = math.cos(radians) * scale, math.sin(radians) * scale
    affine = np.asarray([
        [cosine, -sine, center_x - cosine * width / 2 + sine * height / 2],
        [sine, cosine, center_y - sine * width / 2 - cosine * height / 2],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return affine


def _project(transform, corners):
    import cv2
    import numpy as np

    return cv2.perspectiveTransform(np.asarray(corners, dtype=np.float32).reshape(1, -1, 2), transform).reshape(-1, 2)


def _target_reference(seed: int):
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    image = np.full((240, 320, 3), 215, dtype=np.uint8)
    for _ in range(520):
        x, y = int(rng.integers(6, 314)), int(rng.integers(6, 234))
        radius = int(rng.integers(1, 4))
        color = tuple(int(value) for value in rng.integers(10, 195, 3))
        cv2.circle(image, (x, y), radius, color, -1)
    cv2.rectangle(image, (105, 82), (215, 158), (35, 35, 35), -1)
    cv2.rectangle(image, (125, 100), (195, 140), (190, 190, 190), -1)
    cv2.putText(image, "PM", (137, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2)
    return image


def _board_image():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 7), 20.0, 15.0, dictionary)
    return board.generateImage((640, 896), marginSize=0, borderBits=1)


def _warp_overlay(canvas, source, transform):
    import cv2
    import numpy as np

    height, width = canvas.shape[:2]
    warped = cv2.warpPerspective(source, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    mask = cv2.warpPerspective(np.full(source.shape[:2], 255, dtype=np.uint8), transform, (width, height))
    canvas[mask > 0] = warped[mask > 0]


def _encode(image, jpeg_quality: int) -> tuple[bytes, str, str]:
    import cv2

    if jpeg_quality:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        extension, content_type = ".jpg", "image/jpeg"
    else:
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        extension, content_type = ".png", "image/png"
    if not ok:
        raise SyntheticSceneError("SYNTHETIC_ENCODING_FAILED")
    return encoded.tobytes(), extension, content_type


def _render(spec: SceneSpec, reference):
    import cv2
    import numpy as np

    output_width, output_height = 960, 1040
    canvas = np.full((output_height, output_width, 3), 245, dtype=np.uint8)
    board_gray = _board_image()
    board = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
    board_pose = _pose(640, 896, 455 + spec.board_dx, 515 + spec.board_dy, spec.board_scale, spec.board_rotation)
    global_transform = np.asarray([
        [1.0, spec.perspective_x / output_height, 0.0],
        [spec.perspective_y / output_width, 1.0, 0.0],
        [spec.perspective_x / 2_000_000, spec.perspective_y / 2_000_000, 1.0],
    ], dtype=np.float64)
    board_to_image = global_transform @ board_pose
    _warp_overlay(canvas, board, board_to_image)

    # Reference target sits in a board-normalized location, then receives its own
    # pose.  Board-only motion changes only board_to_image; target-only motion
    # changes only target_to_image, making independence explicit in ground truth.
    target_to_plane = _pose(
        320, 240, 470 + spec.target_dx, 520 + spec.target_dy,
        0.78 * spec.target_scale, spec.target_rotation,
    )
    target_to_image = global_transform @ target_to_plane

    rendered_target = reference.copy()
    inspection_mask = np.zeros(rendered_target.shape[:2], dtype=np.uint8)
    inspection_mask[82:159, 105:216] = 255
    before_defect = rendered_target.copy()
    if spec.defect_type == "MISSING_COMPONENT":
        rendered_target[100:141, 125:196] = (215, 215, 215)
    elif spec.defect_type == "SHIFTED_COMPONENT":
        component = rendered_target[100:141, 125:196].copy()
        rendered_target[100:141, 125:196] = (35, 35, 35)
        rendered_target[108:149, 135:206] = component
    defect_delta = cv2.absdiff(before_defect, rendered_target)
    if spec.held_out_parallax_px:
        shift = int(round(spec.held_out_parallax_px))
        left = rendered_target[80:160, 0:95].copy()
        right = rendered_target[80:160, 225:320].copy()
        rendered_target[80:160, 0:95] = 215
        rendered_target[80:160, 225:320] = 215
        rendered_target[80:160, max(0, shift):95] = left[:, :95 - max(0, shift)]
        rendered_target[80:160, 225:320 - max(0, shift)] = right[:, max(0, shift):]
    if spec.local_warp_px:
        wave = int(round(spec.local_warp_px))
        rendered_target[80:160] = np.roll(rendered_target[80:160], wave, axis=1)

    if spec.target_state != "ABSENT":
        _warp_overlay(canvas, rendered_target, target_to_image)
    duplicate = None
    if spec.target_state == "DUPLICATE":
        duplicate = target_to_image.copy()
        duplicate[0, 2] += 300
        duplicate[1, 2] -= 100
        _warp_overlay(canvas, rendered_target, duplicate)
    if spec.target_state == "PARTIAL":
        projected = _project(target_to_image, ((0, 0), (320, 0), (320, 240), (0, 240)))
        x0, y0 = np.floor(projected.min(axis=0)).astype(int)
        x1, y1 = np.ceil(projected.max(axis=0)).astype(int)
        cv2.rectangle(canvas, (max(0, x0 + (x1 - x0) // 4), max(0, y0)),
                      (min(output_width - 1, x1), min(output_height - 1, y1)), (245, 245, 245), -1)
    if spec.board_occlusion:
        height = int(output_height * spec.board_occlusion)
        cv2.rectangle(canvas, (0, 0), (output_width, height), (245, 245, 245), -1)

    if spec.shadow_ratio:
        shadow_width = max(1, int(output_width * spec.shadow_ratio))
        gradient = np.linspace(0.45, 1.0, shadow_width, dtype=np.float32)[None, :, None]
        canvas[:, :shadow_width] = np.clip(canvas[:, :shadow_width] * gradient, 0, 255).astype(np.uint8)
    if spec.glare_ratio:
        axes = (int(output_width * spec.glare_ratio / 2), int(output_height * spec.glare_ratio / 3))
        cv2.ellipse(canvas, (output_width // 2, output_height // 2), axes, 0, 0, 360, (255, 255, 255), -1)
    canvas = np.clip(canvas.astype(np.float32) * spec.exposure_gain, 0, 255).astype(np.uint8)
    if spec.blur_sigma:
        kernel = max(3, int(math.ceil(spec.blur_sigma * 6)) | 1)
        canvas = cv2.GaussianBlur(canvas, (kernel, kernel), spec.blur_sigma)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    charuco = cv2.aruco.CharucoBoard((5, 7), 20.0, 15.0, dictionary)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(charuco).detectBoard(gray)
    detected = 0 if ids is None else len(ids)
    outside = cv2.countNonZero(cv2.cvtColor(defect_delta, cv2.COLOR_BGR2GRAY) & cv2.bitwise_not(inspection_mask))
    defect_gray = cv2.cvtColor(defect_delta, cv2.COLOR_BGR2GRAY)
    defect_support = np.argwhere(defect_gray > 0)
    target_corners = ((0, 0), (320, 0), (320, 240), (0, 240))
    board_corners = ((0, 0), (640, 0), (640, 896), (0, 896))
    inspection_corners = ((105, 82), (216, 82), (216, 159), (105, 159))
    missing_polygon = ((125, 100), (196, 100), (196, 141), (125, 141))
    shifted_polygon = ((125, 100), (206, 100), (206, 149), (125, 149))
    defect_source = [] if not len(defect_support) else [missing_polygon if spec.defect_type == "MISSING_COMPONENT" else shifted_polygon]
    geometry = {
        "planeSize": {"width": 640, "height": 896},
        "boardObjectToPlaneTransform": _matrix(board_pose),
        "targetReferenceToPlaneTransform": _matrix(target_to_plane),
        "planeToImageTransform": _matrix(global_transform),
        "boardObjectToImageTransform": _matrix(board_to_image),
        "targetReferenceToImageTransform": _matrix(target_to_image),
        "boardProjectedCorners": _points(_project(board_to_image, board_corners)),
        "targetProjectedCorners": _points(_project(target_to_image, target_corners)),
        "inspectionSupportPolygon": _points(_project(target_to_image, inspection_corners)),
        "defectSourcePolygons": [[list(point) for point in polygon] for polygon in defect_source],
        "defectSupportPolygons": [_points(_project(target_to_image, polygon)) for polygon in defect_source],
        "secondaryTargetReferenceToImageTransform": None if duplicate is None else _matrix(duplicate),
        "secondaryTargetProjectedCorners": [] if duplicate is None else _points(_project(duplicate, target_corners)),
        "occlusionSupportPolygon": (
            _points(((max(0, x0 + (x1 - x0) // 4), max(0, y0)), (min(output_width - 1, x1), max(0, y0)),
                     (min(output_width - 1, x1), min(output_height - 1, y1)),
                     (max(0, x0 + (x1 - x0) // 4), min(output_height - 1, y1))))
            if spec.target_state == "PARTIAL" else []
        ),
        "boardOcclusionSupportPolygon": (
            _points(((0, 0), (output_width, 0), (output_width, int(output_height * spec.board_occlusion)),
                     (0, int(output_height * spec.board_occlusion))))
            if spec.board_occlusion else []
        ),
    }
    return canvas, geometry, {
        "laplacianVariance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
        "overExposureRatio": round(float(np.count_nonzero(gray >= 250) / gray.size), 9),
        "detectedCharucoCorners": detected,
        "defectChangedPixels": int(cv2.countNonZero(defect_gray)),
        "defectChangedPixelsOutsideInspection": int(outside),
    }


def artifact_inputs(reference_bytes: bytes) -> dict[str, object]:
    return {
        "board": {
            "squaresX": 5, "squaresY": 7, "squareLengthMm": 20.0, "markerLengthMm": 15.0,
            "dictionary": "DICT_4X4_50", "canonicalWidth": 640, "canonicalHeight": 896,
        },
        "stillGate": {"minCharucoCorners": 6, "minLaplacianVariance": 35.0, "maxOverExposureRatio": 0.38},
        "targetAlignment": {
            "method": "TARGET_AFFINE", "referenceImageBase64": base64.b64encode(reference_bytes).decode("ascii"),
            "referenceImageSha256": _sha(reference_bytes), "canonicalWidth": 320, "canonicalHeight": 240,
            "alignmentRegions": [
                {"x": 0, "y": 0, "width": 320, "height": 70},
                {"x": 0, "y": 170, "width": 320, "height": 70},
            ],
            "heldOutRegions": [
                {"x": 0, "y": 80, "width": 95, "height": 80},
                {"x": 225, "y": 80, "width": 95, "height": 80},
            ],
            "inspectionRegions": [{"x": 105, "y": 82, "width": 111, "height": 77}],
            "minMatches": 16, "minInliers": 10, "minInlierRatio": 0.38, "minCoverageRatio": 0.06,
            "maxReprojectionErrorPx": 4.0, "minScale": 0.65, "maxScale": 1.35,
            "maxRotationDegrees": 16.0, "maxShear": 0.05, "maxTranslationPx": 800.0,
            "maxSecondaryInlierRatio": 0.35, "minHeldOutMatches": 8,
            "maxHeldOutReprojectionErrorPx": 18.0,
        },
    }


def generate_suite(output_dir: Path, seed: int = DEFAULT_SEED) -> Path:
    import cv2
    import numpy as np
    from . import __version__

    if seed < 0 or seed > 2**32 - 1:
        raise SyntheticSceneError("SEED_OUT_OF_RANGE")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise SyntheticSceneError("OUTPUT_DIRECTORY_NOT_EMPTY")
    reference = _target_reference(seed)
    ok, reference_encoded = cv2.imencode(".png", reference, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise SyntheticSceneError("REFERENCE_ENCODING_FAILED")
    reference_bytes = reference_encoded.tobytes()
    (output_dir / "target-reference.png").write_bytes(reference_bytes)
    scenes: list[dict[str, object]] = []
    for index, spec in enumerate(SUITE_SCENES):
        image, geometry, measured = _render(spec, reference)
        encoded, extension, content_type = _encode(image, spec.jpeg_quality)
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        charuco = cv2.aruco.CharucoBoard((5, 7), 20.0, 15.0, dictionary)
        _, ids, _, _ = cv2.aruco.CharucoDetector(charuco).detectBoard(gray)
        measured.update({
            "laplacianVariance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
            "overExposureRatio": round(float(np.count_nonzero(gray >= 250) / gray.size), 9),
            "detectedCharucoCorners": 0 if ids is None else len(ids),
        })
        filename = spec.scene_id + extension
        (output_dir / filename).write_bytes(encoded)
        scenes.append({
            "sceneId": spec.scene_id,
            "image": {"path": filename, "sha256": _sha(encoded), "contentType": content_type,
                      "width": image.shape[1], "height": image.shape[0]},
            "geometry": geometry,
            "conditions": {
                "requested": {
                    "blurSigma": spec.blur_sigma, "exposureGain": spec.exposure_gain,
                    "glareRatio": spec.glare_ratio, "shadowRatio": spec.shadow_ratio,
                    "jpegQuality": spec.jpeg_quality,
                },
                "measured": measured,
            },
            "faults": {
                "targetState": spec.target_state, "defectType": spec.defect_type,
                "heldOutParallaxPx": spec.held_out_parallax_px, "localWarpPx": spec.local_warp_px,
                "boardOcclusion": spec.board_occlusion,
            },
            "expected": {
                "captureState": spec.expected_capture_state,
                "analysisState": "RUN" if spec.expected_capture_state == "ACCEPTED" else "NOT_RUN",
                "reasonClass": spec.reason_class,
            },
        })
    runtime = {
        "packageVersion": __version__, "pythonVersion": platform.python_version(),
        "opencvVersion": cv2.__version__, "numpyVersion": np.__version__,
        "sourceSha256": _sha(Path(__file__).read_bytes()),
    }
    runtime_bytes = json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8")
    document = {
        "schemaVersion": GENERATOR_VERSION,
        "evidenceClass": "SYNTHETIC_ENGINEERING_ONLY",
        "generatorVersion": _sha(GENERATOR_VERSION.encode("utf-8") + b"|" + runtime_bytes),
        "generatorRuntime": runtime,
        "seed": seed,
        "targetReference": {"path": "target-reference.png", "sha256": _sha(reference_bytes),
                            "width": 320, "height": 240},
        "scenes": scenes,
        "artifactInputs": artifact_inputs(reference_bytes),
    }
    manifest = output_dir / "suite.manifest.json"
    manifest.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return manifest


def verify_suite_manifest(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticSceneError("MANIFEST_INVALID") from exc
    if document.get("schemaVersion") != GENERATOR_VERSION or set(document) != {
        "schemaVersion", "evidenceClass", "generatorVersion", "generatorRuntime", "seed", "targetReference", "scenes", "artifactInputs"
    }:
        raise SyntheticSceneError("MANIFEST_SCHEMA_INVALID")
    if document["evidenceClass"] != "SYNTHETIC_ENGINEERING_ONLY":
        raise SyntheticSceneError("MANIFEST_EVIDENCE_CLASS_INVALID")
    import cv2
    import numpy as np
    from . import __version__

    expected_runtime = {
        "packageVersion": __version__, "pythonVersion": platform.python_version(),
        "opencvVersion": cv2.__version__, "numpyVersion": np.__version__,
        "sourceSha256": _sha(Path(__file__).read_bytes()),
    }
    if document["generatorRuntime"] != expected_runtime:
        raise SyntheticSceneError("MANIFEST_GENERATOR_RUNTIME_INVALID")
    runtime_bytes = json.dumps(document["generatorRuntime"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected_generator = _sha(GENERATOR_VERSION.encode("utf-8") + b"|" + runtime_bytes)
    if document["generatorVersion"] != expected_generator:
        raise SyntheticSceneError("MANIFEST_GENERATOR_VERSION_INVALID")
    root = path.parent.resolve()
    files = [document["targetReference"], *(scene["image"] for scene in document["scenes"])]
    for record in files:
        candidate = (root / record["path"]).resolve()
        try:
            candidate.relative_to(root)
            content = candidate.read_bytes()
        except (ValueError, OSError) as exc:
            raise SyntheticSceneError("MANIFEST_FILE_INVALID") from exc
        if _sha(content) != record["sha256"]:
            raise SyntheticSceneError("MANIFEST_FILE_DIGEST_MISMATCH")
    # Recompute every redundant composition claimed by the truth manifest.
    def assert_points(actual, expected, reason: str) -> None:
        if np.asarray(actual).shape != np.asarray(expected).shape or not np.allclose(actual, expected, atol=1e-4):
            raise SyntheticSceneError(reason)

    reference_content = (root / document["targetReference"]["path"]).read_bytes()
    reference_image = cv2.imdecode(np.frombuffer(reference_content, dtype=np.uint8), cv2.IMREAD_COLOR)

    for scene in document["scenes"]:
        geometry = scene["geometry"]
        plane = np.asarray(geometry["planeToImageTransform"], dtype=np.float64).reshape(3, 3)
        board_plane = np.asarray(geometry["boardObjectToPlaneTransform"], dtype=np.float64).reshape(3, 3)
        target_plane = np.asarray(geometry["targetReferenceToPlaneTransform"], dtype=np.float64).reshape(3, 3)
        if not np.allclose(plane @ board_plane, np.asarray(geometry["boardObjectToImageTransform"]).reshape(3, 3), atol=1e-7):
            raise SyntheticSceneError("MANIFEST_BOARD_TRANSFORM_MISMATCH")
        if not np.allclose(plane @ target_plane, np.asarray(geometry["targetReferenceToImageTransform"]).reshape(3, 3), atol=1e-7):
            raise SyntheticSceneError("MANIFEST_TARGET_TRANSFORM_MISMATCH")
        board_image = plane @ board_plane
        target_image = plane @ target_plane
        assert_points(
            geometry["boardProjectedCorners"],
            _points(_project(board_image, ((0, 0), (640, 0), (640, 896), (0, 896)))),
            "MANIFEST_BOARD_CORNERS_MISMATCH",
        )
        assert_points(
            geometry["targetProjectedCorners"],
            _points(_project(target_image, ((0, 0), (320, 0), (320, 240), (0, 240)))),
            "MANIFEST_TARGET_CORNERS_MISMATCH",
        )
        assert_points(
            geometry["inspectionSupportPolygon"],
            _points(_project(target_image, ((105, 82), (216, 82), (216, 159), (105, 159)))),
            "MANIFEST_INSPECTION_POLYGON_MISMATCH",
        )
        defect_type = scene["faults"]["defectType"]
        if defect_type == "NONE":
            expected_source_defects = []
        elif defect_type == "MISSING_COMPONENT":
            expected_source_defects = [[(125, 100), (196, 100), (196, 141), (125, 141)]]
        else:
            expected_source_defects = [[(125, 100), (206, 100), (206, 149), (125, 149)]]
        if geometry["defectSourcePolygons"] != [[list(point) for point in item] for item in expected_source_defects]:
            raise SyntheticSceneError("MANIFEST_DEFECT_SOURCE_MISMATCH")
        expected_defects = [_points(_project(target_image, polygon)) for polygon in expected_source_defects]
        if len(expected_defects) != len(geometry["defectSupportPolygons"]):
            raise SyntheticSceneError("MANIFEST_DEFECT_POLYGON_MISMATCH")
        for actual, expected in zip(geometry["defectSupportPolygons"], expected_defects, strict=True):
            assert_points(actual, expected, "MANIFEST_DEFECT_POLYGON_MISMATCH")
        secondary = geometry["secondaryTargetReferenceToImageTransform"]
        expected_secondary = None
        if scene["faults"]["targetState"] == "DUPLICATE":
            expected_secondary = target_image.copy()
            expected_secondary[0, 2] += 300
            expected_secondary[1, 2] -= 100
        if (secondary is None) != (expected_secondary is None) or (
            secondary is not None and not np.allclose(secondary, expected_secondary.reshape(-1), atol=1e-7)
        ):
            raise SyntheticSceneError("MANIFEST_SECONDARY_TARGET_MISMATCH")
        if secondary is None:
            if geometry["secondaryTargetProjectedCorners"]:
                raise SyntheticSceneError("MANIFEST_SECONDARY_TARGET_MISMATCH")
        else:
            assert_points(
                geometry["secondaryTargetProjectedCorners"],
                _points(_project(np.asarray(secondary).reshape(3, 3), ((0, 0), (320, 0), (320, 240), (0, 240)))),
                "MANIFEST_SECONDARY_TARGET_MISMATCH",
            )
        if scene["faults"]["targetState"] == "PARTIAL":
            projected = np.asarray(geometry["targetProjectedCorners"])
            x0, y0 = np.floor(projected.min(axis=0)).astype(int)
            x1, y1 = np.ceil(projected.max(axis=0)).astype(int)
            expected_occlusion = _points((
                (max(0, x0 + (x1 - x0) // 4), max(0, y0)), (min(959, x1), max(0, y0)),
                (min(959, x1), min(1039, y1)), (max(0, x0 + (x1 - x0) // 4), min(1039, y1)),
            ))
        else:
            expected_occlusion = []
        assert_points(geometry["occlusionSupportPolygon"], expected_occlusion, "MANIFEST_OCCLUSION_MISMATCH")
        board_occlusion = scene["faults"]["boardOcclusion"]
        expected_board_occlusion = (
            _points(((0, 0), (960, 0), (960, int(1040 * board_occlusion)), (0, int(1040 * board_occlusion))))
            if board_occlusion else []
        )
        assert_points(
            geometry["boardOcclusionSupportPolygon"], expected_board_occlusion,
            "MANIFEST_BOARD_OCCLUSION_MISMATCH",
        )
        image_path = root / scene["image"]["path"]
        decoded = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape[1] != scene["image"]["width"] or decoded.shape[0] != scene["image"]["height"]:
            raise SyntheticSceneError("MANIFEST_IMAGE_DIMENSIONS_MISMATCH")
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        charuco = cv2.aruco.CharucoBoard((5, 7), 20.0, 15.0, dictionary)
        _, ids, _, _ = cv2.aruco.CharucoDetector(charuco).detectBoard(gray)
        measured = scene["conditions"]["measured"]
        checks = {
            "laplacianVariance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
            "overExposureRatio": round(float(np.count_nonzero(gray >= 250) / gray.size), 9),
            "detectedCharucoCorners": 0 if ids is None else len(ids),
        }
        if any(measured[key] != value for key, value in checks.items()):
            raise SyntheticSceneError("MANIFEST_MEASUREMENT_MISMATCH")
        expected_target = reference_image.copy()
        if defect_type == "MISSING_COMPONENT":
            expected_target[100:141, 125:196] = (215, 215, 215)
        elif defect_type == "SHIFTED_COMPONENT":
            component = expected_target[100:141, 125:196].copy()
            expected_target[100:141, 125:196] = (35, 35, 35)
            expected_target[108:149, 135:206] = component
        defect_gray = cv2.cvtColor(cv2.absdiff(reference_image, expected_target), cv2.COLOR_BGR2GRAY)
        expected_changed = int(cv2.countNonZero(defect_gray))
        if measured["defectChangedPixels"] != expected_changed or measured["defectChangedPixelsOutsideInspection"] != 0:
            raise SyntheticSceneError("MANIFEST_DEFECT_MEASUREMENT_MISMATCH")
    return document


def evaluate_suite_alignment(path: Path) -> dict[str, dict[str, object]]:
    """Run the real plane/target normalizer without DINO for engineering gates."""
    from .artifacts import ProductionArtifact
    from .decoder import DecodedImage
    from .production import OpenCvCharucoNormalizer

    document = verify_suite_manifest(path)
    component = "sha256:" + "1" * 64
    artifact = ProductionArtifact.model_validate({
        "schemaVersion": "1.1", "recipeId": "synthetic-recipe", "machineId": "synthetic-machine",
        "boardId": "synthetic-board", "goldenSetVersion": component,
        "normalizationPipelineVersion": component, "analyzerModelVersion": component,
        "decisionPolicyVersion": component, "analyzerRuntimeVersion": component,
        "modelRepositoryVersion": component, "boardInstallationVersion": component,
        "modelWeightsSha256": component, **document["artifactInputs"],
        "goldenEmbeddings": [{"id": "synthetic", "sourceSha256": component, "values": [1.0, 0.0, 0.0]}],
    })
    normalizer = OpenCvCharucoNormalizer()
    observations: dict[str, dict[str, object]] = {}
    for scene in document["scenes"]:
        record = scene["image"]
        content = (path.parent / record["path"]).read_bytes()
        decoded = DecodedImage(
            data=content, width=record["width"], height=record["height"],
            format="JPEG" if record["contentType"] == "image/jpeg" else "PNG", elapsed_ms=0,
        )
        result = normalizer.normalize(decoded, artifact)
        observations[scene["sceneId"]] = {
            "captureState": "ACCEPTED" if not result.reason_codes else "RECAPTURE_REQUIRED",
            "reasonCodes": list(result.reason_codes),
            "alignmentState": None if result.alignment is None else result.alignment.state,
        }
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic phone_dino full-scene safety fixtures")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    try:
        manifest = generate_suite(arguments.output_dir.resolve(), arguments.seed)
        verify_suite_manifest(manifest)
    except SyntheticSceneError as exc:
        raise SystemExit(f"Synthetic suite generation failed: {exc}") from exc
    print(str(manifest))


if __name__ == "__main__":
    main()
