"""Replay frozen alignment evidence or explicitly unverified ad-hoc cases.

``--cohort`` is the reproducible path: it binds raw stills, saved requests and
responses, an artifact, and a readiness snapshot before replaying the runtime
normalizer.  The older ``--case`` path remains useful for local triage, but its
records are deliberately labelled ``UNVERIFIED_AD_HOC``.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

from phone_dino.artifacts import (
    ProductionArtifactV14,
    ProductionArtifactV15,
    ProductionArtifactV16,
    ProductionArtifactV17,
    ProductionArtifactV18,
    ProductionArtifactV19,
)
from phone_dino.physical_alignment_replay import PhysicalAlignmentReplayError, run_cohort
from phone_dino.production import OpenCvTargetAligner


ARTIFACT_MODELS = {
    "1.4": ProductionArtifactV14,
    "1.5": ProductionArtifactV15,
    "1.6": ProductionArtifactV16,
    "1.7": ProductionArtifactV17,
    "1.8": ProductionArtifactV18,
    "1.9": ProductionArtifactV19,
}


def _load_artifact(path: Path):
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    model = ARTIFACT_MODELS.get(document.get("schemaVersion"))
    if model is None:
        raise ValueError("subject-alignment replay requires artifact schema 1.4 through 1.9")
    return model.model_validate(document)


def _decode_reference(artifact):
    import cv2
    import numpy as np

    payload = base64.b64decode(artifact.target_alignment.reference_image_base64, validate=True)
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("target reference image cannot be decoded")
    return image


def _generated_cases(artifact) -> list[tuple[str, Any, str | None]]:
    import cv2
    import numpy as np

    reference = _decode_reference(artifact)
    height, width = reference.shape[:2]
    inspection = artifact.target_alignment.inspection_regions[0]
    cases: list[tuple[str, Any, str | None]] = [
        ("generated/exact", reference.copy(), "ALIGNED"),
        ("generated/lighting-shift", cv2.convertScaleAbs(reference, alpha=0.72, beta=38), "ALIGNED"),
    ]

    inspection_change = reference.copy()
    left = inspection.x + max(4, inspection.width // 5)
    right = inspection.x + inspection.width - max(4, inspection.width // 5)
    top = inspection.y + inspection.height // 3
    bottom = min(inspection.y + inspection.height - 1, top + max(12, inspection.height // 10))
    cv2.rectangle(inspection_change, (left, top), (right, bottom), (0, 0, 255), -1)
    cases.append(("generated/inspection-change", inspection_change, "ALIGNED"))

    background_changed = reference.copy()
    background_changed[:, :inspection.x] = 245
    background_changed[:, inspection.x + inspection.width:] = 245
    cases.append(("generated/background-replaced", background_changed, "ALIGNED"))

    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    inset = max(12, round(width * 0.10))
    destination = np.float32([
        [inset, 0], [width - inset - 1, 0], [width - 1, height - 1], [0, height - 1],
    ])
    perspective = cv2.getPerspectiveTransform(source, destination)
    perspective_mismatch = cv2.warpPerspective(
        reference, perspective, (width, height), borderValue=(220, 220, 220),
    )
    # This is deliberately observation-only: whether the recipe should accept
    # this degree of projective change must be calibrated on real defects.
    cases.append(("generated/perspective-mismatch", perspective_mismatch, None))
    cases.append(("generated/missing-target", np.full_like(reference, 220), "NOT_ALIGNED"))
    return cases


def _real_cases(values: list[str]) -> list[tuple[str, Any, str | None]]:
    import cv2

    cases = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --case {value!r}; expected LABEL=IMAGE_PATH")
        label, raw_path = value.split("=", 1)
        image = cv2.imread(str(Path(raw_path).resolve()))
        if image is None:
            raise ValueError(f"cannot decode replay image: {raw_path}")
        cases.append((f"real/{label}", image, None))
    return cases


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--cohort", type=Path, help="external frozen physical-alignment cohort JSON")
    parser.add_argument("--output", type=Path, help="new external cohort replay report JSON")
    parser.add_argument("--case", action="append", default=[], metavar="LABEL=IMAGE_PATH")
    parser.add_argument("--generated", action="store_true", help="also run deterministic generated cases")
    arguments = parser.parse_args()
    if arguments.cohort is not None:
        if arguments.artifact is not None or arguments.case or arguments.generated or arguments.output is None:
            parser.error("--cohort requires --output and cannot be combined with --artifact, --case, or --generated")
        try:
            report = run_cohort(arguments.cohort, arguments.output)
        except PhysicalAlignmentReplayError as error:
            parser.error(str(error))
        print(json.dumps({
            "reportPath": str(arguments.output),
            "reportSha256": report["reportSha256"],
            "evidenceClassification": report["evidenceClassification"],
        }, ensure_ascii=False))
        return 2 if any(not item["expectationMet"] for item in report["cases"]) else 0
    if arguments.output is not None:
        parser.error("--output is only valid with --cohort")
    if arguments.artifact is None:
        parser.error("--artifact is required unless --cohort is used")
    if not arguments.case and not arguments.generated:
        parser.error("provide at least one --case or --generated")

    artifact = _load_artifact(arguments.artifact.resolve())
    aligner = OpenCvTargetAligner(allow_contour_anchor_alignment=True)
    failures = 0
    cases = [*_real_cases(arguments.case), *(_generated_cases(artifact) if arguments.generated else [])]
    for label, image_bgr, expected in cases:
        result = aligner.align(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), artifact)
        alignment = result.alignment
        state = alignment.state if alignment is not None else "NOT_ALIGNED"
        expectation_met = expected is None or state == expected
        failures += int(not expectation_met)
        print(json.dumps({
            "provenance": "UNVERIFIED_AD_HOC",
            "qualificationStatement": "NOT_PHYSICAL_QUALIFICATION_OR_PRODUCTION_AUTHORIZATION",
            "case": label,
            "expectedState": expected,
            "expectationMet": expectation_met,
            "state": state,
            "method": alignment.method if alignment is not None else None,
            "correlationOrInlierRatio": alignment.inlier_ratio if alignment is not None else 0.0,
            "heldOutOrReprojectionResidualPx": (
                alignment.reprojection_error_px if alignment is not None else 1000.0
            ),
            "coverageRatio": alignment.coverage_ratio if alignment is not None else 0.0,
            "reasonCodes": list(result.reason_codes),
        }, ensure_ascii=False))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
