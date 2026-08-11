"""Run an external V2 synthetic-stimulus response-only report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_synthetic_stress_v2 import run_synthetic_stress_v2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an offline SYNTHETIC_ONLY V2 stimulus-response report. It accepts only successor FIT parents and "
            "a validated V2 package; it does not estimate real anomaly performance."
        )
    )
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--augmentation-manifest", required=True, type=Path)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-repo", type=Path, default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2")
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2_vits14_pretrain.pth",
    )
    arguments = parser.parse_args()
    report = run_synthetic_stress_v2(
        arguments.parent_holdout,
        arguments.parent_selection_contract,
        arguments.plan,
        arguments.envelope,
        arguments.augmentation_manifest,
        arguments.output,
        source_root=arguments.source_root,
        recipe_path=arguments.recipe,
        model_repo=arguments.model_repo,
        model_weights=arguments.model_weights,
        device="cpu",
    )
    print(json.dumps({
        "output": str(arguments.output),
        "syntheticStressReportSha256": report["syntheticStressReportSha256"],
        "metricScope": report["metricScope"],
        "realAnomalyPerformance": report["realAnomalyPerformance"],
        "aggregate": report["aggregate"],
        "categories": {
            category: {
                "thresholdFromRawCalibration": values["thresholdFromRawCalibration"],
                "responseCounts": values["responseCounts"],
                "responseRates": values["responseRates"],
                "pairedScoreDeltaSummary": values["pairedScoreDeltaSummary"],
            }
            for category, values in report["categories"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
