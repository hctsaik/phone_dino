"""Freeze a JSON-only selection contract before opening normal selection images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_fresh_normal_selection import (  # noqa: E402
    FRESH_NORMAL_SELECTION_OBJECTIVE,
    create_fresh_normal_selection_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one immutable fresh normal selection contract without opening any image bytes."
    )
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--augmentation-manifest", required=True, type=Path)
    parser.add_argument("--development-report", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-above-threshold-rate", required=True, type=float)
    parser.add_argument("--max-p95-score-minus-threshold", required=True, type=float)
    parser.add_argument("--max-maximum-score-minus-threshold", required=True, type=float)
    arguments = parser.parse_args()
    contract = create_fresh_normal_selection_contract(
        arguments.holdout,
        arguments.augmentation_manifest,
        arguments.development_report,
        arguments.output,
        selection_gates={
            "maxAboveThresholdRate": arguments.max_above_threshold_rate,
            "maxP95ScoreMinusThreshold": arguments.max_p95_score_minus_threshold,
            "maxMaximumScoreMinusThreshold": arguments.max_maximum_score_minus_threshold,
        },
        selection_objective=dict(FRESH_NORMAL_SELECTION_OBJECTIVE),
    )
    print(json.dumps({
        "output": str(arguments.output),
        "contractSha256": contract["contractSha256"],
        "candidateUniverseIdentitySha256": contract["candidateUniverseIdentitySha256"],
        "normalSelectionInputIdentitySha256": contract["normalSelectionInputIdentitySha256"],
        "normalConfirmationInputIdentitySha256": contract["normalConfirmationInputIdentitySha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
