"""Predeclare exact MVTec normal-holdout group quotas before scoring candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_normal_holdout import NormalHoldoutError, create_normal_holdout_plan


def _parse_quota(value: str) -> tuple[str, dict[str, int]]:
    try:
        category, counts = value.split("=", 1)
        fields = [int(part) for part in counts.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("quota must be CATEGORY=FIT,TUNING,SELECTION,CONFIRMATION,RESERVE") from error
    if not category or len(fields) != 5:
        raise argparse.ArgumentTypeError("quota must be CATEGORY=FIT,TUNING,SELECTION,CONFIRMATION,RESERVE")
    return category, {
        "fitGroupCount": fields[0],
        "thresholdTuningGroupCount": fields[1],
        "normalSelectionGroupCount": fields[2],
        "normalConfirmationGroupCount": fields[3],
        "reserveUntouchedGroupCount": fields[4],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a normal-only partition plan.  Every eligible group must be assigned exactly once."
    )
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-metadata", required=True, type=Path)
    parser.add_argument("--quota", action="append", required=True, type=_parse_quota)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    quotas: dict[str, dict[str, int]] = {}
    for category, quota in arguments.quota:
        if category in quotas:
            raise NormalHoldoutError(f"category quota was provided more than once: {category}")
        quotas[category] = quota
    document = create_normal_holdout_plan(
        arguments.pool,
        arguments.ledger,
        arguments.output,
        source_root=arguments.source_root,
        source_metadata_path=arguments.source_metadata,
        category_group_quotas=quotas,
    )
    print(json.dumps({
        "output": str(arguments.output),
        "categories": [value["category"] for value in document["categoryQuotas"]],
        "partitionSeedSha256": document["partitionSeedSha256"],
        "normalHoldoutPlanSha256": document["normalHoldoutPlanSha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
