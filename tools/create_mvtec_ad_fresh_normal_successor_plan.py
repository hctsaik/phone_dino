"""Freeze the fixed reserve-only allocation for the V2 successor screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_normal_successor import create_fresh_normal_successor_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the immutable, deterministic reserve-only successor allocation plan without opening images."
    )
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    plan = create_fresh_normal_successor_plan(
        arguments.parent_holdout,
        arguments.parent_selection_contract,
        arguments.output,
    )
    print(json.dumps({
        "output": str(arguments.output),
        "successorPlanSha256": plan["successorPlanSha256"],
        "allocationSeedSha256": plan["allocationSeedSha256"],
        "categoryQuotas": plan["categoryQuotas"],
    }, indent=2))


if __name__ == "__main__":
    main()
