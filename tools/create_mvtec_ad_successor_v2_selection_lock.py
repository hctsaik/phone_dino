"""Create the JSON-only reserve-successor V2 selection lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_successor_selection_v2 import create_successor_v2_selection_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the JSON-only reserve-successor V2 selection lock.")
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    arguments = parser.parse_args()
    lock = create_successor_v2_selection_lock(
        arguments.parent_holdout, arguments.parent_selection_contract, arguments.plan, arguments.envelope, arguments.contract
    )
    print(json.dumps({"selectionLockSha256": lock["selectionLockSha256"], "decision": lock["decision"]}, indent=2))


if __name__ == "__main__":
    main()
