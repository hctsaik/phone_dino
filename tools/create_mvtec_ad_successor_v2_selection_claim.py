"""Reserve the fixed JSON-only successor V2 selection claim slot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_successor_selection_v2 import create_successor_v2_selection_claim


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a JSON-only one-time successor V2 selection claim.")
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    arguments = parser.parse_args()
    claim = create_successor_v2_selection_claim(
        arguments.parent_holdout, arguments.parent_selection_contract, arguments.plan, arguments.envelope, arguments.contract
    )
    print(json.dumps({"claimSha256": claim["claimSha256"], "claimSlot": claim["claimSlot"]}, indent=2))


if __name__ == "__main__":
    main()
