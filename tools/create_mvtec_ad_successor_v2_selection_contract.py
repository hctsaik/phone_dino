"""Freeze the reserve-successor V2 candidate universe before selection opens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_successor_selection_v2 import create_successor_v2_selection_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a JSON-only reserve-successor V2 selection contract.")
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--development-report", required=True, action="append", type=Path)
    parser.add_argument("--augmentation-manifest", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--development-evidence-ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    contract = create_successor_v2_selection_contract(
        arguments.parent_holdout, arguments.parent_selection_contract, arguments.plan, arguments.envelope,
        arguments.development_report, arguments.augmentation_manifest, arguments.recipe,
        arguments.development_evidence_ledger, arguments.output,
    )
    print(json.dumps({"output": str(arguments.output), "contractSha256": contract["contractSha256"]}, indent=2))


if __name__ == "__main__":
    main()
