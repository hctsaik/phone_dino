"""Freeze an external V2 development-evidence ledger for Git review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_successor_selection_v2 import freeze_successor_v2_development_evidence_ledger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an external V2 development-evidence ledger for review before patching the fixed tracked Git path."
    )
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--development-report", required=True, action="append", type=Path)
    parser.add_argument("--augmentation-manifest", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    ledger = freeze_successor_v2_development_evidence_ledger(
        arguments.parent_holdout, arguments.parent_selection_contract, arguments.plan, arguments.envelope,
        arguments.development_report, arguments.augmentation_manifest, arguments.recipe, arguments.output,
    )
    print(json.dumps({
        "output": str(arguments.output),
        "developmentEvidenceLedgerSha256": ledger["developmentEvidenceLedgerSha256"],
        "contractBindingProjectionSha256": ledger["contractBindingProjectionSha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
