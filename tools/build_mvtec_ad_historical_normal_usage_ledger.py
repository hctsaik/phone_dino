"""Create an external exclusion ledger from normal-only MVTec reports only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_normal_holdout import build_historical_normal_usage_ledger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze source-hash exclusions from non-authoritative normal-only reports; no images are opened."
    )
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    document = build_historical_normal_usage_ledger(arguments.report, arguments.output)
    print(json.dumps({
        "output": str(arguments.output),
        "reportCount": len(document["evidence"]),
        "normalSourceCount": len(document["normalSourceSha256"]),
        "historicalNormalUsageLedgerSha256": document["historicalNormalUsageLedgerSha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
