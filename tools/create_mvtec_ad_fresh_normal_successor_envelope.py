"""Materialize the fixed reserve-only V2 successor phase envelope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_normal_successor import create_fresh_normal_successor_envelope  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the immutable V2 successor envelope without opening source image bytes."
    )
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    envelope = create_fresh_normal_successor_envelope(
        arguments.parent_holdout,
        arguments.parent_selection_contract,
        arguments.plan,
        arguments.output,
    )
    print(json.dumps({
        "output": str(arguments.output),
        "successorEnvelopeSha256": envelope["successorEnvelopeSha256"],
        "resultLabel": envelope["resultLabel"],
        "independenceLabel": envelope["independenceLabel"],
        "successorPartitionIdentities": envelope["successorPartitionIdentities"],
    }, indent=2))


if __name__ == "__main__":
    main()
