"""Seal the unopened reserve of a failed fresh normal-selection cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_normal_successor import (  # noqa: E402
    create_fresh_normal_successor_seal,
    fresh_normal_successor_seal_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically delegate only a failed parent cohort's tool-mediated-unconsumed reserve; "
            "no source image is opened."
        )
    )
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    arguments = parser.parse_args()
    seal = create_fresh_normal_successor_seal(
        arguments.parent_holdout,
        arguments.parent_selection_contract,
    )
    print(json.dumps({
        "output": str(fresh_normal_successor_seal_path(arguments.parent_holdout)),
        "successorSealSha256": seal["successorSealSha256"],
        "delegatedReserveInputIdentitySha256": seal["delegatedReserveInputIdentitySha256"],
        "resultLabel": seal["resultLabel"],
        "delegationPolicy": seal["delegationPolicy"],
    }, indent=2))


if __name__ == "__main__":
    main()
