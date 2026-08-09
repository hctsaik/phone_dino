"""Create the JSON-only lock from an aggregate fresh selection observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_fresh_normal_observation import (  # noqa: E402
    create_fresh_normal_selection_lock,
    fresh_selection_lock_path,
)
from phone_dino.mvtec_fresh_normal_selection import load_validated_fresh_selection_contract  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute fresh normal selection gates/objective from JSON only; never starts confirmation."
    )
    parser.add_argument("--contract", type=Path, required=True)
    arguments = parser.parse_args()
    lock = create_fresh_normal_selection_lock(arguments.contract)
    contract, _ = load_validated_fresh_selection_contract(arguments.contract)
    print(json.dumps({
        "lock": str(fresh_selection_lock_path(contract)),
        "selectionLockSha256": lock["selectionLockSha256"],
        "state": lock["decision"]["state"],
        "selectedCandidateId": lock["decision"]["selectedCandidateId"],
        "automaticConfirmation": lock["decision"]["automaticConfirmation"],
    }, indent=2))


if __name__ == "__main__":
    main()
