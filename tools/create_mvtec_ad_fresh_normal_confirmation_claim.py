"""Explicitly claim the one fresh NORMAL_CONFIRMATION observation slot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_fresh_normal_observation import (  # noqa: E402
    create_fresh_normal_confirmation_claim,
    fresh_confirmation_claim_path,
)
from phone_dino.mvtec_fresh_normal_selection import load_validated_fresh_selection_contract  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly create the fixed confirmation claim after a JSON-only eligible selection lock."
    )
    parser.add_argument("--contract", type=Path, required=True)
    arguments = parser.parse_args()
    claim = create_fresh_normal_confirmation_claim(arguments.contract)
    contract, _ = load_validated_fresh_selection_contract(arguments.contract)
    print(json.dumps({
        "claim": str(fresh_confirmation_claim_path(contract)),
        "confirmationClaimSha256": claim["confirmationClaimSha256"],
        "selectedCandidateId": claim["selectedCandidateId"],
    }, indent=2))


if __name__ == "__main__":
    main()
