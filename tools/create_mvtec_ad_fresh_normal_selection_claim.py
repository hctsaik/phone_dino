"""Create the immutable claim that precedes one normal-selection observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_fresh_normal_selection import (  # noqa: E402
    create_fresh_normal_selection_claim,
    fresh_selection_claim_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically claim one frozen NORMAL_SELECTION partition without opening an image."
    )
    parser.add_argument("--contract", required=True, type=Path)
    arguments = parser.parse_args()
    claim = create_fresh_normal_selection_claim(arguments.contract)
    print(json.dumps({
        "output": str(fresh_selection_claim_path(arguments.contract)),
        "claimSha256": claim["claimSha256"],
        "contractDeclaredSha256": claim["contractDeclaredSha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
