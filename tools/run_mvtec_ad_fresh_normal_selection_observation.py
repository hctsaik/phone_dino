"""Consume the one fresh NORMAL_SELECTION partition under its fixed receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_fresh_normal_observation import (  # noqa: E402
    fresh_selection_observation_path,
    fresh_selection_receipt_path,
    run_fresh_normal_selection_observation,
)
from phone_dino.mvtec_fresh_normal_selection import load_validated_fresh_selection_contract  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the fresh selection receipt, then score every frozen candidate on raw NORMAL_SELECTION once."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--augmentation-manifest", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2")
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2_vits14_pretrain.pth",
    )
    arguments = parser.parse_args()
    observation = run_fresh_normal_selection_observation(
        arguments.contract,
        arguments.holdout,
        arguments.augmentation_manifest,
        arguments.recipe,
        source_root=arguments.source_root,
        model_repo=arguments.model_repo,
        model_weights=arguments.model_weights,
    )
    contract, _ = load_validated_fresh_selection_contract(arguments.contract)
    print(json.dumps({
        "receipt": str(fresh_selection_receipt_path(contract)),
        "observation": str(fresh_selection_observation_path(contract)),
        "selectionObservationSha256": observation["selectionObservationSha256"],
        "candidateIds": [item["candidateId"] for item in observation["candidateObservations"]],
    }, indent=2))


if __name__ == "__main__":
    main()
