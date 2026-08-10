"""Consume the sealed V2 selection partition once after FIT/R3 preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_successor_selection_v2 import run_successor_v2_selection_observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-time aggregate reserve-successor V2 normal selection observation.")
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--augmentation-manifest", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--model-repo", type=Path, default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2")
    parser.add_argument("--model-weights", type=Path, default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2_vits14_pretrain.pth")
    arguments = parser.parse_args()
    observation = run_successor_v2_selection_observation(
        arguments.parent_holdout, arguments.parent_selection_contract, arguments.plan, arguments.envelope,
        arguments.contract, arguments.augmentation_manifest, arguments.recipe, source_root=arguments.source_root,
        model_repo=arguments.model_repo, model_weights=arguments.model_weights,
    )
    print(json.dumps({"selectionObservationSha256": observation["selectionObservationSha256"], "queryInputCount": observation["normalOnlyEvidence"]["queryInputCount"]}, indent=2))


if __name__ == "__main__":
    main()
