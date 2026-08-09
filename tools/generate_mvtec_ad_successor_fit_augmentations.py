"""Generate the external R3 successor-V2 FIT-only camera package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_successor_fit_augmentation_v2 import generate_successor_fit_augmentations


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic R3 component-separated generic camera variants from the sealed successor FIT "
            "partition only. This is offline exploratory research, not device calibration or production authorization."
        )
    )
    parser.add_argument("--parent-holdout", required=True, type=Path, help="External closed parent normal_holdout.json")
    parser.add_argument(
        "--parent-selection-contract",
        required=True,
        type=Path,
        help="External closed schema-1.1 parent selection contract",
    )
    parser.add_argument("--plan", required=True, type=Path, help="External closed successor allocation plan")
    parser.add_argument("--envelope", required=True, type=Path, help="External closed successor phase envelope")
    parser.add_argument("--source-root", required=True, type=Path, help="External root containing sealed parent source bytes")
    parser.add_argument("--recipe", required=True, type=Path, help="Closed successor V2 R3 camera recipe")
    parser.add_argument("--output", required=True, type=Path, help="New external output directory; it must not exist")
    arguments = parser.parse_args()
    document = generate_successor_fit_augmentations(
        arguments.parent_holdout,
        arguments.parent_selection_contract,
        arguments.plan,
        arguments.envelope,
        source_root=arguments.source_root,
        recipe_path=arguments.recipe,
        output_dir=arguments.output,
    )
    records = document["records"]
    print(json.dumps({
        "output": str(arguments.output),
        "parentPartition": document["parentPartition"],
        "fitParentCount": len({record["parentCaseId"] for record in records}),
        "augmentationRecordCount": len(records),
        "variantsPerParent": document["variantsPerParent"],
        "components": sorted({record["component"] for record in records}),
        "augmentationManifestSha256": document["augmentationManifestSha256"],
        "authoritative": document["authoritative"],
        "productionAuthorized": document["productionAuthorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
