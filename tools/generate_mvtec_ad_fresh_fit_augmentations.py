"""Generate an external, FIT-only fresh-cohort MVTec camera package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_fresh_fit_augmentation import generate_fresh_fit_augmentations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic FIT-only generic camera variants from a frozen fresh normal holdout."
    )
    parser.add_argument("--holdout", required=True, type=Path, help="External frozen normal_holdout.json")
    parser.add_argument("--source-root", required=True, type=Path, help="External root containing only frozen source bytes")
    parser.add_argument("--recipe", required=True, type=Path, help="Closed fresh FIT V1 camera recipe")
    parser.add_argument("--variants-per-parent", required=True, type=int, help="1 through 8 deterministic variants per FIT parent")
    parser.add_argument("--output", required=True, type=Path, help="Fresh external output directory; it must not exist")
    arguments = parser.parse_args()
    document = generate_fresh_fit_augmentations(
        arguments.holdout,
        arguments.source_root,
        arguments.recipe,
        arguments.output,
        variants_per_parent=arguments.variants_per_parent,
    )
    records = document["records"]
    print(json.dumps({
        "output": str(arguments.output),
        "parentPartition": document["parentPartition"],
        "fitParentCount": len({record["parentCaseId"] for record in records}),
        "augmentationRecordCount": len(records),
        "variantsPerParent": document["variantsPerParent"],
        "augmentationManifestSha256": document["augmentationManifestSha256"],
        "authoritative": document["authoritative"],
        "productionAuthorized": document["productionAuthorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
