"""Materialize deterministic normal-only MVTec camera augmentations outside Git."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_research import generate_normal_augmentations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline, non-production MVTec normal camera/lighting augmentations"
    )
    parser.add_argument("manifest", type=Path, help="frozen phone-dino.mvtec-ad-smoke/1.0 manifest")
    parser.add_argument("output_dir", type=Path, help="new or empty directory outside this Git worktree")
    parser.add_argument(
        "--recipe",
        type=Path,
        default=REPOSITORY_ROOT / "tools" / "mvtec_ad_camera_lighting_recipe_v1.json",
    )
    parser.add_argument("--variants-per-parent", type=int, default=1)
    arguments = parser.parse_args()
    document = generate_normal_augmentations(
        arguments.manifest,
        arguments.recipe,
        arguments.output_dir,
        variants_per_parent=arguments.variants_per_parent,
    )
    print(json.dumps({
        "output": str(arguments.output_dir / "augmentation_manifest.json"),
        "records": len(document["records"]),
        "authoritative": document["authoritative"],
        "blindPolicy": document["blindPolicy"],
    }, indent=2))


if __name__ == "__main__":
    main()
