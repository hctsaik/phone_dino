"""Generate a post-V1 synthetic-only MVTec stress augmentation package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_synthetic_anomaly_stress_v2 import generate_synthetic_anomaly_stress_v2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 108 deterministic post-V1 synthetic stress stimuli from sealed successor-V2 FIT parents only. "
            "This is response-only synthetic engineering research, not real-anomaly performance, comparison, promotion, or production authorization."
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
    parser.add_argument("--recipe", required=True, type=Path, help="Closed post-V1 synthetic-only stress recipe")
    parser.add_argument("--output", required=True, type=Path, help="New external output directory; it must not exist")
    arguments = parser.parse_args()
    document = generate_synthetic_anomaly_stress_v2(
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
        "fitQueryParentCount": len({record["parentCaseId"] for record in records}),
        "syntheticStimulusRecordCount": len(records),
        "renderIntensityLevels": sorted({record["renderIntensityLevel"] for record in records}),
        "syntheticDefectFamilies": sorted({record["syntheticDefectFamily"] for record in records}),
        "augmentationManifestSha256": document["augmentationManifestSha256"],
        "syntheticOnly": document["syntheticOnly"],
        "postV1Exploratory": document["postV1Exploratory"],
        "comparisonOrPromotionAllowed": document["comparisonOrPromotionAllowed"],
        "authoritative": document["authoritative"],
        "productionAuthorized": document["productionAuthorized"],
        "resultLabel": document["resultLabel"],
    }, indent=2))


if __name__ == "__main__":
    main()
