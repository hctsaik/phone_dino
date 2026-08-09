"""Run one immutable reserve-successor V2 development observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_successor_evaluator_v2 import (
    PRE_REGISTERED_CANDIDATES,
    RAW_FIT_PLUS_AUGMENTATION_R3,
    pre_registered_candidate_configuration,
    run_successor_v2_development_evaluation,
    run_successor_v2_development_evaluations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pre-registered reserve-successor V2 FIT-plus-raw-tuning "
            "development-only observation. This command accepts no selection or confirmation image input."
        )
    )
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, action="append", type=Path)
    parser.add_argument(
        "--candidate-id",
        required=True,
        action="append",
        choices=[item["id"] for item in PRE_REGISTERED_CANDIDATES],
    )
    parser.add_argument("--augmentation-manifest", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--model-repo", type=Path, default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2")
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2_vits14_pretrain.pth",
    )
    arguments = parser.parse_args()
    if len(arguments.output) != len(arguments.candidate_id):
        parser.error("provide exactly one --output for each --candidate-id")
    if len(arguments.candidate_id) != len(set(arguments.candidate_id)):
        parser.error("each --candidate-id may be supplied at most once")
    configurations = [pre_registered_candidate_configuration(identifier) for identifier in arguments.candidate_id]
    needs_augmentation = any(
        configuration["prototypeInputPolicy"] == RAW_FIT_PLUS_AUGMENTATION_R3 for configuration in configurations
    )
    if needs_augmentation != (arguments.augmentation_manifest is not None and arguments.recipe is not None):
        parser.error(
            "a set containing R3 candidates requires both --augmentation-manifest and --recipe; "
            "a raw-only invocation must receive neither"
        )
    if len(configurations) == 1:
        configuration = configurations[0]
        report = run_successor_v2_development_evaluation(
            arguments.parent_holdout,
            arguments.parent_selection_contract,
            arguments.plan,
            arguments.envelope,
            arguments.output[0],
            source_root=arguments.source_root,
            model_repo=arguments.model_repo,
            model_weights=arguments.model_weights,
            device="cpu",
            candidate_configuration=configuration,
            augmentation_manifest_path=arguments.augmentation_manifest,
            recipe_path=arguments.recipe,
        )
        reports = {configuration["id"]: report}
    else:
        reports = run_successor_v2_development_evaluations(
            arguments.parent_holdout,
            arguments.parent_selection_contract,
            arguments.plan,
            arguments.envelope,
            {configuration["id"]: output for configuration, output in zip(configurations, arguments.output, strict=True)},
            source_root=arguments.source_root,
            model_repo=arguments.model_repo,
            model_weights=arguments.model_weights,
            device="cpu",
            candidate_configurations=configurations,
            augmentation_manifest_path=arguments.augmentation_manifest,
            recipe_path=arguments.recipe,
        )
    print(json.dumps({
        identifier: {
            "output": str(next(output for configuration, output in zip(configurations, arguments.output, strict=True) if configuration["id"] == identifier)),
            "candidateConfigurationSha256": report["candidateConfigurationSha256"],
            "developmentReportSha256": report["developmentReportSha256"],
            "thresholds": report["thresholds"],
            "categories": report["categories"],
        }
        for identifier, report in reports.items()
    }, indent=2))


if __name__ == "__main__":
    main()
