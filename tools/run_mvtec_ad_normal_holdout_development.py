"""Run one standalone fresh normal-holdout development observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_normal_holdout_evaluator import run_development_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one FIT-plus-raw-tuning development-only fresh normal-holdout observation."
    )
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--augmentation-manifest", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-repo", type=Path, default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2")
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2_vits14_pretrain.pth",
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--max-prototype-patches", type=int, required=True)
    parser.add_argument("--top-k-most-anomalous-patches", type=int, default=5)
    parser.add_argument("--prototype-block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    arguments = parser.parse_args()
    configuration = {
        "id": arguments.candidate_id,
        "algorithmId": "DINOV2_PATCH_NEAREST_NORMAL_COSINE_TOPK_V1",
        "memoryBankSelection": "DETERMINISTIC_EVENLY_SPACED_PATCH_SUBSET_AFTER_CASEID_SORT",
        "maxPrototypePatches": arguments.max_prototype_patches,
        "topKMostAnomalousPatches": arguments.top_k_most_anomalous_patches,
        "prototypeBlockSize": arguments.prototype_block_size,
        "batchSize": arguments.batch_size,
    }
    report = run_development_evaluation(
        arguments.holdout,
        arguments.augmentation_manifest,
        arguments.recipe,
        arguments.output,
        source_root=arguments.source_root,
        model_repo=arguments.model_repo,
        model_weights=arguments.model_weights,
        device="cpu",
        candidate_configuration=configuration,
    )
    print(json.dumps({
        "output": str(arguments.output),
        "candidateConfigurationSha256": report["candidateConfigurationSha256"],
        "developmentReportSha256": report["developmentReportSha256"],
        "thresholds": report["thresholds"],
        "categories": report["categories"],
    }, indent=2))


if __name__ == "__main__":
    main()
