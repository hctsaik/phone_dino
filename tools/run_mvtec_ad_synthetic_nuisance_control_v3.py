"""Run the external V3 paired generic-capture nuisance-control audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino import mvtec_synthetic_nuisance_control_v3 as nuisance_control
from phone_dino import mvtec_cohort_quarantine as quarantine
from phone_dino import sealed_dino_snapshot as sealed_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one offline V3 paired synthetic-stimulus vs generic-capture-control audit. "
            "It is response-only evidence and does not estimate real anomaly performance."
        )
    )
    parser.add_argument(
        "--sealed-model-snapshot",
        required=True,
        type=Path,
        help="External, hash-validated sealed DINO snapshot directory required for all DINO work.",
    )
    parser.add_argument(
        "--expected-sealed-model-snapshot-manifest-sha256",
        required=True,
        help="Independently approved SHA-256 pin for the sealed snapshot manifest.",
    )
    parser.add_argument("--parent-holdout", required=True, type=Path)
    parser.add_argument(
        "--cohort-quarantine-incident",
        required=True,
        type=Path,
        help=(
            "External immutable JSON-only incident record for the permanently quarantined "
            "fresh_normal_holdout_v1 cohort."
        ),
    )
    parser.add_argument(
        "--expected-cohort-quarantine-incident-sha256",
        required=True,
        help="Independently retained SHA-256 pin for the immutable cohort-quarantine incident.",
    )
    parser.add_argument("--parent-selection-contract", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--stimulus-augmentation-manifest", required=True, type=Path)
    parser.add_argument("--capture-control-augmentation-manifest", required=True, type=Path)
    parser.add_argument(
        "--stimulus-recipe",
        type=Path,
        default=REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
    )
    parser.add_argument(
        "--capture-control-recipe",
        type=Path,
        default=REPOSITORY_ROOT / "tools" / "mvtec_ad_successor_fit_camera_recipe_v2.json",
    )
    parser.add_argument(
        "--registry-root",
        required=True,
        type=Path,
        help="External persistent one-time receipt directory; never put it in the Git worktree.",
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    # Fail before snapshot activation as well as inside the direct API.  The
    # duplicate API-level gate closes callers that do not use this CLI.
    quarantine.assert_v3_parent_holdout_not_quarantined(
        arguments.parent_holdout,
        quarantine_incident_path=arguments.cohort_quarantine_incident,
        expected_quarantine_incident_sha256=arguments.expected_cohort_quarantine_incident_sha256,
        repository_root=REPOSITORY_ROOT,
    )
    sealed = sealed_snapshot.load_sealed_dino_snapshot(
        arguments.sealed_model_snapshot,
        expected_manifest_sha256=arguments.expected_sealed_model_snapshot_manifest_sha256,
        repository_root=REPOSITORY_ROOT,
    )
    # The complete audit (including raw and child feature extraction) stays
    # inside the active sealed snapshot.  Its evidence identity therefore
    # cannot silently name a mutable worktree model.
    with sealed.activate(
        expected_manifest_sha256=arguments.expected_sealed_model_snapshot_manifest_sha256,
        repository_root=REPOSITORY_ROOT,
    ) as activation:
        report = nuisance_control.run_synthetic_nuisance_control_v3(
            arguments.parent_holdout,
            arguments.parent_selection_contract,
            arguments.plan,
            arguments.envelope,
            arguments.stimulus_augmentation_manifest,
            arguments.capture_control_augmentation_manifest,
            arguments.output,
            source_root=arguments.source_root,
            stimulus_recipe_path=arguments.stimulus_recipe,
            capture_control_recipe_path=arguments.capture_control_recipe,
            registry_root=arguments.registry_root,
            quarantine_incident_path=arguments.cohort_quarantine_incident,
            expected_quarantine_incident_sha256=arguments.expected_cohort_quarantine_incident_sha256,
            model_repo=activation.snapshot.repository,
            model_weights=activation.snapshot.weights,
            device="cpu",
            identity_factory=sealed_snapshot.sealed_snapshot_identity_factory(
                nuisance_control._feature_extractor_identity,
                activation,
            ),
        )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "syntheticNuisanceControlReportSha256": report["syntheticNuisanceControlReportSha256"],
                "metricScope": report["metricScope"],
                "evidenceClass": report["evidenceClass"],
                "realAnomalyPerformance": report["realAnomalyPerformance"],
                "realPrecisionRecall": report["realPrecisionRecall"],
                "aggregate": report["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
