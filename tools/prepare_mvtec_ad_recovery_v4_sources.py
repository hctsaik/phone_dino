"""Freeze and acquire the closed Recovery V4 MVTec source set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino import mvtec_recovery_v4_source_acquisition as recovery


def _print_document(document: dict[str, object], *, output: Path, digest_field: str) -> None:
    print(json.dumps({
        "output": str(output),
        "declaredSha256": document[digest_field],
        "authoritative": document["authoritative"],
        "productionAuthorized": document["productionAuthorized"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recovery V4 source foundation. Planning is JSON-only; acquisition downloads only the 96 exact "
            "tracked allowlist paths into a new external root."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ledger = commands.add_parser("freeze-non-overlap-ledger", help="freeze JSON-only history/quarantine exclusions")
    ledger.add_argument("--historical-usage-ledger", required=True, type=Path)
    ledger.add_argument("--quarantine-incident", required=True, type=Path)
    ledger.add_argument("--quarantined-cohort-manifest", required=True, type=Path)
    ledger.add_argument("--output", required=True, type=Path)

    plan = commands.add_parser("create-source-plan", help="validate metadata and freeze the exact allowlist plan")
    plan.add_argument("--source-metadata", required=True, type=Path)
    plan.add_argument("--non-overlap-ledger", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)

    acquire = commands.add_parser("acquire", help="resolve/download only the frozen exact source paths")
    acquire.add_argument("--plan", required=True, type=Path)
    acquire.add_argument("--non-overlap-ledger", required=True, type=Path)
    acquire.add_argument("--source-root", required=True, type=Path)
    acquire.add_argument("--source-manifest-output", required=True, type=Path)
    acquire.add_argument("--timeout-seconds", type=float, default=60.0)

    arguments = parser.parse_args()
    if arguments.command == "freeze-non-overlap-ledger":
        document = recovery.freeze_recovery_v4_non_overlap_metadata_ledger(
            arguments.historical_usage_ledger,
            arguments.quarantine_incident,
            arguments.quarantined_cohort_manifest,
            arguments.output,
        )
        _print_document(
            document,
            output=arguments.output,
            digest_field="recoveryV4NonOverlapMetadataLedgerSha256",
        )
    elif arguments.command == "create-source-plan":
        document = recovery.create_recovery_v4_source_acquisition_plan(
            arguments.source_metadata,
            arguments.non_overlap_ledger,
            arguments.output,
        )
        _print_document(
            document,
            output=arguments.output,
            digest_field="recoveryV4SourceAcquisitionPlanSha256",
        )
    elif arguments.command == "acquire":
        document = recovery.acquire_recovery_v4_sources(
            arguments.plan,
            arguments.non_overlap_ledger,
            arguments.source_root,
            arguments.source_manifest_output,
            timeout_seconds=arguments.timeout_seconds,
        )
        _print_document(
            document,
            output=arguments.source_manifest_output,
            digest_field="recoveryV4AcquiredSourceManifestSha256",
        )
    else:  # pragma: no cover - argparse's required subcommand is exhaustive.
        raise AssertionError("unknown Recovery V4 source command")


if __name__ == "__main__":
    main()
