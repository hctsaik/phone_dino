"""Allocate one frozen external MVTec normal holdout from a predeclared plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_normal_holdout import build_normal_holdout_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allocate an external normal-only holdout once; it cannot consume blind or anomaly records."
    )
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    document = build_normal_holdout_manifest(
        arguments.pool,
        arguments.ledger,
        arguments.plan,
        arguments.output,
        source_root=arguments.source_root,
        source_metadata_path=arguments.source_metadata,
    )
    counts: dict[str, int] = {}
    for record in document["records"]:
        counts[record["partition"]] = counts.get(record["partition"], 0) + 1
    print(json.dumps({
        "output": str(arguments.output),
        "partitionRecordCounts": counts,
        "normalHoldoutManifestSha256": document["normalHoldoutManifestSha256"],
        "authoritative": document["authoritative"],
        "productionAuthorized": document["productionAuthorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
