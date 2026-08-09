"""Download only fresh pinned MVTec AD train/good files into an external root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_normal_source_acquisition import acquire_fresh_normal_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire only pinned MVTec train/good source bytes. The source root must be a new external directory."
        )
    )
    parser.add_argument("--source-metadata", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--candidate-output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    arguments = parser.parse_args()
    document = acquire_fresh_normal_sources(
        arguments.source_metadata,
        arguments.ledger,
        arguments.source_root,
        arguments.candidate_output,
        workers=arguments.workers,
        timeout_seconds=arguments.timeout_seconds,
    )
    counts: dict[str, int] = {}
    for record in document["records"]:
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    print(json.dumps({
        "candidateOutput": str(arguments.candidate_output),
        "sourceRoot": str(arguments.source_root),
        "recordsByCategory": counts,
        "candidateManifestSha256": document["candidateManifestSha256"],
        "authoritative": document["authoritative"],
        "productionAuthorized": document["productionAuthorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
