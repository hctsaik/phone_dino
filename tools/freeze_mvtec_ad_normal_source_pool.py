"""Freeze a rehashed external MVTec train/good source pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino.mvtec_normal_holdout import freeze_normal_source_pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an external, pinned MVTec train/good source pool; never writes into Git."
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    document = freeze_normal_source_pool(
        arguments.candidates,
        arguments.output,
        source_root=arguments.source_root,
        source_metadata_path=arguments.source_metadata,
    )
    print(json.dumps({
        "output": str(arguments.output),
        "records": len(document["records"]),
        "normalSourcePoolSha256": document["normalSourcePoolSha256"],
        "authoritative": document["authoritative"],
        "productionAuthorized": document["productionAuthorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
