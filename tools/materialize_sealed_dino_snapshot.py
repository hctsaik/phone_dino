"""Create one external, hash-pinned local DINO snapshot explicitly.

The default source paths are the checked-in local engineering model layout.
They are inputs only: this command never mutates them, and its output must be
an unused directory outside this Git worktree. Retain the printed manifest
digest outside that directory and supply it as the approved pin to an audit
CLI; a snapshot is not trusted merely because it can self-validate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phone_dino import sealed_dino_snapshot as sealed_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize exactly one new external sealed DINOv2 snapshot from local pinned source bytes. "
            "Expected source digests are required; retain the printed snapshot manifest SHA-256 as the next command's approved pin."
        )
    )
    parser.add_argument(
        "--source-model-repository",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2",
        help="Local DINO repository source (defaults to the engineering worktree path).",
    )
    parser.add_argument(
        "--source-model-weights",
        type=Path,
        default=REPOSITORY_ROOT / "runtime" / "models" / "dinov2_vits14_pretrain.pth",
        help="Local DINO weights source (defaults to the engineering worktree path).",
    )
    parser.add_argument(
        "--expected-repository-sha256",
        required=True,
        help="Pre-approved SHA-256 pin for the included local DINO repository content.",
    )
    parser.add_argument(
        "--expected-weights-sha256",
        required=True,
        help="Pre-approved SHA-256 pin for the local DINO weights file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Unused external directory to consume as the immutable sealed snapshot slot.",
    )
    arguments = parser.parse_args()
    snapshot = sealed_snapshot.materialize_sealed_dino_snapshot(
        arguments.source_model_repository,
        arguments.source_model_weights,
        arguments.output,
        expected_repository_sha256=arguments.expected_repository_sha256,
        expected_weights_sha256=arguments.expected_weights_sha256,
        repository_root=REPOSITORY_ROOT,
    )
    print(
        json.dumps(
            {
                "output": str(snapshot.root),
                "schemaVersion": sealed_snapshot.SEALED_DINO_SNAPSHOT_SCHEMA,
                "snapshotManifestSha256": snapshot.manifest_sha256,
                "snapshotRepositorySha256": snapshot.repository_sha256,
                "snapshotWeightsSha256": snapshot.weights_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
