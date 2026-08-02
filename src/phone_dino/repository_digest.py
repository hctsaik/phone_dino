from __future__ import annotations

import argparse
from pathlib import Path

from .security import digest_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Content-address an approved local model source tree")
    parser.add_argument("repository", type=Path)
    arguments = parser.parse_args()
    try:
        print(digest_directory(arguments.repository.resolve()))
    except OSError as exc:
        raise SystemExit(f"Model repository digest failed: {exc}") from exc
