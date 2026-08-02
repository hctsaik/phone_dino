from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCENARIOS: dict[str, dict[str, object]] = {
    "pass": {"outcome": "pass", "globalDistance": 0.1, "nearestGoldenDistance": 0.1, "uncertainty": 0.05},
    "fail": {"outcome": "fail", "globalDistance": 0.7, "nearestGoldenDistance": 0.7, "uncertainty": 0.05},
    "review": {"outcome": "review", "globalDistance": 0.35, "nearestGoldenDistance": 0.35, "uncertainty": 0.1},
    "recapture": {"outcome": "recapture", "reasonCodes": ["STILL_GATE_REJECTED"]},
    "system-error": {"outcome": "system-error"},
}


def prepare(input_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for scenario, record in SCENARIOS.items():
        source = input_dir / f"{scenario}.png"
        if not source.is_file():
            raise FileNotFoundError(f"required fixture is missing: {source}")
        raw_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = output_dir / f"{raw_sha256}.json"
        destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare hash-addressed phone_dino engineering fixtures")
    parser.add_argument("input_dir", type=Path, help="Directory containing pass/fail/review/recapture/system-error.png")
    parser.add_argument("output_dir", type=Path, help="Directory in which hash-addressed JSON records are written")
    args = parser.parse_args()
    for path in prepare(args.input_dir.resolve(), args.output_dir.resolve()):
        print(path)


if __name__ == "__main__":
    main()

