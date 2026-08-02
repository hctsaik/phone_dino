from __future__ import annotations

import hashlib
import json

from phone_dino.fixture_cli import SCENARIOS, prepare


def test_prepare_creates_hash_addressed_records(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    for scenario in SCENARIOS:
        (source / f"{scenario}.png").write_bytes(f"image:{scenario}".encode())

    written = prepare(source, output)

    assert len(written) == 5
    for scenario, expected in SCENARIOS.items():
        digest = hashlib.sha256(f"image:{scenario}".encode()).hexdigest()
        assert json.loads((output / f"{digest}.json").read_text(encoding="utf-8")) == expected

