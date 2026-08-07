from pathlib import Path


def test_real_dino_launcher_reads_the_phonecv_topology() -> None:
    launcher = (Path(__file__).parents[1] / "start-engineering-real-dino.ps1").read_text(encoding="utf-8")

    assert "local-engineering-topology.json" in launcher
    assert "phoneDinoReal" in launcher
    assert 'ENGINEERING_REAL_DINO' in launcher
    assert "--port 8082" not in launcher
    assert 'PHONE_DINO_DEPTH_ANYTHING_ENCODER = "vits"' in launcher
    assert "PHONE_DINO_DEPTH_ANYTHING_REPOSITORY_VERSION" in launcher
    assert "PHONE_DINO_DEPTH_ANYTHING_WEIGHTS_SHA256" in launcher
