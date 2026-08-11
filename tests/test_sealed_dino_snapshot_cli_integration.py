from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_tool(relative_path: str, *, module_name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _FakeActivation:
    def __init__(self, sealed: SimpleNamespace, events: list[str]) -> None:
        self.snapshot = sealed
        self._events = events
        self.entered = False

    def __enter__(self) -> "_FakeActivation":
        self.entered = True
        self._events.append("activation-enter")
        return self

    def __exit__(self, *_args: object) -> None:
        self.entered = False
        self._events.append("activation-exit")


class _FakeSealedSnapshot(SimpleNamespace):
    def __init__(self, *, repository: Path, weights: Path, events: list[str]) -> None:
        super().__init__(repository=repository, weights=weights)
        self._events = events
        self.activation: _FakeActivation | None = None

    def activate(self, *, expected_manifest_sha256: str, repository_root: Path) -> _FakeActivation:
        assert expected_manifest_sha256 == "sha256:" + "f" * 64
        assert repository_root == REPOSITORY_ROOT
        self._events.append("activate")
        self.activation = _FakeActivation(self, self._events)
        return self.activation


def test_materialize_cli_requires_explicit_pins_and_uses_worktree_model_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool(
        "tools/materialize_sealed_dino_snapshot.py",
        module_name="test_materialize_sealed_dino_snapshot_tool",
        monkeypatch=monkeypatch,
    )
    captured: dict[str, Any] = {}
    fake = SimpleNamespace(
        root=tmp_path / "external" / "snapshot",
        manifest_sha256="sha256:" + "a" * 64,
        repository_sha256="sha256:" + "b" * 64,
        weights_sha256="sha256:" + "c" * 64,
    )

    def materialize(
        source_repository: Path,
        source_weights: Path,
        output_directory: Path,
        **kwargs: object,
    ) -> SimpleNamespace:
        captured.update(
            source_repository=source_repository,
            source_weights=source_weights,
            output_directory=output_directory,
            **kwargs,
        )
        return fake

    monkeypatch.setattr(tool.sealed_snapshot, "materialize_sealed_dino_snapshot", materialize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(tool.__file__),
            "--expected-repository-sha256",
            "sha256:" + "1" * 64,
            "--expected-weights-sha256",
            "sha256:" + "2" * 64,
            "--output",
            str(tmp_path / "external" / "snapshot"),
        ],
    )

    tool.main()

    assert captured == {
        "source_repository": REPOSITORY_ROOT / "runtime" / "models" / "dinov2",
        "source_weights": REPOSITORY_ROOT / "runtime" / "models" / "dinov2_vits14_pretrain.pth",
        "output_directory": tmp_path / "external" / "snapshot",
        "expected_repository_sha256": "sha256:" + "1" * 64,
        "expected_weights_sha256": "sha256:" + "2" * 64,
        "repository_root": REPOSITORY_ROOT,
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed["snapshotManifestSha256"] == fake.manifest_sha256
    assert printed["snapshotRepositorySha256"] == fake.repository_sha256
    assert printed["snapshotWeightsSha256"] == fake.weights_sha256


def test_v2_cli_uses_required_sealed_snapshot_for_full_run_and_provenance_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool(
        "tools/run_mvtec_ad_synthetic_stress_v2.py",
        module_name="test_run_mvtec_ad_synthetic_stress_v2_tool",
        monkeypatch=monkeypatch,
    )
    events: list[str] = []
    sealed = _FakeSealedSnapshot(
        repository=tmp_path / "external" / "snapshot" / "repository",
        weights=tmp_path / "external" / "snapshot" / "model_weights.pth",
        events=events,
    )
    identity_factory_marker = object()
    captured: dict[str, Any] = {}

    def load(path: Path, *, expected_manifest_sha256: str, repository_root: Path) -> _FakeSealedSnapshot:
        assert path == tmp_path / "external" / "snapshot"
        assert expected_manifest_sha256 == "sha256:" + "f" * 64
        assert repository_root == REPOSITORY_ROOT
        events.append("load")
        return sealed

    def identity_wrapper(base_factory: object, activation: _FakeActivation) -> object:
        assert base_factory is tool.stress._feature_extractor_identity
        assert activation is sealed.activation
        events.append("identity-wrapper")
        return identity_factory_marker

    def run(*args: object, **kwargs: object) -> dict[str, object]:
        assert sealed.activation is not None and sealed.activation.entered
        events.append("run")
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "syntheticStressReportSha256": "sha256:" + "d" * 64,
            "metricScope": "SYNTHETIC_STIMULUS_RESPONSE_ONLY",
            "realAnomalyPerformance": "NOT_ESTIMATED",
            "aggregate": {},
            "categories": {},
        }

    monkeypatch.setattr(tool.sealed_snapshot, "load_sealed_dino_snapshot", load)
    monkeypatch.setattr(tool.sealed_snapshot, "sealed_snapshot_identity_factory", identity_wrapper)
    monkeypatch.setattr(tool.stress, "run_synthetic_stress_v2", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(tool.__file__),
            "--sealed-model-snapshot", str(tmp_path / "external" / "snapshot"),
            "--expected-sealed-model-snapshot-manifest-sha256", "sha256:" + "f" * 64,
            "--parent-holdout", str(tmp_path / "chain" / "holdout.json"),
            "--parent-selection-contract", str(tmp_path / "chain" / "contract.json"),
            "--plan", str(tmp_path / "chain" / "plan.json"),
            "--envelope", str(tmp_path / "chain" / "envelope.json"),
            "--source-root", str(tmp_path / "source"),
            "--augmentation-manifest", str(tmp_path / "package" / "augmentation_manifest.json"),
            "--output", str(tmp_path / "external" / "report.json"),
        ],
    )

    tool.main()

    assert events == ["load", "activate", "activation-enter", "identity-wrapper", "run", "activation-exit"]
    assert captured["kwargs"] == {
        "source_root": tmp_path / "source",
        "recipe_path": REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
        "model_repo": sealed.repository,
        "model_weights": sealed.weights,
        "device": "cpu",
        "identity_factory": identity_factory_marker,
    }
    assert json.loads(capsys.readouterr().out)["syntheticStressReportSha256"] == "sha256:" + "d" * 64


def test_v3_cli_uses_required_sealed_snapshot_for_full_run_and_provenance_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool(
        "tools/run_mvtec_ad_synthetic_nuisance_control_v3.py",
        module_name="test_run_mvtec_ad_synthetic_nuisance_control_v3_tool",
        monkeypatch=monkeypatch,
    )
    events: list[str] = []
    sealed = _FakeSealedSnapshot(
        repository=tmp_path / "external" / "snapshot" / "repository",
        weights=tmp_path / "external" / "snapshot" / "model_weights.pth",
        events=events,
    )
    identity_factory_marker = object()
    captured: dict[str, Any] = {}

    def load(path: Path, *, expected_manifest_sha256: str, repository_root: Path) -> _FakeSealedSnapshot:
        assert path == tmp_path / "external" / "snapshot"
        assert expected_manifest_sha256 == "sha256:" + "f" * 64
        assert repository_root == REPOSITORY_ROOT
        events.append("load")
        return sealed

    def identity_wrapper(base_factory: object, activation: _FakeActivation) -> object:
        assert base_factory is tool.nuisance_control._feature_extractor_identity
        assert activation is sealed.activation
        events.append("identity-wrapper")
        return identity_factory_marker

    def run(*args: object, **kwargs: object) -> dict[str, object]:
        assert sealed.activation is not None and sealed.activation.entered
        events.append("run")
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "syntheticNuisanceControlReportSha256": "sha256:" + "e" * 64,
            "metricScope": "SYNTHETIC_NUISANCE_CONTROL_RESPONSE_ONLY",
            "evidenceClass": "SYNTHETIC_ENGINEERING_ONLY",
            "realAnomalyPerformance": "NOT_ESTIMATED",
            "realPrecisionRecall": "NOT_ESTIMATED",
            "aggregate": {},
        }

    monkeypatch.setattr(tool.sealed_snapshot, "load_sealed_dino_snapshot", load)
    monkeypatch.setattr(tool.sealed_snapshot, "sealed_snapshot_identity_factory", identity_wrapper)
    monkeypatch.setattr(tool.nuisance_control, "run_synthetic_nuisance_control_v3", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(tool.__file__),
            "--sealed-model-snapshot", str(tmp_path / "external" / "snapshot"),
            "--expected-sealed-model-snapshot-manifest-sha256", "sha256:" + "f" * 64,
            "--parent-holdout", str(tmp_path / "chain" / "holdout.json"),
            "--parent-selection-contract", str(tmp_path / "chain" / "contract.json"),
            "--plan", str(tmp_path / "chain" / "plan.json"),
            "--envelope", str(tmp_path / "chain" / "envelope.json"),
            "--source-root", str(tmp_path / "source"),
            "--stimulus-augmentation-manifest", str(tmp_path / "stimulus" / "augmentation_manifest.json"),
            "--capture-control-augmentation-manifest", str(tmp_path / "controls" / "augmentation_manifest.json"),
            "--registry-root", str(tmp_path / "external" / "registry"),
            "--output", str(tmp_path / "external" / "report.json"),
        ],
    )

    tool.main()

    assert events == ["load", "activate", "activation-enter", "identity-wrapper", "run", "activation-exit"]
    assert captured["kwargs"] == {
        "source_root": tmp_path / "source",
        "stimulus_recipe_path": REPOSITORY_ROOT / "tools" / "mvtec_ad_synthetic_anomaly_stress_recipe_v2.json",
        "capture_control_recipe_path": REPOSITORY_ROOT / "tools" / "mvtec_ad_successor_fit_camera_recipe_v2.json",
        "registry_root": tmp_path / "external" / "registry",
        "model_repo": sealed.repository,
        "model_weights": sealed.weights,
        "device": "cpu",
        "identity_factory": identity_factory_marker,
    }
    assert json.loads(capsys.readouterr().out)["syntheticNuisanceControlReportSha256"] == "sha256:" + "e" * 64
