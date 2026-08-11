from __future__ import annotations

import importlib
import json
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from phone_dino import mvtec_synthetic_stress_v2 as stress
from phone_dino import sealed_dino_snapshot as snapshot
from phone_dino.security import digest_directory


def _source_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    worktree = tmp_path / "phone-dino-worktree"
    worktree.mkdir(parents=True)
    repository = tmp_path / "external-dinov2"
    (repository / "dinov2").mkdir(parents=True)
    (repository / "hubconf.py").write_text(
        "from dinov2 import marker\n\n"
        "def dinov2_vits14(*, pretrained=False):\n"
        "    return marker\n",
        encoding="utf-8",
    )
    (repository / "dinov2" / "__init__.py").write_text("marker = object()\n", encoding="utf-8")
    (repository / "dinov2" / "layers.py").write_text("LAYER = 'sealed'\n", encoding="utf-8")
    weights = tmp_path / "external-weights.pth"
    weights.write_bytes(b"sealed-dinov2-fixture-weights\x00\x01")
    return worktree, repository, weights


def _worktree_source_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create the real-style ``runtime/models`` layout inside a worktree."""

    worktree = tmp_path / "phone-dino-worktree"
    repository = worktree / "runtime" / "models" / "dinov2"
    (repository / "dinov2").mkdir(parents=True)
    (repository / "hubconf.py").write_text(
        "from dinov2 import marker\n\n"
        "def dinov2_vits14(*, pretrained=False):\n"
        "    return marker\n",
        encoding="utf-8",
    )
    (repository / "dinov2" / "__init__.py").write_text("marker = object()\n", encoding="utf-8")
    weights = worktree / "runtime" / "models" / "dinov2_vits14_pretrain.pth"
    weights.write_bytes(b"worktree-sealed-dinov2-fixture-weights\x00\x01")
    return worktree, repository, weights


def _materialize(tmp_path: Path, *, destination_name: str = "snapshot") -> tuple[snapshot.SealedDinoSnapshot, Path, Path, Path]:
    worktree, repository, weights = _source_layout(tmp_path)
    result = snapshot.materialize_sealed_dino_snapshot(
        repository,
        weights,
        tmp_path / destination_name,
        expected_repository_sha256=snapshot.sealed_repository_sha256(repository),
        expected_weights_sha256=snapshot.sha256_file(weights),
        repository_root=worktree,
    )
    return result, worktree, repository, weights


def test_materializes_new_cache_free_snapshot_with_self_authenticating_manifest(tmp_path: Path) -> None:
    worktree, repository, weights = _source_layout(tmp_path)
    (repository / ".git").mkdir()
    (repository / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (repository / "__pycache__").mkdir()
    cache = repository / "__pycache__" / "hubconf.cpython-311.pyc"
    cache.write_bytes(b"source-cache-must-stay-put")
    bytecode = repository / "dinov2" / "layers.pyc"
    bytecode.write_bytes(b"source-bytecode-must-stay-put")
    expected_repository = snapshot.sealed_repository_sha256(repository)
    expected_weights = snapshot.sha256_file(weights)

    sealed = snapshot.materialize_sealed_dino_snapshot(
        repository,
        weights,
        tmp_path / "sealed-output",
        expected_repository_sha256=expected_repository,
        expected_weights_sha256=expected_weights,
        repository_root=worktree,
    )

    assert digest_directory(repository) == expected_repository
    assert sealed.repository_sha256 == expected_repository
    assert sealed.weights_sha256 == expected_weights
    assert cache.read_bytes() == b"source-cache-must-stay-put"
    assert bytecode.read_bytes() == b"source-bytecode-must-stay-put"
    assert not (sealed.repository / ".git").exists()
    assert not (sealed.repository / "__pycache__").exists()
    assert not (sealed.repository / "dinov2" / "layers.pyc").exists()
    manifest = json.loads(sealed.manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    unsigned.pop("snapshotManifestSha256")
    assert manifest["snapshotManifestSha256"] == snapshot.canonical_json_sha256(unsigned)
    assert sealed.manifest_sha256 == manifest["snapshotManifestSha256"]
    assert {item["relativePath"] for item in manifest["repositoryFiles"]} == {
        "dinov2/__init__.py",
        "dinov2/layers.py",
        "hubconf.py",
    }
    assert snapshot.load_sealed_dino_snapshot(
        sealed.root,
        expected_manifest_sha256=sealed.manifest_sha256,
        repository_root=worktree,
    ) == sealed


def test_materializes_regular_model_sources_inside_worktree_but_keeps_output_external(tmp_path: Path) -> None:
    worktree, repository, weights = _worktree_source_layout(tmp_path)
    expected_repository = snapshot.sealed_repository_sha256(repository)
    expected_weights = snapshot.sha256_file(weights)

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="outside the Git working tree"):
        snapshot.materialize_sealed_dino_snapshot(
            repository,
            weights,
            worktree / "runtime" / "sealed-output",
            expected_repository_sha256=expected_repository,
            expected_weights_sha256=expected_weights,
            repository_root=worktree,
        )

    sealed = snapshot.materialize_sealed_dino_snapshot(
        repository,
        weights,
        tmp_path / "external-sealed-output",
        expected_repository_sha256=expected_repository,
        expected_weights_sha256=expected_weights,
        repository_root=worktree,
    )

    assert sealed.repository_sha256 == expected_repository
    assert sealed.weights_sha256 == expected_weights
    assert repository.is_dir()
    assert weights.is_file()


def test_rejects_link_or_reparse_source_root_inside_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree, repository, weights = _worktree_source_layout(tmp_path)
    original = snapshot._is_link_or_reparse_point

    def reported_reparse(path: Path) -> bool:
        if path.absolute() in {repository.absolute(), weights.absolute()}:
            return True
        return original(path)

    monkeypatch.setattr(snapshot, "_is_link_or_reparse_point", reported_reparse)
    with pytest.raises(snapshot.SealedDinoSnapshotError, match="symbolic link or reparse point"):
        snapshot.materialize_sealed_dino_snapshot(
            repository,
            weights,
            tmp_path / "external-sealed-output",
            expected_repository_sha256="sha256:" + "0" * 64,
            expected_weights_sha256="sha256:" + "1" * 64,
            repository_root=worktree,
        )


def test_identity_factory_binds_active_snapshot_and_rejects_model_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    for name in tuple(sys.modules):
        if name == "dinov2" or name.startswith("dinov2."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    calls: list[tuple[Path, Path, str]] = []

    def base_identity(*, model_repo: Path, model_weights: Path, device: str) -> dict[str, object]:
        calls.append((model_repo, model_weights, device))
        return {"baseIdentity": "fixture", "device": device}

    with sealed.activate(expected_manifest_sha256=sealed.manifest_sha256, repository_root=worktree) as activation:
        identity_factory = snapshot.sealed_snapshot_identity_factory(base_identity, activation)
        identity = identity_factory(
            model_repo=activation.snapshot.repository,
            model_weights=activation.snapshot.weights,
            device="cpu",
        )
        binding = identity[snapshot.SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_FIELD]
        assert binding == {
            "schemaVersion": snapshot.SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_SCHEMA,
            "snapshotSchemaVersion": snapshot.SEALED_DINO_SNAPSHOT_SCHEMA,
            "snapshotManifestSha256": sealed.manifest_sha256,
            "snapshotRepositorySha256": sealed.repository_sha256,
            "repositoryDigestAlgorithm": snapshot.SEALED_DINO_REPOSITORY_DIGEST_ALGORITHM,
            "snapshotWeightsSha256": sealed.weights_sha256,
            "weightsDigestAlgorithm": snapshot.SEALED_DINO_WEIGHTS_DIGEST_ALGORITHM,
        }
        with pytest.raises(snapshot.SealedDinoSnapshotError, match="must use the active sealed snapshot"):
            identity_factory(
                model_repo=tmp_path / "other-repository",
                model_weights=activation.snapshot.weights,
                device="cpu",
            )

    assert calls == [(sealed.repository, sealed.weights, "cpu")]


def test_sealed_identity_wrapper_matches_the_v2_report_contract_without_digest_algorithm_confusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    for name in tuple(sys.modules):
        if name == "dinov2" or name.startswith("dinov2."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    def base_identity(*, model_repo: Path, model_weights: Path, device: str) -> dict[str, object]:
        assert model_repo == sealed.repository
        assert model_weights == sealed.weights
        return {
            "schemaVersion": "phone-dino.mvtec-ad-synthetic-stimulus-feature-extractor/2.0",
            "modelWeightsSha256": "sha256:" + "1" * 64,
            # The legacy evaluator uses a distinct directory-digest algorithm;
            # it must not be compared directly to the sealed manifest digest.
            "modelRepositorySha256": "sha256:" + "2" * 64,
            "modelEntrypoint": "dinov2_vits14",
            "device": device,
            "preprocessingId": stress.knn.SUCCESSOR_V2_PREPROCESSING_ID,
            "algorithmId": stress.SYNTHETIC_STRESS_V2_ALGORITHM,
            "prototypeSelection": stress.SYNTHETIC_STRESS_V2_PROTOTYPE_SELECTION,
            "syntheticStressEvaluatorModuleSha256": "sha256:" + "3" * 64,
            "syntheticStressAugmentationModuleSha256": "sha256:" + "4" * 64,
            "successorModuleSha256": "sha256:" + "5" * 64,
            "patchKnnModuleSha256": "sha256:" + "6" * 64,
            "productionModuleSha256": "sha256:" + "7" * 64,
            "enginesModuleSha256": "sha256:" + "8" * 64,
            "pythonVersion": "fixture-python",
            "numpyVersion": "fixture-numpy",
            "torchVersion": "fixture-torch",
            "torchvisionVersion": "fixture-torchvision",
        }

    with sealed.activate(expected_manifest_sha256=sealed.manifest_sha256, repository_root=worktree) as activation:
        identity = snapshot.sealed_snapshot_identity_factory(base_identity, activation)(
            model_repo=sealed.repository,
            model_weights=sealed.weights,
            device="cpu",
        )

    sealed_provenance = identity[snapshot.SEALED_DINO_FEATURE_EXTRACTOR_PROVENANCE_FIELD]
    assert isinstance(sealed_provenance, dict)
    assert sealed_provenance["snapshotRepositorySha256"] != identity["modelRepositorySha256"]
    assert stress._validate_feature_extractor(identity) is identity


def test_rejects_existing_output_before_consuming_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree, repository, weights = _source_layout(tmp_path)
    output = tmp_path / "already-sealed"
    output.mkdir()
    expected_repository = snapshot.sealed_repository_sha256(repository)
    expected_weights = snapshot.sha256_file(weights)
    calls: list[str] = []

    def forbidden_scan(*_args: object, **_kwargs: object) -> object:
        calls.append("source")
        raise AssertionError("an existing output must reject before source scanning")

    monkeypatch.setattr(snapshot, "_scan_repository", forbidden_scan)

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="already exists"):
        snapshot.materialize_sealed_dino_snapshot(
            repository,
            weights,
            output,
            expected_repository_sha256=expected_repository,
            expected_weights_sha256=expected_weights,
            repository_root=worktree,
        )

    assert calls == []
    assert (repository / "hubconf.py").is_file()
    assert output.is_dir()


def test_expected_digest_mismatch_rejects_before_creating_output(tmp_path: Path) -> None:
    worktree, repository, weights = _source_layout(tmp_path)
    output = tmp_path / "mismatch-output"

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="repository digest"):
        snapshot.materialize_sealed_dino_snapshot(
            repository,
            weights,
            output,
            expected_repository_sha256="sha256:" + "0" * 64,
            expected_weights_sha256=snapshot.sha256_file(weights),
            repository_root=worktree,
        )

    assert not output.exists()


def test_load_requires_an_independent_manifest_pin_after_complete_slot_replacement(tmp_path: Path) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    replacement_root = tmp_path / "replacement-source"
    replacement_worktree, replacement_repository, replacement_weights = _source_layout(replacement_root)
    (replacement_repository / "dinov2" / "layers.py").write_text("LAYER = 'replacement'\n", encoding="utf-8")
    replacement = snapshot.materialize_sealed_dino_snapshot(
        replacement_repository,
        replacement_weights,
        replacement_root / "replacement-snapshot",
        expected_repository_sha256=snapshot.sealed_repository_sha256(replacement_repository),
        expected_weights_sha256=snapshot.sha256_file(replacement_weights),
        repository_root=replacement_worktree,
    )

    # Simulate an attacker replacing every self-consistent slot entry.  The
    # original external manifest pin, not its self digest, must be the trust
    # anchor when this path is loaded again.
    for relative in ("repository/dinov2/__init__.py", "repository/dinov2/layers.py", "repository/hubconf.py", "model_weights.pth", "sealed_dino_snapshot.json"):
        source = replacement.root.joinpath(*relative.split("/"))
        destination = sealed.root.joinpath(*relative.split("/"))
        destination.write_bytes(source.read_bytes())

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="approved SHA-256 pin"):
        snapshot.load_sealed_dino_snapshot(
            sealed.root,
            expected_manifest_sha256=sealed.manifest_sha256,
            repository_root=worktree,
        )

    loaded = snapshot.load_sealed_dino_snapshot(
        sealed.root,
        expected_manifest_sha256=replacement.manifest_sha256,
        repository_root=worktree,
    )
    assert loaded.manifest_sha256 == replacement.manifest_sha256


def test_activation_does_not_trust_a_handcrafted_snapshot_dataclass_pin(tmp_path: Path) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    forged = snapshot.SealedDinoSnapshot(
        root=sealed.root,
        repository=sealed.repository,
        weights=sealed.weights,
        manifest_path=sealed.manifest_path,
        repository_sha256=sealed.repository_sha256,
        weights_sha256=sealed.weights_sha256,
        manifest_sha256="sha256:" + "0" * 64,
    )

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="approved SHA-256 pin"):
        with snapshot.activate_sealed_dino_snapshot(
            forged,
            expected_manifest_sha256=forged.manifest_sha256,
            repository_root=worktree,
        ):
            pass
    assert str(sealed.repository) not in sys.path


def test_rejects_worktree_and_source_containment(tmp_path: Path) -> None:
    worktree, repository, weights = _source_layout(tmp_path)
    expected_repository = snapshot.sealed_repository_sha256(repository)
    expected_weights = snapshot.sha256_file(weights)

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="outside the Git working tree"):
        snapshot.materialize_sealed_dino_snapshot(
            repository,
            weights,
            worktree / "snapshot-under-worktree",
            expected_repository_sha256=expected_repository,
            expected_weights_sha256=expected_weights,
            repository_root=worktree,
        )
    with pytest.raises(snapshot.SealedDinoSnapshotError, match="must not contain one another"):
        snapshot.materialize_sealed_dino_snapshot(
            repository,
            weights,
            repository / "snapshot-under-source",
            expected_repository_sha256=expected_repository,
            expected_weights_sha256=expected_weights,
            repository_root=worktree,
        )
    assert not (worktree / "snapshot-under-worktree").exists()
    assert not (repository / "snapshot-under-source").exists()


def test_rejects_included_link_without_modifying_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree, repository, weights = _source_layout(tmp_path)
    linked = repository / "dinov2" / "linked.py"
    reparse_target = linked
    linked_created = False
    try:
        linked.symlink_to(repository / "dinov2" / "layers.py")
        linked_created = True
    except OSError:  # pragma: no cover - Windows developer-mode dependent
        # The same fail-closed path is exercised when Windows has no symlink
        # privilege: emulate a reparse-point report for another included file.
        original = snapshot._is_link_or_reparse_point
        reparse_target = repository / "dinov2" / "layers.py"

        def reported_reparse(path: Path) -> bool:
            if path.absolute() == reparse_target.absolute():
                return True
            return original(path)

        monkeypatch.setattr(snapshot, "_is_link_or_reparse_point", reported_reparse)

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="symbolic link or reparse"):
        snapshot.materialize_sealed_dino_snapshot(
            repository,
            weights,
            tmp_path / "link-output",
            expected_repository_sha256="sha256:" + "1" * 64,
            expected_weights_sha256=snapshot.sha256_file(weights),
            repository_root=worktree,
        )

    if linked_created:
        assert linked.is_symlink()
    else:
        assert reparse_target.read_text(encoding="utf-8") == "LAYER = 'sealed'\n"
    assert not (tmp_path / "link-output").exists()


def test_snapshot_tamper_or_cache_artifact_is_rejected_on_load(tmp_path: Path) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    (sealed.repository / "dinov2" / "layers.py").write_text("LAYER = 'tampered'\n", encoding="utf-8")

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="files do not match|digest"):
        snapshot.load_sealed_dino_snapshot(
            sealed.root,
            expected_manifest_sha256=sealed.manifest_sha256,
            repository_root=worktree,
        )

    # A separate complete snapshot demonstrates that cache/VCS artifacts are
    # forbidden after sealing too, rather than merely omitted on source copy.
    intact, worktree_two, _source_two, _weights_two = _materialize(tmp_path / "second", destination_name="sealed")
    (intact.repository / "__pycache__").mkdir()
    (intact.repository / "__pycache__" / "unexpected.pyc").write_bytes(b"tampered-cache")
    with pytest.raises(snapshot.SealedDinoSnapshotError, match="forbidden cache or VCS"):
        snapshot.load_sealed_dino_snapshot(
            intact.root,
            expected_manifest_sha256=intact.manifest_sha256,
            repository_root=worktree_two,
        )


def test_manifest_self_digest_detects_metadata_tamper(tmp_path: Path) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    manifest = json.loads(sealed.manifest_path.read_text(encoding="utf-8"))
    manifest["expectedWeightsSha256"] = "sha256:" + "0" * 64
    sealed.manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="self digest"):
        snapshot.load_sealed_dino_snapshot(
            sealed.root,
            expected_manifest_sha256=sealed.manifest_sha256,
            repository_root=worktree,
        )


def test_rejects_preloaded_dinov2_from_outside_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    foreign = tmp_path / "foreign" / "dinov2.py"
    foreign.parent.mkdir()
    foreign.write_text("foreign = True\n", encoding="utf-8")
    preloaded = types.ModuleType("dinov2")
    preloaded.__file__ = str(foreign)
    monkeypatch.setitem(sys.modules, "dinov2", preloaded)
    before = sys.dont_write_bytecode

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="refuses preloaded dinov2 modules"):
        with snapshot.activate_sealed_dino_snapshot(
            sealed,
            expected_manifest_sha256=sealed.manifest_sha256,
            repository_root=worktree,
        ):
            pass

    assert sys.dont_write_bytecode is before
    assert str(sealed.repository) not in sys.path


def test_activation_rejects_even_a_snapshot_origin_preloaded_dinov2_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    preloaded = types.ModuleType("dinov2")
    preloaded.__file__ = str(sealed.repository / "dinov2" / "__init__.py")
    monkeypatch.setitem(sys.modules, "dinov2", preloaded)

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="refuses preloaded dinov2 modules"):
        with snapshot.activate_sealed_dino_snapshot(
            sealed,
            expected_manifest_sha256=sealed.manifest_sha256,
            repository_root=worktree,
        ):
            pass


def test_activation_rejects_a_centralized_bytecode_cache_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "central-pycache"))

    with pytest.raises(snapshot.SealedDinoSnapshotError, match="centralized Python bytecode cache prefix"):
        with snapshot.activate_sealed_dino_snapshot(
            sealed,
            expected_manifest_sha256=sealed.manifest_sha256,
            repository_root=worktree,
        ):
            pass

    monkeypatch.setattr(sys, "pycache_prefix", None)
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path / "environment-central-pycache"))
    with pytest.raises(snapshot.SealedDinoSnapshotError, match="centralized Python bytecode cache prefix"):
        with snapshot.activate_sealed_dino_snapshot(
            sealed,
            expected_manifest_sha256=sealed.manifest_sha256,
            repository_root=worktree,
        ):
            pass


def test_activation_cleans_verified_dino_imports_for_a_later_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    for name in tuple(sys.modules):
        if name == "dinov2" or name.startswith("dinov2."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    with sealed.activate(expected_manifest_sha256=sealed.manifest_sha256, repository_root=worktree):
        loaded = importlib.import_module("dinov2")
        assert Path(str(loaded.__file__)).resolve() == (sealed.repository / "dinov2" / "__init__.py").resolve()

    assert not any(name == "dinov2" or name.startswith("dinov2.") for name in sys.modules)

    with sealed.activate(expected_manifest_sha256=sealed.manifest_sha256, repository_root=worktree):
        reloaded = importlib.import_module("dinov2")
        assert Path(str(reloaded.__file__)).resolve() == (sealed.repository / "dinov2" / "__init__.py").resolve()

    assert not any(name == "dinov2" or name.startswith("dinov2.") for name in sys.modules)


def test_activation_serializes_process_global_import_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    for name in tuple(sys.modules):
        if name == "dinov2" or name.startswith("dinov2."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    attempted = threading.Event()
    entered = threading.Event()
    errors: list[BaseException] = []

    def activate_in_other_thread() -> None:
        attempted.set()
        try:
            with sealed.activate(expected_manifest_sha256=sealed.manifest_sha256, repository_root=worktree):
                entered.set()
        except BaseException as error:  # pragma: no cover - assertion below inspects the thread result
            errors.append(error)

    worker = threading.Thread(target=activate_in_other_thread)
    with sealed.activate(expected_manifest_sha256=sealed.manifest_sha256, repository_root=worktree):
        worker.start()
        assert attempted.wait(timeout=2.0)
        assert not entered.wait(timeout=0.15)
    assert entered.wait(timeout=2.0)
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert errors == []


class _FakeHub:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return {"fake": "model"}


class _FakeTorch:
    def __init__(self) -> None:
        self.hub = _FakeHub()


def test_activation_sets_no_bytecode_and_torch_hub_receives_snapshot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    # Make the test independent of any DINO package a wider test process may
    # have imported; the activation itself is responsible for the safety check.
    for name in tuple(sys.modules):
        if name == "dinov2" or name.startswith("dinov2."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    fake_torch = _FakeTorch()
    before = sys.dont_write_bytecode

    with snapshot.activate_sealed_dino_snapshot(
        sealed.root,
        expected_manifest_sha256=sealed.manifest_sha256,
        repository_root=worktree,
    ) as activation:
        assert sys.dont_write_bytecode is True
        assert sys.path[0] == str(sealed.repository)
        assert activation.load_torch_hub_model(torch_module=fake_torch) == {"fake": "model"}

    assert sys.dont_write_bytecode is before
    assert str(sealed.repository) not in sys.path
    assert fake_torch.hub.calls == [
        ((str(sealed.repository), "dinov2_vits14"), {"source": "local", "pretrained": False})
    ]


def test_full_loader_uses_sealed_paths_and_rechecks_after_weight_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sealed, worktree, _repository, _weights = _materialize(tmp_path)
    for name in tuple(sys.modules):
        if name == "dinov2" or name.startswith("dinov2."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    class FakeModel:
        def __init__(self) -> None:
            self.loaded: tuple[object, bool] | None = None
            self.evaluated = False

        def load_state_dict(self, state_dict: object, *, strict: bool) -> None:
            self.loaded = (state_dict, strict)

        def eval(self) -> "FakeModel":
            self.evaluated = True
            return self

    class FakeTorchWithWeights(_FakeTorch):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeModel()
            self.weight_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def load(*args: object, **kwargs: object) -> object:
                self.weight_calls.append((args, kwargs))
                return {"model": {"fixture": 1}}

            self.load = load  # type: ignore[method-assign]
            self.hub.load = lambda *args, **kwargs: self.model  # type: ignore[method-assign]

    fake_torch = FakeTorchWithWeights()
    model = snapshot.load_sealed_dinov2_vits14(
        sealed,
        torch_module=fake_torch,
        expected_manifest_sha256=sealed.manifest_sha256,
        repository_root=worktree,
    )

    assert model is fake_torch.model
    assert fake_torch.model.loaded == ({"fixture": 1}, True)
    assert fake_torch.model.evaluated is True
    assert fake_torch.weight_calls == [
        ((str(sealed.weights),), {"map_location": "cpu", "weights_only": True})
    ]
