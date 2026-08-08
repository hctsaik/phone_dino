from __future__ import annotations

import importlib.util
import math
from pathlib import Path


def _tool_module():
    path = Path(__file__).parents[1] / "tools" / "run_mvtec_ad_smoke.py"
    spec = importlib.util.spec_from_file_location("mvtec_ad_smoke", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_auroc_handles_perfect_and_tied_scores() -> None:
    tool = _tool_module()
    assert tool.image_auroc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert tool.image_auroc([False, True], [0.5, 0.5]) == 0.5
    assert tool.image_auroc([False, False], [0.1, 0.2]) is None


def test_metric_summary_does_not_derive_its_threshold_from_blind_labels() -> None:
    tool = _tool_module()
    summary = tool.metric_summary([
        {"kind": "NOMINAL", "score": 0.2},
        {"kind": "ANOMALY", "score": 0.8},
        {"kind": "ANOMALY", "score": 0.1},
    ], threshold=0.5)
    assert summary["thresholdFromNominalTuning"] == 0.5
    assert summary["nominalAboveThresholdRate"] == 0.0
    assert summary["anomalyAboveThresholdRate"] == 0.5
    assert summary["imageAuRoc"] == 0.5


def test_patch_memory_bank_selection_is_evenly_spaced_and_deterministic() -> None:
    tool = _tool_module()
    assert tool.deterministic_prototype_indices(4, 8) == [0, 1, 2, 3]
    assert tool.deterministic_prototype_indices(10, 4) == [1, 3, 6, 8]


def test_patch_knn_score_uses_only_the_most_anomalous_patches() -> None:
    import numpy as np

    tool = _tool_module()
    score = tool.patch_knn_score(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        top_k=2,
    )
    assert score["maxPatchDistance"] == 1.0
    assert math.isclose(score["meanNearestPatchDistance"], 1.0 / 3.0, rel_tol=1e-6)
    assert score["score"] == 0.5
