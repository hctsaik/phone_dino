from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _tool_module():
    path = Path(__file__).parents[1] / "tools" / "compare_mvtec_ad_reports.py"
    spec = importlib.util.spec_from_file_location("mvtec_ad_report_comparison", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(score: float, *, source_sha256: str = f"sha256:{'1' * 64}") -> dict:
    return {
        "schemaVersion": "phone-dino.mvtec-ad-iteration-report/1.0",
        "authoritative": False,
        "inputManifestDeclaredSha256": f"sha256:{'a' * 64}",
        "inputManifestFileSha256": f"sha256:{'b' * 64}",
        "algorithm": {"id": "CANDIDATE"},
        "categories": {"capsule": {"imageAuRoc": 0.5}},
        "scores": [
            {
                "caseId": "case-1", "category": "capsule", "role": "BLIND", "kind": "ANOMALY", "defect": "crack",
                "sourceSha256": source_sha256, "score": score,
            }
        ],
    }


def test_compare_joins_by_case_and_reports_blind_deltas_without_winner() -> None:
    tool = _tool_module()
    result = tool.compare_reports(
        _report(0.2), _report(0.6), baseline_report_sha256=f"sha256:{'c' * 64}", candidate_report_sha256=f"sha256:{'d' * 64}"
    )
    assert result["selectionProtocol"] == "COMPARISON_ONLY_NO_BLIND_BASED_MODEL_SELECTION"
    assert result["caseDeltas"][0]["scoreDelta"] == pytest.approx(0.4)
    assert result["blindAnomalyDefects"] == [{
        "category": "capsule", "defect": "crack", "blindAnomalyCases": 1, "meanScoreDelta": pytest.approx(0.4), "medianScoreDelta": pytest.approx(0.4)
    }]


def test_compare_rejects_mismatched_source_or_case_membership() -> None:
    tool = _tool_module()
    with pytest.raises(tool.ComparisonError, match="source digest"):
        tool.compare_reports(
            _report(0.2), _report(0.6, source_sha256=f"sha256:{'2' * 64}"),
            baseline_report_sha256=f"sha256:{'c' * 64}", candidate_report_sha256=f"sha256:{'d' * 64}",
        )
    candidate = _report(0.6)
    candidate["scores"] = []
    with pytest.raises(tool.ComparisonError, match="same immutable caseId"):
        tool.compare_reports(
            _report(0.2), candidate,
            baseline_report_sha256=f"sha256:{'c' * 64}", candidate_report_sha256=f"sha256:{'d' * 64}",
        )
