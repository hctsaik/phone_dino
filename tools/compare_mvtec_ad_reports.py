"""Join two immutable MVTec research reports without selecting a winner.

The frozen blind set is reporting-only.  This command makes per-case and
per-defect deltas auditable, but explicitly refuses to rank or promote a
candidate from blind metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_SCHEMA_VERSION = "phone-dino.mvtec-ad-report-comparison/1.0"


class ComparisonError(ValueError):
    """Raised when immutable report identities cannot safely be joined."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _is_under(directory: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _read_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"Unable to read report: {path}") from error
    if not isinstance(report, dict) or report.get("authoritative") is not False:
        raise ComparisonError("report must explicitly declare authoritative=false")
    schema = report.get("schemaVersion")
    if not isinstance(schema, str) or not schema.startswith("phone-dino.mvtec-ad-"):
        raise ComparisonError("report is not a supported MVTec AD research report")
    if not isinstance(report.get("scores"), list) or not isinstance(report.get("categories"), dict):
        raise ComparisonError("report has no comparable scores/categories")
    return report


def _manifest_identity(report: dict[str, Any]) -> tuple[str, str | None]:
    declared = report.get("inputManifestDeclaredSha256", report.get("inputManifestSha256"))
    if not isinstance(declared, str) or not declared.startswith("sha256:"):
        raise ComparisonError("report has no frozen manifest identity")
    file_digest = report.get("inputManifestFileSha256")
    if file_digest is not None and (not isinstance(file_digest, str) or not file_digest.startswith("sha256:")):
        raise ComparisonError("report has an invalid input manifest file digest")
    return declared, file_digest


def _score_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in report["scores"]:
        if not isinstance(raw, dict):
            raise ComparisonError("report score is not an object")
        case_id = raw.get("caseId")
        if not isinstance(case_id, str) or not case_id or case_id in indexed:
            raise ComparisonError("report score caseId is invalid or duplicated")
        source_sha256 = raw.get("sourceSha256")
        score = raw.get("score")
        if not isinstance(source_sha256, str) or not source_sha256.startswith("sha256:"):
            raise ComparisonError("report score sourceSha256 is invalid")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ComparisonError("report score is invalid")
        indexed[case_id] = raw
    return indexed


def _category_metric(report: dict[str, Any], category: str, name: str) -> float | None:
    value = report["categories"].get(category)
    if not isinstance(value, dict):
        return None
    metric = value.get(name)
    return float(metric) if isinstance(metric, (int, float)) and not isinstance(metric, bool) else None


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_report_sha256: str,
    candidate_report_sha256: str,
) -> dict[str, Any]:
    """Strictly join reports by immutable case ID and source digest."""

    baseline_manifest, baseline_manifest_file = _manifest_identity(baseline)
    candidate_manifest, candidate_manifest_file = _manifest_identity(candidate)
    if baseline_manifest != candidate_manifest:
        raise ComparisonError("reports were not scored from the same frozen manifest")
    if baseline_manifest_file is not None and candidate_manifest_file is not None and baseline_manifest_file != candidate_manifest_file:
        raise ComparisonError("reports have different frozen manifest file bytes")
    baseline_scores = _score_index(baseline)
    candidate_scores = _score_index(candidate)
    if baseline_scores.keys() != candidate_scores.keys():
        raise ComparisonError("reports do not contain the same immutable caseId set")

    case_deltas: list[dict[str, Any]] = []
    per_category: dict[str, list[float]] = defaultdict(list)
    per_defect: dict[tuple[str, str], list[float]] = defaultdict(list)
    for case_id in sorted(baseline_scores):
        before = baseline_scores[case_id]
        after = candidate_scores[case_id]
        if before["sourceSha256"] != after["sourceSha256"]:
            raise ComparisonError(f"source digest differs for {case_id}")
        if before.get("category") != after.get("category") or before.get("kind") != after.get("kind"):
            raise ComparisonError(f"case identity fields differ for {case_id}")
        delta = float(after["score"]) - float(before["score"])
        category = str(before.get("category"))
        defect = str(before.get("defect"))
        per_category[category].append(delta)
        if before.get("role") == "BLIND" and before.get("kind") == "ANOMALY":
            per_defect[(category, defect)].append(delta)
        case_deltas.append({
            "caseId": case_id,
            "category": category,
            "role": before.get("role"),
            "kind": before.get("kind"),
            "defect": before.get("defect"),
            "sourceSha256": before["sourceSha256"],
            "baselineScore": float(before["score"]),
            "candidateScore": float(after["score"]),
            "scoreDelta": delta,
        })
    categories = []
    for category in sorted(per_category):
        baseline_auroc = _category_metric(baseline, category, "imageAuRoc")
        candidate_auroc = _category_metric(candidate, category, "imageAuRoc")
        baseline_threshold = _category_metric(baseline, category, "thresholdFromNominalTuning")
        candidate_threshold = _category_metric(candidate, category, "thresholdFromNominalTuning")
        categories.append({
            "category": category,
            "cases": len(per_category[category]),
            "meanScoreDelta": statistics.fmean(per_category[category]),
            "medianScoreDelta": statistics.median(per_category[category]),
            "baselineImageAuRoc": baseline_auroc,
            "candidateImageAuRoc": candidate_auroc,
            "imageAuRocDelta": None if baseline_auroc is None or candidate_auroc is None else candidate_auroc - baseline_auroc,
            "baselineNormalThreshold": baseline_threshold,
            "candidateNormalThreshold": candidate_threshold,
            "normalThresholdDelta": None if baseline_threshold is None or candidate_threshold is None else candidate_threshold - baseline_threshold,
        })
    defects = [
        {
            "category": category,
            "defect": defect,
            "blindAnomalyCases": len(deltas),
            "meanScoreDelta": statistics.fmean(deltas),
            "medianScoreDelta": statistics.median(deltas),
        }
        for (category, defect), deltas in sorted(per_defect.items())
    ]
    return {
        "schemaVersion": COMPARISON_SCHEMA_VERSION,
        "authoritative": False,
        "productionAuthorized": False,
        "selectionProtocol": "COMPARISON_ONLY_NO_BLIND_BASED_MODEL_SELECTION",
        "disclaimer": "This joins already-fixed offline research reports. Blind metrics and per-defect deltas must not be used to select or promote a PhoneDINO production model/threshold.",
        "inputManifestDeclaredSha256": baseline_manifest,
        "inputManifestFileSha256": baseline_manifest_file or candidate_manifest_file,
        "baseline": {
            "schemaVersion": baseline["schemaVersion"],
            "algorithm": baseline.get("algorithm", {}).get("id"),
            "reportSha256": baseline_report_sha256,
        },
        "candidate": {
            "schemaVersion": candidate["schemaVersion"],
            "algorithm": candidate.get("algorithm", {}).get("id"),
            "reportSha256": candidate_report_sha256,
        },
        "categories": categories,
        "blindAnomalyDefects": defects,
        "caseDeltas": case_deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two frozen MVTec AD research reports")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if _is_under(REPOSITORY_ROOT, arguments.output):
        parser.error("--output must stay outside the Git working tree")
    if arguments.output.exists():
        parser.error("--output already exists; reports are immutable")
    baseline = _read_report(arguments.baseline)
    candidate = _read_report(arguments.candidate)
    report = compare_reports(
        baseline,
        candidate,
        baseline_report_sha256=sha256_file(arguments.baseline),
        candidate_report_sha256=sha256_file(arguments.candidate),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "categories": report["categories"]}, indent=2))


if __name__ == "__main__":
    main()
