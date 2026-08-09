"""Pinned, file-level acquisition of fresh normal MVTec AD research sources.

This is the only normal-holdout component allowed to access the public dataset
inventory or network.  It mechanically emits only pinned `train/good` source
records and never downloads a category archive, test image, ground-truth mask,
or anomaly image.
"""

from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from phone_dino.mvtec_normal_holdout import (
    NORMAL_SOURCE_CANDIDATES_SCHEMA,
    PINNED_MVTEC_DATASET_ID,
    PINNED_MVTEC_LICENSE_NOTICE,
    PINNED_MVTEC_MIRROR_REVISION,
    PINNED_MVTEC_MIRROR_SOURCE_URI,
    PINNED_MVTEC_OFFICIAL_SOURCE_URI,
    PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1,
    PINNED_MVTEC_SAMPLES_SHA256,
    REPOSITORY_ROOT,
    SOURCE_CANDIDATES_PURPOSE,
    NormalHoldoutError,
    _load_pinned_source_metadata,
    _is_under,
    _reject_links_on_existing_path,
    _require_decodable_image,
    _require_external_source_root,
    _safe_relative_path,
    _write_json,
    canonical_json_sha256,
    historical_normal_usage_identity,
    load_historical_normal_usage_ledger,
    sha256_file,
)


TARGET_CATEGORIES = ("capsule", "metal_nut", "tile")
EXPECTED_TRAIN_GOOD_COUNTS = {"capsule": 219, "metal_nut": 220, "tile": 230}
EXPECTED_FRESH_COUNTS = {"capsule": 155, "metal_nut": 156, "tile": 166}
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class NormalSourceAcquisitionError(NormalHoldoutError):
    """Raised when public file-level MVTec source acquisition is not pinned."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request: Request, fp: Any, code: int, message: str, headers: Any, newurl: str) -> None:
        return None


def _origin(*, prior_subset_source_identity_sha256: str) -> dict[str, str]:
    return {
        "datasetId": PINNED_MVTEC_DATASET_ID,
        "officialSourceUri": PINNED_MVTEC_OFFICIAL_SOURCE_URI,
        "mirrorSourceUri": PINNED_MVTEC_MIRROR_SOURCE_URI,
        "mirrorRevision": PINNED_MVTEC_MIRROR_REVISION,
        "sourceMetadataFileSha256": PINNED_MVTEC_SAMPLES_SHA256,
        "sourceMetadataGitBlobSha1": PINNED_MVTEC_SAMPLES_GIT_BLOB_SHA1,
        "sourceCriterion": "OFFICIAL_TRAIN_GOOD_ONLY",
        "licenseNotice": PINNED_MVTEC_LICENSE_NOTICE,
        "priorSubsetSourceIdentitySha256": prior_subset_source_identity_sha256,
    }


def _prepare_new_source_root(path: Path) -> None:
    if path.exists():
        raise NormalSourceAcquisitionError("fresh normal source root already exists; choose a new external directory")
    if _is_under(REPOSITORY_ROOT, path) or _is_under(path, REPOSITORY_ROOT):
        raise NormalSourceAcquisitionError("fresh normal source root must stay outside the Git working tree")
    _reject_links_on_existing_path(path.parent, description="fresh normal source root")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_links_on_existing_path(path.parent, description="fresh normal source root")
    path.mkdir(exist_ok=False)
    _reject_links_on_existing_path(path, description="fresh normal source root")
    _require_external_source_root(path)


def _header(headers: Any, name: str) -> str:
    value = headers.get(name)
    if not isinstance(value, str) or not value.strip():
        raise NormalSourceAcquisitionError(f"MVTec source redirect is missing {name}")
    return value.strip()


def _safe_redirect_location(location: str, *, resolve_url: str) -> str:
    destination = urljoin(resolve_url, location)
    parsed = urlparse(destination)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        raise NormalSourceAcquisitionError("MVTec source redirect is not HTTPS")
    if not (host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co")):
        raise NormalSourceAcquisitionError("MVTec source redirect host is not an approved Hugging Face host")
    return destination


def _resolve_remote_identity(remote_path: str, *, timeout_seconds: float) -> tuple[str, int, str]:
    quoted_path = quote(remote_path, safe="/")
    resolve_url = (
        f"{PINNED_MVTEC_MIRROR_SOURCE_URI}/resolve/{PINNED_MVTEC_MIRROR_REVISION}/{quoted_path}?download=true"
    )
    request = Request(resolve_url, headers={"User-Agent": "phone-dino-normal-holdout/1.0"})
    opener = build_opener(_NoRedirect())
    try:
        response = opener.open(request, timeout=timeout_seconds)
    except HTTPError as error:
        if error.code not in {301, 302, 303, 307, 308}:
            raise NormalSourceAcquisitionError(f"MVTec source resolve request failed with HTTP {error.code}") from error
        response = error
    except URLError as error:
        raise NormalSourceAcquisitionError("MVTec source resolve request failed") from error
    try:
        if getattr(response, "code", getattr(response, "status", None)) not in {301, 302, 303, 307, 308}:
            raise NormalSourceAcquisitionError("MVTec source resolve request did not return a redirect")
        if _header(response.headers, "X-Repo-Commit") != PINNED_MVTEC_MIRROR_REVISION:
            raise NormalSourceAcquisitionError("MVTec source redirect revision does not match")
        etag = _header(response.headers, "X-Linked-ETag").strip('"')
        if not _SHA256_HEX.fullmatch(etag):
            raise NormalSourceAcquisitionError("MVTec source redirect raw digest is invalid")
        size_text = _header(response.headers, "X-Linked-Size")
        try:
            size = int(size_text)
        except ValueError as error:
            raise NormalSourceAcquisitionError("MVTec source redirect raw size is invalid") from error
        if size <= 0:
            raise NormalSourceAcquisitionError("MVTec source redirect raw size is invalid")
        location = _safe_redirect_location(_header(response.headers, "Location"), resolve_url=resolve_url)
        return f"sha256:{etag}", size, location
    finally:
        response.close()


def _stream_verified_file(
    location: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    timeout_seconds: float,
) -> None:
    if destination.exists():
        raise NormalSourceAcquisitionError("fresh normal source destination already exists")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise NormalSourceAcquisitionError("fresh normal source partial destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_links_on_existing_path(destination.parent, description="fresh normal source destination")
    digest = hashlib.sha256()
    byte_count = 0
    request = Request(location, headers={"User-Agent": "phone-dino-normal-holdout/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response, partial.open("xb") as stream:
            final_url = urlparse(response.geturl())
            if final_url.scheme != "https":
                raise NormalSourceAcquisitionError("MVTec source content was not served over HTTPS")
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
                stream.write(chunk)
    except (OSError, URLError) as error:
        raise NormalSourceAcquisitionError("MVTec source download failed") from error
    actual_sha256 = f"sha256:{digest.hexdigest()}"
    if actual_sha256 != expected_sha256 or byte_count != expected_bytes:
        raise NormalSourceAcquisitionError("MVTec source content does not match the pinned redirect identity")
    _require_decodable_image(partial, description="downloaded MVTec normal source")
    # The source root is new and exclusive to this invocation.  `destination`
    # was checked absent before streaming, so promotion cannot overwrite a
    # previously frozen input.
    os.replace(partial, destination)


def _candidate_remote_paths(inventory: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    candidates = [
        (remote_path, metadata)
        for remote_path, metadata in inventory.items()
        if metadata["category"] in TARGET_CATEGORIES
        and metadata["split"] == "train"
        and metadata["defect"] == "good"
        and not metadata["hasMask"]
    ]
    counts = {category: sum(metadata["category"] == category for _, metadata in candidates) for category in TARGET_CATEGORIES}
    if counts != EXPECTED_TRAIN_GOOD_COUNTS:
        raise NormalSourceAcquisitionError("pinned MVTec metadata train/good category counts do not match the protocol")
    return sorted(candidates, key=lambda item: item[0])


def acquire_fresh_normal_sources(
    source_metadata_path: Path,
    ledger_path: Path,
    source_root: Path,
    candidate_output_path: Path,
    *,
    workers: int = 4,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Download only fresh pinned train/good files and emit a closed candidate manifest."""

    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 8:
        raise NormalSourceAcquisitionError("workers must be an integer between 1 and 8")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise NormalSourceAcquisitionError("timeout_seconds must be positive")
    ledger, _, historical_hashes = load_historical_normal_usage_ledger(ledger_path)
    prior_identity = historical_normal_usage_identity(ledger, historical_hashes)
    origin = _origin(prior_subset_source_identity_sha256=prior_identity)
    inventory = _load_pinned_source_metadata(source_metadata_path, origin=origin)
    _prepare_new_source_root(source_root)
    selected: list[dict[str, Any]] = []
    for remote_path, metadata in _candidate_remote_paths(inventory):
        expected_sha256, expected_bytes, location = _resolve_remote_identity(remote_path, timeout_seconds=float(timeout_seconds))
        if expected_sha256 in historical_hashes:
            continue
        suffix = Path(remote_path).suffix.lower()
        if suffix != ".png":
            raise NormalSourceAcquisitionError("pinned MVTec normal source is not a PNG")
        source_sha_hex = expected_sha256[7:]
        selected.append({
            "caseId": f"mvtec-ad/{metadata['category']}/train-good/{source_sha_hex}",
            "category": metadata["category"],
            "relativePath": f"images/{metadata['category']}/{source_sha_hex}{suffix}",
            "sourceGroupId": f"CONTENT_SHA256:{source_sha_hex}",
            "acquisitionStratum": "OFFICIAL_MVTEC_TRAIN_GOOD",
            "sourceRemotePath": remote_path,
            "expectedRemoteSha256": expected_sha256,
            "expectedRemoteBytes": expected_bytes,
            "_location": location,
        })
    counts = {category: sum(record["category"] == category for record in selected) for category in TARGET_CATEGORIES}
    if counts != EXPECTED_FRESH_COUNTS:
        raise NormalSourceAcquisitionError("fresh MVTec normal source counts do not match the protocol")
    selected.sort(key=lambda record: record["caseId"])

    def download(record: dict[str, Any]) -> None:
        relative = _safe_relative_path(record["relativePath"], name="acquired relativePath")
        destination = source_root.joinpath(*relative.parts)
        _stream_verified_file(
            record["_location"],
            destination,
            expected_sha256=record["expectedRemoteSha256"],
            expected_bytes=record["expectedRemoteBytes"],
            timeout_seconds=float(timeout_seconds),
        )
        if sha256_file(destination) != record["expectedRemoteSha256"]:
            raise NormalSourceAcquisitionError("downloaded MVTec source digest changed after verification")

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mvtec-normal-download") as executor:
        futures = [executor.submit(download, record) for record in selected]
        for future in as_completed(futures):
            future.result()

    records = [{key: value for key, value in record.items() if key != "_location"} for record in selected]
    document: dict[str, Any] = {
        "schemaVersion": NORMAL_SOURCE_CANDIDATES_SCHEMA,
        "authoritative": False,
        "productionAuthorized": False,
        "purpose": SOURCE_CANDIDATES_PURPOSE,
        "origin": origin,
        "groupingStrength": "EXACT_CONTENT_ONLY",
        "records": records,
    }
    document["candidateManifestSha256"] = canonical_json_sha256(document)
    _write_json(candidate_output_path, document, description="fresh normal source candidates output")
    return document
