from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

from fastapi import HTTPException, status

from .contracts import AnalyzeRequest


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prefixed_sha256(data: bytes) -> str:
    return f"sha256:{sha256_hex(data)}"


def canonical_bundle_digest(request: AnalyzeRequest) -> str:
    canonical = json.dumps(
        request.execution_bundle.model_dump(by_alias=True, mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return prefixed_sha256(canonical)


def verify_bundle_digest(request: AnalyzeRequest) -> None:
    computed = canonical_bundle_digest(request)
    if not hmac.compare_digest(computed, request.execution_bundle_digest):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="EXECUTION_BUNDLE_DIGEST_MISMATCH")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def digest_directory(path: Path) -> str:
    """Content-address a reviewed source tree without VCS/cache metadata."""
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and ".git" not in candidate.relative_to(path).parts
        and "__pycache__" not in candidate.relative_to(path).parts
        and candidate.suffix != ".pyc"
    )
    if not files:
        raise OSError("source repository is empty")
    for candidate in files:
        if candidate.is_symlink():
            raise OSError("source repository contains a symbolic link")
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"
