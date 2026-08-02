from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import time

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from .config import Settings


@dataclass(frozen=True, slots=True)
class DecodedImage:
    data: bytes
    width: int
    height: int
    format: str
    elapsed_ms: int


async def read_and_validate_image(upload: UploadFile, content_type: str, settings: Settings) -> DecodedImage:
    if upload.content_type != content_type:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CONTENT_TYPE_MISMATCH")

    started = time.perf_counter()
    data = bytearray()
    while True:
        chunk = await upload.read(min(1024 * 1024, settings.max_image_bytes + 1))
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > settings.max_image_bytes:
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="IMAGE_BYTES_LIMIT_EXCEEDED")
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="EMPTY_IMAGE")

    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        with Image.open(BytesIO(data)) as probe:
            expected_format = "JPEG" if content_type == "image/jpeg" else "PNG"
            if probe.format != expected_format:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="IMAGE_FORMAT_MISMATCH")
            width, height = probe.size
            if width <= 0 or height <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="INVALID_IMAGE_DIMENSIONS")
            if width > settings.max_image_width or height > settings.max_image_height or width * height > settings.max_image_pixels:
                raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="IMAGE_PIXELS_LIMIT_EXCEEDED")
            probe.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="IMAGE_DECODE_FAILED") from exc

    return DecodedImage(
        data=bytes(data),
        width=width,
        height=height,
        format=expected_format,
        elapsed_ms=max(0, round((time.perf_counter() - started) * 1000)),
    )
