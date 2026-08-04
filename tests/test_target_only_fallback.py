from io import BytesIO

import numpy as np
from PIL import Image

from phone_dino.decoder import DecodedImage
from phone_dino.production import (
    NormalizedCapture,
    OpenCvCharucoNormalizer,
    PlaneMetricCalibration,
    PlaneNormalizedCapture,
)


class _PlaneFailure:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def normalize_plane(self, _image: object, _artifact: object) -> PlaneNormalizedCapture:
        return PlaneNormalizedCapture(rgb=None, reason_codes=(self.reason,))


class _TargetRecorder:
    def __init__(self) -> None:
        self.rgb: object | None = None

    def align(self, rgb: object, _artifact: object) -> NormalizedCapture:
        self.rgb = rgb
        return NormalizedCapture(rgb=rgb, encoded=b"target-only")


class _PlaneSuccess:
    def normalize_plane(self, _image: object, _artifact: object) -> PlaneNormalizedCapture:
        return PlaneNormalizedCapture(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            source_rgb=np.ones((2, 3, 3), dtype=np.uint8),
            input_to_plane=np.diag((2.0, 3.0, 1.0)),
            metric_calibration=PlaneMetricCalibration(10.0, 11.0, 12, 10, 0.4),
        )


class _PlaneThenRawTarget:
    def __init__(self) -> None:
        self.calls = 0

    def align(self, rgb: object, _artifact: object) -> NormalizedCapture:
        self.calls += 1
        if self.calls == 1:
            return NormalizedCapture(rgb=None, encoded=b"", reason_codes=("TARGET_MATCHES_INSUFFICIENT",))
        return NormalizedCapture(
            rgb=rgb,
            encoded=b"raw-target",
            target_from_input=np.asarray(((4.0, 0.0, 0.0), (0.0, 5.0, 0.0))),
        )


def _image() -> DecodedImage:
    stream = BytesIO()
    Image.new("RGB", (3, 2), (12, 34, 56)).save(stream, "PNG")
    return DecodedImage(data=stream.getvalue(), width=3, height=2, format="PNG", elapsed_ms=0)


def test_engineering_target_only_fallback_accepts_invalid_board_homography() -> None:
    target = _TargetRecorder()
    normalizer = OpenCvCharucoNormalizer(target_aligner=target, allow_target_only_alignment=True)
    normalizer._plane = _PlaneFailure("CHARUCO_HOMOGRAPHY_INVALID")  # type: ignore[assignment]

    result = normalizer.normalize(_image(), object())  # type: ignore[arg-type]

    assert result.reason_codes == ()
    assert result.encoded == b"target-only"
    assert isinstance(target.rgb, np.ndarray)
    assert target.rgb.shape == (2, 3, 3)


def test_target_only_fallback_keeps_quality_failures_closed() -> None:
    target = _TargetRecorder()
    normalizer = OpenCvCharucoNormalizer(target_aligner=target, allow_target_only_alignment=True)
    normalizer._plane = _PlaneFailure("BLUR")  # type: ignore[assignment]

    result = normalizer.normalize(_image(), object())  # type: ignore[arg-type]

    assert result.reason_codes == ("BLUR",)
    assert target.rgb is None


def test_target_only_fallback_retries_raw_target_after_plane_target_mismatch() -> None:
    target = _PlaneThenRawTarget()
    normalizer = OpenCvCharucoNormalizer(target_aligner=target, allow_target_only_alignment=True)
    normalizer._plane = _PlaneSuccess()  # type: ignore[assignment]

    result = normalizer.normalize(_image(), object())  # type: ignore[arg-type]

    assert result.reason_codes == ()
    assert result.encoded == b"raw-target"
    assert target.calls == 2
    assert result.metric_calibration is not None
    np.testing.assert_allclose(
        result.metric_calibration.target_to_plane,
        np.diag((0.5, 0.6, 1.0)),
    )
