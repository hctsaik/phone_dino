from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Callable

from .artifacts import ProductionArtifactV19, ReferenceBoardPolicy
from .contracts import ReferenceBoardEvidence
from .decoder import DecodedImage


QrDecoder = Callable[["np.ndarray"], tuple[str, "np.ndarray | None"]]


def _opencv_qr_decoder(image_bgr: np.ndarray) -> tuple[str, np.ndarray | None]:
    import cv2
    import numpy as np

    payload, corners, _ = cv2.QRCodeDetector().detectAndDecode(image_bgr)
    if corners is None:
        return "", None
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return payload, points if points.shape == (4, 2) else None


def _mean_side_px(points: np.ndarray) -> float:
    import numpy as np

    return float(np.mean(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)))


def qr_to_charuco_residual_mm(mapped_qr: np.ndarray, expected_qr: np.ndarray) -> float:
    """RMS corner error independent of the QR detector's start corner/order."""

    import numpy as np

    if mapped_qr.shape != (4, 2) or expected_qr.shape != (4, 2):
        raise ValueError("QR quadrilaterals must contain four XY corners")
    candidates: list[float] = []
    for reversed_order in (False, True):
        ordered = mapped_qr[::-1] if reversed_order else mapped_qr
        for shift in range(4):
            shifted = np.roll(ordered, shift, axis=0)
            candidates.append(float(np.sqrt(np.mean(np.sum(np.square(shifted - expected_qr), axis=1)))))
    return min(candidates)


@dataclass(frozen=True)
class _CharucoPlane:
    homography_image_to_mm: np.ndarray
    corner_count: int
    inlier_count: int


class ReferenceBoardVerifier:
    """Fail-closed, same-still QR + ChArUco visual verifier.

    This is intentionally an observation component. It validates no signature,
    lifecycle or station authorization; those remain PhoneCV responsibilities.
    """

    def __init__(self, qr_decoder: QrDecoder | None = None):
        self._qr_decoder = qr_decoder or _opencv_qr_decoder

    @staticmethod
    def _rejected(policy: ReferenceBoardPolicy, *reason_codes: str, **metrics: float | int | None) -> ReferenceBoardEvidence:
        return ReferenceBoardEvidence(
            state="REJECTED",
            metricScaleSource="CHARUCO_ONLY",
            qrPayloadSha256=policy.qr_payload_sha256,
            qrSidePx=metrics.get("qr_side_px"),
            charucoCornerCount=metrics.get("charuco_corner_count"),
            charucoInlierCount=metrics.get("charuco_inlier_count"),
            qrToCharucoResidualMm=metrics.get("residual_mm"),
            reasonCodes=list(reason_codes),
        )

    @staticmethod
    def _charuco_plane(image_bgr: np.ndarray, artifact: ProductionArtifactV19) -> _CharucoPlane | None:
        import cv2
        import numpy as np

        board_config = artifact.board
        try:
            dictionary_id = getattr(cv2.aruco, board_config.dictionary)
        except AttributeError:
            return None
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        marker_ids = None if board_config.marker_ids is None else np.asarray(board_config.marker_ids, dtype=np.int32)
        board = cv2.aruco.CharucoBoard(
            (board_config.squares_x, board_config.squares_y),
            board_config.square_length_mm, board_config.marker_length_mm,
            dictionary, marker_ids,
        )
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
        corner_count = 0 if corners is None else len(corners)
        if ids is None or corners is None or corner_count < artifact.reference_board.min_charuco_corners:
            return None
        image_points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        object_points = board.getChessboardCorners()[ids.flatten()][:, :2].astype(np.float32)
        object_points[:, 0] += artifact.reference_board.charuco_origin_mm[0]
        object_points[:, 1] += artifact.reference_board.charuco_origin_mm[1]
        homography, inlier_mask = cv2.findHomography(image_points, object_points, cv2.RANSAC, 3.0)
        inlier_count = 0 if inlier_mask is None else int(np.asarray(inlier_mask).sum())
        if homography is None or inlier_count < artifact.reference_board.min_charuco_corners:
            return None
        return _CharucoPlane(homography, corner_count, inlier_count)

    def verify(self, image: DecodedImage, artifact: ProductionArtifactV19) -> ReferenceBoardEvidence:
        import cv2
        import numpy as np

        policy = artifact.reference_board
        source = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if source is None:
            return self._rejected(policy, "REFERENCE_IMAGE_DECODE_FAILED")
        payload, qr_points = self._qr_decoder(source)
        if qr_points is None or not payload:
            return self._rejected(policy, "REFERENCE_QR_NOT_DETECTED")
        qr_side_px = _mean_side_px(qr_points)
        if qr_side_px < policy.min_qr_side_px:
            return self._rejected(policy, "REFERENCE_QR_TOO_SMALL", qr_side_px=qr_side_px)
        payload_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(payload_sha256, policy.qr_payload_sha256):
            return self._rejected(policy, "REFERENCE_QR_PAYLOAD_MISMATCH", qr_side_px=qr_side_px)
        plane = self._charuco_plane(source, artifact)
        if plane is None:
            return self._rejected(
                policy, "REFERENCE_CHARUCO_CORNERS_INSUFFICIENT", qr_side_px=qr_side_px,
            )
        mapped = cv2.perspectiveTransform(
            qr_points.reshape(1, 4, 2).astype(np.float32), plane.homography_image_to_mm,
        ).reshape(4, 2)
        bounds = policy.qr_symbol_bounds_mm
        expected = np.asarray([
            [bounds.x, bounds.y],
            [bounds.x + bounds.width, bounds.y],
            [bounds.x + bounds.width, bounds.y + bounds.height],
            [bounds.x, bounds.y + bounds.height],
        ], dtype=np.float32)
        residual = qr_to_charuco_residual_mm(mapped, expected)
        if residual > policy.max_qr_to_charuco_residual_mm:
            return self._rejected(
                policy, "REFERENCE_QR_CHARUCO_COLOCATION_FAILED", qr_side_px=qr_side_px,
                charuco_corner_count=plane.corner_count, charuco_inlier_count=plane.inlier_count,
                residual_mm=residual,
            )
        return ReferenceBoardEvidence(
            state="VERIFIED",
            metricScaleSource="CHARUCO_ONLY",
            qrPayloadSha256=policy.qr_payload_sha256,
            qrSidePx=qr_side_px,
            charucoCornerCount=plane.corner_count,
            charucoInlierCount=plane.inlier_count,
            qrToCharucoResidualMm=residual,
            reasonCodes=[],
        )
