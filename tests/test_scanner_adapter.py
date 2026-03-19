from __future__ import annotations

import cv2
import numpy as np

from uniscan.core.scanner_adapter import (
    DETECTOR_BACKEND_OPENCV,
    scan_with_document_detector,
)


def _perspective_doc() -> np.ndarray:
    image = np.full((700, 900, 3), 35, dtype=np.uint8)
    quad = np.array([[170, 90], [760, 130], [700, 610], [130, 560]], dtype=np.int32)
    cv2.fillConvexPoly(image, quad, (245, 245, 245))
    cv2.polylines(image, [quad], isClosed=True, color=(15, 15, 15), thickness=8)
    for y in range(170, 540, 40):
        cv2.line(image, (230, y), (640, y), (40, 40, 40), 4)
    return image


def test_scanner_adapter_detects_quad_with_opencv_fallback() -> None:
    image = _perspective_doc()

    result = scan_with_document_detector(
        image,
        enabled=True,
        backends=(DETECTOR_BACKEND_OPENCV,),
    )

    assert result.backend == DETECTOR_BACKEND_OPENCV
    assert result.contour is not None
    assert result.warped is not None
    assert result.warped.shape[0] > 350
    assert result.warped.shape[1] > 350


def test_scanner_adapter_disabled_returns_original() -> None:
    image = _perspective_doc()

    result = scan_with_document_detector(image, enabled=False)

    assert result.contour is None
    assert result.backend is None
    assert np.array_equal(result.warped, image)


def test_scanner_adapter_gracefully_returns_no_contour() -> None:
    blank = np.zeros((240, 320, 3), dtype=np.uint8)

    result = scan_with_document_detector(
        blank,
        enabled=True,
        backends=(DETECTOR_BACKEND_OPENCV,),
    )

    assert result.contour is None
    assert result.backend is None
    assert np.array_equal(result.warped, blank)
