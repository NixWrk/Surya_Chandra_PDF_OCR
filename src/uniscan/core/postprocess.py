"""Image post-processing registry used across the unified app."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

PostprocessFn = Callable[[np.ndarray], np.ndarray]


def dummy(image: np.ndarray) -> np.ndarray:
    """Return image unchanged."""
    return image


def sharpen(image: np.ndarray) -> np.ndarray:
    """Apply mild sharpening."""
    blurred = cv2.GaussianBlur(src=image, ksize=(0, 0), sigmaX=3)
    return cv2.addWeighted(src1=image, alpha=1.5, src2=blurred, beta=-0.5, gamma=0)


def grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def black_and_white(image: np.ndarray) -> np.ndarray:
    """High-contrast black-and-white effect suitable for documents."""
    gray = grayscale(image=image)
    sharp = sharpen(image=gray)
    return cv2.adaptiveThreshold(
        src=sharp,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=15,
    )


POSTPROCESSING_OPTIONS: dict[str, PostprocessFn] = {
    "None": dummy,
    "Sharpen": sharpen,
    "Grayscale": grayscale,
    "Black and White": black_and_white,
}
