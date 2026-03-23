"""Image preprocessing pipeline and text cleaning utilities before/after OCR.

Three modes:
- ``none``: pass image through unchanged.
- ``basic``: normalize to target DPI (resize) and convert to greyscale.
- ``full``: basic + Otsu binarization + deskew.

All functions accept and return ``numpy.ndarray`` (HxW or HxWxC, uint8).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

PreprocessingMode = Literal["none", "basic", "full"]
PREPROCESSING_MODES: tuple[PreprocessingMode, ...] = ("none", "basic", "full")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_MD_STRIP_PATTERNS = (
    # Fenced code blocks  ```...```
    (r"```[^\n]*\n.*?```", ""),
    # ATX headings  # ## ### …
    (r"^#{1,6}\s+", ""),
    # Bold / italic  **x** *x* __x__ _x_
    (r"\*{1,2}([^*]+)\*{1,2}", r"\1"),
    (r"_{1,2}([^_]+)_{1,2}", r"\1"),
    # Inline code  `x`
    (r"`([^`]+)`", r"\1"),
    # Markdown links  [text](url)
    (r"\[([^\]]+)\]\([^)]*\)", r"\1"),
    # HTML tags
    (r"<[^>]+>", ""),
    # Horizontal rules  --- *** ___
    (r"^[-*_]{3,}\s*$", ""),
    # Leading list markers  - * + or 1.
    (r"^[\s]*[-*+]\s+", ""),
    (r"^[\s]*\d+\.\s+", ""),
)


def _strip_markdown(text: str) -> str:
    """Remove common markdown formatting, leaving plain text."""
    for pattern, repl in _MD_STRIP_PATTERNS:
        flags = re.DOTALL if ".*?" in pattern else re.MULTILINE
        text = re.sub(pattern, repl, text, flags=flags)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _require_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for preprocessing. Install with: pip install opencv-python-headless"
        ) from exc


def _require_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except ImportError as exc:
        raise RuntimeError("numpy is required for preprocessing.") from exc


def to_greyscale(image):
    """Convert BGR / RGB image to greyscale. Returns HxW uint8 array."""
    cv2 = _require_cv2()
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_dpi(image, *, from_dpi: int, to_dpi: int):
    """Resize image proportionally to match target DPI.

    If ``from_dpi == to_dpi`` the original array is returned unchanged.
    """
    if from_dpi <= 0 or to_dpi <= 0:
        raise ValueError(f"DPI values must be positive, got: from={from_dpi} to={to_dpi}")
    if from_dpi == to_dpi:
        return image
    cv2 = _require_cv2()
    scale = to_dpi / from_dpi
    h, w = image.shape[:2]
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    interp = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def binarize_otsu(grey_image):
    """Apply Otsu's thresholding to a greyscale image. Returns HxW binary uint8."""
    cv2 = _require_cv2()
    if grey_image.ndim != 2:
        raise ValueError("binarize_otsu requires a greyscale (2-D) image.")
    _, binary = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def deskew(grey_image):
    """Detect and correct skew angle for a greyscale image.

    Uses the projection-profile method (Hough transform on edges).
    Falls back to identity if angle cannot be determined.
    """
    cv2 = _require_cv2()
    np = _require_numpy()

    if grey_image.ndim != 2:
        raise ValueError("deskew requires a greyscale (2-D) image.")

    edges = cv2.Canny(grey_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, math.pi / 180, threshold=100)
    if lines is None or len(lines) == 0:
        return grey_image

    angles: list[float] = []
    for line in lines:
        rho, theta = line[0]
        # Convert to degrees relative to horizontal
        angle_deg = math.degrees(theta) - 90.0
        if abs(angle_deg) < 45:
            angles.append(angle_deg)

    if not angles:
        return grey_image

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.1:
        return grey_image

    h, w = grey_image.shape
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        grey_image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def apply_preprocessing(
    image,
    *,
    mode: PreprocessingMode,
    render_dpi: int = 0,
    ocr_dpi: int = 0,
):
    """Apply preprocessing pipeline to *image* according to *mode*.

    Parameters
    ----------
    image:
        Input image as ``numpy.ndarray`` (HxW or HxWxC uint8).
    mode:
        ``"none"`` — return *image* unchanged.
        ``"basic"`` — greyscale conversion + DPI normalisation (if ``render_dpi != ocr_dpi``).
        ``"full"`` — basic + Otsu binarisation + deskew.
    render_dpi:
        Source DPI of *image*. Used for DPI normalisation (``basic`` / ``full``).
        Pass ``0`` to skip DPI normalisation.
    ocr_dpi:
        Target DPI for OCR. Pass ``0`` to skip DPI normalisation.

    Returns
    -------
    numpy.ndarray
        Pre-processed image (uint8).
    """
    if mode == "none":
        return image

    result = image

    # Greyscale conversion
    result = to_greyscale(result)

    # DPI normalisation
    if render_dpi > 0 and ocr_dpi > 0 and render_dpi != ocr_dpi:
        result = normalize_dpi(result, from_dpi=render_dpi, to_dpi=ocr_dpi)

    if mode == "basic":
        return result

    # full: binarise + deskew
    result = binarize_otsu(result)
    result = deskew(result)
    return result


def _cv2_imread_unicode(path: Path):
    """Load an image from a path that may contain non-ASCII characters.

    ``cv2.imread`` uses the C runtime ``fopen`` on Windows, which does not
    support non-ASCII paths.  We work around this by reading the raw bytes
    with Python and decoding via ``cv2.imdecode``.
    """
    cv2 = _require_cv2()
    np = _require_numpy()
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return image


def _cv2_imwrite_unicode(path: Path, image) -> bool:
    """Write *image* to *path* even if the path contains non-ASCII characters."""
    cv2 = _require_cv2()
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True


def preprocess_image_file(
    src: Path,
    dst: Path,
    *,
    mode: PreprocessingMode,
    render_dpi: int = 0,
    ocr_dpi: int = 0,
) -> Path:
    """Load *src*, apply preprocessing, write to *dst*. Returns *dst*."""
    try:
        image = _cv2_imread_unicode(src)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"Failed to load image: {src}") from exc
    if image is None:
        raise RuntimeError(f"Failed to load image: {src}")
    processed = apply_preprocessing(image, mode=mode, render_dpi=render_dpi, ocr_dpi=ocr_dpi)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not _cv2_imwrite_unicode(dst, processed):
        raise RuntimeError(f"Failed to write preprocessed image: {dst}")
    return dst
