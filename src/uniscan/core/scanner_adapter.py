"""Adapter layer for document detection/scanner integration."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from .geometry import order_quad_points, warp_perspective_from_points

DETECTOR_BACKEND_CAMSCAN = "camscan"
DETECTOR_BACKEND_OPENCV = "opencv_quad"


class ScanAdapterError(RuntimeError):
    """Raised when scanner backend cannot be loaded or used."""


@dataclass(slots=True)
class ScanOutput:
    """Normalized scanner output."""

    warped: np.ndarray | None
    contour: np.ndarray | None
    backend: str | None
    raw_result: Any


def _import_scanner_with_optional_root(optional_root: Path | None = None) -> ModuleType:
    if optional_root is None:
        try:
            return importlib.import_module("camscan.scanner")
        except Exception as exc:  # pragma: no cover - import is environment-dependent
            raise ScanAdapterError(
                "Cannot import camscan.scanner. Ensure camscan is installed or vendored."
            ) from exc

    root_str = str(optional_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        return importlib.import_module("camscan.scanner")
    except Exception as exc:  # pragma: no cover - import is environment-dependent
        raise ScanAdapterError(
            f"Cannot import camscan.scanner from optional root: {optional_root}"
        ) from exc


def _resize_for_detection(image: np.ndarray, *, max_side: int = 1600) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(max_side / max(1, height), max_side / max(1, width), 1.0)
    if scale >= 1.0:
        return image, 1.0
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _candidate_maps(gray: np.ndarray) -> list[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        15,
    )
    adaptive_inv = cv2.bitwise_not(adaptive)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = cv2.bitwise_not(otsu)
    edges = cv2.Canny(blurred, 60, 180)

    kernel = np.ones((5, 5), dtype=np.uint8)
    closed_maps = []
    for candidate in (adaptive, adaptive_inv, otsu, otsu_inv, edges):
        closed_maps.append(cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2))
    return closed_maps


def _contour_score(contour: np.ndarray, image_area: float) -> float:
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return -1.0
    x, y, width, height = cv2.boundingRect(contour)
    rect_area = float(max(1, width * height))
    fill_ratio = area / rect_area
    coverage = area / max(1.0, image_area)
    return (coverage * 10.0) + fill_ratio


def _find_quad_contour(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    image_area = float(gray.shape[0] * gray.shape[1])
    min_area = image_area * 0.12
    best_quad: np.ndarray | None = None
    best_score = -1.0
    candidate_maps = _candidate_maps(gray)

    for candidate_map in candidate_maps:
        contours, _hierarchy = cv2.findContours(candidate_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            score = _contour_score(approx, image_area)
            if score > best_score:
                best_score = score
                best_quad = approx.reshape(4, 2).astype(np.float32)

        if best_quad is not None:
            break

    if best_quad is not None:
        return order_quad_points(best_quad)

    best_rect: np.ndarray | None = None
    best_rect_score = -1.0
    contours, _hierarchy = cv2.findContours(candidate_maps[0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect)
        points = order_quad_points(points.astype(np.float32))
        score = _contour_score(points.reshape(-1, 1, 2), image_area)
        if score > best_rect_score:
            best_rect_score = score
            best_rect = points
    return best_rect


def _opencv_document_detector(image: np.ndarray) -> ScanOutput:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if float(np.std(gray)) < 5.0:
        return ScanOutput(warped=image, contour=None, backend=None, raw_result={"opencv": "low_variance"})

    resized, scale = _resize_for_detection(image)
    contour = _find_quad_contour(resized)
    if contour is None:
        return ScanOutput(warped=image, contour=None, backend=None, raw_result={"opencv": "no_contour"})

    if scale != 1.0:
        contour = contour / scale
    contour = order_quad_points(contour.astype(np.float32))
    warped = warp_perspective_from_points(image, contour)
    return ScanOutput(warped=warped, contour=contour, backend=DETECTOR_BACKEND_OPENCV, raw_result=None)


def _camscan_document_detector(image: np.ndarray, *, scanner_root: Path | None = None) -> ScanOutput:
    scanner_module = _import_scanner_with_optional_root(optional_root=scanner_root)
    result = scanner_module.main(image)
    warped = getattr(result, "warped", None)
    contour = getattr(result, "contour", None)
    if contour is not None:
        contour_arr = np.array(contour, dtype=np.float32).reshape(-1, 2)
        if contour_arr.shape[0] == 4:
            contour = order_quad_points(contour_arr)
        elif contour_arr.shape[0] > 4:
            rect = cv2.minAreaRect(contour_arr.astype(np.float32))
            contour = order_quad_points(cv2.boxPoints(rect).astype(np.float32))
        else:
            contour = None
    if warped is None:
        return ScanOutput(warped=image, contour=contour, backend=DETECTOR_BACKEND_CAMSCAN, raw_result=result)
    return ScanOutput(warped=warped, contour=contour, backend=DETECTOR_BACKEND_CAMSCAN, raw_result=result)


def scan_with_document_detector(
    image: np.ndarray,
    *,
    enabled: bool = True,
    scanner_root: Path | None = None,
    backends: tuple[str, ...] | None = None,
) -> ScanOutput:
    """
    Run document detector and return normalized output.

    If detection is disabled, returns the input image as warped.
    """
    if not enabled:
        return ScanOutput(warped=image, contour=None, backend=None, raw_result=None)

    selected_backends = backends or (DETECTOR_BACKEND_CAMSCAN, DETECTOR_BACKEND_OPENCV)
    errors: list[str] = []

    for backend in selected_backends:
        try:
            if backend == DETECTOR_BACKEND_CAMSCAN:
                result = _camscan_document_detector(image, scanner_root=scanner_root)
            elif backend == DETECTOR_BACKEND_OPENCV:
                result = _opencv_document_detector(image)
            else:
                raise ScanAdapterError(f"Unsupported detector backend: {backend}")
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            continue

        if result.contour is not None:
            return result

    return ScanOutput(
        warped=image,
        contour=None,
        backend=None,
        raw_result={"errors": errors},
    )
