"""Adapter layer for document detection/scanner integration."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


class ScanAdapterError(RuntimeError):
    """Raised when scanner backend cannot be loaded or used."""


@dataclass(slots=True)
class ScanOutput:
    """Normalized scanner output."""

    warped: np.ndarray | None
    contour: np.ndarray | None
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


def scan_with_document_detector(
    image: np.ndarray,
    *,
    enabled: bool = True,
    scanner_root: Path | None = None,
) -> ScanOutput:
    """
    Run document detector and return normalized output.

    If detection is disabled, returns the input image as warped.
    """
    if not enabled:
        return ScanOutput(warped=image, contour=None, raw_result=None)

    scanner_module = _import_scanner_with_optional_root(optional_root=scanner_root)
    result = scanner_module.main(image)
    warped = getattr(result, "warped", None)
    contour = getattr(result, "contour", None)
    if warped is None:
        return ScanOutput(warped=image, contour=contour, raw_result=result)
    return ScanOutput(warped=warped, contour=contour, raw_result=result)
