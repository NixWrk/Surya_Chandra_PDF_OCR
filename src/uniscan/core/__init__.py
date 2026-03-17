"""Core processing primitives for unified scanner."""

from .postprocess import POSTPROCESSING_OPTIONS
from .scanner_adapter import ScanAdapterError, scan_with_document_detector

__all__ = [
    "POSTPROCESSING_OPTIONS",
    "ScanAdapterError",
    "scan_with_document_detector",
]
