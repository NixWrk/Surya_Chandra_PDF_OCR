"""Core processing primitives for unified scanner."""

from .pipeline import PipelineOptions, build_pdf_from_images, process_loaded_items, split_spread
from .postprocess import POSTPROCESSING_OPTIONS
from .scanner_adapter import ScanAdapterError, scan_with_document_detector

__all__ = [
    "PipelineOptions",
    "POSTPROCESSING_OPTIONS",
    "ScanAdapterError",
    "build_pdf_from_images",
    "process_loaded_items",
    "scan_with_document_detector",
    "split_spread",
]
