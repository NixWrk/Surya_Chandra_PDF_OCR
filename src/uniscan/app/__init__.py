"""Application-level services for OCR workflows."""

from .ocr_pipeline import (
    DEFAULT_BASIC_GUI_LANG,
    MODE_BOTH,
    MODE_HYBRID,
    MODE_SURYA,
    MODE_TO_ENGINES,
    BasicOcrRunSummary,
    ChandraGeometryVariantsSummary,
    build_chandra_geometry_variants,
    run_basic_ocr_benchmark,
)
from .page_spec import parse_page_numbers

__all__ = [
    "DEFAULT_BASIC_GUI_LANG",
    "MODE_BOTH",
    "MODE_HYBRID",
    "MODE_SURYA",
    "MODE_TO_ENGINES",
    "BasicOcrRunSummary",
    "ChandraGeometryVariantsSummary",
    "build_chandra_geometry_variants",
    "parse_page_numbers",
    "run_basic_ocr_benchmark",
]

