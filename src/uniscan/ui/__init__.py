"""UI package for OCR pipeline.

Keep imports lazy so ``python -m uniscan.ui.basic_ocr_gui`` does not emit
runpy re-import warnings.
"""

from __future__ import annotations

from typing import Any


def run_basic_gui() -> int:
    from .basic_ocr_gui import main

    return main()


def __getattr__(name: str) -> Any:
    if name == "BasicOcrGui":
        from .basic_ocr_gui import BasicOcrGui

        return BasicOcrGui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BasicOcrGui", "run_basic_gui"]
