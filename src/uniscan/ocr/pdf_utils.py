"""Shared PDF rendering helpers for OCR workflows."""

from __future__ import annotations

import os
from pathlib import Path

from uniscan.io.loaders import _safe_render_dpi


def _textless_jpeg_quality() -> int:
    raw = os.getenv("UNISCAN_TEXTLESS_JPEG_QUALITY", "").strip()
    if not raw:
        return 85
    try:
        quality = int(raw)
    except ValueError:
        return 85
    if quality <= 0:
        return 0
    return min(quality, 100)


def _build_textless_source_pdf(*, source_pdf: Path, out_pdf: Path, dpi: int = 300) -> Path:
    """Render PDF pages into an image-only PDF to remove existing text."""
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError(
            "Removing original text layer requires PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    requested_dpi = max(int(dpi), 72)
    jpeg_quality = _textless_jpeg_quality()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    src_doc = fitz.open(str(source_pdf))
    dst_doc = fitz.open()
    try:
        for src_page in src_doc:
            safe_dpi = _safe_render_dpi(src_page.rect, requested_dpi)
            scale = float(safe_dpi) / 72.0
            matrix = fitz.Matrix(scale, scale)
            pix = src_page.get_pixmap(matrix=matrix, alpha=False)
            dst_page = dst_doc.new_page(
                width=float(src_page.rect.width),
                height=float(src_page.rect.height),
            )
            if jpeg_quality > 0:
                image_stream = pix.tobytes("jpeg", jpg_quality=jpeg_quality)
                dst_page.insert_image(dst_page.rect, stream=image_stream, keep_proportion=False)
            else:
                dst_page.insert_image(dst_page.rect, pixmap=pix, keep_proportion=False)
        dst_doc.save(str(out_pdf), garbage=4, deflate=True)
    finally:
        src_doc.close()
        dst_doc.close()

    return out_pdf
