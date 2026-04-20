"""Minimal PDF export helpers for OCR pipeline tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from uniscan.core.pipeline import build_pdf_from_images
from uniscan.io.loaders import imwrite_unicode


def export_pages_as_pdf(
    pages: Sequence[np.ndarray],
    *,
    out_pdf: Path,
    dpi: int = 300,
) -> Path:
    """Export image arrays into one merged PDF."""
    if len(pages) == 0:
        raise ValueError("No pages to export.")

    out_pdf = out_pdf.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="uniscan_pdf_") as tmp:
        tmp_dir = Path(tmp)
        image_paths: list[Path] = []
        for idx, page in enumerate(pages, start=1):
            page_path = tmp_dir / f"{idx:05d}.png"
            if not imwrite_unicode(page_path, page):
                raise RuntimeError(f"Failed to write temporary page image: {page_path}")
            image_paths.append(page_path)
        build_pdf_from_images(image_paths, out_pdf=out_pdf, dpi=int(dpi))

    return out_pdf
