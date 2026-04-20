"""OCR pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import img2pdf


def build_pdf_from_images(image_paths: list[Path], out_pdf: Path, dpi: int) -> None:
    """Build one merged PDF from image paths."""
    if not image_paths:
        raise ValueError("No image paths were provided.")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as file:
        try:
            payload = img2pdf.convert([str(p) for p in image_paths], dpi=dpi)
        except TypeError:
            layout = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
            payload = img2pdf.convert([str(p) for p in image_paths], layout_fun=layout)
        file.write(payload)
