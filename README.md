# img_2_pdf

Unified local toolkit for building convenient PDFs from photos/PDFs with optional OCR.

## Recommended entrypoint

For your target workflow ("images folder or single file -> preprocessed PDF -> optional OCR"), run:

```powershell
python unified_pdf_tool.py
```

`unified_pdf_tool.py` combines the practical pieces into one GUI app:

1. Input mode: `Images folder` or `Single file (image/PDF)`
2. Image preprocessing: auto perspective, deskew + crop, optional spread split
3. Output profiles: `Fast`, `Balanced`, `Best quality`
4. Optional OCR stage via `ocrmypdf`

## Script map

| File | Purpose | Input -> Output | Main dependencies |
|---|---|---|---|
| `unified_pdf_tool.py` | Primary unified workflow for this repo | Images folder / image / PDF -> PDF (optional OCR) | `opencv-python`, `numpy`, `img2pdf`, optional `ocrmypdf` |
| `fast.py` | Advanced OCR GUI with batch PDF mode and detailed OCRmyPDF integration | Images/PDF -> searchable PDF | `ocrmypdf`, `pypdf`, `img2pdf`, Tesseract, Ghostscript, qpdf |
| `img_2_pdf.py` | Photo -> PDF app with OpenCV preprocessing and optional OCR | Images folder -> PDF | `opencv-python`, `numpy`, `img2pdf`, optional `ocrmypdf` |
| `only_tesseract.py` | OCR pipeline without `ocrmypdf` (direct `tesseract.exe`) | Images/PDF -> searchable PDF | Tesseract, `pypdf`, Poppler (`pdftoppm`, `pdfunite`) |
| `imgs_and_pdfs_ocr_fast_STABLE.py` | Stable previous version of OCR GUI | Images/PDF -> searchable PDF | `ocrmypdf`, `pypdf`, Tesseract |
| `prepare pdf to tesseract.py` | PDF preconditioning (render + downscale + JPEG) | PDF -> prepared PDF | `PyMuPDF`, `Pillow` |
| `naps2-7.5.3-win.exe` | NAPS2 installer | - | - |

## Quick start

```powershell
python unified_pdf_tool.py
```

Typical use:

1. Choose input mode (`Images folder` or `Single file`).
2. Select output PDF path.
3. Keep profile as `Balanced` for default quality/speed.
4. Leave preprocessing enabled for photo scans.
5. Enable OCR only when searchable text is needed.

## Dependencies

Python packages (depending on selected script):

```powershell
pip install ocrmypdf pypdf img2pdf opencv-python numpy pillow pymupdf
```

External tools (required for OCR-heavy flows):

1. Tesseract OCR
2. Ghostscript
3. qpdf
4. Poppler (`pdftoppm`, `pdfunite`) for `only_tesseract.py` PDF mode

## Last modified by git history

Latest script update in repository history:

1. `img_2_pdf.py` -> commit `0cfb6e7` (`2026-02-11 01:02:50 +03:00`)
2. `fast.py`, `imgs_and_pdfs_ocr_fast_STABLE.py`, `only_tesseract.py`, `prepare pdf to tesseract.py` -> commit `5c408eb` (`2026-02-11 01:01:40 +03:00`)
