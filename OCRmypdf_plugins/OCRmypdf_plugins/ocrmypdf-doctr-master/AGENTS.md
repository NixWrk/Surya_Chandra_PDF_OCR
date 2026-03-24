# AGENTS.md

## Project Overview

This is an OCRmyPDF plugin that uses docTR (Document Text Recognition by Mindee) as the OCR backend. It follows the same plugin architecture as the reference implementations in `references/ocrmypdf-paddleocr` and `references/ocrmypdf-azure`.

## Architecture

- **Plugin pattern**: Module-level `@hookimpl` functions (`add_options`, `check_options`, `get_ocr_engine`) register the plugin with OCRmyPDF
- **OcrEngine subclass**: `DoctrOCREngine` implements the OCR interface with `generate_hocr()` as the core method
- **hOCR intermediate**: OCR results are converted to hOCR XML, then to PDF via OCRmyPDF's `HocrTransform`
- **Nix build**: `flake.nix` builds python-doctr v1.0.1 from GitHub source and wraps the ocrmypdf binary with the plugin pre-loaded

## Key Files

| File | Purpose |
|---|---|
| `src/ocrmypdf_doctr/plugin.py` | Main plugin: CLI options, model setup, hOCR generation, PDF output |
| `src/ocrmypdf_doctr/__init__.py` | Package init with version from setuptools_scm |
| `pyproject.toml` | Python package metadata and `[project.entry-points.ocrmypdf]` registration |
| `flake.nix` | Nix flake: builds python-doctr, the plugin, wrapped binary, and dev shell |

## Build & Test

```bash
nix build . --extra-experimental-features 'nix-command flakes'
./result/bin/ocrmypdf --force-ocr -l deu test_pdfs/0301_250718151728_001.pdf output.pdf
```

## Key Technical Details

### docTR coordinate system
docTR returns **normalized** bounding box coordinates `((xmin, ymin), (xmax, ymax))` where values are 0.0-1.0 relative to page dimensions. Multiply by `width`/`height` to get pixel coordinates.

### docTR output hierarchy
`Document > Page > Block > Line > Word` — each level has `.geometry` (normalized bbox) and words have `.value` (text) and `.confidence` (0-1 float).

### Word bbox gap adjustment
docTR's word bounding boxes are tight around text and often touch or overlap with no whitespace gap. OCRmyPDF's `HocrTransform` derives inter-word spaces from the gap between consecutive word bboxes (`next.x_min - curr.x_max`). If this gap is zero or negative, no space character is rendered in the PDF text layer, causing `pdftotext` to concatenate words. The plugin enforces a minimum 5px gap between consecutive word bboxes by symmetrically shrinking adjacent boxes before emitting hOCR.

### opencv-python shim
docTR depends on `opencv-python` but nixpkgs provides `opencv4`. The flake uses `pythonRemoveDeps` to strip the requirement from the wheel metadata; the actual `cv2` module comes from nixpkgs' `opencv4`.

### Recognition model: multilingual PARSeq from HF Hub
The default recognition model is `Felix92/doctr-torch-parseq-multilingual-v1`, loaded from Hugging Face Hub via `from_hub()`. It supports 13 European languages including German. docTR's built-in models (e.g. `crnn_vgg16_bn`) only support the French character set. If `--doctr-reco-arch` contains `/`, it's treated as an HF Hub model ID; otherwise as a built-in architecture name.

## Reference Plugins

- `references/ocrmypdf-paddleocr/` — PaddleOCR plugin (local model, most similar pattern)
- `references/ocrmypdf-azure/` — Azure Document Intelligence plugin (cloud API)
- `references/doctr/` — docTR v1.0.2a0 source (for API reference only, not used in build)
