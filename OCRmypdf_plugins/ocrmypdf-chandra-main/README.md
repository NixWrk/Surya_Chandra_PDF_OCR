# ocrmypdf-chandra

Chandra OCR plugin for OCRmyPDF.

## What it does

1. Runs `chandra-ocr` in layout mode (`ocr_layout` by default).
2. Converts Chandra layout chunks (`bbox + content`) to hOCR.
3. Produces searchable PDF and plain text through OCRmyPDF plugin bridge.

## Local development install

```powershell
.\.venv_latest_chandra\Scripts\python.exe -m pip install -e .\OCRmypdf_plugins\ocrmypdf-chandra-main
```

## CLI usage with OCRmyPDF

```powershell
ocrmypdf --plugin ocrmypdf_chandra --force-ocr --language rus input.pdf output.pdf
```

## Notes

1. Default mode is `--chandra-method hf`.
2. `--chandra-method vllm` is supported if your vLLM endpoint is available.
