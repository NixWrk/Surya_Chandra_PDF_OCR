# 2026-04-13: Chandra Text With Two Geometry Modes

Goal: build two **real searchable PDFs** with the same `chandra` text layer and different geometry sources.

## Inputs

1. Benchmark run root:
   - `outputs/basic_gui_runs/ГОСТ с плохим качеством скана_20260410_182529`
2. Source PDF root:
   - `O:\OBS_TEST\PDF2OBS\PDFS`
3. Python executable:
   - `.\.venv\Scripts\python.exe`

## One-command Reproduction

```powershell
.\scripts\compare_chandra_geometry_variants.ps1 `
  -RunRoot "D:\Git_Code\Surya_Chandra_PDF_OCR\outputs\basic_gui_runs\ГОСТ с плохим качеством скана_20260410_182529" `
  -PdfRoot "O:\OBS_TEST\PDF2OBS\PDFS"
```

This script does three stages:

1. `prepare-compare-txt` for `chandra` and `surya`.
2. Build variant A: `chandra` text + `chandra` geometry.
3. Build variant B: `chandra` text + `surya` geometry (`UNISCAN_CHANDRA_GEOMETRY_DIR=<run>\surya`).

Direct CLI equivalent (single command):

```powershell
.\.venv\Scripts\python.exe -m uniscan compare-chandra-geometry `
  --run-root "D:\Git_Code\Surya_Chandra_PDF_OCR\outputs\basic_gui_runs\ГОСТ с плохим качеством скана_20260410_182529" `
  --pdf-root "O:\OBS_TEST\PDF2OBS\PDFS" `
  --strict
```

## Manual Commands (Equivalent)

```powershell
$run = "D:\Git_Code\Surya_Chandra_PDF_OCR\outputs\basic_gui_runs\ГОСТ с плохим качеством скана_20260410_182529"

.\.venv\Scripts\python.exe -m uniscan prepare-compare-txt `
  --benchmark-root "$run" `
  --output "$run\_compare_txt" `
  --engines chandra surya `
  --strict

.\.venv\Scripts\python.exe -m uniscan build-searchable-from-artifacts `
  --compare-dir "$run\_compare_txt" `
  --pdf-root "O:\OBS_TEST\PDF2OBS\PDFS" `
  --output "$run\searchable_pdf_geometry_compare\chandra_text__chandra_geometry" `
  --engines chandra `
  --strict

$env:UNISCAN_CHANDRA_GEOMETRY_DIR = "$run\surya"
.\.venv\Scripts\python.exe -m uniscan build-searchable-from-artifacts `
  --compare-dir "$run\_compare_txt" `
  --pdf-root "O:\OBS_TEST\PDF2OBS\PDFS" `
  --output "$run\searchable_pdf_geometry_compare\chandra_text__surya_geometry" `
  --engines chandra `
  --strict
```

## Output Files

1. Chandra text + Chandra geometry:
   - `...\searchable_pdf_geometry_compare\chandra_text__chandra_geometry\ГОСТ с плохим качеством скана\ГОСТ с плохим качеством скана__chandra_searchable.pdf`
2. Chandra text + Surya geometry:
   - `...\searchable_pdf_geometry_compare\chandra_text__surya_geometry\ГОСТ с плохим качеством скана\ГОСТ с плохим качеством скана__chandra_searchable.pdf`

## Important Runtime Notes

1. For stable overlay quality, `surya` must produce `geometry_file` entries in `surya/pages.json`.
2. For `chandra` quality, do not force CLI-only degradation path unless intentionally debugging fallback behavior.
3. Build with `--strict` to fail fast on missing artifacts.
