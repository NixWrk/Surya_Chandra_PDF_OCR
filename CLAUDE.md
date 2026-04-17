# Surya Chandra PDF OCR Project

This is a hybrid OCR project that combines:
1. OCR text from "Chandra" 
2. Geometry/coordinates from "Surya"
3. Building searchable PDFs from artifacts

## Key Components

- `src/uniscan/app/page_spec.py` — unified page parser (1,3,5-8)
- `src/uniscan/app/ocr_pipeline.py` — OCR/artifact workflow orchestration
- Web-ready application layer with HTTP API
- Support for multiple modes: chandra, surya, chandra+surya

## Commands

```powershell
python -m uniscan benchmark-ocr --help
python -m uniscan prepare-compare-txt --help
python -m uniscan build-searchable-from-artifacts --help
python -m uniscan compare-chandra-geometry --help
python -m uniscan searchable-pdf --help
python -m uniscan serve-http --help
```

## Modes

1. `chandra` - OCR text only
2. `surya` - Geometry only  
3. `chandra+surya` (default) - Hybrid approach

## Web API

- GET / - web GUI
- GET /health - health check
- POST /api/jobs - create async OCR job
- GET /api/jobs/{id} - get job status
- GET /api/jobs/{id}/result - download result
- POST /searchable-pdf - sync endpoint

## Key Files

- `unified_pdf_tool.py` - main PDF processing tool
- `camscan_hybrid_tool.py` - hybrid OCR tool
- `img_2_pdf.py` - image to PDF conversion
- `only_tesseract.py` - tesseract-only OCR