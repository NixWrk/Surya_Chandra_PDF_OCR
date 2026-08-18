# Offline synthetic benchmark v1

This directory contains a deterministic, privacy-safe recipe and metadata for
procedural OCR fixtures. It contains no private PDFs, downloaded models, or
external assets. Generated PDFs are written only to the caller-selected output
directory and are not repository artifacts.

Generate and validate into an ignored or temporary directory:

```powershell
python -m runpy benchmarks.synthetic.v1.generate --output .tmp_runtime\synthetic-v1
python -m runpy benchmarks.synthetic.v1.generate --output .tmp_runtime\synthetic-v1 --check
```

The generator uses only Python's standard library and the repository's existing
PyMuPDF dependency. It fixes the random seed, PDF metadata, and document IDs;
rerunning with the same Python/PyMuPDF versions yields identical fixture bytes
and SHA-256 values. `manifest.json` records the generator, provenance, hashes,
fixture matrix, cases, and `metrics.model_status: not_run`.

The `degraded-vector-text` fixture is intentionally vector text with low
contrast, deterministic line noise, skew metadata, and a source-DPI claim. It
is not a raster scan; the name and metadata make that limitation explicit.
The `native-text-layer` fixture intentionally contains visible text plus an
invisible render-mode-3 duplicate layer. It is a contract fixture for removing
or replacing a pre-existing text layer before OCR, not a clean OCR input.
