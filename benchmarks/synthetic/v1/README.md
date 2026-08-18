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

## Model-free evaluation

The model-free evaluator consumes a JSON run record and never imports or invokes
Surya, Chandra, Torch, or another OCR model. It compares caller-provided
searchable text (for example, text extracted from an output PDF) with the
versioned Ground Truth:

~~~powershell
python -m runpy benchmarks.synthetic.v1.evaluate --corpus .tmp_runtime\synthetic-v1 --run .tmp_runtime\run.json --output .tmp_runtime\evaluation.json
~~~

The input contract is formalized in evaluation.schema.json. Each run must
include stage timing for surya, chandra, pdf_build, and validation; peak
RAM/VRAM; engine invocation and render counts; chunk reuse status, reused-chunk
count, and rerun reasons; and output status, path, SHA-256, bytes, and page
count. Measurements that were not collected use not_measured; stages that did
not run use not_run.

The evaluator reports codepoint CER, whitespace-token WER, exact normalized
searchable-text retention, source-to-output page mapping, and expected/observed
page outcomes. accepted_zero_output is an accepted page outcome alongside text,
verified_blank, and explicit_nontext. A report is measurement data, not an
accepted OCR quality or performance baseline.

Output path/hash/bytes/page_count fields are externally attested input metadata;
the evaluator does not open, re-hash, or infer page counts from that file.
Geometry and resource fields remain contract-only when their value is a sentinel.

A minimal run record (with page text obtained by the caller) is:

~~~json
{
  "schema": "uniscan.synthetic-evaluation-input.v1",
  "run_id": "offline-contract-v1",
  "model_status": "not_run",
  "fixtures": [{
    "fixture_id": "clean-en",
    "pages": [{
      "source_page": 1,
      "output_page": 1,
      "searchable_text": "Ground Truth text",
      "outcome": "text"
    }]
  }],
  "measurements": {
    "stage_timing_seconds": {
      "surya": "not_run",
      "chandra": "not_run",
      "pdf_build": "not_run",
      "validation": "not_run"
    },
    "peak_ram_bytes": "not_measured",
    "peak_vram_bytes": "not_measured",
    "engine_invocations": "not_run",
    "render_count": "not_run",
    "chunk_reuse": {
      "status": "not_run",
      "reused_chunks": "not_run",
      "rerun_reasons": "not_run"
    }
  },
  "output": {
    "status": "not_run",
    "path": "not_run",
    "sha256": "not_run",
    "bytes": "not_measured",
    "page_count": "not_measured"
  }
}
~~~
