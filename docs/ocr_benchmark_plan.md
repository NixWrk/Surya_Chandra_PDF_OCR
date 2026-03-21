# OCR Benchmark Plan

## Goal

Use the provided real-world fixture
`J:\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf`
as the canonical benchmark document for OCR comparison work.

This file is large, so the benchmark must work in a sampled / ranged mode:

1. Do not load the full document into memory in a single test.
2. Use a bounded page window for automated tests.
3. Keep the full document available for manual benchmark runs.

## Benchmark Question

For the same source document, compare:

1. Recognition quality.
2. Searchable PDF quality where supported.
3. Runtime per page.
4. Memory pressure.
5. Failure mode clarity.

## Engines In Scope

### Searchable PDF engines

1. `pytesseract`
2. `ocrmypdf`
3. `pymupdf`

### Extraction / readiness engines

1. `paddleocr`
2. `surya`
3. `mineru`

## Fixture Contract

1. The canonical PDF fixture stays outside git.
2. Tests reference the file by absolute path or an overridable env var.
3. Automated tests operate on a sample page range, not the whole document.
4. Manual benchmark runs may use the full document.

Recommended default sample window:

1. First 5 pages.
2. Middle 5 pages.
3. Last 5 pages.

## Output Contract

Each benchmark run should produce:

1. A per-engine PDF or text artifact.
2. A small JSON or CSV report with timings.
3. A summary of missing dependencies and unsupported modes.

## Commit Plan

1. `docs(plan): add OCR benchmark fixture contract`
Write the benchmark rules for the external PDF fixture, including the page-sampling policy and output contract.

2. `feat(ocr-bench): add ranged PDF benchmark runner`
Add a small runner that can take a PDF path, sample page ranges, and execute the selected OCR engines on those pages.

3. `test(ocr-bench): add fixture smoke tests`
Use the provided PDF fixture in ranged mode and assert that the benchmark runner returns stable summaries.

4. `perf(ocr-bench): add timing and memory metrics`
Measure per-engine runtime and memory deltas on the sampled pages.

5. `chore(ocr-bench): add benchmark report export`
Persist the benchmark summary in a human-readable report so regressions can be compared over time.

## Exit Criteria

1. The benchmark can run on the provided PDF without loading it wholly into RAM.
2. The benchmark reports per-engine timings.
3. The benchmark produces deterministic fixture-based smoke tests.
4. The benchmark makes it obvious which engines are searchable-PDF capable and which are extraction-only.
