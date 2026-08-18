# OCR benchmark runbook

No engine, model, preprocessing, prompt, geometry, retry, or performance change
may be accepted without a versioned before/after benchmark.

## Offline corpus contract

Generate the procedural corpus only into a temporary or ignored directory:

```powershell
python -m runpy benchmarks.synthetic.v1.generate --output <temporary-path>
python -m runpy benchmarks.synthetic.v1.generate --output <temporary-path> --check
```

This step is deterministic and model-free. Its manifest must report
`model_status: not_run`. The current degraded fixture is vector text, not a
raster scan, so it cannot establish real scanned-document accuracy.

## Accepted real baseline

A real corpus must have explicit provenance, privacy/license status, immutable
source hashes, page ranges, Ground Truth version, language/layout groups, and
historical failure fixtures. Do not commit private source documents.

Record before execution:

- commit and dirty state;
- command and resolved run configuration;
- Python, package, engine, model, PyTorch, CUDA, driver, and GPU identity;
- cold/warm model and application cache state;
- source, Ground Truth, and manifest hashes.

Record machine-readable results for:

- CER/WER where Ground Truth exists;
- omissions, hallucinations, and accepted page outcomes;
- exact searchable-text retention, page mapping, reading order, and geometry;
- Chandra, Surya, rendering, reconciliation, PDF build, and validation time;
- peak RAM/VRAM, invocation/render counts, cache hits, reused/rerun chunks;
- result hash, bytes, and page count;
- every error, retry, skip, and unavailable metric.

## Experiment discipline

Change one attributable variable at a time. Run enough repetitions to report a
median and tail latency. Compare cold and warm runs separately. Reject a variant
that improves aggregate CER or speed while introducing a page-level omission,
evidence failure, exact-retention regression, or unacceptable VRAM increase.

Store raw results immutably and create a new accepted baseline; never overwrite
a rejected or previous baseline. Surya/Chandra upgrades require release-note and
compatibility review plus the same benchmark gate.

## Verification

Before publishing a baseline, validate corpus/result hashes, rerun within the
documented tolerance, and run the repository test/static-check gate. State
clearly which metrics were not measured. A synthetic contract-only run is not a
production accuracy or performance baseline.
