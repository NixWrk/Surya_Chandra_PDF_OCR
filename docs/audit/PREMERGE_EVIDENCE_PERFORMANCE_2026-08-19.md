# Premerge Evidence Reuse Performance Checkpoint — 2026-08-19

## Scope

This checkpoint addresses only the approximately 549 seconds of the preserved
`long-23p` run that were not accounted for by its recorded engine, PDF-build,
and validation stages. It does not change OCR engines, models, preprocessing,
retry policy, reconciliation, searchable-text policy, or storage retention.

The preserved before-run is under
`outputs/audit_synthetic_baseline/v1_0_1/postfix-c89b593/`. The accepted
after-run is under
`outputs/audit_synthetic_baseline/v1_0_1/premerge-fix-1cb708c-r1/`.
Both are ignored local evidence directories; neither is a document registry.

## Reproducer and cause

Commit `2ea0f6d` adds a model-free regression test. With two fresh one-page
chunks, the complete-chunk sealing phase correctly performed one full
`_required_chunk_evidence()` pass per chunk. The premerge phase performed two:

```text
seal:    [1, 2]
premerge before fix: [1, 1, 2, 2]
premerge required:   [1, 2]
```

The duplicate was structural, not inferred from timing. In
`_build_searchable_pdf_chunked_unlocked()`, `_reusable_chunk_summary()` first
computed and validated `_RequiredChunkEvidence`; the caller then immediately
discarded that value and recomputed `_required_chunk_evidence()` for the same
chunk. A full pass can rebuild a reference textless PDF, render the processing
PDF, compare pixels, verify evidence ownership and fingerprints, and validate
the evidence manifest. SHA-256 is only part of that work.

## Minimal fix and retained fences

Commit `1cb708c` makes the reusable-chunk validator return a frozen
`_ValidatedReusableChunk` containing both the summary and the evidence object.
The premerge caller reuses that object within the same validation boundary.

No integrity check was removed:

- output fingerprint and PDF/page validation still run before reuse;
- the full required-evidence pass still runs once immediately before merge;
- `_premerge_chunk_state()` still checks the previously validated inner and
  outer file fingerprints before snapshotting;
- `_validate_complete_chunk_evidence_manifest()` still runs after the snapshot;
- runtime-configuration drift checks still fence every chunk and publication.

The regression test also requires a `done` chunk manifest with sealed evidence
manifest hashes, so it cannot pass merely by skipping complete-chunk sealing.

## Software verification

```text
RED before fix:
1 failed, 71 deselected; premerge observed [1, 1, 2, 2]

Targeted GREEN:
3 passed, 69 deselected

OCR pipeline file:
70 passed, 2 skipped, 5 warnings in 152.22 s

Full repository:
679 passed, 9 skipped, 5 warnings in 208.99 s
Ruff: clean
mypy: clean across 24 source files
```

## Real-engine before/after

Both measurements use the same 23-page source SHA-256
`9e2f178711ae9aeb9e1a8b434386128c044c8851788b4559e6ad8c03663082e4`,
three chunks of 10, 10, and 3 pages, and immutable image
`sha256:f470cf1520e43ae67b70bf63e5dded12235ebde07e5139c68307cb867b06bdc0`.
The after-run used clean commit `1cb708c09f8ef5242b79d55efa5d9c9cff37ef31`,
`--pull never`, `--network none`, offline library settings, GPU0, and pre-existing
read-only model caches. No package, image, or model download occurred.

| Signal | Before | After | Observed delta |
|---|---:|---:|---:|
| Wall time | 922.991 s | 463.709 s | -459.282 s (-49.760%) |
| Surya sum | 71.043 s | 78.772 s | +7.729 s |
| Chandra sum | 297.778 s | 322.051 s | +24.274 s |
| PDF-build sum | 5.238 s | 1.822 s | -3.416 s |
| Recorded-stage residual | 548.733 s | 61.064 s | -487.669 s (-88.872%) |
| Partial page failures | 0 | 0 | unchanged |
| Output pages | 23 | 23 | unchanged |

The after residual is wall time minus the measured Surya, Chandra, and PDF-build
sums. It includes validation because validation was not separately timed in the
after-run. Engine time was about 32 seconds higher in the after-run, yet total
wall time was about 459 seconds lower.

The model-free evaluator reports:

- CER `0` and WER `0` across all 23 pages;
- exact searchable-text retention `pass` with 23 of 23 pages matched;
- page mapping `pass`;
- 23 expected and 23 observed text outcomes.

The published after-PDF is 2,954,784 bytes with SHA-256
`08243e9fd8b66e86efb79af2cba43785e8cfa39bec8f9222c720b893976b1a9e`.
The byte hash differs from the earlier PDF, but extracted text, mapping, page
count, and expected outcomes are exact.

## Limitations and exclusions

This is one comparable after-run, not a median/tail series. It proves that the
duplicate pass is gone and records a large wall-time improvement on this run;
it does not claim that every machine or run will improve by exactly 49.760%.
Peak RAM and VRAM remain unmeasured.

The image declares nested volumes including `/data/work`. Because the repeat
bound only their parent `/data`, the transient chunk work tree was not retained
after the `--rm` container exited. The final PDF, metadata, evaluator input and
report, and exact runner are retained locally. This limits a second mtime-based
stage decomposition but does not affect the recorded wall time or model-free
quality evaluation.

An excluded setup attempt explicitly invoked `/usr/local/bin/python` and failed
immediately on missing `fitz`, before OCR or model loading. The accepted run used
the image's normal `python` from `PATH`, matching the earlier command template.
