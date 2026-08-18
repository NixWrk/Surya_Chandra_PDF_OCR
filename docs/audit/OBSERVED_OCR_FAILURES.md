# Observed OCR Failures

This log records production OCR failures observed during the repository audit.
It is evidence, not a diagnosis or a statement that the current cause is fixed.

Audit baseline commit: `bbebe4bbb58c0e3a384558e24f22bc06663093c0`.

## 2026-08-18 12:17 MSK — Chandra punctuation-only geometry rejected

### Input and execution

- Document: `пульты и сигналка.pdf`.
- Source size: `24,637,424` bytes.
- Source SHA-256: `7bf57e63406ec4a9268ce328a93474ba51e3c26c9e278e1217dbf092eeb4fd85`.
- Pages: `30`.
- Mode: `chandra+surya`, language `rus+eng`, strict mode enabled.
- Original text-layer removal: enabled.
- Chunk size: `10` pages.
- Run key: `97d0f95718c86d59d4969b81d2b7b5d35e459992a23fda09b8cf2f3857a2b631`.
- First-chunk input artifacts were written at approximately `11:03:50 MSK`.
- Failure manifest was written at `12:17:30 MSK`.

The full external source path is intentionally omitted from this tracked file.
It remains present in the local ignored run manifest.

### Failure signature

Chunk 1, source pages 1–10, failed during strict page reconciliation:

```text
Strict OCR benchmark is incomplete: failed: page reconciliation: unresolved pages: [3]
```

Both OCR engines themselves reported success for all ten pages:

- Surya: `status=ok`, `109.0266914` seconds, `39,047` text characters.
- Chandra: `status=ok`, `4,296.2809659` seconds, `86,895` text characters.

Source page 3 was rejected with:

```text
reason: invalid_chandra_attempt_evidence
retry_evidence_error: Chandra attempt 1 has invalid line text
```

Chandra attempt 1 contained 36 parsed geometry lines. Five lines—indices
23, 26, 29, 32, and 35—contained only U+2014 EM DASH (`—`). The strict
validator canonicalizes a line to alphanumeric characters and rejects it when
that canonical value is empty. Therefore these punctuation-only table cells
made otherwise successful page OCR evidence invalid.

This is an observed failure mechanism. Whether punctuation-only geometry
should be retained, filtered before sealing, or represented differently must
be decided with a reproducing test and searchable-text retention checks.

### Outcome and retained evidence

- Chunk 1 status: `error`.
- Chunks 2 and 3 remained `pending`; they were not processed.
- No final or candidate searchable PDF was published.
- The source PDF still has the recorded size and SHA-256, so no overwrite was observed.
- The run directory retains 114 files totalling `142,380,999` bytes.

Primary local evidence paths, relative to the repository:

- `outputs/service_runs/hybrid_chunk_cache/hybrid_97d0f95718c86d59d4969b81d2b7b5d35e459992a23fda09b8cf2f3857a2b631/chunk_manifest.json`
- `outputs/service_runs/hybrid_chunk_cache/hybrid_97d0f95718c86d59d4969b81d2b7b5d35e459992a23fda09b8cf2f3857a2b631/chunk_runs/chunk_0001/chunk_0001_p0001_0010_20260818_110354_3e2b388c/page_reconciliation.json`
- `outputs/service_runs/hybrid_chunk_cache/hybrid_97d0f95718c86d59d4969b81d2b7b5d35e459992a23fda09b8cf2f3857a2b631/chunk_runs/chunk_0001/chunk_0001_p0001_0010_20260818_110354_3e2b388c/chandra/chandra/page_0003.chandra-attempts/attempt_1/chandra_attempt.json`

These paths are ignored by Git and must be preserved until a minimal fixture
and reproducing test have been derived.

### Difference from the preceding failure

The preceding production failure was job `810c83aa4b53` for `instruction.pdf`,
recorded at `2026-08-18 10:09:54 MSK`. It failed in chunk 2 (source pages
11–20) during searchable-PDF validation:

```text
Output PDF page 3 failed exact searchable text retention.
```

The new incident is a different document, chunk, source page, validation
stage, and error signature. They must remain separate benchmark/regression
cases.
