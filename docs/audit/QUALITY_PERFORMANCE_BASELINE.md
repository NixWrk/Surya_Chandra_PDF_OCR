# Quality and Performance Baseline

Audit code baseline: `bbebe4bbb58c0e3a384558e24f22bc06663093c0`.

Status: the initial software snapshot below is retained for audit history. The
current accepted checkpoints and limitations are summarized here; exact raw
evidence and provenance are in `SYNTHETIC_OCR_BASELINE_2026-08-18.md` and
`PRIVATE_OCR_BASELINE_2026-08-18.md`.

## Current accepted checkpoint

| Signal | Result |
|---|---|
| Full software suite | 679 passed, 9 skipped, 5 warnings in 208.99 s through `1cb708c` |
| Static checks | Ruff and mypy clean |
| Synthetic corpus | Version 1.0.1, nine fixtures with hashed sources and Ground Truth |
| Real-engine coverage | All nine fixtures exercised at least once across two immutable offline checkpoints |
| Added four-fixture accuracy | CER 0, WER 0, exact retention pass, page mapping pass |
| Known accuracy target | `mixed-layout`: CER 0.287293, WER 0.366667, exact retention fail, mapping pass |
| Long-document after fix | 23 pages, three chunks, all checks pass, 463.709 s wall; 49.760% faster than the preserved run |
| Resource/latency limitation | Peak RAM/VRAM and median/tail latency not measured |

The second checkpoint used clean source commit `3acda85`, immutable image
`sha256:f470cf1520e43ae67b70bf63e5dded12235ebde07e5139c68307cb867b06bdc0`,
`--pull never`, `--network none`, offline library settings, and read-only model
caches. Source hashes match the corpus manifest; no download occurred.

The 23-page gap was diagnosed as repeated strict evidence work. A deterministic
test proved two full evidence passes per chunk in the immediate premerge loop.
Commit `1cb708c` reuses the first validated evidence result without removing the
sealing, fingerprint, manifest, or runtime-drift fences. On one comparable
offline after-run, wall time fell from 922.991 to 463.709 seconds while CER/WER
remained zero and exact retention and mapping passed. The residual not explained
by measured engine/PDF-build stages fell from 548.733 to 61.064 seconds; the new
residual includes separately unmeasured validation. This is accepted evidence
for the narrow fix, not a median/tail latency claim. See
`PREMERGE_EVIDENCE_PERFORMANCE_2026-08-19.md`.

The versioned corpus includes procedural English, Russian, mixed layout,
retention, graphics/blank, degraded vector text, rotation, native text, and a
23-page chunked document. It is sufficient to block unmeasured OCR tuning, but
not to claim production accuracy: representative raster scans, denser layouts,
reviewed private Ground Truth, repeat runs, and resource peaks are still missing.

Resume provenance is intentionally narrow. It identifies only an active or
explicitly retained same-deployment run so interrupted chunks cannot be mixed
with another source/configuration. It is not a persistent history of user
documents; successful HTTP working data is removed by default.

## Current acceptance rule

Do not change Surya/Chandra versions or OCR policy until a written hypothesis is
tested against the affected fixture and unchanged fixtures, with OCR quality,
page integrity, runtime, and available resource deltas recorded. Profile before
removing or weakening strict validation.

## Initial software-quality snapshot (historical)

| Signal | Result |
|---|---|
| Full suite | 617 passed, 7 skipped, 5 warnings |
| Test duration | 248.78 seconds; 250.966 seconds wall time |
| Branch-coverage suite | Completed in 261.91 seconds |
| Total branch coverage | 74% |
| Ruff | Clean |
| mypy | Clean across 24 checked files |
| Container/check-out source comparison | 23 of 23 tracked Python source files match; `MismatchCount=0` |

These results establish regression protection for existing tested contracts. They
do not establish OCR accuracy or production throughput.

## Initial OCR quality state (historical)

There is no tracked representative PDF corpus, Ground Truth manifest, CER/WER
baseline or accepted layout-scoring baseline. Therefore no accuracy improvement
claim is currently measurable from a clean clone.

The repository does have strict structural and evidence checks, including:

- source-raster identity and retry lineage;
- Chandra/Surya page bijection;
- explicit blank/non-text/zero-output outcomes;
- exact searchable-text retention;
- visual retention for accepted textless graphics;
- chunk output/evidence/source revalidation before merge.

These are correctness guards, not substitutes for transcription and layout
quality metrics.

## Observed production evidence

Two separate failures are recorded in `docs/audit/OBSERVED_OCR_FAILURES.md`.

### Punctuation-only Chandra geometry

- 30-page document, strict `chandra+surya`, chunk size 10.
- Chunk 1 failed on source page 3.
- Surya reported success in 109.0266914 seconds with 39,047 characters.
- Chandra reported success in 4,296.2809659 seconds with 86,895 characters.
- Reconciliation rejected Chandra attempt evidence because five geometry lines
  contained only U+2014 EM DASH and canonicalized to no alphanumeric text.
- No final PDF was published; source hash/size remained unchanged.

The timings are an incident observation, not a representative performance
benchmark: the run failed, used one document and did not complete all chunks.

### Exact searchable-text retention failure

The preceding `instruction.pdf` incident failed in a different document and chunk
at final searchable-PDF validation:

```text
Output PDF page 3 failed exact searchable text retention.
```

It must remain a separate regression case from punctuation-only evidence rejection.

## Missing quality measurements

An accepted benchmark corpus must include:

- Russian, English and mixed-language pages;
- clean born-digital pages and noisy scans;
- skew, rotation and low contrast;
- tables and punctuation-only cells;
- blank pages and textless graphics;
- existing native text layers;
- minimal reproductions of both recorded incidents;
- small, medium and long documents for chunk/resume measurements.

Per-document measurements should record:

| Dimension | Minimum metric/evidence |
|---|---|
| Text accuracy | CER and WER against Ground Truth where licensing/privacy permits |
| Searchability | Extracted-text retention and page-level searchable-text presence |
| Layout | Reading-order and region/line placement assertions |
| Page policy | Counts of text, verified blank, explicit non-text, accepted retry and unresolved pages |
| Runtime | Wall time by render, Chandra, Surya, reconcile, assembly, validation and merge |
| Resources | Peak RAM and VRAM |
| Rework | Render count, engine invocation count, reused/rerun chunks and cache-hit reason |
| Output | Page count, byte size, validation result and source/output hashes |

## Benchmark provenance

Every accepted result must record:

- repository commit and dirty-state flag;
- source document hash and fixture/ground-truth revision;
- resolved configuration and page selection;
- Python and package versions;
- Chandra and Surya model revisions/digests;
- PyTorch, CUDA, driver and GPU identity;
- pipeline, artifact and validator schema revisions;
- warm/cold cache state;
- command and raw machine-readable output.

Without this provenance, comparisons are exploratory only.

## Initial quality gate (historical)

Before any accuracy or speed modification:

1. Add a minimal reproducing test for corrupt/tampered job-result recovery.
2. Add a minimal reproducing test for job-result symlink escape.
3. Add a deterministic reproducing test for stale-worker publication after
   watchdog reclamation.
4. Apply only the minimal fixes proven necessary by those tests.
5. Derive privacy-safe fixtures for the two OCR incidents.
6. Record the first representative quality/performance baseline.

The first three items protect result integrity and user data; they precede OCR
algorithm tuning.

## Acceptance rule for OCR changes

An OCR behavior change is accepted only when:

- its hypothesis is written before implementation;
- the affected fixture reproduces the old failure;
- unchanged fixtures do not regress beyond an agreed threshold;
- searchable-text and page-count validators pass;
- runtime/resource deltas are recorded;
- the result is attributable to one small reversible change.

## Assumptions

- Retained production artifacts may contain private data and stay local/ignored.
- Minimal fixtures will be anonymized or cropped only after preserving the failure
  signature and obtaining an equivalent reproducer.
- No engine/model upgrade is justified by the current software test baseline alone.
- The measured full-suite and coverage times are machine-specific reference points,
  not deployment SLAs.
