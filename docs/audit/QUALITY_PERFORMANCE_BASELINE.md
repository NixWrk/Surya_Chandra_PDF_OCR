# Quality and Performance Baseline

Audit code baseline: `bbebe4bbb58c0e3a384558e24f22bc06663093c0`.

## Current verified software-quality baseline

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

## Current OCR quality baseline

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

## Immediate quality gate

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
