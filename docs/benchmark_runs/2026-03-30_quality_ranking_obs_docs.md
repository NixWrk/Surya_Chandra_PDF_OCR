# OCR Quality Ranking: OBS Documents (2026-03-30)

## Fixed Result

Quality ranking (manual review):

1. `chandra`
2. `olmocr`
3. `surya`

## Scope

Documents:

1. `ГОСТ с плохим качеством скана.pdf`
2. `Старая книга с частично рукописным текстом.pdf`

Compared engines:

1. `chandra`
2. `olmocr`
3. `surya`

## What Was Compared

Primary comparison was done on final text artifacts (`.txt`) for the same source PDFs.

Main comparison folder:

- `artifacts/ocr_obs_gost_oldbook_20260327_165121/_compare_txt`

Latest fixed GOCT/ГОСТ olmocr artifact additionally used:

- `artifacts/ocr_olmocr_gost_20260330_fix/olmocr/ГОСТ с плохим качеством скана_olmocr.txt`

## Evaluation Parameters

1. OCR text readability and semantic correctness (Russian content quality).
2. Completeness of extracted content (missing fragments/pages).
3. Structural fidelity (headings/lists/tables/formatted fragments represented as readable text).
4. Noise level (garbage symbols, broken escapes, hallucinated fragments).

Benchmark metadata (`status`, `elapsed_seconds`, `text_chars`, `artifact_path`) was used as supporting signals, but ranking above is quality-first.

## Notes

1. This ranking is for current OBS document set and current engine/runtime configuration.
2. It is not a speed ranking.
