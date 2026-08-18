# Private real-engine OCR baseline — 2026-08-18

Status: accepted as an incident-regression and stage-timing baseline. It is not
an accepted CER/WER accuracy baseline because reviewed Ground Truth is not yet
available.

Private source PDFs and raw OCR outputs remain ignored under `outputs/`. This
tracked record contains logical identifiers, hashes, selected pages, results,
and limitations, but no external private source path or extracted document text.

## Runtime identity

- Source commit: `800041e965c8c02bc624d34e727354004db70a21`.
- Source worktree: clean before and after both runs.
- Source loading: repository mounted read-only with `PYTHONPATH=/app/src`; import
  attestation resolved to `/app/src/uniscan/__init__.py`.
- Runtime image: `surya-chandra-ocr@sha256:f470cf1520e43ae67b70bf63e5dded12235ebde07e5139c68307cb867b06bdc0`.
- Python: `3.11.15`; PyTorch: `2.11.0+cu126`.
- GPU: NVIDIA RTX 6000 Ada Generation, container-visible `cuda:0`.
- Engines: `surya-ocr==0.17.1`, `chandra-ocr==0.2.0`.
- Mode: `chandra+surya`; language: `rus+eng`; strict: enabled.
- Model caches: warm and pre-existing; network/model downloads: none.
- Application/engine processes: new container and subprocesses per case. The
  repeated punctuation run reused model caches but did not reuse a live process.

The full exact-retention runs loaded OCR source commit
`14e81878f02331e3a05c8dc5a1841e8856ae765e`. Later HTTP-only admission commits
`d0085fb` and `51ca830` did not affect that loaded OCR pipeline.

The first attempted run stopped before engine execution because the benchmark
helper tried to create `/app/.tmp_runtime` under the read-only source mount. Both
accepted runs used a dedicated ignored writable mount at that path. This exposes
a runtime-temp/repository coupling risk; it did not alter OCR policy.

## Case 1 — exact-retention incident page

- Logical source: `exact-retention-810c83aa4b53`.
- Source SHA-256: `263f72216aba6d6f4ef2b57d68235bf5c94e0494e4e89ba7dd597e4a3456766c`.
- Source bytes/pages: `9,617,844` / `21`.
- Selected source page: `13` (chunk 2 local/output page 3 in the historical
  pages 11–20 failure).
- Chandra: `ok`, `156.308895686` seconds, `2,200` reported text characters,
  `1,848.656` MB memory delta, one attempt.
- Surya: `ok`, `24.978869870` seconds, `2,177` reported text characters,
  memory delta unavailable, one attempt.
- Searchable-PDF assembly: `ok`, `3.101515069` seconds.
- Strict page reconciliation: `ok`.
- Result bytes/pages: `13,893,973` / `21`.
- Result SHA-256: `f0dc9b55d0d777c26cc5268a2d7d495ab7f388c61901b279cc68fa85dea2d7a8`.

The selected-page run did not reproduce the historical exact-retention error.
It did not by itself close the incident because the historical failure occurred
while assembling the full pages 11–20 chunk. The full-context result follows.

### Full 21-page chunked regression

The same logical source was processed end-to-end in three chunks: pages 1–10,
pages 11–20, and page 21. The final manifest reported `done`, output page count
`21`, and `0` partial-page failures. Every chunk passed strict page
reconciliation:

- Chunk 1: Surya `49.880837014` seconds; Chandra `535.800193873` seconds;
  searchable-PDF assembly `5.696970019` seconds.
- Chunk 2 (source pages 11–20): input SHA-256
  `c7859d320bfcd9bf50a2927306e5759327279686822c4154d5c95276add3f66e`;
  Surya `45.244601126` seconds; Chandra `588.082708783` seconds;
  searchable-PDF assembly `5.218301469` seconds. The historical
  `Output PDF page 3 failed exact searchable text retention` error did not recur.
- Chunk 3 (source page 21): Surya `13.447052580` seconds; Chandra
  `58.989545369` seconds; searchable-PDF assembly `0.966234060` seconds. Its
  blank/textless warning was expected.

The cold final PDF was `14,386,082` bytes and `21` pages, with SHA-256
`1fb841d00428dbcd33ecff3158f768e49ebdf68a094751d47d58ae6d32da57f1`.
End-to-end elapsed time was approximately `1,881` seconds; recorded engine and
assembly stages sum to `1,303.326444293` seconds. This is a successful
full-context regression and timing observation, not proof of the original root
cause: the failed candidate was not preserved.

### Full-context chunk-cache hit

A repeat with the same durable work/cache identity reused all three chunks. Its
manifest reported `done`, output page count `21`, and `0` partial-page failures.
End-to-end wall time was `328.924` seconds, about `5.72x` faster than the cold
run, but still shows approximately five and a half minutes of non-engine work.

The cache-hit PDF was `14,386,082` bytes, with SHA-256
`1e6d7d860e9f40e15f9cc7dddab4f3b42501220c63814e26c0132efc11cafefd`.
Extracted text on every page, all 21 page renders at 72 DPI, and PDF metadata
were exactly equal to the cold result; serialized PDF bytes differed. This
measures durable chunk reuse, not in-process engine warm latency.

## Case 2 — punctuation-only Chandra geometry page

- Logical source: `punctuation-hybrid-97d0-chunk-01`.
- Source SHA-256: `78e2cc769b8184e71609b23c84dc87de6218332a3bd318bb7072c458e273b51a`.
- Source bytes/pages: `8,993,123` / `10`.
- Selected source page: `3`.
- Chandra: `ok`, `112.495581785` seconds, `1,701` report-level text
  characters, `1,852.336` MB memory delta, one attempt.
- Surya: `ok`, `18.123892879` seconds, `2,144` reported text characters,
  memory delta unavailable, one attempt.
- Searchable-PDF assembly: `ok`, `2.231652421` seconds.
- Strict page reconciliation: `ok`.
- Result bytes/pages: `13,014,129` / `10`.
- Result SHA-256: `02109585c63a207e78fa144e6421ee3bc4b803048f3b6a7ae3fe3538539b63d2`.

This case confirms that the fixed punctuation-only page completes through strict
reconciliation on a real engine run. Visual/table fidelity still requires human
review or approved Ground Truth.

### Process-warm-cache repeat

The same source page was repeated in a new container with the same pre-existing
model caches and identical runtime identity:

- Chandra: `ok`, `118.282162525` seconds, `1,701` report-level text
  characters, `1,854.547` MB memory delta, one attempt.
- Surya: `ok`, `17.920884581` seconds, `2,144` reported text characters,
  memory delta unavailable, one attempt.
- Searchable-PDF assembly: `ok`, `2.585374948` seconds.
- Strict page reconciliation: `ok`.
- Result bytes/pages: `13,014,129` / `10`.
- Result SHA-256: `02109585c63a207e78fa144e6421ee3bc4b803048f3b6a7ae3fe3538539b63d2`.

The result is byte-for-byte identical to the first run. Stage timing differences
were Chandra `+5.1%`, Surya `-1.1%`, and assembly `+15.9%`; one repeat is not
enough for a latency distribution. Because each run started fresh processes,
this measures process-cold execution with warm model caches, not an in-process
warm path.

## Metrics not measured

- CER and WER: no reviewed Ground Truth.
- Peak VRAM: not captured; GPU identity and successful CUDA allocation only.
- Peak RAM: not captured; Chandra reports process memory delta and Surya reports
  no memory value.
- In-process warm-run median/tail latency: not measured; the repeat used a new
  container and new engine processes.
- General chunk-reuse latency distribution and invalidation reasons: only one
  three-chunk cache-hit repeat was measured.
- Exact-retention root cause: the full 21-page run passed and the historical
  failure did not recur; its original failed candidate is unavailable.

No accuracy, model, preprocessing, prompt, retry, or performance change may use
this checkpoint as a CER/WER improvement claim. It is suitable for detecting
stage failures, large timing regressions, result-hash drift under an identical
runtime, and recurrence of the punctuation rejection.

## Docker deployment observation

A `--no-cache` build of the same source was intentionally stopped at the user's
request after it began downloading duplicate multi-gigabyte CUDA wheels. The
build had completed base system packages and was installing the first engine
venv; it did not fail. Existing image dependencies and local model caches were
reused for the accepted runs instead.

Docker image cleanup removed 15 obsolete unreferenced OCR tags plus two dangling
images in the first conservative pass. The second pass removed 11 independently
verified exited/status-0 bind-only containers and three obsolete image tags:
`surya-chandra-ocr:037c339efc50`, `surya-chandra-ocr:08d2cceb2daa`, and
`surya-chandra-ocr:f7a6c218f49a`. Reported image usage fell from `329.5 GB` to
`211.6 GB`, releasing about `117.9 GB`.

The final `docker system df` snapshot reported Images `25`, active `12`,
`211.6 GB`, reclaimable `129 GB`; Containers `22`, active `4`, `65.65 MB`;
Volumes `46`, active `7`, `44.57 GB`; and Build Cache `174`, `34.43 GB`,
reclaimable `20.69 GB`. The current image digest `sha256:f470cf1520e4...`,
immediate rollback tag `surya-chandra-ocr:00d0499`, all Exited137 incident
candidates, volumes, model caches, and build cache were preserved to avoid
future downloads. No source PDFs, OCR outputs, or user-generated artifacts were
deleted.
