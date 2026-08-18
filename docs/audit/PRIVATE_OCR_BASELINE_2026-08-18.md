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
This does not close the incident: its failure occurred while assembling the full
11–20 chunk, so that context remains required for the exact regression run.

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
- Chunk reuse and rerun counts: selected-page direct runs do not exercise the
  durable ten-page chunk cache.
- Exact-retention full-context result: pages 11–20 not rerun in this checkpoint.

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
images. Reported image usage fell from `329.5 GB` to `244.5 GB` (about `85 GB`).
Current, immediate rollback, and every container-referenced image were preserved.
Volumes and model caches were not removed. Build cache was retained to avoid
future dependency downloads.
