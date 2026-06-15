# HTTP Job Durability Plan

Last updated: 2026-06-15

## Context

The HTTP service already exposes an asynchronous OCR API:

1. `POST /api/jobs`
2. `GET /api/jobs/<job_id>`
3. `GET /api/jobs/<job_id>/result`

This is now the preferred integration path for external orchestrators such as
`zotero-worker`, because long OCR runs can be observed through polling instead
of one silent synchronous response.

Current status: the HTTP service now writes durable per-job metadata under
`UNISCAN_WORK_ROOT/jobs/<job_id>/metadata.json`, appends progress events to
`events.jsonl`, and keeps completed `result.pdf` files discoverable after
restart. Active in-memory work cannot be resumed yet; `queued` or `running`
jobs are restored as `interrupted`, which lets external orchestrators fail fast
and retry from their own durable queues.

## Goal

Make the async HTTP job API restart-tolerant enough for long OCR documents and
external queue orchestrators.

An OCR API restart should not erase knowledge of jobs that were created before
the restart. Completed results should remain discoverable, and interrupted
running jobs should be reported explicitly instead of looking like unknown job
IDs. This baseline is implemented; the remaining work is deeper metadata,
cleanup, and long-document restart verification.

## Persistent Job Store

Implemented baseline store under `UNISCAN_WORK_ROOT`, with this layout:

```text
<work_root>/jobs/<job_id>/result.pdf
<work_root>/jobs/<job_id>/events.jsonl
<work_root>/jobs/<job_id>/metadata.json
```

Future SQLite-backed scheduling can add:

1. `job_id`
2. `status`: `queued`, `running`, `done`, `error`, `interrupted`, `cancelled`
3. `progress`
4. `message`
5. `error`
6. `mode`
7. `pages`
8. `lang`
9. `strict`
10. `delete_original_text_layer`
11. `filename`
12. `input_path`
13. `result_path`
14. `run_dir`
15. `created_at`
16. `started_at`
17. `updated_at`
18. `finished_at`
19. `worker_pid` or process identity
20. `heartbeat_at`

Optional future work: store request input bytes as `input.pdf` before returning
`202 Accepted`, so a job can be resumed by the OCR container itself instead of
being retried by an external orchestrator.

## Result Metadata

Write a stable `metadata.json` next to each result with:

1. request parameters;
2. OCR engine versions where available;
3. output PDF path and byte size;
4. elapsed seconds;
5. final text character count if cheaply available;
6. run directory;
7. error details for failed runs;
8. service version or git commit if available.

`GET /api/jobs/<job_id>` should include enough metadata for a caller to decide
whether to retry, download, or mark a document for manual review.

## Restart Recovery

On service startup:

1. open or migrate the persistent store;
2. scan jobs that were `queued` or `running`;
3. if `result.pdf` exists, mark the job `done`;
4. if no result exists, mark previously `running` jobs as `interrupted`;
5. leave `queued` jobs queued only if a future worker loop will actually pick
   them up automatically;
6. keep completed result metadata available until retention cleanup removes it.

The first implementation may choose not to resume interrupted OCR automatically.
It is enough to report `interrupted` clearly so the external orchestrator can
retry the source document.

## API Changes

Keep the existing API stable:

1. `POST /api/jobs` still returns `202` and a serialized job.
2. `GET /api/jobs/<job_id>` still returns one serialized job.
3. `GET /api/jobs/<job_id>/result` still returns the PDF when ready.

Extend serialized jobs with:

1. `created_at`
2. `started_at`
3. `updated_at`
4. `finished_at`
5. `heartbeat_at`
6. `input_bytes`
7. `result_bytes`
8. `metadata_url` if a separate metadata endpoint is added.

Optional future endpoint:

```http
GET /api/jobs/<job_id>/metadata
```

## Retention And Cleanup

Add retention controls so the work root does not grow forever:

```env
UNISCAN_JOB_RETENTION_DAYS=30
UNISCAN_FAILED_JOB_RETENTION_DAYS=90
UNISCAN_JOB_CLEANUP_ON_START=1
```

Cleanup must never remove a currently `running` job. Completed result cleanup
should remove both database rows and job directory files in one transaction-like
operation where possible.

## Verification Plan

Manual long-document smoke test:

1. start `surya-chandra-ocr-api`;
2. submit a long PDF through `POST /api/jobs`;
3. verify `GET /api/jobs/<job_id>` shows progress and heartbeat updates;
4. restart the OCR container while the job is running;
5. verify `GET /api/jobs/<job_id>` returns `interrupted`, not `404`;
6. resubmit the same input and let it finish;
7. restart the container after completion;
8. verify `GET /api/jobs/<job_id>` still returns `done`;
9. verify `GET /api/jobs/<job_id>/result` still downloads a valid PDF;
10. verify external callers such as `zotero-worker` can distinguish `done`,
    `error`, and `interrupted`.

Automated tests:

1. creating a job persists a row and input file before OCR starts;
2. progress updates survive a store reload;
3. completed result metadata survives a service restart;
4. running jobs are recovered as `interrupted`;
5. missing result files for `done` jobs are reported as a clear server-side
   consistency error;
6. retention cleanup preserves active jobs and removes expired jobs.

## Implementation Notes

Keep the initial implementation conservative:

1. introduce a small `JobStore` module instead of growing `web/service.py`;
2. use SQLite from the standard library;
3. use atomic writes for `metadata.json` and `events.jsonl` appends;
4. keep in-memory locks around job updates, but make SQLite the source of truth;
5. avoid automatic concurrent OCR after restart until scheduling semantics are
   explicit and tested.
