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

For the cross-project OCR request shape and GPU coordination rules with the LLM
orchestrator, see [OCR Orchestrator And GPU Contract](ORCHESTRATOR_GPU_CONTRACT.md).

Current status: the HTTP service now writes durable per-job inputs and metadata
under `UNISCAN_WORK_ROOT/jobs/<job_id>/`, appends progress events to
`events.jsonl`, keeps a SQLite job index, and keeps completed `result.pdf` files
discoverable after restart. Job metadata now includes the shared
`uniscan-ocr-job.v1` protocol fields described in
[UniScan OCR Job Protocol](UNISCAN_JOB_PROTOCOL.md), including caller identity,
idempotency, priority, and coarse GPU estimates. `GET /api/jobs` exposes queue
counts, active jobs, recent jobs, and `worker_concurrency: 1`. The service may
accept jobs from multiple callers, but only one OCR document is processed at a
time. `queued` jobs with durable `input.pdf` are requeued after restart. Active
in-memory `running` work cannot be resumed yet; running jobs are restored as
`interrupted`, which lets external orchestrators fail fast and retry from their
own durable queues.

## Goal

Make the async HTTP job API restart-tolerant enough for long OCR documents and
external queue orchestrators.

An OCR API restart should not erase knowledge of jobs that were created before
the restart. Completed results should remain discoverable, and interrupted
running jobs should be reported explicitly instead of looking like unknown job
IDs. This baseline is implemented.

## Persistent Job Store

Implemented baseline store under `UNISCAN_WORK_ROOT`, with this layout:

```text
<work_root>/jobs/jobs.sqlite3
<work_root>/jobs/<job_id>/input.pdf
<work_root>/jobs/<job_id>/result.pdf
<work_root>/jobs/<job_id>/events.jsonl
<work_root>/jobs/<job_id>/metadata.json
```

The SQLite job index stores the serialized job record plus indexed status,
idempotency key, priority, and timestamps. A future migration may make SQLite
the only source of truth, but the current implementation intentionally keeps
human-readable sidecars.

The stored metadata includes:

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
21. `protocol_version`
22. `project_id`
23. `service_id`
24. `task_id`
25. `request_id`
26. `idempotency_key`
27. `priority`
28. `gpu_policy`
29. `estimated_vram_gb`
30. `estimated_pages`
31. `ttl_seconds`
32. `input_sha256`
33. `request_fingerprint`

Inputs are stored as `input.pdf` before returning `202 Accepted`, so queued jobs
can be resumed by the OCR container itself.

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
4. requeue `queued` jobs when `input.pdf` exists;
5. mark `queued` jobs without `input.pdf` as `interrupted`;
6. mark previously `running` jobs as `interrupted`;
7. keep completed result metadata available until retention cleanup removes it.

The service does not resume a partially running OCR process. It reports
`interrupted` clearly so the external orchestrator can retry the source
document.

## API Changes

Keep the existing API stable:

1. `POST /api/jobs` still returns `202` and a serialized job.
2. `GET /api/jobs/<job_id>` still returns one serialized job.
3. `GET /api/jobs/<job_id>/result` still returns the PDF when ready.

Extend serialized jobs with:

Implemented baseline fields:

1. `created_at`
2. `started_at`
3. `updated_at`
4. `finished_at`
5. `heartbeat_at`
6. `input_bytes`
7. `result_bytes`
8. protocol metadata from `uniscan-ocr-job.v1`
9. `metadata_url` for completed jobs

Implemented endpoint:

```http
GET /api/jobs/<job_id>/metadata
```

Implemented queue summary endpoints:

```http
GET /api/jobs
GET /api/queue
```

Implemented cancellation endpoint:

```http
POST /api/jobs/<job_id>/cancel
```

## Retention And Cleanup

Retention controls keep the work root from growing forever:

```env
UNISCAN_JOB_RETENTION_DAYS=30
UNISCAN_FAILED_JOB_RETENTION_DAYS=90
UNISCAN_JOB_CLEANUP_ON_START=1
```

Cleanup never removes `queued` or `running` jobs. Terminal job cleanup removes
the job directory and the SQLite index row. Per-job `ttl_seconds` overrides the
default retention windows.

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

1. creating a job persists a SQLite row and input file before OCR starts;
2. queued jobs with input survive a store reload and are requeueable;
3. completed result metadata survives a service restart;
4. running jobs are recovered as `interrupted`;
5. waiting jobs are priority-ordered without preempting running work;
6. queued jobs can be cancelled before processing starts;
7. retention cleanup preserves active jobs and removes expired jobs;
8. GPU reserve/release hooks preserve reservation metadata.

## Implementation Notes

Implementation remains conservative:

1. move `JobStore` to its own module if `web/service.py` grows further;
2. use SQLite from the standard library;
3. use atomic writes for `metadata.json` and `events.jsonl` appends;
4. keep in-memory locks around job updates and mirror updates into SQLite;
5. keep OCR execution serialized unless a future scheduler proves explicit,
   tested multi-document GPU scheduling is safe.
