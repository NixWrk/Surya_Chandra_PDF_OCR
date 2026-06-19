# UniScan OCR Job Protocol

Version: `uniscan-ocr-job.v1`

Last updated: 2026-06-19

## Purpose

This is the shared task protocol for sending OCR work to the Surya Chandra PDF
OCR service from other Elvis projects, services, scripts, and containers.

Callers submit raw PDF bytes to the HTTP job API. The OCR service owns accepted
OCR job state, durable metadata, progress, and result files. Callers may keep
their own higher-level queues, but after an OCR job is accepted they should poll
the OCR API instead of running Chandra or Surya directly.

## Stable Boundary

Use the async API for cross-project integration:

```http
POST /api/jobs
GET /api/jobs
GET /api/jobs/<job_id>
GET /api/jobs/<job_id>/metadata
GET /api/jobs/<job_id>/result
POST /api/jobs/<job_id>/cancel
GET /health
```

The sync API, `POST /searchable-pdf`, is useful for manual checks and short
scripts, but it is not the preferred boundary for containers or queues.

## Queue Model

The service accepts OCR jobs from multiple independent callers, or "employers",
such as different projects, workers, containers, and manual tools. Accepted jobs
share one OCR-owned priority queue.

The OCR worker concurrency is fixed at `1`: no more than one document is
processed at a time. This is intentional because Chandra, Surya, and local LLM
work may contend for the same GPU memory. HTTP requests may arrive concurrently,
but actual OCR execution is serialized by the service.

`GET /api/jobs` exposes `worker_concurrency: 1` along with queue counts and
active jobs. Waiting jobs are ordered by `priority` and creation time:
`interactive`, then `normal`, then `batch`, then `low`. The service does not
preempt a running OCR document.

## Create Job

```http
POST http://surya-chandra-ocr-api:8000/api/jobs?mode=chandra+surya&lang=rus+eng&strict=1&filename=input.pdf
Content-Type: application/pdf
X-UniScan-Protocol: uniscan-ocr-job.v1
X-Project-ID: zotero
X-Service-ID: zotero-worker
X-Task-ID: zotero:item:ABCD1234:ocr
X-Request-ID: 8cb04ca2-34a6-45a7-bf3d-fc4c83e8db3c
X-Idempotency-Key: zotero:item:ABCD1234:ocr:v1
X-Priority: batch
X-GPU-Policy: auto
X-Estimated-VRAM-GB: 8
X-Estimated-Pages: 42
X-TTL-Seconds: 86400
```

The request body is the raw PDF. The service persists `<job>/input.pdf`,
`metadata.json`, `events.jsonl`, and a SQLite job index before returning
`202 Accepted`.

On the host machine, use `http://127.0.0.1:8000`. From a Docker container on the
shared Compose network, use `http://surya-chandra-ocr-api:8000`.

## OCR Parameters

Query parameters:

| Name | Default | Meaning |
| --- | --- | --- |
| `mode` | `chandra+surya` | OCR mode: `chandra+surya`, `chandra`, or `surya`. |
| `lang` | service default, usually `rus+eng` | Language string passed to OCR engines. |
| `pages` | all pages | Optional page selection such as `1,3,5-8`. |
| `strict` | `1` | Fail the job if required OCR artifacts are missing. |
| `filename` | `document.pdf` | Original filename used for metadata and result download name. |
| `delete_text_layer` | `1` | Remove the existing text layer before building the output. |
| `delete_original_text_layer` | same as `delete_text_layer` | Legacy alias accepted by the service. |

## Protocol Metadata

Headers are canonical for container-to-container calls. The same values can also
be passed as query parameters for simple scripts.

| Header | Query fallback | Required for orchestrators | Meaning |
| --- | --- | --- | --- |
| `X-UniScan-Protocol` | `protocol_version`, `schema_version` | Recommended | Protocol version. Current value: `uniscan-ocr-job.v1`. |
| `X-Project-ID` | `project_id`, `project` | Yes | Product or project owner, for example `zotero`. |
| `X-Service-ID` | `service_id`, `service` | Yes | Calling service/container, for example `zotero-worker`. |
| `X-Task-ID` | `task_id`, `task`, `caller_job_id` | Yes | Stable logical task id in the caller's domain. |
| `X-Request-ID` | `request_id` | Yes | One id per HTTP submission attempt. Auto-generated if omitted. |
| `X-Idempotency-Key` | `idempotency_key` | Yes | Deterministic retry key for the exact OCR request. |
| `X-Priority` | `priority` | Yes | `interactive`, `normal`, `batch`, or `low`. |
| `X-GPU-Policy` | `gpu_policy`, `gpu` | Recommended | `auto`, `cuda`, `cpu`, or `none`. |
| `X-Estimated-VRAM-GB` | `estimated_vram_gb`, `vram_gb` | Recommended | Coarse VRAM estimate for external schedulers. |
| `X-Estimated-Pages` | `estimated_pages` | Recommended | Coarse page count estimate. |
| `X-TTL-Seconds` | `ttl_seconds` | Optional | Caller-requested metadata/result retention hint. |

`task_id`, `request_id`, and `idempotency_key` are intentionally different:

1. `task_id` identifies the caller's logical work item.
2. `request_id` identifies one HTTP attempt.
3. `idempotency_key` identifies one exact OCR request and should include a
   version suffix when OCR parameters may change.

Example idempotency key:

```text
zotero:item:ABCD1234:ocr:v1
```

## Idempotency

If `X-Idempotency-Key` is present:

1. First accepted request returns `202 Accepted` with a generated OCR `job_id`.
2. A repeat request with the same key, same PDF bytes, and same OCR parameters
   returns `200 OK`, the existing job, and `idempotent_replay: true`.
3. A repeat request with the same key but different PDF bytes or OCR parameters
   returns `409 Conflict`.

Callers should treat `job_id` as the OCR service id. They should store it after
the first accepted response and use it for polling. Callers must not assume that
their `task_id` is the same as the OCR `job_id`.

## Job Response

New job response:

```json
{
  "job_id": "4c60b5e7f1ad",
  "status": "queued",
  "progress": 0,
  "message": "Queued",
  "mode": "chandra+surya",
  "pages": "",
  "lang": "rus+eng",
  "strict": true,
  "delete_original_text_layer": true,
  "filename": "input.pdf",
  "created_at": "2026-06-19T12:00:00+00:00",
  "updated_at": "2026-06-19T12:00:00+00:00",
  "input_bytes": 123456,
  "input_path": "/data/work/jobs/4c60b5e7f1ad/input.pdf",
  "result_bytes": 0,
  "protocol_version": "uniscan-ocr-job.v1",
  "project_id": "zotero",
  "service_id": "zotero-worker",
  "task_id": "zotero:item:ABCD1234:ocr",
  "request_id": "8cb04ca2-34a6-45a7-bf3d-fc4c83e8db3c",
  "idempotency_key": "zotero:item:ABCD1234:ocr:v1",
  "priority": "batch",
  "gpu_policy": "auto",
  "estimated_vram_gb": 8,
  "estimated_pages": 42,
  "ttl_seconds": 86400,
  "input_sha256": "<sha256>",
  "request_fingerprint": "<sha256>"
}
```

Completed jobs include:

```json
{
  "status": "done",
  "progress": 100,
  "result_url": "/api/jobs/4c60b5e7f1ad/result",
  "metadata_url": "/api/jobs/4c60b5e7f1ad/metadata",
  "result_bytes": 654321,
  "completed_at": "2026-06-19T12:08:00+00:00",
  "finished_at": "2026-06-19T12:08:00+00:00"
}
```

## Status Semantics

| Status | Meaning | Caller action |
| --- | --- | --- |
| `queued` | Accepted and waiting to run. | Poll. |
| `running` | OCR is processing. | Poll; inspect `heartbeat_at` if needed. |
| `done` | Result PDF is ready. | Download `result_url`. |
| `error` | OCR failed. | Decide whether to retry, downgrade mode, or mark manual review. |
| `interrupted` | Service restarted before completion. | Resubmit from the caller's durable source queue. |
| `cancelled` | Queued job was cancelled before processing started. | Treat as terminal. |

## Cancellation

Queued jobs can be cancelled before OCR starts:

```http
POST /api/jobs/<job_id>/cancel
```

If the job is still `queued`, the response is `200 OK` and the job becomes
`cancelled`. Running OCR jobs are not forcibly stopped; cancelling a `running`
job returns `409 Conflict` because the current OCR pipeline is not
cooperative-cancellable.

## Queue Summary

`GET /api/jobs` and `GET /api/queue` return counts, active jobs, and recent jobs:

```json
{
  "ok": true,
  "protocol_version": "uniscan-ocr-job.v1",
  "jobs": 3,
  "worker_concurrency": 1,
  "counts": {
    "running": 1,
    "done": 2
  },
  "active_jobs": [],
  "recent_jobs": []
}
```

External schedulers can use this with the persisted `priority`, `gpu_policy`,
`estimated_vram_gb`, and `estimated_pages` fields. The current implementation
uses priority for waiting jobs and keeps execution serialized.

## Durability And Retention

Accepted jobs are stored under `UNISCAN_WORK_ROOT/jobs`:

```text
jobs/
  jobs.sqlite3
  <job_id>/
    input.pdf
    metadata.json
    events.jsonl
    result.pdf
```

On restart:

1. `done` jobs with `result.pdf` stay discoverable.
2. `queued` jobs with `input.pdf` are requeued automatically.
3. `running` jobs are marked `interrupted`.
4. `queued` jobs without `input.pdf` are marked `interrupted`.

Retention cleanup is controlled by:

```env
UNISCAN_JOB_CLEANUP_ON_START=1
UNISCAN_JOB_RETENTION_DAYS=30
UNISCAN_FAILED_JOB_RETENTION_DAYS=90
```

Per-job `ttl_seconds` overrides the default retention window for terminal jobs.
Cleanup never removes `queued` or `running` jobs.

## GPU Resource Metadata

OCR does not reserve, lease, or arbitrate GPU resources with other systems. It
only stores and exposes resource hints supplied by the caller:
`gpu_policy`, `estimated_vram_gb`, and `estimated_pages`.

An external orchestrator may use those fields before submitting OCR work. Once a
job is accepted, the OCR service processes it according to its own single-worker
queue and does not call back to an LLM/GPU orchestrator.

## Ownership Rules

1. The OCR service owns accepted OCR job metadata, job status, progress events,
   and result files under `UNISCAN_WORK_ROOT/jobs`.
2. Calling projects own source documents, upstream queues, and decisions about
   retrying `error` or `interrupted` jobs.
3. Other services should call the HTTP job API and should not import or run
   Chandra/Surya internals directly.
4. External orchestrators own coordination between OCR and local LLM inference.
   OCR only reports queue/resource metadata and processes accepted jobs.
5. Interactive tasks should use `priority=interactive`; background ingestion
   should use `priority=batch` or `priority=low`.

## Curl Example

```bash
curl -X POST "http://127.0.0.1:8000/api/jobs?mode=chandra+surya&lang=rus+eng&strict=1&filename=input.pdf" \
  -H "Content-Type: application/pdf" \
  -H "X-UniScan-Protocol: uniscan-ocr-job.v1" \
  -H "X-Project-ID: zotero" \
  -H "X-Service-ID: zotero-worker" \
  -H "X-Task-ID: zotero:item:ABCD1234:ocr" \
  -H "X-Request-ID: $(uuidgen)" \
  -H "X-Idempotency-Key: zotero:item:ABCD1234:ocr:v1" \
  -H "X-Priority: batch" \
  -H "X-GPU-Policy: auto" \
  -H "X-Estimated-VRAM-GB: 8" \
  -H "X-Estimated-Pages: 42" \
  --data-binary "@input.pdf"
```

Then poll:

```bash
curl "http://127.0.0.1:8000/api/jobs/<job_id>"
curl "http://127.0.0.1:8000/api/jobs"
curl -L "http://127.0.0.1:8000/api/jobs/<job_id>/result" -o output.searchable.pdf
```
