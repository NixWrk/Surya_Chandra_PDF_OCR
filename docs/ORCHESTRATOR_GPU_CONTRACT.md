# OCR Orchestrator And GPU Contract

Last updated: 2026-06-19

## Purpose

Surya Chandra PDF OCR is a universal OCR service. It should be callable by any
Elvis project, not only Zotero automation. Callers submit PDF OCR jobs through
the HTTP job API and track durable job metadata; they do not run Chandra or
Surya directly unless they are doing local manual/debug work.

The canonical cross-container task protocol is now documented in
[UniScan OCR Job Protocol](UNISCAN_JOB_PROTOCOL.md). This document keeps the
GPU and ownership rules that sit around that protocol.

## Canonical OCR Job Request

```http
POST http://localhost:8000/api/jobs?mode=chandra+surya&lang=rus+eng&strict=1&filename=input.pdf
Content-Type: application/pdf
X-UniScan-Protocol: uniscan-ocr-job.v1
X-Project-ID: zotero
X-Service-ID: zotero-worker
X-Task-ID: zotero:item:ABCD1234:ocr
X-Request-ID: <uuid-or-deterministic-id>
X-Idempotency-Key: zotero:item:ABCD1234:ocr:v1
X-Priority: batch
X-GPU-Policy: auto
X-Estimated-VRAM-GB: 8
X-Estimated-Pages: 42
```

The PDF bytes are sent as the request body. The response returns a job id and a
serialized job summary. Callers then poll:

```http
GET /api/jobs/<job_id>
GET /api/jobs/<job_id>/metadata
GET /api/jobs/<job_id>/result
```

## Queue Ownership Rule

The OCR container owns OCR job state:

1. accept or reject incoming OCR jobs;
2. persist job metadata before long processing starts;
3. accept jobs from multiple projects, workers, containers, and manual tools;
4. process no more than one OCR document at a time;
5. expose queued/running/done/error/interrupted status;
6. keep completed results discoverable until retention cleanup;
7. report interrupted jobs clearly after restart.

Large projects may keep their own higher-level queues, but they should treat the
OCR service as the source of truth for an accepted OCR job. Retrying after
network timeouts must use `X-Idempotency-Key`. Reusing the same idempotency key
with the same PDF bytes and OCR parameters returns the existing OCR job; reusing
it for different bytes or parameters returns `409 Conflict`.

## GPU Coordination With LLM Orchestrator

OCR and local LLM inference both use GPU memory. They must not behave as two
independent GPU-heavy systems that only discover conflicts through CUDA OOM.

Target integration:

1. OCR exposes current queued/running jobs, estimated page count, selected mode,
   coarse GPU need, and `worker_concurrency: 1`.
2. LLM orchestrator exposes GPU inventory, running LLM backends, active request
   counts, reserved VRAM, and draining state.
3. Before starting a large OCR job, a project-level orchestrator can check
   whether a GPU slot is available.
4. Before starting or scaling an LLM backend, the LLM lifecycle service should
   consider external OCR reservations when available.
5. Priority policy decides conflicts: interactive work may preempt batch work;
   long batch translation and long OCR should normally queue rather than fight
   for the same VRAM.

The OCR service keeps OCR-specific scheduling. The LLM orchestrator keeps
LLM-specific scheduling. Shared GPU visibility and reservation metadata are the
coordination layer between them.

## OCR Resource Metadata

`POST /api/jobs` accepts orchestration metadata through headers or query
fallbacks. The persisted job metadata includes the equivalent of:

```json
{
  "protocol_version": "uniscan-ocr-job.v1",
  "project_id": "zotero",
  "service_id": "zotero-worker",
  "task_id": "zotero:item:ABCD1234:ocr",
  "request_id": "<uuid-per-submission>",
  "idempotency_key": "zotero:item:ABCD1234:ocr:v1",
  "priority": "batch",
  "gpu_policy": "auto",
  "estimated_vram_gb": 8,
  "estimated_pages": 42,
  "ttl_seconds": 86400
}
```

This metadata is stored with `metadata.json` and surfaced through
`GET /api/jobs/<job_id>/metadata`.

`GET /api/jobs` and `GET /api/queue` expose queue counts, active jobs, recent
jobs, `worker_concurrency: 1`, and the same coarse GPU metadata for schedulers.

## Elvis Projects Layout

The expected local grouping is:

```text
D:/Elvis_projects/
  Zotero_Automation/
    Zotero_automatization/
    zotero-ingest-worker/
    zotero-html-translate-worker/
    zotero-file-relay/
  Surya_Chandra_PDF_OCR/
  LLM_Orchestrator/
```

`Surya_Chandra_PDF_OCR` and `LLM_Orchestrator` are universal infrastructure
folders. Product folders such as `Zotero_Automation` call them through HTTP APIs
and should not vendor their runtime logic.

## Implementation Tasks

1. Promote the current file-backed durable metadata into a SQLite-backed job
   store when concurrent scheduling needs stronger guarantees.
2. Add a long-document restart smoke test to prove status/result durability.
3. Document project-level priority rules for OCR vs LLM work.
4. Add actual GPU reservation handshakes once the LLM orchestrator exposes a
   compatible reservation endpoint.
