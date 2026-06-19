# OCR Orchestrator And GPU Contract

Last updated: 2026-06-19

## Purpose

Surya Chandra PDF OCR is a universal OCR service. It should be callable by any
Elvis project, not only Zotero automation. Callers submit PDF OCR jobs through
the HTTP job API and track durable job metadata; they do not run Chandra or
Surya directly unless they are doing local manual/debug work.

## Canonical OCR Job Request

```http
POST http://localhost:8000/api/jobs?mode=chandra+surya&lang=rus+eng&strict=1&filename=input.pdf
Content-Type: application/pdf
X-Project-ID: zotero
X-Service-ID: zotero-worker
X-Task-ID: zotero:item:ABCD1234:ocr
X-Request-ID: <uuid-or-deterministic-id>
X-Priority: batch
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
3. expose queued/running/done/error/interrupted status;
4. keep completed results discoverable until retention cleanup;
5. report interrupted jobs clearly after restart.

Large projects may keep their own higher-level queues, but they should treat the
OCR service as the source of truth for an accepted OCR job. Retrying after
network timeouts must use a deterministic task id or caller idempotency key once
that endpoint is implemented.

## GPU Coordination With LLM Orchestrator

OCR and local LLM inference both use GPU memory. They must not behave as two
independent GPU-heavy systems that only discover conflicts through CUDA OOM.

Target integration:

1. OCR exposes current queued/running jobs, estimated page count, selected mode,
   and coarse GPU need.
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

## Proposed OCR Resource Metadata

Future `POST /api/jobs` requests should accept optional headers or a JSON sidecar
for orchestration metadata:

```json
{
  "schema_version": "2026-06-19",
  "project": "zotero",
  "service": "zotero-worker",
  "task": "ocr",
  "job_id": "zotero:item:ABCD1234:ocr",
  "idempotency_key": "zotero:item:ABCD1234:ocr:v1",
  "priority": "batch",
  "gpu": "auto",
  "estimated_vram_gb": 8,
  "estimated_pages": 42,
  "ttl_seconds": 86400
}
```

This metadata should be stored with `metadata.json` and surfaced through
`GET /api/jobs/<job_id>/metadata`.

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

1. Add request-id, project-id, service-id, task-id, and priority fields to OCR
   job metadata.
2. Add an idempotency key for safe retries.
3. Promote the current file-backed durable metadata into a SQLite-backed job
   store when concurrent scheduling needs stronger guarantees.
4. Expose queue summary with queued/running/done/error/interrupted counts.
5. Expose coarse GPU reservation metadata for queued and running jobs.
6. Add a long-document restart smoke test to prove status/result durability.
7. Document project-level priority rules for OCR vs LLM work.
