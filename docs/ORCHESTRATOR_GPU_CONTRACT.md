# External Orchestrator Responsibility

Last updated: 2026-06-19

## Purpose

Surya Chandra PDF OCR is a universal OCR service. Its responsibility is narrow:
accept OCR jobs, persist them durably, process PDFs, and expose status/results.

The OCR service does not coordinate GPU ownership with local LLM services. It
does not reserve GPU slots, preempt LLM work, inspect LLM backends, or decide
which project is globally more important.

The canonical OCR task protocol is documented in
[UniScan OCR Job Protocol](UNISCAN_JOB_PROTOCOL.md).

## OCR Responsibility

OCR owns accepted OCR job state:

1. accept or reject incoming OCR jobs;
2. persist `input.pdf`, metadata, event log, and job index before returning
   `202 Accepted`;
3. accept jobs from multiple projects, workers, containers, and manual tools;
4. process no more than one OCR document at a time;
5. expose `queued`, `running`, `done`, `error`, `interrupted`, and `cancelled`
   status;
6. requeue durable `queued` jobs after restart;
7. mark interrupted `running` jobs clearly after restart;
8. keep completed results discoverable until OCR-owned retention cleanup removes
   them.

## External Orchestrator Responsibility

Any coordination between OCR, LLM inference, translation, ingestion, and other
GPU-heavy work belongs outside this repository.

External orchestrators should:

1. decide when to submit OCR jobs;
2. decide whether GPU/CPU capacity is available before submission;
3. choose caller metadata, priority, `gpu_policy`, `estimated_vram_gb`, and
   `estimated_pages`;
4. call `POST /api/jobs` only when OCR should accept and process the task;
5. poll OCR status/result endpoints after acceptance;
6. retry from their own durable source queue when OCR reports `error` or
   `interrupted`.

Once OCR returns `202 Accepted`, OCR will process the job according to its own
single-worker queue. OCR will not call back to an external scheduler for
permission.

## Stable OCR Boundary

```http
POST /api/jobs
GET /api/jobs
GET /api/jobs/<job_id>
GET /api/jobs/<job_id>/metadata
GET /api/jobs/<job_id>/result
POST /api/jobs/<job_id>/cancel
GET /health
```

## Resource Metadata

OCR persists and exposes resource hints for external schedulers:

```json
{
  "priority": "batch",
  "gpu_policy": "auto",
  "estimated_vram_gb": 8,
  "estimated_pages": 42,
  "worker_concurrency": 1
}
```

These fields are informational inside OCR except that `priority` orders waiting
OCR jobs. GPU arbitration remains an external responsibility.

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

## Remaining External Work

1. Define the real LLM/GPU orchestrator scheduling policy outside this repo.
2. Add a live cross-container smoke test from that orchestrator to OCR.
3. Document project-level priority rules for conflicts outside OCR, such as
   local LLM interactive sessions versus batch translation.
