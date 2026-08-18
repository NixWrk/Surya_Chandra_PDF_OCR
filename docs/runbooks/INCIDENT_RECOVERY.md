# OCR incident and recovery runbook

Use this workflow when a job fails, is interrupted, stalls, or produces a result
that cannot be downloaded or validated.

## Preserve first

Do not rerun setup, delete caches, edit metadata, remove chunks, or overwrite the
source/result before evidence capture. Record:

- local and UTC timestamps;
- Git commit, dirty state, image ID, and deployment mode;
- job ID, sanitized filename, input SHA-256, page count, and selected pages;
- exact status, stage, chunk/page, error text, and progress events;
- GPU/driver/CUDA, engine/model versions, and cache state;
- relevant file paths, sizes, hashes, and modification times.

Private PDFs and engine artifacts stay ignored. Add only a minimized,
privacy-safe reproducer to Git.

## Query the stable API

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/api/jobs/<job_id>
curl.exe http://127.0.0.1:8000/api/jobs/<job_id>/metadata
curl.exe http://127.0.0.1:8000/api/jobs
```

Download `/result` only when status is `done`. A missing or rejected result is
evidence; do not manually copy an internal candidate into place.

## Inspect durable evidence

Within the configured job root, inspect without modifying:

- `metadata.json` and `events.jsonl`;
- the SQLite job index through read-only queries where possible;
- chunk manifest and chunk output/evidence hashes;
- Chandra/Surya reports and page reconciliation records;
- candidate/merge validation errors;
- result file type, containment, size, hash, and PDF page count.

Distinguish the observed failing boundary from a hypothesized root cause. For
example, an exact-retention error proves merge validation rejected a page; it
does not prove why the hidden text differed if the failed candidate is absent.

## Recovery decisions

- `queued`: leave it for the worker or cancel through the API.
- `running`: do not force-delete artifacts; current cancellation may return 409.
- `interrupted` or `error`: resubmit from the caller's durable source with a new
  request ID; preserve the same idempotency key only for the exact same bytes and
  parameters.
- `done`: trust only the service result endpoint after its containment and
  validation checks.

Restart recovery may requeue valid queued jobs and marks abandoned running jobs
interrupted. A stale worker is not permitted to publish after reclamation.

## Reproduce before repair

Minimize at the narrowest responsible boundary: evidence parsing,
reconciliation, chunk validation, merge validation, recovery, or HTTP result
serving. Add a deterministic test that fails against the parent revision. Keep
separate incidents separate. After a minimal fix, run the targeted test
repeatedly where concurrency is involved, then the full suite and static checks.

Record confirmed incidents in `docs/audit/OBSERVED_OCR_FAILURES.md` with facts,
inferences, commands, hashes, and the regression-test commit.
