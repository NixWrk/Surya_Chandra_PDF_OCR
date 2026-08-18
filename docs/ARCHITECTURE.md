# UniScan architecture

This document describes the current production contract and the incremental
boundaries used for maintenance. It is not authorization for a rewrite.

## Runtime flow

```text
CLI / GUI / HTTP
       |
       v
build_searchable_pdf application facade
       |
       +--> normalize source and remove existing text layer
       +--> split into deterministic page chunks
       +--> Chandra text + Surya geometry engine evidence
       +--> reconcile every requested page
       +--> build same-filesystem candidate PDF
       +--> validate pages, text retention, and visual retention
       +--> publish atomically under the owning run/job
```

Production is strict hybrid OCR. Single-engine execution remains available only
for diagnostics and controlled benchmarks.

## Components and ownership

| Component | Current owner | Contract |
| --- | --- | --- |
| CLI | `src/uniscan/cli.py` | Parse commands and call application functions |
| GUI | `src/uniscan/ui/basic_ocr_gui.py` | Collect local input and call the facade |
| Application facade | `src/uniscan/app/ocr_pipeline.py` | Resolve a run, orchestrate chunks, reconcile, publish |
| Engine execution | `src/uniscan/ocr/benchmark.py` | Invoke Chandra/Surya and preserve evidence |
| PDF construction | `src/uniscan/ocr/artifact_searchable.py` | Build candidates and run strict validators |
| Durable jobs/API | `src/uniscan/web/service.py` | Persist jobs, serialize GPU work, expose HTTP |
| Deployment | `docker-compose.yml`, `Dockerfile`, Windows scripts | Reproduce isolated engine runtimes and GPU0 policy |

`build_searchable_pdf` remains the stable internal application boundary used by
CLI, GUI, and HTTP. New modules are private until a compatibility contract is
explicitly accepted.

## Page and chunk states

Every requested page must reconcile to one explicit accepted outcome:

```text
text | verified_blank | explicit_nontext | accepted_zero_output
```

`unresolved` is terminal for strict publication. Punctuation-only geometry may
be retained as layout evidence, but it cannot by itself prove semantic page
text.

Chunk lifecycle:

```text
pending -> running -> validating -> done
                    \-> error
running -> interrupted
```

Reuse requires compatible source/config identity plus valid output and evidence
hashes. Missing, changed, interrupted, or failed chunks run again.

## Durable job states

```text
queued -> running -> done
   |         |       \-> retained result
   |         +-> error / interrupted
   +-> cancelled
```

Execution is intentionally serialized (`worker_concurrency: 1`). Queued jobs
can be cancelled. Running cancellation remains unsupported until engine
processes have a proven cooperative stop boundary.

Publication is fenced by job ownership: a stale or reclaimed worker cannot copy
its result or mark a newer attempt done. Recovery and result serving accept only
root-contained regular non-link files. Result seals and complete runtime
identity remain incremental hardening work where noted in the audit backlog.

## Validation boundaries

1. Inputs must be PDFs within configured resource policy.
2. Engine output must have attributable page evidence.
3. Reconciliation must account for every requested page.
4. Candidate page count must match the source/selection.
5. Extracted searchable text must exactly retain accepted page text.
6. Required visual-retention checks must pass.
7. Publication must occur atomically inside the owning root.
8. Durable result serving must recheck containment and file type.

Failures preserve evidence and never publish a partially validated candidate.

## Filesystem ownership

- The caller owns original source documents.
- A run owns only artifacts beneath its run root.
- An HTTP job owns only its service-created job directory and metadata.
- Shared chunk caches are application-owned but are not disposable until
  identity, retention, and recovery contracts prove that deletion is safe.
- Model caches are deployment assets and must not be removed by diagnostics.

## Deployment topology

Docker Compose is canonical. The base file creates a project-local network and
binds HTTP to localhost. The shared-network override attaches the same service
to an explicitly pre-created external network. Windows uses two venvs because
the accepted Surya and Chandra dependency sets differ.

Physical GPU0 is selected by a host-local `UNISCAN_GPU_DEVICE_ID`; the container
sees only CUDA device index 0. Tracked configuration contains no host UUID.

## Agent and integration boundary

Agents should prefer the HTTP job protocol for operational work and repository
commands for tests/diagnostics. A thin MCP is intentionally deferred: it is only
justified if a real consumer needs typed wrappers around health, submit, status,
result, and queued cancellation. It must not import OCR internals or expose
arbitrary filesystem access.

For open risks and sequencing, see `docs/audit/AUDIT_REPORT.md`,
`docs/audit/TARGET_ARCHITECTURE.md`, and
`docs/audit/IMPLEMENTATION_BACKLOG.md`.
