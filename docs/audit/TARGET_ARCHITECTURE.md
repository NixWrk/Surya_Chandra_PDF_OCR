# Target Architecture

Status: incremental target; not a rewrite authorization.

Audit code baseline: `bbebe4bbb58c0e3a384558e24f22bc06663093c0`.

## Goals

1. Preserve the strict `chandra+surya` production contract.
2. Make result recovery and publication trustworthy.
3. Make accuracy changes benchmark-gated.
4. Reduce duplicate OCR through attributable cache/resume behavior.
5. Make Windows and Docker deployment reproducible on a new PC.
6. Give CLI, GUI, HTTP and any future MCP one application boundary.
7. Deliver changes as small reversible commits.

## Non-goals

- Replacing Chandra or Surya without benchmark evidence.
- Introducing a distributed workflow framework or external queue.
- Rewriting the HTTP service.
- Moving all large modules at once.
- Deleting retained run artifacts or user PDFs.
- Adding MCP before the HTTP boundary is stable.

## Logical architecture

```text
CLI / GUI / HTTP / optional thin MCP
                 |
                 v
        stable application façade
          build_searchable_pdf
                 |
       +---------+----------+
       |                    |
       v                    v
 run/chunk state      OCR stage orchestration
       |                    |
       v           +--------+-------+
 artifact store    | Chandra adapter |
 / manifests       | Surya adapter   |
                   +--------+-------+
                            |
                            v
                  page reconciliation policy
                            |
                            v
                 PDF assembly + validators
                            |
                            v
                   atomic publication
```

The logical boundaries should first be documented and tested inside existing
files. Physical extraction happens only when it enables a specific verified fix
or reduces measured duplicate work.

## Stable application façade

`build_searchable_pdf` at `src/uniscan/app/ocr_pipeline.py:5742` remains the
compatibility façade used by CLI, GUI and HTTP.

Existing parameters, progress behavior, strict hybrid mode and return summary
remain stable. New internal modules are not public APIs until explicitly accepted.

## Ownership boundaries

| Boundary | Current owner | Incremental target |
|---|---|---|
| Transport syntax | CLI, GUI, web | Adapters parse requests; application validates semantic invariants |
| Run configuration | Environment reads across modules | One immutable resolved configuration persisted before work |
| Engine execution | `ocr/benchmark.py` | Stable Chandra/Surya adapter contract shared by production and benchmark |
| Retry evidence | `benchmark.py`, `ocr_pipeline.py` | Adapter produces; reconciliation validates |
| Page decision policy | `ocr_pipeline.py` | One explicit exhaustive reconciliation policy |
| PDF assembly/validation | `artifact_searchable.py` | Retain owner; construction and validation stay separate phases |
| Chunk cache/state | `ocr_pipeline.py` | Explicit state machine plus environment-complete identity |
| Job persistence | `web/service.py` | Root-contained repository with sealed results |
| Job scheduling | `web/service.py` | Explicit transitions and attempt/lease ownership |
| HTTP transport | `web/service.py` | Request/response mapping; no direct OCR-state mutation |
| Observability | Reports/events/logs | Common run/job IDs and stage/runtime provenance |

## Conservative physical evolution

No file needs to move for the first integrity fixes. If characterization tests
later justify extraction, use this order:

1. Keep `app/ocr_pipeline.py` as the public façade.
2. Isolate immutable run configuration and run identity.
3. Isolate page reconciliation without schema/behavior changes.
4. Separate web persistence/state transitions from HTTP handler dispatch.
5. Separate production engine adapters from benchmark report orchestration while
   preserving wrapper functions.
6. Leave geometry/overlay implementation in `artifact_searchable.py` until a
   measured defect needs a narrower boundary.

Illustrative names, not mandatory scaffolding:

```text
src/uniscan/app/run_config.py
src/uniscan/app/run_identity.py
src/uniscan/ocr/reconciliation.py
src/uniscan/web/jobs.py
src/uniscan/ocr/adapters/
```

Do not create these modules merely to match the diagram.

## Immutable run contract

Each production run should resolve once to a record containing:

- source SHA-256, byte size and page count;
- selected pages;
- mode, strict/textless flags, DPI and chunk size;
- Chandra and Surya executable identity;
- exact package versions;
- model repository plus resolved revision/digest;
- PyTorch, CUDA, driver and GPU identity;
- pipeline/evidence/artifact/validator schema revisions;
- warm/cold cache state where relevant.

Environment variables remain deployment inputs; their resolved values are captured
before processing and do not silently change a running job.

## Engine adapter contract

An engine adapter accepts sealed ordered source rasters, immutable run
configuration and progress/cancellation hooks. It returns ordered page outcomes,
text/geometry, source identity, retry evidence, timings, resource observations and
explicit failure details.

Benchmarking consumes the same adapter contract. Existing functions in
`ocr/benchmark.py` may implement the contract before any code movement.

## Reconciliation contract

Reconciliation owns the decision whether evidence is publishable. It neither runs
models nor publishes files.

Every requested page ends in one explicit state:

```text
text
verified_blank
explicit_nontext
accepted_zero_output
unresolved
```

Only accepted states reach PDF assembly.

Punctuation-only geometry must be represented as a separate tested concern from
searchable semantic text. The choice to retain, filter or transform it requires a
minimal fixture plus searchable-text and visual-layout assertions.

## PDF assembly and publication

```text
validated page decisions
  -> same-filesystem temporary candidate
  -> page-count validation
  -> exact searchable-text retention
  -> required visual-retention validation
  -> output seal
  -> atomic publication
```

A failed candidate never replaces a prior successful output.

The output seal contains SHA-256, size, page count, complete run identity and
validator revision. Recovery validates this seal before restoring `done`.

## Chunk cache and resume

Target states:

```text
pending -> running -> validating -> done
                    \-> error
running -> interrupted
```

Reuse requires an identical complete run identity, valid schema, page count,
chunk-PDF hash, immutable evidence manifest and matching source-page identities.
Package/model/runtime digests must enter identity before reuse across upgrades is
trusted.

Automatic deletion remains out of scope until ownership/retention is explicit.

## Durable job architecture

Logical components may initially remain in `web/service.py`:

- Job repository: metadata, SQLite index, events and root-contained paths;
- Scheduler: priority and allowed transitions;
- Executor: invokes the stable application façade;
- HTTP handler: request/response translation.

### Recovery integrity

A recovered result becomes `done` only when:

- its resolved path remains inside the job directory;
- it is a regular non-link file;
- a persisted result seal exists;
- SHA-256 and size match;
- PDF parsing and expected page count succeed.

These requirements address recovery at `service.py:374-445`, path resolution at
`411-412` and serving at `2008-2019`.

### Attempt fencing

Each running attempt receives an immutable attempt/lease identifier. Before
publication at the current `service.py:1577-1611` boundary, the worker proves that
its job is still running and its attempt owns the lease. A worker reclaimed by the
watchdog (`service.py:215-258`) may finish computation but cannot publish or delete
another attempt's artifacts.

### Cancellation

Queued cancellation remains supported. Running cancellation is added only after
model/subprocess boundaries can stop cooperatively and cleanup is proven. Until
then the API continues returning a clear conflict.

## Artifact ownership rules

1. User-owned input is never removed by audit or cache cleanup.
2. Every generated artifact belongs to one run/job root.
3. Resolved paths remain within the owning root.
4. Symlink, junction and reparse escapes are rejected at trust boundaries.
5. Evidence used for publication is immutable.
6. Candidates use same-filesystem atomic replacement.
7. Successful outputs carry a verifiable seal.
8. Cleanup operates only on proven generated roots.
9. Incident artifacts remain preserved until minimal fixtures exist.

## Reproducible deployment

Target runtime profiles:

- common application dependencies;
- Surya runtime;
- Chandra runtime;
- developer/test runtime.

Requirements:

- lock exact transitive dependencies or record hash-verified constraints;
- pin accepted container bases by digest;
- record model revision/digest;
- replace tracked machine UUID with explicit local configuration/discovery;
- keep `.env.example` machine-neutral;
- support standalone Compose without a pre-created external network;
- place shared-network integration in an explicit override;
- align evaluation scripts with `.venv_surya`/`.venv_chandra`.

Container labels are metadata only. Source parity is established by content hashes;
the inspected container already matches all 23 tracked Python source files.

## Quality and performance gates

No OCR behavior change is accepted without:

- a written hypothesis;
- a reproducing fixture;
- Ground Truth or explicit structural/layout assertions;
- before/after accuracy and retention results;
- wall-time and RAM/VRAM results;
- environment/model provenance;
- one small attributable change.

The two incidents in `OBSERVED_OCR_FAILURES.md` remain separate fixtures.

## Interface compatibility

- CLI `searchable-pdf` remains strict hybrid.
- GUI stays a thin façade caller.
- HTTP routes remain stable.
- persisted schemas receive explicit versions/migrations.
- response changes are additive until consumers migrate.
- absolute filesystem paths should leave public responses or require a privileged
  diagnostic mode.

## Optional MCP

MCP is not required for OCR correctness. If agent workflows demonstrate value,
it should be a thin HTTP client exposing only submit, status/queue, result, queued
cancel and health. It must not import OCR internals, manipulate manifests, browse
work roots or control GPU processes directly.

Implementation waits for an authentication decision, sealed result recovery and
stable HTTP schemas.

## Incremental migration order

1. Add the three required durable-job reproducing tests.
2. Apply one minimal integrity fix per proven failure.
3. Derive privacy-safe OCR incident fixtures.
4. Establish the quality/performance baseline.
5. Complete run identity and result seals.
6. Remove machine-specific deployment assumptions.
7. Lock runtime profiles and verify new-PC setup.
8. Extract logical seams only where tests/measurements justify them.
9. Reassess MCP last.

## Assumptions and open decisions

- Strict hybrid remains the only production mode.
- Serialized single-GPU execution remains intentional.
- Existing CLI/HTTP contracts may have external consumers.
- Ignored run artifacts may contain private user data.
- Open: trusted-network-only service versus built-in authentication.
- Open: lock mechanism and model digest format.
- Open: punctuation-only geometry policy.
- Open: cooperative cancellation boundary.
- Open: whether MCP produces measurable operational value.
