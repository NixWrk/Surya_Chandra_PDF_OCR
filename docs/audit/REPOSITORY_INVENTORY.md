# Repository Inventory

Audit code baseline: `bbebe4bbb58c0e3a384558e24f22bc06663093c0`.

The later `0cff047` commit adds only `docs/audit/OBSERVED_OCR_FAILURES.md`.
Production source remained unchanged. This is a read-only inventory of the code
baseline, updated with the verified test/runtime measurements in `BASELINE.md`.

## Evidence commands

```powershell
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git diff --name-status bbebe4b..0cff047
git ls-files
rg -n "^(class|def|async def) " src tests
rg -n "add_parser\(|do_GET|do_POST|status|chunk|manifest" src
Get-Content -LiteralPath <tracked-file> | Measure-Object -Line
python -m pytest -q
python -m pytest --cov=uniscan --cov-branch --cov-report=term
python -m ruff check .
python -m mypy src scripts/compare_ocr_results.py
```

## Complete tracked-file classification

At `bbebe4b`: 60 tracked files and 41,459 physical lines.

### Production source — 23 files, 20,708 lines

```text
src/uniscan/__init__.py
src/uniscan/__main__.py
src/uniscan/app/__init__.py
src/uniscan/app/ocr_pipeline.py
src/uniscan/app/page_spec.py
src/uniscan/cli.py
src/uniscan/core/__init__.py
src/uniscan/core/pipeline.py
src/uniscan/export/__init__.py
src/uniscan/export/exporters.py
src/uniscan/io/__init__.py
src/uniscan/io/loaders.py
src/uniscan/ocr/__init__.py
src/uniscan/ocr/artifact_searchable.py
src/uniscan/ocr/benchmark.py
src/uniscan/ocr/canonical.py
src/uniscan/ocr/engine.py
src/uniscan/ocr/pdf_utils.py
src/uniscan/ocr/preprocessing.py
src/uniscan/ui/__init__.py
src/uniscan/ui/basic_ocr_gui.py
src/uniscan/web/__init__.py
src/uniscan/web/service.py
```

Largest modules:

| File | Lines | Primary responsibility |
|---|---:|---|
| `src/uniscan/app/ocr_pipeline.py` | 6,489 | Production orchestration, reconciliation, chunks and publication |
| `src/uniscan/ocr/benchmark.py` | 5,455 | Engine execution, retry/evidence and benchmark reports |
| `src/uniscan/ocr/artifact_searchable.py` | 3,944 | Geometry/overlay construction and PDF validation |
| `src/uniscan/web/service.py` | 2,131 | HTTP, durable jobs, queue, worker and watchdog |
| `src/uniscan/ocr/engine.py` | 738 | Engine registry/detection and legacy adapters |
| `src/uniscan/cli.py` | 502 | Command contracts and dispatch |

### Tests — 12 files, 16,377 lines

```text
tests/conftest.py
tests/test_app_searchable_pdf.py
tests/test_gpu0_runtime_contract.py
tests/test_loaders.py
tests/test_ocr_artifact_searchable.py
tests/test_ocr_benchmark.py
tests/test_ocr_canonical.py
tests/test_ocr_engine.py
tests/test_ocr_page_reconciliation.py
tests/test_ocr_preprocessing.py
tests/test_ocr_zero_output_retry.py
tests/test_web_service.py
```

The verified full suite completed with 617 passed, 7 skipped and 5 warnings.
Branch coverage is 74%. This is the full current suite, not a partial run.

Coverage areas:

- chunk manifests, cache reuse and merge validation;
- atomic output publication and source mutation detection;
- strict page reconciliation and retry provenance;
- Chandra/Surya zero-output behavior;
- geometry/overlay construction and text/visual retention;
- engine registry, cache/device checks and benchmark artifacts;
- durable job persistence, recovery, idempotency, queue and HTTP handlers;
- loaders, preprocessing and canonical packaging.

Known gaps:

- real Surya/Chandra GPU execution in a clean-clone acceptance test;
- GUI automation;
- Docker build/Compose startup acceptance;
- Ground Truth, CER/WER and layout metrics;
- corrupt/tampered result recovery;
- result symlink/reparse escape;
- stale-worker publication after watchdog reclamation;
- malicious/encrypted/oversized PDF matrix;
- operational-script smoke coverage.

### Operational files — 10 files, 2,374 lines

```text
run_basic_gui.cmd
setup_dual_venv.cmd
scripts/benchmark_ocr_matrix.ps1
scripts/bootstrap_new_pc.ps1
scripts/compare_chandra_geometry_variants.ps1
scripts/compare_ocr_results.py
scripts/docker-entrypoint.sh
scripts/full_hybrid_geometry_eval.ps1
scripts/gpu0_contract.ps1
scripts/run_hybrid_gpu_smoke.ps1
```

### Configuration, documentation and repository metadata — 15 files

```text
.dockerignore
.env.example
.gitattributes
.gitignore
Dockerfile
README.md
docker-compose.yml
docs/CLEANUP_REFACTOR_PLAN.md
docs/FULL_REPOSITORY_AUDIT_PLAN.md
docs/HTTP_JOB_DURABILITY_PLAN.md
docs/ORCHESTRATOR_GPU_CONTRACT.md
docs/REPO_INVENTORY_KEEP_REMOVE.md
docs/UNISCAN_JOB_PROTOCOL.md
pyproject.toml
pytest.ini
```

No tracked PDF fixture, Ground Truth, dependency lock, CI workflow, `AGENTS.md`,
repo-local skill or MCP implementation existed at the code baseline.

## Module and import boundaries

```text
python -m uniscan
  -> src/uniscan/__main__.py
  -> src/uniscan/cli.py
     -> uniscan.app -> app/ocr_pipeline.py
     -> uniscan.ocr -> benchmark/artifact/engine/canonical
     -> uniscan.web -> web/service.py

GUI  -> app.build_searchable_pdf
HTTP -> app.build_searchable_pdf
CLI  -> application, benchmark and artifact APIs
```

`app/ocr_pipeline.py` is the production façade. `app/__init__.py` and
`ocr/__init__.py` eagerly export broad APIs. Engine-specific heavyweight imports
are generally deferred inside execution functions.

`ocr/benchmark.py` currently owns both experimental benchmarking and production
Chandra/Surya execution. This is a confirmed mixed responsibility, not evidence
that an immediate rewrite is safe.

## CLI contract

Seven commands are registered at `src/uniscan/cli.py:40-339`:

- `benchmark-ocr`;
- `benchmark-ocr-canonical`;
- `build-searchable-from-artifacts`;
- `prepare-compare-txt`;
- `compare-chandra-geometry`;
- `searchable-pdf`;
- `serve-http`.

Production `searchable-pdf` accepts only `chandra+surya`
(`src/uniscan/cli.py:293-310`) and is always strict
(`src/uniscan/cli.py:333-336`). It accepts one PDF, optional page selection and
an optional work root; the documented default is input overwrite.

## GUI contract

The GUI exposes only `Chandra + Surya` (`src/uniscan/ui/basic_ocr_gui.py:20`).
It invokes `build_searchable_pdf` from a daemon thread
(`src/uniscan/ui/basic_ocr_gui.py:161-184`) with strict mode and original text
removal enabled. There is no running-job cancellation control.

## HTTP contract

Routes at `src/uniscan/web/service.py:2034-2077`:

| Method | Route | Purpose |
|---|---|---|
| GET | `/`, `/index.html` | Embedded UI |
| GET | `/health` | Service/store health |
| POST | `/searchable-pdf` | Synchronous OCR |
| POST | `/api/jobs` | Durable asynchronous submission |
| GET | `/api/jobs`, `/api/queue` | Queue summary |
| GET | `/api/jobs/{id}` | Job state |
| GET | `/api/jobs/{id}/metadata` | Durable metadata |
| GET | `/api/jobs/{id}/result` | Result PDF |
| POST | `/api/jobs/{id}/cancel` | Queued-job cancellation |

No built-in authentication, authorization or rate limiting was found. Serialized
jobs include absolute host paths (`src/uniscan/web/service.py:576-635`).

## Production OCR flow

`build_searchable_pdf` begins at `src/uniscan/app/ocr_pipeline.py:5742`:

```text
PDF path or bytes
  -> staged source
  -> optional textless raster PDF
  -> page rendering
  -> Chandra text + Surya geometry
  -> strict attempt/evidence validation
  -> page reconciliation
  -> compare-text artifacts
  -> searchable-PDF candidate
  -> text/page/visual validation
  -> atomic publication
```

Production supports strict `chandra+surya` only. Chandra supplies text and Surya
supplies geometry. Failed validation prevents final publication.

## Page reconciliation state

`_reconcile_mode_both_pages` starts at
`src/uniscan/app/ocr_pipeline.py:3074`. It requires exact requested-page evidence
from both engines and accepts explicit outcomes for text, verified blank,
non-text graphics and bounded zero-output retry cases. An unresolved page fails
the strict run.

The punctuation-only incident at `docs/audit/OBSERVED_OCR_FAILURES.md:42-53`
shows a real distinction between semantic searchable text and geometry containing
punctuation-only table cells. No policy change is implied without a reproducer.

## Chunk/resume state

```text
source identity -> run identity -> manifest
chunk pending -> running -> done | error
all done -> revalidate chunks/evidence/source -> merge candidate
candidate valid -> atomic final publication
```

Facts:

- default chunk size is 10 pages;
- schema is `uniscan.hybrid-chunks.v4`;
- pipeline revision is `chandra-surya-resumable-v10`;
- run locking and atomic manifest writes exist;
- done chunks are revalidated before reuse and merge;
- `_hybrid_run_identity` at `ocr_pipeline.py:3716-3744` omits package,
  executable, model and CUDA-runtime digests.

## Durable job state

Terminal states are declared at `src/uniscan/web/service.py:48`; active states at
`src/uniscan/web/service.py:125`.

```text
create -> queued
queued -> running | cancelled
running -> done | error | interrupted
```

Running cancellation is rejected at `src/uniscan/web/service.py:259-283`.

Restart recovery is implemented at `src/uniscan/web/service.py:374-445`:

- an existing local result promotes an active job to `done` (`409-420`);
- queued input may be requeued;
- active jobs without a result become `interrupted`;
- done jobs without a result become `interrupted`.

## Artifact ownership

| Artifact | Owner | Current lifecycle |
|---|---|---|
| User source PDF | User/caller | Preserved unless explicit atomic overwrite succeeds |
| Uploaded `input.pdf` | Job store | Stored under `jobs/<id>` |
| Rendered pages/textless source | OCR run | Derived evidence/intermediate |
| Engine attempts and sidecars | OCR run | Validated retry/source evidence |
| `page_reconciliation.json` | OCR run | Strict page-decision record |
| Compare TXT | OCR run | Assembly intermediate |
| Candidate PDF | OCR run | Must validate before publication |
| Chunk cache/manifest | Pipeline | Run-key-owned resumable artifacts |
| `metadata.json`, `events.jsonl`, SQLite | Job store | Durable state/index/event evidence |
| `result.pdf` | Job store | Served for `done`; recovery seal is currently insufficient |
| Incident artifacts | User/audit evidence | Ignored and preserved pending fixture derivation |

## Dependency and deployment boundaries

Dependency declarations are split among `pyproject.toml`, `Dockerfile`,
`setup_dual_venv.cmd` and `scripts/benchmark_ocr_matrix.ps1`; there is no lock.

A machine-specific GPU UUID is tracked in `.env.example`, Compose, Windows
launch/setup scripts, the GPU contract script and `ocr/benchmark.py:82`.

`full_hybrid_geometry_eval.ps1:9` and
`compare_chandra_geometry_variants.ps1:8` default to `.venv`, while
`setup_dual_venv.cmd:6-7` creates `.venv_surya` and `.venv_chandra`.

All 23 tracked Python source files in the inspected container match checkout
SHA-256 content. Only container labels are stale; source mismatch is not a finding.

## Documentation contradictions

- `docs/REPO_INVENTORY_KEEP_REMOVE.md:14,100` describes three GUI modes;
  current GUI/production exposes one.
- The same document references absent
  `scripts/install_local_ocrmypdf_plugins.ps1` at line 90.

## Assumptions

- User PDFs, ignored job/run artifacts and model caches are out of cleanup scope.
- Strict hybrid remains the supported production mode.
- The single-GPU serialized execution policy remains intentional.
- Security/race impacts described in the audit report require reproducing tests
  before minimal fixes.
