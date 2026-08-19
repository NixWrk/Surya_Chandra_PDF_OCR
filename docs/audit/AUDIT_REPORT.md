# Repository Audit Report

Audit code baseline: `bbebe4bbb58c0e3a384558e24f22bc06663093c0`.

Status: accepted complete for the current production OCR scope on 2026-08-19.
The audit, inventory, integrity repairs, reproducible baseline, measured
performance work, deployment documentation, production promotion, and storage
optimization are complete. No OCR task remains active. Evidence gaps that need
another PC, new reviewed Ground Truth, a new incident, or a new product
requirement are accepted as non-blocking limitations in
`OCR_ACCEPTANCE_CLOSURE_2026-08-19.md`.

## Current disposition

The three highest-risk durable-job boundaries found by the initial audit were
reproduced before repair and are now protected by tests:

1. recovered results require an integrity seal and valid PDF/page evidence;
2. symlink, reparse, hardlink, and external-path escapes are rejected;
3. a stale watchdog-reclaimed attempt cannot publish over a newer attempt.

Additional accepted fixes cover malformed/encrypted upload admission, bounded
HTTP runtime shutdown, punctuation-only Chandra geometry, immutable per-run
hybrid configuration, and sealed verified-blank artifact publication. The latest
complete local software suite through storage commit `5d7b7d2` reports:

```text
680 passed, 9 skipped, 5 warnings in 216.47s
```

Ruff, mypy, and `git diff --check` are clean for the accepted changes. Detailed
test-signal classification is in `TEST_SIGNAL_CLASSIFICATION.md`.

The repository now has a versioned synthetic corpus and evaluator. Two immutable
offline checkpoints exercise all nine fixtures at least once with the real
Surya/Chandra image. The added degraded, rotated, native-text, and 23-page cases
all passed CER/WER, exact-retention, and page-mapping checks. `mixed-layout`
remains the measured accuracy target at CER `0.287293` and WER `0.366667`; no OCR
policy or engine version change has been accepted.

The 23-page run completed correctly but exposed a structural duplicate: every
chunk recomputed the same full evidence object twice in the immediate premerge
loop. A test-first narrow fix retained all manifest, fingerprint, snapshot, and
runtime-drift fences. One comparable offline after-run fell from 922.991 to
463.709 seconds (-49.760%) with CER/WER zero, exact retention pass, page mapping
pass, and no partial failures.

A follow-up production-like series rebuilt and attested all 26 tracked source
inputs in image `sha256:72ad02bb45d...`, then ran three independent fresh-cache
checkpoints on the production Windows bind. All quality gates passed. Wall
median was 596.799 seconds, the observed maximum was 628.505 seconds, observed
container RAM peaked at 4.22 GB, and total GPU0 VRAM peaked 11,046 MiB above
background. The earlier anonymous-volume run remains valid for the narrow code
comparison but is not a production-storage baseline. Exact evidence is in
`PRODUCTION_PROMOTION_2026-08-19.md`.

The remaining 256.788-second Windows-bind residual was then attributed without
weakening validation. A fail-closed cache-hit A/B reduced validation/merge from
198.584 seconds on the bind to 31.031 seconds on a Docker-managed volume while
SHA-256 time stayed essentially unchanged. Commit `5d7b7d2` moves only the
same-deployment hybrid chunk cache to that volume; durable inputs, job metadata,
retained evidence, and results remain host files. Production HTTP job
`4eb0ea9d0611` completed the fresh 23-page fixture in 433.897 seconds with exact
retention and zero partial failures. This is one accepted after-run, not a p95
claim. Exact evidence is in `RESIDUAL_STORAGE_PROFILE_2026-08-19.md`.

Deployment is materially less machine-specific: tracked GPU UUIDs were removed,
the permitted GPU is configured locally, Compose has a standalone default,
preflight/smoke/runbooks exist, and an offline no-cache source-layer Docker build
passed without downloads. Per user decision, that build is accepted as partial
new-PC evidence. It does not prove clean dependency/model provisioning or the
currently incomplete Windows Surya venv. A metadata-corrected derivative of the
exact measured image is now the local production `latest` and immutable
`prod-771b5de` at `sha256:b774e4aa955df...`; the old image is preserved as
`rollback-f470cf-20260819`. The production async HTTP smoke completed and exact
retention passed.

CI, `AGENTS.md`, architecture/runbook documentation, and the repository-local
operator skill are present. A thin MCP was assessed and intentionally deferred:
the HTTP API plus existing operator surfaces are sufficient until a concrete
consumer requires MCP.

No global registry or permanent history of processed documents is part of the
design. Run identity, source/chunk hashes, and configuration exist only inside an
active/resumable or explicitly retained run. Successful HTTP working data is
removed by default. Those hashes prevent unsafe resume after interruption; they
are not a document catalog.

Local Docker cleanup reclaimed approximately 117.9 GB earlier in the audit.
Current/rollback/incident images, model caches, user PDFs, jobs, and retained
benchmark evidence were preserved. A final inspection found no additional large
object that was both clearly disposable and unrelated to historical stopped OCR
containers or rollback/audit evidence. Additional broad pruning is therefore not
part of closure.

## Accepted non-blocking future candidates

These are not active OCR tasks. Reopen them only for a new requirement, incident,
or benchmark decision:

1. Add reviewed raster-scan and dense mixed-layout Ground Truth and repeat those
   representative accuracy cases.
2. Run controlled `mixed-layout` experiments before accepting any OCR-policy
   change.
3. Replace the accepted partial clean-build evidence only when a genuinely clean
   machine or prepared cache is available.
4. Measure queue/page limits, design cooperative running cancellation, and add
   bounded quotas only for explicit persistent benchmark/resume caches.
5. Capture the rejected candidate if the historical private exact-retention
   incident recurs; its root cause cannot be reconstructed from the final PDF.

## Initial audit snapshot (historical)

The repository has unusually strong defensive coverage around OCR evidence,
retry lineage, chunk reuse and PDF validation. The complete suite is green:
617 passed, 7 skipped, 5 warnings; Ruff and mypy are clean; branch coverage is
74%. The inspected container has exact SHA-256 content parity for all 23 tracked
Python source files (`MismatchCount=0`).

The immediate risk is not a source/container mismatch or a generally broken test
suite. It is three untested durable-job integrity boundaries, followed by missing
reproducible OCR quality/performance and deployment baselines.

Required immediate order:

1. reproduce corrupt/tampered recovery-result behavior;
2. reproduce result symlink escape;
3. reproduce stale-worker publication after watchdog reclamation;
4. implement the smallest fixes proven by those tests.

No large refactor or engine/model upgrade is justified before these tests and an
accuracy/performance baseline exist.

## Evidence summary

| Evidence | Result |
|---|---|
| Tracked baseline | 60 files, 41,459 physical lines |
| Production source | 23 files, 20,708 lines |
| Tests | 12 files, 16,377 lines |
| Operations | 10 files, 2,374 lines |
| Full pytest | 617 passed, 7 skipped, 5 warnings, 248.78 s; 250.966 s wall |
| Coverage pytest | Same suite, 261.91 s, 74% total branch coverage |
| Ruff | Clean |
| mypy | Clean across 24 files |
| Container source parity | 23/23 SHA-256 matches; zero mismatches |

Evidence commands and limitations are recorded in `BASELINE.md`.

## Immediate findings

### P1 — Recovery trusts an existing result without an integrity seal

Status: confirmed code behavior; corrupt/tampered-file outcome needs a reproducer.

Evidence:

- recovery is implemented at `src/uniscan/web/service.py:374-445`;
- `result_candidate` is accepted based on `is_file()` and resolved at `409-412`;
- size is read and an active job is promoted to `done` at `413-420`;
- the result endpoint serves a `done` job when the path exists at `2008-2019`.

No restart-time PDF parse, stored SHA-256 comparison, expected page-count check or
publication-completion seal is required. The full suite does not currently prove
behavior for a corrupt, truncated or replaced `result.pdf`.

Required next action: add a focused recovery test with a tampered/corrupt result,
then make the smallest recovery-validation change needed for the test.

### P1 — Recovery follows a result symlink outside the job root

Status: confirmed path behavior; external-file serving must be demonstrated by a
safe temporary-directory test.

Evidence:

- `input_candidate.is_file()` and `result_candidate.is_file()` follow links;
- both are assigned using `.resolve()` at `src/uniscan/web/service.py:411-412`;
- no post-resolution containment or non-link check is present there;
- the resolved result is served at `src/uniscan/web/service.py:2008-2019`.

The existing persisted-external-path coverage is not equivalent to a local
`jobs/<id>/result.pdf` symlink or Windows reparse-point test.

Required next action: add a root-containment/symlink recovery test before changing
path handling.

### P1 — A watchdog-reclaimed worker can reach publication code

Status: confirmed control-flow possibility; interleaving and resulting state need
a deterministic concurrency test.

Evidence:

- stale running jobs are reclaimed at `src/uniscan/web/service.py:215-258`;
- the watchdog invokes reclamation at `1707-1720`;
- the worker's successful result copy/publication and done update occur at
  `1577-1611`;
- the old worker is not terminated when the state is changed to `interrupted`.

Terminal-transition protection can reject a late state update, but it does not by
itself prove that the stale worker cannot copy a result or clean artifacts.

Required next action: create a barrier-controlled test that reclaims a running
attempt, lets it resume and asserts that it cannot publish.

## Other confirmed findings

### P1 — Runtime reproducibility is split

`pyproject.toml`, `Dockerfile`, `setup_dual_venv.cmd` and
`scripts/benchmark_ocr_matrix.ps1` contain overlapping but different dependency
truth. There is no tracked lock/checksum set. Model revision/digest is not part of
the run identity.

### P1 — Deployment is machine-specific

One GPU UUID is tracked across `.env.example`, Compose, Windows setup/launcher,
the GPU contract script and `src/uniscan/ocr/benchmark.py:82`. Compose also assumes
a pre-created external network. Evaluation scripts default to `.venv` although
setup creates `.venv_surya` and `.venv_chandra`.

### P1/P2 — HTTP trust is external

The HTTP service has no built-in authentication/authorization/rate limiting and
the container binds to `0.0.0.0`. Absolute host paths are included in serialized
job data (`src/uniscan/web/service.py:576-635`). Severity depends on the intended
network trust boundary, which is not yet explicit.

### P2 — Chunk identity is incomplete across runtime upgrades

`_hybrid_run_identity` at `src/uniscan/app/ocr_pipeline.py:3716-3744` includes
source/configuration and a manual pipeline revision but omits package, executable,
model and CUDA-runtime digests. Reuse across an environment change is therefore
not fully attributable.

### P2 — Compare-text discovery may mix runs

When no summary is supplied, `build_compare_txt_from_benchmark` at
`src/uniscan/ocr/artifact_searchable.py:3812` discovers reports by modification
time independently per engine. A common run identity is not enforced.

### P2 — Running jobs are not cancellable

`src/uniscan/web/service.py:259-283` explicitly rejects running cancellation.
The protocol documents the limitation. The GUI likewise has no running stop.

### P2 — Accuracy/performance baseline is absent

There is no tracked representative corpus, Ground Truth, CER/WER or accepted
layout/runtime/VRAM baseline. The two production incidents are evidence, but
neither is a complete benchmark.

### P2 — Documentation and packaging drift

- `docs/REPO_INVENTORY_KEEP_REMOVE.md:14,100` claims three GUI modes; current
  production exposes only hybrid mode.
- It references absent `scripts/install_local_ocrmypdf_plugins.ps1` at line 90.
- No CI workflow, `AGENTS.md`, repo-local skill or MCP exists.

## OCR incident evidence

`docs/audit/OBSERVED_OCR_FAILURES.md` records two distinct failures:

1. punctuation-only Chandra geometry rejected during strict page reconciliation;
2. exact searchable-text retention failure during PDF validation.

They differ by source document, chunk, stage and signature. Their ignored local
artifacts must be preserved until privacy-safe minimal fixtures reproduce each
failure independently.

## Confirmed facts versus hypotheses

Confirmed:

- exact recovery/path/publication control flow cited above;
- complete green suite and static checks;
- container source parity;
- absence of tracked corpus/locks/CI/agent tooling;
- dependency and deployment configuration split;
- the two observed production failure signatures.

Hypotheses requiring reproducing tests:

- exact corrupt/tampered result accepted and served after restart;
- exact external target reachable through a result symlink/reparse point;
- exact stale-worker publication/cleanup interleaving;
- measurable stale-cache reuse after package/model changes;
- accuracy benefit of any punctuation-only policy change.

## Decision gates

- Do not fix the top three risks until a focused test fails for the expected reason.
- Do not update Surya/Chandra or tune OCR without Ground Truth and runtime metrics.
- Do not delete ignored artifacts until fixture derivation and ownership review.
- Do not split large modules unless a tested boundary enables a small reversible
  change or removes measured duplicate work.
- Do not add MCP until the HTTP contract, authentication decision and result
  integrity are stable.

## Recommended sequence

1. Add the three required reproducing tests in the stated order.
2. Apply one minimal job-integrity fix per proven failure.
3. Derive minimal fixtures for both OCR incidents.
4. Record representative OCR accuracy/performance baseline.
5. Complete dependency/model/runtime provenance and cache identity.
6. Remove machine-specific deployment assumptions and validate a new-PC runbook.
7. Add CI, `AGENTS.md`, runbooks and a repo-local audit/operations skill.
8. Reassess a thin HTTP-backed MCP only after the service boundary is stable.

## Assumptions

- Strict `chandra+surya` remains the only production OCR mode.
- Single-GPU serialized execution remains intended.
- HTTP may currently rely on a trusted network; this must be decided explicitly.
- User documents and ignored run artifacts are private data and must be preserved.
