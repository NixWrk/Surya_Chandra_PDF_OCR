# Implementation Backlog

This backlog is evidence-first. It does not authorize a large refactor, engine
upgrade or deletion of user/run data.

Priority order within the immediate integrity block is mandatory.

## Execution disposition — 2026-08-18

The original ranked matrix below remains the audit trail. Current disposition:

- Completed with tests/evidence: durable recovery and publication fencing
  (items 1–3 and 11), both historical incident regression paths (4–5), versioned
  synthetic corpus/metrics/first baseline (6–8), host-configured GPU identity
  (12), standalone Compose networking (15), trusted-network decision (16), CPU CI
  (24), `AGENTS.md`/runbooks/repo skill (25–27), MCP assessment (28), and test
  signal classification (30).
- Completed: immutable resolved environment snapshot and drift fencing for item
  9. Partially complete: richer runtime/cache provenance (10), dependency/runtime
  alignment (13–14), resource limits (18), and stale documentation (29).
- Deferred by evidence: thin MCP remains unnecessary without a consumer;
  running cancellation (19) still depends on cooperative engine termination.

The current open order, which supersedes the numerical order for future work, is:

1. Clean dependency-resolution evidence for Docker and Windows without changing
   OCR versions; the offline source-layer build is not a clean dependency build.
2. Decide the minimal executable/package/model/CUDA provenance needed for an
   explicitly retained resumable cache. Environment drift is already fenced;
   do not create a global document registry, permanent history, or full model-file
   hash inventory without evidence that it is needed.
3. Repeat the real-engine benchmark for median/tail latency and peak RAM/VRAM;
   extend fixtures for rotated, skewed, noisy, low-resolution, and existing-text
   inputs.
4. Run controlled accuracy experiments against `mixed-layout` (baseline CER
   `0.287293`, WER `0.366667`) before changing OCR policy.
5. Capture a preserved rejected candidate if the private exact-retention failure
   recurs; do not infer its historical root cause.
6. Measure and define queue/upload/page limits and cooperative cancellation.
7. Add age and size quotas for explicit persistent benchmark/resume caches while
   preserving active jobs, retained failures, model caches, and user sources.
8. Incrementally reduce module size and duplicate validation only after the
   above behavior is protected by benchmarks.

Storage constraint: successful HTTP run/chunk working data is removed by default
unless `UNISCAN_KEEP_JOB_RUNS=1`; terminal job retention defaults to 30 days for
success and 90 days for failure. Hashes and run manifests are local cache/result
integrity evidence, not a repository-wide document catalog.

## Ranked matrix

Scales: severity is `P1`, `P2`, or `P3`; effort is `S`, `M`, or `L`.
Risk describes implementation/regression risk, not the severity of leaving an item open.

| Rank | Item | Severity | Impact | Confidence | Effort | Risk | Dependencies | Verification method | Rollback strategy |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | Reproduce corrupt/tampered result recovery | P1 | High: result integrity | High | S | Low | None | Restart test rejects invalid result | Revert test-only commit if premise is invalid |
| 2 | Reproduce result symlink/reparse escape | P1 | High: containment/data exposure | High | S | Low | Link-capable temp directory | Recovery/GET test rejects external target | Revert test; remove only temp artifacts |
| 3 | Reproduce stale-worker publication after reclamation | P1 | High: state/publication integrity | Medium-high | M | Medium | Barrier/event harness | Repeated deterministic race test | Revert isolated test harness |
| 4 | Derive punctuation-only geometry fixture | P1 | High: observed OCR failure | High | M | Medium | Preserved incident evidence | Reproduces old failure plus text/layout assertions | Remove derived fixture only; retain evidence |
| 5 | Derive exact-retention fixture | P1 | High: PDF correctness | High | M | Medium | Preserved second incident | Reproduces exact-retention failure | Remove derived fixture only; retain evidence |
| 6 | Create versioned benchmark manifest | P1 | High: attributable quality work | High | M | Low | Items 4-5 | Schema/hash/Ground Truth validation | Revert manifest; preserve fixtures |
| 7 | Add benchmark metrics | P1 | High: accuracy/speed/rework | Medium-high | L | Medium | Item 6 | Golden output tests and recorded run | Revert metrics; retain raw output |
| 8 | Capture first accepted baseline | P1 | High: OCR change gate | Medium-high | L | Low | Items 6-7 | Re-run within documented tolerance | Supersede; never overwrite rejected baseline |
| 9 | Define immutable run configuration | P1 | High: reproducibility/cache | High | M | Medium | Behavior characterization | Resolution/serialization tests | Revert behind existing facade |
| 10 | Complete hybrid run identity | P1 | High: cache validity | High | M | Medium | Item 9; identity format | Each identity mutation causes cache miss | Revert/bump revision; do not delete caches |
| 11 | Seal successful job results | P1 | High: recovery integrity | High | M | Medium-high | Items 1-2 | Tamper/truncate/page-count/recovery tests | Revert reader/writer together |
| 12 | Remove tracked GPU UUID | P1 | High: new-PC deployment | High | M | Medium | Device policy | Current/new-PC setup and smoke | Restore local override, not tracked UUID |
| 13 | Align runtime profiles | P1 | High: dependency reproducibility | High | L | High | Compatibility matrix | Clean builds/imports/GPU smoke/full suite | Revert locks atomically |
| 14 | Align evaluation scripts | P1 | Medium-high: benchmark operability | High | S | Low | Item 13 paths | Syntax/dry-run/scoped smoke | Revert individual script |
| 15 | Make Compose standalone-capable | P1 | High: deployment | High | M | Medium | Items 12-13 | Build/health/submit/result smoke | Revert Compose/runbook change |
| 16 | Decide network trust/authentication | P2 | High if externally reachable | Medium | M-L | High | Threat model/consumers | Anonymous/authorized contract tests | Disable auth; restore trusted binding |
| 17 | Remove public absolute paths | P2 | Medium: disclosure/coupling | High | S-M | Medium | Consumer inventory; item 16 | Schema and compatibility tests | Temporary versioned diagnostic field |
| 18 | Define request/resource limits | P2 | High: resilience | Medium | M | Medium-high | Adversarial measurements | Bytes/pages/encryption/resource tests | Revert thresholds independently |
| 19 | Add safe running cancellation | P2 | Medium-high: operations | High | L | High | Item 3; cooperative engines | Stage-specific deterministic cancel tests | Disable and retain conflict response |
| 20 | Separate adapters from benchmark reporting | P2 | Medium: maintainability | High | L | High | Items 4-8 | Adapter/report schema parity | Revert extraction; keep wrappers |
| 21 | Isolate reconciliation policy | P2 | High: OCR decisions | High | L | High | Items 4-5 | Behavior/schema-equivalent suite | Revert behavior-neutral extraction |
| 22 | Split web responsibilities incrementally | P2 | Medium-high: job maintainability | High | L | High | Items 1-3, 11 | Web/recovery/race/API parity suite | Revert one boundary at a time |
| 23 | Fix mixed-run compare-text discovery | P2 | Medium-high: provenance | High | S-M | Low-medium | Items 6, 9 | Mixed run rejected; single run accepted | Restore explicit legacy opt-in |
| 24 | Add CI | P2 | High: regression gate | High | M | Low | Stable install; preferably item 13 | Ruff/mypy/full non-GPU suite in CI | Revert workflow only |
| 25 | Add repository `AGENTS.md` | P2 | Medium: safe agent work | High | S | Low | Accepted audit rules | Review and agent dry run | Revert documentation |
| 26 | Add runbooks | P2 | High: operations | High | M | Low | Items 12-15 | Fresh-user/new-PC walkthrough | Revert/supersede section |
| 27 | Add repo-local skill | P2 | Medium: repeatable evidence | Medium | M | Medium | Items 25-26 | Skill validation and dry run | Remove skill; runtime unchanged |
| 28 | Assess thin MCP last | P2 | Low until consumer exists | Medium-low | M-L | High | Items 11, 16-17 | HTTP-only consumer acceptance | Remove adapter; HTTP unchanged |
| 29 | Reconcile stale documentation | P3 | Medium: operator confusion | High | S | Low | Confirm hybrid-only contract | Link/path/current-help checks | Revert/supersede docs |
| 30 | Classify warnings and skips | P3 | Medium: CI signal | High | S | Low | None | Full suite accounts for every warning/skip | Revert policy; retain notes |


## P1 — Immediate durable-job integrity

### 1. Reproduce corrupt/tampered result recovery

Evidence:

- recovery: `src/uniscan/web/service.py:374-445`;
- result acceptance/promotion: `409-420`;
- result serving: `2008-2019`.

Test first:

- create a durable job record in a temporary job root;
- place a corrupt, truncated or replaced `result.pdf` at the expected path;
- restart/load the store;
- assert the job does not become a downloadable `done` result.

Minimal-fix acceptance:

- recovery requires a verifiable result seal and valid PDF/page count;
- invalid result becomes an explicit recoverable/terminal error state;
- no external/user file is deleted;
- existing recovery tests remain green.

Verification:

```powershell
python -m pytest -q tests/test_web_service.py
python -m pytest -q
```

Commit boundary: reproducer, then minimal fix as separate reviewable commits if
practical.

### 2. Reproduce result symlink/reparse escape

Evidence:

- recovery follows `input_candidate`/`result_candidate` with `.resolve()` at
  `src/uniscan/web/service.py:411-412`;
- result serving is at `2008-2019`.

Test first:

- create an external harmless temporary PDF/file;
- place a symlink/reparse-point result under `jobs/<id>`;
- load the store and request the result;
- assert the external target is neither trusted nor served;
- skip only when the platform cannot create the link, with the skip reason visible.

Minimal-fix acceptance:

- job input/result paths are regular non-link files contained in the job root;
- containment is checked after canonical resolution;
- behavior is defined on Windows junction/reparse boundaries;
- cleanup never follows an external target.

Verification: targeted web tests, then full suite.

### 3. Reproduce stale-worker publication after watchdog reclamation

Evidence:

- reclaim transition: `src/uniscan/web/service.py:215-258`;
- watchdog: `1707-1720`;
- worker publication/done boundary: `1577-1611`.

Test first:

- pause a worker after OCR but before publication;
- advance/reclaim its heartbeat deterministically;
- resume the stale worker;
- assert it cannot publish, mark done or remove artifacts owned by a newer attempt.

Minimal-fix acceptance:

- each running attempt has a lease/generation ID;
- publication performs an atomic ownership check;
- terminal-transition rejection is not the only fence;
- the test uses barriers/events, not timing sleeps.

Verification: targeted concurrency test repeated, then full suite.

## P1 — OCR incident regression fixtures

### 4. Derive punctuation-only geometry fixture

- Preserve the private ignored evidence named in `OBSERVED_OCR_FAILURES.md`.
- Create the smallest privacy-safe artifact that retains the EM-DASH-only line
  failure.
- Reproduce `invalid_chandra_attempt_evidence` before policy changes.
- Add searchable-text and visual/table-layout assertions.
- Decide retain/filter/represent policy only after benchmark comparison.

### 5. Derive exact-retention failure fixture

- Keep it separate from the punctuation case.
- Reproduce `Output PDF page 3 failed exact searchable text retention`.
- Identify the smallest assembly/validation boundary responsible.
- Add a regression test before any fix.

## P1 — Reproducible quality/performance baseline

### 6. Create a versioned benchmark manifest

Record fixture hashes, page ranges, Ground Truth versions, privacy/source-use status,
expected structural outcomes and benchmark groups.

### 7. Add benchmark metrics

- CER/WER where Ground Truth exists;
- searchable-text retention;
- reading-order/layout assertions;
- page-outcome counts;
- per-stage wall time;
- RAM/VRAM;
- engine invocation/render counts;
- chunk reuse/rerun reasons;
- output hashes, bytes and page count.

### 8. Capture first accepted baseline

Record commit, dirty state, dependency/model/CUDA/GPU identity, warm/cold cache and
raw machine-readable outputs. Do not tune OCR until this artifact exists.

## P1 — Runtime and cache provenance

### 9. Define immutable resolved run configuration

Characterize current environment reads, then capture resolved values once per run.
Preserve the existing application façade.

### 10. Complete hybrid run identity

Add package, executable, model and CUDA/runtime digests. Prove with tests that an
incompatible environment cannot reuse completed chunks.

### 11. Seal successful job results

Persist SHA-256, size, page count, run identity and validator revision as part of
atomic completion. Reuse the seal for restart recovery.

## P1 — New-PC deployment

### 12. Remove tracked machine-specific GPU UUID

- make `.env.example` machine-neutral;
- discover or explicitly configure the permitted device locally;
- retain GPU0 attestation semantics without one repository-owned host identity;
- test on the current machine and one clean/new-PC procedure.

### 13. Align runtime profiles

Choose and document exact common, Surya, Chandra and dev dependency truth. Add
hash-verified lock/constraints artifacts appropriate to the chosen workflow.

### 14. Align scripts

Make geometry evaluation use `.venv_surya`/`.venv_chandra` or explicit interpreter
arguments consistently. Remove machine-specific default PDF roots.

### 15. Make Compose standalone-capable

Provide a default network without pre-creation and an explicit shared-network
integration override. Verify build, health, submit, status and result retrieval.

## P2 — HTTP/API hardening

### 16. Decide network trust and authentication

Document whether the service is trusted-network-only. If not, add the smallest
authentication/authorization layer supported by actual consumers.

### 17. Remove public absolute paths

Replace host filesystem paths in normal API responses with identifiers/URLs.
Retain detailed paths only in an explicitly privileged diagnostic surface.

### 18. Define request/resource limits

Measure upload memory, PDF page/dimension limits, queue pressure and malicious or
encrypted PDF behavior before changing defaults.

### 19. Running cancellation

Defer implementation until engine processes can stop cooperatively and artifact
cleanup/publication fencing is proven. Preserve explicit conflict behavior meanwhile.

## P2 — Architecture cleanup after baseline

### 20. Separate production adapter contract from benchmark reporting

Characterize existing Chandra/Surya functions. Introduce a narrow shared contract
behind wrappers only if it reduces duplicate execution or improves testability.

### 21. Isolate reconciliation policy

Move only after regression fixtures cover current outcomes and schemas. No policy
change in the extraction commit.

### 22. Split web responsibilities incrementally

Separate repository/state transition, scheduler/executor and HTTP transport
logically; physical files are optional and follow tests.

### 23. Fix mixed-run compare-text discovery

Require an explicit summary/run identity or prove all selected engine reports share
one run. Add a mixed-report regression test first.

## P2 — Repository operations and agent support

### 24. Add CI

Start with Ruff, mypy and the full non-GPU suite. Report skips/warnings explicitly.
Add container/GPU acceptance separately rather than pretending it is portable CI.

### 25. Add repository `AGENTS.md`

Document protected user data, baseline commands, strict hybrid contract, artifact
ownership, test-first incident policy and the prohibition on automatic Understand
Anything use.

### 26. Add runbooks

Cover new-PC setup, model cache prewarm, Windows GUI/CLI, Docker, HTTP job recovery,
incident evidence preservation and benchmark execution.

### 27. Add a repo-local skill

Keep it operational and evidence-oriented: baseline capture, safe diagnostics,
benchmark provenance and incident recording. It must not delete artifacts or bypass
the application/API boundary.

### 28. Assess thin MCP last

Proceed only if an agent consumer needs submit/status/result/cancel/health. Implement
over HTTP, not OCR internals or filesystem access.

## P3 — Documentation and packaging hygiene

### 29. Reconcile stale documentation

Update the three-mode claims and absent installer reference only after confirming
the supported contract remains hybrid-only.

### 30. Classify test warnings and skips

Record why all seven skips and five warnings occur and which are acceptable in CI.

## Definition of done for every backlog item

- evidence/reproducer exists before a behavioral fix;
- scope is limited and reversible;
- no user source or retained incident artifact is modified/deleted;
- targeted tests and the complete suite are recorded;
- static checks remain clean where applicable;
- benchmark deltas are recorded for OCR/performance changes;
- documentation names exact commands, paths and assumptions;
- commit contains one coherent change.

## Assumptions

- Items are reordered only when new evidence changes severity or dependency.
- The first three reproducing tests remain the immediate mandatory sequence.
- Source/container mismatch is not an active finding: all 23 source hashes match.
- The 617-passed suite is authoritative and must not be described as partial.
