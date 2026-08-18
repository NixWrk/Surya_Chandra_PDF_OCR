# Audit execution status — 2026-08-18

This is an evidence checkpoint, not a claim that the full audit plan is complete.
The reference start was `bbebe4bbb58c0e3a384558e24f22bc06663093c0`.

## Accepted changes

- Baseline, inventory, ranked backlog, target architecture, and both observed OCR
  incidents are recorded under `docs/audit/`.
- Durable-job recovery now rejects unsafe/tampered artifacts and stale attempts
  cannot publish after watchdog reclamation. Successful result recovery is
  sealed and link/hardlink containment tests remain in the suite.
- Malformed and encrypted PDF uploads are rejected before durable job creation.
  Idempotency conflicts are checked before parsing a replacement payload.
- HTTP worker and watchdog threads now have an explicit bounded shutdown path;
  the full-suite order-dependent thread leak is fixed.
- Chandra punctuation-only geometry is retained. The separate exact-retention
  incident remains a reproducer only; its failed candidate was not preserved, so
  a root cause is not asserted.
- The repository has a deterministic model-free benchmark corpus and evaluator,
  plus an offline real-engine synthetic checkpoint with valid English, Russian,
  mixed-layout, blank/graphics, and three-page retention Ground Truth.
  No OCR accuracy or performance tuning has been accepted without a real engine
  baseline.
- The synthetic run exposed a second OCR publication failure on `blank-graphics`.
  It was reproduced test-first and fixed by requiring sealed, recomputed blank
  raster evidence. The preserved run now rebuilds artifact-only as a valid
  two-page PDF without rerunning OCR.
- A private real-engine incident/timing baseline covers both observed failure
  classes and a full 21-page chunked exact-retention run. The run passed, but the
  historical rejected candidate is unavailable, so its root cause is not proven;
  CER/WER remain unmeasured because reviewed Ground Truth is unavailable.
- A repeat of the punctuation case with warm model caches produced a
  byte-identical PDF. It used fresh processes, so it is not an in-process warm
  latency measurement.
- GPU selection is host-configured, Compose starts standalone by default, and a
  read-only new-PC preflight plus deployment/incident/benchmark runbooks exist.
- CI, `AGENTS.md`, and the repo-local operator skill are present. The HTTP trust
  boundary is documented and MCP is deferred until a real consumer needs it.
- Docker dependency snapshots are preserved as observations only. They are not
  wired into installation and are not described as cross-platform locks.
- Chunked OCR now resolves one runtime-configuration snapshot for a run and
  fails closed before publishing if tracked settings drift. The snapshot stays
  inside that run's resumable cache; it is not a document-history registry.

## Current verification

On local `main`, through `c89b593`:

```text
python -m pytest -q
678 passed, 9 skipped, 5 warnings in 259.49s
```

There are no expected failures. The former malformed/encrypted admission xfails
are ordinary passing tests after `d0085fb` and `51ca830`.

Targeted verification also includes `164 passed, 1 skipped` for page
reconciliation, `95 passed` for searchable-artifact assembly, Ruff, mypy, and
`git diff --check`. A no-cache offline source-layer Docker build passed in both
engine venvs, CLI help passed, and the preserved blank/graphics artifact rebuilt
successfully. This is not a clean dependency build. Windows preflight still
reports that the existing Surya venv lacks the local `uniscan` install; it was
not mutated.

## Open high-value work

Current order after the synthetic baseline and blank-page fix:

Accepted boundary: resumable caches are same-deployment only. The update runbook
requires a fresh cache root after code/runtime/model changes. Richer attestation
is deferred unless cross-upgrade resume becomes a supported requirement; there is
no global document registry or historical model-tree hashing.

1. Validate clean dependency resolutions separately for Docker/Windows and
   `cu126`/`cu128`; the offline source-layer build is only partial evidence.
2. Extend the real-engine baseline with median/tail timing, peak RAM/VRAM,
   rotated/noisy/skewed fixtures, and controlled `mixed-layout` experiments. Its
   current CER/WER are `0.287293`/`0.366667`.
3. Capture a preserved rejected candidate if the historical private
   exact-retention failure recurs; do not infer its root cause from a final PDF.
4. Measure and define queue and page-count limits.
5. Add bounded age-and-size quotas for explicit persistent benchmark/resume
   caches. Successful HTTP run caches already clean up by default; job retention
   remains 30 days success/90 days failure unless overridden.
6. Remove normal-response absolute paths only with a versioned compatibility
   decision; the service remains trusted-network-only.

The earlier checkpoint list is retained below as historical audit context:

1. Extend the accepted real Surya+Chandra incident/timing baseline with reviewed
   Ground Truth, CER/WER, warm-run repetitions, peak RAM/VRAM, and full pages
   11–20 exact-retention context.
2. Reproduce the exact-retention incident from a preserved failed candidate, or
   capture a new equivalent failure. Do not guess a merge fix from the final PDF.
3. Resolve run configuration once per run and complete cache/run identity across
   subprocesses and artifact helpers. A partial environment-key patch is unsafe.
4. Validate clean dependency resolutions separately for Docker/Windows and
   `cu126`/`cu128` before enforcing constraints. Full model-weight hashes have not
   been computed, so model provenance is metadata-and-size evidence only.
5. Replace 14 fake HTTP upload fixtures with valid generated PDFs before enabling
   strict PyMuPDF admission parsing. Do not weaken validation to a `%PDF` prefix.
6. Define measured queue and page-count limits. Current evidence shows one running
   plus 31 queued jobs and acceptance of a 101-page PDF; no threshold is invented.
7. Remove normal-response absolute paths only with a versioned compatibility
   decision. The service remains documented as trusted-network-only.

## Guardrails still in force

- Preserve source PDFs, caches, retained incidents, benchmark raw outputs, and
  user-generated artifacts.
- No Surya/Chandra upgrade or OCR policy tuning before an accepted benchmark.
- Keep changes small, reversible, test-first, and separately committed.
- Do not push, delete artifacts, or invoke Understand Anything without explicit
  user authorization.
- Do not create a permanent catalog of processed user documents. Identity and
  chunk metadata remain scoped to a specific active/resumable cache or retained
  result and are removed according to explicit cleanup/retention policy.
