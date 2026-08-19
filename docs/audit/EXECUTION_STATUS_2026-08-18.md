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
  A second immutable checkpoint exercises the remaining degraded, rotated,
  native-text, and 23-page fixtures. Across both checkpoints, all nine corpus
  fixtures have been exercised at least once. No OCR accuracy or performance
  tuning has been accepted without real-engine evidence.
- The 23-page checkpoint exposed two consecutive full premerge evidence passes
  per chunk. A deterministic RED test and narrow fix now reuse the already
  validated evidence object without removing sealing, TOCTOU, manifest, or
  runtime-drift fences. One comparable offline after-run reduced wall time from
  922.991 to 463.709 seconds while CER/WER remained zero and retention/mapping
  passed.
- The clean `771b5de` source was rebuilt as an offline source/install layer,
  matched against all 26 tracked production inputs, and promoted to the local
  production service as image `sha256:72ad02bb45d...`; the old image remains
  tagged for rollback. Three fresh production-bind `long-23p` runs passed all
  quality gates with 596.799-second median wall time, 628.505-second observed
  maximum, 4.22 GB observed RAM peak, and 11,046 MiB peak GPU0 VRAM above
  background. A real async HTTP smoke also completed with exact retention.
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

The latest complete software suite was run on local `main` through production
commit `1cb708c`:

```text
python -m pytest -q
679 passed, 9 skipped, 5 warnings in 208.99s
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

The extended real-engine checkpoint ran from clean source commit `3acda85` with
image `sha256:f470cf1520e43ae67b70bf63e5dded12235ebde07e5139c68307cb867b06bdc0`,
`--pull never`, `--network none`, offline settings, and read-only model caches.
All four added fixtures completed; their source hashes match the corpus manifest,
CER/WER were zero, exact retention passed, and page mapping passed. Peak RAM/VRAM
and repeat-run latency were not measured.

The premerge evidence-reuse checkpoint used the same `long-23p` source hash and
immutable image from clean commit `1cb708c`, again with `--pull never`,
`--network none`, offline settings, and read-only model caches. Wall time was
463.709 seconds versus 922.991 seconds before. The model-free evaluator reports
CER/WER zero, exact retention pass, page mapping pass, 23 output pages, and zero
partial failures. At that checkpoint, peak RAM/VRAM and median/tail latency
remained open. Exact evidence is in
`PREMERGE_EVIDENCE_PERFORMANCE_2026-08-19.md` and the ignored local benchmark
directory named there.

The follow-up production-like series closes the long-document repeat/resource
measurement gap. It used fresh chunk caches, read-only model caches, network
isolation, and an exact Windows `/data/work` bind. All three runs had CER/WER
zero, retention/mapping pass, and zero partial failures. Wall median was 596.799
seconds; observed maximum was 628.505 seconds; the observed RAM peak was
4,221,952,852 bytes; and total GPU0 VRAM peaked 11,046 MiB above the immediate
background. This observed maximum is not a p95. See
`PRODUCTION_PROMOTION_2026-08-19.md`.

## Open high-value work

Current order after the synthetic baseline and blank-page fix:

Accepted boundary: resumable caches are same-deployment only. The update runbook
requires a fresh cache root after code/runtime/model changes. Richer attestation
is deferred unless cross-upgrade resume becomes a supported requirement; there is
no global document registry or historical model-tree hashing.

1. Validate clean dependency resolutions separately for Docker/Windows and
   `cu126`/`cu128`; the offline source-layer build is only partial evidence.
2. Extend the real-engine baseline with representative raster scans and
   controlled `mixed-layout` experiments. The procedural long fixture now has
   repeat timing/resource evidence; the current `mixed-layout` CER/WER remain
   `0.287293`/`0.366667`.
3. Capture a preserved rejected candidate if the historical private
   exact-retention failure recurs; do not infer its root cause from a final PDF.
4. Measure and define queue and page-count limits.
5. Add bounded age-and-size quotas for explicit persistent benchmark/resume
   caches. Successful HTTP run caches already clean up by default; job retention
   remains 30 days success/90 days failure unless overridden.
6. Remove normal-response absolute paths only with a versioned compatibility
   decision; the service remains trusted-network-only.
7. Instrument the production-bind 23-page residual (256.788-second median,
   282.728-second observed maximum) before any further validation, rendering,
   or storage optimization.

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
