# Audit execution status — 2026-08-18

This is an evidence checkpoint, not a claim that the full audit plan is complete.
The reference start was `bbebe4bbb58c0e3a384558e24f22bc06663093c0`.

## Accepted changes

- Baseline, inventory, ranked backlog, target architecture, and both observed OCR
  incidents are recorded under `docs/audit/`.
- Durable-job recovery now rejects unsafe/tampered artifacts and stale attempts
  cannot publish after watchdog reclamation.
- Chandra punctuation-only geometry is retained. The separate exact-retention
  incident remains a reproducer only; its failed candidate was not preserved, so
  a root cause is not asserted.
- The repository has a deterministic model-free benchmark corpus and evaluator.
  No OCR accuracy or performance tuning has been accepted without a real engine
  baseline.
- A private real-engine incident/timing baseline now covers one page from each
  observed failure class. Both pass strict reconciliation; CER/WER and full
  exact-retention chunk context remain unmeasured.
- A repeat of the punctuation case with warm model caches produced a
  byte-identical PDF. It used fresh processes, so it is not an in-process warm
  latency measurement.
- GPU selection is host-configured, Compose starts standalone by default, and a
  read-only new-PC preflight plus deployment/incident/benchmark runbooks exist.
- CI, `AGENTS.md`, and the repo-local operator skill are present. The HTTP trust
  boundary is documented and MCP is deferred until a real consumer needs it.
- Docker dependency snapshots are preserved as observations only. They are not
  wired into installation and are not described as cross-platform locks.

## Current verification

On local `main`, after the accepted audit commits:

```text
python -m pytest -q
655 passed, 9 skipped, 2 xfailed, 5 warnings in 251.77s
```

The two strict expected failures reproduce early HTTP admission gaps for
malformed and encrypted PDFs. A forced run (`--runxfail`) produced two failures
because the API returned `202` instead of the expected `400`.

Targeted dependency snapshot checks passed (`3 passed`), with Ruff and
`git diff --check` clean. Docker standalone and shared-network Compose config
preflights passed earlier in this execution. Windows preflight found that the
existing Surya venv lacks the local `uniscan` install; it did not mutate it.

## Open high-value work

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
