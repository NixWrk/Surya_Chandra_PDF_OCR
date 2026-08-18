# UniScan repository working agreement

These instructions apply to the whole repository.

## Production invariants

- Production OCR is strict `chandra+surya`: Chandra supplies text and Surya
  supplies geometry. Single-engine modes are diagnostics only.
- Existing text layers are removed before OCR. Every requested page must have
  accepted evidence before PDF assembly and publication.
- A failed candidate must never replace a prior successful result.
- Durable-job publication is ownership-fenced. Never bypass job locks, result
  containment checks, or validation to make a job appear successful.
- GPU work is restricted to the host-configured physical GPU index 0. Never
  commit a host GPU UUID or absolute host path.
- Do not invoke Understand Anything or any knowledge graph unless the user
  explicitly requests it.

## Protected data

Treat `PDFs/`, `outputs/`, job roots, model caches, benchmark outputs, and
retained incident evidence as user data. Do not delete, rename, rewrite, or add
them to Git during cleanup or testing. Generate fixtures only in pytest temp
directories or an explicitly caller-owned ignored directory.

## Evidence-first changes

1. Read `docs/FULL_REPOSITORY_AUDIT_PLAN.md` and
   `docs/audit/IMPLEMENTATION_BACKLOG.md` before audit work.
2. Record the exact failure and add a deterministic reproducer before a
   correctness fix.
3. Do not change engines, model revisions, preprocessing, prompts, geometry, or
   retry policy without a versioned benchmark and before/after metrics.
4. Keep changes small and reversible. Stage only named files because ignored or
   untracked runtime evidence may be present.
5. Support claims with paths, commands, tests, hashes, or measurements.

## Commands

From the repository root:

```powershell
python -m pytest -q
python -m ruff check src tests
python -m mypy
git diff --check
```

Read-only deployment checks:

```powershell
$env:UNISCAN_GPU_DEVICE_ID = "<physical GPU0 UUID>"
.\scripts\preflight_new_pc.ps1 -Target Docker -Json
.\scripts\preflight_new_pc.ps1 -Target Windows -Json
```

Synthetic corpus generation is model-free and must target a temporary or
ignored directory:

```powershell
python -m runpy benchmarks.synthetic.v1.generate --output <temporary-path>
```

Real GPU smoke and benchmark commands run models, may take a long time, and
write artifacts. Use only an approved non-sensitive PDF and report cache state.

## Architecture boundaries

- `src/uniscan/app/ocr_pipeline.py`: stable application facade, chunks,
  reconciliation, and publication orchestration.
- `src/uniscan/ocr/benchmark.py`: engine invocation and diagnostic reports.
- `src/uniscan/ocr/artifact_searchable.py`: searchable-PDF construction and
  validation.
- `src/uniscan/web/service.py`: durable jobs, queue, worker, and HTTP mapping.
- `src/uniscan/ui/`: thin local GUI adapter.

Prefer characterization tests and internal seams over large file moves. HTTP,
CLI, and GUI should call the application facade rather than OCR internals.

## Deployment

Docker Compose is the canonical runtime. `docker-compose.yml` is standalone;
`docker-compose.shared-network.yml` is an explicit opt-in integration layer.
The Windows dual-venv setup is a supported development fallback. Keep Surya and
Chandra dependencies isolated.
