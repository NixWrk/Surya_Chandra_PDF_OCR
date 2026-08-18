---
name: uniscan-ocr-operator
description: Safely diagnose, test, benchmark, deploy, and recover the UniScan hybrid Chandra+Surya PDF OCR repository. Use for OCR job failures, durable-job recovery, GPU/cache preflight, synthetic or real benchmark capture, Docker or Windows setup, smoke checks, and incident evidence recording in this repository.
---

# UniScan OCR Operator

Operate from the repository root. Preserve source PDFs, job directories, caches,
and retained failure evidence. Prefer read-only evidence collection before any
repair.

## Guardrails

- Never invoke Understand Anything or a knowledge graph unless the user
  explicitly requests it.
- Treat chandra+surya as the production OCR contract. Single-engine commands are
  diagnostic or benchmark-only.
- Never delete or overwrite PDFs/, outputs/, model caches, job records, or
  incident artifacts as part of diagnosis.
- Do not upgrade engines, PyTorch, CUDA, preprocessing, prompts, or geometry
  policy without a versioned corpus and an accepted metric comparison.
- Add a reproducing test before fixing a confirmed correctness defect.
- Keep commits small and reversible. Cite paths, commands, tests, hashes, or
  measurements for every operational conclusion.

## Choose the workflow

1. For repository/runtime readiness, run preflight.
2. For a failed job, preserve and inspect incident evidence.
3. For quality or speed claims, run a versioned benchmark.
4. For deployment, use Docker unless the user explicitly needs Windows venvs.
5. For code changes, run the verification workflow before handoff.

## Preflight

Discover physical GPU index 0 without writing configuration:

    nvidia-smi --id=0 --query-gpu=index,uuid,name --format=csv,noheader
    $env:UNISCAN_GPU_DEVICE_ID = "<full GPU0 UUID>"
    .\scripts\preflight_new_pc.ps1 -Target Docker -Json
    .\scripts\preflight_new_pc.ps1 -Target Windows -Json

Use -SharedNetwork only for the opt-in docker-compose.shared-network.yml
integration. Preflight must not install, download, build, start services, or run
OCR.

## Diagnose a job

1. Record time, commit, dirty state, deployment mode, job ID, sanitized document
   name, page/chunk, status, and exact error.
2. Query /health, /api/jobs/<job_id>, and /api/jobs/<job_id>/metadata. Download
   /result only for a done job.
3. Inspect metadata.json, events.jsonl, chunk manifest, engine reports,
   reconciliation evidence, and validation error without altering them.
4. Hash relevant files and distinguish observed facts from inference.
5. Add privacy-safe evidence to docs/audit/OBSERVED_OCR_FAILURES.md; keep private
   source material ignored.
6. Reproduce at the narrowest boundary before proposing a fix.

Queued cancellation uses POST /api/jobs/<job_id>/cancel. Treat rejection of
running-job cancellation as the current explicit contract; do not kill engine
processes or remove their artifacts manually.

## Benchmark

Generate the privacy-safe offline corpus only into a caller-owned temporary or
ignored directory:

    python -m runpy benchmarks.synthetic.v1.generate --output <temporary-path>
    python -m runpy benchmarks.synthetic.v1.generate --output <temporary-path> --check

Synthetic generation does not run OCR and must retain
metrics.model_status=not_run. Do not call its vector degradation fixture a
raster scan. A real accepted baseline must record commit/dirty state,
dependencies/models, GPU/CUDA, cache state, raw outputs, CER/WER, page failures,
exact retention, stage timing, RAM/VRAM, reuse/rerun counts, output hashes,
bytes, and page count.

## Deploy and smoke

Default standalone Docker:

    docker compose build
    docker compose up -d

For the shared Zotero network, add -f docker-compose.shared-network.yml to both
commands. Do not create the external network unless that integration is
requested.

Use setup_dual_venv.cmd only for the Windows fallback. Run
scripts/run_hybrid_gpu_smoke.ps1 only with a user-approved, non-sensitive PDF;
it performs real model work and writes under outputs/gpu_hybrid_smoke.

## Verify changes

Run targeted tests first, then:

    python -m pytest -q
    python -m ruff check src tests
