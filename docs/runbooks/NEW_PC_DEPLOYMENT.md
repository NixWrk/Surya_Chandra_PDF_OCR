# New-PC deployment runbook

Docker is the canonical production path. Windows dual venvs are a development
fallback. This runbook separates read-only checks from mutating installation
and real OCR smoke work.

## Prerequisites

- Git and PowerShell 5.1 or newer;
- Docker Desktop with Compose and NVIDIA container support for Docker;
- an NVIDIA driver exposing the intended physical GPU as index 0;
- enough disk space for the image, model caches, work output, and source PDFs;
- network access for the first image build/model download.

## Configure the local GPU

From the repository root, inspect physical GPU0:

```powershell
nvidia-smi --id=0 --query-gpu=index,uuid,name,driver_version,compute_cap --format=csv,noheader
```

Copy `.env.example` to an untracked `.env` and set the full UUID returned above
as `UNISCAN_GPU_DEVICE_ID`. Never commit `.env` or the host UUID. Existing
`.env` files are operator-owned and must not be overwritten automatically.

## Read-only preflight

Set the same UUID in the current shell, then run:

```powershell
$env:UNISCAN_GPU_DEVICE_ID = "<full GPU0 UUID>"
.\scripts\preflight_new_pc.ps1 -Target Docker -Json
```

The preflight checks GPU0 identity, Docker, Compose parsing, the `ocr-api`
service, and required repository files. It does not build, start, download, or
run OCR.

For an existing shared Zotero Docker network:

```powershell
.\scripts\preflight_new_pc.ps1 -Target Docker -SharedNetwork -Json
```

## Install and start

Standalone:

```powershell
docker compose build
docker compose up -d
docker compose ps
curl.exe http://127.0.0.1:8000/health
```

Compose creates a Docker-managed volume named
`surya-chandra-ocr-hybrid-chunk-cache` for active resumable chunk work. Override
the name only in the untracked `.env` when multiple deployments share one host:

```dotenv
UNISCAN_HYBRID_CACHE_VOLUME=my-uniscan-hybrid-cache
```

The volume does not replace `./outputs`: durable job inputs, SQLite metadata,
published PDFs, and retained job evidence remain on the host bind. A missing or
empty chunk-cache volume can make an interrupted job repeat OCR, but it does not
remove an already published result or its durable input.

Inspect the volume without starting OCR:

```powershell
$hybridCacheVolume = if ($env:UNISCAN_HYBRID_CACHE_VOLUME) {
    $env:UNISCAN_HYBRID_CACHE_VOLUME
} else {
    "surya-chandra-ocr-hybrid-chunk-cache"
}
docker volume inspect $hybridCacheVolume
docker compose config
```

Do not prune this volume while a job is running or while an interrupted run is
expected to resume.

Shared-network integration:

```powershell
docker compose -f docker-compose.yml -f docker-compose.shared-network.yml build
docker compose -f docker-compose.yml -f docker-compose.shared-network.yml up -d
```

The external network must already exist for the shared mode. The standalone
mode must not create or require it.

## Smoke

Use only a non-sensitive, approved fixture. Submit through the HTTP API or use
the local GPU smoke script. Record whether caches were cold or warm, the commit,
image ID, GPU/driver, elapsed time, output hash, page count, and validation
result. A health check alone does not prove model readiness or OCR correctness.

## Windows fallback

Run `setup_dual_venv.cmd` to create the isolated Surya and Chandra environments,
then:

```powershell
.\scripts\preflight_new_pc.ps1 -Target Windows -Json
```

Both venvs must import their engine, PyTorch, and UniScan and complete a tiny
CUDA tensor operation on device 0. The current audited host exposed a stale
`.venv_surya` that lacked `uniscan`; preflight correctly failed that environment
without repairing it.

## Backup and rollback

Before updating, record the Git commit and Docker image ID. Back up operator
configuration plus `outputs/jobs` or the configured `UNISCAN_WORK_ROOT/jobs`.
Retain the named hybrid-cache volume when same-deployment interruption resume
matters; it is not required to serve completed job results. Export it before a
host migration only when preserving resumable work is worth the extra storage.
Model caches may be large; preserve them if redownload time matters. Never copy
a live SQLite database without stopping the service or using a consistent
backup method.

Stop the service before changing code, packages, models, Python, CUDA, the driver,
or the selected GPU runtime. The old `UNISCAN_WORK_ROOT/runs/hybrid_chunk_cache`
is valid only for the old deployment. Preserve individual failed runs when they
are incident evidence, then rename/quarantine the old cache root so startup creates
a fresh one. Do not copy completed chunks into the new root and do not delete job
results or model caches as part of this step.

Rollback by stopping the current service, checking out the recorded commit or
retagging the recorded image, restoring the matching local configuration, and
running preflight before start. Do not delete newer job evidence during
rollback.
