# Surya Chandra PDF OCR

OCR-only repository for building searchable PDF from scanned PDF input.

Supported modes:

1. `surya` (`surya-surya`)
2. `chandra` (`chandra-chandra`)
3. `chandra+surya` (hybrid: Chandra text + Surya geometry)

## Core principles

1. OCR pipeline only (no camera/session UI legacy).
2. Dual-venv runtime to avoid dependency conflicts between engines.
3. Strict geometry behavior for Surya (no silent text-only degradation).

## Repository layout

1. `src/uniscan/app` - orchestration (`build_searchable_pdf`, mode routing)
2. `src/uniscan/ocr` - OCR benchmark + artifact-based searchable build
3. `src/uniscan/ui/basic_ocr_gui.py` - minimal local GUI
4. `run_basic_gui.cmd` - GUI launcher with dual-venv routing
5. `setup_dual_venv.cmd` - one-time environment setup
6. `docs/` - cleanup plan, inventory, operational notes

## Quick start (recommended)

```powershell
cd D:\Git_Code\Surya_Chandra_PDF_OCR
.\setup_dual_venv.cmd
.\run_basic_gui.cmd
```

`setup_dual_venv.cmd` creates:

1. `.venv_surya`
2. `.venv_chandra`

`run_basic_gui.cmd` sets:

1. `UNISCAN_SURYA_PYTHON=<repo>\.venv_surya\Scripts\python.exe`
2. `UNISCAN_CHANDRA_PYTHON=<repo>\.venv_chandra\Scripts\python.exe`

So each OCR engine runs in its own interpreter.

## CLI entrypoints

```powershell
python -m uniscan benchmark-ocr --help
python -m uniscan benchmark-ocr-canonical --help
python -m uniscan prepare-compare-txt --help
python -m uniscan build-searchable-from-artifacts --help
python -m uniscan compare-chandra-geometry --help
python -m uniscan searchable-pdf --help
python -m uniscan serve-http --help
```

## Container quick start (Docker + GPU)

Repository already contains:

1. `Dockerfile` (dual-venv image: Surya + Chandra)
2. `docker-compose.yml` (API service with GPU and cache volumes)
3. `scripts/docker-entrypoint.sh` (runtime env routing)

Build and run:

```powershell
cd D:\Git_Code\Surya_Chandra_PDF_OCR
docker compose build
docker compose up -d
```

Service endpoint:

`http://localhost:8000`

Stop:

```powershell
docker compose down
```

## Container runtime model

Inside container:

1. `/opt/venvs/surya` -> Surya runtime
2. `/opt/venvs/chandra` -> Chandra runtime
3. default API process runs from Chandra venv
4. Surya calls are routed through `UNISCAN_SURYA_PYTHON`
5. Chandra calls are routed through `UNISCAN_CHANDRA_PYTHON`

Mounted persistent volumes (host -> container):

1. `./.hf_cache_chandra` -> `/cache/hf_chandra`
2. `./.hf_cache_surya` -> `/cache/hf_surya`
3. `./.surya_cache` -> `/cache/surya_models`
4. `./.modelscope_cache` -> `/cache/modelscope`
5. `./outputs` -> `/data/work`
6. `./PDFs` -> `/data/in`

This keeps model weights and OCR outputs between container restarts.

## Ways to interact with container

### 1) Built-in web UI

Open:

`http://localhost:8000`

Web page supports:

1. PDF upload
2. mode selection (`surya`, `chandra`, `chandra+surya`)
3. language (`rus+eng` by default)
4. optional pages list
5. strict toggle

### 2) Sync HTTP API

Endpoint:

`POST /searchable-pdf`

Query params:

1. `mode=surya|chandra|chandra+surya`
2. `lang=rus+eng` (or any supported value)
3. `pages=1,3,5-8` (optional)
4. `strict=1|0`

Request body:

`application/pdf` raw bytes

Example:

```bash
curl -X POST "http://localhost:8000/searchable-pdf?mode=chandra+surya&lang=rus+eng&strict=1" \
  -H "Content-Type: application/pdf" \
  --data-binary "@input.pdf" \
  -o output.searchable.pdf
```

### 3) Async HTTP API

Endpoints:

1. `POST /api/jobs` - create job
2. `GET /api/jobs/{job_id}` - status/progress
3. `GET /api/jobs/{job_id}/result` - downloadable PDF

Example:

```bash
# Create async job
curl -X POST "http://localhost:8000/api/jobs?mode=chandra+surya&lang=rus+eng&strict=1&filename=input.pdf" \
  -H "Content-Type: application/pdf" \
  --data-binary "@input.pdf"

# Poll status
curl "http://localhost:8000/api/jobs/<job_id>"

# Download result
curl -L "http://localhost:8000/api/jobs/<job_id>/result" -o output.searchable.pdf
```

### 4) CLI inside running container

```bash
# Check CLI help
docker compose exec ocr-api /opt/venvs/chandra/bin/python -m uniscan --help

# Process PDF from mounted /data/in
docker compose exec ocr-api /opt/venvs/chandra/bin/python -m uniscan searchable-pdf \
  --pdf /data/in/input.pdf \
  --mode chandra+surya \
  --work-root /data/work \
  --strict
```

Result appears in:

`./outputs` (host) and `input.pdf` can be overwritten by command semantics.

## GPU requirements for container

1. NVIDIA driver on host
2. Docker with GPU runtime (`--gpus all`)
3. Docker Compose plugin with GPU support

Quick validation:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

If you need CPU-only debug mode, override:

1. `UNISCAN_CHANDRA_REQUIRE_GPU=0`
2. `UNISCAN_CHANDRA_TORCH_DEVICE=cpu`

## Typical pipeline

```powershell
python -m uniscan searchable-pdf `
  --pdf "D:\path\input.pdf" `
  --mode chandra+surya `
  --strict
```

## Hybrid geometry behavior

In hybrid mode:

1. Text source: Chandra
2. Geometry source: Surya

Surya geometry sidecars are mandatory by default. If geometry sidecars are missing, run fails instead of silently falling back to low-quality text-only behavior.

## Caches and local artifacts

Runtime caches are local to repository root (ignored by git):

1. `.hf_cache*`
2. `.surya_cache`
3. `.modelscope_cache`
4. `.venv*`
5. `.tmp*`
6. `outputs/`

## Troubleshooting

### 1) Chandra fails with `CUDA unavailable`

Check torch stack inside `.venv_chandra`:

```powershell
@'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
'@ | .\.venv_chandra\Scripts\python.exe -
```

If needed, reinstall CUDA wheels:

```powershell
uv pip install --python .\.venv_chandra\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cu128 `
  --upgrade --reinstall `
  "torch==2.11.0+cu128" `
  "torchvision==0.26.0+cu128" `
  "torchaudio==2.11.0+cu128"
```

### 2) Chandra cache preflight fails

Set cache root explicitly before launch:

```powershell
$env:UNISCAN_CHANDRA_HF_HOME = "D:\Git_Code\Surya_Chandra_PDF_OCR\.hf_cache"
.\run_basic_gui.cmd
```

### 3) Surya fallback must stay disabled

Default behavior enforces geometry quality:

1. `UNISCAN_SURYA_ALLOW_TEXT_FALLBACK=0`
2. `UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON=1`

## Cleanup and refactor docs

1. `docs/CLEANUP_REFACTOR_PLAN.md`
2. `docs/REPO_INVENTORY_KEEP_REMOVE.md`

## Rollback checkpoint

Stable checkpoint tag before the large cleanup/refactor pass:

`checkpoint/dual-venv-stable-20260420`
