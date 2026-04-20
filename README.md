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
