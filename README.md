# Surya Chandra PDF OCR

Surya Chandra PDF OCR is a small OCR pipeline for turning scanned PDFs into searchable PDFs.
It is built around two OCR engines:

1. **Chandra** extracts text.
2. **Surya** provides page geometry for accurate invisible text placement.
3. **Hybrid mode** (`chandra+surya`) combines Chandra text with Surya geometry and is the default mode.

The project is meant for people who have scanned documents and want a local, reproducible workflow that produces PDFs with selectable and searchable text. It is especially useful when text-only OCR output is not enough and the text layer needs to align with the scanned page.

## Should You Use This?

Use this project if you need:

1. Searchable PDFs from scanned PDF files.
2. Local processing with Python, Docker, or a simple desktop GUI.
3. Russian/English OCR by default (`rus+eng`), with other language codes passed through where supported.
4. Strict geometry behavior: hybrid output should fail loudly if the geometry sidecar is missing instead of silently producing a low-quality text layer.

This project is probably not the right fit if you need:

1. A general document management system.
2. A camera/scanner capture UI.
3. CPU-only performance on large batches.
4. A polished end-user desktop application with installers and automatic updates.

## Requirements

Recommended local setup:

1. Windows with PowerShell or `cmd.exe`.
2. Python 3.11 available through `py -3.11` or `python`.
3. NVIDIA GPU and current NVIDIA driver.
4. Internet access on first setup to download Python packages and model weights.
5. `uv` is recommended but not required; the setup script falls back to `pip`.

The bootstrap script intentionally creates two separate virtual environments because Surya and Chandra have different dependency stacks. Keeping them isolated prevents one engine from upgrading packages in a way that breaks the other engine.

1. `.venv_surya`
2. `.venv_chandra`

Expected versions after a healthy setup:

1. `.venv_surya`: `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `torchaudio==2.11.0+cu128`, `pillow>=10.2,<11.0`.
2. `.venv_chandra`: `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `torchaudio==2.11.0+cu128`.

`pillow` is pinned below 11 only where Surya needs it. `surya-ocr==0.17.1` requires `pillow>=10.2,<11.0`, while Chandra can currently run with a newer Pillow version. This is one of the reasons the project does not use one shared venv for both engines. The base `uniscan` package allows newer Pillow builds; the Surya venv is pinned separately by `setup_dual_venv.cmd`.

The setup script also downloads and verifies the model weights. Runtime is intentionally strict: the GUI and CLI require the local caches to exist and will fail instead of silently downloading weights or degrading output quality mid-run.

Model cache locations:

1. Chandra weights: `.hf_cache_chandra`, model `datalab-to/chandra-ocr-2`.
2. Surya weights: `.surya_cache`, components `text_detection` and `text_recognition`.
3. Auxiliary HF/ModelScope caches: `.hf_cache_surya` and `.modelscope_cache`.

## Project Status

Repository hardening is complete as of 2026-04-21. The intended user path is now: clone the repository, run `setup_dual_venv.cmd`, then run `run_basic_gui.cmd` or the CLI. Future work should be treated as maintenance, dependency updates, or feature work rather than initial repository rescue.

Completed baseline:

1. Windows dual-venv setup for Surya and Chandra.
2. CUDA PyTorch installation and verification in both venvs.
3. Setup-time model weight downloads and cache verification.
4. Strict runtime behavior when caches or geometry sidecars are missing.
5. English README, menu labels, GUI copy, and web UI copy.
6. Human-readable troubleshooting for expected setup warnings.

## Last Verified Versions

Last checked on 2026-04-21 with Python 3.11 on Windows and CUDA PyTorch `cu128`.

Surya venv:

1. `torch==2.11.0+cu128`
2. `torchvision==0.26.0+cu128`
3. `torchaudio==2.11.0+cu128`
4. `surya-ocr==0.17.1`
5. `pillow==10.4.0`
6. `transformers==4.57.1`
7. `huggingface-hub==0.36.2`
8. `pypdfium2==4.30.0`
9. `pypdf==6.10.2`
10. `reportlab==4.4.10`
11. `setuptools==70.2.0`

Chandra venv:

1. `torch==2.11.0+cu128`
2. `torchvision==0.26.0+cu128`
3. `torchaudio==2.11.0+cu128`
4. `chandra-ocr==0.2.0`
5. `pillow==12.1.1`
6. `transformers==5.5.4`
7. `huggingface-hub==1.11.0`
8. `pypdfium2==4.30.0`
9. `pypdf==6.10.2`
10. `reportlab==4.4.10`
11. `setuptools==70.2.0`

## Quick Start: Local GUI

```powershell
git clone https://github.com/NixWrk/Surya_Chandra_PDF_OCR.git
cd Surya_Chandra_PDF_OCR
.\setup_dual_venv.cmd
.\run_basic_gui.cmd
```

The first setup can take a while. The script installs both environments, installs CUDA builds of PyTorch, downloads the OCR model weights, and verifies that the required caches are ready before it exits successfully.

After setup, the GUI lets you:

1. Choose a PDF.
2. Pick `chandra+surya`, `chandra`, or `surya`.
3. Optionally limit OCR to pages such as `1,3,5-8`.
4. Optionally remove an existing text layer before building the new searchable PDF.

By default, the GUI overwrites the selected input PDF with the searchable version. Intermediate artifacts are written under `outputs/`.

## Verify GPU PyTorch

Run this after setup if OCR is slow or if Chandra reports that CUDA is unavailable:

```powershell
@'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
print("cuda_device_0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
'@ | .\.venv_chandra\Scripts\python.exe -
```

Repeat for Surya:

```powershell
@'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
print("cuda_device_0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
'@ | .\.venv_surya\Scripts\python.exe -
```

A healthy GPU install should show a `torch` version containing `+cu` and `cuda_available: True`. For Surya, `pillow` should stay below `11.0` because `surya-ocr==0.17.1` depends on that range. For Chandra, `pillow 12.x` is acceptable unless Chandra changes its own dependency constraints.

## CLI Usage

Use the Chandra environment for the main CLI:

```powershell
.\.venv_chandra\Scripts\python.exe -m uniscan --help
```

Build a searchable PDF in the default hybrid mode:

```powershell
.\.venv_chandra\Scripts\python.exe -m uniscan searchable-pdf `
  --pdf "D:\path\input.pdf" `
  --mode chandra+surya `
  --lang rus+eng `
  --strict
```

Useful commands:

```powershell
.\.venv_chandra\Scripts\python.exe -m uniscan searchable-pdf --help
.\.venv_chandra\Scripts\python.exe -m uniscan benchmark-ocr --help
.\.venv_chandra\Scripts\python.exe -m uniscan prepare-compare-txt --help
.\.venv_chandra\Scripts\python.exe -m uniscan build-searchable-from-artifacts --help
.\.venv_chandra\Scripts\python.exe -m uniscan serve-http --help
```

## HTTP Service

Start the local web/API service:

```powershell
.\.venv_chandra\Scripts\python.exe -m uniscan serve-http --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Synchronous API:

```bash
curl -X POST "http://127.0.0.1:8000/searchable-pdf?mode=chandra+surya&lang=rus+eng&strict=1" \
  -H "Content-Type: application/pdf" \
  --data-binary "@input.pdf" \
  -o output.searchable.pdf
```

Asynchronous API:

```bash
curl -X POST "http://127.0.0.1:8000/api/jobs?mode=chandra+surya&lang=rus+eng&strict=1&filename=input.pdf" \
  -H "Content-Type: application/pdf" \
  --data-binary "@input.pdf"

curl "http://127.0.0.1:8000/api/jobs/<job_id>"
curl -L "http://127.0.0.1:8000/api/jobs/<job_id>/result" -o output.searchable.pdf
```

## Docker

Docker is useful when you want a repeatable GPU runtime with persistent model caches.

```powershell
docker compose build
docker compose up -d
```

Open:

```text
http://localhost:8000
```

Stop:

```powershell
docker compose down
```

The compose file mounts these local folders:

1. `.hf_cache_chandra` for Chandra Hugging Face weights.
2. `.hf_cache_surya` for Surya Hugging Face weights.
3. `.surya_cache` for Surya model cache.
4. `.modelscope_cache` for ModelScope cache.
5. `outputs` for work artifacts.
6. `PDFs` for optional input files.

Docker GPU requirements:

1. NVIDIA driver on the host.
2. Docker with GPU support.
3. Docker Compose with `gpus: all` support.

Quick GPU check:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## Runtime Caches

The project keeps heavyweight runtime files out of git. These folders are expected to be local:

1. `.venv_surya`
2. `.venv_chandra`
3. `.hf_cache*`
4. `.surya_cache`
5. `.modelscope_cache`
6. `.uv_cache`
7. `.tmp_*`
8. `outputs`

If a model download is interrupted, deleting the incomplete cache for that engine and rerunning setup or OCR is often enough.

## Modes

`chandra+surya`:
The default. Chandra provides text, Surya provides geometry. Best target for searchable PDFs when both engines are available.

`chandra`:
Uses Chandra text and Chandra geometry. Useful when Surya is unavailable or for comparison.

`surya`:
Uses Surya OCR and Surya geometry.

## Troubleshooting

`torch` shows `+cpu`:
Run `.\setup_dual_venv.cmd` again. The current setup script requires CUDA PyTorch and fails if a CUDA build cannot be installed.

`cuda_available: False`:
Check `nvidia-smi` first. If the driver cannot see the GPU, PyTorch cannot use it either.

`pip's dependency resolver does not currently take into account all the packages that are installed`:
This can appear during the forced CUDA PyTorch reinstall. In the Surya venv, PyTorch may temporarily pull `pillow 12.x`, which conflicts with `surya-ocr==0.17.1`; the setup script immediately pins Surya back to `pillow>=10.2,<11.0` after the torch install. Treat the final verification as authoritative: Surya should end on `pillow 10.4.0` or another `<11.0` build, while Chandra can stay on `pillow 12.x`.

`Warning: You are sending unauthenticated requests to the HF Hub`:
This is a Hugging Face rate-limit warning, not a project failure. Chandra weights are large, about 10.6 GB for `datalab-to/chandra-ocr-2`, so unauthenticated downloads can be slow. If needed, set `HF_TOKEN` before running setup to use authenticated Hugging Face requests.

`No module named uniscan`:
Install the package into both venvs:

```powershell
.\.venv_surya\Scripts\python.exe -m pip install -e .
.\.venv_chandra\Scripts\python.exe -m pip install -e .
```

`Chandra cache/weights preflight failed` or `Surya cache/weights preflight failed`:
The project requires local model caches at runtime. Re-run setup to download and verify the weights:

```powershell
.\setup_dual_venv.cmd
```

Expected cache targets are `.hf_cache_chandra` for `datalab-to/chandra-ocr-2` and `.surya_cache` for Surya `text_detection` plus `text_recognition`. If setup cannot download them, fix network/auth/firewall access first; the GUI will not perform lazy model downloads during OCR.

`setup_dual_venv.cmd` cannot find labels such as `ensure_venv`:
Make sure the file has Windows CRLF line endings. A normal git checkout on Windows should handle this.

## Project Layout

1. `src/uniscan/app` - high-level OCR orchestration.
2. `src/uniscan/ocr` - OCR engine adapters, benchmarks, geometry, and searchable PDF assembly.
3. `src/uniscan/ui/basic_ocr_gui.py` - local Tkinter GUI.
4. `src/uniscan/web/service.py` - local HTTP API and web UI.
5. `setup_dual_venv.cmd` - Windows dual-venv bootstrap.
6. `run_basic_gui.cmd` - Windows GUI launcher.
7. `Dockerfile` and `docker-compose.yml` - GPU container runtime.

## Development

Install dev dependencies into a venv and run tests:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

The test suite contains Russian OCR fixture text on purpose. That text is part of OCR behavior coverage, not UI copy.
