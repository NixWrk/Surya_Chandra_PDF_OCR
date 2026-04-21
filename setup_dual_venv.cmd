@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_SURYA=.venv_surya"
set "VENV_CHANDRA=.venv_chandra"
set "UV_CACHE_DIR=%CD%\.uv_cache"
set "TMP_BOOT=%CD%\.tmp_bootstrap"
set "CHANDRA_HF_HOME=%CD%\.hf_cache_chandra"
set "CHANDRA_HF_HUB_CACHE=%CHANDRA_HF_HOME%\hub"
set "SURYA_HF_HOME=%CD%\.hf_cache_surya"
set "SURYA_HF_HUB_CACHE=%SURYA_HF_HOME%\hub"
set "SURYA_MODEL_CACHE_DIR=%CD%\.surya_cache"
set "SURYA_MODELSCOPE_CACHE=%CD%\.modelscope_cache"
set "SETUPTOOLS_VERSION=70.2.0"

if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%"
if not exist "%TMP_BOOT%" mkdir "%TMP_BOOT%"
if not exist "%CHANDRA_HF_HOME%" mkdir "%CHANDRA_HF_HOME%"
if not exist "%CHANDRA_HF_HUB_CACHE%" mkdir "%CHANDRA_HF_HUB_CACHE%"
if not exist "%SURYA_HF_HOME%" mkdir "%SURYA_HF_HOME%"
if not exist "%SURYA_HF_HUB_CACHE%" mkdir "%SURYA_HF_HUB_CACHE%"
if not exist "%SURYA_MODEL_CACHE_DIR%" mkdir "%SURYA_MODEL_CACHE_DIR%"
if not exist "%SURYA_MODELSCOPE_CACHE%" mkdir "%SURYA_MODELSCOPE_CACHE%"
set "UV_CACHE_DIR=%UV_CACHE_DIR%"
set "TEMP=%TMP_BOOT%"
set "TMP=%TMP_BOOT%"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

call :ensure_venv "%VENV_SURYA%"
if errorlevel 1 goto :error
call :ensure_venv "%VENV_CHANDRA%"
if errorlevel 1 goto :error

set "PY_SURYA=%CD%\%VENV_SURYA%\Scripts\python.exe"
set "PY_CHANDRA=%CD%\%VENV_CHANDRA%\Scripts\python.exe"

call :clean_invalid_distributions "%PY_SURYA%" "SURYA"
if errorlevel 1 goto :error
call :clean_invalid_distributions "%PY_CHANDRA%" "CHANDRA"
if errorlevel 1 goto :error

echo [dual-venv] Installing project into SURYA venv ...
"%PY_SURYA%" -m ensurepip --upgrade >nul 2>nul
"%PY_SURYA%" -m pip install --upgrade pip wheel
if errorlevel 1 goto :error
"%PY_SURYA%" -m pip install "setuptools==%SETUPTOOLS_VERSION%"
if errorlevel 1 goto :error
"%PY_SURYA%" -m pip install -e "."
if errorlevel 1 goto :error
"%PY_SURYA%" -m pip install ^
  "pypdf>=4.0" ^
  "reportlab>=4.0" ^
  "surya-ocr==0.17.1" ^
  "transformers==4.57.1" ^
  "huggingface-hub>=0.34,<1.0" ^
  "tokenizers>=0.22,<0.23" ^
  "pypdfium2==4.30.0"
if errorlevel 1 goto :error

call :install_gpu_torch "%PY_SURYA%" "SURYA"
if errorlevel 1 goto :error
"%PY_SURYA%" -m pip install --upgrade "pillow>=10.2,<11.0"
if errorlevel 1 goto :error
call :warm_surya_cache
if errorlevel 1 goto :error

echo [dual-venv] Installing project into CHANDRA venv ...
"%PY_CHANDRA%" -m ensurepip --upgrade >nul 2>nul
"%PY_CHANDRA%" -m pip install --upgrade pip wheel
if errorlevel 1 goto :error
"%PY_CHANDRA%" -m pip install "setuptools==%SETUPTOOLS_VERSION%"
if errorlevel 1 goto :error
"%PY_CHANDRA%" -m pip install -e "."
if errorlevel 1 goto :error
"%PY_CHANDRA%" -m pip install ^
  "pypdf>=4.0" ^
  "reportlab>=4.0" ^
  "chandra-ocr[hf]==0.2.0" ^
  "pypdfium2==4.30.0"
if errorlevel 1 goto :error

call :install_gpu_torch "%PY_CHANDRA%" "CHANDRA"
if errorlevel 1 goto :error
call :warm_chandra_cache
if errorlevel 1 goto :error

echo [dual-venv] Done.
echo [dual-venv] SURYA  python: %PY_SURYA%
echo [dual-venv] CHANDRA python: %PY_CHANDRA%
echo [dual-venv] Run: .\run_basic_gui.cmd
exit /b 0

:ensure_venv
if exist "%~1\Scripts\python.exe" exit /b 0
echo [dual-venv] Creating %~1 ...
where py >nul 2>nul
if not errorlevel 1 (
  py -3.11 -m venv "%~1"
  if not errorlevel 1 exit /b 0
  py -3.12 -m venv "%~1"
  if not errorlevel 1 exit /b 0
)
where python >nul 2>nul
if errorlevel 1 (
  echo [dual-venv] No Python launcher found.
  exit /b 1
)
python -m venv "%~1"
if errorlevel 1 exit /b 1
exit /b 0

:clean_invalid_distributions
set "CLEAN_PY=%~1"
set "CLEAN_NAME=%~2"
echo [dual-venv] Cleaning invalid pip leftovers in %CLEAN_NAME% venv ...
"%CLEAN_PY%" -c "import shutil, site; from pathlib import Path; roots=[Path(p) for p in site.getsitepackages()]; leftovers=[p for r in roots for p in r.glob('~*')]; [shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True) for p in leftovers]; print('removed invalid distribution leftovers:', len(leftovers))"
if errorlevel 1 (
  echo [dual-venv] ERROR: Failed to clean invalid pip leftovers in %CLEAN_NAME% venv.
  exit /b 1
)
exit /b 0

:install_gpu_torch
set "GPU_PY=%~1"
set "GPU_NAME=%~2"
call :verify_gpu_torch_stack "%GPU_PY%" "%GPU_NAME%" >nul 2>nul
if not errorlevel 1 (
  echo [dual-venv] %GPU_NAME% CUDA torch stack already OK; skipping reinstall.
  exit /b 0
)
echo [dual-venv] Installing torch GPU stack into %GPU_NAME% venv (cu128) ...
where uv >nul 2>nul
if not errorlevel 1 (
  uv pip install --python "%GPU_PY%" --index-url https://download.pytorch.org/whl/cu128 --upgrade --reinstall "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" "torchaudio==2.11.0+cu128"
) else (
  "%GPU_PY%" -m pip install --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" "torchaudio==2.11.0+cu128"
)
if errorlevel 1 (
  echo [dual-venv] ERROR: CUDA torch install failed for %GPU_NAME% venv.
  exit /b 1
)
call :verify_gpu_torch "%GPU_PY%" "%GPU_NAME%"
if errorlevel 1 exit /b 1
exit /b 0

:verify_gpu_torch_stack
set "GPU_PY=%~1"
set "GPU_NAME=%~2"
"%GPU_PY%" -c "import sys, importlib.metadata as m; exp=dict(torch='2.11.0+cu128', torchvision='0.26.0+cu128', torchaudio='2.11.0+cu128'); bad=[k+'='+m.version(k) for k,v in exp.items() if m.version(k)!=v]; sys.exit(0 if not bad else 1)"
exit /b %ERRORLEVEL%

:verify_gpu_torch
set "GPU_PY=%~1"
set "GPU_NAME=%~2"
"%GPU_PY%" -c "import sys, torch; v=getattr(torch, '__version__', ''); print(v); sys.exit(0 if '+cu' in v else 1)"
if errorlevel 1 (
  echo [dual-venv] ERROR: %GPU_NAME% torch is not a CUDA build.
  exit /b 1
)
exit /b 0

:warm_surya_cache
echo [dual-venv] Downloading and verifying Surya model caches ...
set "MODEL_CACHE_DIR=%SURYA_MODEL_CACHE_DIR%"
set "HF_HOME=%SURYA_HF_HOME%"
set "HUGGINGFACE_HUB_CACHE=%SURYA_HF_HUB_CACHE%"
set "HF_HUB_CACHE=%SURYA_HF_HUB_CACHE%"
set "MODELSCOPE_CACHE=%SURYA_MODELSCOPE_CACHE%"
"%PY_SURYA%" -c "from surya.settings import settings; from surya.common.s3 import S3DownloaderMixin, download_directory; checkpoints=(settings.DETECTOR_MODEL_CHECKPOINT, settings.RECOGNITION_MODEL_CHECKPOINT); [download_directory(cp.replace('s3://', ''), S3DownloaderMixin.get_local_path(cp)) for cp in checkpoints]"
if errorlevel 1 (
  echo [dual-venv] ERROR: Surya model cache download failed.
  exit /b 1
)
"%PY_SURYA%" -c "from uniscan.ocr.benchmark import _ensure_surya_cache_ready; _ensure_surya_cache_ready(); print('Surya cache ready')"
if errorlevel 1 (
  echo [dual-venv] ERROR: Surya model cache verification failed.
  exit /b 1
)
exit /b 0

:warm_chandra_cache
echo [dual-venv] Downloading and verifying Chandra model cache ...
set "HF_HOME=%CHANDRA_HF_HOME%"
set "HUGGINGFACE_HUB_CACHE=%CHANDRA_HF_HUB_CACHE%"
set "HF_HUB_CACHE=%CHANDRA_HF_HUB_CACHE%"
set "TRANSFORMERS_CACHE="
"%PY_CHANDRA%" -c "from huggingface_hub import snapshot_download; snapshot_download('datalab-to/chandra-ocr-2')"
if errorlevel 1 (
  echo [dual-venv] ERROR: Chandra model cache download failed.
  exit /b 1
)
"%PY_CHANDRA%" -c "from uniscan.ocr.benchmark import _ensure_chandra_cache_ready; _ensure_chandra_cache_ready(); print('Chandra cache ready')"
if errorlevel 1 (
  echo [dual-venv] ERROR: Chandra model cache verification failed.
  exit /b 1
)
exit /b 0

:error
echo [dual-venv] Setup failed.
exit /b 1
