@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_SURYA=.venv_surya"
set "VENV_CHANDRA=.venv_chandra"
set "UV_CACHE_DIR=%CD%\.uv_cache"
set "TMP_BOOT=%CD%\.tmp_bootstrap"

if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%"
if not exist "%TMP_BOOT%" mkdir "%TMP_BOOT%"
set "UV_CACHE_DIR=%UV_CACHE_DIR%"
set "TEMP=%TMP_BOOT%"
set "TMP=%TMP_BOOT%"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

call :ensure_venv "%VENV_SURYA%"
if errorlevel 1 goto :error
call :ensure_venv "%VENV_CHANDRA%"
if errorlevel 1 goto :error

set "PY_SURYA=%CD%\%VENV_SURYA%\Scripts\python.exe"
set "PY_CHANDRA=%CD%\%VENV_CHANDRA%\Scripts\python.exe"

echo [dual-venv] Installing project into SURYA venv ...
"%PY_SURYA%" -m ensurepip --upgrade >nul 2>nul
"%PY_SURYA%" -m pip install --upgrade pip setuptools wheel
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

echo [dual-venv] Installing project into CHANDRA venv ...
"%PY_CHANDRA%" -m ensurepip --upgrade >nul 2>nul
"%PY_CHANDRA%" -m pip install --upgrade pip setuptools wheel
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

:install_gpu_torch
set "GPU_PY=%~1"
set "GPU_NAME=%~2"
echo [dual-venv] Installing torch GPU stack into %GPU_NAME% venv (cu128) ...
where uv >nul 2>nul
if not errorlevel 1 (
  uv pip install --python "%GPU_PY%" --index-url https://download.pytorch.org/whl/cu128 --upgrade --reinstall ^
    "torch==2.11.0+cu128" ^
    "torchvision==0.26.0+cu128" ^
    "torchaudio==2.11.0+cu128"
) else (
  "%GPU_PY%" -m pip install --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall ^
    "torch==2.11.0+cu128" ^
    "torchvision==0.26.0+cu128" ^
    "torchaudio==2.11.0+cu128"
)
if errorlevel 1 (
  echo [dual-venv] ERROR: CUDA torch install failed for %GPU_NAME% venv.
  exit /b 1
)
call :verify_gpu_torch "%GPU_PY%" "%GPU_NAME%"
if errorlevel 1 exit /b 1
exit /b 0

:verify_gpu_torch
set "GPU_PY=%~1"
set "GPU_NAME=%~2"
"%GPU_PY%" -c "import sys, torch; v=getattr(torch, '__version__', ''); print('[dual-venv] %GPU_NAME% torch=' + v); sys.exit(0 if '+cu' in v else 1)"
if errorlevel 1 (
  echo [dual-venv] ERROR: %GPU_NAME% torch is not a CUDA build.
  exit /b 1
)
exit /b 0

:error
echo [dual-venv] Setup failed.
exit /b 1
