@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_CHANDRA=%CD%\.venv_chandra"
set "VENV_SURYA=%CD%\.venv_surya"
set "PY_MAIN=%VENV_CHANDRA%\Scripts\python.exe"
set "PY_SURYA=%VENV_SURYA%\Scripts\python.exe"

if not exist "%PY_MAIN%" (
  echo [OCR GUI] Missing %PY_MAIN%
  echo [OCR GUI] Run setup_dual_venv.cmd first.
  exit /b 1
)
if not exist "%PY_SURYA%" (
  echo [OCR GUI] Missing %PY_SURYA%
  echo [OCR GUI] Run setup_dual_venv.cmd first.
  exit /b 1
)

set "UV_CACHE_DIR=%CD%\.uv_cache"
set "TEMP=%CD%\.tmp_bootstrap"
set "TMP=%TEMP%"
if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%"
if not exist "%TEMP%" mkdir "%TEMP%"

set "UNISCAN_CHANDRA_PYTHON=%PY_MAIN%"
set "UNISCAN_SURYA_PYTHON=%PY_SURYA%"

if not defined UNISCAN_CHANDRA_HF_HOME set "UNISCAN_CHANDRA_HF_HOME=%CD%\.hf_cache_chandra"
set "LEGACY_HF_HOME=%CD%\.hf_cache"
set "CHANDRA_CACHE_READY=0"
if exist "%UNISCAN_CHANDRA_HF_HOME%\models--datalab-to--chandra-ocr-2\snapshots\*\model.safetensors.index.json" set "CHANDRA_CACHE_READY=1"
if exist "%UNISCAN_CHANDRA_HF_HOME%\models--datalab-to--chandra-ocr-2\snapshots\*\model.safetensors" set "CHANDRA_CACHE_READY=1"
if exist "%UNISCAN_CHANDRA_HF_HOME%\hub\models--datalab-to--chandra-ocr-2\snapshots\*\model.safetensors.index.json" set "CHANDRA_CACHE_READY=1"
if exist "%UNISCAN_CHANDRA_HF_HOME%\hub\models--datalab-to--chandra-ocr-2\snapshots\*\model.safetensors" set "CHANDRA_CACHE_READY=1"
if "%CHANDRA_CACHE_READY%"=="0" (
  if exist "%LEGACY_HF_HOME%\models--datalab-to--chandra-ocr-2\snapshots\*\model.safetensors.index.json" (
    echo [OCR GUI] Using legacy Chandra cache from .hf_cache
    set "UNISCAN_CHANDRA_HF_HOME=%LEGACY_HF_HOME%"
    set "CHANDRA_CACHE_READY=1"
  )
)
if "%CHANDRA_CACHE_READY%"=="0" (
  if exist "%LEGACY_HF_HOME%\models--datalab-to--chandra-ocr-2\snapshots\*\model.safetensors" (
    echo [OCR GUI] Using legacy Chandra cache from .hf_cache
    set "UNISCAN_CHANDRA_HF_HOME=%LEGACY_HF_HOME%"
    set "CHANDRA_CACHE_READY=1"
  )
)
if "%CHANDRA_CACHE_READY%"=="0" (
  if exist "%LEGACY_HF_HOME%\hub\models--datalab-to--chandra-ocr-2\snapshots\*\model.safetensors.index.json" (
    echo [OCR GUI] Using legacy Chandra hub cache from .hf_cache\hub
    set "UNISCAN_CHANDRA_HF_HOME=%LEGACY_HF_HOME%"
    set "CHANDRA_CACHE_READY=1"
  )
)
if "%CHANDRA_CACHE_READY%"=="0" (
  if exist "%LEGACY_HF_HOME%\hub\models--datalab-to--chandra-ocr-2\snapshots\*\model.safetensors" (
    echo [OCR GUI] Using legacy Chandra hub cache from .hf_cache\hub
    set "UNISCAN_CHANDRA_HF_HOME=%LEGACY_HF_HOME%"
    set "CHANDRA_CACHE_READY=1"
  )
)
set "UNISCAN_CHANDRA_HUGGINGFACE_HUB_CACHE=%UNISCAN_CHANDRA_HF_HOME%\hub"
set "UNISCAN_CHANDRA_HF_HUB_CACHE=%UNISCAN_CHANDRA_HF_HOME%\hub"

set "UNISCAN_SURYA_HF_HOME=%CD%\.hf_cache_surya"
set "UNISCAN_SURYA_HUGGINGFACE_HUB_CACHE=%UNISCAN_SURYA_HF_HOME%\hub"
set "UNISCAN_SURYA_HF_HUB_CACHE=%UNISCAN_SURYA_HF_HOME%\hub"
set "UNISCAN_SURYA_MODEL_CACHE_DIR=%CD%\.surya_cache"
set "UNISCAN_SURYA_MODELSCOPE_CACHE=%CD%\.modelscope_cache"

set "HF_HOME=%UNISCAN_CHANDRA_HF_HOME%"
set "HUGGINGFACE_HUB_CACHE=%UNISCAN_CHANDRA_HUGGINGFACE_HUB_CACHE%"
set "HF_HUB_CACHE=%UNISCAN_CHANDRA_HF_HUB_CACHE%"
set "TRANSFORMERS_CACHE="
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if not defined UNISCAN_CHANDRA_DEVICE_POLICY set "UNISCAN_CHANDRA_DEVICE_POLICY=cuda"
if /I "%UNISCAN_CHANDRA_DEVICE_POLICY%"=="cuda" (
  set "TORCH_DEVICE=cuda:0"
  set "UNISCAN_CHANDRA_TORCH_DEVICE=cuda:0"
  set "UNISCAN_CHANDRA_PREFER_GPU=1"
  set "UNISCAN_CHANDRA_REQUIRE_GPU=1"
) else if /I "%UNISCAN_CHANDRA_DEVICE_POLICY%"=="cpu" (
  echo [OCR GUI] CPU mode is disabled. UniScan OCR runs Chandra/Surya on CUDA only.
  exit /b 1
) else (
  set "TORCH_DEVICE="
  set "UNISCAN_CHANDRA_TORCH_DEVICE="
  set "UNISCAN_CHANDRA_PREFER_GPU=1"
  set "UNISCAN_CHANDRA_REQUIRE_GPU=1"
)
set "UNISCAN_SURYA_TORCH_DEVICE=cuda:0"
set "UNISCAN_SURYA_REQUIRE_GPU=1"
set "UNISCAN_SURYA_ALLOW_TEXT_FALLBACK=0"
set "UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON=1"

if not exist "%UNISCAN_CHANDRA_HF_HOME%" mkdir "%UNISCAN_CHANDRA_HF_HOME%"
if not exist "%UNISCAN_CHANDRA_HUGGINGFACE_HUB_CACHE%" mkdir "%UNISCAN_CHANDRA_HUGGINGFACE_HUB_CACHE%"
if not exist "%UNISCAN_SURYA_HF_HOME%" mkdir "%UNISCAN_SURYA_HF_HOME%"
if not exist "%UNISCAN_SURYA_HUGGINGFACE_HUB_CACHE%" mkdir "%UNISCAN_SURYA_HUGGINGFACE_HUB_CACHE%"
if not exist "%UNISCAN_SURYA_MODEL_CACHE_DIR%" mkdir "%UNISCAN_SURYA_MODEL_CACHE_DIR%"
if not exist "%UNISCAN_SURYA_MODELSCOPE_CACHE%" mkdir "%UNISCAN_SURYA_MODELSCOPE_CACHE%"

set "PATH=%VENV_CHANDRA%\Scripts;%PATH%"

echo [OCR GUI] Launching with dual-venv engine routing ...
"%PY_MAIN%" -m uniscan.ui.basic_ocr_gui
set "APP_EXIT=%ERRORLEVEL%"
exit /b %APP_EXIT%
