@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "LOCAL_TMP=%CD%\.tmp_bootstrap"
set "UV_CACHE_DIR=%CD%\.uv_cache"

if not exist "%LOCAL_TMP%" mkdir "%LOCAL_TMP%"
if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%"
set "TEMP=%LOCAL_TMP%"
set "TMP=%LOCAL_TMP%"

if not exist "%VENV_PY%" (
  echo [UniScan Basic GUI] Creating virtual environment in .venv ...
  set "VENV_CREATED=0"

  where py >nul 2>nul
  if "%ERRORLEVEL%"=="0" (
    py -3.11 -m venv .venv
    if "%ERRORLEVEL%"=="0" set "VENV_CREATED=1"
  )

  if "%VENV_CREATED%"=="0" (
    where python >nul 2>nul
    if "%ERRORLEVEL%"=="0" (
      python -m venv .venv
      if "%ERRORLEVEL%"=="0" set "VENV_CREATED=1"
    )
  )

  if "%VENV_CREATED%"=="0" (
    where python >nul 2>nul
    if "%ERRORLEVEL%"=="0" (
      python -m virtualenv .venv
      if "%ERRORLEVEL%"=="0" set "VENV_CREATED=1"
    )
  )

  if "%VENV_CREATED%"=="0" (
    where uv >nul 2>nul
    if "%ERRORLEVEL%"=="0" (
      uv venv .venv
      if "%ERRORLEVEL%"=="0" set "VENV_CREATED=1"
    )
  )

  if "%VENV_CREATED%"=="0" goto :error
)

if not exist "%VENV_PY%" (
  echo [UniScan Basic GUI] Python in .venv was not created.
  goto :error
)

echo [UniScan Basic GUI] Installing/updating dependencies ...
"%VENV_PY%" -m pip --version >nul 2>nul
if "%ERRORLEVEL%"=="0" (
  "%VENV_PY%" -m pip install -U pip
  if errorlevel 1 goto :error

  "%VENV_PY%" -m pip install -e ".[ocr]"
  if errorlevel 1 goto :error

  "%VENV_PY%" -m pip install -U ^
    "surya-ocr" ^
    "chandra-ocr[hf]" ^
    "requests" ^
    "transformers==4.57.1" ^
    "tokenizers==0.22.1" ^
    "huggingface-hub==0.34.4"
  if errorlevel 1 goto :error
) else (
  where uv >nul 2>nul
  if errorlevel 1 (
    echo [UniScan Basic GUI] pip is missing in .venv and uv is not available.
    goto :error
  )
  uv pip install --python "%VENV_PY%" -e ".[ocr]"
  if errorlevel 1 goto :error
  uv pip install --python "%VENV_PY%" -U ^
    "surya-ocr" ^
    "chandra-ocr[hf]" ^
    "requests" ^
    "transformers==4.57.1" ^
    "tokenizers==0.22.1" ^
    "huggingface-hub==0.34.4"
  if errorlevel 1 goto :error
)

set "PATH=%CD%\.venv\Scripts;%PATH%"

echo [UniScan Basic GUI] Launching ...
"%VENV_PY%" -m uniscan.ui.basic_ocr_gui
set "APP_EXIT=%ERRORLEVEL%"
exit /b %APP_EXIT%

:error
echo [UniScan Basic GUI] Startup failed.
exit /b 1
