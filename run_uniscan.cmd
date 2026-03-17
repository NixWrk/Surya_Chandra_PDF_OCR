@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [UniScan] Creating virtual environment...
  where py >nul 2>nul
  if "%ERRORLEVEL%"=="0" (
    py -3.11 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto :error
)

echo [UniScan] Installing/updating dependencies...
"%VENV_PY%" -m pip install -U pip
if errorlevel 1 goto :error

"%VENV_PY%" -m pip install -e .
if errorlevel 1 goto :error

echo [UniScan] Launching application...
"%VENV_PY%" -m uniscan.cli
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
  echo [UniScan] Application exited with code %APP_EXIT%.
)

exit /b %APP_EXIT%

:error
echo [UniScan] Startup failed.
exit /b 1
