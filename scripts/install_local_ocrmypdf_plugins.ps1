[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    if (-not [string]::IsNullOrWhiteSpace($PSCommandPath)) {
        $scriptDir = Split-Path -Parent $PSCommandPath
        $RepoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
    }
    else {
        $RepoRoot = (Get-Location).Path
    }
}

$pythonPath = $PythonExe
if (-not [System.IO.Path]::IsPathRooted($pythonPath)) {
    $pythonPath = Join-Path $RepoRoot $pythonPath
}
if (!(Test-Path $pythonPath)) {
    throw "Python executable not found: $pythonPath"
}

$pluginRoot = Join-Path $RepoRoot "OCRmypdf_plugins"
if (!(Test-Path $pluginRoot)) {
    throw "Plugin folder not found: $pluginRoot"
}

$candidates = @(
    "ocrmypdf-paddleocr-master",
    "OCRmyPDF-PaddleOCR-main",
    "OCRmyPDF-EasyOCR-main",
    "ocrmypdf-doctr-master",
    "OCRmyPDF-AppleOCR-main"
)

$installed = @()
foreach ($name in $candidates) {
    $path = Join-Path $pluginRoot $name
    if (!(Test-Path $path)) {
        continue
    }
    Write-Host "Installing plugin from $path"
    & $pythonPath -m pip install -e $path
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install plugin: $path"
    }
    $installed += $name
}

if ($installed.Count -eq 0) {
    Write-Host "No known plugin repos found under $pluginRoot"
    exit 0
}

Write-Host ""
Write-Host "Installed plugin repos:"
$installed | ForEach-Object { Write-Host " - $_" }

