[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [switch]$RefreshNixCopies
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
    $sourcePath = Join-Path $pluginRoot $name
    if (!(Test-Path $sourcePath)) {
        continue
    }

    $nixPath = Join-Path $pluginRoot ("{0}_NIX" -f $name)
    if ($RefreshNixCopies -and (Test-Path $nixPath)) {
        Write-Host "Refreshing NIX copy: $nixPath"
        Remove-Item -Recurse -Force $nixPath
    }
    if (!(Test-Path $nixPath)) {
        Write-Host "Creating NIX copy: $sourcePath -> $nixPath"
        Copy-Item -Path $sourcePath -Destination $nixPath -Recurse -Force
    }
    else {
        Write-Host "Using existing NIX copy: $nixPath"
    }

    Write-Host "Installing plugin from NIX copy $nixPath"
    & $pythonPath -m pip install -e $nixPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install plugin from NIX copy: $nixPath"
    }
    $installed += ("{0}_NIX" -f $name)
}

if ($installed.Count -eq 0) {
    Write-Host "No known plugin repos found under $pluginRoot"
    exit 0
}

Write-Host ""
Write-Host "Installed plugin repos:"
$installed | ForEach-Object { Write-Host " - $_" }
