<#
.SYNOPSIS
    Bootstrap script for setting up the OCR benchmark environment on a new PC.

.DESCRIPTION
    Checks prerequisites (Python, CUDA, Tesseract) and runs a quick smoke test.
    Run this BEFORE benchmark_ocr_matrix.ps1.

.EXAMPLE
    .\scripts\bootstrap_new_pc.ps1
    .\scripts\bootstrap_new_pc.ps1 -PdfPath "C:\path\to\test.pdf"
#>
[CmdletBinding()]
param(
    [string]$PdfPath = "",
    [string]$BootstrapPython = "py",
    [string]$BootstrapVersion = "3.11"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = if (-not [string]::IsNullOrWhiteSpace($PSCommandPath)) {
    (Resolve-Path (Join-Path (Split-Path -Parent $PSCommandPath) "..")).Path
} else {
    (Get-Location).Path
}

Write-Host "=" * 60
Write-Host "  OCR Benchmark Environment Bootstrap"
Write-Host "=" * 60
Write-Host ""

# --- 1. Python ---
Write-Host "[1/5] Python..." -ForegroundColor Cyan
try {
    if ($BootstrapPython -eq "py" -and -not [string]::IsNullOrWhiteSpace($BootstrapVersion)) {
        $pyVer = & $BootstrapPython "-$BootstrapVersion" --version 2>&1
    } else {
        $pyVer = & $BootstrapPython --version 2>&1
    }
    Write-Host "  OK: $pyVer" -ForegroundColor Green
}
catch {
    Write-Host "  FAIL: Python not found. Install Python 3.11+ and ensure 'py' launcher is available." -ForegroundColor Red
    exit 1
}

# --- 2. NVIDIA / CUDA ---
Write-Host "[2/5] GPU / CUDA..." -ForegroundColor Cyan
try {
    $nvOut = & nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($nvOut)) {
        $parts = $nvOut.Split(",")
        $gpuName = $parts[0].Trim()
        $driverVer = $parts[1].Trim()
        $computeCap = [double]($parts[2].Trim())
        Write-Host "  GPU:     $gpuName" -ForegroundColor Green
        Write-Host "  Driver:  $driverVer" -ForegroundColor Green
        Write-Host "  Compute: $computeCap" -ForegroundColor Green

        if ($computeCap -ge 7.5) {
            Write-Host "  -> PaddlePaddle GPU: SUPPORTED" -ForegroundColor Green
        } else {
            Write-Host "  -> PaddlePaddle GPU: NOT supported (need >= 7.5)" -ForegroundColor Yellow
        }
        if ($computeCap -ge 3.5) {
            Write-Host "  -> PyTorch CUDA:     SUPPORTED" -ForegroundColor Green
        } else {
            Write-Host "  -> PyTorch CUDA:     NOT supported (need >= 3.5)" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  No NVIDIA GPU detected. All engines will run on CPU." -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  nvidia-smi not found. All engines will run on CPU." -ForegroundColor Yellow
}

# --- 3. Tesseract ---
Write-Host "[3/5] Tesseract OCR..." -ForegroundColor Cyan
$tessExe = Get-Command tesseract -ErrorAction SilentlyContinue
if ($null -ne $tessExe) {
    $tessVer = & tesseract --version 2>&1 | Select-Object -First 1
    Write-Host "  OK: $tessVer at $($tessExe.Source)" -ForegroundColor Green
    # Check language packs
    $langs = & tesseract --list-langs 2>&1
    $hasRus = ($langs | Where-Object { $_ -eq "rus" }).Count -gt 0
    $hasEng = ($langs | Where-Object { $_ -eq "eng" }).Count -gt 0
    if ($hasRus -and $hasEng) {
        Write-Host "  Languages: eng + rus available" -ForegroundColor Green
    } else {
        $missing = @()
        if (-not $hasEng) { $missing += "eng" }
        if (-not $hasRus) { $missing += "rus" }
        Write-Host "  WARNING: Missing language packs: $($missing -join ', ')" -ForegroundColor Yellow
        Write-Host "  Download from: https://github.com/tesseract-ocr/tessdata" -ForegroundColor Yellow
    }
} else {
    Write-Host "  NOT FOUND. Tesseract-based engines (pytesseract, ocrmypdf, pymupdf) will fail." -ForegroundColor Yellow
    Write-Host "  Install from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
}

# --- 4. Repo install check ---
Write-Host "[4/5] Repository..." -ForegroundColor Cyan
$pyprojectPath = Join-Path $RepoRoot "pyproject.toml"
if (Test-Path $pyprojectPath) {
    Write-Host "  OK: pyproject.toml found at $RepoRoot" -ForegroundColor Green
} else {
    Write-Host "  FAIL: pyproject.toml not found. Are you in the repo root?" -ForegroundColor Red
    exit 1
}

# --- 5. Test PDF ---
Write-Host "[5/5] Test PDF..." -ForegroundColor Cyan
if (-not [string]::IsNullOrWhiteSpace($PdfPath) -and (Test-Path $PdfPath)) {
    Write-Host "  OK: $PdfPath" -ForegroundColor Green
} elseif (-not [string]::IsNullOrWhiteSpace($PdfPath)) {
    Write-Host "  NOT FOUND: $PdfPath" -ForegroundColor Yellow
} else {
    Write-Host "  No PDF specified. Pass -PdfPath when running benchmark_ocr_matrix.ps1." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" * 60
Write-Host "  Bootstrap complete. Next steps:"
Write-Host "=" * 60
Write-Host ""
Write-Host '  1. Copy your test PDF to this PC'
Write-Host '  2. Run the full benchmark:'
Write-Host ''
Write-Host '     Set-ExecutionPolicy RemoteSigned -Scope Process' -ForegroundColor White
Write-Host ''
Write-Host '     .\scripts\benchmark_ocr_matrix.ps1 `' -ForegroundColor White
Write-Host '       -PdfPath "C:\path\to\your.pdf" `' -ForegroundColor White
Write-Host '       -SampleSize 3 -Dpi 300 `' -ForegroundColor White
Write-Host '       -OutputRoot ".\artifacts\ocr_gpu_full"' -ForegroundColor White
Write-Host ''
Write-Host '  3. Generate comparison report:'
Write-Host ''
Write-Host '     py -3.11 scripts/compare_ocr_results.py --input-root .\artifacts\ocr_gpu_full' -ForegroundColor White
Write-Host ''
