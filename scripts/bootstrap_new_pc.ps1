<#
.SYNOPSIS
    Bootstrap diagnostics for the production hybrid OCR environment on a new PC.

.DESCRIPTION
    Checks Python, CUDA, the repository, and a PDF fixture before a hybrid smoke test.
    Production OCR uses Chandra text plus Surya geometry and does not require Tesseract.

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
        Write-Host "  No NVIDIA GPU detected. UniScan OCR requires CUDA GPU for Chandra/Surya." -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  nvidia-smi not found. UniScan OCR requires CUDA GPU for Chandra/Surya." -ForegroundColor Yellow
}

# --- 3. Production OCR contract ---
Write-Host "[3/5] Production OCR mode..." -ForegroundColor Cyan
Write-Host "  chandra+surya only: Chandra text + Surya geometry." -ForegroundColor Green
Write-Host "  Tesseract and single-engine searchable-PDF fallbacks are not required." -ForegroundColor Green

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
