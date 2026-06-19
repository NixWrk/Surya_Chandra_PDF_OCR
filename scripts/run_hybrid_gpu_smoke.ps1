param(
    [Parameter(Mandatory = $true)]
    [string]$InputPdf,

    [string]$Pages = "1",

    [string]$Lang = "rus+eng",

    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "outputs\gpu_hybrid_smoke"
}

$resolvedInput = (Resolve-Path -LiteralPath $InputPdf).Path
$pyChandra = Join-Path $RepoRoot ".venv_chandra\Scripts\python.exe"
$pySurya = Join-Path $RepoRoot ".venv_surya\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pyChandra)) {
    throw "Missing Chandra python: $pyChandra. Run setup_dual_venv.cmd first."
}
if (-not (Test-Path -LiteralPath $pySurya)) {
    throw "Missing Surya python: $pySurya. Run setup_dual_venv.cmd first."
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found. UniScan OCR smoke requires CUDA GPU."
}

$env:UNISCAN_CHANDRA_PYTHON = $pyChandra
$env:UNISCAN_SURYA_PYTHON = $pySurya

$env:UNISCAN_CHANDRA_HF_HOME = Join-Path $RepoRoot ".hf_cache_chandra"
$env:UNISCAN_CHANDRA_HUGGINGFACE_HUB_CACHE = Join-Path $env:UNISCAN_CHANDRA_HF_HOME "hub"
$env:UNISCAN_CHANDRA_HF_HUB_CACHE = $env:UNISCAN_CHANDRA_HUGGINGFACE_HUB_CACHE

$env:UNISCAN_SURYA_HF_HOME = Join-Path $RepoRoot ".hf_cache_surya"
$env:UNISCAN_SURYA_HUGGINGFACE_HUB_CACHE = Join-Path $env:UNISCAN_SURYA_HF_HOME "hub"
$env:UNISCAN_SURYA_HF_HUB_CACHE = $env:UNISCAN_SURYA_HUGGINGFACE_HUB_CACHE
$env:UNISCAN_SURYA_MODEL_CACHE_DIR = Join-Path $RepoRoot ".surya_cache"
$env:UNISCAN_SURYA_MODELSCOPE_CACHE = Join-Path $RepoRoot ".modelscope_cache"

$env:HF_HOME = $env:UNISCAN_CHANDRA_HF_HOME
$env:HUGGINGFACE_HUB_CACHE = $env:UNISCAN_CHANDRA_HUGGINGFACE_HUB_CACHE
$env:HF_HUB_CACHE = $env:UNISCAN_CHANDRA_HF_HUB_CACHE
$env:TRANSFORMERS_CACHE = ""
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$env:UNISCAN_CHANDRA_DEVICE_POLICY = "cuda"
$env:TORCH_DEVICE = "cuda:0"
$env:UNISCAN_CHANDRA_TORCH_DEVICE = "cuda:0"
$env:UNISCAN_CHANDRA_PREFER_GPU = "1"
$env:UNISCAN_CHANDRA_REQUIRE_GPU = "1"
$env:UNISCAN_SURYA_TORCH_DEVICE = "cuda:0"
$env:UNISCAN_SURYA_REQUIRE_GPU = "1"
$env:UNISCAN_SURYA_ALLOW_TEXT_FALLBACK = "0"
$env:UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON = "1"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = Join-Path $OutputRoot "run_$timestamp"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$workPdf = Join-Path $runRoot "input.pdf"
Copy-Item -LiteralPath $resolvedInput -Destination $workPdf -Force

Write-Host "[gpu-smoke] CUDA inventory:" -ForegroundColor Cyan
nvidia-smi

$workRoot = Join-Path $runRoot "work"
$args = @(
    "-m", "uniscan",
    "searchable-pdf",
    "--pdf", $workPdf,
    "--mode", "chandra+surya",
    "--lang", $Lang,
    "--work-root", $workRoot,
    "--strict"
)
if (-not [string]::IsNullOrWhiteSpace($Pages)) {
    $args += @("--pages", $Pages)
}

Write-Host "[gpu-smoke] Running chandra+surya on CUDA only..." -ForegroundColor Cyan
& $pyChandra @args
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid GPU smoke failed with exit code $LASTEXITCODE."
}

Write-Host "[gpu-smoke] output_pdf=$workPdf" -ForegroundColor Green
Write-Host "[gpu-smoke] work_root=$workRoot" -ForegroundColor Green
