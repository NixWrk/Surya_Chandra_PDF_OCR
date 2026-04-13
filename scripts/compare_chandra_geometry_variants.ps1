param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,

    [Parameter(Mandatory = $true)]
    [string]$PdfRoot,

    [string]$Python = ".\\.venv\\Scripts\\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-CheckedPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [string]$Label = "path"
    )
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "${Label} not found: $PathValue"
    }
    return (Resolve-Path -LiteralPath $PathValue).Path
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runRootResolved = Resolve-CheckedPath -PathValue $RunRoot -Label "Run root"
$pdfRootResolved = Resolve-CheckedPath -PathValue $PdfRoot -Label "PDF root"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}

$compareDir = Join-Path $runRootResolved "_compare_txt"
$compareArgs = @(
    "-m", "uniscan", "prepare-compare-txt",
    "--benchmark-root", $runRootResolved,
    "--output", $compareDir,
    "--engines", "chandra", "surya",
    "--strict"
)

Write-Host "[compare] Preparing TXT artifacts ..."
& $Python @compareArgs
if ($LASTEXITCODE -ne 0) {
    throw "prepare-compare-txt failed with exit code $LASTEXITCODE"
}

$outputBase = Join-Path $runRootResolved "searchable_pdf_geometry_compare"
$outChandraGeometry = Join-Path $outputBase "chandra_text__chandra_geometry"
$outSuryaGeometry = Join-Path $outputBase "chandra_text__surya_geometry"

foreach ($target in @($outChandraGeometry, $outSuryaGeometry)) {
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

Write-Host "[compare] Building variant A: chandra text + chandra geometry ..."
$buildA = @(
    "-m", "uniscan", "build-searchable-from-artifacts",
    "--compare-dir", $compareDir,
    "--pdf-root", $pdfRootResolved,
    "--output", $outChandraGeometry,
    "--engines", "chandra",
    "--strict"
)
& $Python @buildA
if ($LASTEXITCODE -ne 0) {
    throw "build-searchable-from-artifacts (chandra geometry) failed with exit code $LASTEXITCODE"
}

Write-Host "[compare] Building variant B: chandra text + surya geometry ..."
$env:UNISCAN_CHANDRA_GEOMETRY_DIR = (Join-Path $runRootResolved "surya")
$buildB = @(
    "-m", "uniscan", "build-searchable-from-artifacts",
    "--compare-dir", $compareDir,
    "--pdf-root", $pdfRootResolved,
    "--output", $outSuryaGeometry,
    "--engines", "chandra",
    "--strict"
)
& $Python @buildB
if ($LASTEXITCODE -ne 0) {
    throw "build-searchable-from-artifacts (surya geometry) failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Geometry comparison outputs:"
Write-Host "  A (Chandra geometry): $outChandraGeometry"
Write-Host "  B (Surya geometry):   $outSuryaGeometry"
