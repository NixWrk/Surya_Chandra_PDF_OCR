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

Write-Host "[compare] Running application workflow ..."
$argsList = @(
    "-m", "uniscan", "compare-chandra-geometry",
    "--run-root", $runRootResolved,
    "--pdf-root", $pdfRootResolved,
    "--strict"
)
& $Python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "compare-chandra-geometry failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Geometry comparison outputs:"
$outputBase = Join-Path $runRootResolved "searchable_pdf_geometry_compare"
Write-Host "  A (Chandra geometry): $(Join-Path $outputBase 'chandra_text__chandra_geometry')"
Write-Host "  B (Surya geometry):   $(Join-Path $outputBase 'chandra_text__surya_geometry')"
