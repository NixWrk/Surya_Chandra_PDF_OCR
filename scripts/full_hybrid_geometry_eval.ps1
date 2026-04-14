param(
    [Parameter(Mandatory = $true)]
    [string]$GostPdf,

    [Parameter(Mandatory = $true)]
    [string]$BookPdf,

    [string]$PdfRoot = "D:\\Git_Code\\PDFS",
    [string]$Python = ".\\.venv\\Scripts\\python.exe",
    [string]$OutputRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-CheckedFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [string]$Label = "file"
    )
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "${Label} not found: $PathValue"
    }
    $item = Get-Item -LiteralPath $PathValue
    if (-not $item.PSIsContainer -and $item.Extension -ieq ".pdf") {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }
    throw "${Label} is not a PDF file: $PathValue"
}

function Resolve-CheckedDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [string]$Label = "directory"
    )
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "${Label} not found: $PathValue"
    }
    $item = Get-Item -LiteralPath $PathValue
    if ($item.PSIsContainer) {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }
    throw "${Label} is not a directory: $PathValue"
}

function Invoke-UniScan {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Step,
        [Parameter(Mandatory = $true)]
        [string[]]$Args,
        [Parameter(Mandatory = $true)]
        [string]$LogFile,
        [Parameter(Mandatory = $true)]
        [string]$PythonExe
    )
    Write-Host ""
    Write-Host "[step] $Step" -ForegroundColor Cyan
    Write-Host "cmd: $PythonExe $($Args -join ' ')" -ForegroundColor DarkGray
    $output = & $PythonExe @Args 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE) at step '$Step'."
    }
    return @($output | ForEach-Object { "$_" })
}

function Get-RunDirFromOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$OutputLines,
        [Parameter(Mandatory = $true)]
        [string]$Step
    )
    $line = $OutputLines | Where-Object { $_ -like "run_dir=*" } | Select-Object -Last 1
    if (-not $line) {
        throw "run_dir line not found in output for step '$Step'."
    }
    $runDir = $line.Substring("run_dir=".Length).Trim()
    if (-not (Test-Path -LiteralPath $runDir)) {
        throw "run_dir path does not exist for step '$Step': $runDir"
    }
    return (Resolve-Path -LiteralPath $runDir).Path
}

function Invoke-HybridVariantBuild {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Step,
        [Parameter(Mandatory = $true)]
        [string]$CompareDir,
        [Parameter(Mandatory = $true)]
        [string]$PdfRootDir,
        [Parameter(Mandatory = $true)]
        [string]$OutputDir,
        [Parameter(Mandatory = $true)]
        [string]$SuryaGeometryDir,
        [Parameter(Mandatory = $true)]
        [ValidateSet("auto", "surya_only", "softline")]
        [string]$Policy,
        [double]$BlendWeight = -1,
        [Parameter(Mandatory = $true)]
        [string]$LogFile,
        [Parameter(Mandatory = $true)]
        [string]$PythonExe
    )
    $oldGeom = $env:UNISCAN_CHANDRA_GEOMETRY_DIR
    $hadGeom = Test-Path Env:UNISCAN_CHANDRA_GEOMETRY_DIR
    try {
        $env:UNISCAN_CHANDRA_GEOMETRY_DIR = $SuryaGeometryDir
        $args = @(
            "-m", "uniscan", "build-searchable-from-artifacts",
            "--compare-dir", $CompareDir,
            "--pdf-root", $PdfRootDir,
            "--output", $OutputDir,
            "--engines", "chandra",
            "--strict",
            "--geometry-debug-log",
            "--chandra-geometry-policy", $Policy
        )
        if ($BlendWeight -ge 0) {
            $args += @("--chandra-blend-weight", ([string]$BlendWeight))
        }
        Invoke-UniScan -Step $Step -Args $args -LogFile $LogFile -PythonExe $PythonExe | Out-Null
    }
    finally {
        if ($hadGeom) {
            $env:UNISCAN_CHANDRA_GEOMETRY_DIR = $oldGeom
        }
        else {
            Remove-Item Env:UNISCAN_CHANDRA_GEOMETRY_DIR -ErrorAction SilentlyContinue
        }
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonResolved = Resolve-CheckedFile -PathValue (Join-Path $repoRoot $Python) -Label "Python"
$gostPdfResolved = Resolve-CheckedFile -PathValue $GostPdf -Label "Gost PDF"
$bookPdfResolved = Resolve-CheckedFile -PathValue $BookPdf -Label "Book PDF"
$pdfRootResolved = Resolve-CheckedDir -PathValue $PdfRoot -Label "PDF root"

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $outputRootResolved = Join-Path $repoRoot "outputs\\full_hybrid_eval_$ts"
}
else {
    $outputRootResolved = $OutputRoot
}
New-Item -ItemType Directory -Force -Path $outputRootResolved | Out-Null
$outputRootResolved = (Resolve-Path -LiteralPath $outputRootResolved).Path
$logFile = Join-Path $outputRootResolved "full_run.log"

$docs = @(
    @{ Tag = "gost"; Source = $gostPdfResolved },
    @{ Tag = "book_handwritten"; Source = $bookPdfResolved }
)

foreach ($doc in $docs) {
    $tag = [string]$doc.Tag
    $sourcePdf = [string]$doc.Source
    $docOut = Join-Path $outputRootResolved $tag
    New-Item -ItemType Directory -Force -Path $docOut | Out-Null

    $suryaNativePdf = Join-Path $docOut "${tag}__surya_native.pdf"
    $chandraNativePdf = Join-Path $docOut "${tag}__chandra_native.pdf"
    Copy-Item -LiteralPath $sourcePdf -Destination $suryaNativePdf -Force
    Copy-Item -LiteralPath $sourcePdf -Destination $chandraNativePdf -Force

    $suryaOut = Invoke-UniScan -Step "$tag / surya native" -Args @(
        "-m", "uniscan", "searchable-pdf",
        "--pdf", $suryaNativePdf,
        "--mode", "surya",
        "--strict"
    ) -LogFile $logFile -PythonExe $pythonResolved
    $suryaRunDir = Get-RunDirFromOutput -OutputLines $suryaOut -Step "$tag / surya native"

    $chandraOut = Invoke-UniScan -Step "$tag / chandra native" -Args @(
        "-m", "uniscan", "searchable-pdf",
        "--pdf", $chandraNativePdf,
        "--mode", "chandra",
        "--strict"
    ) -LogFile $logFile -PythonExe $pythonResolved
    $chandraRunDir = Get-RunDirFromOutput -OutputLines $chandraOut -Step "$tag / chandra native"

    $suryaGeometryDir = Join-Path $suryaRunDir "surya"
    if (-not (Test-Path -LiteralPath $suryaGeometryDir)) {
        throw "Surya geometry directory not found: $suryaGeometryDir"
    }
    $compareDir = Join-Path $chandraRunDir "_compare_txt"
    if (-not (Test-Path -LiteralPath $compareDir)) {
        throw "Chandra compare dir not found: $compareDir"
    }

    $chandraTxt = Get-ChildItem -LiteralPath $compareDir -File | Where-Object {
        $_.Name -like "*__chandra.txt"
    } | Select-Object -First 1
    if (-not $chandraTxt) {
        throw "No '__chandra.txt' found in compare dir: $compareDir"
    }

    $suryaOnlyWorkspace = Join-Path $docOut "_workspace_surya_only"
    $suryaOnlyCompare = Join-Path $suryaOnlyWorkspace "_compare_txt"
    New-Item -ItemType Directory -Force -Path $suryaOnlyCompare | Out-Null
    Copy-Item -LiteralPath $chandraTxt.FullName -Destination (Join-Path $suryaOnlyCompare $chandraTxt.Name) -Force

    Invoke-HybridVariantBuild `
        -Step "$tag / hybrid A (chandra text + surya geometry)" `
        -CompareDir $suryaOnlyCompare `
        -PdfRootDir $pdfRootResolved `
        -OutputDir (Join-Path $docOut "hybrid_a_chandra_text_surya_geometry") `
        -SuryaGeometryDir $suryaGeometryDir `
        -Policy "surya_only" `
        -LogFile $logFile `
        -PythonExe $pythonResolved

    Invoke-HybridVariantBuild `
        -Step "$tag / hybrid B (auto per page)" `
        -CompareDir $compareDir `
        -PdfRootDir $pdfRootResolved `
        -OutputDir (Join-Path $docOut "hybrid_b_auto_per_page") `
        -SuryaGeometryDir $suryaGeometryDir `
        -Policy "auto" `
        -LogFile $logFile `
        -PythonExe $pythonResolved

    Invoke-HybridVariantBuild `
        -Step "$tag / hybrid C (softline blend)" `
        -CompareDir $compareDir `
        -PdfRootDir $pdfRootResolved `
        -OutputDir (Join-Path $docOut "hybrid_c_softline") `
        -SuryaGeometryDir $suryaGeometryDir `
        -Policy "softline" `
        -BlendWeight 0.75 `
        -LogFile $logFile `
        -PythonExe $pythonResolved
}

Write-Host ""
Write-Host "Done. Root: $outputRootResolved" -ForegroundColor Green
Write-Host "Log : $logFile" -ForegroundColor Green
Write-Host ""
Write-Host "Generated PDFs:"
Get-ChildItem -LiteralPath $outputRootResolved -Recurse -Filter "*.pdf" -File | Select-Object FullName,Length,LastWriteTime
Write-Host ""
Write-Host "Geometry logs:"
Get-ChildItem -LiteralPath $outputRootResolved -Recurse -Filter "*_geometry_log.json" -File | Select-Object FullName,Length,LastWriteTime
