<#
.SYNOPSIS
    Read-only deployment preflight for the production hybrid OCR runtime.

.DESCRIPTION
    Verifies repository files, the configured physical GPU0 contract, and
    either the installed Windows dual-venv runtime or Docker Compose runtime.
    It does not install packages, download models, create directories, or run OCR.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Windows", "Docker")]
    [string]$Target,

    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$checks = [Collections.Generic.List[object]]::new()
$gpuSummary = $null
$gpuReady = $false

function Add-PreflightCheck {
    param(
        [Parameter(Mandatory = $true)] [string]$Name,
        [Parameter(Mandatory = $true)] [bool]$Ok,
        [Parameter(Mandatory = $true)] [string]$Detail
    )
    [void]$checks.Add([pscustomobject][ordered]@{ name = $Name; ok = $Ok; detail = $Detail })
}

function Test-RequiredFile {
    param([Parameter(Mandatory = $true)] [string]$RelativePath)
    $path = Join-Path $RepoRoot $RelativePath
    $exists = Test-Path -LiteralPath $path -PathType Leaf
    Add-PreflightCheck -Name "file:$RelativePath" -Ok $exists -Detail $(
        if ($exists) { $path } else { "Missing required file: $path" }
    )
    return $exists
}

function Format-CommandFailure {
    param([object[]]$Output, [int]$ExitCode)
    $text = (($Output | ForEach-Object { "$_" }) -join "`n").Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return "Command failed with exit code $ExitCode."
    }
    return "Command failed with exit code ${ExitCode}: $text"
}

foreach ($requiredFile in @("pyproject.toml", "scripts\gpu0_contract.ps1")) {
    Test-RequiredFile -RelativePath $requiredFile | Out-Null
}

$gpuHelper = Join-Path $PSScriptRoot "gpu0_contract.ps1"
if (Test-Path -LiteralPath $gpuHelper -PathType Leaf) {
    try {
        . $gpuHelper
        $gpu = Assert-UniscanGpu0Contract -AdditionalFields @("name", "driver_version", "compute_cap")
        if ([string]::IsNullOrWhiteSpace("$($gpu.compute_cap)")) {
            throw "GPU0 compute capability was not reported."
        }
        $gpuSummary = [pscustomobject][ordered]@{
            index = "$($gpu.index)"
            uuid = "$($gpu.uuid)"
            name = "$($gpu.name)"
            driver_version = "$($gpu.driver_version)"
            compute_capability = "$($gpu.compute_cap)"
        }
        $gpuReady = $true
        Add-PreflightCheck -Name "gpu0.contract" -Ok $true -Detail (
            "index=$($gpu.index) uuid=$($gpu.uuid) name=$($gpu.name) " +
            "driver=$($gpu.driver_version) compute_cap=$($gpu.compute_cap)"
        )
    }
    catch {
        Add-PreflightCheck -Name "gpu0.contract" -Ok $false -Detail $_.Exception.Message
    }
}

if ($Target -eq "Windows") {
    $windowsEnvironments = @(
        [pscustomobject]@{ name = "chandra"; module = "chandra"; python = Join-Path $RepoRoot ".venv_chandra\Scripts\python.exe" },
        [pscustomobject]@{ name = "surya"; module = "surya"; python = Join-Path $RepoRoot ".venv_surya\Scripts\python.exe" }
    )
    foreach ($environment in $windowsEnvironments) {
        $pythonExists = Test-Path -LiteralPath $environment.python -PathType Leaf
        Add-PreflightCheck -Name "windows.$($environment.name).python" -Ok $pythonExists -Detail $(
            if ($pythonExists) { $environment.python } else { "Missing Python executable: $($environment.python)" }
        )
        if (-not $pythonExists) { continue }
        if (-not $gpuReady) {
            Add-PreflightCheck -Name "windows.$($environment.name).cuda" -Ok $false -Detail "Skipped because the configured GPU0 contract failed."
            continue
        }
        $module = $environment.module
        $probe = @"
import json
import sys
import torch
import uniscan
import $module

ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
value = float(torch.ones((1,), device="cuda:0").item()) if ok else None
payload = {
    "torch": str(torch.__version__),
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "cuda_device_0": torch.cuda.get_device_name(0) if ok else None,
    "tensor_probe": value,
}
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
sys.exit(0 if ok and value == 1.0 else 1)
"@
        $probeOutput = @(& $environment.python -B -c $probe 2>&1)
        $probeExitCode = $LASTEXITCODE
        Add-PreflightCheck -Name "windows.$($environment.name).cuda" -Ok ($probeExitCode -eq 0) -Detail $(
            if ($probeExitCode -eq 0) {
                (($probeOutput | ForEach-Object { "$_" }) -join "`n").Trim()
            } else {
                Format-CommandFailure -Output $probeOutput -ExitCode $probeExitCode
            }
        )
    }
}
else {
    foreach ($requiredFile in @("docker-compose.yml", "Dockerfile", "scripts\docker-entrypoint.sh")) {
        Test-RequiredFile -RelativePath $requiredFile | Out-Null
    }
    $dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
    Add-PreflightCheck -Name "docker.tool" -Ok ($null -ne $dockerCommand) -Detail $(
        if ($null -ne $dockerCommand) { $dockerCommand.Source } else { "docker was not found in PATH." }
    )
    if ($null -ne $dockerCommand) {
        $versionOutput = @(& docker version --format "{{.Server.Version}}" 2>&1)
        $versionExitCode = $LASTEXITCODE
        Add-PreflightCheck -Name "docker.daemon" -Ok ($versionExitCode -eq 0) -Detail $(
            if ($versionExitCode -eq 0) { "Docker server version: $(($versionOutput -join '').Trim())" }
            else { Format-CommandFailure -Output $versionOutput -ExitCode $versionExitCode }
        )
        $composePath = Join-Path $RepoRoot "docker-compose.yml"
        $configOutput = @(& docker compose --project-directory $RepoRoot -f $composePath config 2>&1)
        $configExitCode = $LASTEXITCODE
        Add-PreflightCheck -Name "docker.compose.config" -Ok ($configExitCode -eq 0) -Detail $(
            if ($configExitCode -eq 0) { "Compose configuration resolved successfully." }
            else { Format-CommandFailure -Output $configOutput -ExitCode $configExitCode }
        )
        $servicesOutput = @(& docker compose --project-directory $RepoRoot -f $composePath config --services 2>&1)
        $servicesExitCode = $LASTEXITCODE
        $services = @($servicesOutput | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
        $serviceOk = $servicesExitCode -eq 0 -and $services -contains "ocr-api"
        Add-PreflightCheck -Name "docker.compose.service" -Ok $serviceOk -Detail $(
            if ($serviceOk) { "ocr-api" }
            elseif ($servicesExitCode -ne 0) { Format-CommandFailure -Output $servicesOutput -ExitCode $servicesExitCode }
            else { "Compose services did not include ocr-api: $($services -join ', ')" }
        )
        if ($versionExitCode -eq 0) {
            $networkOutput = @(& docker network inspect zotero-automation 2>&1)
            $networkExitCode = $LASTEXITCODE
            Add-PreflightCheck -Name "docker.network.zotero-automation" -Ok ($networkExitCode -eq 0) -Detail $(
                if ($networkExitCode -eq 0) { "External network zotero-automation exists." }
                else { Format-CommandFailure -Output $networkOutput -ExitCode $networkExitCode }
            )
        } else {
            Add-PreflightCheck -Name "docker.network.zotero-automation" -Ok $false -Detail "Skipped because the Docker daemon check failed."
        }
    }
}

$failedChecks = @($checks | Where-Object { -not $_.ok })
$result = [pscustomobject][ordered]@{
    target = $Target
    ok = $failedChecks.Count -eq 0
    repository = $RepoRoot
    gpu0 = $gpuSummary
    checks = @($checks)
}

if ($Json) { $result | ConvertTo-Json -Depth 6 }
else {
    foreach ($check in $checks) {
        $label = if ($check.ok) { "PASS" } else { "FAIL" }
        Write-Output "[$label] $($check.name): $($check.detail)"
    }
    Write-Output ""
    Write-Output $(if ($result.ok) { "Preflight passed." } else { "Preflight failed." })
}

if (-not $result.ok) { exit 1 }
