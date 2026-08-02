Set-StrictMode -Version Latest

$script:UniscanExpectedGpu0Uuid = "GPU-e6a8c006-5017-6126-01cc-bf9bd972bf4f"

function Assert-UniscanGpu0Contract {
    [CmdletBinding()]
    param(
        [string[]]$AdditionalFields = @()
    )

    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "nvidia-smi was not found; the permitted GPU0 cannot be attested."
    }

    $fields = @("index", "uuid") + $AdditionalFields
    $query = "--query-gpu=$($fields -join ',')"
    $rawRows = @(& nvidia-smi --id=0 $query --format=csv,noheader,nounits 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed while attesting GPU index 0."
    }
    $rows = @($rawRows | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
    if ($rows.Count -ne 1) {
        throw "GPU0 attestation must return exactly one row; got $($rows.Count)."
    }
    $values = @($rows[0].Split(",") | ForEach-Object { $_.Trim() })
    if ($values.Count -ne $fields.Count) {
        throw "GPU0 attestation returned an unexpected field count: $($rows[0])"
    }
    if ($values[0] -ne "0" -or $values[1] -ne $script:UniscanExpectedGpu0Uuid) {
        throw "GPU0 attestation mismatch: $($rows[0])"
    }

    $env:UNISCAN_GPU_DEVICE_ID = $script:UniscanExpectedGpu0Uuid
    $env:CUDA_VISIBLE_DEVICES = "0"

    $result = [ordered]@{}
    for ($index = 0; $index -lt $fields.Count; $index++) {
        $result[$fields[$index]] = $values[$index]
    }
    return [pscustomobject]$result
}
