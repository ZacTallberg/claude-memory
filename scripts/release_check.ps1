param(
    [switch]$SkipLiveChecks
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing virtual environment interpreter: $python"
}

Push-Location $repoRoot
try {
    & $python -m compileall -q claudemem hooks
    if ($LASTEXITCODE -ne 0) { throw "source compilation failed" }

    & $python -m claudemem selftest
    if ($LASTEXITCODE -ne 0) { throw "self-test failed" }

    if (-not $SkipLiveChecks) {
        & $python -m claudemem eval
        if ($LASTEXITCODE -ne 0) { throw "retrieval evaluation failed" }
        & $python -m claudemem integrations
        if ($LASTEXITCODE -ne 0) { throw "integration census failed" }
        & $python -m claudemem delivery-check --load
        if ($LASTEXITCODE -ne 0) { throw "delivery SLO check failed" }
    }
}
finally {
    Pop-Location
}
