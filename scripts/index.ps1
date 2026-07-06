<#
.SYNOPSIS
  Incrementally index transcripts + curated notes into the store.
.DESCRIPTION
  Runs `mem index`. By default this is incremental (tail-reads only new transcript
  bytes via the persisted byte offset). Pass -Full to force a complete reindex
  (e.g. after the embedding model/dim changed). Extra args are forwarded.
.EXAMPLE
  .\scripts\index.ps1
  .\scripts\index.ps1 -Full
#>
param(
  [switch]$Full,
  [Parameter(ValueFromRemainingArguments = $true)] $Rest
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$indexArgs = @("index")
if ($Full) { $indexArgs += "--full" }
if ($Rest) { $indexArgs += $Rest }

& $py -m claudemem @indexArgs
exit $LASTEXITCODE
