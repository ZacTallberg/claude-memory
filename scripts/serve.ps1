<#
.SYNOPSIS
  Start the claude-memory dashboard (warm-server hot path + hub UI).
.DESCRIPTION
  Runs `mem serve`, which launches uvicorn on the configured [server] host/port
  (default http://127.0.0.1:7777) and warms the embedder + store in a background
  thread. Pass -NoBrowser to suppress auto-opening the browser (used by the
  persistence Scheduled Task). Extra args are forwarded to `mem serve`.
.EXAMPLE
  .\scripts\serve.ps1
  .\scripts\serve.ps1 -NoBrowser
  .\scripts\serve.ps1 -Port 7788
#>
param(
  [int]$Port,
  [switch]$NoBrowser,
  [Parameter(ValueFromRemainingArguments = $true)] $Rest
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$serveArgs = @("serve")
if ($PSBoundParameters.ContainsKey("Port")) { $serveArgs += @("--port", "$Port") }
if ($NoBrowser) { $serveArgs += "--no-browser" }
if ($Rest) { $serveArgs += $Rest }

& $py -m claudemem @serveArgs
exit $LASTEXITCODE
