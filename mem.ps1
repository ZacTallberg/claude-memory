<#
.SYNOPSIS
  `mem` launcher (PowerShell): thin wrapper around `python -m claudemem`.
.DESCRIPTION
  Forwards every argument verbatim to the claudemem CLI, run with the project's
  virtualenv interpreter so all deps resolve. PYTHONUTF8=1 forces UTF-8 stdio so
  transcript content with non-ASCII never trips Windows' default cp1252 codec.
.EXAMPLE
  .\mem.ps1 stats
  .\mem.ps1 query "paradedb hybrid" --rerank
  .\mem.ps1 serve --no-browser
#>
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $py -m claudemem @args
exit $LASTEXITCODE
