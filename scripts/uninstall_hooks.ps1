<#
.SYNOPSIS
  Remove the memory hooks from ~/.claude/settings.json.
.DESCRIPTION
  Runs `mem uninstall-hooks`, which removes only our entries (recall / unify /
  index_trigger) and preserves every other hook and setting.
.EXAMPLE
  .\scripts\uninstall_hooks.ps1
#>
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $py -m claudemem uninstall-hooks
exit $LASTEXITCODE
