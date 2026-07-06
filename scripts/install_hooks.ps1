<#
.SYNOPSIS
  Wire the memory hooks into ~/.claude/settings.json (idempotent).
.DESCRIPTION
  Runs `mem install-hooks`, which merges our UserPromptSubmit (recall),
  SessionStart (unify), and SessionEnd/PreCompact (background index) hooks into
  Claude Code's settings.json without clobbering existing entries. Re-running is
  safe; it never duplicates our entries.
.EXAMPLE
  .\scripts\install_hooks.ps1
#>
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $py -m claudemem install-hooks
exit $LASTEXITCODE
