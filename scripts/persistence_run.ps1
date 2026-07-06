<#
.SYNOPSIS
  Keep the claude-memory stack alive. This is the body the Scheduled Task runs at logon.
.DESCRIPTION
  WHY THIS EXISTS:
    ParadeDB runs as a Docker container inside the WSL2 "Ubuntu" VM. WSL2 shuts the VM
    down a few seconds after its last process exits (the WSL "VM idle timeout"). When the
    VM dies, Docker and the ParadeDB container die with it, and the localhost:55432 port
    the store talks to disappears. So we must keep at least one long-lived process running
    inside the VM. We do that by launching `sleep infinity` in WSL and never letting it
    return; that single pinned process keeps the VM (and therefore Docker + ParadeDB) up
    for the whole logon session.

  WHAT IT DOES (in order, then supervises forever):
    1. Pin the WSL VM:  wsl -d Ubuntu -- sleep infinity   (kept running as a child process)
    2. Ensure ParadeDB is up:  wsl -d Ubuntu -- bash /mnt/c/code/claude-memory/scripts/db.sh up
    3. Start the dashboard:  mem serve --no-browser   (warm-server hot path)
    Then it loops: if the WSL pin or the dashboard process dies, it restarts them, and it
    periodically re-runs `db.sh up` (idempotent) so a crashed container comes back.

  This script is normally invoked by the Scheduled Task created by install_persistence.ps1,
  but you can also run it by hand in a console to watch the supervision output.
#>
$ErrorActionPreference = "Continue"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$logDir = Join-Path $root "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "persistence.log"

# With the store pinned to sqlite, the WSL pin + ParadeDB legs are dead weight - supervise
# only the dashboard server. They re-arm automatically if config.toml pins postgres/auto.
$needDb = $true
try {
  $m = Select-String -Path (Join-Path $root "config.toml") -Pattern '^\s*backend\s*=\s*"(\w+)"' | Select-Object -First 1
  if ($m -and $m.Matches[0].Groups[1].Value -eq "sqlite") { $needDb = $false }
} catch {}

function Write-Log([string]$msg) {
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Add-Content -Path $log -Value $line -Encoding utf8
}

# Singleton guard: the scheduled task, the SessionStart hook, and manual runs may all spawn
# this script. Global\ (not Local\) because the task can land in a different logon session.
# AbandonedMutexException = the previous holder died without releasing - WE own it now, so
# continue. Any other failure to acquire (held elsewhere, or unopenable across an elevation/
# ACL boundary) means another instance owns the stack - log the decision and exit.
$script:acquired = $false
try {
  $script:supervisorMutex = New-Object System.Threading.Mutex($false, "Global\ClaudeMemoryPersistence")
  try { $script:acquired = $script:supervisorMutex.WaitOne(0) }
  catch [System.Threading.AbandonedMutexException] { $script:acquired = $true }
} catch {
  Write-Log "singleton: mutex unopenable ($($_.Exception.GetType().Name)) - another instance owns it; exiting (pid $PID)"
  exit 0
}
if (-not $script:acquired) {
  Write-Log "singleton: mutex held elsewhere; exiting (pid $PID)"
  exit 0
}
Write-Log "singleton: mutex acquired (pid $PID)"

Write-Log "persistence supervisor starting (root=$root)"

# --- 1. Pin the WSL VM so Docker + ParadeDB never idle-terminate. ---
function Start-WslPin {
  # `sleep infinity` is a single long-lived process inside the Ubuntu VM. As long as it
  # runs, WSL2 will not shut the VM down. Started hidden, tracked so we can revive it.
  $p = Start-Process -FilePath "wsl.exe" `
        -ArgumentList @("-d", "Ubuntu", "--", "sleep", "infinity") `
        -WindowStyle Hidden -PassThru
  Write-Log "WSL pin started (pid=$($p.Id))"
  return $p
}

# --- 2. Ensure the ParadeDB container is up (idempotent; db.sh waits for healthy). ---
function Ensure-Db {
  Write-Log "ensuring ParadeDB is up via db.sh"
  & wsl.exe -d Ubuntu -- bash /mnt/c/code/claude-memory/scripts/db.sh up *> $null
  Write-Log "db.sh up returned exit=$LASTEXITCODE"
}

# --- 3. Start the dashboard (warm server). ---
function Start-Server {
  $p = Start-Process -FilePath $py `
        -ArgumentList @("-m", "claudemem", "serve", "--no-browser") `
        -WindowStyle Hidden -PassThru
  Write-Log "dashboard server started (pid=$($p.Id))"
  return $p
}

function Test-Server {
  try { return (Invoke-WebRequest -Uri "http://127.0.0.1:7777/api/stats" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 }
  catch { return $false }
}

$pin = $null
if ($needDb) {
  $pin = Start-WslPin
  Ensure-Db
} else {
  Write-Log "backend=sqlite: skipping WSL pin + ParadeDB; supervising the dashboard server only"
}
if (Test-Server) { Write-Log "a dashboard server is already up; not starting a second one"; $srv = $null }
else { $srv = Start-Server }

# --- Supervise forever: revive anything that dies, re-assert the DB periodically. ---
$lastDbCheck = Get-Date
while ($true) {
  Start-Sleep -Seconds 15

  if ($needDb -and ($pin -eq $null -or $pin.HasExited)) {
    Write-Log "WSL pin gone; restarting"
    $pin = Start-WslPin
  }
  if (-not (Test-Server)) {
    Write-Log "dashboard server not responding; (re)starting"
    $srv = Start-Server
  }
  if ($needDb -and ((Get-Date) - $lastDbCheck).TotalSeconds -ge 120) {
    Ensure-Db
    $lastDbCheck = Get-Date
  }
}
