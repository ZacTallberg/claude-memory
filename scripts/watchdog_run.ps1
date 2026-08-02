<#
.SYNOPSIS
  Keep the claude-memory persistence supervisor alive without requiring administrator access.
.DESCRIPTION
  This watchdog is deliberately independent from persistence_run.ps1. It checks the
  supervisor heartbeat and process identity every 30 seconds, starts a replacement when
  either is stale, and owns a different named mutex so duplicate logon/task launches collapse
  into one long-running watchdog.
#>
$ErrorActionPreference = "Continue"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runner = Join-Path $root "scripts\persistence_run.ps1"
$dataDir = Join-Path $root "data"
$logDir = Join-Path $dataDir "logs"
$supervisorHeartbeat = Join-Path $dataDir "persistence-heartbeat.json"
$watchdogHeartbeat = Join-Path $dataDir "watchdog-heartbeat.json"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "watchdog.log"

function Write-Log([string]$message) {
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $message
  Add-Content -LiteralPath $log -Value $line -Encoding utf8
}

function Write-WatchdogHeartbeat {
  try {
    $payload = @{ ts = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); pid = $PID } | ConvertTo-Json -Compress
    [System.IO.File]::WriteAllText($watchdogHeartbeat, $payload, (New-Object System.Text.UTF8Encoding($false)))
  } catch { }
}

function Test-Supervisor {
  if (-not (Test-Path -LiteralPath $supervisorHeartbeat)) { return $false }
  try {
    $state = Get-Content -LiteralPath $supervisorHeartbeat -Raw | ConvertFrom-Json
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $age = $now - [long]$state.ts
    if ($age -lt -300 -or $age -gt 90) { return $false }
    $process = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f [int]$state.pid) -ErrorAction Stop
    return ($null -ne $process -and $process.CommandLine -like "*persistence_run.ps1*")
  } catch { return $false }
}

$acquired = $false
try {
  $watchdogMutex = New-Object System.Threading.Mutex($false, "Global\ClaudeMemoryWatchdog")
  try { $acquired = $watchdogMutex.WaitOne(0) }
  catch [System.Threading.AbandonedMutexException] { $acquired = $true }
} catch {
  Write-Log "singleton mutex unavailable ($($_.Exception.GetType().Name)); exiting pid=$PID"
  exit 0
}
if (-not $acquired) { exit 0 }

Write-Log "watchdog starting pid=$PID root=$root"
while ($true) {
  Write-WatchdogHeartbeat
  if (-not (Test-Supervisor)) {
    try {
      $process = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", $runner) `
        -WindowStyle Hidden -PassThru
      Write-Log "supervisor missing or stale; launched candidate pid=$($process.Id)"
    } catch {
      Write-Log "failed to launch supervisor: $($_.Exception.Message)"
    }
  }
  Start-Sleep -Seconds 30
}

