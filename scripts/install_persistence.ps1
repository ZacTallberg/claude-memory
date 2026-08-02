<#
.SYNOPSIS
  Keep the claude-memory stack alive across logons. Admin-optional.
.DESCRIPTION
  WHY:
    ParadeDB (Postgres) runs in Docker inside the WSL2 "Ubuntu" VM. WSL2 idle-terminates that
    VM shortly after its last process exits, taking Docker + ParadeDB (localhost:55432) down
    with it, so recall fails. We keep a pinned process in the VM, ensure the DB is up, and run
    the warm dashboard server. scripts\persistence_run.ps1 does + supervises all three.

  This installer prefers a hidden At-Logon Scheduled Task. Registering a task can require an
  elevated shell; if that is denied it FALLS BACK to a current-user HKCU "Run" entry, which
  needs NO admin and also starts the supervisor at each logon.
.EXAMPLE
  .\scripts\install_persistence.ps1
#>
$ErrorActionPreference = "Stop"

$TaskName = "ClaudeMemoryPersistence"
$WatchdogTaskName = "ClaudeMemoryWatchdog"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runner = Join-Path $root "scripts\persistence_run.ps1"
$watchdog = Join-Path $root "scripts\watchdog_run.ps1"
if (-not (Test-Path $runner)) { throw "runner not found: $runner" }
if (-not (Test-Path $watchdog)) { throw "watchdog not found: $watchdog" }
$psArgs = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $runner
$watchdogArgs = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $watchdog

function Install-Task {
  $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $psArgs
  $logonTrigger = New-ScheduledTaskTrigger -AtLogOn
  # A child server can survive after its supervising PowerShell process is externally stopped.
  # At-logon alone then leaves memory unsupervised for days. A repeating watchdog trigger is
  # harmless while the task is running (MultipleInstancesPolicy=IgnoreNew) and re-arms it within
  # five minutes if the supervisor itself disappears.
  $watchdogTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
  $settings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1)
  $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($logonTrigger, $watchdogTrigger) -Settings $settings `
    -Principal $principal -Force -ErrorAction Stop `
    -Description "Supervises shared Claude/Codex hybrid memory, live indexing, and verified backups." | Out-Null
  if (-not (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)) {
    throw "task registration silently failed"
  }
}

function Install-WatchdogTask {
  $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $watchdogArgs
  $logonTrigger = New-ScheduledTaskTrigger -AtLogOn
  $repeatTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
  $settings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1)
  $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
  Register-ScheduledTask -TaskName $WatchdogTaskName -Action $action -Trigger @($logonTrigger, $repeatTrigger) `
    -Settings $settings -Principal $principal -Force -ErrorAction Stop `
    -Description "Re-arms the shared Claude/Codex memory supervisor if its heartbeat stops." | Out-Null
}

function Install-WatchdogRunKey {
  $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
  $val = 'powershell.exe ' + $watchdogArgs
  New-ItemProperty -Path $key -Name $WatchdogTaskName -Value $val -PropertyType String -Force | Out-Null
  Remove-ItemProperty -Path $key -Name $TaskName -ErrorAction SilentlyContinue
}

Install-WatchdogRunKey
Write-Host "Installed no-admin HKCU watchdog startup entry."

try {
  Install-Task
  Write-Host "Installed persistence as a hidden At-Logon Scheduled Task '$TaskName'."
  Write-Host "Start now without re-logon:  Start-ScheduledTask -TaskName '$TaskName'"
}
catch {
  Write-Host "Scheduled Task registration unavailable ($($_.Exception.Message.Split([Environment]::NewLine)[0]))."
  Write-Host "The watchdog startup entry will supervise the existing task/process without admin access."
}

try {
  Install-WatchdogTask
  Write-Host "Installed repeating Scheduled Task '$WatchdogTaskName'."
  Start-ScheduledTask -TaskName $WatchdogTaskName -ErrorAction Stop
} catch {
  Write-Host "Repeating watchdog task unavailable ($($_.Exception.Message.Split([Environment]::NewLine)[0]))."
  Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", $watchdog) -WindowStyle Hidden | Out-Null
  Write-Host "Started the no-admin watchdog directly; it will return automatically at next logon."
}
Write-Host "Remove later with:  .\scripts\uninstall_persistence.ps1"
