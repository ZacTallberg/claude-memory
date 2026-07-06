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
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runner = Join-Path $root "scripts\persistence_run.ps1"
if (-not (Test-Path $runner)) { throw "runner not found: $runner" }
$psArgs = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $runner

function Install-Task {
  $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $psArgs
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $settings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
  $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Force -ErrorAction Stop `
    -Description "Keeps the claude-memory stack alive (WSL pin + ParadeDB + warm server)." | Out-Null
  if (-not (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)) {
    throw "task registration silently failed"
  }
}

function Install-RunKey {
  $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
  $val = 'powershell.exe ' + $psArgs
  New-ItemProperty -Path $key -Name "ClaudeMemoryPersistence" -Value $val -PropertyType String -Force | Out-Null
}

try {
  Install-Task
  Write-Host "Installed persistence as a hidden At-Logon Scheduled Task '$TaskName'."
  Write-Host "Start now without re-logon:  Start-ScheduledTask -TaskName '$TaskName'"
}
catch {
  Write-Host "Scheduled Task registration unavailable ($($_.Exception.Message.Split([Environment]::NewLine)[0]))."
  Write-Host "Falling back to a no-admin HKCU Run entry (starts at each logon)."
  Install-RunKey
  Write-Host "Installed HKCU\...\Run\ClaudeMemoryPersistence."
  Write-Host "Start now without re-logon:  powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runner`""
}
Write-Host "Remove later with:  .\scripts\uninstall_persistence.ps1"
