<#
.SYNOPSIS
  Remove the "ClaudeMemoryPersistence" Scheduled Task created by install_persistence.ps1.
.DESCRIPTION
  Stops the task if running and unregisters it. Does NOT stop an already-running dashboard
  server or the WSL VM pin from the current session — those exit on next logoff, or you can
  end them manually. Safe to run if the task does not exist.

  NOTE: may require an elevated (Administrator) PowerShell, same as install.
.EXAMPLE
  .\scripts\uninstall_persistence.ps1
#>
$ErrorActionPreference = "Continue"
$TaskName = "ClaudeMemoryPersistence"
$WatchdogTaskName = "ClaudeMemoryWatchdog"

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $task) {
  try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch {}
  try { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false; Write-Host "Removed Scheduled Task '$TaskName'." }
  catch { Write-Host "Scheduled Task removal failed (try elevated): $($_.Exception.Message)" }
} else {
  Write-Host "Scheduled Task '$TaskName' not found."
}

$watchdogTask = Get-ScheduledTask -TaskName $WatchdogTaskName -ErrorAction SilentlyContinue
if ($null -ne $watchdogTask) {
  try { Stop-ScheduledTask -TaskName $WatchdogTaskName -ErrorAction SilentlyContinue } catch {}
  try { Unregister-ScheduledTask -TaskName $WatchdogTaskName -Confirm:$false; Write-Host "Removed Scheduled Task '$WatchdogTaskName'." }
  catch { Write-Host "Watchdog task removal failed: $($_.Exception.Message)" }
}

# Also remove the no-admin HKCU Run fallback entry, if present.
$key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
if (Get-ItemProperty -Path $key -Name "ClaudeMemoryPersistence" -ErrorAction SilentlyContinue) {
  Remove-ItemProperty -Path $key -Name "ClaudeMemoryPersistence" -ErrorAction SilentlyContinue
  Write-Host "Removed HKCU Run entry 'ClaudeMemoryPersistence'."
}
if (Get-ItemProperty -Path $key -Name $WatchdogTaskName -ErrorAction SilentlyContinue) {
  Remove-ItemProperty -Path $key -Name $WatchdogTaskName -ErrorAction SilentlyContinue
  Write-Host "Removed HKCU Run entry '$WatchdogTaskName'."
}
Write-Host "(The dashboard server / WSL pin from the current session, if any, were left running;"
Write-Host " they will not be restarted at next logon.)"
