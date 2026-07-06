<#
.SYNOPSIS
  Download the dashboard's vendored front-end JS into static/js/vendor/ (offline-first).
.DESCRIPTION
  The hub ships fully offline: htmx, Alpine, Cytoscape, ECharts, and markdown-it are
  vendored rather than pulled from a CDN at runtime (no network dependency, stable
  versions, works behind a firewall). This script fetches each library with
  Invoke-WebRequest into:
      claudemem/dashboard/static/js/vendor/

  Idempotent: a file that already exists and is non-empty is left untouched. Use
  -Force to re-download everything (e.g. to bump a version).
.EXAMPLE
  .\scripts\fetch_vendor.ps1
  .\scripts\fetch_vendor.ps1 -Force
#>
param(
  [switch]$Force
)
$ErrorActionPreference = "Stop"
# Modern TLS for older Windows PowerShell defaults.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$vendor = Join-Path $root "claudemem\dashboard\static\js\vendor"
New-Item -ItemType Directory -Force -Path $vendor | Out-Null

# filename -> pinned CDN URL. Pinned versions keep the build reproducible.
$assets = [ordered]@{
  "htmx.min.js"        = "https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js"
  "alpine.min.js"      = "https://unpkg.com/alpinejs@3.14.8/dist/cdn.min.js"
  "cytoscape.min.js"   = "https://unpkg.com/cytoscape@3.31.0/dist/cytoscape.min.js"
  "echarts.min.js"     = "https://unpkg.com/echarts@5.5.1/dist/echarts.min.js"
  "markdown-it.min.js" = "https://unpkg.com/markdown-it@14.1.0/dist/markdown-it.min.js"
}

$fetched = 0; $skipped = 0; $failed = 0
foreach ($name in $assets.Keys) {
  $dest = Join-Path $vendor $name
  $have = (Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)
  if ($have -and -not $Force) {
    Write-Host ("skip   {0} (exists, {1:N0} bytes)" -f $name, (Get-Item $dest).Length)
    $skipped++
    continue
  }
  $url = $assets[$name]
  Write-Host ("fetch  {0}  <-  {1}" -f $name, $url)
  try {
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing -TimeoutSec 60
    if ((Get-Item $dest).Length -le 0) { throw "downloaded file is empty" }
    Write-Host ("   ok  {0:N0} bytes" -f (Get-Item $dest).Length)
    $fetched++
  }
  catch {
    Write-Warning ("failed {0}: {1}" -f $name, $_.Exception.Message)
    if ((Test-Path $dest) -and (Get-Item $dest).Length -le 0) { Remove-Item $dest -Force }
    $failed++
  }
}

Write-Host ("`nvendor: {0} fetched, {1} skipped, {2} failed  ->  {3}" -f $fetched, $skipped, $failed, $vendor)
if ($failed -gt 0) { exit 1 }
