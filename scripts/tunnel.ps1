#requires -Version 5.1
<#
.SYNOPSIS
  Start a Cloudflare quick tunnel that makes the local pipeline (:7860) reachable by
  ANYONE with the printed link -- no Cloudflare account, no domain.

.DESCRIPTION
  Runs `cloudflared tunnel --url http://localhost:<Port>`, which prints a random
  https://<random>.trycloudflare.com URL and proxies it to the local port. That URL is
  the shareable public link. The tunnel only carries the HTTPS page + /api/offer
  signaling; the mic/avatar MEDIA is UDP P2P over ICE and needs .env WEBRTC_PUBLIC=1
  (STUN) to be reachable off-tailnet -- the tunnel alone is not enough.

  The URL is RANDOM and changes every time the tunnel restarts (inherent to zero-setup
  quick tunnels). Standalone, run with -Wait to hold the window open (closing it stops
  the tunnel). launch.ps1 calls this without -Wait and gets back {Process, Url} to track
  and shut down with the rest of the stack.

.NOTES
  cloudflared v2025.8.x is expected at C:\Program Files (x86)\cloudflared, else on PATH.
  ASCII-only source (PS 5.1 parser), same house style as launch.ps1 / run.ps1.
#>
param(
    [int]$Port = 7860,
    [switch]$Wait
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$log = Join-Path $logs "cloudflared.log"

# Locate cloudflared: PATH first, then the known winget install location.
$exe = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $exe) {
    $known = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
    if (Test-Path $known) { $exe = $known }
}
if (-not $exe) {
    Write-Host "ERROR: cloudflared not found (PATH or 'C:\Program Files (x86)\cloudflared'). Install it, then re-run." -ForegroundColor Red
    return $null
}

# Recommended client path follows the live avatar preset (leo -> /studio/, else /nimbus/).
$preset = $null
$envFile = Join-Path $repo ".env"
if (Test-Path $envFile) {
    $m = Select-String -Path $envFile -Pattern '^\s*AVATAR_PRESET\s*=\s*(.+?)\s*(?:#.*)?$' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { $preset = $m.Matches[0].Groups[1].Value.Trim() }
}
$clientPath = if ($preset -eq "leo") { "/studio/" } else { "/nimbus/" }

# Fresh log so we only parse THIS run's URL, then start the tunnel detached.
Set-Content -Path $log -Value "" -Encoding ascii
$args = @("tunnel", "--no-autoupdate", "--url", ("http://localhost:{0}" -f $Port))
$proc = Start-Process -FilePath $exe -ArgumentList $args -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError (Join-Path $logs "cloudflared.err.log")

Write-Host "Starting Cloudflare quick tunnel to http://localhost:$Port ..." -ForegroundColor Cyan

# Poll the log for the assigned trycloudflare URL (~30s budget; cloudflared prints it in a banner).
$url = $null
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    foreach ($file in @($log, (Join-Path $logs "cloudflared.err.log"))) {
        if (Test-Path $file) {
            $hit = Select-String -Path $file -Pattern 'https://[-a-z0-9]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($hit) { $url = $hit.Matches[0].Value; break }
        }
    }
    if ($url) { break }
    if ($proc.HasExited) { break }
}

if (-not $url) {
    Write-Host "Tunnel did not report a URL in time -- check $log (and cloudflared.err.log)." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "  PUBLIC LINK (share this): $url$clientPath" -ForegroundColor Green
    Write-Host "  (base $url ; also /nimbus/ , /client/)" -ForegroundColor DarkGray
    Write-Host ""
}

if ($Wait) {
    Read-Host "Tunnel running. Press Enter to STOP it (closing this window also stops it)"
    try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
    return $null
}

return [pscustomobject]@{ Process = $proc; Url = $url; ClientPath = $clientPath }
