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

# Only one quick tunnel should exist, and a lingering cloudflared from a previous session keeps
# logs\cloudflared.log OPEN -- then Set-Content below throws (ErrorAction Stop) and the whole
# script aborts BEFORE starting the tunnel, so the launcher reports "not up" even though the URL
# was available (the real "PUBLIC link not up" bug). Kill any stale cloudflared first: it frees
# the log AND prevents a second, stale tunnel. (This box uses cloudflared only for this.)
Get-Process -Name cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 400   # let the OS release the file handles before we truncate the logs

$errLog = Join-Path $logs "cloudflared.err.log"
# Fresh logs so we only ever parse THIS run's URL (cloudflared APPENDS; it writes the banner to
# STDERR, so the err log is the one that matters). Best-effort truncate -- never abort on a lock.
foreach ($f in @($log, $errLog)) { try { Set-Content -Path $f -Value "" -Encoding ascii -ErrorAction Stop } catch { } }

# --url 127.0.0.1 (NOT localhost): on Windows cloudflared resolves "localhost" to ::1 (IPv6) FIRST,
# but the pipeline binds IPv4 -- so every request 502'd ("dial tcp [::1]:7860: refused" in the log)
# and the public link, even when shown, reached nothing. 127.0.0.1 forces IPv4. (Same fix run.ps1
# already applies to its health probes.)
$cfArgs = @("tunnel", "--no-autoupdate", "--url", ("http://127.0.0.1:{0}" -f $Port))
$proc = Start-Process -FilePath $exe -ArgumentList $cfArgs -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $errLog

Write-Host "Starting Cloudflare quick tunnel to http://127.0.0.1:$Port ..." -ForegroundColor Cyan

# Poll for the assigned trycloudflare URL (~30s budget; cloudflared prints it to STDERR). Take the
# LAST match so a stale line can never win (logs are freshly cleared, but stay safe).
$url = $null
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    foreach ($file in @($errLog, $log)) {
        if (Test-Path $file) {
            $hit = Select-String -Path $file -Pattern 'https://[-a-z0-9]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -Last 1
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
