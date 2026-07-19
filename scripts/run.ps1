#requires -Version 5.1
<#
.SYNOPSIS
  Start the VisualLLm stack (MuseTalk avatar server + pipeline) with a durable log
  file per process, wait until each is healthy, and print the URLs + log paths.

.DESCRIPTION
  Starts the MuseTalk avatar server in its `musetalk` conda env and the pipeline in
  the system python, and propagates the avatar knobs (AVATAR_REF / size / fps / lead /
  tail) from .env to the OS environment -- the avatar server reads OS env ONLY (no
  python-dotenv in its conda env), so without this it would use its built-in defaults
  and mismatch the pipeline/transport.

  Each process writes two kinds of log under logs\ :
    <name>.log              structured loguru (rotated, full tracebacks, uvicorn)
    <name>.out/.err.log     raw stdout/stderr (also catches native mediapipe/onnx spew)

.PARAMETER MusetalkPython
  Path to the python.exe of the 'musetalk' conda env (default E:\miniconda3\envs\musetalk).

.EXAMPLE
  .\scripts\run.ps1
#>
param(
    [string]$MusetalkPython = "E:\miniconda3\envs\musetalk\python.exe"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$pids = @()
$envFile = Join-Path $repo ".env"

# Read a single KEY=value from .env (trimmed, no inline-comment handling needed for
# the simple values we propagate). Returns $null if absent.
function Get-EnvVal([string]$key) {
    if (-not (Test-Path $envFile)) { return $null }
    $m = Select-String -Path $envFile -Pattern ("^\s*{0}\s*=\s*(.+?)\s*(?:#.*)?$" -f [regex]::Escape($key)) -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value.Trim() }
    return $null
}

# Propagate avatar knobs from .env to BOTH child processes. The MuseTalk server reads
# these from the OS environment ONLY (no .env loading in its conda env); without this
# it would use its built-in defaults and mismatch the pipeline/transport.
function Set-EnvFromDotenv([string]$key) {
    $v = Get-EnvVal $key
    if ($v) {
        Set-Item -Path ("Env:{0}" -f $key) -Value $v
        Write-Host ("  {0}={1} (from .env -> both processes)" -f $key, $v) -ForegroundColor DarkCyan
    }
}
Write-Host "Avatar engine: musetalk" -ForegroundColor Cyan
Set-EnvFromDotenv "AVATAR_REF"
Set-EnvFromDotenv "MUSETALK_SIZE"
Set-EnvFromDotenv "MUSETALK_BASE_MAX"
Set-EnvFromDotenv "MUSETALK_FPS"
Set-EnvFromDotenv "MUSETALK_LEAD_FRAMES"
Set-EnvFromDotenv "MUSETALK_END_TAIL_FRAMES"
Set-EnvFromDotenv "MUSETALK_IDLE_MOTION"
Set-EnvFromDotenv "MUSETALK_TRT"
Set-EnvFromDotenv "MUSETALK_GPU_COMPOSITE"
# Split mode: the avatar server reads MUSETALK_SPLIT from OS env only, and the pipeline reads it
# from .env. If the launcher forwards it to the pipeline but NOT here, the pipeline sizes the video
# track to the crop (256) while the server keeps streaming full 512 frames -- a size mismatch that
# renders NO video (voice fine). Forward both so the two stay consistent (see MUSETALK_SPLIT docs).
Set-EnvFromDotenv "MUSETALK_SPLIT"
Set-EnvFromDotenv "MUSETALK_SPLIT_SIZE"
# (No STT server to start: deepgram is cloud, sherpa runs in-process in the pipeline.
#  The funasr server + its launch block were removed 2026-07-14 with the provider branch.)

function Test-PortBusy([int]$port) {
    # netstat (native), NOT Get-NetTCPConnection: the CIM cmdlet hangs tens of
    # seconds under CPU load on this box (the windows-process-tools issue; the
    # config panel uses netstat for the same reason).
    $needle = ":{0} " -f $port
    $out = netstat -ano | Select-String -SimpleMatch $needle -ErrorAction SilentlyContinue
    return [bool]($out | Where-Object { $_ -match 'LISTENING' })
}

# Stop whatever LISTENS on a port. Same netstat + Stop-Process shape as launch.ps1's
# Stop-Port, and for the same reason as Test-PortBusy above: the CIM cmdlets hang tens of
# seconds under CPU load on this box. NOTE $procId, never $pid -- $pid is the automatic
# variable holding OUR pid, and assigning it would target this very script.
function Clear-Port([int]$port) {
    $needle = ":{0} " -f $port
    $procs = netstat -ano | Select-String -SimpleMatch $needle |
        Where-Object { $_ -match 'LISTENING' } |
        ForEach-Object { ($_.ToString().Trim() -split '\s+')[-1] } |
        Sort-Object -Unique
    foreach ($procId in $procs) {
        Write-Host ("  port {0}: stopping leftover listener PID {1}" -f $port, $procId) -ForegroundColor Yellow
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
}

# RECLAIM a port a previous run left behind, rather than refusing to start. Both ports are
# ours (7860 = pipeline/client, 8002 = avatar server) and the code below is about to start a
# fresh process on each with the CURRENT .env, so a leftover listener is never worth keeping
# -- it is stale by definition (2026-07-17: an avatar server orphaned ~2h earlier was still
# holding 8002 with pre-.env-edit settings).
# Aborting was far more expensive than it looked: this guard runs BEFORE the pipeline
# Start-Process below, so ONE orphaned avatar server meant NO pipeline at all -- and launch.ps1
# calls this script with & (which ignores the exit code), so it sailed on and published a
# Cloudflare tunnel over the dead origin. Every request 502'd, with the true cause (a port
# collision) never reaching a log. Still aborts when the port will NOT free: that is the
# original silent-bind-error guard, and it is kept.
foreach ($p in @(7860, 8002)) {
    if (-not (Test-PortBusy $p)) { continue }
    Write-Host ("Port {0} is already in use -- reclaiming it (a stale listener from a previous run)." -f $p) -ForegroundColor Yellow
    Clear-Port $p
    $freed = $false
    for ($i = 0; $i -lt 10; $i++) {
        Start-Sleep -Milliseconds 500
        if (-not (Test-PortBusy $p)) { $freed = $true; break }
    }
    if (-not $freed) {
        Write-Host "ERROR: port $p is in use and could not be freed. Stop that process by hand first." -ForegroundColor Red
        exit 1
    }
    Write-Host ("  port {0} freed." -f $p) -ForegroundColor Green
}

# 1) MuseTalk avatar server -- own conda env, unbuffered (-u) so logs are live.
if (-not (Test-Path $MusetalkPython)) {
    Write-Host "ERROR: musetalk python not found at $MusetalkPython (pass -MusetalkPython <path>)." -ForegroundColor Red
    exit 1
}
Write-Host "Starting musetalk server -> logs\musetalk.out.log"
$av = Start-Process -FilePath $MusetalkPython `
    -ArgumentList '-u', '-m', 'local_services.musetalk_server.app' `
    -WorkingDirectory $repo -NoNewWindow -PassThru `
    -RedirectStandardOutput (Join-Path $logs "musetalk.out.log") `
    -RedirectStandardError  (Join-Path $logs "musetalk.err.log")
$pids += $av.Id
Write-Host ("  musetalk PID {0}; waiting for models to load (/health)..." -f $av.Id)
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    try {
        # 127.0.0.1 (not localhost): the server binds IPv4-only, and Windows
        # resolves localhost to ::1 (IPv6) first, which would never answer.
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8002/health" -TimeoutSec 3
        if ($h.ok) { $ok = $true; break }
    } catch { }
}
if ($ok) { Write-Host "  musetalk ready." -ForegroundColor Green }
else { Write-Host "  musetalk not healthy in ~120s; check logs\musetalk.err.log" -ForegroundColor Yellow }

# 2) Pipeline (serves /client at :7860).
Write-Host "Starting pipeline -> logs\pipeline.out.log"
$pipe = Start-Process -FilePath "python" `
    -ArgumentList '-m', 'pipeline.main' `
    -WorkingDirectory $repo -NoNewWindow -PassThru `
    -RedirectStandardOutput (Join-Path $logs "pipeline.out.log") `
    -RedirectStandardError  (Join-Path $logs "pipeline.err.log")
$pids += $pipe.Id
Write-Host ("  Pipeline PID {0}; waiting for /client..." -f $pipe.Id)
Start-Sleep -Seconds 2   # let the process import Pipecat before the first probe
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:7860/client/" -TimeoutSec 2 -UseBasicParsing -DisableKeepAlive
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if ($ok) { Write-Host "  Pipeline ready." -ForegroundColor Green }
else { Write-Host "  Client not up yet; check logs\pipeline.err.log" -ForegroundColor Yellow }

Write-Host ""
Write-Host "== VisualLLm running ==" -ForegroundColor Cyan
Write-Host "  Client : http://localhost:7860/client/"
Write-Host "  Logs (structured): $logs\pipeline.log  $logs\musetalk.log"
Write-Host "  Raw stdout/stderr (native spew + banner): logs\*.out.log / *.err.log"
Write-Host ""
Write-Host ("Stop with:  Stop-Process -Id {0}" -f ($pids -join ','))
