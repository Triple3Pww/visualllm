#requires -Version 5.1
<#
.SYNOPSIS
  Start the VisualLLm stack (local avatar server + pipeline) with a durable log
  file per process, wait until each is healthy, and print the URLs + log paths.

.DESCRIPTION
  The avatar engine is chosen by AVATAR in .env (musetalk = default, ditto, none).
  This script starts the matching server in its own conda env and propagates the
  engine-specific knobs (AVATAR_REF / size / fps) from .env to the OS environment,
  because the avatar servers read OS env ONLY (no python-dotenv in their conda envs).

  Each process writes two kinds of log under logs\ :
    <name>.log              structured loguru (rotated, full tracebacks, uvicorn)
    <name>.out/.err.log     raw stdout/stderr (also catches native mediapipe/onnx spew)
  The structured logs are the ones the dashboard tails at http://localhost:7861/debug.

.PARAMETER AvatarNone
  Force audio-only (AVATAR=none): the client renders the face, so no avatar server starts.

.PARAMETER MusetalkPython
  Path to the python.exe of the 'musetalk' conda env (default E:\miniconda3\envs\musetalk).

.PARAMETER DittoPython
  Path to the python.exe of the 'ditto' conda env (default E:\miniconda3\envs\ditto).

.EXAMPLE
  .\scripts\run.ps1
  .\scripts\run.ps1 -AvatarNone
#>
param(
    [switch]$AvatarNone,
    [string]$MusetalkPython = "E:\miniconda3\envs\musetalk\python.exe",
    [string]$DittoPython = "E:\miniconda3\envs\ditto\python.exe"
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

# Resolve the active avatar engine: -AvatarNone wins, else AVATAR from .env (default musetalk).
if ($AvatarNone) {
    $avatarMode = "none"
} else {
    $avatarMode = Get-EnvVal "AVATAR"
    if (-not $avatarMode) { $avatarMode = "musetalk" }
    $avatarMode = $avatarMode.ToLower()
}
Write-Host ("Avatar engine: {0}" -f $avatarMode) -ForegroundColor Cyan

# Propagate engine-specific knobs from .env to BOTH child processes. The avatar servers
# read these from the OS environment ONLY (no .env loading in their conda envs); without
# this they would use their built-in defaults and mismatch the pipeline/transport.
function Set-EnvFromDotenv([string]$key) {
    $v = Get-EnvVal $key
    if ($v) {
        Set-Item -Path ("Env:{0}" -f $key) -Value $v
        Write-Host ("  {0}={1} (from .env -> both processes)" -f $key, $v) -ForegroundColor DarkCyan
    }
}
if ($avatarMode -eq "musetalk") {
    Set-EnvFromDotenv "AVATAR_REF"
    Set-EnvFromDotenv "MUSETALK_SIZE"
    Set-EnvFromDotenv "MUSETALK_FPS"
    Set-EnvFromDotenv "MUSETALK_LEAD_FRAMES"
    Set-EnvFromDotenv "MUSETALK_END_TAIL_FRAMES"
} elseif ($avatarMode -eq "ditto") {
    Set-EnvFromDotenv "AVATAR_REF"
    Set-EnvFromDotenv "DITTO_SIZE"
}

function Test-PortBusy([int]$port) {
    $c = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    return [bool]$c
}

# Refuse to start over a port that is already taken (the silent bind error that
# wasted time before). 7860 = client, 7861 = dashboard, 8002 = avatar server.
$ports = @(7860, 7861)
if ($avatarMode -ne "none") { $ports += 8002 }
foreach ($p in $ports) {
    if (Test-PortBusy $p) {
        Write-Host "ERROR: port $p is already in use. Stop the existing process first." -ForegroundColor Red
        exit 1
    }
}

# 1) Avatar server -- own conda env, unbuffered (-u) so logs are live. Engine-specific.
if ($avatarMode -ne "none") {
    if ($avatarMode -eq "musetalk") {
        $avPython = $MusetalkPython; $avModule = "local_services.musetalk_server.app"
    } else {
        $avPython = $DittoPython;    $avModule = "local_services.ditto_server.app"
    }
    if (-not (Test-Path $avPython)) {
        Write-Host "ERROR: $avatarMode python not found at $avPython (pass -MusetalkPython/-DittoPython <path>)." -ForegroundColor Red
        exit 1
    }
    Write-Host ("Starting {0} server -> logs\{0}.out.log" -f $avatarMode)
    $av = Start-Process -FilePath $avPython `
        -ArgumentList '-u', '-m', $avModule `
        -WorkingDirectory $repo -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $logs ("{0}.out.log" -f $avatarMode)) `
        -RedirectStandardError  (Join-Path $logs ("{0}.err.log" -f $avatarMode))
    $pids += $av.Id
    Write-Host ("  {0} PID {1}; waiting for models to load (/health)..." -f $avatarMode, $av.Id)
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        try {
            # 127.0.0.1 (not localhost): the servers bind IPv4-only, and Windows
            # resolves localhost to ::1 (IPv6) first, which would never answer.
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:8002/health" -TimeoutSec 3
            if ($h.ok) { $ok = $true; break }
        } catch { }
    }
    if ($ok) { Write-Host ("  {0} ready." -f $avatarMode) -ForegroundColor Green }
    else { Write-Host ("  {0} not healthy in ~120s; check logs\{0}.err.log" -f $avatarMode) -ForegroundColor Yellow }
}

# 2) Pipeline (+ dashboard). Audio-only sets AVATAR=none for the child process.
if ($avatarMode -eq "none") { $env:AVATAR = "none" }
Write-Host "Starting pipeline -> logs\pipeline.out.log"
$pipe = Start-Process -FilePath "python" `
    -ArgumentList '-m', 'pipeline.main' `
    -WorkingDirectory $repo -NoNewWindow -PassThru `
    -RedirectStandardOutput (Join-Path $logs "pipeline.out.log") `
    -RedirectStandardError  (Join-Path $logs "pipeline.err.log")
$pids += $pipe.Id
Write-Host ("  Pipeline PID {0}; waiting for dashboard..." -f $pipe.Id)
Start-Sleep -Seconds 2   # let the process import Pipecat before the first probe
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:7861/debug" -TimeoutSec 2 -UseBasicParsing -DisableKeepAlive
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if ($ok) { Write-Host "  Pipeline + dashboard ready." -ForegroundColor Green }
else { Write-Host "  Dashboard not up yet; check logs\pipeline.err.log" -ForegroundColor Yellow }

Write-Host ""
Write-Host "== VisualLLm running ==" -ForegroundColor Cyan
Write-Host "  Client    : http://localhost:7860/client/"
Write-Host "  Dashboard : http://localhost:7861/debug"
Write-Host "  Logs (structured, tailed by the dashboard):"
Write-Host "    $logs\pipeline.log"
if ($avatarMode -ne "none") { Write-Host ("    {0}\{1}.log" -f $logs, $avatarMode) }
Write-Host "  Raw stdout/stderr (native spew + banner): logs\*.out.log / *.err.log"
Write-Host ""
Write-Host ("Stop with:  Stop-Process -Id {0}" -f ($pids -join ','))
