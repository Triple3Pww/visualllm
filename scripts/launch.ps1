#requires -Version 5.1
<#
.SYNOPSIS
  One-click full-stack launcher for VisualLLm. Started by "Run VisualLLm.exe".

.DESCRIPTION
  Brings up the whole system in order and opens the client in the browser:
    1. CosyVoice TTS server in WSL  (:8001, only if TTS_PROVIDER=cosyvoice)
    2. MuseTalk avatar + pipeline   (delegates to scripts\run.ps1 -> :8002 + :7860)
    3. Web config panel             (:7870)
    4. Cloudflare quick tunnel      (public https://<random>.trycloudflare.com link)
    5. Opens http://localhost:7860/client/

  The launcher window stays open as the system's "running" indicator: press Enter
  in it (or close it) to shut every service down. Per-process logs land in logs\.

.NOTES
  TTS lives in WSL and is reached over the WSL IP in .env COSYVOICE_URL (NOT
  localhost -- WSL2's localhost relay buffers the audio stream). If the WSL IP
  changed after a `wsl --shutdown`, update COSYVOICE_URL (get it via `wsl hostname -I`).
#>
param(
    [string]$MusetalkPython = "E:\miniconda3\envs\musetalk\python.exe",
    [string]$WslDistro      = "Ubuntu",
    [string]$CosyRunScript  = "/mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$logs = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$envFile = Join-Path $repo ".env"

# Read a single KEY=value from .env (same parser as run.ps1).
function Get-EnvVal([string]$key) {
    if (-not (Test-Path $envFile)) { return $null }
    $m = Select-String -Path $envFile -Pattern ("^\s*{0}\s*=\s*(.+?)\s*(?:#.*)?$" -f [regex]::Escape($key)) -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value.Trim() }
    return $null
}

# True when an HTTP GET returns 200 (health probes). A 503/refused throws -> $false.
function Test-Url([string]$u) {
    try { return ((Invoke-WebRequest -Uri $u -TimeoutSec 3 -UseBasicParsing -DisableKeepAlive).StatusCode -eq 200) }
    catch { return $false }
}

# Stop whatever LISTENS on a port (best-effort shutdown). Uses native netstat,
# NOT Get-NetTCPConnection -- the CIM cmdlet hangs tens of seconds under CPU load
# on this box (the windows-process-tools issue; run.ps1/config_panel do the same).
function Stop-Port([int]$port) {
    $needle = ":{0} " -f $port
    $procs = netstat -ano | Select-String -SimpleMatch $needle |
        Where-Object { $_ -match 'LISTENING' } |
        ForEach-Object { ($_.ToString().Trim() -split '\s+')[-1] } |
        Sort-Object -Unique
    foreach ($procId in $procs) { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue }
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   VisualLLm -- full-stack one-click launcher"   -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

$startedWsl = $null

# ---------------------------------------------------------------------------
# 1) CosyVoice TTS in WSL (only when it is the selected provider).
# ---------------------------------------------------------------------------
$ttsProvider = Get-EnvVal "TTS_PROVIDER"
$cosyUrl     = Get-EnvVal "COSYVOICE_URL"
if ($ttsProvider -eq "cosyvoice" -and $cosyUrl) {
    $health = "$cosyUrl/health"
    Write-Host "[1/5] CosyVoice TTS ($cosyUrl)" -ForegroundColor Cyan
    if (Test-Url $health) {
        Write-Host "  already up -- reusing." -ForegroundColor Green
    } else {
        Write-Host "  starting in WSL ($WslDistro) -- a separate window will show its logs..."
        # COSYVOICE_MODEL (.env) selects the model: v2 (default) or v3 (Fun-CosyVoice3-0.5B).
        # Forwarded as an env prefix; run_vllm_server.sh expands it into MODEL_DIR + PROMPT_TEXT.
        $cosyModel = Get-EnvVal "COSYVOICE_MODEL"
        if ($cosyModel) {
            Write-Host "  COSYVOICE_MODEL=$cosyModel" -ForegroundColor DarkGray
            $inner = "COSYVOICE_MODEL=$cosyModel bash $CosyRunScript"
        } else {
            $inner = "bash $CosyRunScript"
        }
        $cmd = ('-d {0} -e bash -c "{1}"' -f $WslDistro, $inner)
        $startedWsl = Start-Process -FilePath "wsl.exe" -ArgumentList $cmd -PassThru
        Write-Host "  loading the TTS model (this takes ~1-3 min on first start)..."
        $ok = $false
        for ($i = 0; $i -lt 120; $i++) {   # ~240s budget
            Start-Sleep -Seconds 2
            if (Test-Url $health) { $ok = $true; break }
        }
        if ($ok) { Write-Host "  CosyVoice ready." -ForegroundColor Green }
        else { Write-Host "  TTS not healthy yet -- the bot may be silent until it finishes loading (check the WSL window)." -ForegroundColor Yellow }
    }
} else {
    Write-Host "[1/5] TTS_PROVIDER=$ttsProvider -- skipping WSL CosyVoice start." -ForegroundColor DarkGray
}
Write-Host ""

# ---------------------------------------------------------------------------
# 2) MuseTalk avatar + pipeline (run.ps1 owns the env propagation + health waits).
# ---------------------------------------------------------------------------
Write-Host "[2/5] Avatar server + pipeline (via run.ps1)" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "run.ps1") -MusetalkPython $MusetalkPython
if (-not (Test-Url "http://127.0.0.1:7860/client/")) {
    Write-Host "  WARNING: pipeline client did not come up -- check logs\pipeline.err.log" -ForegroundColor Yellow
}
Write-Host ""

# ---------------------------------------------------------------------------
# 3) Web config panel (:7870), system python, logged to logs\config_panel.*.
# ---------------------------------------------------------------------------
Write-Host "[3/5] Config panel (http://localhost:7870)" -ForegroundColor Cyan
$cp = $null
if (Test-Url "http://127.0.0.1:7870/") {
    Write-Host "  already up -- reusing." -ForegroundColor Green
} else {
    $cp = Start-Process -FilePath "python" `
        -ArgumentList "-m", "local_services.config_panel.server" `
        -WorkingDirectory $repo -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $logs "config_panel.out.log") `
        -RedirectStandardError  (Join-Path $logs "config_panel.err.log")
    $ok = $false
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Url "http://127.0.0.1:7870/") { $ok = $true; break }
    }
    if ($ok) { Write-Host "  config panel ready." -ForegroundColor Green }
    else { Write-Host "  config panel not up -- check logs\config_panel.err.log" -ForegroundColor Yellow }
}
Write-Host ""

# ---------------------------------------------------------------------------
# 4) Cloudflare quick tunnel -- the PUBLIC link anyone can open. tunnel.ps1 starts
#    cloudflared, waits for the https://<random>.trycloudflare.com URL, and returns
#    {Process, Url}. The tunnel carries only the page + signaling; media reachability
#    is .env WEBRTC_PUBLIC=1 (STUN). Never fails the launch -- local stack still runs.
# ---------------------------------------------------------------------------
Write-Host "[4/5] Cloudflare public tunnel" -ForegroundColor Cyan
$tunnel = $null
try {
    $tunnel = & (Join-Path $PSScriptRoot "tunnel.ps1") -Port 7860
    if (-not ($tunnel -and $tunnel.Url)) {
        Write-Host "  no public URL yet -- see logs\cloudflared.log; the local stack is still up." -ForegroundColor Yellow
    }
} catch {
    Write-Host "  tunnel failed to start ($($_.Exception.Message)) -- local stack is still up." -ForegroundColor Yellow
}
Write-Host ""

# ---------------------------------------------------------------------------
# 5) Open the client.
# ---------------------------------------------------------------------------
Write-Host "[5/5] Opening the client in your browser..." -ForegroundColor Cyan
Start-Process "http://localhost:7860/client/"
Write-Host ""

Write-Host "===============================================" -ForegroundColor Green
Write-Host "   VisualLLm is RUNNING"                          -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host "  Client      : http://localhost:7860/client/"
Write-Host "  Config panel: http://localhost:7870"
if ($tunnel -and $tunnel.Url) {
    Write-Host "  PUBLIC link : $($tunnel.Url)$($tunnel.ClientPath)  <-- share this" -ForegroundColor Green
} else {
    Write-Host "  PUBLIC link : (not up -- check logs\cloudflared.log)" -ForegroundColor Yellow
}
Write-Host "  Logs        : $logs"
Write-Host ""
Read-Host "Press Enter to STOP everything and exit (closing this window also stops it)"

Write-Host "Shutting down..." -ForegroundColor Yellow
Stop-Port 7860      # pipeline / client
Stop-Port 8002      # musetalk avatar server
Stop-Port 7870      # config panel
if ($tunnel -and $tunnel.Process) { try { Stop-Process -Id $tunnel.Process.Id -Force -ErrorAction SilentlyContinue } catch { } }
if ($cp) { try { Stop-Process -Id $cp.Id -Force -ErrorAction SilentlyContinue } catch { } }
if ($startedWsl) { try { Stop-Process -Id $startedWsl.Id -Force -ErrorAction SilentlyContinue } catch { } }
Write-Host "Stopped. (CosyVoice inside WSL may keep running -- 'wsl --shutdown' to fully stop it.)" -ForegroundColor DarkGray
Start-Sleep -Seconds 1
