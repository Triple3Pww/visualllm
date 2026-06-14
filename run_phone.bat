@echo off
REM ============================================================================
REM  VisualLLm — run it for PHONE access over Tailscale (HTTPS).
REM
REM  Avatar mode is read from avatar_mode.txt (set by switch_local.bat /
REM  switch_cloud.bat). Default = local MuseTalk. Local mode also starts the
REM  GPU avatar server; cloud mode (Simli) does not.
REM
REM  ONE-TIME PREREQUISITE: enable HTTPS in the Tailscale admin once:
REM    https://login.tailscale.com/admin/dns  ->  Enable HTTPS
REM ============================================================================
setlocal
set TS="C:\Program Files\Tailscale\tailscale.exe"
set PY=E:\miniconda3\envs\musetalk\python.exe
set PHONE_URL=https://porsche-pc.tail21bb8a.ts.net/client
cd /d "%~dp0"

set MODE=musetalk_local
if exist avatar_mode.txt set /p MODE=<avatar_mode.txt

echo ============================================================
echo   Selected avatar: %MODE%
echo   (Switch with switch_local.bat / switch_cloud.bat, then
echo    reconnect the phone -- no need to restart for a switch.)
echo ============================================================

echo %MODE% | findstr /I "musetalk" >nul
if %errorlevel%==0 (
  echo [*] Starting the MuseTalk avatar server ^(GPU^)...
  set AVATAR_REF=%CD%\assets\avatar.png
  start "MuseTalk server" cmd /k "%PY% -u -m local_services.musetalk_server.app"
) else (
  echo [*] Cloud avatar selected - no local GPU server needed.
)

echo [*] Starting the VisualLLm pipeline...
start "VisualLLm pipeline" cmd /k "python -m pipeline.main"

echo [*] Exposing the pipeline over Tailscale HTTPS...
%TS% serve --bg --https=443 http://127.0.0.1:7860 >nul 2>&1

echo.
echo ============================================================
echo   On your PHONE (Tailscale app ON, same account), open:
echo.
echo       %PHONE_URL%
echo.
echo   Allow the microphone, wait for the face, then talk.
echo ============================================================
echo.
%TS% serve status
echo.
echo (Leave this window and the server/pipeline windows open while you use it.)
pause
