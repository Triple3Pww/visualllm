@echo off
REM Start the local MuseTalk lip-sync avatar server (Phase 3).
REM Runs in the dedicated 'musetalk' conda env (cu128 torch + face_alignment).
REM The portrait it drives is AVATAR_REF (default assets\avatar.png).

cd /d "%~dp0\..\.."
set AVATAR_REF=%CD%\assets\avatar.png
echo Starting MuseTalk server on http://localhost:8002  (avatar: %AVATAR_REF%)
conda run --no-capture-output -n musetalk python -u -m local_services.musetalk_server.app
