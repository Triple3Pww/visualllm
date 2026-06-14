@echo off
REM Switch the avatar to the LOCAL model (MuseTalk on the 5060 Ti). Double-click me.
cd /d "%~dp0"
echo musetalk_local>avatar_mode.txt
echo ============================================================
echo   Avatar switched to:  LOCAL  (MuseTalk on the 5060 Ti)
echo.
echo   Make sure the "MuseTalk server" window is running
echo   (run_phone.bat starts it), then RECONNECT on your phone.
echo ============================================================
echo.
pause
