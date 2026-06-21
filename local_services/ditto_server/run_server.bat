@echo off
REM Start the Ditto lip-sync server in the dedicated 'ditto' conda env.
cd /d %~dp0\..\..
conda run -n ditto python -m local_services.ditto_server.app
