@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
echo ---VCVARS_EXIT %ERRORLEVEL%---
where cl
where dumpbin
where lib
