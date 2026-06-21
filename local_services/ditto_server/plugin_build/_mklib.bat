@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "E:\Claude\VisualLLm\local_services\ditto_server\plugin_build\trt_lib"
lib /DEF:nvinfer_10.def /MACHINE:X64 /OUT:nvinfer_10.lib /NOLOGO
echo LIB_EXIT %ERRORLEVEL%
