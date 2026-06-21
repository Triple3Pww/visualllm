@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >/dev/null 2>&1
set DLL=E:\miniconda3\envs\ditto\Lib\site-packages\tensorrt_libs\nvinfer_10.dll
set OUT=E:\Claude\VisualLLm\local_services\ditto_server\plugin_build\trt_lib
dumpbin /EXPORTS "%DLL%" > "%OUT%\nvinfer_10.exports.txt"
echo DUMPBIN_DONE exit %ERRORLEVEL%
where dumpbin lib
