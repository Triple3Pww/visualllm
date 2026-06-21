@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
dumpbin /EXPORTS "E:\miniconda3\envs\ditto\Lib\site-packages\tensorrt_libs\nvinfer_10.dll" > "E:\Claude\VisualLLm\local_services\ditto_server\plugin_build\trt_lib\nvinfer_10.exports.txt"
echo DUMPBIN_EXIT %ERRORLEVEL%
