@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d "E:\Claude\VisualLLm\local_services\ditto_server\plugin_build"
rmdir /s /q build 2>/dev/null
cmake -S grid-sample3d-trt-plugin -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTensorRT_ROOT=E:/Claude/VisualLLm/local_services/ditto_server/plugin_build/trt_root -DCMAKE_CUDA_ARCHITECTURES=120
echo CONFIGURE_EXIT %ERRORLEVEL%
cmake --build build --config Release
echo BUILD_EXIT %ERRORLEVEL%
