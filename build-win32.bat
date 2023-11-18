@echo off
set VS_DEV_CMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
call %VS_DEV_CMD%
cd /d %~dp0
mkdir build
cd build
cmake .. -A win32 -DONNXRUNTIME_ROOTDIR="path/to/onnxruntime-win-x86"
cmake --build . --config Release
xcopy Release\* ..\
pause
