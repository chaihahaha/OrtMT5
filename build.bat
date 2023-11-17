@echo off
set VS_DEV_CMD="D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
set ARCH="amd64"
call %VS_DEV_CMD% -arch=%ARCH%
cd /d %~dp0
mkdir build
cd build
cmake ..
cmake --build . --config Release
xcopy Release\* ..\
pause
