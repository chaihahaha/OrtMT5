@echo off
set VS_DEV_CMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
set ARCH="amd64"
call %VS_DEV_CMD% -arch=%ARCH%
cd /d %~dp0
mkdir build
mkdir bin
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
xcopy Debug\* ..\bin
pause
