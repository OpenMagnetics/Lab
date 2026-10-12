@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set CM="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set NJ=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja
set PATH=%NJ%;%PATH%
rem usage: build_faraday.bat FARADAY_DIR  (a clone of github.com/OpenConverters/Faraday)
if "%~1"=="" (echo usage: build_faraday.bat FARADAY_DIR & exit /b 1)
cd /d %~1
%CM% -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFARADAY_BUILD_TESTS=OFF "-DCMAKE_CXX_FLAGS=/D_USE_MATH_DEFINES /EHsc /utf-8" || exit /b 1
%CM% --build build --target faraday_cli || exit /b 1
