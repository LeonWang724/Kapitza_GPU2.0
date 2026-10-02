@REM Load project-local and standard Windows CUDA build paths.
@REM This file intentionally does not use setlocal: callers need these values.
@echo off

if exist "%ProgramFiles%\Python312\python.exe" set "PATH=%ProgramFiles%\Python312;%PATH%"
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PATH=%LocalAppData%\Programs\Python\Python312;%PATH%"
@REM Project packages must take priority over system Python and other projects.
set "GPE_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "GPE_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "PATH=%~dp0.venv\Scripts;%PATH%"
if exist "%ProgramFiles%\CMake\bin\cmake.exe" set "PATH=%ProgramFiles%\CMake\bin;%PATH%"
if exist "%LocalAppData%\Microsoft\WinGet\Links\ninja.exe" set "PATH=%LocalAppData%\Microsoft\WinGet\Links;%PATH%"
if exist "%~dp0.venv\Library\lib\cmake\mkl\MKLConfig.cmake" (
  set "MKLROOT=%~dp0.venv\Library"
  set "MKL_DIR=%~dp0.venv\Library\lib\cmake\mkl"
  set "PATH=%~dp0.venv\Library\bin;%PATH%"
)

@REM Honor a valid CUDA_PATH for both compilation and runtime DLL lookup.
@REM Pascal GPUs need CUDA 12.x; an installed 13.3 must not override that choice.
if defined CUDA_PATH if not exist "%CUDA_PATH%\bin\nvcc.exe" set "CUDA_PATH="
if not defined CUDA_PATH if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
if not defined CUDA_PATH if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
if not defined CUDA_PATH if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
if not defined CUDA_PATH if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
if not defined CUDA_PATH if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not defined CUDA_PATH if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\x64" set "PATH=%CUDA_PATH%\bin\x64;%PATH%"

if defined HDF5_ROOT if not exist "%HDF5_ROOT%\include\hdf5.h" set "HDF5_ROOT="
if not defined HDF5_ROOT if exist "%ProgramFiles%\HDF_Group\HDF5\2.1.1\include\hdf5.h" set "HDF5_ROOT=%ProgramFiles%\HDF_Group\HDF5\2.1.1"
if not defined HDF5_ROOT if exist "%ProgramFiles%\HDF_Group\HDF5\2.1.0\include\hdf5.h" set "HDF5_ROOT=%ProgramFiles%\HDF_Group\HDF5\2.1.0"
if not defined HDF5_ROOT if exist "%ProgramFiles%\HDF_Group\HDF5\2.0.0\include\hdf5.h" set "HDF5_ROOT=%ProgramFiles%\HDF_Group\HDF5\2.0.0"
if not defined HDF5_ROOT if exist "%ProgramFiles%\HDF_Group\HDF5\1.14.6\include\hdf5.h" set "HDF5_ROOT=%ProgramFiles%\HDF_Group\HDF5\1.14.6"
if defined HDF5_ROOT set "PATH=%HDF5_ROOT%\bin;%PATH%"

if /I not "%~1"=="--with-compiler" exit /b 0
@REM CUDA temporary files can fail under a Windows username containing spaces.
@REM Change only the build/check process environment, never global TEMP/TMP.
if not defined GPE_CUDA_TEMP set "GPE_CUDA_TEMP=%PUBLIC%\KapitzaCudaTemp"
if not exist "%GPE_CUDA_TEMP%" mkdir "%GPE_CUDA_TEMP%" >nul 2>nul
if exist "%GPE_CUDA_TEMP%" set "TEMP=%GPE_CUDA_TEMP%"
if exist "%GPE_CUDA_TEMP%" set "TMP=%GPE_CUDA_TEMP%"
where cl.exe >nul 2>nul && exit /b 0

set "VS_INSTALL="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
if exist "%VSWHERE%" for /f "usebackq delims=" %%I in (`vswhere.exe -latest -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALL=%%I"
if not defined VS_INSTALL if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" set "VS_INSTALL=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools"
if not defined VS_INSTALL if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" set "VS_INSTALL=%ProgramFiles%\Microsoft Visual Studio\2022\Community"
if defined VS_INSTALL call "%VS_INSTALL%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>nul
exit /b 0
