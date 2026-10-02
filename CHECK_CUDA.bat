@REM Codex CUDA Port: verify the Windows CUDA, build, HDF5, and Python environment.
@echo off
setlocal
cd /d "%~dp0"

call "%~dp0LOAD_CUDA_ENV.bat" --with-compiler
call :check
set "CHECK_RESULT=%ERRORLEVEL%"
echo.
if "%CHECK_RESULT%"=="0" (
  echo CHECK PASSED. The required build environment is available.
) else (
  echo CHECK FAILED with exit code %CHECK_RESULT%.
  echo Run SETUP_WINDOWS.bat to install or repair the dependencies.
)
if not defined GPE_NO_PAUSE pause
exit /b %CHECK_RESULT%

:check

echo === NVIDIA driver and GPU ===
nvidia-smi
if errorlevel 1 exit /b 1
echo.
echo === CUDA compiler ===
nvcc --version
if errorlevel 1 exit /b 1
echo.
echo === Ninja build tool ===
ninja --version
if errorlevel 1 exit /b 1
echo.
echo === CMake ===
cmake --version
if errorlevel 1 exit /b 1
echo.
echo === Visual Studio C++ x64 compiler ===
where cl.exe
if errorlevel 1 exit /b 1
echo.
echo === HDF5 development files ===
if not defined HDF5_ROOT (
  echo ERROR: HDF5_ROOT is not defined.
  exit /b 1
)
if not exist "%HDF5_ROOT%\include\hdf5.h" (
  echo ERROR: "%HDF5_ROOT%\include\hdf5.h" does not exist.
  exit /b 1
)
if not exist "%HDF5_ROOT%\lib" (
  echo ERROR: "%HDF5_ROOT%\lib" does not exist.
  exit /b 1
)
echo HDF5_ROOT=%HDF5_ROOT%
echo.
echo === Intel oneMKL CPU-reference files ===
if not defined MKL_DIR (
  echo ERROR: MKL_DIR is not defined. Rerun SETUP_WINDOWS.bat.
  exit /b 1
)
if not exist "%MKL_DIR%\MKLConfig.cmake" (
  echo ERROR: "%MKL_DIR%\MKLConfig.cmake" does not exist.
  exit /b 1
)
echo MKLROOT=%MKLROOT%
echo.
echo === Python orchestration dependencies ===
"%GPE_PYTHON%" --version
if errorlevel 1 exit /b 1
"%GPE_PYTHON%" -c "import sys; import numpy, scipy, tables, matplotlib; print('Python dependencies: OK'); print('Interpreter:', sys.executable)"
if errorlevel 1 exit /b 1
"%GPE_PYTHON%" -c "import sys; sys.path.insert(0, r'phase_diagram\simulation_core'); import create_initial_state_function, compact_sweep, phase_metric; print('Phase-diagram modules: OK')"
if errorlevel 1 exit /b 1
"%GPE_PYTHON%" -c "import os, pathlib, tempfile; p=os.environ['TEMP']; print('CUDA temporary directory:', p); assert ' ' not in p, 'Use GPE_CUDA_TEMP to select a writable path without spaces'; f=tempfile.TemporaryFile(dir=p); f.close()"
if errorlevel 1 exit /b 1

if exist "%~dp0build\bin\gpe1d_cuda.exe" (
  echo.
  echo === Native solver build ===
  "%~dp0build\bin\gpe1d_cuda.exe" --version-json
  if errorlevel 1 exit /b 1
  "%~dp0build\bin\gpe1d_cuda.exe" --device-info 0
  if errorlevel 1 exit /b 1
) else (
  echo WARNING: build\bin\gpe1d_cuda.exe does not exist yet.
)
if not exist "%~dp0build\bin\gpe1d_cpu_reference.exe" (
  echo WARNING: build\bin\gpe1d_cpu_reference.exe does not exist yet. Run BUILD_CUDA.bat.
)
exit /b 0
