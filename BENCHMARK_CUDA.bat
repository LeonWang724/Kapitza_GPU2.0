@REM Time the solver on this GPU and recommend POINTS_PER_BATCH.
@echo off
setlocal
cd /d "%~dp0"
call "%~dp0LOAD_CUDA_ENV.bat"
"%GPE_PYTHON%" validation\benchmark_speed.py %*
set "BENCHMARK_RESULT=%ERRORLEVEL%"
echo.
if not "%BENCHMARK_RESULT%"=="0" echo BENCHMARK FAILED with exit code %BENCHMARK_RESULT%.
if not defined GPE_NO_PAUSE pause
exit /b %BENCHMARK_RESULT%
