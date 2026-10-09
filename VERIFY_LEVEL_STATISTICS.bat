@REM Check the level-statistics module (r_n, <r>, <r^2>, eta) and print one example point.
@echo off
setlocal
cd /d "%~dp0"
call "%~dp0LOAD_CUDA_ENV.bat"
"%GPE_PYTHON%" phase_diagram\simulation_core\level_statistics.py --verify %*
set "VERIFY_RESULT=%ERRORLEVEL%"
echo.
if "%VERIFY_RESULT%"=="0" (
  echo LEVEL STATISTICS CHECKS PASSED.
) else (
  echo LEVEL STATISTICS CHECKS FAILED with exit code %VERIFY_RESULT%.
)
if not defined GPE_NO_PAUSE pause
exit /b %VERIFY_RESULT%
