@REM Check the fused and batched solver loops against the original loop on this GPU.
@echo off
setlocal
cd /d "%~dp0"
call "%~dp0LOAD_CUDA_ENV.bat"
"%GPE_PYTHON%" validation\validate_fast_solver.py %*
set "VALIDATION_RESULT=%ERRORLEVEL%"
echo.
if "%VALIDATION_RESULT%"=="0" (
  echo FAST SOLVER VALIDATION PASSED.
) else (
  echo FAST SOLVER VALIDATION FAILED with exit code %VALIDATION_RESULT%.
)
if not defined GPE_NO_PAUSE pause
exit /b %VALIDATION_RESULT%
