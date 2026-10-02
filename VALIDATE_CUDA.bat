@REM Codex CUDA Port: run short-time checkpoint and 3x3 phase-diagram validation.
@echo off
setlocal
cd /d "%~dp0"
call "%~dp0LOAD_CUDA_ENV.bat"
"%GPE_PYTHON%" validation\run_validation.py %*
set "VALIDATION_RESULT=%ERRORLEVEL%"
echo.
if "%VALIDATION_RESULT%"=="0" (
  echo VALIDATION COMPLETED SUCCESSFULLY.
) else (
  echo VALIDATION FAILED with exit code %VALIDATION_RESULT%.
)
if not defined GPE_NO_PAUSE pause
exit /b %VALIDATION_RESULT%
