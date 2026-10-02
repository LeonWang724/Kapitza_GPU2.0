@echo off
setlocal
cd /d "%~dp0"
call "%~dp0LOAD_CUDA_ENV.bat"
"%GPE_PYTHON%" validation\validate_compact.py %*
set "VALIDATION_RESULT=%ERRORLEVEL%"
echo.
if "%VALIDATION_RESULT%"=="0" (
  echo COMPACT OUTPUT VALIDATION PASSED.
) else (
  echo COMPACT OUTPUT VALIDATION FAILED with exit code %VALIDATION_RESULT%.
)
if not defined GPE_NO_PAUSE pause
exit /b %VALIDATION_RESULT%
