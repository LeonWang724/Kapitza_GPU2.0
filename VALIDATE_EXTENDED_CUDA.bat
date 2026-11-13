@REM Check the 6.0 extended-metric sums of this GPU against NumPy.
@echo off
setlocal
cd /d "%~dp0"
call "%~dp0LOAD_CUDA_ENV.bat"
"%GPE_PYTHON%" validation\validate_extended.py %*
set "VALIDATION_RESULT=%ERRORLEVEL%"
echo.
if "%VALIDATION_RESULT%"=="0" (
  echo EXTENDED METRICS VALIDATION PASSED.
) else (
  echo EXTENDED METRICS VALIDATION FAILED with exit code %VALIDATION_RESULT%.
)
if not defined GPE_NO_PAUSE pause
exit /b %VALIDATION_RESULT%
