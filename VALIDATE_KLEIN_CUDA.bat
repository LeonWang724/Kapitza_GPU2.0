@echo off
setlocal
cd /d "%~dp0"
call "%~dp0LOAD_CUDA_ENV.bat"
"%GPE_PYTHON%" -m klein_tunneling.validate %*
set "VALIDATION_RESULT=%ERRORLEVEL%"
echo.
if "%VALIDATION_RESULT%"=="0" (
  echo KLEIN SCATTERING VALIDATION PASSED.
) else (
  echo KLEIN SCATTERING VALIDATION FAILED with exit code %VALIDATION_RESULT%.
)
if not defined GPE_NO_PAUSE pause
exit /b %VALIDATION_RESULT%
