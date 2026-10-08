@echo off
setlocal
cd /d "%~dp0"
call "%~dp0LOAD_CUDA_ENV.bat"
"%GPE_PYTHON%" -m klein_tunneling.analyze %*
exit /b %errorlevel%
