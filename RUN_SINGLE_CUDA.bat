@REM Codex CUDA Port: run one config with the compiled CUDA executable.
@echo off
setlocal
cd /d "%~dp0"
call "%~dp0LOAD_CUDA_ENV.bat"
"%GPE_PYTHON%" phase_diagram\simulation_core\run_single_CUDA.py %*
exit /b %errorlevel%
