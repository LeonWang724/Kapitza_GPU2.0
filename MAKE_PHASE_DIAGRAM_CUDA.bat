@REM Codex CUDA Port: analyze one completed manifest without a second parameter grid.
@echo off
setlocal
cd /d "%~dp0"
call "%~dp0LOAD_CUDA_ENV.bat"
"%GPE_PYTHON%" phase_diagram\simulation_core\make_phase_diagram_CUDA.py %*
exit /b %errorlevel%
