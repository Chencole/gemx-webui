@echo off
setlocal
cd /d "%~dp0"
set "MOTION_DEVICE=cuda"
if not defined MOTION_BACKEND_ROOT set "MOTION_BACKEND_ROOT=%~dp0..\GEM-X"
uv run --locked python app.py %*
