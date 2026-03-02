@echo off
cd /d "%~dp0"
echo ============================================
echo   PyTorch / CUDA Manager for SwarmUI
echo   Opening http://localhost:9090
echo ============================================
echo.
dlbackend\comfy\python_embeded\python.exe pytorch_manager.py
pause
