@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Inspector de toallas - deje esta ventana abierta
set "VPY=%LOCALAPPDATA%\InspectorToallas\entorno\Scripts\python.exe"
if not exist "%VPY%" (
  echo Primero ejecute INSTALAR.
  pause
  exit /b 1
)
"%VPY%" programa\iniciar.py
pause
