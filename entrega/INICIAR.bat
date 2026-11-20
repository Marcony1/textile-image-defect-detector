@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Inspector de toallas - deje esta ventana abierta
if not exist "entorno\Scripts\python.exe" (
  echo Primero ejecute INSTALAR.
  pause
  exit /b 1
)
"entorno\Scripts\python.exe" programa\iniciar.py
pause
