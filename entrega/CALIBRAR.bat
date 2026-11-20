@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Calibrar - Inspector de toallas
set "VPY=%LOCALAPPDATA%\InspectorToallas\entorno\Scripts\python.exe"
if not exist "%VPY%" (
  echo Primero ejecute INSTALAR.
  pause
  exit /b 1
)
rem Uso: copie 30-60 fotos de toallas BUENAS a la carpeta "calibracion" y ejecute este archivo.
"%VPY%" programa\calibrar.py %*
pause
