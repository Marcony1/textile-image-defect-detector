@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Calibrar - Inspector de toallas
if not exist "entorno\Scripts\python.exe" (
  echo Primero ejecute INSTALAR.
  pause
  exit /b 1
)
rem Uso: copie 30-60 fotos de toallas BUENAS a la carpeta "calibracion" y ejecute este archivo.
"entorno\Scripts\python.exe" programa\calibrar.py %*
pause
