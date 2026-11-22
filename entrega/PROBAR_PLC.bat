@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Probar conexion PLC - Inspector de toallas
set "VPY=%LOCALAPPDATA%\InspectorToallas\entorno\Scripts\python.exe"
if not exist "%VPY%" (
  echo Primero ejecute INSTALAR.
  pause
  exit /b 1
)
rem Simula un PLC que lee las decisiones del inspector por Modbus TCP.
rem Uso: PROBAR_PLC.bat            (inspector en esta computadora, puerto 502)
rem      PROBAR_PLC.bat 192.168.1.50 502
"%VPY%" programa\probar_plc.py %*
pause
