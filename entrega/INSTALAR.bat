@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Instalar - Inspector de toallas
echo ================================================================
echo  Instalacion del inspector de toallas
echo  Necesita internet. Tarda entre 5 y 20 minutos (descarga ~3 GB).
echo ================================================================
echo.

rem ---- 1. Buscar Python 3.11 (o 3.10 / 3.12)
set "PY="
for %%V in (3.11 3.12 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo No se encontro Python 3.11.
  echo Instalelo desde https://www.python.org/downloads/release/python-3119/
  echo  - descargue "Windows installer (64-bit)"
  echo  - marque la casilla "Add python.exe to PATH" antes de instalar
  echo Despues vuelva a ejecutar INSTALAR.
  pause
  exit /b 1
)
echo [1/4] Python encontrado: %PY%

rem ---- 2. Entorno propio del programa (no modifica otros programas)
if not exist "entorno\Scripts\python.exe" (
  %PY% -m venv entorno || goto error
)
set "VPY=%~dp0entorno\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip --quiet || goto error
echo [2/4] Entorno creado en la carpeta "entorno"

rem ---- 3. PyTorch: version para tarjeta NVIDIA o para procesador
nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo [3/4] Tarjeta grafica NVIDIA detectada: se instala la version con GPU
  set "IDX=https://download.pytorch.org/whl/cu121"
) else (
  echo [3/4] Sin tarjeta NVIDIA: se instala la version para procesador - mas lenta
  set "IDX=https://download.pytorch.org/whl/cpu"
)
"%VPY%" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url %IDX% || goto error

rem ---- 4. Resto de librerias
echo [4/4] Instalando librerias...
"%VPY%" -m pip install -r programa\requirements.txt || goto error

echo.
"%VPY%" -c "import torch, timm, ultralytics, fastapi; print(' Prueba OK. Tarjeta grafica disponible:', 'SI' if torch.cuda.is_available() else 'NO (usara el procesador)')" || goto error
echo.
echo ================================================================
echo  Instalacion terminada. Ahora abra INICIAR.
echo ================================================================
pause
exit /b 0

:error
echo.
echo ================================================================
echo  Hubo un error durante la instalacion.
echo  Revise su conexion a internet y vuelva a ejecutar INSTALAR.
echo  Si se repite, envie una foto de esta ventana al equipo de soporte.
echo ================================================================
pause
exit /b 1
