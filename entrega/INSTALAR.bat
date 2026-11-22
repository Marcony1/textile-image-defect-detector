@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Instalar - Inspector de toallas
echo ================================================================
echo  Instalacion del inspector de toallas
echo  Necesita internet. Tarda entre 5 y 20 minutos (descarga ~3 GB).
echo ================================================================
echo.

rem ---- 0. No correr desde dentro del zip (Windows lo abre en una carpeta temporal)
echo %~dp0 | findstr /i /c:".zip" /c:"Temp1_" /c:"Temp2_" >nul
if %errorlevel%==0 goto en_zip
if not exist "programa\iniciar.py" goto en_zip

rem ---- 1. Buscar Python 3.11 (la version con la que se probo todo)
set "BASE=%LOCALAPPDATA%\InspectorToallas"
set "PYEXE="
set "PYOK=(3, 11)"
call :buscar
if defined PYEXE goto py_ok

echo No se encontro Python 3.11 en esta computadora.
echo Se descargara una copia PORTATIL de Python 3.11 (17 MB) en:
echo   %BASE%\python311
echo No se ejecuta ningun instalador, no necesita permisos de administrador
echo y no modifica otras versiones de Python que ya existan.
echo.
call :python_portatil
if defined PYEXE goto py_ok

echo.
echo No se pudo usar la copia portatil. Probando con winget, el instalador de Windows...
where winget >nul 2>nul
if %errorlevel%==0 winget install -e --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
call :buscar
if defined PYEXE goto py_ok

rem ultimo recurso: Python 3.12 si ya existe
set "PYOK=(3, 11), (3, 12)"
call :buscar
if defined PYEXE goto py_ok
goto sin_python

:py_ok
echo [1/4] Python encontrado: %PYEXE%

rem ---- 2. Entorno propio del programa (no modifica otros programas)
rem Se crea en una ruta corta y fija: PyTorch trae archivos con nombres muy largos y Windows
rem no acepta rutas de mas de 260 caracteres (falla si la carpeta esta en OneDrive o muy anidada).
set "ENTORNO=%BASE%\entorno"
echo %~dp0 | findstr /i "OneDrive" >nul
if %errorlevel%==0 (
  echo.
  echo AVISO: esta carpeta esta dentro de OneDrive. Funciona, pero se recomienda
  echo copiarla a un disco local, por ejemplo C:\Inspector_Toallas, para que OneDrive
  echo no sincronice las fotos de la linea.
  echo.
)
rem Un entorno a medias o creado con otra version de Python se vuelve a crear
if exist "%ENTORNO%\Scripts\python.exe" (
  "%ENTORNO%\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3, 11), (3, 12)] else 1)" >nul 2>nul
  if errorlevel 1 rmdir /s /q "%ENTORNO%"
)
if exist "%ENTORNO%" if not exist "%ENTORNO%\Scripts\python.exe" rmdir /s /q "%ENTORNO%"
if not exist "%ENTORNO%\Scripts\python.exe" (
  "%PYEXE%" -m venv "%ENTORNO%" || goto error
)
set "VPY=%ENTORNO%\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip --quiet || goto error
echo [2/4] Entorno listo en %ENTORNO%

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

:en_zip
echo ================================================================
echo  Este programa se esta ejecutando DESDE DENTRO DEL ZIP.
echo  Primero hay que descomprimirlo:
echo   1. Cierre esta ventana.
echo   2. Clic derecho sobre Inspector_Toallas.zip  -  "Extraer todo..."
echo   3. Elija una carpeta local, por ejemplo C:\  (quedara C:\Inspector_Toallas)
echo   4. Abra la carpeta descomprimida y ejecute INSTALAR de nuevo.
echo ================================================================
pause
exit /b 1

:python_portatil
rem Python 3.11 portatil oficial (paquete NuGet de python.org): un zip, no un instalador.
rem OJO: el entorno del programa depende de esta carpeta; no la borre.
set "PYPKG=https://www.nuget.org/api/v2/package/python/3.11.9"
if not exist "%BASE%\descargas" mkdir "%BASE%\descargas"
set "PYZIP=%BASE%\descargas\python311_%RANDOM%%RANDOM%.zip"
curl.exe -L --fail --retry 3 -o "%PYZIP%" "%PYPKG%"
if not exist "%PYZIP%" powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri '%PYPKG%' -OutFile '%PYZIP%'"
if not exist "%PYZIP%" exit /b 0
if exist "%BASE%\python311" rmdir /s /q "%BASE%\python311"
powershell -NoProfile -Command "Expand-Archive -LiteralPath '%PYZIP%' -DestinationPath '%BASE%\python311' -Force"
del "%PYZIP%" >nul 2>nul
call :probar "%BASE%\python311\tools\python.exe"
exit /b 0

:sin_python
echo ================================================================
echo  No se pudo conseguir Python 3.11 automaticamente.
echo  Causas comunes: sin internet, un proxy de la empresa, o el
echo  antivirus bloqueando la descarga (revise sus avisos).
echo  Instalelo a mano desde:
echo    https://www.python.org/downloads/release/python-3119/
echo  - descargue "Windows installer (64-bit)" y ejecutelo
echo  - marque la casilla "Add python.exe to PATH" antes de instalar
echo  Despues vuelva a ejecutar INSTALAR.
echo ================================================================
pause
exit /b 1

:error
echo.
echo ================================================================
echo  Hubo un error durante la instalacion.
echo  Revise su conexion a internet y vuelva a ejecutar INSTALAR.
echo  Si se repite, envie una foto de esta ventana al equipo de soporte.
echo ================================================================
pause
exit /b 1

:buscar
rem Busca un Python aceptable (PYOK) en: copia portatil, lanzador py, PATH, python.org, Miniconda/Anaconda
call :probar "%LOCALAPPDATA%\InspectorToallas\python311\tools\python.exe"
for %%V in (3.11 3.12) do (
  for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)" 2^>nul') do call :probar "%%P"
)
for /f "delims=" %%P in ('where python 2^>nul') do call :probar "%%P"
for %%D in (Python311 Python312) do (
  call :probar "%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
  call :probar "%ProgramFiles%\%%D\python.exe"
)
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
  call :probar "%%~D\python.exe"
)
exit /b 0

:probar
rem Acepta el Python indicado si existe, es real (no el acceso directo de la Tienda) y su version esta en PYOK
if defined PYEXE exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys; sys.exit(0 if sys.version_info[:2] in [%PYOK%] else 1)" >nul 2>nul
if %errorlevel%==0 set "PYEXE=%~1"
exit /b 0
