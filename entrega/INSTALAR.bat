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

rem ---- 1. Python 3.11 portatil INCLUIDO en el paquete (carpeta python311): no se descarga nada
rem Se copia a una carpeta fija para que el programa siga funcionando aunque esta carpeta se mueva o se borre.
set "BASE=%LOCALAPPDATA%\InspectorToallas"
set "PYEXE="
set "PYOK=(3, 11)"
call :probar "%BASE%\python311\tools\python.exe"
if defined PYEXE goto py_ok
if not exist "python311\tools\python.exe" goto sin_incluido
echo Copiando Python 3.11 incluido en el paquete a %BASE%\python311 ...
if not exist "%BASE%" mkdir "%BASE%"
xcopy "python311" "%BASE%\python311\" /e /i /q /y >nul
call :probar "%BASE%\python311\tools\python.exe"
if defined PYEXE goto py_ok

:sin_incluido
rem Respaldo: un Python 3.11 (o 3.12) ya instalado en la computadora
call :buscar
if defined PYEXE goto py_ok
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

rem ---- 3. PyTorch segun la tarjeta grafica
rem Cada version de PyTorch trae codigo solo para ciertas generaciones de tarjetas NVIDIA:
rem   CUDA 12.1 (PyTorch 2.5.1): de la serie GTX 900 a la RTX 40
rem   CUDA 12.8 (PyTorch 2.7.1): RTX 50 (arquitectura 12.x, "Blackwell") y anteriores recientes
rem Se elige por la arquitectura y despues se PRUEBA en la tarjeta; si falla, se prueba la otra y al final el procesador.
set "T_CU121=torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121"
set "T_CU128=torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128"
set "T_CPU=torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu"
set "GPUOK=0"
nvidia-smi >nul 2>nul
if errorlevel 1 goto torch_cpu
set "GPUNAME=desconocida"
set "GPUCC=0.0"
for /f "tokens=1,2 delims=," %%A in ('nvidia-smi --query-gpu^=name^,compute_cap --format^=csv^,noheader 2^>nul') do (
  set "GPUNAME=%%A"
  set "GPUCC=%%B"
)
set "CCMAJ=0"
for /f "tokens=1 delims=. " %%M in ("%GPUCC%") do set "CCMAJ=%%M"
echo [3/4] Tarjeta grafica: %GPUNAME% - arquitectura %GPUCC%
set "PRIMERO=%T_CU121%"
set "SEGUNDO=%T_CU128%"
if %CCMAJ% GEQ 12 set "PRIMERO=%T_CU128%"
if %CCMAJ% GEQ 12 set "SEGUNDO=%T_CU121%"
if %CCMAJ% GEQ 1 if %CCMAJ% LSS 5 goto torch_vieja

echo Instalando PyTorch para la tarjeta grafica...
"%VPY%" -m pip install %PRIMERO% || goto error
call :probar_gpu
if "%GPUOK%"=="1" goto torch_ok
echo.
echo La version instalada no funciona con esta tarjeta. Probando la otra version de PyTorch...
"%VPY%" -m pip install %SEGUNDO% || goto error
call :probar_gpu
if "%GPUOK%"=="1" goto torch_ok
echo.
echo AVISO: la tarjeta grafica no funciona con PyTorch (driver antiguo o tarjeta no compatible).
echo Se usara el procesador: funciona, pero mas lento. Actualizar el driver de NVIDIA y
echo volver a ejecutar INSTALAR puede resolverlo.
goto torch_cpu

:torch_vieja
echo AVISO: esta tarjeta grafica es demasiado antigua para PyTorch; se usara el procesador.

:torch_cpu
echo [3/4] Se instala PyTorch para procesador - mas lento
"%VPY%" -m pip install %T_CPU% || goto error

:torch_ok
if "%GPUOK%"=="1" echo        Tarjeta grafica verificada: funciona correctamente.

rem ---- 4. Resto de librerias
echo [4/4] Instalando librerias...
"%VPY%" -m pip install -r programa\requirements.txt || goto error

echo.
"%VPY%" -c "import torch, timm, ultralytics, fastapi; import sys; sys.path.insert(0, 'programa'); from galia.models import has_gpu; print(' Prueba OK. PyTorch', torch.__version__, '- tarjeta grafica en uso:', 'SI' if has_gpu() else 'NO (usara el procesador)')" || goto error
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

:sin_python
echo ================================================================
echo  No se encontro Python 3.11.
echo  La carpeta "python311" del paquete falta o el antivirus la bloqueo.
echo  - Descargue y descomprima de nuevo el zip completo, o
echo  - revise el baul / cuarentena del antivirus y restaure los archivos.
echo  Si no es posible, instale Python 3.11 a mano desde:
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

:probar_gpu
rem Convolucion real en la tarjeta: detecta "no kernel image is available" y drivers viejos
set "GPUOK=0"
"%VPY%" -c "import torch, torch.nn.functional as F; x = torch.randn(1, 3, 16, 16, device='cuda'); F.conv2d(x, torch.randn(4, 3, 3, 3, device='cuda')).sum().item()" >nul 2>nul
if %errorlevel%==0 set "GPUOK=1"
exit /b 0

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
