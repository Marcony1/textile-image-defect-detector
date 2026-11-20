#!/usr/bin/env bash
# Instalación en Linux (Ubuntu/Debian, PC normal). Uso: bash instalar.sh
set -e
cd "$(dirname "$0")"
PY=""
for v in python3.11 python3.12 python3.10 python3; do
  if command -v $v >/dev/null && $v -c 'import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)'; then PY=$v; break; fi
done
if [ -z "$PY" ]; then echo "Instale Python 3.11 (sudo apt install python3.11 python3.11-venv)"; exit 1; fi
echo "[1/4] Python: $PY"
[ -x entorno/bin/python ] || $PY -m venv entorno
VPY=entorno/bin/python
$VPY -m pip install --upgrade pip --quiet
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  echo "[2/4] Tarjeta NVIDIA detectada"; IDX=https://download.pytorch.org/whl/cu121
else
  echo "[2/4] Sin tarjeta NVIDIA: versión para procesador"; IDX=https://download.pytorch.org/whl/cpu
fi
echo "[3/4] PyTorch"; $VPY -m pip install torch==2.5.1 torchvision==0.20.1 --index-url $IDX
echo "[4/4] Librerías"; $VPY -m pip install -r programa/requirements.txt
$VPY -c "import torch, timm, ultralytics, fastapi; print('Prueba OK. GPU:', torch.cuda.is_available())"
echo "Instalación terminada. Ejecute: bash iniciar.sh"
