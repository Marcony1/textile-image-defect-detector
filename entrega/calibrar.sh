#!/usr/bin/env bash
# Copie 30-60 fotos de toallas BUENAS a la carpeta "calibracion" y ejecute: bash calibrar.sh
cd "$(dirname "$0")"
[ -x entorno/bin/python ] || { echo "Primero ejecute: bash instalar.sh"; exit 1; }
exec entorno/bin/python programa/calibrar.py "$@"
