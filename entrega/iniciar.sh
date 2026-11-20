#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -x entorno/bin/python ] || { echo "Primero ejecute: bash instalar.sh"; exit 1; }
exec entorno/bin/python programa/iniciar.py
