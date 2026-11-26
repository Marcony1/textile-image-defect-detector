# Regenera los archivos derivados del repositorio. Todo es Python: no hace falta un asistente de IA.
#
#   make              lista de comandos
#   make todo         diagramas + curva + reportes + propuesta + zip
#   make zip          solo el paquete para la empresa
#   make version V=1.4.0 T="Título" C="Cambio 1|Cambio 2"   registra una versión nueva en cambios.json
#
# Sin make (Windows sin herramientas):  python scripts/actualizar.py [pasos...]
# Use el entorno del proyecto:          conda activate galia_v2   (o PY=ruta/a/python make ...)

PY ?= python

.PHONY: ayuda todo diagramas curva reportes propuesta zip version servir

ayuda:
	@echo "make todo        diagramas + curva + reportes + propuesta + zip"
	@echo "make diagramas   docs/diagramas/*.svg (red neuronal)"
	@echo "make curva       models/curva_operacion.json (Exigencia y probabilidad orientativa)"
	@echo "make reportes    docs/reportes/*.html y *.pdf desde docs/reportes/fuente/"
	@echo "make propuesta   docs/propuesta_estacion_captura_v*.pdf"
	@echo "make zip         dist/Inspector_Toallas_v<version>.zip"
	@echo "make version V=1.4.0 T=\"Titulo\" C=\"Cambio 1|Cambio 2\"   nueva version en cambios.json"
	@echo "make servir      tablero local con la demo en http://localhost:8000"
	@$(PY) scripts/nueva_version.py --ver

todo:
	$(PY) scripts/actualizar.py

diagramas curva propuesta zip:
	$(PY) scripts/actualizar.py $@

# los reportes incrustan los diagramas: si cambió el generador, se regeneran primero
reportes:
	$(PY) scripts/actualizar.py diagramas reportes

version:
	@test -n "$(V)" || (echo "Falta V=MAYOR.MENOR.PARCHE"; exit 1)
	$(PY) -c "import subprocess,sys; sys.exit(subprocess.call([sys.executable,'scripts/nueva_version.py','$(V)','$(T)',*'$(C)'.split('|')]))"

servir:
	$(PY) scripts/serve.py --operating-point recall_95
