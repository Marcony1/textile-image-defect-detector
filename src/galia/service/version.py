"""Program version and changelog (single source: static/cambios.json, newest first)."""
import json
from pathlib import Path

CHANGES = json.loads((Path(__file__).parent / "static" / "cambios.json").read_text(encoding="utf-8"))
VERSION = CHANGES[0]["version"]
