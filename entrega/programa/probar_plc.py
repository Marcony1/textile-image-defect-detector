"""Simula un PLC: se conecta al inspector por Modbus TCP y muestra cada decisión.

Sirve para comprobar la conexión antes de programar el PLC real, y como ejemplo
de lo que el PLC debe hacer:

  1. Leer los registros 0-13 cada ~100-200 ms.
  2. Cuando el registro 2 (contador de resultados) cambia, hay una toalla nueva:
     leer la decisión (registro 3) y el ID de toalla (registros 4-5).
  3. Escribir un valor que cambie en el registro 20 (latido del PLC).

    python probar_plc.py                  # inspector en esta misma computadora
    python probar_plc.py 192.168.1.50 502 # inspector en otra computadora
"""
import socket
import struct
import sys
import time

DEC = {0: "sin resultados aún", 1: "OK", 2: "NO OK", 3: "ERROR de foto"}
DEFECT = {0: "", 1: "Mancha", 2: "Hilo", 3: "Costura", 4: "Tejido", 5: "Teñido", 6: "Empalme"}
STATUS = {0: "iniciando", 1: "listo", 2: "ERROR: no se cargó el modelo", 3: "inspección detenida"}


class ModbusClient:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.tid = 0

    def _call(self, pdu):
        self.tid = (self.tid + 1) % 65536
        self.sock.sendall(struct.pack(">HHHB", self.tid, 0, len(pdu) + 1, 1) + pdu)
        head = self._recv(7)
        _, _, length, _ = struct.unpack(">HHHB", head)
        resp = self._recv(length - 1)
        if resp[0] & 0x80:
            raise RuntimeError(f"Excepción Modbus {resp[1]}")
        return resp

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("conexión cerrada")
            buf += chunk
        return buf

    def read(self, addr, qty):
        resp = self._call(struct.pack(">BHH", 3, addr, qty))
        return list(struct.unpack(f">{qty}H", resp[2:2 + 2 * qty]))

    def write(self, addr, value):
        self._call(struct.pack(">BHH", 6, addr, value))


def s16(v):
    return v - 65536 if v >= 32768 else v


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 502
    print(f"Conectando a {host}:{port} (Modbus TCP)…  Ctrl+C para salir.\n")
    try:
        c = ModbusClient(host, port)
    except OSError as e:
        print(f"No se pudo conectar: {e}\n"
              "¿Está abierto INICIAR con  [plc] habilitar = si  en configuracion.ini?\n"
              "¿El firewall de Windows permite el puerto?")
        return 1
    last_seq, last_status, beat = None, None, 0
    while True:
        r = c.read(0, 14)
        beat = (beat + 1) % 65536
        c.write(20, beat)  # latido del PLC
        if r[1] != last_status:
            print(f"Estado del inspector: {STATUS.get(r[1], r[1])}  (latido {r[0]})")
            last_status = r[1]
        if last_seq is None:
            last_seq = r[2]
            print(f"Conectado. Resultados hasta ahora: OK {r[11]} · NO OK {r[12]} · errores {r[13]}\n")
        elif r[2] != last_seq:
            last_seq = r[2]
            tid = (r[4] << 16) | r[5]
            tipo = DEFECT.get(r[10], "")
            print(f"{time.strftime('%H:%M:%S')}  toalla {tid:>8}  {DEC.get(r[3], r[3]):<8}"
                  f"{'  ' + tipo if tipo else ''}  score {s16(r[6]) / 100:.2f} (umbral {s16(r[7]) / 100:.2f})"
                  f"  {r[8]} ms  cola {r[9]}")
        time.sleep(0.2)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nTerminado.")
    except (ConnectionError, OSError) as e:
        print(f"\nSe perdió la conexión: {e}")
