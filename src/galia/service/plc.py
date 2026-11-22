"""Modbus TCP server so a line PLC can read each towel's decision.

The inspector is the Modbus *server* (slave); the PLC is the *client* (master)
and polls the holding registers (function 3; function 4 returns the same table).
No extra library: only the small subset of Modbus that PLCs use for this.

Register map (0-based addresses; some PLCs show them as 40001, 40002, ...):

  0  heartbeat            +1 every second while the program runs (watchdog)
  1  status               0 starting, 1 ready, 2 model failed to load, 3 inspection stopped
  2  result counter       +1 for every towel analysed (wraps at 65535): "a new result is ready"
  3  last decision        0 none yet, 1 OK, 2 NO OK, 3 photo could not be read
  4  towel id (high word) id = number in the photo file name (e.g. toalla_000123.jpg -> 123)
  5  towel id (low word)
  6  score x 100          signed; the towel is NO OK when score >= threshold
  7  threshold x 100      signed
  8  analysis time (ms)
  9  photos waiting in the queue
 10  main defect type     0 none / not classified, 1 Mancha, 2 Hilo, 3 Costura, 4 Tejido, 5 Tenido, 6 Empalme
 11  OK count             since the program started (wraps)
 12  NO OK count
 13  error count
 20  PLC heartbeat        WRITTEN by the PLC (any changing value): shows "PLC conectado" on the dashboard
 21  free register        written by the PLC, shown on the dashboard (e.g. a line state code)

Writes are only accepted at addresses 20-29.
"""
from __future__ import annotations

import logging
import re
import socketserver
import struct
import threading
import time
from datetime import datetime

from .. import config as C

log = logging.getLogger("galia.plc")
N_REGS = 64
WRITABLE = range(20, 30)
DECISION_CODE = {"OK": 1, "NO OK": 2, "ERROR": 3}


def towel_id_from_name(name: str, fallback: int) -> int:
    nums = re.findall(r"\d+", name.rsplit(".", 1)[0])
    return int(nums[-1]) % (2 ** 32) if nums else fallback % (2 ** 32)


def _s16(v: float) -> int:
    v = int(round(max(-32768, min(32767, v))))
    return v & 0xFFFF


class PLCServer:
    def __init__(self, port: int = 502, host: str = "0.0.0.0", allowed: set[str] | None = None):
        self.port, self.host, self.allowed = port, host, allowed or set()
        self.regs = [0] * N_REGS
        self.lock = threading.Lock()
        self.started = None
        self.error: str | None = None
        self.last_read: str | None = None
        self.last_client: str | None = None
        self.plc_hb_changed: float | None = None
        self.counts = {"OK": 0, "NO OK": 0, "ERROR": 0}
        self.seq = 0
        self._srv = None

    # ------------------------------------------------------------ registers
    def set_status(self, code: int):
        with self.lock:
            self.regs[1] = code

    def publish(self, row: dict, queue: int = 0):
        """Called by the worker after every towel."""
        dec = row.get("decision", "ERROR")
        self.counts[dec] = self.counts.get(dec, 0) + 1
        self.seq = (self.seq + 1) % 65536
        tid = towel_id_from_name(row.get("file", ""), row.get("id", 0))
        top = row.get("top_defect")
        with self.lock:
            r = self.regs
            r[3] = DECISION_CODE.get(dec, 3)
            r[4], r[5] = tid >> 16, tid & 0xFFFF
            r[6] = _s16((row.get("score") or 0) * 100)
            r[7] = _s16((row.get("threshold") or 0) * 100)
            r[8] = int(min(65535, row.get("elapsed_ms") or 0))
            r[9] = min(65535, queue)
            r[10] = C.CLASSES.index(top) if top in C.CLASSES else 0
            r[11], r[12], r[13] = (self.counts["OK"] % 65536, self.counts["NO OK"] % 65536,
                                   self.counts["ERROR"] % 65536)
            r[2] = self.seq  # last: the PLC sees the counter change only when the rest is consistent

    def state(self) -> dict:
        now = time.time()
        with self.lock:
            regs = list(self.regs[:22])
        return dict(enabled=True, port=self.port, running=self._srv is not None and self.error is None,
                    error=self.error, last_client=self.last_client, last_read=self.last_read,
                    plc_connected=bool(self.plc_hb_changed and now - self.plc_hb_changed < 10),
                    plc_value=regs[21], registers=regs)

    # --------------------------------------------------------------- server
    def start(self):
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                ip = self.client_address[0]
                if outer.allowed and ip not in outer.allowed:
                    log.warning("Modbus: connection from %s refused (not in ip_permitidas)", ip)
                    return
                sock = self.request
                sock.settimeout(60)
                while True:
                    try:
                        head = _recv(sock, 7)
                        if not head:
                            return
                        tid, pid, length, unit = struct.unpack(">HHHB", head)
                        pdu = _recv(sock, length - 1)
                        if pdu is None:
                            return
                    except OSError:
                        return
                    resp = outer._handle_pdu(pdu, ip)
                    sock.sendall(struct.pack(">HHHB", tid, 0, len(resp) + 1, unit) + resp)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        try:
            self._srv = Server((self.host, self.port), Handler)
        except OSError as e:
            self.error = f"No se pudo abrir el puerto Modbus {self.port}: {e}"
            log.error(self.error)
            self._srv = None
            return False
        threading.Thread(target=self._srv.serve_forever, daemon=True, name="modbus").start()
        threading.Thread(target=self._heartbeat, daemon=True, name="modbus-heartbeat").start()
        self.started = datetime.now().isoformat(timespec="seconds")
        log.info("Modbus TCP server on port %d", self.port)
        return True

    def stop(self):
        if self._srv:
            self._srv.shutdown()
            self._srv = None

    status_fn = None  # callable -> status code (0 starting, 1 ready, 2 model error, 3 stopped)

    def _heartbeat(self):
        while self._srv:
            time.sleep(1)
            code = self.status_fn() if self.status_fn else 1
            with self.lock:
                self.regs[0] = (self.regs[0] + 1) % 65536
                self.regs[1] = code

    def _handle_pdu(self, pdu: bytes, ip: str) -> bytes:
        fc = pdu[0]
        try:
            if fc in (3, 4):
                addr, qty = struct.unpack(">HH", pdu[1:5])
                if not 1 <= qty <= 125 or addr + qty > N_REGS:
                    return bytes([fc | 0x80, 2])
                with self.lock:
                    vals = self.regs[addr:addr + qty]
                self.last_read, self.last_client = datetime.now().isoformat(timespec="seconds"), ip
                return bytes([fc, 2 * qty]) + struct.pack(f">{qty}H", *vals)
            if fc == 6:
                addr, val = struct.unpack(">HH", pdu[1:5])
                if addr not in WRITABLE:
                    return bytes([fc | 0x80, 2])
                self._write(addr, [val], ip)
                return pdu[:5]
            if fc == 16:
                addr, qty, nbytes = struct.unpack(">HHB", pdu[1:6])
                if qty < 1 or any(a not in WRITABLE for a in range(addr, addr + qty)) or nbytes != 2 * qty:
                    return bytes([fc | 0x80, 2])
                self._write(addr, list(struct.unpack(f">{qty}H", pdu[6:6 + nbytes])), ip)
                return pdu[:5]
        except struct.error:
            return bytes([fc | 0x80, 3])
        return bytes([fc | 0x80, 1])  # illegal function

    def _write(self, addr: int, vals: list[int], ip: str):
        with self.lock:
            for i, v in enumerate(vals):
                if addr + i == 20 and self.regs[20] != v:
                    self.plc_hb_changed = time.time()
                self.regs[addr + i] = v
        self.last_client = ip


def _recv(sock, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None if not buf else None
        buf += chunk
    return buf
