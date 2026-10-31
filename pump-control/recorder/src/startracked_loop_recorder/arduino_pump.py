"""Exclusive, watchdog-safe serial owner for the parallel Arduino pump sink."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Protocol

from .config import CAMERAS


STARTUP_MARKER = "MULTICAM_PUMP_READY"
PROTOCOL = "PUMPS:<mask>\\n"
VALID_MASK = 0x0F


class SerialPort(Protocol):
    @property
    def in_waiting(self) -> int: ...

    def write(self, data: bytes) -> int: ...

    def read(self, size: int) -> bytes: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...

def encode_mask(mask: int) -> bytes:
    if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask > VALID_MASK:
        raise ValueError("pump mask must be an integer from 0 through 15")
    return f"PUMPS:{mask}\n".encode("ascii")


def default_serial_factory(path: str, baud: int) -> SerialPort:
    import serial

    return serial.Serial(
        path,
        baud,
        timeout=0,
        write_timeout=0.2,
        exclusive=True,
    )


class ArduinoPumpWorker(threading.Thread):
    """Own the serial port and coalesce authoritative logical demand updates."""

    def __init__(
        self,
        settings: Callable[[], dict[str, Any]],
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        serial_factory: Callable[[str, int], SerialPort] = default_serial_factory,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(name="arduino-pump-serial", daemon=True)
        self.settings = settings
        self.event_callback = event_callback
        self.serial_factory = serial_factory
        self.clock = clock
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop_requested = threading.Event()
        self._zero_confirmed = threading.Event()
        self._serial: SerialPort | None = None
        self._line_buffer = bytearray()
        self._session_token = ""
        self._session_armed = False
        self._camera_demand = {camera: False for camera in CAMERAS}
        self._desired_mask = 0
        self._confirmed_mask: int | None = None
        self._connected = False
        self._ready = False
        self._startup_seen = False
        self._zero_handshake = False
        self._watchdog = False
        self._last_event = ""
        self._last_error = ""
        self._last_write_at: float | None = None
        self._last_sent_mask: int | None = None
        self._reconnect_count = 0

    def set_session(self, armed: bool, session_token: str = "") -> None:
        config = self.settings()
        with self._lock:
            self._session_armed = bool(armed)
            self._session_token = session_token if armed else ""
            if not armed:
                self._camera_demand = {camera: False for camera in CAMERAS}
            self._refresh_mask_locked(config)
        self._wake.set()

    def set_camera_demand(self, camera: str, desired: bool, session_token: str) -> None:
        if camera not in self._camera_demand:
            return
        config = self.settings()
        with self._lock:
            if not self._session_armed or not session_token or session_token != self._session_token:
                return
            self._camera_demand[camera] = bool(desired)
            self._refresh_mask_locked(config)
        self._wake.set()

    def _refresh_mask_locked(self, config: dict[str, Any]) -> None:
        mapping = config.get("mapping", {})
        if not self._session_armed or not config.get("enabled", False):
            self._desired_mask = 0
        else:
            self._desired_mask = 0
            for camera, demanded in self._camera_demand.items():
                if demanded:
                    self._desired_mask |= int(mapping.get(camera, 0))
        if self._desired_mask == 0:
            if self._connected and self._confirmed_mask == 0:
                self._zero_confirmed.set()
            else:
                self._zero_confirmed.clear()

    def desired_mask(self) -> int:
        with self._lock:
            return self._desired_mask

    def wait_for_zero(self, timeout: float = 0.45) -> bool:
        return self._zero_confirmed.wait(timeout)

    def close(self, timeout: float = 3.0) -> None:
        self.set_session(False)
        self._wake.set()
        if self.ident is not None and self.is_alive():
            self._zero_confirmed.wait(timeout=0.45)
        self._stop_requested.set()
        self._wake.set()
        if self.ident is not None:
            self.join(timeout=timeout)

    def status(self) -> dict[str, Any]:
        config = self.settings()
        with self._lock:
            confirmed = self._confirmed_mask
            desired = self._desired_mask
            return {
                "enabled": bool(config.get("enabled", False)),
                "serial_path": config.get("serial_path", ""),
                "baud": config.get("baud", 115200),
                "heartbeat_seconds": config.get("heartbeat_seconds", 0.15),
                "protocol": PROTOCOL,
                "mapping": dict(config.get("mapping", {})),
                "connected": self._connected,
                "ready": self._ready,
                "watchdog": self._watchdog,
                "desired_mask": desired,
                "confirmed_mask": confirmed,
                "agreement": confirmed == desired if confirmed is not None else False,
                "session_armed": self._session_armed,
                "last_event": self._last_event,
                "last_error": self._last_error,
                "last_write_monotonic": self._last_write_at,
                "reconnect_count": self._reconnect_count,
            }

    def _emit(self, event: str, **details: Any) -> None:
        callback = self.event_callback
        if callback is not None:
            try:
                callback(event, details)
            except Exception:
                # Serial watchdog service must never depend on audit consumers.
                pass

    def run(self) -> None:
        backoff = 0.25
        while not self._stop_requested.is_set():
            config = self.settings()
            if not config.get("enabled", False):
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            try:
                self._connect_and_run(config)
                backoff = 0.25
            except Exception as exc:
                self._drop_serial(str(exc))
                if self._stop_requested.wait(backoff):
                    break
                backoff = min(float(config.get("reconnect_max_seconds", 5.0)), backoff * 2)
        self._drop_serial(None)

    def _connect_and_run(self, config: dict[str, Any]) -> None:
        port = self.serial_factory(str(config["serial_path"]), int(config["baud"]))
        with self._lock:
            self._serial = port
            self._connected = True
            self._ready = False
            self._startup_seen = False
            self._zero_handshake = False
            self._watchdog = False
            self._confirmed_mask = None
            self._last_sent_mask = None
            self._last_write_at = None
            self._last_error = ""
            self._line_buffer.clear()
            self._reconnect_count += 1
        self._emit("arduino_connected", serial_path=config["serial_path"])
        self._write(port, 0, "reconnect_zero")
        opened_at = self.clock()
        heartbeat = float(config["heartbeat_seconds"])
        startup_timeout = float(config.get("startup_timeout_seconds", 5.0))

        while not self._stop_requested.is_set():
            now = self.clock()
            for line in self._read_lines(port):
                self._apply_line(line)
            with self._lock:
                if self._startup_seen and self._zero_handshake:
                    self._ready = True
                ready = self._ready
                startup_seen = self._startup_seen
                desired = self._desired_mask if ready else 0
                due = self._last_write_at is None or now - self._last_write_at >= heartbeat
                changed = desired != self._last_sent_mask
            if not ready and now - opened_at > startup_timeout:
                raise TimeoutError("Arduino READY/zero handshake timed out")
            # The initial zero may reach the Mega bootloader. Do not stream
            # heartbeats until the sketch announces READY; repeated bytes can
            # keep the bootloader from handing control to the firmware.
            if startup_seen and (changed or due):
                self._write(port, desired, "desired_change" if changed else "heartbeat")
            self._wake.wait(min(0.02, heartbeat / 4))
            self._wake.clear()

        self._neutralize(port)

    def _write(self, port: SerialPort, mask: int, reason: str) -> None:
        payload = encode_mask(mask)
        written = port.write(payload)
        if written != len(payload):
            raise OSError("short Arduino serial write")
        port.flush()
        now = self.clock()
        with self._lock:
            changed = mask != self._last_sent_mask
            self._last_sent_mask = mask
            self._last_write_at = now
        if changed or reason != "heartbeat":
            self._emit("arduino_command", mask=mask, reason=reason)

    def _read_lines(self, port: SerialPort) -> list[str]:
        waiting = int(port.in_waiting)
        if waiting <= 0:
            return []
        chunk = bytes(port.read(min(waiting, 512)))
        lines: list[str] = []
        for byte in chunk:
            if byte == 10:
                line = self._line_buffer.decode("ascii", "replace").rstrip("\r")
                self._line_buffer.clear()
                if line:
                    lines.append(line)
            elif (byte == 13 or 0x20 <= byte <= 0x7E) and len(self._line_buffer) < 128:
                self._line_buffer.append(byte)
            else:
                self._line_buffer.clear()
                self._emit("arduino_malformed_event", reason="invalid_serial_bytes")
        return lines

    def _apply_line(self, line: str) -> None:
        valid = True
        details: dict[str, Any] = {"firmware_event": line}
        with self._lock:
            self._last_event = line
            if line == STARTUP_MARKER:
                self._startup_seen = True
            elif line == "EVENT BOOT":
                self._confirmed_mask = 0
            elif line == "EVENT REARMED":
                self._zero_handshake = True
                self._confirmed_mask = 0
                self._watchdog = False
                self._zero_confirmed.set()
            elif line.startswith("EVENT STATE "):
                try:
                    mask = int(line.removeprefix("EVENT STATE "))
                    if mask < 0 or mask > VALID_MASK:
                        raise ValueError
                except ValueError:
                    valid = False
                else:
                    self._confirmed_mask = mask
                    details["mask"] = mask
                    if mask == 0:
                        self._zero_handshake = True
                        self._zero_confirmed.set()
            elif line == "EVENT WATCHDOG":
                self._confirmed_mask = 0
                self._ready = False
                self._zero_handshake = False
                self._watchdog = True
                self._last_sent_mask = None
                self._zero_confirmed.set()
            elif line in {
                "EVENT REJECT MALFORMED",
                "EVENT REJECT OVERFLOW",
                "EVENT IGNORED LATCHED",
            }:
                self._confirmed_mask = 0
                self._ready = False
                self._zero_handshake = False
                self._last_sent_mask = None
                self._last_error = line
                self._zero_confirmed.set()
            else:
                valid = False
            if not valid:
                self._last_error = f"malformed Arduino event: {line}"[:240]
        self._emit("arduino_event" if valid else "arduino_malformed_event", **details)

    def _neutralize(self, port: SerialPort) -> None:
        try:
            self._write(port, 0, "shutdown_zero")
            deadline = self.clock() + 0.4
            while self.clock() < deadline:
                for line in self._read_lines(port):
                    self._apply_line(line)
                with self._lock:
                    if self._confirmed_mask == 0:
                        break
                time.sleep(0.01)
        except Exception:
            pass
        try:
            port.close()
        except Exception:
            pass

    def _drop_serial(self, error: str | None) -> None:
        with self._lock:
            port = self._serial
            self._serial = None
            self._connected = False
            self._ready = False
            self._startup_seen = False
            self._zero_handshake = False
            self._confirmed_mask = None
            self._last_sent_mask = None
            if error:
                self._last_error = error[:240]
        if port is not None:
            self._neutralize(port)
        if error:
            self._emit("arduino_error", error=error[:240])
