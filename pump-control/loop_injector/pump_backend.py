"""Fail-closed serial owner and scheduler for MultiCameraPumpLCD."""
from __future__ import annotations

from dataclasses import dataclass
import secrets
import threading
import time
from typing import Callable, Protocol

from .automations import AutomationStore
from .pump_log import PumpActivityLog


SERIAL_PATH = (
    "/dev/serial/by-id/"
    "usb-Arduino__www.arduino.cc__0042_00000000000000000000-if00"
)
STARTUP_MARKER = "MULTICAM_PUMP_READY"
HEARTBEAT_SECONDS = 0.15
LEASE_SECONDS = 2.0
RECONNECT_SECONDS = 0.5
READY_TIMEOUT_SECONDS = 5.0


class SerialPort(Protocol):
    @property
    def in_waiting(self) -> int: ...

    def write(self, data: bytes) -> int: ...

    def read(self, size: int) -> bytes: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


def encode_mask(mask: int) -> bytes:
    if isinstance(mask, bool) or not isinstance(mask, int) or not 0 <= mask <= 15:
        raise ValueError("pump mask must be an integer from 0 through 15")
    return f"PUMPS:{mask}\n".encode("ascii")


def default_serial_factory() -> SerialPort:
    import serial

    return serial.Serial(
        SERIAL_PATH,
        115200,
        timeout=0,
        write_timeout=2,
        exclusive=True,
    )


@dataclass(frozen=True)
class Schedule:
    order: tuple[int, ...]
    burst_seconds: float
    total_seconds: float
    started_at: float


class PumpController:
    """Own the serial port and publish only fresh browser-authorized state."""

    def __init__(
        self,
        serial_factory: Callable[[], SerialPort] = default_serial_factory,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        lease_seconds: float = LEASE_SECONDS,
        heartbeat: float = HEARTBEAT_SECONDS,
        automation_store: AutomationStore | None = None,
        pump_log: PumpActivityLog | None = None,
    ) -> None:
        self._serial_factory = serial_factory
        self._clock = clock
        self._sleep = sleep
        self._lease_seconds = float(lease_seconds)
        self._heartbeat = float(heartbeat)
        self._automation_store = automation_store
        self._pump_log = pump_log
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial: SerialPort | None = None
        self._line_buffer = bytearray()
        self._connected = False
        self._ready = False
        self._mode = "idle"
        self._transient_mode = "idle"
        self._transient_mask = 0
        self._effective_mask = 0
        self._acknowledged_mask = 0
        self._lease_deadline = 0.0
        self._schedule: Schedule | None = None
        self._emergency_latched = False
        self._recorder_token: str | None = None
        self._recorder_mask = 0
        self._recorder_lease_deadline = 0.0
        self._last_event: str | None = None
        self._last_error: str | None = None
        self._last_write_at = 0.0
        self._automation_state = self._empty_automation_state()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="loop-injector-serial", daemon=True
            )
            self._thread.start()

    def close(self) -> None:
        with self._lock:
            self._schedule = None
            self._recorder_token = None
            self._recorder_mask = 0
            self._recorder_lease_deadline = 0.0
            self._transient_mask = 0
            self._effective_mask = 0
            self._transient_mode = "idle"
            self._mode = "idle"
            self._lease_deadline = 0.0
        self._stop_event.set()
        thread = self._thread
        if thread:
            thread.join(timeout=3)
        with self._lock:
            port = self._serial
            self._serial = None
            self._connected = False
            self._ready = False
        if port is not None:
            self._neutral_and_close(port)
        if self._pump_log is not None:
            self._pump_log.close()

    def renew_lease(self) -> dict:
        with self._lock:
            self._lease_deadline = self._clock() + self._lease_seconds
            return self._status_locked()

    def set_manual_mask(self, mask: int) -> dict:
        encode_mask(mask)
        with self._lock:
            self._require_live_lease()
            if self._schedule is not None:
                raise ValueError("stop the active schedule before manual control")
            self._emergency_latched = False
            self._transient_mode = "manual" if mask else "idle"
            self._transient_mask = mask
            return self._status_locked()

    def start_schedule(
        self, order: tuple[int, ...], burst_seconds: float, total_seconds: float
    ) -> dict:
        if not order or len(order) > 16:
            raise ValueError("schedule order must contain 1 through 16 pumps")
        if any(isinstance(pump, bool) or pump not in (1, 2, 3, 4) for pump in order):
            raise ValueError("schedule order entries must be pump numbers 1 through 4")
        burst = float(burst_seconds)
        total = float(total_seconds)
        if not 0.5 <= burst <= 3600:
            raise ValueError("burst_seconds must be between 0.5 and 3600")
        if not 0.5 <= total <= 86400:
            raise ValueError("total_seconds must be between 0.5 and 86400")
        with self._lock:
            self._require_live_lease()
            now = self._clock()
            self._schedule = Schedule(tuple(order), burst, total, now)
            self._transient_mode = "schedule"
            self._transient_mask = 1 << (order[0] - 1)
            return self._status_locked(now)

    def stop_schedule(self) -> dict:
        with self._lock:
            self._schedule = None
            self._transient_mask = 0
            self._transient_mode = "idle"
            return self._status_locked()

    def emergency_stop(self) -> dict:
        if self._automation_store is not None:
            try:
                self._automation_store.set_paused(True)
            except Exception as exc:
                with self._lock:
                    self._last_error = (
                        f"failed to persist automation pause: {exc}"
                    )[:240]
        with self._lock:
            self._emergency_latched = True
            self._schedule = None
            self._transient_mask = 0
            self._recorder_mask = 0
            self._recorder_lease_deadline = 0.0
            self._effective_mask = 0
            self._transient_mode = "idle"
            self._mode = "idle"
            self._lease_deadline = 0.0
            return self._status_locked()

    def start_recorder_session(self) -> tuple[str, dict]:
        with self._lock:
            self._recorder_token = secrets.token_urlsafe(32)
            self._recorder_mask = 0
            self._recorder_lease_deadline = self._clock() + self._lease_seconds
            return self._recorder_token, self._status_locked()

    def set_recorder_demand(self, token: str, mask: int) -> dict:
        encode_mask(mask)
        with self._lock:
            self._require_recorder_token(token)
            self._recorder_mask = mask
            self._recorder_lease_deadline = self._clock() + self._lease_seconds
            return self._status_locked()

    def stop_recorder_session(self, token: str) -> dict:
        with self._lock:
            self._require_recorder_token(token)
            self._recorder_token = None
            self._recorder_mask = 0
            self._recorder_lease_deadline = 0.0
            return self._status_locked()

    def pause_automations(self) -> dict:
        if self._automation_store is not None:
            self._automation_store.set_paused(True)
        with self._lock:
            self._schedule = None
            self._transient_mask = 0
            self._transient_mode = "idle"
            return self._status_locked()

    def resume_automations(self) -> dict:
        with self._lock:
            self._schedule = None
            self._transient_mask = 0
            self._transient_mode = "idle"
            self._lease_deadline = 0.0
        if self._automation_store is not None:
            self._automation_store.set_paused(False)
        return self.status()

    def save_automation(self, payload: dict) -> dict:
        store = self._require_automation_store()
        store.save(payload)
        return self.status()

    def delete_automation(self, rule_id: str) -> dict:
        self._require_automation_store().delete(rule_id)
        return self.status()

    def set_automation_enabled(self, rule_id: str, enabled: bool) -> dict:
        self._require_automation_store().set_enabled(rule_id, enabled)
        return self.status()

    def query_pump_log(self, **filters) -> dict:
        if self._pump_log is None:
            raise ValueError("pump activity log is not configured")
        return self._pump_log.query_intervals(**filters)

    def export_pump_log_csv(self, **filters) -> bytes:
        if self._pump_log is None:
            raise ValueError("pump activity log is not configured")
        return self._pump_log.export_csv(**filters)

    def status(self) -> dict:
        with self._lock:
            self._refresh_desired_locked(self._clock())
            return self._status_locked()

    def _require_live_lease(self) -> None:
        if self._clock() >= self._lease_deadline:
            raise ValueError("control lease is not active")

    def _require_recorder_token(self, token: str) -> None:
        if (
            not isinstance(token, str)
            or not token
            or self._recorder_token is None
            or not secrets.compare_digest(token, self._recorder_token)
        ):
            raise PermissionError("invalid recorder session token")

    def _require_automation_store(self) -> AutomationStore:
        if self._automation_store is None:
            raise ValueError("automation storage is not configured")
        return self._automation_store

    @staticmethod
    def _empty_automation_state() -> dict:
        return {
            "paused": True,
            "mask": 0,
            "active_ids": [],
            "next_start": None,
            "rules": [],
            "last_error": None,
            "local_time": None,
            "timezone": None,
        }

    def _refresh_desired_locked(self, now: float) -> None:
        if now >= self._lease_deadline:
            self._schedule = None
            self._transient_mask = 0
            self._transient_mode = "idle"
        schedule = self._schedule
        if schedule is not None:
            elapsed = max(0.0, now - schedule.started_at)
            if elapsed >= schedule.total_seconds:
                self._schedule = None
                self._transient_mask = 0
                self._transient_mode = "idle"
            else:
                index = int(elapsed / schedule.burst_seconds) % len(schedule.order)
                self._transient_mask = 1 << (schedule.order[index] - 1)

        if now >= self._recorder_lease_deadline:
            self._recorder_mask = 0

        store = self._automation_store
        automation = store.evaluate() if store is not None else self._empty_automation_state()
        self._automation_state = automation
        automation_mask = automation["mask"]
        if self._emergency_latched:
            self._effective_mask = 0
            self._mode = "emergency"
        elif self._transient_mask or self._schedule is not None:
            self._effective_mask = self._transient_mask
            self._mode = self._transient_mode
        elif self._recorder_mask:
            self._effective_mask = self._recorder_mask
            self._mode = "recorder"
        elif automation_mask:
            self._effective_mask = automation_mask
            self._mode = "automation"
        else:
            self._effective_mask = 0
            self._mode = "idle"

    def _status_locked(self, now: float | None = None) -> dict:
        current = self._clock() if now is None else now
        self._refresh_desired_locked(current)
        schedule = self._schedule
        schedule_status = None
        if schedule is not None:
            elapsed = min(schedule.total_seconds, max(0.0, current - schedule.started_at))
            index = int(elapsed / schedule.burst_seconds) % len(schedule.order)
            schedule_status = {
                "order": list(schedule.order),
                "burst_seconds": schedule.burst_seconds,
                "total_seconds": schedule.total_seconds,
                "elapsed_seconds": elapsed,
                "remaining_seconds": max(0.0, schedule.total_seconds - elapsed),
                "active_pump": schedule.order[index],
            }
        return {
            "connected": self._connected,
            "ready": self._ready,
            "mode": self._mode,
            "requested_mask": self._effective_mask,
            "acknowledged_mask": self._acknowledged_mask,
            "lease_active": current < self._lease_deadline,
            "lease_remaining_seconds": max(0.0, self._lease_deadline - current),
            "emergency_latched": self._emergency_latched,
            "arbitration_source": self._mode,
            "recorder": {
                "session_active": self._recorder_token is not None,
                "connected": (
                    self._recorder_token is not None
                    and current < self._recorder_lease_deadline
                ),
                "lease_active": current < self._recorder_lease_deadline,
                "lease_remaining_seconds": max(
                    0.0, self._recorder_lease_deadline - current
                ),
                "demand_mask": self._recorder_mask,
            },
            "schedule": schedule_status,
            "automations": self._automation_state,
            "pump_log": (
                self._pump_log.status()
                if self._pump_log is not None
                else {
                    "healthy": False,
                    "total_intervals": 0,
                    "open_intervals": 0,
                    "last_error": "pump activity log is not configured",
                }
            ),
            "last_event": self._last_event,
            "last_error": self._last_error,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._connect_and_run()
            except Exception as exc:
                self._drop_serial(str(exc))
                self._stop_event.wait(RECONNECT_SECONDS)
        self._drop_serial(None)

    def _connect_and_run(self) -> None:
        port = self._serial_factory()
        with self._lock:
            self._serial = port
            self._connected = True
            self._ready = False
            self._last_error = None
            self._line_buffer.clear()

        self._write(port, 0)
        opened_at = self._clock()
        while not self._stop_event.is_set():
            events = self._read_events(port)
            self._apply_events(events)
            if STARTUP_MARKER in events:
                if self._pump_log is not None:
                    self._pump_log.record_boot_zero(
                        firmware_event=STARTUP_MARKER
                    )
                self._write(port, 0)
                with self._lock:
                    self._ready = True
                    self._last_write_at = self._clock()
                break
            if self._clock() - opened_at > READY_TIMEOUT_SECONDS:
                raise TimeoutError("Arduino startup marker not received")
            self._stop_event.wait(0.02)

        while not self._stop_event.is_set():
            now = self._clock()
            events = self._read_events(port)
            self._apply_events(events)
            if "EVENT WATCHDOG" in events:
                raise OSError("Arduino watchdog triggered")
            with self._lock:
                self._refresh_desired_locked(now)
                desired = self._effective_mask if self._ready else 0
                due = now - self._last_write_at >= self._heartbeat
            if due:
                self._write(port, desired)
                with self._lock:
                    self._last_write_at = now
            self._stop_event.wait(0.02)

        self._confirm_neutral_and_close(port)
        with self._lock:
            if self._serial is port:
                self._serial = None
                self._connected = False
                self._ready = False

    def _confirm_neutral_and_close(self, port: SerialPort) -> None:
        try:
            with self._lock:
                needs_confirmation = self._acknowledged_mask != 0
            self._write(port, 0)
            if needs_confirmation:
                deadline = self._clock() + 0.4
                while self._clock() < deadline:
                    events = self._read_events(port)
                    self._apply_events(events)
                    with self._lock:
                        if self._acknowledged_mask == 0:
                            break
                    self._sleep(0.01)
        except Exception:
            pass
        try:
            port.close()
        except Exception:
            pass

    def _write(self, port: SerialPort, mask: int) -> None:
        payload = encode_mask(mask)
        written = port.write(payload)
        if written != len(payload):
            raise OSError("short serial write")
        port.flush()

    def _read_events(self, port: SerialPort) -> list[str]:
        waiting = int(port.in_waiting)
        if waiting <= 0:
            return []
        chunk = bytes(port.read(min(waiting, 512)))
        events: list[str] = []
        for byte in chunk:
            if byte == 10:
                line = self._line_buffer.decode("ascii", "replace").rstrip("\r")
                self._line_buffer.clear()
                if line:
                    events.append(line)
            elif len(self._line_buffer) < 128:
                self._line_buffer.append(byte)
            else:
                self._line_buffer.clear()
        return events

    def _apply_events(self, events: list[str]) -> None:
        for event in events:
            confirmed_mask: int | None = None
            if event == "EVENT WATCHDOG" and self._automation_store is not None:
                try:
                    self._automation_store.set_paused(True)
                except Exception:
                    pass
            with self._lock:
                self._last_event = event
                if event.startswith("EVENT STATE "):
                    try:
                        confirmed_mask = int(
                            event.removeprefix("EVENT STATE ")
                        )
                        self._acknowledged_mask = confirmed_mask
                    except ValueError:
                        pass
                elif event == "EVENT WATCHDOG":
                    confirmed_mask = 0
                    self._acknowledged_mask = 0
                    self._ready = False
                    self._transient_mask = 0
                    self._effective_mask = 0
                    self._schedule = None
                    self._transient_mode = "idle"
                    self._recorder_mask = 0
                    self._recorder_lease_deadline = 0.0
                    self._emergency_latched = True
                    self._mode = "emergency"
                mode = self._mode
                automation_ids = tuple(
                    self._automation_state.get("active_ids", ())
                )
            if confirmed_mask is not None and self._pump_log is not None:
                try:
                    self._pump_log.record_confirmed_mask(
                        confirmed_mask,
                        firmware_event=event,
                        control_mode=mode,
                        automation_ids=automation_ids,
                        end_reason=(
                            "firmware_watchdog"
                            if event == "EVENT WATCHDOG"
                            else "confirmed_state_change"
                        ),
                    )
                except Exception as exc:
                    self._handle_log_failure(exc)
                    raise

    def _handle_log_failure(self, error: Exception) -> None:
        store = self._automation_store
        if store is not None:
            try:
                store.set_paused(True)
            except Exception:
                pass
        with self._lock:
            self._schedule = None
            self._transient_mask = 0
            self._effective_mask = 0
            self._transient_mode = "idle"
            self._mode = "idle"
            self._lease_deadline = 0.0
            self._recorder_mask = 0
            self._recorder_lease_deadline = 0.0
            self._emergency_latched = True
            self._last_error = f"pump activity log failure: {error}"[:240]

    def _drop_serial(self, error: str | None) -> None:
        if error and self._automation_store is not None:
            try:
                self._automation_store.set_paused(True)
            except Exception:
                pass
        with self._lock:
            port = self._serial
            self._serial = None
            self._connected = False
            self._ready = False
            self._acknowledged_mask = 0
            if error:
                self._last_error = error[:240]
                self._schedule = None
                self._transient_mask = 0
                self._recorder_mask = 0
                self._recorder_lease_deadline = 0.0
                self._effective_mask = 0
                self._emergency_latched = True
                self._mode = "emergency"
        if port is not None:
            self._neutral_and_close(port)

    @staticmethod
    def _neutral_and_close(port: SerialPort) -> None:
        try:
            payload = encode_mask(0)
            port.write(payload)
            port.flush()
        except Exception:
            pass
        try:
            port.close()
        except Exception:
            pass
