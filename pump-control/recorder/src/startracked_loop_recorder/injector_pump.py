"""HTTP pump client for the single-owner Loop Injector service."""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request

from .config import CAMERAS


PROTOCOL = "Loop Injector recorder lease API"
MAPPING = {"A": 2, "B": 1, "C": 8, "D": 4}


class InjectorPumpClient(threading.Thread):
    """Translate recorder camera demand into a renewable Injector lease."""

    def __init__(
        self,
        settings: Callable[[], dict[str, Any]],
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(name="injector-pump-http", daemon=True)
        self.settings = settings
        self.event_callback = event_callback
        self.opener = opener
        self.clock = clock
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop_requested = threading.Event()
        self._zero_confirmed = threading.Event()
        self._armed = False
        self._recording_session = ""
        self._remote_token = ""
        self._camera_demand = {camera: False for camera in CAMERAS}
        self._desired_mask = 0
        self._confirmed_mask: int | None = None
        self._connected = False
        self._ready = False
        self._last_error = ""
        self._last_event = ""
        self._last_write_at: float | None = None
        self._reconnect_count = 0
        self._arbitration_source = "idle"
        self._emergency_latched = False

    def set_session(self, armed: bool, session_token: str = "") -> None:
        with self._lock:
            changed = self._armed != bool(armed) or (
                armed and session_token != self._recording_session
            )
            self._armed = bool(armed)
            self._recording_session = session_token if armed else ""
            if changed:
                self._camera_demand = {camera: False for camera in CAMERAS}
                self._desired_mask = 0
            if not armed:
                self._zero_confirmed.clear()
        self._wake.set()

    def set_camera_demand(self, camera: str, desired: bool, session_token: str) -> None:
        if camera not in self._camera_demand:
            return
        with self._lock:
            if not self._armed or not session_token or session_token != self._recording_session:
                return
            self._camera_demand[camera] = bool(desired)
            self._desired_mask = sum(
                MAPPING[name] for name, active in self._camera_demand.items() if active
            )
            if self._desired_mask:
                self._zero_confirmed.clear()
        self._wake.set()

    def desired_mask(self) -> int:
        with self._lock:
            return self._desired_mask

    def wait_for_zero(self, timeout: float = 0.75) -> bool:
        return self._zero_confirmed.wait(timeout)

    def close(self, timeout: float = 3.0) -> None:
        self.set_session(False)
        if self.ident is not None and self.is_alive():
            self._zero_confirmed.wait(timeout=0.75)
        self._stop_requested.set()
        self._wake.set()
        if self.ident is not None:
            self.join(timeout=timeout)

    def status(self) -> dict[str, Any]:
        config = self.settings()
        with self._lock:
            return {
                "enabled": bool(config.get("enabled", False)),
                "mode": "injector",
                "injector_url": config.get("injector_url", ""),
                "serial_path": f"injector:{config.get('injector_url', '')}",
                "baud": None,
                "heartbeat_seconds": config.get("injector_heartbeat_seconds", 0.5),
                "protocol": PROTOCOL,
                "mapping": dict(MAPPING),
                "connected": self._connected,
                "ready": self._ready,
                "watchdog": False,
                "desired_mask": self._desired_mask,
                "confirmed_mask": self._confirmed_mask,
                "agreement": (
                    self._confirmed_mask == self._desired_mask
                    if self._confirmed_mask is not None else False
                ),
                "session_armed": self._armed,
                "last_event": self._last_event,
                "last_error": self._last_error,
                "last_write_monotonic": self._last_write_at,
                "reconnect_count": self._reconnect_count,
                "arbitration_source": self._arbitration_source,
                "emergency_latched": self._emergency_latched,
            }

    def _emit(self, event: str, **details: Any) -> None:
        if self.event_callback is not None:
            try:
                self.event_callback(event, details)
            except Exception:
                pass

    def _request(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        config = self.settings()
        base = str(config["injector_url"]).rstrip("/")
        request = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener(
                request, timeout=float(config.get("injector_timeout_seconds", 0.5))
            ) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Injector HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Injector unavailable: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Injector returned malformed JSON")
        return payload

    def _get_status(self) -> dict[str, Any]:
        config = self.settings()
        request = urllib.request.Request(
            str(config["injector_url"]).rstrip("/") + "/api/status",
            method="GET",
        )
        try:
            with self.opener(
                request, timeout=float(config.get("injector_timeout_seconds", 0.5))
            ) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Injector unavailable: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Injector returned malformed JSON")
        return payload

    def _apply_status(self, status: dict[str, Any]) -> None:
        requested = status.get("requested_mask")
        acknowledged = status.get("acknowledged_mask")
        recorder = status.get("recorder", {})
        if (
            isinstance(requested, bool) or not isinstance(requested, int)
            or not 0 <= requested <= 15
            or isinstance(acknowledged, bool) or not isinstance(acknowledged, int)
            or not 0 <= acknowledged <= 15
            or not isinstance(recorder, dict)
        ):
            raise RuntimeError("Injector status is malformed")
        with self._lock:
            self._connected = bool(status.get("connected"))
            self._ready = bool(status.get("ready"))
            self._confirmed_mask = acknowledged
            self._arbitration_source = str(
                status.get("arbitration_source", status.get("mode", "idle"))
            )
            self._emergency_latched = bool(status.get("emergency_latched"))
            self._last_event = str(status.get("last_event") or "")
            self._last_error = str(status.get("last_error") or "")
            self._last_write_at = self.clock()
            if self._desired_mask == 0 and acknowledged == 0:
                self._zero_confirmed.set()

    def _start_remote_session(self) -> None:
        payload = self._request("/api/recorder/session", {})
        token = payload.get("token")
        status = payload.get("status")
        if not isinstance(token, str) or not token or not isinstance(status, dict):
            raise RuntimeError("Injector session response is malformed")
        with self._lock:
            self._remote_token = token
            self._reconnect_count += 1
        self._apply_status(status)
        self._emit("injector_session_started")

    def _send_demand(self, mask: int) -> None:
        with self._lock:
            token = self._remote_token
        if not token:
            raise RuntimeError("Injector session is not established")
        status = self._request(
            "/api/recorder/demand", {"token": token, "mask": mask}
        )
        self._apply_status(status)
        self._emit(
            "injector_command", mask=mask,
            arbitration_source=self._arbitration_source,
            emergency_latched=self._emergency_latched,
        )

    def _stop_remote_session(self) -> None:
        with self._lock:
            token = self._remote_token
            self._remote_token = ""
        if not token:
            self._zero_confirmed.set()
            return
        try:
            self._send_with_token("/api/recorder/demand", token, 0)
            status = self._request("/api/recorder/session/stop", {"token": token})
            self._apply_status(status)
        finally:
            with self._lock:
                self._confirmed_mask = 0
                self._zero_confirmed.set()

    def _send_with_token(self, path: str, token: str, mask: int) -> None:
        status = self._request(path, {"token": token, "mask": mask})
        self._apply_status(status)

    def run(self) -> None:
        while not self._stop_requested.is_set():
            config = self.settings()
            heartbeat = float(config.get("injector_heartbeat_seconds", 0.5))
            with self._lock:
                enabled = bool(config.get("enabled", False))
                armed = self._armed
                token = self._remote_token
                mask = self._desired_mask
            try:
                if not enabled or not armed:
                    if token:
                        self._stop_remote_session()
                    else:
                        self._zero_confirmed.set()
                        if enabled:
                            self._apply_status(self._get_status())
                else:
                    if not token:
                        self._start_remote_session()
                    self._send_demand(mask)
            except Exception as exc:
                with self._lock:
                    self._connected = False
                    self._ready = False
                    self._confirmed_mask = None
                    self._last_error = str(exc)[:240]
                    self._remote_token = ""
                self._emit("injector_error", error=str(exc)[:240])
            self._wake.wait(heartbeat)
            self._wake.clear()
        try:
            self._stop_remote_session()
        except Exception:
            pass
