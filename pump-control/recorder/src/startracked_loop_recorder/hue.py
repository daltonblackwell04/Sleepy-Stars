from __future__ import annotations

import json
import queue
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .arduino_pump import ArduinoPumpWorker
from .config import CAMERAS


HUE_PAIRING_DEVICE_TYPE = "startracked#plates"
SUNFLOWER_CLASS_NAMES = ("sunflower_star", "Pycnopodia_helianthoides")
HOURLY_PUMP_CAMERAS = ("A", "B", "C", "D")
HOURLY_PULSE_SECONDS = 60.0
HOURLY_SCHEDULE_POLICY = {
    "version": 4,
    "cameras": list(HOURLY_PUMP_CAMERAS),
    "opportunity": "next_local_clock_top_of_hour_then_each_local_hour_while_armed",
    "pulse_seconds": HOURLY_PULSE_SECONDS,
    "occupancy": "hysteresis_confirmed_sunflower_star",
    "occupancy_class_names": list(SUNFLOWER_CLASS_NAMES),
    "fail_closed": True,
    "occupied_or_unknown_action": "skip_without_retry_until_next_hour",
    "hue_arbitration": "primary_occupancy_or_active_secondary_pulse",
    "arduino_arbitration": "primary_occupancy_only",
    "pulse_actuator": "hue_only",
    "arm_state": "runtime_only",
}
_HUE_DEVICE_TYPE_MAX_LENGTH = 40
_HUE_APPLICATION_NAME_MAX_LENGTH = 20
_HUE_DEVICE_NAME_MAX_LENGTH = 19


def _validate_hue_device_type(devicetype: str) -> str:
    parts = devicetype.split("#")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Hue devicetype must be application_name#device_name")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in devicetype):
        raise ValueError("Hue devicetype must not contain control characters")
    application_name, device_name = parts
    if (
        len(devicetype) > _HUE_DEVICE_TYPE_MAX_LENGTH
        or len(application_name) > _HUE_APPLICATION_NAME_MAX_LENGTH
        or len(device_name) > _HUE_DEVICE_NAME_MAX_LENGTH
    ):
        raise ValueError("Hue devicetype exceeds bridge length limits")
    return devicetype


def is_sunflower_class(name: str) -> bool:
    """Recognize exact legacy/native names without assuming a model class ID."""
    return name in SUNFLOWER_CLASS_NAMES


@dataclass(frozen=True)
class InferenceEvent:
    camera: str
    frame_index: int
    frame_ts: float
    received_monotonic: float
    recorded: bool
    polygon_valid: bool
    detections: tuple[tuple[str, float, float], ...]
    events_jsonl: Path | None
    session_token: str = ""
    inference_ok: bool = True
    pressure_polygon: tuple[tuple[float, float], ...] = ()
    lane: str = "control"
    capture_monotonic: float = 0.0
    enqueue_monotonic: float = 0.0
    inference_started_monotonic: float = 0.0
    inference_finished_monotonic: float = 0.0

    @property
    def occupied(self) -> bool:
        return any(is_sunflower_class(name) for name, _x, _y in self.detections)


class HueBridge:
    """Small Hue v2 client. HTTPS verification is disabled only for the local
    bridge's manufacturer-installed self-signed certificate."""

    def __init__(self, address: str, app_key: str = "", timeout: float = 2.0):
        self.address = address.strip()
        self.app_key = app_key.strip()
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Any = None, *, authenticated=True):
        scheme = "https" if authenticated else "http"
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if authenticated:
            if not self.app_key:
                raise RuntimeError("Hue application key is not configured")
            headers["hue-application-key"] = self.app_key
        request = urllib.request.Request(
            f"{scheme}://{self.address}{path}", data=data, headers=headers, method=method
        )
        context = ssl._create_unverified_context() if authenticated else None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=context) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Hue HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Hue bridge unavailable: {exc}") from exc

    def pair(self) -> str:
        response = self._request(
            "POST", "/api",
            {"devicetype": _validate_hue_device_type(HUE_PAIRING_DEVICE_TYPE)},
            authenticated=False,
        )
        first = response[0] if isinstance(response, list) and response else {}
        username = first.get("success", {}).get("username")
        if username:
            return str(username)
        error = first.get("error", {}).get("description", "pairing failed")
        raise RuntimeError(
            "Hue pairing was not authorized; press the bridge link button, then retry"
            if "link button" in str(error).lower() else str(error)
        )

    def discover_outputs(self) -> list[dict[str, str]]:
        payload = self._request("GET", "/clip/v2/resource/light")
        outputs = []
        for item in payload.get("data", []):
            metadata = item.get("metadata", {})
            archetype = str(metadata.get("archetype", "")).lower()
            outputs.append({
                "id": str(item["id"]),
                "name": str(metadata.get("name") or item["id"]),
                "kind": "smart_plug" if archetype in {"plug", "socket"} else "light",
            })
        return sorted(outputs, key=lambda item: (item["name"].lower(), item["id"]))

    def set_output(self, resource_id: str, on: bool) -> None:
        self._request("PUT", f"/clip/v2/resource/light/{resource_id}", {"on": {"on": on}})


class PressurePlateService(threading.Thread):
    """Owns state machines and all Hue network I/O away from recorder workers."""

    def __init__(
        self, settings: Callable[[], dict[str, Any]], shutdown: threading.Event,
        bridge_factory: Callable[[str, str], HueBridge] = HueBridge,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        arduino: ArduinoPumpWorker | None = None,
    ):
        super().__init__(name="hue-pressure-plates", daemon=True)
        self.settings = settings
        self.shutdown = shutdown
        self.bridge_factory = bridge_factory
        self.wall_clock = wall_clock
        self.monotonic_clock = monotonic_clock
        self.arduino = arduino
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._condition = threading.Condition()
        self._submitted = self._processed = 0
        self._armed = False
        self._session_token = ""
        self._hourly_armed = False
        self._next_hourly_at: float | None = None
        self._states = {
            camera: {
                "occupied": False, "on": False, "candidate": None, "candidate_since": 0.0,
                "last_frame_ts": float("-inf"), "last_seen": 0.0, "error": "",
                "events_jsonl": None,
                "logical_primary_demand": False,
                "hue_desired": False,
                "arduino_desired": False,
                "last_command": None,
                "retry_at": 0.0, "retry_delay": 0.5,
                "output_id": "",
                "inference_available": False,
                "inference_reason": "unavailable",
                "secondary_active": False,
                "secondary_deadline": None,
                "last_opportunity_at": None,
                "last_opportunity_result": "",
                "last_skip_reason": "",
                "pending_events": [],
                "split_warning": False,
                "last_control_provenance": {},
            }
            for camera in CAMERAS
        }
        self._outputs: list[dict[str, str]] = []
        self._bridge_status = "unconfigured"
        self._bridge_error = ""
        self._closed = False

    def submit(self, event: InferenceEvent) -> None:
        with self._condition:
            if self._closed:
                return
            self._submitted += 1
        self._queue.put(("inference", event))

    def submit_arduino_event(self, event: str, details: dict[str, Any]) -> None:
        """Accept serial audit events without ever blocking its watchdog thread."""
        self._queue.put(("arduino_event", (event, details)))

    def arm(self, armed: bool, session_token: str = "") -> None:
        self._queue.put(("arm", (bool(armed), session_token)))

    def set_hourly_schedule(
        self, armed: bool, session_token: str = "", timeout: float = 5.0
    ) -> None:
        """Queue a runtime-only schedule change; the worker owns reconciliation."""
        reply: queue.Queue[bool] = queue.Queue(maxsize=1)
        self._queue.put(("hourly_arm", (bool(armed), session_token, reply)))
        try:
            accepted = reply.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("timed out changing hourly pump schedule") from exc
        if not accepted:
            raise ValueError("hourly pump schedule requires the active recording session")

    def reconfigure(self) -> None:
        self._queue.put(("reconfigure", None))

    def set_event_sink(self, camera: str, path: Path, session_token: str) -> None:
        self._queue.put(("event_sink", (camera, path, session_token)))

    def discover(self, timeout: float = 8.0) -> list[dict[str, str]]:
        reply: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._queue.put(("discover", reply))
        result = reply.get(timeout=timeout)
        if isinstance(result, Exception):
            raise result
        return result

    def pair(self, timeout: float = 8.0) -> str:
        reply: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._queue.put(("pair", reply))
        result = reply.get(timeout=timeout)
        if isinstance(result, Exception):
            raise result
        return result

    def flush(self, timeout: float = 10.0) -> None:
        if self.ident is None:
            return
        reply: queue.Queue[bool] = queue.Queue(maxsize=1)
        self._queue.put(("barrier", reply))
        try:
            reply.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("timed out flushing pressure-plate events") from exc

    def close(self) -> None:
        self._closed = True
        self._queue.put(("close", None))

    def _bridge(self) -> HueBridge:
        settings = self.settings()
        return self.bridge_factory(settings["bridge_address"], settings["app_key"])

    def _write(self, camera: str, event: str, **details: Any) -> None:
        state = self._states[camera]
        provenance = (
            state["last_control_provenance"]
            if event in {"occupancy", "actuator_targets", "hue_command"} else {}
        )
        row = {
            "event": event, "camera": camera, "ts": self.wall_clock(),
            "monotonic": self.monotonic_clock(), **provenance, **details,
        }
        path = state["events_jsonl"]
        if path is None:
            state["pending_events"].append(row)
            return
        self._append_event(path, row)

    @staticmethod
    def _append_event(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()

    def _camera_plate_valid(self, camera: str) -> bool:
        polygon = self.settings().get("polygons", {}).get(camera, [])
        return isinstance(polygon, list) and 3 <= len(polygon) <= 64

    @staticmethod
    def _next_local_hour(now: float) -> float:
        """Find the next real epoch whose local representation is HH:00:00.

        Searching real minutes handles non-whole-hour UTC offsets and DST gaps
        and folds without constructing nonexistent local datetimes.
        """
        candidate = (int(now) // 60 + 1) * 60
        for _ in range(26 * 60):
            local = datetime.fromtimestamp(candidate).astimezone()
            if local.minute == 0 and local.second == 0 and candidate > now:
                return float(candidate)
            candidate += 60
        raise RuntimeError("could not find next local top-of-hour")

    @staticmethod
    def _local_display(timestamp: float | None) -> str | None:
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp).astimezone().isoformat()

    def _logical_primary_demand(self, camera: str) -> bool:
        state = self._states[camera]
        settings = self.settings()
        assigned = bool(
            settings["outputs"].get(camera, "")
            or (
                self.arduino is not None
                and int(self.arduino.status()["mapping"].get(camera, 0))
            )
        )
        return bool(
            state["occupied"] and self._armed and settings["enabled"] and assigned
            and self._camera_plate_valid(camera)
        )

    def _reconcile(self, camera: str, reason: str, force=False) -> None:
        state = self._states[camera]
        settings = self.settings()
        output = settings["outputs"].get(camera, "")
        previous_hue_desired = state["hue_desired"]
        previous_targets = (
            state["logical_primary_demand"],
            state["hue_desired"],
            state["arduino_desired"],
        )
        state["logical_primary_demand"] = self._logical_primary_demand(camera)
        common_hue_gate = bool(
            self._armed and settings["enabled"] and output
            and self._camera_plate_valid(camera)
        )
        state["hue_desired"] = bool(
            common_hue_gate
            and (state["logical_primary_demand"] or state["secondary_active"])
        )
        arduino = self.arduino
        arduino_bit = int(
            arduino.status()["mapping"].get(camera, 0) if arduino is not None else 0
        )
        state["arduino_desired"] = bool(
            state["logical_primary_demand"] and arduino_bit
        )
        if arduino is not None:
            arduino.set_camera_demand(
                camera, state["arduino_desired"], self._session_token
            )
        targets = (
            state["logical_primary_demand"],
            state["hue_desired"],
            state["arduino_desired"],
        )
        if targets != previous_targets:
            self._write(
                camera, "actuator_targets", reason=reason,
                logical_primary_demand=bool(state["logical_primary_demand"]),
                secondary_hourly_pulse_active=bool(state["secondary_active"]),
                hue_desired=bool(state["hue_desired"]),
                arduino_desired=bool(state["arduino_desired"]),
            )
        try:
            self._reconcile_hue(
                camera, reason, force, previous_hue_desired, output
            )
        finally:
            self._update_split(camera, reason)

    def _reconcile_hue(
        self, camera: str, reason: str, force: bool,
        previous_hue_desired: bool, output: str,
    ) -> None:
        state = self._states[camera]
        wanted = state["hue_desired"]
        previous_output = state["output_id"]
        if previous_output and previous_output != output:
            try:
                self._bridge().set_output(previous_output, False)
                self._write(
                    camera, "hue_command", output_id=previous_output, on=False,
                    reason="output_unassigned_or_changed",
                )
                state["on"] = False
                state["last_command"] = None
                state["output_id"] = ""
            except Exception as exc:
                state["error"] = str(exc)
                state["retry_at"] = self.monotonic_clock() + state["retry_delay"]
                state["retry_delay"] = min(5.0, state["retry_delay"] * 2)
                self._bridge_status = "error"
                self._bridge_error = str(exc)
                return
        if not output:
            state["on"] = False
            state["last_command"] = False
            return
        if (
            state["error"] and not force and wanted == previous_hue_desired
            and self.monotonic_clock() < state["retry_at"]
        ):
            return
        if not force and state["last_command"] is wanted and not state["error"]:
            return
        try:
            self._bridge().set_output(output, wanted)
            changed = state["on"] != wanted or state["last_command"] is None
            state["on"] = wanted
            state["last_command"] = wanted
            state["output_id"] = output
            state["error"] = ""
            state["retry_at"] = 0.0
            state["retry_delay"] = 0.5
            self._bridge_status = "connected"
            self._bridge_error = ""
            if changed:
                self._write(camera, "hue_command", output_id=output, on=wanted, reason=reason)
        except Exception as exc:
            state["error"] = str(exc)
            state["retry_at"] = self.monotonic_clock() + state["retry_delay"]
            state["retry_delay"] = min(5.0, state["retry_delay"] * 2)
            self._bridge_status = "error"
            self._bridge_error = str(exc)
            self._write(
                camera, "hue_command_error", output_id=output, on=wanted,
                reason=reason, error=str(exc),
            )

    def _update_split(self, camera: str, reason: str) -> None:
        state = self._states[camera]
        if camera not in HOURLY_PUMP_CAMERAS or self.arduino is None:
            state["split_warning"] = bool(
                state["error"] or state["on"] != state["hue_desired"]
            )
            return
        arduino = self.arduino.status()
        bit = int(arduino["mapping"].get(camera, 0))
        confirmed = arduino["confirmed_mask"]
        arduino_on = None if confirmed is None else bool(confirmed & bit)
        split = bool(
            state["error"]
            or state["on"] != state["hue_desired"]
            or not arduino["connected"]
            or not arduino["ready"]
            or arduino["last_error"]
            or arduino_on is None
            or arduino_on != state["arduino_desired"]
        )
        if split != state["split_warning"]:
            state["split_warning"] = split
            self._write(
                camera, "split_actuator_warning",
                active=split, reason=reason,
                logical_primary_demand=bool(state["logical_primary_demand"]),
                secondary_hourly_pulse_active=bool(state["secondary_active"]),
                hue_desired=bool(state["hue_desired"]),
                hue_on=bool(state["on"]), hue_error=state["error"],
                arduino_desired=bool(state["arduino_desired"]),
                arduino_worker_desired=bool(arduino["desired_mask"] & bit),
                arduino_confirmed=arduino_on,
                arduino_connected=arduino["connected"],
                arduino_ready=arduino["ready"],
                arduino_error=arduino["last_error"],
            )

    def _record_arduino_event(self, event: str, details: dict[str, Any]) -> None:
        if not self._armed:
            return
        for camera in HOURLY_PUMP_CAMERAS:
            self._write(
                camera, event, session_token=self._session_token,
                mapping_bit=int(
                    self.arduino.status()["mapping"].get(camera, 0)
                    if self.arduino is not None else 0
                ),
                **details,
            )
            self._update_split(camera, event)

    def _disarm_all(self, reason: str) -> None:
        self._set_hourly_armed(False, reason)
        self._armed = False
        for camera, state in self._states.items():
            if state["occupied"]:
                state["occupied"] = False
                self._write(camera, "occupancy", occupied=False, reason=reason)
            state["candidate"] = None
            state["inference_available"] = False
            state["inference_reason"] = reason
            self._reconcile(camera, reason, force=True)
            state["events_jsonl"] = None

    def _set_hourly_armed(self, armed: bool, reason: str) -> None:
        if armed:
            if not self._armed:
                return
            if self._hourly_armed:
                return
            self._hourly_armed = True
            self._next_hourly_at = self._next_local_hour(self.wall_clock())
            for camera in HOURLY_PUMP_CAMERAS:
                self._write(
                    camera, "hourly_schedule_arm",
                    next_opportunity_at=self._next_hourly_at,
                    next_opportunity_local=self._local_display(self._next_hourly_at),
                    policy=HOURLY_SCHEDULE_POLICY,
                )
            return
        was_armed = self._hourly_armed
        self._hourly_armed = False
        self._next_hourly_at = None
        for camera in HOURLY_PUMP_CAMERAS:
            state = self._states[camera]
            was_active = state["secondary_active"]
            if was_active:
                state["secondary_active"] = False
                state["secondary_deadline"] = None
                self._write(
                    camera, "hourly_pulse_end", reason=reason, cancelled=True,
                    actuator="hue_only",
                )
            if was_armed:
                self._write(camera, "hourly_schedule_disarm", reason=reason)
            # Schedule cancellation changes only Hue secondary demand. Recompute
            # both explicit actuator targets on every disarm request,
            # including retries and idempotent requests, without touching the
            # recording arm/session or primary hysteresis state.
            self._reconcile(camera, "hourly_schedule_disarm")
            self._write(
                camera, "hourly_schedule_reconcile", reason=reason,
                logical_primary_demand=bool(state["logical_primary_demand"]),
                secondary_hourly_pulse_active=bool(state["secondary_active"]),
                hue_desired=bool(state["hue_desired"]),
                arduino_desired=bool(state["arduino_desired"]),
                hue_on=bool(state["on"]),
            )

    def _hourly_skip_reason(self, camera: str, now: float) -> str:
        state = self._states[camera]
        settings = self.settings()
        if not settings["enabled"]:
            return "pressure_service_disabled"
        if not settings["outputs"].get(camera, ""):
            return "output_missing"
        if not self._camera_plate_valid(camera):
            return "plate_invalid"
        if not state["inference_available"]:
            return state["inference_reason"] or "starseg_unavailable"
        if not state["last_seen"]:
            return "occupancy_unknown"
        if now - state["last_seen"] > settings["stale_seconds"]:
            return "starseg_stale"
        if state["occupied"]:
            return "occupied"
        return ""

    def _hourly_tick(self) -> None:
        now_mono = self.monotonic_clock()
        for camera in HOURLY_PUMP_CAMERAS:
            state = self._states[camera]
            deadline = state["secondary_deadline"]
            if state["secondary_active"] and deadline is not None and now_mono >= deadline:
                state["secondary_active"] = False
                state["secondary_deadline"] = None
                self._write(
                    camera, "hourly_pulse_end", reason="duration_complete",
                    cancelled=False, actuator="hue_only",
                )
                self._reconcile(camera, "hourly_pulse_end")
        if not self._hourly_armed or self._next_hourly_at is None:
            return
        now_wall = self.wall_clock()
        if now_wall < self._next_hourly_at:
            return
        opportunity_at = self._next_hourly_at
        # Process at most one opportunity, even after suspension or wall jumps.
        self._next_hourly_at = self._next_local_hour(now_wall)
        for camera in HOURLY_PUMP_CAMERAS:
            state = self._states[camera]
            state["last_opportunity_at"] = opportunity_at
            skip_reason = self._hourly_skip_reason(camera, now_mono)
            if skip_reason:
                state["last_opportunity_result"] = "skipped"
                state["last_skip_reason"] = skip_reason
                self._write(
                    camera, "hourly_opportunity", result="skipped", reason=skip_reason,
                    scheduled_at=opportunity_at,
                    scheduled_local=self._local_display(opportunity_at),
                )
                continue
            state["last_opportunity_result"] = "pulse_started"
            state["last_skip_reason"] = ""
            state["secondary_active"] = True
            state["secondary_deadline"] = now_mono + HOURLY_PULSE_SECONDS
            self._write(
                camera, "hourly_opportunity", result="pulse_started",
                scheduled_at=opportunity_at,
                scheduled_local=self._local_display(opportunity_at),
            )
            self._write(
                camera, "hourly_pulse_start", duration_seconds=HOURLY_PULSE_SECONDS,
                deadline_monotonic=state["secondary_deadline"],
                actuator="hue_only", arduino_desired=False,
            )
            self._reconcile(camera, "hourly_pulse_start")

    def _process_event(self, event: InferenceEvent) -> None:
        state = self._states[event.camera]
        now = self.monotonic_clock()
        settings = self.settings()
        if event.lane != "control":
            return
        if event.session_token != self._session_token:
            return
        if (
            not event.recorded or not self._armed or not settings["enabled"]
            or not event.polygon_valid
        ):
            state["inference_available"] = False
            state["inference_reason"] = (
                "plate_invalid" if not event.polygon_valid else "unarmed_or_invalid"
            )
            self._reconcile(event.camera, "unarmed_or_invalid")
            return
        if not event.inference_ok:
            state["inference_available"] = False
            state["inference_reason"] = "starseg_unavailable"
            if state["occupied"]:
                state["occupied"] = False
                self._write(
                    event.camera, "occupancy", occupied=False, reason="starseg_unavailable",
                    pressure_polygon=[list(point) for point in event.pressure_polygon],
                )
            state["candidate"] = None
            self._reconcile(event.camera, "starseg_unavailable", force=True)
            return
        if event.frame_ts <= state["last_frame_ts"]:
            return
        capture_monotonic = event.capture_monotonic or event.received_monotonic
        frame_age = max(0.0, now - capture_monotonic)
        if frame_age > settings["stale_seconds"]:
            state["inference_available"] = False
            state["inference_reason"] = "stale_inference"
            self._reconcile(event.camera, "stale_inference")
            return
        state["last_frame_ts"] = event.frame_ts
        state["last_seen"] = now
        state["inference_available"] = True
        state["inference_reason"] = ""
        state["last_control_provenance"] = {
            "lane": event.lane, "frame_index": event.frame_index,
            "frame_ts": event.frame_ts,
            "capture_age_ms": round(frame_age * 1000, 2),
            "inference_latency_ms": round(
                max(
                    0.0,
                    event.inference_finished_monotonic
                    - event.inference_started_monotonic,
                ) * 1000,
                2,
            ),
        }
        observed = event.occupied
        if observed == state["occupied"]:
            state["candidate"] = None
            if state["error"]:
                self._reconcile(event.camera, "retry")
            return
        if state["candidate"] is not observed:
            state["candidate"] = observed
            state["candidate_since"] = now
            return
        threshold = settings["enter_seconds"] if observed else settings["exit_seconds"]
        if now - state["candidate_since"] >= threshold:
            state["occupied"] = observed
            state["candidate"] = None
            self._write(
                event.camera, "occupancy", occupied=observed, frame_ts=event.frame_ts,
                frame_index=event.frame_index, lane=event.lane,
                capture_age_ms=round(frame_age * 1000, 2),
                inference_latency_ms=round(
                    max(
                        0.0,
                        event.inference_finished_monotonic
                        - event.inference_started_monotonic,
                    ) * 1000,
                    2,
                ),
                pressure_polygon=[list(point) for point in event.pressure_polygon],
                occupancy_class_names=list(SUNFLOWER_CLASS_NAMES),
                occupancy_policy=(
                    "sunflower_star_roi_test_point_inside_pressure_plate_boundary_inclusive"
                ),
            )
            self._reconcile(event.camera, "occupancy")

    def _stale_sweep(self) -> None:
        now = self.monotonic_clock()
        threshold = self.settings()["stale_seconds"]
        for camera, state in self._states.items():
            if self._armed and state["last_seen"] and now - state["last_seen"] > threshold:
                if state["occupied"]:
                    state["occupied"] = False
                    self._write(camera, "occupancy", occupied=False, reason="starseg_stale")
                state["candidate"] = None
                state["inference_available"] = False
                state["inference_reason"] = "starseg_stale"
                self._reconcile(camera, "starseg_stale")
            elif state["error"]:
                self._reconcile(camera, "retry")

    def run(self) -> None:
        while True:
            try:
                action, value = self._queue.get(timeout=0.2)
            except queue.Empty:
                self._stale_sweep()
                self._hourly_tick()
                if self.shutdown.is_set() and self._closed:
                    break
                continue
            try:
                if action == "inference":
                    self._process_event(value)
                    with self._condition:
                        self._processed += 1
                        self._condition.notify_all()
                elif action == "arduino_event":
                    event, details = value
                    self._record_arduino_event(event, details)
                elif action == "arm":
                    armed, session_token = value
                    if armed:
                        self._armed = True
                        self._session_token = session_token
                        if self.arduino is not None:
                            self.arduino.set_session(True, session_token)
                        for camera, state in self._states.items():
                            state["last_frame_ts"] = float("-inf")
                            state["last_seen"] = 0.0
                            state["inference_available"] = False
                            state["inference_reason"] = "occupancy_unknown"
                            state["last_control_provenance"] = {}
                            self._reconcile(camera, "recording_armed", force=True)
                    else:
                        if self.arduino is not None:
                            self.arduino.set_session(False)
                            zero_confirmed = self.arduino.wait_for_zero()
                            arduino_status = self.arduino.status()
                            for camera in HOURLY_PUMP_CAMERAS:
                                self._write(
                                    camera, "arduino_recording_stop_zero",
                                    desired_mask=0,
                                    confirmed_mask=arduino_status["confirmed_mask"],
                                    acknowledged=zero_confirmed,
                                    serial_path=arduino_status["serial_path"],
                                )
                        self._disarm_all("recording_stopped")
                        self._session_token = ""
                elif action == "hourly_arm":
                    armed, session_token, reply = value
                    if armed and (not self._armed or session_token != self._session_token):
                        reply.put(False)
                        continue
                    self._set_hourly_armed(armed, "operator_disarm")
                    reply.put(True)
                elif action == "event_sink":
                    camera, path, session_token = value
                    if session_token == self._session_token:
                        state = self._states[camera]
                        state["events_jsonl"] = path
                        for row in state["pending_events"]:
                            self._append_event(path, row)
                        state["pending_events"].clear()
                elif action == "barrier":
                    value.put(True)
                elif action == "reconfigure":
                    for camera in CAMERAS:
                        self._reconcile(camera, "reconfigured", True)
                elif action in {"discover", "pair"}:
                    reply = value
                    try:
                        if action == "discover":
                            result = self._bridge().discover_outputs()
                            self._outputs = result
                            self._bridge_status = "connected"
                        else:
                            result = self._bridge().pair()
                        self._bridge_error = ""
                        reply.put(result)
                    except Exception as exc:
                        self._bridge_status = "error"
                        self._bridge_error = str(exc)
                        reply.put(exc)
                elif action == "close":
                    if self.arduino is not None:
                        self.arduino.set_session(False)
                        self.arduino.wait_for_zero()
                    self._disarm_all("shutdown")
                    break
                self._hourly_tick()
            finally:
                self._queue.task_done()

    def status(self) -> dict[str, Any]:
        settings = self.settings()
        arduino = self.arduino.status() if self.arduino is not None else {
            "enabled": False, "connected": False, "ready": False, "watchdog": False,
            "desired_mask": 0, "confirmed_mask": None, "mapping": {},
            "last_error": "Arduino worker is not configured",
        }
        return {
            "enabled": settings["enabled"], "armed": self._armed,
            "hourly_schedule": {
                "armed": self._hourly_armed,
                "next_opportunity_at": self._next_hourly_at,
                "next_opportunity_local": self._local_display(self._next_hourly_at),
                "pulse_seconds": HOURLY_PULSE_SECONDS,
                "cameras": list(HOURLY_PUMP_CAMERAS),
                "policy": HOURLY_SCHEDULE_POLICY,
            },
            "bridge": {
                "address": settings["bridge_address"], "configured": bool(settings["app_key"]),
                "status": self._bridge_status, "error": self._bridge_error,
            },
            "arduino": arduino,
            "split_actuator_warning": any(
                bool(self._states[camera]["split_warning"])
                for camera in HOURLY_PUMP_CAMERAS
            ),
            "outputs": list(self._outputs),
            "cameras": {
                camera: {
                    "output_id": settings["outputs"].get(camera, ""),
                    "armed": (
                        self._armed and settings["enabled"]
                        and self._camera_plate_valid(camera)
                        and bool(settings["outputs"].get(camera, ""))
                    ),
                    "plate_valid": self._camera_plate_valid(camera),
                    "occupied": bool(state["occupied"]), "on": bool(state["on"]),
                    # Compatibility alias: desired has always described the Hue
                    # command target. New consumers should use hue_desired.
                    "desired": bool(state["hue_desired"]),
                    "logical_primary_demand": bool(state["logical_primary_demand"]),
                    "primary_demand": bool(state["logical_primary_demand"]),
                    "hue_desired": bool(state["hue_desired"]),
                    "secondary_pulse_active": bool(state["secondary_active"]),
                    "secondary_pulse_deadline_monotonic": state["secondary_deadline"],
                    "last_opportunity_at": state["last_opportunity_at"],
                    "last_opportunity_local": self._local_display(
                        state["last_opportunity_at"]
                    ),
                    "last_opportunity_result": state["last_opportunity_result"],
                    "last_skip_reason": state["last_skip_reason"],
                    "occupancy_fresh": bool(
                        state["inference_available"] and state["last_seen"]
                        and self.monotonic_clock() - state["last_seen"]
                        <= settings["stale_seconds"]
                    ),
                    "error": state["error"],
                    "hue_commanded": state["last_command"],
                    "hue_reported": bool(state["on"]),
                    "hue_error": state["error"],
                    "arduino_desired": bool(state["arduino_desired"]),
                    "arduino_worker_desired": bool(
                        arduino["desired_mask"]
                        & int(arduino["mapping"].get(camera, 0))
                    ),
                    "arduino_confirmed": (
                        None if arduino["confirmed_mask"] is None else bool(
                            arduino["confirmed_mask"]
                            & int(arduino["mapping"].get(camera, 0))
                        )
                    ),
                    "hue_target_satisfied": bool(
                        not state["error"]
                        and state["on"] == state["hue_desired"]
                    ),
                    "arduino_target_satisfied": (
                        None
                        if not int(arduino["mapping"].get(camera, 0))
                        else bool(
                            arduino["connected"] and arduino["ready"]
                            and not arduino["last_error"]
                            and arduino["confirmed_mask"] is not None
                            and bool(
                                arduino["confirmed_mask"]
                                & int(arduino["mapping"].get(camera, 0))
                            ) == state["arduino_desired"]
                        )
                    ),
                    "actuator_difference_expected": bool(
                        state["hue_desired"] != state["arduino_desired"]
                    ),
                    "actuator_targets_satisfied": not bool(state["split_warning"]),
                    "split_warning": bool(state["split_warning"]),
                }
                for camera, state in self._states.items()
            },
        }

    def selectable_output_ids(self) -> set[str]:
        return {item["id"] for item in self._outputs}
