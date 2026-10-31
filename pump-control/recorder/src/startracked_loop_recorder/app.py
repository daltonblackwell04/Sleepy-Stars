from __future__ import annotations

import copy
import html
import json
import os
import queue
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2

from .arduino_pump import ArduinoPumpWorker
from .camera import Frame, UVCCamera
from .config import (
    CAMERAS, DEFAULT_CONFIG_PATH, ConfigError, atomic_json_write,
    default_deployment, default_user_config, load_json, validate_deployment,
    validate_user_config,
)
from .hue import PressurePlateService
from .injector_pump import InjectorPumpClient
from .starseg import StarSegJob, StarSegWorker
from .storage import (
    BackupWorker, Segment, finalize_segment, recover_staging, safe_trial_name, timestamp,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_PATH = Path(
    os.getenv("BASLER_DEPLOYMENT_CONFIG", PROJECT_ROOT / "deployment.json")
)


def parse_bool(value: str) -> bool:
    if value not in {"0", "1", "true", "false", "on", "off"}:
        raise ConfigError("expected boolean 1|0")
    return value in {"1", "true", "on"}


def validate_camera(value: str) -> str:
    if value not in CAMERAS:
        raise ConfigError("camera must be A, B, C, or D")
    return value


class Recorder:
    def __init__(self, deployment_path: Path = DEPLOYMENT_PATH, config_path: Path = DEFAULT_CONFIG_PATH):
        if not deployment_path.exists():
            atomic_json_write(deployment_path, default_deployment(PROJECT_ROOT))
        self.deployment = load_json(
            deployment_path, default_deployment(PROJECT_ROOT), validate_deployment
        )
        self.config_path = config_path
        self.config = load_json(config_path, default_user_config(), validate_user_config)
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        self.recording = False
        self.session = ""
        self.session_started = 0.0
        self.active_trial_name = ""
        self.stop_requested = False
        self.queues = {c: queue.Queue(maxsize=120) for c in CAMERAS}
        self.segments: dict[str, Segment | None] = {c: None for c in CAMERAS}
        self.next_record_at = {c: 0.0 for c in CAMERAS}
        self.control_frame_indices = {c: 0 for c in CAMERAS}
        self.next_control_at = {c: 0.0 for c in CAMERAS}
        self.cameras: dict[str, UVCCamera] = {}
        self.encoders: list[threading.Thread] = []
        self.finalizers: set[threading.Thread] = set()
        self.errors: list[str] = []
        self.finalized_segments = 0
        self.recovery = recover_staging(Path(self.config["record_root"]))
        self.arduino = self._make_pump_worker()
        self.pressure = PressurePlateService(
            self._pressure_settings, self.shutdown_event, arduino=self.arduino
        )
        self.arduino.event_callback = self.pressure.submit_arduino_event
        self.starseg = StarSegWorker(
            lambda: copy.deepcopy(self.config["starseg_display"]), self.shutdown_event,
            self.pressure.submit,
        )
        self.starseg.configure_model(self.config.get("starseg_model", ""))
        self.starseg.request_reload()
        self.backup: BackupWorker | None = None

    def _make_pump_worker(self):
        mode = self.config["arduino_pump"].get("mode", "injector")
        if mode == "direct":
            return ArduinoPumpWorker(self._arduino_settings)
        return InjectorPumpClient(self._arduino_settings)

    def _pressure_settings(self) -> dict[str, Any]:
        with self.lock:
            settings = copy.deepcopy(self.config["pressure_plate"])
            settings["polygons"] = copy.deepcopy(self.config["pressure_plate_polygons"])
            return settings

    def _arduino_settings(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.config["arduino_pump"])

    def save_config(self) -> None:
        validate_user_config(self.config)
        atomic_json_write(self.config_path, self.config)

    def start_workers(self) -> None:
        self.arduino.start()
        self.pressure.start()
        self.starseg.start()
        self._configure_backup()
        for letter in CAMERAS:
            spec = self.deployment["cameras"][letter]
            camera = UVCCamera(
                letter, spec["path"], self.deployment["width"], self.deployment["height"],
                self.queues[letter],
                lambda c=letter: copy.deepcopy(self.config["camera_controls"][c]),
                self.shutdown_event,
                self._submit_control_frame,
            )
            self.cameras[letter] = camera
            camera.start()
            encoder = threading.Thread(
                target=self._encoder_loop, args=(letter,), name=f"encoder-{letter}", daemon=True
            )
            self.encoders.append(encoder)
            encoder.start()

    def _configure_backup(self) -> None:
        settings = self.config["backup"]
        if not settings["enabled"]:
            if self.backup:
                self.backup.close()
                self.backup.join(timeout=10)
                self.backup = None
            return
        candidate = BackupWorker(
            Path(self.config["record_root"]), Path(settings["root"]), settings["throttle_mbps"]
        )
        candidate.validate()
        if self.backup:
            self.backup.close()
            self.backup.join(timeout=10)
        self.backup = candidate
        candidate.start()

    def _jpeg(self, image, quality: int) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return encoded.tobytes()

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _new_segment(self, letter: str, frame: Frame) -> Segment:
        controls = self.config["camera_controls"][letter]
        spec = self.deployment["cameras"][letter]
        segment = Segment.create(
            camera=letter, serial=spec["path"], model=spec.get("model", "Arducam UVC"),
            trial_name=self.active_trial_name, session=self.session,
            session_started=self.session_started, start_ts=frame.timestamp,
            record_root=Path(self.config["record_root"]),
            fps=float(controls.get("fps", self.deployment["record_fps"])),
            requested_seconds=float(self.deployment["segment_seconds"]),
            width=int(frame.image.shape[1]), height=int(frame.image.shape[0]),
            polygon=copy.deepcopy(self.config["camera_polygons"][letter]),
            pressure_polygon=copy.deepcopy(self.config["pressure_plate_polygons"][letter]),
        )
        self.pressure.set_event_sink(
            letter, segment.staging / "pressure_plate_events.jsonl", self.session
        )
        return segment

    def _should_roll(self, segment: Segment) -> bool:
        return (
            time.time() - segment.start_ts >= segment.requested_seconds
            or segment.record_root != Path(self.config["record_root"])
            or segment.session != self.session
        )

    def _submit_control_frame(self, frame: Frame) -> None:
        """Capture-thread fast path; never waits for the archival queue."""
        with self.lock:
            if not self.config["starseg_display"]["enabled"]:
                return
            camera = frame.camera
            captured = frame.capture_monotonic or time.monotonic()
            if captured < self.next_control_at[camera]:
                return
            self.next_control_at[camera] = captured + 1.0
            recording = self.recording and not self.stop_requested
            segment = self.segments[camera] if recording else None
            frame_index = self.control_frame_indices[camera]
            self.control_frame_indices[camera] += 1
            session = self.session if recording else ""
            polygon = copy.deepcopy(
                segment.polygon if segment else self.config["camera_polygons"][camera]
            )
            pressure_polygon = copy.deepcopy(
                segment.pressure_polygon
                if segment else self.config["pressure_plate_polygons"][camera]
            )
            trial_name = (
                segment.trial_name if segment
                else self.active_trial_name or self.config["trial_name"]
            )
            jsonl = (
                segment.staging / "starseg_frames.jsonl"
                if segment is not None else None
            )
        self.starseg.submit_control(StarSegJob(
            camera=camera, frame_index=frame_index, frame_ts=frame.timestamp,
            frame_file="", image=frame.image.copy(), polygon=polygon,
            trial_name=trial_name, jsonl=jsonl, recorded=recording,
            session_token=session, pressure_polygon=pressure_polygon,
            capture_monotonic=captured,
        ))

    def _finish_async(self, segment: Segment, end_ts: float) -> None:
        def work():
            try:
                try:
                    self.starseg.wait_recorded(segment.staging / "starseg_frames.jsonl")
                except RuntimeError as exc:
                    # wait_recorded reports inference failures only after jobs drain.
                    # TimeoutError must still abort: jobs may be writing sidecars.
                    atomic_json_write(segment.staging / "starseg_inference_failure.json", {
                        "inference_status": "failed", "inference_error": str(exc),
                    })
                    with self.lock:
                        self.errors.append(f"{segment.camera} recorded inference: {exc}")
                self.pressure.flush()
                files = finalize_segment(
                    segment, end_ts, self.deployment["x264_crf"],
                    self.deployment["x264_preset"], self.deployment["delete_failed_segments"],
                )
                if self.backup:
                    metadata = json.loads(files[1].read_text(encoding="utf-8"))
                    folder = Path(metadata["folder"])
                    extras = []
                    for key in (
                        "pressure_plate_events_jsonl",
                        "hydra_frames_jsonl",
                        "hydra_summary_json",
                    ):
                        raw = metadata.get(key)
                        if raw:
                            extras.append(Path(raw))
                    backup_files = []
                    seen: set[Path] = set()
                    candidates = [*files, *extras]
                    if folder.is_dir():
                        candidates.extend(sorted(folder.iterdir()))
                    for path in candidates:
                        if not path.is_file() or path.name.startswith("."):
                            continue
                        resolved = path.resolve()
                        if resolved in seen:
                            continue
                        seen.add(resolved)
                        backup_files.append(path)
                    for path in backup_files:
                        self.backup.enqueue(path)
                with self.lock:
                    self.finalized_segments += 1
            except Exception as exc:
                with self.lock:
                    self.errors.append(f"{segment.camera} finalization: {exc}")
            finally:
                with self.lock:
                    self.finalizers.discard(threading.current_thread())
        thread = threading.Thread(target=work, name=f"finalize-{segment.camera}", daemon=True)
        with self.lock:
            self.finalizers.add(thread)
        thread.start()

    def _finish_remaining_segments(self) -> None:
        with self.lock:
            remaining = [
                segment for segment in self.segments.values() if segment is not None
            ]
            self.segments = {c: None for c in CAMERAS}
            self.stop_requested = False
        for segment in remaining:
            end_ts = (
                segment.frame_timestamps[-1]
                if segment.frame_timestamps
                else time.time()
            )
            self._finish_async(segment, end_ts)

    def _encoder_loop(self, letter: str) -> None:
        while not self.shutdown_event.is_set() or not self.queues[letter].empty():
            try:
                frame = self.queues[letter].get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                preview = self._jpeg(frame.image, self.deployment["preview_jpeg_quality"])
                preview_path = Path(self.config["record_root"]) / "preview" / f"{letter}.jpg"
                self._atomic_bytes(preview_path, preview)
                with self.lock:
                    recording = self.recording
                    segment = self.segments[letter]
                    stop = self.stop_requested
                if segment and (stop or self._should_roll(segment)):
                    self._finish_async(segment, frame.timestamp)
                    with self.lock:
                        self.segments[letter] = None
                    segment = None
                if recording and not stop:
                    if segment is None:
                        segment = self._new_segment(letter, frame)
                        with self.lock:
                            self.segments[letter] = segment
                        self.next_record_at[letter] = 0
                    if frame.timestamp >= self.next_record_at[letter]:
                        jpeg = self._jpeg(frame.image, self.deployment["jpeg_quality"])
                        frame_path = segment.add_frame(jpeg, frame.timestamp)
                        enabled = bool(self.config["starseg_display"]["enabled"])
                        if enabled:
                            self.starseg.submit(StarSegJob(
                                camera=letter, frame_index=segment.frames - 1,
                                frame_ts=frame.timestamp, frame_file=frame_path.name,
                                image=frame.image.copy(), polygon=copy.deepcopy(segment.polygon),
                                trial_name=segment.trial_name,
                                jsonl=segment.staging / "starseg_frames.jsonl",
                                recorded=True, session_token=self.session,
                                pressure_polygon=copy.deepcopy(segment.pressure_polygon),
                                capture_monotonic=(
                                    frame.capture_monotonic or time.monotonic()
                                ),
                            ))
                        self.next_record_at[letter] = frame.timestamp + (1 / segment.fps)
                with self.lock:
                    if self.stop_requested and all(v is None for v in self.segments.values()):
                        self.stop_requested = False
            except Exception as exc:
                with self.lock:
                    self.errors.append(f"{letter} encoder: {exc}")
            finally:
                self.queues[letter].task_done()

    def set_recording(self, enabled: bool) -> None:
        with self.lock:
            if enabled:
                if self.recording:
                    return
                self.starseg.clear_control()
                self.starseg.request_reload()
                self.control_frame_indices = {c: 0 for c in CAMERAS}
                self.next_control_at = {c: 0.0 for c in CAMERAS}
                self.session_started = time.time()
                self.active_trial_name = self.config["trial_name"]
                self.session = (
                    f"{safe_trial_name(self.active_trial_name)}"
                    f"_recording_start-{timestamp(self.session_started)}"
                )
                self.recording = True
                self.stop_requested = False
                self.pressure.arm(True, self.session)
            else:
                self.starseg.clear_control(self.session)
                self.pressure.arm(False)
                self.recording = False
                self.stop_requested = any(v is not None for v in self.segments.values())

    def set_hourly_schedule(self, enabled: bool) -> None:
        with self.lock:
            if enabled and (not self.recording or self.stop_requested):
                raise ConfigError("hourly pump schedule requires an active recording")
            session = self.session
        self.pressure.set_hourly_schedule(enabled, session)

    def status(self) -> dict[str, Any]:
        with self.lock:
            camera_status = {}
            for c in CAMERAS:
                status = copy.deepcopy(self.cameras[c].status) if c in self.cameras else {}
                status.update({
                    "queue_size": self.queues[c].qsize(),
                    "active_segment": str(self.segments[c].staging) if self.segments[c] else None,
                    "controls": copy.deepcopy(self.config["camera_controls"][c]),
                })
                camera_status[c] = status
            return {
                "product": "Startracked Loop Recorder", "healthy": not self.shutdown_event.is_set(),
                "recording": self.recording, "stop_requested": self.stop_requested,
                "session": self.session, "record_root": self.config["record_root"],
                "trial_name": self.config["trial_name"],
                "active_trial_name": self.active_trial_name if self.recording or self.stop_requested else "",
                "cameras": camera_status,
                "camera_polygons": copy.deepcopy(self.config["camera_polygons"]),
                "pressure_plate_polygons": copy.deepcopy(
                    self.config["pressure_plate_polygons"]
                ),
                "starseg": self.starseg.status(),
                "pressure_plate": self.pressure.status(),
                "backup": {
                    **copy.deepcopy(self.config["backup"]),
                    "status": copy.deepcopy(self.backup.metrics) if self.backup else
                    {"disabled": True, "reason": "No validated external mount configured"},
                },
                "finalizers": len(self.finalizers), "finalized_segments": self.finalized_segments,
                "errors": self.errors[-20:], "recovery": self.recovery,
            }

    def pair_hue(self) -> dict[str, Any]:
        key = self.pressure.pair()
        with self.lock:
            self.config["pressure_plate"]["app_key"] = key
            self.save_config()
            self.pressure.reconfigure()
        return self.status()

    def discover_hue(self) -> dict[str, Any]:
        self.pressure.discover()
        return self.status()

    def apply_control(self, fields: dict[str, list[str]]) -> dict[str, Any]:
        def one(name: str) -> str | None:
            values = fields.get(name)
            return values[-1] if values else None
        if one("hourly_pumps_armed") is not None:
            if set(fields) != {"hourly_pumps_armed"}:
                raise ConfigError("hourly pump schedule must be changed as a separate action")
            self.set_hourly_schedule(parse_bool(one("hourly_pumps_armed")))
            return self.status()
        with self.lock:
            if one("trial_name") is not None:
                if self.recording or self.stop_requested:
                    raise ConfigError("trial name is immutable while recording is active")
                trial_name = one("trial_name").strip()
                if not trial_name or len(trial_name) > 120:
                    raise ConfigError("trial_name must be 1-120 characters")
                self.config["trial_name"] = trial_name
            if one("recording") is not None:
                self.set_recording(parse_bool(one("recording")))
            if one("record_root") is not None:
                root = Path(one("record_root")).expanduser()
                if not root.is_absolute():
                    raise ConfigError("record_root must be absolute")
                root.mkdir(parents=True, exist_ok=True)
                self.config["record_root"] = str(root)
            camera = validate_camera(one("camera")) if one("camera") else None
            if camera:
                controls = self.config["camera_controls"][camera]
                ranges = {"fps": (0.1, 120), "exposure_us": (1, 10_000_000),
                          "gain": (0, 36), "brightness": (0, 1)}
                for key, limits in ranges.items():
                    if one(key) is not None:
                        value = float(one(key))
                        if not limits[0] <= value <= limits[1]:
                            raise ConfigError(f"{key} outside {limits}")
                        if key == "exposure_us" and camera in self.cameras:
                            probed = self.cameras[camera].status.get("uvc", {}).get(
                                "ranges", {}
                            ).get("exposure_us", {})
                            if probed and not probed.get("min", value) <= value <= probed.get("max", value):
                                raise ConfigError(
                                    f"exposure_us outside probed range "
                                    f"{probed.get('min')}-{probed.get('max')}"
                                )
                        controls[key] = value
                for key, options in {
                    "exposure_mode": {"Auto", "Manual"},
                    "gain_auto": {"Off", "Once", "Continuous"},
                    "white_balance": {"Off", "Once", "Continuous"},
                }.items():
                    if one(key) is not None:
                        if one(key) not in options:
                            raise ConfigError(f"invalid {key}")
                        controls[key] = one(key)
            if one("camera_polygon") is not None:
                if not camera:
                    raise ConfigError("camera required for polygon")
                if self.recording or self.stop_requested:
                    raise ConfigError("geometry is immutable while recording is active")
                try:
                    points = json.loads(one("points") or "")
                except json.JSONDecodeError as exc:
                    raise ConfigError("points must be JSON") from exc
                test = copy.deepcopy(self.config)
                test["camera_polygons"][camera] = points
                validate_user_config(test)
                self.config["camera_polygons"][camera] = points
            if one("pressure_plate_polygon") is not None:
                if not camera:
                    raise ConfigError("camera required for pressure-plate polygon")
                if self.recording or self.stop_requested:
                    raise ConfigError("geometry is immutable while recording is active")
                try:
                    points = json.loads(one("pressure_points") or "")
                except json.JSONDecodeError as exc:
                    raise ConfigError("pressure_points must be JSON") from exc
                test = copy.deepcopy(self.config)
                test["pressure_plate_polygons"][camera] = points
                validate_user_config(test)
                self.config["pressure_plate_polygons"][camera] = points
                self.pressure.reconfigure()
            if one("starseg_display") is not None:
                display = self.config["starseg_display"]
                display["enabled"] = parse_bool(one("starseg_display"))
                for key in ("overlay", "show_boxes", "show_masks", "show_labels"):
                    if one(key) is not None:
                        display[key] = parse_bool(one(key))
                if one("confidence") is not None:
                    value = float(one("confidence"))
                    if not 0 <= value <= 1:
                        raise ConfigError("confidence must be 0-1")
                    display["confidence"] = value
                if one("imgsz") is not None:
                    value = int(one("imgsz"))
                    if not 128 <= value <= 4096:
                        raise ConfigError("imgsz must be 128-4096")
                    display["imgsz"] = value
            if one("starseg_model") is not None:
                model = one("starseg_model")
                if model and not Path(model).is_absolute():
                    raise ConfigError("starseg model must be absolute")
                self.config["starseg_model"] = model
                self.starseg.configure_model(model)
            pressure_changed = any(one(key) is not None for key in (
                "pressure_enabled", "hue_bridge_address", "hue_app_key",
                "hue_output", "pressure_enter_seconds", "pressure_exit_seconds",
                "pressure_stale_seconds",
            ))
            if pressure_changed:
                if self.recording or self.stop_requested:
                    raise ConfigError("Hue configuration is immutable while recording is active")
                candidate = copy.deepcopy(self.config["pressure_plate"])
                if one("pressure_enabled") is not None:
                    candidate["enabled"] = parse_bool(one("pressure_enabled"))
                if one("hue_bridge_address") is not None:
                    candidate["bridge_address"] = one("hue_bridge_address").strip()
                if one("hue_app_key") is not None:
                    supplied_key = one("hue_app_key").strip()
                    if supplied_key:
                        candidate["app_key"] = supplied_key
                if one("hue_output") is not None:
                    if not camera:
                        raise ConfigError("camera required for Hue output assignment")
                    output_id = one("hue_output").strip()
                    if output_id and output_id not in self.pressure.selectable_output_ids():
                        raise ConfigError("Hue output is not a discovered switchable light or plug")
                    candidate["outputs"][camera] = output_id
                for field, key in (
                    ("pressure_enter_seconds", "enter_seconds"),
                    ("pressure_exit_seconds", "exit_seconds"),
                    ("pressure_stale_seconds", "stale_seconds"),
                ):
                    if one(field) is not None:
                        candidate[key] = float(one(field))
                test = copy.deepcopy(self.config)
                test["pressure_plate"] = candidate
                validate_user_config(test)
                self.config["pressure_plate"] = candidate
                self.pressure.reconfigure()
            backup_changed = any(one(key) is not None for key in (
                "backup_enabled", "backup_root", "backup_throttle_mbps"
            ))
            backup_candidate = copy.deepcopy(self.config["backup"])
            if one("backup_enabled") is not None:
                backup_candidate["enabled"] = parse_bool(one("backup_enabled"))
            if one("backup_root") is not None:
                root = one("backup_root")
                if root and not Path(root).is_absolute():
                    raise ConfigError("backup root must be absolute")
                backup_candidate["root"] = root
            if one("backup_throttle_mbps") is not None:
                throttle = float(one("backup_throttle_mbps"))
                if throttle < 0:
                    raise ConfigError("backup throttle cannot be negative")
                backup_candidate["throttle_mbps"] = throttle
            if backup_changed and backup_candidate["enabled"]:
                BackupWorker(
                    Path(self.config["record_root"]), Path(backup_candidate["root"]),
                    backup_candidate["throttle_mbps"],
                ).validate()
            if backup_changed:
                self.config["backup"] = backup_candidate
                self._configure_backup()
            self.save_config()
        return self.status()

    def close(self) -> None:
        self.set_recording(False)
        deadline = time.monotonic() + 10
        while self.stop_requested and time.monotonic() < deadline:
            time.sleep(0.1)
        self.shutdown_event.set()
        for camera in self.cameras.values():
            camera.join(timeout=3)
        for encoder in self.encoders:
            encoder.join(timeout=3)
        self._finish_remaining_segments()
        self.starseg.close()
        for finalizer in list(self.finalizers):
            finalizer.join(timeout=60)
        if self.starseg.ident is not None:
            self.starseg.join(timeout=60)
        if self.starseg.is_alive():
            with self.lock:
                self.errors.append("StarSeg worker did not stop cleanly")
        self.pressure.close()
        if self.pressure.ident is not None:
            self.pressure.join(timeout=10)
        if self.pressure.is_alive():
            with self.lock:
                self.errors.append("Hue pressure-plate worker did not stop cleanly")
        self.arduino.close()
        if self.arduino.is_alive():
            with self.lock:
                self.errors.append("Arduino pump worker did not stop cleanly")
        if self.backup:
            self.backup.close()
            self.backup.join(timeout=10)


def page(popout: str | None = None) -> bytes:
    cards = "".join(
        f"""<section class="card" data-camera="{c}">
        <header><b>Camera {c}</b><span class="pill" id="state-{c}">connecting</span>
        <button onclick="window.open('/popout/{c}','camera-{c}')">Popout</button>
        <button onclick="fullscreen('{c}')">Fullscreen</button></header>
        <div class="view" id="view-{c}"><img id="img-{c}" src="/preview/{c}.jpg"><svg id="svg-{c}" preserveAspectRatio="none"></svg></div>
        <div class="tools"><button onclick="zoom('{c}',1.2)">+</button><button onclick="zoom('{c}',.83)">−</button>
        <button onclick="resetView('{c}')">Reset view</button><button id="add-{c}" class="mode" aria-pressed="false" onclick="toggleAddMode('{c}')">Add vertex</button>
        <button onclick="removeVertex('{c}')">Remove selected</button><button onclick="undoPolygon('{c}')">Undo</button>
        <button onclick="resetPolygon('{c}')">Reset polygon</button><button onclick="cancelPolygon('{c}')">Cancel changes</button>
        <button id="polygon-apply-{c}" class="primary" onclick="applyPolygon('{c}')">Apply</button>
        <span id="edit-{c}"></span></div>
        <div class="tools plate-tools"><b>Pressure plate</b>
        <button id="plate-add-{c}" class="mode" aria-pressed="false" onclick="togglePlateAddMode('{c}')">Add plate vertex</button>
        <button onclick="removePlateVertex('{c}')">Remove selected</button><button onclick="undoPlate('{c}')">Undo</button>
        <button onclick="resetPlate('{c}')">Reset pressure plate</button><button onclick="cancelPlate('{c}')">Cancel plate changes</button>
        <button id="plate-apply-{c}" class="primary" onclick="applyPlate('{c}')">Apply plate</button>
        <span id="plate-edit-{c}"></span></div>
        <small>StarSeg ROI is cyan. The separate pressure plate is amber and must stay entirely inside the ROI. Boundary points count as occupied.</small>
        <details><summary>Status / Controls</summary>
        <label>FPS <input id="fps-{c}" type="number" min=".1" max="120" step=".1" value="1" oninput="markControlDirty('{c}','fps')"></label>
        <label>Exposure <select id="exp-mode-{c}" onchange="markControlDirty('{c}','mode');exposureMode('{c}')"><option>Auto</option><option>Manual</option></select></label>
        <label>Exposure µs <input id="exp-{c}" type="number" value="10000" disabled oninput="markControlDirty('{c}','exposure')"></label>
        <label>Gain dB <input id="gain-{c}" type="number" min="0" max="36" value="0" oninput="markControlDirty('{c}','gain')"></label>
        <button id="apply-{c}" onclick="applyCamera('{c}')">Apply</button>
        <label>Hue output <select id="hue-{c}" onchange="assignHue('{c}')"><option value="">Unassigned (OFF)</option></select></label>
        <span id="plate-{c}" class="pill">Disarmed · Empty · Hue off · Arduino unknown</span>
        <span id="control-status-{c}" role="status" aria-live="polite"></span>
        <pre id="info-{c}"></pre></details></section>"""
        for c in (CAMERAS if popout is None else (popout,))
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Startracked Loop Recorder</title>
<style>
:root{{--bg:#07131b;--panel:#102631;--ink:#e7f5f6;--cyan:#46d5ce;--red:#ff6272}}*{{box-sizing:border-box}}
body{{margin:0;background:linear-gradient(135deg,#061117,#0c202b);color:var(--ink);font:14px system-ui}}
.trialbar{{position:sticky;top:0;z-index:5;padding:12px 18px;background:#06363a;display:flex;gap:12px;align-items:center}}
.trialbar input{{font-size:18px;flex:1;min-width:220px}}.top{{position:sticky;top:57px;z-index:4;padding:12px 18px;background:#08171eee;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
h1{{font-size:20px;margin:0}}input,select,button{{background:#173541;color:var(--ink);border:1px solid #315666;border-radius:5px;padding:7px}}
button{{cursor:pointer}}button.primary{{background:#087c77}}button.stop{{background:#9d2635}}
button.mode[aria-pressed="true"]{{background:#ffcf40;color:#07131b;border-color:#fff;box-shadow:0 0 0 2px #ffcf4066}}
.grid{{padding:14px;display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:14px}}.card{{background:var(--panel);padding:10px;border-radius:8px}}
.card header,.tools{{display:flex;gap:8px;align-items:center;margin-bottom:8px}}.card header b{{margin-right:auto}}.pill{{color:var(--cyan)}}.warning{{color:#fff;background:#9d2635;padding:6px;border-radius:5px;font-weight:700}}
.view{{position:relative;overflow:hidden;background:#000;aspect-ratio:16/9;touch-action:none}}.view img,.view svg{{position:absolute;width:100%;height:100%;object-fit:contain;transform-origin:center}}
.view svg{{z-index:2}}circle.vertex{{fill:#46d5ce;stroke:#111;stroke-width:2;cursor:grab;pointer-events:all}}circle.vertex.selected{{fill:#fff;stroke:#ff6272;stroke-width:3}}.boundary{{fill:none;stroke:#46d5ce;stroke-width:2;pointer-events:none}}.pressure-boundary{{fill:#ffcf4022;stroke:#ffcf40;stroke-width:3;stroke-dasharray:8 4;pointer-events:none}}circle.pressure-vertex{{fill:#ffcf40;stroke:#111;stroke-width:2;cursor:grab;pointer-events:all}}circle.pressure-vertex.selected{{fill:#fff;stroke:#ff8a00;stroke-width:3}}.plate-tools{{border-top:1px solid #315666;padding-top:8px}}.overlay{{pointer-events:none}}
pre{{white-space:pre-wrap;font-size:11px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="trialbar"><b>Trial name</b><input id="trialName" value="Startracked Loop Recorder" maxlength="120" aria-label="Trial name"><span id="trialLock"></span></div>
<div class="top"><h1>Startracked Loop Recorder</h1>
<button class="primary" onclick="control({{trial_name:trialName.value,recording:1}})">Start</button><button class="stop" onclick="control({{recording:0}})">Stop</button>
<button id="hourlyPumpArm" onclick="toggleHourlyPumps()">Arm hourly A+B+C+D Hue pumps</button>
<span id="hourlyPumpStatus">Hourly A+B+C+D Hue pumps disarmed</span>
<input id="root" size="38" aria-label="Record root"><button onclick="saveGlobals()">Save</button>
<label><input id="starsegEnabled" type="checkbox"> StarSeg</label><input id="starsegModel" size="24" placeholder="/absolute/model.pt">
<label><input id="backupEnabled" type="checkbox"> Backup</label><input id="backupRoot" size="24" placeholder="/mnt/backup">
<input id="backupThrottle" type="number" min="0" value="8" title="Backup MB/s" style="width:65px">
<label><input id="pressureEnabled" type="checkbox"> Hue plates</label>
<input id="hueBridge" size="15" value="192.0.2.10" aria-label="Hue bridge address">
<input id="hueKey" type="password" size="18" placeholder="application key" aria-label="Hue application key">
<button onclick="pairHue()">Pair (press bridge button)</button><button onclick="discoverHue()">Discover outputs</button>
<span id="hueStatus"></span>
<span id="arduinoStatus"></span>
<span id="global"></span></div><main class="grid">{cards}</main>
<script>
const cameras={json.dumps(list(CAMERAS if popout is None else (popout,)))}, views={{}}, polygonDefaults=[[.08,.08],[.92,.08],[.92,.92],[.08,.92]],svgNS='http://www.w3.org/2000/svg';
function geometryIsLocked(){{return typeof trialName!=='undefined'&&Boolean(trialName.disabled)}}
async function control(data){{let b=new URLSearchParams(data);let r=await fetch('/control',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body:b}}),j=await r.json();if(!r.ok){{let message=j.error||'Request failed ('+r.status+')';alert(message);throw new Error(message)}}return j}}
function snapshot(c){{let v=views[c];v.history.push(structuredClone(v.polygon));if(v.history.length>50)v.history.shift()}}
function setEditStatus(c,text){{document.getElementById('edit-'+c).textContent=text}}
function updatePolygonGuidance(c,prefix=''){{let v=views[c],count=v.polygon.length,needed=Math.max(0,3-count),button=document.getElementById('polygon-apply-'+c),noun=count===1?'vertex':'vertices',text=count+' '+noun;if(needed)text+=' · Add '+needed+' more to apply';else text+=' · Ready to apply';if(v.dirty)text+=' · Unsaved';if(v.addMode)text='Adding vertices · '+text;button.disabled=geometryIsLocked()||needed>0;setEditStatus(c,prefix?prefix+' · '+text:text)}}
function dirty(c){{let v=views[c];v.dirty=true;v.editing=true}}
function previewGeometry(c){{let v=views[c],s=document.getElementById('svg-'+c),img=document.getElementById('img-'+c),width=Math.max(1,s.clientWidth||s.getBoundingClientRect().width),height=Math.max(1,s.clientHeight||s.getBoundingClientRect().height),imageWidth=img.naturalWidth||v.imageWidth||16,imageHeight=img.naturalHeight||v.imageHeight||9,scale=Math.min(width/imageWidth,height/imageHeight),drawWidth=imageWidth*scale,drawHeight=imageHeight*scale;return{{width,height,x:(width-drawWidth)/2,y:(height-drawHeight)/2,drawWidth,drawHeight}}}}
function prepareSvg(s,g){{s.setAttribute('viewBox',`0 0 ${{g.width}} ${{g.height}}`);s.setAttribute('preserveAspectRatio','none')}}
function renderPoint(g,p){{return[g.x+p[0]*g.drawWidth,g.y+p[1]*g.drawHeight]}}
function svgPoint(c,event){{let s=document.getElementById('svg-'+c),r=s.getBoundingClientRect(),g=previewGeometry(c),x=(event.clientX-r.left)*g.width/r.width,y=(event.clientY-r.top)*g.height/r.height;return[Math.max(0,Math.min(1,(x-g.x)/g.drawWidth)),Math.max(0,Math.min(1,(y-g.y)/g.drawHeight))]}}
function ensureSvgGroups(s){{let boundary=s.querySelector('[data-layer="boundary"]'),pressure=s.querySelector('[data-layer="pressure"]'),overlay=s.querySelector('[data-layer="overlay"]');if(!boundary){{boundary=document.createElementNS(svgNS,'g');boundary.dataset.layer='boundary';s.appendChild(boundary)}}if(!pressure){{pressure=document.createElementNS(svgNS,'g');pressure.dataset.layer='pressure';s.appendChild(pressure)}}if(!overlay){{overlay=document.createElementNS(svgNS,'g');overlay.dataset.layer='overlay';overlay.setAttribute('class','overlay');s.appendChild(overlay)}}return{{boundary,pressure,overlay}}}}
function draw(c,pts){{let v=views[c],s=document.getElementById('svg-'+c),g=previewGeometry(c),layers=ensureSvgGroups(s);prepareSvg(s,g);v.polygon=pts;layers.boundary.replaceChildren();let rendered=pts.map(p=>renderPoint(g,p));if(rendered.length){{let shape=document.createElementNS(svgNS,pts.length>=3?'polygon':'polyline');shape.setAttribute('class','boundary');shape.setAttribute('points',rendered.map(p=>p[0]+','+p[1]).join(' '));layers.boundary.appendChild(shape)}}rendered.forEach((p,i)=>{{let e=document.createElementNS(svgNS,'circle');e.setAttribute('class','vertex'+(v.selected===i?' selected':''));e.dataset.vertex=String(i);e.setAttribute('cx',String(p[0]));e.setAttribute('cy',String(p[1]));e.setAttribute('r',v.selected===i?'10':'7');layers.boundary.appendChild(e)}});updatePolygonGuidance(c)}}
function updatePlateGuidance(c,prefix=''){{let v=views[c],count=v.pressurePolygon.length,button=document.getElementById('plate-apply-'+c),text=count?(count+' vertices · '+(count<3?'Add '+(3-count)+' more to apply':'Ready to apply')):'0 vertices · Empty (fail closed)';if(v.pressureDirty)text+=' · Unsaved';if(v.plateAddMode)text='Adding plate vertices · '+text;button.disabled=geometryIsLocked()||(count>0&&count<3);document.getElementById('plate-edit-'+c).textContent=prefix?prefix+' · '+text:text}}
function drawPlate(c,pts){{let v=views[c],s=document.getElementById('svg-'+c),g=previewGeometry(c),layer=ensureSvgGroups(s).pressure;prepareSvg(s,g);v.pressurePolygon=pts;layer.replaceChildren();let rendered=pts.map(p=>renderPoint(g,p));if(rendered.length){{let shape=document.createElementNS(svgNS,pts.length>=3?'polygon':'polyline');shape.setAttribute('class','pressure-boundary');shape.setAttribute('points',rendered.map(p=>p[0]+','+p[1]).join(' '));layer.appendChild(shape)}}rendered.forEach((p,i)=>{{let e=document.createElementNS(svgNS,'circle');e.setAttribute('class','pressure-vertex'+(v.pressureSelected===i?' selected':''));e.dataset.pressureVertex=String(i);e.setAttribute('cx',String(p[0]));e.setAttribute('cy',String(p[1]));e.setAttribute('r',v.pressureSelected===i?'10':'7');layer.appendChild(e)}});updatePlateGuidance(c)}}
function zoom(c,v){{views[c].z*=v;transform(c)}}function transform(c){{let v=views[c],value=`translate(${{v.x}}px,${{v.y}}px) scale(${{v.z}})`;document.getElementById('img-'+c).style.transform=value;document.getElementById('svg-'+c).style.transform=value}}
function resetView(c){{views[c].z=1;views[c].x=views[c].y=0;transform(c)}}
function setAddMode(c,enabled){{if(enabled&&geometryIsLocked())return;let v=views[c];if(enabled&&v.plateAddMode)setPlateAddMode(c,false);v.addMode=enabled;v.editing=enabled||v.dirty;let button=document.getElementById('add-'+c);button.setAttribute('aria-pressed',String(enabled));button.textContent=enabled?'Exit add mode':'Add vertex';updatePolygonGuidance(c)}}
function toggleAddMode(c){{setAddMode(c,!views[c].addMode)}}
function appendVertex(c,point){{if(geometryIsLocked())return;let v=views[c];if(v.polygon.length>=64)return alert('Maximum 64 vertices');snapshot(c);v.polygon.push(point);v.selected=v.polygon.length-1;dirty(c);draw(c,v.polygon)}}
function removeVertex(c){{if(geometryIsLocked())return;let v=views[c];if(v.selected==null)return alert('Select a vertex first');snapshot(c);v.polygon.splice(v.selected,1);v.selected=null;dirty(c);draw(c,v.polygon)}}
function undoPolygon(c){{if(geometryIsLocked())return;let v=views[c],p=v.history.pop();if(p){{v.polygon=p;v.selected=null;dirty(c);draw(c,p)}}}}
function resetPolygon(c){{if(geometryIsLocked())return;snapshot(c);views[c].selected=null;dirty(c);draw(c,[])}}
function cancelPolygon(c){{let v=views[c];v.polygon=structuredClone(v.persisted);v.dirty=false;v.editing=false;v.history=[];v.selected=null;setAddMode(c,false);draw(c,v.polygon);setEditStatus(c,'Changes discarded')}}
async function applyPolygon(c){{let v=views[c],button=document.getElementById('polygon-apply-'+c);if(v.polygon.length<3){{updatePolygonGuidance(c,'Cannot apply');return}}button.disabled=true;setEditStatus(c,'Validating…');try{{let j=await control({{camera_polygon:1,camera:c,points:JSON.stringify(v.polygon)}}),saved=structuredClone(j.camera_polygons[c]);v.persisted=saved;v.serverPolygon=structuredClone(saved);v.polygon=structuredClone(saved);v.dirty=false;v.editing=false;v.history=[];v.selected=null;setAddMode(c,false);draw(c,v.polygon);setEditStatus(c,'Applied · '+v.polygon.length+' vertices')}}catch(error){{setEditStatus(c,'Apply failed: '+error.message)}}finally{{button.disabled=v.polygon.length<3}}}}
function plateSnapshot(c){{let v=views[c];v.pressureHistory.push(structuredClone(v.pressurePolygon));if(v.pressureHistory.length>50)v.pressureHistory.shift()}}
function plateDirty(c){{views[c].pressureDirty=true;views[c].pressureEditing=true}}
function setPlateAddMode(c,enabled){{if(enabled&&geometryIsLocked())return;let v=views[c];if(enabled&&v.addMode)setAddMode(c,false);v.plateAddMode=enabled;v.pressureEditing=enabled||v.pressureDirty;let b=document.getElementById('plate-add-'+c);b.setAttribute('aria-pressed',String(enabled));b.textContent=enabled?'Exit plate add mode':'Add plate vertex';updatePlateGuidance(c)}}
function togglePlateAddMode(c){{setPlateAddMode(c,!views[c].plateAddMode)}}
function appendPlateVertex(c,point){{if(geometryIsLocked())return;let v=views[c];if(v.pressurePolygon.length>=64)return alert('Maximum 64 vertices');plateSnapshot(c);v.pressurePolygon.push(point);v.pressureSelected=v.pressurePolygon.length-1;plateDirty(c);drawPlate(c,v.pressurePolygon)}}
function removePlateVertex(c){{if(geometryIsLocked())return;let v=views[c];if(v.pressureSelected==null)return alert('Select a pressure-plate vertex first');plateSnapshot(c);v.pressurePolygon.splice(v.pressureSelected,1);v.pressureSelected=null;plateDirty(c);drawPlate(c,v.pressurePolygon)}}
function undoPlate(c){{if(geometryIsLocked())return;let v=views[c],p=v.pressureHistory.pop();if(p){{v.pressurePolygon=p;v.pressureSelected=null;plateDirty(c);drawPlate(c,p)}}}}
function resetPlate(c){{if(geometryIsLocked())return;plateSnapshot(c);views[c].pressureSelected=null;plateDirty(c);drawPlate(c,[])}}
function cancelPlate(c){{let v=views[c];v.pressurePolygon=structuredClone(v.pressurePersisted);v.pressureDirty=false;v.pressureEditing=false;v.pressureHistory=[];v.pressureSelected=null;setPlateAddMode(c,false);drawPlate(c,v.pressurePolygon);updatePlateGuidance(c,'Changes discarded')}}
async function applyPlate(c){{let v=views[c],button=document.getElementById('plate-apply-'+c);if(v.pressurePolygon.length>0&&v.pressurePolygon.length<3){{updatePlateGuidance(c,'Cannot apply');return}}button.disabled=true;updatePlateGuidance(c,'Validating…');try{{let j=await control({{pressure_plate_polygon:1,camera:c,pressure_points:JSON.stringify(v.pressurePolygon)}}),saved=structuredClone(j.pressure_plate_polygons[c]);v.pressurePersisted=saved;v.serverPressurePolygon=structuredClone(saved);v.pressurePolygon=structuredClone(saved);v.pressureDirty=false;v.pressureEditing=false;v.pressureHistory=[];v.pressureSelected=null;setPlateAddMode(c,false);drawPlate(c,v.pressurePolygon);updatePlateGuidance(c,'Applied')}}catch(error){{updatePlateGuidance(c,'Apply failed: '+error.message)}}finally{{button.disabled=v.pressurePolygon.length>0&&v.pressurePolygon.length<3}}}}
function fullscreen(c){{document.getElementById('view-'+c).requestFullscreen()}}
function markControlDirty(c,key){{let v=views[c];v.controlDirty[key]=true;v.controlVersion[key]++;document.getElementById('control-status-'+c).textContent='Unsaved changes'}}
function exposureMode(c){{document.getElementById('exp-'+c).disabled=document.getElementById('exp-mode-'+c).value==='Auto'}}
function cameraControlElements(c){{return{{fps:document.getElementById('fps-'+c),mode:document.getElementById('exp-mode-'+c),exposure:document.getElementById('exp-'+c),gain:document.getElementById('gain-'+c)}}}}
function cameraControlValues(c){{let e=cameraControlElements(c);return{{fps:e.fps.value,exposure_mode:e.mode.value,exposure_us:e.exposure.value,gain:e.gain.value}}}}
function sameNumber(a,b){{return Number(a)===Number(b)}}
function cameraApplyConfirmed(x,wanted){{let controls=x.controls||{{}},u=x.uvc||{{}},live=u.live||{{}};if(!sameNumber(controls.fps,wanted.fps)||controls.exposure_mode!==wanted.exposure_mode||!sameNumber(controls.exposure_us,wanted.exposure_us)||!sameNumber(controls.gain,wanted.gain))return false;if(u.error)throw new Error(u.error);if(!u.available)throw new Error('Live UVC controls unavailable');if(live.exposure_mode!==wanted.exposure_mode)return false;return wanted.exposure_mode!=='Manual'||sameNumber(live.exposure_us,wanted.exposure_us)}}
async function waitForCameraApply(c,wanted,timeoutMs=8000){{let deadline=Date.now()+timeoutMs;while(Date.now()<deadline){{let r=await fetch('/status.json',{{cache:'no-store'}});if(!r.ok)throw new Error('Status check failed ('+r.status+')');let s=await r.json(),x=s.cameras?.[c];if(!x)throw new Error('Camera '+c+' missing from status');if(cameraApplyConfirmed(x,wanted))return x;await new Promise(resolve=>setTimeout(resolve,250))}}throw new Error('Timed out waiting for confirmed live camera state')}}
async function applyCamera(c){{let v=views[c],e=cameraControlElements(c),wanted=cameraControlValues(c),versions={{...v.controlVersion}},button=document.getElementById('apply-'+c),status=document.getElementById('control-status-'+c);for(let input of Object.values(e))if(!input.checkValidity()){{input.reportValidity();return}}v.applying=true;button.disabled=true;status.textContent='Applying…';try{{let j=await control({{camera:c,...wanted}});if(j.error)throw new Error(j.error);await waitForCameraApply(c,wanted);for(let key of Object.keys(v.controlDirty))if(v.controlVersion[key]===versions[key])v.controlDirty[key]=false;status.textContent=Object.values(v.controlDirty).some(Boolean)?'Applied; newer edits unsaved':'Applied and confirmed'}}catch(error){{status.textContent='Apply failed: '+error.message}}finally{{v.applying=false;button.disabled=false}}}}
function saveGlobals(){{control({{trial_name:trialName.value,record_root:root.value,starseg_display:starsegEnabled.checked?1:0,starseg_model:starsegModel.value,backup_enabled:backupEnabled.checked?1:0,backup_root:backupRoot.value,backup_throttle_mbps:backupThrottle.value}})}}
async function savePressure(){{await control({{pressure_enabled:pressureEnabled.checked?1:0,hue_bridge_address:hueBridge.value,hue_app_key:hueKey.value}})}}
async function hueAction(path){{let r=await fetch(path,{{method:'POST'}}),j=await r.json();if(!r.ok){{alert(j.error||'Hue request failed');throw new Error(j.error)}}return j}}
async function pairHue(){{await savePressure();await hueAction('/hue/pair');hueKey.value='';await discoverHue()}}
async function discoverHue(){{await savePressure();await hueAction('/hue/discover');await poll()}}
async function assignHue(c){{await control({{camera:c,hue_output:document.getElementById('hue-'+c).value}})}}
let hourlyActionPending=false,hourlyArmed=false;
async function toggleHourlyPumps(){{if(hourlyActionPending)return;hourlyActionPending=true;hourlyPumpArm.disabled=true;hourlyPumpStatus.textContent=hourlyArmed?'Disarming hourly A+B+C+D pumps…':'Arming hourly A+B+C+D pumps…';try{{let j=await control({{hourly_pumps_armed:hourlyArmed?0:1}});updateHourlySchedule(j)}}catch(error){{hourlyPumpStatus.textContent='Hourly pump action failed: '+error.message}}finally{{hourlyActionPending=false;hourlyPumpArm.disabled=false}}}}
function updateHourlySchedule(s){{let h=s.pressure_plate.hourly_schedule;hourlyArmed=Boolean(h.armed);hourlyPumpArm.textContent=hourlyArmed?'Disarm hourly A+B+C+D Hue pumps':'Arm hourly A+B+C+D Hue pumps';hourlyPumpArm.classList.toggle('stop',hourlyArmed);let next=h.next_opportunity_local?new Date(h.next_opportunity_at*1000).toLocaleString():'none';hourlyPumpStatus.textContent=hourlyArmed?'Armed · next '+next+' · 60 s Hue-only pulses':'Hourly A+B+C+D Hue pumps disarmed';hourlyPumpArm.disabled=hourlyActionPending||(!s.recording&&!hourlyArmed)}}
function overlay(c,row){{let v=views[c],s=document.getElementById('svg-'+c),g=previewGeometry(c),layer=ensureSvgGroups(s).overlay;v.overlayRow=row;prepareSvg(s,g);layer.replaceChildren();if(!row)return;let w=row.width||1,h=row.height||1,counts={{}};for(let d of row.detections||[]){{counts[d.class_name]=(counts[d.class_name]||0)+1;if(d.segment_xy?.length){{let p=document.createElementNS(svgNS,'polygon');p.setAttribute('points',d.segment_xy.map(x=>renderPoint(g,[x[0]/w,x[1]/h]).join(',')).join(' '));p.style.fill='#46d5ce44';layer.appendChild(p)}}if(d.box_xyxy){{let b=d.box_xyxy,a=renderPoint(g,[b[0]/w,b[1]/h]),z=renderPoint(g,[b[2]/w,b[3]/h]),r=document.createElementNS(svgNS,'rect');r.setAttribute('x',String(a[0]));r.setAttribute('y',String(a[1]));r.setAttribute('width',String(z[0]-a[0]));r.setAttribute('height',String(z[1]-a[1]));r.style.cssText='fill:none;stroke:#ff6272;stroke-width:2';layer.appendChild(r)}}}}if(Object.keys(counts).length)document.getElementById('info-'+c).textContent+='\\nStarSeg '+JSON.stringify(counts)}}
function refreshPristineControl(v,key,element,value){{if(!v.controlDirty[key]&&!v.applying&&document.activeElement!==element)element.value=value}}
async function poll(){{try{{let [s,l]=await Promise.all([fetch('/status.json').then(r=>r.json()),fetch('/starseg/latest.json').then(r=>r.json())]);if(document.activeElement!==trialName)trialName.value=s.active_trial_name||s.trial_name;let geometryLocked=s.recording||s.stop_requested;trialName.disabled=geometryLocked;trialLock.textContent=geometryLocked?'Trial and geometry locked for active recording':'';root.value=s.record_root;starsegEnabled.checked=s.starseg.enabled;starsegModel.value=s.starseg.desired_model||'';backupEnabled.checked=s.backup.enabled;backupRoot.value=s.backup.root||'';backupThrottle.value=s.backup.throttle_mbps;let p=s.pressure_plate,a=p.arduino;if(!hourlyActionPending)updateHourlySchedule(s);pressureEnabled.checked=p.enabled;hueBridge.value=p.bridge.address;if(!p.bridge.configured&&document.activeElement!==hueKey)hueKey.value='';hueStatus.textContent='Hue '+p.bridge.status+(p.bridge.error?' · '+p.bridge.error:'');let owner=a.mode==='injector'?'Injector':'Direct Arduino';arduinoStatus.textContent=owner+' '+(a.connected?(a.ready?'ready':'not ready'):'disconnected')+' · arbitration '+(a.arbitration_source||'unknown')+' · confirmed mask '+(a.confirmed_mask==null?'unknown':a.confirmed_mask)+(a.emergency_latched?' · EMERGENCY LATCHED':'')+(a.last_error?' · '+a.last_error:'');arduinoStatus.className=p.split_actuator_warning||a.emergency_latched?'warning':'';let sg=s.starseg,archiveWarning=sg.recorded_queue_depth>0&&sg.archival_oldest_age_ms>3000,controlWarning=s.recording&&(!sg.control_available||sg.last_control_frame_age_ms>3000||sg.control_last_error);global.textContent=(s.recording?'● RECORDING':'Idle')+' · Control q'+sg.control_queue_depth+' age '+Math.round(sg.last_control_frame_age_ms)+'ms'+(sg.control_last_error?' ERROR '+sg.control_last_error:'')+' · Archival q'+sg.recorded_queue_depth+' oldest '+Math.round(sg.archival_oldest_age_ms)+'ms'+(archiveWarning?' BACKLOG GROWING':'')+' · Backup '+(s.backup.enabled?'enabled':'disabled');global.className=controlWarning||archiveWarning?'warning':'';for(let c of cameras){{let x=s.cameras[c],v=views[c],u=x.uvc||{{}},e=cameraControlElements(c),server=structuredClone(s.camera_polygons[c]),serverPlate=structuredClone(s.pressure_plate_polygons[c]),plate=p.cameras[c],select=document.getElementById('hue-'+c);for(let id of ['add-','polygon-apply-','plate-add-','plate-apply-'])document.getElementById(id+c).disabled=geometryLocked;document.getElementById('state-'+c).textContent=x.connected?'online':'offline';let secondary=plate.secondary_pulse_active?' · Hue-only hourly pulse active':plate.last_opportunity_result==='skipped'?' · Hourly skipped: '+plate.last_skip_reason:'';let arduinoState=plate.arduino_confirmed==null?'unknown':plate.arduino_confirmed?'on':'off';let plateStatus=document.getElementById('plate-'+c);plateStatus.textContent=(plate.armed?'Armed':'Disarmed')+' · '+(plate.plate_valid?'Plate valid':'No plate')+' · '+(plate.occupied?'Occupied':'Empty')+' · Primary target '+(plate.logical_primary_demand?'on':'off')+' · Hue '+(plate.hue_reported?'on':'off')+' (target '+(plate.hue_desired?'on':'off')+') · Arduino '+arduinoState+' (target '+(plate.arduino_desired?'on':'off')+')'+secondary+(plate.split_warning?' · ACTUATOR TARGET WARNING':'')+(plate.error?' · Error':'');plateStatus.className='pill'+(plate.split_warning?' warning':'');let selected=plate.output_id,known=new Set([...select.options].map(o=>o.value));for(let output of p.outputs)if(!known.has(output.id))select.add(new Option(output.name+' ('+output.kind+')',output.id));select.value=selected;document.getElementById('info-'+c).textContent=JSON.stringify(x,null,2);document.getElementById('img-'+c).src='/preview/'+c+'.jpg?t='+Date.now();if(x.resolution?.length===2){{v.imageWidth=Number(x.resolution[0])||v.imageWidth;v.imageHeight=Number(x.resolution[1])||v.imageHeight}}v.serverPolygon=server;if(!v.editing&&!v.dirty&&!v.vertexDrag){{v.persisted=structuredClone(server);draw(c,structuredClone(server))}}v.serverPressurePolygon=serverPlate;if(!v.pressureEditing&&!v.pressureDirty&&!v.pressureVertexDrag){{v.pressurePersisted=structuredClone(serverPlate);drawPlate(c,structuredClone(serverPlate))}}refreshPristineControl(v,'fps',e.fps,x.controls.fps);refreshPristineControl(v,'mode',e.mode,x.controls.exposure_mode||'Auto');refreshPristineControl(v,'exposure',e.exposure,x.controls.exposure_us);refreshPristineControl(v,'gain',e.gain,x.controls.gain);exposureMode(c);let range=u.ranges?.exposure_us||{{}};if(range.min!=null)e.exposure.min=range.min;if(range.max!=null)e.exposure.max=range.max;if(range.step!=null)e.exposure.step=range.step;overlay(c,l[c])}}}}catch(e){{global.textContent='API unavailable'}}}}
for(let c of cameras){{views[c]={{z:1,x:0,y:0,imageWidth:16,imageHeight:9,polygon:structuredClone(polygonDefaults),persisted:structuredClone(polygonDefaults),serverPolygon:structuredClone(polygonDefaults),dirty:false,editing:false,addMode:false,vertexDrag:null,pressurePolygon:[],pressurePersisted:[],serverPressurePolygon:[],pressureDirty:false,pressureEditing:false,plateAddMode:false,pressureVertexDrag:null,pressureHistory:[],pressureSelected:null,overlayRow:null,controlDirty:{{fps:false,mode:false,exposure:false,gain:false}},controlVersion:{{fps:0,mode:0,exposure:0,gain:0}},applying:false,history:[],selected:null}};let area=document.getElementById('view-'+c),s=document.getElementById('svg-'+c),pan=null;draw(c,views[c].polygon);drawPlate(c,[]);if(window.ResizeObserver)new ResizeObserver(()=>{{draw(c,views[c].polygon);drawPlate(c,views[c].pressurePolygon);overlay(c,views[c].overlayRow)}}).observe(area);s.onpointerdown=e=>{{let pressureIndex=e.target.dataset.pressureVertex,index=e.target.dataset.vertex,v=views[c];if(pressureIndex!=null){{e.preventDefault();e.stopPropagation();plateSnapshot(c);v.pressureSelected=Number(pressureIndex);v.pressureEditing=true;v.pressureVertexDrag={{index:Number(pressureIndex),pointerId:e.pointerId}};s.setPointerCapture(e.pointerId);drawPlate(c,v.pressurePolygon)}}else if(index!=null){{e.preventDefault();e.stopPropagation();snapshot(c);v.selected=Number(index);v.editing=true;v.vertexDrag={{index:Number(index),pointerId:e.pointerId}};s.setPointerCapture(e.pointerId);draw(c,v.polygon)}}}};s.onpointermove=e=>{{let v=views[c],pressureDrag=v.pressureVertexDrag,drag=v.vertexDrag;if(pressureDrag?.pointerId===e.pointerId){{e.preventDefault();v.pressurePolygon[pressureDrag.index]=svgPoint(c,e);plateDirty(c);drawPlate(c,v.pressurePolygon)}}else if(drag?.pointerId===e.pointerId){{e.preventDefault();v.polygon[drag.index]=svgPoint(c,e);dirty(c);draw(c,v.polygon)}}}};s.onpointerup=s.onpointercancel=e=>{{let v=views[c];if(v.pressureVertexDrag?.pointerId===e.pointerId)v.pressureVertexDrag=null;if(v.vertexDrag?.pointerId===e.pointerId)v.vertexDrag=null;if(s.hasPointerCapture(e.pointerId))s.releasePointerCapture(e.pointerId)}};area.onwheel=e=>{{e.preventDefault();zoom(c,e.deltaY<0?1.1:.9)}};area.onpointerdown=e=>{{let v=views[c];if(e.target.dataset.vertex!=null||e.target.dataset.pressureVertex!=null)return;if(v.plateAddMode){{e.preventDefault();appendPlateVertex(c,svgPoint(c,e));return}}if(v.addMode){{e.preventDefault();appendVertex(c,svgPoint(c,e));return}}pan=[e.pointerId,e.clientX,e.clientY,v.x,v.y];area.setPointerCapture(e.pointerId)}};area.onpointermove=e=>{{if(pan&&pan[0]===e.pointerId){{views[c].x=pan[3]+e.clientX-pan[1];views[c].y=pan[4]+e.clientY-pan[2];transform(c)}}}};area.onpointerup=area.onpointercancel=e=>{{if(pan?.[0]===e.pointerId)pan=null}}}}
document.addEventListener('pointerdown',e=>{{if(geometryIsLocked()&&(e.target.dataset?.vertex!=null||e.target.dataset?.pressureVertex!=null))e.stopImmediatePropagation()}},true);
document.addEventListener('pointermove',e=>{{if(geometryIsLocked()&&(e.target.dataset?.vertex!=null||e.target.dataset?.pressureVertex!=null))e.stopImmediatePropagation()}},true);
poll();setInterval(()=>{{if(!document.hidden)poll()}},2000);trialName.onchange=saveGlobals;pressureEnabled.onchange=savePressure;hueBridge.onchange=savePressure;hueKey.onchange=savePressure;
</script></body></html>""".encode()


class Handler(BaseHTTPRequestHandler):
    recorder: Recorder
    server_version = "StartrackedLoopRecorder/1.0"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        self._send(status, "application/json", json.dumps(data).encode())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            return self._send(200, "text/html; charset=utf-8", page())
        if parsed.path == "/status.json":
            return self._json(200, self.recorder.status())
        if parsed.path == "/starseg/latest.json":
            return self._json(200, self.recorder.starseg.latest)
        if parsed.path.startswith("/popout/") and parsed.path[-1:] in CAMERAS:
            return self._send(200, "text/html; charset=utf-8", page(parsed.path[-1]))
        if parsed.path.startswith("/preview/") and parsed.path[-5:-4] in CAMERAS and parsed.path.endswith(".jpg"):
            camera = parsed.path[-5:-4]
            path = Path(self.recorder.config["record_root"]) / "preview" / f"{camera}.jpg"
            if path.exists():
                return self._send(200, "image/jpeg", path.read_bytes())
            return self._send(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain", b"preview unavailable")
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/hue/pair":
            try:
                return self._json(200, self.recorder.pair_hue())
            except (RuntimeError, queue.Empty) as exc:
                return self._json(503, {"error": str(exc)})
        if path == "/hue/discover":
            try:
                return self._json(200, self.recorder.discover_hue())
            except (RuntimeError, queue.Empty) as exc:
                return self._json(503, {"error": str(exc)})
        if path != "/control":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64 * 1024:
                raise ConfigError("form too large")
            if "application/x-www-form-urlencoded" not in self.headers.get("Content-Type", ""):
                raise ConfigError("URL-encoded form required")
            fields = parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
            self._json(200, self.recorder.apply_control(fields))
        except (ConfigError, ValueError, OSError) as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    recorder = Recorder()
    Handler.recorder = recorder
    server = ThreadingHTTPServer(
        (recorder.deployment["bind"], recorder.deployment["port"]), Handler
    )
    stopping = threading.Event()

    def stop_handler(_signum=None, _frame=None):
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    recorder.start_workers()
    print(
        f"Startracked Loop Recorder: http://{recorder.deployment['bind']}:{recorder.deployment['port']}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        recorder.close()


if __name__ == "__main__":
    main()
