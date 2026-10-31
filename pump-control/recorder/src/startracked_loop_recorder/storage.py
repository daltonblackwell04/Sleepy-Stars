from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import atomic_json_write

HOURLY_PUMP_SCHEDULE_POLICY = {
    "version": 4,
    "cameras": ["A", "B", "C", "D"],
    "opportunity": "next_local_clock_top_of_hour_then_each_local_hour_while_armed",
    "pulse_seconds": 60.0,
    "occupancy": "hysteresis_confirmed_sunflower_star",
    "fail_closed": True,
    "occupied_or_unknown_action": "skip_without_retry_until_next_hour",
    "hue_arbitration": "primary_occupancy_or_active_secondary_pulse",
    "arduino_arbitration": "primary_occupancy_only",
    "pulse_actuator": "hue_only",
    "arm_state": "runtime_only",
}
DUAL_ACTUATOR_POLICY = {
    "version": 3,
    "logical_owner": "PressurePlateService",
    "logical_primary_demand": (
        "hysteresis_confirmed_sunflower_star_occupancy_while_recording_armed_"
        "pressure_enabled_plate_valid_and_actuator_assigned"
    ),
    "hue_arbitration": "logical_primary_demand_or_active_secondary_hourly_pulse",
    "arduino_arbitration": "logical_primary_demand_only",
    "secondary_hourly_pulse_actuator": "hue_only",
    "failure_policy": (
        "independent_sinks_warn_on_mismatch_with_their_own_explicit_target"
    ),
    "hue_outputs": {
        "A": "aaaaaaaa-0000-4000-8000-00000000000a",
        "B": "bbbbbbbb-0000-4000-8000-00000000000b",
        "C": "cccccccc-0000-4000-8000-00000000000c",
        "D": "dddddddd-0000-4000-8000-00000000000d",
    },
    "hue_output_names": {
        "A": "Camera A Pump",
        "B": "Camera B Pump",
        "C": "Camera C Pump",
        "D": "Camera D Pump",
    },
    "arduino": {
        "serial_path": (
            "/dev/serial/by-id/"
            "usb-Arduino__www.arduino.cc__0042_00000000000000000000-if00"
        ),
        "baud": 115200,
        "protocol": "PUMPS:<mask>\\n",
        "heartbeat_seconds": 0.15,
        "firmware_watchdog_seconds": 0.5,
        "mapping": {"A": 2, "B": 1, "C": 8, "D": 4},
        "numbering": "firmware_and_LCD",
        "confirmation": "EVENT STATE acknowledgment; zero handshake accepts EVENT REARMED",
    },
}


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def timestamp(ts: float, milliseconds: bool = True) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y%m%dT%H%M%S") + (f"{dt.microsecond // 1000:03d}" if milliseconds else "")


def safe_trial_name(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in value.strip())
    return cleaned[:120] or "recording"


safe_heading = safe_trial_name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Segment:
    camera: str
    serial: str
    model: str
    trial_name: str
    session: str
    session_started: float
    start_ts: float
    record_root: Path
    fps: float
    requested_seconds: float
    width: int
    height: int
    polygon: list[list[float]]
    staging: Path
    pressure_polygon: list[list[float]] = field(default_factory=list)
    frames: int = 0
    frame_timestamps: list[float] = field(default_factory=list)

    @classmethod
    def create(cls, **kwargs: Any) -> "Segment":
        root = Path(kwargs["record_root"])
        camera = kwargs["camera"]
        start_ts = kwargs["start_ts"]
        staging = root / "_active_segments" / camera / str(int(start_ts * 1000))
        staging.mkdir(parents=True, exist_ok=False)
        segment = cls(staging=staging, **kwargs)
        atomic_json_write(staging / "segment_state.json", segment.state())
        return segment

    def state(self) -> dict[str, Any]:
        return {
            "schema_version": 3, "camera": self.camera, "serial": self.serial,
            "model": self.model, "trial_name": self.trial_name, "session": self.session,
            "session_started": self.session_started, "start_ts": self.start_ts,
            "record_root": str(self.record_root), "fps": self.fps,
            "requested_seconds": self.requested_seconds, "width": self.width,
            "height": self.height, "polygon": self.polygon,
            "pressure_polygon": self.pressure_polygon, "frames": self.frames,
            "frame_timestamps": self.frame_timestamps,
        }

    def add_frame(self, jpeg: bytes, frame_ts: float) -> Path:
        destination = self.staging / f"{self.frames:08d}.jpg"
        fd, temp = tempfile.mkstemp(prefix=".frame-", dir=self.staging)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(jpeg)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, destination)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        self.frame_timestamps.append(frame_ts)
        self.frames += 1
        atomic_json_write(self.staging / "segment_state.json", self.state())
        return destination

    def final_paths(self, end_ts: float) -> tuple[str, Path, Path]:
        duration = max(0.0, end_ts - self.start_ts)
        stem = (
            f"{safe_trial_name(self.trial_name)}_camera-{self.camera}_start-{timestamp(self.start_ts)}"
            f"_duration-{duration:.3f}s_fps-{self.fps:g}_frames-{self.frames}"
            f"_size-{self.width}x{self.height}"
        )
        folder = self.record_root / self.camera / self.session / stem
        return stem, folder, folder / f"{stem}.mp4"


def derive_starseg_summary(jsonl: Path, expected_frames: int | None = None) -> dict[str, Any]:
    classes: dict[str, int] = {}
    detections = 0
    frames = 0
    models: list[str] = []
    model_provenance = None
    conf = None
    imgsz = None
    if jsonl.exists():
        with jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                frames += 1
                model = row.get("model")
                if model and model not in models:
                    models.append(model)
                model_provenance = row.get("model_provenance", model_provenance)
                conf = row.get("confidence_threshold", conf)
                imgsz = row.get("imgsz", imgsz)
                for detection in row.get("detections", []):
                    detections += 1
                    name = detection.get("class_name", "unknown")
                    classes[name] = classes.get(name, 0) + 1
    return {
        "frames": frames, "detections": detections, "classes": classes,
        "model": models[-1] if models else None, "models": models,
        "model_provenance": model_provenance,
        "confidence_threshold": conf, "imgsz": imgsz, "frames_jsonl": str(jsonl),
    }


def finalize_segment(
    segment: Segment, end_ts: float, crf: int, preset: str,
    delete_failed: bool = False, ffmpeg: str = "ffmpeg",
) -> list[Path]:
    if segment.frames < 1:
        raise RuntimeError("cannot finalize an empty segment")
    stem, folder, video = segment.final_paths(end_ts)
    folder.mkdir(parents=True, exist_ok=True)
    temp_video = folder / f".{stem}.part.mp4"
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(segment.fps), "-i", str(segment.staging / "%08d.jpg"),
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_video),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not temp_video.exists() or temp_video.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced no video")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height", "-of", "json", str(temp_video)],
            check=True, capture_output=True, text=True,
        )
        if not json.loads(probe.stdout).get("streams"):
            raise RuntimeError("final MP4 has no video stream")
        os.replace(temp_video, video)
        staged_jsonl = segment.staging / "starseg_frames.jsonl"
        final_jsonl = folder / f"{stem}_starseg_frames.jsonl"
        if staged_jsonl.exists():
            shutil.copy2(staged_jsonl, final_jsonl)
        else:
            final_jsonl.write_text("", encoding="utf-8")
        summary_path = folder / f"{stem}_starseg_summary.json"
        summary = derive_starseg_summary(final_jsonl, segment.frames)
        inference_failure = segment.staging / "starseg_inference_failure.json"
        if inference_failure.exists():
            failure = json.loads(inference_failure.read_text(encoding="utf-8"))
            summary.update({
                "inference_status": "failed", "inference_error": failure["inference_error"],
            })
        atomic_json_write(summary_path, summary)
        staged_pressure = segment.staging / "pressure_plate_events.jsonl"
        pressure_path = folder / f"{stem}_pressure_plate_events.jsonl"
        if staged_pressure.exists():
            shutil.copy2(staged_pressure, pressure_path)
        else:
            pressure_path.write_text("", encoding="utf-8")
        absolute = [[round(x * segment.width), round(y * segment.height)] for x, y in segment.polygon]
        pressure_absolute = [
            [round(x * segment.width), round(y * segment.height)]
            for x, y in segment.pressure_polygon
        ]
        metadata_path = folder / f"{stem}.json"
        metadata = {
            "schema_version": 3, "trial_name": segment.trial_name, "camera": segment.camera,
            "serial": segment.serial, "model": segment.model,
            "recording_session": segment.session,
            "recording_started_iso": iso(segment.session_started),
            "start_ts": segment.start_ts, "started_iso": iso(segment.start_ts),
            "end_ts": end_ts, "ended_iso": iso(end_ts),
            "requested_segment_seconds": segment.requested_seconds,
            "duration_seconds": max(0, end_ts - segment.start_ts), "fps": segment.fps,
            "width": segment.width, "height": segment.height, "frames": segment.frames,
            "record_root": str(segment.record_root), "file": str(video),
            "folder": str(folder),
            "starseg_roi": {
                "normalized": segment.polygon, "absolute": absolute,
                "filter_policy": "segmentation_centroid_or_box_center_inside_polygon",
            },
            "pressure_plate": {
                "normalized": segment.pressure_polygon,
                "absolute": pressure_absolute,
                "occupancy_policy": (
                    "sunflower_star_roi_test_point_inside_pressure_plate_boundary_inclusive"
                ),
                "valid": len(segment.pressure_polygon) >= 3,
            },
            "secondary_hourly_pump_schedule": HOURLY_PUMP_SCHEDULE_POLICY,
            "dual_actuator_pump_policy": DUAL_ACTUATOR_POLICY,
            "frame_timestamps": segment.frame_timestamps,
            "starseg_frames_jsonl": str(final_jsonl),
            "starseg_summary_json": str(summary_path), "starseg_summary": summary,
            "pressure_plate_events_jsonl": str(pressure_path),
        }
        atomic_json_write(metadata_path, metadata)
        shutil.rmtree(segment.staging)
        return [video, metadata_path, final_jsonl, summary_path]
    except Exception:
        temp_video.unlink(missing_ok=True)
        if delete_failed:
            shutil.rmtree(segment.staging, ignore_errors=True)
        raise


def recover_staging(root: Path, finalize_callback=None) -> dict[str, int]:
    active = root / "_active_segments"
    report = {"recovered": 0, "quarantined": 0}
    if not active.exists():
        return report
    quarantine = root / "_quarantined_segments"
    for state_path in active.glob("*/*/segment_state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            jpgs = sorted(state_path.parent.glob("[0-9]" * 8 + ".jpg"))
            if not jpgs:
                raise ValueError("no complete frames")
            if finalize_callback:
                finalize_callback(state_path.parent, state)
                report["recovered"] += 1
            else:
                raise ValueError("automatic finalizer unavailable")
        except Exception:
            destination = quarantine / state_path.parent.parent.name / state_path.parent.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.move(str(state_path.parent), destination)
            report["quarantined"] += 1
    return report


class BackupWorker(threading.Thread):
    def __init__(self, primary: Path, root: Path, throttle_mbps: float = 8):
        super().__init__(name="backup-worker", daemon=True)
        self.primary, self.root = primary.resolve(), root
        self.throttle = max(0.0, throttle_mbps) * 1024 * 1024
        self.jobs: queue.Queue[Path | None] = queue.Queue(maxsize=2000)
        self.stop_event = threading.Event()
        self.metrics = {"copied": 0, "errors": 0, "high_water": 0, "last_error": ""}

    def validate(self) -> None:
        resolved = self.root.resolve()
        if not str(resolved).startswith(("/media/", "/mnt/")):
            raise ValueError("backup root must be under /media or /mnt")
        if not os.path.ismount(resolved):
            raise ValueError("backup root is not a mounted filesystem")

    def enqueue(self, path: Path) -> None:
        relative = path.resolve().relative_to(self.primary)
        if relative.parts[0] in {"preview", "_active_segments"}:
            raise ValueError("transient files cannot be backed up")
        self.jobs.put(path)
        self.metrics["high_water"] = max(self.metrics["high_water"], self.jobs.qsize())

    def copy(self, source: Path) -> Path:
        relative = source.resolve().relative_to(self.primary)
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_sha256(source) == file_sha256(destination):
            return destination
        temp = destination.with_name(f".{destination.name}.{os.getpid()}.part")
        started = time.monotonic()
        written = 0
        with source.open("rb") as src, temp.open("wb") as dst:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(block)
                written += len(block)
                if self.throttle:
                    wait = written / self.throttle - (time.monotonic() - started)
                    if wait > 0:
                        time.sleep(wait)
            dst.flush()
            os.fsync(dst.fileno())
        shutil.copystat(source, temp)
        os.replace(temp, destination)
        return destination

    def run(self) -> None:
        while not self.stop_event.is_set() or not self.jobs.empty():
            try:
                job = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                self.jobs.task_done()
                break
            try:
                self.validate()
                self.copy(job)
                self.metrics["copied"] += 1
            except Exception as exc:
                self.metrics["errors"] += 1
                self.metrics["last_error"] = str(exc)
            finally:
                self.jobs.task_done()

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
