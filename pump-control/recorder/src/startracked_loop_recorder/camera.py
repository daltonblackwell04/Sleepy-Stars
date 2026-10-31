from __future__ import annotations

import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2


@dataclass
class Frame:
    camera: str
    timestamp: float
    image: Any
    capture_monotonic: float = 0.0


class UVCCamera(threading.Thread):
    """Reconnectable capture with a bounded latest-frame encoder handoff."""

    def __init__(
        self, letter: str, path: str, width: int, height: int,
        output: queue.Queue[Frame], controls: Callable[[], dict[str, Any]],
        shutdown: threading.Event,
        on_capture: Callable[[Frame], None] | None = None,
    ):
        super().__init__(name=f"capture-{letter}", daemon=True)
        self.letter, self.path = letter, path
        self.width, self.height = width, height
        self.output, self.controls, self.shutdown = output, controls, shutdown
        self.on_capture = on_capture
        self.status: dict[str, Any] = {
            "connected": False, "frames": 0, "errors": 0, "last_error": "",
            "encoder_queue_drops": 0,
            "last_frame_ts": None, "path": path, "resolution": [width, height],
            "uvc": {
                "available": False, "ranges": {}, "requested": {},
                "applied": {}, "live": {}, "error": "",
            },
        }
        self._capture = None

    def _enqueue_latest(self, frame: Frame) -> None:
        """Keep capture/control live if an encoder is blocked by archival work."""
        try:
            self.output.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self.output.get_nowait()
            self.output.task_done()
            self.status["encoder_queue_drops"] += 1
        except queue.Empty:
            pass
        try:
            self.output.put_nowait(frame)
        except queue.Full:
            # A consumer/producer race can refill the bounded handoff. Dropping
            # this unrecorded frame is preferable to stalling live control.
            self.status["encoder_queue_drops"] += 1

    def _open(self):
        capture = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"could not open {self.path}")
        self._capture = capture
        self.status["connected"] = True

    @staticmethod
    def _auto_value(value: str) -> int:
        return {"Manual": 1, "Auto": 3, "Off": 1, "Once": 1, "Continuous": 3}.get(value, 3)

    @staticmethod
    def _parse_controls(text: str) -> dict[str, dict[str, int]]:
        parsed: dict[str, dict[str, int]] = {}
        for line in text.splitlines():
            match = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\([^)]*\)\s*:\s*(.*)", line, re.I)
            if not match:
                continue
            values = {
                key: int(value)
                for key, value in re.findall(r"\b(min|max|step|default|value)=(-?\d+)", match.group(2))
            }
            if values:
                parsed[match.group(1)] = values
        return parsed

    def probe_controls(self) -> dict[str, Any]:
        if shutil.which("v4l2-ctl") is None:
            return {
                "available": False, "ranges": {}, "requested": {}, "applied": {},
                "live": {}, "error": "v4l2-ctl unavailable; install v4l-utils",
            }
        result = subprocess.run(
            ["v4l2-ctl", "-d", self.path, "--list-ctrls-menus"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            return {
                "available": False, "ranges": {}, "requested": {}, "applied": {},
                "live": {}, "error": result.stderr.strip() or "UVC control probe failed",
            }
        controls = self._parse_controls(result.stdout)
        exposure = controls.get("exposure_time_absolute", {})
        auto_control = next(
            (name for name in ("auto_exposure", "exposure_auto") if name in controls),
            "exposure_auto",
        )
        ranges = {}
        if exposure:
            ranges["exposure_us"] = {
                key: value * 100 for key, value in exposure.items()
                if key in {"min", "max", "step", "default", "value"}
            }
        return {
            "available": True, "ranges": ranges, "requested": {}, "applied": {},
            "live": {}, "error": "", "exposure_auto_control": auto_control,
        }

    def _set_control(self, control: str, value: int) -> None:
        result = subprocess.run(
            ["v4l2-ctl", "-d", self.path, "--set-ctrl", f"{control}={value}"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"{control} rejected: {result.stderr.strip() or result.stdout.strip() or 'device error'}"
            )

    def _read_live_controls(self, auto_control: str) -> dict[str, Any]:
        result = subprocess.run(
            ["v4l2-ctl", "-d", self.path, "--get-ctrl",
             f"{auto_control},exposure_time_absolute"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "failed to read live exposure controls")
        values = dict(re.findall(r"(\w+):\s*(-?\d+)", result.stdout))
        auto = int(values.get(auto_control, 3))
        absolute = int(values.get("exposure_time_absolute", 0))
        return {
            "exposure_mode": "Manual" if auto == 1 else "Auto",
            "exposure_us": absolute * 100,
        }

    def apply_controls(self) -> None:
        if not self.status["uvc"].get("available"):
            self.status["uvc"] = self.probe_controls()
        if not self.status["uvc"].get("available"):
            return
        values = self.controls()
        mode = str(values.get("exposure_mode", values.get("exposure_auto", "Auto")))
        requested = {
            "exposure_mode": "Manual" if mode in {"Manual", "Off"} else "Auto",
            "exposure_us": int(float(values.get("exposure_us", 10000))),
        }
        self.status["uvc"]["requested"] = requested
        auto_control = self.status["uvc"].get("exposure_auto_control", "exposure_auto")
        try:
            self._set_control(auto_control, self._auto_value(requested["exposure_mode"]))
            applied = {"exposure_mode": requested["exposure_mode"]}
            if requested["exposure_mode"] == "Manual":
                self._set_control(
                    "exposure_time_absolute", max(1, round(requested["exposure_us"] / 100))
                )
                applied["exposure_us"] = requested["exposure_us"]
            self.status["uvc"]["applied"] = applied
            self.status["uvc"]["live"] = self._read_live_controls(auto_control)
            self.status["uvc"]["error"] = ""
        except RuntimeError as exc:
            self.status["uvc"]["error"] = str(exc)

    def run(self) -> None:
        last_controls = 0.0
        while not self.shutdown.is_set():
            try:
                if self._capture is None:
                    self._open()
                now = time.monotonic()
                if now - last_controls >= 1:
                    self.apply_controls()
                    last_controls = now
                ok, image = self._capture.read()
                if not ok:
                    raise RuntimeError("camera read failed")
                captured = time.monotonic()
                ts = time.time()
                frame = Frame(self.letter, ts, image, captured)
                if self.on_capture is not None:
                    self.on_capture(frame)
                self._enqueue_latest(frame)
                self.status.update(
                    connected=True, frames=self.status["frames"] + 1,
                    last_frame_ts=ts, resolution=[image.shape[1], image.shape[0]],
                )
            except Exception as exc:
                self.status.update(
                    connected=False, errors=self.status["errors"] + 1, last_error=str(exc)
                )
                if self._capture is not None:
                    self._capture.release()
                    self._capture = None
                self.shutdown.wait(2)
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def probe_uvc(path: str) -> dict[str, Any]:
    device = Path(path)
    result = {"path": path, "exists": device.exists(), "formats": "", "controls": ""}
    if not device.exists():
        return result
    if shutil.which("v4l2-ctl") is None:
        result["usable"] = False
        result["error"] = "v4l2-ctl is not installed"
        return result
    formats = subprocess.run(
        ["v4l2-ctl", "-d", path, "--list-formats-ext"],
        capture_output=True, text=True, check=False,
    )
    controls = subprocess.run(
        ["v4l2-ctl", "-d", path, "--list-ctrls-menus"],
        capture_output=True, text=True, check=False,
    )
    result["formats"] = formats.stdout
    result["controls"] = controls.stdout
    result["control_ranges"] = UVCCamera._parse_controls(controls.stdout)
    result["usable"] = formats.returncode == 0
    return result
