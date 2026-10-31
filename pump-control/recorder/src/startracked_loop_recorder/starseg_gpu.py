"""StarSeg GPU inference in a replaceable child process.

Why: on 2026-10-04 two GPU hangs (NVRM Xid 8) left the recorder's own CUDA context
permanently broken. CUDA cannot recover a context in the same process, so every later
inference failed ("launch timed out") for six days until the whole recorder was restarted.

Here the model lives in a spawned child. The recorder process never initialises CUDA.
After a fatal CUDA error, a hang (no reply within `timeout`), or a child crash, the child is
killed and a fresh one is started on the next call (after `backoff` seconds), so recovery
takes seconds instead of needing a recording stop.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from typing import Any

FATAL_MARKERS = (
    "cuda error", "cuda out of memory", "cublas", "cudnn", "cufft", "nvrtc",
    "device-side assert", "illegal memory access", "launch timed out", "unspecified launch failure",
    "acceleratorerror",
)


def is_fatal_gpu_error(message: str) -> bool:
    m = (message or "").lower()
    return any(marker in m for marker in FATAL_MARKERS)


class YoloBackend:
    """Real backend: Ultralytics YOLO on CUDA when available (inside the child only)."""

    def __init__(self, model_path: str):
        import torch
        from ultralytics import YOLO
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.names = {str(k): str(v) for k, v in self.model.names.items()}
        self.task = str(self.model.task)

    def predict(self, images, conf, imgsz):
        results = self.model.predict(list(images), conf=conf, imgsz=imgsz, device=self.device, verbose=False)
        out = []
        for result in results:
            dets = []
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                poly = []
                if result.masks is not None and i < len(result.masks.xy):
                    poly = [[float(x), float(y)] for x, y in result.masks.xy[i].tolist()]
                dets.append({
                    "class_id": cls_id, "class_name": str(result.names[cls_id]),
                    "confidence": float(result.boxes.conf[i].item()),
                    "box_xyxy": [float(v) for v in result.boxes.xyxy[i].tolist()],
                    "segment_xy": poly,
                })
            out.append(dets)
        return out


def _load_backend(spec: str, model_path: str):
    if spec == "yolo":
        return YoloBackend(model_path)
    module, _, name = spec.partition(":")
    import importlib
    return getattr(importlib.import_module(module), name)(model_path)


def _child(conn, model_path: str, backend_spec: str) -> None:
    try:
        os.nice(0)
        backend = _load_backend(backend_spec, model_path)
        conn.send(("ready", {"names": backend.names, "task": backend.task, "device": backend.device}))
    except BaseException as exc:      # noqa: BLE001 - report everything to the parent
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        images, conf, imgsz = msg
        try:
            conn.send(("ok", backend.predict(images, conf, imgsz)))
        except BaseException as exc:  # noqa: BLE001
            conn.send(("error", str(exc)))


class GpuPredictor:
    """Thread-safe front end; one child process, replaced on fatal errors/hangs/crashes."""

    def __init__(self, model_path: str, backend: str = "yolo", timeout: float = 60.0,
                 ready_timeout: float = 180.0, backoff: float = 5.0):
        self.model_path, self.backend = model_path, backend
        self.timeout, self.ready_timeout, self.backoff = timeout, ready_timeout, backoff
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._proc = None
        self._conn = None
        self._info: dict[str, Any] = {}
        self._dead_since = 0.0
        self.restarts = 0
        self.last_restart_reason = ""
        self.last_restart_at = 0.0
        self._ever_started = False

    # -- lifecycle -------------------------------------------------------------
    def _kill(self, reason: str) -> None:
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if proc is not None:
            proc.kill()
            proc.join(10)
        if reason:
            self.restarts += 1
            self.last_restart_reason = reason[:300]
            self.last_restart_at = time.time()
            self._dead_since = time.monotonic()

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        if self._proc is not None:
            self._kill("GPU child exited unexpectedly")
        if self._ever_started and self.backoff and time.monotonic() - self._dead_since < self.backoff:
            raise RuntimeError("StarSeg GPU process restarting after a fatal error; retry shortly")
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=_child, args=(child, self.model_path, self.backend),
                                 name="starseg-gpu", daemon=True)
        proc.start()
        child.close()
        self._proc, self._conn, self._ever_started = proc, parent, True
        if not parent.poll(self.ready_timeout):
            self._kill("GPU child did not become ready")
            raise RuntimeError("StarSeg GPU process timed out while loading the model")
        try:
            kind, payload = parent.recv()
        except (EOFError, OSError):
            self._kill("GPU child exited while loading")
            raise RuntimeError("StarSeg GPU process exited while loading the model")
        if kind != "ready":
            self._kill(f"model load failed: {payload}")
            raise RuntimeError(f"StarSeg model load failed: {payload}")
        self._info = payload

    def info(self) -> dict[str, Any]:
        with self._lock:
            self._ensure()
            return dict(self._info)

    def predict(self, images, conf: float, imgsz: int):
        """Returns (per-image detection lists, device). Raises RuntimeError on failure."""
        with self._lock:
            self._ensure()
            try:
                self._conn.send((list(images), float(conf), int(imgsz)))
            except (OSError, ValueError, BrokenPipeError) as exc:
                self._kill(f"send failed: {exc}")
                raise RuntimeError(f"StarSeg GPU process exited: {exc}") from exc
            if not self._conn.poll(self.timeout):
                self._kill(f"inference timed out after {self.timeout:.0f}s")
                raise RuntimeError(f"StarSeg GPU inference timed out after {self.timeout:.0f}s; process replaced")
            try:
                kind, payload = self._conn.recv()
            except (EOFError, OSError) as exc:
                self._kill("GPU child exited during inference")
                raise RuntimeError("StarSeg GPU process exited during inference; process replaced") from exc
            if kind == "ok":
                return payload, self._info.get("device", "")
            if is_fatal_gpu_error(payload):
                self._kill(payload)
            raise RuntimeError(payload)

    def status(self) -> dict[str, Any]:
        proc = self._proc
        return {"gpu_restarts": self.restarts, "gpu_last_restart_reason": self.last_restart_reason,
                "gpu_last_restart_at": self.last_restart_at,
                "gpu_child_pid": proc.pid if proc is not None and proc.is_alive() else None}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send(None)
                except (OSError, ValueError):
                    pass
            if self._proc is not None:
                self._proc.join(5)
            self._kill("")
