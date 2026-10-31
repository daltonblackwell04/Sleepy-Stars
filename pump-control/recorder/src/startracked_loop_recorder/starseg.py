from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .hue import InferenceEvent, SUNFLOWER_CLASS_NAMES, is_sunflower_class
from .starseg_gpu import GpuPredictor


@dataclass
class StarSegJob:
    camera: str
    frame_index: int
    frame_ts: float
    frame_file: str
    image: Any
    polygon: list[list[float]]
    trial_name: str
    jsonl: Path | None
    recorded: bool
    session_token: str = ""
    pressure_polygon: list[list[float]] | None = None
    capture_monotonic: float = field(default_factory=time.monotonic)
    enqueue_monotonic: float = field(default_factory=time.monotonic)


FILTER_POLICY = "segmentation_centroid_or_box_center_inside_polygon"


def polygon_centroid(points: list[list[float]]) -> tuple[float, float]:
    if len(points) < 3:
        return (0.0, 0.0)
    cross_sum = cx = cy = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        cross = point[0] * following[1] - following[0] * point[1]
        cross_sum += cross
        cx += (point[0] + following[0]) * cross
        cy += (point[1] + following[1]) * cross
    if abs(cross_sum) < 1e-12:
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
    return cx / (3 * cross_sum), cy / (3 * cross_sum)


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        cross = (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])
        if abs(cross) <= 1e-9 and min(a[0], b[0]) - 1e-9 <= x <= max(a[0], b[0]) + 1e-9 and min(a[1], b[1]) - 1e-9 <= y <= max(a[1], b[1]) + 1e-9:
            return True
        if (a[1] > y) != (b[1] > y):
            intersection = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < intersection:
                inside = not inside
    return inside


def filter_detections_absolute(
    detections: list[dict[str, Any]], absolute: list[list[float]]
) -> list[dict[str, Any]]:
    retained = []
    for detection in detections:
        mask = detection.get("segment_xy") or []
        if len(mask) >= 3:
            center = polygon_centroid(mask)
            source = "segmentation_centroid"
        else:
            x1, y1, x2, y2 = detection["box_xyxy"]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            source = "box_center"
        if point_in_polygon(center, absolute):
            item = dict(detection)
            item["roi_test_point"] = [float(center[0]), float(center[1])]
            item["roi_test_source"] = source
            retained.append(item)
    return retained


def filter_detections(
    detections: list[dict[str, Any]], polygon: list[list[float]], width: int, height: int
) -> list[dict[str, Any]]:
    absolute = [[x * width, y * height] for x, y in polygon]
    return filter_detections_absolute(detections, absolute)


def detections_from_result(result) -> list[dict[str, Any]]:
    detections = []
    for index in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[index].item())
        polygon = []
        if result.masks is not None and index < len(result.masks.xy):
            polygon = [[float(x), float(y)] for x, y in result.masks.xy[index].tolist()]
        detections.append({
            "class_id": cls_id, "class_name": str(result.names[cls_id]),
            "confidence": float(result.boxes.conf[index].item()),
            "box_xyxy": [float(v) for v in result.boxes.xyxy[index].tolist()],
            "segment_xy": polygon,
        })
    return detections


class StarSegWorker(threading.Thread):
    def __init__(
        self, settings, shutdown: threading.Event,
        on_inference: Callable[[InferenceEvent], None] | None = None,
    ):
        super().__init__(name="starseg-worker", daemon=True)
        self.settings, self.shutdown = settings, shutdown
        self.max_recorded_queue = 240
        self._condition = threading.Condition()
        self._recorded_jobs: deque[StarSegJob] = deque()
        self._control_jobs: dict[str, StarSegJob] = {}
        self._control_order: deque[str] = deque()
        self._recorded_pending = 0
        self._recorded_by_output: dict[str, int] = {}
        self._recorded_failures: dict[str, list[str]] = {}
        self._active_recorded = False
        self._active_control = 0
        self._closed = False
        self._started_monotonic = time.monotonic()
        self._recorded_submitted = 0
        self._control_latency_samples: deque[float] = deque(maxlen=512)
        self._control_age_samples: deque[float] = deque(maxlen=512)
        self._inference_condition = threading.Condition()
        self._inference_active = False
        self._inference_waiters = {"control": 0, "archival": 0}
        self._last_inference_lane = ""
        self.desired_model = ""
        self.active_model = ""
        self.reload_requested = False
        self.control_reload_requested = False
        self.model = None
        self.control_model = None
        # 2026-10-10: GPU work runs in a replaceable child process (see starseg_gpu.py).
        self.gpu_factory: Callable[[str], Any] = lambda path: GpuPredictor(path)
        self.gpu: Any = None
        self._gpu_lock = threading.Lock()
        self.control_active_model = ""
        self.control_model_provenance: dict[str, Any] = {}
        self._control_thread = threading.Thread(
            target=self._run_control, name="starseg-control", daemon=True
        )
        self.on_inference = on_inference
        self.model_provenance: dict[str, Any] = {}
        self.latest: dict[str, Any] = {}
        self.metrics = {
            "processed": 0, "processed_recorded": 0,
            "control_processed": 0, "control_batches": 0,
            "control_replacements": 0, "control_dropped": 0,
            "control_errors": 0,
            "recorded_waits": 0, "recorded_high_water": 0,
            "errors": 0, "high_water": 0, "last_latency_ms": 0,
            "last_recorded_latency_ms": 0,
            "last_control_latency_ms": 0, "last_control_frame_age_ms": 0,
            "max_control_frame_age_ms": 0,
            "last_error": "", "control_last_error": "",
        }

    def configure_model(self, path: str) -> None:
        if path != self.desired_model:
            self.desired_model = path
            self.reload_requested = False
            self.control_reload_requested = False

    def request_reload(self) -> None:
        if self.desired_model != self.active_model:
            self.reload_requested = True
        if self.desired_model != self.control_active_model:
            self.control_reload_requested = True

    @staticmethod
    def _output_key(job: StarSegJob) -> str:
        return str(job.jsonl.resolve()) if job.jsonl is not None else ""

    def _update_high_water_locked(self) -> None:
        recorded_depth = len(self._recorded_jobs)
        total_depth = recorded_depth + len(self._control_jobs)
        self.metrics["recorded_high_water"] = max(
            self.metrics["recorded_high_water"], recorded_depth
        )
        self.metrics["high_water"] = max(self.metrics["high_water"], total_depth)

    def _submit_control_locked(self, job: StarSegJob) -> bool:
        if self._closed:
            self.metrics["control_dropped"] += 1
            return False
        if job.camera in self._control_jobs:
            self.metrics["control_replacements"] += 1
            self.metrics["control_dropped"] += 1
        else:
            self._control_order.append(job.camera)
        self._control_jobs[job.camera] = job
        return True

    def submit(self, job: StarSegJob) -> bool:
        with self._condition:
            if job.recorded:
                waited = False
                while len(self._recorded_jobs) >= self.max_recorded_queue and not self._closed:
                    if not waited:
                        self.metrics["recorded_waits"] += 1
                        waited = True
                    self._condition.wait()
                if self._closed:
                    raise RuntimeError("StarSeg worker is closed")
                self._recorded_jobs.append(job)
                self._recorded_pending += 1
                self._recorded_submitted += 1
                key = self._output_key(job)
                self._recorded_by_output[key] = self._recorded_by_output.get(key, 0) + 1
            else:
                if not self._submit_control_locked(job):
                    return False
            self._update_high_water_locked()
            # notify_all: archival and control threads share this condition; notify() could
            # wake only the control thread and leave a recorded job unserved.
            self._condition.notify_all()
            return True

    def submit_control(self, job: StarSegJob) -> bool:
        """Nonblocking latest-frame submission, independent of archival pressure."""
        with self._condition:
            accepted = self._submit_control_locked(job)
            self._update_high_water_locked()
            self._condition.notify_all()
            return accepted

    def wait_recorded(self, output: Path | None = None, timeout: float | None = None) -> None:
        key = str(output.resolve()) if output is not None else None
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while (
                self._recorded_pending if key is None
                else self._recorded_by_output.get(key, 0)
            ):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for recorded StarSeg jobs")
                self._condition.wait(remaining)
            if key is None:
                failures = [
                    message
                    for values in self._recorded_failures.values()
                    for message in values
                ]
                self._recorded_failures.clear()
            else:
                failures = self._recorded_failures.pop(key, [])
            if failures:
                raise RuntimeError(f"recorded StarSeg inference failed: {failures[-1]}")

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self.metrics["control_dropped"] += len(self._control_jobs)
            self._control_jobs.clear()
            self._control_order.clear()
            self._condition.notify_all()

    def clear_control(self, session_token: str | None = None) -> None:
        """Discard queued live work on recording stop/session rollover."""
        with self._condition:
            cameras = [
                camera for camera, job in self._control_jobs.items()
                if session_token is None or job.session_token == session_token
            ]
            for camera in cameras:
                self._control_jobs.pop(camera, None)
            if cameras:
                self.metrics["control_dropped"] += len(cameras)
                self._control_order = deque(
                    camera for camera in self._control_order
                    if camera in self._control_jobs
                )
            self._condition.notify_all()

    def _next_job(self) -> StarSegJob | None:
        with self._condition:
            while not self._recorded_jobs:
                if self._closed:
                    return None
                self._condition.wait()
            if self._recorded_jobs:
                job = self._recorded_jobs.popleft()
                self._active_recorded = True
                self._condition.notify_all()
                return job
            return None

    def _next_control_batch(self) -> list[StarSegJob] | None:
        with self._condition:
            while not self._control_jobs:
                if self._closed:
                    return None
                self._condition.wait()
            jobs = []
            while self._control_order:
                camera = self._control_order.popleft()
                job = self._control_jobs.pop(camera, None)
                if job is not None:
                    jobs.append(job)
            self._active_control = len(jobs)
            return jobs

    @contextmanager
    def _inference_slot(self, lane: str):
        """Serialize CUDA use and alternate lanes when both are waiting."""
        other = "archival" if lane == "control" else "control"
        with self._inference_condition:
            self._inference_waiters[lane] += 1
            try:
                while (
                    self._inference_active
                    or (
                        self._inference_waiters[other] > 0
                        and self._last_inference_lane == lane
                    )
                ):
                    self._inference_condition.wait()
                self._inference_active = True
            finally:
                self._inference_waiters[lane] -= 1
        try:
            yield
        finally:
            with self._inference_condition:
                self._inference_active = False
                self._last_inference_lane = lane
                self._inference_condition.notify_all()

    def _complete_job(self, job: StarSegJob, error: str = "") -> None:
        with self._condition:
            if job.recorded:
                self._active_recorded = False
                self._recorded_pending -= 1
                key = self._output_key(job)
                remaining = self._recorded_by_output.get(key, 0) - 1
                if remaining:
                    self._recorded_by_output[key] = remaining
                else:
                    self._recorded_by_output.pop(key, None)
                if error:
                    self._recorded_failures.setdefault(key, []).append(error)
            self._condition.notify_all()

    def _record_error(self, job: StarSegJob, exc: Exception) -> str:
        message = str(exc)
        self.metrics["errors"] += 1
        self.metrics["last_error"] = message
        return message

    def _publish(self, job: StarSegJob, record: dict[str, Any] | None) -> None:
        if self.on_inference is None:
            return
        pressure_polygon = job.pressure_polygon or []
        width = float((record or {}).get("width", 1))
        height = float((record or {}).get("height", 1))
        absolute_plate = [[x * width, y * height] for x, y in pressure_polygon]
        detections = tuple(
            (
                str(item.get("class_name", "")),
                float((item.get("roi_test_point") or [0, 0])[0]),
                float((item.get("roi_test_point") or [0, 0])[1]),
            )
            for item in (record or {}).get("detections", ())
            if (
                not is_sunflower_class(str(item.get("class_name", "")))
                or (
                    len(absolute_plate) >= 3
                    and point_in_polygon(
                        (
                            float((item.get("roi_test_point") or [0, 0])[0]),
                            float((item.get("roi_test_point") or [0, 0])[1]),
                        ),
                        absolute_plate,
                    )
                )
            )
        )
        event_path = (
            job.jsonl.parent / "pressure_plate_events.jsonl"
            if job.recorded and job.jsonl is not None else None
        )
        self.on_inference(InferenceEvent(
            camera=job.camera, frame_index=job.frame_index, frame_ts=job.frame_ts,
            received_monotonic=time.monotonic(), recorded=job.recorded,
            polygon_valid=len(pressure_polygon) >= 3, detections=detections,
            events_jsonl=event_path, session_token=job.session_token,
            inference_ok=record is not None,
            pressure_polygon=tuple((float(x), float(y)) for x, y in pressure_polygon),
            lane="control", capture_monotonic=job.capture_monotonic,
            enqueue_monotonic=job.enqueue_monotonic,
            inference_started_monotonic=float(
                (record or {}).get("_inference_started_monotonic", 0.0)
            ),
            inference_finished_monotonic=float(
                (record or {}).get("_inference_finished_monotonic", time.monotonic())
            ),
        ))

    def _predictor(self) -> Any:
        """Shared GPU predictor for the desired model (created or replaced on model change)."""
        if not self.desired_model or not Path(self.desired_model).is_file():
            raise RuntimeError("StarSeg model file is unavailable")
        with self._gpu_lock:
            if self.gpu is None or getattr(self.gpu, "model_path", None) != self.desired_model:
                if self.gpu is not None:
                    self.gpu.close()
                self.gpu = self.gpu_factory(self.desired_model)
            return self.gpu

    def _provenance(self, info: dict[str, Any]) -> dict[str, Any]:
        digest = hashlib.sha256()
        with Path(self.desired_model).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": str(Path(self.desired_model).resolve()),
            "sha256": digest.hexdigest(),
            "task": str(info.get("task", "")),
            "classes": {str(k): str(v) for k, v in (info.get("names") or {}).items()},
        }

    def _load(self) -> None:
        info = self._predictor().info()
        self.model = self.gpu
        self.model_provenance = self._provenance(info)
        self.active_model = self.desired_model
        self.reload_requested = False

    def _load_control(self) -> None:
        info = self._predictor().info()
        self.control_model = self.gpu
        self.control_model_provenance = self._provenance(info)
        self.control_active_model = self.desired_model
        self.control_reload_requested = False

    def _infer(self, job: StarSegJob) -> dict[str, Any]:
        settings = self.settings()
        if self.reload_requested:
            try:
                self._load()
            except Exception:
                if self.model is not None:
                    self.reload_requested = False
                raise
        if self.model is None:
            raise RuntimeError("StarSeg model activation is pending the next recording start")
        batches, device = self.model.predict(
            [job.image], float(settings["confidence"]), int(settings["imgsz"]),
        )
        width, height = int(job.image.shape[1]), int(job.image.shape[0])
        all_detections = batches[0]
        detections = filter_detections(all_detections, job.polygon, width, height)
        pressure_polygon = job.pressure_polygon or []
        return {
            "camera": job.camera, "frame_index": job.frame_index,
            "frame_ts": job.frame_ts, "frame_file": job.frame_file,
            "model": self.active_model,
            "model_provenance": dict(self.model_provenance),
            "device": device,
            "imgsz": int(settings["imgsz"]),
            "confidence_threshold": float(settings["confidence"]),
            "width": width, "height": height,
            "trial_name": job.trial_name,
            "roi": {
                "normalized": job.polygon,
                "absolute": [[round(x * width), round(y * height)] for x, y in job.polygon],
                "filter_policy": FILTER_POLICY,
                "detections_before_filter": len(all_detections),
                "detections_after_filter": len(detections),
            },
            "pressure_plate": {
                "normalized": pressure_polygon,
                "absolute": [
                    [round(x * width), round(y * height)] for x, y in pressure_polygon
                ],
                "valid": len(pressure_polygon) >= 3,
                "occupancy_class_names": list(SUNFLOWER_CLASS_NAMES),
                "occupancy_policy": (
                    "sunflower_star_roi_test_point_inside_pressure_plate_boundary_inclusive"
                ),
            },
            "detections": detections,
            "lane": "archival",
            "capture_monotonic": job.capture_monotonic,
            "enqueue_monotonic": job.enqueue_monotonic,
        }

    def _infer_control_batch(
        self, jobs: list[StarSegJob]
    ) -> list[dict[str, Any]]:
        settings = self.settings()
        if self.control_reload_requested:
            self._load_control()
        if self.control_model is None:
            raise RuntimeError(
                "StarSeg control model activation is pending the next recording start"
            )
        started = time.monotonic()
        results, device = self.control_model.predict(
            [job.image for job in jobs], float(settings["confidence"]), int(settings["imgsz"]),
        )
        finished = time.monotonic()
        records = []
        if len(results) != len(jobs):
            raise RuntimeError("StarSeg control batch returned an unexpected result count")
        for job, result in zip(jobs, results):
            width, height = int(job.image.shape[1]), int(job.image.shape[0])
            all_detections = result
            detections = filter_detections(
                all_detections, job.polygon, width, height
            )
            pressure_polygon = job.pressure_polygon or []
            records.append({
                "camera": job.camera, "frame_index": job.frame_index,
                "frame_ts": job.frame_ts, "frame_file": job.frame_file,
                "model": self.control_active_model,
                "model_provenance": dict(self.control_model_provenance),
                "device": device, "imgsz": int(settings["imgsz"]),
                "confidence_threshold": float(settings["confidence"]),
                "width": width, "height": height,
                "trial_name": job.trial_name,
                "roi": {
                    "normalized": job.polygon,
                    "absolute": [
                        [round(x * width), round(y * height)]
                        for x, y in job.polygon
                    ],
                    "filter_policy": FILTER_POLICY,
                    "detections_before_filter": len(all_detections),
                    "detections_after_filter": len(detections),
                },
                "pressure_plate": {
                    "normalized": pressure_polygon,
                    "absolute": [
                        [round(x * width), round(y * height)]
                        for x, y in pressure_polygon
                    ],
                    "valid": len(pressure_polygon) >= 3,
                    "occupancy_class_names": list(SUNFLOWER_CLASS_NAMES),
                    "occupancy_policy": (
                        "sunflower_star_roi_test_point_inside_pressure_plate_boundary_inclusive"
                    ),
                },
                "detections": detections, "lane": "control",
                "capture_monotonic": job.capture_monotonic,
                "enqueue_monotonic": job.enqueue_monotonic,
                "_inference_started_monotonic": started,
                "_inference_finished_monotonic": finished,
            })
        return records

    def start(self) -> None:
        self._control_thread.start()
        super().start()

    def join(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        super().join(timeout)
        remaining = (
            None if deadline is None else max(0.0, deadline - time.monotonic())
        )
        if self._control_thread.ident is not None:
            self._control_thread.join(remaining)
        if not self.is_alive() and not self._control_thread.is_alive():
            with self._gpu_lock:
                if self.gpu is not None:
                    self.gpu.close()
                    self.gpu = None

    def control_is_alive(self) -> bool:
        return self._control_thread.is_alive()

    def run(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                break
            started = time.monotonic()
            error = ""
            try:
                with self._inference_slot("archival"):
                    record = self._infer(job)
                self.latest[job.camera] = record
                if job.recorded and job.jsonl:
                    job.jsonl.parent.mkdir(parents=True, exist_ok=True)
                    with job.jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                        handle.flush()
                self.metrics["processed"] += 1
                self.metrics["processed_recorded"] += 1
            except Exception as exc:
                error = self._record_error(job, exc)
            finally:
                latency = round((time.monotonic() - started) * 1000, 2)
                self.metrics["last_latency_ms"] = latency
                self.metrics["last_recorded_latency_ms"] = latency
                self._complete_job(job, error)

    def _run_control(self) -> None:
        while True:
            jobs = self._next_control_batch()
            if jobs is None:
                break
            started = time.monotonic()
            try:
                with self._inference_slot("control"):
                    records = self._infer_control_batch(jobs)
                for job, record in zip(jobs, records):
                    self.latest[job.camera] = {
                        key: value for key, value in record.items()
                        if not key.startswith("_")
                    }
                    self._publish(job, record)
                self.metrics["control_processed"] += len(jobs)
                self.metrics["control_batches"] += 1
                self.metrics["control_last_error"] = ""
            except Exception as exc:
                message = str(exc)
                self.metrics["control_errors"] += len(jobs)
                self.metrics["control_last_error"] = message
                # A failed live batch is explicit fail-closed input. The
                # archival model/thread remains untouched.
                for job in jobs:
                    self._publish(job, None)
                # The GPU predictor replaces its child process itself after a fatal CUDA
                # error or hang; the next batch simply runs in the fresh process.
            finally:
                finished = time.monotonic()
                latency_ms = round((finished - started) * 1000, 2)
                ages = [
                    max(0.0, finished - job.capture_monotonic) * 1000
                    for job in jobs
                ]
                with self._condition:
                    self.metrics["last_control_latency_ms"] = latency_ms
                    self._control_latency_samples.extend(
                        [latency_ms] * max(1, len(jobs))
                    )
                    if ages:
                        self._control_age_samples.extend(ages)
                        self.metrics["last_control_frame_age_ms"] = round(max(ages), 2)
                        self.metrics["max_control_frame_age_ms"] = round(max(
                            self.metrics["max_control_frame_age_ms"], max(ages)
                        ), 2)
                    self._active_control = 0
                    self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        settings = self.settings()
        available = bool(self.desired_model and Path(self.desired_model).is_file())
        with self._condition:
            recorded_queue_depth = len(self._recorded_jobs)
            control_queue_depth = len(self._control_jobs)
            total_pending = (
                self._recorded_pending + control_queue_depth + self._active_control
            )
            now = time.monotonic()
            archival_oldest_age_ms = round(
                max(0.0, now - self._recorded_jobs[0].enqueue_monotonic) * 1000, 2
            ) if self._recorded_jobs else 0
            control_oldest_age_ms = round(
                max(
                    (max(0.0, now - job.capture_monotonic) for job in self._control_jobs.values()),
                    default=0.0,
                ) * 1000,
                2,
            )
            def percentile(values: deque[float], fraction: float) -> float:
                if not values:
                    return 0.0
                ordered = sorted(values)
                index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
                return round(ordered[index], 2)

            elapsed = max(0.001, now - self._started_monotonic)
            return {
                "enabled": bool(settings.get("enabled")), "available": available,
                "desired_model": self.desired_model, "active_model": self.active_model,
                "active_model_provenance": dict(self.model_provenance),
                "unavailable_reason": "" if available else "Model file is unavailable",
                "queue_size": total_pending,
                "total_pending": total_pending,
                "recorded_queue_depth": recorded_queue_depth,
                "recorded_pending": self._recorded_pending,
                "recorded_inflight": int(self._active_recorded),
                "archival_oldest_age_ms": archival_oldest_age_ms,
                "archival_submitted": self._recorded_submitted,
                "archival_throughput_fps": round(
                    self.metrics["processed_recorded"] / elapsed, 3
                ),
                "control_queue_depth": control_queue_depth,
                "control_inflight": self._active_control,
                "control_oldest_frame_age_ms": control_oldest_age_ms,
                "control_available": bool(
                    available and self._control_thread.is_alive()
                    and not self.metrics["control_last_error"]
                ),
                "control_active_model": self.control_active_model,
                "control_model_provenance": dict(self.control_model_provenance),
                "control_latency_p50_ms": percentile(
                    self._control_latency_samples, 0.50
                ),
                "control_latency_p95_ms": percentile(
                    self._control_latency_samples, 0.95
                ),
                "control_frame_age_p50_ms": percentile(
                    self._control_age_samples, 0.50
                ),
                "control_frame_age_p95_ms": percentile(
                    self._control_age_samples, 0.95
                ),
                # Compatibility aliases for existing status consumers.
                "preview_queue_depth": control_queue_depth,
                "preview_inflight": self._active_control,
                "preview_replacements": self.metrics["control_replacements"],
                "preview_dropped": self.metrics["control_dropped"],
                "processed_preview": self.metrics["control_processed"],
                "last_preview_latency_ms": self.metrics["last_control_latency_ms"],
                **(self.gpu.status() if self.gpu is not None else {
                    "gpu_restarts": 0, "gpu_last_restart_reason": "",
                    "gpu_last_restart_at": 0.0, "gpu_child_pid": None}),
                **self.metrics,
            }
