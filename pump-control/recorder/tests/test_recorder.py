from __future__ import annotations

import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from startracked_loop_recorder.app import Handler, Recorder, page, parse_bool, validate_camera
from startracked_loop_recorder.backcast import canonical_videos, main as backcast_main, process
from startracked_loop_recorder.camera import UVCCamera
from startracked_loop_recorder.config import (
    ConfigError, atomic_json_write, default_deployment, default_user_config,
    load_json, validate_contained_polygon, validate_deployment, validate_polygon,
    validate_user_config,
)
from startracked_loop_recorder.starseg import (
    FILTER_POLICY, StarSegJob, StarSegWorker, filter_detections,
)
from startracked_loop_recorder.storage import (
    BackupWorker, Segment, file_sha256, finalize_segment, recover_staging,
    safe_heading,
)
from startracked_loop_recorder.sync_sidecars import sync


MODEL_DIR = Path(__file__).parents[1] / "models" / "target_oversample"


@pytest.mark.skipif(
    not (MODEL_DIR / "best.pt").is_file(),
    reason="segmentation weights (best.pt, 54 MB) are not distributed with the repository",
)
def test_staged_starseg_manifest_and_checkpoint_hash():
    model_dir = MODEL_DIR
    checkpoint = model_dir / "best.pt"
    manifest = json.loads((model_dir / "manifest.json").read_text())
    assert manifest["artifact"]["file"] == checkpoint.name
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest["artifact"]["sha256"]
    assert manifest["artifact"]["source_archive_sha256"] == (
        "172079257efe6b288f892350ce79a0b449a90dd23dec522fef07533cce7fc2ab"
    )
    assert manifest["model"] == {
        "task": "segment",
        "architecture": "yolo26m-seg.pt",
        "imgsz": 1024,
        "classes": {
            "0": "sunflower_star",
            "1": "other_sea_star",
            "2": "live_prey",
            "3": "dead_prey",
        },
    }


def test_deployment_requires_four_unique_stable_paths(tmp_path):
    deployment = default_deployment(tmp_path)
    assert set(validate_deployment(deployment)["cameras"]) == set("ABCD")
    deployment["cameras"]["D"]["path"] = deployment["cameras"]["A"]["path"]
    with pytest.raises(ConfigError, match="unique"):
        validate_deployment(deployment)


def test_user_config_and_atomic_write(tmp_path):
    config = default_user_config()
    config["record_root"] = str(tmp_path / "recordings")
    assert validate_user_config(config) is config
    path = tmp_path / "nested" / "config.json"
    atomic_json_write(path, config)
    assert json.loads(path.read_text())["schema_version"] == 7
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(path.parent.glob(".config.json.*"))


def test_control_validation():
    assert validate_camera("D") == "D"
    with pytest.raises(ConfigError):
        validate_camera("E")
    assert parse_bool("1") is True
    assert parse_bool("off") is False
    with pytest.raises(ConfigError):
        parse_bool("maybe")


def make_segment(tmp_path: Path, fps=2.0) -> Segment:
    segment = Segment.create(
        camera="A", serial="/dev/v4l/by-path/test-video-index0", model="Arducam UVC",
        trial_name="Startracked Loop Recorder", session="recording_start-20260814T100000000",
        session_started=1000.0, start_ts=1000.0, record_root=tmp_path, fps=fps,
        requested_seconds=600, width=64, height=48,
        polygon=default_user_config()["camera_polygons"]["A"],
    )
    for index in range(3):
        image = np.full((48, 64, 3), index * 50, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        segment.add_frame(encoded.tobytes(), 1000 + index / fps)
    return segment


def test_segment_naming_metadata_and_mp4(tmp_path):
    segment = make_segment(tmp_path)
    files = finalize_segment(segment, 1001.5, 34, "medium")
    video, metadata_path, frames_path, summary_path = files
    assert "camera-A" in video.name and "frames-3" in video.name
    assert not segment.staging.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["frames"] == 3
    assert metadata["trial_name"] == "Startracked Loop Recorder"
    assert len(metadata["starseg_roi"]["absolute"]) == 4
    assert metadata["starseg_roi"]["filter_policy"] == FILTER_POLICY
    assert json.loads(summary_path.read_text())["model"] is None
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {"width": 64, "height": 48, "nb_read_frames": "3"}
    assert frames_path.read_text() == ""


def test_finalize_failure_retains_staging(tmp_path):
    segment = make_segment(tmp_path)
    with pytest.raises(Exception):
        finalize_segment(segment, 1002, 34, "medium", delete_failed=False, ffmpeg="/missing/ffmpeg")
    assert segment.staging.exists()
    assert len(list(segment.staging.glob("*.jpg"))) == 3


def test_staging_recovery_quarantines(tmp_path):
    segment = make_segment(tmp_path)
    report = recover_staging(tmp_path)
    assert report == {"recovered": 0, "quarantined": 1}
    assert not segment.staging.exists()
    assert len(list((tmp_path / "_quarantined_segments").glob("A/*"))) == 1


def test_backup_checksums_and_transient_refusal(tmp_path):
    primary, backup = tmp_path / "primary", tmp_path / "backup"
    source = primary / "A" / "session" / "segment" / "sidecar.json"
    source.parent.mkdir(parents=True)
    source.write_text("one")
    worker = BackupWorker(primary, backup, throttle_mbps=0)
    destination = worker.copy(source)
    assert file_sha256(source) == file_sha256(destination)
    destination.write_text("two")
    worker.copy(source)
    assert destination.read_text() == "one"
    transient = primary / "preview" / "A.jpg"
    transient.parent.mkdir(parents=True)
    transient.write_bytes(b"x")
    with pytest.raises(ValueError):
        worker.enqueue(transient)


def starseg_job(
    camera="A", frame_index=-1, *, recorded=False, jsonl=None, frame_ts=None
):
    return StarSegJob(
        camera, frame_index, time.time() if frame_ts is None else frame_ts, "",
        np.full((1, 1, 3), max(0, frame_index) % 255, dtype=np.uint8),
        [[0, 0], [1, 0], [1, 1]], "Trial", jsonl, recorded,
    )


def fake_starseg_record(job):
    return {
        "camera": job.camera, "frame_index": job.frame_index,
        "frame_ts": job.frame_ts, "recorded": job.recorded,
    }


def test_starseg_preview_is_bounded_and_latest_frame_wins():
    worker = StarSegWorker(lambda: {"enabled": False}, threading.Event())
    entered, release = threading.Event(), threading.Event()

    def infer(jobs):
        if any(job.camera == "A" and job.frame_index == 0 for job in jobs):
            entered.set()
            assert release.wait(2)
        now = time.monotonic()
        return [
            {
                **fake_starseg_record(job),
                "_inference_started_monotonic": now,
                "_inference_finished_monotonic": now,
            }
            for job in jobs
        ]

    worker._infer_control_batch = infer
    worker.start()
    worker.submit_control(starseg_job("A", 0))
    assert entered.wait(2)
    for index in range(1, 101):
        worker.submit_control(starseg_job("A", index))
    for camera in "BCD":
        worker.submit_control(starseg_job(camera, 100))

    status = worker.status()
    assert status["preview_queue_depth"] == 4
    assert status["preview_inflight"] == 1
    assert status["total_pending"] == 5
    assert status["preview_replacements"] == 99
    assert status["preview_dropped"] == 99

    release.set()
    deadline = time.monotonic() + 2
    while (
        worker.latest.get("A", {}).get("frame_index") != 100
        or any(camera not in worker.latest for camera in "BCD")
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.latest["A"]["frame_index"] == 100
    assert all(worker.latest[camera]["frame_index"] == 100 for camera in "BCD")
    worker.close()
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_starseg_recorded_jobs_have_priority_and_complete_jsonl(tmp_path):
    worker = StarSegWorker(lambda: {"enabled": False}, threading.Event())
    output = tmp_path / "starseg_frames.jsonl"
    seen = []
    preview_started, release_preview = threading.Event(), threading.Event()

    def infer(job):
        seen.append((job.recorded, job.camera, job.frame_index))
        return fake_starseg_record(job)

    worker._infer = infer
    def infer_control(jobs):
        preview_started.set()
        assert release_preview.wait(2)
        now = time.monotonic()
        return [
            {
                **fake_starseg_record(job),
                "_inference_started_monotonic": now,
                "_inference_finished_monotonic": now,
            }
            for job in jobs
        ]

    worker._infer_control_batch = infer_control
    for camera in "ABCD":
        worker.submit_control(starseg_job(camera, 100))
    for index in range(12):
        worker.submit(starseg_job("A", index, recorded=True, jsonl=output))

    worker.start()
    assert preview_started.wait(2)
    assert worker.status()["preview_inflight"] == 4
    release_preview.set()
    worker.wait_recorded(output, timeout=2)
    assert seen == [(True, "A", index) for index in range(12)]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["frame_index"] for row in rows] == list(range(12))
    assert worker.status()["recorded_pending"] == 0
    worker.close()
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_starseg_recorded_backpressure_is_lossless_and_join_is_clean(tmp_path):
    worker = StarSegWorker(lambda: {"enabled": False}, threading.Event())
    worker.max_recorded_queue = 2
    output = tmp_path / "starseg_frames.jsonl"
    release = threading.Event()

    def infer(job):
        assert release.wait(2)
        return fake_starseg_record(job)

    worker._infer = infer

    def produce():
        for index in range(10):
            worker.submit(starseg_job("A", index, recorded=True, jsonl=output))

    producer = threading.Thread(target=produce)
    producer.start()
    deadline = time.monotonic() + 2
    while worker.status()["recorded_queue_depth"] < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.status()["recorded_queue_depth"] == 2
    assert producer.is_alive()

    worker.start()
    deadline = time.monotonic() + 2
    while worker.metrics["recorded_waits"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.metrics["recorded_waits"] > 0
    release.set()
    producer.join(timeout=2)
    assert not producer.is_alive()
    worker.wait_recorded(output, timeout=2)
    worker.close()
    worker.join(timeout=2)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["frame_index"] for row in rows] == list(range(10))
    assert worker.metrics["processed_recorded"] == 10
    assert not worker.is_alive()


def test_starseg_control_lane_runs_while_archival_is_saturated(tmp_path):
    worker = StarSegWorker(lambda: {"enabled": False}, threading.Event())
    worker.max_recorded_queue = 1
    output = tmp_path / "starseg_frames.jsonl"
    archival_entered = threading.Event()
    release_archival = threading.Event()
    control_processed = threading.Event()

    def infer_archival(job):
        archival_entered.set()
        assert release_archival.wait(2)
        return fake_starseg_record(job)

    def infer_control(jobs):
        control_processed.set()
        now = time.monotonic()
        return [
            {
                **fake_starseg_record(job),
                "_inference_started_monotonic": now,
                "_inference_finished_monotonic": now,
            }
            for job in jobs
        ]

    worker._infer = infer_archival
    worker._infer_control_batch = infer_control
    worker.submit(starseg_job("A", 1, recorded=True, jsonl=output))
    worker.start()
    assert archival_entered.wait(1)
    worker.submit(starseg_job("A", 2, recorded=True, jsonl=output))
    assert worker.status()["recorded_queue_depth"] == 1
    for camera in "ABCD":
        worker.submit_control(starseg_job(camera, 10))
    assert not control_processed.is_set()
    release_archival.set()
    assert control_processed.wait(1)
    assert set(worker.latest) == set("ABCD")
    assert worker.status()["control_queue_depth"] == 0
    worker.wait_recorded(output, timeout=2)
    worker.close()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert not worker.control_is_alive()


def test_starseg_control_queue_is_bounded_to_one_latest_per_camera():
    worker = StarSegWorker(lambda: {"enabled": False}, threading.Event())
    for index in range(100):
        for camera in "ABCD":
            worker.submit_control(starseg_job(camera, index))
    status = worker.status()
    assert status["control_queue_depth"] == 4
    assert status["control_replacements"] == 396
    assert status["control_dropped"] == 396
    worker.clear_control()
    assert worker.status()["control_queue_depth"] == 0


def test_starseg_control_failure_does_not_kill_archival(tmp_path):
    events = []
    failure_published = threading.Event()

    def publish(event):
        events.append(event)
        failure_published.set()

    worker = StarSegWorker(
        lambda: {"enabled": False}, threading.Event(), publish
    )
    worker._infer = fake_starseg_record
    worker._infer_control_batch = lambda _jobs: (_ for _ in ()).throw(
        RuntimeError("control oom")
    )
    output = tmp_path / "starseg_frames.jsonl"
    worker.submit(starseg_job("A", 1, recorded=True, jsonl=output))
    worker.submit_control(starseg_job("A", 2, recorded=True, jsonl=output))
    worker.start()
    assert failure_published.wait(1)
    worker.wait_recorded(output, timeout=2)
    assert json.loads(output.read_text())["frame_index"] == 1
    assert events[-1].lane == "control"
    assert events[-1].inference_ok is False
    assert worker.status()["control_errors"] == 1
    assert worker.status()["processed_recorded"] == 1
    worker.close()
    worker.join(timeout=2)


def test_starseg_model_reload_waits_for_recording_start(tmp_path):
    worker = StarSegWorker(lambda: {"enabled": True}, threading.Event())
    worker.active_model = str(tmp_path / "old.pt")
    worker.model = object()
    worker.reload_requested = True
    worker.configure_model(str(tmp_path / "new.pt"))
    assert worker.desired_model.endswith("new.pt")
    assert worker.reload_requested is False
    worker.request_reload()
    assert worker.reload_requested is True
    assert worker.control_reload_requested is True


def test_starseg_load_captures_exact_model_provenance(tmp_path, monkeypatch):
    # 2026-10-10: the model now lives in a GPU child process (starseg_gpu.GpuPredictor);
    # provenance (path, sha256, task, classes) must be unchanged.
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"checkpoint")

    class FakePredictor:
        def __init__(self, path):
            self.model_path = path

        def info(self):
            return {"task": "segment", "device": "0",
                    "names": {0: "sunflower_star", 1: "other_sea_star", 2: "live_prey", 3: "dead_prey"}}

    worker = StarSegWorker(lambda: {"enabled": False}, threading.Event())
    worker.gpu_factory = FakePredictor
    worker.configure_model(str(model_path))
    worker._load()
    assert isinstance(worker.model, FakePredictor)
    assert worker.model_provenance == {
        "path": str(model_path.resolve()),
        "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "task": "segment",
        "classes": {
            "0": "sunflower_star",
            "1": "other_sea_star",
            "2": "live_prey",
            "3": "dead_prey",
        },
    }


def test_shutdown_sweep_finalizes_segment_without_another_frame(tmp_path):
    deployment = default_deployment(tmp_path)
    deployment_path = tmp_path / "deployment.json"
    atomic_json_write(deployment_path, deployment)
    config = default_user_config()
    config["record_root"] = str(tmp_path / "recordings")
    config_path = tmp_path / "config.json"
    atomic_json_write(config_path, config)
    recorder = Recorder(deployment_path, config_path)
    segment = make_segment(Path(config["record_root"]))
    recorder.segments["A"] = segment
    recorder.stop_requested = True
    recorder._finish_remaining_segments()
    for finalizer in list(recorder.finalizers):
        finalizer.join(timeout=30)
    assert recorder.finalized_segments == 1
    assert recorder.stop_requested is False
    assert not segment.staging.exists()


def test_control_form_and_polygon_persistence(tmp_path):
    deployment = default_deployment(tmp_path)
    deployment_path = tmp_path / "deployment.json"
    atomic_json_write(deployment_path, deployment)
    config = default_user_config()
    config["record_root"] = str(tmp_path / "recordings")
    config_path = tmp_path / "config.json"
    atomic_json_write(config_path, config)
    recorder = Recorder(deployment_path, config_path)
    points = [[0.1, 0.1], [0.9, 0.1], [0.7, 0.8], [0.2, 0.9]]
    recorder.apply_control({
        "trial_name": ["Trial 7"], "camera": ["D"], "fps": ["2.5"],
        "camera_polygon": ["1"], "points": [json.dumps(points)],
    })
    for camera in "ABCD":
        recorder.apply_control({
            "camera": [camera], "exposure_mode": ["Manual"], "exposure_us": ["1200"],
        })
    saved = json.loads(config_path.read_text())
    assert saved["trial_name"] == "Trial 7"
    assert saved["camera_controls"]["D"]["fps"] == 2.5
    assert saved["camera_polygons"]["D"] == points
    assert all(
        controls["exposure_mode"] == "Manual" and controls["exposure_us"] == 1200
        for controls in saved["camera_controls"].values()
    )
    recorder.shutdown_event.set()


def test_invalid_polygon_apply_never_mutates_memory_or_disk(tmp_path):
    deployment_path, config_path = tmp_path / "deployment.json", tmp_path / "config.json"
    atomic_json_write(deployment_path, default_deployment(tmp_path))
    config = default_user_config()
    config["record_root"] = str(tmp_path / "recordings")
    atomic_json_write(config_path, config)
    recorder = Recorder(deployment_path, config_path)
    before_memory = json.loads(json.dumps(recorder.config["camera_polygons"]["A"]))
    before_disk = config_path.read_bytes()

    with pytest.raises(ConfigError, match="self-intersect"):
        recorder.apply_control({
            "camera_polygon": ["1"],
            "camera": ["A"],
            "points": [json.dumps([[0, 0], [1, 1], [0, 1], [1, 0]])],
        })
    with pytest.raises(ConfigError, match="3-64"):
        recorder.apply_control({
            "camera_polygon": ["1"],
            "camera": ["A"],
            "points": ["[]"],
        })

    assert recorder.config["camera_polygons"]["A"] == before_memory
    assert config_path.read_bytes() == before_disk
    recorder.shutdown_event.set()


def test_http_api_forms_and_no_store(tmp_path):
    deployment = default_deployment(tmp_path)
    deployment_path = tmp_path / "deployment.json"
    atomic_json_write(deployment_path, deployment)
    config = default_user_config()
    config["record_root"] = str(tmp_path / "recordings")
    config_path = tmp_path / "config.json"
    atomic_json_write(config_path, config)
    Handler.recorder = Recorder(deployment_path, config_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{url}/") as response:
            assert b"Startracked Loop Recorder" in response.read()
        with urllib.request.urlopen(f"{url}/status.json") as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response)["product"] == "Startracked Loop Recorder"
        request = urllib.request.Request(
            f"{url}/control", data=urllib.parse.urlencode({"trial_name": "HTTP Trial"}).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["trial_name"] == "HTTP Trial"
    finally:
        server.shutdown()
        server.server_close()
        Handler.recorder.shutdown_event.set()


def test_backcast_dry_run_resume_and_sidecar_sync(tmp_path, capsys):
    primary, backup = tmp_path / "primary", tmp_path / "backup"
    folder = primary / "A" / "recording_start-20260814T100000000" / "segment"
    folder.mkdir(parents=True)
    video = folder / "segment.mp4"
    video.write_bytes(b"not decoded in dry run")
    assert canonical_videos(primary, {"A"}) == [video]
    model = tmp_path / "missing.pt"
    assert backcast_main([
        "--root", str(primary), "--model", str(model), "--dry-run", "--min-age", "0"
    ]) == 0
    assert str(video) in capsys.readouterr().out
    sidecar = folder / "segment_starseg_summary.json"
    sidecar.write_text('{"model": "old"}')
    assert sync(primary, backup) == (1, 0)
    assert sync(primary, backup) == (0, 1)


def test_heading_sanitization():
    assert safe_heading(" My experiment #7 ") == "My_experiment__7"


def test_schema_one_migrates_heading_controls_and_outer_polygon(tmp_path):
    old = default_user_config()
    old["schema_version"] = 1
    old["heading"] = old.pop("trial_name")
    old["aquaria_corners"] = {
        camera: points + [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
        for camera, points in old.pop("camera_polygons").items()
    }
    for controls in old["camera_controls"].values():
        controls["exposure_auto"] = "Off"
        controls.pop("exposure_mode")
    path = tmp_path / "config.json"
    atomic_json_write(path, old)
    migrated = load_json(path, default_user_config(), validate_user_config)
    assert migrated["schema_version"] == 7
    assert migrated["pressure_plate_polygons"] == {camera: [] for camera in "ABCD"}
    assert migrated["trial_name"] == "Startracked Loop Recorder"
    assert all(len(points) == 4 for points in migrated["camera_polygons"].values())
    assert all(
        controls["exposure_mode"] == "Manual"
        for controls in migrated["camera_controls"].values()
    )
    assert "heading" not in json.loads(path.read_text())


@pytest.mark.parametrize("points,message", [
    ([[0, 0], [1, 1]], "3-64"),
    ([[0, 0], [1, 1], [0, 1], [1, 0]], "self-intersect"),
    ([[0, 0], [0.5, 0.5], [1, 1]], "nonzero"),
    ([[0, 0], [2, 0], [0, 1]], "normalized"),
])
def test_polygon_validation(points, message):
    with pytest.raises(ConfigError, match=message):
        validate_polygon(points)


def test_pressure_plate_containment_handles_concave_crossing_and_boundary():
    concave_roi = [
        [0, 0], [1, 0], [1, 1], [.6, 1],
        [.6, .4], [.4, .4], [.4, 1], [0, 1],
    ]
    crossing = [[.2, .8], [.8, .8], [.5, .2]]
    with pytest.raises(ConfigError, match="edges"):
        validate_contained_polygon(crossing, concave_roi)
    boundary_touching = [[0, 0], [1, 0], [.5, .3]]
    assert validate_contained_polygon(boundary_touching, concave_roi) == boundary_touching


def test_pressure_plate_invalid_apply_is_atomic_and_roi_is_independent(tmp_path):
    deployment_path, config_path = tmp_path / "deployment.json", tmp_path / "config.json"
    atomic_json_write(deployment_path, default_deployment(tmp_path))
    config = default_user_config()
    config["record_root"] = str(tmp_path / "recordings")
    config["camera_polygons"]["A"] = [
        [0, 0], [1, 0], [1, 1], [.6, 1],
        [.6, .4], [.4, .4], [.4, 1], [0, 1],
    ]
    atomic_json_write(config_path, config)
    recorder = Recorder(deployment_path, config_path)
    before_memory = json.loads(json.dumps(recorder.config))
    before_disk = config_path.read_bytes()
    with pytest.raises(ConfigError, match="edges"):
        recorder.apply_control({
            "camera": ["A"], "pressure_plate_polygon": ["1"],
            "pressure_points": [json.dumps([[.2, .8], [.8, .8], [.5, .2]])],
        })
    assert recorder.config == before_memory
    assert config_path.read_bytes() == before_disk
    assert recorder.config["camera_polygons"]["A"] == before_memory["camera_polygons"]["A"]
    recorder.shutdown_event.set()


def test_pressure_geometry_locks_and_snapshots_while_recording(tmp_path):
    deployment_path, config_path = tmp_path / "deployment.json", tmp_path / "config.json"
    atomic_json_write(deployment_path, default_deployment(tmp_path))
    config = default_user_config()
    config["record_root"] = str(tmp_path / "recordings")
    plate = [[.2, .2], [.8, .2], [.5, .8]]
    config["pressure_plate_polygons"]["A"] = plate
    atomic_json_write(config_path, config)
    recorder = Recorder(deployment_path, config_path)
    recorder.set_recording(True)
    with pytest.raises(ConfigError, match="geometry"):
        recorder.apply_control({
            "camera": ["A"], "pressure_plate_polygon": ["1"],
            "pressure_points": ["[]"],
        })
    frame = SimpleNamespace(timestamp=time.time(), image=np.zeros((48, 64, 3), dtype=np.uint8))
    segment = recorder._new_segment("A", frame)
    assert segment.pressure_polygon == plate
    recorder.config["pressure_plate_polygons"]["A"] = []
    assert segment.pressure_polygon == plate
    with pytest.raises(ConfigError, match="geometry"):
        recorder.apply_control({
            "camera": ["A"], "camera_polygon": ["1"],
            "points": [json.dumps(default_user_config()["camera_polygons"]["A"])],
        })
    recorder.set_recording(False)
    recorder.shutdown_event.set()


def test_centroid_roi_filter_uses_mask_then_box_center():
    detections = [
        {"box_xyxy": [80, 80, 90, 90], "segment_xy": [[10, 10], [20, 10], [15, 20]]},
        {"box_xyxy": [10, 10, 20, 20], "segment_xy": []},
        {"box_xyxy": [80, 80, 90, 90], "segment_xy": []},
    ]
    polygon = [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]
    kept = filter_detections(detections, polygon, 100, 100)
    assert [item["roi_test_source"] for item in kept] == [
        "segmentation_centroid", "box_center",
    ]


@pytest.mark.parametrize("auto_control", ["auto_exposure", "exposure_auto"])
def test_exposure_probe_order_range_live_and_unavailable(monkeypatch, auto_control):
    controls = {"exposure_mode": "Manual", "exposure_us": 1200}
    camera = UVCCamera("A", "/dev/video-test", 64, 48, queue.Queue(), lambda: controls, threading.Event())
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--list-ctrls-menus" in command:
            return SimpleNamespace(
                returncode=0, stderr="",
                stdout=(
                    f"{auto_control} 0x009a0901 (menu) : min=0 max=3 default=3 value=1\n"
                    "exposure_time_absolute 0x009a0902 (int) : "
                    "min=1 max=1000 step=1 default=100 value=12\n"
                ),
            )
        if "--get-ctrl" in command:
            return SimpleNamespace(
                returncode=0, stderr="",
                stdout=f"{auto_control}: 1\nexposure_time_absolute: 12\n",
            )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("startracked_loop_recorder.camera.shutil.which", lambda _: "/usr/bin/v4l2-ctl")
    monkeypatch.setattr("startracked_loop_recorder.camera.subprocess.run", run)
    camera.apply_controls()
    sets = [call[-1] for call in calls if "--set-ctrl" in call]
    assert sets[:2] == [f"{auto_control}=1", "exposure_time_absolute=12"]
    assert camera.status["uvc"]["ranges"]["exposure_us"]["max"] == 100000
    assert camera.status["uvc"]["live"] == {"exposure_mode": "Manual", "exposure_us": 1200}

    unavailable = UVCCamera("B", "/dev/video-test", 64, 48, queue.Queue(), lambda: controls, threading.Event())
    monkeypatch.setattr("startracked_loop_recorder.camera.shutil.which", lambda _: None)
    unavailable.apply_controls()
    assert "install v4l-utils" in unavailable.status["uvc"]["error"]


def test_camera_encoder_handoff_is_bounded_and_keeps_latest_frame():
    output = queue.Queue(maxsize=2)
    camera = UVCCamera(
        "A", "/dev/video-test", 64, 48, output, lambda: {}, threading.Event()
    )
    for index in range(10):
        camera._enqueue_latest(SimpleNamespace(index=index))
    assert output.qsize() == 2
    assert [output.get_nowait().index for _ in range(2)] == [8, 9]
    assert camera.status["encoder_queue_drops"] == 8


def test_camera_editors_are_refreshed_independently_only_when_pristine():
    html = page().decode()
    for control in ("fps", "mode", "exposure", "gain"):
        assert f"markControlDirty('A','{control}')" in html
    assert 'id="exp-A" type="number" value="10000" disabled' in html
    assert "document.activeElement!==element" in html
    assert "!v.controlDirty[key]&&!v.applying" in html
    assert "refreshPristineControl(v,'fps',e.fps,x.controls.fps)" in html
    assert "refreshPristineControl(v,'mode',e.mode,x.controls.exposure_mode||'Auto')" in html
    assert "refreshPristineControl(v,'exposure',e.exposure,x.controls.exposure_us)" in html
    assert "refreshPristineControl(v,'gain',e.gain,x.controls.gain)" in html


def test_camera_apply_waits_for_backend_and_live_uvc_confirmation():
    html = page().decode()
    assert "await waitForCameraApply(c,wanted)" in html
    assert "cameraApplyConfirmed(x,wanted)" in html
    assert "controls.exposure_mode!==wanted.exposure_mode" in html
    assert "live.exposure_mode!==wanted.exposure_mode" in html
    assert "sameNumber(live.exposure_us,wanted.exposure_us)" in html
    assert "Live UVC controls unavailable" in html
    assert "Apply failed: " in html
    assert "v.controlVersion[key]===versions[key]" in html
    assert "Applied; newer edits unsaved" in html


def test_polygon_editor_javascript_syntax_and_draft_state_machine():
    html = page("A").decode()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    checked = subprocess.run(
        ["node", "--check", "-"],
        input=script,
        text=True,
        capture_output=True,
    )
    assert checked.returncode == 0, checked.stderr

    editor = script.split("function markControlDirty", 1)[0]
    exercise = r"""
const assert = require('node:assert/strict');
const elements = new Map();
global.document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      textContent: '', disabled: false, attrs: {},
      setAttribute(name, value) { this.attrs[name] = value; }
    });
    return elements.get(id);
  }
};
draw = (c, points) => {
  views[c].polygon = points;
  updatePolygonGuidance(c);
};
views.A = {
  polygon: [[.1,.1],[.9,.1],[.9,.9],[.1,.9]],
  persisted: [[.1,.1],[.9,.1],[.9,.9],[.1,.9]],
  serverPolygon: [[.1,.1],[.9,.1],[.9,.9],[.1,.9]],
  dirty: false, editing: false, addMode: false, history: [], selected: null
};
let writes = [];
control = async payload => {
  writes.push(structuredClone(payload));
  return {camera_polygons: {A: structuredClone(views.A.polygon)}};
};
(async () => {
  resetPolygon('A');
  assert.deepEqual(views.A.polygon, []);
  assert.equal(views.A.persisted.length, 4);
  assert.equal(elements.get('polygon-apply-A').disabled, true);
  assert.match(elements.get('edit-A').textContent, /0 vertices · Add 3 more to apply/);
  await applyPolygon('A');
  assert.equal(writes.length, 0);
  assert.equal(views.A.persisted.length, 4);
  assert.match(elements.get('edit-A').textContent, /Cannot apply/);

  toggleAddMode('A');
  assert.equal(views.A.addMode, true);
  assert.equal(elements.get('add-A').attrs['aria-pressed'], 'true');
  appendVertex('A', [.2,.2]);
  assert.equal(views.A.polygon.length, 1);
  assert.equal(elements.get('polygon-apply-A').disabled, true);
  assert.match(elements.get('edit-A').textContent, /1 vertex · Add 2 more to apply/);
  appendVertex('A', [.8,.2]);
  assert.equal(views.A.polygon.length, 2);
  assert.equal(elements.get('polygon-apply-A').disabled, true);
  assert.match(elements.get('edit-A').textContent, /2 vertices · Add 1 more to apply/);
  assert.equal(views.A.addMode, true);
  assert.equal(writes.length, 0);
  cancelPolygon('A');
  assert.equal(views.A.polygon.length, 4);
  assert.equal(views.A.dirty, false);
  assert.equal(views.A.addMode, false);
  assert.equal(writes.length, 0);
  resetPolygon('A');
  toggleAddMode('A');
  appendVertex('A', [.2,.2]);
  appendVertex('A', [.8,.2]);
  appendVertex('A', [.5,.8]);
  assert.equal(elements.get('polygon-apply-A').disabled, false);
  await applyPolygon('A');
  assert.equal(writes.length, 1);
  assert.equal(views.A.persisted.length, 3);
  assert.equal(views.A.dirty, false);
  assert.equal(views.A.editing, false);
  assert.equal(views.A.addMode, false);
  resetPolygon('A');
  cancelPolygon('A');
  assert.equal(views.A.polygon.length, 3);
  assert.equal(writes.length, 1);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    exercised = subprocess.run(
        ["node", "-"],
        input=editor + exercise,
        text=True,
        capture_output=True,
    )
    assert exercised.returncode == 0, exercised.stderr
    assert 'button.mode[aria-pressed="true"]' in html
    assert "if(!v.editing&&!v.dirty&&!v.vertexDrag)" in html
    assert "s.setPointerCapture(e.pointerId)" in html
    assert "if(v.addMode)" in html
    assert "pressurePolygon:[]" in html
    assert "Reset pressure plate" in html
    assert "data.pressureVertex" not in html
    assert "dataset.pressureVertex" in html
    assert "draw(c,views[c].polygon);drawPlate(c,[])" in html


def test_polygon_renderer_uses_parseable_responsive_user_space_points():
    html = page("A").decode()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    renderer = script.split("function zoom", 1)[0]
    exercise = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.dataset = {};
    this.attrs = {};
    this.clientWidth = 0;
    this.clientHeight = 0;
    this.points = {numberOfItems: 0};
    this.style = {};
  }
  setAttribute(name, value) {
    value = String(value);
    this.attrs[name] = value;
    if (name === 'points') {
      const number = String.raw`-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?`;
      const pair = new RegExp(`^${number},${number}$`, 'i');
      const values = value.trim() ? value.trim().split(/\s+/) : [];
      this.points.numberOfItems = values.length && values.every(x => pair.test(x))
        ? values.length : 0;
    }
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  querySelector(selector) {
    if (selector === '[data-layer="boundary"]')
      return this.children.find(x => x.dataset.layer === 'boundary') || null;
    if (selector === '[data-layer="overlay"]')
      return this.children.find(x => x.dataset.layer === 'overlay') || null;
    return null;
  }
  getBoundingClientRect() {
    return {left: 0, top: 0, width: this.clientWidth, height: this.clientHeight};
  }
}
const svg = new Element('svg');
svg.clientWidth = 400;
svg.clientHeight = 300;
const image = new Element('img');
image.naturalWidth = 1920;
image.naturalHeight = 1080;
const apply = new Element('button');
const edit = new Element('span');
global.document = {
  getElementById(id) {
    if (id === 'svg-A') return svg;
    if (id === 'img-A') return image;
    if (id === 'polygon-apply-A') return apply;
    if (id === 'edit-A') return edit;
    throw new Error(`Unexpected id ${id}`);
  },
  createElementNS(namespace, tag) { return new Element(tag); }
};
views.A = {
  imageWidth: 16, imageHeight: 9, selected: null,
  polygon: [], dirty: false, addMode: false
};
draw('A', []);
let boundary = svg.querySelector('[data-layer="boundary"]');
assert.equal(boundary.children.length, 0);
assert.equal(apply.disabled, true);
assert.match(edit.textContent, /0 vertices/);

const polygon = [[.08,.08],[.92,.08],[.92,.92],[.08,.92]];
draw('A', polygon);
boundary = svg.querySelector('[data-layer="boundary"]');
let shape = boundary.children[0];
assert.equal(shape.tagName, 'polygon');
assert.equal(shape.points.numberOfItems, 4);
assert.equal(shape.attrs.points.includes('%'), false);
assert.equal(svg.attrs.viewBox, '0 0 400 300');
assert.deepEqual(shape.attrs.points.split(' ')[0].split(',').map(Number), [32, 55.5]);
for (const vertex of boundary.children.slice(1)) {
  assert.equal(vertex.attrs.cx.includes('%'), false);
  assert.equal(vertex.attrs.cy.includes('%'), false);
}

svg.clientWidth = 800;
svg.clientHeight = 400;
draw('A', polygon);
boundary = svg.querySelector('[data-layer="boundary"]');
shape = boundary.children[0];
assert.equal(shape.points.numberOfItems, 4);
assert.equal(svg.attrs.viewBox, '0 0 800 400');
const resizedFirst = shape.attrs.points.split(' ')[0].split(',').map(Number);
assert.ok(Math.abs(resizedFirst[0] - 101.33333333333333) < 1e-9);
assert.equal(resizedFirst[1], 32);
"""
    exercised = subprocess.run(
        ["node", "-"],
        input=renderer + exercise,
        text=True,
        capture_output=True,
    )
    assert exercised.returncode == 0, exercised.stderr
    assert "new ResizeObserver" in html
    assert "x[0]*100+'%,'" not in html


def test_pressure_plate_editor_has_independent_draft_reset_and_apply_state():
    html = page("A").decode()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    editor = script.split("function fullscreen", 1)[0]
    exercise = r"""
const assert = require('node:assert/strict');
const elements = new Map();
global.document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      textContent: '', disabled: false, attrs: {},
      setAttribute(name, value) { this.attrs[name] = value; }
    });
    return elements.get(id);
  }
};
drawPlate = (c, points) => {
  views[c].pressurePolygon = points;
  updatePlateGuidance(c);
};
draw = (c, points) => { views[c].polygon = points; };
views.A = {
  polygon: [[0,0],[1,0],[1,1],[0,1]], addMode: false,
  pressurePolygon: [], pressurePersisted: [], serverPressurePolygon: [],
  pressureDirty: false, pressureEditing: false, plateAddMode: false,
  pressureHistory: [], pressureSelected: null
};
let writes = [];
control = async payload => {
  writes.push(structuredClone(payload));
  return {pressure_plate_polygons: {A: structuredClone(views.A.pressurePolygon)}};
};
(async () => {
  togglePlateAddMode('A');
  appendPlateVertex('A', [.2,.2]);
  appendPlateVertex('A', [.8,.2]);
  appendPlateVertex('A', [.5,.8]);
  assert.equal(views.A.pressurePolygon.length, 3);
  assert.equal(views.A.polygon.length, 4);
  await applyPlate('A');
  assert.equal(writes.length, 1);
  assert.equal(views.A.pressurePersisted.length, 3);
  resetPlate('A');
  assert.deepEqual(views.A.pressurePolygon, []);
  assert.equal(views.A.polygon.length, 4);
  await applyPlate('A');
  assert.equal(writes.length, 2);
  assert.deepEqual(views.A.pressurePersisted, []);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    exercised = subprocess.run(
        ["node", "-"], input=editor + exercise, text=True, capture_output=True,
    )
    assert exercised.returncode == 0, exercised.stderr
    assert "v.pressurePolygon[pressureDrag.index]=svgPoint(c,e)" in html
    assert "drawPlate(c,views[c].pressurePolygon)" in html
    assert "document.addEventListener('pointermove'" in html


def test_status_ui_separates_control_health_from_archival_backlog():
    html = page("A").decode()
    assert "Control q" in html
    assert "Archival q" in html
    assert "BACKLOG GROWING" in html
    assert "last_control_frame_age_ms" in html
    assert "archival_oldest_age_ms" in html


def test_polygon_transform_keeps_image_space_positions_invariant():
    html = page("A").decode()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    geometry = script.split("function setAddMode", 1)[0]
    exercise = r"""
const assert = require('node:assert/strict');
const svg = {
  clientWidth: 400, clientHeight: 300, style: {},
  getBoundingClientRect() {
    const v = views.A;
    return {
      left: (this.clientWidth - this.clientWidth * v.z) / 2 + v.x,
      top: (this.clientHeight - this.clientHeight * v.z) / 2 + v.y,
      width: this.clientWidth * v.z,
      height: this.clientHeight * v.z
    };
  }
};
const image = {naturalWidth: 1920, naturalHeight: 1080, style: {}};
global.document = {
  getElementById(id) {
    if (id === 'svg-A') return svg;
    if (id === 'img-A') return image;
    throw new Error(`Unexpected id ${id}`);
  }
};
views.A = {z: 1, x: 0, y: 0, imageWidth: 1920, imageHeight: 1080};
const normalized = [.23, .71];

function eventForNormalized(point) {
  const g = previewGeometry('A');
  const local = renderPoint(g, point);
  const rect = svg.getBoundingClientRect();
  return {
    clientX: rect.left + local[0] * views.A.z,
    clientY: rect.top + local[1] * views.A.z
  };
}
function assertRegistered() {
  transform('A');
  assert.equal(svg.style.transform, image.style.transform);
  const recovered = svgPoint('A', eventForNormalized(normalized));
  assert.ok(Math.abs(recovered[0] - normalized[0]) < 1e-12);
  assert.ok(Math.abs(recovered[1] - normalized[1]) < 1e-12);
}

assertRegistered();
for (const state of [
  [1.2, 37, -19], [2.4, -83, 61], [.75, 12, 44],
  [3.1, 145, -97], [1, 0, 0]
]) {
  [views.A.z, views.A.x, views.A.y] = state;
  assertRegistered();
}

// Fullscreen-sized layout uses the same image-space mapping.
svg.clientWidth = 1280;
svg.clientHeight = 720;
views.A.z = 2.2;
views.A.x = -211;
views.A.y = 108;
assertRegistered();
"""
    exercised = subprocess.run(
        ["node", "-"],
        input=geometry + exercise,
        text=True,
        capture_output=True,
    )
    assert exercised.returncode == 0, exercised.stderr
    assert ".style.transform=value" in html


def test_trial_name_snapshots_and_is_immutable_while_recording(tmp_path):
    deployment_path, config_path = tmp_path / "deployment.json", tmp_path / "config.json"
    atomic_json_write(deployment_path, default_deployment(tmp_path))
    config = default_user_config()
    config["record_root"] = str(tmp_path / "recordings")
    atomic_json_write(config_path, config)
    recorder = Recorder(deployment_path, config_path)
    recorder.apply_control({"trial_name": ["Raw Trial #9"], "recording": ["1"]})
    assert recorder.active_trial_name == "Raw Trial #9"
    assert recorder.session.startswith("Raw_Trial__9_recording_start-")
    with pytest.raises(ConfigError, match="immutable"):
        recorder.apply_control({"trial_name": ["Changed"]})
    recorder.set_recording(False)
    recorder.shutdown_event.set()


def test_backcast_applies_same_roi_policy(tmp_path, monkeypatch):
    folder = tmp_path / "A" / "Trial_recording_start-1" / "segment"
    folder.mkdir(parents=True)
    video = folder / "segment.mp4"
    video.write_bytes(b"x")
    metadata = video.with_suffix(".json")
    atomic_json_write(metadata, {
        "camera": "A", "trial_name": "Trial", "frame_timestamps": [1.0],
        "starseg_roi": {
            "normalized": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]],
            "absolute": [[0, 0], [50, 0], [50, 50], [0, 50]],
        },
    })
    raw = [
        {"class_id": 0, "class_name": "star", "confidence": .9,
         "box_xyxy": [10, 10, 20, 20], "segment_xy": []},
        {"class_id": 0, "class_name": "star", "confidence": .8,
         "box_xyxy": [80, 80, 90, 90], "segment_xy": []},
    ]
    monkeypatch.setattr("startracked_loop_recorder.backcast.detections_from_result", lambda _: raw)
    model = SimpleNamespace(predict=lambda **_: iter([SimpleNamespace(orig_shape=(100, 100))]))
    args = SimpleNamespace(conf=.25, imgsz=1024, device="cpu", model=tmp_path / "model.pt", frame_delay=0)
    process(video, model, args)
    row = json.loads(video.with_name("segment_starseg_frames.jsonl").read_text())
    assert len(row["detections"]) == 1
    assert row["roi"]["filter_policy"] == FILTER_POLICY
