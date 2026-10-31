"""2026-10-10: StarSeg must recover from a broken CUDA context without a recorder restart.

Oct 4: two GPU hangs (NVRM Xid 8) left the recorder process's CUDA context permanently broken;
every later inference failed with "launch timed out" for 6 days because recovery only retried
inside the same process. Inference now runs in a child process that is replaced after a fatal
CUDA error, a hang, or a crash.
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from startracked_loop_recorder.starseg import StarSegJob, StarSegWorker
from startracked_loop_recorder.starseg_gpu import GpuPredictor, is_fatal_gpu_error

BACKEND = "fake_gpu_backend:Backend"


def script(tmp_path, actions, names=None):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"names": names or {"0": "sunflower_star"}, "task": "segment",
                                "actions": actions}))
    return str(path)


def image():
    return np.zeros((4, 6, 3), dtype=np.uint8)


def predictor(tmp_path, actions, **kw):
    kw.setdefault("timeout", 5.0); kw.setdefault("ready_timeout", 30.0); kw.setdefault("backoff", 0.0)
    return GpuPredictor(script(tmp_path, actions), backend=BACKEND, **kw)


@pytest.mark.parametrize("message,fatal", [
    ("CUDA error: the launch timed out and was terminated", True),
    ("CUDA error: an illegal memory access was encountered", True),
    ("CUDA out of memory. Tried to allocate 20.00 MiB", True),
    ("CUBLAS_STATUS_EXECUTION_FAILED when calling cublasSgemm", True),
    ("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR", True),
    ("bad input frame", False),
])
def test_fatal_classification(message, fatal):
    assert is_fatal_gpu_error(message) is fatal


def test_predict_returns_detections_and_model_info(tmp_path):
    p = predictor(tmp_path, ["ok"])
    try:
        info = p.info()
        assert info["names"] == {"0": "sunflower_star"} and info["task"] == "segment"
        dets, device = p.predict([image(), image()], 0.25, 640)
        assert device == "cuda-fake" and len(dets) == 2
        assert dets[0][0]["shape"] == [4, 6, 3]
        assert p.restarts == 0
    finally:
        p.close()


def test_fatal_cuda_error_replaces_the_process(tmp_path):
    p = predictor(tmp_path, ["ok", "cuda", "ok"])
    try:
        first_pid = p.predict([image()], 0.25, 640)[0][0][0]["pid"]
        with pytest.raises(RuntimeError, match="launch timed out"):
            p.predict([image()], 0.25, 640)
        dets, _ = p.predict([image()], 0.25, 640)
        assert dets[0][0]["pid"] != first_pid
        assert p.restarts == 1 and "launch timed out" in p.last_restart_reason
    finally:
        p.close()


def test_ordinary_error_keeps_the_process(tmp_path):
    p = predictor(tmp_path, ["ok", "value", "ok"])
    try:
        pid = p.predict([image()], 0.25, 640)[0][0][0]["pid"]
        with pytest.raises(RuntimeError, match="bad input frame"):
            p.predict([image()], 0.25, 640)
        assert p.predict([image()], 0.25, 640)[0][0][0]["pid"] == pid
        assert p.restarts == 0
    finally:
        p.close()


def test_hang_is_killed_and_replaced(tmp_path):
    p = predictor(tmp_path, ["ok", "hang", "ok"], timeout=1.0)
    try:
        pid = p.predict([image()], 0.25, 640)[0][0][0]["pid"]
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="timed out"):
            p.predict([image()], 0.25, 640)
        assert time.monotonic() - started < 5
        assert p.predict([image()], 0.25, 640)[0][0][0]["pid"] != pid
        assert p.restarts == 1
    finally:
        p.close()


def test_crashed_child_is_replaced(tmp_path):
    p = predictor(tmp_path, ["die", "ok"])
    try:
        with pytest.raises(RuntimeError, match="exited"):
            p.predict([image()], 0.25, 640)
        assert len(p.predict([image()], 0.25, 640)[0]) == 1
        assert p.restarts == 1
    finally:
        p.close()


def test_restart_backoff_fails_fast_instead_of_thrashing(tmp_path):
    p = predictor(tmp_path, ["cuda", "ok"], backoff=30.0)
    try:
        with pytest.raises(RuntimeError, match="CUDA"):
            p.predict([image()], 0.25, 640)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="restarting"):
            p.predict([image()], 0.25, 640)
        assert time.monotonic() - started < 1          # no child spawned during backoff
    finally:
        p.close()


def test_close_stops_the_child(tmp_path):
    p = predictor(tmp_path, ["ok"])
    p.predict([image()], 0.25, 640)
    proc = p._proc
    p.close()
    proc.join(5)
    assert not proc.is_alive()


# ---- worker level: the recorder's StarSeg worker recovers without a restart ----

def make_worker(tmp_path, actions):
    model = script(tmp_path, actions)
    settings = {"enabled": True, "confidence": 0.25, "imgsz": 640}
    worker = StarSegWorker(lambda: dict(settings), threading.Event())
    worker.gpu_factory = lambda path: GpuPredictor(path, backend=BACKEND, timeout=5.0,
                                                   ready_timeout=30.0, backoff=0.0)
    worker.configure_model(model)
    worker.request_reload()
    return worker


def job(camera, index, jsonl=None, recorded=False):
    return StarSegJob(camera, index, time.time(), f"{index:08d}.jpg", image(),
                      [[0, 0], [1, 0], [1, 1], [0, 1]], "trial_test", jsonl, recorded)


def test_recorded_lane_recovers_after_fatal_cuda_error(tmp_path):
    out = tmp_path / "starseg_frames.jsonl"
    worker = make_worker(tmp_path, ["ok", "cuda", "ok", "ok"])
    worker.start()
    try:
        for i in range(4):
            worker.submit(job("A", i, out, recorded=True))
        with pytest.raises(RuntimeError, match="launch timed out"):
            worker.wait_recorded(out, timeout=60)
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert [r["frame_index"] for r in rows] == [0, 2, 3]       # only the failing frame is lost
        assert rows[-1]["device"] == "cuda-fake"
        status = worker.status()
        assert status["errors"] == 1 and status["gpu_restarts"] == 1
        assert status["active_model_provenance"]["classes"] == {"0": "sunflower_star"}
    finally:
        worker.close(); worker.join(10)


def test_control_lane_recovers_after_fatal_cuda_error(tmp_path):
    worker = make_worker(tmp_path, ["cuda", "ok", "ok"])
    worker.start()
    try:
        worker.submit_control(job("A", 1))
        deadline = time.monotonic() + 30
        while worker.status()["control_errors"] < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        worker.submit_control(job("A", 2))
        while worker.status()["control_processed"] < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        s = worker.status()
        assert s["control_errors"] == 1 and s["control_processed"] == 1
        assert s["control_last_error"] == "" and s["gpu_restarts"] == 1
    finally:
        worker.close(); worker.join(10)


def test_recorder_process_never_initialises_cuda(tmp_path):
    import sys
    worker = make_worker(tmp_path, ["ok"])
    worker.start()
    try:
        out = tmp_path / "f.jsonl"
        worker.submit(job("A", 0, out, recorded=True)); worker.wait_recorded(out, timeout=60)
        torch = sys.modules.get("torch")
        assert torch is None or not torch.cuda.is_initialized()
    finally:
        worker.close(); worker.join(10)
