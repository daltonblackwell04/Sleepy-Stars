"""CPU-only finalization regressions; no camera, model or actuator workers start."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from startracked_loop_recorder.app import Recorder
from startracked_loop_recorder.starseg import StarSegWorker
from startracked_loop_recorder.storage import Segment


ERROR = "recorded StarSeg inference failed: CUDA error: unspecified launch failure"


def make_recorder_segment(tmp_path):
    segment = Segment.create(
        camera="A", serial="test", model="test", trial_name="test",
        session="test-session", session_started=1000.0, start_ts=1000.0,
        record_root=tmp_path, fps=1.0, requested_seconds=2.0,
        width=32, height=32, polygon=[],
    )
    ok, jpeg = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    segment.add_frame(jpeg.tobytes(), 1000.0)
    segment.add_frame(jpeg.tobytes(), 1001.0)
    recorder = Recorder.__new__(Recorder)  # Never construct hardware services.
    recorder.lock = threading.RLock()
    recorder.finalizers = set()
    recorder.errors = []
    recorder.finalized_segments = 0
    recorder.deployment = {
        "x264_crf": 28, "x264_preset": "ultrafast", "delete_failed_segments": False,
    }
    recorder.starseg = StarSegWorker(lambda: {}, threading.Event())
    recorder.pressure = SimpleNamespace(flush=Mock())
    recorder.backup = SimpleNamespace(enqueue=Mock())
    return recorder, segment


def finish(recorder, segment):
    # Hold the lock so the thread remains registered until we capture it.
    with recorder.lock:
        recorder._finish_async(segment, 1002.0)
        threads = list(recorder.finalizers)
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()
    assert not recorder.finalizers


@pytest.mark.parametrize("partial_rows", [0, 1])
def test_recorded_inference_failure_finalizes_video_with_honest_summary(tmp_path, partial_rows):
    recorder, segment = make_recorder_segment(tmp_path)
    output = segment.staging / "starseg_frames.jsonl"
    rows = [{"frame_index": 0, "detections": [], "model": "test-model"}] * partial_rows
    if rows:
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    recorder.starseg._recorded_failures[str(output.resolve())] = [
        "CUDA error: unspecified launch failure"
    ]

    finish(recorder, segment)

    assert recorder.finalized_segments == 1, recorder.errors
    _, folder, video = segment.final_paths(1002.0)
    assert video.stat().st_size > 0
    metadata = json.loads(video.with_suffix(".json").read_text())
    summary = json.loads((folder / f"{video.stem}_starseg_summary.json").read_text())
    assert summary["inference_status"] == "failed"
    assert summary["inference_error"] == ERROR
    assert summary["frames"] == partial_rows
    assert metadata["starseg_summary"] == summary
    final_rows = (folder / f"{video.stem}_starseg_frames.jsonl").read_text()
    assert [json.loads(line) for line in final_rows.splitlines()] == rows
    assert any(ERROR in error for error in recorder.errors)
    recorder.pressure.flush.assert_called_once()
    backed_up = {call.args[0] for call in recorder.backup.enqueue.call_args_list}
    assert {video, video.with_suffix(".json"), folder / f"{video.stem}_starseg_summary.json"} <= backed_up
    assert not segment.staging.exists()


@pytest.mark.parametrize("error", [TimeoutError("jobs still writing"), ValueError("unexpected failure")])
def test_undrained_or_unexpected_wait_error_keeps_staging(tmp_path, error):
    recorder, segment = make_recorder_segment(tmp_path)
    recorder.starseg.wait_recorded = Mock(side_effect=error)
    finish(recorder, segment)
    assert recorder.finalized_segments == 0
    assert segment.staging.exists()
    assert len(list(segment.staging.glob("*.jpg"))) == 2
    assert not segment.final_paths(1002.0)[2].exists()
    recorder.pressure.flush.assert_not_called()
    recorder.backup.enqueue.assert_not_called()
    assert recorder.errors == [f"A finalization: {error}"]


@pytest.mark.parametrize("site", ["pressure", "video"])
def test_runtime_error_outside_recorded_wait_still_aborts(tmp_path, monkeypatch, site):
    recorder, segment = make_recorder_segment(tmp_path)
    fail = Mock(side_effect=RuntimeError("not an inference failure"))
    if site == "pressure":
        recorder.pressure.flush = fail
    else:
        monkeypatch.setattr("startracked_loop_recorder.app.finalize_segment", fail)
    finish(recorder, segment)
    assert recorder.finalized_segments == 0
    assert segment.staging.exists()
    assert not (segment.staging / "starseg_inference_failure.json").exists()
    assert recorder.errors == ["A finalization: not an inference failure"]
    recorder.backup.enqueue.assert_not_called()


def test_successful_wait_keeps_normal_summary(tmp_path):
    recorder, segment = make_recorder_segment(tmp_path)
    finish(recorder, segment)
    assert recorder.finalized_segments == 1, recorder.errors
    metadata = json.loads(segment.final_paths(1002.0)[2].with_suffix(".json").read_text())
    assert "inference_error" not in metadata["starseg_summary"]
    assert recorder.errors == []
