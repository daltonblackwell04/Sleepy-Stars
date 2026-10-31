from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from startracked_loop_recorder.backcast import (
    MissingArchivedROI,
    already_current,
    archived_roi_absolute,
    inventory_empty_starseg,
    process,
    resolve_live_weights,
    starseg_is_empty,
)
from startracked_loop_recorder.config import atomic_json_write
from startracked_loop_recorder.starseg import FILTER_POLICY, filter_detections_absolute


def _clip(root: Path, *, camera="A", trial="trial_101__session", stem="seg",
          absolute=None, frames=0, jsonl_rows=None, width=100, height=100):
    folder = root / camera / trial / stem
    folder.mkdir(parents=True)
    video = folder / f"{stem}.mp4"
    video.write_bytes(b"x")
    if absolute is None:
        absolute = [[0, 0], [50, 0], [50, 50], [0, 50]]
    metadata = {
        "camera": camera,
        "width": width,
        "height": height,
        "starseg_roi": {
            "normalized": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]],
            "absolute": absolute,
        },
    }
    atomic_json_write(video.with_suffix(".json"), metadata)
    summary = folder / f"{stem}_starseg_summary.json"
    atomic_json_write(summary, {"frames": frames, "model": str(root / "live.pt")})
    jsonl = folder / f"{stem}_starseg_frames.jsonl"
    if jsonl_rows is None:
        jsonl.write_text("" if frames == 0 else json.dumps({"frame_index": 0}) + "\n")
    else:
        jsonl.write_text(jsonl_rows)
    return video


def test_archived_roi_rejects_aquaria_and_unit_square():
    with pytest.raises(MissingArchivedROI, match="absolute"):
        archived_roi_absolute({"aquaria_corners": {"normalized": [[0, 0], [1, 0], [1, 1], [0, 1]]}})
    with pytest.raises(MissingArchivedROI, match="absolute"):
        archived_roi_absolute({"starseg_roi": {"normalized": [[0, 0], [1, 0], [1, 1], [0, 1]]}})


def test_filter_uses_stored_absolute_not_unit_square():
    detections = [
        {"class_id": 0, "class_name": "star", "confidence": 0.9,
         "box_xyxy": [70, 40, 80, 50], "segment_xy": []},
    ]
    left_half = [[0, 0], [50, 0], [50, 100], [0, 100]]
    assert filter_detections_absolute(detections, left_half) == []


def test_already_current_requires_nonempty_matching_model(tmp_path):
    model = tmp_path / "live.pt"
    model.write_bytes(b"w")
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"frames": 0, "model": str(model)}))
    labeled = tmp_path / "labeled.json"
    labeled.write_text(json.dumps({"frames": 12, "model": str(model)}))
    assert already_current(empty, model) is False
    assert already_current(labeled, model) is True


def test_inventory_skips_no_roi_and_nonempty(tmp_path):
    empty = _clip(tmp_path, stem="empty", frames=0)
    labeled = _clip(tmp_path, camera="B", stem="labeled", frames=8)
    missing = _clip(tmp_path, camera="C", stem="noroI", frames=0, absolute=[])
    # overwrite missing ROI
    atomic_json_write(missing.with_suffix(".json"), {"width": 100, "height": 100, "starseg_roi": {}})
    eligible, skipped = inventory_empty_starseg(tmp_path, set("ABCD"))
    keys = {clip.key for clip in eligible}
    assert f"A:{empty.stem}" in keys
    assert f"B:{labeled.stem}" not in keys
    statuses = {item.status for item in skipped}
    assert "nonempty_starseg" in statuses
    assert "missing_roi" in statuses


def test_process_refuses_full_frame_fallback(tmp_path, monkeypatch):
    folder = tmp_path / "A" / "trial" / "segment"
    folder.mkdir(parents=True)
    video = folder / "segment.mp4"
    video.write_bytes(b"x")
    atomic_json_write(video.with_suffix(".json"), {
        "camera": "A",
        "width": 100,
        "height": 100,
        "aquaria_corners": {"normalized": [[0, 0], [1, 0], [1, 1], [0, 1]]},
    })
    monkeypatch.setattr(
        "startracked_loop_recorder.backcast.detections_from_result",
        lambda _: [{"class_id": 0, "class_name": "star", "confidence": 0.9,
                    "box_xyxy": [10, 10, 20, 20], "segment_xy": []}],
    )
    model = SimpleNamespace(predict=lambda **_: iter([SimpleNamespace(orig_shape=(100, 100))]))
    args = SimpleNamespace(conf=0.25, imgsz=1024, device="cpu", model=tmp_path / "model.pt", frame_delay=0)
    with pytest.raises(MissingArchivedROI):
        process(video, model, args)
    assert not video.with_name("segment_starseg_frames.jsonl").exists()


def test_process_keeps_archived_absolute(tmp_path, monkeypatch):
    folder = tmp_path / "A" / "trial" / "segment"
    folder.mkdir(parents=True)
    video = folder / "segment.mp4"
    video.write_bytes(b"x")
    absolute = [[0, 0], [50, 0], [50, 50], [0, 50]]
    atomic_json_write(video.with_suffix(".json"), {
        "camera": "A",
        "width": 100,
        "height": 100,
        "starseg_roi": {
            "normalized": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "absolute": absolute,
        },
    })
    raw = [
        {"class_id": 0, "class_name": "star", "confidence": 0.9,
         "box_xyxy": [10, 10, 20, 20], "segment_xy": []},
        {"class_id": 0, "class_name": "star", "confidence": 0.8,
         "box_xyxy": [80, 80, 90, 90], "segment_xy": []},
    ]
    monkeypatch.setattr("startracked_loop_recorder.backcast.detections_from_result", lambda _: raw)
    model = SimpleNamespace(predict=lambda **_: iter([SimpleNamespace(orig_shape=(100, 100))]))
    args = SimpleNamespace(conf=0.25, imgsz=1024, device="cpu", model=tmp_path / "model.pt", frame_delay=0)
    process(video, model, args)
    row = json.loads(video.with_name("segment_starseg_frames.jsonl").read_text())
    assert row["roi"]["absolute"] == [[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]]
    assert row["roi"]["normalized"] == [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert len(row["detections"]) == 1
    assert row["roi"]["filter_policy"] == FILTER_POLICY


def test_starseg_is_empty_treats_nonempty_jsonl_as_labeled(tmp_path):
    frames = tmp_path / "frames.jsonl"
    summary = tmp_path / "summary.json"
    frames.write_text('{"frame_index":0}\n')
    summary.write_text('{"frames":0}')
    assert starseg_is_empty(frames, summary) is False


def test_resolve_live_weights_prefers_status_then_fallback(tmp_path, monkeypatch):
    live = tmp_path / "live.pt"
    live.write_bytes(b"w")
    fallback = tmp_path / "fallback.pt"
    fallback.write_bytes(b"f")

    def fake_urlopen(_url, timeout=0.4):
        class Response:
            def read(self):
                return json.dumps({"starseg": {"active_model": str(live)}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return Response()

    monkeypatch.setattr("startracked_loop_recorder.backcast.urllib.request.urlopen", fake_urlopen)
    assert resolve_live_weights("http://127.0.0.1:8000/status.json", fallback) == live.resolve()

    def down(_url, timeout=0.4):
        raise OSError("down")

    monkeypatch.setattr("startracked_loop_recorder.backcast.urllib.request.urlopen", down)
    assert resolve_live_weights("http://127.0.0.1:8000/status.json", fallback) == fallback.resolve()
