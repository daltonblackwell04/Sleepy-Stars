from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import atomic_json_write
from .starseg import FILTER_POLICY, detections_from_result, filter_detections_absolute
from .storage import derive_starseg_summary

STOP = False
DEFAULT_ARCHIVE_ROOT = Path.home() / "Videos" / "Startracked Loop Recordings"
DEFAULT_LIVE_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "target_oversample" / "best.pt"
DEFAULT_STATUS_URL = "http://127.0.0.1:8000/status.json"


class MissingArchivedROI(ValueError):
    """Finalized metadata does not contain a usable starseg_roi.absolute."""


@dataclass(frozen=True)
class EmptyStarsegClip:
    key: str
    camera: str
    trial: str
    video: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EmptyStarsegIssue:
    key: str
    video: str
    status: str
    error: str


def canonical_videos(root: Path, cameras: set[str]) -> list[Path]:
    return sorted(
        path for camera in cameras for path in (root / camera).glob("*recording_start-*/*/*.mp4")
        if "_active_segments" not in path.parts and "preview" not in path.parts
    )


def archive_videos(root: Path, cameras: set[str]) -> list[Path]:
    videos: list[Path] = []
    for camera in sorted(cameras):
        camera_root = root / camera
        if not camera_root.is_dir():
            continue
        for video in camera_root.glob("*/*/*.mp4"):
            if video.name != f"{video.parent.name}.mp4":
                continue
            if "_active_segments" in video.parts or "preview" in video.parts:
                continue
            videos.append(video)
    videos.sort()
    return videos


def target_paths(video: Path) -> tuple[Path, Path, Path]:
    stem = video.stem
    return (
        video.with_name(f"{stem}_starseg_frames.jsonl"),
        video.with_name(f"{stem}_starseg_summary.json"),
        video.with_suffix(".json"),
    )


def already_current(summary: Path, model: Path) -> bool:
    if not summary.exists():
        return False
    try:
        payload = json.loads(summary.read_text())
        if int(payload.get("frames") or 0) <= 0:
            return False
        return Path(payload["model"]).resolve() == model.resolve()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def starseg_is_empty(frames_path: Path, summary_path: Path) -> bool:
    if summary_path.is_file():
        try:
            if int(json.loads(summary_path.read_text()).get("frames") or 0) > 0:
                return False
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    if not frames_path.is_file():
        return True
    try:
        with frames_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return False
    except OSError:
        return True
    return True


def _finite_xy(value: Any) -> list[float] | None:
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x, y]


def archived_roi_absolute(
    metadata: dict[str, Any],
    width: int | None = None,
    height: int | None = None,
) -> list[list[float]]:
    roi = metadata.get("starseg_roi")
    if not isinstance(roi, dict) or "absolute" not in roi:
        raise MissingArchivedROI("metadata is missing starseg_roi.absolute")
    raw = roi.get("absolute")
    if not isinstance(raw, list):
        raise MissingArchivedROI("metadata is missing starseg_roi.absolute")
    vertices: list[list[float]] = []
    in_frame = 0
    for item in raw:
        point = _finite_xy(item)
        if point is None:
            continue
        vertices.append(point)
        if width and height and 0 <= point[0] <= width and 0 <= point[1] <= height:
            in_frame += 1
    if len(vertices) < 3:
        raise MissingArchivedROI("starseg_roi.absolute requires at least 3 finite vertices")
    if width and height and in_frame < 3:
        raise MissingArchivedROI(
            "starseg_roi.absolute requires at least 3 finite in-frame vertices"
        )
    return vertices


def inventory_empty_starseg(
    root: Path,
    cameras: set[str],
) -> tuple[list[EmptyStarsegClip], list[EmptyStarsegIssue]]:
    eligible: list[EmptyStarsegClip] = []
    skipped: list[EmptyStarsegIssue] = []
    for video in archive_videos(root, cameras):
        camera = video.parents[2].name
        stem = video.stem
        key = f"{camera}:{stem}"
        trial = video.parents[1].name
        metadata_path = video.with_suffix(".json")
        frames_path, summary_path, _ = target_paths(video)
        try:
            metadata = json.loads(metadata_path.read_text())
        except FileNotFoundError:
            skipped.append(EmptyStarsegIssue(key, str(video), "missing_metadata", "metadata not found"))
            continue
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append(EmptyStarsegIssue(key, str(video), "malformed_metadata", str(exc)))
            continue
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        try:
            archived_roi_absolute(metadata, width, height)
        except MissingArchivedROI as exc:
            skipped.append(EmptyStarsegIssue(key, str(video), "missing_roi", str(exc)))
            continue
        if not starseg_is_empty(frames_path, summary_path):
            skipped.append(
                EmptyStarsegIssue(
                    key, str(video), "nonempty_starseg", "refusing to overwrite nonempty StarSeg"
                )
            )
            continue
        eligible.append(EmptyStarsegClip(key, camera, trial, video, metadata_path, metadata))
    return eligible, skipped


def resolve_live_weights(status_url: str, fallback: Path) -> Path:
    try:
        with urllib.request.urlopen(status_url, timeout=0.4) as response:
            payload = json.loads(response.read().decode())
        starseg = payload.get("starseg") or {}
        for key in ("active_model", "desired_model"):
            raw = starseg.get(key)
            if not raw:
                continue
            path = Path(str(raw))
            if path.is_file():
                return path.resolve()
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError, TypeError):
        pass
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"live StarSeg weights unavailable: {fallback}")


def apply_os_isolation(*, nice: int = 10, ionice_idle: bool = True) -> dict[str, Any]:
    applied: dict[str, Any] = {"nice": None, "ionice": False}
    try:
        os.nice(nice)
        applied["nice"] = os.nice(0)
    except OSError:
        pass
    if ionice_idle:
        result = subprocess.run(
            ["ionice", "-c3", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
        )
        applied["ionice"] = result.returncode == 0
    return applied


def process(
    video: Path,
    model,
    args,
    on_frame: Callable[[], None] | None = None,
) -> None:
    frames_path, summary_path, metadata_path = target_paths(video)
    old_meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    meta_w, meta_h = int(old_meta.get("width") or 0), int(old_meta.get("height") or 0)
    if meta_w and meta_h:
        archived_roi_absolute(old_meta, meta_w, meta_h)
    else:
        archived_roi_absolute(old_meta)
    temporary = frames_path.with_name(f".{frames_path.name}.{os.getpid()}.part")
    count = 0
    results = model.predict(
        source=str(video), stream=True, conf=args.conf, imgsz=args.imgsz,
        device=args.device, verbose=False,
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for result in results:
                if STOP:
                    raise KeyboardInterrupt
                if on_frame is not None:
                    on_frame()
                timestamps = old_meta.get("frame_timestamps", [])
                frame_ts = timestamps[count] if count < len(timestamps) else None
                width, height = int(result.orig_shape[1]), int(result.orig_shape[0])
                polygon = archived_roi_absolute(old_meta, width, height)
                stored = old_meta.get("starseg_roi") if isinstance(old_meta.get("starseg_roi"), dict) else {}
                stored_normalized = stored.get("normalized") if isinstance(stored.get("normalized"), list) else []
                all_detections = detections_from_result(result)
                kept = filter_detections_absolute(all_detections, polygon)
                row = {
                    "camera": old_meta.get("camera", video.parents[2].name),
                    "frame_index": count, "frame_ts": frame_ts,
                    "frame_file": f"{count:08d}.jpg", "model": str(args.model.resolve()),
                    "device": args.device, "imgsz": args.imgsz,
                    "confidence_threshold": args.conf,
                    "width": width, "height": height,
                    "trial_name": old_meta.get("trial_name", old_meta.get("heading", "")),
                    "roi": {
                        "normalized": stored_normalized,
                        "absolute": polygon,
                        "filter_policy": FILTER_POLICY,
                        "detections_before_filter": len(all_detections),
                        "detections_after_filter": len(kept),
                    },
                    "detections": kept,
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                count += 1
                if args.frame_delay:
                    time.sleep(args.frame_delay)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, frames_path)
    except Exception:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        raise
    summary = derive_starseg_summary(frames_path, count)
    atomic_json_write(summary_path, summary)
    old_meta.update({
        "starseg_frames_jsonl": str(frames_path),
        "starseg_summary_json": str(summary_path), "starseg_summary": summary,
        "starseg_backcast": {
            "completed_at": time.time(), "model": str(args.model.resolve()),
            "confidence_threshold": args.conf, "imgsz": args.imgsz, "device": args.device,
            "filter_policy": FILTER_POLICY,
        },
    })
    atomic_json_write(metadata_path, old_meta)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resumable StarSeg backcast for finalized segments")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cameras", default="A,B,C,D")
    p.add_argument("--frame-delay", type=float, default=0)
    p.add_argument("--video-delay", type=float, default=0)
    p.add_argument("--threads", type=int, default=1, help="Reserved; inference remains batch-size one")
    p.add_argument("--nice", type=int, default=10)
    p.add_argument("--min-age", type=float, default=300, help="Minimum video age in seconds")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--state-file", type=Path)
    return p


def main(argv=None) -> int:
    global STOP
    args = parser().parse_args(argv)
    if not args.root.is_absolute() or not args.model.is_absolute():
        raise SystemExit("--root and --model must be absolute")
    cameras = {c.strip() for c in args.cameras.split(",")}
    if not cameras <= {"A", "B", "C", "D"}:
        raise SystemExit("--cameras must contain only A,B,C,D")
    os.nice(max(0, args.nice))
    completed = set()
    if args.state_file and args.state_file.exists():
        completed = set(json.loads(args.state_file.read_text()).get("completed", []))
    candidates = []
    now = time.time()
    for video in canonical_videos(args.root, cameras):
        frames_path, summary_path, _ = target_paths(video)
        if str(video) in completed or now - video.stat().st_mtime < args.min_age:
            continue
        if not args.force and already_current(summary_path, args.model):
            continue
        candidates.append(video)
    if args.limit is not None:
        candidates = candidates[:args.limit]
    for video in candidates:
        print(video)
    if args.dry_run:
        return 0
    if not args.model.is_file():
        raise SystemExit(f"model unavailable: {args.model}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed; install the starseg extra") from exc
    model = YOLO(str(args.model))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
    for video in candidates:
        if STOP:
            break
        try:
            process(video, model, args)
        except KeyboardInterrupt:
            break
        completed.add(str(video))
        if args.state_file:
            atomic_json_write(args.state_file, {"schema_version": 1, "completed": sorted(completed)})
        if args.video_delay:
            time.sleep(args.video_delay)
    return 0
