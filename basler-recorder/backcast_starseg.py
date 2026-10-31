#!/usr/bin/env python3
"""Back-cast a new StarSeg/YOLO model over previously finalized recordings.

This is a fully standalone background process. It does NOT import or talk to the
live recorder; it only reads the model weights and rewrites the per-video StarSeg
sidecar files in place. Running it never stops or interrupts recording.

For each finalized video folder it:
  1. decodes frames from `<stem>.mp4` (the only surviving frame source, since the
     per-frame JPEG staging is deleted at finalize),
  2. runs the new model on every frame (one frame at a time -- batch of 1),
  3. overwrites `<stem>_starseg_frames.jsonl` and `<stem>_starseg_summary.json`
     using the exact same schema, and refreshes the `starseg_summary` block inside
     the `<stem>.json` sidecar (adding a small `starseg_backcast` provenance marker).

Design goals (recording runs 24/7 for weeks):
  - Gentle on resources: streaming batch-of-1 inference, lowest CPU priority
    (`os.nice(19)`), capped torch CPU threads, and optional per-frame / per-video
    sleeps so the live StarSeg worker keeps getting GPU time.
  - Crash/restart safe: writes go to a temp file in the same directory and are
    atomically `os.replace`d over the originals, so a reader (or a kill -9) never
    sees a half-written file. Original file mode is preserved.
  - Idempotent & resumable: a video whose summary already records the target model
    is skipped (unless `--force`), so you can stop and re-run at any time.

Examples:
  # Dry run: show what would be processed, newest data unaffected
  ./backcast_starseg.py --dry-run

  # Run gently in the background alongside live recording
  nohup ./backcast_starseg.py >> /tmp/basler_loop_recorder_state/backcast.log 2>&1 &

  # Re-do everything (e.g. after another model change)
  ./backcast_starseg.py --force
"""
import argparse
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(os.environ.get("BASLER_RECORD_ROOT", "~/Videos/Basler Loop Recordings")).expanduser()
DEFAULT_MODEL = Path(os.environ.get("BASLER_STARSEG_MODEL", str(PROJECT_DIR / "models" / "best.pt"))).expanduser()
DEFAULT_CONF = float(os.environ.get("BASLER_STARSEG_CONF", "0.25"))
DEFAULT_IMGSZ = int(os.environ.get("BASLER_STARSEG_IMGSZ", "1024"))
SKIP_DIRS = {"_active_segments", "preview"}

_stop = False


def request_stop(signum, frame):
    global _stop
    _stop = True
    log("stop requested; finishing current video then exiting")


def iso(ts):
    return time.strftime("%Y%m%dT%H%M%S", time.localtime(ts)) + f"{int((ts % 1) * 1000):03d}"


def log(msg):
    print(f"[{iso(time.time())}] {msg}", flush=True)


def atomic_write(path, data, mode=None):
    """Write text to path atomically, preserving the original file mode if present."""
    path = Path(path)
    if mode is None and path.exists():
        try:
            mode = path.stat().st_mode & 0o777
        except OSError:
            mode = None
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def find_videos(root, cameras=None):
    """Yield finalized video folders as (mp4_path, stem). Skips active staging."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not name.endswith(".mp4"):
                continue
            mp4 = Path(dirpath) / name
            stem = name[:-4]
            # A finalized video folder is named exactly after its stem.
            if mp4.parent.name != stem:
                continue
            if cameras:
                cam = camera_from_stem(stem)
                if cam not in cameras:
                    continue
            yield mp4, stem


def camera_from_stem(stem):
    marker = "_camera-"
    if marker in stem:
        rest = stem.split(marker, 1)[1]
        return rest.split("_", 1)[0]
    return ""


def load_old_frame_meta(jsonl_path):
    """Return {frame_index: {camera, frame_ts, frame_file}} from an existing jsonl.

    Only the fields we want to preserve are kept (not detections), to bound memory.
    """
    out = {}
    p = Path(jsonl_path)
    if not p.exists():
        return out
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                idx = rec.get("frame_index")
                if idx is None:
                    continue
                out[int(idx)] = {
                    "camera": rec.get("camera", ""),
                    "frame_ts": rec.get("frame_ts"),
                    "frame_file": rec.get("frame_file"),
                }
    except OSError:
        pass
    return out


def build_detections(result, names):
    boxes = result.boxes
    masks = result.masks
    detections = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        clss = [int(x) for x in boxes.cls.cpu().tolist()]
        polys = masks.xy if masks is not None else []
        for i, cls_id in enumerate(clss):
            seg = []
            if i < len(polys):
                seg = [[round(float(x), 2), round(float(y), 2)] for x, y in polys[i].tolist()]
            detections.append({
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": round(float(confs[i]), 6),
                "box_xyxy": [round(float(v), 2) for v in xyxy[i]],
                "segment_xy": seg,
            })
    return detections


def process_video(model, names, model_str, device_str, conf, imgsz, mp4, stem, args):
    video_dir = mp4.parent
    frames_jsonl = video_dir / f"{stem}_starseg_frames.jsonl"
    summary_json = video_dir / f"{stem}_starseg_summary.json"
    sidecar_json = video_dir / f"{stem}.json"

    if not args.force and summary_json.exists():
        try:
            existing = json.loads(summary_json.read_text())
            if existing.get("model") == model_str:
                return "skip-done"
        except Exception:
            pass

    age = time.time() - mp4.stat().st_mtime
    if age < args.min_age:
        return "skip-fresh"

    if args.dry_run:
        return "would-process"

    old_meta = load_old_frame_meta(frames_jsonl)
    sidecar = {}
    if sidecar_json.exists():
        try:
            sidecar = json.loads(sidecar_json.read_text())
        except Exception:
            sidecar = {}
    cam_fallback = camera_from_stem(stem) or sidecar.get("camera", "")
    start_ts = sidecar.get("start_ts")
    fps = sidecar.get("fps") or 1.0
    try:
        fps = float(fps) or 1.0
    except (TypeError, ValueError):
        fps = 1.0

    summary = {"frames": 0, "detections": 0, "classes": {}}
    fd, tmp = tempfile.mkstemp(dir=str(video_dir), prefix="." + frames_jsonl.name + ".", suffix=".tmp")
    processed = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            results = model.predict(source=str(mp4), stream=True, imgsz=imgsz, conf=conf, verbose=False)
            for idx, result in enumerate(results):
                if _stop:
                    raise KeyboardInterrupt()
                detections = build_detections(result, names)
                h, w = result.orig_shape[:2] if getattr(result, "orig_shape", None) else (0, 0)
                meta = old_meta.get(idx, {})
                cam = meta.get("camera") or cam_fallback
                frame_file = meta.get("frame_file") or f"{idx:08d}.jpg"
                frame_ts = meta.get("frame_ts")
                if frame_ts is None and start_ts is not None:
                    frame_ts = float(start_ts) + (idx / fps)
                rec = {
                    "camera": cam,
                    "frame_index": idx,
                    "frame_ts": frame_ts if frame_ts is not None else 0,
                    "frame_file": frame_file,
                    "model": model_str,
                    "device": device_str,
                    "imgsz": imgsz,
                    "confidence_threshold": conf,
                    "width": int(w),
                    "height": int(h),
                    "detections": detections,
                }
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                summary["frames"] += 1
                for det in detections:
                    summary["detections"] += 1
                    name = det["class_name"]
                    summary["classes"][name] = summary["classes"].get(name, 0) + 1
                processed += 1
                if args.frame_delay > 0:
                    time.sleep(args.frame_delay)
            out.flush()
            os.fsync(out.fileno())
        mode = frames_jsonl.stat().st_mode & 0o777 if frames_jsonl.exists() else 0o664
        os.chmod(tmp, mode)
        os.replace(tmp, frames_jsonl)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    summary.update({
        "model": model_str,
        "confidence_threshold": conf,
        "imgsz": imgsz,
        "frames_jsonl": str(frames_jsonl),
    })
    atomic_write(summary_json, json.dumps(summary, indent=2))

    if sidecar:
        previous_model = (sidecar.get("starseg_summary") or {}).get("model")
        sidecar["starseg_summary"] = summary
        sidecar["starseg_frames_jsonl"] = str(frames_jsonl)
        sidecar["starseg_summary_json"] = str(summary_json)
        sidecar["starseg_backcast"] = {
            "model": model_str,
            "previous_model": previous_model,
            "confidence_threshold": conf,
            "imgsz": imgsz,
            "device": device_str,
            "ran_iso": iso(time.time()),
            "frames": summary["frames"],
        }
        atomic_write(sidecar_json, json.dumps(sidecar, indent=2))

    return f"ok ({processed} frames, {summary['detections']} detections)"


def load_state(path):
    if not path:
        return {}
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_state(path, state):
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(state, indent=2))
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description="Back-cast a new StarSeg model over old recordings.")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="recording root to scan")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="path to the .pt model to apply")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="inference image size")
    ap.add_argument("--device", default="auto", help="'auto', 'cpu', or a CUDA index like '0'")
    ap.add_argument("--cameras", default="", help="comma-separated camera filter, e.g. A,B,C")
    ap.add_argument("--frame-delay", type=float, default=0.1, help="seconds to sleep after each frame (gentle GPU sharing)")
    ap.add_argument("--video-delay", type=float, default=2.0, help="seconds to sleep between videos")
    ap.add_argument("--threads", type=int, default=2, help="cap torch CPU threads")
    ap.add_argument("--nice", type=int, default=19, help="process niceness (0-19, higher = lower priority)")
    ap.add_argument("--min-age", type=float, default=120.0, help="skip mp4s modified within this many seconds (avoid racing finalize)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N videos (0 = no limit)")
    ap.add_argument("--force", action="store_true", help="reprocess even if already at the target model")
    ap.add_argument("--dry-run", action="store_true", help="list what would be processed and exit")
    ap.add_argument("--state-file", default=str(Path("~/.config/basler-loop-recorder/backcast_state.json").expanduser()), help="progress/state file (use '' to disable)")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        os.nice(args.nice)
    except OSError:
        pass

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        log(f"ERROR: model not found: {model_path}")
        return 2
    model_str = str(model_path)
    cameras = {c.strip() for c in args.cameras.split(",") if c.strip()} or None

    log(f"scanning {args.root}")
    videos = sorted(find_videos(args.root, cameras), key=lambda t: str(t[0]))
    log(f"found {len(videos)} finalized videos")

    if args.dry_run:
        # Avoid loading torch in dry-run; just classify each video cheaply.
        counts = {"would-process": 0, "skip-done": 0, "skip-fresh": 0}
        for mp4, stem in videos:
            summary_json = mp4.parent / f"{stem}_starseg_summary.json"
            status = "would-process"
            if not args.force and summary_json.exists():
                try:
                    if json.loads(summary_json.read_text()).get("model") == model_str:
                        status = "skip-done"
                except Exception:
                    pass
            if status == "would-process" and (time.time() - mp4.stat().st_mtime) < args.min_age:
                status = "skip-fresh"
            counts[status] += 1
        log(f"dry-run: {counts}")
        return 0

    import torch
    from ultralytics import YOLO
    try:
        torch.set_num_threads(max(1, args.threads))
    except Exception:
        pass
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = int(args.device)
    device_str = str(device)
    log(f"loading model {model_str} on device {device_str}")
    model = YOLO(model_str)
    model.to(device)
    names = {int(k): v for k, v in model.names.items()}
    log(f"model task={getattr(model, 'task', '?')} names={names}")

    state = load_state(args.state_file)
    state.setdefault("model", model_str)
    if state.get("model") != model_str:
        state = {"model": model_str}
    state.setdefault("processed", 0)
    state.setdefault("errors", 0)
    state.setdefault("done", [])
    done_set = set(state["done"])

    started = time.time()
    n_ok = n_skip = n_err = 0
    for mp4, stem in videos:
        if _stop:
            break
        if args.limit and n_ok >= args.limit:
            log(f"reached --limit {args.limit}")
            break
        key = str(mp4)
        try:
            result = process_video(model, names, model_str, device_str, args.conf, args.imgsz, mp4, stem, args)
        except KeyboardInterrupt:
            log(f"interrupted during {stem}")
            break
        except Exception as e:
            n_err += 1
            state["errors"] = state.get("errors", 0) + 1
            log(f"ERROR {stem}: {e!r}")
            save_state(args.state_file, state)
            continue
        if result.startswith("ok"):
            n_ok += 1
            state["processed"] = state.get("processed", 0) + 1
            if key not in done_set:
                done_set.add(key)
                state["done"].append(key)
            log(f"[{n_ok}] {stem}: {result}")
            if n_ok % 20 == 0:
                save_state(args.state_file, state)
            if args.video_delay > 0:
                time.sleep(args.video_delay)
        else:
            n_skip += 1

    save_state(args.state_file, state)
    elapsed = time.time() - started
    log(f"done: processed={n_ok} skipped={n_skip} errors={n_err} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
