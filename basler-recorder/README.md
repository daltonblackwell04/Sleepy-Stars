# Basler loop recorder: three-camera 24/7 tank recording

The recorder used for the multi-day trials: three Basler USB cameras (A, B, C) over the tanks, recorded continuously as 10-minute H.264 segments. A sunflower-star segmentation model runs live on every camera, and each segment gets JSON sidecars describing exactly how, when and where it was recorded. A browser UI shows live previews with segmentation overlays, per-camera controls, and the recorder's health.

```
basler-recorder/
├── basler_loop_video_recorder.py   recorder + web UI (http://<host>:8000/)
├── backcast_starseg.py             re-run a newer model over finalized videos, in place
├── sync_starseg_to_backup.py       copy rewritten StarSeg sidecars to the backup drive
├── extract_starseg_food_frames.py  build food/prey training frames from annotated video
├── open-basler-loop-recorder.sh    start the recorder if needed and open the UI
├── run-backcast.sh                 background back-cast with logs and pid file
└── run-backcast-lacie.sh           bring the backup drive up to the new model
```

## How a recording is built

1. **Acquisition.** One thread per camera (`acq`) grabs frames through pypylon at the configured rate. Cameras are matched by serial number, and per-camera exposure, gain, white balance and frame-rate settings are applied and persisted.
2. **Staging.** Frames are written as JPEGs into `_active_segments/<camera>/…` and handed to the StarSeg GPU worker.
3. **Segmentation.** StarSeg (YOLO segmentation) runs on each frame. Detections and mask polygons go to `<video>_starseg_frames.jsonl`, and counts go to `<video>_starseg_summary.json`.
4. **Finalization.** A segment closes every `BASLER_SEGMENT_SECONDS` (default 600 s), or sooner if the recording session or save folder changes. Finalization waits for that segment's inference to drain, then encodes the frames to `<video>.mp4` with ffmpeg/x264. The file name itself records the trial heading, camera, start time, duration, frame rate, frame count and resolution. A `<video>.json` sidecar records the trial heading, camera letter and serial, session, start/end times, frame count, aquarium geometry, and links to the StarSeg files with the model and settings used. Only then are the staged JPEGs removed.
5. **Backup.** Only finalized files (never staging or previews) are queued to a second drive by a throttled copier (`copy_file_atomic`). Each file is written to a temp name, fsynced and atomically renamed, so the backup never holds half-copied files. Files already present with the same size are skipped. A backup root under `/media` or `/mnt` must actually be mounted, so an unplugged drive can't silently fill the system disk.

```
<record root>/C/recording_start-20260602T120000000/<video>/
    <video>.mp4
    <video>.json                    sidecar: timing, camera, geometry, provenance
    <video>_starseg_frames.jsonl    per-frame detections and polygons
    <video>_starseg_summary.json    per-segment counts
```

## Aquarium geometry

Each camera view has an **Edit corners** tool for placing the 8 corners of the tank, which seen from above is a truncated pyramid. Points 1–4 mark the top rim, clockwise from top-left. Points 5–8 mark the floor, in the same order. Recording can't start until all three cameras have their corners set. Every segment's sidecar stores the geometry it was recorded with, in normalized and pixel coordinates, so wall-versus-floor position can be computed later for any clip.

## Model rollover and back-casting

The desired model can be changed while recording. It takes effect at the next Stop → Start, so a running experiment never switches models mid-trial, and `/status.json` reports both the active and the pending model. `backcast_starseg.py` later re-runs a new model over finalized videos. It decodes frames from the MP4, atomically rewrites the StarSeg sidecars, and adds a `starseg_backcast` provenance block recording the new and previous model. It runs at `nice 19` with capped threads so it can share the machine with live recording. It is resumable and idempotent: videos already processed with the target model are skipped, as are videos still being finalized. `run-backcast-lacie.sh` waits for the primary back-cast, copies the rewritten sidecars to the backup drive (`sync_starseg_to_backup.py`), and back-casts only the videos that exist on the backup alone.

## Running

Requires a Python environment with `pypylon` (plus `torch` and `ultralytics` for StarSeg), `ffmpeg`, and Basler cameras.

```bash
export BASLER_SERIAL_A=<serial> BASLER_SERIAL_B=<serial> BASLER_SERIAL_C=<serial>
export BASLER_STARSEG_MODEL=/path/to/model.pt     # or BASLER_STARSEG_ENABLED=0
BASLER_PYTHON=/path/to/venv/bin/python ./open-basler-loop-recorder.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `BASLER_SERIAL_A/B/C` | placeholders | camera serial numbers for letters A, B, C |
| `BASLER_RECORD_ROOT` | `~/Videos/Basler Loop Recordings` | primary recording folder (also settable in the UI) |
| `BASLER_BACKUP_ROOT` | `/media/$USER/LaCie/Basler Loop Recordings` | backup copy destination |
| `BASLER_BACKUP_THROTTLE_MBPS` | `8` | backup copy rate limit |
| `BASLER_SEGMENT_SECONDS` | `600` | segment length |
| `BASLER_STREAM_FPS` | `1` | acquisition rate (frames/s) |
| `BASLER_X264_CRF` / `BASLER_X264_PRESET` | `34` / `medium` | encoder settings |
| `BASLER_STARSEG_ENABLED` / `_MODEL` / `_CONF` / `_IMGSZ` | on / `models/best.pt` / – / – | segmentation settings |
| `BASLER_STREAM_PORT` | `8000` | web UI port |

Settings changed in the UI persist in `~/.config/basler-loop-recorder/config.json`. Model weights aren't included in this repository.

Note: the web UI listens on all interfaces (`0.0.0.0`), so keep the recording computer on a trusted network.
