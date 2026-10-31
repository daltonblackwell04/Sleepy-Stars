# Basler Annotation Interface: ethogram scoring from video

A local web app for scoring sunflower-star behavior from the multi-day tank recordings. Each camera-day becomes a smooth timelapse (1 real hour = 1 second of playback). The observer marks state changes with single keys while it plays, and the app stores the result as a time-stamped ethogram next to the original footage.

It only reads the recordings. It never controls cameras, never calls the recorder, and never modifies source MP4s.

## Scoring

| Key | Behavioral state |
|---|---|
| `q` | moving |
| `w` | still alert (SA) |
| `e` | quiescent |
| `r` | poking (stimulus being applied) |

| Key | Feeding state |
|---|---|
| `z` | eating |
| `x` | not eating |

Space plays/pauses, the left/right arrow keys step, and Ctrl+Z / Ctrl+Y undo and redo. A state holds until the next mark. Recording gaps (missing clips) are drawn explicitly in the timelapse and the ethogram rather than silently closed up.

## Views

- **Trial / date / camera selection**, including aggregate views of the first 72 h and the post-observation period.
- **C/A/B behavioral-state comparison:** the three tanks of a trial on one time axis. Treatment labels as in the code: C = nonstress, A = control, B = stress.
- **C/A/B feeding comparison**
- **Intertrial comparison:** the same camera across all trials or any two selected trials.
- State durations split by light/dark phase (per-cohort light regimes) and by disturbance phase.

## Output

Each save goes to the app's annotation store (one record per trial, date and camera, under the cache folder) and is mirrored as two sidecars into the camera's recording-session folder:

- `<clip>.annotations.json`: the marked events, with a revision number. A save based on a stale revision is refused (HTTP 409) instead of overwriting someone else's work. An unreadable file locks into read-only recovery mode (HTTP 423) and offers the replicas to restore from.
- `<clip>.ethogram.csv`: the ethogram resampled at 60 rows per playback second. Columns are `video_time_s, source_elapsed_s, wall_clock, in_gap, behavior, behavior_code, behavior_label, color, feeding, feeding_code, feeding_label, feeding_color`.

Writes are atomic. A third copy goes to a backup root (`BASLER_ANNOTATION_BACKUP`) if that folder exists. A failed replica never undoes the canonical save.

## Running

Needs Python 3.11+ (standard library only) and `ffmpeg`/`ffprobe`.

```bash
BASLER_RECORD_ROOT="$HOME/Videos/Basler Loop Recordings" python3 annotation_interface.py
# open http://127.0.0.1:8010/
```

| Variable | Default | Meaning |
|---|---|---|
| `BASLER_RECORD_ROOT` | `~/Videos/Basler Loop Recordings` | finalized recorder output, one folder per camera |
| `BASLER_ANNOTATION_PORT` | `8010` | HTTP port (bound to 127.0.0.1 only) |
| `BASLER_ANNOTATION_CACHE` | `~/.cache/basler-annotation-interface` | timelapse and annotation cache |
| `BASLER_ANNOTATION_BACKUP` | `~/Basler Annotation Backup/annotations` | optional backup replica (used only if it exists) |
| `BASLER_ANNOTATION_ENCODER` | `auto` | `libx264` to force CPU encoding |

`systemd/basler-annotation-interface.service` runs it as a user service. `open-basler-annotation-interface.sh` starts or restarts the service and opens the page.

## Feeding-centric frame extraction

`extract_feeding_centric_frames.py` builds a 100-image training/validation set from the human feeding ethograms. It picks frames scored *eating* that also have a sunflower-star segmentation detection, stratifies them across cameras, spreads them out in position and time, and crops a square twice the detection width around each star. `--plan-only` previews the selection without extracting anything.

## Tests

```bash
python3 -m unittest -v test_annotation_store
```

They cover atomic and revisioned saves, conflict and corrupt-file recovery, replica failures, the comparison caches, and the light/dark window statistics (gap exclusion, pooling by classified hours).
