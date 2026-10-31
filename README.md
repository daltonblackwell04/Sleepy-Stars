# Sleepy Stars

Behavioral quiescence and sleep-like recovery in sunflower stars (*Pycnopodia helianthoides*).

This repository holds the software behind a laboratory study of sleep-like quiescence in sunflower stars: the multi-camera recorder used for the multi-day trials, a live logger for the arousal ("poke") trials, a tool for scoring behavioral states from the recordings, and a closed-loop system in which a camera-based segmentation model drives peristaltic pumps through an Arduino.

## What's here

| Folder | What it does |
|---|---|
| [`poke-logger/`](poke-logger/) | Browser-based logger for live poke trials. One keypress per event (entry, first poke, first movement, movement away, exit), the star's state before poking (quiescent / alert / moving), a poke timer, per-camera rest periods, an audit trail of every edit, and a live report of response times. Python standard library only. |
| [`annotation-interface/`](annotation-interface/) | Web app for scoring the three behavioral states (quiescent, still alert, moving), plus poking and feeding, from timelapses of the multi-day recordings. It produces time-stamped ethograms for each tank and compares tanks and trials on a shared time axis. Python standard library + ffmpeg. |
| [`basler-recorder/`](basler-recorder/) | The 24/7 three-camera Basler recorder used for the multi-day trials. It writes 10-minute H.264 segments with live star segmentation, self-describing file names, JSON sidecars (timing, camera, aquarium geometry, model provenance), throttled atomic backup to a second drive, and resumable model back-casting. |
| [`pump-control/`](pump-control/) | Camera → segmentation → Arduino → peristaltic pumps. Arduino firmware, the pump service that owns the serial port, and the multi-camera recorder that runs the segmentation model and decides when each tank's pump runs. |
| [`sample-data/`](sample-data/) | A small anonymized excerpt of the poke-trial log, with column documentation. |

## How the closed loop works

```
 camera A-D frames
        │
        ▼
 StarSeg (YOLO segmentation, 4 classes)          recorder/src/startracked_loop_recorder/starseg.py
        │  sunflower-star masks, filtered to each camera's ROI polygon
        ▼
 virtual pressure plate + hysteresis             recorder/src/startracked_loop_recorder/hue.py
        │  "is a star on this camera's trigger zone?"  → per-camera pump demand
        ▼
 4-bit pump mask (A=2, B=1, C=8, D=4)             injector_pump.py  /  arduino_pump.py
        │  HTTP heartbeat to the local pump service, or direct serial
        ▼
 loop_injector service (owns the serial port)     loop_injector/pump_backend.py
        │  "PUMPS:<0-15>\n" at 115200 baud, resent every 150 ms
        ▼
 Arduino Mega + 16x2 LCD                          firmware/MultiCameraPumpLCD/MultiCameraPumpLCD.ino
        │  pins 50-53 → pump drivers, LCD shows P1-P4 ON/OFF
        ▼
 peristaltic pumps (one per tank)
```

The firmware fails closed: if no valid command arrives within 500 ms, or it receives a malformed or oversized line, it switches every pump off and stays latched off until the host sends an explicit `PUMPS:0` to re-arm. Every occupancy change, pump command, acknowledgement and error is logged, and each finalized video segment carries a `_pressure_plate_events.jsonl` sidecar, so pump activity can be lined up with the footage afterwards. Each plate can also switch an assigned Philips Hue smart plug, and an optional hourly schedule pulses the Hue outputs only (never the Arduino pumps).

Details, wiring and protocol: [`pump-control/README.md`](pump-control/README.md).

## Running the tests

```bash
# Poke logger (standard library only)
cd poke-logger && python3 -m unittest -v test_poke_logger

# Ethogram annotation interface (standard library only)
cd annotation-interface && python3 -m unittest -v test_annotation_store

# Pump service (standard library only)
cd pump-control && python3 -m unittest discover -s loop_injector/tests -t .

# Recorder + segmentation-to-pump bridge (Python 3.11-3.12)
cd pump-control/recorder
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
```

The recorder tests use a fake serial port, a fake GPU backend and in-process HTTP servers, so they run without cameras, an Arduino, a Hue bridge or the model weights.

## License

MIT. See [LICENSE](LICENSE).
