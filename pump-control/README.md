# Pump control: camera → segmentation → Arduino → peristaltic pumps

Four tanks, four cameras (A–D), and one small peristaltic pump per tank. When the segmentation model sees a sunflower star on a camera's trigger zone (a "virtual pressure plate" drawn on the video), that tank's pump runs; when the star leaves, or anything in the chain goes stale, the pump stops.

```
pump-control/
├── firmware/MultiCameraPumpLCD/   Arduino sketch: serial protocol, watchdog, LCD
├── loop_injector/                 local pump service that owns the Arduino's serial port
│   ├── public/                    its browser UI (manual control, schedules, pump log)
│   └── tests/
├── recorder/                      multi-camera recorder + StarSeg segmentation + plate logic
│   ├── src/startracked_loop_recorder/
│   ├── tests/
│   └── models/target_oversample/manifest.json   model card (weights not included)
├── systemd/loop-injector.service  user unit for the pump service
└── launch-loop-injector.sh        starts the service and opens its UI
```

## 1. Segmentation (`recorder/src/startracked_loop_recorder/starseg.py`, `starseg_gpu.py`)

Live frames from each camera go to StarSeg, a YOLO segmentation model (1024 px input; classes `sunflower_star`, `other_sea_star`, `live_prey`, `dead_prey`). See `models/target_oversample/manifest.json` for training settings and validation metrics. Inference runs in a separate GPU worker process that the recorder can restart, with a CPU fallback.

Each detection's test point is the centroid of its segmentation mask (or the box center if there is no mask). Detections whose test point falls outside the camera's cyan ROI polygon are dropped.

## 2. Virtual pressure plates (`hue.py`)

Each camera has an amber "pressure plate" polygon drawn in the recorder's web UI. It must lie entirely inside the ROI, and an empty plate disables that camera (fail closed). A plate is occupied when a `sunflower_star` detection's test point is inside it or on its boundary. Occupancy is confirmed with hysteresis so a single flickering frame doesn't toggle the pump, and stale or archival inference results can never actuate anything.

Occupancy sets that camera's pump demand. The same demand can also switch an assigned Philips Hue smart plug. An optional hourly schedule pulses the Hue outputs for 60 s at the top of each hour when the plate is clear; the hourly pulse never turns on an Arduino pump.

## 3. Host → Arduino (`injector_pump.py`, `arduino_pump.py`, `loop_injector/`)

The four per-camera demands are combined into a 4-bit mask:

| Camera | Mask bit | Pump (LCD) | Arduino pin |
|---|---|---|---|
| B | 1 | P1 | 50 |
| A | 2 | P2 | 51 |
| D | 4 | P3 | 52 |
| C | 8 | P4 | 53 |

By default the recorder sends demand to the **loop_injector** service on `127.0.0.1:8081` (`injector_pump.py`). It holds a session lease and heartbeats every 0.5 s. The service is the only process that opens the serial port. It merges recorder demand with manual control and scheduled automations from its own UI, logs every pump on/off interval to SQLite (exportable as CSV), and provides an emergency stop. `arduino_pump.py` is the alternative direct-serial mode, with no separate service.

## 4. Firmware (`firmware/MultiCameraPumpLCD/MultiCameraPumpLCD.ino`)

Arduino Mega with a 16×2 HD44780 LCD (`LiquidCrystal(12, 11, 5, 4, 3, 2)`: RS, E, D4–D7). Pins 50–53 each switch one pump's driver stage. The LCD shows `P1:ON  P2:OFF` / `P3:OFF P4:OFF`.

Serial protocol, 115200 baud, one command per line:

| Host → Arduino | Meaning |
|---|---|
| `PUMPS:<0-15>\n` | set all four pumps at once from the mask (the host resends it every 150 ms) |

| Arduino → host | Meaning |
|---|---|
| `MULTICAM_PUMP_READY` | sketch booted. The host waits for this before sending anything, so the bootloader never receives commands. |
| `EVENT BOOT` | sent right after the ready marker |
| `EVENT REARMED` | `PUMPS:0` received while latched off; outputs may now be switched |
| `EVENT STATE <mask>` | outputs changed to `<mask>` |
| `EVENT WATCHDOG` | no valid command for 500 ms; all pumps off, latched |
| `EVENT REJECT MALFORMED` / `EVENT REJECT OVERFLOW` | bad or over-long line; all pumps off, latched |
| `EVENT IGNORED LATCHED` | non-zero mask received while latched off |

The firmware starts latched off. After a watchdog timeout or a rejected line it switches everything off and ignores non-zero masks until the host sends `PUMPS:0` again. If the computer, the service or the USB cable fails, the pumps stop within half a second.

To flash: open `firmware/MultiCameraPumpLCD/` in the Arduino IDE (2.x), board "Arduino Mega or Mega 2560", and upload. The only library is `LiquidCrystal`, which ships with the Arduino IDE (with arduino-cli: `arduino-cli lib install LiquidCrystal`). The sketch uses about 4.6 KB of flash and 257 bytes of RAM on a Mega.

## Running

```bash
# Pump service (standard library + pyserial)
cd pump-control
python3 -m loop_injector.service          # UI at http://127.0.0.1:8081

# Recorder (Python 3.11-3.12; see recorder/pyproject.toml)
cd recorder
python3 -m venv .venv && .venv/bin/pip install -e '.[starseg]'
.venv/bin/startracked-loop-recorder        # UI at http://127.0.0.1:8000
```

The Arduino is opened by its `/dev/serial/by-id/usb-Arduino...` path; set it for your board in `loop_injector/pump_backend.py` and `recorder/src/startracked_loop_recorder/config.py`. The segmentation weights (`best.pt`) aren't included. Place them at `recorder/models/target_oversample/best.pt`; the manifest records the expected SHA-256.

## Tests

```bash
cd pump-control && python3 -m unittest discover -s loop_injector/tests -t .
cd pump-control/recorder && .venv/bin/pip install -e '.[test]' && .venv/bin/python -m pytest
```

The tests cover the firmware contract (handshake, watchdog re-arm, malformed lines, every camera/mask combination), lease expiry and reconnect behavior, plate validation, hysteresis, and the hourly schedule. They use a fake serial port, a fake GPU backend and in-process HTTP servers, so they need no hardware.
