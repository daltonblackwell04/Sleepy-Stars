from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
USER_SCHEMA_VERSION = 7
CAMERAS = ("A", "B", "C", "D")
ARDUINO_SERIAL_PATH = (
    "/dev/serial/by-id/"
    "usb-Arduino__www.arduino.cc__0042_00000000000000000000-if00"
)
ARDUINO_PUMP_MAPPING = {"A": 2, "B": 1, "C": 8, "D": 4}
DEFAULT_CORNERS = [
    [0.08, 0.08], [0.92, 0.08], [0.92, 0.92], [0.08, 0.92],
]
DEFAULT_CONFIG_PATH = Path(
    os.getenv("BASLER_CONFIG", "~/.config/startracked-loop-recorder/config.json")
).expanduser()
DEFAULT_RECORD_ROOT = Path(
    os.getenv("BASLER_RECORD_ROOT", "~/Videos/Startracked Loop Recordings")
).expanduser()


class ConfigError(ValueError):
    pass


def default_deployment(project_root: Path) -> dict[str, Any]:
    base = "/dev/v4l/by-path"
    return {
        "schema_version": SCHEMA_VERSION,
        "cameras": {
            "A": {"path": f"{base}/pci-0000:21:00.0-usb-0:2.3:1.0-video-index0", "model": "Arducam UVC"},
            "B": {"path": f"{base}/pci-0000:21:00.0-usb-0:2.4:1.0-video-index0", "model": "Arducam UVC"},
            "C": {"path": f"{base}/pci-0000:30:00.3-usb-0:1.4.1:1.0-video-index0", "model": "Arducam UVC"},
            "D": {"path": f"{base}/pci-0000:30:00.3-usb-0:1.4.4:1.0-video-index0", "model": "Arducam UVC"},
        },
        "width": 1920,
        "height": 1080,
        "source_fourcc": "MJPG",
        "record_fps": float(os.getenv("BASLER_STREAM_FPS", "1")),
        "segment_seconds": float(os.getenv("BASLER_SEGMENT_SECONDS", "600")),
        "port": int(os.getenv("BASLER_STREAM_PORT", "8000")),
        "bind": os.getenv("BASLER_BIND", "127.0.0.1"),
        "jpeg_quality": int(os.getenv("BASLER_JPEG_QUALITY", "80")),
        "preview_jpeg_quality": int(os.getenv("BASLER_PREVIEW_JPEG_QUALITY", "45")),
        "x264_crf": int(os.getenv("BASLER_X264_CRF", "34")),
        "x264_preset": os.getenv("BASLER_X264_PRESET", "medium"),
        "delete_failed_segments": os.getenv("BASLER_DELETE_FAILED_SEGMENTS", "0") == "1",
        "project_root": str(project_root),
    }


def default_user_config() -> dict[str, Any]:
    return {
        "schema_version": USER_SCHEMA_VERSION,
        "record_root": str(DEFAULT_RECORD_ROOT),
        "trial_name": "Startracked Loop Recorder",
        "camera_controls": {
            c: {
                "fps": 1.0,
                "exposure_mode": "Auto",
                "exposure_us": 10000,
                "gain_auto": "Continuous",
                "gain": 0,
                "white_balance": "Continuous",
                "brightness": 0.5,
            }
            for c in CAMERAS
        },
        "starseg_display": {
            "enabled": False, "overlay": True, "confidence": 0.25,
            "imgsz": 1024, "mask_opacity": 0.35, "line_width": 2,
            "show_boxes": True, "show_masks": True, "show_labels": True,
        },
        "starseg_model": "",
        "backup": {"enabled": False, "root": "", "throttle_mbps": 8.0},
        "camera_polygons": {c: copy.deepcopy(DEFAULT_CORNERS) for c in CAMERAS},
        "pressure_plate_polygons": {c: [] for c in CAMERAS},
        # Conservative for the normal 1 Hz inference cadence: require two
        # observations to enter, allow two missed observations to leave.
        "pressure_plate": {
            "enabled": False,
            "bridge_address": "192.0.2.10",
            "app_key": "",
            "outputs": {c: "" for c in CAMERAS},
            "enter_seconds": 0.5,
            "exit_seconds": 1.5,
            "stale_seconds": 3.0,
        },
        "arduino_pump": {
            "enabled": True,
            "mode": "injector",
            "injector_url": "http://127.0.0.1:8081",
            "injector_heartbeat_seconds": 0.5,
            "injector_timeout_seconds": 0.5,
            "serial_path": ARDUINO_SERIAL_PATH,
            "baud": 115200,
            "heartbeat_seconds": 0.15,
            "reconnect_max_seconds": 5.0,
            "startup_timeout_seconds": 5.0,
            "mapping": copy.deepcopy(ARDUINO_PUMP_MAPPING),
        },
    }


def validate_deployment(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError("unsupported deployment schema")
    cameras = data.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != set(CAMERAS):
        raise ConfigError("deployment must map exactly cameras A-D")
    paths = []
    for camera in CAMERAS:
        path = cameras[camera].get("path", "")
        if not path.startswith("/dev/v4l/by-path/") or not path.endswith("video-index0"):
            raise ConfigError(f"camera {camera} must use a stable video-index0 by-path")
        paths.append(path)
    if len(set(paths)) != 4:
        raise ConfigError("camera paths must be unique")
    if not 0.1 <= float(data["record_fps"]) <= 120:
        raise ConfigError("record_fps must be 0.1-120")
    if float(data["segment_seconds"]) <= 0:
        raise ConfigError("segment_seconds must be positive")
    return data


def _orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d) -> bool:
    eps = 1e-12
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if (
        (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps)
        and (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps)
    ):
        return True
    def on_segment(p, q, r):
        return (
            abs(_orientation(p, q, r)) <= eps
            and min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps
            and min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps
        )
    return (
        on_segment(a, b, c) or on_segment(a, b, d)
        or on_segment(c, d, a) or on_segment(c, d, b)
    )


def validate_polygon(points: Any) -> list[list[float]]:
    if not isinstance(points, list) or not 3 <= len(points) <= 64:
        raise ConfigError("polygon must contain 3-64 vertices")
    if any(not isinstance(p, list) or len(p) != 2 for p in points):
        raise ConfigError("polygon vertices must be [x, y] pairs")
    try:
        normalized = [[float(p[0]), float(p[1])] for p in points]
    except (TypeError, ValueError) as exc:
        raise ConfigError("polygon coordinates must be numbers") from exc
    if any(not 0 <= value <= 1 for point in normalized for value in point):
        raise ConfigError("polygon coordinates must be normalized from 0 to 1")
    if len({(point[0], point[1]) for point in normalized}) != len(normalized):
        raise ConfigError("polygon must not repeat vertices")
    count = len(normalized)
    for i in range(count):
        for j in range(i + 1, count):
            if j in {i, (i + 1) % count} or i == (j + 1) % count:
                continue
            if _segments_intersect(
                normalized[i], normalized[(i + 1) % count],
                normalized[j], normalized[(j + 1) % count],
            ):
                raise ConfigError("polygon must not self-intersect")
    area = abs(sum(
        normalized[i][0] * normalized[(i + 1) % len(normalized)][1]
        - normalized[(i + 1) % len(normalized)][0] * normalized[i][1]
        for i in range(len(normalized))
    )) / 2
    if area <= 1e-9:
        raise ConfigError("polygon must have nonzero area")
    return normalized


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        cross = _orientation(a, b, point)
        if (
            abs(cross) <= 1e-10
            and min(a[0], b[0]) - 1e-10 <= x <= max(a[0], b[0]) + 1e-10
            and min(a[1], b[1]) - 1e-10 <= y <= max(a[1], b[1]) + 1e-10
        ):
            return True
        if (a[1] > y) != (b[1] > y):
            crossing_x = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < crossing_x:
                inside = not inside
    return inside


def _segment_boundary_parameters(
    a: list[float], b: list[float], c: list[float], d: list[float]
) -> list[float]:
    """Return parameters where AB meets CD, including collinear endpoints."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    denominator = rx * sy - ry * sx
    qx, qy = c[0] - a[0], c[1] - a[1]
    eps = 1e-10
    if abs(denominator) > eps:
        t = (qx * sy - qy * sx) / denominator
        u = (qx * ry - qy * rx) / denominator
        return [min(1.0, max(0.0, t))] if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps else []
    if abs(qx * ry - qy * rx) > eps:
        return []
    length_squared = rx * rx + ry * ry
    if length_squared <= eps:
        return [0.0]
    values = []
    for point in (c, d):
        t = ((point[0] - a[0]) * rx + (point[1] - a[1]) * ry) / length_squared
        if -eps <= t <= 1 + eps:
            values.append(min(1.0, max(0.0, t)))
    return values


def validate_contained_polygon(
    points: Any, container: list[list[float]]
) -> list[list[float]]:
    """Validate a plate lies in the closed interior of a simple ROI."""
    normalized = validate_polygon(points)
    roi = validate_polygon(container)
    if any(not _point_in_polygon(point, roi) for point in normalized):
        raise ConfigError("pressure plate must remain entirely inside the StarSeg ROI")
    for index, a in enumerate(normalized):
        b = normalized[(index + 1) % len(normalized)]
        parameters = [0.0, 1.0]
        for roi_index, c in enumerate(roi):
            d = roi[(roi_index + 1) % len(roi)]
            parameters.extend(_segment_boundary_parameters(a, b, c, d))
        ordered = sorted(set(round(value, 12) for value in parameters))
        for start, end in zip(ordered, ordered[1:]):
            if end - start <= 1e-12:
                continue
            t = (start + end) / 2
            midpoint = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
            if not _point_in_polygon(midpoint, roi):
                raise ConfigError("pressure plate edges must remain entirely inside the StarSeg ROI")
    return normalized


def migrate_user_config(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    version = data.get("schema_version", 1)
    if version == USER_SCHEMA_VERSION:
        return data, False
    if version not in {1, 2, 3, 4, 5, 6}:
        raise ConfigError("unsupported config schema")
    migrated = copy.deepcopy(data)
    if version == 1:
        migrated["trial_name"] = migrated.pop("heading", "Startracked Loop Recorder")
        old = migrated.pop("aquaria_corners", {})
        migrated["camera_polygons"] = {
            camera: copy.deepcopy(old.get(camera, DEFAULT_CORNERS))[:4]
            for camera in CAMERAS
        }
        for controls in migrated.get("camera_controls", {}).values():
            old_mode = controls.pop("exposure_auto", "Continuous")
            controls["exposure_mode"] = "Manual" if old_mode == "Off" else "Auto"
    migrated["schema_version"] = USER_SCHEMA_VERSION
    migrated.setdefault("pressure_plate", copy.deepcopy(default_user_config()["pressure_plate"]))
    migrated.setdefault("pressure_plate_polygons", {camera: [] for camera in CAMERAS})
    migrated.setdefault("arduino_pump", copy.deepcopy(default_user_config()["arduino_pump"]))
    pump = migrated["arduino_pump"]
    pump.setdefault("mode", "injector")
    pump.setdefault("injector_url", "http://127.0.0.1:8081")
    pump.setdefault("injector_heartbeat_seconds", 0.5)
    pump.setdefault("injector_timeout_seconds", 0.5)
    # Schema 6 commissions the two previously unused firmware channels. Keep
    # every output assignment and geometry exactly as stored while replacing
    # the fixed, safety-validated logical-to-firmware mapping.
    migrated["arduino_pump"]["mapping"] = copy.deepcopy(ARDUINO_PUMP_MAPPING)
    return migrated, True


def validate_user_config(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema_version") != USER_SCHEMA_VERSION:
        raise ConfigError("unsupported config schema")
    root = Path(str(data.get("record_root", ""))).expanduser()
    if not root.is_absolute():
        raise ConfigError("record_root must be absolute")
    if (
        not isinstance(data.get("trial_name"), str)
        or not data["trial_name"].strip()
        or len(data["trial_name"]) > 120
    ):
        raise ConfigError("trial_name must be 1-120 characters")
    polygons = data.get("camera_polygons", {})
    if not isinstance(polygons, dict) or set(polygons) != set(CAMERAS):
        raise ConfigError("camera_polygons must map exactly cameras A-D")
    for camera in CAMERAS:
        validate_polygon(polygons[camera])
    plates = data.get("pressure_plate_polygons", {})
    if not isinstance(plates, dict) or set(plates) != set(CAMERAS):
        raise ConfigError("pressure_plate_polygons must map exactly cameras A-D")
    for camera in CAMERAS:
        plate = plates[camera]
        if plate != []:
            validate_contained_polygon(plate, polygons[camera])
    backup = data.get("backup", {})
    if backup.get("enabled") and not Path(str(backup.get("root", ""))).is_absolute():
        raise ConfigError("enabled backup_root must be absolute")
    pressure = data.get("pressure_plate")
    if not isinstance(pressure, dict):
        raise ConfigError("pressure_plate must be an object")
    if not isinstance(pressure.get("enabled"), bool):
        raise ConfigError("pressure_plate enabled must be boolean")
    address = pressure.get("bridge_address")
    if not isinstance(address, str) or any(c in address for c in "/ \t\r\n") or not address:
        raise ConfigError("Hue bridge address must be a host or IP without a URL scheme")
    if not isinstance(pressure.get("app_key"), str):
        raise ConfigError("Hue app_key must be a string")
    outputs = pressure.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(CAMERAS):
        raise ConfigError("pressure_plate outputs must map exactly cameras A-D")
    if any(not isinstance(value, str) for value in outputs.values()):
        raise ConfigError("Hue output IDs must be strings")
    assigned = [value for value in outputs.values() if value]
    if len(set(assigned)) != len(assigned):
        raise ConfigError("each Hue output may be assigned to only one camera")
    try:
        enter = float(pressure["enter_seconds"])
        exit_ = float(pressure["exit_seconds"])
        stale = float(pressure["stale_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("pressure-plate timing values must be numbers") from exc
    if not 0 <= enter <= 10 or not 0 <= exit_ <= 30:
        raise ConfigError("pressure-plate enter/exit seconds are outside safe ranges")
    if stale <= max(enter, exit_) or stale > 60:
        raise ConfigError("pressure-plate stale seconds must exceed hysteresis and be at most 60")
    arduino = data.get("arduino_pump")
    if not isinstance(arduino, dict):
        raise ConfigError("arduino_pump must be an object")
    if not isinstance(arduino.get("enabled"), bool):
        raise ConfigError("arduino_pump enabled must be boolean")
    if arduino.get("mode") not in {"injector", "direct"}:
        raise ConfigError("Arduino pump mode must be injector or direct")
    injector_url = arduino.get("injector_url")
    try:
        parsed_injector = urlsplit(injector_url)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Injector URL must be loopback HTTP") from exc
    if (
        parsed_injector.scheme != "http"
        or parsed_injector.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed_injector.path not in {"", "/"}
        or parsed_injector.query
        or parsed_injector.fragment
        or parsed_injector.username is not None
        or parsed_injector.password is not None
        or parsed_injector.port is None
    ):
        raise ConfigError("Injector URL must be loopback HTTP with an explicit port")
    try:
        injector_heartbeat = float(arduino["injector_heartbeat_seconds"])
        injector_timeout = float(arduino["injector_timeout_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("Injector timing values must be numbers") from exc
    if not 0.1 <= injector_heartbeat <= 1.0 or not 0.1 <= injector_timeout <= 1.5:
        raise ConfigError("Injector timing is outside safe ranges")
    serial_path = arduino.get("serial_path")
    if (
        not isinstance(serial_path, str)
        or not serial_path.startswith("/dev/serial/by-id/")
        or serial_path != ARDUINO_SERIAL_PATH
    ):
        raise ConfigError("Arduino must use the approved stable serial by-id path")
    if arduino.get("baud") != 115200:
        raise ConfigError("Arduino baud must be 115200")
    if arduino.get("mapping") != ARDUINO_PUMP_MAPPING:
        raise ConfigError("Arduino mapping must be A=2, B=1, C=8, D=4")
    try:
        heartbeat = float(arduino["heartbeat_seconds"])
        reconnect = float(arduino["reconnect_max_seconds"])
        startup_timeout = float(arduino["startup_timeout_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("Arduino timing values must be numbers") from exc
    if not 0.05 <= heartbeat <= 0.25:
        raise ConfigError("Arduino heartbeat must be 0.05-0.25 seconds")
    if not 0.25 <= reconnect <= 10 or not 1 <= startup_timeout <= 15:
        raise ConfigError("Arduino reconnect/startup timing is outside safe ranges")
    return data


def atomic_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_json(path: Path, default: dict[str, Any], validator) -> dict[str, Any]:
    if not path.exists():
        data = copy.deepcopy(default)
        validator(data)
        atomic_json_write(path, data)
        return data
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if validator is validate_user_config:
        data, changed = migrate_user_config(data)
        validator(data)
        if changed:
            atomic_json_write(path, data)
        return data
    return validator(data)
