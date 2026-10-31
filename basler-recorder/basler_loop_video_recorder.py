#!/usr/bin/env python3
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from pypylon import genicam, pylon

PORT = int(os.environ.get("BASLER_STREAM_PORT", "8000"))
FPS = float(os.environ.get("BASLER_STREAM_FPS", "1"))
SEGMENT_SECONDS = int(os.environ.get("BASLER_SEGMENT_SECONDS", "600"))
DEFAULT_ROOT = Path(os.environ.get("BASLER_RECORD_ROOT", "~/Videos/Basler Loop Recordings")).expanduser()
CONFIG = Path(os.environ.get("BASLER_CONFIG", "~/.config/basler-loop-recorder/config.json")).expanduser()
PROJECT_DIR = Path(__file__).resolve().parent
STARSEG_MODEL = Path(os.environ.get("BASLER_STARSEG_MODEL", str(PROJECT_DIR / "models" / "best.pt"))).expanduser()
STARSEG_ENABLED = os.environ.get("BASLER_STARSEG_ENABLED", "1") not in {"0", "false", "False", "no"}
STARSEG_CONF_DEFAULT = float(os.environ.get("BASLER_STARSEG_CONF", "0.25"))
STARSEG_IMGSZ_DEFAULT = int(os.environ.get("BASLER_STARSEG_IMGSZ", "1024"))
JPEG_QUALITY = int(os.environ.get("BASLER_JPEG_QUALITY", "80"))
PREVIEW_JPEG_QUALITY = int(os.environ.get("BASLER_PREVIEW_JPEG_QUALITY", "45"))
X264_CRF = int(os.environ.get("BASLER_X264_CRF", "34"))
X264_PRESET = os.environ.get("BASLER_X264_PRESET", "medium")
DELETE_FAILED_SEGMENTS = os.environ.get("BASLER_DELETE_FAILED_SEGMENTS", "1") not in {"0", "false", "False", "no"}
BACKUP_ENABLED_DEFAULT = os.environ.get("BASLER_BACKUP_ENABLED", "1") not in {"0", "false", "False", "no"}
BACKUP_ROOT_DEFAULT = Path(os.environ.get("BASLER_BACKUP_ROOT", f"/media/{os.environ.get('USER', 'user')}/LaCie/Basler Loop Recordings")).expanduser()
BACKUP_THROTTLE_MBPS_DEFAULT = float(os.environ.get("BASLER_BACKUP_THROTTLE_MBPS", "8"))
# Camera letter -> Basler serial number. Set BASLER_SERIAL_A/B/C for your cameras.
CAMERAS = {letter: os.environ.get(f"BASLER_SERIAL_{letter}", default) for letter, default in (("A", "00000001"), ("B", "00000002"), ("C", "00000003"))}
ORDER = ["C", "A", "B"]

control_lock = threading.Lock()
status_lock = threading.Lock()
camera_control_lock = threading.Lock()
queues = {k: queue.Queue(maxsize=120) for k in CAMERAS}
starseg_queue = queue.Queue(maxsize=240)
backup_queue = queue.Queue(maxsize=2000)
backup_config_lock = threading.Lock()
backup_config = {
    "enabled": BACKUP_ENABLED_DEFAULT,
    "root": str(BACKUP_ROOT_DEFAULT),
    "throttle_mbps": BACKUP_THROTTLE_MBPS_DEFAULT,
}
backup_status = {
    "enabled": BACKUP_ENABLED_DEFAULT,
    "root": str(BACKUP_ROOT_DEFAULT),
    "state": "starting",
    "queued": 0,
    "copied_files": 0,
    "copied_bytes": 0,
    "skipped_files": 0,
    "errors": 0,
    "last_source": "",
    "last_destination": "",
    "last_error": "",
    "last_completed_ts": 0,
}
starseg_pending = {}
starseg_lock = threading.Condition()
starseg_status = {
    "enabled": STARSEG_ENABLED,
    "model": str(STARSEG_MODEL),
    "desired_model": str(STARSEG_MODEL),
    "reload_requested": False,
    "device": "unknown",
    "state": "starting" if STARSEG_ENABLED else "disabled",
    "queued": 0,
    "processed": 0,
    "errors": 0,
    "last_camera": "",
    "last_frame": "",
    "last_detections": 0,
    "last_error": "",
}
starseg_active_model = str(STARSEG_MODEL)
starseg_latest = {
    k: {
        "camera": k,
        "frame_index": None,
        "frame_ts": 0,
        "updated_ts": 0,
        "width": 0,
        "height": 0,
        "detections": [],
    }
    for k in CAMERAS
}
starseg_display = {
    "inference_enabled": STARSEG_ENABLED,
    "overlay_enabled": True,
    "conf": STARSEG_CONF_DEFAULT,
    "imgsz": STARSEG_IMGSZ_DEFAULT,
    "mask_opacity": 0.22,
    "line_width": 2.0,
    "show_boxes": True,
    "show_masks": True,
    "show_labels": True,
}
DEFAULT_CAMERA_CONTROLS = {
    "fps": FPS,
    "exposure_auto": "Off",
    "exposure_time_us": 20000.0,
    "auto_exposure_max_us": 100000.0,
    "gain_auto": "Off",
    "gain_db": 0.0,
    "auto_gain_max_db": 24.0,
    "target_brightness": 0.5,
    "white_balance_auto": "Continuous",
    "light_source_preset": "Daylight6500K",
}
camera_controls = {k: dict(DEFAULT_CAMERA_CONTROLS) for k in CAMERAS}
CORNER_COUNT = 8
aquaria_corners_lock = threading.Lock()
# Per-camera aquarium corner annotations in normalized image coordinates (0..1).
# None until set in the UI. Order: 1-4 = top rim (outer quad, clockwise from
# top-left), 5-8 = bottom floor (inner quad, matching order).
aquaria_corners = {k: None for k in CAMERAS}
control = {
    "recording": False,
    "heading": "recording",
    "record_root": str(DEFAULT_ROOT),
    "default_record_root": str(DEFAULT_ROOT),
    "recording_session": "",
    "recording_started_iso": "",
    "root_error": "",
}
status = {
    k: {
        "letter": k,
        "serial": s,
        "state": "starting",
        "acquired": 0,
        "encoded": 0,
        "queued": 0,
        "dropped": 0,
        "late": 0,
        "segment_seconds": SEGMENT_SECONDS,
        "folder": str(DEFAULT_ROOT / k),
    }
    for k, s in CAMERAS.items()
}


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip() or "recording")[:80]


def iso(ts):
    return time.strftime("%Y%m%dT%H%M%S", time.localtime(ts)) + f"{int((ts % 1) * 1000):03d}"


def normalize_root(value):
    p = Path(value.strip() or str(DEFAULT_ROOT)).expanduser()
    if not p.is_absolute():
        p = DEFAULT_ROOT / p
    return p.resolve()


def preview_dir(root=None):
    return Path(root or get_root()) / "preview"


def active_dir(root=None):
    return Path(root or get_root()) / "_active_segments"


def save_jpeg(img, path, quality=None):
    opts = pylon.ImagePersistenceOptions()
    opts.SetQuality(max(1, min(100, int(JPEG_QUALITY if quality is None else quality))))
    img.Save(pylon.ImageFileFormat_Jpeg, str(path), opts)


def ensure_root(root):
    root = Path(root)
    for p in [root / "preview", root / "_active_segments", *[root / k for k in CAMERAS]]:
        p.mkdir(parents=True, exist_ok=True)
    return root


def load_config():
    try:
        if CONFIG.exists():
            data = json.loads(CONFIG.read_text())
            if data.get("record_root"):
                control["record_root"] = str(normalize_root(data["record_root"]))
            if data.get("heading"):
                control["heading"] = safe(data["heading"])
            for letter, values in data.get("camera_controls", {}).items():
                if letter in camera_controls and isinstance(values, dict):
                    camera_controls[letter].update({k: values[k] for k in values if k in DEFAULT_CAMERA_CONTROLS})
            if isinstance(data.get("starseg_display"), dict):
                starseg_display.update({k: data["starseg_display"][k] for k in data["starseg_display"] if k in starseg_display})
            if data.get("starseg_model"):
                with status_lock:
                    starseg_status["desired_model"] = str(Path(str(data["starseg_model"])).expanduser())
            if isinstance(data.get("aquaria_corners"), dict):
                with aquaria_corners_lock:
                    for letter, pts in data["aquaria_corners"].items():
                        if letter not in aquaria_corners or not isinstance(pts, list) or len(pts) != CORNER_COUNT:
                            continue
                        clean = []
                        ok = True
                        for i, p in enumerate(pts):
                            try:
                                clean.append({"n": i + 1, "x": min(1.0, max(0.0, float(p["x"]))), "y": min(1.0, max(0.0, float(p["y"])))})
                            except Exception:
                                ok = False
                                break
                        if ok:
                            aquaria_corners[letter] = clean
            if isinstance(data.get("backup"), dict):
                with backup_config_lock:
                    b = data["backup"]
                    if "enabled" in b:
                        backup_config["enabled"] = bool(b["enabled"])
                    if b.get("root"):
                        backup_config["root"] = str(Path(str(b["root"])).expanduser().resolve())
                    if "throttle_mbps" in b:
                        backup_config["throttle_mbps"] = max(0.0, float(b["throttle_mbps"]))
                with status_lock:
                    backup_status.update(enabled=backup_config["enabled"], root=backup_config["root"], throttle_mbps=backup_config["throttle_mbps"])
    except Exception as e:
        control["root_error"] = f"config load error: {e!r}"


def save_config():
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with control_lock, camera_control_lock, backup_config_lock:
        with aquaria_corners_lock:
            corners = {k: (list(v) if v else None) for k, v in aquaria_corners.items()}
        with status_lock:
            starseg_model = starseg_status.get("desired_model", str(STARSEG_MODEL))
        data = {"record_root": control["record_root"], "heading": control["heading"], "camera_controls": camera_controls, "starseg_display": starseg_display, "starseg_model": starseg_model, "backup": backup_config, "aquaria_corners": corners}
    CONFIG.write_text(json.dumps(data, indent=2))


def get_root():
    with control_lock:
        return Path(control["record_root"])


def set_record_root(value):
    root = normalize_root(value)
    ensure_root(root)
    with control_lock:
        if control["recording"]:
            raise RuntimeError("stop recording before changing the save folder")
        control["record_root"] = str(root)
        control["root_error"] = ""
    with status_lock:
        for letter in CAMERAS:
            status[letter]["folder"] = str(root / letter)
    save_config()
    return root


def parse_float(value, name, lo=None, hi=None):
    try:
        v = float(value)
    except Exception:
        raise RuntimeError(f"{name} must be a number")
    if lo is not None and v < lo:
        raise RuntimeError(f"{name} must be >= {lo}")
    if hi is not None and v > hi:
        raise RuntimeError(f"{name} must be <= {hi}")
    return v


def set_camera_controls(letter, data):
    if letter not in CAMERAS:
        raise RuntimeError(f"unknown camera {letter}")
    enum_allowed = {
        "exposure_auto": {"Off", "Once", "Continuous"},
        "gain_auto": {"Off", "Once", "Continuous"},
        "white_balance_auto": {"Off", "Once", "Continuous"},
        "light_source_preset": {"Off", "Tungsten", "Daylight5000K", "Daylight6500K", "FactoryLED6000K"},
    }
    updates = {}
    for key in DEFAULT_CAMERA_CONTROLS:
        if key not in data:
            continue
        value = data[key][0] if isinstance(data[key], list) else data[key]
        if key == "fps":
            updates[key] = parse_float(value, key, 0.1, 120.0)
        elif key == "exposure_time_us":
            updates[key] = parse_float(value, key, 1.0, 10000000.0)
        elif key == "auto_exposure_max_us":
            updates[key] = parse_float(value, key, 1.0, 10000000.0)
        elif key == "gain_db":
            updates[key] = parse_float(value, key, 0.0, 36.0)
        elif key == "auto_gain_max_db":
            updates[key] = parse_float(value, key, 0.0, 36.0)
        elif key == "target_brightness":
            updates[key] = parse_float(value, key, 0.0, 1.0)
        elif key in enum_allowed:
            if value not in enum_allowed[key]:
                raise RuntimeError(f"{key} must be one of {sorted(enum_allowed[key])}")
            updates[key] = value
    with camera_control_lock:
        camera_controls[letter].update(updates)
        current = dict(camera_controls[letter])
    save_config()
    return current


def get_camera_controls():
    with camera_control_lock:
        return {k: dict(v) for k, v in camera_controls.items()}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def set_starseg_display(data):
    updates = {}
    bools = {"inference_enabled", "overlay_enabled", "show_boxes", "show_masks", "show_labels"}
    for key in bools:
        if key in data:
            value = data[key][0] if isinstance(data[key], list) else data[key]
            updates[key] = parse_bool(value)
    if "starseg_conf" in data:
        updates["conf"] = parse_float(data["starseg_conf"][0] if isinstance(data["starseg_conf"], list) else data["starseg_conf"], "StarSeg confidence", 0.01, 1.0)
    if "starseg_imgsz" in data:
        updates["imgsz"] = int(parse_float(data["starseg_imgsz"][0] if isinstance(data["starseg_imgsz"], list) else data["starseg_imgsz"], "StarSeg image size", 320, 2600))
    if "mask_opacity" in data:
        updates["mask_opacity"] = parse_float(data["mask_opacity"][0] if isinstance(data["mask_opacity"], list) else data["mask_opacity"], "Mask opacity", 0.0, 1.0)
    if "line_width" in data:
        updates["line_width"] = parse_float(data["line_width"][0] if isinstance(data["line_width"], list) else data["line_width"], "Line width", 0.5, 10.0)
    with status_lock:
        starseg_display.update(updates)
        current = dict(starseg_display)
        starseg_status.update(display=current, inference_enabled=bool(starseg_display.get("inference_enabled", False)))
        if not starseg_display.get("inference_enabled", False):
            for latest in starseg_latest.values():
                latest["detections"] = []
                latest["updated_ts"] = time.time()
    save_config()
    return current


def set_starseg_model(data):
    raw = data["starseg_model"][0] if isinstance(data.get("starseg_model"), list) else data.get("starseg_model")
    if not raw or not str(raw).strip():
        raise RuntimeError("starseg_model path is required")
    path = Path(str(raw).strip()).expanduser()
    if path.suffix.lower() != ".pt":
        raise RuntimeError(f"model must be a .pt file: {path}")
    if not path.exists():
        raise RuntimeError(f"model not found: {path}")
    with status_lock:
        starseg_status["desired_model"] = str(path)
        current = {
            "model": starseg_status.get("model"),
            "desired_model": starseg_status.get("desired_model"),
            "reload_requested": bool(starseg_status.get("reload_requested", False)),
        }
    save_config()
    return current


def list_starseg_models():
    """Return a sorted, de-duplicated list of selectable .pt model paths.

    Includes every .pt under PROJECT_DIR/models plus the currently active and
    desired model paths (in case they live elsewhere), so the UI dropdown can
    always show the model in use.
    """
    paths = []
    seen = set()

    def add(p):
        if not p:
            return
        s = str(Path(str(p)).expanduser())
        if s not in seen:
            seen.add(s)
            paths.append(s)

    models_dir = PROJECT_DIR / "models"
    try:
        found = sorted(str(p) for p in models_dir.glob("*.pt"))
    except Exception:
        found = []
    for p in found:
        add(p)
    # Read without acquiring status_lock: callers (e.g. the status.json builder)
    # may already hold it, and threading.Lock is not reentrant. Reading two
    # string keys from the dict is atomic enough for display purposes.
    add(starseg_status.get("model"))
    add(starseg_status.get("desired_model"))
    return paths


def get_aquaria_corners():
    with aquaria_corners_lock:
        return {k: (list(v) if v else None) for k, v in aquaria_corners.items()}


def missing_corner_cameras():
    with aquaria_corners_lock:
        return [k for k in ORDER if not (aquaria_corners.get(k) and len(aquaria_corners[k]) == CORNER_COUNT)]


def set_aquaria_corners(data):
    letter = data["camera"][0] if isinstance(data.get("camera"), list) else data.get("camera")
    if letter not in CAMERAS:
        raise RuntimeError(f"unknown camera {letter}")
    raw = data.get("points")
    raw = raw[0] if isinstance(raw, list) else raw
    try:
        pts = json.loads(raw)
    except Exception:
        raise RuntimeError("corner points must be a JSON array")
    if not isinstance(pts, list) or len(pts) != CORNER_COUNT:
        raise RuntimeError(f"exactly {CORNER_COUNT} corner points are required")
    clean = []
    for i, p in enumerate(pts):
        try:
            x = float(p["x"])
            y = float(p["y"])
        except Exception:
            raise RuntimeError("each corner needs numeric x and y")
        clean.append({"n": i + 1, "x": min(1.0, max(0.0, x)), "y": min(1.0, max(0.0, y))})
    with aquaria_corners_lock:
        aquaria_corners[letter] = clean
    save_config()
    return clean


load_config()
try:
    ensure_root(get_root())
except Exception as e:
    control["root_error"] = repr(e)

HTML = """<!doctype html>
<html><head><meta charset='utf-8'><title>Basler Loop Recorder</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:sans-serif}header{padding:12px;background:#222;position:sticky;top:0;z-index:9}input,select{background:#111;color:#eee;border:1px solid #555;border-radius:4px;padding:5px}button{background:#444;color:#eee;border:0;border-radius:4px;padding:6px 10px;margin-left:4px;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:10px}.cam{background:#222}.cam h2{margin:0;padding:8px;background:#2c2c2c}.view{height:45vh;overflow:hidden;background:#000;cursor:grab;touch-action:none;position:relative}.stage{position:relative;width:100%;transform-origin:0 0}.stage img{width:100%;display:block;user-select:none;-webkit-user-drag:none}.overlay{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}.legend{position:absolute;left:8px;top:8px;background:rgba(0,0,0,.55);padding:4px 6px;border-radius:4px;font-size:12px;color:#eee;pointer-events:none}.tools{padding:8px;display:flex;gap:4px;flex-wrap:wrap}.tools button{margin-left:0}.status{font-family:monospace;font-size:12px;white-space:pre-wrap;padding:8px;max-height:28vh;overflow:auto}.row{margin-top:8px}.small{font-size:12px;color:#bbb}.err{color:#ff8888}.ok{color:#88ff88}.tabs{display:flex;gap:4px;padding:8px 8px 0}.tab{background:#333}.tab.active{background:#666}.pane{display:none;padding:8px}.pane.active{display:block}.ctrlgrid{display:grid;grid-template-columns:auto 1fr;gap:6px;align-items:center;font-size:13px}.ctrlgrid input,.ctrlgrid select{width:100%}.ctrlgrid input[type=range]{padding:0}.ctrlmsg{font-size:12px;min-height:1.2em;color:#bbb}.muted{color:#aaa;font-size:12px}
</style></head><body>
<header>
  <div>Basler loop recorder — display order C-A-B — <input id='head' placeholder='file heading' value='recording' oninput='queueHeadingSave()' title='Video filename prefix; autosaves after typing'> <button onclick='ctl(true)'>Start recording</button><button onclick='ctl(false)'>Stop recording</button> <button onclick='applyAllCameras()'>Apply All Controls</button> <span id='rec'></span></div>
  <div class='row'>Save folder: <input id='root' style='width:520px' placeholder='~/Videos/Basler Loop Recordings'> <button onclick='saveRoot()'>Set save folder</button> <button onclick='resetRoot()'>Default Videos folder</button></div>
  <div class='row'>Backup: <select id='backup_enabled'><option value='1'>On</option><option value='0'>Off</option></select> <input id='backup_root' style='width:520px' placeholder='/media/&lt;user&gt;/LaCie/Basler Loop Recordings'> <label class='small'>MB/s</label> <input id='backup_throttle_mbps' style='width:70px' type='number' min='0' step='1'> <button onclick='saveBackup()'>Set backup</button> <span id='backup_status' class='small'></span></div>
  <div id='msg' class='small'></div>
</header>
<div id='g' class='grid'></div>
<script>
const O=['C','A','B'], V={A:{z:1,x:0,y:0},B:{z:1,x:0,y:0},C:{z:1,x:0,y:0}};
const CORN={A:null,B:null,C:null}, EDIT={A:false,B:false,C:false}, lastLatest={};
const CORNER_EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
function defaultCorners(){return [[.12,.12],[.88,.12],[.88,.88],[.12,.88],[.28,.28],[.72,.28],[.72,.72],[.28,.72]].map((p,i)=>({n:i+1,x:p[0],y:p[1]}))}
function drawCorners(l,cv,ctx){let pts=CORN[l];if(!pts||pts.length!==8)return;let w=cv.width,h=cv.height;ctx.save();ctx.lineWidth=1.5;ctx.strokeStyle=EDIT[l]?'#00e5ff':'rgba(0,229,255,.85)';ctx.beginPath();for(const e of CORNER_EDGES){ctx.moveTo(pts[e[0]].x*w,pts[e[0]].y*h);ctx.lineTo(pts[e[1]].x*w,pts[e[1]].y*h)}ctx.stroke();for(const p of pts){let x=p.x*w,y=p.y*h;ctx.beginPath();ctx.arc(x,y,9,0,7);ctx.fillStyle=p.n<=4?'#00e5ff':'#ffd400';ctx.fill();ctx.lineWidth=1;ctx.strokeStyle='#001';ctx.stroke();ctx.fillStyle='#000';ctx.font='bold 11px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(p.n),x,y)}ctx.restore();ctx.textAlign='start';ctx.textBaseline='alphabetic'}
function updateCornerStatus(){O.forEach(l=>{let el=document.getElementById('cstat'+l);if(!el)return;let ok=CORN[l]&&CORN[l].length===8;el.textContent=EDIT[l]?'editing — drag dots':(ok?'corners set':'corners NOT set');el.className=ok?'small ok':'small err'})}
function toggleCorners(l){EDIT[l]=!EDIT[l];let cv=document.getElementById('ov'+l),btn=document.getElementById('cbtn'+l);if(EDIT[l]){if(!CORN[l])CORN[l]=defaultCorners();cv.style.pointerEvents='auto';cv.style.cursor='crosshair';btn.textContent='Done editing';btn.style.background='#066';saveCorners(l)}else{cv.style.pointerEvents='none';cv.style.cursor='';btn.textContent='Edit corners';btn.style.background=''}drawOverlay(l,lastLatest[l]||{});updateCornerStatus()}
function resetCorners(l){CORN[l]=defaultCorners();drawOverlay(l,lastLatest[l]||{});updateCornerStatus();saveCorners(l)}
async function saveCorners(l){try{await post({aquaria_corners:'1',camera:l,points:JSON.stringify(CORN[l].map(p=>({x:p.x,y:p.y})))});updateCornerStatus()}catch(e){let el=document.getElementById('cstat'+l);if(el){el.textContent=e.message;el.className='small err'}}}
const enums={exposure_auto:['Off','Once','Continuous'],gain_auto:['Off','Once','Continuous'],white_balance_auto:['Off','Once','Continuous'],light_source_preset:['Off','Tungsten','Daylight5000K','Daylight6500K','FactoryLED6000K']};
function opt(name){return enums[name].map(x=>`<option value="${x}">${x}</option>`).join('')}
function controlPane(l){return `<div class="ctrlgrid">
<label>FPS</label><input id="${l}_fps" type="number" step="0.1" min="0.1">
<label>Exposure auto</label><select id="${l}_exposure_auto">${opt('exposure_auto')}</select>
<label>Manual exposure (us)</label><input id="${l}_exposure_time_us" type="number" step="1" min="1">
<label>Auto exposure max (us)</label><input id="${l}_auto_exposure_max_us" type="number" step="1" min="1">
<label>Gain auto</label><select id="${l}_gain_auto">${opt('gain_auto')}</select>
<label>Manual gain (dB)</label><input id="${l}_gain_db" type="number" step="0.1" min="0">
<label>Auto gain max (dB)</label><input id="${l}_auto_gain_max_db" type="number" step="0.1" min="0">
<label>Target brightness</label><input id="${l}_target_brightness" type="number" step="0.01" min="0" max="1">
<label>White balance auto</label><select id="${l}_white_balance_auto">${opt('white_balance_auto')}</select>
<label>Light preset</label><select id="${l}_light_source_preset">${opt('light_source_preset')}</select>
</div><h3>StarSeg display / model</h3><div class="ctrlgrid">
<label>Model</label><select id="${l}_starseg_model"></select>
<label>Inference enabled</label><select id="${l}_inference_enabled"><option value="1">On</option><option value="0">Off</option></select>
<label>Overlay enabled</label><select id="${l}_overlay_enabled"><option value="1">On</option><option value="0">Off</option></select>
<label>Confidence</label><input id="${l}_starseg_conf" type="number" step="0.01" min="0.01" max="1">
<label>Inference image size</label><input id="${l}_starseg_imgsz" type="number" step="32" min="320" max="2600">
<label>Mask opacity</label><input id="${l}_mask_opacity" type="number" step="0.05" min="0" max="1">
<label>Line width</label><input id="${l}_line_width" type="number" step="0.5" min="0.5" max="10">
<label>Show boxes</label><select id="${l}_show_boxes"><option value="1">On</option><option value="0">Off</option></select>
<label>Show masks</label><select id="${l}_show_masks"><option value="1">On</option><option value="0">Off</option></select>
<label>Show labels</label><select id="${l}_show_labels"><option value="1">On</option><option value="0">Off</option></select>
</div><div class="row"><button onclick="applyCam('${l}')">Apply Camera ${l} Controls</button><button onclick="applyStarseg('${l}')">Apply StarSeg Display</button><button onclick="applyModel('${l}')">Apply model</button><button onclick="copyToAll('${l}')">Copy to all</button></div><div id="${l}_modelmsg" class="ctrlmsg"></div><div id="${l}_ctrlmsg" class="ctrlmsg"></div><div class="muted">Camera changes are per-camera. StarSeg inference/display/model settings are global, apply immediately, and are saved in config. Turning inference off stops new StarSeg GPU jobs and sidecar detections until turned back on. Model changes take effect on the next Stop then Start (recording restart).</div>`}
g.innerHTML=O.map(l=>`<section class="cam"><h2>Camera ${l}</h2><div class="view" id="view${l}" data-cam="${l}"><div class="stage" id="stg${l}"><img id="i${l}"><canvas class="overlay" id="ov${l}"></canvas><div class="legend" id="lg${l}"></div></div></div><div class="tools"><button onclick="Z('${l}',1.25)">+</button><button onclick="Z('${l}',.8)">-</button><button onclick="R('${l}')">Reset</button><button onclick="FS('${l}')">Full screen</button><button onclick="PO('${l}')">Pop out</button><button id="cbtn${l}" onclick="toggleCorners('${l}')">Edit corners</button><button onclick="resetCorners('${l}')">Reset corners</button><span id="cstat${l}" class="small"></span></div><div class="tabs"><button id="${l}_tab_status" class="tab active" onclick="tab('${l}','status')">Status</button><button id="${l}_tab_controls" class="tab" onclick="tab('${l}','controls')">Controls</button></div><div id="${l}_pane_status" class="pane active"><pre class="status" id="s${l}"></pre></div><div id="${l}_pane_controls" class="pane">${controlPane(l)}</div></section>`).join('');
function tab(l,t){['status','controls'].forEach(x=>{document.getElementById(`${l}_tab_${x}`).classList.toggle('active',x==t);document.getElementById(`${l}_pane_${x}`).classList.toggle('active',x==t)})}
function A(l){let v=V[l];document.getElementById('stg'+l).style.transform=`translate(${v.x}px,${v.y}px) scale(${v.z})`}
function Z(l,f){V[l].z=Math.max(.25,Math.min(16,V[l].z*f));A(l)}
function R(l){V[l]={z:1,x:0,y:0};A(l)}
function FS(l){let e=document.getElementById('view'+l);if(e.requestFullscreen)e.requestFullscreen();else if(e.webkitRequestFullscreen)e.webkitRequestFullscreen()}
function PO(l){window.open(`/popout/${l}`,`basler_camera_${l}`,'popup=yes,width=1400,height=1000')}
document.querySelectorAll('[data-cam]').forEach(e=>{let l=e.dataset.cam,d=0,sx=0,sy=0,ox=0,oy=0;e.onwheel=x=>{x.preventDefault();Z(l,x.deltaY<0?1.2:.833)};e.onpointerdown=x=>{d=1;sx=x.clientX;sy=x.clientY;ox=V[l].x;oy=V[l].y;e.setPointerCapture(x.pointerId)};e.onpointermove=x=>{if(!d)return;V[l].x=ox+x.clientX-sx;V[l].y=oy+x.clientY-sy;A(l)};e.onpointerup=()=>d=0});
O.forEach(l=>{let cv=document.getElementById('ov'+l),drag=-1;function toCv(x){let r=cv.getBoundingClientRect();return{x:(x.clientX-r.left)*cv.width/r.width,y:(x.clientY-r.top)*cv.height/r.height}}cv.addEventListener('pointerdown',x=>{if(!EDIT[l]||!CORN[l])return;x.stopPropagation();x.preventDefault();let c=toCv(x);drag=-1;let best=1e9;CORN[l].forEach((p,i)=>{let dx=p.x*cv.width-c.x,dy=p.y*cv.height-c.y,dd=dx*dx+dy*dy;if(dd<best&&dd<=400){best=dd;drag=i}});if(drag>=0)cv.setPointerCapture(x.pointerId)});cv.addEventListener('pointermove',x=>{if(!EDIT[l]||drag<0)return;x.stopPropagation();let c=toCv(x);CORN[l][drag].x=Math.min(1,Math.max(0,c.x/cv.width));CORN[l][drag].y=Math.min(1,Math.max(0,c.y/cv.height));drawOverlay(l,lastLatest[l]||{})});cv.addEventListener('pointerup',x=>{if(!EDIT[l]||drag<0)return;x.stopPropagation();drag=-1;saveCorners(l)});});
async function post(data){let r=await fetch('/control',{method:'POST',body:new URLSearchParams(data)}); let t=await r.text(); if(!r.ok) throw new Error(t); return t}
function setIfClean(id,value){let el=document.getElementById(id); if(el && document.activeElement!==el && !dirty[id]) el.value=value}
let headingTimer=null;
function queueHeadingSave(){dirty.head=true; clearTimeout(headingTimer); headingTimer=setTimeout(async()=>{try{await post({heading:head.value}); dirty.head=false; msg.textContent='Video name saved'; msg.className='small ok'}catch(e){msg.textContent=e.message;msg.className='small err'}},600)}
async function ctl(on){try{await post({recording:on?'1':'0',heading:head.value}); dirty.head=false; msg.textContent='';}catch(e){msg.textContent=e.message;msg.className='small err'} T()}
async function saveRoot(){try{await post({heading:head.value,record_root:root.value}); msg.textContent='Save folder set';msg.className='small ok';}catch(e){msg.textContent=e.message;msg.className='small err'} T()}
async function saveBackup(){try{await post({backup_enabled:backup_enabled.value,backup_root:backup_root.value,backup_throttle_mbps:backup_throttle_mbps.value}); msg.textContent='Backup settings saved';msg.className='small ok'}catch(e){msg.textContent=e.message;msg.className='small err'} T()}
async function resetRoot(){try{let j=await fetch('/status.json').then(r=>r.json()); root.value=j.control.default_record_root; await saveRoot();}catch(e){msg.textContent=e.message;msg.className='small err'}}
function readCam(l){let d={camera:l}; ['fps','exposure_auto','exposure_time_us','auto_exposure_max_us','gain_auto','gain_db','auto_gain_max_db','target_brightness','white_balance_auto','light_source_preset'].forEach(k=>d[k]=document.getElementById(`${l}_${k}`).value); return d}
function readStarseg(l){let d={starseg_display:'1'}; ['inference_enabled','overlay_enabled','starseg_conf','starseg_imgsz','mask_opacity','line_width','show_boxes','show_masks','show_labels'].forEach(k=>d[k]=document.getElementById(`${l}_${k}`).value); return d}
async function applyStarseg(l){try{await post(readStarseg(l)); document.getElementById(`${l}_ctrlmsg`).textContent='StarSeg display applied'; document.getElementById(`${l}_ctrlmsg`).className='ctrlmsg ok'}catch(e){document.getElementById(`${l}_ctrlmsg`).textContent=e.message; document.getElementById(`${l}_ctrlmsg`).className='ctrlmsg err'} T()}
async function applyModel(l){let sel=document.getElementById(`${l}_starseg_model`),m=document.getElementById(`${l}_modelmsg`);if(!sel||!sel.value){if(m){m.textContent='No model selected';m.className='ctrlmsg err'}return}try{await post({starseg_model:sel.value}); dirty[sel.id]=false; if(m){m.textContent='Model set — takes effect on next Stop then Start';m.className='ctrlmsg ok'}}catch(e){if(m){m.textContent=e.message;m.className='ctrlmsg err'}} T()}
async function applyCam(l){try{await post(readCam(l)); document.getElementById(`${l}_ctrlmsg`).textContent='Applied'; document.getElementById(`${l}_ctrlmsg`).className='ctrlmsg ok'}catch(e){document.getElementById(`${l}_ctrlmsg`).textContent=e.message; document.getElementById(`${l}_ctrlmsg`).className='ctrlmsg err'} T()}
async function applyAllCameras(){msg.textContent='Applying controls to all cameras…';msg.className='small';let ok=0,errs=[];for(const l of O){try{await post(readCam(l)); ok++; document.getElementById(`${l}_ctrlmsg`).textContent='Applied'; document.getElementById(`${l}_ctrlmsg`).className='ctrlmsg ok'}catch(e){errs.push(`Camera ${l}: ${e.message}`); document.getElementById(`${l}_ctrlmsg`).textContent=e.message; document.getElementById(`${l}_ctrlmsg`).className='ctrlmsg err'}} if(errs.length){msg.textContent=`Applied ${ok}/${O.length}; `+errs.join(' | ');msg.className='small err'}else{msg.textContent='Applied controls to all cameras';msg.className='small ok'} T()}
async function copyToAll(src){for(const l of O){if(l==src)continue; ['fps','exposure_auto','exposure_time_us','auto_exposure_max_us','gain_auto','gain_db','auto_gain_max_db','target_brightness','white_balance_auto','light_source_preset'].forEach(k=>document.getElementById(`${l}_${k}`).value=document.getElementById(`${src}_${k}`).value); await applyCam(l)} await applyCam(src)}
let dirty={}; document.addEventListener('input',e=>{if(e.target&&e.target.id) dirty[e.target.id]=true});
function fillControls(j){for(const l of O){let c=j.camera_controls[l]||{}; for(const [k,v] of Object.entries(c)){let el=document.getElementById(`${l}_${k}`); if(el && !dirty[el.id]) el.value=v}let sd=j.starseg.display||j.starseg_display||{};let map={conf:'starseg_conf',imgsz:'starseg_imgsz'};for(const [k,v] of Object.entries(sd)){let id=map[k]||k,el=document.getElementById(`${l}_${id}`); if(el && !dirty[el.id]) el.value=(typeof v==='boolean')?(v?'1':'0'):v}}}
function modelLabel(p){return String(p||'').split('/').pop()}
function fillModels(j){let ss=j.starseg||{},avail=ss.available_models||[],desired=ss.desired_model||ss.model||'',active=ss.model||'';for(const l of O){let sel=document.getElementById(`${l}_starseg_model`);if(!sel)continue;let want=avail.join('|');if(sel.dataset.opts!==want){sel.innerHTML=avail.map(p=>`<option value="${p}">${modelLabel(p)}</option>`).join('');sel.dataset.opts=want}if(!dirty[sel.id]){sel.value=desired;let hint=document.getElementById(`${l}_modelmsg`);if(hint){if(active&&desired&&active!==desired){hint.textContent=`pending: ${modelLabel(desired)} (active ${modelLabel(active)}) — Stop then Start to apply`;hint.className='ctrlmsg'}else{hint.textContent='';hint.className='ctrlmsg'}}}}}
const clsColor={'Other_species':'#ffcc33','Prey':'#33ccff','Pycnopodia_helianthoides':'#ff4d88'};
let starsegDisplay={inference_enabled:true,overlay_enabled:true,mask_opacity:.22,line_width:2,show_boxes:true,show_masks:true,show_labels:true};
function drawOverlay(l,latest){let img=document.getElementById('i'+l),cv=document.getElementById('ov'+l),lg=document.getElementById('lg'+l),ctx=cv.getContext('2d');let w=img.clientWidth,h=img.clientHeight;if(!w||!h){return} if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h} ctx.clearRect(0,0,w,h);drawCorners(l,cv,ctx);if(!starsegDisplay.inference_enabled){lg.textContent='StarSeg inference off';return}if(!starsegDisplay.overlay_enabled){lg.textContent='StarSeg overlay off';return}let det=(latest&&latest.detections)||[],iw=latest&&latest.width||img.naturalWidth||w,ih=latest&&latest.height||img.naturalHeight||h,sx=w/iw,sy=h/ih,counts={};for(const d of det){let name=d.class_name||String(d.class_id),col=clsColor[name]||'#66ff66';counts[name]=(counts[name]||0)+1;ctx.lineWidth=Number(starsegDisplay.line_width||2);ctx.strokeStyle=col;ctx.fillStyle=col;let seg=d.segment_xy||[];if(starsegDisplay.show_masks&&seg.length){ctx.beginPath();seg.forEach((p,i)=>{let x=p[0]*sx,y=p[1]*sy;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath();ctx.globalAlpha=Number(starsegDisplay.mask_opacity??.22);ctx.fill();ctx.globalAlpha=1;ctx.stroke()}let b=d.box_xyxy||[];if(b.length===4&&starsegDisplay.show_boxes){let x=b[0]*sx,y=b[1]*sy,bw=(b[2]-b[0])*sx,bh=(b[3]-b[1])*sy;ctx.strokeRect(x,y,bw,bh);if(starsegDisplay.show_labels){let label=`${name} ${(100*(d.confidence||0)).toFixed(0)}%`;ctx.font='13px sans-serif';let tw=ctx.measureText(label).width+8;ctx.fillRect(x,Math.max(0,y-18),tw,18);ctx.fillStyle='#000';ctx.fillText(label,x+4,Math.max(13,y-5));ctx.fillStyle=col}}}let summary=Object.entries(counts).map(([k,v])=>`${k}:${v}`).join('  ');lg.textContent=summary||'StarSeg live: no detections'}
async function loadOverlays(t){try{let latest=await fetch('/starseg/latest.json?t='+t).then(r=>r.json());O.forEach(l=>{lastLatest[l]=latest[l]||{};drawOverlay(l,lastLatest[l])})}catch(e){}}
async function T(){let t=Date.now();O.forEach(l=>window['i'+l].src=`/preview/${l}.jpg?t=${t}`);let j=await fetch('/status.json?t='+t).then(r=>r.json());starsegDisplay=j.starseg.display||starsegDisplay;setIfClean('head',j.control.heading);setIfClean('root',j.control.record_root);let b=j.backup||{};setIfClean('backup_enabled',b.enabled?'1':'0');setIfClean('backup_root',b.root||'');setIfClean('backup_throttle_mbps',b.throttle_mbps??'');backup_status.textContent=`${b.state||''} q=${b.queued||0} copied=${b.copied_files||0} errors=${b.errors||0}`;backup_status.className=(b.errors?'small err':'small ok');rec.textContent=j.control.recording?' RECORDING':' STOPPED';rec.style.color=j.control.recording?'#6f6':'#f66';if(j.control.root_error){msg.textContent=j.control.root_error;msg.className='small err'};fillControls(j);fillModels(j);if(j.aquaria_corners){O.forEach(l=>{if(!EDIT[l]&&j.aquaria_corners[l])CORN[l]=j.aquaria_corners[l].map((p,i)=>({n:i+1,x:p.x,y:p.y}))})}updateCornerStatus();O.forEach(l=>document.getElementById('s'+l).textContent=JSON.stringify(j.cameras[l],null,2));loadOverlays(t)}
T();setInterval(()=>{if(document.hidden)return;T()},2000);
</script></body></html>"""


def popout_html(letter):
    if letter not in CAMERAS:
        letter = "C"
    body = """<!doctype html>
<html><head><meta charset='utf-8'><title>Camera __CAM__ Popout</title>
<style>
html,body{margin:0;width:100%;height:100%;background:#000;color:#eee;font-family:sans-serif;overflow:hidden}#bar{position:fixed;left:0;right:0;top:0;z-index:5;background:rgba(20,20,20,.85);padding:8px;display:flex;gap:8px;align-items:center}button{background:#444;color:#eee;border:0;border-radius:4px;padding:6px 10px;cursor:pointer}#wrap{position:absolute;left:0;right:0;top:42px;bottom:0;overflow:hidden;touch-action:none;cursor:grab}#stage{position:relative;width:100%;transform-origin:0 0}img{width:100%;display:block;user-select:none;-webkit-user-drag:none}canvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}#legend{position:absolute;left:10px;top:52px;z-index:4;background:rgba(0,0,0,.6);padding:5px 7px;border-radius:4px;font-size:13px}#status{margin-left:auto;font-family:monospace;font-size:12px;color:#bbb}
</style></head><body>
<div id='bar'><strong>Camera __CAM__</strong><button onclick='Z(1.25)'>+</button><button onclick='Z(.8)'>-</button><button onclick='R()'>Reset</button><button onclick='FS()'>Full screen</button><button onclick='window.close()'>Close</button><span id='status'></span></div>
<div id='legend'></div><div id='wrap'><div id='stage'><img id='img'><canvas id='ov'></canvas></div></div>
<script>
const CAM='__CAM__';let V={z:1,x:0,y:0},starsegDisplay={inference_enabled:true,overlay_enabled:true,mask_opacity:.22,line_width:2,show_boxes:true,show_masks:true,show_labels:true};const clsColor={'Other_species':'#ffcc33','Prey':'#33ccff','Pycnopodia_helianthoides':'#ff4d88'};let CORN=null;const CE=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
function A(){stage.style.transform=`translate(${V.x}px,${V.y}px) scale(${V.z})`}function Z(f){V.z=Math.max(.25,Math.min(16,V.z*f));A()}function R(){V={z:1,x:0,y:0};A()}function FS(){let e=document.documentElement;if(e.requestFullscreen)e.requestFullscreen();else if(e.webkitRequestFullscreen)e.webkitRequestFullscreen()}
wrap.onwheel=e=>{e.preventDefault();Z(e.deltaY<0?1.2:.833)};let d=0,sx=0,sy=0,ox=0,oy=0;wrap.onpointerdown=e=>{d=1;sx=e.clientX;sy=e.clientY;ox=V.x;oy=V.y;wrap.setPointerCapture(e.pointerId)};wrap.onpointermove=e=>{if(!d)return;V.x=ox+e.clientX-sx;V.y=oy+e.clientY-sy;A()};wrap.onpointerup=()=>d=0;
function draw(latest){let cv=ov,ctx=cv.getContext('2d'),w=img.clientWidth,h=img.clientHeight;if(!w||!h)return;if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h}ctx.clearRect(0,0,w,h);if(CORN&&CORN.length===8){ctx.save();ctx.lineWidth=1.5;ctx.strokeStyle='rgba(0,229,255,.85)';ctx.beginPath();for(const e of CE){ctx.moveTo(CORN[e[0]].x*w,CORN[e[0]].y*h);ctx.lineTo(CORN[e[1]].x*w,CORN[e[1]].y*h)}ctx.stroke();for(let i=0;i<8;i++){let p=CORN[i],x=p.x*w,y=p.y*h;ctx.beginPath();ctx.arc(x,y,9,0,7);ctx.fillStyle=i<4?'#00e5ff':'#ffd400';ctx.fill();ctx.lineWidth=1;ctx.strokeStyle='#001';ctx.stroke();ctx.fillStyle='#000';ctx.font='bold 11px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(i+1),x,y)}ctx.restore();ctx.textAlign='start';ctx.textBaseline='alphabetic'}if(!starsegDisplay.inference_enabled){legend.textContent='StarSeg inference off';return}if(!starsegDisplay.overlay_enabled){legend.textContent='StarSeg overlay off';return}let det=(latest&&latest.detections)||[],iw=latest&&latest.width||img.naturalWidth||w,ih=latest&&latest.height||img.naturalHeight||h,sx=w/iw,sy=h/ih,counts={};for(const q of det){let name=q.class_name||String(q.class_id),col=clsColor[name]||'#66ff66';counts[name]=(counts[name]||0)+1;ctx.lineWidth=Number(starsegDisplay.line_width||2);ctx.strokeStyle=col;ctx.fillStyle=col;let seg=q.segment_xy||[];if(starsegDisplay.show_masks&&seg.length){ctx.beginPath();seg.forEach((p,i)=>{let x=p[0]*sx,y=p[1]*sy;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath();ctx.globalAlpha=Number(starsegDisplay.mask_opacity??.22);ctx.fill();ctx.globalAlpha=1;ctx.stroke()}let b=q.box_xyxy||[];if(b.length===4&&starsegDisplay.show_boxes){let x=b[0]*sx,y=b[1]*sy,bw=(b[2]-b[0])*sx,bh=(b[3]-b[1])*sy;ctx.strokeRect(x,y,bw,bh);if(starsegDisplay.show_labels){let label=`${name} ${(100*(q.confidence||0)).toFixed(0)}%`;ctx.font='13px sans-serif';let tw=ctx.measureText(label).width+8;ctx.fillRect(x,Math.max(0,y-18),tw,18);ctx.fillStyle='#000';ctx.fillText(label,x+4,Math.max(13,y-5));ctx.fillStyle=col}}}legend.textContent=Object.entries(counts).map(([k,v])=>`${k}:${v}`).join('  ')||'StarSeg live: no detections'}
async function tick(){let t=Date.now();img.src=`/preview/${CAM}.jpg?t=${t}`;try{let s=await fetch('/status.json?t='+t).then(r=>r.json());starsegDisplay=s.starseg.display||starsegDisplay;if(s.aquaria_corners)CORN=s.aquaria_corners[CAM]||null;let c=s.cameras[CAM]||{};status.textContent=`${c.state||''} fps=${c.control_fps||''} dropped=${c.dropped||0} late=${c.late||0}`;let latest=await fetch('/starseg/latest.json?t='+t).then(r=>r.json());draw(latest[CAM]||{})}catch(e){status.textContent=e.message}}
tick();setInterval(()=>{if(document.hidden)return;tick()},2000);
</script></body></html>"""
    return body.replace("__CAM__", letter)


class H(BaseHTTPRequestHandler):
    def send_bytes(self, body, content_type="application/octet-stream", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            return
        if path.startswith("/popout/"):
            letter = Path(path).name.upper()
            if letter not in CAMERAS:
                self.send_error(404)
                return
            self.send_bytes(popout_html(letter).encode(), "text/html; charset=utf-8")
            return
        if path == "/status.json":
            with status_lock, control_lock:
                ss = dict(starseg_status)
                ss["display"] = dict(starseg_display)
                ss["available_models"] = list_starseg_models()
                ss["inference_enabled"] = bool(starseg_display.get("inference_enabled", False))
                if STARSEG_ENABLED and not ss["inference_enabled"]:
                    ss["state"] = "paused"
                body = json.dumps({"control": control, "camera_controls": get_camera_controls(), "starseg": ss, "backup": dict(backup_status), "starseg_display": starseg_display, "aquaria_corners": get_aquaria_corners(), "cameras": status}, indent=2).encode()
            self.send_bytes(body, "application/json")
            return
        if path == "/starseg/latest.json":
            with status_lock:
                body = json.dumps(starseg_latest, separators=(",", ":")).encode()
            self.send_bytes(body, "application/json")
            return
        if path.startswith("/preview/"):
            name = Path(path).name
            if name not in {f"{letter}.jpg" for letter in CAMERAS}:
                self.send_error(404)
                return
            f = preview_dir() / name
            if not f.exists():
                self.send_error(404)
                return
            self.send_bytes(f.read_bytes(), mimetypes.guess_type(str(f))[0] or "image/jpeg")
            return
        self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/control"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        data = urllib.parse.parse_qs(self.rfile.read(n).decode())
        try:
            if "recording" in data and data["recording"][0] == "1":
                missing = missing_corner_cameras()
                if missing:
                    raise RuntimeError("Set all 8 aquarium corners before recording (missing: " + ", ".join(missing) + ")")
            with control_lock:
                if "heading" in data:
                    control["heading"] = safe(data["heading"][0])
                if "recording" in data:
                    want_recording = data["recording"][0] == "1"
                    was_recording = control["recording"]
                    control["recording"] = want_recording
                    if want_recording and not was_recording:
                        started = time.time()
                        session = f"recording_start-{iso(started)}"
                        control["recording_session"] = session
                        control["recording_started_iso"] = iso(started)
                        with status_lock:
                            if starseg_status.get("desired_model") != starseg_status.get("model"):
                                starseg_status["reload_requested"] = True
                    elif not want_recording and was_recording:
                        control["recording_session"] = ""
                        control["recording_started_iso"] = ""
            if "starseg_display" in data:
                current = set_starseg_display(data)
                body = json.dumps(current, indent=2).encode()
            elif "starseg_model" in data:
                current = set_starseg_model(data)
                body = json.dumps(current, indent=2).encode()
            elif "aquaria_corners" in data:
                current = set_aquaria_corners(data)
                body = json.dumps(current).encode()
            elif any(k in data for k in ("backup_enabled", "backup_root", "backup_throttle_mbps")):
                current = set_backup_config(data)
                body = json.dumps(current, indent=2).encode()
            elif "camera" in data:
                current = set_camera_controls(data["camera"][0], data)
                body = json.dumps(current, indent=2).encode()
            elif "record_root" in data:
                root = set_record_root(data["record_root"][0])
                body = f"save folder set to {root}".encode()
            else:
                save_config()
                body = b"ok"
            self.send_bytes(body, "text/plain")
        except Exception as e:
            with control_lock:
                control["root_error"] = str(e)
            self.send_bytes(str(e).encode(), "text/plain", status=400)

    def log_message(self, *args):
        pass


def set_backup_config(data):
    updates = {}
    if "backup_enabled" in data:
        updates["enabled"] = parse_bool(data["backup_enabled"][0] if isinstance(data["backup_enabled"], list) else data["backup_enabled"])
    if "backup_root" in data:
        value = data["backup_root"][0] if isinstance(data["backup_root"], list) else data["backup_root"]
        root = Path(str(value).strip()).expanduser()
        if not root.is_absolute():
            raise RuntimeError("backup folder must be an absolute path")
        updates["root"] = str(root.resolve())
    if "backup_throttle_mbps" in data:
        value = data["backup_throttle_mbps"][0] if isinstance(data["backup_throttle_mbps"], list) else data["backup_throttle_mbps"]
        updates["throttle_mbps"] = parse_float(value, "backup_throttle_mbps", 0.0, 1000.0)
    with backup_config_lock:
        backup_config.update(updates)
        current = dict(backup_config)
    with status_lock:
        backup_status.update(enabled=current["enabled"], root=current["root"], throttle_mbps=current["throttle_mbps"], queued=backup_queue.qsize())
    save_config()
    return current


def backup_snapshot():
    with backup_config_lock:
        cfg = dict(backup_config)
    return cfg


def backup_destination(src, source_root, backup_root):
    src = Path(src).resolve()
    source_root = Path(source_root).resolve()
    backup_root = Path(backup_root).expanduser().resolve()
    rel = src.relative_to(source_root)
    if rel.parts and rel.parts[0] in {"preview", "_active_segments"}:
        raise ValueError(f"refusing to back up transient recorder path: {rel}")
    return backup_root / rel


def backup_root_ready(root):
    root = Path(root).expanduser().resolve()
    if str(root).startswith("/media/") or str(root).startswith("/mnt/"):
        parts = root.parts
        candidates = [Path(*parts[:i]) for i in range(len(parts), 1, -1)]
        if not any(p.exists() and os.path.ismount(p) for p in candidates):
            raise RuntimeError(f"backup target is under /media or /mnt but no mounted filesystem was found for {root}")
    elif not root.parent.exists():
        raise RuntimeError(f"backup target parent does not exist: {root.parent}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def copy_file_atomic(src, dst, throttle_mbps=0.0):
    src = Path(src)
    dst = Path(dst)
    st = src.stat()
    if dst.exists() and dst.stat().st_size == st.st_size:
        return "skipped", st.st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    chunk = 1024 * 1024
    copied = 0
    started = time.monotonic()
    with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
        while True:
            buf = fsrc.read(chunk)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)
            if throttle_mbps and throttle_mbps > 0:
                target_elapsed = copied / (throttle_mbps * 1024 * 1024)
                actual_elapsed = time.monotonic() - started
                if target_elapsed > actual_elapsed:
                    time.sleep(min(0.5, target_elapsed - actual_elapsed))
        fdst.flush()
        os.fsync(fdst.fileno())
    shutil.copystat(src, tmp, follow_symlinks=True)
    os.replace(tmp, dst)
    return "copied", copied


def enqueue_backup(paths, source_root):
    cfg = backup_snapshot()
    if not cfg.get("enabled"):
        return False
    backup_root = Path(cfg.get("root") or "")
    if not backup_root:
        return False
    if isinstance(paths, (str, Path)):
        paths = [paths]
    enqueued = 0
    for p in paths:
        p = Path(p)
        if not p.exists() or not p.is_file():
            continue
        try:
            dst = backup_destination(p, source_root, backup_root)
        except Exception as e:
            with status_lock:
                backup_status.update(state="error", errors=backup_status["errors"] + 1, last_error=repr(e))
            continue
        try:
            backup_queue.put_nowait({"src": str(p), "dst": str(dst), "source_root": str(source_root)})
            enqueued += 1
        except queue.Full:
            with status_lock:
                backup_status.update(state="queue full", errors=backup_status["errors"] + 1, last_error=f"backup queue full; dropped {p}")
    with status_lock:
        backup_status.update(queued=backup_queue.qsize())
    return enqueued > 0


def backup_worker():
    while True:
        job = backup_queue.get()
        try:
            cfg = backup_snapshot()
            if not cfg.get("enabled"):
                with status_lock:
                    backup_status.update(state="disabled", queued=backup_queue.qsize())
                continue
            src = Path(job["src"])
            backup_root_ready(cfg["root"])
            dst = Path(job["dst"])
            with status_lock:
                backup_status.update(state="copying", queued=backup_queue.qsize(), root=cfg["root"], throttle_mbps=cfg["throttle_mbps"], last_source=str(src), last_destination=str(dst), last_error="")
            result, nbytes = copy_file_atomic(src, dst, float(cfg.get("throttle_mbps", 0.0)))
            with status_lock:
                if result == "skipped":
                    backup_status["skipped_files"] += 1
                else:
                    backup_status["copied_files"] += 1
                    backup_status["copied_bytes"] += nbytes
                backup_status.update(state="idle", queued=backup_queue.qsize(), last_completed_ts=time.time(), last_error="")
        except Exception as e:
            try:
                tmp = Path(job.get("dst", "")).with_name(f".{Path(job.get('dst', '')).name}.tmp")
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            with status_lock:
                backup_status.update(state="error", queued=backup_queue.qsize(), errors=backup_status["errors"] + 1, last_error=repr(e))
        finally:
            backup_queue.task_done()


def setn(c, n, v):
    try:
        x = getattr(c, n)
        if genicam.IsWritable(x):
            x.SetValue(v)
            return True
    except Exception:
        pass
    return False


def getn(c, n):
    try:
        x = getattr(c, n)
        if genicam.IsReadable(x):
            return x.GetValue()
    except Exception:
        pass
    return None


def starseg_inc(segdir):
    if not STARSEG_ENABLED:
        return
    key = str(segdir)
    with starseg_lock:
        starseg_pending[key] = starseg_pending.get(key, 0) + 1


def starseg_done(segdir):
    if not STARSEG_ENABLED:
        return
    key = str(segdir)
    with starseg_lock:
        starseg_pending[key] = max(0, starseg_pending.get(key, 0) - 1)
        if starseg_pending[key] == 0:
            starseg_pending.pop(key, None)
        starseg_lock.notify_all()


def wait_starseg(segdir):
    if not STARSEG_ENABLED:
        return
    key = str(segdir)
    with starseg_lock:
        while starseg_pending.get(key, 0) > 0:
            starseg_lock.wait(timeout=0.5)


def enqueue_starseg(letter, segdir, frame_index, ts, frame_path, record_result=True, block=True):
    if not STARSEG_ENABLED:
        return False
    with status_lock:
        inference_enabled = bool(starseg_display.get("inference_enabled", False))
    if not inference_enabled:
        with status_lock:
            starseg_status.update(queued=starseg_queue.qsize(), inference_enabled=False)
            status[letter]["starseg_inference_enabled"] = False
        return False
    if record_result and segdir is not None:
        starseg_inc(segdir)
    job = {
        "camera": letter,
        "segdir": str(segdir) if segdir is not None else "",
        "frame_index": frame_index,
        "frame_ts": ts,
        "frame_path": str(frame_path),
        "result_path": str(Path(segdir) / "starseg_frames.jsonl") if record_result and segdir is not None else "",
        "record_result": bool(record_result and segdir is not None),
    }
    try:
        starseg_queue.put(job, block=block)
        with status_lock:
            starseg_status["queued"] = starseg_queue.qsize()
            starseg_status["inference_enabled"] = True
            status[letter]["starseg_queued"] = starseg_queue.qsize()
            status[letter]["starseg_inference_enabled"] = True
        return True
    except queue.Full:
        if record_result and segdir is not None:
            starseg_done(segdir)
        with status_lock:
            starseg_status["queued"] = starseg_queue.qsize()
            status[letter]["starseg_live_dropped"] = status[letter].get("starseg_live_dropped", 0) + 1
        return False
    except Exception:
        if record_result and segdir is not None:
            starseg_done(segdir)
        raise


def summarize_starseg(jsonl_path):
    summary = {"frames": 0, "detections": 0, "classes": {}}
    path = Path(jsonl_path)
    if not path.exists():
        return summary
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        summary["frames"] += 1
        for det in rec.get("detections", []):
            summary["detections"] += 1
            name = det.get("class_name", str(det.get("class_id")))
            summary["classes"][name] = summary["classes"].get(name, 0) + 1
    return summary


def load_starseg_model(path):
    """Load a YOLO model from path, updating starseg_status, and return (model, names, device).

    Raises on failure so callers can decide whether to keep a previously loaded model.
    """
    import torch
    from ultralytics import YOLO
    path = Path(path).expanduser()
    if not path.exists():
        raise RuntimeError(f"model not found: {path}")
    with status_lock:
        starseg_status.update(state="reloading")
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(path))
    names = {int(k): v for k, v in model.names.items()}
    with status_lock:
        starseg_status.update(state="ready", model=str(path), device=str(device), task=str(getattr(model, "task", "")), names=names, inference_enabled=bool(starseg_display.get("inference_enabled", False)), display=dict(starseg_display))
    return model, names, device


def starseg_worker():
    global starseg_active_model
    if not STARSEG_ENABLED:
        with status_lock:
            starseg_status.update(state="disabled", device="none")
        return
    with status_lock:
        startup_model = starseg_status.get("desired_model", str(STARSEG_MODEL))
    if not Path(startup_model).expanduser().exists():
        with status_lock:
            starseg_status.update(state="disabled", device="none", last_error=f"model not found: {startup_model}")
        return
    try:
        model, names, device = load_starseg_model(startup_model)
        starseg_active_model = str(Path(startup_model).expanduser())
    except Exception as e:
        with status_lock:
            starseg_status.update(state="error", last_error=repr(e), errors=starseg_status["errors"] + 1)
        return
    while True:
        job = starseg_queue.get()
        letter = job["camera"]
        segdir = Path(job["segdir"])
        try:
            with status_lock:
                reload_target = starseg_status.get("desired_model") if starseg_status.get("reload_requested") else None
            if reload_target and str(Path(reload_target).expanduser()) != starseg_active_model:
                try:
                    model, names, device = load_starseg_model(reload_target)
                    starseg_active_model = str(Path(reload_target).expanduser())
                    with status_lock:
                        starseg_status["reload_requested"] = False
                except Exception as e:
                    with status_lock:
                        starseg_status.update(state="error", last_error=f"model reload failed, keeping {starseg_active_model}: {e!r}", errors=starseg_status["errors"] + 1, reload_requested=False)
            else:
                with status_lock:
                    if starseg_status.get("reload_requested"):
                        starseg_status["reload_requested"] = False
            with status_lock:
                predict_conf = float(starseg_display.get("conf", STARSEG_CONF_DEFAULT))
                predict_imgsz = int(starseg_display.get("imgsz", STARSEG_IMGSZ_DEFAULT))
            result = model.predict(source=job["frame_path"], device=device, imgsz=predict_imgsz, conf=predict_conf, verbose=False)[0]
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
            h, w = result.orig_shape[:2] if getattr(result, "orig_shape", None) else (0, 0)
            rec = {
                "camera": letter,
                "frame_index": job["frame_index"],
                "frame_ts": job["frame_ts"],
                "frame_file": Path(job["frame_path"]).name,
                "model": starseg_active_model,
                "device": str(device),
                "imgsz": predict_imgsz,
                "confidence_threshold": predict_conf,
                "width": int(w),
                "height": int(h),
                "detections": detections,
            }
            if job.get("result_path"):
                with open(job["result_path"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            with status_lock:
                starseg_latest[letter] = {**rec, "updated_ts": time.time()}
                starseg_status.update(state="running", queued=starseg_queue.qsize(), processed=starseg_status["processed"] + 1, last_camera=letter, last_frame=job["frame_path"], last_detections=len(detections), last_error="", inference_enabled=bool(starseg_display.get("inference_enabled", False)), display=dict(starseg_display))
                status[letter].update(starseg_processed=status[letter].get("starseg_processed", 0) + 1, starseg_last_detections=len(detections), starseg_last_frame=Path(job["frame_path"]).name, starseg_inference_enabled=bool(starseg_display.get("inference_enabled", False)))
        except Exception as e:
            with status_lock:
                starseg_status.update(state="error", queued=starseg_queue.qsize(), errors=starseg_status["errors"] + 1, last_error=repr(e))
                status[letter].update(starseg_error=repr(e))
        finally:
            if job.get("record_result"):
                starseg_done(segdir)
            starseg_queue.task_done()


def finalize(letter, segdir, meta):
    segdir = Path(segdir)
    outfile = None
    try:
        wait_starseg(segdir)
        root = Path(meta["record_root"])
        ensure_root(root)
        duration = max(0, meta["end_ts"] - meta["start_ts"])
        heading = safe(meta["heading"])
        size = f"{meta.get('width')}x{meta.get('height')}"
        stem = f"{heading}_camera-{letter}_start-{iso(meta['start_ts'])}_duration-{duration:.3f}s_fps-{meta['fps']:g}_frames-{meta['frames']}_size-{size}"
        session = meta.get("recording_session") or iso(meta["start_ts"])
        video_dir = root / letter / session / stem
        video_dir.mkdir(parents=True, exist_ok=True)
        outfile = video_dir / f"{stem}.mp4"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(meta["fps"]), "-i", str(segdir / "%08d.jpg"), "-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(X264_CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(outfile)]
        subprocess.run(cmd, check=True)
        starseg_src = Path(segdir) / "starseg_frames.jsonl"
        starseg_out = video_dir / f"{stem}_starseg_frames.jsonl"
        starseg_summary_out = video_dir / f"{stem}_starseg_summary.json"

        if starseg_src.exists():
            shutil.move(str(starseg_src), str(starseg_out))
        else:
            starseg_out.write_text("")

        starseg_summary = summarize_starseg(starseg_out)
        with status_lock:
            summary_conf = float(starseg_display.get("conf", STARSEG_CONF_DEFAULT))
            summary_imgsz = int(starseg_display.get("imgsz", STARSEG_IMGSZ_DEFAULT))
        starseg_summary.update({"model": str(STARSEG_MODEL), "confidence_threshold": summary_conf, "imgsz": summary_imgsz, "frames_jsonl": str(starseg_out)})
        starseg_summary_out.write_text(json.dumps(starseg_summary, indent=2))
        meta.update({"duration_seconds": duration, "file": str(outfile), "folder": str(video_dir), "ended_iso": iso(meta["end_ts"]), "starseg_frames_jsonl": str(starseg_out), "starseg_summary_json": str(starseg_summary_out), "starseg_summary": starseg_summary})
        meta_out = outfile.with_suffix(".json")
        meta_out.write_text(json.dumps(meta, indent=2))
        # Only finalized artifacts are enqueued for backup. Active JPEG staging,
        # previews, and tmp files are never copied.
        enqueue_backup([outfile, starseg_out, starseg_summary_out, meta_out], root)
        shutil.rmtree(segdir, ignore_errors=True)
        with status_lock:
            status[letter].pop("finalize_error", None)
            status[letter]["last_finalized_file"] = str(outfile)

            status[letter]["last_finalized_cleanup"] = "deleted staging after successful mp4"
    except Exception as e:
        cleanup = "kept failed staging"
        if DELETE_FAILED_SEGMENTS:
            shutil.rmtree(segdir, ignore_errors=True)
            cleanup = "deleted failed staging"
        with status_lock:
            status[letter]["finalize_error"] = repr(e)
            status[letter]["last_finalized_file"] = str(outfile) if outfile else ""
            status[letter]["last_finalized_cleanup"] = cleanup


def set_if_writable(cam, node, value):
    try:
        x = getattr(cam, node)
        if genicam.IsWritable(x):
            x.SetValue(value)
            return True
    except Exception:
        pass
    return False


def apply_camera_controls(cam, letter):
    with camera_control_lock:
        cfg = dict(camera_controls[letter])
    set_if_writable(cam, "TestImageSelector", "Off")
    set_if_writable(cam, "AcquisitionFrameRateEnable", True)
    set_if_writable(cam, "AcquisitionFrameRate", float(cfg["fps"]))
    set_if_writable(cam, "ExposureAuto", cfg["exposure_auto"])
    set_if_writable(cam, "AutoExposureTimeUpperLimit", float(cfg["auto_exposure_max_us"]))
    if cfg["exposure_auto"] == "Off":
        set_if_writable(cam, "ExposureTime", float(cfg["exposure_time_us"]))
    set_if_writable(cam, "GainAuto", cfg["gain_auto"])
    set_if_writable(cam, "AutoGainUpperLimit", float(cfg["auto_gain_max_db"]))
    if cfg["gain_auto"] == "Off":
        set_if_writable(cam, "Gain", float(cfg["gain_db"]))
    set_if_writable(cam, "AutoTargetBrightness", float(cfg["target_brightness"]))
    set_if_writable(cam, "BalanceWhiteAuto", cfg["white_balance_auto"])
    set_if_writable(cam, "BslLightSourcePreset", cfg["light_source_preset"])
    live = {
        "control_fps": cfg["fps"],
        "exposure_auto_value": getn(cam, "ExposureAuto"),
        "exposure_time_us": getn(cam, "ExposureTime"),
        "auto_exposure_max_us": getn(cam, "AutoExposureTimeUpperLimit"),
        "gain_auto_value": getn(cam, "GainAuto"),
        "gain_db": getn(cam, "Gain"),
        "auto_gain_max_db": getn(cam, "AutoGainUpperLimit"),
        "target_brightness": getn(cam, "AutoTargetBrightness"),
        "white_balance_auto_value": getn(cam, "BalanceWhiteAuto"),
        "light_source_preset_value": getn(cam, "BslLightSourcePreset"),
    }
    return cfg, live


def acq(letter, serial):
    f = pylon.TlFactory.GetInstance()
    while True:
        cam = None
        try:
            dev = [d for d in f.EnumerateDevices() if d.GetSerialNumber() == serial]
            if not dev:
                with status_lock:
                    status[letter].update(state="not found", error="device not enumerated", ts=time.time())
                time.sleep(2)
                continue
            cam = pylon.InstantCamera(f.CreateDevice(dev[0]))
            cam.Open()
            cfg, live = apply_camera_controls(cam, letter)
            with status_lock:
                status[letter].update(state="acquiring", error="", model=dev[0].GetModelName(), width=getn(cam, "Width"), height=getn(cam, "Height"), pixel_format=getn(cam, "PixelFormat"), camera_controls=cfg, **live)
            cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
            due = time.monotonic()
            seq = 0
            last_apply = 0
            while cam.IsGrabbing():
                now = time.monotonic()
                if now - last_apply >= 1.0:
                    cfg, live = apply_camera_controls(cam, letter)
                    period = 1 / max(0.1, float(cfg["fps"]))
                    last_apply = now
                    with status_lock:
                        status[letter].update(camera_controls=cfg, **live)
                r = cam.RetrieveResult(int(max(3000, period * 2000)), pylon.TimeoutHandling_ThrowException)
                try:
                    if r.GrabSucceeded():
                        queues[letter].put((seq, time.time(), r))
                        r = None
                        seq += 1
                        due += period
                    with status_lock:
                        status[letter].update(acquired=seq, queued=queues[letter].qsize(), ts=time.time())
                finally:
                    if r is not None:
                        r.Release()
                if due - time.monotonic() > 0:
                    time.sleep(due - time.monotonic())
        except Exception as e:
            with status_lock:
                status[letter].update(state="acq error", error=repr(e), ts=time.time())
            time.sleep(2)
        finally:
            try:
                if cam and cam.IsGrabbing():
                    cam.StopGrabbing()
                if cam and cam.IsOpen():
                    cam.Close()
            except Exception:
                pass

def enc(letter):
    conv = pylon.ImageFormatConverter()
    conv.OutputPixelFormat = pylon.PixelType_RGB8packed
    conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    segdir = None
    meta = None
    local = 0
    while True:
        seq, ts, r = queues[letter].get()
        try:
            with control_lock:
                rec = control["recording"]
                heading = control["heading"]
                root = Path(control["record_root"])
                recording_session = control.get("recording_session") or f"recording_start-{iso(ts)}"
                recording_started_iso = control.get("recording_started_iso") or iso(ts)
            ensure_root(root)
            img = conv.Convert(r)
            ptmp = preview_dir(root) / f".{letter}.jpg.tmp"
            # Preview JPEGs are for the browser only; keep them smaller/lighter than
            # recorded frames so a dashboard left open does not compete with capture.
            save_jpeg(img, ptmp, PREVIEW_JPEG_QUALITY)
            preview_path = preview_dir(root) / f"{letter}.jpg"
            os.replace(ptmp, preview_path)
            if not rec:
                enqueue_starseg(letter, None, seq, ts, preview_path, record_result=False, block=False)
                if segdir is not None:
                    meta["end_ts"] = ts
                    meta["frames"] = local
                    threading.Thread(target=finalize, args=(letter, segdir, meta.copy()), daemon=True).start()
                    segdir = None
                    meta = None
                    local = 0
                with status_lock:
                    status[letter].update(state="preview only", queued=queues[letter].qsize(), recording=False, folder=str(root / letter), recording_session="", recording_folder="", ts=time.time())
            else:
                if segdir is None or ts - meta["start_ts"] >= SEGMENT_SECONDS or Path(meta["record_root"]) != root or meta.get("recording_session") != recording_session:
                    if segdir is not None:
                        meta["end_ts"] = ts
                        meta["frames"] = local
                        threading.Thread(target=finalize, args=(letter, segdir, meta.copy()), daemon=True).start()
                    local = 0
                    segdir = active_dir(root) / letter / f"{int(ts)}"
                    segdir.mkdir(parents=True, exist_ok=True)
                    with status_lock:
                        width = status[letter].get("width")
                        height = status[letter].get("height")
                        model = status[letter].get("model")
                        serial = status[letter].get("serial")
                        meta_fps = status[letter].get("control_fps", FPS)
                    corner_pts = get_aquaria_corners().get(letter)
                    aquaria = None
                    if corner_pts:
                        aquaria = {
                            "marker_scheme": "1-4 top rim (outer quad, clockwise from top-left), 5-8 bottom floor (inner quad, matching order)",
                            "image_width": width,
                            "image_height": height,
                            "normalized": corner_pts,
                            "absolute": [{"n": p["n"], "x": (p["x"] * width) if width else None, "y": (p["y"] * height) if height else None} for p in corner_pts],
                        }
                    meta = {"heading": heading, "camera": letter, "serial": serial, "model": model, "recording_session": recording_session, "recording_started_iso": recording_started_iso, "start_ts": ts, "started_iso": iso(ts), "requested_segment_seconds": SEGMENT_SECONDS, "fps": meta_fps, "width": width, "height": height, "record_root": str(root), "aquaria_corners": aquaria}
                frame = segdir / f"{local:08d}.jpg"
                tmp = frame.with_suffix(".tmp")
                save_jpeg(img, tmp)
                os.replace(tmp, frame)
                enqueue_starseg(letter, segdir, local, ts, frame)
                local += 1
                with status_lock:
                    status[letter].update(state="recording", recording=True, encoded=status[letter]["encoded"] + 1, queued=queues[letter].qsize(), folder=str(root / letter), recording_session=recording_session, recording_folder=str(root / letter / recording_session), active_segment=str(segdir), current_segment_frame=local, heading=heading, ts=time.time())
        except Exception as e:
            with status_lock:
                status[letter].update(state="enc error", error=repr(e))
        finally:
            try:
                r.Release()
            except Exception:
                pass
            queues[letter].task_done()


def main():
    threading.Thread(target=starseg_worker, daemon=True).start()
    threading.Thread(target=backup_worker, daemon=True).start()
    with status_lock:
        backup_status.update(state="idle", queued=backup_queue.qsize())
    for l, s in CAMERAS.items():
        threading.Thread(target=acq, args=(l, s), daemon=True).start()
        threading.Thread(target=enc, args=(l,), daemon=True).start()
    print(f"Basler loop recorder at http://127.0.0.1:{PORT}/ root={get_root()} segment_seconds={SEGMENT_SECONDS}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


if __name__ == "__main__":
    main()


