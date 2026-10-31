#!/usr/bin/env python3
"""Poke Trial Logger: log live poke-trial events from a browser page into a CSV file.

Run it with:

    python3 poke_logger.py

It needs only the Python standard library. The page opens at http://127.0.0.1:8766,
and only this computer can reach it. See README.md for details.
"""

import argparse
import csv
import errno
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import urllib.request
import webbrowser
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

TANKS = ("starbase", "starbot", "starsense")
CAMERAS = ("A", "B", "C", "D")
CAMERA_D_TANK = "starbot"  # camera D exists only on this tank
POKERS = ("Poker A", "Poker B")
STAR_STATES = ("quiescent", "alert", "moving")  # optional: the annotator's call on the star before poking
EVENTS = (
    ("entry_time", "Entry"),
    ("first_poke_time", "First poke"),
    ("first_movement_time", "First movement"),
    ("movement_away_time", "Movement away"),
    ("exit_time", "Exit"),
)
EVENT_FIELDS = tuple(field for field, _ in EVENTS)
REQUIRED_EVENTS = ("entry_time",)  # the rest can be left blank if they didn't happen
TIMER_START_EVENT = "first_poke_time"  # the on-page poke timer counts from this event...
TIMER_STOP_EVENTS = ("movement_away_time", "exit_time")  # ...until the first of these is recorded
POKE_LIMIT_SECONDS = 5 * 60  # informational: when the timer says it's time to stop poking
CAMERA_REST_SECONDS = 30 * 60  # a camera can't be chosen again until this long after its last trial's last event
COLUMNS = (
    "record_id",
    "trial_number",
    "tank_id",
    "camera_id",
    "poker_id",
    "pre_poke_state",
    *EVENT_FIELDS,
    "notes",
    "logged_at",
    "edited_at",
)
DEFAULT_CSV = Path.home() / "poke_logs" / "poke_log.csv"
DEFAULT_PORT = 8766
ICON_PATH = Path(__file__).resolve().with_name("icon.svg")
CAMERAS_CONFIG = Path(__file__).resolve().with_name("cameras.json")
REFERENCE_DIR = Path(__file__).resolve().with_name("reference_images")
MAX_NOTES = 1000
MAX_BODY = 64 * 1024
ACTIONS_SHOWN = 500  # how many activity-log entries the page loads


class ValidationError(ValueError):
    """A submitted record broke one or more rules."""

    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


class StoreError(RuntimeError):
    """The CSV file can't be used."""


class RecordNotFound(LookupError):
    """No record has the requested ID."""


class RequestError(Exception):
    """A request the server refuses, with the status to send back."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def now_iso():
    """Current local time in ISO 8601, with milliseconds and the UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def validate_record(data):
    """Check a submitted record and return its cleaned fields, or raise ValidationError."""
    if not isinstance(data, dict):
        raise ValidationError(["Expected a JSON object."])
    errors = []
    clean = {}

    trial = str(data.get("trial_number", "")).strip()
    if re.fullmatch(r"[0-9]+", trial) and int(trial) > 0:
        clean["trial_number"] = str(int(trial))
    else:
        errors.append("Trial number must be a whole number of 1 or more.")

    tank = data.get("tank_id")
    if tank in TANKS:
        clean["tank_id"] = tank
    else:
        errors.append(f"Tank must be one of: {', '.join(TANKS)}.")

    camera = data.get("camera_id")
    if camera not in CAMERAS:
        errors.append(f"Camera must be one of: {', '.join(CAMERAS)}.")
    elif camera == "D" and tank != CAMERA_D_TANK:
        errors.append(f"Camera D is only available on {CAMERA_D_TANK}.")
    else:
        clean["camera_id"] = camera

    poker = data.get("poker_id")
    if poker in POKERS:
        clean["poker_id"] = poker
    else:
        errors.append(f"Poker must be one of: {', '.join(POKERS)}.")

    star_state = data.get("pre_poke_state") or ""
    if star_state in ("", *STAR_STATES):
        clean["pre_poke_state"] = star_state
    else:
        errors.append(f"Star state before poking must be one of: {', '.join(STAR_STATES)} (or left blank).")

    for field, label in EVENTS:
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            # Later events may never happen (e.g. the star doesn't move before poking stops).
            if field in REQUIRED_EVENTS:
                errors.append(f"{label} time is missing.")
            else:
                clean[field] = ""
            continue
        if not isinstance(value, str):
            errors.append(f"{label} time is not a valid time.")
            continue
        try:
            moment = datetime.fromisoformat(value.strip())
        except ValueError:
            errors.append(f"{label} time is not a valid time.")
            continue
        if moment.tzinfo is None:
            errors.append(f"{label} time needs a UTC offset.")
            continue
        clean[field] = moment.isoformat(timespec="milliseconds")

    notes = data.get("notes", "")
    if not isinstance(notes, str):
        errors.append("Notes must be text.")
    else:
        # Keep notes on one line so spreadsheets show one row per record.
        notes = " ".join(notes.split())
        if len(notes) > MAX_NOTES:
            errors.append(f"Notes can be at most {MAX_NOTES} characters.")
        else:
            clean["notes"] = notes

    if errors:
        raise ValidationError(errors)
    return clean


def append_line(path, line):
    """Durably append one line (ending in a newline) to a file.

    If the file doesn't end in a newline (a write cut off by a crash or power
    failure), the new line starts fresh instead of being glued to the broken one.
    If this write itself fails part-way, the file is cut back to its old size so
    no half-written line is left behind.
    """
    data = line.encode("utf-8")
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        size = os.lseek(fd, 0, os.SEEK_END)
        if size and os.pread(fd, 1, size - 1) != b"\n":
            data = b"\r\n" + data if line.endswith("\r\n") else b"\n" + data
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        except BaseException:
            os.ftruncate(fd, size)
            raise
    finally:
        os.close(fd)


def fsync_dir(directory):
    """Make a rename in this directory survive a power failure."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _as_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class RecordStore:
    """The CSV file of records, plus a change log (JSON lines) of every edit and deletion."""

    def __init__(self, csv_path):
        self.path = Path(csv_path).expanduser().resolve()
        self.changes_path = self.path.with_name(self.path.stem + "_changes.jsonl")
        self.actions_path = self.path.with_name(self.path.stem + "_actions.jsonl")
        self._lock = threading.RLock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > 0:
                self._check_header()
            else:
                self._write_rows([])
            # Unfinished copies from a rewrite interrupted by a crash; the CSV itself was never touched.
            for leftover in self.path.parent.glob(f".{self.path.name}.*.tmp"):
                leftover.unlink(missing_ok=True)
        except OSError as exc:
            raise StoreError(f"Can't use {self.path}: {exc.strerror or exc}") from exc
        self._next_id = self._find_next_id()

    def _check_header(self):
        with self.path.open(newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), [])
        if tuple(header) == COLUMNS:
            return
        if self._is_older_layout(header):
            self._upgrade(header)
            return
        raise StoreError(
            f"{self.path} has different columns from the ones this logger writes, "
            "so it won't add to it. Move that file aside, or choose another file with --csv."
        )

    @staticmethod
    def _is_older_layout(header):
        """A file from an earlier version: only known columns, in order, just missing newer ones."""
        return (len(set(header)) == len(header) and set(header) < set(COLUMNS)
                and {"record_id", "entry_time"} <= set(header)
                and list(header) == [c for c in COLUMNS if c in header])

    def _upgrade(self, header):
        """Add newly introduced columns (blank for existing rows), keeping a copy of the original."""
        added = [c for c in COLUMNS if c not in header]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = self.path.with_name(f"{self.path.name}.before-upgrade-{stamp}")
        shutil.copy2(self.path, backup)
        self._write_rows(self.read_rows())
        print(f"Upgraded {self.path.name}: added column(s) {', '.join(added)}. "
              f"The original is kept as {backup.name}.", file=sys.stderr)

    def _find_next_id(self):
        """One more than the highest ID ever used, including deleted records."""
        ids = [_as_id(row.get("record_id")) for row in self.read_rows()]
        if self.changes_path.exists():
            with self.changes_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        ids.append(_as_id(json.loads(line).get("record_id")))
                    except (ValueError, AttributeError):
                        pass
        return max((i for i in ids if i is not None), default=0) + 1

    def read_rows(self):
        with self._lock, self.path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def add(self, data):
        clean = validate_record(data)
        with self._lock:
            row = {"record_id": str(self._next_id), **clean, "logged_at": now_iso(), "edited_at": ""}
            text = io.StringIO()
            csv.DictWriter(text, fieldnames=COLUMNS).writerow(row)
            append_line(self.path, text.getvalue())
            self._next_id += 1
        return row

    def update(self, record_id, data):
        clean = validate_record(data)
        with self._lock:
            rows = self.read_rows()
            index = self._index_of(rows, record_id)
            before = rows[index]
            after = {**before, **clean, "edited_at": now_iso()}
            rows[index] = after
            # Log first: if the rewrite then fails, the old values are still on record.
            self._log_change("edit", before, after)
            self._write_rows(rows)
        return {key: after.get(key, "") for key in COLUMNS}

    def delete(self, record_id):
        with self._lock:
            rows = self.read_rows()
            before = rows.pop(self._index_of(rows, record_id))
            self._log_change("delete", before, None)
            self._write_rows(rows)
        return {key: before.get(key, "") for key in COLUMNS}

    @staticmethod
    def _index_of(rows, record_id):
        wanted = _as_id(record_id)
        for index, row in enumerate(rows):
            if wanted is not None and _as_id(row.get("record_id")) == wanted:
                return index
        raise RecordNotFound(record_id)

    def _write_rows(self, rows):
        """Replace the whole CSV safely: write a temporary file, then swap it in."""
        mode = self.path.stat().st_mode & 0o777 if self.path.exists() else 0o644
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", restval="")
                writer.writeheader()
                writer.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, mode)
            os.replace(tmp, self.path)
            fsync_dir(self.path.parent)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _log_change(self, action, before, after):
        change = {
            "at": now_iso(),
            "action": action,
            "record_id": before.get("record_id", ""),
            "before": {key: before.get(key, "") for key in COLUMNS},
        }
        if after is not None:
            change["after"] = {key: after.get(key, "") for key in COLUMNS}
        append_line(self.changes_path, json.dumps(change) + "\n")

    def log_action(self, data):
        """Append one page action (with the form as it stood afterwards) to the activity log."""
        if not isinstance(data, dict):
            raise ValidationError(["Expected a JSON object."])
        action, summary, form = data.get("action"), data.get("summary", ""), data.get("form", {})
        if not isinstance(action, str) or not 0 < len(action) <= 40:
            raise ValidationError(["Action must be a short name."])
        if not isinstance(summary, str) or len(summary) > 1000:
            raise ValidationError(["Summary must be text of at most 1000 characters."])
        if not isinstance(form, dict):
            raise ValidationError(["Form must be a JSON object."])
        entry = {"at": now_iso(), "action": action, "summary": summary, "form": form}
        with self._lock:
            append_line(self.actions_path, json.dumps(entry) + "\n")
        return entry

    def recent_actions(self, limit=ACTIONS_SHOWN):
        """The newest `limit` activity-log entries, oldest first."""
        if not self.actions_path.exists():
            return []
        entries = []
        with self._lock, self.actions_path.open(encoding="utf-8") as f:
            for line in deque(f, maxlen=limit):
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    pass
        return entries


def cameras_of(tank):
    return [c for c in CAMERAS if c != "D" or tank == CAMERA_D_TANK]


def reference_images(directory=None):
    """{"<tank>_<camera>": time the still was grabbed} for every reference image on disk."""
    directory = directory or REFERENCE_DIR
    found = {}
    for tank in TANKS:
        for camera in cameras_of(tank):
            path = directory / f"{tank}_{camera}.jpg"
            if path.is_file():
                found[f"{tank}_{camera}"] = (
                    datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"))
    return found


def grab_references(config_path=CAMERAS_CONFIG, directory=REFERENCE_DIR):
    """Save one still per camera from the camera monitor, as reference_images/<tank>_<camera>.jpg."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    monitor = config["monitor"].rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    directory.mkdir(exist_ok=True)
    ok = True
    for tank in TANKS:
        for camera in cameras_of(tank):
            feed = config["feeds"].get(tank, {}).get(camera, "")
            name = f"{tank} camera {camera}"
            if not feed:
                print(f"- {name}: no feed set in {config_path.name}, skipped")
                continue
            host, _, feed_camera = feed.partition("/")
            url = f"{monitor}/feeds/{host}/preview/{urllib.parse.quote(feed_camera)}.jpg"
            try:
                with opener.open(url, timeout=15) as res:
                    image = res.read()
                if not image.startswith(b"\xff\xd8"):
                    raise ValueError("the monitor didn't return a JPEG image")
            except (OSError, ValueError) as exc:
                print(f"- {name}: couldn't grab from {url} ({exc})")
                ok = False
                continue
            target = directory / f"{tank}_{camera}.jpg"
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(image)
            os.replace(tmp, target)
            print(f"- {name}: saved {target.name} ({len(image) // 1024} KB)")
    return ok


def display_path(path):
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


class Handler(BaseHTTPRequestHandler):
    server_version = "PokeTrialLogger/1.0"

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def log_request(self, code="-", size="-"):
        # Keep the terminal quiet; only report failed requests.
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)

    def _dispatch(self, method):
        store = self.server.store
        try:
            self._check_source()
            path = urlsplit(self.path).path
            record_path = re.fullmatch(r"/api/records/([0-9]+)", path)
            if method == "GET" and path == "/":
                self._send(HTTPStatus.OK, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if method == "GET" and path == "/icon.svg" and ICON_PATH.is_file():
                self._send(HTTPStatus.OK, ICON_PATH.read_bytes(), "image/svg+xml")
                return
            reference = re.fullmatch(r"/reference/([a-z]+_[A-Z])\.jpg", path)
            if method == "GET" and reference and reference[1] in reference_images():
                self._send(HTTPStatus.OK, (REFERENCE_DIR / f"{reference[1]}.jpg").read_bytes(), "image/jpeg")
                return
            if method == "GET" and path == "/api/meta":
                status, body = HTTPStatus.OK, self._meta()
            elif method == "GET" and path == "/api/records":
                status, body = HTTPStatus.OK, {"records": store.read_rows()}
            elif method == "GET" and path == "/api/actions":
                status, body = HTTPStatus.OK, {"actions": store.recent_actions()}
            elif method == "POST" and path == "/api/actions":
                status, body = HTTPStatus.CREATED, {"action": store.log_action(self._read_json())}
            elif method == "POST" and path == "/api/records":
                status, body = HTTPStatus.CREATED, {"record": store.add(self._read_json())}
            elif method == "PUT" and record_path:
                status, body = HTTPStatus.OK, {"record": store.update(record_path[1], self._read_json())}
            elif method == "DELETE" and record_path:
                status, body = HTTPStatus.OK, {"record": store.delete(record_path[1])}
            else:
                raise RequestError(HTTPStatus.NOT_FOUND, "Not found.")
        except RequestError as exc:
            status, body = exc.status, {"errors": [exc.message]}
        except ValidationError as exc:
            status, body = HTTPStatus.UNPROCESSABLE_ENTITY, {"errors": exc.errors}
        except RecordNotFound:
            status, body = HTTPStatus.NOT_FOUND, {"errors": ["That record no longer exists."]}
        except OSError as exc:
            status, body = HTTPStatus.INTERNAL_SERVER_ERROR, {"errors": [f"Couldn't use the CSV file: {exc}"]}
        self._send(status, json.dumps(body).encode("utf-8"), "application/json")

    def _check_source(self):
        """Refuse requests that didn't come from this logger's own page.

        Checking Host blocks DNS-rebinding tricks; checking Origin stops other
        websites open in the browser from writing to the log.
        """
        port = self.server.server_port
        allowed = {f"localhost:{port}", f"127.0.0.1:{port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in allowed or (origin is not None and urlsplit(origin).netloc not in allowed):
            raise RequestError(HTTPStatus.FORBIDDEN, "Requests must come from the logger page.")

    def _read_json(self):
        if self.headers.get_content_type() != "application/json":
            raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send the record as JSON.")
        length = _as_id(self.headers.get("Content-Length"))
        if length is None or not 0 < length <= MAX_BODY:
            raise RequestError(HTTPStatus.BAD_REQUEST, "The request body is missing or too large.")
        try:
            return json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            raise RequestError(HTTPStatus.BAD_REQUEST, "The request body isn't valid JSON.") from None

    def _meta(self):
        return {
            "tanks": TANKS,
            "cameras": CAMERAS,
            "camera_d_tank": CAMERA_D_TANK,
            "pokers": POKERS,
            "star_states": STAR_STATES,
            "events": [{"field": field, "label": label} for field, label in EVENTS],
            "required_events": REQUIRED_EVENTS,
            "timer": {"start": TIMER_START_EVENT, "stop": TIMER_STOP_EVENTS, "limit_seconds": POKE_LIMIT_SECONDS},
            "camera_rest_seconds": CAMERA_REST_SECONDS,
            "csv_path": display_path(self.server.store.path),
            "actions_path": display_path(self.server.store.actions_path),
            "references": reference_images(),
        }

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def make_server(store, port=DEFAULT_PORT):
    """Create the web server, reachable from this computer only."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.store = store
    return server


def logger_running(port):
    """True if a Poke Trial Logger is already answering on this port."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/api/meta", timeout=2) as res:
            data = json.load(res)
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and "events" in data


def report_error(message):
    """Print an error, and also show it as a desktop notification when there's no terminal to see it."""
    print(f"Error: {message}", file=sys.stderr)
    if not sys.stderr.isatty() and shutil.which("notify-send"):
        try:
            subprocess.run(["notify-send", "Poke Trial Logger", message], timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            pass


def open_page(url, browser=None):
    """Open the page in the named browser (or the default one) without holding up the server."""
    def run():
        try:
            (webbrowser.get(browser) if browser else webbrowser).open(url)
        except webbrowser.Error:
            report_error(f"Couldn't open {browser}. Open {url} in a browser yourself.")

    threading.Timer(0.5, run).start()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Log live poke-trial events to a CSV file.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help=f"CSV file to write (default: {display_path(DEFAULT_CSV)})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port for the page (default: {DEFAULT_PORT})")
    parser.add_argument("--browser", help="browser to open the page in, e.g. firefox (default: your default browser)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the page in a browser")
    parser.add_argument("--grab-references", action="store_true",
                        help="save a reference still from each camera (see cameras.json), then exit")
    args = parser.parse_args(argv)

    if args.grab_references:
        return 0 if grab_references() else 1

    try:
        store = RecordStore(args.csv)
    except StoreError as exc:
        report_error(str(exc))
        return 1
    try:
        server = make_server(store, args.port)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        if not logger_running(args.port):
            report_error(f"Port {args.port} is in use by another program. Pick another port with --port.")
            return 1
        # Already running (e.g. the desktop icon was clicked again): just show the page.
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Poke Trial Logger is already running at {url}")
        if not args.no_browser:
            open_page(url, args.browser)
        return 0

    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Poke Trial Logger is running at {url}")
    print(f"Saving to {store.path}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        open_page(url, args.browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Poke Trial Logger</title>
<link rel="icon" type="image/svg+xml" href="/icon.svg">
<style>
  :root {
    --bg: #f4f5f7; --panel: #ffffff; --text: #1d2430; --muted: #5d6877; --border: #d5dae1;
    --accent: #2456c4; --accent-text: #ffffff; --done-bg: #e6f2ea; --done: #2b7445;
    --error: #b3261e; --error-bg: #fdeceb; --edit-bg: #fff3d1; --edit: #7a5600;
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --track: #c4c9d1; --grid: #e6e8ec;
    color-scheme: light;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14171c; --panel: #1d2128; --text: #e7eaee; --muted: #9aa4b1; --border: #363c46;
      --accent: #6d9cff; --accent-text: #0d1320; --done-bg: #17321f; --done: #6fcf97;
      --error: #ff8a80; --error-bg: #3a1d1b; --edit-bg: #3a3017; --edit: #f0cf6a;
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --track: #4b515b; --grid: #2c313a;
      color-scheme: dark;
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 16px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { display: flex; justify-content: space-between; align-items: baseline; gap: 16px;
           max-width: 980px; margin: 0 auto; padding: 20px 16px 10px; }
  h1 { font-size: 1.4rem; margin: 0; }
  h2 { font-size: 1.05rem; margin: 0; }
  .clock { font: 600 1.5rem ui-monospace, "SFMono-Regular", Menlo, monospace; }
  main { max-width: 980px; margin: 0 auto; padding: 0 16px 32px; display: grid; gap: 16px; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; min-width: 0; }
  .label, legend { display: block; font-weight: 600; font-size: .85rem; color: var(--muted);
                   margin: 0 0 6px; padding: 0; text-transform: uppercase; letter-spacing: .04em; }
  small { display: block; color: var(--muted); font-size: .85rem; margin-top: 6px; }

  #setup { display: grid; grid-template-columns: minmax(150px, 1fr) repeat(3, minmax(170px, 1.3fr)); gap: 16px 20px; }
  @media (max-width: 760px) { #setup { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 460px) { #setup { grid-template-columns: 1fr; } }
  fieldset { border: 0; margin: 0; padding: 0; min-width: 0; }
  .choices { display: flex; flex-wrap: wrap; gap: 6px; }
  .choice { position: relative; }
  .choice input { position: absolute; opacity: 0; pointer-events: none; }
  .choice span { display: inline-block; padding: 8px 14px; border: 1px solid var(--border); border-radius: 8px;
                 cursor: pointer; user-select: none; background: var(--bg); }
  .choice input:checked + span { background: var(--accent); color: var(--accent-text); border-color: var(--accent); font-weight: 600; }
  .choice input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 2px; }
  .choice input:disabled + span { opacity: .4; cursor: not-allowed; }
  .choice.resting span { background: var(--edit-bg); color: var(--edit); border: 1px dashed var(--edit); }
  .choice.resting input:disabled + span { opacity: 1; }
  .choice .wait { font-style: normal; font-size: .8rem; margin-left: 6px; font-variant-numeric: tabular-nums; }
  .rest-hint { color: var(--edit); }

  input[type=text] { font: inherit; color: inherit; background: var(--bg); border: 1px solid var(--border);
                     border-radius: 8px; padding: 8px 10px; width: 100%; }
  input[type=text]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  #trial { font-size: 1.35rem; font-weight: 700; padding-right: 40px; }
  .combo { position: relative; max-width: 170px; }
  .combo-toggle { position: absolute; right: 4px; top: 50%; transform: translateY(-50%); width: 32px; height: 32px;
                  border: 0; border-radius: 6px; background: transparent; color: var(--muted); font-size: 1rem; cursor: pointer; }
  .combo-toggle:hover { color: var(--text); background: var(--panel); }
  .combo-list { position: absolute; z-index: 10; left: 0; top: calc(100% + 4px); min-width: 220px; max-height: 260px; overflow-y: auto; margin: 0; padding: 4px; list-style: none; background: var(--panel);
                border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 8px 24px rgba(0, 0, 0, .18); }
  .combo-list li { display: flex; justify-content: space-between; gap: 16px; padding: 8px 10px; border-radius: 6px; cursor: pointer; }
  .combo-list li:hover { background: var(--bg); }
  .combo-list li.active { background: var(--accent); color: var(--accent-text); }
  .combo-list .count { color: var(--muted); font-size: .85rem; }
  .combo-list li.active .count { color: inherit; }
  .combo-list li.none { color: var(--muted); cursor: default; background: none; font-size: .9rem; }

  #record-panel.editing { border-color: var(--edit); box-shadow: 0 0 0 2px var(--edit-bg); }
  .banner { background: var(--edit-bg); color: var(--edit); border-radius: 8px; padding: 10px 12px;
            margin-bottom: 12px; font-weight: 600; }
  .timer { margin-bottom: 14px; padding: 10px 14px 12px; border: 1px solid var(--border); border-radius: 10px; background: var(--bg); }
  .timer-row { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .timer .label { margin: 0; }
  .timer-value { font: 700 2rem/1.1 ui-monospace, "SFMono-Regular", Menlo, monospace; }
  .timer-limit { color: var(--muted); font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; }
  .timer-note { margin-left: auto; color: var(--muted); font-size: .9rem; }
  .timer-track { height: 8px; margin-top: 8px; border-radius: 4px; background: var(--grid); overflow: hidden; }
  .timer-fill { height: 100%; width: 0; border-radius: 4px; background: var(--accent); }
  .timer.idle .timer-value { color: var(--muted); }
  .timer.warn { border-color: var(--edit); background: var(--edit-bg); }
  .timer.warn .timer-value, .timer.warn .timer-note { color: var(--edit); font-weight: 700; }
  .timer.warn .timer-fill { background: var(--edit); }
  .timer.over { border-color: var(--error); background: var(--error-bg); }
  .timer.over .timer-value, .timer.over .timer-note { color: var(--error); font-weight: 700; }
  .timer.over .timer-fill { background: var(--error); }
  .timer.stopped .timer-fill { background: var(--muted); }
  .star-state { display: flex; align-items: center; gap: 8px 12px; flex-wrap: wrap; margin-bottom: 12px; }
  .star-state .label { margin: 0; }
  .state-btn { font: inherit; padding: 8px 14px; border: 1px solid var(--border); border-radius: 8px;
               background: var(--bg); color: var(--text); cursor: pointer; }
  .state-btn:hover { border-color: var(--accent); }
  .state-btn[aria-pressed="true"] { background: var(--accent); color: var(--accent-text); border-color: var(--accent); font-weight: 600; }
  .state-btn[aria-pressed="true"] kbd { background: transparent; color: inherit; border-color: currentColor; }
  .event { display: grid; grid-template-columns: minmax(180px, 240px) 160px 90px 40px; gap: 10px;
           align-items: center; padding: 5px 0; }
  .stamp { font: inherit; font-weight: 600; text-align: left; padding: 12px 14px; border-radius: 8px;
           border: 1px solid var(--border); background: var(--bg); color: var(--text); cursor: pointer; }
  .stamp:hover:not(:disabled) { border-color: var(--accent); }
  .stamp:disabled { cursor: default; opacity: .6; }
  .event.done .stamp { background: var(--done-bg); border-color: var(--done); }
  .event.done .stamp::after { content: "  \2713"; color: var(--done); }
  .time { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; }
  .time.invalid { border-color: var(--error); background: var(--error-bg); }
  .delta { color: var(--muted); font: .9rem ui-monospace, "SFMono-Regular", Menlo, monospace; white-space: nowrap; }
  .delta.backwards { color: var(--error); font-weight: 700; }
  .clear { font: inherit; font-size: 1.2rem; line-height: 1; width: 36px; height: 36px; border-radius: 8px;
           border: 1px solid transparent; background: transparent; color: var(--muted); cursor: pointer; }
  .clear:hover { border-color: var(--border); color: var(--text); }
  @media (max-width: 620px) {
    .event { grid-template-columns: 1fr 80px 40px; padding: 8px 0; }
    .stamp { grid-column: 1 / -1; }
  }
  .flash { animation: flash .7s; }
  @keyframes flash { 30% { box-shadow: 0 0 0 4px var(--edit); } }
  kbd { font: 600 .78rem ui-monospace, "SFMono-Regular", Menlo, monospace; border: 1px solid var(--border);
        border-bottom-width: 2px; border-radius: 4px; padding: 0 5px; margin-right: 10px; color: var(--muted);
        background: var(--panel); }

  .notes { margin-top: 12px; }
  .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
  .btn { font: inherit; padding: 10px 18px; border-radius: 8px; border: 1px solid var(--border);
         background: var(--panel); color: var(--text); cursor: pointer; }
  .btn:hover { border-color: var(--accent); }
  .btn.primary { background: var(--accent); color: var(--accent-text); border-color: var(--accent); font-weight: 700; }
  .btn.primary kbd { background: transparent; color: inherit; border-color: currentColor; }
  .btn.danger { color: var(--error); border-color: var(--error); margin-left: auto; }
  #message { margin-top: 14px; padding: 10px 12px; border-radius: 8px; white-space: pre-line; }
  #message:empty { display: none; }
  #message.ok { background: var(--done-bg); color: var(--done); }
  #message.warn { background: var(--edit-bg); color: var(--edit); }
  #message.error { background: var(--error-bg); color: var(--error); font-weight: 600; }

  .records-head { display: flex; justify-content: space-between; align-items: center; gap: 12px;
                  flex-wrap: wrap; margin-bottom: 10px; }
  .muted { color: var(--muted); font-weight: 400; font-size: .9rem; }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .88rem; white-space: nowrap; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  th { color: var(--muted); font-weight: 600; }
  td.t { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; }
  td.n { white-space: normal; min-width: 140px; }
  tr.current td { background: var(--edit-bg); }
  td .btn { padding: 4px 12px; font-size: .85rem; }
  .status { color: var(--muted); }
  .path { color: var(--muted); font-size: .85rem; margin: 12px 0 0; }
  .intro { color: var(--muted); font-size: .9rem; margin: 0 0 10px; }
  .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .seg button { font: inherit; font-size: .85rem; padding: 6px 12px; border: 0; background: var(--panel);
                color: var(--text); cursor: pointer; }
  .seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-text); font-weight: 600; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 6px; }
  .tile { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
  .tile .k { font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; font-weight: 600; }
  .tile .v { font-size: 1.6rem; font-weight: 700; font-variant-numeric: tabular-nums; }
  .tile .s { font-size: .8rem; color: var(--muted); }
  .chart-title { font-size: .95rem; margin: 18px 0 6px; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: .85rem; color: var(--muted); margin-bottom: 4px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .chart svg { display: block; }
  .chart text { fill: var(--muted); font-size: 11px; font-family: inherit; }
  .chart text.row-label { fill: var(--text); font-size: 12px; }
  .report-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 0 28px; }
  .hbar { display: grid; grid-template-columns: minmax(80px, 11em) 1fr 4.6em; align-items: center; gap: 8px;
          font-size: .85rem; padding: 4px 0; }
  .hbar:hover { background: var(--bg); }
  .hbar .fill { height: 12px; border-radius: 0 4px 4px 0; background: var(--series-1); min-width: 2px; }
  .hbar .val { font-variant-numeric: tabular-nums; text-align: right; }
  .tooltip { position: fixed; z-index: 20; pointer-events: none; background: var(--panel); color: var(--text);
             border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: .82rem;
             box-shadow: 0 6px 18px rgba(0, 0, 0, .18); white-space: pre-line; max-width: 300px; }
  .views { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
  .view { font: inherit; color: inherit; text-align: left; padding: 0; border: 2px solid var(--border);
          border-radius: 10px; background: var(--bg); overflow: hidden; cursor: pointer; }
  .view:hover { border-color: var(--accent); }
  .view.selected { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }
  .view img, .view .missing { display: block; width: 100%; aspect-ratio: 16 / 10; object-fit: contain; background: #000; }
  .view .missing { display: flex; align-items: center; justify-content: center; padding: 12px; text-align: center;
                   background: var(--panel); color: var(--muted); font-size: .85rem; }
  .view .caption { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; padding: 8px 10px; font-weight: 600; }
  .view .caption small { margin: 0; font-weight: 400; }
  .view.selected .caption { background: var(--accent); color: var(--accent-text); }
  .view.selected .caption small { color: inherit; }
  .view.resting, .view.resting:hover { cursor: not-allowed; border-color: var(--edit); border-style: dashed; }
  .view.resting img, .view.resting .missing { opacity: .35; filter: grayscale(1); }
  .view.resting .caption, .view.resting .caption small { background: var(--edit-bg); color: var(--edit); }
  .activity { list-style: none; margin: 0; padding: 0; max-height: 380px; overflow-y: auto; }
  .activity li { display: grid; grid-template-columns: 9.5em 1fr auto; gap: 12px; align-items: center;
                 padding: 6px 4px; border-bottom: 1px solid var(--border); font-size: .9rem; }
  .activity time { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; color: var(--muted); }
  .activity li.pending .summary::after { content: " (not yet written to the log)"; color: var(--error); }
  .activity .btn { padding: 4px 12px; font-size: .85rem; }
  @media (max-width: 620px) { .activity li { grid-template-columns: 1fr auto; } .activity time { grid-column: 1 / -1; } }
</style>
</head>
<body>
<header>
  <h1>Poke Trial Logger</h1>
  <div class="clock" id="clock" aria-label="Current time">--:--:--</div>
</header>
<main>
  <section class="panel" id="setup" aria-label="Session">
    <div>
      <label class="label" for="trial">Trial number</label>
      <div class="combo">
        <input type="text" id="trial" inputmode="numeric" maxlength="6" autocomplete="off"
               role="combobox" aria-expanded="false" aria-controls="trial-list" aria-autocomplete="list">
        <button type="button" class="combo-toggle" id="trial-toggle" tabindex="-1"
                aria-label="Show saved trials" title="Show saved trials">&#9662;</button>
        <ul class="combo-list" id="trial-list" role="listbox" hidden></ul>
      </div>
      <small id="trial-hint"></small>
    </div>
    <fieldset><legend>Tank</legend><div class="choices" id="tank"></div></fieldset>
    <fieldset><legend>Camera</legend><div class="choices" id="camera"></div><small id="camera-hint"></small>
      <small class="rest-hint" id="camera-rest" hidden></small></fieldset>
    <fieldset><legend>Poker</legend><div class="choices" id="poker"></div></fieldset>
  </section>

  <section class="panel" id="views" aria-label="Camera views" hidden>
    <div class="records-head">
      <h2>Camera views <span class="muted" id="views-tank"></span></h2>
      <span class="muted">Reference stills. Click one to choose that camera.</span>
    </div>
    <div class="views" id="views-grid"></div>
  </section>

  <section class="panel" id="record-panel" aria-label="Record">
    <div class="banner" id="edit-banner" hidden></div>
    <div class="timer" id="poke-timer" role="timer" aria-live="off">
      <div class="timer-row">
        <span class="label">Poke timer</span>
        <span class="timer-value" id="timer-value">0:00</span>
        <span class="timer-limit" id="timer-limit"></span>
        <span class="timer-note" id="timer-note"></span>
      </div>
      <div class="timer-track"><div class="timer-fill" id="timer-fill"></div></div>
    </div>
    <div class="star-state">
      <span class="label" id="star-state-label">Star before poking</span>
      <div class="choices" id="star-state" role="group" aria-labelledby="star-state-label"></div>
      <span class="muted">Optional. Click again to clear.</span>
    </div>
    <div id="events"></div>
    <div class="notes">
      <label class="label" for="notes">Notes</label>
      <input type="text" id="notes" maxlength="1000" autocomplete="off">
    </div>
    <div class="actions">
      <button type="button" class="btn primary" id="save"><kbd>Enter</kbd><span id="save-label">Save record</span></button>
      <button type="button" class="btn" id="reset"><kbd>Esc</kbd><span id="reset-label">Reset</span></button>
      <button type="button" class="btn danger" id="delete" hidden>Delete record</button>
    </div>
    <div id="message" role="status" aria-live="polite"></div>
  </section>

  <section class="panel" aria-label="Saved records">
    <div class="records-head">
      <h2>Saved records <span class="muted" id="count"></span></h2>
      <label><input type="checkbox" id="show-all"> Show all</label>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr id="records-head"></tr></thead>
        <tbody id="records-body"></tbody>
      </table>
    </div>
    <p class="status" id="records-status" hidden></p>
    <p class="path">Saving to <code id="csv-path"></code></p>
  </section>

  <section class="panel" aria-label="Activity log">
    <div class="records-head">
      <h2>Activity log <span class="muted" id="activity-count"></span></h2>
      <label><input type="checkbox" id="activity-all"> Show all</label>
    </div>
    <p class="intro">Everything done on this page is logged, along with what was in the form at that moment
      (for resets, cancelled edits and deletions: just before). <strong>Restore</strong> puts those inputs back in the form.</p>
    <ol class="activity" id="activity-list"></ol>
    <p class="status" id="activity-status" hidden></p>
    <p class="path">Logging to <code id="actions-path"></code></p>
  </section>

  <section class="panel" id="report" aria-label="Report">
    <div class="records-head">
      <h2>Report <span class="muted" id="report-scope"></span></h2>
      <div class="seg" role="group" aria-label="Report covers">
        <button type="button" data-scope="trial" aria-pressed="true">Current trial</button>
        <button type="button" data-scope="all" aria-pressed="false">All trials</button>
      </div>
    </div>
    <p class="status" id="report-empty" hidden></p>
    <div id="report-body" hidden>
      <div class="tiles" id="report-tiles"></div>
      <h3 class="chart-title">Each record: seconds after entry <span class="muted" id="timeline-note"></span></h3>
      <div class="legend" id="timeline-legend"></div>
      <div class="chart" id="timeline"></div>
      <div class="report-cols">
        <div><h3 class="chart-title">Median time between events</h3><div id="phases"></div></div>
        <div><h3 class="chart-title">Records by tank and camera</h3><div id="by-camera"></div></div>
        <div><h3 class="chart-title">Records by poker</h3><div id="by-poker"></div></div>
        <div><h3 class="chart-title">Star before poking</h3><div id="by-star-state"></div></div>
      </div>
    </div>
  </section>
</main>
<div class="tooltip" id="tooltip" hidden></div>
<script>
"use strict";
const RECENT = 15;
const $ = (selector) => document.querySelector(selector);
const state = { meta: null, fields: [], times: {}, invalid: {}, starState: "", editing: null, saved: null, records: [], busy: false };

// ---- Time helpers -------------------------------------------------------
const pad = (n, width = 2) => String(n).padStart(width, "0");
const formatTime = (d) =>
  `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
const formatDate = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

function toIso(d) {
  const offset = -d.getTimezoneOffset();
  const sign = offset >= 0 ? "+" : "-";
  const abs = Math.abs(offset);
  return `${formatDate(d)}T${formatTime(d)}${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`;
}

// Accepts HH:MM, HH:MM:SS or HH:MM:SS.mmm, on the same day as `base`.
function parseTyped(text, base) {
  const m = text.trim().match(/^(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,3}))?)?$/);
  if (!m) return null;
  const [h, min, s] = [Number(m[1]), Number(m[2]), Number(m[3] || 0)];
  if (h > 23 || min > 59 || s > 59) return null;
  const ms = Number((m[4] || "0").padEnd(3, "0"));
  return new Date(base.getFullYear(), base.getMonth(), base.getDate(), h, min, s, ms);
}

function showIso(iso, format) {
  const d = new Date(iso);
  return iso && !Number.isNaN(d.getTime()) ? format(d) : iso || "";
}

// ---- Server -------------------------------------------------------------
async function api(method, url, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw new Error("Couldn't reach the logger. Check that poke_logger.py is still running in its terminal, " +
      "then try again. Nothing on this page has been lost.");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data.errors || [`The logger returned error ${res.status}.`]).join("\n"));
  return data;
}

// ---- Messages -----------------------------------------------------------
function showMessage(text, kind = "ok") {
  const el = $("#message");
  el.textContent = text;
  el.className = text ? kind : "";
}

function flash(el) {
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}

const labelOf = (field) => state.meta.events.find((e) => e.field === field).label;
const rowOf = (field) => document.querySelector(`.event[data-field="${field}"]`);

// ---- Activity log -------------------------------------------------------
// Every action on the page goes to the server's activity log together with the
// form's contents, so inputs lost by mistake can be restored.
const ACTIVITY_RECENT = 25;
const activity = { entries: [], queue: [], sending: false, last: { trial: "", notes: "" } };

function formState() {
  const form = {
    trial_number: $("#trial").value.trim(), tank_id: getChoice("tank"), camera_id: getChoice("camera"),
    poker_id: getChoice("poker"), pre_poke_state: state.starState, notes: $("#notes").value,
  };
  for (const f of state.fields) form[f] = state.times[f] ? toIso(state.times[f]) : "";
  form.editing_record = state.editing ? state.editing.record_id : "";
  return form;
}

function logAction(action, summary) {
  const entry = { action, summary, form: formState(), at: toIso(new Date()), pending: true };
  activity.entries.push(entry);
  activity.queue.push(entry);
  renderActivity();
  sendQueued();
}

// Send queued entries one at a time, in order. If the logger can't be reached,
// they stay queued and are retried.
async function sendQueued() {
  if (activity.sending) return;
  activity.sending = true;
  const status = $("#activity-status");
  try {
    while (activity.queue.length) {
      const { action, summary, form } = activity.queue[0];
      await api("POST", "/api/actions", { action, summary, form });
      activity.queue.shift().pending = false;
    }
    status.hidden = true;
  } catch {
    status.hidden = false;
    status.textContent = `${activity.queue.length} action(s) haven't been written to the log yet. ` +
      "They'll be retried automatically.";
  } finally {
    activity.sending = false;
    renderActivity();
  }
}

function logTrialChange(how = "") {
  const trial = $("#trial").value.trim();
  if (trial === activity.last.trial) return;
  activity.last.trial = trial;
  logAction("trial_number", trial ? `Trial number set to ${trial}${how}` : "Trial number cleared");
}

function logNotesChange() {
  const notes = $("#notes").value.trim();
  if (notes === activity.last.notes) return;
  activity.last.notes = notes;
  logAction("notes", notes ? `Notes changed to "${notes}"` : "Notes cleared");
}

const hasInputs = (form) =>
  ["trial_number", "tank_id", "camera_id", "poker_id", "pre_poke_state", "notes", ...state.fields].some((k) => form[k]);

function describeTime(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso || "";
  const time = formatTime(d).slice(0, 8);
  return formatDate(d) === formatDate(new Date()) ? time : `${formatDate(d)} ${time}`;
}

function renderActivity() {
  const list = $("#activity-list");
  list.textContent = "";
  const entries = [...activity.entries].reverse();
  const shown = $("#activity-all").checked ? entries : entries.slice(0, ACTIVITY_RECENT);
  for (const entry of shown) {
    const li = document.createElement("li");
    if (entry.pending) li.className = "pending";
    const time = document.createElement("time");
    time.textContent = describeTime(entry.at);
    time.title = entry.at;
    const summary = document.createElement("span");
    summary.className = "summary";
    summary.textContent = entry.summary;
    const cell = document.createElement("span");
    if (entry.form && hasInputs(entry.form)) {
      const restore = document.createElement("button");
      restore.type = "button";
      restore.className = "btn";
      restore.textContent = "Restore";
      restore.title = "Put the inputs from this moment back in the form";
      restore.addEventListener("click", () => restoreFrom(entry));
      cell.append(restore);
    }
    li.append(time, summary, cell);
    list.append(li);
  }
  $("#activity-count").textContent = entries.length > shown.length
    ? `(newest ${shown.length} of ${entries.length})`
    : `(${entries.length})`;
}

function restoreFrom(entry) {
  const form = entry.form;
  const when = describeTime(entry.at);
  const inUse = state.editing || state.fields.some((f) => state.times[f]) || $("#notes").value.trim();
  if (inUse && !confirm(`Replace what's in the form now with the inputs from ${when}?\n\n` +
      "What's in the form now is already in the activity log, so it can be restored later.")) return;
  if (state.editing) exitEdit();
  const record = form.editing_record && state.records.find((r) => r.record_id === form.editing_record);
  if (record) startEdit(record);
  const times = {};
  for (const f of state.fields) times[f] = form[f] ? new Date(form[f]) : null;
  fill({ trial: form.trial_number || "", tank: form.tank_id || "", camera: form.camera_id || "",
         poker: form.poker_id || "", starState: form.pre_poke_state || "", notes: form.notes || "", times });
  logAction("restore", `Restored inputs from ${when} (${entry.summary})`);
  showMessage(record
    ? `Restored your edit of record #${record.record_id} from ${when}. Click Save changes to keep it.`
    : `Restored the inputs from ${when}. Save the record to keep them.`, "ok");
  $("#record-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadActivity() {
  try {
    activity.entries = (await api("GET", "/api/actions")).actions;
  } catch (err) {
    $("#activity-status").hidden = false;
    $("#activity-status").textContent = `Couldn't load the activity log. ${err.message}`;
  }
  renderActivity();
}

// ---- Session choices ----------------------------------------------------
function buildChoices(name, values) {
  const box = $(`#${name}`);
  for (const value of values) {
    const label = document.createElement("label");
    label.className = "choice";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = name;
    input.value = value;
    const text = document.createElement("span");
    text.textContent = value;
    label.append(input, text);
    box.append(label);
  }
}

const getChoice = (name) => (document.querySelector(`input[name="${name}"]:checked`) || {}).value || "";
function setChoice(name, value) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((el) => { el.checked = el.value === value; });
}

function applyCameraRule() {
  const dTank = state.meta.camera_d_tank;
  const d = document.querySelector('input[name="camera"][value="D"]');
  if (!d) return;
  renderRest();
  const allowed = getChoice("tank") === dTank;
  if (!allowed && d.checked) {
    d.checked = false;
    showMessage(`Camera D is only on ${dTank}, so the camera choice was cleared.`, "warn");
    return true;
  }
  return false;
}

// ---- Camera rest --------------------------------------------------------
// After a trial, its camera on that tank can't be chosen for a new record until
// camera_rest_seconds after the trial's last recorded event, so students can see which
// stars are still resting. Saving a trial that already happened, and editing saved records, aren't limited.
const rest = { views: "" };
const restUntil = (r) => formatTime(new Date(Math.ceil(r.until / 60000) * 60000)).slice(0, 5);
const restKey = (tank, resting) => `${tank}|${Object.entries(resting).map(([c, r]) => c + r.until.getTime()).sort().join()}`;

function lastEventTime(record) {
  let latest = -Infinity;
  for (const f of state.fields) {
    const t = record[f] ? new Date(record[f]).getTime() : NaN;
    if (t > latest) latest = t;
  }
  return latest;
}

// {camera: {until, record}} for each camera of the chosen tank that is still resting.
function restingCameras() {
  const tank = getChoice("tank");
  const resting = {};
  if (!tank || state.editing) return resting;
  const now = Date.now();
  for (const r of state.records) {
    if (r.tank_id !== tank) continue;
    const until = lastEventTime(r) + state.meta.camera_rest_seconds * 1000;
    if (until > now && !(resting[r.camera_id]?.until >= until)) resting[r.camera_id] = { until: new Date(until), record: r };
  }
  return resting;
}

const restTitle = (camera, r) => `${getChoice("tank")} camera ${camera} is resting until ${formatTime(r.until).slice(0, 8)}: ` +
  `${state.meta.camera_rest_seconds / 60} minutes after record #${r.record.record_id} (trial ${r.record.trial_number}) ended.`;

// Disable the cameras that can't be chosen (resting, or D on a tank without it) and say when they're free.
function renderRest() {
  if (!state.meta) return;
  const tank = getChoice("tank");
  const resting = restingCameras();
  document.querySelectorAll('input[name="camera"]').forEach((input) => {
    const r = resting[input.value];
    const span = input.nextElementSibling;
    let wait = span.querySelector(".wait");
    if (!wait) {
      wait = document.createElement("i");
      wait.className = "wait";
      span.append(wait);
    }
    input.disabled = Boolean(r) || (input.value === "D" && tank !== state.meta.camera_d_tank);
    input.parentElement.classList.toggle("resting", Boolean(r));
    input.parentElement.title = r ? restTitle(input.value, r) : "";
    wait.textContent = r ? `${Math.max(1, Math.ceil((r.until - Date.now()) / 60000))} min` : "";
  });
  const list = Object.keys(resting).sort().map((camera) => `${camera} until ${restUntil(resting[camera])}`);
  $("#camera-rest").hidden = !list.length;
  $("#camera-rest").textContent = `Resting ${state.meta.camera_rest_seconds / 60} min after a trial: ${list.join(" · ")}`;
  if (restKey(tank, resting) !== rest.views) renderViews();
}

// A resting camera can't stay chosen for a new record. It's kept only if the form already
// has event times (that trial happened, so it can still be saved). Returns a note if it was cleared.
function releaseRestingCamera() {
  const camera = getChoice("camera");
  const r = restingCameras()[camera];
  if (!r || state.fields.some((f) => state.times[f])) return "";
  setChoice("camera", "");
  renderViews();
  const where = `${getChoice("tank")} camera ${camera}`;
  logAction("camera_resting", `Camera choice cleared: ${where} is resting until ${restUntil(r)}`);
  return `${capitalize(where)} is resting until ${restUntil(r)}, so the camera choice was cleared. Choose another camera.`;
}

// Runs every second, and when the records or the tank change. A note about a cleared camera
// is added to the message already showing, or replaces it when `fresh`.
function refreshCameras(fresh = false) {
  renderRest();
  const note = releaseRestingCamera();
  if (!note) return;
  const el = $("#message");
  if (fresh || !el.textContent) showMessage(note, "warn");
  else showMessage(`${el.textContent}\n${note}`, el.className);
}

// One reference still per camera of the chosen tank (cameras don't move during trials).
function renderViews() {
  const tank = getChoice("tank");
  const resting = restingCameras();
  rest.views = restKey(tank, resting);
  $("#views").hidden = !tank;
  if (!tank) return;
  $("#views-tank").textContent = `(${tank})`;
  const grid = $("#views-grid");
  grid.textContent = "";
  const chosen = getChoice("camera");
  const cameras = state.meta.cameras.filter((c) => c !== "D" || tank === state.meta.camera_d_tank);
  for (const camera of cameras) {
    const key = `${tank}_${camera}`;
    const grabbed = state.meta.references[key];
    const r = resting[camera];
    const card = document.createElement("button");
    card.type = "button";
    card.className = ["view", camera === chosen && "selected", r && "resting"].filter(Boolean).join(" ");
    card.disabled = Boolean(r);
    card.title = r ? restTitle(camera, r)
      : `Choose camera ${camera}` + (grabbed ? ` (reference still grabbed ${describeTime(grabbed)})` : "");
    if (grabbed) {
      const img = document.createElement("img");
      img.src = `/reference/${key}.jpg?v=${encodeURIComponent(grabbed)}`;
      img.alt = `Reference view from ${tank} camera ${camera}`;
      card.append(img);
    } else {
      const missing = document.createElement("div");
      missing.className = "missing";
      missing.textContent = "No reference image yet. See README: Camera reference images.";
      card.append(missing);
    }
    const caption = document.createElement("div");
    caption.className = "caption";
    caption.textContent = `Camera ${camera}`;
    if (r) {
      const until = document.createElement("small");
      until.textContent = `Resting until ${restUntil(r)}`;
      caption.append(until);
    }
    card.append(caption);
    card.addEventListener("click", () => {
      card.blur();
      if (getChoice("camera") === camera) return;
      setChoice("camera", camera);
      $("#camera").dispatchEvent(new Event("change"));
    });
    grid.append(card);
  }
}

function renderTrialHint() {
  const trial = $("#trial").value.trim();
  const parts = [];
  if (/^\d+$/.test(trial) && Number(trial) > 0) {
    const n = String(Number(trial));
    const count = state.records.filter((r) => r.trial_number === n).length;
    parts.push(`${count} saved for trial ${n}`);
  }
  const last = state.records[state.records.length - 1];
  parts.push(last ? `last record was trial ${last.trial_number}` : "no records in the file yet");
  const text = parts.join(" · ");
  $("#trial-hint").textContent = text.charAt(0).toUpperCase() + text.slice(1);
  renderReport();
}

// ---- Trial number combo box ---------------------------------------------
// Type any number, or pick one of the trials already in the file.
const combo = { open: false, items: [], active: -1 };

function savedTrials() {
  const counts = new Map();
  for (const r of state.records) counts.set(r.trial_number, (counts.get(r.trial_number) || 0) + 1);
  return [...counts].sort((a, b) => Number(b[0]) - Number(a[0]));
}

function renderTrialList(filter) {
  const list = $("#trial-list");
  list.textContent = "";
  combo.items = savedTrials().filter(([trial]) => !filter || trial.startsWith(filter));
  combo.active = -1;
  $("#trial").removeAttribute("aria-activedescendant");
  if (!combo.items.length) {
    const li = document.createElement("li");
    li.className = "none";
    li.textContent = filter ? `No saved trial starts with ${filter}. It will be a new trial.`
                            : "No saved trials yet. Type a number to start one.";
    list.append(li);
  }
  combo.items.forEach(([trial, count], i) => {
    const li = document.createElement("li");
    li.id = `trial-option-${i}`;
    li.setAttribute("role", "option");
    const name = document.createElement("span");
    name.textContent = `Trial ${trial}`;
    const records = document.createElement("span");
    records.className = "count";
    records.textContent = `${count} record${count === 1 ? "" : "s"}`;
    li.append(name, records);
    // mousedown + preventDefault keeps focus in the box, so the list doesn't close first.
    li.addEventListener("mousedown", (e) => { e.preventDefault(); chooseTrial(trial); });
    list.append(li);
  });
}

function openTrialList(filter = "") {
  renderTrialList(filter);
  $("#trial-list").hidden = false;
  combo.open = true;
  $("#trial").setAttribute("aria-expanded", "true");
}

function closeTrialList() {
  $("#trial-list").hidden = true;
  combo.open = false;
  $("#trial").setAttribute("aria-expanded", "false");
  $("#trial").removeAttribute("aria-activedescendant");
}

function chooseTrial(trial) {
  $("#trial").value = trial;
  closeTrialList();
  renderTrialHint();
  logTrialChange(" (picked from the list)");
}

function moveActive(step) {
  if (!combo.items.length) return;
  combo.active = (combo.active + step + combo.items.length) % combo.items.length;
  [...$("#trial-list").children].forEach((li, i) => li.classList.toggle("active", i === combo.active));
  const li = $(`#trial-option-${combo.active}`);
  li.scrollIntoView({ block: "nearest" });
  $("#trial").setAttribute("aria-activedescendant", li.id);
}

function setUpTrialCombo() {
  const input = $("#trial");
  input.addEventListener("input", () => {
    input.value = input.value.replace(/\D/g, "");
    renderTrialHint();
    openTrialList(input.value);
  });
  input.addEventListener("click", () => { if (!combo.open) openTrialList(""); });
  input.addEventListener("blur", closeTrialList);
  input.addEventListener("keydown", (e) => {
    // Keys handled here don't reach the page shortcuts (Enter = save, Esc = reset).
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!combo.open) openTrialList("");
      else moveActive(e.key === "ArrowDown" ? 1 : -1);
    } else if (e.key === "Enter" && combo.open) {
      if (combo.active >= 0) chooseTrial(combo.items[combo.active][0]);
      else closeTrialList();
    } else if (e.key === "Escape" && combo.open) {
      closeTrialList();
    } else {
      return;
    }
    e.preventDefault();
    e.stopPropagation();
  });
  const toggle = $("#trial-toggle");
  toggle.addEventListener("mousedown", (e) => e.preventDefault());
  toggle.addEventListener("click", () => {
    if (combo.open) { closeTrialList(); return; }
    input.focus();
    openTrialList("");
  });
}

// ---- Event times --------------------------------------------------------
function buildEvents() {
  const box = $("#events");
  state.meta.events.forEach(({ field, label }, i) => {
    const row = document.createElement("div");
    row.className = "event";
    row.dataset.field = field;

    const stampButton = document.createElement("button");
    stampButton.type = "button";
    stampButton.className = "stamp";
    const key = document.createElement("kbd");
    key.textContent = String(i + 1);
    stampButton.append(key, label);
    stampButton.addEventListener("click", () => { stamp(field); stampButton.blur(); });

    const input = document.createElement("input");
    input.type = "text";
    input.className = "time";
    input.placeholder = "HH:MM:SS.mmm";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.setAttribute("aria-label", `${label} time`);
    input.addEventListener("change", () => commitTyped(field));

    const delta = document.createElement("span");
    delta.className = "delta";

    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "clear";
    clear.textContent = "×";
    clear.title = `Clear ${label} time`;
    clear.setAttribute("aria-label", `Clear ${label} time`);
    clear.addEventListener("click", () => {
      const was = state.times[field];
      setTime(field, null);
      clear.blur();
      if (was) logAction("clear_time", `${label} cleared (was ${formatTime(was)})`);
    });

    row.append(stampButton, input, delta, clear);
    box.append(row);
  });
}

function setTime(field, date) {
  state.times[field] = date;
  state.invalid[field] = false;
  const row = rowOf(field);
  const input = row.querySelector(".time");
  input.value = date ? formatTime(date) : "";
  input.title = date ? toIso(date) : "";
  input.classList.remove("invalid");
  row.classList.toggle("done", Boolean(date));
  renderDeltas();
  renderTimer();
}

function stamp(field) {
  if (state.editing) {
    showMessage("You're editing a saved record. Type corrected times, or press Esc to cancel the edit.", "warn");
    return;
  }
  if (state.times[field]) {
    flash(rowOf(field).querySelector(".time"));
    showMessage(`${labelOf(field)} is already recorded. Clear it with × first if you need to record it again.`, "warn");
    logAction("press_ignored", `${labelOf(field)} pressed again; ignored (already ${formatTime(state.times[field])})`);
    return;
  }
  const now = new Date();
  setTime(field, now);
  logAction("record_time", `${labelOf(field)} recorded at ${formatTime(now)}`);
}

function baseDate(field) {
  if (state.times[field]) return state.times[field];
  const entry = state.times[state.fields[0]];
  if (entry) return entry;
  if (state.editing) return new Date(state.editing[state.fields[0]]);
  return new Date();
}

function commitTyped(field) {
  const input = rowOf(field).querySelector(".time");
  const text = input.value.trim();
  const current = state.times[field];
  const was = current ? formatTime(current) : "blank";
  if (!text) {
    setTime(field, null);
    if (current) logAction("clear_time", `${labelOf(field)} cleared by deleting the text (was ${was})`);
    return;
  }
  if (current && text === formatTime(current)) return;
  const date = parseTyped(text, baseDate(field));
  if (date) {
    setTime(field, date);
    logAction("type_time", `${labelOf(field)} typed as ${formatTime(date)} (was ${was})`);
  } else {
    state.invalid[field] = true;
    input.classList.add("invalid");
    if (input.dataset.loggedBad !== text) {
      input.dataset.loggedBad = text;
      logAction("invalid_time", `${labelOf(field)}: "${text}" isn't a valid time, so it wasn't applied`);
    }
  }
}

// Fields whose time is earlier than an event listed above them.
function outOfOrder() {
  const problems = [];
  let latest = null;
  for (const field of state.fields) {
    const t = state.times[field];
    if (!t) continue;
    if (latest && t < latest.t) problems.push({ field, before: latest.field });
    else latest = { field, t };
  }
  return problems;
}

function renderDeltas() {
  const entry = state.times[state.fields[0]];
  const backwards = new Set(outOfOrder().map((p) => p.field));
  state.fields.forEach((field, i) => {
    const el = rowOf(field).querySelector(".delta");
    const t = state.times[field];
    if (i > 0 && entry && t) {
      const secs = (t - entry) / 1000;
      el.textContent = `${secs < 0 ? "−" : "+"}${Math.abs(secs).toFixed(2)} s`;
    } else {
      el.textContent = "";
    }
    el.classList.toggle("backwards", backwards.has(field));
  });
}

// ---- Star state before poking -------------------------------------------
// Optional per-record call: was the star quiescent, alert or moving before poking? Keys Q / A / M.
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function buildStarStates() {
  const box = $("#star-state");
  for (const value of state.meta.star_states) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "state-btn";
    button.dataset.value = value;
    button.setAttribute("aria-pressed", "false");
    const key = document.createElement("kbd");
    key.textContent = value.charAt(0).toUpperCase();
    button.append(key, capitalize(value));
    button.addEventListener("click", () => {
      button.blur();
      chooseStarState(state.starState === value ? "" : value);
    });
    box.append(button);
  }
}

function setStarState(value) {
  state.starState = value || "";
  document.querySelectorAll(".state-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.value === state.starState)));
}

function chooseStarState(value) {
  if (value === state.starState) return;
  const was = state.starState;
  setStarState(value);
  logAction("star_state", value ? `Star before poking: ${value}` : `Star-before-poking choice cleared (was ${was})`);
}

// ---- Poke timer ---------------------------------------------------------
// For information only: counts from first poke toward the ~5-minute limit,
// and stops once the star moves away or exits.
const fmtClock = (secs) => `${Math.floor(secs / 60)}:${pad(Math.floor(secs % 60))}`;

function renderTimer() {
  if (!state.meta) return;
  const box = $("#poke-timer");
  box.hidden = Boolean(state.editing);
  if (state.editing) return;
  const { start, stop, limit_seconds: limit } = state.meta.timer;
  const startedAt = state.times[start];
  const stopField = stop.find((f) => state.times[f]);
  const stoppedAt = stopField ? state.times[stopField] : null;
  const secs = startedAt ? Math.max(0, ((stoppedAt || new Date()) - startedAt) / 1000) : 0;
  let kind, note;
  if (!startedAt) {
    kind = "idle";
    note = `Starts at ${labelOf(start)} (key ${state.fields.indexOf(start) + 1})`;
  } else if (stoppedAt) {
    kind = "stopped";
    note = `Stopped at ${labelOf(stopField)}`;
  } else if (secs >= limit) {
    kind = "over";
    note = `${fmtClock(limit)} reached: time to stop poking`;
  } else if (secs >= limit - 60) {
    kind = "warn";
    note = `${fmtClock(Math.ceil(limit - secs))} left`;
  } else {
    kind = "running";
    note = `${fmtClock(Math.ceil(limit - secs))} left`;
  }
  box.className = `timer ${kind}`;
  $("#timer-value").textContent = fmtClock(secs);
  $("#timer-limit").textContent = `/ ${fmtClock(limit)}`;
  $("#timer-note").textContent = note;
  $("#timer-fill").style.width = `${Math.min(100, (secs / limit) * 100)}%`;
}

function clearForm() {
  state.fields.forEach((f) => setTime(f, null));
  $("#notes").value = "";
  activity.last.notes = "";
  setStarState("");
}

// ---- Saving -------------------------------------------------------------
function collect() {
  state.fields.forEach(commitTyped);
  const errors = [];
  const trial = $("#trial").value.trim();
  if (!/^\d+$/.test(trial) || Number(trial) < 1) errors.push("Enter a trial number.");
  if (!getChoice("tank")) errors.push("Choose a tank.");
  if (!getChoice("camera")) errors.push("Choose a camera.");
  if (!getChoice("poker")) errors.push("Choose a poker.");
  const invalid = state.fields.filter((f) => state.invalid[f]);
  const missing = state.fields.filter((f) => !state.invalid[f] && !state.times[f]);
  const required = missing.filter((f) => state.meta.required_events.includes(f));
  if (invalid.length) errors.push(`Fix the time format (HH:MM:SS.mmm) for: ${invalid.map(labelOf).join(", ")}.`);
  if (required.length) errors.push(`${required.map(labelOf).join(", ")} must be recorded before saving.`);

  const payload = {
    trial_number: trial,
    tank_id: getChoice("tank"),
    camera_id: getChoice("camera"),
    poker_id: getChoice("poker"),
    pre_poke_state: state.starState,
    notes: $("#notes").value,
  };
  for (const f of state.fields) payload[f] = state.times[f] ? toIso(state.times[f]) : "";
  return { errors, missing, payload };
}

function unchanged(payload, record) {
  const same = (a, b) => String(a) === String(b);
  return same(Number(payload.trial_number), record.trial_number) &&
    ["tank_id", "camera_id", "poker_id", "pre_poke_state", ...state.fields].every((k) => same(payload[k], record[k] ?? "")) &&
    same(payload.notes.split(/\s+/).filter(Boolean).join(" "), record.notes);
}

async function save() {
  if (state.busy) return;
  logTrialChange();
  logNotesChange();
  const { errors, missing, payload } = collect();
  if (errors.length) {
    showMessage(errors.join("\n"), "error");
    logAction("save_blocked", `Save blocked: ${errors.join(" ")}`);
    return;
  }
  // Events that didn't happen (e.g. no movement before poking stopped) can be left blank, but ask
  // first so an early Enter doesn't save a half-finished record by accident.
  if (missing.length && !(state.editing && missing.every((f) => !state.editing[f]))) {
    const list = missing.map(labelOf).join(", ");
    if (!confirm(`Not recorded: ${list}.\n\nSave this record with ${missing.length === 1 ? "that event" : "those events"} left blank?`)) {
      logAction("save_cancelled", `Save cancelled; not recorded: ${list}`);
      return;
    }
  }
  if (state.editing && unchanged(payload, state.editing)) {
    const id = state.editing.record_id;
    exitEdit();
    showMessage("No changes to save.", "ok");
    logAction("edit_closed", `Closed record #${id} with no changes`);
    return;
  }
  const problems = outOfOrder();
  if (problems.length) {
    const lines = problems.map((p) => `${labelOf(p.field)} is before ${labelOf(p.before)}`);
    if (!confirm(`These times are out of order:\n\n${lines.map((l) => `• ${l}`).join("\n")}\n\nSave anyway?`)) {
      logAction("save_cancelled", `Save cancelled because times were out of order (${lines.join("; ")})`);
      return;
    }
  }
  state.busy = true;
  try {
    if (state.editing) {
      const { record } = await api("PUT", `/api/records/${state.editing.record_id}`, payload);
      logAction("save_edit", `Saved changes to record #${record.record_id} (trial ${record.trial_number})`);
      exitEdit();
      showMessage(`Saved changes to record #${record.record_id}.`, "ok");
    } else {
      const { record } = await api("POST", "/api/records", payload);
      logAction("save", `Saved record #${record.record_id} (trial ${record.trial_number})`);
      clearForm();
      showMessage(`Saved record #${record.record_id} (trial ${record.trial_number}, ` +
        `entry at ${showIso(record[state.fields[0]], formatTime)}).`, "ok");
    }
  } catch (err) {
    showMessage(err.message, "error");
    logAction("save_failed", `Save failed: ${err.message}`);
  } finally {
    state.busy = false;
  }
  await loadRecords();
}

function reset() {
  if (state.editing) {
    // Logged before leaving edit mode, so Restore can bring back anything typed into the edit.
    logAction("edit_cancelled", `Cancelled editing record #${state.editing.record_id}; changes not saved`);
    exitEdit();
    showMessage("Edit cancelled. Nothing was changed.", "ok");
    return;
  }
  const recorded = state.fields.filter((f) => state.times[f]).map((f) => `${labelOf(f)} ${formatTime(state.times[f])}`);
  const notes = $("#notes").value.trim();
  const star = state.starState;
  if (!recorded.length && !notes && !star) { clearForm(); showMessage(""); return; }
  if (!confirm("Clear the times, star state and notes for this record?")) {
    logAction("reset_cancelled", "Reset cancelled");
    return;
  }
  // Logged before clearing, so Restore on this entry brings the cleared inputs back.
  logAction("reset", `Reset: discarded ${[star && `star ${star}`, ...recorded, notes && `notes "${notes}"`].filter(Boolean).join(", ")}`);
  clearForm();
  showMessage("");
}

// ---- Editing saved records ----------------------------------------------
function snapshot() {
  return {
    trial: $("#trial").value, tank: getChoice("tank"), camera: getChoice("camera"),
    poker: getChoice("poker"), starState: state.starState, notes: $("#notes").value, times: { ...state.times },
  };
}

function fill({ trial, tank, camera, poker, starState = "", notes, times }) {
  $("#trial").value = trial;
  setChoice("camera", "");
  setChoice("tank", tank);
  applyCameraRule();
  setChoice("camera", camera);
  setChoice("poker", poker);
  setStarState(starState);
  $("#notes").value = notes;
  state.fields.forEach((f) => setTime(f, times[f] || null));
  renderTrialHint();
  renderViews();
  activity.last.trial = trial.trim();
  activity.last.notes = notes.trim();
}

function setEditMode(record) {
  const editing = Boolean(record);
  $("#record-panel").classList.toggle("editing", editing);
  $("#edit-banner").hidden = !editing;
  if (editing) {
    $("#edit-banner").textContent = `Editing saved record #${record.record_id}. ` +
      "Type corrected times or change any field, then save. Live recording is paused.";
  }
  $("#delete").hidden = !editing;
  $("#save-label").textContent = editing ? "Save changes" : "Save record";
  $("#reset-label").textContent = editing ? "Cancel edit" : "Reset";
  document.querySelectorAll(".stamp").forEach((b) => { b.disabled = editing; });
  renderTimer();
}

function startEdit(record) {
  if (!state.editing) state.saved = snapshot();
  state.editing = record;
  const times = {};
  for (const f of state.fields) times[f] = record[f] ? new Date(record[f]) : null;
  fill({ trial: record.trial_number, tank: record.tank_id, camera: record.camera_id,
         poker: record.poker_id, starState: record.pre_poke_state || "", notes: record.notes || "", times });
  setEditMode(record);
  renderTable();
  showMessage("");
  logAction("edit_started", `Started editing record #${record.record_id} (trial ${record.trial_number})`);
  $("#record-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function exitEdit() {
  const saved = state.saved;
  state.editing = null;
  state.saved = null;
  setEditMode(null);
  if (saved) fill(saved);
  renderTable();
}

async function deleteRecord() {
  const record = state.editing;
  if (!record) return;
  if (!confirm(`Delete record #${record.record_id} (trial ${record.trial_number})?\n\n` +
      "It will be removed from the CSV file. A copy is kept in the change log and the activity log.")) {
    logAction("delete_cancelled", `Delete of record #${record.record_id} cancelled`);
    return;
  }
  try {
    await api("DELETE", `/api/records/${record.record_id}`);
    // Logged before leaving edit mode, so the entry holds the deleted record's values.
    logAction("delete", `Deleted record #${record.record_id} (trial ${record.trial_number})`);
    exitEdit();
    showMessage(`Deleted record #${record.record_id}.`, "ok");
  } catch (err) {
    showMessage(err.message, "error");
    logAction("delete_failed", `Delete of record #${record.record_id} failed: ${err.message}`);
  }
  await loadRecords();
}

// ---- Saved records table ------------------------------------------------
function buildTableHead() {
  const labels = ["#", "Trial", "Date", "Tank", "Cam", "Poker", "Star before", ...state.meta.events.map((e) => e.label), "Notes", "Edited", ""];
  const head = $("#records-head");
  for (const text of labels) {
    const th = document.createElement("th");
    th.textContent = text;
    head.append(th);
  }
}

function renderTable() {
  const body = $("#records-body");
  body.textContent = "";
  const rows = [...state.records].reverse();
  const shown = $("#show-all").checked ? rows : rows.slice(0, RECENT);
  for (const r of shown) {
    const tr = document.createElement("tr");
    if (state.editing && state.editing.record_id === r.record_id) tr.className = "current";
    const cells = [
      [r.record_id], [r.trial_number], [showIso(r[state.fields[0]], formatDate)],
      [r.tank_id], [r.camera_id], [r.poker_id], [r.pre_poke_state],
      ...state.fields.map((f) => [showIso(r[f], formatTime), "t"]),
      [r.notes, "n"], [r.edited_at ? "yes" : ""],
    ];
    for (const [text, cls] of cells) {
      const td = document.createElement("td");
      td.textContent = text || "";
      if (cls) td.className = cls;
      tr.append(td);
    }
    if (r.edited_at) tr.lastChild.title = `Last edited ${r.edited_at}`;
    const td = document.createElement("td");
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "btn";
    edit.textContent = "Edit";
    edit.addEventListener("click", () => startEdit(r));
    td.append(edit);
    tr.append(td);
    body.append(tr);
  }
  $("#count").textContent = rows.length > shown.length
    ? `(newest ${shown.length} of ${rows.length})`
    : `(${rows.length})`;
}

async function loadRecords() {
  const status = $("#records-status");
  try {
    state.records = (await api("GET", "/api/records")).records;
    status.hidden = state.records.length > 0;
    status.textContent = "No records saved yet.";
  } catch (err) {
    status.hidden = false;
    status.textContent = `Couldn't load saved records. ${err.message}`;
  }
  renderTable();
  renderTrialHint();
  refreshCameras();
}

// ---- Report -------------------------------------------------------------
// A live summary of saved records, for the current trial or all trials.
const report = { scope: "trial" };
const TIMELINE_MAX = 30;
// The three events between entry and exit, drawn as markers on each record's entry→exit bar.
const MARKERS = [
  { index: 1, color: "var(--series-1)", shape: "circle" },
  { index: 2, color: "var(--series-2)", shape: "square" },
  { index: 3, color: "var(--series-3)", shape: "diamond" },
];

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const secondsAfterEntry = (r, field) => (new Date(r[field]) - new Date(r[state.fields[0]])) / 1000;
const fmtSeconds = (s) => (s == null || !Number.isFinite(s) ? "–" : `${s.toFixed(Math.abs(s) < 10 ? 2 : 1)} s`);

function median(values) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return null;
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

function niceTicks(lo, hi, count) {
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let t = Math.floor(lo / step) * step; t <= Math.ceil(hi / step) * step + step / 2; t += step) {
    ticks.push(Math.round(t * 1000) / 1000);
  }
  return ticks;
}

function markerSvg(shape, cx, cy, color, r = 5) {
  const ring = 'stroke="var(--panel)" stroke-width="2"';
  if (shape === "circle") return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${color}" ${ring}/>`;
  if (shape === "square") return `<rect x="${cx - r + 0.5}" y="${cy - r + 0.5}" width="${2 * r - 1}" height="${2 * r - 1}" rx="1.5" fill="${color}" ${ring}/>`;
  const d = r + 1;
  return `<path d="M${cx} ${cy - d}L${cx + d} ${cy}L${cx} ${cy + d}L${cx - d} ${cy}Z" fill="${color}" ${ring}/>`;
}

function showTip(e, text) {
  const tip = $("#tooltip");
  tip.textContent = text;
  tip.hidden = false;
  const pad = 14, rect = tip.getBoundingClientRect();
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + rect.width > innerWidth - 8) x = e.clientX - rect.width - pad;
  if (y + rect.height > innerHeight - 8) y = e.clientY - rect.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => { $("#tooltip").hidden = true; };

function currentTrial() {
  const t = $("#trial").value.trim();
  return /^\d+$/.test(t) && Number(t) > 0 ? String(Number(t)) : "";
}

function renderReport() {
  if (!state.meta) return;
  const trial = currentTrial();
  const rows = report.scope === "trial" ? state.records.filter((r) => trial && r.trial_number === trial) : state.records;
  $("#report-scope").textContent = report.scope === "all" ? "(all trials)" : trial ? `(trial ${trial})` : "";
  document.querySelectorAll("#report .seg button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.scope === report.scope)));
  const empty = $("#report-empty");
  $("#report-body").hidden = !rows.length;
  empty.hidden = Boolean(rows.length);
  if (!rows.length) {
    empty.textContent = report.scope === "trial" && !trial
      ? "Enter a trial number to see its report."
      : `No records saved${report.scope === "trial" ? ` for trial ${trial}` : ""} yet. The report fills in as records are saved.`;
    return;
  }
  renderTiles(rows);
  renderTimeline(rows);
  renderPhases(rows);
  renderCounts("#by-camera", rows, (r) => `${r.tank_id} · camera ${r.camera_id}`);
  renderCounts("#by-poker", rows, (r) => r.poker_id);
  renderCounts("#by-star-state", rows, (r) => (r.pre_poke_state ? capitalize(r.pre_poke_state) : "Not recorded"));
}

function renderTiles(rows) {
  const f = state.fields;
  const trials = new Set(rows.map((r) => r.trial_number)).size;
  const tiles = [
    ["Records", String(rows.length), report.scope === "all" ? `across ${trials} trial${trials === 1 ? "" : "s"}` : `in trial ${rows[0].trial_number}`],
    ["Median time in view", fmtSeconds(median(rows.map((r) => secondsAfterEntry(r, f[4])))), `${labelOf(f[0])} → ${labelOf(f[4])}`],
    ["Median to first poke", fmtSeconds(median(rows.map((r) => secondsAfterEntry(r, f[1])))), `${labelOf(f[0])} → ${labelOf(f[1])}`],
    ["Median poke response", fmtSeconds(median(rows.map((r) => secondsAfterEntry(r, f[2]) - secondsAfterEntry(r, f[1])))), `${labelOf(f[1])} → ${labelOf(f[2])}`],
    ["Moved away", `${rows.filter((r) => r[f[3]]).length} of ${rows.length}`, `records with ${labelOf(f[3]).toLowerCase()} recorded`],
  ];
  const box = $("#report-tiles");
  box.textContent = "";
  for (const [k, v, s] of tiles) {
    const tile = document.createElement("div");
    tile.className = "tile";
    for (const [cls, text] of [["k", k], ["v", v], ["s", s]]) {
      const div = document.createElement("div");
      div.className = cls;
      div.textContent = text;
      tile.append(div);
    }
    box.append(tile);
  }
}

function renderTimeline(rows) {
  const f = state.fields;
  const shown = rows.slice(-TIMELINE_MAX).reverse();
  $("#timeline-note").textContent = rows.length > shown.length ? `(newest ${shown.length} of ${rows.length})` : "";
  $("#timeline-legend").innerHTML =
    `<span><svg width="26" height="10"><rect x="1" y="2" width="24" height="6" rx="3" fill="var(--track)"/></svg>${esc(labelOf(f[0]))} to ${esc(labelOf(f[4]))}</span>` +
    `<span><svg width="26" height="10"><rect x="1.75" y="2.75" width="22.5" height="4.5" rx="2.25" fill="none" stroke="var(--track)" stroke-width="1.5" stroke-dasharray="3 2"/></svg>no ${esc(labelOf(f[4]).toLowerCase())} recorded</span>` +
    MARKERS.map((m) => `<span><svg width="14" height="14">${markerSvg(m.shape, 7, 7, m.color)}</svg>${esc(labelOf(f[m.index]))}</span>`).join("");

  const box = $("#timeline");
  const W = Math.max(300, box.clientWidth), left = 128, right = 18, top = 24, rowH = 24;
  const H = top + shown.length * rowH + 4;
  const data = shown.map((r) => ({ r, t: f.map((field) => secondsAfterEntry(r, field)) }));
  const values = data.flatMap((d) => d.t).filter(Number.isFinite);
  const ticks = niceTicks(Math.min(0, ...values), Math.max(1, ...values), Math.max(2, Math.floor((W - left - right) / 80)));
  const lo = ticks[0], hi = ticks[ticks.length - 1];
  const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Event times for each record, in seconds after entry">`;
  for (const t of ticks) {
    svg += `<line x1="${x(t)}" x2="${x(t)}" y1="${top - 6}" y2="${H}" stroke="var(--grid)"/>` +
           `<text x="${x(t)}" y="12" text-anchor="middle">${t} s</text>`;
  }
  data.forEach(({ r, t }, i) => {
    const y = top + i * rowH + rowH / 2;
    svg += `<rect class="hit" data-i="${i}" x="0" y="${y - rowH / 2}" width="${W}" height="${rowH}" fill="transparent"/>`;
    svg += `<text class="row-label" x="${left - 10}" y="${y + 4}" text-anchor="end">#${esc(r.record_id)} ${esc(r.tank_id)} ${esc(r.camera_id)}</text>`;
    // Without an exit time, the bar runs to the last recorded event and is drawn hollow.
    const recorded = t.filter(Number.isFinite);
    const hasExit = Number.isFinite(t[4]);
    const a = Math.min(0, ...recorded), b = Math.max(0, ...(hasExit ? [t[4]] : recorded));
    svg += hasExit
      ? `<rect x="${x(a)}" y="${y - 3}" width="${Math.max(2, x(b) - x(a))}" height="6" rx="3" fill="var(--track)" pointer-events="none"/>`
      : `<rect x="${x(a) + 0.75}" y="${y - 2.25}" width="${Math.max(2, x(b) - x(a) - 1.5)}" height="4.5" rx="2.25" fill="none" stroke="var(--track)" stroke-width="1.5" stroke-dasharray="3 2" pointer-events="none"/>`;
    for (const m of MARKERS) {
      if (Number.isFinite(t[m.index])) svg += markerSvg(m.shape, x(t[m.index]), y, m.color).replace("/>", ' pointer-events="none"/>');
    }
  });
  box.innerHTML = `${svg}</svg>`;
  box.querySelectorAll(".hit").forEach((hit) => {
    const { r, t } = data[Number(hit.dataset.i)];
    const text = [`Record #${r.record_id} · trial ${r.trial_number}`, `${r.tank_id} camera ${r.camera_id} · poker ${r.poker_id}`,
      `Star before poking: ${r.pre_poke_state || "not recorded"}`,
      `${labelOf(f[0])} at ${showIso(r[f[0]], formatTime)}`,
      ...f.slice(1).map((field, k) => `${labelOf(field)}: ${Number.isFinite(t[k + 1]) ? `${fmtSeconds(t[k + 1])} after entry` : "not recorded"}`)].join("\n");
    hit.addEventListener("mousemove", (e) => { hit.setAttribute("fill", "var(--grid)"); showTip(e, text); });
    hit.addEventListener("mouseleave", () => { hit.setAttribute("fill", "transparent"); hideTip(); });
  });
}

function hbars(selector, items, format) {
  const box = $(selector);
  box.textContent = "";
  const max = Math.max(...items.map((i) => i.value), 0) || 1;
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "hbar";
    const label = document.createElement("span");
    label.textContent = item.label;
    const track = document.createElement("div");
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${(Math.max(0, item.value) / max) * 100}%`;
    track.append(fill);
    const val = document.createElement("span");
    val.className = "val";
    val.textContent = format(item.value);
    row.append(label, track, val);
    row.addEventListener("mousemove", (e) => showTip(e, item.tip));
    row.addEventListener("mouseleave", hideTip);
    box.append(row);
  }
}

function renderPhases(rows) {
  const f = state.fields;
  const items = f.slice(0, -1).map((from, k) => {
    const to = f[k + 1];
    const spans = rows.map((r) => secondsAfterEntry(r, to) - secondsAfterEntry(r, from)).filter(Number.isFinite);
    const m = median(spans);
    return {
      label: `${labelOf(from)} → ${labelOf(to)}`, value: m ?? 0,
      tip: `${labelOf(from)} → ${labelOf(to)}\nMedian ${fmtSeconds(m)} over ${spans.length} record${spans.length === 1 ? "" : "s"}` +
           `\nShortest ${fmtSeconds(Math.min(...spans))} · longest ${fmtSeconds(Math.max(...spans))}`,
    };
  });
  hbars("#phases", items, fmtSeconds);
}

function renderCounts(selector, rows, keyOf) {
  const counts = new Map();
  for (const r of rows) counts.set(keyOf(r), (counts.get(keyOf(r)) || 0) + 1);
  const items = [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([label, value]) => ({ label, value, tip: `${label}: ${value} record${value === 1 ? "" : "s"} (${Math.round((value / rows.length) * 100)}%)` }));
  hbars(selector, items, String);
}

// ---- Keyboard -----------------------------------------------------------
document.addEventListener("keydown", (e) => {
  if (!state.meta || e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
  const t = e.target;
  const typing = t instanceof HTMLInputElement && t.type === "text";
  if (e.key === "Enter") {
    if (t instanceof HTMLButtonElement || (t instanceof HTMLInputElement && t.type === "checkbox")) return;
    e.preventDefault();
    save();
  } else if (e.key === "Escape") {
    e.preventDefault();
    if (typing) t.blur();
    reset();
  } else if (!typing && !e.repeat && state.meta.star_states.some((s) => s.charAt(0) === e.key.toLowerCase())) {
    e.preventDefault();
    const value = state.meta.star_states.find((s) => s.charAt(0) === e.key.toLowerCase());
    chooseStarState(state.starState === value ? "" : value);
  } else if (!typing && /^[0-9]$/.test(e.key)) {
    const n = Number(e.key);
    if (n < 1 || n > state.fields.length) return;
    e.preventDefault();
    if (!e.repeat) stamp(state.fields[n - 1]);
  }
});

window.addEventListener("beforeunload", (e) => {
  if ((!state.editing && state.fields.some((f) => state.times[f])) || activity.queue.length) {
    e.preventDefault();
    e.returnValue = "";
  }
});

// ---- Start --------------------------------------------------------------
async function init() {
  setInterval(() => {
    $("#clock").textContent = formatTime(new Date()).slice(0, 8);
    renderTimer();
  }, 200);
  try {
    state.meta = await api("GET", "/api/meta");
  } catch (err) {
    showMessage(err.message, "error");
    return;
  }
  const meta = state.meta;
  state.fields = meta.events.map((e) => e.field);
  buildChoices("tank", meta.tanks);
  buildChoices("camera", meta.cameras);
  buildChoices("poker", meta.pokers);
  buildStarStates();
  $("#camera-hint").textContent = `D is only on ${meta.camera_d_tank}`;
  buildEvents();
  buildTableHead();
  applyCameraRule();
  $("#csv-path").textContent = meta.csv_path;
  $("#actions-path").textContent = meta.actions_path;

  for (const [name, label] of [["tank", "Tank"], ["camera", "Camera"], ["poker", "Poker"]]) {
    $(`#${name}`).addEventListener("change", () => {
      const cameraCleared = name === "tank" && applyCameraRule();
      logAction("choose", `${label} set to ${getChoice(name)}${cameraCleared ? " (camera D cleared)" : ""}`);
      renderViews();
      if (name === "tank") refreshCameras(true);
    });
  }
  setUpTrialCombo();
  $("#trial").addEventListener("change", () => logTrialChange());
  $("#notes").addEventListener("change", logNotesChange);
  $("#save").addEventListener("click", save);
  $("#reset").addEventListener("click", reset);
  $("#delete").addEventListener("click", deleteRecord);
  $("#show-all").addEventListener("change", renderTable);
  $("#activity-all").addEventListener("change", renderActivity);
  document.querySelectorAll("#report .seg button").forEach((b) => b.addEventListener("click", () => {
    report.scope = b.dataset.scope;
    renderReport();
  }));
  let resizeFrame = 0;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(renderReport);
  });
  setInterval(() => { if (activity.queue.length) sendQueued(); }, 10000);
  setInterval(() => refreshCameras(), 1000);
  await Promise.all([loadRecords(), loadActivity()]);
  logAction("page_opened", "Page opened");
}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
