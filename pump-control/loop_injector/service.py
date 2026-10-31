"""Loopback-only HTTP service for the loop-injector interface."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .automations import AutomationStore
from .pump_backend import PumpController
from .pump_log import PumpActivityLog


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 8192
STATIC_ROUTES = {
    "/": ("public/index.html", "text/html; charset=utf-8"),
    "/public/app.js": ("public/app.js", "text/javascript; charset=utf-8"),
    "/public/styles.css": ("public/styles.css", "text/css; charset=utf-8"),
    "/public/loop-injector-icon.png": ("public/loop-injector-icon.png", "image/png"),
}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class InjectorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, controller: PumpController):
        super().__init__(address, InjectorRequestHandler)
        self.controller = controller


class InjectorRequestHandler(BaseHTTPRequestHandler):
    server: InjectorHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path in STATIC_ROUTES:
            relative, content_type = STATIC_ROUTES[path]
            self._send_bytes(
                200, (PROJECT_ROOT / relative).read_bytes(), content_type=content_type
            )
            return
        if path == "/health":
            self._send_json(200, {"ok": True})
            return
        if path == "/api/status":
            self._send_json(200, self.server.controller.status())
            return
        if path == "/api/automations":
            self._send_json(
                200, self.server.controller.status().get("automations", {})
            )
            return
        if path in ("/api/pump-log", "/api/pump-log.csv"):
            try:
                filters = self._pump_log_filters(
                    parsed.query, export=path.endswith(".csv")
                )
                if path.endswith(".csv"):
                    self._send_bytes(
                        200,
                        self.server.controller.export_pump_log_csv(**filters),
                        content_type="text/csv; charset=utf-8",
                        extra_headers={
                            "Content-Disposition": (
                                'attachment; filename="loop-injector-pump-log.csv"'
                            )
                        },
                    )
                else:
                    self._send_json(
                        200, self.server.controller.query_pump_log(**filters)
                    )
            except (TypeError, ValueError) as exc:
                self._send_json(400, {"error": str(exc)})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            path = urlsplit(self.path).path
            body = self._read_json()
            if path == "/api/lease":
                self._require_keys(body, set())
                result = self.server.controller.renew_lease()
            elif path == "/api/manual":
                self._require_keys(body, {"mask"})
                result = self.server.controller.set_manual_mask(body["mask"])
            elif path == "/api/schedule/start":
                self._require_keys(
                    body, {"order", "burst_seconds", "total_seconds"}
                )
                order = body["order"]
                if not isinstance(order, list):
                    raise ApiError(400, "order must be an array")
                result = self.server.controller.start_schedule(
                    tuple(order), body["burst_seconds"], body["total_seconds"]
                )
            elif path == "/api/schedule/stop":
                self._require_keys(body, set())
                result = self.server.controller.stop_schedule()
            elif path == "/api/automations/save":
                result = self.server.controller.save_automation(body)
            elif path == "/api/automations/delete":
                self._require_keys(body, {"id"})
                result = self.server.controller.delete_automation(body["id"])
            elif path == "/api/automations/enabled":
                self._require_keys(body, {"id", "enabled"})
                result = self.server.controller.set_automation_enabled(
                    body["id"], body["enabled"]
                )
            elif path == "/api/automations/pause":
                self._require_keys(body, set())
                result = self.server.controller.pause_automations()
            elif path == "/api/automations/resume":
                self._require_keys(body, set())
                result = self.server.controller.resume_automations()
            elif path == "/api/emergency-stop":
                self._require_keys(body, set())
                result = self.server.controller.emergency_stop()
            elif path == "/api/recorder/session":
                self._require_keys(body, set())
                token, status = self.server.controller.start_recorder_session()
                result = {"token": token, "status": status}
            elif path == "/api/recorder/demand":
                self._require_keys(body, {"token", "mask"})
                result = self.server.controller.set_recorder_demand(
                    body["token"], body["mask"]
                )
            elif path == "/api/recorder/session/stop":
                self._require_keys(body, {"token"})
                result = self.server.controller.stop_recorder_session(body["token"])
            else:
                raise ApiError(404, "not found")
            self._send_json(200, result)
        except ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except PermissionError as exc:
            self._send_json(403, {"error": str(exc)})
        except (TypeError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            self._send_json(500, {"error": "internal service error"})

    @staticmethod
    def _require_keys(body: dict[str, Any], required: set[str]) -> None:
        if set(body) != required:
            raise ApiError(400, f"request fields must be exactly {sorted(required)}")

    def _read_json(self) -> dict[str, Any]:
        if self.headers.get_content_type() != "application/json":
            raise ApiError(415, "Content-Type must be application/json")
        length_text = self.headers.get("Content-Length")
        try:
            length = int(length_text) if length_text is not None else 0
        except ValueError as exc:
            raise ApiError(400, "invalid Content-Length") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ApiError(413, "request body too large")
        try:
            decoded = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(400, "request body must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ApiError(400, "request body must be a JSON object")
        return decoded

    @staticmethod
    def _pump_log_filters(query: str, *, export: bool) -> dict[str, Any]:
        parsed = parse_qs(query, keep_blank_values=True)
        allowed = {"pump", "from", "to"} if export else {
            "pump", "from", "to", "limit", "offset"
        }
        unknown = set(parsed) - allowed
        if unknown:
            raise ValueError(f"unknown pump log filters: {sorted(unknown)}")
        if any(len(values) != 1 for values in parsed.values()):
            raise ValueError("pump log filters must not be repeated")
        result: dict[str, Any] = {}
        if "pump" in parsed and parsed["pump"][0]:
            result["pump"] = int(parsed["pump"][0])
        if "from" in parsed and parsed["from"][0]:
            result["started_from"] = parsed["from"][0]
        if "to" in parsed and parsed["to"][0]:
            result["started_to"] = parsed["to"][0]
        if not export:
            result["limit"] = int(parsed.get("limit", ["200"])[0])
            result["offset"] = int(parsed.get("offset", ["0"])[0])
        return result

    def _send_json(self, status: int, payload: object) -> None:
        self._send_bytes(
            status,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def serve_until_interrupted(
    controller: PumpController, *, host: str = "127.0.0.1", port: int = 8081
) -> None:
    server = InjectorHTTPServer((host, port), controller)
    controller.start()
    stopping = threading.Event()

    def request_stop(_signum=None, _frame=None) -> None:
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, request_stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        controller.close()
        for sig, old_handler in old_handlers.items():
            signal.signal(sig, old_handler)


def main() -> None:
    port = int(os.environ.get("LOOP_INJECTOR_PORT", "8081"))
    if not 1 <= port <= 65535:
        raise SystemExit("LOOP_INJECTOR_PORT must be between 1 and 65535")
    default_state_root = Path(__file__).resolve().parents[1] / ".loop-injector"
    state_root = Path(
        os.environ.get("LOOP_INJECTOR_STATE_ROOT", default_state_root)
    ).expanduser().resolve()
    automation_store = AutomationStore(state_root / "automations.json")
    pump_log = PumpActivityLog(state_root / "pump-log.sqlite3")
    print(f"loop-injector listening on http://127.0.0.1:{port}", flush=True)
    serve_until_interrupted(
        PumpController(automation_store=automation_store, pump_log=pump_log),
        port=port,
    )


if __name__ == "__main__":
    main()
