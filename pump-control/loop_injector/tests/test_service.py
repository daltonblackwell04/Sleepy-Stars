from __future__ import annotations

from http.client import HTTPConnection
import json
import threading
import unittest

from loop_injector.service import InjectorHTTPServer


class FakeController:
    def __init__(self):
        self.calls = []
        self.state = {
            "connected": True,
            "ready": True,
            "mode": "idle",
            "requested_mask": 0,
            "acknowledged_mask": 0,
            "lease_active": True,
            "lease_remaining_seconds": 1,
            "schedule": None,
            "automations": {"paused": False, "mask": 0, "rules": []},
            "pump_log": {
                "healthy": True,
                "total_intervals": 1,
                "open_intervals": 0,
                "last_error": None,
            },
            "last_event": "EVENT REARMED",
            "last_error": None,
        }

    def status(self):
        return self.state

    def renew_lease(self):
        self.calls.append(("lease",))
        return self.state

    def set_manual_mask(self, mask):
        self.calls.append(("manual", mask))
        return self.state

    def start_schedule(self, order, burst, total):
        self.calls.append(("schedule", order, burst, total))
        return self.state

    def stop_schedule(self):
        self.calls.append(("schedule-stop",))
        return self.state

    def emergency_stop(self):
        self.calls.append(("emergency-stop",))
        return self.state

    def start_recorder_session(self):
        self.calls.append(("recorder-session",))
        return "secret-token", self.state

    def set_recorder_demand(self, token, mask):
        self.calls.append(("recorder-demand", token, mask))
        if token == "wrong":
            raise PermissionError("invalid recorder session token")
        return self.state

    def stop_recorder_session(self, token):
        self.calls.append(("recorder-stop", token))
        return self.state

    def save_automation(self, body):
        self.calls.append(("automation-save", body))
        return self.state

    def delete_automation(self, rule_id):
        self.calls.append(("automation-delete", rule_id))
        return self.state

    def set_automation_enabled(self, rule_id, enabled):
        self.calls.append(("automation-enabled", rule_id, enabled))
        return self.state

    def pause_automations(self):
        self.calls.append(("automation-pause",))
        return self.state

    def resume_automations(self):
        self.calls.append(("automation-resume",))
        return self.state

    def query_pump_log(self, **filters):
        self.calls.append(("pump-log", filters))
        return {"total": 0, "limit": filters["limit"], "offset": filters["offset"], "rows": []}

    def export_pump_log_csv(self, **filters):
        self.calls.append(("pump-log-csv", filters))
        return b"pump_number,started_at_utc\\n"


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        self.server = InjectorHTTPServer(("127.0.0.1", 0), self.controller)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def request(self, method, path, body=None, content_type="application/json"):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=2)
        headers = {}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers = {"Content-Type": content_type}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        result = json.loads(raw) if response.headers.get_content_type() == "application/json" else raw
        headers = dict(response.headers)
        status = response.status
        connection.close()
        return status, headers, result

    def test_static_assets_health_and_status_are_loopback_http_surfaces(self):
        for path in (
            "/",
            "/public/app.js",
            "/public/styles.css",
            "/health",
            "/api/status",
            "/api/automations",
        ):
            with self.subTest(path=path):
                status, headers, _body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertEqual(headers["Cache-Control"], "no-store")

    def test_manual_schedule_stop_and_emergency_routes_are_bounded(self):
        cases = (
            ("/api/lease", {}, ("lease",)),
            ("/api/manual", {"mask": 5}, ("manual", 5)),
            (
                "/api/schedule/start",
                {"order": [1, 2, 3, 4], "burst_seconds": 10, "total_seconds": 600},
                ("schedule", (1, 2, 3, 4), 10, 600),
            ),
            ("/api/schedule/stop", {}, ("schedule-stop",)),
            ("/api/emergency-stop", {}, ("emergency-stop",)),
        )
        for path, body, expected in cases:
            with self.subTest(path=path):
                status, _headers, response = self.request("POST", path, body)
                self.assertEqual(status, 200)
                self.assertTrue(response["ready"])
                self.assertEqual(self.controller.calls[-1], expected)

    def test_unknown_fields_wrong_content_type_and_unknown_routes_are_rejected(self):
        status, _headers, _body = self.request(
            "POST", "/api/manual", {"mask": 1, "device": "/dev/ttyACM0"}
        )
        self.assertEqual(status, 400)
        status, _headers, _body = self.request(
            "POST", "/api/manual", {"mask": 1}, content_type="text/plain"
        )
        self.assertEqual(status, 415)
        status, _headers, _body = self.request("GET", "/README.md")
        self.assertEqual(status, 404)

    def test_recorder_session_demand_stop_and_token_rejection(self):
        status, _headers, result = self.request(
            "POST", "/api/recorder/session", {}
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["token"], "secret-token")
        self.assertEqual(result["status"]["mode"], "idle")
        self.assertNotIn("token", result["status"])

        status, _headers, _result = self.request(
            "POST", "/api/recorder/demand",
            {"token": "secret-token", "mask": 10},
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            self.controller.calls[-1],
            ("recorder-demand", "secret-token", 10),
        )
        status, _headers, _result = self.request(
            "POST", "/api/recorder/session/stop", {"token": "secret-token"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(self.controller.calls[-1], ("recorder-stop", "secret-token"))

        status, _headers, _result = self.request(
            "POST", "/api/recorder/demand", {"token": "wrong", "mask": 1}
        )
        self.assertEqual(status, 403)

    def test_automation_crud_pause_and_resume_routes(self):
        rule = {
            "name": "Day dosing",
            "enabled": True,
            "mask": 3,
            "weekdays": [0, 1],
            "start_time": "00:00",
            "end_time": "18:00",
            "on_minutes": 5,
            "block_minutes": 30,
        }
        cases = (
            ("/api/automations/save", rule, ("automation-save", rule)),
            (
                "/api/automations/enabled",
                {"id": "a" * 32, "enabled": False},
                ("automation-enabled", "a" * 32, False),
            ),
            (
                "/api/automations/delete",
                {"id": "a" * 32},
                ("automation-delete", "a" * 32),
            ),
            ("/api/automations/pause", {}, ("automation-pause",)),
            ("/api/automations/resume", {}, ("automation-resume",)),
        )
        for path, body, expected in cases:
            with self.subTest(path=path):
                status, _headers, response = self.request("POST", path, body)
                self.assertEqual(status, 200)
                self.assertTrue(response["ready"])
                self.assertEqual(self.controller.calls[-1], expected)

    def test_pump_log_query_and_csv_export_have_bounded_filters(self):
        status, _headers, result = self.request(
            "GET", "/api/pump-log?pump=2&limit=50&offset=10"
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["limit"], 50)
        self.assertEqual(
            self.controller.calls[-1],
            ("pump-log", {"pump": 2, "limit": 50, "offset": 10}),
        )

        status, headers, body = self.request(
            "GET", "/api/pump-log.csv?pump=2"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertIn(b"pump_number", body)

        status, _headers, _body = self.request(
            "GET", "/api/pump-log?device=/dev/ttyACM0"
        )
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
