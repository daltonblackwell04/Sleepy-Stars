from __future__ import annotations

from io import BytesIO
import json
import time
import urllib.error

import pytest

from startracked_loop_recorder.config import (
    ConfigError, atomic_json_write, default_user_config, load_json,
    validate_user_config,
)
from startracked_loop_recorder.injector_pump import InjectorPumpClient, MAPPING


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    assert predicate()


class Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class FakeInjector:
    def __init__(self):
        self.token_number = 0
        self.token = ""
        self.mask = 0
        self.demands = []
        self.emergency = False
        self.failures = 0

    def status(self):
        effective = 0 if self.emergency else self.mask
        return {
            "connected": True,
            "ready": True,
            "requested_mask": effective,
            "acknowledged_mask": effective,
            "mode": "emergency" if self.emergency else (
                "recorder" if effective else "idle"
            ),
            "arbitration_source": "emergency" if self.emergency else (
                "recorder" if effective else "idle"
            ),
            "emergency_latched": self.emergency,
            "last_event": f"EVENT STATE {effective}",
            "last_error": None,
            "recorder": {
                "session_active": bool(self.token),
                "connected": bool(self.token),
                "lease_active": bool(self.token),
                "lease_remaining_seconds": 2.0,
                "demand_mask": self.mask,
            },
        }

    def open(self, request, timeout):
        assert timeout <= 1.5
        if self.failures:
            self.failures -= 1
            raise urllib.error.URLError("simulated outage")
        path = request.full_url.removeprefix("http://127.0.0.1:8081")
        body = json.loads(request.data) if request.data is not None else {}
        if path == "/api/status":
            payload = self.status()
        elif path == "/api/recorder/session":
            self.token_number += 1
            self.token = f"token-{self.token_number}"
            self.mask = 0
            payload = {"token": self.token, "status": self.status()}
        elif path == "/api/recorder/demand":
            if body["token"] != self.token:
                raise urllib.error.HTTPError(
                    request.full_url, 403, "forbidden", {}, BytesIO(b"bad token")
                )
            self.mask = body["mask"]
            self.demands.append(self.mask)
            payload = self.status()
        elif path == "/api/recorder/session/stop":
            assert body["token"] == self.token
            self.mask = 0
            self.token = ""
            payload = self.status()
        else:
            raise AssertionError(path)
        return Response(json.dumps(payload).encode())


def settings():
    value = default_user_config()["arduino_pump"]
    value["injector_heartbeat_seconds"] = 0.1
    return value


def test_mapping_and_http_session_demand_heartbeat_and_shutdown_zero():
    fake = FakeInjector()
    client = InjectorPumpClient(lambda: settings(), opener=fake.open)
    client.start()
    client.set_session(True, "recording")
    client.set_camera_demand("A", True, "recording")
    client.set_camera_demand("C", True, "recording")
    try:
        wait_for(lambda: client.status()["confirmed_mask"] == 10)
        assert MAPPING == {"A": 2, "B": 1, "C": 8, "D": 4}
        demand_count = len(fake.demands)
        wait_for(lambda: len(fake.demands) > demand_count)
        assert client.status()["agreement"] is True
        assert client.status()["mode"] == "injector"
    finally:
        client.close()
    assert fake.mask == 0
    assert fake.token == ""


def test_idle_client_reports_injector_connectivity_without_opening_session():
    fake = FakeInjector()
    client = InjectorPumpClient(lambda: settings(), opener=fake.open)
    client.start()
    try:
        wait_for(lambda: client.status()["ready"])
        assert client.status()["connected"] is True
        assert fake.token == ""
    finally:
        client.close()


def test_reconnect_reestablishes_session_and_current_demand():
    fake = FakeInjector()
    client = InjectorPumpClient(lambda: settings(), opener=fake.open)
    client.start()
    client.set_session(True, "recording")
    client.set_camera_demand("B", True, "recording")
    try:
        wait_for(lambda: client.status()["confirmed_mask"] == 1)
        original_sessions = fake.token_number
        fake.failures = 1
        wait_for(lambda: not client.status()["connected"])
        wait_for(lambda: fake.token_number > original_sessions)
        wait_for(lambda: client.status()["confirmed_mask"] == 1)
    finally:
        client.close()


def test_emergency_latch_is_reported_and_recorder_cannot_clear_it():
    fake = FakeInjector()
    client = InjectorPumpClient(lambda: settings(), opener=fake.open)
    client.start()
    client.set_session(True, "recording")
    client.set_camera_demand("D", True, "recording")
    try:
        wait_for(lambda: client.status()["confirmed_mask"] == 4)
        fake.emergency = True
        wait_for(lambda: client.status()["emergency_latched"])
        status = client.status()
        assert status["desired_mask"] == 4
        assert status["confirmed_mask"] == 0
        assert status["agreement"] is False
        assert status["arbitration_source"] == "emergency"
        time.sleep(0.2)
        assert fake.mask == 4
        assert fake.emergency is True
    finally:
        client.close()


def test_schema6_migrates_to_injector_and_direct_mode_remains_explicit(tmp_path):
    old = default_user_config()
    old["schema_version"] = 6
    for key in (
        "mode", "injector_url", "injector_heartbeat_seconds",
        "injector_timeout_seconds",
    ):
        old["arduino_pump"].pop(key)
    path = tmp_path / "config.json"
    atomic_json_write(path, old)
    migrated = load_json(path, default_user_config(), validate_user_config)
    assert migrated["schema_version"] == 7
    assert migrated["arduino_pump"]["mode"] == "injector"
    assert migrated["arduino_pump"]["injector_url"] == "http://127.0.0.1:8081"

    migrated["arduino_pump"]["mode"] = "direct"
    validate_user_config(migrated)
    migrated["arduino_pump"]["injector_url"] = "http://192.168.1.2:8081"
    with pytest.raises(ConfigError, match="loopback"):
        validate_user_config(migrated)
