from __future__ import annotations

import threading
import time

import pytest

from startracked_loop_recorder.arduino_pump import ArduinoPumpWorker, encode_mask
from startracked_loop_recorder.config import default_user_config
from startracked_loop_recorder.hue import PressurePlateService


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


class FakeSerial:
    def __init__(self, *, fail_after=None):
        self._lock = threading.Lock()
        # Arduino Serial.println emits CRLF, not bare LF.
        self._incoming = bytearray(b"MULTICAM_PUMP_READY\r\nEVENT BOOT\r\n")
        self.writes: list[tuple[float, int]] = []
        self.closed = False
        self.latched = True
        self.mask = 0
        self.fail_after = fail_after

    @property
    def in_waiting(self):
        with self._lock:
            return len(self._incoming)

    def read(self, size):
        with self._lock:
            chunk = bytes(self._incoming[:size])
            del self._incoming[:size]
            return chunk

    def write(self, payload):
        mask = int(payload.decode("ascii").removeprefix("PUMPS:").strip())
        with self._lock:
            if self.fail_after is not None and len(self.writes) >= self.fail_after:
                raise OSError("simulated serial loss")
            self.writes.append((time.monotonic(), mask))
            if self.latched:
                if mask == 0:
                    self.latched = False
                    self._incoming.extend(b"EVENT REARMED\r\n")
                else:
                    self._incoming.extend(b"EVENT IGNORED LATCHED\r\n")
            elif mask != self.mask:
                self.mask = mask
                self._incoming.extend(f"EVENT STATE {mask}\r\n".encode())
        return len(payload)

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def inject(self, line):
        with self._lock:
            self._incoming.extend((line + "\r\n").encode())


def arduino_settings(**changes):
    settings = default_user_config()["arduino_pump"]
    settings.update(changes)
    return settings


@pytest.mark.parametrize("mask", range(16))
def test_all_mask_encodings(mask):
    assert encode_mask(mask) == f"PUMPS:{mask}\n".encode()


@pytest.mark.parametrize(
    ("camera", "bit"), [("A", 2), ("B", 1), ("C", 8), ("D", 4)]
)
def test_exact_camera_mapping_and_all_camera_mask_15(camera, bit):
    mapping = default_user_config()["arduino_pump"]["mapping"]
    assert mapping[camera] == bit
    assert mapping["A"] | mapping["B"] | mapping["C"] | mapping["D"] == 15


@pytest.mark.parametrize("target_mask", range(16))
def test_every_camera_demand_combination_reaches_exact_mask(target_mask):
    port = FakeSerial()
    worker = ArduinoPumpWorker(
        lambda: arduino_settings(), serial_factory=lambda _path, _baud: port
    )
    worker.start()
    worker.set_session(True, "session")
    wait_for(lambda: worker.status()["ready"])
    try:
        for camera, bit in {"A": 2, "B": 1, "C": 8, "D": 4}.items():
            worker.set_camera_demand(camera, bool(target_mask & bit), "session")
        wait_for(lambda: worker.status()["confirmed_mask"] == target_mask)
        assert worker.desired_mask() == target_mask
    finally:
        worker.close()


@pytest.mark.parametrize(
    ("mask", "payload"), [(0, b"PUMPS:0\n"), (15, b"PUMPS:15\n")]
)
def test_mask_encoding(mask, payload):
    assert encode_mask(mask) == payload


@pytest.mark.parametrize("bad", [-1, 16, True, "1"])
def test_mask_encoding_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        encode_mask(bad)


def test_handshake_mapping_ack_heartbeat_and_shutdown_zero():
    port = FakeSerial()
    events = []
    worker = ArduinoPumpWorker(
        lambda: arduino_settings(), lambda event, details: events.append((event, details)),
        lambda _path, _baud: port,
    )
    worker.start()
    worker.set_session(True, "session")
    wait_for(lambda: worker.status()["ready"])
    worker.set_camera_demand("A", True, "session")
    wait_for(lambda: worker.status()["confirmed_mask"] == 2)
    worker.set_camera_demand("B", True, "session")
    wait_for(lambda: worker.status()["confirmed_mask"] == 3)
    worker.set_camera_demand("C", True, "session")
    wait_for(lambda: worker.status()["confirmed_mask"] == 11)
    worker.set_camera_demand("D", True, "session")
    wait_for(lambda: worker.status()["confirmed_mask"] == 15)
    worker.set_camera_demand("A", False, "session")
    wait_for(lambda: worker.status()["confirmed_mask"] == 13)
    time.sleep(0.34)
    timestamps = [at for at, _mask in port.writes[-3:]]
    assert max(b - a for a, b in zip(timestamps, timestamps[1:])) < 0.23
    worker.close()
    assert port.writes[-1][1] == 0
    assert port.mask == 0
    assert any(event == "arduino_command" for event, _ in events)
    assert any(
        event == "arduino_event" and details["firmware_event"] == "EVENT REARMED"
        for event, details in events
    )


def test_firmware_crlf_lines_are_preserved_and_normalized():
    port = FakeSerial()
    worker = ArduinoPumpWorker(
        lambda: arduino_settings(), serial_factory=lambda _path, _baud: port
    )
    worker.start()
    try:
        wait_for(lambda: worker.status()["ready"])
        assert worker.status()["confirmed_mask"] == 0
        assert worker.status()["last_event"] == "EVENT REARMED"
        assert all(mask == 0 for _at, mask in port.writes)
    finally:
        worker.close()


def test_bootloader_receives_no_heartbeat_stream_before_ready():
    class DelayedBootSerial(FakeSerial):
        def __init__(self):
            super().__init__()
            self._incoming.clear()
            self.booted = False

        def boot(self):
            with self._lock:
                self.booted = True
                self.latched = True
                self._incoming.extend(b"MULTICAM_PUMP_READY\r\nEVENT BOOT\r\n")

        def write(self, payload):
            mask = int(payload.decode("ascii").removeprefix("PUMPS:").strip())
            with self._lock:
                self.writes.append((time.monotonic(), mask))
                if not self.booted:
                    return len(payload)
                if self.latched and mask == 0:
                    self.latched = False
                    self._incoming.extend(b"EVENT REARMED\r\n")
                elif mask != self.mask:
                    self.mask = mask
                    self._incoming.extend(f"EVENT STATE {mask}\r\n".encode())
            return len(payload)

    port = DelayedBootSerial()
    worker = ArduinoPumpWorker(
        lambda: arduino_settings(), serial_factory=lambda _path, _baud: port
    )
    worker.start()
    try:
        time.sleep(0.35)
        assert [mask for _at, mask in port.writes] == [0]
        port.boot()
        wait_for(lambda: worker.status()["ready"])
        assert worker.status()["confirmed_mask"] == 0
        assert len(port.writes) >= 2
    finally:
        worker.close()


def test_reconnect_zero_then_current_authoritative_demand():
    first = FakeSerial(fail_after=4)
    second = FakeSerial()
    ports = iter((first, second))
    worker = ArduinoPumpWorker(
        lambda: arduino_settings(reconnect_max_seconds=0.25),
        serial_factory=lambda _path, _baud: next(ports),
    )
    worker.start()
    worker.set_session(True, "session")
    worker.set_camera_demand("A", True, "session")
    wait_for(lambda: worker.status()["reconnect_count"] >= 2)
    wait_for(lambda: worker.status()["confirmed_mask"] == 2)
    assert second.writes[0][1] == 0
    assert any(mask == 2 for _at, mask in second.writes[1:])
    worker.close()


def test_stale_session_demand_is_not_replayed_after_reconnect():
    first = FakeSerial(fail_after=3)
    second = FakeSerial()
    ports = iter((first, second))
    worker = ArduinoPumpWorker(
        lambda: arduino_settings(reconnect_max_seconds=0.25),
        serial_factory=lambda _path, _baud: next(ports),
    )
    worker.start()
    worker.set_session(True, "old")
    worker.set_camera_demand("B", True, "old")
    wait_for(lambda: worker.status()["confirmed_mask"] == 1)
    worker.set_session(False)
    assert worker.wait_for_zero()
    wait_for(lambda: worker.status()["reconnect_count"] >= 2)
    time.sleep(0.1)
    assert all(mask == 0 for _at, mask in second.writes)
    worker.set_camera_demand("A", True, "old")
    assert worker.desired_mask() == 0
    worker.close()


def test_watchdog_rearms_before_restoring_current_demand_and_malformed_is_audited():
    port = FakeSerial()
    events = []
    worker = ArduinoPumpWorker(
        lambda: arduino_settings(), lambda event, details: events.append((event, details)),
        lambda _path, _baud: port,
    )
    worker.start()
    worker.set_session(True, "session")
    worker.set_camera_demand("A", True, "session")
    wait_for(lambda: worker.status()["confirmed_mask"] == 2)
    writes_before_watchdog = len(port.writes)
    port.latched = True
    port.mask = 0
    port.inject("EVENT WATCHDOG")
    wait_for(lambda: any(e == "arduino_event" and d["firmware_event"] == "EVENT WATCHDOG"
                         for e, d in events))
    wait_for(lambda: worker.status()["confirmed_mask"] == 2)
    after_watchdog = [mask for _at, mask in port.writes[writes_before_watchdog:]]
    assert after_watchdog[0] == 0
    assert 2 in after_watchdog[1:]
    port.inject("EVENT STATE nope")
    wait_for(lambda: any(e == "arduino_malformed_event" for e, _ in events))
    assert "malformed Arduino event" in worker.status()["last_error"]
    worker.close()


def test_slow_hue_request_cannot_interrupt_serial_heartbeat():
    port = FakeSerial()
    serial_worker = ArduinoPumpWorker(
        lambda: arduino_settings(), serial_factory=lambda _path, _baud: port
    )
    entered, release = threading.Event(), threading.Event()

    class BlockingBridge:
        def __init__(self, _address, _key):
            pass

        def set_output(self, _output, _on):
            entered.set()
            assert release.wait(2)

    pressure_settings = default_user_config()["pressure_plate"]
    pressure_settings.update(
        enabled=True, app_key="key", enter_seconds=0.0, exit_seconds=0.0,
        stale_seconds=2.0,
    )
    pressure_settings["outputs"]["A"] = "pump-a"
    pressure_settings["polygons"] = {
        camera: [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        for camera in "ABCD"
    }
    pressure = PressurePlateService(
        lambda: pressure_settings, threading.Event(), BlockingBridge,
        arduino=serial_worker,
    )
    serial_worker.start()
    serial_worker.set_session(True, "session")
    wait_for(lambda: serial_worker.status()["ready"])
    pressure._armed = True
    pressure._session_token = "session"
    pressure._states["A"]["occupied"] = True
    reconcile = threading.Thread(target=pressure._reconcile, args=("A", "occupancy"))
    reconcile.start()
    try:
        assert entered.wait(1)
        wait_for(lambda: serial_worker.status()["confirmed_mask"] == 2)
        start = len(port.writes)
        time.sleep(0.36)
        heartbeat_writes = port.writes[start:]
        assert len(heartbeat_writes) >= 2
        assert max(
            b[0] - a[0] for a, b in zip(heartbeat_writes, heartbeat_writes[1:])
        ) < 0.23
    finally:
        release.set()
        reconcile.join(timeout=2)
        serial_worker.close()
