from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from startracked_loop_recorder.app import Recorder, page
from startracked_loop_recorder.config import (
    ConfigError, atomic_json_write, default_deployment, default_user_config,
    load_json, validate_user_config,
)
from startracked_loop_recorder.hue import (
    HOURLY_PULSE_SECONDS, HUE_PAIRING_DEVICE_TYPE, HueBridge, InferenceEvent,
    PressurePlateService,
    _validate_hue_device_type,
)
from startracked_loop_recorder.starseg import StarSegJob, StarSegWorker
from startracked_loop_recorder.storage import finalize_segment
from test_recorder import make_segment


class FakeBridge:
    calls: list[tuple[str, bool]] = []
    failures = 0

    def __init__(self, address, key):
        self.address, self.key = address, key

    def set_output(self, output, on):
        type(self).calls.append((output, on))
        if type(self).failures:
            type(self).failures -= 1
            raise RuntimeError("temporary bridge failure")

    def discover_outputs(self):
        return [
            {"id": "light-a", "name": "Aquarium A", "kind": "light"},
            {"id": "plug-b", "name": "Pump Plug", "kind": "smart_plug"},
        ]

    def pair(self):
        return "paired-application-key"


class FakeArduino:
    def __init__(self, *, connected=True, ready=True, confirmed_mask=0, error=""):
        self.sessions = []
        self.demands = {camera: False for camera in "ABCD"}
        self.connected = connected
        self.ready = ready
        self.confirmed_mask = confirmed_mask
        self.error = error

    def set_session(self, armed, session_token=""):
        self.sessions.append((armed, session_token))
        if not armed:
            self.demands = {camera: False for camera in "ABCD"}
            self.confirmed_mask = 0

    def set_camera_demand(self, camera, desired, session_token):
        if camera in self.demands:
            self.demands[camera] = desired

    def status(self):
        mapping = {"A": 2, "B": 1, "C": 8, "D": 4}
        desired_mask = 0
        for camera, bit in mapping.items():
            if self.demands[camera]:
                desired_mask |= bit
        return {
            "enabled": True, "connected": self.connected, "ready": self.ready,
            "watchdog": False, "desired_mask": desired_mask,
            "confirmed_mask": self.confirmed_mask, "mapping": mapping,
            "last_error": self.error, "serial_path": "/dev/serial/by-id/test",
            "baud": 115200, "heartbeat_seconds": 0.15, "protocol": "PUMPS:<mask>\\n",
            "agreement": self.confirmed_mask == desired_mask,
        }


def settings(**changes):
    value = default_user_config()["pressure_plate"]
    value.update({
        "enabled": True, "app_key": "key", "enter_seconds": 0.0,
        "exit_seconds": 0.0, "stale_seconds": 0.2,
    })
    value["outputs"]["A"] = "light-a"
    value["outputs"]["B"] = "plug-b"
    value["polygons"] = {
        camera: [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        for camera in "ABCD"
    }
    value.update(changes)
    return value


def event(
    camera="A", ts=1.0, names=("sunflower_star",), *,
    session="session", ok=True, lane="control", capture_monotonic=0.0,
):
    return InferenceEvent(
        camera=camera, frame_index=int(ts * 10), frame_ts=ts,
        received_monotonic=time.monotonic(), recorded=True, polygon_valid=True,
        detections=tuple((name, 10.0, 10.0) for name in names),
        events_jsonl=None, session_token=session, inference_ok=ok,
        lane=lane, capture_monotonic=capture_monotonic,
    )


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


@pytest.fixture
def service():
    FakeBridge.calls = []
    FakeBridge.failures = 0
    current = settings()
    worker = PressurePlateService(lambda: current, threading.Event(), FakeBridge)
    worker.start()
    worker.arm(True, "session")
    wait_for(lambda: worker.status()["armed"])
    FakeBridge.calls = []
    yield worker, current
    worker.close()
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_sunflower_only_hysteresis_idempotency_and_camera_isolation(service):
    worker, _ = service
    worker.submit(event(names=("other_sea_star",), ts=1))
    worker.submit(event(names=("other_sea_star",), ts=2))
    worker.submit(event(ts=3))
    worker.submit(event(ts=4))
    worker.flush()
    assert worker.status()["cameras"]["A"]["occupied"] is True
    assert worker.status()["cameras"]["B"]["occupied"] is False
    assert FakeBridge.calls == [("light-a", True)]

    worker.submit(event(ts=5))
    worker.submit(event(ts=6))
    worker.flush()
    assert FakeBridge.calls == [("light-a", True)]

    worker.submit(event(names=(), ts=7))
    worker.submit(event(names=(), ts=8))
    worker.flush()
    assert FakeBridge.calls[-1] == ("light-a", False)


def test_stale_out_of_order_old_session_and_failure_fail_closed(service):
    worker, _ = service
    worker.submit(event(ts=10))
    worker.submit(event(ts=11))
    worker.flush()
    assert worker.status()["cameras"]["A"]["on"]

    worker.submit(event(ts=9, names=()))
    worker.flush()
    assert worker.status()["cameras"]["A"]["on"]

    worker.submit(event(ts=12, session="old"))
    worker.flush()
    assert worker.status()["cameras"]["A"]["on"] is True

    worker.submit(event(ts=15, ok=False))
    worker.flush()
    assert worker.status()["cameras"]["A"]["on"] is False


def test_archival_and_already_stale_results_cannot_actuate(service):
    worker, current = service
    worker.submit(event(ts=1, lane="archival"))
    worker.submit(event(ts=2, lane="archival"))
    worker.flush()
    assert worker.status()["cameras"]["A"]["occupied"] is False
    assert FakeBridge.calls == []

    current["stale_seconds"] = 0.2
    stale_capture = time.monotonic() - 1.0
    worker.submit(event(ts=3, capture_monotonic=stale_capture))
    worker.submit(event(ts=4, capture_monotonic=stale_capture))
    worker.flush()
    state = worker.status()["cameras"]["A"]
    assert state["occupied"] is False
    assert state["occupancy_fresh"] is False
    assert FakeBridge.calls == []


def test_retry_recovery_disarm_and_stale_reconciliation(service):
    worker, current = service
    current["stale_seconds"] = 2.0
    FakeBridge.failures = 1
    worker.submit(event(ts=1))
    worker.submit(event(ts=2))
    worker.flush()
    time.sleep(0.55)
    worker.submit(event(ts=3))
    worker.flush()
    wait_for(lambda: worker.status()["cameras"]["A"]["on"])
    assert len(FakeBridge.calls) >= 2
    worker.arm(False)
    wait_for(lambda: not worker.status()["armed"])
    assert FakeBridge.calls[-1] == ("plug-b", False)  # all assigned outputs reconciled OFF


def test_discovery_pair_and_light_resource_contract(service):
    worker, current = service
    assert [item["kind"] for item in worker.discover()] == ["light", "smart_plug"]
    assert worker.selectable_output_ids() == {"light-a", "plug-b"}
    assert worker.pair() == "paired-application-key"

    bridge = HueBridge("bridge", "key")
    bridge._request = lambda *args, **kwargs: {"data": [
        {"id": "1", "metadata": {"name": "Lamp", "archetype": "classic_bulb"}},
        {"id": "2", "metadata": {"name": "Socket", "archetype": "plug"}},
    ]}
    assert bridge.discover_outputs() == [
        {"id": "1", "name": "Lamp", "kind": "light"},
        {"id": "2", "name": "Socket", "kind": "smart_plug"},
    ]


def test_hue_pair_uses_bridge_compliant_device_type():
    calls = []
    bridge = HueBridge("192.0.2.10")

    def request(method, path, body=None, **kwargs):
        calls.append((method, path, body, kwargs))
        return [{"success": {"username": "application-key"}}]

    bridge._request = request
    assert bridge.pair() == "application-key"
    assert calls == [(
        "POST", "/api", {"devicetype": "startracked#plates"},
        {"authenticated": False},
    )]

    application_name, device_name = HUE_PAIRING_DEVICE_TYPE.split("#")
    assert len(HUE_PAIRING_DEVICE_TYPE) <= 40
    assert 0 < len(application_name) <= 20
    assert 0 < len(device_name) <= 19


@pytest.mark.parametrize("devicetype", [
    "missing-separator",
    "#device",
    "application#",
    "application#device#extra",
    f"{'a' * 21}#device",
    f"application#{'d' * 20}",
    "application#bad\nname",
])
def test_hue_pairing_device_type_rejects_invalid_identifiers(devicetype):
    with pytest.raises(ValueError, match="Hue devicetype"):
        _validate_hue_device_type(devicetype)


def test_config_validation_v2_migration_and_unique_assignments(tmp_path):
    old = default_user_config()
    old["schema_version"] = 2
    old.pop("pressure_plate")
    path = tmp_path / "config.json"
    atomic_json_write(path, old)
    migrated = load_json(path, default_user_config(), validate_user_config)
    assert migrated["schema_version"] == 7
    assert migrated["arduino_pump"]["mode"] == "injector"
    assert migrated["arduino_pump"]["mapping"] == {"A": 2, "B": 1, "C": 8, "D": 4}
    assert migrated["pressure_plate_polygons"] == {camera: [] for camera in "ABCD"}
    assert migrated["pressure_plate"]["bridge_address"] == "192.0.2.10"

    bad = default_user_config()
    bad["pressure_plate"]["stale_seconds"] = 1
    bad["pressure_plate"]["exit_seconds"] = 2
    with pytest.raises(ConfigError, match="stale"):
        validate_user_config(bad)
    bad = default_user_config()
    bad["pressure_plate"]["outputs"].update(A="same", B="same")
    with pytest.raises(ConfigError, match="only one"):
        validate_user_config(bad)


def test_v3_migration_preserves_enabled_and_live_output_assignments(tmp_path):
    old = default_user_config()
    old["schema_version"] = 3
    old.pop("pressure_plate_polygons")
    old["pressure_plate"]["enabled"] = True
    old["pressure_plate"]["outputs"].update(
        C="bbbbbbbb-0000-4000-8000-00000000000b",
        D="aaaaaaaa-0000-4000-8000-00000000000a",
    )
    path = tmp_path / "config.json"
    atomic_json_write(path, old)
    migrated = load_json(path, default_user_config(), validate_user_config)
    assert migrated["pressure_plate"]["enabled"] is True
    assert migrated["pressure_plate"]["outputs"]["C"].startswith("bbbbbbbb")
    assert migrated["pressure_plate"]["outputs"]["D"].startswith("aaaaaaaa")
    assert migrated["pressure_plate_polygons"] == {camera: [] for camera in "ABCD"}


def test_v5_mapping_migration_preserves_all_assignments_and_geometries(tmp_path):
    old = default_user_config()
    old["schema_version"] = 5
    old["arduino_pump"]["mapping"] = {"A": 2, "B": 1, "C": 0, "D": 0}
    old["pressure_plate"]["outputs"].update(
        A="pump-a", B="pump-b", C="pump-c", D="pump-d",
    )
    old["pressure_plate_polygons"]["C"] = [
        [0.2, 0.2], [0.4, 0.2], [0.3, 0.4],
    ]
    old["camera_polygons"]["D"] = [
        [0.1, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1, 0.8],
    ]
    expected_outputs = json.loads(json.dumps(old["pressure_plate"]["outputs"]))
    expected_plates = json.loads(json.dumps(old["pressure_plate_polygons"]))
    expected_rois = json.loads(json.dumps(old["camera_polygons"]))
    path = tmp_path / "config.json"
    atomic_json_write(path, old)

    migrated = load_json(path, default_user_config(), validate_user_config)
    assert migrated["schema_version"] == 7
    assert migrated["arduino_pump"]["mode"] == "injector"
    assert migrated["arduino_pump"]["mapping"] == {"A": 2, "B": 1, "C": 8, "D": 4}
    assert migrated["pressure_plate"]["outputs"] == expected_outputs
    assert migrated["pressure_plate_polygons"] == expected_plates
    assert migrated["camera_polygons"] == expected_rois


def test_complete_four_camera_pump_assignments_and_mapping():
    config = default_user_config()
    config["pressure_plate"]["outputs"] = {
        "A": "aaaaaaaa-0000-4000-8000-00000000000a",
        "B": "bbbbbbbb-0000-4000-8000-00000000000b",
        "C": "cccccccc-0000-4000-8000-00000000000c",
        "D": "dddddddd-0000-4000-8000-00000000000d",
    }
    validate_user_config(config)
    assert config["pressure_plate"]["outputs"]["A"].startswith("aaaaaaaa")
    assert config["pressure_plate"]["outputs"]["B"].startswith("bbbbbbbb")
    assert config["pressure_plate"]["outputs"]["C"].startswith("cccccccc")
    assert config["pressure_plate"]["outputs"]["D"].startswith("dddddddd")
    assert config["arduino_pump"]["mapping"] == {"A": 2, "B": 1, "C": 8, "D": 4}


def test_starseg_roi_then_nested_pressure_plate_occupancy_and_empty_fail_closed():
    published = []
    worker = StarSegWorker(lambda: {"enabled": True}, threading.Event(), published.append)
    record = {
        "width": 100, "height": 100,
        "detections": [
            {"class_name": "sunflower_star", "roi_test_point": [25, 25]},
            {"class_name": "sunflower_star", "roi_test_point": [75, 75]},
            {"class_name": "other_sea_star", "roi_test_point": [25, 25]},
        ],
    }
    job = StarSegJob(
        "A", 1, 1.0, "", None, [[0, 0], [1, 0], [1, 1], [0, 1]],
        "Trial", None, True, "session", [[.1, .1], [.5, .1], [.5, .5], [.1, .5]],
    )
    worker._publish(job, record)
    assert published[-1].occupied is True
    assert [item[0] for item in published[-1].detections].count("sunflower_star") == 1
    job.pressure_polygon = []
    worker._publish(job, record)
    assert published[-1].polygon_valid is False
    assert published[-1].occupied is False
    assert len(record["detections"]) == 3  # StarSeg ROI output/provenance is untouched.


def test_recorder_lifecycle_ui_and_output_validation(tmp_path):
    deployment_path, config_path = tmp_path / "deployment.json", tmp_path / "config.json"
    atomic_json_write(deployment_path, default_deployment(tmp_path))
    config = default_user_config()
    config["record_root"] = str(tmp_path / "recordings")
    atomic_json_write(config_path, config)
    recorder = Recorder(deployment_path, config_path)
    recorder.pressure.bridge_factory = FakeBridge
    recorder.pressure.start()
    try:
        assert not recorder.status()["pressure_plate"]["armed"]
        recorder.set_recording(True)
        wait_for(lambda: recorder.status()["pressure_plate"]["armed"])
        recorder.set_recording(False)
        wait_for(lambda: not recorder.status()["pressure_plate"]["armed"])
        with pytest.raises(ConfigError, match="discovered"):
            recorder.apply_control({"camera": ["A"], "hue_output": ["sensor-id"]})
        recorder.pressure.discover()
        recorder.apply_control({"camera": ["A"], "hue_output": ["light-a"]})
    finally:
        recorder.pressure.close()
        recorder.pressure.join(timeout=2)
        recorder.shutdown_event.set()

    html = page().decode()
    assert "/hue/pair" in html and "/hue/discover" in html
    assert "Pair (press bridge button)" in html
    assert "Hue output" in html and "Occupied" in html


def test_pressure_events_finalize_safely_with_and_without_events(tmp_path):
    segment = make_segment(tmp_path / "with")
    segment.pressure_polygon = [[.2, .2], [.8, .2], [.5, .8]]
    event_path = segment.staging / "pressure_plate_events.jsonl"
    event_path.write_text('{"event":"occupancy","occupied":true}\n')
    files = finalize_segment(segment, 1001.5, 34, "medium")
    metadata = json.loads(files[1].read_text())
    assert metadata["pressure_plate"]["normalized"] == [[.2, .2], [.8, .2], [.5, .8]]
    assert metadata["pressure_plate"]["valid"] is True
    final_events = Path(metadata["pressure_plate_events_jsonl"])
    assert "occupancy" in final_events.read_text()

    segment = make_segment(tmp_path / "without")
    files = finalize_segment(segment, 1001.5, 34, "medium")
    metadata = json.loads(files[1].read_text())
    assert Path(metadata["pressure_plate_events_jsonl"]).read_text() == ""


def test_hue_network_delay_never_blocks_starseg_worker(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class BlockingBridge(FakeBridge):
        def set_output(self, output, on):
            entered.set()
            assert release.wait(2)

    current = settings(stale_seconds=2.0)
    shutdown = threading.Event()
    pressure = PressurePlateService(lambda: current, shutdown, BlockingBridge)
    starseg = StarSegWorker(lambda: {"enabled": True}, shutdown, pressure.submit)
    starseg._infer = lambda job: {
        "camera": "A", "frame_index": job.frame_index, "frame_ts": job.frame_ts,
        "detections": [{"class_name": "sunflower_star", "roi_test_point": [1, 1]}],
    }
    pressure.start()
    pressure.arm(True, "session")
    starseg.start()
    output = tmp_path / "starseg_frames.jsonl"
    try:
        for index in range(2):
            starseg.submit(StarSegJob(
                "A", index, float(index + 1), "", None,
                [[0, 0], [1, 0], [0, 1]], "Trial", output, True, "session",
            ))
        starseg.wait_recorded(output, timeout=0.5)
        assert starseg.metrics["processed_recorded"] == 2
        assert entered.wait(1)
    finally:
        release.set()
        starseg.close()
        starseg.join(timeout=2)
        pressure.close()
        pressure.join(timeout=2)


class FakeClock:
    def __init__(self, wall=1_800_000_123.0, monotonic=100.0):
        self.wall = wall
        self.monotonic = monotonic

    def time(self):
        return self.wall

    def mono(self):
        return self.monotonic

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic += seconds


def hourly_service(clock, tmp_path=None):
    FakeBridge.calls = []
    FakeBridge.failures = 0
    current = settings(stale_seconds=10.0)
    current["outputs"].update({
        camera: f"camera-{camera.lower()}-pump" for camera in "ABCD"
    })
    worker = PressurePlateService(
        lambda: current, threading.Event(), FakeBridge,
        wall_clock=clock.time, monotonic_clock=clock.mono,
    )
    worker._armed = True
    worker._session_token = "session"
    for camera in "ABCD":
        state = worker._states[camera]
        state["last_seen"] = clock.monotonic
        state["inference_available"] = True
        state["inference_reason"] = ""
        state["last_command"] = False
        state["output_id"] = current["outputs"][camera]
        if tmp_path:
            state["events_jsonl"] = tmp_path / camera / "pressure_plate_events.jsonl"
    return worker, current


def confirm_hourly_occupied(worker, clock, camera="A", start_ts=1.0):
    for frame_ts in (start_ts, start_ts + 1):
        worker._process_event(InferenceEvent(
            camera=camera, frame_index=int(frame_ts), frame_ts=frame_ts,
            received_monotonic=clock.monotonic, recorded=True, polygon_valid=True,
            detections=(("sunflower_star", 10.0, 10.0),), events_jsonl=None,
            session_token="session", inference_ok=True,
        ))
    assert worker._states[camera]["occupied"] is True


def start_secondary_pulse(worker, camera="A"):
    state = worker._states[camera]
    state["secondary_active"] = True
    state["secondary_deadline"] = worker.monotonic_clock() + HOURLY_PULSE_SECONDS
    worker._reconcile(camera, "hourly_pulse_start")


def test_hourly_alignment_independent_skip_and_missed_boundary_no_catchup(tmp_path):
    clock = FakeClock()
    worker, _ = hourly_service(clock, tmp_path)
    worker._states["B"]["occupied"] = True
    worker._states["C"]["inference_available"] = False
    worker._states["C"]["inference_reason"] = "starseg_unavailable"
    worker._states["D"]["last_seen"] = clock.monotonic - 11
    worker._set_hourly_armed(True, "test")
    first = worker._next_hourly_at
    assert first > clock.wall
    local = time.localtime(first)
    assert (local.tm_min, local.tm_sec) == (0, 0)

    clock.wall = first
    worker._hourly_tick()
    assert worker._states["A"]["secondary_active"] is True
    assert worker._states["B"]["secondary_active"] is False
    assert worker._states["B"]["last_skip_reason"] == "occupied"
    assert worker._states["C"]["secondary_active"] is False
    assert worker._states["D"]["secondary_active"] is False
    assert worker._states["C"]["last_skip_reason"] == "starseg_unavailable"
    assert worker._states["D"]["last_skip_reason"] == "starseg_stale"
    assert ("camera-a-pump", True) in FakeBridge.calls
    assert not any(output != "camera-a-pump" and on for output, on in FakeBridge.calls)

    previous = worker._next_hourly_at
    worker._states["A"]["secondary_active"] = False
    worker._states["A"]["secondary_deadline"] = None
    clock.advance(3 * 3600)
    worker._hourly_tick()
    assert worker._states["A"]["last_opportunity_at"] == previous
    assert worker._next_hourly_at > clock.wall
    rows = [
        json.loads(line)
        for line in (tmp_path / "B" / "pressure_plate_events.jsonl").read_text().splitlines()
    ]
    assert any(row["event"] == "hourly_opportunity" and row["reason"] == "occupied"
               for row in rows)


def test_hourly_exact_monotonic_pulse_primary_or_and_manual_disarm(tmp_path):
    clock = FakeClock()
    worker, _ = hourly_service(clock, tmp_path)
    worker._set_hourly_armed(True, "test")
    clock.wall = worker._next_hourly_at
    worker._hourly_tick()
    assert worker._states["A"]["secondary_deadline"] == clock.monotonic + HOURLY_PULSE_SECONDS
    assert FakeBridge.calls[-4:] == [
        ("camera-a-pump", True), ("camera-b-pump", True),
        ("camera-c-pump", True), ("camera-d-pump", True),
    ]

    worker._states["A"]["occupied"] = True
    worker._reconcile("A", "primary_enter")
    a_calls = [call for call in FakeBridge.calls if call[0] == "camera-a-pump"]
    assert a_calls == [("camera-a-pump", True)]
    clock.advance(59.999)
    worker._hourly_tick()
    assert worker._states["A"]["secondary_active"] is True
    clock.advance(0.001)
    worker._hourly_tick()
    assert worker._states["A"]["secondary_active"] is False
    assert worker._states["A"]["on"] is True
    assert [call for call in FakeBridge.calls if call[0] == "camera-a-pump"] == a_calls

    worker._states["A"]["secondary_active"] = True
    worker._states["A"]["secondary_deadline"] = clock.monotonic + 60
    worker._set_hourly_armed(False, "operator_disarm")
    assert worker._states["A"]["secondary_active"] is False
    assert worker._states["A"]["on"] is True
    worker._states["A"]["occupied"] = False
    worker._reconcile("A", "primary_exit")
    assert FakeBridge.calls[-1] == ("camera-a-pump", False)
    rows = [
        json.loads(line)
        for line in (tmp_path / "A" / "pressure_plate_events.jsonl").read_text().splitlines()
    ]
    event_names = {row["event"] for row in rows}
    assert {
        "hourly_schedule_arm", "hourly_opportunity", "hourly_pulse_start",
        "hourly_pulse_end", "hourly_schedule_disarm", "hourly_schedule_reconcile",
        "hue_command",
    } <= event_names
    disarm_events = [
        row for row in rows
        if row["event"] in {
            "hourly_pulse_end", "hourly_schedule_disarm", "hourly_schedule_reconcile",
        } and row.get("reason") == "operator_disarm"
    ]
    assert [row["event"] for row in disarm_events] == [
        "hourly_pulse_end", "hourly_schedule_disarm", "hourly_schedule_reconcile",
    ]
    assert disarm_events[-1]["logical_primary_demand"] is True
    assert disarm_events[-1]["secondary_hourly_pulse_active"] is False
    assert disarm_events[-1]["hue_desired"] is True
    assert disarm_events[-1]["arduino_desired"] is False
    assert disarm_events[-1]["hue_on"] is True


def test_hourly_disarm_active_pulse_with_primary_occupied_has_zero_off_calls():
    clock = FakeClock()
    worker, _ = hourly_service(clock)
    worker._hourly_armed = True
    start_secondary_pulse(worker)
    confirm_hourly_occupied(worker, clock)
    calls_before = list(FakeBridge.calls)
    worker._set_hourly_armed(False, "operator_disarm")

    state = worker._states["A"]
    assert state["occupied"] is True
    assert state["secondary_active"] is False
    assert state["hue_desired"] is True
    assert state["on"] is True
    assert state["last_command"] is True
    assert FakeBridge.calls == calls_before
    assert not any(not on for output, on in FakeBridge.calls if output == "camera-a-pump")


def test_hourly_disarm_active_pulse_without_primary_turns_off_immediately():
    clock = FakeClock()
    worker, _ = hourly_service(clock)
    worker._hourly_armed = True
    start_secondary_pulse(worker)
    worker._set_hourly_armed(False, "operator_disarm")

    state = worker._states["A"]
    assert state["occupied"] is False
    assert state["secondary_active"] is False
    assert state["hue_desired"] is False
    assert state["on"] is False
    assert state["last_command"] is False
    assert FakeBridge.calls == [
        ("camera-a-pump", True),
        ("camera-a-pump", False),
    ]


def test_primary_entering_during_secondary_pulse_survives_schedule_disarm():
    clock = FakeClock()
    worker, _ = hourly_service(clock)
    worker._hourly_armed = True
    start_secondary_pulse(worker)
    confirm_hourly_occupied(worker, clock)
    worker._set_hourly_armed(False, "operator_disarm")

    a_calls = [call for call in FakeBridge.calls if call[0] == "camera-a-pump"]
    assert a_calls == [("camera-a-pump", True)]
    assert worker._states["A"]["occupied"] is True
    assert worker._states["A"]["on"] is True


def test_primary_active_before_secondary_demand_and_disarm_stays_continuously_on():
    clock = FakeClock()
    worker, _ = hourly_service(clock)
    confirm_hourly_occupied(worker, clock)
    worker._hourly_armed = True
    start_secondary_pulse(worker)
    worker._set_hourly_armed(False, "operator_disarm")

    assert [call for call in FakeBridge.calls if call[0] == "camera-a-pump"] == [
        ("camera-a-pump", True),
    ]
    assert worker._states["A"]["hue_desired"] is True
    assert worker._states["A"]["last_command"] is True


def test_hourly_disarm_is_idempotent_and_preserves_primary_session_state():
    clock = FakeClock()
    worker, _ = hourly_service(clock)
    confirm_hourly_occupied(worker, clock)
    worker._hourly_armed = True
    start_secondary_pulse(worker)
    worker._set_hourly_armed(False, "operator_disarm")
    calls_after_first = list(FakeBridge.calls)
    worker._set_hourly_armed(False, "operator_disarm")

    assert FakeBridge.calls == calls_after_first
    assert worker._armed is True
    assert worker._session_token == "session"
    assert worker._states["A"]["occupied"] is True
    assert worker._states["A"]["hue_desired"] is True
    assert worker._states["A"]["last_command"] is True


def test_hourly_disarm_reconciles_failed_secondary_on_to_primary_on():
    clock = FakeClock()
    worker, _ = hourly_service(clock)
    worker._hourly_armed = True
    FakeBridge.failures = 1
    start_secondary_pulse(worker)
    confirm_hourly_occupied(worker, clock)
    clock.advance(0.5)
    worker._set_hourly_armed(False, "operator_disarm")

    assert not any(not on for output, on in FakeBridge.calls if output == "camera-a-pump")
    assert worker._states["A"]["hue_desired"] is True
    assert worker._states["A"]["on"] is True
    assert worker._states["A"]["error"] == ""


def test_queued_primary_confirmation_precedes_schedule_disarm_without_off():
    clock = FakeClock()
    worker, _ = hourly_service(clock)
    worker._hourly_armed = True
    start_secondary_pulse(worker)
    worker.start()
    try:
        for frame_ts in (1.0, 2.0):
            worker.submit(InferenceEvent(
                camera="A", frame_index=int(frame_ts), frame_ts=frame_ts,
                received_monotonic=clock.monotonic, recorded=True, polygon_valid=True,
                detections=(("sunflower_star", 10.0, 10.0),), events_jsonl=None,
                session_token="session", inference_ok=True,
            ))
        worker.set_hourly_schedule(False, "session")
        worker.flush()
        assert worker._states["A"]["occupied"] is True
        assert worker._states["A"]["hue_desired"] is True
        assert worker._states["A"]["on"] is True
        assert [call for call in FakeBridge.calls if call[0] == "camera-a-pump"] == [
            ("camera-a-pump", True),
        ]
    finally:
        worker.close()
        worker.join(timeout=2)


def test_hourly_off_reconciliation_bypasses_failed_on_retry_delay():
    clock = FakeClock()
    worker, _ = hourly_service(clock)
    worker._states["A"]["secondary_active"] = True
    FakeBridge.failures = 1
    worker._reconcile("A", "hourly_pulse_start")
    assert worker._states["A"]["error"]
    assert worker._states["A"]["retry_at"] > clock.monotonic
    worker._states["A"]["secondary_active"] = False
    worker._reconcile("A", "hourly_pulse_end")
    assert FakeBridge.calls[-1] == ("camera-a-pump", False)
    assert worker._states["A"]["error"] == ""


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda worker, current: worker._states["A"].update(
            inference_available=False, inference_reason="starseg_unavailable"
        ), "starseg_unavailable"),
        (lambda worker, current: worker._states["A"].update(last_seen=1.0), "starseg_stale"),
        (lambda worker, current: current["outputs"].update(A=""), "output_missing"),
        (lambda worker, current: current["polygons"].update(A=[]), "plate_invalid"),
    ],
)
def test_hourly_fail_closed_skip_reasons(mutation, reason):
    clock = FakeClock()
    worker, current = hourly_service(clock)
    mutation(worker, current)
    worker._set_hourly_armed(True, "test")
    clock.wall = worker._next_hourly_at
    worker._hourly_tick()
    assert worker._states["A"]["secondary_active"] is False
    assert worker._states["A"]["last_skip_reason"] == reason


def test_hourly_trial_stop_forces_off_and_restart_never_rearms():
    clock = FakeClock()
    worker, current = hourly_service(clock)
    worker._states["A"].update(occupied=True, secondary_active=True, secondary_deadline=160.0)
    worker._reconcile("A", "combined")
    worker._hourly_armed = True
    worker._next_hourly_at = clock.wall + 100
    worker._disarm_all("recording_stopped")
    assert worker._hourly_armed is False
    assert worker._armed is False
    assert worker._states["A"]["occupied"] is False
    assert worker._states["A"]["hue_desired"] is False
    assert worker._states["A"]["on"] is False
    assert ("camera-a-pump", False) in FakeBridge.calls

    restarted = PressurePlateService(
        lambda: current, threading.Event(), FakeBridge,
        wall_clock=clock.time, monotonic_clock=clock.mono,
    )
    assert restarted.status()["hourly_schedule"]["armed"] is False
    assert restarted.status()["hourly_schedule"]["next_opportunity_at"] is None


def test_hourly_scheduler_progresses_under_inference_queue_load():
    clock = FakeClock()
    worker, current = hourly_service(clock)
    worker.start()
    try:
        worker.arm(True, "session")
        wait_for(lambda: worker.status()["armed"])
        for camera in "ABCD":
            worker.submit(InferenceEvent(
                camera, 1, 1.0, clock.monotonic, True, True, (),
                None, "session", True,
            ))
        worker.flush()
        worker.set_hourly_schedule(True, "session")
        boundary = worker.status()["hourly_schedule"]["next_opportunity_at"]
        clock.wall = boundary
        for index in range(500):
            worker.submit(InferenceEvent(
                "A", index + 1, float(index + 1), clock.monotonic, True, True, (),
                None, "session", True,
            ))
        wait_for(lambda: worker.status()["cameras"]["A"]["secondary_pulse_active"])
    finally:
        worker.close()
        worker.join(timeout=2)


def test_hourly_api_ui_audit_and_final_metadata(tmp_path):
    deployment_path, config_path = tmp_path / "deployment.json", tmp_path / "config.json"
    atomic_json_write(deployment_path, default_deployment(tmp_path))
    config = default_user_config()
    config["record_root"] = str(tmp_path / "recordings")
    atomic_json_write(config_path, config)
    recorder = Recorder(deployment_path, config_path)
    recorder.pressure.bridge_factory = FakeBridge
    recorder.pressure.start()
    try:
        with pytest.raises(ConfigError, match="active recording"):
            recorder.apply_control({"hourly_pumps_armed": ["1"]})
        recorder.set_recording(True)
        wait_for(lambda: recorder.pressure.status()["armed"])
        result = recorder.apply_control({"hourly_pumps_armed": ["1"]})
        assert result["pressure_plate"]["hourly_schedule"]["armed"] is True
        recorder.set_recording(False)
        wait_for(lambda: not recorder.pressure.status()["hourly_schedule"]["armed"])
    finally:
        recorder.pressure.close()
        recorder.pressure.join(timeout=2)
        recorder.shutdown_event.set()

    html = page().decode()
    assert "Arm hourly A+B+C+D Hue pumps" in html
    assert "60 s Hue-only pulses" in html
    assert "Hourly A+B+C+D Hue pumps disarmed" in html
    assert "hourlyActionPending" in html
    assert "if(!hourlyActionPending)updateHourlySchedule(s)" in html

    segment = make_segment(tmp_path / "metadata")
    files = finalize_segment(segment, 1001.5, 34, "medium")
    metadata = json.loads(files[1].read_text())
    policy = metadata["secondary_hourly_pump_schedule"]
    assert policy["version"] == 4
    assert policy["cameras"] == ["A", "B", "C", "D"]
    assert policy["pulse_seconds"] == 60.0
    assert policy["arm_state"] == "runtime_only"
    assert policy["pulse_actuator"] == "hue_only"
    assert policy["arduino_arbitration"] == "primary_occupancy_only"
    dual = metadata["dual_actuator_pump_policy"]
    assert dual["version"] == 3
    assert dual["arduino"]["mapping"] == {"A": 2, "B": 1, "C": 8, "D": 4}
    assert dual["hue_output_names"]["C"] == "Camera C Pump"
    assert dual["hue_outputs"]["D"] == "dddddddd-0000-4000-8000-00000000000d"
    assert dual["arduino_arbitration"] == "logical_primary_demand_only"
    assert dual["secondary_hourly_pulse_actuator"] == "hue_only"


def test_hourly_pulse_is_hue_only_and_arduino_mask_stays_zero():
    clock = FakeClock()
    arduino = FakeArduino(confirmed_mask=0)
    worker, _ = hourly_service(clock)
    worker.arduino = arduino
    worker._set_hourly_armed(True, "test")
    clock.wall = worker._next_hourly_at
    worker._hourly_tick()
    assert FakeBridge.calls[-4:] == [
        ("camera-a-pump", True), ("camera-b-pump", True),
        ("camera-c-pump", True), ("camera-d-pump", True),
    ]
    assert arduino.status()["desired_mask"] == 0
    assert all(
        worker.status()["cameras"][camera]["split_warning"] is False
        for camera in "ABCD"
    )
    clock.advance(HOURLY_PULSE_SECONDS)
    worker._hourly_tick()
    assert FakeBridge.calls[-4:] == [
        ("camera-a-pump", False), ("camera-b-pump", False),
        ("camera-c-pump", False), ("camera-d-pump", False),
    ]
    assert arduino.status()["desired_mask"] == 0
    assert all(
        worker.status()["cameras"][camera]["split_warning"] is False
        for camera in "ABCD"
    )


def test_primary_arduino_mapping_a_bit2_b_bit1_and_both_mask3():
    clock = FakeClock()
    arduino = FakeArduino(confirmed_mask=0)
    worker, _ = hourly_service(clock)
    worker.arduino = arduino
    worker._states["A"]["occupied"] = True
    worker._reconcile("A", "primary_enter")
    assert arduino.status()["desired_mask"] == 2
    worker._states["B"]["occupied"] = True
    worker._reconcile("B", "primary_enter")
    assert arduino.status()["desired_mask"] == 3
    worker._states["A"]["occupied"] = False
    worker._reconcile("A", "primary_exit")
    assert arduino.status()["desired_mask"] == 1


def test_c_and_d_primary_occupancy_drive_independent_dual_actuators_to_mask12():
    clock = FakeClock()
    arduino = FakeArduino(confirmed_mask=0)
    worker, _ = hourly_service(clock)
    worker.arduino = arduino

    worker._states["C"]["occupied"] = True
    worker._reconcile("C", "primary_enter")
    assert ("camera-c-pump", True) in FakeBridge.calls
    assert arduino.demands["C"] is True
    assert arduino.status()["desired_mask"] == 8

    worker._states["D"]["occupied"] = True
    worker._reconcile("D", "primary_enter")
    assert ("camera-d-pump", True) in FakeBridge.calls
    assert arduino.demands["D"] is True
    assert arduino.status()["desired_mask"] == 12

    arduino.confirmed_mask = 12
    worker._update_split("C", "arduino_event")
    worker._update_split("D", "arduino_event")
    assert worker.status()["cameras"]["C"]["split_warning"] is False
    assert worker.status()["cameras"]["D"]["split_warning"] is False


def test_c_and_d_invalid_plates_fail_closed_for_primary_and_hourly():
    clock = FakeClock()
    arduino = FakeArduino(confirmed_mask=0)
    worker, current = hourly_service(clock)
    worker.arduino = arduino
    current["polygons"]["C"] = []
    current["polygons"]["D"] = []
    worker._states["C"]["occupied"] = True
    worker._states["D"]["occupied"] = True
    worker._reconcile("C", "primary_enter")
    worker._reconcile("D", "primary_enter")
    assert arduino.status()["desired_mask"] == 0
    assert not any(on for output, on in FakeBridge.calls if output in {
        "camera-c-pump", "camera-d-pump",
    })

    worker._set_hourly_armed(True, "test")
    clock.wall = worker._next_hourly_at
    worker._hourly_tick()
    assert worker._states["C"]["last_skip_reason"] == "plate_invalid"
    assert worker._states["D"]["last_skip_reason"] == "plate_invalid"
    assert arduino.status()["desired_mask"] == 0


def test_primary_starts_during_hourly_and_survives_pulse_end_and_disarm():
    clock = FakeClock()
    arduino = FakeArduino(confirmed_mask=0)
    worker, _ = hourly_service(clock)
    worker.arduino = arduino
    worker._hourly_armed = True
    start_secondary_pulse(worker)
    assert arduino.status()["desired_mask"] == 0
    confirm_hourly_occupied(worker, clock)
    assert arduino.status()["desired_mask"] == 2
    arduino.confirmed_mask = 2
    calls_before = list(FakeBridge.calls)
    worker._set_hourly_armed(False, "operator_disarm")
    camera = worker.status()["cameras"]["A"]
    assert camera["logical_primary_demand"] is True
    assert camera["hue_desired"] is True
    assert camera["arduino_desired"] is True
    assert camera["split_warning"] is False
    assert arduino.status()["desired_mask"] == 2
    assert FakeBridge.calls == calls_before


def test_schedule_only_intentional_difference_is_not_split_but_failures_warn():
    clock = FakeClock()
    arduino = FakeArduino(confirmed_mask=0)
    worker, _ = hourly_service(clock)
    worker.arduino = arduino
    start_secondary_pulse(worker)
    camera = worker.status()["cameras"]["A"]
    assert camera["hue_desired"] is True
    assert camera["arduino_desired"] is False
    assert camera["hue_reported"] is True
    assert camera["arduino_confirmed"] is False
    assert camera["hue_target_satisfied"] is True
    assert camera["arduino_target_satisfied"] is True
    assert camera["actuator_difference_expected"] is True
    assert camera["actuator_targets_satisfied"] is True
    assert camera["split_warning"] is False

    arduino.connected = arduino.ready = False
    arduino.error = "serial lost"
    worker._update_split("A", "arduino_error")
    assert worker.status()["cameras"]["A"]["split_warning"] is True


def test_hue_failure_does_not_block_arduino_demand_and_sets_split_warning():
    clock = FakeClock()
    arduino = FakeArduino(confirmed_mask=2)
    worker, _ = hourly_service(clock)
    worker.arduino = arduino
    FakeBridge.failures = 1
    worker._states["A"]["occupied"] = True
    worker._reconcile("A", "occupancy")
    camera = worker.status()["cameras"]["A"]
    assert arduino.demands["A"] is True
    assert camera["hue_error"]
    assert camera["arduino_confirmed"] is True
    assert camera["split_warning"] is True


def test_serial_failure_does_not_block_hue_and_split_warning_clears():
    clock = FakeClock()
    arduino = FakeArduino(connected=False, ready=False, confirmed_mask=None, error="serial lost")
    worker, _ = hourly_service(clock)
    worker.arduino = arduino
    worker._states["A"]["occupied"] = True
    worker._reconcile("A", "occupancy")
    assert ("camera-a-pump", True) in FakeBridge.calls
    assert worker.status()["cameras"]["A"]["split_warning"] is True
    arduino.connected = arduino.ready = True
    arduino.error = ""
    arduino.confirmed_mask = 2
    worker._update_split("A", "arduino_event")
    assert worker.status()["cameras"]["A"]["split_warning"] is False


def test_ui_and_status_expose_dual_actuator_warning():
    html = page().decode()
    assert "Arduino unknown" in html
    assert "ACTUATOR TARGET WARNING" in html
    assert "logical_primary_demand" in html
    assert "confirmed mask" in html


def test_arduino_commands_acknowledgements_and_errors_are_written_to_jsonl(tmp_path):
    arduino = FakeArduino()
    worker = PressurePlateService(
        lambda: settings(), threading.Event(), FakeBridge, arduino=arduino
    )
    worker._armed = True
    worker._session_token = "session"
    for camera in "ABCD":
        worker._states[camera]["events_jsonl"] = tmp_path / camera / "events.jsonl"
    worker._record_arduino_event("arduino_command", {"mask": 15, "reason": "desired_change"})
    worker._record_arduino_event(
        "arduino_event", {"firmware_event": "EVENT STATE 15", "mask": 15}
    )
    worker._record_arduino_event("arduino_error", {"error": "serial lost"})
    for camera, bit in (("A", 2), ("B", 1), ("C", 8), ("D", 4)):
        rows = [
            json.loads(line)
            for line in (tmp_path / camera / "events.jsonl").read_text().splitlines()
        ]
        assert [row["event"] for row in rows[:3]] == [
            "arduino_command", "arduino_event", "arduino_error",
        ]
        assert all(row["mapping_bit"] == bit for row in rows[:3])
