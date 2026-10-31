from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest

from loop_injector.automations import AutomationStore
from loop_injector.pump_backend import PumpController, encode_mask
from loop_injector.pump_log import PumpActivityLog


class FakeSerial:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._incoming = bytearray(b"MULTICAM_PUMP_READY\r\nEVENT BOOT\r\n")
        self.writes: list[bytes] = []
        self.closed = False
        self.current_mask = 0
        self.rearmed = False

    @property
    def in_waiting(self) -> int:
        with self._lock:
            return len(self._incoming)

    def read(self, size: int) -> bytes:
        with self._lock:
            result = bytes(self._incoming[:size])
            del self._incoming[:size]
            return result

    def write(self, data: bytes) -> int:
        with self._lock:
            self.writes.append(bytes(data))
            mask = int(data.decode("ascii").removeprefix("PUMPS:").strip())
            if mask == 0 and not self.rearmed:
                self.rearmed = True
                self._incoming.extend(b"EVENT REARMED\r\n")
            elif self.rearmed and mask != self.current_mask:
                self.current_mask = mask
                self._incoming.extend(f"EVENT STATE {mask}\r\n".encode())
            return len(data)

    def flush(self) -> None:
        return

    def close(self) -> None:
        self.closed = True


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition was not reached before timeout")


class PumpControllerTests(unittest.TestCase):
    def make_controller(self, **kwargs):
        fake = FakeSerial()
        controller = PumpController(lambda: fake, **kwargs)
        controller.start()
        self.addCleanup(controller.close)
        wait_for(lambda: controller.status()["ready"])
        return controller, fake

    def test_mask_encoding_is_exact_and_bounded(self):
        self.assertEqual(encode_mask(0), b"PUMPS:0\n")
        self.assertEqual(encode_mask(15), b"PUMPS:15\n")
        for invalid in (-1, 16, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    encode_mask(invalid)

    def test_manual_masks_are_acknowledged_and_shutdown_is_neutral(self):
        controller, fake = self.make_controller()
        controller.renew_lease()
        controller.set_manual_mask(1)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 1)
        controller.set_manual_mask(10)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 10)
        controller.emergency_stop()
        wait_for(lambda: controller.status()["acknowledged_mask"] == 0)
        controller.close()
        self.assertEqual(fake.writes[-1], b"PUMPS:0\n")
        self.assertTrue(fake.closed)

    def test_schedule_moves_through_mask_bits_and_finishes_off(self):
        controller, fake = self.make_controller(lease_seconds=3)
        controller.renew_lease()
        controller.start_schedule((1, 2), 0.5, 1.1)
        wait_for(lambda: b"PUMPS:1\n" in fake.writes)
        wait_for(lambda: b"PUMPS:2\n" in fake.writes)
        wait_for(lambda: controller.status()["mode"] == "idle", timeout=2)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 0)

    def test_expired_browser_lease_forces_off_and_cancels_schedule(self):
        controller, _fake = self.make_controller(lease_seconds=0.2)
        controller.renew_lease()
        controller.start_schedule((1, 2), 0.5, 10)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 1)
        wait_for(lambda: not controller.status()["lease_active"])
        wait_for(lambda: controller.status()["acknowledged_mask"] == 0)
        status = controller.status()
        self.assertEqual(status["mode"], "idle")
        self.assertIsNone(status["schedule"])

    def test_open_failure_is_reported_fail_closed(self):
        controller = PumpController(lambda: (_ for _ in ()).throw(OSError("busy")))
        controller.start()
        self.addCleanup(controller.close)
        status = wait_for(lambda: controller.status()["last_error"])
        self.assertIn("busy", status)
        self.assertEqual(controller.status()["requested_mask"], 0)
        self.assertFalse(controller.status()["ready"])

    def test_persistent_automation_runs_without_lease_and_emergency_pauses_it(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        now = datetime(2026, 8, 10, 0, 27, tzinfo=timezone.utc)
        store = AutomationStore(
            Path(temporary.name) / "automations.json", wall_clock=lambda: now
        )
        store.save(
            {
                "name": "Unattended",
                "enabled": True,
                "mask": 9,
                "weekdays": [0],
                "start_time": "00:00",
                "end_time": "01:00",
                "on_minutes": 5,
                "block_minutes": 25,
            }
        )
        fake = FakeSerial()
        controller = PumpController(lambda: fake, automation_store=store)
        controller.start()
        self.addCleanup(controller.close)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 9)
        self.assertFalse(controller.status()["lease_active"])
        self.assertEqual(controller.status()["mode"], "automation")

        controller.renew_lease()
        controller.set_manual_mask(1)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 1)
        controller.emergency_stop()
        wait_for(lambda: controller.status()["acknowledged_mask"] == 0)
        self.assertTrue(store.paused)

        controller.renew_lease()
        controller.set_manual_mask(1)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 1)
        controller.resume_automations()
        wait_for(lambda: controller.status()["acknowledged_mask"] == 9)
        controller.close()
        self.assertFalse(store.paused, "normal shutdown must not pause saved automations")
        self.assertEqual(fake.writes[-1], b"PUMPS:0\n")

    def test_recorder_lease_expires_and_priority_is_manual_then_recorder_then_automation(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        now = datetime(2026, 8, 10, 0, 27, tzinfo=timezone.utc)
        store = AutomationStore(
            Path(temporary.name) / "automations.json", wall_clock=lambda: now
        )
        store.save({
            "name": "Background", "enabled": True, "mask": 8,
            "weekdays": [0], "start_time": "00:00", "end_time": "01:00",
            "on_minutes": 5, "block_minutes": 25,
        })
        controller, _fake = self.make_controller(
            lease_seconds=0.2, automation_store=store
        )
        token, status = controller.start_recorder_session()
        self.assertNotIn("token", status["recorder"])
        controller.set_recorder_demand(token, 2)
        wait_for(lambda: controller.status()["requested_mask"] == 2)
        self.assertEqual(controller.status()["arbitration_source"], "recorder")

        controller.renew_lease()
        controller.set_manual_mask(1)
        wait_for(lambda: controller.status()["requested_mask"] == 1)
        self.assertEqual(controller.status()["arbitration_source"], "manual")
        controller.set_manual_mask(0)
        wait_for(lambda: controller.status()["requested_mask"] == 2)
        wait_for(lambda: not controller.status()["recorder"]["lease_active"])
        wait_for(lambda: controller.status()["requested_mask"] == 8)
        self.assertEqual(controller.status()["arbitration_source"], "automation")

    def test_emergency_latch_only_explicit_manual_command_clears(self):
        controller, _fake = self.make_controller(lease_seconds=1)
        token, _status = controller.start_recorder_session()
        controller.set_recorder_demand(token, 4)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 4)
        controller.emergency_stop()
        wait_for(lambda: controller.status()["acknowledged_mask"] == 0)
        self.assertTrue(controller.status()["emergency_latched"])

        controller.set_recorder_demand(token, 8)
        controller.resume_automations()
        controller.renew_lease()
        controller.start_schedule((1,), 0.5, 10)
        time.sleep(0.1)
        self.assertEqual(controller.status()["requested_mask"], 0)
        self.assertTrue(controller.status()["emergency_latched"])
        controller.stop_schedule()
        controller.set_manual_mask(1)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 1)
        self.assertFalse(controller.status()["emergency_latched"])

    def test_recorder_tokens_are_validated_and_session_stop_is_fail_closed(self):
        controller, _fake = self.make_controller()
        token, _status = controller.start_recorder_session()
        with self.assertRaises(PermissionError):
            controller.set_recorder_demand("wrong", 1)
        with self.assertRaises(ValueError):
            controller.set_recorder_demand(token, 16)
        controller.set_recorder_demand(token, 3)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 3)
        controller.stop_recorder_session(token)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 0)
        self.assertFalse(controller.status()["recorder"]["session_active"])

    def test_firmware_acknowledgements_are_written_to_interval_database(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        activity_log = PumpActivityLog(
            Path(temporary.name) / "pump-log.sqlite3"
        )
        fake = FakeSerial()
        controller = PumpController(lambda: fake, pump_log=activity_log)
        controller.start()
        self.addCleanup(controller.close)
        wait_for(lambda: controller.status()["ready"])
        controller.renew_lease()
        controller.set_manual_mask(2)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 2)
        time.sleep(0.05)
        controller.set_manual_mask(0)
        wait_for(lambda: controller.status()["acknowledged_mask"] == 0)
        result = controller.query_pump_log(pump=2)
        self.assertEqual(result["total"], 1)
        self.assertGreater(result["rows"][0]["duration_seconds"], 0)
        self.assertEqual(result["rows"][0]["pump_number"], 2)
        self.assertTrue(controller.status()["pump_log"]["healthy"])


if __name__ == "__main__":
    unittest.main()
