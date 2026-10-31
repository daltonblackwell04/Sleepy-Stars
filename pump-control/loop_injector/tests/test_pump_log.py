from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from loop_injector.pump_log import PumpActivityLog


class PumpActivityLogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "pump-log.sqlite3"
        self.now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
        self.log = PumpActivityLog(self.path, clock=lambda: self.now)

    def tearDown(self):
        try:
            self.log.close()
        except sqlite3.ProgrammingError:
            pass
        self.temporary.cleanup()

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)

    def test_confirmed_masks_create_per_pump_intervals_with_duration(self):
        self.log.record_boot_zero(occurred_at=self.now)
        self.advance(5)
        self.log.record_confirmed_mask(
            5,
            firmware_event="EVENT STATE 5",
            control_mode="automation",
            automation_ids=["rule-a"],
            occurred_at=self.now,
        )
        self.advance(10)
        self.log.record_confirmed_mask(
            4,
            firmware_event="EVENT STATE 4",
            control_mode="automation",
            automation_ids=["rule-a"],
            occurred_at=self.now,
        )
        self.advance(10)
        self.log.record_confirmed_mask(
            0,
            firmware_event="EVENT STATE 0",
            control_mode="idle",
            automation_ids=[],
            occurred_at=self.now,
        )
        result = self.log.query_intervals()
        self.assertEqual(result["total"], 2)
        by_pump = {row["pump_number"]: row for row in result["rows"]}
        self.assertEqual(by_pump[1]["duration_seconds"], 10)
        self.assertEqual(by_pump[3]["duration_seconds"], 20)
        self.assertEqual(by_pump[1]["control_mode"], "automation")
        self.assertEqual(by_pump[1]["automation_ids"], ["rule-a"])
        self.assertEqual(self.log.status()["open_intervals"], 0)

    def test_repeated_acknowledgement_does_not_create_duplicate_interval(self):
        self.log.record_boot_zero()
        self.log.record_confirmed_mask(
            1,
            firmware_event="EVENT STATE 1",
            control_mode="manual",
            automation_ids=[],
        )
        self.log.record_confirmed_mask(
            1,
            firmware_event="EVENT STATE 1",
            control_mode="manual",
            automation_ids=[],
        )
        self.assertEqual(self.log.query_intervals()["total"], 1)
        self.assertEqual(self.log.status()["open_intervals"], 1)

    def test_boot_reconciles_interval_left_open_by_process_exit(self):
        self.log.record_boot_zero()
        self.log.record_confirmed_mask(
            1,
            firmware_event="EVENT STATE 1",
            control_mode="automation",
            automation_ids=["rule-a"],
        )
        self.log.close()
        self.advance(12)
        self.log = PumpActivityLog(self.path, clock=lambda: self.now)
        self.log.record_boot_zero(occurred_at=self.now)
        row = self.log.query_intervals()["rows"][0]
        self.assertEqual(row["duration_seconds"], 12)
        self.assertEqual(row["end_reason"], "firmware_boot")
        self.assertEqual(self.log.status()["open_intervals"], 0)

    def test_filters_and_csv_export_are_referenceable(self):
        self.log.record_boot_zero()
        for pump, mask in ((1, 1), (2, 2)):
            self.advance(1)
            self.log.record_confirmed_mask(
                mask,
                firmware_event=f"EVENT STATE {mask}",
                control_mode="manual",
                automation_ids=[],
            )
            self.advance(2)
            self.log.record_confirmed_mask(
                0,
                firmware_event="EVENT STATE 0",
                control_mode="idle",
                automation_ids=[],
            )
        only_two = self.log.query_intervals(pump=2)
        self.assertEqual(only_two["total"], 1)
        self.assertEqual(only_two["rows"][0]["pump_number"], 2)
        csv_text = self.log.export_csv().decode()
        self.assertIn("pump_number", csv_text)
        self.assertIn("manual", csv_text)


if __name__ == "__main__":
    unittest.main()
