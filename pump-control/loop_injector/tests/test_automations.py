from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from loop_injector.automations import AutomationRule, AutomationStore, rule_runtime


def payload(**overrides):
    result = {
        "name": "Day dosing",
        "enabled": True,
        "mask": 5,
        "weekdays": [0, 1, 2, 3, 4, 5, 6],
        "start_time": "00:00",
        "end_time": "18:00",
        "on_minutes": 5,
        "block_minutes": 25,
    }
    result.update(overrides)
    return result


class AutomationRuleTests(unittest.TestCase):
    def test_twenty_five_minute_gap_then_five_minute_injection(self):
        rule = AutomationRule.from_dict(payload(), generate_id=True)
        monday = datetime(2026, 8, 10, tzinfo=timezone.utc)
        self.assertFalse(rule_runtime(rule, monday.replace(hour=0, minute=24))["active"])
        self.assertTrue(rule_runtime(rule, monday.replace(hour=0, minute=25))["active"])
        self.assertTrue(rule_runtime(rule, monday.replace(hour=0, minute=29))["active"])
        self.assertFalse(rule_runtime(rule, monday.replace(hour=0, minute=30))["active"])
        self.assertTrue(rule_runtime(rule, monday.replace(hour=0, minute=55))["active"])
        self.assertTrue(rule_runtime(rule, monday.replace(hour=17, minute=55))["active"])
        self.assertFalse(rule_runtime(rule, monday.replace(hour=18))["active"])

    def test_weekdays_and_same_day_windows_are_validated(self):
        monday_rule = AutomationRule.from_dict(
            payload(weekdays=[0]), generate_id=True
        )
        monday = datetime(2026, 8, 10, 0, 26, tzinfo=timezone.utc)
        tuesday = monday.replace(day=11)
        self.assertTrue(rule_runtime(monday_rule, monday)["active"])
        self.assertFalse(rule_runtime(monday_rule, tuesday)["active"])
        for invalid in (
            payload(weekdays=[]),
            payload(start_time="18:00", end_time="06:00"),
            payload(start_time="17:00", end_time="18:00", on_minutes=40, block_minutes=25),
            payload(mask=0),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    AutomationRule.from_dict(invalid, generate_id=True)


class AutomationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "automations.json"
        self.now = datetime(2026, 8, 10, 0, 27, tzinfo=timezone.utc)

    def tearDown(self):
        self.temporary.cleanup()

    def make_store(self):
        return AutomationStore(self.path, wall_clock=lambda: self.now)

    def test_overlapping_rules_or_masks_and_persist_across_reload(self):
        store = self.make_store()
        first = store.save(payload(mask=1))
        second = store.save(payload(name="Second", mask=4))
        state = store.evaluate()
        self.assertEqual(state["mask"], 5)
        self.assertEqual(set(state["active_ids"]), {first["id"], second["id"]})

        reloaded = self.make_store()
        self.assertEqual(reloaded.evaluate()["mask"], 5)
        self.assertEqual(len(reloaded.evaluate()["rules"]), 2)

    def test_pause_is_persistent_and_fails_closed(self):
        store = self.make_store()
        store.save(payload(mask=3))
        store.set_paused(True)
        self.assertEqual(store.evaluate()["mask"], 0)
        self.assertFalse(store.evaluate()["rules"][0]["active"])
        self.assertTrue(self.make_store().evaluate()["paused"])
        store.set_paused(False)
        self.assertEqual(store.evaluate()["mask"], 3)

    def test_update_enable_and_delete(self):
        store = self.make_store()
        saved = store.save(payload())
        updated = {**saved, "enabled": False, "name": "Disabled"}
        store.save(updated)
        self.assertEqual(store.evaluate()["mask"], 0)
        store.set_enabled(saved["id"], True)
        self.assertEqual(store.evaluate()["mask"], 5)
        store.delete(saved["id"])
        self.assertEqual(store.evaluate()["rules"], [])

    def test_malformed_persisted_state_is_rejected_and_paused(self):
        self.path.write_text(json.dumps({"version": 1, "paused": False, "rules": [{}]}))
        store = self.make_store()
        state = store.evaluate()
        self.assertEqual(state["mask"], 0)
        self.assertTrue(state["paused"])
        self.assertIn("rejected", state["last_error"])

    def test_legacy_cycle_period_is_migrated_to_off_gap(self):
        legacy = payload(block_minutes=30)
        legacy["id"] = "a" * 32
        self.path.write_text(
            json.dumps({"version": 1, "paused": False, "rules": [legacy]})
        )
        store = self.make_store()
        state = store.evaluate()
        self.assertEqual(state["rules"][0]["block_minutes"], 25)
        self.assertEqual(state["mask"], 5)
        persisted = json.loads(self.path.read_text())
        self.assertEqual(persisted["version"], 2)
        self.assertEqual(persisted["rules"][0]["block_minutes"], 25)


if __name__ == "__main__":
    unittest.main()
