"""Tests for poke_logger.py. Run them with:  python3 -m unittest -v test_poke_logger"""

import csv
import json
import socket
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

import poke_logger as pl


def sample(**changes):
    record = {
        "trial_number": "3",
        "tank_id": "starbot",
        "camera_id": "D",
        "poker_id": "Poker A",
        "entry_time": "2026-10-14T13:00:00.000-04:00",
        "first_poke_time": "2026-10-14T13:00:05.250-04:00",
        "first_movement_time": "2026-10-14T13:00:06.000-04:00",
        "movement_away_time": "2026-10-14T13:00:09.500-04:00",
        "exit_time": "2026-10-14T13:00:12.125-04:00",
        "notes": "",
    }
    record.update(changes)
    return record


class ValidateRecordTests(unittest.TestCase):
    def assertRejected(self, record, fragment):
        with self.assertRaises(pl.ValidationError) as ctx:
            pl.validate_record(record)
        self.assertTrue(any(fragment in e for e in ctx.exception.errors), ctx.exception.errors)

    def test_valid_record_is_cleaned(self):
        clean = pl.validate_record(sample(trial_number="007", notes="  slow\nstart "))
        self.assertEqual(clean["trial_number"], "7")
        self.assertEqual(clean["notes"], "slow start")
        self.assertEqual(clean["first_poke_time"], "2026-10-14T13:00:05.250-04:00")

    def test_camera_d_only_on_starbot(self):
        self.assertRejected(sample(tank_id="starbase"), "Camera D")
        self.assertRejected(sample(tank_id="starsense"), "Camera D")
        for camera in "ABC":
            for tank in pl.TANKS:
                pl.validate_record(sample(tank_id=tank, camera_id=camera))

    def test_unknown_values_are_rejected(self):
        self.assertRejected(sample(tank_id="tank9"), "Tank")
        self.assertRejected(sample(tank_id=None), "Tank")
        self.assertRejected(sample(camera_id="E"), "Camera")
        self.assertRejected(sample(poker_id="gray"), "Poker")

    def test_trial_number_must_be_a_positive_whole_number(self):
        for bad in ("", "0", "-1", "2.5", "abc", None):
            with self.subTest(bad=bad):
                self.assertRejected(sample(trial_number=bad), "Trial number")

    def test_only_entry_is_required(self):
        self.assertRejected(sample(entry_time=""), "Entry time is missing")
        self.assertRejected(sample(entry_time=None), "Entry time is missing")
        # e.g. poking stopped at 5 minutes before the star moved away or exited
        clean = pl.validate_record(sample(first_movement_time="", movement_away_time=None, exit_time="  "))
        self.assertEqual((clean["first_movement_time"], clean["movement_away_time"], clean["exit_time"]), ("", "", ""))
        clean = pl.validate_record({k: v for k, v in sample().items() if k not in pl.EVENT_FIELDS[1:]} | {"entry_time": sample()["entry_time"]})
        self.assertTrue(all(clean[f] == "" for f in pl.EVENT_FIELDS[1:]))

    def test_star_state_is_optional_but_must_be_known(self):
        self.assertEqual(pl.validate_record(sample())["pre_poke_state"], "")
        for value in pl.STAR_STATES:
            self.assertEqual(pl.validate_record(sample(pre_poke_state=value))["pre_poke_state"], value)
        self.assertRejected(sample(pre_poke_state="sleepy"), "Star state before poking")

    def test_optional_event_must_still_be_a_valid_time(self):
        self.assertRejected(sample(exit_time="soon"), "Exit time is not a valid time")
        self.assertRejected(sample(exit_time=12), "Exit time is not a valid time")

    def test_times_need_a_valid_format_and_offset(self):
        self.assertRejected(sample(exit_time="2026-10-14T13:00:12"), "UTC offset")
        self.assertRejected(sample(exit_time="13:00"), "not a valid time")

    def test_out_of_order_times_are_allowed(self):
        pl.validate_record(sample(first_movement_time="2026-10-14T13:00:01.000-04:00"))

    def test_rejects_non_object(self):
        self.assertRejected(["not", "a", "record"], "JSON object")


class RecordStoreTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "logs" / "poke_log.csv"

    def read_csv(self):
        with self.path.open(newline="", encoding="utf-8") as f:
            return list(csv.reader(f))

    def changes(self, store):
        return [json.loads(line) for line in store.changes_path.read_text().splitlines()]

    def test_new_file_gets_header_row(self):
        pl.RecordStore(self.path)
        self.assertEqual(self.read_csv(), [list(pl.COLUMNS)])

    def test_refuses_a_file_with_other_columns(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("a,b,c\n1,2,3\n")
        with self.assertRaises(pl.StoreError):
            pl.RecordStore(self.path)
        self.assertEqual(self.path.read_text(), "a,b,c\n1,2,3\n")

    def test_upgrades_a_file_from_before_the_star_state_column(self):
        old_columns = [c for c in pl.COLUMNS if c != "pre_poke_state"]
        self.path.parent.mkdir(parents=True)
        old_row = {c: "" for c in old_columns} | {"record_id": "4", "trial_number": "2", "poker_id": "Poker A",
                                                   "entry_time": "2026-10-14T13:00:00.000-07:00", "notes": "kept"}
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=old_columns)
            writer.writeheader()
            writer.writerow(old_row)
        original = self.path.read_bytes()

        with mock.patch("sys.stderr"):
            store = pl.RecordStore(self.path)
        self.assertEqual(self.read_csv()[0], list(pl.COLUMNS))
        [row] = store.read_rows()
        self.assertEqual((row["record_id"], row["notes"], row["pre_poke_state"]), ("4", "kept", ""))
        [backup] = self.path.parent.glob("poke_log.csv.before-upgrade-*")
        self.assertEqual(backup.read_bytes(), original)
        self.assertEqual(store.add(sample(pre_poke_state="alert"))["record_id"], "5")

    def test_add_appends_rows_with_increasing_ids(self):
        store = pl.RecordStore(self.path)
        first = store.add(sample())
        second = store.add(sample(trial_number="4"))
        self.assertEqual((first["record_id"], second["record_id"]), ("1", "2"))
        rows = pl.RecordStore(self.path).read_rows()
        self.assertEqual([r["trial_number"] for r in rows], ["3", "4"])
        self.assertTrue(rows[0]["logged_at"])
        self.assertEqual(rows[0]["edited_at"], "")
        self.assertEqual(self.read_csv()[0], list(pl.COLUMNS))

    def test_partial_record_is_saved_with_blank_events(self):
        store = pl.RecordStore(self.path)
        store.add(sample(movement_away_time="", exit_time=""))
        [row] = store.read_rows()
        self.assertEqual((row["movement_away_time"], row["exit_time"]), ("", ""))
        self.assertTrue(row["first_movement_time"])

    def test_invalid_record_is_not_written(self):
        store = pl.RecordStore(self.path)
        with self.assertRaises(pl.ValidationError):
            store.add(sample(entry_time=""))
        self.assertEqual(store.read_rows(), [])

    def test_update_changes_one_row_and_logs_it(self):
        store = pl.RecordStore(self.path)
        original = store.add(sample())
        store.add(sample(trial_number="4"))
        updated = store.update("1", sample(poker_id="Poker B"))

        rows = store.read_rows()
        self.assertEqual(rows[0]["poker_id"], "Poker B")
        self.assertEqual(rows[0]["logged_at"], original["logged_at"])
        self.assertEqual(rows[0]["edited_at"], updated["edited_at"])
        self.assertTrue(updated["edited_at"])
        self.assertEqual(rows[1]["poker_id"], "Poker A")

        [change] = self.changes(store)
        self.assertEqual((change["action"], change["record_id"]), ("edit", "1"))
        self.assertEqual(change["before"]["poker_id"], "Poker A")
        self.assertEqual(change["after"]["poker_id"], "Poker B")

    def test_invalid_update_leaves_file_unchanged(self):
        store = pl.RecordStore(self.path)
        store.add(sample())
        before = self.path.read_text()
        with self.assertRaises(pl.ValidationError):
            store.update("1", sample(camera_id="D", tank_id="starbase"))
        self.assertEqual(self.path.read_text(), before)
        self.assertFalse(store.changes_path.exists())

    def test_delete_removes_row_and_ids_are_never_reused(self):
        store = pl.RecordStore(self.path)
        store.add(sample())
        store.add(sample())
        store.delete("2")
        self.assertEqual([r["record_id"] for r in store.read_rows()], ["1"])
        [change] = self.changes(store)
        self.assertEqual((change["action"], change["record_id"]), ("delete", "2"))
        self.assertNotIn("after", change)

        reopened = pl.RecordStore(self.path)
        self.assertEqual(reopened.add(sample())["record_id"], "3")

    def test_activity_log_appends_and_returns_newest(self):
        store = pl.RecordStore(self.path)
        for i in range(5):
            store.log_action({"action": "record_time", "summary": f"step {i}", "form": {"notes": str(i)}})
        recent = store.recent_actions(limit=3)
        self.assertEqual([a["summary"] for a in recent], ["step 2", "step 3", "step 4"])
        self.assertEqual(recent[-1]["form"], {"notes": "4"})
        self.assertTrue(recent[-1]["at"])
        self.assertEqual(len(store.actions_path.read_text().splitlines()), 5)

    def test_activity_log_rejects_bad_entries(self):
        store = pl.RecordStore(self.path)
        for bad in ({"summary": "no action"}, {"action": "x" * 41}, {"action": "a", "form": []}, ["not", "an", "object"]):
            with self.subTest(bad=bad), self.assertRaises(pl.ValidationError):
                store.log_action(bad)
        self.assertEqual(store.recent_actions(), [])

    def test_half_written_tails_dont_swallow_new_entries(self):
        store = pl.RecordStore(self.path)
        store.add(sample())
        store.log_action({"action": "a", "summary": "before"})
        with self.path.open("a", encoding="utf-8") as f:
            f.write("2,3,starbot,D,Gr")  # a row cut off mid-write
        with store.actions_path.open("a", encoding="utf-8") as f:
            f.write('{"at": "2026-10')
        store.add(sample(notes="after"))
        store.log_action({"action": "a", "summary": "after"})
        rows = self.read_csv()
        self.assertEqual(rows[-1][pl.COLUMNS.index("notes")], "after")
        self.assertEqual(len(rows[-1]), len(pl.COLUMNS))
        self.assertEqual([a["summary"] for a in store.recent_actions()], ["before", "after"])

    def test_failed_append_leaves_no_partial_line(self):
        store = pl.RecordStore(self.path)
        store.add(sample())
        before = self.path.read_bytes()
        real_write = pl.os.write

        def write_half_then_fail(fd, data):
            real_write(fd, bytes(data[: len(data) // 2]))
            raise OSError(28, "No space left on device")

        with mock.patch.object(pl.os, "write", write_half_then_fail), self.assertRaises(OSError):
            store.add(sample(notes="lost"))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(store.add(sample())["record_id"], "2")

    def test_failed_rewrite_keeps_original_file_and_logs_old_values(self):
        store = pl.RecordStore(self.path)
        store.add(sample())
        before = self.path.read_bytes()
        with mock.patch.object(pl.os, "replace", side_effect=OSError("disk error")), self.assertRaises(OSError):
            store.update("1", sample(poker_id="Poker B"))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(list(self.path.parent.glob("*.tmp")))
        [change] = self.changes(store)
        self.assertEqual(change["before"]["poker_id"], "Poker A")

    def test_startup_clears_copies_left_by_an_interrupted_rewrite(self):
        pl.RecordStore(self.path).add(sample())
        leftover = self.path.with_name(f".{self.path.name}.abc123.tmp")
        leftover.write_text("partial copy")
        self.assertEqual(len(pl.RecordStore(self.path).read_rows()), 1)
        self.assertFalse(leftover.exists())

    def test_missing_record(self):
        store = pl.RecordStore(self.path)
        with self.assertRaises(pl.RecordNotFound):
            store.update("9", sample())
        with self.assertRaises(pl.RecordNotFound):
            store.delete("9")


class ServerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = pl.RecordStore(Path(tmp.name) / "poke_log.csv")
        self.server = pl.make_server(self.store, port=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.host = f"localhost:{self.server.server_port}"

    def request(self, method, path, body=None, headers=None):
        conn = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        all_headers = {"Host": self.host}
        if body is not None:
            all_headers["Content-Type"] = "application/json"
            body = json.dumps(body)
        all_headers.update(headers or {})
        conn.request(method, path, body=body, headers=all_headers)
        res = conn.getresponse()
        data = res.read()
        conn.close()
        return res.status, data

    def test_serves_the_page_and_meta(self):
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Poke Trial Logger", page)
        status, meta = self.request("GET", "/api/meta")
        meta = json.loads(meta)
        self.assertEqual(meta["tanks"], list(pl.TANKS))
        self.assertEqual([e["field"] for e in meta["events"]], list(pl.EVENT_FIELDS))
        self.assertEqual(meta["camera_rest_seconds"], 30 * 60)

    def test_create_list_edit_delete(self):
        status, body = self.request("POST", "/api/records", sample())
        self.assertEqual(status, 201)
        record_id = json.loads(body)["record"]["record_id"]

        status, body = self.request("PUT", f"/api/records/{record_id}", sample(notes="fixed poker"))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["record"]["notes"], "fixed poker")

        status, body = self.request("GET", "/api/records")
        [row] = json.loads(body)["records"]
        self.assertEqual(row["notes"], "fixed poker")

        status, _ = self.request("DELETE", f"/api/records/{record_id}")
        self.assertEqual(status, 200)
        status, _ = self.request("DELETE", f"/api/records/{record_id}")
        self.assertEqual(status, 404)
        self.assertEqual(self.store.read_rows(), [])

    def test_invalid_record_returns_errors(self):
        status, body = self.request("POST", "/api/records", sample(poker_id="Nobody", entry_time=""))
        self.assertEqual(status, 422)
        errors = json.loads(body)["errors"]
        self.assertTrue(any("Poker" in e for e in errors))
        self.assertTrue(any("Entry" in e for e in errors))
        self.assertEqual(self.store.read_rows(), [])

    def test_refuses_requests_from_other_sites(self):
        status, _ = self.request("POST", "/api/records", sample(), {"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.request("GET", "/api/records", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.store.read_rows(), [])

    def test_requires_json(self):
        status, _ = self.request("POST", "/api/records", sample(), {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_serves_the_icon(self):
        status, body = self.request("GET", "/icon.svg")
        self.assertEqual(status, 200)
        self.assertIn(b"<svg", body)

    def test_activity_log_endpoints(self):
        entry = {"action": "reset", "summary": "Reset: discarded Entry", "form": {"entry_time": "x"}}
        status, _ = self.request("POST", "/api/actions", entry)
        self.assertEqual(status, 201)
        status, body = self.request("GET", "/api/actions")
        [logged] = json.loads(body)["actions"]
        self.assertEqual((logged["action"], logged["form"]), ("reset", {"entry_time": "x"}))
        status, _ = self.request("POST", "/api/actions", {"summary": "missing action"})
        self.assertEqual(status, 422)

    def test_serves_reference_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            refs = Path(tmp)
            (refs / "starbot_D.jpg").write_bytes(b"\xff\xd8 fake jpeg")
            (refs / "starbase_D.jpg").write_bytes(b"\xff\xd8 starbase has no camera D")
            with mock.patch.object(pl, "REFERENCE_DIR", refs):
                status, body = self.request("GET", "/reference/starbot_D.jpg")
                self.assertEqual((status, body), (200, b"\xff\xd8 fake jpeg"))
                self.assertEqual(self.request("GET", "/reference/starbase_D.jpg")[0], 404)
                self.assertEqual(self.request("GET", "/reference/starbot_A.jpg")[0], 404)
                meta = json.loads(self.request("GET", "/api/meta")[1])
                self.assertEqual(list(meta["references"]), ["starbot_D"])


class GrabReferencesTests(unittest.TestCase):
    def test_saves_one_still_per_configured_camera(self):
        class FakeMonitor(pl.BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"\xff\xd8" + self.path.encode() if self.path != "/feeds/starbot/preview/B.jpg" else b"oops"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        monitor = pl.ThreadingHTTPServer(("127.0.0.1", 0), FakeMonitor)
        threading.Thread(target=monitor.serve_forever, daemon=True).start()
        self.addCleanup(monitor.server_close)
        self.addCleanup(monitor.shutdown)

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "cameras.json"
            config.write_text(json.dumps({
                "monitor": f"http://127.0.0.1:{monitor.server_port}",
                "feeds": {"starbot": {"A": "starbot/A", "B": "starbot/B", "D": "starbot/D"},
                          "starsense": {"A": "deepstar/FCG1"}},
            }))
            refs = Path(tmp) / "refs"
            with mock.patch("sys.stdout"):
                ok = pl.grab_references(config, refs)
            self.assertFalse(ok)  # starbot B returned something that isn't a JPEG
            self.assertEqual(sorted(p.name for p in refs.iterdir()), ["starbot_A.jpg", "starbot_D.jpg", "starsense_A.jpg"])
            self.assertEqual((refs / "starsense_A.jpg").read_bytes(), b"\xff\xd8/feeds/deepstar/preview/FCG1.jpg")
            self.assertEqual(set(pl.reference_images(refs)), {"starbot_A", "starbot_D", "starsense_A"})


class StartupTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.csv = str(Path(tmp.name) / "poke_log.csv")

    def test_second_start_reuses_the_running_logger(self):
        server = pl.make_server(pl.RecordStore(self.csv), port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_port

        self.assertTrue(pl.logger_running(port))
        with mock.patch.object(pl, "open_page") as open_page:
            self.assertEqual(pl.main(["--csv", self.csv, "--port", str(port), "--browser", "firefox"]), 0)
        open_page.assert_called_once_with(f"http://127.0.0.1:{port}/", "firefox")

    def test_port_taken_by_another_program(self):
        other = socket.socket()
        other.bind(("127.0.0.1", 0))
        other.listen()
        self.addCleanup(other.close)
        port = other.getsockname()[1]

        self.assertFalse(pl.logger_running(port))
        with mock.patch.object(pl, "report_error") as report_error:
            self.assertEqual(pl.main(["--csv", self.csv, "--port", str(port), "--no-browser"]), 1)
        self.assertIn("in use by another program", report_error.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
