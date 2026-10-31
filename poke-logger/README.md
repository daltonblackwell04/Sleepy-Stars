# Poke Trial Logger

A small page for logging live poke trials. Choose the trial number, tank, camera and poker, press one key per event as it happens, and save. Each save adds one row to a CSV file.

It needs only Python 3 (3.11 or newer). There's nothing to install.

## Start it

From this folder:

```bash
python3 poke_logger.py
```

The page opens at http://127.0.0.1:8766. Only this computer can reach the page. Ctrl+C stops the logger, and saved records are already on disk.

Options:

- `--csv PATH`: the CSV file to write (default `~/poke_logs/poke_log.csv`)
- `--port N`: use another port if 8766 is taken
- `--browser NAME`: open the page in this browser, e.g. `firefox` (default: your default browser)
- `--no-browser`: don't open a browser

On the lab computers it was launched from a desktop icon that runs `python3 poke_logger.py --browser firefox` and uses `icon.svg` from this folder. Clicking the icon again while the logger is running just reopens the page, and if it can't start (for example, the port is taken by another program), a desktop notification says why.

## Logging a record

1. **Session setup.** Set the trial number and choose the tank, camera and poker. Nothing is chosen for you, and camera D can only be picked when the tank is starbot. These choices stay in place for every record you save until you change them, so the trial number applies to all records from the session.
   - **Trial number** is a combo box. Type a new number, or click the box (or its ▾, or press ↓) to pick one of the trials already in the file, with its record count.
   - **Camera views.** Once a tank is chosen, a reference still from each of its cameras appears below. Click one to choose that camera (see [Camera reference images](#camera-reference-images)).
   - **Resting cameras.** After a trial, that camera on that tank rests for 30 minutes, counted from the trial's last recorded event (usually Exit). Until then it can't be chosen. Its button is amber and dashed, with the minutes left, and the line under the cameras says when each one is free again (for example `Resting 30 min after a trial: B until 17:37`). Hover over it to see which record started the rest. Saving a record clears the camera choice, since that camera is now resting, so pick the next free one. A record that already has event times can still be saved on a camera that's resting, and editing saved records isn't limited. Deleting a record ends its rest. To change the 30 minutes, edit `CAMERA_REST_SECONDS` at the top of `poke_logger.py`.
2. **Star before poking (optional).** Press **Q** (quiescent), **A** (alert) or **M** (moving), or click the button, for your read of the star before poking starts. Press the same one again to clear it. It's saved with the record, isn't required, and resets for the next record.
3. **Record the events live.** Press each key as the event happens:

   | Key | Event |
   |---|---|
   | 1 | Entry |
   | 2 | First poke |
   | 3 | First movement |
   | 4 | Movement away |
   | 5 | Exit |

   You can also click the buttons. A second press of an event that's already recorded is ignored, so a double tap can't overwrite a time. To redo one, clear it with ×. You can also type a correction in the time box, as `HH:MM:SS.mmm`.
4. **Watch the poke timer.** It starts counting at First poke, toward the 5-minute poking limit:
   - **Last minute:** it turns amber.
   - **At 5:00:** it turns red and says it's time to stop poking.
   - **Movement away or Exit recorded:** it stops.

   It's for information only and doesn't stop or save anything by itself.
5. **Save** with Enter. Only Entry is required. Stars don't always wake up and move away before poking stops, so the other events can be left blank. When some are blank, you're asked to confirm, so an early Enter can't save a half-finished record by accident. If the times are out of order (for example exit before entry), you're asked to confirm. Esc clears the current record.

## Fixing a mistake

Every saved record appears under **Saved records**, newest first (tick **Show all** to see older ones). Click **Edit** on a record to load it into the form, change any field, and click **Save changes**. You can also **Delete record** from there. Esc or **Cancel edit** discards changes, and live recording picks up where you left off.

Edits and deletions never lose data silently. Each one is written to `poke_log_changes.jsonl` next to the CSV, along with the record before the change (and after it, for edits). Record numbers are never reused, even after a deletion.

## Activity log: recovering inputs

Everything done on the page is written to `poke_log_actions.jsonl` next to the CSV. That includes every event time recorded, cleared or typed, every choice of trial, tank, camera and poker, notes, saves (including blocked or failed ones), resets, edits, deletions and restores. Each entry also stores what was in the form at that moment. For resets, cancelled edits and deletions, that's what was in the form just before, so the lost inputs are kept.

The **Activity log** panel shows the newest entries (tick **Show all** for the last 500). Click **Restore** on an entry to put those inputs back in the form, then save. If you restore an entry from the middle of an edit, the record reopens in edit mode with those values. Restoring is logged too, so it can be undone the same way.

If the logger can't be reached for a moment, entries wait on the page and are sent automatically once it's back.

## Camera reference images

The cameras don't move during trials, so the page shows one stored still per camera rather than a live feed. The stills live in `reference_images/` as `<tank>_<camera>.jpg` (not included in this repository; the page works without them). To grab fresh ones (for example after a camera is moved), make sure the Live Experiments monitor is running, then run:

```bash
python3 poke_logger.py --grab-references
```

`cameras.json` says where each camera's still comes from on the monitor. Feeds named by camera letter work as-is; a camera whose feed has another name must be assigned to its letter in that file before its still can be grabbed. Reload the page to see new stills.

## Report

The **Report** at the bottom of the page fills in as records are saved. It covers the current trial, or switch to **All trials**. It shows:

- **Totals:** the record count; medians for time in view (entry → exit), time to first poke, and response time (first poke → first movement); and how many records reached movement away. Medians only count records that have both events.
- **Each record** as a bar from entry to exit, with markers for first poke ●, first movement ■ and movement away ◆ (the newest 30 records). A record with no exit is drawn as a dashed outline up to its last recorded event.
- **Median time between each pair of events**
- **Record counts** by tank and camera, by poker, and by star state before poking

Hover over any bar for details.

## The CSV file

One row per record:

| Column | Contents |
|---|---|
| `record_id` | Unique number for the row (1, 2, 3, ...) |
| `trial_number` | Trial number set for the session |
| `tank_id` | starbase, starbot or starsense |
| `camera_id` | A, B, C or D (D only with starbot) |
| `poker_id` | Poker A or Poker B (edit `POKERS` at the top of `poke_logger.py`) |
| `pre_poke_state` | quiescent, alert or moving: the star before poking (blank if not chosen) |
| `entry_time` | Entry (always filled) |
| `first_poke_time` | First poke |
| `first_movement_time` | First movement |
| `movement_away_time` | Movement away |
| `exit_time` | Exit |
| `notes` | Optional free text |
| `logged_at` | When the row was first saved |
| `edited_at` | When it was last edited (blank if never) |

The event columns after `entry_time` are blank when that event wasn't recorded.

Times are local time in ISO 8601 with milliseconds and the UTC offset, for example `2026-10-14T13:05:22.481-04:00`. Excel, LibreOffice, pandas and R can all read them.

Each save is written to disk right away, so stopping the program or a crash loses at most the record you're in the middle of. Opening the CSV in a spreadsheet to look at it is fine. Make corrections through the page instead, so they're recorded in the change log. If the CSV comes from an earlier version that's only missing newer columns (such as `pre_poke_state`), the logger upgrades it on start, adding the column blank for existing rows. It first keeps a copy of the original as `poke_log.csv.before-upgrade-<date>`. If the columns differ in any other way, the logger refuses to start rather than mix formats.

## Tests

```bash
python3 -m unittest -v test_poke_logger
```
