# Sample data

`poke_log_sample.csv` is a 12-record excerpt of the poke-trial log written by the [Poke Trial Logger](../poke-logger/). It covers trials 1–3, all three tanks, and all three pre-poke states. It is meant for trying the logger and for seeing the data format. It is not the full dataset.

Anonymization: operator names are replaced with `Poker A` / `Poker B`, and the free-text `notes` column is blanked. Everything else is as recorded.

To open it in the logger:

```bash
cp poke_log_sample.csv /tmp/poke_log.csv
python3 ../poke-logger/poke_logger.py --csv /tmp/poke_log.csv
```

## Columns

One row per poke trial:

| Column | Contents |
|---|---|
| `record_id` | Unique row number. Never reused, even after a deletion, so gaps are normal. |
| `trial_number` | Trial number set for the session |
| `tank_id` | Tank (`starbase`, `starbot` or `starsense`, named after the recording computer) |
| `camera_id` | Camera over the animal: `A`, `B`, `C`, or `D` (D exists only on `starbot`) |
| `poker_id` | Person delivering the stimulus (anonymized) |
| `pre_poke_state` | Observer's call before poking: `quiescent`, `alert` or `moving`; blank if not chosen |
| `entry_time` | Entry: start of the trial (always filled) |
| `first_poke_time` | First poke |
| `first_movement_time` | First movement of the star |
| `movement_away_time` | Movement away |
| `exit_time` | Exit: end of the trial |
| `notes` | Free text (blanked here) |
| `logged_at` | When the row was first saved |
| `edited_at` | When it was last edited (blank if never) |

Event columns after `entry_time` are blank when the event didn't happen. Not every star moves away before the 5-minute poking limit.

Times are local time in ISO 8601 with milliseconds and UTC offset, e.g. `2026-10-15T13:38:28.829-07:00`.

Derived measures shown in the logger's report:

- **Response time:** `first_movement_time − first_poke_time`
- **Time to first poke:** `first_poke_time − entry_time`
- **Time in view:** `exit_time − entry_time`
