"""Durable append-only SQLite log of Arduino-confirmed pump activity."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Callable


SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("log timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")


class PumpActivityLog:
    """Record confirmed mask transitions as per-pump intervals without pruning."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._path = Path(path)
        self._clock = clock
        self._lock = threading.RLock()
        self._last_error: str | None = None
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self._path.parent, 0o700)
        except OSError:
            pass
        self._connection = sqlite3.connect(
            self._path,
            timeout=5,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._create_schema()
        row = self._connection.execute(
            "SELECT observed_mask FROM mask_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self._current_mask = int(row["observed_mask"]) if row else 0

    def _configure(self) -> None:
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mask_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at_utc TEXT NOT NULL,
                observed_mask INTEGER NOT NULL CHECK(observed_mask BETWEEN 0 AND 15),
                firmware_event TEXT NOT NULL,
                control_mode TEXT NOT NULL,
                automation_ids_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pump_intervals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pump_number INTEGER NOT NULL CHECK(pump_number BETWEEN 1 AND 4),
                started_at_utc TEXT NOT NULL,
                ended_at_utc TEXT,
                duration_seconds REAL,
                start_event_id INTEGER NOT NULL REFERENCES mask_events(id),
                end_event_id INTEGER REFERENCES mask_events(id),
                start_mode TEXT NOT NULL,
                automation_ids_json TEXT NOT NULL,
                end_reason TEXT,
                CHECK (
                    (ended_at_utc IS NULL AND duration_seconds IS NULL AND end_event_id IS NULL)
                    OR
                    (ended_at_utc IS NOT NULL AND duration_seconds IS NOT NULL AND end_event_id IS NOT NULL)
                )
            );
            CREATE INDEX IF NOT EXISTS pump_intervals_pump_started
                ON pump_intervals(pump_number, started_at_utc DESC);
            CREATE INDEX IF NOT EXISTS pump_intervals_started
                ON pump_intervals(started_at_utc DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS pump_intervals_one_open_per_pump
                ON pump_intervals(pump_number) WHERE ended_at_utc IS NULL;
            """
        )
        existing = self._connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        if existing is None:
            self._connection.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        elif int(existing["value"]) != SCHEMA_VERSION:
            raise RuntimeError("unsupported pump log schema version")

    def record_boot_zero(
        self,
        *,
        firmware_event: str = "MULTICAM_PUMP_READY",
        occurred_at: datetime | None = None,
    ) -> None:
        """Record firmware boot as authoritative all-off and reconcile open rows."""
        self.record_confirmed_mask(
            0,
            firmware_event=firmware_event,
            control_mode="boot",
            automation_ids=(),
            occurred_at=occurred_at,
            end_reason="firmware_boot",
            force_event=True,
        )

    def record_confirmed_mask(
        self,
        mask: int,
        *,
        firmware_event: str,
        control_mode: str,
        automation_ids: tuple[str, ...] | list[str],
        occurred_at: datetime | None = None,
        end_reason: str = "confirmed_state_change",
        force_event: bool = False,
    ) -> None:
        if isinstance(mask, bool) or not isinstance(mask, int) or not 0 <= mask <= 15:
            raise ValueError("confirmed mask must be an integer from 0 through 15")
        if not isinstance(firmware_event, str) or not firmware_event:
            raise ValueError("firmware_event must be non-empty")
        if not isinstance(control_mode, str) or not control_mode:
            raise ValueError("control_mode must be non-empty")
        ids = tuple(automation_ids)
        if any(not isinstance(rule_id, str) for rule_id in ids):
            raise ValueError("automation ids must be strings")
        timestamp = occurred_at or self._clock()
        timestamp_text = _iso_utc(timestamp)
        ids_json = json.dumps(ids, separators=(",", ":"))

        with self._lock:
            previous = self._current_mask
            if mask == previous and not force_event:
                return
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                cursor = self._connection.execute(
                    """
                    INSERT INTO mask_events(
                        occurred_at_utc, observed_mask, firmware_event,
                        control_mode, automation_ids_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp_text,
                        mask,
                        firmware_event[:160],
                        control_mode[:40],
                        ids_json,
                    ),
                )
                event_id = int(cursor.lastrowid)
                for pump in range(1, 5):
                    bit = 1 << (pump - 1)
                    was_on = bool(previous & bit)
                    is_on = bool(mask & bit)
                    if was_on and not is_on:
                        self._close_interval(
                            pump, timestamp, timestamp_text, event_id, end_reason
                        )
                    elif is_on and not was_on:
                        self._connection.execute(
                            """
                            INSERT INTO pump_intervals(
                                pump_number, started_at_utc, start_event_id,
                                start_mode, automation_ids_json
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (pump, timestamp_text, event_id, control_mode[:40], ids_json),
                        )
                if mask == 0 and force_event:
                    for pump in range(1, 5):
                        self._close_interval(
                            pump, timestamp, timestamp_text, event_id, end_reason
                        )
                self._connection.execute("COMMIT")
                self._current_mask = mask
                self._last_error = None
            except Exception as exc:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                self._last_error = str(exc)[:240]
                raise

    def _close_interval(
        self,
        pump: int,
        ended_at: datetime,
        ended_text: str,
        event_id: int,
        reason: str,
    ) -> None:
        row = self._connection.execute(
            """
            SELECT id, started_at_utc
            FROM pump_intervals
            WHERE pump_number = ? AND ended_at_utc IS NULL
            ORDER BY id DESC LIMIT 1
            """,
            (pump,),
        ).fetchone()
        if row is None:
            return
        started = datetime.fromisoformat(row["started_at_utc"])
        duration = max(0.0, (ended_at.astimezone(timezone.utc) - started).total_seconds())
        self._connection.execute(
            """
            UPDATE pump_intervals
            SET ended_at_utc = ?, duration_seconds = ?,
                end_event_id = ?, end_reason = ?
            WHERE id = ?
            """,
            (ended_text, duration, event_id, reason[:80], row["id"]),
        )

    def query_intervals(
        self,
        *,
        pump: int | None = None,
        started_from: str | None = None,
        started_to: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict:
        if pump is not None and pump not in (1, 2, 3, 4):
            raise ValueError("pump filter must be 1 through 4")
        if isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be nonnegative")
        clauses = []
        params: list[object] = []
        if pump is not None:
            clauses.append("pump_number = ?")
            params.append(pump)
        if started_from is not None:
            clauses.append("started_at_utc >= ?")
            params.append(_iso_utc(datetime.fromisoformat(started_from)))
        if started_to is not None:
            clauses.append("started_at_utc < ?")
            params.append(_iso_utc(datetime.fromisoformat(started_to)))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            total = int(
                self._connection.execute(
                    f"SELECT COUNT(*) AS count FROM pump_intervals {where}", params
                ).fetchone()["count"]
            )
            rows = self._connection.execute(
                f"""
                SELECT id, pump_number, started_at_utc, ended_at_utc,
                       duration_seconds, start_mode, automation_ids_json,
                       end_reason
                FROM pump_intervals
                {where}
                ORDER BY started_at_utc DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "rows": [
                {
                    "id": row["id"],
                    "pump_number": row["pump_number"],
                    "started_at_utc": row["started_at_utc"],
                    "ended_at_utc": row["ended_at_utc"],
                    "duration_seconds": row["duration_seconds"],
                    "control_mode": row["start_mode"],
                    "automation_ids": json.loads(row["automation_ids_json"]),
                    "end_reason": row["end_reason"],
                }
                for row in rows
            ],
        }

    def export_csv(self, **filters) -> bytes:
        filters = dict(filters)
        filters["limit"] = min(int(filters.get("limit", 500)), 500)
        filters.setdefault("offset", 0)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            (
                "interval_id",
                "pump_number",
                "started_at_utc",
                "ended_at_utc",
                "duration_seconds",
                "control_mode",
                "automation_ids",
                "end_reason",
            )
        )
        offset = 0
        while True:
            page = self.query_intervals(**{**filters, "limit": 500, "offset": offset})
            for row in page["rows"]:
                writer.writerow(
                    (
                        row["id"],
                        row["pump_number"],
                        row["started_at_utc"],
                        row["ended_at_utc"] or "",
                        "" if row["duration_seconds"] is None else row["duration_seconds"],
                        row["control_mode"],
                        "|".join(row["automation_ids"]),
                        row["end_reason"] or "",
                    )
                )
            offset += len(page["rows"])
            if not page["rows"] or offset >= page["total"]:
                break
        return output.getvalue().encode("utf-8")

    def status(self) -> dict:
        with self._lock:
            total = int(
                self._connection.execute(
                    "SELECT COUNT(*) AS count FROM pump_intervals"
                ).fetchone()["count"]
            )
            open_count = int(
                self._connection.execute(
                    "SELECT COUNT(*) AS count FROM pump_intervals WHERE ended_at_utc IS NULL"
                ).fetchone()["count"]
            )
            return {
                "healthy": self._last_error is None,
                "total_intervals": total,
                "open_intervals": open_count,
                "last_error": self._last_error,
            }

    def close(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA wal_checkpoint(FULL)")
            self._connection.close()
