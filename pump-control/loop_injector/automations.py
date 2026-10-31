"""Persistent local-wall-clock automation rules for loop-injector."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as wall_time, timedelta
import json
import math
import os
from pathlib import Path
import threading
import uuid
from typing import Any, Callable


STATE_VERSION = 2
RULE_FIELDS = {
    "id",
    "name",
    "enabled",
    "mask",
    "weekdays",
    "start_time",
    "end_time",
    "on_minutes",
    "block_minutes",
}


def _bounded_number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return result


def _parse_time(value: object, name: str) -> int:
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        raise ValueError(f"{name} must use HH:MM")
    try:
        hour = int(value[:2])
        minute = int(value[3:])
    except ValueError as exc:
        raise ValueError(f"{name} must use HH:MM") from exc
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"{name} must be a valid local time")
    return hour * 60 + minute


def _format_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


@dataclass(frozen=True, slots=True)
class AutomationRule:
    id: str
    name: str
    enabled: bool
    mask: int
    weekdays: tuple[int, ...]
    start_minute: int
    end_minute: int
    on_seconds: float
    gap_seconds: float

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, generate_id: bool = False
    ) -> "AutomationRule":
        if not isinstance(data, dict):
            raise ValueError("automation must be an object")
        allowed = RULE_FIELDS - ({"id"} if generate_id else set())
        if set(data) != allowed:
            raise ValueError(f"automation fields must be exactly {sorted(allowed)}")

        rule_id = uuid.uuid4().hex if generate_id else data["id"]
        if (
            not isinstance(rule_id, str)
            or len(rule_id) != 32
            or any(char not in "0123456789abcdef" for char in rule_id)
        ):
            raise ValueError("automation id is invalid")
        name = data["name"]
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("name must contain 1 through 80 characters")
        if not isinstance(data["enabled"], bool):
            raise ValueError("enabled must be true or false")
        mask = data["mask"]
        if isinstance(mask, bool) or not isinstance(mask, int) or not 1 <= mask <= 15:
            raise ValueError("mask must select at least one pump")
        raw_weekdays = data["weekdays"]
        if not isinstance(raw_weekdays, list) or not raw_weekdays:
            raise ValueError("weekdays must be a non-empty array")
        if any(
            isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6
            for day in raw_weekdays
        ):
            raise ValueError("weekdays entries must be integers from 0 through 6")
        weekdays = tuple(sorted(set(raw_weekdays)))
        if len(weekdays) != len(raw_weekdays):
            raise ValueError("weekdays must not contain duplicates")

        start_minute = _parse_time(data["start_time"], "start_time")
        end_minute = _parse_time(data["end_time"], "end_time")
        if end_minute <= start_minute:
            raise ValueError("end_time must be later than start_time on the same day")
        on_minutes = _bounded_number(data["on_minutes"], "on_minutes", 0.1, 1440)
        block_minutes = _bounded_number(
            data["block_minutes"], "block_minutes", 0.1, 1440
        )
        window_seconds = (end_minute - start_minute) * 60
        if (on_minutes + block_minutes) * 60 > window_seconds:
            raise ValueError(
                "the OFF gap and ON duration must fit inside the daily time window"
            )

        return cls(
            rule_id,
            name.strip(),
            data["enabled"],
            mask,
            weekdays,
            start_minute,
            end_minute,
            on_minutes * 60,
            block_minutes * 60,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "mask": self.mask,
            "weekdays": list(self.weekdays),
            "start_time": _format_time(self.start_minute),
            "end_time": _format_time(self.end_minute),
            "on_minutes": self.on_seconds / 60,
            "block_minutes": self.gap_seconds / 60,
        }

    def with_enabled(self, enabled: bool) -> "AutomationRule":
        data = self.to_dict()
        data["enabled"] = enabled
        return AutomationRule.from_dict(data)


def _at_local_minute(day: date, minute: int, tzinfo) -> datetime:
    return datetime.combine(
        day,
        wall_time(hour=minute // 60, minute=minute % 60),
        tzinfo=tzinfo,
    )


def rule_runtime(rule: AutomationRule, now: datetime) -> dict[str, Any]:
    if now.tzinfo is None:
        raise ValueError("wall clock must be timezone-aware")
    active = False
    active_until: datetime | None = None
    if rule.enabled and now.weekday() in rule.weekdays:
        start = _at_local_minute(now.date(), rule.start_minute, now.tzinfo)
        end = _at_local_minute(now.date(), rule.end_minute, now.tzinfo)
        first_on = start + timedelta(seconds=rule.gap_seconds)
        cycle_seconds = rule.gap_seconds + rule.on_seconds
        if first_on <= now < end:
            elapsed = (now - first_on).total_seconds()
            block_start = start + timedelta(
                seconds=rule.gap_seconds
                + math.floor(elapsed / cycle_seconds) * cycle_seconds
            )
            block_end = min(end, block_start + timedelta(seconds=rule.on_seconds))
            active = block_start <= now < block_end
            if active:
                active_until = block_end

    next_start: datetime | None = None
    if rule.enabled:
        for day_offset in range(8):
            candidate_day = now.date() + timedelta(days=day_offset)
            if candidate_day.weekday() not in rule.weekdays:
                continue
            start = _at_local_minute(candidate_day, rule.start_minute, now.tzinfo)
            end = _at_local_minute(candidate_day, rule.end_minute, now.tzinfo)
            candidate = start + timedelta(seconds=rule.gap_seconds)
            while candidate + timedelta(seconds=rule.on_seconds) <= end:
                if candidate > now:
                    next_start = candidate
                    break
                candidate += timedelta(
                    seconds=rule.gap_seconds + rule.on_seconds
                )
            if next_start is not None:
                break

    return {
        "active": active,
        "active_until": active_until.isoformat() if active_until else None,
        "next_start": next_start.isoformat() if next_start else None,
    }


class AutomationStore:
    """Thread-safe, atomically persisted automation rule authority."""

    def __init__(
        self,
        state_path: Path,
        *,
        wall_clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        self._path = Path(state_path)
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        self._rules: tuple[AutomationRule, ...] = ()
        self._paused = False
        self._last_error: str | None = None
        self._load()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def has_enabled_rules(self) -> bool:
        with self._lock:
            return any(rule.enabled for rule in self._rules)

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rule_id = payload.get("id") if isinstance(payload, dict) else None
            if rule_id is None:
                rule = AutomationRule.from_dict(payload, generate_id=True)
                self._rules = (*self._rules, rule)
            else:
                rule = AutomationRule.from_dict(payload)
                if not any(existing.id == rule.id for existing in self._rules):
                    raise ValueError("automation does not exist")
                self._rules = tuple(
                    rule if existing.id == rule.id else existing
                    for existing in self._rules
                )
            self._persist_locked()
            return rule.to_dict()

    def delete(self, rule_id: str) -> None:
        with self._lock:
            remaining = tuple(rule for rule in self._rules if rule.id != rule_id)
            if len(remaining) == len(self._rules):
                raise ValueError("automation does not exist")
            self._rules = remaining
            self._persist_locked()

    def set_enabled(self, rule_id: str, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be true or false")
        with self._lock:
            found = False
            updated = []
            for rule in self._rules:
                if rule.id == rule_id:
                    found = True
                    updated.append(rule.with_enabled(enabled))
                else:
                    updated.append(rule)
            if not found:
                raise ValueError("automation does not exist")
            self._rules = tuple(updated)
            self._persist_locked()

    def set_paused(self, paused: bool) -> None:
        if not isinstance(paused, bool):
            raise ValueError("paused must be true or false")
        with self._lock:
            self._paused = paused
            self._persist_locked()

    def evaluate(self, now: datetime | None = None) -> dict[str, Any]:
        with self._lock:
            current = self._wall_clock() if now is None else now
            rules = self._rules
            paused = self._paused
            error = self._last_error
        rule_statuses = []
        active_ids = []
        mask = 0
        next_candidates = []
        for rule in rules:
            runtime = rule_runtime(rule, current)
            scheduled_active = runtime["active"]
            status = {
                **rule.to_dict(),
                **runtime,
                "active": scheduled_active and not paused and not error,
            }
            rule_statuses.append(status)
            if scheduled_active and not paused and not error:
                active_ids.append(rule.id)
                mask |= rule.mask
            if runtime["next_start"]:
                next_candidates.append(runtime["next_start"])
        return {
            "paused": paused,
            "mask": 0 if paused or error else mask,
            "active_ids": active_ids if not paused and not error else [],
            "next_start": min(next_candidates) if next_candidates else None,
            "rules": rule_statuses,
            "last_error": error,
            "local_time": current.isoformat(),
            "timezone": current.tzname(),
        }

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or set(raw) != {"version", "paused", "rules"}:
                raise ValueError("automation state has invalid fields")
            if raw["version"] not in (1, STATE_VERSION) or not isinstance(raw["paused"], bool):
                raise ValueError("automation state version or paused value is invalid")
            if not isinstance(raw["rules"], list):
                raise ValueError("automation rules must be an array")
            rule_data = raw["rules"]
            migrated = raw["version"] == 1
            if migrated:
                rule_data = []
                for item in raw["rules"]:
                    if not isinstance(item, dict):
                        raise ValueError("automation must be an object")
                    converted = dict(item)
                    old_period = _bounded_number(
                        converted.get("block_minutes"),
                        "block_minutes",
                        0.1,
                        1440,
                    )
                    on_minutes = _bounded_number(
                        converted.get("on_minutes"), "on_minutes", 0.1, 1440
                    )
                    gap_minutes = old_period - on_minutes
                    if gap_minutes < 0.1:
                        raise ValueError(
                            "legacy block duration cannot be migrated to an OFF gap"
                        )
                    converted["block_minutes"] = gap_minutes
                    rule_data.append(converted)
            rules = tuple(AutomationRule.from_dict(item) for item in rule_data)
            if len({rule.id for rule in rules}) != len(rules):
                raise ValueError("automation ids must be unique")
            self._rules = rules
            self._paused = raw["paused"]
            if migrated:
                self._persist_locked()
        except Exception as exc:
            self._rules = ()
            self._paused = True
            self._last_error = f"automation state rejected: {exc}"[:240]

    def _persist_locked(self) -> None:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self._path.parent, 0o700)
        except OSError:
            pass
        payload = {
            "version": STATE_VERSION,
            "paused": self._paused,
            "rules": [rule.to_dict() for rule in self._rules],
        }
        temporary = self._path.with_name(f".{self._path.name}.{uuid.uuid4().hex}.tmp")
        try:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
            directory = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self._last_error = None
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
