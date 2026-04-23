"""Input time parsing and normalization."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo


def parse_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def normalize_to_utc(dt: datetime, timezone_hint: Optional[str] = None) -> datetime:
    if dt.tzinfo is None:
        if timezone_hint:
            dt = dt.replace(tzinfo=ZoneInfo(timezone_hint))
        else:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_utc_datetime(value: str, timezone_hint: Optional[str] = None) -> datetime:
    return normalize_to_utc(parse_datetime(value), timezone_hint=timezone_hint)

