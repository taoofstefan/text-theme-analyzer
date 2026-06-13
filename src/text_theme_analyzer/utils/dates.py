"""Tolerant frontmatter / filename date parsing.

Priority:
1. Frontmatter `date` / `created` / `published` (string, datetime, or date)
2. Filename regex `YYYY-MM-DD` (e.g. `2025-04-01-braindump.md`)
3. File mtime
4. None (excluded from time-series, still in clusters/themes)
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dateutil import parser as dateutil_parser

_FILENAME_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def coerce_to_date(value: Any) -> date | None:
    """Best-effort conversion of a frontmatter value to a date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        # Treat as Unix timestamp.
        try:
            return datetime.fromtimestamp(float(value)).date()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return dateutil_parser.parse(text).date()
        except (ValueError, TypeError, OverflowError):
            return None
    return None


def date_from_filename(path: Path) -> date | None:
    match = _FILENAME_DATE_RE.search(path.name)
    if not match:
        return None
    return coerce_to_date(match.group(1))


def date_from_mtime(path: Path) -> date | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).date()
    except OSError:
        return None


def resolve_note_date(
    *,
    frontmatter: dict,
    path: Path,
) -> date | None:
    for key in ("date", "created", "published"):
        if key in frontmatter:
            d = coerce_to_date(frontmatter[key])
            if d is not None:
                return d
    d = date_from_filename(path)
    if d is not None:
        return d
    return date_from_mtime(path)


def has_authoritative_date(frontmatter: dict, path: Path) -> bool:
    """Return True if the note has an explicit date (frontmatter or filename).

    File mtime is intentionally excluded — it's a filesystem fallback, not an
    authorial date. Used by the --require-dates check (T2.1).
    """
    for key in ("date", "created", "published"):
        if key in frontmatter and coerce_to_date(frontmatter[key]) is not None:
            return True
    return date_from_filename(path) is not None
