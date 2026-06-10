"""Ingest CSV files (e.g. content review queues) as Notes.

Each row becomes a Note. The first row is the header. Recognized columns
get folded into the title/body/date/tags so the rest of the pipeline
(cleaning, keyword extraction, embedding, clustering) can treat them the
same as markdown notes.

Default shape (option 1 in the design discussion):

- title:  "[<id>] <theme>"  (e.g. "[2026-05-22-001] AI operator workflow")
- date:   from `date_created` (column name is configurable)
- tags:   `theme` + `platform` (configurable)
- body:   multi-line, with draft/hook/your_comment/notes_for_revision
          as labeled sections so the LLM and keyphrase extractor have
          full context.

Unknown columns are kept under the "extra" frontmatter key and ignored
by the rest of the pipeline.
"""

from __future__ import annotations

import csv
from pathlib import Path

from text_theme_analyzer.pipeline.model import Note
from text_theme_analyzer.utils.dates import coerce_to_date

DEFAULT_COLUMN_MAP: dict[str, str] = {
    "id": "id",
    "date": "date_created",
    "source": "source",
    "theme": "theme",
    "platform": "platform",
    "format": "format",
    "draft": "draft",
    "hook": "hook",
    "tone": "tone",
    "personal_level": "personal_level",
    "private_risk": "private_risk",
    "status": "status",
    "comment": "your_comment",
    "posted_url": "posted_url",
    "reuse_as": "reuse_as",
    "notes": "notes_for_revision",
}


def _row_field(row: dict, *names: str) -> str:
    """Return the first non-empty value among the named columns."""
    for n in names:
        v = row.get(n)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _row_to_note(
    path: Path,
    row_index: int,
    row: dict,
    column_map: dict[str, str],
) -> Note | None:
    """Build a Note from one CSV row. Returns None if the row is empty
    (no id, no draft) so we don't pollute the pipeline with noise.
    """

    def cell(*logical: str) -> str:
        """Return the first non-empty value across the given logical fields."""
        for k in logical:
            csv_col = column_map.get(k)
            if not csv_col:
                continue
            v = row.get(csv_col)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    id_ = cell("id")
    draft = cell("draft")
    if not id_ and not draft:
        return None  # blank row

    theme = cell("theme")
    title = f"[{id_}] {theme}" if id_ and theme else (theme or id_ or f"row {row_index + 1}")

    # Body: labeled sections so the keyword extractor and LLM can see
    # the structure. Skip empty fields.
    sections: list[tuple[str, str]] = []
    if theme:
        sections.append(("Theme", theme))
    for label, key in (
        ("Source", "source"),
        ("Platform", "platform"),
        ("Format", "format"),
        ("Tone", "tone"),
        ("Personal level", "personal_level"),
        ("Private risk", "private_risk"),
        ("Status", "status"),
        ("Reuse as", "reuse_as"),
    ):
        v = cell(key)
        if v:
            sections.append((label, v))
    if draft:
        sections.append(("Draft", draft))
    for label, key in (("Hook", "hook"), ("Your comment", "comment"), ("Notes for revision", "notes")):
        v = cell(key)
        if v:
            sections.append((label, v))

    body = "\n\n".join(f"{label}: {text}" for label, text in sections)

    # Date
    raw_date = cell("date")
    note_date = coerce_to_date(raw_date) if raw_date else None

    # Tags: theme + platform (kept distinct so clusters can split on either).
    tags: list[str] = []
    for k in ("theme", "platform"):
        v = cell(k)
        if v and v not in tags:
            tags.append(v)

    # Pass unknown columns through as frontmatter extras (useful for
    # debug/diff, harmless to the rest of the pipeline).
    known_csv_cols = {c for c in column_map.values() if c}
    extras = {k: v for k, v in row.items() if k not in known_csv_cols and v not in (None, "")}

    # Stable ID: hash on path + row index so the same row maps to the
    # same note across re-runs (matches the spirit of note_id_for_path).
    from hashlib import sha1
    nid = sha1(f"{path.resolve()}::{row_index}".encode()).hexdigest()[:12]

    return Note(
        id=nid,
        path=path,
        title=title,
        body=body,
        date=note_date,
        tags=tags,
        frontmatter={
            "id": id_,
            "date": raw_date or None,
            "theme": theme,
            "platform": cell("platform"),
            "format": cell("format"),
            "status": cell("status"),
            "tone": cell("tone"),
            "personal_level": cell("personal_level"),
            "private_risk": cell("private_risk"),
            "source": cell("source"),
            "extras": extras,
        },
    )


def load_csv(
    path: Path,
    *,
    column_map: dict[str, str] | None = None,
) -> list[Note]:
    """Parse a CSV file into a list of Notes (one per non-empty row).

    `column_map` overrides which CSV column feeds which logical field.
    Keys are logical names ("id", "date", "draft", ...); values are the
    actual CSV column names. See `DEFAULT_COLUMN_MAP` for defaults.
    """
    column_map = column_map or DEFAULT_COLUMN_MAP
    # Tolerate BOM and any encoding hiccup the frontmatter loader might miss.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    notes: list[Note] = []
    for i, row in enumerate(rows):
        note = _row_to_note(path, i, row, column_map)
        if note is not None:
            notes.append(note)
    return notes
