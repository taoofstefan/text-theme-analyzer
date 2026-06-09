"""Ingest markdown notes from a folder.

Walks a directory tree, filters by include/exclude globs, parses YAML
frontmatter, and produces a list of `Note` dataclasses.

CSV files (e.g. content review queues) are also supported when included
via `--include "**/*.csv"`. Each row becomes a Note; the loader is in
`pipeline.csv_ingest`.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import frontmatter

from text_theme_analyzer.pipeline.csv_ingest import load_csv
from text_theme_analyzer.pipeline.model import Note
from text_theme_analyzer.utils.dates import resolve_note_date
from text_theme_analyzer.utils.hashing import note_id_for_path

_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_INLINE_TAG_RE = re.compile(r"#([a-zA-Z][\w\-/]+)")


def _matches_any(path: Path, patterns: list[str], root: Path) -> bool:
    rel = str(path.relative_to(root)).replace("\\", "/")
    name = path.name
    for p in patterns:
        # fnmatch's `**` is not special, so normalize `**/*.md` -> `*.md`
        # so it can match a top-level file.
        normalized = p
        while normalized.startswith("**/"):
            normalized = normalized[3:]
        if fnmatch.fnmatch(name, normalized) or fnmatch.fnmatch(rel, p):
            return True
    return False


def discover_notes(
    root: Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Path]:
    """Walk `root` and return markdown paths matching include/exclude globs."""
    if include is None:
        include = ["**/*.md", "**/*.markdown"]
    if exclude is None:
        exclude = []

    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if not _matches_any(path, include, root):
            continue
        if exclude and _matches_any(path, exclude, root):
            continue
        found.append(path)
    return found


def _parse_title(fm: dict, body: str, path: Path) -> str:
    if isinstance(fm.get("title"), str) and fm["title"].strip():
        return fm["title"].strip()
    match = _H1_RE.search(body)
    if match:
        return match.group(1).strip()
    return path.stem


def _parse_tags(fm: dict, body: str) -> list[str]:
    tags: list[str] = []
    raw = fm.get("tags") or fm.get("tag")
    if raw is None:
        pass
    elif isinstance(raw, str):
        tags = [t.strip() for t in re.split(r"[,;]", raw) if t.strip()]
    elif isinstance(raw, list):
        tags = [str(t).strip() for t in raw if str(t).strip()]
    inline = _INLINE_TAG_RE.findall(body)
    for t in inline:
        if t not in tags:
            tags.append(t)
    return tags


def load_note(path: Path) -> Note:
    """Parse a single markdown file into a `Note`."""
    text = path.read_text(encoding="utf-8", errors="replace")
    post = frontmatter.loads(text)
    fm = dict(post.metadata or {})
    body = post.content
    return Note(
        id=note_id_for_path(path),
        path=path,
        title=_parse_title(fm, body, path),
        body=body,
        date=resolve_note_date(frontmatter=fm, path=path),
        tags=_parse_tags(fm, body),
        frontmatter=fm,
    )


def load_notes(
    root: Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Note]:
    """Discover + load all notes under `root`.

    Dispatches by extension: `.csv` files go through `load_csv` (one
    Note per row); everything else is parsed as markdown. The user has
    to opt into CSV via `--include "**/*.csv"`.
    """
    paths = discover_notes(root, include=include, exclude=exclude)
    notes: list[Note] = []
    for p in paths:
        if p.suffix.lower() == ".csv":
            notes.extend(load_csv(p))
        else:
            notes.append(load_note(p))
    return notes
