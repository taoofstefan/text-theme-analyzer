"""Stable content hashes used for cache keys and note IDs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace so trivial edits don't bust the cache."""
    return re.sub(r"\s+", " ", text.strip().lower())


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def note_id_for_path(path: Path) -> str:
    """Stable 12-char ID for a note path. Uses the absolute, normalized path."""
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
