"""Thin progress-bar wrapper. Falls back gracefully if tqdm/rich are missing."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore


@contextmanager
def progress(
    iterable: Iterator,
    *,
    desc: str = "",
    quiet: bool = False,
    total: int | None = None,
) -> Iterator:
    """Wrap an iterable in a progress bar (or pass through if quiet/missing)."""
    if quiet or tqdm is None:
        yield from iterable
        return
    yield from tqdm(iterable, desc=desc, total=total, file=sys.stderr)


def log(message: str, *, quiet: bool = False) -> None:
    """Emit a one-line status message to stderr."""
    if quiet:
        return
    print(message, file=sys.stderr)
