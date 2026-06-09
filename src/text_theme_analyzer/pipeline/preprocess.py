"""Strip code blocks / normalize whitespace, then chunk long bodies."""

from __future__ import annotations

import re

from text_theme_analyzer.pipeline.model import Note, NoteChunk

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BLANK_LINE_RE = re.compile(r"\n\s*\n+")


def clean_body(text: str) -> str:
    """Remove code, images, HTML tags; normalize whitespace."""
    text = _FENCED_CODE_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    text = _IMAGE_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def chunk_text(text: str, *, max_chars: int = 4000, overlap: int = 200) -> list[tuple[str, int]]:
    """Split text on paragraph boundaries; return list of (chunk, char_offset)."""
    if len(text) <= max_chars:
        return [(text, 0)]
    paragraphs = text.split("\n\n")
    chunks: list[tuple[str, int]] = []
    buf = ""
    offset = 0
    cursor = 0
    for para in paragraphs:
        if not para.strip():
            cursor += len(para) + 2
            continue
        if buf and len(buf) + len(para) + 2 > max_chars:
            chunks.append((buf.strip(), offset))
            # Start next chunk with overlap from the tail of the previous.
            tail = buf[-overlap:] if overlap > 0 else ""
            offset = cursor - len(tail)
            buf = tail + "\n\n" + para
        else:
            buf = (buf + "\n\n" + para) if buf else para
        cursor += len(para) + 2
    if buf.strip():
        chunks.append((buf.strip(), offset))
    return chunks


def preprocess_note(note: Note, *, max_chars: int = 4000) -> list[NoteChunk]:
    """Clean + chunk a note. Always returns at least one chunk (the cleaned body)."""
    cleaned = clean_body(note.body)
    pieces = chunk_text(cleaned, max_chars=max_chars)
    return [
        NoteChunk(
            note_id=note.id,
            chunk_index=i,
            text=piece,
            char_offset=offset,
        )
        for i, (piece, offset) in enumerate(pieces)
    ]
