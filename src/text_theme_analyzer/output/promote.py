"""Render and apply a "promote to project" stub from a stale verdict.

This module is intentionally small and pure-function-heavy so the
write logic is easy to test. The CLI subcommand (`tta promote`)
loads `themes.json`, looks up the verdict by `promote_key`, calls
`render_promote_stub()` to get the markdown, then `apply_promotion()`
to write it to the configured target file (default: `promoted-projects.md`
next to the input folder).

File structure:
- The target file is a flat list of `## ` *buckets* (e.g. `## Promoted`,
  `## To start`).
- Each bucket contains one or more *stubs*. A stub starts with an H3
  (`### `) heading (the project title), followed by a body that
  begins with a `<!-- promote_key: ... -->` marker line.
- The marker is the stable id used for idempotent re-promotion: the
  write logic scans the file for the marker and replaces the stub
  in place rather than appending a duplicate.

Vault-agnostic design:
- Source-note links use standard markdown syntax `[title](path)`,
  never `[[wikilinks]]`.
- Section headings are plain `## ` / `### ` — the Obsidian Kanban
  plugin renders `## ` headings as columns for free, but no plugin
  is required to read or write the file.
- The marker is an invisible HTML comment, so the rendered output
  is clean in any markdown viewer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Marker format used to re-locate a previously-written stub. Appears as
# the first non-blank line of the stub's body. Invisible in any
# markdown renderer.
PROMOTE_MARKER = "<!-- promote_key: {key} -->"
PROMOTE_MARKER_RE = re.compile(r"<!--\s*promote_key:\s*([^\s]+?)\s*-->")


@dataclass(frozen=True)
class ClusterContext:
    """The slice of cluster data the stub needs.

    Built by the CLI from the JSON `promote_keys` map + the `files[]`
    list. Kept minimal so the helper has no Analysis dependency.
    """

    cluster_id: int
    theme: str
    reasoning: str
    last_seen: str | None
    first_seen: str | None
    frequency: int | None
    severity: str | None
    keywords: list[str]
    # (note_id, date_iso, title, relpath) tuples in representative order.
    representative_notes: list[tuple[str, str | None, str, str]]


# --- Stub rendering ---

def _marker_block(promote_key: str, context: ClusterContext) -> str:
    """The invisible re-location markers + a debug-info comment line.

    The `<!-- promote_key: ... -->` line is the stable id used by
    `apply_promotion()` to find this stub on re-promote.
    """
    return "\n".join([
        PROMOTE_MARKER.format(key=promote_key),
        f"<!-- cluster_id: {context.cluster_id}, last_seen: {context.last_seen or '?'}, "
        f"frequency: {context.frequency or '?'} -->",
    ])


def _intro(context: ClusterContext) -> str:
    """1-2 sentence framing of the verdict."""
    return f"**Why now**: _{context.reasoning.strip()}_"


def _metadata_lines(context: ClusterContext) -> list[str]:
    """Key phrases + the user-visible metadata block."""
    if not context.keywords:
        return []
    return ["**Key phrases**: " + ", ".join(context.keywords)]


def _excerpts_section(context: ClusterContext) -> list[str]:
    """The 'Supporting notes' subsection. Uses standard markdown links."""
    if not context.representative_notes:
        return []
    lines = ["", "#### Supporting notes", ""]
    for _nid, date_iso, title, relpath in context.representative_notes:
        date_prefix = f"{date_iso} — " if date_iso else ""
        # Forward slashes in markdown link targets even on Windows;
        # both Obsidian and most editors normalize on render.
        link = f"[{title}]({relpath.replace(chr(92), '/')})"
        lines.append(f"- {date_prefix}{link}")
    return lines


def render_promote_stub(promote_key: str, context: ClusterContext) -> str:
    """Render a single promote stub as a markdown fragment.

    The fragment is a `### ` heading (H3) plus a body. It is meant to
    live inside a `## ` bucket in the target file (the bucket is added
    by `apply_promotion`). The body's first non-blank line is the
    `<!-- promote_key: ... -->` marker used for idempotent re-promote.
    """
    # H3 not H2: the target file's `## ` is the bucket, the stub is a
    # sub-section within it.
    heading = f"### {context.theme.strip()}"
    marker = _marker_block(promote_key, context)
    intro = _intro(context)
    meta = _metadata_lines(context)
    excerpts = _excerpts_section(context)
    body_lines = [heading, "", marker, "", intro, ""]
    body_lines.extend(meta)
    body_lines.extend(excerpts)
    return "\n".join(body_lines)


# --- File-level apply ---

def _select_bucket_heading(
    existing_buckets: list[str],
    target_section: str | None,
) -> str:
    """Decide which `## ` bucket heading a new stub should live under.

    Behavior (matches the plan's "option A" — simple default):
    - If `target_section` is given, use that exact heading.
    - Otherwise, use "## Promoted" (single bucket).
    """
    if target_section:
        return f"## {target_section.strip()}"
    return "## Promoted"


def _find_stub_span(
    lines: list[str], promote_key: str
) -> tuple[int, int] | None:
    """Return (start_line, end_line_exclusive) for the stub with this key.

    Scans for the `<!-- promote_key: ... -->` marker; the stub spans
    from the `### ` heading immediately before the marker (or the
    marker itself if no preceding heading) to the line before the
    next `### ` or `## ` heading (or EOF).
    """
    # Find the marker line.
    marker_re = re.compile(r"<!--\s*promote_key:\s*" + re.escape(promote_key) + r"\s*-->")
    marker_idx = None
    for i, line in enumerate(lines):
        if marker_re.search(line):
            marker_idx = i
            break
    if marker_idx is None:
        return None
    # Walk backwards to find the start of the stub (the `### ` heading).
    start = marker_idx
    for i in range(marker_idx - 1, -1, -1):
        if lines[i].startswith("### "):
            start = i
            break
        if lines[i].startswith("## "):
            # Shouldn't happen — marker is inside a bucket. Start at the marker.
            start = marker_idx
            break
        if lines[i].strip() == "":
            continue
        # Encountered a non-blank, non-heading line before finding `### `.
        # Treat the marker as the start of the stub.
        start = marker_idx
        break
    else:
        start = marker_idx
    # Walk forward to find the end (next `### ` or `## ` heading).
    end = len(lines)
    for i in range(marker_idx + 1, len(lines)):
        if lines[i].startswith("### ") or lines[i].startswith("## "):
            end = i
            break
    return start, end


def _find_bucket_span(
    lines: list[str], bucket_heading: str
) -> tuple[int, int] | None:
    """Return (start_line, end_line_exclusive) for the bucket heading.

    The bucket spans from its `## ` heading to the next `## ` (or EOF).
    """
    start = None
    for i, line in enumerate(lines):
        if line == bucket_heading:
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    return start, end


def apply_promotion(
    target_path: Path,
    stub: str,
    promote_key: str,
    target_section: str | None = None,
) -> None:
    """Idempotently write (or replace) a promote stub in the target file.

    Behavior:
    - If the file does not exist: create it with the bucket heading
      followed by the stub.
    - If the file exists and contains a stub with the same `promote_key`:
      replace that stub in place (preserves the bucket heading and any
      other stubs in the same bucket).
    - If the file exists but has no matching stub: append the stub under
      the bucket heading (creating the bucket if it doesn't exist).
    - `target_section` overrides the default bucket name ("Promoted").
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not target_path.exists():
        bucket = _select_bucket_heading([], target_section)
        content = f"{bucket}\n\n{stub}\n"
        target_path.write_text(content, encoding="utf-8")
        return

    content = target_path.read_text(encoding="utf-8")
    # Preserve trailing newline shape: read lines, but track whether the
    # file ended with `\n` for round-tripping.
    ends_with_newline = content.endswith("\n")
    lines = content.splitlines()

    # 1. Look for an existing stub with this key.
    span = _find_stub_span(lines, promote_key)
    if span is not None:
        start, end = span
        # Replace lines[start:end] with the stub.
        new_lines = lines[:start] + stub.splitlines() + lines[end:]
        _write_lines(target_path, new_lines, ends_with_newline)
        return

    # 2. No matching stub — append under the chosen bucket.
    bucket = _select_bucket_heading(
        [l for l in lines if l.startswith("## ")], target_section,
    )
    bucket_span = _find_bucket_span(lines, bucket)
    if bucket_span is not None:
        # Append the stub at the end of the existing bucket.
        start, end = bucket_span
        # Trim trailing blank lines from the bucket so we get a single
        # blank line between the last stub and the new one.
        insert_at = end
        while insert_at > start and lines[insert_at - 1].strip() == "":
            insert_at -= 1
        # The new stub: blank line + stub content.
        new_chunk = ["", *stub.splitlines()]
        new_lines = lines[:insert_at] + new_chunk + lines[insert_at:]
        _write_lines(target_path, new_lines, ends_with_newline)
        return

    # 3. No such bucket — create it at the end of the file.
    # Ensure exactly one blank line between the new bucket and any prior content.
    if lines and lines[-1].strip() != "":
        new_lines = lines + ["", bucket, "", *stub.splitlines()]
    else:
        new_lines = lines + [bucket, "", *stub.splitlines()]
    _write_lines(target_path, new_lines, ends_with_newline)


def _write_lines(
    target_path: Path,
    lines: list[str],
    ends_with_newline: bool,
) -> None:
    """Serialize `lines` back to the target file.

    Preserves the original file's trailing-newline shape. The list
    itself has no trailing empty string from `splitlines()`; we
    re-add the newline explicitly so writers/viewers that care
    (most of them) stay happy.
    """
    content = "\n".join(lines)
    if ends_with_newline or content:
        content += "\n"
    target_path.write_text(content, encoding="utf-8")
