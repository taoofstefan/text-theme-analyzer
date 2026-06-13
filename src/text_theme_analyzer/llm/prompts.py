"""Prompt templates for LLM enrichment."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a thoughtful editor helping a writer understand what they keep \
circling around in their private notes. You receive structured metadata about \
clusters of notes (keywords, representative titles, sample quotes, time series).

You must return a single JSON object that exactly matches the provided schema. \
Be specific, name actual tensions you can defend from the evidence, and prefer \
concrete language over generic platitudes.

Rules:
- Do not invent quotes. Only use quotes that appear in the input. If a candidate \
  quote isn't strong enough, leave the top_quotes list shorter.
- Tensions connect two DIFFERENT clusters that pull in opposing directions. \
  Do not list tensions within a single cluster.
- For the `evidence` field, paraphrase the supporting material. Do not invent \
  quotes there either.
- For emotional_tone, use 1-3 hyphenated words (e.g. "frustrated-energized").
- If a cluster's `tags` list contains words the user already uses in their own \
  note-tagging, prefer those words (or close synonyms) in `name` and `summary`. \
  The user's existing tag vocabulary is signal — match it where it fits, don't \
  override it with generic labels.
"""


def _format_cluster_block(clusters_for_prompt: list[dict]) -> str:
    parts: list[str] = []
    for c in clusters_for_prompt:
        kw = ", ".join(c.get("keywords", [])) or "(no keywords)"
        kp = ", ".join(c.get("keyphrases", [])) or "(none)"
        tags = ", ".join(c.get("tags", [])) or "(none)"
        titles = "; ".join(c.get("representative_titles", [])) or "(no titles)"
        quotes = "\n    ".join(f'"{q}"' for q in c.get("representative_quotes", [])) or "(no quotes)"
        excerpts = c.get("excerpts") or []
        if excerpts:
            ex_lines = []
            for ex in excerpts:
                ex_lines.append(
                    f'    [{ex.get("date","?")}] {ex.get("title","?")}:\n'
                    f'      {ex.get("body","")}'
                )
            excerpts_block = "\n".join(ex_lines)
        else:
            excerpts_block = "    (no excerpts)"
        parts.append(
            f"- #{c['id']} ({c['size']} notes, "
            f"{c.get('first_seen','?')} -> {c.get('last_seen','?')}):\n"
            f"  keywords: {kw}\n"
            f"  keyphrases: {kp}\n"
            f"  tags: {tags}\n"
            f"  titles: {titles}\n"
            f"  sample quotes:\n    {quotes}\n"
            f"  representative excerpts:\n{excerpts_block}"
        )
    return "\n\n".join(parts)


def build_user_prompt(
    *,
    total_notes: int,
    date_range: tuple[str | None, str | None],
    clusters: list[dict],
    spikes: list[dict],
    stale_candidates: list[dict],
    promote_sections: list[str] | None = None,
) -> str:
    """Build the user-prompt bundle sent to the LLM."""
    clusters_block = _format_cluster_block(clusters)
    spike_block = "\n".join(
        f"- cluster {s['cluster_id']}: {s['bucket']} count={s['count']} (+{s['delta']:.1f})"
        for s in spikes
    ) or "(none)"
    stale_block = "\n".join(
        f"- cluster {s['cluster_id']}: {s['first_seen']} -> {s['last_seen']} freq={s['frequency']}"
        for s in stale_candidates
    ) or "(none)"
    section_hint = ""
    if promote_sections:
        section_hint = (
            "\nWhen a stale verdict is 'promote_to_project', pick a `target_section` "
            f"from the user's configured project-board sections: {promote_sections!r}. "
            "Choose the section that best fits the actionability of the idea. "
            "Return the section name exactly as shown. If none fit, leave it null.\n"
        )
    lo, hi = date_range
    return (
        f"Total notes: {total_notes} ({lo or '?'} to {hi or '?'})\n\n"
        f"Top clusters (id, size, keywords, first/last seen, sample titles+quotes):\n\n"
        f"{clusters_block}\n\n"
        f"Recent spikes:\n{spike_block}\n\n"
        f"Stale-but-recurring candidates:\n{stale_block}\n\n"
        f"{section_hint}"
        f"Return a single JSON object with this exact shape:\n"
        f"{{\n"
        f"  \"clusters\": [\n"
        f"    {{\"cluster_id\": int, \"name\": str, \"summary\": str, "
        f"\"top_quotes\": [str], \"emotional_tone\": str}}\n"
        f"  ],\n"
        f"  \"tensions\": [\n"
        f"    {{\"title\": str, \"pole_a\": str, \"pole_b\": str, "
        f"\"evidence\": [str], \"note\": str}}\n"
        f"  ],\n"
        f"  \"article_candidates\": [\n"
        f"    {{\"title\": str, \"angle\": str, \"supporting_cluster_ids\": [int]}}\n"
        f"  ],\n"
        f"  \"stale_recurring\": [\n"
        f"    {{\"cluster_id\": int, \"theme\": str, "
        f"\"verdict\": \"promote_to_project|archive|keep_observing\", \"reasoning\": str, "
        f"\"target_section\": str | null}}\n"
        f"  ]\n"
        f"}}\n"
    )
