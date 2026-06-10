# Follow-up ideas for text-theme-analyzer

This file collects the next-step ideas that came out of running the
analyzer end-to-end on a real Obsidian vault (~150 notes, 6 weeks of
history) on 2026-06-09. They are ordered by ROI for the current user
(the author of the vault) and grouped by effort.

None of this is committed as work yet. The current state of the repo
is at commit `a01442d`: 112/112 tests green, dashboard renders
cleanly on real data, gitleaks workflow active.

---

## Where the current run landed (for context)

- **Corpus**: 150 notes from `data/memory/` after excluding
  `06 Reading/`, `.obsidian/`, `90 System/`, `07 Media/`. The vault
  is ~6 weeks old, so time-series features (spikes, stale detection)
  have little signal to work with yet.
- **Pipeline output**: 205 chunks, 25 clusters (sizes 20 → 3),
  0 outliers, 11 LLM-named clusters, 5 tensions, 7 article candidates.
- **Real-tier bug fixes shipped in `a01442d`** (all surfaced because
  the default 19-note sample masked them):
  - `max_tokens` bumped from 4096 to 12288 (real-sized JSON was
    hitting the ceiling mid-string).
  - Schema `max_length` on `name`/`emotional_tone`/`summary`
    loosened (60/80/600 → 120/160/1000).
  - `TEXTHEME_OLLAMA_TIMEOUT` env var is now actually read by
    `apply_env_overrides` — was documented in `.env.example` but
    silently ignored.
- **Known limitations of the run itself** (not bugs, design ceiling):
  - 6-week date range (older notes lack `date:` frontmatter).
  - 10 of 22 LLM-suggested quotes dropped by the strict verbatim
    validator.

---

## What the output told us about the vault

These are observations worth keeping alongside the run artifacts
in `out/themes-report.md`:

- The vault has **~7 stable lanes** of thought, not "scattered
  notes": regulated-banking AI consulting, the memory-vault-as-life-OS,
  the two passion projects (BarcodeCine, Oura Lab), the anti-bullshit
  agency frame, recovery-era self-reflection, the games/childhood
  memory lane, and the financial-pressure-vs-freedom thread. Daily
  notes and the question game are *feeders* for these, not lanes
  themselves.
- The **two most actionable tensions** the LLM pulled out that the
  vault doesn't name explicitly anywhere:
  1. The gap between *what you can credibly claim* (premium
     regulated-banking AI consulting) and *what you can actually bill
     today* (Fiverr's $20-$125/hr reality).
  2. The gap between the passion projects being *finally buildable*
     and the income question being *urgent*.
- **What is missing from the output is also informative**: 0 stale
  verdicts, and spikes clustered in the last 6 weeks. Both are
  because the corpus is young — these features will earn their keep
  over the next 3-6 months.

---

## What is frontmatter, and why the date situation matters

Frontmatter is the YAML block at the very top of a markdown file,
between two `---` lines:

```markdown
---
date: 2026-06-09
title: Anti-bullshit agency work
tags: [consulting, philosophy, life-direction]
---

# Anti-bullshit agency work

Body of the note starts here...
```

It is a convention shared by Obsidian, Jekyll, Hugo, and most
static-site generators. The analyzer's `date_resolution_priority`
in `README.md` is: frontmatter `date` → `created` → `published` →
filename regex `YYYY-MM-DD` → file mtime → `None`.

The vault has `date:` on some recent notes (daily notes, `MEMORY.md`)
but not on older Self/Themes notes. With no `date:` the tool falls
back to file modified time, which is "when did you last save the
file", not "when did you write the thought". For a personal vault
that is a bad proxy and collapses the apparent time range.

**Cheapest fix**: add a `date:` line to the ~50 notes that already
have a YYYY-MM-DD prefix in their filename. Obsidian's find-and-
replace, or a one-shot script over the filename → frontmatter, is
~15 minutes of work and pays off forever for the time-series and
stale-detection features.

---

## Tier 1 — Better signal, hours of work

### T1.1 — Tag-weighted clustering and LLM prompt

> **DONE** (T1.1 implementation). New CLI flags: `--tag-weight FLOAT`
> (default `0.0` = off) and `--top-n-tags INT` (default `20`). The
> per-cluster tag distribution is also passed to the LLM in the
> enrichment prompt; the system prompt now asks the model to prefer
> the user's existing tag vocabulary when naming clusters.
> Implementation: `pipeline/clustering.py::build_tag_matrix`,
> `llm/enrichment.py::_cluster_tags`. Tests in
> `tests/test_t11_tag_weighted.py` (14 tests).

Right now clusters are formed purely by embedding similarity, and the
LLM enrichment bundle has no tag information. If notes are tagged
(`#consulting`, `#game-design`, `#self-reflection`, `#berichteki`,
etc.), the tool could:

- Weight tags heavily during clustering (treat a tag as a strong
  prior that two notes belong near each other).
- Pass the tag distribution per cluster to the LLM so cluster
  narratives reference the tag vocabulary the user already has.
- Let the user query "show me what's happening in the consulting
  lane this month" instead of getting one global view.

Implementation sketch: add `tags` to the `Note` dataclass (it
already exists in `model.py`), propagate through `build_bundle()`,
and add a `--tag-weight` CLI flag that reweights embeddings by tag
overlap before clustering.

**Highest leverage change for the current user's day-to-day use.**

### T1.1a — Reconcile the tag-string vs. tag-matrix design split

> **DONE** (T1.1a implementation). `build_tag_matrix` now returns
> `tuple[np.ndarray, list[str]]` — the matrix plus the corpus's
> top-N tag ordering in frequency-desc order (the string→column
> map). The orchestrator unpacks the tuple; the `tag_columns` list
> is captured but unused today, with a comment marking it as the
> future per-tag-weight hook. The dead `mid.metadata["tag_matrix"]`
> stash is gone (it was never read by `_cluster_tags`; removing it
> shrinks the JSON artifact by ~16KB on a 205-chunk corpus with 20
> tags). Tests in `tests/test_t11_tag_weighted.py` cover the new
> shape (frequency order, dedup, columns/width invariant, alignment
> to matrix indices, empty-corpus fallback). 156/156 tests green.

> **TODO** (flagged at end of T1.1a implementation). The
> `build_tag_matrix` signature is now ready for per-tag weights,
> but we don't expose a way to set them. The next step is a
> `tag_weights: dict[str, float]` config (YAML map under
> `tag_weights:`), plumbed through `Config` and applied in
> `cluster_chunks` as a per-column scale on the tag contribution
> (`tag_matrix[:, j] * tag_weights[tag_columns[j]]`). Revisit
> when the user actually has differential weight requirements;
> the current corpus is flat.

**File pointers** (for whoever picks T1.1b up):
- `src/text_theme_analyzer/pipeline/clustering.py:build_tag_matrix`
  (the tuple-return contract; tag_columns is the string→index map)
- `src/text_theme_analyzer/pipeline/clustering.py:cluster_chunks`
  (the `np.hstack([embeddings, tag_matrix * tag_weight])` line —
  the per-column scale goes here)
- `src/text_theme_analyzer/pipeline/orchestrator.py:run` (the
  `tag_matrix, _tag_columns = build_tag_matrix(...)` call; replace
  the `_` with `tag_columns` and pass to a new config field)
- `src/text_theme_analyzer/config.py` (add a `tag_weights: dict[str, float]`
  field; YAML override branch mirroring the `ollama` /
  `promote` blocks)

### T1.2 — A "promote to project" action on stale verdicts

> **DONE** (T1.2 implementation). The dashboard's stale-but-recurring
> table now renders a **Copy command** button next to every
> `promote_to_project` verdict. Clicking it copies a `tta promote
> <promote_key> --from-run <output-dir>` invocation to the clipboard
> with a 2-second "Copied ✓" confirmation. Running the command
> writes a pre-filled project stub to `promote.target_file`
> (default: `./promoted-projects.md` next to the input folder).
>
> Re-promoting the same `promote_key` replaces the existing stub
> in place via an invisible `<!-- promote_key: ... -->` HTML
> comment marker (no wikilinks, no frontmatter, no Obsidian
> assumptions). The file structure is `## <bucket>` → `### <project
> title>` → body with marker + reasoning + supporting notes.
> Configurable via `promote.target_file` and `promote.sections` in
> `text-theme-analyzer.yml`. CLI: `tta promote <key> [--from-run
> PATH] [--output-dir PATH] [--target-file PATH] [--section TEXT]`.
>
> Implementation: `output/promote.py::render_promote_stub` +
> `output/promote.py::apply_promotion`; `cli.py::promote_cmd`;
> `llm/enrichment.py::build_bundle` (adds `promote_key` per cluster);
> `output/json_report.py::_build_promote_keys` (top-level
> `promote_keys` map for the CLI to consume);
> `output/html_dashboard.py::render_html` (button + JS clipboard
> handler); `output/templates/dashboard.html.j2` (CSS + button +
> JS). Tests: `tests/test_t12_promote.py` (26 tests).

The model already produces `stale_recurring` verdicts with a
`Literal["promote_to_project", "archive", "keep_observing"]` field
(`src/text_theme_analyzer/llm/schemas.py`). The dashboard renders
them as read-only text.

The dashboard could render each verdict as a clickable button. The
"promote" action would either:

- Copy a pre-filled note stub (title from the verdict, body from
  the supporting cluster excerpts) to a destination folder the
  user configures, **or**
- Append to a `Memory Vault Kanban.md`-style file the user already
  maintains (the vault has one in `90 System/`).

This is blocked on a design decision — **where do promoted projects
live?** — which the user has to make before any of this is
implementable. Worth a 10-minute conversation.

**Design rule (must hold for any implementation): vault-agnostic.**
The promote action must not assume Obsidian. Obsidian-specific
features — Kanban plugin rendering, `[[wikilinks]]`, the `.obsidian/`
config directory, the vault's `00 Inbox/` convention — are *free
upgrades* for users who happen to use Obsidian, not requirements
for the tool to work. Concretely: the dashboard writes standard
markdown to a path the user configures (`promote.target_file` in
`text-theme-analyzer.yml`, defaulting to a `promoted-projects.md`
next to the input folder); optional `promote.sections` headings
(letting the user set up a Kanban-style column structure) are
plain `## ` headings, no plugin needed to *write* to them. Any
links to source notes use standard markdown link syntax
(`[excerpt](path/to/source.md)`), not `[[wikilinks]]`. The
Obsidian Kanban plugin will render the headings as columns for
free; a non-Obsidian user gets a normal markdown file in any
editor.

### T1.2a — Smarter section routing for promote stubs

> **TODO** (flagged at end of T1.2 implementation). The current
> default routes every new stub to `promote.sections[0]` (or
> `## Promoted` if no sections are configured). That's fine for a
> single-bucket Kanban but is opinionated when the user has set
> up multiple buckets ("To start" / "In progress" / "Archive").

Today the choice is "always land in the first bucket" — the user
moves entries manually in any markdown editor. That's a defensible
default (it matches the vault-agnostic spirit of "let the user
decide"), but it means a `tta promote` invocation never lands in
"In progress" or "Archive" by itself.

Two reasonable extensions, in order of effort:

1. **`--section` is per-invocation today (CLI override).** Wire it
   into a per-cluster override too: a `target_section` field on the
   `StaleVerdict` schema, with the LLM picking a section name from
   `promote.sections` (and falling back to `sections[0]` if it
   can't decide). The cost is a small schema change + LLM prompt
   nudge; the benefit is "I clicked Copy → ran the command → the
   stub landed in the right column without me touching it."

2. **Deterministic severity-based heuristic.** When `sections` is
   non-empty, route by `StaleIdea.severity`:
   - `strong` → `sections[0]` (the "To start" equivalent)
   - `medium` → `sections[0]`
   - `weak` → `sections[-1]` (the "Archive" equivalent)
   No LLM call, but opinionated. Useful as a stepping-stone
   before option 1.

**When to revisit**: the moment a user complains "I have to move
half my promotes by hand" or "the archive bucket is full of stuff
that's still alive". Not a real user request yet.

**File pointers**:
- `src/text_theme_analyzer/output/promote.py::_select_bucket_heading`
- `src/text_theme_analyzer/llm/schemas.py::StaleVerdict` (option 1)
- `src/text_theme_analyzer/pipeline/model.py::StaleIdea.severity`
  (option 2)

### T1.3 — Multi-run diff

> **DONE** (T1.3 implementation). `tta diff OLD NEW` now matches
> clusters across runs by IDF-weighted cosine similarity of their
> top-8 c-TF-IDF keywords (the new `cluster_fingerprints` field on
> `RunSnapshot`). The `Diff` dataclass gains a `stable_clusters`
> category and a `matched_pairs` list; `render_diff` adds a
> "stable:" line and a "Matched: N (avg similarity 0.XX)" line.
> Pre-T1.3 snapshots (no `cluster_fingerprints`) fall back to the
> old raw-ID matching, so old runs and old diffs keep working.
> Schema is additive (`HISTORY_SCHEMA_VERSION` stays at `1.0`).
>
> New `--html PATH` flag on `tta diff` writes a self-contained
> 2-column side-by-side dashboard (`output/diff_dashboard.py` +
> `output/templates/diff_dashboard.html.j2`). The "2-column" is
> the matched-pairs table — not two literal dashboards side by
> side. Static HTML, no JS, no external CSS, XSS-safe via
> Jinja `autoescape=True`.
>
> New `--match-threshold FLOAT` option (default `0.3`) controls
> the cosine-similarity cutoff. Lower = more aggressive
> matching; higher = stricter. Range 0.0-1.0.
>
> Tests: 19 new in `tests/test_history.py` (fingerprint field,
> round-trip, ID/IDF, similarity, matching, stable category,
> threshold knob, backwards compat with old snapshots), 12 new
> in `tests/test_diff_dashboard.py` (HTML self-containment,
> matched/added/removed/keyphrase sections, summary block, XSS
> escape, CLI wiring for `--html` / `--match-threshold` /
> unknown-snapshot errors). 185/185 tests green.
>
> **File pointers** (for whoever picks T1.3 follow-ups up):
> - `src/text_theme_analyzer/output/history.py:_match_clusters`
>   (greedy + symmetric match; raw-ID fallback for old snapshots)
> - `src/text_theme_analyzer/output/history.py:_idf_from_runs`
>   (smoothed IDF over the union of fingerprints)
> - `src/text_theme_analyzer/output/diff_dashboard.py:render_diff_html`
> - `src/text_theme_analyzer/output/templates/diff_dashboard.html.j2`
> - `src/text_theme_analyzer/cli.py:diff_runs` (the updated
>   subcommand with `--html` and `--match-threshold`)

---

## Tier 2 — Substantial features, days of work

### T2.1 — Frontmatter-anchored ingest

The right discipline is: the tool should **require or strongly
prefer** `date:` in frontmatter, and warn loudly (or fail) when a
note has no date. This is a one-time content discipline change
(add `date:` to the older notes) plus a small config change
(turn the warning into a hard error in the analyzer, or a
`--require-dates` flag).

Worth bundling with T1.1 since the user's tagging is probably
correlated with the user's dating.

### T2.2 — Per-folder / per-corpus runs

The user's Reading folder is a fundamentally different kind of
corpus than the Self folder, and trying to do per-lane work in
one global view dilutes both.

The right shape is probably: `text-theme-analyzer` as a tool you
point at *one* focused corpus, with a thin wrapper script that
runs it on each of `Self/`, `Themes/`, `Daily/`, `projects/`
separately and renders four mini-dashboards. This gives you
per-lane "thinking radars" instead of one global view that mixes
them.

Implementation: a `corpora` config that maps name → include/exclude
+ output-dir, and a `tta run-all` subcommand that loops over them
and emits a `corpora/index.html` that links to each mini-dashboard.

This is the change that would make the most *qualitative*
difference to day-to-day use. Probably the highest-value single
feature on this whole list.

### T2.3 — Persistent cluster names across runs

Right now the LLM names clusters fresh each time. If you re-run
next month, cluster #4 might be a different theme because
BERTopic doesn't have stable IDs across runs (the cluster IDs are
arbitrary integers assigned during HDBSCAN fitting).

Adding a name-persistence layer would let the time-series and
multi-run diff actually be comparable:

- After a run, save `{cluster_id: llm_name, embedding_centroid}`.
- On the next run, for each new cluster, find the closest old
  cluster by centroid cosine similarity. If similarity > 0.85,
  reuse the old name. Otherwise it's a genuinely new cluster.

This is what makes T1.3 actually usable.

---

## Tier 3 — A different shape, weeks of work, only worth it for sharing

### T3.1 — Static dashboard served from the vault itself

Drop `out/dashboard.html` into the vault as something like
`00 Inbox/dashboard-latest.html`, link it from `MEMORY.md`, and
you have a self-updating thinking radar you can open in Obsidian.
The HTML is already self-contained (~90KB), no server needed.

The piece that's missing is a tiny `render-to-vault.py` script
that:

- Runs the analyzer.
- Copies `dashboard.html` to a configurable vault path.
- Updates a "last rendered" line in `MEMORY.md`.
- Optional: copies `themes.json` next to the HTML so Obsidian's
  Dataview plugin can show a quick cluster summary in the sidebar.

### T3.2 — Smarter stale detection

The current stale detection uses an 8-week rolling window. A
smarter approach:

- A cluster is "stale" if its new-note rate has been below the
  corpus median for some period **and** the cluster has high
  cosine similarity to a *newer* cluster (meaning you've started
  thinking about the same thing under a different name).

This is much closer to what "stale" actually means to a human —
a thought you keep circling back to but aren't naming the same
way anymore. The second condition (similarity to a newer cluster)
is what the current implementation is missing.

### T3.3 — Obsidian plugin

Same tool, but invoked from a ribbon button or command-palette
action. The repo already has `text-theme-analyzer.yml`
discoverable from any folder, so a thin plugin that:

- Shells out to the CLI.
- Renders the result in an Obsidian pane.
- Has a "re-run" button.

...would make this a daily-driver tool rather than a "run it when
I think of it" tool.

Worth doing only if you decide to share this with people who
already use Obsidian. The current CLI is good enough for personal
use and for one or two friends willing to install it the same way.

---

## What was *not* a problem, worth noting

So that future-me doesn't re-investigate:

- **Embeddings cache works.** Re-running on the same 150 notes is
  near-instant on the embedding stage, so iterative re-runs to
  tune `min_cluster_size` etc. are cheap.
- **BERTopic on 205 chunks is fast.** ~5s on CPU. UMAP is the
  slow part if you tune `umap_n_neighbors` aggressively.
- **The LLM stage degrades gracefully.** When the LLM call fails
  (timeout, parse error, missing key) the pipeline still writes
  the no-LLM artifacts. The error is logged once, not raised.
  Good.
- **Gitleaks is in place.** The secret-scan workflow is active on
  the public repo. A real `TEXTHEME_OLLAMA_API_KEY` would have
  been blocked.
- **`.gitignore` is sufficient.** `data/`, `out/`, `.env`,
  `text-theme-analyzer.yml` are all properly ignored. The
  `text-theme-analyzer-sample-*/` pattern is a safety net for
  any future private corpus.
