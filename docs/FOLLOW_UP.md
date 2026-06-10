# Follow-up ideas for text-theme-analyzer

This file collects the next-step ideas that came out of running the
analyzer end-to-end on a real Obsidian vault (~150 notes, 6 weeks of
history) on 2026-06-09, plus the gaps surfaced by the 2026-06-10
T1.1–T1.3 ship + lint cleanup. They are ordered by ROI for the
current user (the author of the vault) and grouped by effort.

**Current state of the repo** (commit `48773e7`):
- 185/185 tests green locally and on CI.
- Test workflow (`.github/workflows/test.yml`) runs pytest + ruff
  on a 3.11/3.12 × ubuntu/windows matrix; gitleaks workflow
  (`.github/workflows/secret-scan.yml`) runs on every push to main.
- Tier 1 (T1.1, T1.1a, T1.2, T1.3) is fully shipped. T1.1b and
  T1.2a remain as Tier-1 follow-ups; T1.4 is the new entry
  (yake zero-dep gap, see below).

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

**How to use it (2026-06-10, post-T1.1):**

The flag is off by default so existing runs are unchanged. To turn
it on, pass `--tag-weight 0.3` (or similar; tune by re-running and
eyeballing the dashboard's tag column). The matrix scales the tag
contribution to the embedding vector, so a value of `0.0` is the
pre-T1.1 behavior, `1.0` makes tags as influential as one
embedding dimension, and anything in between is a soft nudge. The
LLM side is independent: cluster names will pull from the user's
tag vocabulary regardless of `--tag-weight`.

**File pointers:**
- `src/text_theme_analyzer/pipeline/clustering.py::build_tag_matrix`
  + `cluster_chunks` (the `np.hstack([embeddings, tag_matrix * tag_weight])` line)
- `src/text_theme_analyzer/llm/enrichment.py::_cluster_tags`
  (per-cluster tag distribution)
- `src/text_theme_analyzer/llm/prompts.py` (the new system-prompt
  nudge to prefer existing tag vocabulary)

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
> to matrix indices, empty-corpus fallback). 185/185 tests green.

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

**How to use it (2026-06-10, post-T1.2):**

Two flows, both available now:

1. **From the dashboard.** Open `out/dashboard.html` in a browser,
   find the `stale-but-recurring` table, and click the **Copy
   command** button next to any `promote_to_project` verdict. The
   button copies a `tta promote <key> --from-run <output-dir>`
   invocation to the clipboard with a 2-second "Copied ✓" toast.
   Paste it into a terminal and run it.

2. **From a fresh `analyze` run.** The cluster-level `promote_key`
   field is now in `themes.json` (top-level `promote_keys` map) and
   in the LLM enrichment bundle, so external scripts (or the
   Obsidian Templater plugin) can iterate over them without parsing
   the dashboard.

The output file is configured via `promote.target_file` in
`text-theme-analyzer.yml` (default: `./promoted-projects.md` next
to the input folder). Re-promoting the same `promote_key`
*replaces* the existing stub in place via an invisible HTML-comment
marker — no wikilinks, no frontmatter, no Obsidian assumptions.

**Design rule (vault-agnostic, must hold for any future work):**
The promote action does not assume Obsidian. Optional `promote.sections`
headings are plain `## ` markdown headings, not Obsidian Kanban
plugin syntax. Links to source notes use standard markdown
`[excerpt](path/to/source.md)` syntax, not `[[wikilinks]]`. The
Obsidian Kanban plugin renders the headings as columns for free;
a non-Obsidian user gets a normal markdown file in any editor.

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

### T1.4 — Real zero-dep fallback for the keyword extractor

> **OPEN** (gap surfaced 2026-06-10, see commit `48773e7` and the
> `project-extract-with-yake-not-zero-dep` memory). Currently
> *worked around* with `pytest.importorskip("yake")` in 4
> end-to-end tests so CI's lean install passes — but the
> underlying bug is unfixed.

The pipeline's `keywords.py::extract_with_yake` has a docstring
that says "YAKE fallback: zero-dep, but lower quality. Scores are
inverted (higher = better)." It is **not** zero-dep — the function
body does `import yake` and crashes if the package is missing. The
orchestrator (`pipeline/orchestrator.py:66`) calls
`extract_keyphrases(..., method="yake")` by default, so a lean
install (`pip install -e ".[dev]"` without the heavy extras)
crashes the *entire* pipeline on `ModuleNotFoundError: No module
named 'yake'`. The `extract_keyphrases` wrapper has a
`try: extract_with_keybert ... except ImportError: extract_with_yake`
path that *was* meant to be the fallback chain, but the orchestrator
bypasses it by passing `method="yake"` directly.

The right fix is two changes:

1. **Replace `extract_with_yake` with a real zero-dep fallback.**
   A small TF-IDF-lite or word-frequency + noun-phrase extractor
   (~50 lines, no deps). It will be lower quality than yake/keybert
   but actually zero-dep, so the docstring claim becomes true.
2. **Change the orchestrator default to `method="keybert"`.** The
   `try/except ImportError` in `extract_keyphrases` then actually
   fires on lean installs, and keybert-heavy users get the better
   result. The default works on every install.

After the fix, the `pytest.importorskip("yake")` gates on the 4
end-to-end tests can be removed (the tests no longer need to skip
on lean installs because the pipeline no longer crashes).

**Why this matters:** today, anyone who installs the package via
`pip install text-theme-analyzer` (not `pip install -e ".[heavy]"`)
and runs `tta analyze` will hit this. It's the one "looks like it
should work but doesn't" install path left in the project.

**File pointers:**
- `src/text_theme_analyzer/pipeline/keywords.py:152`
  (`extract_with_yake` — the docstring lie is here)
- `src/text_theme_analyzer/pipeline/keywords.py:171`
  (`extract_keyphrases` — the existing `try/except ImportError`
  fallback chain; just needs the orchestrator to use it)
- `src/text_theme_analyzer/pipeline/orchestrator.py:66`
  (`method="yake"` default; should be `method="keybert"`)
- `tests/test_cli_glob_recovery.py:148`,
  `tests/test_config_precedence.py:66, 81`,
  `tests/test_m5.py:127` — the 4 importorskip gates to remove
  after the fix lands.

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
