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

### T1.2 — A "promote to project" action on stale verdicts

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

### T1.3 — Multi-run diff

The CLI has a `runs` subcommand stub (visible in the glob-recovery
test's argv repair logic), but it doesn't currently do meaningful
diff between runs. Running on the same vault a month apart and
seeing *what changed* — new clusters, clusters that grew, clusters
that went quiet — is the actual long-term value of having a
thinking radar at all.

Implementation sketch: the analyzer already writes a snapshot
to `{output_dir}/run-history/{timestamp}.json` on every run.
Add a `tta diff <run_a> <run_b>` subcommand that:

- Matches clusters between runs by cosine similarity of their
  c-TF-IDF centroids.
- Reports: new clusters, removed clusters, grew-since-last-run,
  shrank-since-last-run, stable.
- Optionally renders a 2-column dashboard with the two runs side
  by side.

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
